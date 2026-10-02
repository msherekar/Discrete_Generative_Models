#!/usr/bin/env python3
"""Sample a finished run again, at any guidance setting, without retraining.

Sampling is the half of a run that scales with the arm set: twelve arms at 100
samples with --endpoint-guidance is hours of GPU time, while training is fixed.
Keeping both in one OSPool job pushes the total against the 20-hour ceiling of
+JobDurationCategory "Long" and makes every extra arm a retraining risk.

This separates them. Train once with a small --samples, push results.pt to
OSDF, then run the heavy arm set here -- in its own short job, or locally, or
again next week with more arms. The trained field is read from results.pt, so
nothing is retrained and the arms are directly comparable to the original run's.

Compared with dgm-resample-cfg, which sweeps conditioning weight and condition
label only, this accepts the full arm set: reward strength, lambda mixes and
setpoints as well.

Usage:
  dgm-resample --run-dir results/base_s11 --samples 100 \\
      --cfg-weight 0 1 2 4 --reward-eta 30 100 --reward-lambda 0.3 0.5 0.7
  dgm-resample --run-dir results/base_s11 --setpoint -0.4 0.0 0.15 --samples 100
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from dgm.common.paths import cache_dir, outputs_dir

from . import run_experiment as R
from .resample_cfg import rebuild_model, rebuild_objective


def path_from_config(config):
    """The PathSpec a saved run was trained with.

    Reads the new keys when present and falls back to the legacy `interpolant`,
    so a results.pt written before paths.py was factored still re-samples along
    the path it was actually trained on.
    """
    from .pipeline.paths import PathSpec, resolve_path
    if config.get("path_geometry"):
        return PathSpec(config["path_geometry"], config.get("time_schedule", "linear"),
                        config.get("path_scale") or 1.0,
                        config.get("sample_schedule"))
    geometry, schedule = resolve_path(config.get("interpolant") or "linear")
    return PathSpec(geometry, schedule)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="A finished run holding flow/ and diffusion/.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write. Default: outputs/<run-dir>_resampled")
    parser.add_argument("--samples", type=int, default=None,
                        help="Samples per arm. Default: whatever the run used.")
    parser.add_argument("--cfg-weight", type=float, nargs="+", default=[2.0],
                        metavar="W", help="Conditioning weights; 0 is the "
                                          "unconditional control.")
    parser.add_argument("--reward-eta", type=float, nargs="+", default=[0.0],
                        metavar="ETA", help="Reward strengths for single/multi arms.")
    parser.add_argument("--reward-lambda", type=float, nargs="+", default=None,
                        metavar="L1", help="Brightness weight per lam arm.")
    parser.add_argument("--setpoint", type=float, nargs="+", default=None,
                        metavar="Y", help="Target brightness values, RAW units. "
                                          "Percentiles are not accepted here: "
                                          "the training distribution is not in "
                                          "results.pt.")
    parser.add_argument("--setpoint-saturation", type=float, default=4.0)
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="Default: the seed the run used.")
    parser.add_argument("--guidance-clip", type=float, default=None)
    parser.add_argument("--no-normalize-guidance", action="store_true",
                        help="Override the run's --normalize-guidance.")
    parser.add_argument("--no-endpoint-guidance", action="store_true",
                        help="Override the run's --endpoint-guidance.")
    parser.add_argument("--oracle", type=Path, default=None,
                        help="Scoring oracle. Default: the one the run used.")
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="ESM weight cache. Default: the project's.")
    return parser


def build_arms(args, objective, n_obj, stats):
    """Every arm to sample, named exactly as run_experiment would name them."""
    base = (1.,) + (0.,) * (n_obj - 1)
    mixed = ((0.7, 0.3) + (0.,) * (n_obj - 2)) if n_obj >= 2 else (1.,)

    def tag(value):
        return f"{value:g}"

    arms = {f"cfg{'' if i == 0 else '@' + tag(w)}":
            dict(c=1, w=w, eta=0.0, lambdas=base, objective=objective)
            for i, w in enumerate(args.cfg_weight)}
    for position, eta in enumerate(args.reward_eta):
        suffix = "" if position == 0 else f"@{tag(eta)}"
        arms[f"single{suffix}"] = dict(c=1, w=0.0, eta=eta, lambdas=base,
                                       objective=objective)
        if n_obj >= 2:
            arms[f"multi{suffix}"] = dict(c=1, w=0.0, eta=eta, lambdas=mixed,
                                          objective=objective)
        for value in (args.reward_lambda or []):
            if n_obj < 2:
                raise SystemExit("--reward-lambda needs two objective columns")
            arms[f"lam{value:g}{suffix}"] = dict(
                c=1, w=0.0, eta=eta,
                lambdas=(value, 1.0 - value) + (0.,) * (n_obj - 2),
                objective=objective)
        for value in (args.setpoint or []):
            arms[f"sp{value:g}{suffix}"] = dict(
                c=1, w=0.0, eta=eta, lambdas=base,
                objective=R.Objective(
                    n_props=objective.n_props,
                    setpoint=R.Objective.from_raw(value, 0, stats),
                    senses=objective.senses,
                    constraint_index=objective.constraint_index,
                    constraint_threshold=objective.constraint_threshold,
                    rho=objective.rho,
                    setpoint_saturation=args.setpoint_saturation))
    return arms


def main():
    args = build_parser().parse_args()
    device = R.DEVICE
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir or outputs_dir(create=True) / f"{run_dir.name}_resampled"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir == run_dir:
        raise SystemExit("--out-dir must differ from --run-dir")

    esm = tokenizer = None
    cache = args.cache_dir or cache_dir(create=True)
    found = False

    for method in ("flow", "diffusion"):
        path = run_dir / method / "results.pt"
        if not path.is_file() or path.stat().st_size == 0:
            print(f"  [skip] no {method}/results.pt")
            continue
        found = True
        saved = torch.load(path, weights_only=False, map_location="cpu")
        config = saved.get("config") or {}
        stats = saved["stats"]
        n_props = stats["r_mean"].numel()
        model, reward, schedule = rebuild_model(method, saved, config, device)
        objective = rebuild_objective(config, n_props)

        if esm is None:
            esm_id = saved["esm_name"]
            print(f"  loading {esm_id} from {cache}")
            tokenizer = AutoTokenizer.from_pretrained(esm_id, cache_dir=str(cache))
            esm = EsmForMaskedLM.from_pretrained(
                esm_id, cache_dir=str(cache), use_safetensors=True
            ).to(device).eval().requires_grad_(False)

        reference = config.get("reference")
        budgets = config.get("mut_budgets") or [config.get("mut_budget")]
        frozen = tuple(config.get("freeze_positions") or ())
        support = None
        if config.get("restrict_support"):
            support = R.load_support_mask(Path(config["restrict_support"]),
                                          reference,
                                          config.get("support_level", "substitution"))
        if reference is None:
            decode_fns = {None: lambda z: R.decode(z, esm, tokenizer, stats,
                                                   config.get("min_polar", 0))}
        else:
            decode_fns = {
                b: (lambda z, b=b: R.decode_budget(
                    z, esm, tokenizer, stats, reference, b,
                    config.get("decode_temperature", 0.0), frozen,
                    config.get("exact_mutations", False), support))
                for b in budgets if b is not None}
            decode_fns = decode_fns or {
                None: lambda z: R.decode(z, esm, tokenizer, stats,
                                         config.get("min_polar", 0))}
        primary = next(iter(decode_fns))

        oracle_path = None if args.no_oracle else (args.oracle or config.get("oracle"))
        oracle = R.load_brightness_oracle(Path(oracle_path)) if oracle_path else None

        anchor_kwargs = {}
        if config.get("anchor_strength"):
            anchor_kwargs = {"anchor": R.encode_reference(reference, esm,
                                                          tokenizer, stats),
                             "strength": float(config["anchor_strength"])}
        guide = {
            "clip": (args.guidance_clip if args.guidance_clip is not None
                     else config.get("guidance_clip", 0.0)),
            "normalize": (False if args.no_normalize_guidance
                          else bool(config.get("normalize_guidance"))),
            "endpoint_guidance": (False if args.no_endpoint_guidance
                                  else bool(config.get("endpoint_guidance"))),
            "seed": args.sample_seed or int(config.get("sample_seed", 123)),
        }
        n = args.samples or int(config.get("samples", 8))
        n_obj = int(config.get("n_objectives") or n_props)
        arms = build_arms(args, objective, n_obj, stats)

        print(f"\n{method}: {len(arms)} arms x {n} samples")
        latents = {}
        for name, cfg in arms.items():
            if method == "flow":
                z = R.sample_flow(model, reward, n=n, **cfg, **anchor_kwargs,
                                  **guide, path=path_from_config(config))
            else:
                betas, alphas, alpha_bars, post_vars = schedule
                z = R.sample_diffusion(model, reward, alpha_bars, betas, alphas,
                                       post_vars, n=n, **cfg, **anchor_kwargs,
                                       **guide)
            entry = R.decode_all_budgets(z, decode_fns, primary, oracle)
            R.report_arm(name, entry["sequences"], reference, entry["oracle"], z)
            latents[name] = entry

        resampled = dict(config, samples=n, resampled_from=str(run_dir),
                         cfg_weight=list(args.cfg_weight),
                         reward_etas=list(args.reward_eta))
        R.save_results(out_dir, method, latents, model, reward, stats,
                       saved["esm_name"], saved["length"], saved["dim"],
                       config.get("min_polar", 0), saved.get("losses", []),
                       resampled)

    if not found:
        raise SystemExit(f"no results.pt under {run_dir}/{{flow,diffusion}}")
    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
