#!/usr/bin/env python3
"""Re-sample a finished run at new classifier-free guidance weights.

The CFG weight is a sampling-time knob: Lecture 3.4's guided field is

    F_CFG = F_uncond + w * (F_cond - F_uncond)

and w appears only when the field is evaluated, never in the training loss. So
a sweep over w costs one sampling pass per value, not one training run per
value -- and because every arm reads the same saved weights, a difference
between them is the weight and nothing else. run_experiment.py --cfg-weight now
sweeps w within a run; this script recovers the same sweep from a run that has
already finished, without retraining.

Why it matters: a run whose every arm uses w=2 cannot say what the conditioning
contributed. w=0 drops the conditional term entirely and is the unconditional
control, so the gap between w=0 and w>0 is the part of the result the condition
is responsible for. Without it, a gain over a random baseline could equally come
from the generative model, the conditioning, or a support-restricted decoder.

Everything else -- decode budget, temperature, frozen positions, support
restriction, anchor strength, seeds -- is read back from the saved config, so
the new arms are directly comparable to the original ones.

Usage:
  python resample_cfg.py --run-dir outputs/esm2_8m_calib3 --cfg-weight 0 1 2 4
  python resample_cfg.py --run-dir outputs/esm2_8m_calib3 --cfg-weight 0 2 \
      --samples 100 --out-dir outputs/esm2_8m_calib3_cfg

Then score it exactly like a normal run:
  python gfp_metrics.py --run-dir outputs/esm2_8m_calib3_cfg \
      --oracle data/avgfp_oracle_v2.npz \
      --embedding-oracle data/avgfp_metl_oracle_v2.npz --baseline-n 50
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import run_experiment as R

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="A finished run holding flow/ and/or diffusion/.")
    parser.add_argument("--cfg-weight", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0],
                        metavar="W",
                        help="Weights to sample (default: 0 1 2 4). 0 is the "
                             "unconditional control, 1 the ordinary conditional "
                             "field, above 1 extrapolates.")
    parser.add_argument("--samples", type=int, default=None,
                        help="Samples per arm (default: whatever the run used).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write (default: <run-dir>_cfg). Never the "
                             "run directory itself, so the original stays intact.")
    parser.add_argument("--oracle", type=Path, default=None,
                        help="Scoring oracle (default: the one the run used).")
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="Override the run's sampling seed.")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    return parser.parse_args()


def rebuild_model(method, saved, config, device):
    """Reconstruct the trained field from its saved weights."""
    length, dim = saved["length"], saved["dim"]
    hidden = int(config.get("hidden", R.HIDDEN))
    arch = config.get("arch", "mlp")
    schedule = None
    if method == "flow":
        model = R.FlowModel(length, dim, hidden, arch)
    else:
        steps = int(config.get("diffusion_steps") or 1000)
        schedule = R.make_ddpm_schedule(steps)
        betas, alphas, alpha_bars, post_vars = schedule
        model = R.DiffusionModel(length, dim, alpha_bars, hidden, arch,
                                 config.get("predict", "eps"))
    model.load_state_dict(saved["model"])
    model = model.to(device).eval()

    n_props = saved["stats"]["r_mean"].numel()
    reward = R.RewardModel(length, dim, hidden, arch, n_props)
    reward.load_state_dict(saved["reward_model"])
    reward = reward.to(device).eval()
    return model, reward, schedule


def rebuild_objective(config, n_props):
    """The run's objective, so the lambda vector has the right width.

    With eta=0 the reward gradient is never evaluated, but the samplers still
    build their weight vector up front, so the objective has to agree on how
    many columns are scalarized.
    """
    index = config.get("constraint_property")
    index = None if index is None or not config.get("constraint_rho") else index - 1
    senses = config.get("objective_senses")
    return R.Objective(
        n_props=n_props,
        setpoint=None,
        senses=[float(s) for s in senses] if senses else None,
        constraint_index=index,
        constraint_threshold=config.get("constraint_threshold_standardized"),
        rho=float(config.get("constraint_rho") or 0.0),
    )


def main():
    args = parse_args()
    device = R.DEVICE
    out_dir = args.out_dir or args.run_dir.parent / f"{args.run_dir.name}_cfg"
    if out_dir.resolve() == args.run_dir.resolve():
        raise SystemExit("--out-dir must differ from --run-dir")

    methods = [m for m in ("flow", "diffusion")
               if (args.run_dir / m / "results.pt").is_file()]
    if not methods:
        raise SystemExit(f"No results.pt under {args.run_dir}/{{flow,diffusion}}/")

    print(f"\nRe-sampling {args.run_dir.name} at cfg weights {args.cfg_weight}")
    print(f"  methods   : {', '.join(methods)}")
    print(f"  output    : {out_dir}")

    esm = tokenizer = None
    table = []

    for method in methods:
        saved = torch.load(args.run_dir / method / "results.pt",
                           weights_only=False, map_location="cpu")
        config = saved.get("config", {})
        stats = saved["stats"]
        n_samples = args.samples or int(config.get("samples", 8))
        seed = args.sample_seed if args.sample_seed is not None \
            else int(config.get("sample_seed", 123))

        if esm is None:
            from transformers import AutoTokenizer, EsmForMaskedLM
            esm_id = saved["esm_name"]
            print(f"\n  loading {esm_id}")
            tokenizer = AutoTokenizer.from_pretrained(esm_id, cache_dir=str(args.cache_dir))
            esm = EsmForMaskedLM.from_pretrained(
                esm_id, cache_dir=str(args.cache_dir), use_safetensors=True
            ).to(device).eval().requires_grad_(False)

        model, reward, schedule = rebuild_model(method, saved, config, device)
        objective = rebuild_objective(config, stats["r_mean"].numel())
        lambdas = (1.0,) + (0.0,) * (objective.n_obj - 1)

        reference = config.get("reference")
        if reference is None:
            raise SystemExit(f"{method}: the run saved no reference sequence; "
                             f"this script reproduces budgeted decoding only")
        frozen = tuple(config.get("freeze_positions") or ())
        support = None
        if config.get("restrict_support"):
            support = R.load_support_mask(Path(config["restrict_support"]), reference,
                                          config.get("support_level", "substitution"))
        budgets = config.get("mut_budgets") or [config.get("mut_budget")]
        decode_fns = {
            b: (lambda z, b=b: R.decode_budget(
                z, esm, tokenizer, stats, reference, b,
                float(config.get("decode_temperature", 0.0)), frozen,
                bool(config.get("exact_mutations", False)), support))
            for b in budgets}
        primary = next(iter(decode_fns))

        anchor, strength = None, 1.0
        if config.get("anchor_strength"):
            anchor = R.encode_reference(reference, esm, tokenizer, stats)
            strength = float(config["anchor_strength"])

        oracle_path = args.oracle or (Path(config["oracle"]) if config.get("oracle") else None)
        oracle = R.load_brightness_oracle(oracle_path) if oracle_path else None

        print(f"\n  {method}: {n_samples} samples/arm, seed {seed}, budget {primary}"
              f"{', anchored ' + str(strength) if anchor is not None else ''}")

        latents = {}
        for i, w in enumerate(args.cfg_weight):
            name = "cfg" if i == 0 else f"cfg@{w:g}"
            common = dict(n=n_samples, c=1, w=float(w), eta=0.0, lambdas=lambdas,
                          anchor=anchor, strength=strength, seed=seed,
                          objective=objective)
            if method == "flow":
                z = R.sample_flow(model, reward, interpolant=config.get("interpolant", "linear"),
                                  **common)
            else:
                betas, alphas, alpha_bars, post_vars = schedule
                z = R.sample_diffusion(model, reward, alpha_bars, betas, alphas,
                                       post_vars, **common)
            entry = R.decode_all_budgets(z, decode_fns, primary, oracle)
            latents[name] = entry
            scores = entry.get("oracle")
            line = (f"    {name:9s} w={w:<5g} unique {len(set(entry['sequences']))}"
                    f"/{len(entry['sequences'])}")
            if scores is not None:
                scores = np.asarray(scores)
                line += (f"   oracle mean {scores.mean():+.4f}"
                         f"   best {scores.max():+.4f}")
                table.append({"method": method, "w": w, "arm": name,
                              "mean": float(scores.mean()), "best": float(scores.max()),
                              "unique": len(set(entry["sequences"]))})
            print(line)

        R.save_results(out_dir, method, latents, model, reward, stats,
                       saved["esm_name"], saved["length"], saved["dim"],
                       saved.get("min_polar", 12), saved.get("losses", []),
                       {**config, "cfg_weight": list(args.cfg_weight),
                        "samples": n_samples, "sample_seed": seed,
                        "resampled_from": str(args.run_dir)})

    if table:
        print(f"\n  CFG sweep (w=0 is the unconditional control)")
        print(f"  {'method':11s}{'w':>6}{'mean':>10}{'best':>10}{'unique':>9}"
              f"{'delta vs w=0':>14}")
        for method in methods:
            rows = [r for r in table if r["method"] == method]
            base = next((r["mean"] for r in rows if r["w"] == 0.0), None)
            for r in rows:
                delta = "" if base is None else f"{r['mean'] - base:+.4f}"
                print(f"  {r['method']:11s}{r['w']:>6g}{r['mean']:>10.4f}"
                      f"{r['best']:>10.4f}{r['unique']:>9d}{delta:>14}")
        if not any(r["w"] == 0.0 for r in table):
            print("\n  No w=0 arm: without the unconditional control this sweep "
                  "shows how the result varies with w, not what the conditioning "
                  "contributes.")

    print(f"\nDone. Score it with:\n"
          f"  python gfp_metrics.py --run-dir {out_dir} \\\n"
          f"      --oracle data/avgfp_oracle_v2.npz \\\n"
          f"      --embedding-oracle data/avgfp_metl_oracle_v2.npz --baseline-n 50")


if __name__ == "__main__":
    main()
