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
  dgm-resample-cfg --run-dir outputs/esm2_8m_calib3 --cfg-weight 0 1 2 4
  dgm-resample-cfg --run-dir outputs/esm2_8m_calib3 --cfg-weight 0 2 \
      --samples 100 --out-dir outputs/esm2_8m_calib3_cfg

Then score it exactly like a normal run:
  dgm-gfp-metrics --run-dir outputs/esm2_8m_calib3_cfg \
      --oracle data/avgfp_oracle_v2.npz \
      --embedding-oracle data/avgfp_metl_oracle_v2.npz --baseline-n 50
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from dgm.project1 import run_experiment as R

from dgm.common.paths import project_dir

ROOT = project_dir()


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
    parser.add_argument("--condition", type=int, nargs="+", default=[1],
                        metavar="C",
                        help="Condition labels to sample (default: 1). On the "
                             "avGFP split c=1 is bright (training mean r1 -0.175) "
                             "and c=0 is dark (-2.282), so '0 1' contrasts two "
                             "labels the data separates by 2.1 brightness units. "
                             "That is the test of whether the learned conditional "
                             "direction is SEMANTICALLY right rather than merely "
                             "nonzero: a w sweep on one label shows only that the "
                             "field moves, while c=0 against c=1 at the same w "
                             "shows whether it moves the way the label means. If "
                             "the two are indistinguishable, the condition was "
                             "never learned; if they separate in the latent but "
                             "not after decoding, the decode is what erases it.")
    parser.add_argument("--samples", type=int, default=None,
                        help="Samples per arm (default: whatever the run used).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write (default: <run-dir>_cfg). Never the "
                             "run directory itself, so the original stays intact.")
    parser.add_argument("--oracle", type=Path, default=None,
                        help="Scoring oracle (default: the one the run used).")
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="Override the run's sampling seed.")
    parser.add_argument("--clean", action="store_true",
                        help="Delete arms already in --out-dir that this run does "
                             "not itself produce. Without it they are kept and "
                             "reported, because they may be deliberate; but a "
                             "scoring tool reads whatever FASTA files it finds, so "
                             "arms left from an earlier invocation with different "
                             "settings will be scored as though they belonged to "
                             "this one.")
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

        # At w=0 the samplers evaluate the null field and never look at c
        # (v = model(z, t, null); the conditional term is gated on `if w`), so
        # every condition collapses to one arm there. Emitting it once keeps the
        # table honest rather than printing identical rows under different names.
        arms, seen_unconditional = [], False
        for c in args.condition:
            for w in args.cfg_weight:
                if w == 0.0:
                    if seen_unconditional:
                        continue
                    seen_unconditional = True
                    arms.append((None, 0.0))
                else:
                    arms.append((c, w))
        multi = len(args.condition) > 1

        latents = {}
        for i, (c, w) in enumerate(arms):
            if c is None:
                name = "cfg" if i == 0 else "uncond"
            elif multi:
                name = f"c{c}" + ("" if w == 1.0 else f"@{w:g}")
            else:
                name = "cfg" if i == 0 else f"cfg@{w:g}"
            common = dict(n=n_samples, c=int(c or 0), w=float(w), eta=0.0,
                          lambdas=lambdas,
                          anchor=anchor, strength=strength, seed=seed,
                          objective=objective)
            if method == "flow":
                z = R.sample_flow(model, reward, path=path_from_config(config),
                                  **common)
            else:
                betas, alphas, alpha_bars, post_vars = schedule
                z = R.sample_diffusion(model, reward, alpha_bars, betas, alphas,
                                       post_vars, **common)
            entry = R.decode_all_budgets(z, decode_fns, primary, oracle)
            latents[name] = entry
            scores = entry.get("oracle")
            label = "uncond" if c is None else f"c={c}"
            line = (f"    {name:9s} {label:7s} w={w:<5g} "
                    f"unique {len(set(entry['sequences']))}"
                    f"/{len(entry['sequences'])}")
            if scores is not None:
                scores = np.asarray(scores)
                line += (f"   oracle mean {scores.mean():+.4f}"
                         f"   best {scores.max():+.4f}")
                table.append({"method": method, "w": w, "arm": name, "c": c,
                              "latent": entry["latent"].detach().cpu(),
                              "mean": float(scores.mean()), "best": float(scores.max()),
                              "unique": len(set(entry["sequences"]))})
            print(line)

        stale = sorted(f.stem for f in (out_dir / method).glob("*.fasta")
                       if f.stem not in latents) if (out_dir / method).is_dir() else []
        if stale:
            if args.clean:
                for name in stale:
                    (out_dir / method / f"{name}.fasta").unlink()
                print(f"    removed {len(stale)} arm(s) from an earlier run: "
                      f"{', '.join(stale)}")
            else:
                print(f"    [warn] {len(stale)} arm(s) already in {out_dir / method} "
                      f"are not produced by this run: {', '.join(stale)}")
                print(f"           gfp_metrics.py scores every FASTA it finds, so "
                      f"these will appear in the table as if they came from these "
                      f"settings. Pass --clean to remove them.")
        R.save_results(out_dir, method, latents, model, reward, stats,
                       saved["esm_name"], saved["length"], saved["dim"],
                       saved.get("min_polar", 12), saved.get("losses", []),
                       {**config, "cfg_weight": list(args.cfg_weight),
                        "samples": n_samples, "sample_seed": seed,
                        "resampled_from": str(args.run_dir)})

    if table:
        print(f"\n  Sweep (w=0 is the unconditional control; c is the condition label)")
        print(f"  {'method':11s}{'arm':9s}{'c':>5}{'w':>5}{'mean':>10}{'best':>10}"
              f"{'unique':>8}{'delta vs uncond':>17}")
        for method in methods:
            rows = [r for r in table if r["method"] == method]
            base = next((r["mean"] for r in rows if r["w"] == 0.0), None)
            for r in rows:
                delta = "" if base is None else f"{r['mean'] - base:+.4f}"
                label = "-" if r["c"] is None else str(r["c"])
                print(f"  {r['method']:11s}{r['arm']:9s}{label:>5}{r['w']:>5g}"
                      f"{r['mean']:>10.4f}{r['best']:>10.4f}{r['unique']:>8d}{delta:>17}")

        # The contrast the condition sweep exists for: same w, opposite labels.
        # A w sweep on one label only shows that the field moves; this shows
        # whether it moves the way the label means.
        pairs = []
        for method in methods:
            rows = [r for r in table if r["method"] == method and r["c"] is not None]
            for w in sorted({r["w"] for r in rows}):
                at_w = {r["c"]: r for r in rows if r["w"] == w}
                if 0 in at_w and 1 in at_w:
                    # Separating these two tells you WHERE the condition is lost.
                    # A latent gap with no property gap means the field carried
                    # the condition and the decode discarded it; no latent gap
                    # means the field never learned it.
                    gap = float((at_w[1]["latent"] - at_w[0]["latent"])
                                .flatten(1).norm(dim=1).mean())
                    scale = float(at_w[1]["latent"].flatten(1).norm(dim=1).mean())
                    pairs.append((method, w, at_w[1]["mean"], at_w[0]["mean"],
                                  gap, scale))
        if pairs:
            print(f"\n  Condition contrast: does conditioning move the property "
                  f"the way the label means?")
            print(f"  {'method':11s}{'w':>5}{'c=1 (bright)':>15}{'c=0 (dark)':>13}"
                  f"{'separation':>13}{'latent gap':>13}{'% of |z|':>10}")
            for method, w, bright, dark, gap, scale in pairs:
                print(f"  {method:11s}{w:>5g}{bright:>15.4f}{dark:>13.4f}"
                      f"{bright - dark:>+13.4f}{gap:>13.2f}"
                      f"{gap / max(scale, 1e-9):>9.1%}")
            print(f"  The avGFP training labels are separated by 2.107 (c=1 mean "
                  f"r1 -0.175, c=0 mean -2.282).")
            print(f"  Read the two columns together. A latent gap with no "
                  f"property separation means the field")
            print(f"  carried the condition and the decode discarded it; no "
                  f"latent gap means the field never")
            print(f"  learned it. Compare the gap against the decode projection "
                  f"that check_reward_transfer.py")
            print(f"  reports for the same run.")
        elif len(args.condition) > 1:
            print("\n  No c=0/c=1 pair at a shared w>0, so no contrast to report.")
        if not any(r["w"] == 0.0 for r in table):
            print("\n  No w=0 arm: without the unconditional control this sweep "
                  "shows how the result varies with w, not what the conditioning "
                  "contributes.")

    print(f"\nDone. Score it with:\n"
          f"  dgm-gfp-metrics --run-dir {out_dir} \\\n"
          f"      --oracle data/avgfp_oracle_v2.npz \\\n"
          f"      --embedding-oracle data/avgfp_metl_oracle_v2.npz --baseline-n 50")


if __name__ == "__main__":
    main()
