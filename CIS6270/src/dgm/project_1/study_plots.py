#!/usr/bin/env python3
"""The three project-specific figures, from a finished run directory.

  11_setpoint_calibration   achieved brightness against requested, with slope
  12_reward_vs_oracle       reward head against an independent oracle, per eta
  13_lambda_pareto          brightness against substitutions, across lambda

Each writes a PNG and the CSV behind it. They are numbered to continue the
sequence dgm-evaluate and run_experiment --plot produce, so one directory holds
the whole figure set in order.

A figure is skipped, with a reason, when the run lacks the arms it needs:
calibration needs --setpoint arms, the Pareto front needs --reward-lambda arms,
and the reward comparison needs a saved reward head and latents.

Usage:
  dgm-study-plots --run-dir outputs/esm2_8m_long250
  dgm-study-plots --run-dir outputs/esm2_8m_long250 \\
      --oracle data/avgfp_oracle_v2.npz --outdir plots/long250_study
"""
import argparse
from pathlib import Path

from dgm.common.paths import data_dir, plots_dir
from dgm.common.tracking import load_results

from .evaluation.study import lambda_pareto, reward_vs_oracle, setpoint_calibration


def arm_scores(runs, oracle_path=None, embedding_path=None):
    """A brightness score per sample, per arm, per modality.

    Preference order: the scores the run itself saved, then an embedding
    oracle, then the indicator oracle. The saved ones come free; a run made
    with --no-oracle, which is how OSG jobs run, has none and must be scored
    here.
    """
    scores, source = {}, None
    for method, results in runs.items():
        saved = results.get("oracle_brightness") or {}
        if saved and all(v is not None for v in saved.values()):
            scores[method] = saved
            source = source or "saved with the run"
            continue
        sequences = results.get("sequences") or {}
        if embedding_path is not None:
            from . import embedding_oracle as embo
            oracle = embo.load_oracle(embedding_path)
            scores[method] = {a: embo.score_sequences(list(s), oracle)
                              for a, s in sequences.items()}
            source = f"embedding oracle {embedding_path.name}"
        elif oracle_path is not None:
            from .gfp_oracle import load_oracle, score_sequences
            oracle = load_oracle(oracle_path)
            scores[method] = {a: score_sequences(list(s), oracle)
                              for a, s in sequences.items()}
            source = f"indicator oracle {oracle_path.name}"
        else:
            scores[method] = {}
    return scores, source


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="A run directory holding flow/ and diffusion/.")
    parser.add_argument("--oracle", type=Path, default=None, metavar="NPZ",
                        help="Indicator oracle from gfp_oracle.py. Defaults to "
                             "data/avgfp_oracle_v2.npz, then avgfp_oracle.npz, "
                             "when either exists. Only used when the run saved "
                             "no scores of its own.")
    parser.add_argument("--embedding-oracle", type=Path, default=None,
                        metavar="NPZ",
                        help="Embedding oracle from embedding_oracle.py, "
                             "preferred over the indicator one when given. "
                             "Needs METL for the metl backend.")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Where the figures go. Default: "
                             "plots/<run-dir name>_study/")
    parser.add_argument("--prefix", default=None,
                        help="Filename prefix. Default: the run directory name.")
    return parser


def _default_oracle(given):
    if given is not None:
        return given
    for name in ("avgfp_oracle_v2.npz", "avgfp_oracle.npz"):
        candidate = data_dir() / name
        if candidate.is_file():
            return candidate
    return None


def main():
    args = build_parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()

    runs = load_results(run_dir)
    if not runs:
        raise SystemExit(f"no results.pt under {run_dir}/{{flow,diffusion}}")
    print(f"run       : {run_dir}")
    print(f"modalities: {', '.join(sorted(runs))}")

    oracle_path = _default_oracle(args.oracle)
    scores, source = arm_scores(runs, oracle_path, args.embedding_oracle)
    if source is None:
        raise SystemExit(
            "no brightness scores available: the run saved none and no oracle "
            "was found. Pass --oracle or --embedding-oracle.")
    print(f"scores    : {source}")

    outdir = args.outdir or plots_dir(create=True) / f"{run_dir.name}_study"
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or run_dir.name

    written = 0
    for label, function, needs in (
            ("setpoint calibration", setpoint_calibration, "--setpoint arms"),
            ("reward vs oracle", reward_vs_oracle, "a saved reward head and latents"),
            ("lambda Pareto front", lambda_pareto, "--reward-lambda arms"),
    ):
        rows = function(runs, scores, outdir, prefix)
        if rows:
            print(f"  wrote {label}  ({len(rows)} rows)")
            written += 1
        else:
            print(f"  [skip] {label}: this run has no {needs}")

    if not written:
        print("\nNothing to plot. These figures need a run with --setpoint, "
              "--reward-lambda, or several --reward-eta values.")
        return
    png = len(list(outdir.glob(f"{prefix}_1*.png")))
    print(f"\n{png} figures + {written} CSVs in {outdir}/")


if __name__ == "__main__":
    main()
