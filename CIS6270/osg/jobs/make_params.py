#!/usr/bin/env python3
"""Generate the Condor params files for the five innovation axes.

One row per job, written from a single baseline dict so that every axis row
differs from the control in exactly one column -- which is the whole point of
an ablation and is not something a hand-edited 150-line table preserves.
"""
import argparse
from pathlib import Path

# Column order must match the `queue ... from` line in gfp_sweep.sub.
COLUMNS = ("tag", "METHOD", "seed", "epochs", "batch", "hidden", "conditioning",
           "modulation", "predict", "beta", "weighting", "geometry",
           "tschedule", "coupling", "fsolver", "dsolver", "steps", "scaling",
           "censor", "ema", "warmup", "K")

# The Stage 3 control. Every axis row below is this with one field replaced.
#
# steps=100 rather than 50. The baseline sweep measured the response of every
# arm against step count: the cfg and spp90 arms are flat, but the high-eta
# reward arms climb steeply and only plateau by 100 (spp99@60 runs
# 0.02/0.13/0.36/0.51/0.54 at 10/20/50/100/200). A control at 50 sits on the
# steepest part of that curve, so any axis change that happens to shift
# effective integration accuracy reads as an effect on those arms. 100 is the
# first point where flow has settled, and gfp_base_{f,d}100_s11..15 already
# ran there -- so --no-ctl below reuses them instead of recomputing.
BASELINE = {
    "METHOD": "flow", "seed": 11, "epochs": 250, "batch": 256, "hidden": 256,
    "conditioning": "continuous", "modulation": "adaln", "predict": "x0",
    "beta": "cosine", "weighting": "min-snr", "geometry": "segment",
    "tschedule": "linear", "coupling": "independent", "fsolver": "euler",
    "dsolver": "ddim", "steps": 100, "scaling": "score",
    "censor": -2.418182, "ema": 0.999, "warmup": 500, "K": 1000,
}

# Diffusion stops improving far sooner than flow; see Stage 0.5.3.
DIFFUSION_EPOCHS = 150

# The five-replicate protocol.
SEEDS = (11, 12, 13, 14, 15)

# Each axis is a list of (short label, overrides). The first entry of every
# axis is the control, so a reader can see what the comparison is against.
AXES = {
    "path": [                                    # Axis A
        ("ctl", {}),
        ("blin", {"beta": "linear"}),
        ("bsig", {"beta": "sigmoid"}),
        ("arc", {"geometry": "arc"}),
        ("darc", {"geometry": "data-arc"}),
        ("tquad", {"tschedule": "quadratic"}),
        ("tcos", {"tschedule": "cosine"}),
        ("therm", {"tschedule": "hermite"}),
        ("tstep", {"tschedule": "smoothstep"}),
    ],
    "coup": [                                    # Axis B
        ("ctl", {}),
        ("ot", {"coupling": "ot"}),
        ("aux", {"coupling": "aux"}),
        ("inf", {"coupling": "informed"}),
        # The two controls that separate information from scale, which is what
        # makes an informed-coupling claim falsifiable.
        ("infsh", {"coupling": "informed-shuffled"}),
        ("scaled", {"coupling": "scaled"}),
    ],
    "obj": [                                     # Axis C
        ("ctl", {}),
        ("v", {"predict": "v"}),
        ("eps", {"predict": "eps"}),
        ("wnone", {"weighting": "none"}),
        ("wsnr", {"weighting": "snr"}),
        ("wsig", {"weighting": "sigma"}),
        # Stage 0.1: plain MSE on the assay floor, the censoring control.
        ("nocens", {"censor": "none"}),
        ("binary", {"conditioning": "binary"}),
    ],
    "guid": [                                    # Axis D
        ("ctl", {}),
        ("legacy", {"scaling": "legacy"}),
        ("const", {"scaling": "constant"}),
    ],
    "samp": [                                    # Axis E
        ("ctl", {}),
        ("heun", {"fsolver": "heun", "dsolver": "heun"}),
        ("mid", {"fsolver": "midpoint"}),
        ("ddpm", {"dsolver": "ddpm"}),
        # No n10/n20/n50/n200 arms: the baseline sweep ran all five step counts
        # for both methods at these settings, so they are already measured.
    ],
    "arch": [                                    # Stage 1
        ("ctl", {}),
        ("token", {"modulation": "token"}),
        ("noema", {"ema": 0}),
        ("nowarm", {"warmup": 0}),
    ],
}

# Which columns each method's code path actually reads, from pipeline/wiring.py:
# flow_train_kwargs adds the path (geometry, tschedule) and the coupling;
# diffusion_train_kwargs adds predict, beta_schedule, loss_weighting and K; and
# each sampler reads only its own solver. A row that varies a column the chosen
# method never reads reproduces that axis's control exactly, so it is not
# queued -- 75 such jobs were about to be submitted across the six axes.
FLOW_ONLY = {"geometry", "tschedule", "coupling", "fsolver"}
DIFFUSION_ONLY = {"predict", "beta", "weighting", "dsolver", "K"}

METHODS = ("flow", "diffusion")


def reaches(method, overrides):
    """False when every column this variant changes is one `method` ignores."""
    if not overrides:
        return True                      # the control
    ignored = DIFFUSION_ONLY if method == "flow" else FLOW_ONLY
    return not set(overrides) <= ignored


def rows_for(axis, variants, seeds, methods, skip_ctl=False):
    """Every job row for one axis, as dicts keyed by COLUMNS."""
    out = []
    for method in methods:
        for label, overrides in variants:
            if not reaches(method, overrides):
                continue
            if skip_ctl and not overrides:
                continue
            for seed in seeds:
                row = dict(BASELINE)
                row.update(overrides)
                row["METHOD"] = method
                row["seed"] = seed
                if method == "diffusion":
                    row["epochs"] = DIFFUSION_EPOCHS
                row["tag"] = f"gfp_{axis}_{label}_{method[0]}_s{seed}"
                out.append(row)
    return out


def write(path, axis, rows):
    """A params file with a header naming the control and the column order."""
    lines = [
        f"# Axis '{axis}': {len(rows)} jobs. Generated by make_params.py --"
        f" edit that, not this.",
        "#",
        "# Every row differs from the 'ctl' row in exactly one column, so a "
        "difference",
        "# in the result is attributable to that column and nothing else.",
        "#",
        "# " + ", ".join(COLUMNS),
        "",
    ]
    for row in rows:
        lines.append(", ".join(str(row[c]) for c in COLUMNS))
    path.write_text("\n".join(lines) + "\n")
    return len(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axes", nargs="+", default=sorted(AXES),
                        choices=sorted(AXES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        choices=list(METHODS))
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--no-ctl", action="store_true",
                        help="Omit the 'ctl' rows. At steps=100 they are "
                             "byte-identical to gfp_base_{f,d}100_s11..15, "
                             "which already completed, and runs are "
                             "bit-reproducible -- so reuse those results as "
                             "the control instead of spending 10 jobs per "
                             "axis recomputing them.")
    parser.add_argument("--one-seed", action="store_true",
                        help="Emit only the first seed, for a cheap dry run of "
                             "the whole ladder before committing five "
                             "replicates per cell.")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = args.seeds[:1] if args.one_seed else args.seeds
    total = 0
    for axis in args.axes:
        rows = rows_for(axis, AXES[axis], seeds, args.methods,
                        skip_ctl=args.no_ctl)
        path = args.outdir / f"gfp_{axis}_params.txt"
        count = write(path, axis, rows)
        total += count
        print(f"  {path.name:<28} {count:>4} jobs")
    print(f"\n  {total} jobs total across {len(args.axes)} axes.")
    print("  Submit one axis at a time:")
    print("    condor_submit gfp_sweep.sub -append 'PARAMS = gfp_path_params.txt'")


if __name__ == "__main__":
    main()
