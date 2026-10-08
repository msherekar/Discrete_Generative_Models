#!/usr/bin/env python3
"""Relabel the training CSV with an epistatic oracle, to test for reward gaming.

The question this answers. Everything in the baseline sweep was guided toward
r1, which is the measured DMS score, and then judged mostly by a ridge on
substitution indicators. That oracle is additive: each substitution contributes
a fixed amount no matter what else is present. Two of its highest-coefficient
substitutions already push a variant past the top of its observed range, and a
greedy top-15 stack reaches +4.26 against a clamp at +0.40. So "the model found
bright designs" and "the model found a direction the additive oracle rewards"
are not distinguishable on that oracle alone.

The three-oracle comparison made the worry concrete rather than theoretical:
3.21% of the baseline's designs beat wild type on the indicator ridge, 2.55% on
the METL-embedding ridge, and 0.45% on METL ft-1d-64 -- a seven-fold drop on the
one judge that is both near-independent and non-additive. Either the designs
exploit additivity, or ft-1d-64 is simply stricter. Holding ft-1d-64 out as a
judge cannot separate those.

Guiding toward it can. This writes a copy of the training CSV whose r1 column is
ft-1d-64's prediction instead of the measured assay score, leaving every other
column alone. Train on that and the reward head learns an objective with
interaction structure, and the existing pipeline needs no change -- it just
reads --dataset. Then:

  * if flow's reward arms still work, "reward guidance works" survives an
    objective that cannot be gamed additively;
  * if they collapse the way diffusion's do, the baseline result was partly an
    artifact of a linear reward, and that is worth knowing before six more axes
    are run against it.

The honest alternative reading, which the measurement also has to allow for: a
relabelled r1 is a *predicted* score, so its noise is different from the assay's
and part of any drop is that. Hence --keep-measured, which writes the measured
score to a spare column so a run can report both.

Usage:
  dgm-relabel-oracle --dataset data/avgfp_train_props.csv --target ft-1d-64
  dgm-relabel-oracle --dataset data/avgfp_val.csv --target ft-1d-64 --out data/avgfp_val_ft64.csv
"""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

BATCH = 256


def relabel(rows, wt, target, column, keep_measured, batch=BATCH):
    """Replace `column` with the target model's prediction, row by row.

    Returns (rows, stats) where stats carries the before/after agreement, which
    is the number that says whether the new objective is a different problem or
    the same one with extra noise.
    """
    from dgm.project1.oracles import metl_finetuned as mf
    seqs = [r["sequence"] for r in rows]
    pred = []
    for start in range(0, len(seqs), batch):
        chunk = seqs[start:start + batch]
        pred.append(np.asarray(mf.predict(chunk, wt, target), float))
        done = min(start + batch, len(seqs))
        print(f"  scored {done}/{len(seqs)}", end="\r", flush=True)
    pred = np.concatenate(pred)
    print()
    measured = np.array([float(r[column]) for r in rows])
    for r, v in zip(rows, pred):
        if keep_measured:
            r[f"{column}_measured"] = f"{float(r[column]):.6f}"
        r[column] = f"{v:.6f}"
        if "score" in r and column != "score":
            pass
    # Spearman without scipy.
    rx = {v: i for i, v in enumerate(sorted(measured))}
    ry = {v: i for i, v in enumerate(sorted(pred))}
    n = len(pred)
    d = sum((rx[a] - ry[b]) ** 2 for a, b in zip(measured, pred))
    stats = {"n": n,
             "spearman_vs_measured": 1 - 6 * d / (n * (n * n - 1)),
             "measured_mean": float(measured.mean()),
             "relabelled_mean": float(pred.mean()),
             "measured_range": (float(measured.min()), float(measured.max())),
             "relabelled_range": (float(pred.min()), float(pred.max()))}
    return rows, stats


def copy_sidecars(dataset, out):
    """Carry the wt_attributes.json across, which the constraint arm reads.

    run_experiment resolves the constraint reference from
    <dataset stem>.wt_attributes.json, so a relabelled dataset without it falls
    back to needing METL at training time -- which is exactly what failed on
    OSG in the baseline run.
    """
    src = dataset.with_name(dataset.stem + ".wt_attributes.json")
    if not src.is_file():
        print(f"  [warn] no sidecar at {src.name}; the --constraint-property "
              f"arm will have no reference")
        return None
    dst = out.with_name(out.stem + ".wt_attributes.json")
    shutil.copy2(src, dst)
    print(f"  sidecar -> {dst.name}")
    return dst


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    from dgm.common.paths import project_dir
    root = project_dir()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path,
                   default=root / "data" / "avgfp_train_props.csv",
                   help="Training CSV from prepare_gfp.py / add_properties.py.")
    p.add_argument("--out", type=Path, default=None,
                   help="Default: <dataset stem>_<target>.csv beside the input.")
    p.add_argument("--target", default="ft-1d-64",
                   choices=("ft-1d", "ft-3d", "ft-1d-64", "ft-3d-64"),
                   help="Released finetuned METL model. ft-1d-64 is finetuned "
                        "on 64 examples and is the near-independent one; "
                        "ft-1d saw 80%% of this DMS and is not.")
    p.add_argument("--column", default="r1",
                   help="Property column to overwrite (default r1, the "
                        "brightness objective the sweep maximises).")
    p.add_argument("--keep-measured", action="store_true", default=True,
                   help="Also write <column>_measured, so a run can report "
                        "against both the new objective and the assay.")
    p.add_argument("--no-keep-measured", dest="keep_measured",
                   action="store_false")
    p.add_argument("--wt", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    wt = (args.wt or args.dataset.parent / "avgfp_wt.txt").read_text().strip()
    rows = list(csv.DictReader(open(args.dataset)))
    if not rows:
        raise SystemExit(f"{args.dataset} is empty")
    if args.column not in rows[0]:
        raise SystemExit(f"no column {args.column!r} in {args.dataset.name}; "
                         f"have {list(rows[0])}")
    out = args.out or args.dataset.with_name(
        f"{args.dataset.stem}_{args.target.replace('-', '')}.csv")
    print(f"relabelling {args.column!r} of {len(rows)} rows in "
          f"{args.dataset.name} with METL {args.target}")

    rows, stats = relabel(rows, wt, args.target, args.column,
                          args.keep_measured)
    fields = list(rows[0])
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out}")
    copy_sidecars(args.dataset, out)

    print(f"\n  n                      {stats['n']}")
    print(f"  Spearman vs measured   {stats['spearman_vs_measured']:+.4f}")
    print(f"  mean  measured {stats['measured_mean']:+.4f}"
          f"   relabelled {stats['relabelled_mean']:+.4f}")
    print(f"  range measured [{stats['measured_range'][0]:+.3f},"
          f"{stats['measured_range'][1]:+.3f}]"
          f"   relabelled [{stats['relabelled_range'][0]:+.3f},"
          f"{stats['relabelled_range'][1]:+.3f}]")
    print("\n  Train against it with:")
    print(f"    --dataset {out.relative_to(out.parent.parent) if out.is_absolute() else out}")
    print("  and report BOTH objectives when scoring, since r1 is now a "
          "prediction rather than the assay.")


if __name__ == "__main__":
    main()
