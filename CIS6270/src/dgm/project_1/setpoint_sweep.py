#!/usr/bin/env python3
"""Sweep the setpoint on a fine grid, from checkpoints, without retraining.

Why. The baseline's setpoint arms are its most interesting positive result and
its weakest measurement at the same time. Flow respects the requested ordering
-- p50 lands below p90 lands below p99 -- but the achieved spread is only
+0.07 to +0.18 against a requested +0.58, and p90 and p99 come out within 0.03
of each other at every step count. Three requested levels, two distinguishable
outcomes. Both near-independent oracles also rank the spp90 arms first overall,
so this is the arm set worth characterising properly rather than another
training-hyperparameter ablation.

Three levels cannot tell a saturating controller from a coarse one. Six can.
This asks for 50/70/80/90/95/99 and reports achieved against requested, so the
shape of the response is visible: a controller that saturates above the
training distribution's upper tail looks different from one whose gain is just
too low everywhere.

It costs no training. dgm-resample reads the trained field and the reward head
out of results.pt, so each model is a sampling pass. The baseline's step axis
does not touch training -- its 50 jobs trained only 10 distinct models, one per
method and seed -- so ten checkpoints cover the whole grid.

Note on units: dgm-resample takes --setpoint in RAW score units, not
percentiles, because the training distribution is not stored in results.pt.
Percentiles are converted here against --val-dataset, which is the same split
and the same column the sweep itself used to place its three setpoints.

Usage:
  dgm-setpoint-sweep --runs 'outputs/gfp_base_*100_s*' --percentiles 50 70 80 90 95 99
  dgm-setpoint-sweep --runs 'outputs/gfp_base_f100_s11' --dry-run
"""
import argparse
import csv
import glob
import json
import re
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TAG = re.compile(r"(gfp_\w*?_?([fd])(\d+)_s(\d+))")


def percentile_targets(val_csv, percentiles, column="score"):
    """Raw score at each requested percentile of the validation split.

    The sweep's own setpoint arms were placed with --setpoint-percentile against
    this split, so converting here keeps the fine grid on the same scale as the
    three coarse points already measured.
    """
    vals = np.array([float(r[column]) for r in csv.DictReader(open(val_csv))])
    return {p: float(np.percentile(vals, p)) for p in percentiles}


def resample_one(run_dir, targets, samples, saturation, oracle, out_root,
                 extra=(), dry_run=False):
    """One dgm-resample call: every setpoint in `targets`, one sampling pass."""
    out_dir = out_root / f"{Path(run_dir).name}_setpoints"
    cmd = [sys.executable, "-m", "dgm.project1.resample",
           "--run-dir", str(run_dir), "--out-dir", str(out_dir),
           "--setpoint", *[f"{targets[p]:.6f}" for p in sorted(targets)],
           "--setpoint-saturation", str(saturation),
           "--samples", str(samples)]
    if oracle:
        cmd += ["--oracle", str(oracle)]
    cmd += list(extra)
    if dry_run:
        print("  " + " ".join(cmd))
        return out_dir, None
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        print(f"  [fail] {Path(run_dir).name} rc={done.returncode}")
        print("    " + (done.stderr or done.stdout).strip().splitlines()[-1][:300])
        return out_dir, None
    return out_dir, done.stdout


def score_setpoints(out_dirs, requested, wt, data_dir):
    """Achieved score per (method, seed, requested percentile), three oracles.

    Scored from the FASTA that dgm-resample writes, so this needs no second
    results.pt and matches how the baseline was re-scored.
    """
    from dgm.project1.gfp_oracle import load_oracle, score_sequences
    from dgm.project1.embedding_oracle import load_oracle as load_embed
    from dgm.project1.embedding_oracle import score_sequences as score_embed
    from dgm.project1.oracles import metl_finetuned as mf
    ind = load_oracle(data_dir / "avgfp_oracle_v2.npz")
    emb = load_embed(data_dir / "avgfp_metl_oracle_v2.npz")
    raw_to_pct = {f"{v:.6f}": p for p, v in requested.items()}
    rows = []
    for out_dir in out_dirs:
        meta = TAG.search(str(out_dir))
        if meta is None:
            continue
        method = {"f": "flow", "d": "diffusion"}[meta.group(2)]
        seed = int(meta.group(4))
        for fasta in sorted(Path(out_dir).glob("*/*.fasta")):
            arm = fasta.stem
            if not arm.startswith("spp"):
                continue
            seqs = [l.strip() for l in fasta.read_text().splitlines()
                    if l and not l.startswith(">")]
            if not seqs:
                continue
            pct = raw_to_pct.get(arm.replace("spp", ""), None)
            scored = {
                "indicator": np.asarray(score_sequences(seqs, oracle=ind), float),
                "metl_embed": np.asarray(score_embed(seqs, oracle=emb), float),
                "ft1d64": np.asarray(mf.predict(seqs, wt, "ft-1d-64"), float)}
            rows.append(dict(method=method, seed=seed, arm=arm,
                             requested_pct=pct, n=len(seqs),
                             **{f"achieved_{c}": float(np.mean(v))
                                for c, v in scored.items()},
                             **{f"sd_{c}": float(np.std(v))
                                for c, v in scored.items()}))
    return rows


def report(rows, requested):
    """Achieved against requested, averaged over seeds -- the calibration test.

    Prints the realised slope as well as the levels, because a controller that
    tracks with gain 0.2 and one that saturates at the 90th percentile both
    produce a monotone table and are different failures.
    """
    print("\n=== achieved vs requested, mean over seeds ===")
    by = defaultdict(list)
    for r in rows:
        if r["requested_pct"] is not None:
            by[(r["method"], r["requested_pct"])].append(r)
    pcts = sorted({k[1] for k in by})
    for method in ("flow", "diffusion"):
        ks = [(method, p) for p in pcts if (method, p) in by]
        if not ks:
            continue
        print(f"\n-- {method} --")
        print(f"{'requested':>10}{'raw target':>12}" +
              "".join(f"{('ach ' + c):>14}" for c in
                      ("indicator", "metl_embed", "ft1d64")))
        xs, ys = [], []
        for _, p in ks:
            group = by[(method, p)]
            line = f"{p:>9.0f}%{requested[p]:>12.3f}"
            for c in ("indicator", "metl_embed", "ft1d64"):
                line += f"{st.mean(r['achieved_' + c] for r in group):>14.3f}"
            print(line)
            xs.append(requested[p])
            ys.append(st.mean(r["achieved_indicator"] for r in group))
        if len(xs) > 1:
            slope = np.polyfit(xs, ys, 1)[0]
            span_req = max(xs) - min(xs)
            span_ach = max(ys) - min(ys)
            print(f"  realised slope {slope:+.3f}   requested span "
                  f"{span_req:+.3f}   achieved span {span_ach:+.3f}   "
                  f"({100 * span_ach / span_req:.0f}% of requested)")
            distinct = sum(1 for a, b in zip(ys, ys[1:]) if abs(b - a) > 0.05)
            print(f"  {distinct + 1} distinguishable levels out of "
                  f"{len(ys)} requested (gap > 0.05)")


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    from dgm.common.paths import project_dir
    root = project_dir()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True,
                   help="Glob of finished run directories holding flow/ or "
                        "diffusion/ with results.pt. One per method and seed "
                        "is enough: the step axis does not affect training.")
    p.add_argument("--percentiles", type=float, nargs="+",
                   default=[50, 70, 80, 90, 95, 99],
                   help="Requested levels. The baseline measured 50/90/99 and "
                        "could not separate 90 from 99.")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--setpoint-saturation", type=float, default=6.0,
                   help="Match the sweep's value so the comparison holds.")
    p.add_argument("--val-dataset", type=Path, default=None,
                   help="Split the percentiles are taken against "
                        "(default: data/avgfp_val.csv).")
    p.add_argument("--oracle", type=Path, default=None,
                   help="Indicator oracle passed through to dgm-resample.")
    p.add_argument("--out-root", type=Path, default=root / "outputs")
    p.add_argument("--table", type=Path,
                   default=root / "plots" / "setpoint_sweep.csv")
    p.add_argument("--wt", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=root / "data")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the dgm-resample commands and stop.")
    p.add_argument("--score-only", action="store_true",
                   help="Skip resampling; score *_setpoints directories that "
                        "are already there.")
    return p.parse_args()


def main():
    args = parse_args()
    val = args.val_dataset or args.data_dir / "avgfp_val.csv"
    wt = (args.wt or args.data_dir / "avgfp_wt.txt").read_text().strip()
    requested = percentile_targets(val, args.percentiles)
    print("requested setpoints (raw units, from "
          f"{val.name}):")
    for p in sorted(requested):
        print(f"  p{p:<5.0f} -> {requested[p]:+.4f}")

    runs = sorted(glob.glob(args.runs))
    if not runs:
        raise SystemExit(f"--runs matched nothing: {args.runs}")
    print(f"\n{len(runs)} run(s) to resample")

    out_dirs = []
    if args.score_only:
        out_dirs = [args.out_root / f"{Path(r).name}_setpoints" for r in runs]
    else:
        for n, run in enumerate(runs, 1):
            out_dir, _ = resample_one(run, requested, args.samples,
                                      args.setpoint_saturation, args.oracle,
                                      args.out_root, dry_run=args.dry_run)
            if not args.dry_run:
                print(f"  [{n}/{len(runs)}] {Path(run).name} -> {out_dir.name}",
                      flush=True)
            out_dirs.append(out_dir)
    if args.dry_run:
        return

    rows = score_setpoints([d for d in out_dirs if d.is_dir()], requested,
                           wt, args.data_dir)
    if not rows:
        raise SystemExit("no setpoint FASTA found; did dgm-resample succeed?")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    with open(args.table, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {args.table}  ({len(rows)} rows)")
    report(rows, requested)
    json.dump({"requested": requested}, open(
        args.table.with_suffix(".meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
