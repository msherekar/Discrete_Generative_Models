#!/usr/bin/env python3
"""Score a finished sweep with all three oracles at once, and plot the result.

Why this exists. The sweep's own per-run figures each show one run judged by
one oracle, and the headline numbers turned out to depend on which oracle that
was. Scored with the indicator ridge, 3.21% of the baseline's 138,243 designs
beat wild type; with the METL-embedding ridge, 2.55%; with METL ft-1d-64, which
is finetuned on 64 examples and is the only near-independent judge available,
0.45%. The indicator oracle is additive over substitution indicators and is fit
on the generator's own training variants, so stacking two of its favourite
substitutions already saturates its range -- which is the mechanism behind the
seven-fold gap. A figure that shows one oracle hides that.

What it plots, from the same tidy table:

  A  mean score against step count, per oracle, flow and diffusion
  B  three-way rank agreement over arm x step cells, with Spearman
  C  fraction of designs above wild type, per oracle
  D  arm ranking under each oracle, as a bump chart

Panels A and B are the robustness argument: the structural findings (flow's
reward arms recover with step count, diffusion's never do) hold at rho ~ 0.96
across all three. Panels C and D are the caveat: the absolute gain over wild
type and the identity of the best arm do not.

Usage:
  dgm-three-oracle --runs 'results/gfp_base_*' --out plots/three_oracle.png
  dgm-three-oracle --table scored.csv --out plots/three_oracle.png
"""
import argparse
import csv
import glob
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ORACLES = ("indicator", "metl_embed", "ft1d64")
LABELS = {"indicator": "indicator ridge\n(fit on train, additive)",
          "metl_embed": "METL embedding\n(independent representation)",
          "ft1d64": "METL ft-1d-64\n(64 examples, near-independent)"}
TAG = re.compile(r"(gfp_\w*?_?([fd])(\d+)_s(\d+))")


# ══════════════════════════════════════════════════════════════════════════════
# Building the tidy table
# ══════════════════════════════════════════════════════════════════════════════

def parse_tag(path):
    """(method, steps, seed) from a run directory name, or None."""
    m = TAG.search(str(path))
    if not m:
        return None
    return ({"f": "flow", "d": "diffusion"}[m.group(2)],
            int(m.group(3)), int(m.group(4)))


def wt_references(wt, data_dir):
    """Each oracle's score for the wild type, the baseline every gain is over.

    The indicator ridge has no substitutions to sum for the wild type, so its
    value is exactly the intercept; the other two are a forward pass.
    """
    from dgm.project1.gfp_oracle import load_oracle, score_sequences
    from dgm.project1.oracles import metl_finetuned as mf
    from dgm.project1.embedding_oracle import load_oracle as load_embed
    from dgm.project1.embedding_oracle import score_sequences as score_embed
    ind = load_oracle(data_dir / "avgfp_oracle_v2.npz")
    emb = load_embed(data_dir / "avgfp_metl_oracle_v2.npz")
    return {"indicator": float(score_sequences([wt], oracle=ind)[0]),
            "metl_embed": float(score_embed([wt], oracle=emb)[0]),
            "ft1d64": float(mf.predict([wt], wt, "ft-1d-64")[0])}


def score_runs(run_dirs, wt, data_dir, out_csv):
    """Score every design in every run with all three oracles; write the table.

    The per-sequence CSV each run already carries supplies the sequences, so
    nothing is resampled and no results.pt is needed -- which matters because
    the sweep remapped those to OSDF and only the tarballs came back.
    """
    from dgm.project1.gfp_oracle import load_oracle, score_sequences
    from dgm.project1.embedding_oracle import load_oracle as load_embed
    from dgm.project1.embedding_oracle import score_sequences as score_embed
    from dgm.project1.oracles import metl_finetuned as mf
    ind = load_oracle(data_dir / "avgfp_oracle_v2.npz")
    emb = load_embed(data_dir / "avgfp_metl_oracle_v2.npz")
    rows = []
    for n, run in enumerate(run_dirs, 1):
        meta = parse_tag(run)
        if meta is None:
            print(f"  [skip] cannot parse method/steps/seed from {run}")
            continue
        found = sorted(Path(run).glob("metrics/*_gfp_sequences.csv"))
        if not found:
            print(f"  [skip] no metrics/*_gfp_sequences.csv in {run}")
            continue
        recs = [r for r in csv.DictReader(open(found[0]))
                if r["method"] not in ("training", "random")]
        seqs = [r["sequence"] for r in recs]
        scored = {"indicator": np.asarray(score_sequences(seqs, oracle=ind), float),
                  "metl_embed": np.asarray(score_embed(seqs, oracle=emb), float),
                  "ft1d64": np.asarray(mf.predict(seqs, wt, "ft-1d-64"), float)}
        for i, r in enumerate(recs):
            rows.append(dict(method=meta[0], steps=meta[1], seed=meta[2],
                             arm=r["mode"], is_wt=int(r["is_wildtype"]),
                             **{c: f"{scored[c][i]:.6f}" for c in ORACLES}))
        print(f"  [{n}/{len(run_dirs)}] {Path(run).name}  {len(recs)} designs",
              flush=True)
    if not rows:
        raise SystemExit("nothing scored; check --runs")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out_csv}  ({len(rows)} rows)")
    return rows


def read_table(path):
    out = []
    for r in csv.DictReader(open(path)):
        r["steps"] = int(r["steps"])
        r["seed"] = int(r["seed"])
        r["is_wt"] = int(r["is_wt"])
        for c in ORACLES:
            r[c] = float(r[c])
        out.append(r)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════

def spearman(xs, ys):
    """Rank correlation without scipy, so this runs wherever numpy does."""
    n = len(xs)
    rx = {v: i for i, v in enumerate(sorted(xs))}
    ry = {v: i for i, v in enumerate(sorted(ys))}
    d = sum((rx[a] - ry[b]) ** 2 for a, b in zip(xs, ys))
    return 1 - 6 * d / (n * (n * n - 1))


def cell_means(rows):
    """{(method, steps, arm): {oracle: mean}} -- one point per sweep cell."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for c in ORACLES:
            acc[(r["method"], r["steps"], r["arm"])][c].append(r[c])
    return {k: {c: st.mean(v[c]) for c in ORACLES} for k, v in acc.items()}


def arm_means(rows, method):
    """{arm: {oracle: mean}} pooled over every step count and seed."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] != method:
            continue
        for c in ORACLES:
            acc[r["arm"]][c].append(r[c])
    return {k: {c: st.mean(v[c]) for c in ORACLES} for k, v in acc.items()}


def above_wt(rows, wt_ref):
    """Fraction of non-wild-type designs scoring above wild type, per oracle."""
    live = [r for r in rows if not r["is_wt"]]
    return {c: sum(1 for r in live if r[c] > wt_ref[c]) / len(live)
            for c in ORACLES}, len(live)


# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

def plot(rows, wt_ref, out_path, arms_a=("cfg", "cfg@4", "spp90", "spp99@60")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = cell_means(rows)
    steps = sorted({k[1] for k in cells})
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.28,
                          height_ratios=[1.15, 1.0, 1.0])
    colour = dict(zip(ORACLES, ("#1b6ca8", "#c1583a", "#2e7d5b")))

    # A -- response to step count, one column per oracle, both methods
    for j, c in enumerate(ORACLES):
        ax = fig.add_subplot(gs[0, j])
        for arm, ls in zip(arms_a, ("-", "--", "-.", ":")):
            for meth, mk in (("flow", "o"), ("diffusion", "s")):
                y = [cells.get((meth, s, arm), {}).get(c) for s in steps]
                if any(v is None for v in y):
                    continue
                ax.plot(steps, y, ls, marker=mk, ms=4, lw=1.4,
                        color=colour[c], alpha=1.0 if meth == "flow" else 0.38,
                        label=f"{arm} ({meth[0]})")
        ax.axhline(wt_ref[c], color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(steps)
        ax.set_xticklabels(steps)
        ax.set_xlabel("integration steps")
        if j == 0:
            ax.set_ylabel("mean oracle score")
        ax.set_title(LABELS[c], fontsize=9)
        ax.legend(fontsize=5.5, ncol=2, loc="lower right")
        ax.grid(alpha=0.25)
    fig.text(0.5, 0.965, "A   solid = flow, faded = diffusion; dotted line = "
             "wild type.  Flow's high-eta arms climb with step count on all "
             "three oracles; diffusion's never do.",
             ha="center", fontsize=9)

    # B -- pairwise rank agreement over sweep cells
    pairs = (("indicator", "metl_embed"), ("indicator", "ft1d64"),
             ("metl_embed", "ft1d64"))
    for j, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, j])
        for meth, mk, al in (("flow", "o", 0.8), ("diffusion", "s", 0.45)):
            ks = [k for k in cells if k[0] == meth]
            xs = [cells[k][a] for k in ks]
            ys = [cells[k][b] for k in ks]
            ax.scatter(xs, ys, s=12, marker=mk, alpha=al,
                       label=f"{meth} rho={spearman(xs, ys):+.3f}")
        lo = min(cells[k][a] for k in cells) - 0.1
        hi = max(cells[k][b] for k in cells) + 0.1
        ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.set_title("B   rank agreement, one point per arm x step cell",
                         fontsize=9, loc="left")

    # C -- the headline that moves
    ax = fig.add_subplot(gs[2, 0])
    frac, n = above_wt(rows, wt_ref)
    bars = ax.bar(range(3), [100 * frac[c] for c in ORACLES],
                  color=[colour[c] for c in ORACLES])
    for r, c in zip(bars, ORACLES):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(),
                f"{100 * frac[c]:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["indicator", "METL\nembedding", "ft-1d-64"], fontsize=8)
    ax.set_ylabel("% of designs above wild type")
    ax.set_title(f"C   n={n:,} designs", fontsize=9, loc="left")
    ax.grid(alpha=0.25, axis="y")

    # D -- does the best arm survive the oracle swap?
    ax = fig.add_subplot(gs[2, 1:])
    am = arm_means(rows, "flow")
    order = {c: sorted(am, key=lambda a: -am[a][c]) for c in ORACLES}
    top = order["ft1d64"][:10]
    for arm in top:
        ys = [order[c].index(arm) + 1 for c in ORACLES]
        ax.plot(range(3), ys, marker="o", ms=4, lw=1.2,
                color="#c1583a" if arm.startswith("cfg") else "#2e7d5b")
        ax.annotate(arm, (2.02, ys[-1]), fontsize=7, va="center")
    ax.invert_yaxis()
    ax.set_xticks(range(3))
    ax.set_xticklabels(["indicator", "METL embedding", "ft-1d-64"], fontsize=8)
    ax.set_xlim(-0.15, 2.6)
    ax.set_ylabel("rank among flow arms")
    ax.set_title("D   red = cfg arms, green = setpoint/reward arms.  cfg@4 is "
                 "first under the indicator oracle and seventh under both "
                 "independent ones.", fontsize=9, loc="left")
    ax.grid(alpha=0.25, axis="y")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"  figure -> {out_path}")
    return frac


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    from dgm.common.paths import project_dir
    root = project_dir()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=None,
                   help="Glob of run directories, each holding "
                        "metrics/*_gfp_sequences.csv. Scores them with all "
                        "three oracles and caches the table at --table.")
    p.add_argument("--table", type=Path, default=root / "plots" / "three_oracle.csv",
                   help="Tidy per-design table. Read when --runs is omitted, "
                        "written when it is given.")
    p.add_argument("--out", type=Path, default=root / "plots" / "three_oracle.png")
    p.add_argument("--wt", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=root / "data")
    return p.parse_args()


def main():
    args = parse_args()
    wt = (args.wt or args.data_dir / "avgfp_wt.txt").read_text().strip()
    wt_ref = wt_references(wt, args.data_dir)
    print("wild-type reference per oracle: "
          + "  ".join(f"{c} {wt_ref[c]:+.4f}" for c in ORACLES))
    if args.runs:
        runs = sorted(glob.glob(args.runs))
        if not runs:
            raise SystemExit(f"--runs matched nothing: {args.runs}")
        args.table.parent.mkdir(parents=True, exist_ok=True)
        rows = score_runs(runs, wt, args.data_dir, args.table)
        rows = read_table(args.table)
    else:
        if not args.table.is_file():
            raise SystemExit(f"no table at {args.table}; pass --runs to build it")
        rows = read_table(args.table)
        print(f"  read {len(rows)} rows from {args.table}")
    frac, n = above_wt(rows, wt_ref)
    cells = cell_means(rows)
    for meth in ("flow", "diffusion"):
        ks = [k for k in cells if k[0] == meth]
        print(f"  {meth:<10} {len(ks)} cells   "
              + "  ".join(f"{a}~{b} {spearman([cells[k][a] for k in ks], [cells[k][b] for k in ks]):+.3f}"
                          for a, b in (("indicator", "metl_embed"),
                                       ("indicator", "ft1d64"),
                                       ("metl_embed", "ft1d64"))))
    print("  above wild type: "
          + "  ".join(f"{c} {100 * frac[c]:.2f}%" for c in ORACLES))
    plot(rows, wt_ref, args.out)


if __name__ == "__main__":
    main()
