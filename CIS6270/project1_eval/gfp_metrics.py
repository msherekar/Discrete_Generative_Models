#!/usr/bin/env python3
"""GFP-specific evaluation of generated sequences: oracle brightness, distance to
wild type, chromophore preservation, and novelty.

Replaces the composition proxies in evaluate.py with function-grounded metrics for
the avGFP study. Reads the FASTA files written by run_experiment.py.

Usage:
  python gfp_metrics.py --run-dir outputs/esm2_8m_avgfp_train_4000
  python gfp_metrics.py --run-dir outputs/... --train data/avgfp_train_4000.csv
"""
import argparse, csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gfp_oracle import in_domain, load_fasta, load_oracle, score_sequences

ROOT        = Path(__file__).resolve().parent
METHODS     = ("flow", "diffusion")
MODES       = ("cfg", "single", "multi")
CHROMOPHORE = (63, 64, 65)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def hamming(sequences: list[str], wt: str) -> np.ndarray:
    return np.array([sum(a != b for a, b in zip(seq, wt)) for seq in sequences])


def chromophore_intact(sequences: list[str], wt: str) -> np.ndarray:
    target = "".join(wt[i] for i in CHROMOPHORE)
    return np.array([float("".join(seq[i] for i in CHROMOPHORE) == target) for seq in sequences])


def score_set(sequences: list[str], wt: str, oracle, known: set[str]) -> list[dict]:
    brightness = score_sequences(sequences, oracle)
    distance   = hamming(sequences, wt)
    intact     = chromophore_intact(sequences, wt)
    inside     = in_domain(sequences, oracle)
    return [{"oracle_brightness": round(float(b), 4), "hamming_to_wt": int(d),
             "chromophore_intact": int(c), "novel": int(s not in known),
             "in_domain": int(k), "sequence": s}
            for s, b, d, c, k in zip(sequences, brightness, distance, intact, inside)]


def collect(run_dir: Path, wt: str, oracle, known: set[str]) -> list[dict]:
    rows = []
    for method in METHODS:
        for mode in MODES:
            fasta = run_dir / method / f"{mode}.fasta"
            if not fasta.exists():
                continue
            for row in score_set(load_fasta(fasta), wt, oracle, known):
                rows.append({"method": method, "mode": mode, **row})
    if not rows:
        raise FileNotFoundError(f"No FASTA files under {run_dir}/{{flow,diffusion}}/")
    return rows


def summarize(rows: list[dict], train_rows: list[dict], threshold: float) -> list[dict]:
    summary = []
    for group, label in [(train_rows, ("training", "data"))] + [
        ([r for r in rows if (r["method"], r["mode"]) == (m, g)], (m, g))
        for m in METHODS for g in MODES
    ]:
        if not group:
            continue
        summary.append({
            "method": label[0], "mode": label[1], "n": len(group),
            "mean_brightness":      round(float(np.mean([r["oracle_brightness"] for r in group])), 4),
            "frac_bright":          round(float(np.mean([r["oracle_brightness"] > threshold for r in group])), 3),
            "mean_hamming":         round(float(np.mean([r["hamming_to_wt"] for r in group])), 2),
            "frac_chromophore":     round(float(np.mean([r["chromophore_intact"] for r in group])), 3),
            "frac_novel":           round(float(np.mean([r["novel"] for r in group])), 3),
            "frac_in_domain":       round(float(np.mean([r["in_domain"] for r in group])), 3),
        })
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_brightness(rows, train_rows, out_dir: Path, prefix: str, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [("training\ndata", [r["oracle_brightness"] for r in train_rows])]
    groups += [(f"{m[:4]}\n{g}", [r["oracle_brightness"] for r in rows
                                  if (r["method"], r["mode"]) == (m, g)])
               for m in METHODS for g in MODES]
    groups = [(label, values) for label, values in groups if values]
    ax.boxplot([v for _, v in groups], tick_labels=[l for l, _ in groups])
    ax.axhline(threshold, color="crimson", ls="--", lw=1,
               label=f"bright/dark threshold ({threshold:+.1f})")
    ax.set_ylabel("oracle brightness (wild-type centered)")
    ax.set_title("Predicted brightness by method and guidance mode")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_gfp_brightness.png", dpi=150); plt.close(fig)


def plot_pareto(rows, train_rows, out_dir: Path, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([r["hamming_to_wt"] for r in train_rows],
               [r["oracle_brightness"] for r in train_rows],
               s=8, c="lightgray", label="training data")
    for method, marker in zip(METHODS, ("o", "^")):
        for mode, color in zip(MODES, ("tab:blue", "tab:orange", "tab:green")):
            group = [r for r in rows if (r["method"], r["mode"]) == (method, mode)]
            if group:
                ax.scatter([r["hamming_to_wt"] for r in group],
                           [r["oracle_brightness"] for r in group],
                           marker=marker, c=color, s=45, edgecolors="k", linewidths=.4,
                           label=f"{method}/{mode}")
    ax.set_xlabel("Hamming distance to wild type"); ax.set_ylabel("oracle brightness")
    ax.set_title("Brightness vs. mutational parsimony")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_gfp_pareto.png", dpi=150); plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run output directory holding flow/ and diffusion/")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "avgfp_train_4000.csv",
                        help="Training CSV from prepare_gfp.py (novelty + reference cloud)")
    parser.add_argument("--oracle", type=Path, default=ROOT / "data" / "avgfp_oracle.npz",
                        help="Fitted oracle from gfp_oracle.py --fit")
    parser.add_argument("--threshold", type=float, default=-1.0,
                        help="Bright/dark cutoff; must match prepare_gfp.py --threshold")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Plot/CSV directory (default: plots/<run-dir name>)")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    oracle = load_oracle(args.oracle)
    wt     = oracle[2]
    domain = oracle[3]
    with args.train.open(newline="") as f:
        train_seqs = [r["sequence"].strip().upper() for r in csv.DictReader(f)]
    known      = set(train_seqs)
    train_rows = score_set(train_seqs, wt, oracle, known)
    rows       = collect(args.run_dir, wt, oracle, known)

    out_dir = args.outdir or ROOT / "plots" / args.run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix  = args.run_dir.name
    summary = summarize(rows, train_rows, args.threshold)
    write_csv(out_dir / f"{prefix}_gfp_sequences.csv", rows)
    write_csv(out_dir / f"{prefix}_gfp_summary.csv", summary)
    plot_brightness(rows, train_rows, out_dir, prefix, args.threshold)
    plot_pareto(rows, train_rows, out_dir, prefix)

    header = (f"{'method':<10}{'mode':<9}{'n':>4}{'bright':>9}{'%bright':>9}"
              f"{'hamming':>9}{'%chromo':>9}{'%novel':>8}{'%indom':>8}")
    print("\n" + header + "\n" + "-" * len(header))
    for row in summary:
        print(f"{row['method']:<10}{row['mode']:<9}{row['n']:>4}{row['mean_brightness']:>9.3f}"
              f"{row['frac_bright']:>9.2f}{row['mean_hamming']:>9.1f}"
              f"{row['frac_chromophore']:>9.2f}{row['frac_novel']:>8.2f}"
              f"{row['frac_in_domain']:>8.2f}")

    generated = [r for r in rows]
    outside   = sum(1 for r in generated if not r["in_domain"])
    if outside:
        print(f"\nWarning: {outside}/{len(generated)} generated sequences carry more than "
              f"{domain['max_mutations']} substitutions, beyond what the oracle was fit on. "
              f"Their brightness is clamped to [{domain['score_min']:.2f}, "
              f"{domain['score_max']:.2f}] and ranks them no better than 'dead'. "
              f"Tighten --mut-budget to keep comparisons inside the measured domain.")
    print(f"\nWrote 2 plots + 2 CSVs to {out_dir}/")


if __name__ == "__main__":
    main()
