#!/usr/bin/env python3
"""GFP-specific evaluation of generated sequences: oracle brightness, distance to
wild type, chromophore preservation, and novelty.

Replaces the composition proxies in evaluate.py with function-grounded metrics for
the avGFP study. Reads the FASTA files written by run_experiment.py.

Usage:
  dgm-gfp-metrics --run-dir outputs/esm2_8m_avgfp_train_4000
  dgm-gfp-metrics --run-dir outputs/... --train data/avgfp_train_4000.csv
"""
import argparse, csv, random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .gfp_oracle import in_domain, load_fasta, load_oracle, score_sequences

from dgm.project1 import embedding_oracle as embo

from dgm.common.paths import project_dir

ROOT        = project_dir()
METHODS     = ("flow", "diffusion")
MODES       = ("cfg", "single", "multi")
CHROMOPHORE = (63, 64, 65)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def hamming(sequences: list[str], wt: str) -> np.ndarray:
    return np.array([sum(a != b for a, b in zip(seq, wt)) for seq in sequences])


def chromophore_intact(sequences: list[str], wt: str) -> np.ndarray:
    target = "".join(wt[i] for i in CHROMOPHORE)
    return np.array([float("".join(seq[i] for i in CHROMOPHORE) == target) for seq in sequences])


def oracle_support(oracle) -> set[tuple[int, str]]:
    """(position, residue) substitutions the oracle was actually fit on.

    A ridge oracle on mutation indicators has a nonzero coefficient only where it
    saw data; everywhere else it contributes nothing and the prediction falls back
    to the intercept. This set is what "observed" means for that oracle.
    """
    coef, _, wt, _ = oracle
    return {(p, AMINO_ACIDS[a]) for p in range(len(wt)) for a in range(len(AMINO_ACIDS))
            if abs(float(coef[p * len(AMINO_ACIDS) + a])) > 1e-9}


def random_baseline(wt: str, counts, support: set[tuple[int, str]],
                    observed: bool, seed: int = 0) -> list[str]:
    """Random variants matched to `counts` substitutions each.

    `observed` draws only substitutions the oracle has seen; otherwise it draws
    only substitutions it has never seen. This mirrors the Observed AA and
    Unobserved AA design settings in the METL study, whose random variants exist
    to show that a design's score reflects the model rather than chance. If a
    guided sample scores no better than its matched random baseline, the oracle
    is not measuring the thing the guidance was supposed to improve.
    """
    rng = random.Random(seed)
    by_position: dict[int, list[str]] = {}
    for p in range(len(wt)):
        options = [a for a in AMINO_ACIDS
                   if a != wt[p] and (((p, a) in support) == observed)]
        if options:
            by_position[p] = options
    if not by_position:
        return []
    positions = sorted(by_position)
    out = []
    for k in counts:
        k = max(1, min(int(k), len(positions)))
        seq = list(wt)
        for p in rng.sample(positions, k):
            seq[p] = rng.choice(by_position[p])
        out.append("".join(seq))
    return out


def score_set(sequences: list[str], wt: str, oracle, known: set[str],
              embedding=None) -> list[dict]:
    brightness = score_sequences(sequences, oracle)
    distance   = hamming(sequences, wt)
    intact     = chromophore_intact(sequences, wt)
    inside     = in_domain(sequences, oracle)
    # The indicator oracle returns its intercept for any substitution the assay
    # never measured, so it cannot rank novel designs. The embedding oracle can.
    emb = (embo.score_sequences(list(sequences), embedding)
           if embedding is not None else [None] * len(sequences))
    out = []
    for s, b, d, c, k, e in zip(sequences, brightness, distance, intact, inside, emb):
        # The wild type is absent from the training set -- every measured variant
        # carries at least one mutation -- so a plain set-membership test calls it
        # novel. Returning the unmutated wild type is not a design.
        row = {"oracle_brightness": round(float(b), 4), "hamming_to_wt": int(d),
               "chromophore_intact": int(c),
               "novel": int(d > 0 and s not in known),
               "is_wildtype": int(d == 0),
               "in_domain": int(k)}
        if e is not None:
            row["embedding_brightness"] = round(float(e), 4)
        row["sequence"] = s
        out.append(row)
    return out


def training_set_from_run(run_dir: Path) -> Path | None:
    """The dataset a run was actually trained on, as recorded in results.pt."""
    import torch
    for method in METHODS:
        results = run_dir / method / "results.pt"
        if results.is_file():
            config = torch.load(results, weights_only=False, map_location="cpu").get("config", {})
            if config.get("dataset"):
                return Path(config["dataset"])
    return None


def arms_in(run_dir: Path, method: str) -> list[str]:
    """Every arm a run actually wrote, base modes first.

    MODES is only the three arms every run has. A sweep adds more -- 'cfg@0',
    'single@100', 'lam0.5', 'spp99' -- and scoring just the fixed three silently
    drops them: a run with 26 arms reported 3, which hides the very comparison
    the sweep was for. So read the directory instead, and keep MODES at the front
    so the ordering callers expect still holds.
    """
    found = sorted(p.stem for p in (run_dir / method).glob("*.fasta"))
    return [m for m in MODES if m in found] + [m for m in found if m not in MODES]


def collect(run_dir: Path, wt: str, oracle, known: set[str],
            embedding=None) -> list[dict]:
    rows = []
    for method in METHODS:
        if not (run_dir / method).is_dir():
            continue
        for mode in arms_in(run_dir, method):
            fasta = run_dir / method / f"{mode}.fasta"
            if not fasta.exists():
                continue
            for row in score_set(load_fasta(fasta), wt, oracle, known, embedding):
                rows.append({"method": method, "mode": mode, **row})
    if not rows:
        raise FileNotFoundError(f"No FASTA files under {run_dir}/{{flow,diffusion}}/")
    return rows


def summarize(rows: list[dict], train_rows: list[dict], threshold: float) -> list[dict]:
    seen: list[tuple[str, str]] = []
    for r in rows:
        if (r["method"], r["mode"]) not in seen:
            seen.append((r["method"], r["mode"]))
    summary = []
    for group, label in [(train_rows, ("training", "data"))] + [
        ([r for r in rows if (r["method"], r["mode"]) == key], key) for key in seen
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
            "frac_wildtype":        round(float(np.mean([r["is_wildtype"] for r in group])), 3),
            "frac_in_domain":       round(float(np.mean([r["in_domain"] for r in group])), 3),
            **({"mean_embedding_brightness":
                round(float(np.mean([r["embedding_brightness"] for r in group])), 4)}
               if "embedding_brightness" in group[0] else {}),
        })
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_brightness(rows, train_rows, out_dir: Path, prefix: str, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [("training\ndata", [r["oracle_brightness"] for r in train_rows])]
    keys = list(dict.fromkeys((r["method"], r["mode"]) for r in rows))
    groups += [(f"{m[:4]}\n{g}", [r["oracle_brightness"] for r in rows
                                  if (r["method"], r["mode"]) == (m, g)])
               for m, g in keys]
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
    markers = {"flow": "o", "diffusion": "^", "random": "x"}
    colors  = {"cfg": "tab:blue", "single": "tab:orange", "multi": "tab:green",
               "observed": "tab:red", "unobserved": "tab:purple"}
    for method, mode in dict.fromkeys((r["method"], r["mode"]) for r in rows):
            marker = markers.get(method, "s")
            color  = colors.get(mode, "tab:gray")
            group = [r for r in rows if (r["method"], r["mode"]) == (method, mode)]
            if group:
                edge = {} if marker in "x+" else {"edgecolors": "k", "linewidths": .4}
                ax.scatter([r["hamming_to_wt"] for r in group],
                           [r["oracle_brightness"] for r in group],
                           marker=marker, c=color, s=45, **edge,
                           label=f"{method}/{mode}")
    ax.set_xlabel("Hamming distance to wild type"); ax.set_ylabel("oracle brightness")
    ax.set_title("Brightness vs. mutational parsimony")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_gfp_pareto.png", dpi=150); plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader(); writer.writerows(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run output directory holding flow/ and diffusion/")
    parser.add_argument("--train", type=Path, default=None,
                        help="Training CSV from prepare_gfp.py (novelty + reference "
                             "cloud). Defaults to whatever the run itself recorded in "
                             "results.pt, so it cannot silently disagree with the model.")
    parser.add_argument("--oracle", type=Path, default=ROOT / "data" / "avgfp_oracle.npz",
                        help="Fitted oracle from gfp_oracle.py --fit")
    parser.add_argument("--embedding-oracle", type=Path, default=None, metavar="NPZ",
                        help="Second oracle from embedding_oracle.py (e.g. "
                             "data/avgfp_metl_oracle.npz). Unlike the indicator oracle "
                             "it can rank substitutions the assay never measured.")
    parser.add_argument("--baseline-n", type=int, default=0, metavar="N",
                        help="Add N random variants per setting (observed / unobserved "
                             "substitutions), matched to the generated Hamming distances. "
                             "0 disables. This is the control that shows whether a guided "
                             "sample beats chance at the same mutational distance.")
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
    embedding = embo.load_oracle(args.embedding_oracle) if args.embedding_oracle else None
    if embedding is not None:
        print(f"  Second oracle: {embedding[3]} backend, from {args.embedding_oracle.name}")

    train_path = args.train or training_set_from_run(args.run_dir)
    if train_path is None:
        raise SystemExit(
            f"{args.run_dir} records no training set, so novelty and the "
            f"reference cloud cannot be computed; pass --train")
    if not train_path.is_file():
        raise SystemExit(
            f"the training set this run recorded is missing: {train_path}\n"
            f"It has moved or was never on this machine; pass --train with its "
            f"current location.")
    print(f"  Training set: {train_path.name}")

    with train_path.open(newline="") as f:
        train_seqs = [r["sequence"].strip().upper() for r in csv.DictReader(f)]
    known      = set(train_seqs)
    train_rows = score_set(train_seqs, wt, oracle, known, embedding)
    rows       = collect(args.run_dir, wt, oracle, known, embedding)

    if args.baseline_n:
        support = oracle_support(oracle)
        counts  = [r["hamming_to_wt"] for r in rows] or [4]
        matched = [counts[i % len(counts)] for i in range(args.baseline_n)]
        for observed, mode in ((True, "observed"), (False, "unobserved")):
            seqs = random_baseline(wt, matched, support, observed, seed=7)
            if not seqs:
                print(f"  [skip] no {mode} substitutions available for the baseline")
                continue
            for row in score_set(seqs, wt, oracle, known, embedding):
                rows.append({"method": "random", "mode": mode, **row})
        print(f"  Added {args.baseline_n} random variants per setting, "
              f"Hamming-matched to the generated samples")

    out_dir = args.outdir or ROOT / "plots" / args.run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix  = args.run_dir.name
    summary = summarize(rows, train_rows, args.threshold)
    write_csv(out_dir / f"{prefix}_gfp_sequences.csv", rows)
    write_csv(out_dir / f"{prefix}_gfp_summary.csv", summary)
    plot_brightness(rows, train_rows, out_dir, prefix, args.threshold)
    plot_pareto(rows, train_rows, out_dir, prefix)

    has_emb = any("mean_embedding_brightness" in r for r in summary)
    header = (f"{'method':<10}{'mode':<9}{'n':>4}{'indicator':>11}"
              + (f"{'embedding':>11}" if has_emb else "")
              + f"{'%bright':>9}{'hamming':>9}{'%chromo':>9}{'%novel':>8}"
                f"{'%wt':>6}{'%indom':>8}")
    print("\n" + header + "\n" + "-" * len(header))
    for row in summary:
        line = (f"{row['method']:<10}{row['mode']:<9}{row['n']:>4}"
                f"{row['mean_brightness']:>11.3f}")
        if has_emb:
            line += f"{row.get('mean_embedding_brightness', float('nan')):>11.3f}"
        line += (f"{row['frac_bright']:>9.2f}{row['mean_hamming']:>9.1f}"
                 f"{row['frac_chromophore']:>9.2f}{row['frac_novel']:>8.2f}"
                 f"{row['frac_wildtype']:>6.2f}{row['frac_in_domain']:>8.2f}")
        print(line)

    generated = [r for r in rows if r["method"] != "random"]
    wildtype  = sum(r["is_wildtype"] for r in generated)
    if wildtype:
        print(f"\nNote: {wildtype}/{len(generated)} generated samples are the unmutated "
              f"wild type. They score near 0 by construction and inflate every mean. "
              f"Use --exact-mutations in run_experiment.py to remove them.")
    outside   = sum(1 for r in generated if not r["in_domain"])
    if outside:
        # in_domain fails on either of two conditions, and they need different
        # fixes, so say which one actually fired. Reporting the count when the
        # support check is what failed sends you to --mut-budget, which cannot
        # help: a run at exactly 5 substitutions is nowhere near the 15 limit.
        too_many = sum(1 for r in generated
                       if r["hamming_to_wt"] > domain["max_mutations"])
        unmeasured = outside - too_many
        print(f"\nWarning: {outside}/{len(generated)} generated sequences fall outside "
              f"the oracle's domain. Their brightness is clamped to "
              f"[{domain['score_min']:.2f}, {domain['score_max']:.2f}] and ranks them "
              f"no better than 'dead'.")
        if too_many:
            print(f"  {too_many} carry more than {domain['max_mutations']} "
                  f"substitutions. Tighten --mut-budget.")
        if unmeasured:
            print(f"  {unmeasured} contain a substitution the assay never measured. "
                  f"--mut-budget will not help. Either decode with "
                  f"--restrict-support, or check that --oracle here is the SAME "
                  f"file the run restricted to: an oracle saved without a recorded "
                  f"support recovers it from its nonzero coefficients, which is "
                  f"approximate and reports in-domain sequences as outside it.")
    print(f"\nWrote 2 plots + 2 CSVs to {out_dir}/")


if __name__ == "__main__":
    main()
