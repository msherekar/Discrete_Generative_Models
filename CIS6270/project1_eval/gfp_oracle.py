#!/usr/bin/env python3
"""Held-out brightness oracle for the avGFP DMS study.

Ridge regression on sparse mutation indicators (position, amino acid) relative to
the wild type — the standard DMS baseline. Fit on the oracle split written by
prepare_gfp.py, which is disjoint from the generator's training variants, so
oracle scores of generated sequences are not self-evaluation.

Usage:
  python gfp_oracle.py --fit                       # fit + report held-out Spearman
  python gfp_oracle.py --score some.fasta          # score sequences with the saved oracle

Import:
  from gfp_oracle import load_oracle, score_sequences
"""
import argparse, csv
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT        = Path(__file__).resolve().parent
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX    = {a: i for i, a in enumerate(AMINO_ACIDS)}


# ══════════════════════════════════════════════════════════════════════════════
# Featurization
# ══════════════════════════════════════════════════════════════════════════════

def featurize(sequences: list[str], wt: str) -> csr_matrix:
    """One indicator per (position, substituted residue) that differs from wt."""
    n_features = len(wt) * len(AMINO_ACIDS)
    indices, indptr = [], [0]
    for seq in sequences:
        for position, (ref, alt) in enumerate(zip(wt, seq)):
            if alt != ref and alt in AA_INDEX:
                indices.append(position * len(AMINO_ACIDS) + AA_INDEX[alt])
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    return csr_matrix((data, indices, indptr), shape=(len(sequences), n_features))


def read_split(csv_path: Path) -> tuple[list[str], np.ndarray]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    sequences = [r["sequence"].strip().upper() for r in rows]
    scores    = np.array([float(r["score"]) for r in rows], dtype=np.float32)
    return sequences, scores


def load_fasta(path: Path) -> list[str]:
    sequences, current = [], ""
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current:
                sequences.append(current)
            current = ""
        else:
            current += line.strip()
    return sequences + ([current] if current else [])


# ══════════════════════════════════════════════════════════════════════════════
# Fit / predict
# ══════════════════════════════════════════════════════════════════════════════

def fit_oracle(sequences, scores, wt, alpha=1.0, val_fraction=0.2, seed=7):
    """Fit ridge on a random subset.

    Returns (model, Spearman, MAE, domain) where `domain` records the range the
    oracle was actually fit on: the observed score bounds and the largest
    mutation count seen. Ridge on mutation indicators is linear and unbounded,
    so predictions far outside that domain are extrapolation, not measurement.
    """
    rng     = np.random.default_rng(seed)
    order   = rng.permutation(len(sequences))
    n_val   = int(len(order) * val_fraction)
    val, fit = order[:n_val], order[n_val:]
    features = featurize(sequences, wt)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(features[fit], scores[fit])
    predicted = model.predict(features[val])
    rho = spearmanr(predicted, scores[val]).statistic
    mae = float(np.abs(predicted - scores[val]).mean())
    distances = np.asarray([sum(a != b for a, b in zip(wt, s)) for s in sequences])
    domain = {"score_min": float(scores.min()), "score_max": float(scores.max()),
              "max_mutations": int(distances.max())}
    return model, rho, mae, domain


def save_oracle(path: Path, model: Ridge, wt: str, rho: float, mae: float,
                domain: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, coef=model.coef_.astype(np.float32),
             intercept=np.float32(model.intercept_), wt=wt, rho=rho, mae=mae,
             score_min=domain["score_min"], score_max=domain["score_max"],
             max_mutations=domain["max_mutations"])


def load_oracle(path: Path = ROOT / "data" / "avgfp_oracle.npz"):
    """Return (coef, intercept, wt, domain) for the fitted oracle."""
    saved = np.load(path, allow_pickle=False)
    domain = {"score_min": float(saved["score_min"]), "score_max": float(saved["score_max"]),
              "max_mutations": int(saved["max_mutations"])}
    return saved["coef"], float(saved["intercept"]), str(saved["wt"]), domain


def score_sequences(sequences: list[str], oracle=None, clip: bool = True) -> np.ndarray:
    """Predicted wild-type-centered brightness for each sequence.

    Ridge on mutation indicators extrapolates without bound: a sequence 100
    substitutions from the wild type scores far below the dimmest variant ever
    measured. With `clip` (the default) predictions are held to the observed
    score range, so a dead sequence reads as dead rather than as an arbitrarily
    large negative number. Use `in_domain()` to tell real ranking from clamping.
    """
    coef, intercept, wt, domain = oracle if oracle is not None else load_oracle()
    predicted = np.asarray(featurize(sequences, wt) @ coef + intercept, dtype=np.float64)
    if clip:
        predicted = predicted.clip(domain["score_min"], domain["score_max"])
    return predicted


def in_domain(sequences: list[str], oracle=None) -> np.ndarray:
    """True where a sequence is within the mutation count the oracle was fit on."""
    _, _, wt, domain = oracle if oracle is not None else load_oracle()
    return np.asarray([sum(a != b for a, b in zip(wt, s)) <= domain["max_mutations"]
                       for s in sequences])


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, default=ROOT / "data" / "avgfp_oracle.csv",
                        help="Oracle-split CSV from prepare_gfp.py")
    parser.add_argument("--oracle", type=Path, default=ROOT / "data" / "avgfp_oracle.npz",
                        help="Where the fitted oracle is saved / loaded")
    parser.add_argument("--wt", type=Path, default=None,
                        help="Wild-type sequence file (default: <split dir>/avgfp_wt.txt)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge penalty")
    parser.add_argument("--fit", action="store_true", help="Fit and save the oracle")
    parser.add_argument("--score", type=Path, default=None, help="FASTA to score")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    if args.fit:
        sequences, scores = read_split(args.split)
        wt = (args.wt or args.split.parent / "avgfp_wt.txt").read_text().strip()
        model, rho, mae, domain = fit_oracle(sequences, scores, wt, args.alpha)
        save_oracle(args.oracle, model, wt, rho, mae, domain)
        print(f"Oracle fit on {len(sequences)} held-out variants "
              f"({len(wt)} positions x 20 features)")
        print(f"  validation Spearman {rho:.3f} | MAE {mae:.3f} | saved to {args.oracle}")
        print(f"  domain: score in [{domain['score_min']:.2f}, {domain['score_max']:.2f}], "
              f"up to {domain['max_mutations']} substitutions from wild type")
    if args.score is not None:
        sequences = load_fasta(args.score)
        oracle    = load_oracle(args.oracle)
        predicted = score_sequences(sequences, oracle)
        inside    = in_domain(sequences, oracle)
        for sequence, value, ok in zip(sequences, predicted, inside):
            print(f"  {value:+.3f}{'' if ok else '  [extrapolated]'}  {sequence[:40]}...")
        print(f"mean predicted brightness {predicted.mean():+.3f} "
              f"({int(inside.sum())}/{len(inside)} within the oracle's domain)")
    if not args.fit and args.score is None:
        print("Nothing to do: pass --fit and/or --score")


if __name__ == "__main__":
    main()
