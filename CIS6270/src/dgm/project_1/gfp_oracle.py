#!/usr/bin/env python3
"""Held-out brightness oracle for the avGFP DMS study.

Ridge regression on sparse mutation indicators (position, amino acid) relative to
the wild type — the standard DMS baseline. Fit on the oracle split written by
prepare_gfp.py, which is disjoint from the generator's training variants, so
oracle scores of generated sequences are not self-evaluation.

Usage:
  dgm-gfp-oracle --fit                       # fit + report held-out Spearman
  dgm-gfp-oracle --score some.fasta          # score sequences with the saved oracle

Import:
  from dgm.project1.gfp_oracle import load_oracle, score_sequences
"""
import argparse, csv
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from dgm.common.paths import project_dir

ROOT        = project_dir()
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


def support_of(sequences, wt):
    """The (position, residue) substitutions present in a set of sequences.

    An additive oracle can only score what it was fitted on: a substitution with
    no column in the design matrix keeps a zero coefficient, so a sequence
    carrying it falls back to the intercept. Recording the support explicitly is
    more reliable than recovering it from the fitted coefficients afterwards --
    ridge on a sparse design leaves small nonzero values for columns it never
    really saw, so the two disagree by about a hundred entries on avGFP.
    """
    mask = np.zeros((len(wt), len(AMINO_ACIDS)), dtype=bool)
    for seq in sequences:
        for position, (ref, alt) in enumerate(zip(wt, seq)):
            if alt != ref and alt in AA_INDEX:
                mask[position, AA_INDEX[alt]] = True
    return mask


def evaluate(model, sequences, scores, wt, support):
    """Spearman and MAE overall, and again over only the in-support sequences.

    The restricted number is the one that describes this oracle in use. A
    sequence with any unfitted substitution is scored partly by the intercept, so
    including those measures how often the assay happened to cover a variant, not
    how well the oracle ranks brightness.
    """
    predicted = model.predict(featurize(sequences, wt))
    inside = np.array([all(support[p, AA_INDEX[b]]
                           for p, (a, b) in enumerate(zip(wt, s))
                           if a != b and b in AA_INDEX)
                       for s in sequences])
    out = {"n": len(sequences),
           "rho": float(spearmanr(predicted, scores).statistic),
           "mae": float(np.abs(predicted - scores).mean()),
           "in_support": float(inside.mean())}
    if inside.sum() > 2:
        out["rho_in_support"] = float(spearmanr(predicted[inside], scores[inside]).statistic)
        out["mae_in_support"] = float(np.abs(predicted[inside] - scores[inside]).mean())
    return out


def fit_on_splits(train, val, test, wt, alphas=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0)):
    """Fit on train, choose the ridge penalty on val, report once on test.

    The model is never refitted on train+val. Refitting would move the support
    away from the train split that the generator is also constrained to, and the
    extra 15% of rows buys little for a model with 4,503 parameters and 41,372
    observations. Test is read exactly once, for the numbers in the report.
    """
    train_x = featurize(train[0], wt)
    support = support_of(train[0], wt)
    print(f"  selecting the ridge penalty on {len(val[0])} validation variants")
    tried = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(train_x, train[1])
        tried.append((model, evaluate(model, val[0], val[1], wt, support), alpha))
    best = max(tried, key=lambda t: t[1]["rho"])
    for candidate in tried:
        _, stats, alpha = candidate
        print(f"    alpha {alpha:>5}: val Spearman {stats['rho']:.4f}  "
              f"MAE {stats['mae']:.4f}"
              + ("  <-- chosen" if candidate is best else ""))
    model, val_stats, alpha = best
    test_stats = evaluate(model, test[0], test[1], wt, support)
    distances = np.array([sum(a != b for a, b in zip(wt, s)) for s in train[0]])
    domain = {"score_min": float(train[1].min()), "score_max": float(train[1].max()),
              "max_mutations": int(distances.max())}
    return model, support, alpha, {"val": val_stats, "test": test_stats}, domain


def save_oracle(path: Path, model: Ridge, wt: str, rho: float, mae: float,
                domain: dict, support=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if support is None else {"support": support.astype(bool)}
    np.savez(path, coef=model.coef_.astype(np.float32),
             intercept=np.float32(model.intercept_), wt=wt, rho=rho, mae=mae,
             score_min=domain["score_min"], score_max=domain["score_max"],
             max_mutations=domain["max_mutations"], **extra)


def load_oracle(path: Path = ROOT / "data" / "avgfp_oracle.npz"):
    """Return (coef, intercept, wt, domain) for the fitted oracle."""
    saved = np.load(path, allow_pickle=False)
    domain = {"score_min": float(saved["score_min"]), "score_max": float(saved["score_max"]),
              "max_mutations": int(saved["max_mutations"]),
              "support": saved["support"] if "support" in saved.files else None}
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
    """True where the oracle can actually score a sequence.

    Two conditions, both necessary. The mutation count must be within the range
    the oracle saw, and EVERY substitution must be one the assay measured: an
    indicator oracle has a zero coefficient for anything else, so an unmeasured
    substitution contributes nothing and the prediction silently falls back to
    the intercept. Checking only the count reports such sequences as in-domain
    when the oracle is really returning a constant for them.
    """
    coef, _, wt, domain = oracle if oracle is not None else load_oracle()
    recorded = domain.get("support")
    if recorded is not None:
        # The support the oracle was actually fitted on, saved at fit time.
        support = {(p, AMINO_ACIDS[a]) for p in range(len(wt))
                   for a in range(len(AMINO_ACIDS)) if recorded[p, a]}
    else:
        # Older oracles predate the recorded support; recover it from the
        # coefficients, which is approximate (see support_of).
        support = {(p, AMINO_ACIDS[a])
                   for p in range(len(wt)) for a in range(len(AMINO_ACIDS))
                   if abs(float(coef[p * len(AMINO_ACIDS) + a])) > 1e-9}
    out = []
    for s in sequences:
        subs = [(p, b) for p, (a, b) in enumerate(zip(wt, s)) if a != b]
        out.append(len(subs) <= domain["max_mutations"]
                   and all(x in support for x in subs))
    return np.asarray(out)


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=None,
                        help="Training CSV from split_gfp.py (default: "
                             "data/avgfp_train.csv when it exists). Given --train, the "
                             "penalty is chosen on --val and reported on --test.")
    parser.add_argument("--val", type=Path, default=None,
                        help="Validation CSV; selects the ridge penalty.")
    parser.add_argument("--test", type=Path, default=None,
                        help="Test CSV; read once, for the reported accuracy.")
    parser.add_argument("--split", type=Path, default=None,
                        help="Legacy single-CSV mode: fit on this file with an internal "
                             "random 80/20 split. Superseded by --train/--val/--test, "
                             "which reuse the stratified partition.")
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
    default_train = ROOT / "data" / "avgfp_train.csv"
    if args.fit and args.split is None and args.train is None and default_train.is_file():
        args.train = default_train
        args.val = args.val or ROOT / "data" / "avgfp_val.csv"
        args.test = args.test or ROOT / "data" / "avgfp_test.csv"
    if args.fit and args.train is not None:
        for name, path in (("--val", args.val), ("--test", args.test)):
            if path is None or not path.is_file():
                raise SystemExit(f"{name} is required with --train (got {path}). "
                                 f"Build the splits with: dgm-split-gfp")
        wt = (args.wt or args.train.parent / "avgfp_wt.txt").read_text().strip()
        train, val, test = (read_split(p) for p in (args.train, args.val, args.test))
        print(f"\nFitting the oracle on {len(train[0])} training variants "
              f"({len(wt)} positions x 20 = {len(wt)*20} features)")
        model, support, alpha, stats, domain = fit_on_splits(train, val, test, wt)
        save_oracle(args.oracle, model, wt, stats["val"]["rho"],
                    stats["val"]["mae"], domain, support)
        n_possible = len(wt) * (len(AMINO_ACIDS) - 1)
        print(f"\n  chosen penalty alpha = {alpha}")
        print(f"  support: {int(support.sum())} of {n_possible} substitutions "
              f"({100*support.sum()/n_possible:.1f}%)")
        for name in ("val", "test"):
            s = stats[name]
            line = (f"  {name:<5} n={s['n']:<6} Spearman {s['rho']:.4f}  "
                    f"MAE {s['mae']:.4f}  in-support {100*s['in_support']:.1f}%")
            if "rho_in_support" in s:
                line += (f"  |  in-support only: Spearman {s['rho_in_support']:.4f}  "
                         f"MAE {s['mae_in_support']:.4f}")
            print(line)
        print(f"  domain: score in [{domain['score_min']:.2f}, {domain['score_max']:.2f}], "
              f"up to {domain['max_mutations']} substitutions from wild type")
        print(f"  saved to {args.oracle}")
    elif args.fit:
        if args.split is None:
            raise SystemExit("--fit needs either --train/--val/--test or --split")
        sequences, scores = read_split(args.split)
        wt = (args.wt or args.split.parent / "avgfp_wt.txt").read_text().strip()
        model, rho, mae, domain = fit_oracle(sequences, scores, wt, args.alpha)
        save_oracle(args.oracle, model, wt, rho, mae, domain,
                    support_of(sequences, wt))
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
