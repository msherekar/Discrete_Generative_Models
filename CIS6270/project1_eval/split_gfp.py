#!/usr/bin/env python3
"""Partition the avGFP DMS assay into train / validation / test, stratified by
mutation count.

Why this exists. prepare_gfp.py drew its two splits with a round-robin over
buckets, which equalizes rather than stratifies: it cycles through buckets taking
one row at a time, so a small bucket drains completely into whichever split is
filled first. On avGFP that put all 1,084 single mutants into the oracle split
and left the generator's 35,000-variant training set with none, while
over-weighting the oracle toward 6-9 mutation variants. This script partitions
instead of sampling, so every split has the same mutation-count profile as the
assay and no variant is dropped.

  train  80%   fits the generator and the brightness oracle
  val    15%   selects the oracle's ridge penalty; monitors generator training
  test    5%   touched once, for the oracle's honest accuracy

Stratification is on mutation count because that is the axis the assay is
unbalanced on -- 2.1% of variants carry one mutation and 24.7% carry two -- and
because it is what the generation task is defined against (--mut-budget k). The
brightness balance follows closely enough on its own that a second stratum is
usually unnecessary; --stratify-by mutations+class adds it and the report prints
the bright fraction per split either way, so the choice is visible.

The oracle and the generator deliberately share the train split. They were
disjoint before, which is why the oracle could only score 1,551 of the 1,693
substitutions the generator had seen, and why only ~5% of generated 3-mutation
variants were inside the oracle's support. Sharing makes the two supports
identical. It is safe here because the oracle is additive over 4,503 mutation
indicators fitted on ~41,000 rows: it has no way to memorize a sequence, only an
average per-substitution effect, and its honest accuracy is measured on test.
Pass --oracle-holdout to carve the oracle its own disjoint slice instead.

Usage:
  python split_gfp.py
  python split_gfp.py --test-frac 0.05 --val-frac 0.15 --seed 7
  python split_gfp.py --stratify-by mutations+class
"""
import argparse
import collections
import csv
import json
import random
from pathlib import Path

import numpy as np

from prepare_gfp import CHROMOPHORE, FIELDS, annotate, read_variants, read_wt

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT.parent.parent / "metl" / "data" / "dms_data"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def stratum(row, mode):
    """The bucket a variant is stratified within."""
    k = int(row["num_mutations"])
    return (k,) if mode == "mutations" else (k, int(row["c"]))


def partition(rows, test_frac, val_frac, seed, mode):
    """Split rows into (train, val, test), preserving each stratum's proportions.

    A partition, not three samples: every row lands in exactly one split, so the
    splits are disjoint by construction rather than by a post-hoc check, and
    nothing is left over.

    Strata too small to fill a slice give it up rather than borrowing from
    another stratum -- with 2 variants at 15 mutations a 5% test slice rounds to
    zero, and inventing one would misreport the test set's composition. Those
    rows go to train, and the report names every stratum it happened to.
    """
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[stratum(row, mode)].append(row)
    rng = random.Random(seed)
    train, val, test, undersized = [], [], [], []
    for key in sorted(buckets):
        bucket = buckets[key][:]
        rng.shuffle(bucket)
        n = len(bucket)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if n_test == 0 or n_val == 0:
            undersized.append((key, n, n_test, n_val))
        test += bucket[:n_test]
        val += bucket[n_test:n_test + n_val]
        train += bucket[n_test + n_val:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test, undersized


def substitution_support(rows, wt):
    """The distinct substitutions a split contains, as a (length, 20) bool mask.

    This is what an additive oracle fitted on the split can score: a substitution
    absent here has no column to fit, so its coefficient stays zero and any
    sequence carrying it silently falls back to the intercept.
    """
    index = {a: i for i, a in enumerate(AMINO_ACIDS)}
    mask = np.zeros((len(wt), len(AMINO_ACIDS)), dtype=bool)
    for row in rows:
        for token in row["variant"].split(","):
            if token:
                position, alt = int(token[1:-1]), token[-1]
                if alt in index:
                    mask[position, index[alt]] = True
    return mask


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def report(splits, rows, wt, undersized):
    """Print everything needed to audit the split, and return it as a dict."""
    total = len(rows)
    full = collections.Counter(int(r["num_mutations"]) for r in rows)
    n_possible = len(wt) * (len(AMINO_ACIDS) - 1)

    print(f"\n{'':<9}{'rows':>8}{'share':>8}{'bright':>9}{'subs':>8}"
          f"{'coverage':>10}{'mean score':>12}")
    print("-" * 64)
    summary = {}
    for name, split in splits.items():
        support = substitution_support(split, wt)
        scores = np.array([float(r["score"]) for r in split])
        bright = np.mean([int(r["c"]) for r in split])
        summary[name] = {"rows": len(split), "share": len(split) / total,
                         "bright_fraction": float(bright),
                         "substitutions": int(support.sum()),
                         "coverage": float(support.sum() / n_possible),
                         "mean_score": float(scores.mean())}
        print(f"{name:<9}{len(split):>8}{100*len(split)/total:>7.1f}%"
              f"{100*bright:>8.1f}%{int(support.sum()):>8}"
              f"{100*support.sum()/n_possible:>9.1f}%{scores.mean():>+12.4f}")
    print(f"{'assay':<9}{total:>8}{100.0:>7.1f}%"
          f"{100*np.mean([int(r['c']) for r in rows]):>8.1f}%"
          f"{int(substitution_support(rows, wt).sum()):>8}"
          f"{100*substitution_support(rows, wt).sum()/n_possible:>9.1f}%"
          f"{np.mean([float(r['score']) for r in rows]):>+12.4f}")

    print("\nmutation-count composition (% of each split)")
    print(f"{'k':>3}{'assay':>9}" + "".join(f"{n:>9}" for n in splits))
    counts = {n: collections.Counter(int(r["num_mutations"]) for r in s)
              for n, s in splits.items()}
    worst = 0.0
    for k in sorted(full):
        line = f"{k:>3}{100*full[k]/total:>8.1f}%"
        for name, split in splits.items():
            share = 100 * counts[name][k] / len(split)
            worst = max(worst, abs(share - 100 * full[k] / total))
            line += f"{share:>8.1f}%"
        print(line)
    print(f"\nlargest deviation from the assay's profile: {worst:.2f} percentage points")

    # Disjointness is structural, but an assertion costs nothing and a silent
    # leak would invalidate every number downstream.
    keys = {n: {r["variant"] for r in s} for n, s in splits.items()}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = keys[a] & keys[b]
            assert not shared, f"{a} and {b} share {len(shared)} variants"
        assert len(keys[a]) == len(splits[a]), f"{a} has duplicate variants"
    print(f"splits are disjoint and cover {sum(len(s) for s in splits.values())} "
          f"of {total} variants")

    # The generator can only be asked to produce what the oracle can score.
    train_support = substitution_support(splits["train"], wt)
    for name in ("val", "test"):
        other = substitution_support(splits[name], wt)
        unseen = int((other & ~train_support).sum())
        print(f"  {name}: {unseen} substitutions absent from train "
              f"({100*unseen/max(1, other.sum()):.1f}% of its support) — an "
              f"additive oracle scores these by intercept alone")
    if undersized:
        print("\nstrata too small to fill every slice (their rows went to train):")
        for key, n, n_test, n_val in undersized:
            print(f"  stratum {key}: {n} variants -> test {n_test}, val {n_val}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-key", default="avgfp")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="METL dms_data directory holding datasets.yml")
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--stratify-by", default="mutations",
                        choices=("mutations", "mutations+class"),
                        help="Stratum definition (default: mutations). The report "
                             "prints the bright fraction per split either way, so the "
                             "cost of the simpler choice is visible.")
    parser.add_argument("--threshold", type=float, default=-1.0,
                        help="Brightness score above which c=1 (default: -1.0)")
    parser.add_argument("--max-mutations", type=int, default=None)
    parser.add_argument("--oracle-holdout", type=float, default=0.0, metavar="FRAC",
                        help="Carve this fraction of train into a disjoint oracle "
                             "split (default: 0 = oracle and generator share train). "
                             "Disjoint splits are what left only ~5%% of generated "
                             "3-mutation variants inside the oracle's support.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=Path, default=ROOT / "data")
    return parser.parse_args()


def main():
    args = parse_args()
    key = args.dataset_key
    wt = read_wt(args.data_root / "datasets.yml", key)
    raw = read_variants(args.data_root / key / f"{key}.tsv")
    rows = annotate(raw, wt, args.threshold, args.max_mutations)
    print(f"\nSplitting {key}: wild type {len(wt)} residues, chromophore "
          f"{''.join(wt[i] for i in CHROMOPHORE)} at {CHROMOPHORE}")
    print(f"  {len(raw)} measured variants, {len(rows)} after filtering")
    print(f"  strata: {args.stratify_by}   seed: {args.seed}")

    train, val, test, undersized = partition(
        rows, args.test_frac, args.val_frac, args.seed, args.stratify_by)
    splits = {"train": train, "val": val, "test": test}
    if args.oracle_holdout > 0:
        cut = int(round(len(train) * args.oracle_holdout))
        splits = {"train": train[cut:], "oracle": train[:cut],
                  "val": val, "test": test}

    summary = report(splits, rows, wt, undersized)

    print()
    for name, split in splits.items():
        path = args.outdir / f"{key}_{name}.csv"
        write_csv(path, split)
        print(f"  wrote {path.relative_to(ROOT)} ({len(split)} variants)")
    (args.outdir / f"{key}_wt.txt").write_text(wt + "\n")

    manifest = {"dataset": key, "wt_length": len(wt), "seed": args.seed,
                "stratify_by": args.stratify_by, "threshold": args.threshold,
                "test_frac": args.test_frac, "val_frac": args.val_frac,
                "oracle_holdout": args.oracle_holdout,
                "source": str(args.data_root / key / f"{key}.tsv"),
                "splits": summary}
    path = args.outdir / f"{key}_splits.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  wrote {path.relative_to(ROOT)}")
    print(f"\nNext: fit the oracle on train, select on val, report on test:\n"
          f"  python gfp_oracle.py --fit --train data/{key}_train.csv "
          f"--val data/{key}_val.csv --test data/{key}_test.csv")


if __name__ == "__main__":
    main()
