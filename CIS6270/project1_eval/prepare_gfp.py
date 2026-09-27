#!/usr/bin/env python3
"""Build a Project-1 input CSV from the avGFP DMS dataset (Sarkisyan et al., METL format).

Applies each variant to the wild-type sequence and writes sequence,c,r1,r2 for
run_experiment.py, plus a disjoint remainder split used to fit the brightness
oracle in gfp_oracle.py.

Usage:
  python prepare_gfp.py                          # 4000 training variants, oracle split
  python prepare_gfp.py --n 8000 --threshold -1.0
  python prepare_gfp.py --max-mutations 8 --oracle-n 20000

Rewards (larger is better):
  r1 = brightness score (log-scale, wild-type centered)
  r2 = -num_mutations  (mutational parsimony)
"""
import argparse, csv, random, re
from collections import defaultdict
from pathlib import Path

ROOT        = Path(__file__).resolve().parent
DATA_ROOT   = ROOT.parent.parent / "Data_GFP" / "data" / "dms_data"
CHROMOPHORE = (63, 64, 65)          # 0-indexed S-Y-G in the METL avGFP numbering
FIELDS      = ["sequence", "c", "r1", "r2", "variant", "num_mutations", "score"]


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def read_wt(yml_path: Path, key: str) -> str:
    """Pull wt_aa for one dataset out of METL's datasets.yml (no yaml dependency)."""
    block = ("\n" + yml_path.read_text()).split(f"\n{key}:", 1)[-1]
    match = re.search(r"wt_aa:\s*([A-Z]+)", block)
    if not match:
        raise ValueError(f"No wt_aa entry for '{key}' in {yml_path}")
    return match.group(1)


def read_variants(tsv_path: Path) -> list[dict]:
    with tsv_path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def apply_variant(wt: str, variant: str) -> str:
    """Substitute 0-indexed mutations such as 'K1E,A108D' into the wild type."""
    seq = list(wt)
    for token in variant.split(","):
        ref, position, alt = token[0], int(token[1:-1]), token[-1]
        if wt[position] != ref:
            raise ValueError(f"{token}: wild type has {wt[position]} at index {position}")
        seq[position] = alt
    return "".join(seq)


# ══════════════════════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════════════════════

def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin over (class, mutation-count) buckets so both stay balanced."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["c"], min(int(row["num_mutations"]), 8))].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    picked, keys = [], sorted(buckets)
    while len(picked) < n and any(buckets[k] for k in keys):
        for key in keys:
            if buckets[key] and len(picked) < n:
                picked.append(buckets[key].pop())
    rng.shuffle(picked)
    return picked


def annotate(rows: list[dict], wt: str, threshold: float, max_mutations: int | None) -> list[dict]:
    out = []
    for row in rows:
        n_mut, score = int(row["num_mutations"]), float(row["score"])
        if max_mutations is not None and n_mut > max_mutations:
            continue
        out.append({
            "sequence":      apply_variant(wt, row["variant"]),
            "c":             int(score > threshold),
            "r1":            f"{score:.6f}",
            "r2":            f"{-n_mut:d}",
            "variant":       row["variant"],
            "num_mutations": n_mut,
            "score":         f"{score:.6f}",
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    bright = sum(r["c"] for r in rows)
    try:
        shown = path.relative_to(ROOT.parent)
    except ValueError:            # --outdir outside the project tree
        shown = path
    print(f"  {shown}: {len(rows)} variants "
          f"({bright} bright / {len(rows) - bright} dark)")


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-key", default="avgfp", help="Dataset name in datasets.yml")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="METL dms_data directory (holds datasets.yml)")
    parser.add_argument("--n", type=int, default=4000,
                        help="Variants for the generator training CSV (default: 4000)")
    parser.add_argument("--oracle-n", type=int, default=20000,
                        help="Variants for the disjoint oracle split (default: 20000)")
    parser.add_argument("--threshold", type=float, default=-1.0,
                        help="Brightness score above which c=1 (default: -1.0)")
    parser.add_argument("--max-mutations", type=int, default=None,
                        help="Drop variants with more mutations than this")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=Path, default=ROOT / "data",
                        help="Where to write the CSVs (default: project1_eval/data)")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    key  = args.dataset_key
    wt   = read_wt(args.data_root / "datasets.yml", key)
    rows = read_variants(args.data_root / key / f"{key}.tsv")
    print(f"\nPreparing {key}: wild type is {len(wt)} residues, "
          f"chromophore {''.join(wt[i] for i in CHROMOPHORE)} at {CHROMOPHORE}")
    print(f"  {len(rows)} measured variants in {key}.tsv")

    annotated = annotate(rows, wt, args.threshold, args.max_mutations)
    random.Random(args.seed).shuffle(annotated)
    train  = stratified_sample(annotated, args.n, args.seed)
    used   = {r["variant"] for r in train}
    oracle = [r for r in annotated if r["variant"] not in used][:args.oracle_n]

    write_csv(args.outdir / f"{key}_train_{len(train)}.csv", train)
    write_csv(args.outdir / f"{key}_oracle.csv", oracle)
    (args.outdir / f"{key}_wt.txt").write_text(wt + "\n")
    print(f"\nNext: python run_experiment.py --esm-model esm2_8m "
          f"--dataset data/{key}_train_{len(train)}.csv --max-length {len(wt)} "
          f"--min-polar 0 --mut-budget 12 --plot --ablate")


if __name__ == "__main__":
    main()
