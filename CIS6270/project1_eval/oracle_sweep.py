#!/usr/bin/env python3
"""Compare candidate brightness oracles on two tasks with different difficulty.

Every oracle here is the same ridge regression; only the representation changes.
The comparison is deliberately narrow: which representation lets a linear model
rank variants the deep mutational scan never measured?

Three tasks, each using METL's own published splits so the partition is not ours
to get wrong:

  in-support    splits/standard/...  37,232 train / 4,655 test. 98% of test
                                     substitutions also appear in train.
                                     Ordinary interpolation.
  unseen-subs   splits/mutation/...  21,670 train / 791 test. 100% of test
                                     substitutions are absent from train.
                                     Generalization across amino acid types.
  unseen-pos    splits/position/...  22,975 train / 655 test. 100% of test
                                     POSITIONS are absent from train. The
                                     hardest setting; the METL paper reports a
                                     one-hot baseline of 0.001 here.

Each task fits on its own train file and evaluates on its own test file, so the
three are independent experiments. The splits overlap EACH OTHER (they are
separate partitions of the same 51,714 rows, ~72% cross-contaminated), which is
exactly why a single run never mixes them. standard/val and standard/stest
(9,827 variants) are touched by nothing here.

Usage:
  python oracle_sweep.py                         # all tasks, all default models
  python oracle_sweep.py --models metl --append  # one model at a time, resumable
  python oracle_sweep.py --train-n 20000         # cap training if memory is tight
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gfp_oracle import featurize                      # noqa: E402
from embedding_oracle import ESM_HF, encode_metl, encode_esm, seq_to_variant  # noqa: E402

DMS = ROOT.parent.parent / "Data_GFP" / "data" / "dms_data" / "avgfp"
TASKS = {
    "in-support":     "splits/standard/standard_tr0.8_tu0.1_te0.1_w1abc2f4e9a64_r3597",
    "unseen-subs":    "splits/mutation/mutation_tr-muts0.8_tu0.1_r4419",
    "unseen-pos":     "splits/position/position_tr-pos0.8_tu0.1_r6822",
}


def read_dataset():
    rows = [l.rstrip().split("\t") for l in (DMS / "avgfp.tsv").read_text().splitlines()[1:]]
    wt = (ROOT / "data" / "avgfp_wt.txt").read_text().strip()
    return rows, wt


def apply_variant(wt: str, variant: str) -> str:
    seq = list(wt)
    for m in variant.split(","):
        seq[int(m[1:-1])] = m[-1]
    return "".join(seq)


def read_indices(path: Path) -> list[int]:
    """Split files hold row indices into avgfp.tsv, not variant strings."""
    return [int(x) for x in path.read_text().split()]


def evaluate(tag, params_m, feature_fn, selection, scores, results):
    for task, (train, test) in selection.items():
        try:
            x_train, x_test = feature_fn(train), feature_fn(test)
            if hasattr(x_train, "nbytes"):
                # Ridge solves the dual when features outnumber samples. Keeping
                # float32 matters: a float64 upcast doubles an already large
                # matrix and is the one thing likely to exhaust memory here.
                print(f"    [{x_train.shape[0]}x{x_train.shape[1]} "
                      f"{x_train.dtype}, {x_train.nbytes/1e9:.1f} GB]", flush=True)
            y_train = np.array([scores[i] for i in train])
            y_test  = np.array([scores[i] for i in test])
            model = Ridge(alpha=1.0).fit(x_train, y_train)
            predicted = model.predict(x_test)
            # A representation with no information about the test substitutions
            # yields a constant prediction, for which Spearman is undefined.
            constant = bool(np.ptp(predicted) < 1e-9)
            rho = None if constant else round(float(spearmanr(predicted, y_test).statistic), 4)
            results.append({"model": tag, "params_m": params_m, "task": task,
                            "spearman": rho, "constant": constant,
                            "mae": round(float(np.abs(predicted - y_test).mean()), 4),
                            "features": int(x_train.shape[1])})
            print(f"  {tag:<22}{task:<17}"
                  f"rho={'CONSTANT' if constant else f'{rho:+.4f}'}  "
                  f"features={x_train.shape[1]}", flush=True)
            del x_train, x_test
            gc.collect()
        except Exception as exc:                       # keep the sweep going
            print(f"  {tag:<22}{task:<17}FAILED: {type(exc).__name__}: {exc}", flush=True)
            results.append({"model": tag, "task": task, "error": f"{type(exc).__name__}: {exc}"[:200]})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+",
                        default=["onehot", "metl", "esm2_8m", "esm2_35m",
                                 "esm2_150m", "esm2_650m"],
                        help="onehot, metl, and/or ESM tags: " + ", ".join(ESM_HF))
    parser.add_argument("--train-n", type=int, default=None,
                        help="Variants from each task's train file. Default: all of them "
                             "(37,232 / 21,670 / 22,975). Lower it if memory is tight.")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS),
                        choices=list(TASKS), help="Which tasks to run")
    parser.add_argument("--append", action="store_true",
                        help="Merge into an existing --out file instead of replacing it, "
                             "so models can be run one at a time and resumed.")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "oracle_sweep.json")
    args = parser.parse_args()

    rows, wt = read_dataset()
    limit = args.train_n if args.train_n else None
    selection = {name: (read_indices(DMS / TASKS[name] / "train.txt")[:limit],
                        read_indices(DMS / TASKS[name] / "test.txt"))
                 for name in args.tasks}
    for name, (train, test) in selection.items():
        assert not set(train) & set(test), f"{name}: train and test overlap"
        print(f"{name:<16} train {len(train):>6}  test {len(test):>6}")

    index = sorted({i for tr, te in selection.values() for i in tr + te})
    sequences = {i: apply_variant(wt, rows[i][0]) for i in index}
    scores = {i: float(rows[i][2]) for i in index}
    print(f"{len(index)} unique sequences to encode\n")

    results = []
    if args.append and args.out.is_file():
        results = json.loads(args.out.read_text())
        done = {(r["model"], r["task"]) for r in results}
        print(f"resuming: {len(results)} existing rows in {args.out.name}\n")
    else:
        done = set()

    for tag in args.models:
        print(f"[{tag}]", flush=True)
        started = time.time()
        if tag == "onehot":
            evaluate("onehot", 0.0,
                     lambda ii: featurize([sequences[i] for i in ii], wt),
                     selection, scores, results)
        elif tag == "metl":
            emb = encode_metl([sequences[i] for i in index], wt)
            lookup = {i: emb[j] for j, i in enumerate(index)}
            print(f"  encoded in {time.time()-started:.0f}s, dim={emb.shape[1]}", flush=True)
            evaluate("METL (fc1)", 2.46,
                     lambda ii: np.stack([lookup[i] for i in ii]),
                     selection, scores, results)
            del emb, lookup
        else:
            # encode_esm already returns the per-residue difference from wild
            # type, flattened; mean pooling is computed from the same tensors.
            params = {"esm2_8m": 8, "esm2_35m": 35, "esm2_150m": 150,
                      "esm2_650m": 650, "esm2_3b": 3000, "esm2_15b": 15000}[tag]
            batch = 32 if params <= 150 else 8
            delta = encode_esm([sequences[i] for i in index], wt, tag, batch_size=batch)
            dim = delta.shape[1] // len(wt)
            print(f"  encoded in {time.time()-started:.0f}s, per-residue dim={dim}", flush=True)
            lookup = {i: delta[j] for j, i in enumerate(index)}
            evaluate(f"{tag} meanpool", params,
                     lambda ii: np.stack([lookup[i].reshape(len(wt), dim).mean(0) for i in ii]),
                     selection, scores, results)
            evaluate(f"{tag} posdelta", params,
                     lambda ii: np.stack([lookup[i] for i in ii]),
                     selection, scores, results)
            del delta, lookup
        gc.collect()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))

    print(f"\nWrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
