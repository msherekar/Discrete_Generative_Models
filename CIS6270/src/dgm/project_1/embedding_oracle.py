#!/usr/bin/env python3
"""Embedding-based brightness oracles for the avGFP study.

The ridge oracle in gfp_oracle.py scores a variant from indicators of its exact
(position, residue) substitutions. That works well inside the measured library
and not at all outside it: the avGFP deep mutational scan contains only 1,810 of
4,503 possible single substitutions, so any substitution the assay never sampled
contributes a zero and the prediction collapses to the intercept. Generated
sequences are mostly made of such substitutions.

This module fits the same ridge regression on a learned representation instead,
which assigns a value to substitutions it has never seen.

Two backends:

  metl   METL-L-2M-3D-GFP, a 2.5M-parameter transformer pretrained to predict 55
         Rosetta biophysical attributes from sequence (Gelman et al., Nature
         Methods 2025). We read its 256-dimensional pooled representation. It
         shares no architecture, tokenizer or training corpus with the ESM-2
         encoder the generator samples in, so it is an independent judge.

  esm2_* Per-residue ESM-2 embeddings, taken as the difference from wild type and
         flattened, keeping positional resolution. Mean pooling over 237 residues
         destroys the signal from a handful of substitutions and is not used.
         Note this backend shares its representation with the generator, so a
         score it assigns to a generated sequence is not fully independent.

Usage:
  dgm-embedding-oracle --fit --backend metl
  dgm-embedding-oracle --fit --backend esm2_150m
  dgm-embedding-oracle --score some.fasta --backend metl

Import:
  from dgm.project1.embedding_oracle import load_oracle, score_sequences

The backends and the fitting code live in oracles/; this file is the command
line over them and the import surface the rest of the project uses.
"""
import argparse
from pathlib import Path

from .oracles.common import (AMINO_ACIDS, ESM_HF, METL_CKPT, METL_PDB,
                            METL_ROOT, ROOT, chdir, device, seq_to_variant)
from .oracles.encoders import (encode, encode_esm, encode_esm_pca,
                              fit_projection, parse_backend)
from .oracles.fitting import (fit_on_splits, fit_oracle, load_fasta,
                             load_oracle, read_split, save_oracle,
                             score_sequences)
from .oracles.metl import (attribute_index, attribute_names, encode_metl,
                          metl_attributes, metl_attributes_wt, metl_imports)

__all__ = [
    "AMINO_ACIDS", "ESM_HF", "METL_CKPT", "METL_PDB", "METL_ROOT", "ROOT",
    "attribute_index", "attribute_names", "chdir", "device", "encode",
    "encode_esm", "encode_esm_pca", "encode_metl", "fit_on_splits",
    "fit_oracle", "fit_projection", "load_fasta", "load_oracle",
    "metl_attributes", "metl_attributes_wt", "metl_imports", "parse_backend",
    "read_split", "save_oracle", "score_sequences", "seq_to_variant",
]


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="metl",
                        help=f"metl (default) or one of: {', '.join(ESM_HF)}")
    parser.add_argument("--train", type=Path, default=None,
                        help="Training CSV from split_gfp.py (default: "
                             "data/avgfp_train.csv when it exists). With --train the "
                             "penalty is chosen on --val and reported on --test.")
    parser.add_argument("--val", type=Path, default=None,
                        help="Validation CSV; selects the ridge penalty.")
    parser.add_argument("--test", type=Path, default=None,
                        help="Test CSV; read once, for the reported accuracy.")
    parser.add_argument("--split", type=Path, default=ROOT / "data" / "avgfp_oracle.csv",
                        help="Oracle-split CSV from prepare_gfp.py")
    parser.add_argument("--wt", type=Path, default=None,
                        help="Wild-type file (default: <split dir>/avgfp_wt.txt)")
    parser.add_argument("--oracle", type=Path, default=None,
                        help="Where to save/load (default: data/avgfp_<backend>_oracle.npz)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use at most this many variants from the split")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--score", type=Path, default=None, help="FASTA to score")
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.oracle or ROOT / "data" / f"avgfp_{args.backend}_oracle.npz"
    default_train = ROOT / "data" / "avgfp_train.csv"
    if args.fit and args.train is None and default_train.is_file():
        args.train = default_train
        args.val = args.val or ROOT / "data" / "avgfp_val.csv"
        args.test = args.test or ROOT / "data" / "avgfp_test.csv"
    if args.fit and args.train is not None:
        for name, path in (("--val", args.val), ("--test", args.test)):
            if path is None or not path.is_file():
                raise SystemExit(f"{name} is required with --train (got {path}). "
                                 f"Build the splits with: dgm-split-gfp")
        wt = (args.wt or args.train.parent / "avgfp_wt.txt").read_text().strip()
        train, val, test = (read_split(q) for q in (args.train, args.val, args.test))
        if args.limit:
            train = (train[0][:args.limit], train[1][:args.limit])
        print(f"\nFitting the {args.backend} oracle on {len(train[0])} "
              f"training variants...")
        model, alpha, stats, domain, n_features, projection = fit_on_splits(
            train, val, test, wt, args.backend)
        save_oracle(out, model, wt, args.backend, stats["val"]["rho"],
                    stats["val"]["mae"], domain, n_features, projection)
        print(f"\n  representation: {n_features} features")
        print(f"  chosen penalty alpha = {alpha}")
        for name in ("val", "test"):
            s = stats[name]
            print(f"  {name:<5} n={s['n']:<6} Spearman {s['rho']:.4f}  "
                  f"MAE {s['mae']:.4f}")
        print(f"  saved to {out}")
    elif args.fit:
        sequences, scores = read_split(args.split)
        if args.limit:
            sequences, scores = sequences[:args.limit], scores[:args.limit]
        wt = (args.wt or args.split.parent / "avgfp_wt.txt").read_text().strip()
        print(f"Fitting {args.backend} oracle on {len(sequences)} variants...")
        model, rho, mae, domain, n_features = fit_oracle(
            sequences, scores, wt, args.backend, args.alpha)
        save_oracle(out, model, wt, args.backend, rho, mae, domain, n_features)
        print(f"  representation: {n_features} features")
        print(f"  validation Spearman {rho:.3f} | MAE {mae:.3f} | saved to {out}")
    if args.score is not None:
        oracle = load_oracle(out)
        sequences = load_fasta(args.score)
        predicted = score_sequences(sequences, oracle)
        for sequence, value in zip(sequences, predicted):
            print(f"  {value:+.3f}  {sequence[:40]}...")
        print(f"mean predicted brightness {predicted.mean():+.3f}")
    if not args.fit and args.score is None:
        print("Nothing to do: pass --fit and/or --score")




if __name__ == "__main__":
    main()
