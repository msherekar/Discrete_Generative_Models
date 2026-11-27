#!/usr/bin/env python3
"""Encode a split's ESM-2 latents once and save them as a float16 tensor.

Every OSG job currently re-encodes all 41,372 avGFP sequences. Doing it once
and shipping the result removes that work from every job and removes the ESM-2
weights from the staged payload.
"""
import argparse
from pathlib import Path

import torch

from dgm.common.esm_models import get_model
from dgm.common.paths import data_dir
from dgm.project1.pipeline.config import resolve_cache_dir
from dgm.project1.pipeline.data import load_data

# Bump when the on-disk layout changes, so a stale cache is refused rather than
# silently reinterpreted.
FORMAT_VERSION = 1


def cache_path(cache_dir: Path, csv_path: Path, esm_model: str, max_length: int):
    """Where a given (split, encoder, length) combination is stored."""
    return cache_dir / f"latents_{csv_path.stem}_{esm_model}_L{max_length}.pt"


def encode(csv_path, esm_model, cache_dir, max_length, encode_batch):
    """Encode one split and return the payload to save.

    The standardization statistics travel with the latents. They have to: a
    split standardized by its own mean and standard deviation is not on the
    same scale as the training split, so a validation loss computed that way
    would not be comparable, and a decode would be wrong.
    """
    info = get_model(esm_model)
    dataset, _, _, stats, sequences = load_data(
        csv_path, info["hf_id"], cache_dir, max_length,
        encode_batch=encode_batch, store_dtype=torch.float16,
        store_device="cpu")
    z, c, r = dataset.tensors
    return {"version": FORMAT_VERSION, "z": z, "c": c.cpu(), "r": r.cpu(),
            "stats": {k: (v.cpu() if torch.is_tensor(v) else v)
                      for k, v in stats.items()},
            "sequences": sequences, "esm_model": esm_model,
            "hf_id": info["hf_id"], "max_length": max_length,
            "csv": csv_path.name}


def load_cached(path: Path):
    """Read a cached payload back, refusing a version this code cannot read."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != FORMAT_VERSION:
        raise ValueError(f"{path} was written by format version "
                         f"{payload.get('version')}, this code reads "
                         f"{FORMAT_VERSION}. Re-run dgm-cache-latents.")
    return payload


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, nargs="+",
                        default=[data_dir() / "avgfp_train_props.csv",
                                 data_dir() / "avgfp_val.csv",
                                 data_dir() / "avgfp_test.csv"],
                        help="CSVs to encode (default: the three avGFP splits).")
    parser.add_argument("--esm-model", default="esm2_8m")
    parser.add_argument("--max-length", type=int, default=237)
    parser.add_argument("--encode-batch", type=int, default=256,
                        help="Sequences per ESM-2 forward pass (default: 256).")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-encode even if the cache file already exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = resolve_cache_dir(args.cache_dir)
    for csv_path in args.splits:
        if not csv_path.is_file():
            print(f"  [skip] {csv_path} not found")
            continue
        out = cache_path(cache_dir, csv_path, args.esm_model, args.max_length)
        if out.is_file() and not args.force:
            size = out.stat().st_size / 1024 ** 3
            print(f"  [have] {out.name}  ({size:.2f} GiB)  --force to redo")
            continue
        print(f"\nEncoding {csv_path.name} ...")
        payload = encode(csv_path, args.esm_model, cache_dir, args.max_length,
                         args.encode_batch)
        torch.save(payload, out)
        size = out.stat().st_size / 1024 ** 3
        print(f"  saved {out}  ({size:.2f} GiB)")
    print("\nStage these alongside esm-cache.tar.gz via OSDF; a job that finds "
          "them skips both the encode and the ESM-2 download.")


if __name__ == "__main__":
    main()
