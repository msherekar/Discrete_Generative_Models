"""Dataset loading: CSV in, standardized ESM-2 latents out.

The composition proxies stand in for measured properties when the CSV carries
no rN columns, so the teaching set and a real assay both flow through the same
path.
"""
import csv
from pathlib import Path

import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, EsmForMaskedLM

from .config import AMINO_ACIDS, BATCH_SIZE, DEVICE, POLAR_RESIDUES

# ══════════════════════════════════════════════════════════════════════════════
# Composition proxies
# ══════════════════════════════════════════════════════════════════════════════

def composition_proxies(sequences):
    return torch.tensor([
        [(sum(a in "KR" for a in s) - sum(a in "DE" for a in s)) / len(s),
         sum(a in POLAR_RESIDUES for a in s) / len(s)]
        for s in sequences
    ], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def load_data(csv_path: Path, esm_hf_id: str, cache_dir: Path,
              max_length: int = 128, encode_batch: int = None):
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    sequences = [r["sequence"].strip().upper() for r in rows]
    if len(rows) < 4 or any(not s or set(s) - set(AMINO_ACIDS) for s in sequences):
        raise ValueError("Supply at least four sequences using the 20 standard amino acids")
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1 or max(lengths) > max_length:
        raise ValueError(f"Sequences must all be the same length, at most {max_length}")
    c = torch.tensor([int(r["c"]) for r in rows], dtype=torch.long)
    # However many rN columns the file carries, in order. add_properties.py
    # writes r1=brightness, r2=aggregation proxy, r3=stability, where r3 exists
    # to be constrained rather than scalarized. Two columns is the older layout
    # and still works.
    names = [f"r{i}" for i in range(1, 100)]
    names = names[:next((k for k, n in enumerate(names) if n not in rows[0]), 0)]
    if len(names) >= 2:
        r = torch.tensor([[float(row[n]) for n in names] for row in rows])
    else:
        names = ["r1", "r2"]
        r = composition_proxies(sequences)
    if set(c.tolist()) != {0, 1} or not torch.isfinite(r).all():
        raise ValueError("Both c=0 and c=1 must be present; r1/r2 must be finite")

    print(f"  Loading {esm_hf_id} from cache: {cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(esm_hf_id, cache_dir=cache_dir)
    esm = EsmForMaskedLM.from_pretrained(
        esm_hf_id, cache_dir=cache_dir, use_safetensors=True
    ).to(DEVICE).eval().requires_grad_(False)
    hidden_size = esm.config.hidden_size
    print(f"  ESM-2 hidden size: {hidden_size}  |  sequences: {len(sequences)}  |  length: {max(lengths)}")

    # The encode pass used the module-level BATCH_SIZE of 16 regardless of
    # --batch-size, so 41,372 sequences meant 2,586 tiny forward passes and a
    # CPU-bound phase with the GPU near idle. Every sequence has the same
    # length, so batching introduces no padding and changes no result.
    encode_batch = encode_batch or BATCH_SIZE
    encoded = []
    for start in range(0, len(sequences), encode_batch):
        toks = tokenizer(sequences[start:start + encode_batch], return_tensors="pt")
        toks = {k: v.to(DEVICE) for k, v in toks.items()}
        h = esm.esm(**toks).last_hidden_state
        encoded.append(h[:, 1:-1].cpu())
    z = torch.cat(encoded)                         # [N, L, hidden_size]
    z_mean = z.mean((0, 1), keepdim=True)
    z_std  = z.std((0, 1), correction=0, keepdim=True).clamp_min(1e-4)
    r_mean, r_std = r.mean(0), r.std(0, correction=0).clamp_min(1e-6)
    dataset = TensorDataset((z - z_mean) / z_std, c, (r - r_mean) / r_std)
    stats = {"z_mean": z_mean, "z_std": z_std, "r_mean": r_mean, "r_std": r_std,
             "r_names": names}
    print(f"  Properties: {', '.join(names)}  "
          f"(raw means {', '.join(f'{v:+.3f}' for v in r_mean.tolist())})")
    return dataset, esm, tokenizer, stats, sequences


@torch.no_grad()
def encode_reference(sequence, esm, tokenizer, stats):
    """Standardized ESM-2 latent for one reference sequence, shaped [1, L, dim]."""
    toks = tokenizer([sequence], return_tensors="pt")
    toks = {k: v.to(DEVICE) for k, v in toks.items()}
    h = esm.esm(**toks).last_hidden_state[:, 1:-1].cpu()
    return ((h - stats["z_mean"]) / stats["z_std"]).to(DEVICE)


def consensus(sequences):
    """Per-position most common residue; equals the wild type for DMS variant sets."""
    return "".join(max(AMINO_ACIDS, key=lambda a: sum(s[i] == a for s in sequences))
                   for i in range(len(sequences[0])))
