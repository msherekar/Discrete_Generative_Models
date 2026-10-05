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

# How many sequences to promote to float32 at once while standardizing in
# place. 2,048 x 237 x 320 x 4 B is about 620 MB, small beside the buffer it
# is rewriting and large enough that the loop overhead does not matter.
_STANDARDIZE_CHUNK = 2048

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

def choose_latent_device(n, length, dim, dtype=torch.float16, headroom=0.5):
    """Where the latent buffer should live: the GPU when it comfortably fits.

    Keeping the whole split on the GPU is what removes the training loop's host
    bottleneck -- see training.on_gpu_batches -- but only if there is room left
    for activations. `headroom` is the fraction of free VRAM the buffer is
    allowed to take; the rest covers the model, its gradients, the optimizer
    state and the largest activation.

    avGFP at float16 is 6.27 GiB, which passes on an 80 GB H100 (and on the
    128 GB unified memory of a GB10) and fails on a 12 GB card, where the
    buffer stays on the host and the loop falls back to copying per batch.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    needed = n * length * dim * torch.empty((), dtype=dtype).element_size()
    free, _ = torch.cuda.mem_get_info()
    return DEVICE if needed < headroom * free else torch.device("cpu")


@torch.no_grad()
def load_data(csv_path: Path, esm_hf_id: str, cache_dir: Path,
              max_length: int = 128, encode_batch: int = None,
              store_dtype=torch.float16, store_device=None):
    """CSV in, standardized ESM-2 latents out.

    `store_dtype` and `store_device` control the latent buffer only. Pass
    float32 to reproduce the pre-2026-10 numerics exactly, and a device to
    force residency either way; the default None lets choose_latent_device()
    decide from the free VRAM.
    """
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
    names = names[:next((k for k, n in enumerate(names) if n not in rows[0]), len(names))]
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
    if store_device is None:
        store_device = choose_latent_device(len(sequences), max(lengths),
                                            hidden_size, store_dtype)

    # The encode pass used the module-level BATCH_SIZE of 16 regardless of
    # --batch-size, so 41,372 sequences meant 2,586 tiny forward passes and a
    # CPU-bound phase with the GPU near idle. Every sequence has the same
    # length, so batching introduces no padding and changes no result.
    encode_batch = encode_batch or BATCH_SIZE
    z, z_mean, z_std = _encode_latents(sequences, esm, tokenizer, hidden_size,
                                       encode_batch, store_dtype, store_device)
    r_mean, r_std = r.mean(0), r.std(0, correction=0).clamp_min(1e-6)
    r_tilde = ((r - r_mean) / r_std).to(z.device)
    dataset = TensorDataset(z, c.to(z.device), r_tilde)
    stats = {"z_mean": z_mean, "z_std": z_std, "r_mean": r_mean, "r_std": r_std,
             "r_names": names}
    print(f"  Properties: {', '.join(names)}  "
          f"(raw means {', '.join(f'{v:+.3f}' for v in r_mean.tolist())})")
    return dataset, esm, tokenizer, stats, sequences


@torch.no_grad()
def _encode_latents(sequences, esm, tokenizer, hidden_size, encode_batch,
                    store_dtype, store_device):
    """Standardized ESM-2 latents for a whole split, in two streaming passes.

    This function exists because the obvious four-line version is what held
    job 15833935 on OSPool. It was:

        encoded = [...]                        # list of CPU chunks
        z = torch.cat(encoded)                 # a second full copy
        TensorDataset((z - z_mean) / z_std)    # a third and a fourth

    with `encoded` still in scope for all of it. At avGFP's
    41,372 x 237 x 320 each copy is 12.55 GB in float32, so the peak was about
    50 GB of host RAM against a 60 GB request -- and the job was killed after
    both models had finished training, during plotting, losing everything.

    Here there is one buffer and one chunk live at a time:

      pass 1  encode straight into a preallocated buffer, accumulating the
              channel sums and sums of squares in float64 so the statistics do
              not inherit the storage dtype's precision.
      pass 2  standardize that buffer in place, one chunk at a time, promoting
              each chunk to float32 for the arithmetic before casting back.

    float16 storage halves it again to 6.27 GB, which is small enough to keep
    resident on the GPU (see `store_device`) and skip the host-to-device copy
    that left the GPU idling at 0% between steps. The latents are standardized
    to roughly unit variance, so float16's ~3 decimal digits are ample; the
    statistics themselves stay float32 because decoding multiplies by them.
    """
    n, length = len(sequences), len(sequences[0])
    buffer_device = torch.device(store_device) if store_device else torch.device("cpu")
    z = torch.empty((n, length, hidden_size), dtype=store_dtype, device=buffer_device)
    total = torch.zeros(hidden_size, dtype=torch.float64, device=DEVICE)
    total_sq = torch.zeros(hidden_size, dtype=torch.float64, device=DEVICE)

    for start in range(0, n, encode_batch):
        chunk = sequences[start:start + encode_batch]
        toks = tokenizer(chunk, return_tensors="pt")
        toks = {k: v.to(DEVICE) for k, v in toks.items()}
        # [:, 1:-1] drops the BOS/EOS tokens ESM-2 adds, leaving one row per residue.
        h = esm.esm(**toks).last_hidden_state[:, 1:-1]
        flat = h.reshape(-1, hidden_size).double()
        total += flat.sum(0)
        total_sq += (flat * flat).sum(0)
        z[start:start + len(chunk)] = h.to(device=buffer_device, dtype=store_dtype)

    count = float(n * length)
    mean = (total / count)
    # Population variance, matching the correction=0 the previous code used.
    var = (total_sq / count - mean * mean).clamp_min(0.0)
    z_mean = mean.to(torch.float32).view(1, 1, hidden_size).cpu()
    z_std = var.sqrt().to(torch.float32).view(1, 1, hidden_size).clamp_min(1e-4).cpu()

    scale_mean = z_mean.to(buffer_device)
    scale_std = z_std.to(buffer_device)
    for start in range(0, n, _STANDARDIZE_CHUNK):
        stop = min(start + _STANDARDIZE_CHUNK, n)
        block = z[start:stop].to(torch.float32)
        block = (block - scale_mean) / scale_std
        z[start:stop] = block.to(store_dtype)

    gigabytes = z.numel() * z.element_size() / 1024 ** 3
    print(f"  Latents     : {tuple(z.shape)} {store_dtype} on {buffer_device} "
          f"({gigabytes:.2f} GiB)")
    return z, z_mean, z_std


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


if __name__ == "__main__":
    import torch
    # composition_proxies: shape [N, 2], finite, in [0, 1] for fractions.
    seqs = ["ACDEFGHIKLMN", "KRKRKRKRKRKR", "DEDEVEDEVE__".replace("_", "A")]
    props = composition_proxies(seqs)
    assert props.shape == (3, 2) and props.isfinite().all()
    print(f"  composition_proxies: shape={props.shape}")
    print(f"  charge proxy:    {props[:, 0].tolist()}")
    print(f"  polarity proxy:  {props[:, 1].tolist()}")

    # consensus: returns most-common residue per position.
    seqs2 = ["ACDE", "ACDF", "ACDE"]
    c = consensus(seqs2)
    assert len(c) == 4
    assert c[3] == "E", f"consensus mismatch at pos 3: {c[3]}"
    print(f"  consensus(['ACDE','ACDF','ACDE']): {c!r}")

    # choose_latent_device: returns a valid torch.device.
    dev = choose_latent_device(100, 32, 16)
    assert isinstance(dev, torch.device)
    print(f"  choose_latent_device(100,32,16): {dev}")

    # load_data requires ESM weights; skip here.
    print(f"  load_data: requires ESM download (skipped in smoke test)")
    print("data.py OK")
