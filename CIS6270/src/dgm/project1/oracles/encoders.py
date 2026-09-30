"""The ESM-2 backend: per-residue embeddings as a difference from wild type.

Mean pooling over 237 residues destroys the signal from a handful of
substitutions, so positional resolution is kept and optionally projected down
with a fitted PCA basis.
"""
from pathlib import Path

import numpy as np

from .common import ESM_HF, ROOT, device
from .metl import encode_metl

def encode_esm(sequences: list[str], wt: str, tag: str,
               cache_dir: Path | None = None, batch_size: int = 32) -> np.ndarray:
    """Per-residue ESM-2 embedding minus wild type, flattened to keep position."""
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM
    if tag not in ESM_HF:
        raise ValueError(f"Unknown ESM tag '{tag}'. Known: {', '.join(ESM_HF)}")
    cache_dir = cache_dir or ROOT / "cache"
    tokenizer = AutoTokenizer.from_pretrained(ESM_HF[tag], cache_dir=str(cache_dir))
    model = EsmForMaskedLM.from_pretrained(
        ESM_HF[tag], cache_dir=str(cache_dir), use_safetensors=True
    ).to(device()).eval().requires_grad_(False)

    def embed(batch):
        toks = {k: v.to(device()) for k, v in tokenizer(batch, return_tensors="pt").items()}
        return model.esm(**toks).last_hidden_state[:, 1:-1].cpu().numpy().astype(np.float32)

    with torch.no_grad():
        z_wt = embed([wt])[0]
        out = []
        for start in range(0, len(sequences), batch_size):
            h = embed(sequences[start:start + batch_size])
            out.append((h - z_wt).reshape(len(h), -1))
    return np.concatenate(out)


def parse_backend(backend: str):
    """Split a backend name into (tag, k). 'esm2_8m_pca64' -> ('esm2_8m', 64).

    A bare ESM tag keeps the legacy flattened per-residue delta, which is 237 x d
    features and needs an iterative solver. The _pca<k> form is the one measured
    to work: see esm_compare.py, where a 64-component per-position projection beat
    one-hot indicators on the held-out split by +0.0046 Spearman (paired bootstrap
    95% interval [+0.0015, +0.0078]), while mean pooling over positions scored
    0.448 because it discards which position mutated.
    """
    if "_pca" in backend:
        tag, _, k = backend.partition("_pca")
        return tag, int(k)
    return backend, None


def fit_projection(sequences, wt, tag, k, cache_dir=None, batch_size=32, pca_n=1500):
    """PCA basis for the per-residue delta, shared across positions.

    Returned as (mean, components) so it can be saved beside the ridge weights --
    the projection is part of the featurizer, and an oracle that cannot reproduce
    it cannot score anything.
    """
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(0)
    pick = rng.choice(len(sequences), min(pca_n, len(sequences)), replace=False)
    sample = [sequences[i] for i in pick]
    pooled = np.concatenate([b.reshape(-1, b.shape[-1]) for b in
                             _stream_delta(sample, wt, tag, cache_dir, batch_size)])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(pooled)
    kept = float(np.sum(pca.explained_variance_ratio_))
    print(f"  projection: {k} components, {100*kept:.1f}% of the delta variance")
    return pca.mean_.astype(np.float32), pca.components_.astype(np.float32)


def _stream_delta(sequences, wt, tag, cache_dir=None, batch_size=32):
    """Yield (batch, length, d) per-residue ESM-2 embeddings minus wild type."""
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM
    if tag not in ESM_HF:
        raise ValueError(f"Unknown ESM tag '{tag}'. Known: {', '.join(ESM_HF)}")
    cache_dir = cache_dir or ROOT / "cache"
    tokenizer = AutoTokenizer.from_pretrained(ESM_HF[tag], cache_dir=str(cache_dir))
    model = EsmForMaskedLM.from_pretrained(
        ESM_HF[tag], cache_dir=str(cache_dir), use_safetensors=True
    ).to(device()).eval().requires_grad_(False)

    def embed(batch):
        toks = {k: v.to(device())
                for k, v in tokenizer(batch, return_tensors="pt").items()}
        return model.esm(**toks).last_hidden_state[:, 1:-1]

    with torch.no_grad():
        wt_latent = embed([wt])[0]
        for start in range(0, len(sequences), batch_size):
            yield (embed(sequences[start:start + batch_size])
                   - wt_latent).cpu().numpy().astype(np.float32)


def encode_esm_pca(sequences, wt, tag, projection, cache_dir=None, batch_size=32):
    """Per-position ESM-2 delta projected onto a saved basis: 237 x k features."""
    mean, components = projection
    out = []
    for block in _stream_delta(sequences, wt, tag, cache_dir, batch_size):
        n, length, _ = block.shape
        flat = block.reshape(-1, block.shape[-1]) - mean
        out.append((flat @ components.T).reshape(n, length * len(components)))
    return np.concatenate(out).astype(np.float32)


def encode(sequences: list[str], wt: str, backend: str, projection=None,
           **kwargs) -> np.ndarray:
    if backend == "metl":
        return encode_metl(sequences, wt, **kwargs)
    tag, k = parse_backend(backend)
    if k is not None:
        if projection is None:
            raise ValueError(f"backend '{backend}' needs its saved PCA projection")
        return encode_esm_pca(sequences, wt, tag, projection, **kwargs)
    return encode_esm(sequences, wt, backend, **kwargs)
