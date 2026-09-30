"""Shared constants and the ESM-2 weight cache location.

Everything in this package reads its device, alphabet and default widths from
here, so a change lands in one place rather than in every module.
"""
from pathlib import Path

import torch

ROOT   = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AMINO_ACIDS    = "ACDEFGHIKLMNPQRSTVWY"
POLAR_RESIDUES = "DEHKNQRST"
BATCH_SIZE, HIDDEN, LEARNING_RATE = 16, 128, 1e-3
CONDITION_DROP = 0.2

# One HuggingFace cache for every model and every run. HF already namespaces
# downloads as models--facebook--<name>, so an extra per-model or per-run
# subdirectory only causes the same weights to be fetched again.
DEFAULT_CACHE = ROOT / "cache"


def resolve_cache_dir(override=None) -> Path:
    """Shared ESM-2 weight cache: --cache-dir, else $ESM2_CACHE, else ROOT/cache."""
    import os
    chosen = override or os.environ.get("ESM2_CACHE") or DEFAULT_CACHE
    path = Path(chosen).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
