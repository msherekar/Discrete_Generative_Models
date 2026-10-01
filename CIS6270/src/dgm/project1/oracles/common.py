"""Paths, the ESM-2 registry and the small sequence helpers.

Kept separate so importing the variant notation or the model table does not
drag in METL or torch.
"""
import contextlib
import os
from pathlib import Path

from dgm.common.paths import METL_ROOT, project_dir

ROOT        = project_dir()
METL_CKPT   = METL_ROOT / "pretrained_models" / "Hr4GNHws.pt"
METL_PDB    = "1gfl_cm.pdb"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

ESM_HF = {
    "esm2_8m":   "facebook/esm2_t6_8M_UR50D",
    "esm2_35m":  "facebook/esm2_t12_35M_UR50D",
    "esm2_150m": "facebook/esm2_t30_150M_UR50D",
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
    "esm2_3b":   "facebook/esm2_t36_3B_UR50D",
    "esm2_15b":  "facebook/esm2_t48_15B_UR50D",
}


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# Sequence helpers
# ══════════════════════════════════════════════════════════════════════════════

def seq_to_variant(sequence: str, wt: str) -> str:
    """METL's mutation-string form, e.g. 'K1E,A108D'. '_wt' when identical."""
    muts = [f"{a}{i}{b}" for i, (a, b) in enumerate(zip(wt, sequence)) if a != b]
    return ",".join(muts) if muts else "_wt"


@contextlib.contextmanager
def chdir(path: Path):
    """METL resolves its PDB files relative to the repository root."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
