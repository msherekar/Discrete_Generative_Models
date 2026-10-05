"""Paths, the ESM-2 registry and the small sequence helpers.

Kept separate so importing the variant notation or the model table does not
drag in METL or torch.
"""
import contextlib
import os
from pathlib import Path

from dgm.common.paths import METL_ROOT, project_dir

ROOT        = project_dir()
METL_PDB    = "1gfl_cm.pdb"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# METL-L-2M-3D-GFP, the Rosetta-attribute source model (UUID Hr4GNHws).
METL_SOURCE_UUID = "Hr4GNHws"

# Released finetuned GFP target models, which predict the DMS functional score
# directly rather than Rosetta energies. 1D is the one to prefer: Gelman et al.
# report that 3D relative position embeddings help METL-Global but "do not make
# a difference for METL-Local models", and the 1D model needs no PDB file,
# which removes a staging dependency on the OSG node.
METL_TARGET_UUIDS = {
    "ft-1d":    "HaUuRwfE",     # FT-METL-L-1D-GFP, trained on 80% of the DMS
    "ft-3d":    "LWEY95Yb",     # FT-METL-L-3D-GFP, same but needs 1gfl_cm.pdb
    "ft-1d-64": "YoQkzoLD",     # finetuned on 64 examples, the design experiment
    "ft-3d-64": "PEkeRuxb",
}


def _resolve_metl_ckpt() -> Path:
    """Where the METL source checkpoint lives.

    Search order, first hit wins:

      1. $METL_CKPT, for a path that is neither of the two below.
      2. The project's weight cache. This is the preferred home: cache_dir()
         is already the one directory every run shares and every OSG job
         stages, so the 9.4 MB checkpoint rides along with the ESM-2 weights
         instead of needing the whole metl checkout on the node.
      3. The metl repository's own pretrained_models/, where it ships.

    Returns the cache path when nothing exists yet, so the error message from
    _load_metl() names the location a user should copy the file to.
    """
    override = os.environ.get("METL_CKPT")
    if override:
        return Path(override).expanduser().resolve()
    cached = ROOT / "cache" / f"{METL_SOURCE_UUID}.pt"
    vendored = METL_ROOT / "pretrained_models" / f"{METL_SOURCE_UUID}.pt"
    return cached if cached.is_file() or not vendored.is_file() else vendored


METL_CKPT = _resolve_metl_ckpt()

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
    muts = [f"{a}{i}{b}" for i, (a, b) in enumerate(zip(wt, sequence), start=1) if a != b]
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
