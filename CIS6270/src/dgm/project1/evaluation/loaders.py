"""Reading a finished run back off disk.

Results may come from the lecture_3 scripts or from run_experiment.py, so the
loaders accept both layouts and say plainly which one they found.
"""
import csv
from pathlib import Path

import torch

from .style import GUIDANCE_MODES, LECTURE3, POLAR_RESIDUES

# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def composition_proxies(sequences):
    return torch.tensor([
        [(sum(a in "KR" for a in s) - sum(a in "DE" for a in s)) / len(s),
         sum(a in POLAR_RESIDUES for a in s) / len(s)]
        for s in sequences
    ], dtype=torch.float32)


def load_fasta(path: Path):
    sequences, cur = [], ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    sequences.append(cur)
                cur = ""
            else:
                cur += line
    if cur:
        sequences.append(cur)
    return sequences


def load_results(method: str, out_dir: Path | None = None):
    """Return dict with sequences and latents for one method.

    out_dir: explicit directory containing results.pt + FASTA files.
    Falls back to the lecture_3 output layout when omitted.
    """
    if out_dir is None:
        out_dir = LECTURE3 / f"esm2_{method}_outputs"
        fallback_msg = (
            f"Run lecture_3/esm2_{method}_guidance.py --epochs 200, "
            f"or use run_experiment.py and pass --{method}-outdir."
        )
    else:
        fallback_msg = f"Run run_experiment.py and check --{method}-outdir."

    out_dir = Path(out_dir).resolve()
    pt_path = out_dir / "results.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"{pt_path} not found. {fallback_msg}")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    seqs = {}
    for mode in GUIDANCE_MODES:
        fasta = out_dir / f"{mode}.fasta"
        seqs[mode] = load_fasta(fasta) if fasta.exists() else []
    data["sequences"] = seqs
    return data


def load_training_sequences():
    csv_path = LECTURE3 / "esm2_example.csv"
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    sequences = [r["sequence"].strip().upper() for r in rows]
    c = [int(r["c"]) for r in rows]
    return sequences, c
