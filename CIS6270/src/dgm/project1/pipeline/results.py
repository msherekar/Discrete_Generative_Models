"""Writing a finished run to disk: results.pt plus one FASTA per arm.

The run configuration is saved alongside the samples because an arm name alone
does not identify what was optimized -- the same lambda means a different
tradeoff under a different sense or constraint.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .config import POLAR_RESIDUES

# ══════════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════════

def save_results(out_dir: Path, method: str, latents: dict, model, reward_model,
                 stats, esm_hf_id: str, length: int, dim: int,
                 min_polar: int, losses: list, config: dict | None = None):
    method_dir = out_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    for name, latent in latents.items():
        scores = latent.get("oracle")
        fasta = "".join(
            f">{name}_{i+1}"
            + (f" oracle_brightness={scores[i]:+.4f}" if scores is not None else "")
            + f"\n{seq}\n"
            for i, seq in enumerate(latent["sequences"]))
        (method_dir / f"{name}.fasta").write_text(fasta)
    torch.save({
        "standardized_latents": {k: v["latent"].cpu() for k, v in latents.items()},
        "sequences":    {k: v["sequences"] for k, v in latents.items()},
        "oracle_brightness": {k: v.get("oracle") for k, v in latents.items()},
        "model":        model.state_dict(),
        "reward_model": reward_model.state_dict(),
        "stats":        stats,
        "esm_name":     esm_hf_id,
        "length":       length,
        "dim":          dim,
        "min_polar":    min_polar,
        "polar_residues": POLAR_RESIDUES,
        "losses":       losses,
        "config":       config or {},
    }, method_dir / "results.pt")
    print(f"  Saved {method} results to {method_dir}/")
