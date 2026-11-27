#!/usr/bin/env python3
"""What loss should flow matching actually reach on this data?

The training loss in run_experiment.py plateaus around 0.78 and stops moving.
That is expected, and this script says why by computing the reference points the
logged number should be compared against:

  predict 0                    what a model that learned nothing scores
  predict the dataset mean     what a model that learned only the average scores
  pi/2 = 1.5708                the irreducible floor IF the data were isotropic
                               Gaussian; conditional flow matching cannot beat
                               this on Gaussian data, because the target
                               z1 - z0 keeps a fresh z0 and t every step
  oracle velocity field        the floor if every variant were the same latent

It also reports how the standardized latent variance splits between positions
and sequences. For avGFP that split is ~95/5: variants differ at a handful of
237 positions, so almost all the variance is the shared backbone, which the
model fits quickly. That is why the loss drops fast and then flattens.

Note the logged loss is the SUM of the flow term and the reward term. The reward
term has its own floor near 1.0 at high noise, where brightness is not
predictable from the latent at all, so a large part of the plateau is
irreducible rather than a failure to converge.

Usage:
  python diagnostics/check_loss_floor.py
  python diagnostics/check_loss_floor.py --n 600 --esm-model facebook/esm2_t12_35M_UR50D
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from dgm.common.paths import project_dir

ROOT = project_dir()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "avgfp_train_4000.csv")
    parser.add_argument("--esm-model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--n", type=int, default=600, help="Sequences to encode")
    parser.add_argument("--draws", type=int, default=8000, help="Monte Carlo draws")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.dataset.open(newline="")))[:args.n]
    sequences = [r["sequence"].strip().upper() for r in rows]
    rewards = torch.tensor([[float(r["r1"]), float(r["r2"])] for r in rows])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model, cache_dir=str(ROOT / "cache"))
    esm = EsmForMaskedLM.from_pretrained(
        args.esm_model, cache_dir=str(ROOT / "cache"), use_safetensors=True
    ).to(device).eval().requires_grad_(False)

    encoded = []
    with torch.no_grad():
        for start in range(0, len(sequences), 16):
            toks = {k: v.to(device) for k, v in
                    tokenizer(sequences[start:start + 16], return_tensors="pt").items()}
            encoded.append(esm.esm(**toks).last_hidden_state[:, 1:-1].cpu())
    z = torch.cat(encoded)
    print(f"latents {tuple(z.shape)}")

    # exactly the normalization run_experiment.load_data applies
    z1 = (z - z.mean((0, 1), keepdim=True)) / z.std((0, 1), correction=0, keepdim=True).clamp_min(1e-4)
    r_std = (rewards - rewards.mean(0)) / rewards.std(0, correction=0).clamp_min(1e-6)

    positional = z1.mean(0).var().item()
    per_sequence = z1.var(0).mean().item()
    total = z1.var().item()
    print(f"\nSTANDARDIZED LATENT STRUCTURE (overall var {total:.3f})")
    print(f"  variance across POSITIONS : {positional:.4f}  ({positional/total:.1%})")
    print(f"  variance across SEQUENCES : {per_sequence:.4f}  ({per_sequence/total:.1%})")

    torch.manual_seed(0)
    picked = z1[torch.randint(0, len(z1), (args.draws,))]
    noise  = torch.randn_like(picked)
    t      = torch.rand(args.draws, 1, 1)
    target = picked - noise
    mean_latent = z1.mean(0)

    print("\nFLOW TERM  (MSE against z1 - z0)")
    print(f"  predict 0                        : {target.pow(2).mean():.4f}")
    print(f"  predict the dataset mean latent  : {(target - mean_latent).pow(2).mean():.4f}")
    print(f"  irreducible floor, Gaussian data : {np.pi/2:.4f}   (pi/2)")
    deterministic_zt = (1 - t) * noise + t * mean_latent
    optimal = (mean_latent - deterministic_zt) / (1 - t).clamp_min(1e-3)
    print(f"  oracle v=(c-zt)/(1-t), c=mean    : "
          f"{(optimal - (mean_latent - noise)).pow(2).mean():.6f}"
          f"   <- floor if every variant shared one latent")

    print("\nREWARD TERM  (MSE against standardized r)")
    print(f"  predict 0                        : {r_std.pow(2).mean():.4f}")
    print("  (at t near 0 the latent is pure noise and brightness is not "
          "predictable,\n   so this term cannot reach 0 averaged over t)")

    print(f"\nTRIVIAL TOTAL (mean latent + zero reward): "
          f"{(target - mean_latent).pow(2).mean() + r_std.pow(2).mean():.4f}")
    print("Compare the logged epoch loss against the TRIVIAL TOTAL, not against 0.")


if __name__ == "__main__":
    main()
