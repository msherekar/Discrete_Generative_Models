#!/usr/bin/env python3
"""What does a coupling actually buy, before any model is trained?

A coupling claim is a claim about the REGRESSION TARGET, not about sample
quality. Flow matching regresses the velocity onto z1 - z0, and a coupling that
pairs nearby points shrinks that target. check_loss_floor.py already names the
reference points this should be read against:

    predict 0                 what a model that learned nothing scores, which is
                              exactly mean((z1 - z0)^2) -- the number below
    pi/2 = 1.5708             the irreducible floor IF the data were isotropic
                              Gaussian, which holds ONLY under independent
                              coupling, because that is what keeps z0 fresh and
                              independent of z1 at every step

So the headline number here is the ratio to the independent baseline. A coupling
that does not move it cannot straighten anything, and no amount of sampling will
rescue it -- which is why this runs in seconds and belongs before the sweep, not
after it.

Minibatch OT is biased with respect to true OT and the bias shrinks with batch
size, so batch size is swept rather than fixed: the figure shows how much of the
gain is real transport and how much is small-batch artefact.

Nothing is trained. These are properties of the pairing alone.

Usage:
  python diagnostics/check_coupling.py --modality image
  python diagnostics/check_coupling.py --modality peptide \
      --dataset ../lecture/lecture_3/esm2_example.csv
  python diagnostics/check_coupling.py --modality image --betas 0 0.5 1 2
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dgm.common.paths import SHARED_DATA, plots_dir, project_dir

from ..pipeline.coupling import (InformedSource, couple, pair_distance,
                                 target_moment)

ROOT = project_dir()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PI_OVER_2 = float(np.pi / 2)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modality", default="image", choices=("image", "peptide"))
    p.add_argument("--dataset", type=Path,
                   default=ROOT.parent / "lecture" / "lecture_3" / "esm2_example.csv",
                   help="Peptide CSV (ignored for --modality image).")
    p.add_argument("--esm-model", default="esm2_8m")
    p.add_argument("--data-dir", type=Path, default=SHARED_DATA,
                   help="MNIST root (ignored for --modality peptide).")
    p.add_argument("--limit", type=int, default=4096,
                   help="Images to load; the peptide set is used whole.")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128],
                   help="Minibatch sizes to sweep; OT's bias shrinks with this.")
    p.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                   help="Property-cost weights for the 'aux' coupling.")
    p.add_argument("--repeats", type=int, default=8,
                   help="Independent batches per cell, averaged.")
    p.add_argument("--columns", type=int, nargs="+", default=None,
                   help="0-based property columns in the 'aux' cost. Default: all.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--prefix", default=None,
                   help="Output file prefix. Default: coupling_<modality>.")
    return p.parse_args()


def load(args):
    """(latents, properties) on DEVICE, both standardized, for either modality."""
    if args.modality == "image":
        from ..pipeline.images import load_mnist
        dataset, _ = load_mnist(args.limit, args.data_dir)
        return (dataset.tensors[0].to(DEVICE), dataset.tensors[2].to(DEVICE))
    from dgm.common.esm_models import get_model
    from ..pipeline.config import resolve_cache_dir
    from ..pipeline.data import load_data
    info = get_model(args.esm_model)
    dataset = load_data(args.dataset, info["hf_id"], resolve_cache_dir(None))[0]
    return (dataset.tensors[0].to(DEVICE), dataset.tensors[2].to(DEVICE))


def variants(args):
    """(label, kind, beta) for every coupling the run compares."""
    out = [("independent", "independent", 0.0), ("ot", "ot", 0.0)]
    out += [(f"aux_b{b:g}", "aux", float(b)) for b in args.betas]
    out.append(("informed", "informed", 0.0))
    return out


def measure(z1, aux, kind, beta, batch, repeats, columns, source):
    """Target moment and pair distance for one coupling at one batch size."""
    moments, distances = [], []
    for _ in range(repeats):
        index = torch.randperm(len(z1), device=DEVICE)[:batch]
        data, props = z1[index], aux[index]
        if kind == "informed":
            # Not a permutation: the source itself is different, so there is no
            # pairing to solve. Reported in the same column because it is the
            # alternative being compared, not because it is a coupling.
            noise = source.paired(props)
        else:
            noise = torch.randn_like(data)
            noise = noise[couple(noise, data, kind, beta, props, columns)]
        moments.append(target_moment(noise, data))
        distances.append(pair_distance(noise, data))
    return float(np.mean(moments)), float(np.mean(distances))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    prefix = args.prefix or f"coupling_{args.modality}"
    outdir = args.outdir or plots_dir() / "coupling"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nCoupling diagnostic -- {args.modality}")
    z1, aux = load(args)
    dim = int(np.prod(z1.shape[1:]))
    print(f"  {len(z1)} samples   latent {tuple(z1.shape[1:])} = {dim} dims"
          f"   {aux.shape[1]} property column(s)   device {DEVICE}")
    source = InformedSource.fit(z1, aux)
    print(f"  informed source residual sigma {source.sigma:.4f}")
    print(f"  reference: pi/2 = {PI_OVER_2:.4f} (Gaussian floor, independent "
          f"coupling only)")

    rows = []
    for label, kind, beta in variants(args):
        for batch in args.batch_sizes:
            moment, distance = measure(z1, aux, kind, beta, batch,
                                       args.repeats, args.columns, source)
            rows.append({"modality": args.modality, "coupling": label,
                         "kind": kind, "beta": beta, "batch_size": batch,
                         "latent_dim": dim, "repeats": args.repeats,
                         "target_moment": round(moment, 6),
                         "pair_distance": round(distance, 6),
                         "pi_over_2": round(PI_OVER_2, 6)})

    base = {r["batch_size"]: r["target_moment"] for r in rows
            if r["coupling"] == "independent"}
    for r in rows:
        r["ratio_to_independent"] = round(r["target_moment"]
                                          / base[r["batch_size"]], 4)

    header = (f"\n  {'coupling':<14}{'batch':>7}{'target E[(z1-z0)^2]':>22}"
              f"{'vs independent':>16}")
    print(header + "\n  " + "-" * (len(header) - 3))
    for r in rows:
        print(f"  {r['coupling']:<14}{r['batch_size']:>7}"
              f"{r['target_moment']:>22.4f}{r['ratio_to_independent']:>15.3f}x")

    csv_path = outdir / f"{prefix}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for label, _, _ in variants(args):
        sel = [r for r in rows if r["coupling"] == label]
        ax.plot([r["batch_size"] for r in sel],
                [r["ratio_to_independent"] for r in sel],
                marker="o", label=label)
    ax.axhline(1.0, color="k", lw=0.9, ls=":", label="independent baseline")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("minibatch size (OT's bias shrinks to the right)")
    ax.set_ylabel("target moment, relative to independent")
    ax.set_title(f"{args.modality}: what each coupling does to the "
                 f"regression target\n(lower is a smaller target; "
                 f"no training involved)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = outdir / f"{prefix}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {fig_path}")
    print("\nA coupling that does not move the ratio cannot straighten the path.")
    print("Read the ratio, then test it where it is supposed to pay off: "
          "quality against --steps.")


if __name__ == "__main__":
    main()
