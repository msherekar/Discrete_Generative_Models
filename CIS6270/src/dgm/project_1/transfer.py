#!/usr/bin/env python3
"""Does the innovation's benefit transfer to the second modality?

The project asks for an innovation evaluated on one continuous modality and
then transferred to a second "to test whether the improvement generalizes".
Putting both modalities on one figure takes two corrections, and without them
the comparison argues the wrong way.

1. The property units differ -- peptide net charge against mean pixel ink --
   so a raw gain cannot be compared. This plots a standardized effect size,
   the gain in units of the unguided arm's own spread:

       d = (mean(guided) - mean(unguided)) / sd(unguided)

   measured within each seed against that seed's own eta=0 control, so it
   carries no seed-to-seed model difference.

2. Nominal eta is not comparable across modalities either. cli.py states why:
   under --normalize-guidance the reward gradient is a unit vector, so its step
   is the same absolute size at any dimension while the latent norm grows as
   sqrt(D). A 237x320 protein latent has norm ~275 against MNIST's ~28, so the
   same eta moves a protein about ten times less. The x-axis is therefore
   eta / sqrt(D), the relative size of the guidance step.

The figure shows both axes side by side: nominal eta on the left, where the
two modalities look unrelated, and the rescaled axis on the right, where the
question "does it transfer?" can actually be read.

Flow and diffusion are pooled in each series, because the question asked here
is about the innovation rather than the method; dgm-analyze-sweep already
separates the methods into panels. The CSV keeps the method column, so a
per-method split is a groupby away.

Usage:
  dgm-transfer --protein-prefix ep --image-prefix mn \\
      --variants st ep --etas 1 5 20 50 --seeds 11 12 13 14 15
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from dgm.common.paths import outputs_dir, plots_dir

from .analyze_sweep import COLORS, VARIANT_LABEL, load_run, mean_ci

MODALITY_MARKER = {"protein": "o", "image": "^"}


def fmt_eta(eta):
    """Run directories spell an integral eta without its decimal point.

    A local copy: analyze_sweep defines this inside main(), so it cannot be
    imported, and duplicating four characters of formatting is better than
    reaching into that module.
    """
    return str(int(eta)) if float(eta).is_integer() else str(eta)


def latent_dim(run_dir: Path, method: str) -> int:
    """Dimensionality of the space guidance acts in, for the eta rescaling."""
    import torch
    protein = run_dir / method / "results.pt"
    if protein.is_file():
        saved = torch.load(protein, weights_only=False, map_location="cpu")
        return int(saved["length"]) * int(saved["dim"])
    image = run_dir / "results.pt"
    if image.is_file():
        saved = torch.load(image, weights_only=False, map_location="cpu")
        block = saved["results"][method]
        for mode in ("cfg", "single", "multi"):
            if mode in block:
                return int(np.prod(block[mode]["images"].shape[1:]))
    raise SystemExit(f"cannot read a latent dimension from {run_dir}")


def collect(outputs: Path, prefix: str, variants, etas, seeds, methods):
    """Per-seed standardized effect sizes for one modality's sweep."""
    rows, missing, dims = [], [], {}
    for method in methods:
        for variant in variants:
            for eta in etas:
                for seed in seeds:
                    run = outputs / f"{prefix}_{variant}_{fmt_eta(eta)}_s{seed}"
                    loaded = load_run(run, method)
                    if loaded is None:
                        missing.append(run.name)
                        continue
                    props, _losses, modality = loaded
                    if "cfg" not in props or "single" not in props:
                        continue
                    unguided, guided = props["cfg"], props["single"]
                    spread = float(unguided.std(ddof=1)) if unguided.size > 1 else 0.0
                    if spread <= 0:
                        continue
                    dims.setdefault(modality, latent_dim(run, method))
                    rows.append({
                        "modality": modality, "method": method,
                        "variant": variant, "eta": float(eta), "seed": seed,
                        "raw_gain": float(guided.mean() - unguided.mean()),
                        "effect_size": float(
                            (guided.mean() - unguided.mean()) / spread),
                        "unguided_sd": spread,
                        "latent_dim": dims[modality],
                        "eta_scaled": float(eta) / np.sqrt(dims[modality]),
                    })
    return rows, missing, dims


def _series(rows, modality, variant, etas, key):
    points = []
    for eta in etas:
        values = [r[key] for r in rows
                  if r["modality"] == modality and r["variant"] == variant
                  and r["eta"] == float(eta)]
        points.append(mean_ci(values) if values else (np.nan, 0.0))
    return points


def plot_transfer(rows, variants, etas, dims, outdir: Path, prefix: str):
    """Two panels: nominal eta, then eta rescaled by sqrt(latent dimension)."""
    from .evaluation.style import plt

    modalities = sorted({r["modality"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)

    for ax, (xkey, title) in zip(axes, (
            ("eta", "Nominal eta: the two modalities look unrelated"),
            ("eta_scaled", r"eta / $\sqrt{D}$: comparable guidance step"))):
        for modality in modalities:
            scale = (1.0 if xkey == "eta"
                     else 1.0 / np.sqrt(dims.get(modality, 1)))
            for variant in variants:
                points = _series(rows, modality, variant, etas, "effect_size")
                if all(np.isnan(p[0]) for p in points):
                    continue
                ax.errorbar([float(e) * scale for e in etas],
                            [p[0] for p in points], yerr=[p[1] for p in points],
                            marker=MODALITY_MARKER.get(modality, "s"),
                            color=COLORS.get(variant), capsize=4,
                            ls="-" if modality == "protein" else "--",
                            label=f"{modality} / {VARIANT_LABEL.get(variant, variant)}")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("guidance strength eta" if xkey == "eta"
                      else r"eta / $\sqrt{D}$")
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("effect size: gain / sd of the unguided arm")
    axes[0].legend(fontsize=8)
    fig.suptitle("Does the innovation transfer across modalities?", fontsize=13)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{prefix}_14_cross_modality_transfer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protein-prefix", default="ep",
                        help="Run-name prefix of the protein sweep.")
    parser.add_argument("--image-prefix", default="mn",
                        help="Run-name prefix of the MNIST sweep.")
    parser.add_argument("--variants", nargs="+", default=["st", "ep"],
                        help="The innovation axis, as in dgm-analyze-sweep.")
    parser.add_argument("--etas", type=float, nargs="+",
                        default=[1, 5, 20, 50])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[11, 12, 13, 14, 15])
    parser.add_argument("--methods", nargs="+", default=["flow", "diffusion"])
    parser.add_argument("--outputs", type=Path, default=None,
                        help="Where run directories live. Default: outputs/")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Default: plots/transfer/")
    parser.add_argument("--prefix", default="transfer",
                        help="Filename prefix (default: transfer).")
    return parser


def main():
    args = build_parser().parse_args()
    outputs = args.outputs or outputs_dir()

    rows, missing, dims = [], [], {}
    for prefix in (args.protein_prefix, args.image_prefix):
        got, gone, found = collect(outputs, prefix, args.variants, args.etas,
                                   args.seeds, args.methods)
        rows += got
        missing += gone
        dims.update(found)
        print(f"{prefix:>12}: {len(got)} runs" +
              (f", {len(gone)} missing" if gone else ""))

    if not rows:
        raise SystemExit(
            f"no runs found under {outputs}. Expected directories like "
            f"{args.protein_prefix}_{args.variants[0]}_"
            f"{fmt_eta(args.etas[0])}_s{args.seeds[0]}")
    if len({r["modality"] for r in rows}) < 2:
        print("\n  [warn] only one modality present, so this figure cannot "
              "show transfer. Run the second modality's sweep first "
              "(MODALITY=image bash run_sweep.sh).")

    for modality, dim in sorted(dims.items()):
        print(f"  {modality:<8} latent dim {dim:>7}  sqrt {np.sqrt(dim):>7.1f}")

    outdir = args.outdir or plots_dir(create=True) / "transfer"
    plot_transfer(rows, args.variants, args.etas, dims, outdir, args.prefix)

    columns = ["modality", "method", "variant", "eta", "eta_scaled", "seed",
               "raw_gain", "effect_size", "unguided_sd", "latent_dim"]
    path = outdir / f"{args.prefix}_cross_modality_transfer.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path}")


if __name__ == "__main__":
    main()
