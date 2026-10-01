#!/usr/bin/env python3
"""Aggregate a multi-seed sweep from either modality and produce figures.

Reads run directories named <prefix>_<variant>_<eta>_s<seed> and reports, per
guidance strength, the effect of guidance measured against that run's own
unguided arm:

    gain = property(single) - property(cfg)

cfg is run with eta=0, so it is the control for its own seed and the comparison
is free of seed-to-seed differences in the base model.

Both modalities use a property that is an exact function of the sample, so there
is no oracle error in these numbers:

  protein  r1 = (count(K)+count(R)-count(D)-count(E)) / length      net charge
  image    mean pixel intensity                                     ink

The modality is detected from the contents of results.pt, so the same command
works for peptides and for MNIST.

Usage:
  dgm-analyze-sweep --prefix ep --seeds 11 12 13 14 15 --etas 1 5 20 50
  dgm-analyze-sweep --prefix mn --modality image --seeds 11 12 13
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dgm.common.paths import SHARED_DATA, project_dir

ROOT = project_dir()
VARIANT_LABEL = {"st": "state-based", "ep": "endpoint", "base": "baseline"}
COLORS = {"st": "tab:gray", "ep": "tab:blue", "base": "tab:gray"}
MODES = ("cfg", "single", "multi")


def charge(seq):
    return (sum(a in "KR" for a in seq) - sum(a in "DE" for a in seq)) / len(seq)


def load_run(run_dir, method):
    """Return {mode: property array} and the per-epoch losses, or None."""
    protein = run_dir / method / "results.pt"
    if protein.is_file():
        saved = torch.load(protein, weights_only=False, map_location="cpu")
        seqs = saved.get("sequences")
        if seqs is None:
            return None
        return ({m: np.array([charge(s) for s in seqs[m]]) for m in MODES if m in seqs},
                saved.get("losses", []), "protein")
    image = run_dir / "results.pt"
    if image.is_file():
        saved = torch.load(image, weights_only=False, map_location="cpu")
        block = saved.get("results", {}).get(method)
        if block is None:
            return None
        return ({m: block[m]["properties"][:, 0].numpy() for m in MODES if m in block},
                block.get("losses", []), "image")
    return None


def real_image_reference(data_dir, n=2000):
    """Ink and saturation of real MNIST, the target a guided sample should match.

    Guidance optimizes a property; nothing in the objective asks the sample to
    remain in the data distribution. On a bounded property (peptide net charge,
    capped by sequence length) that cannot go far wrong. On an unbounded one
    (mean pixel intensity) the optimum is a blank white image, which scores
    perfectly and is not a digit. Reporting gain without this reference makes the
    most degenerate setting look like the best one.
    """
    from torchvision import datasets
    from torchvision.transforms import v2
    tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                     v2.Normalize((0.5,), (0.5,))])
    mnist = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=tf)
    x = torch.stack([mnist[i][0] for i in range(min(n, len(mnist)))])
    return {"ink": x.flatten(1).mean().item(),
            "saturation": (x > 0.9).float().mean().item()}


def image_fidelity(run_dir, method, mode="single"):
    """Absolute ink and saturated-pixel fraction of a run's generated images."""
    saved = torch.load(run_dir / "results.pt", weights_only=False, map_location="cpu")
    img = saved["results"][method][mode]["images"]
    return img.flatten(1).mean().item(), (img > 0.9).float().mean().item()


def mean_ci(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return np.nan, 0.0
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(a.size))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True)
    p.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    p.add_argument("--variants", nargs="+", default=["st", "ep"])
    p.add_argument("--etas", type=float, nargs="+", default=[1, 5, 20, 50])
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    p.add_argument("--methods", nargs="+", default=["flow", "diffusion"])
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=SHARED_DATA,
                   help="Where MNIST lives, for the real-data fidelity reference")
    p.add_argument("--images", action="store_true",
                   help="Also save a grid of generated images (image modality only)")
    args = p.parse_args()
    outdir = args.outdir or ROOT / "plots" / f"{args.prefix}_sweep"
    outdir.mkdir(parents=True, exist_ok=True)

    def fmt_eta(e):
        return str(int(e)) if float(e).is_integer() else str(e)

    rows, modality, missing = [], None, []
    for method in args.methods:
        for v in args.variants:
            for e in args.etas:
                for s in args.seeds:
                    d = args.outputs / f"{args.prefix}_{v}_{fmt_eta(e)}_s{s}"
                    loaded = load_run(d, method)
                    if loaded is None:
                        missing.append(d.name)
                        continue
                    props, losses, modality = loaded
                    if "cfg" not in props or "single" not in props:
                        continue
                    row = {"method": method, "variant": v, "eta": e, "seed": s,
                           "gain": props["single"].mean() - props["cfg"].mean(),
                           "unguided": props["cfg"].mean(),
                           "guided": props["single"].mean(),
                           "loss_final": losses[-1] if len(losses) else np.nan}
                    if modality == "image":
                        ink, sat = image_fidelity(d, method)
                        row["ink"] = ink
                        row["saturation"] = sat
                    rows.append(row)
    if not rows:
        raise SystemExit(f"No runs found under {args.outputs} with prefix "
                         f"'{args.prefix}'. Checked e.g. "
                         f"{args.prefix}_{args.variants[0]}_{fmt_eta(args.etas[0])}"
                         f"_s{args.seeds[0]}")
    if missing:
        print(f"  {len(missing)} run(s) missing, e.g. {missing[0]}")
    label = "net charge" if modality == "protein" else "mean intensity (ink)"

    fig, axes = plt.subplots(1, len(args.methods), figsize=(6.2 * len(args.methods), 4.6),
                             squeeze=False)
    for ax, method in zip(axes[0], args.methods):
        print(f"\n=== {method} ({modality}) ===")
        head = f"{'eta':>7}  " + "  ".join(f"{VARIANT_LABEL.get(v, v):>24}"
                                                for v in args.variants)
        print(head + "\n" + "-" * len(head))
        for e in args.etas:
            line = f"{fmt_eta(e):>7}  "
            for v in args.variants:
                g = [r["gain"] for r in rows
                     if r["method"] == method and r["variant"] == v and r["eta"] == e]
                m, ci = mean_ci(g)
                line += f"{f'{m:+.4f} +/- {ci:.4f} (n={len(g)})':>24}  "
            print(line)
        for v in args.variants:
            pts = [mean_ci([r["gain"] for r in rows if r["method"] == method
                            and r["variant"] == v and r["eta"] == e]) for e in args.etas]
            ax.errorbar(args.etas, [x[0] for x in pts], yerr=[x[1] for x in pts],
                        marker="o", capsize=4, color=COLORS.get(v),
                        label=VARIANT_LABEL.get(v, v))
        ax.axhline(0, color="k", lw=0.8, ls=":")
        if min(args.etas) > 0:
            ax.set_xscale("log")
        ax.set_xlabel("guidance strength eta")
        ax.set_ylabel(f"guidance gain in {label}")
        ax.set_title(f"{method}: {len(args.seeds)} seeds, 95% interval")
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{args.prefix}_gain_vs_eta.png", dpi=150)
    plt.close(fig)

    if modality == "image":
        ref = real_image_reference(args.data_dir)
        print(f"\nFIDELITY -- real MNIST: ink {ref['ink']:+.4f}, "
              f"{100*ref['saturation']:.1f}% of pixels > 0.9")
        print(f"{'method':<11}{'var':<5}{'eta':>6}{'gain':>9}{'ink':>9}"
              f"{'sat%':>8}{'sat/real':>10}{'verdict':>11}")
        print("-" * 69)
        for method in args.methods:
            for v in args.variants:
                for e in args.etas:
                    sel = [r for r in rows if r["method"] == method
                           and r["variant"] == v and r["eta"] == e]
                    if not sel:
                        continue
                    g = np.mean([r["gain"] for r in sel])
                    ink = np.mean([r["ink"] for r in sel])
                    sat = np.mean([r["saturation"] for r in sel])
                    ratio = sat / ref["saturation"]
                    verdict = ("ok" if ratio < 1.5 else
                               "drifting" if ratio < 2.5 else "BLOBS")
                    print(f"{method:<11}{v:<5}{fmt_eta(e):>6}{g:>+9.3f}{ink:>9.3f}"
                          f"{100*sat:>7.1f}%{ratio:>9.2f}x{verdict:>11}")
            print()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for method in args.methods:
            for v in args.variants:
                xs, ys = [], []
                for e in args.etas:
                    sel = [r for r in rows if r["method"] == method
                           and r["variant"] == v and r["eta"] == e]
                    if sel:
                        xs.append(np.mean([r["gain"] for r in sel]))
                        ys.append(np.mean([r["saturation"] for r in sel])
                                  / ref["saturation"])
                ax.plot(xs, ys, marker="o",
                        ls="-" if method == "flow" else "--",
                        color=COLORS.get(v),
                        label=f"{method} / {VARIANT_LABEL.get(v, v)}")
        ax.axhline(1.0, color="g", lw=1, ls=":", label="real MNIST")
        ax.axhline(2.5, color="r", lw=1, ls=":", label="degenerate")
        ax.set_xlabel("guidance gain (the optimized property)")
        ax.set_ylabel("saturated pixels, relative to real MNIST")
        ax.set_title("Guidance vs fidelity: up and to the right is reward hacking")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"{args.prefix}_gain_vs_fidelity.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {outdir}/{args.prefix}_gain_vs_fidelity.png")

    with (outdir / f"{args.prefix}_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    if args.images and modality == "image":
        d = args.outputs / (f"{args.prefix}_{args.variants[-1]}_"
                            f"{fmt_eta(args.etas[-1])}_s{args.seeds[0]}")
        saved = torch.load(d / "results.pt", weights_only=False, map_location="cpu")
        fig, axes = plt.subplots(len(args.methods), len(MODES),
                                 figsize=(3.2 * len(MODES), 3.4 * len(args.methods)),
                                 squeeze=False)
        for i, method in enumerate(args.methods):
            for j, mode in enumerate(MODES):
                img = saved["results"][method][mode]["images"][:16]
                grid = img.reshape(4, 4, 28, 28).permute(0, 2, 1, 3).reshape(112, 112)
                axes[i][j].imshow(grid.clamp(-1, 1), cmap="gray", vmin=-1, vmax=1)
                axes[i][j].set_title(f"{method} / {mode}")
                axes[i][j].axis("off")
        fig.suptitle(f"generated samples, eta={fmt_eta(args.etas[-1])}")
        fig.tight_layout()
        fig.savefig(outdir / f"{args.prefix}_samples.png", dpi=150)
        plt.close(fig)

    print(f"\nWrote {outdir}/{args.prefix}_gain_vs_eta.png")
    print(f"      {outdir}/{args.prefix}_sweep.csv")


if __name__ == "__main__":
    main()
