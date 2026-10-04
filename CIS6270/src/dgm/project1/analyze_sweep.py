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


def nfe_for(run_dir, method, steps):
    """Network forward passes this run actually spent, from its saved config.

    The cost unit the matched protocol is defined on. Read from the run rather
    than assumed, because the same `steps` means different costs depending on
    the solver (Heun evaluates twice per step), on whether classifier-free
    guidance was on (two branches), and on whether the diffusion chain was
    ancestral (all K levels, ignoring `steps` entirely).
    """
    from dgm.project1.pipeline.solvers import expected_nfe
    try:
        results = torch.load(run_dir / method / "results.pt", map_location="cpu",
                             weights_only=False)
        config = results.get("config", {}) or {}
    except Exception:
        config = {}
    if method == "diffusion":
        solver = config.get("diffusion_solver", "ddpm")
        if solver == "ddpm":
            # The ancestral chain walks every level whatever --steps says.
            return float(config.get("diffusion_steps", 1000))
        steps = config.get("sample_steps") or steps
        per_step = 2 if solver == "heun" else 1
    else:
        solver = config.get("flow_solver", "euler")
        steps = config.get("steps", steps)
        per_step = 1
        return float(expected_nfe(int(steps), solver,
                                  cfg=bool(config.get("cfg_weight")),
                                  endpoint=bool(config.get("endpoint_guidance")),
                                  guided=bool(config.get("reward_eta"))))
    cfg = 2 if config.get("cfg_weight") else 1
    return float(int(steps) * per_step * cfg)


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


def reference_property(modality, run_dir, method, data_dir, n=2000):
    """The property distribution a good sampler should reproduce.

    A path or coupling study runs at eta=0, where the guided arm IS the unguided
    arm, so `gain` is identically zero and measures nothing. What such a study
    changes is how faithfully the sampler reproduces the DATA at a given step
    count, which needs the data's own distribution as the reference -- this one.

    Real MNIST ink for images; the training CSV's net charge for peptides, read
    from the dataset path the run recorded in its own config.
    """
    if modality == "image":
        from torchvision import datasets
        from torchvision.transforms import v2
        tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                         v2.Normalize((0.5,), (0.5,))])
        mnist = datasets.MNIST(root=str(data_dir), train=True, download=True,
                               transform=tf)
        x = torch.stack([mnist[i][0] for i in range(min(n, len(mnist)))])
        return x.flatten(1).mean(1).numpy()
    saved = torch.load(run_dir / method / "results.pt", weights_only=False,
                       map_location="cpu")
    dataset = Path(saved.get("config", {}).get("dataset", ""))
    if not dataset.is_file():
        dataset = ROOT.parent / "lecture" / "lecture_3" / "esm2_example.csv"
    rows = list(csv.DictReader(dataset.open(newline="")))
    return np.array([charge(r["sequence"].strip().upper()) for r in rows])


def fidelity_w1(generated, reference):
    """Wasserstein-1 distance between two property samples, in reference sds.

    A distribution comparison rather than a difference of means: at a low step
    count a sampler can land the mean and still produce the wrong spread, and
    that is exactly the failure a few-step study is looking for. Zero is a
    perfect match; the scale is the data's own standard deviation, so the number
    is comparable across modalities.
    """
    from scipy.stats import wasserstein_distance
    sd = float(np.std(reference)) or 1.0
    return float(wasserstein_distance(generated, reference)) / sd


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
    p.add_argument("--x-axis", default="eta", choices=("eta", "steps", "nfe"),
                   help="What the run-directory level means and what the figure "
                        "plots against (default: eta). 'steps' is the axis a "
                        "coupling claim lives on: couplings reach the same "
                        "marginal given enough integration steps and differ in "
                        "how few they need, so quality at a fixed 200 steps is "
                        "the one measurement that cannot show the effect. "
                        "'nfe' is network forward passes, the only unit on "
                        "which flow and diffusion are comparable at all -- "
                        "Lecture 2.1: 'every velocity evaluation requires a "
                        "neural-network evaluation'.")
    p.add_argument("--steps", type=int, nargs="+", default=[10, 20, 50, 200],
                   help="Integration-step levels, used when --x-axis steps.")
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

    # One level list drives the directory names, the table and the x axis, so a
    # second axis costs a name and a label rather than a parallel code path.
    levels = [float(v) for v in
              (args.steps if args.x_axis in ("steps", "nfe") else args.etas)]
    x_label = {"steps": "integration steps",
               "nfe": "network forward passes (NFE)",
               "eta": "guidance strength eta"}[args.x_axis]

    if args.x_axis == "steps" and "diffusion" in args.methods:
        # The old behaviour, and still correct for runs made with
        # --diffusion-solver ddpm: the ancestral chain walks all K levels
        # whatever --steps says, so its runs at every level are the same run.
        print("  [skip] diffusion on a steps axis: --steps drives the FLOW "
              "integrator only when the reverse chain is ancestral DDPM. "
              "Re-run the sweep with --diffusion-solver ddim and plot "
              "--x-axis nfe to compare the two methods at matched cost.")
        args.methods = [m for m in args.methods if m != "diffusion"] or ["flow"]

    rows, modality, missing = [], None, []
    reference = {}
    for method in args.methods:
        for v in args.variants:
            for e in levels:
                for s in args.seeds:
                    d = args.outputs / f"{args.prefix}_{v}_{fmt_eta(e)}_s{s}"
                    loaded = load_run(d, method)
                    if loaded is None:
                        missing.append(d.name)
                        continue
                    props, losses, modality = loaded
                    if "cfg" not in props or "single" not in props:
                        continue
                    if method not in reference:
                        reference[method] = reference_property(
                            modality, d, method, args.data_dir)
                    row = {"method": method, "variant": v, "eta": e,
                           "level": nfe_for(d, method, e) if args.x_axis == "nfe"
                                    else e,
                           "x_axis": args.x_axis, "seed": s,
                           "gain": props["single"].mean() - props["cfg"].mean(),
                           "unguided": props["cfg"].mean(),
                           "guided": props["single"].mean(),
                           # Distribution fidelity of the UNGUIDED arm: what a
                           # path or coupling actually changes.
                           "w1": fidelity_w1(props["cfg"], reference[method]),
                           "loss_final": losses[-1] if len(losses) else np.nan}
                    if modality == "image":
                        ink, sat = image_fidelity(d, method)
                        row["ink"] = ink
                        row["saturation"] = sat
                    rows.append(row)
    if not rows:
        raise SystemExit(f"No runs found under {args.outputs} with prefix "
                         f"'{args.prefix}'. Checked e.g. "
                         f"{args.prefix}_{args.variants[0]}_{fmt_eta(levels[0])}"
                         f"_s{args.seeds[0]}")
    if missing:
        print(f"  {len(missing)} run(s) missing, e.g. {missing[0]}")
    label = "net charge" if modality == "protein" else "mean intensity (ink)"
    # On a steps axis the study is about reproducing the data, not about
    # guidance, so the reported quantity changes with the axis.
    metric = "w1" if args.x_axis in ("steps", "nfe") else "gain"
    y_label = (f"W1({label}) to the data, in sds  [lower is better]"
               if metric == "w1" else f"guidance gain in {label}")
    if metric == "gain" and all(r["gain"] == 0 for r in rows):
        print("\n  [warn] every gain is exactly zero: the runs have eta=0, so "
              "the guided arm IS the unguided arm. A guidance metric cannot "
              "measure anything here -- read the w1 column instead.")

    fig, axes = plt.subplots(1, len(args.methods), figsize=(6.2 * len(args.methods), 4.6),
                             squeeze=False)
    for ax, method in zip(axes[0], args.methods):
        print(f"\n=== {method} ({modality}) ===")
        head = f"{args.x_axis:>7}  " + "  ".join(f"{VARIANT_LABEL.get(v, v):>24}"
                                                for v in args.variants)
        print(head + "\n" + "-" * len(head))
        for e in levels:
            line = f"{fmt_eta(e):>7}  "
            for v in args.variants:
                g = [r[metric] for r in rows
                     if r["method"] == method and r["variant"] == v and r["eta"] == e]
                m, ci = mean_ci(g)
                line += f"{f'{m:+.4f} +/- {ci:.4f} (n={len(g)})':>24}  "
            print(line)
        for v in args.variants:
            pts = [mean_ci([r[metric] for r in rows if r["method"] == method
                            and r["variant"] == v and r["eta"] == e]) for e in levels]
            ax.errorbar(levels, [x[0] for x in pts], yerr=[x[1] for x in pts],
                        marker="o", capsize=4, color=COLORS.get(v),
                        label=VARIANT_LABEL.get(v, v))
        ax.axhline(0, color="k", lw=0.8, ls=":",
                   label="perfect match" if metric == "w1" else None)
        if min(levels) > 0:
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{method}: {len(args.seeds)} seeds, 95% interval")
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{args.prefix}_gain_vs_{args.x_axis}.png", dpi=150)
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
                for e in levels:
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
                for e in levels:
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
                            f"{fmt_eta(levels[-1])}_s{args.seeds[0]}")
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
        fig.suptitle(f"generated samples, {args.x_axis}={fmt_eta(levels[-1])}")
        fig.tight_layout()
        fig.savefig(outdir / f"{args.prefix}_samples.png", dpi=150)
        plt.close(fig)

    print(f"\nWrote {outdir}/{args.prefix}_gain_vs_{args.x_axis}.png")
    print(f"      {outdir}/{args.prefix}_sweep.csv")


if __name__ == "__main__":
    main()
