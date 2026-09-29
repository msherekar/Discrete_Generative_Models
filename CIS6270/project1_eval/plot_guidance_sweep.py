#!/usr/bin/env python3
"""Aggregate a multi-seed guidance sweep and plot it.

Reads runs named <prefix>_<variant>_<eta>_s<seed> and reports, for each guidance
strength, the effect of guidance relative to that run's own unguided arm:

    gain = mean r1(single) - mean r1(cfg)

cfg is run with w=0 and eta=0, so it is the unguided control for its own seed,
which removes seed-to-seed differences in the base model from the comparison.

r1 is an exact residue count -- (K+R-D-E)/L -- so there is no oracle error here
and nothing to calibrate. Diffusion additionally reports how many samples
numerically diverged, since that is where the two variants differ most.

Usage:
  python plot_guidance_sweep.py --prefix ep --seeds 11 12 13 14 15 --etas 1 5 20 50
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
POLAR = "DEHKNQRST"
VARIANTS = {"st": "state-based", "ep": "endpoint"}
COLORS = {"st": "tab:gray", "ep": "tab:blue"}


def r1(seq):
    return (sum(a in "KR" for a in seq) - sum(a in "DE" for a in seq)) / len(seq)


def read_fasta(path):
    out, current = [], ""
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current:
                out.append(current)
            current = ""
        else:
            current += line.strip()
    return out + ([current] if current else [])


def run_dir(outputs, prefix, variant, eta, seed):
    return outputs / f"{prefix}_{variant}_{eta}_s{seed}"


def collect(outputs, prefix, variants, etas, seeds, method):
    """gain[variant][eta] -> list over seeds; plus diversity and divergence."""
    gain = {v: {e: [] for e in etas} for v in variants}
    uniq = {v: {e: [] for e in etas} for v in variants}
    diverged = {v: {e: [] for e in etas} for v in variants}
    missing = []
    for v in variants:
        for e in etas:
            for s in seeds:
                d = run_dir(outputs, prefix, v, e, s) / method
                if not (d / "cfg.fasta").is_file() or not (d / "single.fasta").is_file():
                    missing.append(str(d))
                    continue
                c = np.array([r1(x) for x in read_fasta(d / "cfg.fasta")])
                g = read_fasta(d / "single.fasta")
                gain[v][e].append(np.mean([r1(x) for x in g]) - c.mean())
                uniq[v][e].append(len(set(g)) / len(g))
                results = d / "results.pt"
                if results.is_file():
                    saved = torch.load(results, weights_only=False, map_location="cpu")
                    z = saved["standardized_latents"]["single"]
                    n = z.reshape(len(z), -1).norm(dim=1)
                    diverged[v][e].append(float((n > 3 * n.median()).float().mean()))
    return gain, uniq, diverged, missing


def mean_err(values):
    """Mean and a 95% interval; falls back to 0 width for a single replicate."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return np.nan, 0.0
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(a.size))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default="ep")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--etas", type=int, nargs="+", default=[1, 5, 20, 50])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    parser.add_argument("--variants", nargs="+", default=["st", "ep"])
    parser.add_argument("--outdir", type=Path, default=ROOT / "plots" / "guidance_sweep")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    summary_rows = []
    for method, ax in zip(("flow", "diffusion"), axes[:2]):
        gain, uniq, diverged, missing = collect(
            args.outputs, args.prefix, args.variants, args.etas, args.seeds, method)
        if missing:
            print(f"  [{method}] {len(missing)} run(s) missing, e.g. {missing[0]}")
        print(f"\n=== {method} ===")
        print(f"{'eta':>5}" + "".join(f"{VARIANTS.get(v, v):>22}" for v in args.variants))
        for e in args.etas:
            row = f"{e:>5}"
            for v in args.variants:
                m, err = mean_err(gain[v][e])
                row += f"{f'{m:+.4f} +/- {err:.4f}':>22}"
                summary_rows.append({"method": method, "eta": e, "variant": v,
                                     "gain": m, "err": err,
                                     "n_seeds": len(gain[v][e])})
            print(row)
        for v in args.variants:
            m = [mean_err(gain[v][e]) for e in args.etas]
            ax.errorbar(args.etas, [x[0] for x in m], yerr=[x[1] for x in m],
                        marker="o", capsize=4, label=VARIANTS.get(v, v),
                        color=COLORS.get(v))
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("guidance strength eta (log scale)")
        ax.set_ylabel("guidance gain in r1  (guided - unguided)")
        ax.set_title(f"{method}: effect of guidance\n"
                     f"mean over {len(args.seeds)} seeds, 95% interval")
        ax.legend()

    # Divergence is a diffusion-only story; show it beside the two gain panels.
    _, _, diverged, _ = collect(args.outputs, args.prefix, args.variants,
                                args.etas, args.seeds, "diffusion")
    ax = axes[2]
    for v in args.variants:
        m = [mean_err(diverged[v][e]) for e in args.etas]
        ax.errorbar(args.etas, [100 * x[0] for x in m], yerr=[100 * x[1] for x in m],
                    marker="s", capsize=4, label=VARIANTS.get(v, v), color=COLORS.get(v))
    ax.set_xscale("log")
    ax.set_xlabel("guidance strength eta (log scale)")
    ax.set_ylabel("% of samples numerically diverged")
    ax.set_title("diffusion: sampler stability\n(latent norm > 3x median)")
    ax.legend()

    fig.tight_layout()
    out = args.outdir / f"{args.prefix}_guidance_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    import csv
    csv_path = args.outdir / f"{args.prefix}_guidance_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {out}\n      {csv_path}")


if __name__ == "__main__":
    main()
