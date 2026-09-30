"""The four single-metric scaling plots: composition, polar counts, Hamming
distance and positional entropy.

Each reads one metric CSV per model and plots it against parameter count.
"""
from pathlib import Path

import numpy as np

from .loading import _float
from .style import (GUIDANCE_MODES, GUIDE_LS, GUIDE_MARKER, METHOD_COLOR,
                    METHODS, MODEL_ORDER, PARAM_M, _log_xaxis, _save, plt)

# ── Individual plots ─────────────────────────────────────────────────────────

def plot_composition_scaling(data: dict, out_dir: Path):
    """r1 and r2 mean vs model size, split by method and guidance mode."""
    models = [m for m in MODEL_ORDER if m in data]
    xs = [PARAM_M[m] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Composition Proxies vs Model Size", fontweight="bold")

    for ax, metric, ylabel, title in zip(
        axes,
        ["r1", "r2"],
        ["Mean r1 (charge proxy)", "Mean r2 (polar fraction)"],
        ["Charge Proxy (r1)", "Polar Fraction (r2)"],
    ):
        for method in METHODS:
            for gmode in GUIDANCE_MODES:
                ys, errs = [], []
                for m in models:
                    rows = [r for r in data[m]["composition_proxies"]
                            if r["method"] == method and r["guidance_mode"] == gmode]
                    vals = [_float(r[metric]) for r in rows]
                    ys.append(np.nanmean(vals) if vals else float("nan"))
                    errs.append(np.nanstd(vals) if vals else float("nan"))
                label = f"{method}/{gmode}"
                color = METHOD_COLOR[method]
                alpha = 1.0 if method == "flow" else 0.6
                ax.errorbar(xs, ys, yerr=errs,
                            marker=GUIDE_MARKER[gmode], linestyle=GUIDE_LS[gmode],
                            color=color, alpha=alpha, label=label, capsize=3)

        _log_xaxis(ax, models)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "scaling_01_composition_proxies.png")


def plot_polar_count_scaling(data: dict, out_dir: Path):
    """Mean polar count per sequence vs model size."""
    models = [m for m in MODEL_ORDER if m in data]
    xs = [PARAM_M[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Polar Residue Count vs Model Size", fontweight="bold")

    for method in METHODS:
        for gmode in GUIDANCE_MODES:
            ys = []
            for m in models:
                rows = [r for r in data[m]["polar_counts"]
                        if r["method"] == method and r["guidance_mode"] == gmode]
                vals = [_float(r["polar_count"]) for r in rows]
                ys.append(np.nanmean(vals) if vals else float("nan"))
            ax.plot(xs, ys, marker=GUIDE_MARKER[gmode], linestyle=GUIDE_LS[gmode],
                    color=METHOD_COLOR[method],
                    alpha=1.0 if method == "flow" else 0.6,
                    label=f"{method}/{gmode}")

    _log_xaxis(ax, models)
    ax.set_ylabel("Mean polar residues per sequence")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "scaling_02_polar_counts.png")


def plot_hamming_scaling(data: dict, out_dir: Path):
    """Mean Hamming diversity vs model size."""
    models = [m for m in MODEL_ORDER if m in data]
    xs = [PARAM_M[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Sequence Diversity (Mean Hamming) vs Model Size", fontweight="bold")

    for method in METHODS:
        for gmode in GUIDANCE_MODES:
            ys = []
            for m in models:
                rows = [r for r in data[m]["hamming_diversity"]
                        if r["method"] == method and r["guidance_mode"] == gmode]
                vals = [_float(r["hamming"]) for r in rows]
                ys.append(np.nanmean(vals) if vals else float("nan"))
            ax.plot(xs, ys, marker=GUIDE_MARKER[gmode], linestyle=GUIDE_LS[gmode],
                    color=METHOD_COLOR[method],
                    alpha=1.0 if method == "flow" else 0.6,
                    label=f"{method}/{gmode}")

    _log_xaxis(ax, models)
    ax.set_ylabel("Mean pairwise Hamming distance")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "scaling_03_hamming_diversity.png")


def plot_entropy_scaling(data: dict, out_dir: Path):
    """Mean positional entropy vs model size."""
    models = [m for m in MODEL_ORDER if m in data]
    xs = [PARAM_M[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Mean Positional Entropy vs Model Size", fontweight="bold")

    for method in METHODS:
        for gmode in GUIDANCE_MODES:
            ys = []
            for m in models:
                rows = [r for r in data[m]["positional_entropy"]
                        if r["method"] == method and r["guidance_mode"] == gmode]
                vals = [_float(r["entropy_bits"]) for r in rows]
                ys.append(np.nanmean(vals) if vals else float("nan"))
            ax.plot(xs, ys, marker=GUIDE_MARKER[gmode], linestyle=GUIDE_LS[gmode],
                    color=METHOD_COLOR[method],
                    alpha=1.0 if method == "flow" else 0.6,
                    label=f"{method}/{gmode}")

    _log_xaxis(ax, models)
    ax.set_ylabel("Mean positional entropy (bits)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "scaling_04_positional_entropy.png")
