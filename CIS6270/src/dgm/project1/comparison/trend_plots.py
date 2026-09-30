"""The two multi-panel scaling figures: loss convergence and guidance
sensitivity.

Both read several metrics per model, so they are kept apart from the
single-metric plots in scaling_plots.py.
"""
from pathlib import Path

from .loading import _float
from .style import (METHOD_COLOR, METHODS, MODEL_ORDER, PARAM_LABEL, PARAM_M,
                    _log_xaxis, _save, plt)

def plot_loss_convergence_scaling(data: dict, out_dir: Path):
    """Final training loss (last-epoch average) vs model size."""
    models = [m for m in MODEL_ORDER if m in data]
    xs = [PARAM_M[m] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training Loss Convergence vs Model Size", fontweight="bold")

    for ax, loss_key, title, ylabel in zip(
        axes,
        ["flow_loss", "diff_loss"],
        ["Flow Matching — final loss", "Diffusion — final loss"],
        ["Final flow loss", "Final diffusion loss"],
    ):
        final_vals = []
        for m in models:
            rows = data[m]["training_loss"]
            if not rows:
                final_vals.append(float("nan"))
                continue
            # last epoch
            last = rows[-1]
            final_vals.append(_float(last.get(loss_key, "nan")))
        method_key = "flow" if loss_key.startswith("flow") else "diffusion"
        ax.plot(xs, final_vals, marker="o", color=METHOD_COLOR[method_key],
                linewidth=2)
        _log_xaxis(ax, models)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "scaling_05_loss_convergence.png")

    # Also plot full loss curves per model on one panel for each method
    for method, loss_key in [("flow", "flow_loss"), ("diffusion", "diff_loss")]:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        ax2.set_title(f"{method.capitalize()} Training Loss Curves by Model Size",
                      fontweight="bold")
        cmap = plt.colormaps["plasma"].resampled(len(models))
        for i, m in enumerate(models):
            rows = data[m]["training_loss"]
            if not rows:
                continue
            epochs = [_float(r["epoch"]) for r in rows]
            losses = [_float(r[loss_key]) for r in rows]
            ax2.plot(epochs, losses, color=cmap(i), linewidth=1.5,
                     label=PARAM_LABEL[m])
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(title="Model size")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        _save(fig2, out_dir, f"scaling_05b_{method}_loss_curves.png")


def plot_ablation_sensitivity_scaling(data: dict, out_dir: Path):
    """Max change in r1/r2 over cfg_weight sweep vs model size (sensitivity)."""
    models = [m for m in MODEL_ORDER if m in data]
    xs = [PARAM_M[m] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("CFG Guidance Sensitivity vs Model Size\n"
                 "(Δr across cfg_weight sweep)", fontweight="bold")

    for ax, metric, ylabel in zip(
        axes, ["mean_r1", "mean_r2"], ["Δ r1 (max − min)", "Δ r2 (max − min)"]
    ):
        for method in METHODS:
            ys = []
            for m in models:
                rows = [r for r in data[m]["ablation_cfg_weight"]
                        if r["method"] == method]
                vals = [_float(r[metric]) for r in rows]
                if vals:
                    ys.append(max(vals) - min(vals))
                else:
                    ys.append(float("nan"))
            ax.plot(xs, ys, marker="o", color=METHOD_COLOR[method],
                    linewidth=2, label=method)
        _log_xaxis(ax, models)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "scaling_06_cfg_sensitivity.png")

    # Reward eta sensitivity
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Reward-Eta Sensitivity vs Model Size\n"
                  "(Δr across eta sweep)", fontweight="bold")

    for ax, metric, ylabel in zip(
        axes2, ["mean_r1", "mean_r2"], ["Δ r1 (max − min)", "Δ r2 (max − min)"]
    ):
        for method in METHODS:
            ys = []
            for m in models:
                rows = [r for r in data[m]["ablation_reward_eta"]
                        if r["method"] == method]
                vals = [_float(r[metric]) for r in rows]
                if vals:
                    ys.append(max(vals) - min(vals))
                else:
                    ys.append(float("nan"))
            ax.plot(xs, ys, marker="o", color=METHOD_COLOR[method],
                    linewidth=2, label=method)
        _log_xaxis(ax, models)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    _save(fig2, out_dir, "scaling_07_eta_sensitivity.png")
