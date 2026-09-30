"""Cross-model scaling analysis for Project 1.

Reads all esm2_*_<dataset> CSV outputs and produces scaling plots with
x-axis = parameter count (log scale).

Usage:
    python project1_eval/compare_models.py
    python project1_eval/compare_models.py --dataset toy64 --outdir scaling_plots/
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Model registry (param count in millions) ─────────────────────────────────
MODEL_ORDER = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3b"]
PARAM_M = {"esm2_8m": 8, "esm2_35m": 35, "esm2_150m": 150,
           "esm2_650m": 650, "esm2_3b": 3000}
PARAM_LABEL = {"esm2_8m": "8M", "esm2_35m": "35M", "esm2_150m": "150M",
               "esm2_650m": "650M", "esm2_3b": "3B"}

GUIDANCE_MODES = ["cfg", "single", "multi"]
METHODS = ["flow", "diffusion"]

METHOD_COLOR = {"flow": "#2196F3", "diffusion": "#FF5722"}
GUIDE_MARKER = {"cfg": "o", "single": "s", "multi": "^"}
GUIDE_LS = {"cfg": "-", "single": "--", "multi": ":"}


# ── CSV loaders ───────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ── Data aggregation ─────────────────────────────────────────────────────────

def load_all(plots_dir: Path, dataset: str) -> dict:
    """Load all CSVs for every available model into a structured dict."""
    data = {}
    for model in MODEL_ORDER:
        folder = plots_dir / f"{model}_{dataset}"
        if not folder.is_dir():
            continue
        prefix = f"{model}_{dataset}"
        data[model] = {
            "composition_proxies":  _read_csv(folder / f"{prefix}_composition_proxies.csv"),
            "polar_counts":         _read_csv(folder / f"{prefix}_polar_counts.csv"),
            "hamming_diversity":    _read_csv(folder / f"{prefix}_hamming_diversity.csv"),
            "positional_entropy":   _read_csv(folder / f"{prefix}_positional_entropy.csv"),
            "training_loss":        _read_csv(folder / f"{prefix}_training_loss.csv"),
            "ablation_cfg_weight":  _read_csv(folder / f"{prefix}_ablation_cfg_weight.csv"),
            "ablation_reward_eta":  _read_csv(folder / f"{prefix}_ablation_reward_eta.csv"),
        }
    return data


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _log_xaxis(ax, models):
    xs = [PARAM_M[m] for m in models]
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(
        lambda v, _: PARAM_LABEL.get(
            next((m for m in models if PARAM_M[m] == v), ""), f"{int(v)}M"
        )
    ))
    ax.set_xlabel("ESM-2 model size (parameters)")


def _save(fig, out_dir: Path, name: str):
    p = out_dir / name
    fig.savefig(p, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print(f"  Saved {p}")


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


def plot_summary_scaling(data: dict, out_dir: Path):
    """5-panel summary: r1, r2, Hamming, entropy, final-loss — cfg mode only."""
    models = [m for m in MODEL_ORDER if m in data]
    xs = [PARAM_M[m] for m in models]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("ESM-2 Model Scaling — Summary (toy64 dataset, CFG mode)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()

    panels = [
        ("r1", "composition_proxies", "r1", "Charge proxy r1 (CFG)"),
        ("r2", "composition_proxies", "r2", "Polar fraction r2 (CFG)"),
        ("hamming", "hamming_diversity", "hamming", "Mean Hamming diversity (CFG)"),
        ("entropy", "positional_entropy", "entropy_bits", "Mean positional entropy (CFG)"),
        ("flow_loss", "training_loss", "flow_loss", "Final flow loss"),
        ("diff_loss", "training_loss", "diff_loss", "Final diffusion loss"),
    ]

    for ax, (panel_id, csv_key, col, title) in zip(axes, panels):
        for method in METHODS:
            ys = []
            for m in models:
                if panel_id in ("flow_loss", "diff_loss"):
                    rows = data[m][csv_key]
                    if rows:
                        ys.append(_float(rows[-1].get(col, "nan")))
                    else:
                        ys.append(float("nan"))
                else:
                    rows = [r for r in data[m][csv_key]
                            if r.get("method") == method
                            and r.get("guidance_mode") == "cfg"]
                    vals = [_float(r[col]) for r in rows]
                    ys.append(np.nanmean(vals) if vals else float("nan"))

            if panel_id in ("flow_loss", "diff_loss"):
                mk = "flow" if panel_id.startswith("flow") else "diffusion"
                ax.plot(xs, ys, marker="o", linewidth=2, color=METHOD_COLOR[mk])
                break
            else:
                ax.plot(xs, ys, marker="o", linewidth=2,
                        color=METHOD_COLOR[method], label=method)
        ax.set_title(title, fontsize=9)
        _log_xaxis(ax, models)
        ax.grid(True, alpha=0.3)
        if panel_id not in ("flow_loss", "diff_loss"):
            ax.legend(fontsize=7)

    axes[-1].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir, "scaling_00_summary.png")


def save_summary_csv(data: dict, out_dir: Path, dataset: str):
    """Write a single aggregated CSV with one row per (model, method, guidance_mode)."""
    rows_out = []
    for model in MODEL_ORDER:
        if model not in data:
            continue
        for method in METHODS:
            for gmode in GUIDANCE_MODES:
                row = {"model": model, "params_M": PARAM_M[model],
                       "method": method, "guidance_mode": gmode}

                # composition
                cp = [r for r in data[model]["composition_proxies"]
                      if r["method"] == method and r["guidance_mode"] == gmode]
                row["mean_r1"] = np.nanmean([_float(r["r1"]) for r in cp]) if cp else ""
                row["mean_r2"] = np.nanmean([_float(r["r2"]) for r in cp]) if cp else ""

                # polar
                pc = [r for r in data[model]["polar_counts"]
                      if r["method"] == method and r["guidance_mode"] == gmode]
                row["mean_polar_count"] = np.nanmean([_float(r["polar_count"]) for r in pc]) if pc else ""

                # hamming
                hd = [r for r in data[model]["hamming_diversity"]
                      if r["method"] == method and r["guidance_mode"] == gmode]
                row["mean_hamming"] = np.nanmean([_float(r["hamming"]) for r in hd]) if hd else ""

                # entropy
                pe = [r for r in data[model]["positional_entropy"]
                      if r["method"] == method and r["guidance_mode"] == gmode]
                row["mean_entropy_bits"] = np.nanmean([_float(r["entropy_bits"]) for r in pe]) if pe else ""

                rows_out.append(row)

        # losses (not per guidance mode)
        tl = data[model]["training_loss"]
        if tl:
            last = tl[-1]
            rows_out[-len(GUIDANCE_MODES * len(METHODS))]["final_flow_loss"] = _float(last.get("flow_loss", ""))
            rows_out[-len(GUIDANCE_MODES * len(METHODS))]["final_diff_loss"] = _float(last.get("diff_loss", ""))

    if not rows_out:
        return
    out_path = out_dir / f"scaling_summary_{dataset}.csv"
    fieldnames = list(rows_out[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-model scaling analysis")
    parser.add_argument("--plots-dir", default=None,
                        help="Directory containing esm2_*_<dataset> sub-folders "
                             "(default: project1_eval/plots/ relative to script)")
    parser.add_argument("--dataset", default="toy64",
                        help="Dataset tag (default: toy64)")
    parser.add_argument("--outdir", default=None,
                        help="Output directory for scaling plots "
                             "(default: <plots-dir>/scaling_<dataset>/)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    plots_dir = Path(args.plots_dir) if args.plots_dir else script_dir / "plots"
    out_dir = Path(args.outdir) if args.outdir else plots_dir / f"scaling_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSVs from: {plots_dir}")
    data = load_all(plots_dir, args.dataset)

    if not data:
        print(f"No data found for dataset '{args.dataset}' in {plots_dir}.", file=sys.stderr)
        print("Expected folders like: esm2_8m_toy64/, esm2_35m_toy64/, ...", file=sys.stderr)
        sys.exit(1)

    found = list(data.keys())
    print(f"Found {len(found)} model(s): {', '.join(found)}")
    print(f"Writing plots to: {out_dir}\n")

    plot_composition_scaling(data, out_dir)
    plot_polar_count_scaling(data, out_dir)
    plot_hamming_scaling(data, out_dir)
    plot_entropy_scaling(data, out_dir)
    plot_loss_convergence_scaling(data, out_dir)
    plot_ablation_sensitivity_scaling(data, out_dir)
    plot_summary_scaling(data, out_dir)
    save_summary_csv(data, out_dir, args.dataset)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} plots + 1 CSV saved to {out_dir}")


if __name__ == "__main__":
    main()
