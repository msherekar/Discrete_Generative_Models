"""The summary panel and the summary CSV.

The one figure and the one table to read first: every metric for every model
side by side.
"""
import csv
from pathlib import Path

import numpy as np

from .loading import _float
from .style import (GUIDANCE_MODES, METHOD_COLOR, METHODS, MODEL_ORDER,
                    PARAM_M, _log_xaxis, _save, plt)

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
