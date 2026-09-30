"""The model registry, the palette and the two plot helpers.

Parameter counts live here because they are the x-axis of every figure in this
analysis.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
