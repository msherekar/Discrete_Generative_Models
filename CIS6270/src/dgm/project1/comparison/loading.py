"""Reading the per-model CSVs that evaluate.py wrote.

A missing model is simply absent from the result rather than an error, so the
scaling plots draw whatever subset of the sweep has finished.
"""
import csv
from pathlib import Path

from .style import MODEL_ORDER

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
