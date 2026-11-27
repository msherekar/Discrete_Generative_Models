"""Writing a figure or a table to disk.

Every plot and every CSV in this package goes out through these two functions,
so naming and the "saved" line stay uniform.
"""
import csv
from pathlib import Path

from .style import plt

def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path}")


def _save(fig, out_dir, name, prefix=""):
    filename = f"{prefix}_{name}" if prefix else name
    p = Path(out_dir) / filename
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {p}")
