"""Shared constants, colours and the matplotlib style for every figure.

One palette in one place, so the method and guidance-mode colours mean the
same thing in every plot of the study.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import torch                                           # noqa: E402

from dgm.common.paths import lecture_dir, plots_dir     # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
AMINO_ACIDS   = "ACDEFGHIKLMNPQRSTVWY"
POLAR_RESIDUES = "DEHKNQRST"
GUIDANCE_MODES = ["cfg", "single", "multi"]
GUIDANCE_LABELS = {
    "cfg":    "CFG (w=2)",
    "single": "Single-obj (η=1, λ₁=1)",
    "multi":  "Multi-obj  (η=1, λ=0.7/0.3)",
    "train":  "Training data",
}
MODE_COLORS  = {"cfg": "#4CAF50", "single": "#FF9800", "multi": "#9C27B0", "train": "#607D8B"}
METHOD_COLORS = {"flow": "#1565C0", "diffusion": "#B71C1C"}
METHOD_MARKERS = {"flow": "o", "diffusion": "^"}

ROOT      = plots_dir().parent      # the project's artifact directory
LECTURE3  = lecture_dir(3)
ESM_NAME  = "facebook/esm2_t6_8M_UR50D"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE, HIDDEN, LEARNING_RATE = 16, 128, 1e-3
CONDITION_DROP = 0.2

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

