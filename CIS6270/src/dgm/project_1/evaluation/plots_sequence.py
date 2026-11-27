"""Plots 7-8: per-position entropy and pairwise-Hamming diversity.

Both ask whether the samples are actually different from each other, which is
the first thing a collapsed decoder gets wrong.
"""
import numpy as np

from .metrics import _pairwise_hamming, _positional_entropy
from .output import _save
from .style import GUIDANCE_LABELS, GUIDANCE_MODES, MODE_COLORS, plt

def plot_positional_entropy(flow_data, diff_data, train_seqs, out_dir, prefix=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    positions = None

    for ax, data, method_label in zip(
        axes, [flow_data, diff_data], ["Flow Matching", "Diffusion (DDPM)"]
    ):
        train_ent = _positional_entropy(train_seqs)
        positions = range(1, len(train_ent) + 1)
        ax.fill_between(positions, train_ent, alpha=0.2,
                        color=MODE_COLORS["train"], label="Training data")
        ax.plot(positions, train_ent, color=MODE_COLORS["train"],
                lw=1.2, ls="--")
        for mode in GUIDANCE_MODES:
            ent = _positional_entropy(data["sequences"][mode])
            ax.plot(range(1, len(ent) + 1), ent,
                    color=MODE_COLORS[mode], lw=2, label=GUIDANCE_LABELS[mode])
        ax.set_xlabel("Sequence position")
        ax.set_ylabel("Shannon entropy (bits)")
        ax.set_title(method_label)
        ax.legend(fontsize=9)
        ax.set_xlim(1, len(train_ent))

    fig.suptitle("Per-Position Amino Acid Entropy (higher = more diverse)", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir, "07_positional_entropy.png", prefix)



def plot_sequence_diversity(flow_data, diff_data, train_seqs, out_dir, prefix=""):
    modes_plus = GUIDANCE_MODES + ["train"]
    fig, axes  = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, data, method_label in zip(
        axes, [flow_data, diff_data], ["Flow Matching", "Diffusion (DDPM)"]
    ):
        all_dists = []
        labs      = []
        colors    = []
        for m in modes_plus:
            seqs = data["sequences"].get(m, train_seqs) if m != "train" else train_seqs
            d    = _pairwise_hamming(seqs)
            all_dists.append(d)
            labs.append(GUIDANCE_LABELS[m])
            colors.append(MODE_COLORS[m])

        parts = ax.violinplot(all_dists, showmeans=True, showmedians=True)
        for body, c in zip(parts["bodies"], colors):
            body.set_facecolor(c); body.set_alpha(0.7)
        for j, (dists, color) in enumerate(zip(all_dists, colors)):
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(dists))
            ax.scatter(np.full(len(dists), j + 1) + jitter, dists,
                       color=color, s=25, alpha=0.6, zorder=3)
        ax.set_xticks(range(1, len(labs) + 1))
        ax.set_xticklabels(labs, rotation=15, ha="right")
        ax.set_ylabel("Pairwise Hamming distance")
        ax.set_title(method_label)

    fig.suptitle("Sequence Diversity: Pairwise Hamming Distance Distribution", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir, "08_sequence_diversity.png", prefix)
