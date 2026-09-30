"""Plots 1-6: losses, composition, latent PCA, amino-acid usage, polar counts,
and the r1/r2 Pareto view.

These six read only what every run saves, so they work on any results.pt
regardless of which guidance arms it carries.
"""
import math
from collections import Counter

import numpy as np

from .loaders import composition_proxies
from .output import _save
from .style import (AMINO_ACIDS, GUIDANCE_LABELS, GUIDANCE_MODES,
                    METHOD_COLORS, MODE_COLORS, POLAR_RESIDUES, plt)

# ── Plot 1: Training loss curves ───────────────────────────────────────────────
def plot_training_loss(flow_losses, diff_losses, out_dir, prefix=""):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(flow_losses) + 1)
    ax.plot(epochs, flow_losses, color=METHOD_COLORS["flow"],      lw=2, label="Flow Matching")
    ax.plot(epochs, diff_losses, color=METHOD_COLORS["diffusion"], lw=2, label="Diffusion (DDPM)", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Combined Training Loss")
    ax.set_title("Training Loss: Flow Matching vs Diffusion")
    ax.legend()
    ax.set_yscale("log")
    _save(fig, out_dir, "01_training_loss.png", prefix)


# ── Plot 2: Composition proxy bar chart ───────────────────────────────────────
def plot_composition_proxies(flow_data, diff_data, train_seqs, out_dir, prefix=""):
    modes_plus = GUIDANCE_MODES + ["train"]
    r1_flow, r2_flow, r1_diff, r2_diff = [], [], [], []
    for m in GUIDANCE_MODES:
        fp = composition_proxies(flow_data["sequences"][m])
        dp = composition_proxies(diff_data["sequences"][m])
        r1_flow.append(fp[:, 0].mean().item()); r2_flow.append(fp[:, 1].mean().item())
        r1_diff.append(dp[:, 0].mean().item()); r2_diff.append(dp[:, 1].mean().item())
    tp = composition_proxies(train_seqs)
    r1_flow.append(tp[:, 0].mean().item()); r2_flow.append(tp[:, 1].mean().item())
    r1_diff.append(float("nan"));           r2_diff.append(float("nan"))

    x      = np.arange(len(modes_plus))
    width  = 0.22
    labels = [GUIDANCE_LABELS[m] for m in modes_plus]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, r_flow, r_diff, title, ylabel, metric in zip(
        axes,
        [r1_flow, r2_flow], [r1_diff, r2_diff],
        ["r₁: Charge proxy (larger = more K/R, fewer D/E)",
         "r₂: Polar fraction (larger = more polar/charged residues)"],
        ["Mean charge proxy (r₁)", "Mean polar fraction (r₂)"],
        ["r1", "r2"],
    ):
        bars_f = ax.bar(x - width / 2, r_flow, width, label="Flow Matching",
                        color=METHOD_COLORS["flow"], alpha=0.85)
        bars_d = ax.bar(x + width / 2, r_diff, width, label="Diffusion",
                        color=METHOD_COLORS["diffusion"], alpha=0.85)
        for bar in list(bars_f) + list(bars_d):
            h = bar.get_height()
            if not math.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.legend()
    fig.suptitle("Composition Proxy Comparison by Guidance Mode", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "02_composition_proxies.png", prefix)


# ── Plot 3: Latent space PCA ───────────────────────────────────────────────────
def plot_latent_pca(flow_data, diff_data, out_dir, prefix=""):
    from sklearn.decomposition import PCA  # type: ignore[import]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, method_label in zip(
        axes,
        [flow_data, diff_data],
        ["Flow Matching", "Diffusion (DDPM)"],
    ):
        all_latents, all_labels = [], []
        for mode in GUIDANCE_MODES:
            lats = data["standardized_latents"][mode]   # [n, L, D]
            lats_mean = lats.mean(1).numpy()            # [n, D]
            all_latents.append(lats_mean)
            all_labels.extend([mode] * len(lats_mean))

        X = np.concatenate(all_latents, axis=0)
        pca = PCA(n_components=2)
        Z2  = pca.fit_transform(X)
        var = pca.explained_variance_ratio_ * 100

        offset = 0
        for mode in GUIDANCE_MODES:
            n = data["standardized_latents"][mode].shape[0]
            ax.scatter(Z2[offset:offset + n, 0], Z2[offset:offset + n, 1],
                       color=MODE_COLORS[mode], label=GUIDANCE_LABELS[mode],
                       s=80, edgecolors="white", linewidths=0.5, zorder=3)
            offset += n

        ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)")
        ax.set_title(method_label)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Latent Space PCA: Generated Sequences by Guidance Mode", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir, "03_latent_pca.png", prefix)


# ── Plot 4: Amino acid composition heatmap ─────────────────────────────────────
def plot_aa_composition(flow_data, diff_data, train_seqs, out_dir, prefix=""):
    def aa_freq(seq_list):
        total = sum(len(s) for s in seq_list)
        counts = Counter("".join(seq_list))
        return np.array([counts.get(a, 0) / total for a in AMINO_ACIDS])

    row_labels = [f"Flow – {GUIDANCE_LABELS[m]}" for m in GUIDANCE_MODES] + \
                 [f"Diff – {GUIDANCE_LABELS[m]}" for m in GUIDANCE_MODES] + \
                 ["Training data"]
    freqs = [aa_freq(flow_data["sequences"][m]) for m in GUIDANCE_MODES] + \
            [aa_freq(diff_data["sequences"][m]) for m in GUIDANCE_MODES] + \
            [aa_freq(train_seqs)]
    matrix = np.stack(freqs)

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(AMINO_ACIDS)))
    ax.set_xticklabels(list(AMINO_ACIDS))
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    fig.colorbar(im, ax=ax, label="Residue frequency")
    ax.set_xlabel("Amino acid")
    ax.set_title("Amino Acid Composition Heatmap")
    # Divider line between flow and diffusion rows
    ax.axhline(2.5, color="white", lw=2)
    ax.axhline(5.5, color="white", lw=2, linestyle="--")
    fig.tight_layout()
    _save(fig, out_dir, "04_aa_composition_heatmap.png", prefix)


# ── Plot 5: Polar residue count distribution ──────────────────────────────────
def plot_polar_residue_distribution(flow_data, diff_data, out_dir, prefix=""):
    def polar_counts(seq_list):
        return [sum(a in POLAR_RESIDUES for a in s) for s in seq_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, data, method_label in zip(
        axes, [flow_data, diff_data], ["Flow Matching", "Diffusion (DDPM)"]
    ):
        all_counts = [polar_counts(data["sequences"][m]) for m in GUIDANCE_MODES]
        positions  = range(len(GUIDANCE_MODES))
        parts = ax.violinplot(all_counts, positions=list(positions),
                              showmeans=True, showmedians=True, showextrema=True)
        for i, (body, mode) in enumerate(zip(parts["bodies"], GUIDANCE_MODES)):
            body.set_facecolor(MODE_COLORS[mode])
            body.set_alpha(0.7)
        # Scatter individual points
        for i, (counts, mode) in enumerate(zip(all_counts, GUIDANCE_MODES)):
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(counts))
            ax.scatter(np.full(len(counts), i) + jitter, counts,
                       color=MODE_COLORS[mode], s=30, zorder=3, alpha=0.8)
        min_polar = flow_data.get("min_polar", 12)
        ax.axhline(min_polar, color="red", lw=1.5, ls="--", label=f"min_polar = {min_polar}")
        ax.set_xticks(list(positions))
        ax.set_xticklabels([GUIDANCE_LABELS[m] for m in GUIDANCE_MODES], rotation=12, ha="right")
        ax.set_ylabel("Number of polar/charged residues")
        ax.set_title(method_label)
        ax.legend()

    fig.suptitle("Polar/Charged Residue Count Distribution per Guidance Mode", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir, "05_polar_residue_distribution.png", prefix)


# ── Plot 6: r1 vs r2 scatter (multi-objective Pareto view) ───────────────────
def plot_reward_pareto(flow_data, diff_data, train_seqs, out_dir, prefix=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    tp = composition_proxies(train_seqs).numpy()

    for ax, data, method_label in zip(
        axes, [flow_data, diff_data], ["Flow Matching", "Diffusion (DDPM)"]
    ):
        # Training data backdrop
        ax.scatter(tp[:, 0], tp[:, 1], color=MODE_COLORS["train"], s=35,
                   alpha=0.5, marker="s", label="Training data", zorder=1)
        for mode in GUIDANCE_MODES:
            p = composition_proxies(data["sequences"][mode]).numpy()
            ax.scatter(p[:, 0], p[:, 1], color=MODE_COLORS[mode],
                       s=80, label=GUIDANCE_LABELS[mode], edgecolors="white",
                       linewidths=0.5, zorder=3)
            # Annotate centroid
            ax.scatter(p[:, 0].mean(), p[:, 1].mean(),
                       color=MODE_COLORS[mode], s=200, marker="*",
                       edgecolors="black", linewidths=0.5, zorder=4)
        ax.set_xlabel("r₁ (charge proxy)")
        ax.set_ylabel("r₂ (polar fraction)")
        ax.set_title(method_label)
        ax.legend(loc="upper left", fontsize=9)
        ax.axvline(0, color="gray", lw=0.7, ls=":")

    fig.suptitle("Reward Pareto View: r₁ vs r₂ by Guidance Mode\n(★ = centroid)", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir, "06_reward_pareto_scatter.png", prefix)
