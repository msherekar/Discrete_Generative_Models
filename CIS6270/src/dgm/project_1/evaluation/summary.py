"""Plot 10: the one-page summary panel.

The figure to look at first: every headline metric for both methods and all
guidance modes on a single sheet.
"""
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .loaders import composition_proxies
from .metrics import _pairwise_hamming, _positional_entropy
from .output import _save
from .style import (GUIDANCE_MODES, METHOD_COLORS, MODE_COLORS,
                    POLAR_RESIDUES, plt)

def plot_summary_panel(flow_data, diff_data, train_seqs, out_dir, prefix="", run_tag=""):
    """One-page overview figure for the report."""
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    # ── A: r1 by mode ─────────────────────────────────────────────────────────
    ax_r1 = fig.add_subplot(gs[0, 0])
    modes_plus = GUIDANCE_MODES + ["train"]
    tp = composition_proxies(train_seqs)
    x  = np.arange(len(modes_plus))
    w  = 0.3

    r1_f = [composition_proxies(flow_data["sequences"][m])[:, 0].mean().item()
            for m in GUIDANCE_MODES] + [tp[:, 0].mean().item()]
    r1_d = [composition_proxies(diff_data["sequences"][m])[:, 0].mean().item()
            for m in GUIDANCE_MODES] + [float("nan")]
    ax_r1.bar(x - w/2, r1_f, w, color=METHOD_COLORS["flow"],      label="Flow", alpha=0.85)
    ax_r1.bar(x + w/2, r1_d, w, color=METHOD_COLORS["diffusion"], label="Diff", alpha=0.85)
    ax_r1.set_xticks(x)
    ax_r1.set_xticklabels(["CFG", "Single", "Multi", "Train"], rotation=10)
    ax_r1.set_ylabel("Mean r₁"); ax_r1.set_title("(A) Charge proxy r₁")
    ax_r1.legend(fontsize=8); ax_r1.axhline(0, color="gray", lw=0.7, ls=":")

    # ── B: r2 by mode ─────────────────────────────────────────────────────────
    ax_r2 = fig.add_subplot(gs[0, 1])
    r2_f = [composition_proxies(flow_data["sequences"][m])[:, 1].mean().item()
            for m in GUIDANCE_MODES] + [tp[:, 1].mean().item()]
    r2_d = [composition_proxies(diff_data["sequences"][m])[:, 1].mean().item()
            for m in GUIDANCE_MODES] + [float("nan")]
    ax_r2.bar(x - w/2, r2_f, w, color=METHOD_COLORS["flow"],      label="Flow", alpha=0.85)
    ax_r2.bar(x + w/2, r2_d, w, color=METHOD_COLORS["diffusion"], label="Diff", alpha=0.85)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(["CFG", "Single", "Multi", "Train"], rotation=10)
    ax_r2.set_ylabel("Mean r₂"); ax_r2.set_title("(B) Polar fraction r₂")
    ax_r2.legend(fontsize=8)

    # ── C: r1 vs r2 scatter ───────────────────────────────────────────────────
    ax_sc = fig.add_subplot(gs[0, 2])
    ax_sc.scatter(tp[:, 0].numpy(), tp[:, 1].numpy(),
                  color=MODE_COLORS["train"], s=25, alpha=0.5, marker="s", label="Train")
    for mode in GUIDANCE_MODES:
        pf = composition_proxies(flow_data["sequences"][mode]).numpy()
        pd = composition_proxies(diff_data["sequences"][mode]).numpy()
        ax_sc.scatter(pf[:, 0], pf[:, 1], color=MODE_COLORS[mode], s=60,
                      marker="o", alpha=0.8)
        ax_sc.scatter(pd[:, 0], pd[:, 1], color=MODE_COLORS[mode], s=60,
                      marker="^", alpha=0.8)
    legend_els = [Patch(color=MODE_COLORS[m], label=m.upper()) for m in GUIDANCE_MODES] + \
                 [Patch(color=MODE_COLORS["train"], label="Train")] + \
                 [Line2D([0], [0], marker="o", color="gray", label="Flow", linestyle="none"),
                  Line2D([0], [0], marker="^", color="gray", label="Diff", linestyle="none")]
    ax_sc.legend(handles=legend_els, fontsize=7, ncol=2)
    ax_sc.set_xlabel("r₁"); ax_sc.set_ylabel("r₂")
    ax_sc.set_title("(C) r₁ vs r₂ (Pareto view)")

    # ── D: Positional entropy ─────────────────────────────────────────────────
    ax_ent = fig.add_subplot(gs[1, 0])
    train_ent = _positional_entropy(train_seqs)
    ax_ent.plot(range(1, len(train_ent) + 1), train_ent,
                color=MODE_COLORS["train"], lw=1.5, ls="--", label="Train")
    for mode in GUIDANCE_MODES:
        fe = _positional_entropy(flow_data["sequences"][mode])
        ax_ent.plot(range(1, len(fe) + 1), fe,
                    color=MODE_COLORS[mode], lw=1.8, label=f"Flow/{mode.upper()}")
    ax_ent.set_xlabel("Position"); ax_ent.set_ylabel("Shannon entropy (bits)")
    ax_ent.set_title("(D) Per-position entropy (Flow)")
    ax_ent.legend(fontsize=8)

    # ── E: Polar count violin ─────────────────────────────────────────────────
    ax_pol = fig.add_subplot(gs[1, 1])
    for offset, (data, marker) in enumerate([(flow_data, "Flow"), (diff_data, "Diff")]):
        counts = [[sum(a in POLAR_RESIDUES for a in s)
                   for s in data["sequences"][m]] for m in GUIDANCE_MODES]
        pos = [1 + offset * 0.4 + i for i in range(len(GUIDANCE_MODES))]
        parts = ax_pol.violinplot(counts, positions=pos, widths=0.35,
                                  showmeans=True, showmedians=False)
        for body, mode in zip(parts["bodies"], GUIDANCE_MODES):
            body.set_facecolor(MODE_COLORS[mode]); body.set_alpha(0.6)
    min_polar = flow_data.get("min_polar", 12)
    ax_pol.axhline(min_polar, color="red", lw=1.5, ls="--", label=f"threshold={min_polar}")
    ax_pol.set_ylabel("# polar residues"); ax_pol.set_title("(E) Polar residue count")
    ax_pol.legend(fontsize=8)

    # ── F: Hamming diversity ──────────────────────────────────────────────────
    ax_ham = fig.add_subplot(gs[1, 2])
    for method, data, marker in [("flow", flow_data, "o"), ("diffusion", diff_data, "^")]:
        means = [np.mean(_pairwise_hamming(data["sequences"][m])) for m in GUIDANCE_MODES]
        ax_ham.plot(GUIDANCE_MODES, means,
                    color=METHOD_COLORS[method], marker=marker, lw=2,
                    label=method.capitalize())
    ax_ham.set_ylabel("Mean pairwise Hamming"); ax_ham.set_title("(F) Sequence diversity")
    ax_ham.legend(fontsize=9)

    tag_line = f"  [{run_tag}]" if run_tag else ""
    fig.suptitle(
        f"Project 1 Summary: Flow Matching vs Diffusion on ESM-2 Protein Embeddings{tag_line}",
        fontsize=15, fontweight="bold"
    )
    _save(fig, out_dir, "00_summary_panel.png", prefix)
