"""One CSV per metric, written before the figures.

The plots are a view of these tables, not the other way round: whatever a
figure shows can be recomputed or replotted from the CSVs alone.
"""
from collections import Counter
from pathlib import Path
from itertools import combinations

import numpy as np

from .loaders import composition_proxies
from .metrics import _hamming, _positional_entropy
from .output import _write_csv
from .style import AMINO_ACIDS, GUIDANCE_MODES, POLAR_RESIDUES

# ══════════════════════════════════════════════════════════════════════════════
# Raw-data export  (one CSV per metric — load later for custom plots)
# ══════════════════════════════════════════════════════════════════════════════



def save_raw_data(flow_data, diff_data, train_seqs, out_dir, prefix,
                  flow_losses=None, diff_losses=None):
    """Write one CSV per metric into out_dir.  All files share the run prefix.

    Files produced
    ──────────────
    {prefix}_sequences.csv          every generated sequence + r1/r2/polar_count
    {prefix}_composition_proxies.csv  per-sequence r1, r2 incl. training data
    {prefix}_aa_composition.csv     per-amino-acid frequency per method/mode
    {prefix}_polar_counts.csv       polar residue count per sequence
    {prefix}_positional_entropy.csv  Shannon entropy at each sequence position
    {prefix}_hamming_diversity.csv  pairwise Hamming distances within each group
    {prefix}_latent_pca.csv         2-D PCA coordinates (requires scikit-learn)
    {prefix}_training_loss.csv      epoch-level losses (only when losses passed in)
    """
    out_dir = Path(out_dir)

    def _fp(tag):
        return out_dir / f"{prefix}_{tag}.csv"

    # ── 1. Sequences ──────────────────────────────────────────────────────────
    rows = []
    for method, data in [("flow", flow_data), ("diffusion", diff_data)]:
        for mode in GUIDANCE_MODES:
            for idx, seq in enumerate(data["sequences"][mode]):
                p = composition_proxies([seq])[0]
                rows.append({
                    "method": method, "guidance_mode": mode,
                    "sample_idx": idx, "sequence": seq,
                    "r1": round(p[0].item(), 6), "r2": round(p[1].item(), 6),
                    "polar_count": sum(a in POLAR_RESIDUES for a in seq),
                    "seq_length": len(seq),
                })
    for idx, seq in enumerate(train_seqs):
        p = composition_proxies([seq])[0]
        rows.append({
            "method": "training", "guidance_mode": "train",
            "sample_idx": idx, "sequence": seq,
            "r1": round(p[0].item(), 6), "r2": round(p[1].item(), 6),
            "polar_count": sum(a in POLAR_RESIDUES for a in seq),
            "seq_length": len(seq),
        })
    _write_csv(_fp("sequences"), rows)

    # ── 2. Composition proxies (mean ± std per group) ─────────────────────────
    rows = []
    for method, data in [("flow", flow_data), ("diffusion", diff_data)]:
        for mode in GUIDANCE_MODES:
            p = composition_proxies(data["sequences"][mode])
            for idx in range(len(p)):
                rows.append({
                    "method": method, "guidance_mode": mode, "sample_idx": idx,
                    "r1": round(p[idx, 0].item(), 6),
                    "r2": round(p[idx, 1].item(), 6),
                })
    tp = composition_proxies(train_seqs)
    for idx in range(len(tp)):
        rows.append({
            "method": "training", "guidance_mode": "train", "sample_idx": idx,
            "r1": round(tp[idx, 0].item(), 6), "r2": round(tp[idx, 1].item(), 6),
        })
    _write_csv(_fp("composition_proxies"), rows)

    # ── 3. Amino-acid composition ─────────────────────────────────────────────
    rows = []
    sources = [(m, d, g) for m, d in [("flow", flow_data), ("diffusion", diff_data)]
               for g in GUIDANCE_MODES]
    sources += [("training", None, "train")]
    for method, data, mode in sources:
        seqs = data["sequences"][mode] if data else train_seqs
        total = sum(len(s) for s in seqs)
        counts = Counter("".join(seqs))
        for aa in AMINO_ACIDS:
            rows.append({
                "method": method, "guidance_mode": mode,
                "amino_acid": aa,
                "count": counts.get(aa, 0),
                "frequency": round(counts.get(aa, 0) / total, 6) if total else 0.0,
            })
    _write_csv(_fp("aa_composition"), rows)

    # ── 4. Polar residue counts ───────────────────────────────────────────────
    rows = []
    for method, data in [("flow", flow_data), ("diffusion", diff_data)]:
        for mode in GUIDANCE_MODES:
            for idx, seq in enumerate(data["sequences"][mode]):
                rows.append({
                    "method": method, "guidance_mode": mode, "sample_idx": idx,
                    "polar_count": sum(a in POLAR_RESIDUES for a in seq),
                    "min_polar_threshold": data.get("min_polar", 12),
                })
    _write_csv(_fp("polar_counts"), rows)

    # ── 5. Positional entropy ─────────────────────────────────────────────────
    rows = []
    for method, data in [("flow", flow_data), ("diffusion", diff_data),
                          ("training", {"sequences": {"train": train_seqs}})]:
        modes = GUIDANCE_MODES if method != "training" else ["train"]
        for mode in modes:
            seqs = data["sequences"][mode]
            ent  = _positional_entropy(seqs)
            for pos, h in enumerate(ent, start=1):
                rows.append({
                    "method": method, "guidance_mode": mode,
                    "position": pos, "entropy_bits": round(float(h), 6),
                })
    _write_csv(_fp("positional_entropy"), rows)

    # ── 6. Pairwise Hamming distances ─────────────────────────────────────────
    rows = []
    for method, data in [("flow", flow_data), ("diffusion", diff_data)]:
        for mode in GUIDANCE_MODES:
            seqs = data["sequences"][mode]
            for (i, s1), (j, s2) in combinations(enumerate(seqs), 2):
                rows.append({
                    "method": method, "guidance_mode": mode,
                    "seq_i": i, "seq_j": j,
                    "hamming": _hamming(s1, s2),
                })
    for (i, s1), (j, s2) in combinations(enumerate(train_seqs), 2):
        rows.append({
            "method": "training", "guidance_mode": "train",
            "seq_i": i, "seq_j": j, "hamming": _hamming(s1, s2),
        })
    _write_csv(_fp("hamming_diversity"), rows)

    # ── 7. Latent PCA ─────────────────────────────────────────────────────────
    try:
        from sklearn.decomposition import PCA
        rows = []
        for method, data in [("flow", flow_data), ("diffusion", diff_data)]:
            all_lats = np.concatenate(
                [data["standardized_latents"][m].mean(1).numpy() for m in GUIDANCE_MODES]
            )
            pca   = PCA(n_components=2)
            coords = pca.fit_transform(all_lats)
            offset = 0
            for mode in GUIDANCE_MODES:
                n = data["standardized_latents"][mode].shape[0]
                for idx in range(n):
                    rows.append({
                        "method": method, "guidance_mode": mode, "sample_idx": idx,
                        "pc1": round(float(coords[offset + idx, 0]), 6),
                        "pc2": round(float(coords[offset + idx, 1]), 6),
                        "pc1_var_pct": round(float(pca.explained_variance_ratio_[0] * 100), 3),
                        "pc2_var_pct": round(float(pca.explained_variance_ratio_[1] * 100), 3),
                    })
                offset += n
        _write_csv(_fp("latent_pca"), rows)
    except ImportError:
        print("  [skip] latent_pca.csv — install scikit-learn")

    # ── 8. Training loss (only when passed in) ────────────────────────────────
    if flow_losses is not None and diff_losses is not None:
        rows = [
            {"epoch": i + 1, "flow_loss": round(fl, 6), "diff_loss": round(dl, 6)}
            for i, (fl, dl) in enumerate(zip(flow_losses, diff_losses))
        ]
        _write_csv(_fp("training_loss"), rows)
