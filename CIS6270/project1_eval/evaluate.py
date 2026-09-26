#!/usr/bin/env python3
"""Project 1 Evaluation: Flow Matching vs Diffusion on ESM-2 Protein Embeddings.

Loads saved results and produces comparison plots.  Results can come from the
lecture_3 scripts (default) or from run_experiment.py (recommended for new models).

Usage:
  # lecture_3 baseline (esm2_8m, toy dataset)
  python evaluate.py

  # results produced by run_experiment.py
  python evaluate.py --model-name esm2_35m --dataset toy64 \\
      --flow-outdir outputs/esm2_35m_toy64/flow \\
      --diff-outdir outputs/esm2_35m_toy64/diffusion

  # optional extras
  python evaluate.py --retrain 100      # re-train with loss tracking
  python evaluate.py --ablate           # guidance-strength sweep
  python evaluate.py --list-models      # show all known ESM-2 variants
"""
import argparse
import csv
import math
from pathlib import Path
from collections import Counter
from itertools import combinations

from models import get_model, list_models, ESM2_MODELS

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

ROOT      = Path(__file__).resolve().parent
LECTURE3  = ROOT.parent / "lecture_3"
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


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def composition_proxies(sequences):
    return torch.tensor([
        [(sum(a in "KR" for a in s) - sum(a in "DE" for a in s)) / len(s),
         sum(a in POLAR_RESIDUES for a in s) / len(s)]
        for s in sequences
    ], dtype=torch.float32)


def load_fasta(path: Path):
    sequences, cur = [], ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    sequences.append(cur)
                cur = ""
            else:
                cur += line
    if cur:
        sequences.append(cur)
    return sequences


def load_results(method: str, out_dir: Path | None = None):
    """Return dict with sequences and latents for one method.

    out_dir: explicit directory containing results.pt + FASTA files.
    Falls back to the lecture_3 output layout when omitted.
    """
    if out_dir is None:
        out_dir = LECTURE3 / f"esm2_{method}_outputs"
        fallback_msg = (
            f"Run lecture_3/esm2_{method}_guidance.py --epochs 200, "
            f"or use run_experiment.py and pass --{method}-outdir."
        )
    else:
        fallback_msg = f"Run run_experiment.py and check --{method}-outdir."

    out_dir = Path(out_dir).resolve()
    pt_path = out_dir / "results.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"{pt_path} not found. {fallback_msg}")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    seqs = {}
    for mode in GUIDANCE_MODES:
        fasta = out_dir / f"{mode}.fasta"
        seqs[mode] = load_fasta(fasta) if fasta.exists() else []
    data["sequences"] = seqs
    return data


def load_training_sequences():
    csv_path = LECTURE3 / "esm2_example.csv"
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    sequences = [r["sequence"].strip().upper() for r in rows]
    c = [int(r["c"]) for r in rows]
    return sequences, c


# ══════════════════════════════════════════════════════════════════════════════
# Model definitions (needed for re-training / ablation)
# ══════════════════════════════════════════════════════════════════════════════

class FlowModel(nn.Module):
    def __init__(self, length, dim):
        super().__init__()
        self.length, self.dim = length, dim
        self.time = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.skip = nn.Linear(32, 1)
        self.condition = nn.Embedding(3, 16)
        self.net = nn.Sequential(
            nn.Linear(length * dim + 48, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, length * dim),
        )

    def forward(self, z, t, c):
        time = self.time(t[:, None])
        inputs = torch.cat([z.flatten(1), time, self.condition(c)], dim=1)
        return self.skip(time)[:, :, None] * z + self.net(inputs).reshape_as(z)


class DiffusionModel(nn.Module):
    def __init__(self, length, dim, alpha_bars):
        super().__init__()
        self.length, self.dim = length, dim
        self.alpha_bars = alpha_bars
        self.K = len(alpha_bars) - 1
        self.time = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.condition = nn.Embedding(3, 16)
        self.net = nn.Sequential(
            nn.Linear(length * dim + 48, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, length * dim),
        )

    def forward(self, z, t, c):
        time = self.time(t[:, None])
        inputs = torch.cat([z.flatten(1), time, self.condition(c)], dim=1)
        k = (t * self.K).round().long().clamp(0, self.K)
        a = self.alpha_bars[k, None, None]
        return (1 - a).sqrt() * z + a.sqrt() * self.net(inputs).reshape_as(z)


class RewardModel(nn.Module):
    def __init__(self, length, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(length * dim + 1, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 2),
        )

    def forward(self, z, t):
        return self.net(torch.cat([z.flatten(1), t[:, None]], dim=1))


def make_ddpm_schedule(K=1000, device="cpu"):
    betas      = torch.cat([torch.zeros(1), torch.linspace(1e-4, 0.02, K)]).to(device)
    alphas     = 1.0 - betas
    alpha_bars = alphas.cumprod(0)
    previous   = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
    post_vars  = betas * (1 - previous) / (1 - alpha_bars).clamp_min(1e-20)
    return betas, alphas, alpha_bars, post_vars


# ══════════════════════════════════════════════════════════════════════════════
# Re-training with loss tracking
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset_from_csv(csv_path, esm_model_tag="esm2_8m", cache_dir=None):
    """Encode sequences with the specified ESM-2 model; return TensorDataset."""
    from transformers import AutoTokenizer, EsmForMaskedLM
    model_info = get_model(esm_model_tag)
    hf_id = model_info["hf_id"]
    cache = cache_dir or (LECTURE3 / ".esm2_cache")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    sequences = [r["sequence"].strip().upper() for r in rows]
    c = torch.tensor([int(r["c"]) for r in rows], dtype=torch.long)
    if {"r1", "r2"}.issubset(rows[0]):
        r = torch.tensor([[float(r["r1"]), float(r["r2"])] for r in rows])
    else:
        r = composition_proxies(sequences)
    print(f"  Loading {hf_id} (cache: {cache})")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=cache)
    esm = EsmForMaskedLM.from_pretrained(
        hf_id, cache_dir=cache, use_safetensors=True
    ).to(DEVICE).eval().requires_grad_(False)
    encoded = []
    with torch.no_grad():
        for start in range(0, len(sequences), BATCH_SIZE):
            toks = tokenizer(sequences[start:start + BATCH_SIZE], return_tensors="pt")
            toks = {k: v.to(DEVICE) for k, v in toks.items()}
            h = esm.esm(**toks).last_hidden_state
            encoded.append(h[:, 1:-1].cpu())
    z = torch.cat(encoded)
    z_mean = z.mean((0, 1), keepdim=True)
    z_std  = z.std((0, 1), correction=0, keepdim=True).clamp_min(1e-4)
    r_mean, r_std = r.mean(0), r.std(0, correction=0).clamp_min(1e-6)
    stats = {"z_mean": z_mean, "z_std": z_std, "r_mean": r_mean, "r_std": r_std}
    dataset = TensorDataset((z - z_mean) / z_std, c, (r - r_mean) / r_std)
    return dataset, esm, tokenizer, stats


def train_flow(dataset, epochs):
    _, length, dim = dataset.tensors[0].shape
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model  = FlowModel(length, dim).to(DEVICE)
    reward = RewardModel(length, dim).to(DEVICE)
    opt    = torch.optim.Adam(list(model.parameters()) + list(reward.parameters()), lr=LEARNING_RATE)
    losses = []
    for epoch in range(epochs):
        total = 0.0
        for z1, c, r_tilde in loader:
            z1, c, r_tilde = z1.to(DEVICE), c.to(DEVICE), r_tilde.to(DEVICE)
            z0 = torch.randn_like(z1)
            t  = torch.rand(len(z1), device=DEVICE)
            zt = (1 - t[:, None, None]) * z0 + t[:, None, None] * z1
            dropped = c.masked_fill(torch.rand(len(c), device=DEVICE) < CONDITION_DROP, 2)
            loss = F.mse_loss(model(zt, t, dropped), z1 - z0) + F.mse_loss(reward(zt, t), r_tilde)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        if (epoch + 1) % max(1, epochs // 4) == 0 or epoch + 1 == epochs:
            print(f"  [flow]      epoch {epoch+1}/{epochs}: loss {avg:.4f}")
    model.eval().requires_grad_(False)
    reward.eval().requires_grad_(False)
    return model, reward, losses


def train_diffusion(dataset, epochs):
    _, length, dim = dataset.tensors[0].shape
    betas, alphas, alpha_bars, post_vars = make_ddpm_schedule(device=DEVICE)
    K      = len(betas) - 1
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model  = DiffusionModel(length, dim, alpha_bars).to(DEVICE)
    reward = RewardModel(length, dim).to(DEVICE)
    opt    = torch.optim.Adam(list(model.parameters()) + list(reward.parameters()), lr=LEARNING_RATE)
    losses = []
    for epoch in range(epochs):
        total = 0.0
        for z0, c, r_tilde in loader:
            z0, c, r_tilde = z0.to(DEVICE), c.to(DEVICE), r_tilde.to(DEVICE)
            k = torch.randint(1, K + 1, (len(z0),), device=DEVICE)
            t = k.float() / K
            a = alpha_bars[k, None, None]
            eps = torch.randn_like(z0)
            zk  = a.sqrt() * z0 + (1 - a).sqrt() * eps
            dropped = c.masked_fill(torch.rand(len(c), device=DEVICE) < CONDITION_DROP, 2)
            loss = F.mse_loss(model(zk, t, dropped), eps) + F.mse_loss(reward(zk, t), r_tilde)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        if (epoch + 1) % max(1, epochs // 4) == 0 or epoch + 1 == epochs:
            print(f"  [diffusion] epoch {epoch+1}/{epochs}: loss {avg:.4f}")
    model.eval().requires_grad_(False)
    reward.eval().requires_grad_(False)
    return model, reward, losses, alpha_bars, betas, alphas, post_vars


# ══════════════════════════════════════════════════════════════════════════════
# Sampling helpers (for ablation)
# ══════════════════════════════════════════════════════════════════════════════

def reward_gradient(reward_model, z, t, lambdas):
    with torch.enable_grad():
        state = z.detach().requires_grad_(True)
        R = (reward_model(state, t) * lambdas).sum(1)
        grad = torch.autograd.grad(R.sum(), state)[0]
    return grad.detach()


@torch.no_grad()
def sample_flow(model, reward_model, n=8, c=1, w=0.0, eta=0.0, lambdas=(1., 0.), steps=200):
    lam = torch.tensor(lambdas, dtype=torch.float32, device=DEVICE)
    lam = lam / lam.sum()
    torch.manual_seed(123)
    z    = torch.randn(n, model.length, model.dim, device=DEVICE)
    null = torch.full((n,), 2, dtype=torch.long, device=DEVICE)
    cond = torch.full((n,), c, dtype=torch.long, device=DEVICE)
    dt   = 1.0 / steps
    for step in range(steps):
        t = torch.full((n,), step * dt, device=DEVICE)
        v = model(z, t, null)
        if w:
            v = v + w * (model(z, t, cond) - model(z, t, null))
        if eta:
            kappa = eta * 4 * t[:, None, None] * (1 - t[:, None, None])
            v = v + kappa * reward_gradient(reward_model, z, t, lam)
        z = z + dt * v
    return z


@torch.no_grad()
def sample_diffusion(model, reward_model, alpha_bars, betas, alphas, post_vars,
                     n=8, c=1, w=0.0, eta=0.0, lambdas=(1., 0.)):
    lam = torch.tensor(lambdas, dtype=torch.float32, device=DEVICE)
    lam = lam / lam.sum()
    K = len(betas) - 1
    torch.manual_seed(123)
    z    = torch.randn(n, model.length, model.dim, device=DEVICE)
    null = torch.full((n,), 2, dtype=torch.long, device=DEVICE)
    cond = torch.full((n,), c, dtype=torch.long, device=DEVICE)
    for k in range(K, 0, -1):
        t = torch.full((n,), k / K, device=DEVICE)
        eps = model(z, t, null)
        if w:
            eps = eps + w * (model(z, t, cond) - model(z, t, null))
        sigma = (1 - alpha_bars[k]).sqrt()
        if eta:
            eps = eps - eta * sigma * reward_gradient(reward_model, z, t, lam)
        mean = (z - betas[k] * eps / sigma) / alphas[k].sqrt()
        z    = mean + post_vars[k].sqrt() * torch.randn_like(z) if k > 1 else mean
    return z


def predict_reward(reward_model, z, stats):
    """Use reward model at t=1 to predict properties, then unstandardize."""
    z = z.to(DEVICE)
    t = torch.ones(len(z), device=DEVICE)
    with torch.no_grad():
        r_tilde = reward_model(z, t).cpu()
    r_mean = stats["r_mean"]
    r_std  = stats["r_std"]
    return r_tilde * r_std + r_mean


# ══════════════════════════════════════════════════════════════════════════════
# Raw-data export  (one CSV per metric — load later for custom plots)
# ══════════════════════════════════════════════════════════════════════════════

def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path}")


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


# ══════════════════════════════════════════════════════════════════════════════
# Plotting functions
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, out_dir, name, prefix=""):
    filename = f"{prefix}_{name}" if prefix else name
    p = Path(out_dir) / filename
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {p}")


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


# ── Plot 7: Per-position amino acid entropy ───────────────────────────────────
def _positional_entropy(sequences):
    if not sequences:
        return np.array([])
    L = len(sequences[0])
    entropy = []
    for pos in range(L):
        col = [s[pos] for s in sequences if pos < len(s)]
        counts = Counter(col)
        total  = len(col)
        probs  = np.array([v / total for v in counts.values()])
        h = -np.sum(probs * np.log2(probs + 1e-12))
        entropy.append(h)
    return np.array(entropy)


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


# ── Plot 8: Sequence diversity (pairwise Hamming distance) ────────────────────
def _hamming(s1, s2):
    return sum(a != b for a, b in zip(s1, s2))


def _pairwise_hamming(sequences):
    return [_hamming(a, b) for a, b in combinations(sequences, 2)] if len(sequences) >= 2 else [0]


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


# ── Plot 9: Guidance strength ablation ────────────────────────────────────────
def plot_guidance_ablation(flow_model, flow_reward, diff_model, diff_reward,
                            diff_schedule, flow_stats, diff_stats, out_dir, prefix=""):
    """Vary CFG weight w and reward eta; plot + save predicted r1, r2."""
    betas, alphas, alpha_bars, post_vars = diff_schedule
    out_dir  = Path(out_dir)

    w_vals   = [0.0, 0.5, 1.0, 2.0, 3.0]
    eta_vals = [0.0, 0.5, 1.0, 2.0]

    def get_pred(z, reward_model, stats):
        return predict_reward(reward_model, z, stats).numpy()

    # Collect raw numbers for both sweeps so we can write CSVs
    cfg_rows, eta_rows = [], []

    # --- CFG sweep (eta=0) ---
    cfg_results = {}   # method -> {"means_r1": [...], "stds_r1": [...], ...}
    for method, model, reward, stats, sampler in [
        ("flow",      flow_model,  flow_reward,  flow_stats,
         lambda m, r, w: sample_flow(m, r, n=8, c=1, w=w, eta=0)),
        ("diffusion", diff_model, diff_reward, diff_stats,
         lambda m, r, w: sample_diffusion(m, r, alpha_bars, betas, alphas, post_vars,
                                          n=8, c=1, w=w, eta=0)),
    ]:
        mr1, sr1, mr2, sr2 = [], [], [], []
        for w in w_vals:
            pred = get_pred(sampler(model, reward, w), reward, stats)
            mr1.append(pred[:, 0].mean()); sr1.append(pred[:, 0].std())
            mr2.append(pred[:, 1].mean()); sr2.append(pred[:, 1].std())
            cfg_rows.append({
                "method": method, "cfg_weight_w": w,
                "mean_r1": round(float(pred[:, 0].mean()), 6),
                "std_r1":  round(float(pred[:, 0].std()),  6),
                "mean_r2": round(float(pred[:, 1].mean()), 6),
                "std_r2":  round(float(pred[:, 1].std()),  6),
            })
        cfg_results[method] = {
            "mr1": np.array(mr1), "sr1": np.array(sr1),
            "mr2": np.array(mr2), "sr2": np.array(sr2),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key_m, key_s, ylabel in zip(
        axes,
        ["mr1", "mr2"], ["sr1", "sr2"],
        ["Predicted r₁ (charge proxy)", "Predicted r₂ (polar fraction)"],
    ):
        for method in ("flow", "diffusion"):
            r = cfg_results[method]
            ax.plot(w_vals, r[key_m], color=METHOD_COLORS[method], marker="o",
                    lw=2, label=method.capitalize())
            ax.fill_between(w_vals, r[key_m] - r[key_s], r[key_m] + r[key_s],
                            alpha=0.15, color=METHOD_COLORS[method])
        ax.set_xlabel("CFG weight w"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs CFG weight (η=0)")
        ax.legend(); ax.axhline(0, color="gray", lw=0.7, ls=":")
    fig.suptitle("Ablation: CFG Guidance Strength (predicted rewards)", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir, "09a_ablation_cfg_weight.png", prefix)
    _write_csv(out_dir / f"{prefix}_ablation_cfg_weight.csv", cfg_rows)

    # --- Reward eta sweep (w=0, lambda=(1,0)) ---
    eta_results = {}
    for method, model, reward, stats, sampler in [
        ("flow",      flow_model,  flow_reward,  flow_stats,
         lambda m, r, e: sample_flow(m, r, n=8, c=1, w=0, eta=e, lambdas=(1., 0.))),
        ("diffusion", diff_model, diff_reward, diff_stats,
         lambda m, r, e: sample_diffusion(m, r, alpha_bars, betas, alphas, post_vars,
                                          n=8, c=1, w=0, eta=e, lambdas=(1., 0.))),
    ]:
        mr1, sr1, mr2, sr2 = [], [], [], []
        for eta in eta_vals:
            pred = get_pred(sampler(model, reward, eta), reward, stats)
            mr1.append(pred[:, 0].mean()); sr1.append(pred[:, 0].std())
            mr2.append(pred[:, 1].mean()); sr2.append(pred[:, 1].std())
            eta_rows.append({
                "method": method, "reward_eta": eta,
                "mean_r1": round(float(pred[:, 0].mean()), 6),
                "std_r1":  round(float(pred[:, 0].std()),  6),
                "mean_r2": round(float(pred[:, 1].mean()), 6),
                "std_r2":  round(float(pred[:, 1].std()),  6),
            })
        eta_results[method] = {
            "mr1": np.array(mr1), "sr1": np.array(sr1),
            "mr2": np.array(mr2), "sr2": np.array(sr2),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key_m, key_s, ylabel in zip(
        axes,
        ["mr1", "mr2"], ["sr1", "sr2"],
        ["Predicted r₁ (charge proxy)", "Predicted r₂ (polar fraction)"],
    ):
        for method in ("flow", "diffusion"):
            r = eta_results[method]
            ax.plot(eta_vals, r[key_m], color=METHOD_COLORS[method], marker="^",
                    lw=2, linestyle="--", label=method.capitalize())
            ax.fill_between(eta_vals, r[key_m] - r[key_s], r[key_m] + r[key_s],
                            alpha=0.15, color=METHOD_COLORS[method])
        ax.set_xlabel("Reward steering strength η"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs reward strength (w=0, λ=(1,0))")
        ax.legend(); ax.axhline(0, color="gray", lw=0.7, ls=":")
    fig.suptitle("Ablation: Reward Steering Strength (predicted rewards)", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir, "09b_ablation_reward_eta.png", prefix)
    _write_csv(out_dir / f"{prefix}_ablation_reward_eta.csv", eta_rows)


# ── Plot 10: Summary comparison panel ─────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ── Naming ────────────────────────────────────────────────────────────────
    parser.add_argument("--model-name",   default="esm2_8m",
                        help="ESM-2 model tag (default: esm2_8m). Use --list-models to see all.")
    parser.add_argument("--dataset",      default="toy64",
                        help="Short label for the dataset (default: toy64). "
                             "Used in folder name and every filename.")
    # ── Result sources ────────────────────────────────────────────────────────
    parser.add_argument("--flow-outdir",  type=Path, default=None,
                        help="Directory with flow results.pt + FASTA files. "
                             "Default: lecture_3/esm2_flow_outputs/  (lecture_3 baseline).")
    parser.add_argument("--diff-outdir",  type=Path, default=None,
                        help="Directory with diffusion results.pt + FASTA files. "
                             "Default: lecture_3/esm2_diffusion_outputs/  (lecture_3 baseline).")
    # ── Optional extras ───────────────────────────────────────────────────────
    parser.add_argument("--retrain",      type=int, metavar="EPOCHS",
                        help="Re-train both models for N epochs to capture loss curves.")
    parser.add_argument("--ablate",       action="store_true",
                        help="Run guidance-strength ablation (loads saved model weights).")
    parser.add_argument("--outdir",       type=Path, default=None,
                        help="Override plot output directory. "
                             "Default: plots/<model_name>_<dataset>/")
    parser.add_argument("--list-models",  action="store_true",
                        help="Print all known ESM-2 model tags and exit.")
    args = parser.parse_args()

    if args.list_models:
        print(list_models())
        return

    # Build run tag and prefix from model+dataset
    run_tag = f"{args.model_name}_{args.dataset}"
    prefix  = run_tag
    outdir  = args.outdir if args.outdir else ROOT / "plots" / run_tag
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Run tag       : {run_tag}")
    print(f"Output dir    : {outdir}")
    print(f"File prefix   : {prefix}_<plotname>.png")
    if args.flow_outdir:
        print(f"Flow results  : {args.flow_outdir}")
    if args.diff_outdir:
        print(f"Diff results  : {args.diff_outdir}")

    # ── Load results ──────────────────────────────────────────────────────────
    print("\nLoading results...")
    flow_data = load_results("flow", args.flow_outdir)
    diff_data = load_results("diffusion", args.diff_outdir)
    train_seqs, train_classes = load_training_sequences()
    print(f"  Flow guidance modes : {list(flow_data['sequences'].keys())}")
    print(f"  Diff guidance modes : {list(diff_data['sequences'].keys())}")
    print(f"  Training sequences  : {len(train_seqs)}")

    # ── Raw data export ───────────────────────────────────────────────────────
    print("\nSaving raw metric CSVs...")
    save_raw_data(flow_data, diff_data, train_seqs, outdir, prefix)

    # ── Static plots (no re-training needed) ──────────────────────────────────
    print("\nGenerating static plots...")
    plot_composition_proxies(flow_data, diff_data, train_seqs, outdir, prefix)
    try:
        plot_latent_pca(flow_data, diff_data, outdir, prefix)
    except ImportError:
        print("  [skip] latent PCA — install scikit-learn: pip install scikit-learn")
    plot_aa_composition(flow_data, diff_data, train_seqs, outdir, prefix)
    plot_polar_residue_distribution(flow_data, diff_data, outdir, prefix)
    plot_reward_pareto(flow_data, diff_data, train_seqs, outdir, prefix)
    plot_positional_entropy(flow_data, diff_data, train_seqs, outdir, prefix)
    plot_sequence_diversity(flow_data, diff_data, train_seqs, outdir, prefix)
    plot_summary_panel(flow_data, diff_data, train_seqs, outdir, prefix, run_tag)

    # ── Optional: re-train with loss tracking ─────────────────────────────────
    if args.retrain:
        print(f"\nRe-training for {args.retrain} epochs "
              f"(model: {args.model_name}, requires ESM-2 weights)...")
        csv_path  = LECTURE3 / "esm2_example.csv"
        cache_dir = (args.flow_outdir / ".." / "cache") if args.flow_outdir else None
        dataset, esm, tokenizer, stats = load_dataset_from_csv(
            csv_path, esm_model_tag=args.model_name, cache_dir=cache_dir
        )
        torch.manual_seed(7)
        _, _, flow_losses = train_flow(dataset, args.retrain)
        torch.manual_seed(7)
        _, _, diff_losses, *_ = train_diffusion(dataset, args.retrain)
        plot_training_loss(flow_losses, diff_losses, outdir, prefix)
        save_raw_data(flow_data, diff_data, train_seqs, outdir, prefix,
                      flow_losses=flow_losses, diff_losses=diff_losses)

    # ── Optional: guidance strength ablation ──────────────────────────────────
    if args.ablate:
        print("\nRunning guidance ablation (loading saved model weights)...")
        length = flow_data["length"]
        dim    = flow_data["dim"]
        K      = 1000
        betas, alphas, alpha_bars, post_vars = make_ddpm_schedule(K, device=DEVICE)

        flow_model  = FlowModel(length, dim).to(DEVICE)
        flow_reward = RewardModel(length, dim).to(DEVICE)
        flow_model.load_state_dict(flow_data["model"])
        flow_reward.load_state_dict(flow_data["reward_model"])
        flow_model.eval().requires_grad_(False)
        flow_reward.eval().requires_grad_(False)

        diff_model  = DiffusionModel(length, dim, alpha_bars).to(DEVICE)
        diff_reward = RewardModel(length, dim).to(DEVICE)
        diff_model.load_state_dict(diff_data["model"])
        diff_reward.load_state_dict(diff_data["reward_model"])
        diff_model.eval().requires_grad_(False)
        diff_reward.eval().requires_grad_(False)

        plot_guidance_ablation(
            flow_model, flow_reward, diff_model, diff_reward,
            (betas, alphas, alpha_bars, post_vars),
            flow_data["stats"], diff_data["stats"],
            outdir, prefix,
        )

    n_png = len(list(outdir.glob("*.png")))
    n_csv = len(list(outdir.glob("*.csv")))
    print(f"\nDone. {n_png} plots + {n_csv} CSV files saved to {outdir}/")


if __name__ == "__main__":
    main()
