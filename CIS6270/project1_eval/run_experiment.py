#!/usr/bin/env python3
"""Run the full Project 1 pipeline for any ESM-2 model variant.

Encodes sequences with the chosen ESM-2 model, trains flow-matching and
diffusion models, samples under three guidance modes, decodes to amino-acid
sequences, and saves results.pt + FASTA files in a structured output directory
ready for evaluate.py.

Usage:
  python run_experiment.py --esm-model esm2_8m  --dataset ../lecture_3/esm2_example.csv
  python run_experiment.py --esm-model esm2_35m --dataset /path/to/my_data.csv --epochs 300
  python run_experiment.py --list-models

Output layout:
  outputs/<esm_model>_<dataset_tag>/
    flow/       results.pt, cfg.fasta, single.fasta, multi.fasta
    diffusion/  results.pt, cfg.fasta, single.fasta, multi.fasta

  cache/        HuggingFace weights, shared by every model and every run
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, EsmForMaskedLM

from models import get_model, list_models, ESM2_MODELS

ROOT   = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AMINO_ACIDS    = "ACDEFGHIKLMNPQRSTVWY"
POLAR_RESIDUES = "DEHKNQRST"
BATCH_SIZE, HIDDEN, LEARNING_RATE = 16, 128, 1e-3
CONDITION_DROP = 0.2

# One HuggingFace cache for every model and every run. HF already namespaces
# downloads as models--facebook--<name>, so an extra per-model or per-run
# subdirectory only causes the same weights to be fetched again.
DEFAULT_CACHE = ROOT / "cache"


def resolve_cache_dir(override=None) -> Path:
    """Shared ESM-2 weight cache: --cache-dir, else $ESM2_CACHE, else ROOT/cache."""
    import os
    chosen = override or os.environ.get("ESM2_CACHE") or DEFAULT_CACHE
    path = Path(chosen).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Composition proxies
# ══════════════════════════════════════════════════════════════════════════════

def composition_proxies(sequences):
    return torch.tensor([
        [(sum(a in "KR" for a in s) - sum(a in "DE" for a in s)) / len(s),
         sum(a in POLAR_RESIDUES for a in s) / len(s)]
        for s in sequences
    ], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def load_data(csv_path: Path, esm_hf_id: str, cache_dir: Path, max_length: int = 128):
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    sequences = [r["sequence"].strip().upper() for r in rows]
    if len(rows) < 4 or any(not s or set(s) - set(AMINO_ACIDS) for s in sequences):
        raise ValueError("Supply at least four sequences using the 20 standard amino acids")
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1 or max(lengths) > max_length:
        raise ValueError(f"Sequences must all be the same length, at most {max_length}")
    c = torch.tensor([int(r["c"]) for r in rows], dtype=torch.long)
    if {"r1", "r2"}.issubset(rows[0]):
        r = torch.tensor([[float(r["r1"]), float(r["r2"])] for r in rows])
    else:
        r = composition_proxies(sequences)
    if set(c.tolist()) != {0, 1} or not torch.isfinite(r).all():
        raise ValueError("Both c=0 and c=1 must be present; r1/r2 must be finite")

    print(f"  Loading {esm_hf_id} from cache: {cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(esm_hf_id, cache_dir=cache_dir)
    esm = EsmForMaskedLM.from_pretrained(
        esm_hf_id, cache_dir=cache_dir, use_safetensors=True
    ).to(DEVICE).eval().requires_grad_(False)
    hidden_size = esm.config.hidden_size
    print(f"  ESM-2 hidden size: {hidden_size}  |  sequences: {len(sequences)}  |  length: {max(lengths)}")

    encoded = []
    for start in range(0, len(sequences), BATCH_SIZE):
        toks = tokenizer(sequences[start:start + BATCH_SIZE], return_tensors="pt")
        toks = {k: v.to(DEVICE) for k, v in toks.items()}
        h = esm.esm(**toks).last_hidden_state
        encoded.append(h[:, 1:-1].cpu())
    z = torch.cat(encoded)                         # [N, L, hidden_size]
    z_mean = z.mean((0, 1), keepdim=True)
    z_std  = z.std((0, 1), correction=0, keepdim=True).clamp_min(1e-4)
    r_mean, r_std = r.mean(0), r.std(0, correction=0).clamp_min(1e-6)
    dataset = TensorDataset((z - z_mean) / z_std, c, (r - r_mean) / r_std)
    stats = {"z_mean": z_mean, "z_std": z_std, "r_mean": r_mean, "r_std": r_std}
    return dataset, esm, tokenizer, stats, sequences


# ══════════════════════════════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════════════════════════════

class FlowModel(nn.Module):
    def __init__(self, length, dim, hidden=HIDDEN):
        super().__init__()
        self.length, self.dim = length, dim
        self.time      = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.skip      = nn.Linear(32, 1)
        self.condition = nn.Embedding(3, 16)
        self.net = nn.Sequential(
            nn.Linear(length * dim + 48, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, length * dim),
        )

    def forward(self, z, t, c):
        time   = self.time(t[:, None])
        inputs = torch.cat([z.flatten(1), time, self.condition(c)], dim=1)
        return self.skip(time)[:, :, None] * z + self.net(inputs).reshape_as(z)


class DiffusionModel(nn.Module):
    def __init__(self, length, dim, alpha_bars, hidden=HIDDEN):
        super().__init__()
        self.length, self.dim = length, dim
        self.alpha_bars = alpha_bars
        self.K          = len(alpha_bars) - 1
        self.time      = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.condition = nn.Embedding(3, 16)
        self.net = nn.Sequential(
            nn.Linear(length * dim + 48, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, length * dim),
        )

    def forward(self, z, t, c):
        time   = self.time(t[:, None])
        inputs = torch.cat([z.flatten(1), time, self.condition(c)], dim=1)
        k = (t * self.K).round().long().clamp(0, self.K)
        a = self.alpha_bars[k, None, None]
        return (1 - a).sqrt() * z + a.sqrt() * self.net(inputs).reshape_as(z)


class RewardModel(nn.Module):
    def __init__(self, length, dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(length * dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2),
        )

    def forward(self, z, t):
        return self.net(torch.cat([z.flatten(1), t[:, None]], dim=1))


def make_ddpm_schedule(K=1000):
    betas      = torch.cat([torch.zeros(1), torch.linspace(1e-4, 0.02, K)]).to(DEVICE)
    alphas     = 1.0 - betas
    alpha_bars = alphas.cumprod(0)
    previous   = torch.cat([torch.ones(1, device=DEVICE), alpha_bars[:-1]])
    post_vars  = betas * (1 - previous) / (1 - alpha_bars).clamp_min(1e-20)
    return betas, alphas, alpha_bars, post_vars


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_flow(dataset, epochs, batch_size=BATCH_SIZE, hidden=HIDDEN):
    _, length, dim = dataset.tensors[0].shape
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model  = FlowModel(length, dim, hidden).to(DEVICE)
    reward = RewardModel(length, dim, hidden).to(DEVICE)
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
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        if (epoch + 1) % max(1, epochs // 4) == 0 or epoch + 1 == epochs:
            print(f"  [flow]      epoch {epoch+1:>4}/{epochs}: loss {avg:.4f}")
    model.eval().requires_grad_(False)
    reward.eval().requires_grad_(False)
    return model, reward, losses


def train_diffusion(dataset, epochs, batch_size=BATCH_SIZE, hidden=HIDDEN):
    _, length, dim = dataset.tensors[0].shape
    betas, alphas, alpha_bars, post_vars = make_ddpm_schedule()
    K      = len(betas) - 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model  = DiffusionModel(length, dim, alpha_bars, hidden).to(DEVICE)
    reward = RewardModel(length, dim, hidden).to(DEVICE)
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
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        if (epoch + 1) % max(1, epochs // 4) == 0 or epoch + 1 == epochs:
            print(f"  [diffusion] epoch {epoch+1:>4}/{epochs}: loss {avg:.4f}")
    model.eval().requires_grad_(False)
    reward.eval().requires_grad_(False)
    return model, reward, losses, alpha_bars, betas, alphas, post_vars


# ══════════════════════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════════════════════

def reward_gradient(reward_model, z, t, lambdas):
    with torch.enable_grad():
        state = z.detach().requires_grad_(True)
        R = (reward_model(state, t) * lambdas).sum(1)
        grad = torch.autograd.grad(R.sum(), state)[0]
    return grad.detach()


@torch.no_grad()
def sample_flow(model, reward_model, n=8, c=1, w=0.0, eta=0.0, lambdas=(1., 0.), steps=200,
                anchor=None, strength=1.0):
    """Integrate the velocity field from t=0 to t=1.

    With `anchor` (a standardized reference latent) the trajectory starts from a
    partially noised anchor at t = 1 - strength instead of pure noise at t = 0.
    Every measured avGFP variant sits within 15 substitutions of the wild type,
    a vanishingly small region of a 237x320 latent space, so integrating from
    N(0, I) rarely lands on it. strength=1.0 reproduces the unanchored path.
    """
    lam = torch.tensor(lambdas, dtype=torch.float32, device=DEVICE)
    lam = lam / lam.sum()
    torch.manual_seed(123)
    start = 0.0
    if anchor is None:
        z = torch.randn(n, model.length, model.dim, device=DEVICE)
    else:
        start = 1.0 - float(strength)
        z1    = anchor.to(DEVICE).expand(n, -1, -1)
        z0    = torch.randn(n, model.length, model.dim, device=DEVICE)
        z     = (1 - start) * z0 + start * z1      # the path's own interpolant at t=start
    null = torch.full((n,), 2, dtype=torch.long, device=DEVICE)
    cond = torch.full((n,), c, dtype=torch.long, device=DEVICE)
    dt   = (1.0 - start) / steps
    for step in range(steps):
        t = torch.full((n,), start + step * dt, device=DEVICE)
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
                     n=8, c=1, w=0.0, eta=0.0, lambdas=(1., 0.),
                     anchor=None, strength=1.0):
    """Run the reverse chain from step K down to 1.

    With `anchor` the chain starts at step round(strength * K) from the forward-
    noised anchor (SDEdit), rather than at K from pure noise. strength=1.0
    reproduces the unanchored chain.
    """
    lam = torch.tensor(lambdas, dtype=torch.float32, device=DEVICE)
    lam = lam / lam.sum()
    K    = len(betas) - 1
    torch.manual_seed(123)
    start = K
    if anchor is None:
        z = torch.randn(n, model.length, model.dim, device=DEVICE)
    else:
        start = max(1, min(K, int(round(float(strength) * K))))
        a     = alpha_bars[start]
        z1    = anchor.to(DEVICE).expand(n, -1, -1)
        z     = a.sqrt() * z1 + (1 - a).sqrt() * torch.randn(n, model.length, model.dim,
                                                             device=DEVICE)
    null = torch.full((n,), 2, dtype=torch.long, device=DEVICE)
    cond = torch.full((n,), c, dtype=torch.long, device=DEVICE)
    for k in range(start, 0, -1):
        t   = torch.full((n,), k / K, device=DEVICE)
        eps = model(z, t, null)
        if w:
            eps = eps + w * (model(z, t, cond) - model(z, t, null))
        sigma = (1 - alpha_bars[k]).sqrt()
        if eta:
            eps = eps - eta * sigma * reward_gradient(reward_model, z, t, lam)
        mean = (z - betas[k] * eps / sigma) / alphas[k].sqrt()
        z    = mean + post_vars[k].sqrt() * torch.randn_like(z) if k > 1 else mean
    return z


# ══════════════════════════════════════════════════════════════════════════════
# Decoding
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def decode(z, esm, tokenizer, stats, min_polar=12):
    latent  = z * stats["z_std"].to(z.device) + stats["z_mean"].to(z.device)
    logits  = esm.lm_head(latent)
    aa_ids  = torch.tensor(tokenizer.convert_tokens_to_ids(list(AMINO_ACIDS)), device=z.device)
    logits  = logits.index_select(-1, aa_ids)
    polar   = torch.tensor([a in POLAR_RESIDUES for a in AMINO_ACIDS], device=z.device)
    polar_ids = polar.nonzero().flatten()
    best_scores, choices = logits.max(dim=-1)
    polar_scores, local  = logits[..., polar_ids].max(dim=-1)
    polar_choices = polar_ids[local]
    for i in range(len(z)):
        already = polar[choices[i]]
        missing = max(0, min_polar - int(already.sum()))
        if missing:
            cost = (best_scores[i] - polar_scores[i]).masked_fill(already, float("inf"))
            positions = cost.topk(missing, largest=False).indices
            choices[i, positions] = polar_choices[i, positions]
    return ["".join(AMINO_ACIDS[i] for i in row) for row in choices.cpu().tolist()]


@torch.no_grad()
def encode_reference(sequence, esm, tokenizer, stats):
    """Standardized ESM-2 latent for one reference sequence, shaped [1, L, dim]."""
    toks = tokenizer([sequence], return_tensors="pt")
    toks = {k: v.to(DEVICE) for k, v in toks.items()}
    h = esm.esm(**toks).last_hidden_state[:, 1:-1].cpu()
    return ((h - stats["z_mean"]) / stats["z_std"]).to(DEVICE)


def consensus(sequences):
    """Per-position most common residue; equals the wild type for DMS variant sets."""
    return "".join(max(AMINO_ACIDS, key=lambda a: sum(s[i] == a for s in sequences))
                   for i in range(len(sequences[0])))


@torch.no_grad()
def decode_budget(z, esm, tokenizer, stats, reference, budget,
                  temperature=0.0, frozen=(), exact=False):
    """Decode as a variant of `reference` with at most `budget` substitutions.

    With temperature=0 this takes the argmax and keeps only positions where that
    argmax beats the reference residue. The rule is deterministic, and on deep
    mutational scanning data it collapses: ESM's head puts about 0.71 of its mass
    on the reference residue with ~1.34 nats of entropy, so the reference is the
    mode at essentially every position even though the distribution is broad.
    The argmax discards that mass, and fifty different latents decode to a
    handful of sequences -- measured here as 3 distinct from 50 draws.

    With temperature>0 it instead samples `budget` positions, preferring those
    the model is least certain about, and samples a residue at each from the
    per-position distribution. On the same latents, T=0.7 recovers 40 distinct
    sequences from 50 draws at an unchanged mean Hamming distance.

    `frozen` positions are never substituted. Position 0 is a common choice for
    avGFP: this wild-type sequence omits the initiator methionine, so ESM wants
    to put one back, and that single substitution dominates otherwise.

    `budget` is normally a ceiling, not a target: each chosen position keeps the
    reference residue whenever the sample lands on it, which it does about 70% of
    the time, so a budget of 5 yields ~1.5 substitutions and leaves ~18% of
    samples identical to the reference. With `exact=True` the reference residue
    is excluded at the chosen positions, so every sample carries exactly `budget`
    substitutions. That makes the mutational distance a controlled variable
    rather than a confound: comparisons against a baseline no longer have to be
    matched after the fact, and no sample can score well by declining to mutate.
    """
    device = z.device
    latent = z * stats["z_std"].to(device) + stats["z_mean"].to(device)
    aa_ids = torch.tensor(tokenizer.convert_tokens_to_ids(list(AMINO_ACIDS)), device=device)
    logits = esm.lm_head(latent).index_select(-1, aa_ids)
    ref_ids = torch.tensor([AMINO_ACIDS.index(a) for a in reference], device=device)
    n, length = len(z), logits.shape[1]
    budget = min(budget, length)
    decoded = ref_ids.expand(n, -1).clone()
    frozen_mask = torch.zeros(length, dtype=torch.bool, device=device)
    for position in frozen:
        if 0 <= position < length:
            frozen_mask[position] = True

    if temperature <= 0:
        # Exclude the reference residue so "best" always means a real substitution.
        masked = logits.scatter(-1, ref_ids.expand(n, -1)[..., None], -float("inf"))
        alt_scores, alt_choices = masked.max(dim=-1)
        ref_scores = logits.gather(-1, ref_ids.expand(n, -1)[..., None]).squeeze(-1)
        margin = (alt_scores - ref_scores).masked_fill(frozen_mask, -float("inf"))
        keep = margin.topk(budget, dim=1).indices
        for i in range(n):
            positions = keep[i] if exact else keep[i][margin[i, keep[i]] > 0]
            decoded[i, positions] = alt_choices[i, positions]
    else:
        # A sample whose latent diverged (large --cfg-weight or --reward-eta can do
        # this) produces non-finite logits. Sanitize rather than crash: the run
        # already warns about diverged latents, and one bad sample should not take
        # the whole batch down.
        scaled = torch.nan_to_num(logits / temperature, nan=0.0,
                                  posinf=30.0, neginf=-30.0)
        probs = torch.softmax(scaled, dim=-1)
        p_ref = probs.gather(-1, ref_ids.expand(n, -1)[..., None]).squeeze(-1)
        # Prefer positions the model is least sure about; never pick a frozen one.
        weight = (1.0 - p_ref).clamp_min(1e-6).masked_fill(frozen_mask, 0.0)
        weight = torch.nan_to_num(weight, nan=1e-6).clamp_min(0.0)
        # multinomial without replacement needs at least `budget` positive weights
        # in every row; fall back to replacement if some row is degenerate.
        eligible = int((weight > 0).sum(dim=1).min())
        positions = torch.multinomial(weight, budget,
                                      replacement=budget > eligible)
        if exact:
            # Zero the reference residue at every position and renormalize, so a
            # draw at a chosen position is always a substitution.
            alt = scaled.scatter(-1, ref_ids.expand(n, -1)[..., None], -float("inf"))
            drawn = torch.distributions.Categorical(logits=alt).sample()
        else:
            drawn = torch.distributions.Categorical(logits=scaled).sample()
        decoded.scatter_(1, positions, drawn.gather(1, positions))

    return ["".join(AMINO_ACIDS[i] for i in row) for row in decoded.cpu().tolist()]


# ══════════════════════════════════════════════════════════════════════════════
# Oracle scoring
# ══════════════════════════════════════════════════════════════════════════════

def load_brightness_oracle(path: Path):
    """Load a fitted oracle from embedding_oracle.py, or return None.

    Kept optional and lazily imported: the METL backend pulls in the metl
    repository and its dependencies, which the toy peptide runs do not need.
    """
    if path is None or not Path(path).is_file():
        return None
    try:
        import embedding_oracle
        oracle = embedding_oracle.load_oracle(Path(path))
    except Exception as exc:
        print(f"  [skip] could not load oracle {path}: {type(exc).__name__}: {exc}")
        return None
    print(f"  Oracle: {oracle[3]} backend from {Path(path).name}")
    return oracle


def score_with_oracle(sequences, oracle):
    """Predicted brightness for decoded sequences, or None if no oracle."""
    if oracle is None:
        return None
    import embedding_oracle
    return embedding_oracle.score_sequences(list(sequences), oracle)


# ══════════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════════

def save_results(out_dir: Path, method: str, latents: dict, model, reward_model,
                 stats, esm_hf_id: str, length: int, dim: int,
                 min_polar: int, losses: list, config: dict | None = None):
    method_dir = out_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    for name, latent in latents.items():
        scores = latent.get("oracle")
        fasta = "".join(
            f">{name}_{i+1}"
            + (f" oracle_brightness={scores[i]:+.4f}" if scores is not None else "")
            + f"\n{seq}\n"
            for i, seq in enumerate(latent["sequences"]))
        (method_dir / f"{name}.fasta").write_text(fasta)
    torch.save({
        "standardized_latents": {k: v["latent"].cpu() for k, v in latents.items()},
        "sequences":    {k: v["sequences"] for k, v in latents.items()},
        "oracle_brightness": {k: v.get("oracle") for k, v in latents.items()},
        "model":        model.state_dict(),
        "reward_model": reward_model.state_dict(),
        "stats":        stats,
        "esm_name":     esm_hf_id,
        "length":       length,
        "dim":          dim,
        "min_polar":    min_polar,
        "polar_residues": POLAR_RESIDUES,
        "losses":       losses,
        "config":       config or {},
    }, method_dir / "results.pt")
    print(f"  Saved {method} results to {method_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--esm-model",  default="esm2_8m",
                        help=f"ESM-2 model tag. (default: esm2_8m)")
    parser.add_argument("--dataset",    type=Path,
                        default=Path(__file__).resolve().parent.parent / "lecture_3" / "esm2_example.csv",
                        help="Path to input CSV (sequence,c[,r1,r2]). (default: lecture_3/esm2_example.csv)")
    parser.add_argument("--dataset-tag", default=None,
                        help="Short label for the dataset used in the output dir name. "
                             "Defaults to the CSV filename stem.")
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--samples",    type=int, default=8)
    parser.add_argument("--min-polar",  type=int, default=12)
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum (and shared) sequence length. Raise to 237 for avGFP.")
    parser.add_argument("--mut-budget", type=int, default=None,
                        help="Decode as variants of a reference with at most this many "
                             "substitutions (replaces the min-polar constraint). "
                             "The reference defaults to the training-set consensus.")
    parser.add_argument("--reference",  type=Path, default=None,
                        help="File holding the reference sequence for --mut-budget "
                             "(e.g. data/avgfp_wt.txt). Default: training consensus.")
    parser.add_argument("--outdir",     type=Path, default=None,
                        help="Override output root. Default: outputs/<esm_model>_<dataset_tag>/")
    parser.add_argument("--anchor-strength", type=float, default=None, metavar="S",
                        help="Start sampling from the partially noised reference instead of "
                             "pure noise. S in (0,1]: 0.3 keeps the reference largely intact, "
                             "1.0 is the unanchored baseline. Requires --reference or "
                             "--mut-budget (which supplies the consensus reference).")
    parser.add_argument("--hidden", type=int, default=HIDDEN, metavar="H",
                        help=f"Width of the velocity/noise network (default: {HIDDEN}). "
                             "The network maps length*dim -> H -> H -> length*dim, so H "
                             "caps the rank of the learned field. At 237x320 the default "
                             "compresses 75,840 dimensions to 128, which may be what "
                             "limits GFP; the 64-sequence teaching set it was chosen for "
                             "is only 7,680.")
    parser.add_argument("--exact-mutations", action="store_true",
                        help="Give every sample exactly --mut-budget substitutions "
                             "instead of at most that many. Without it the budget is a "
                             "ceiling and the reference residue usually wins the draw, "
                             "so a budget of 5 yields ~1.5 substitutions and ~18% of "
                             "samples are the unmutated reference. Fixing the count "
                             "makes mutational distance a controlled variable.")
    parser.add_argument("--decode-temperature", type=float, default=0.0, metavar="T",
                        help="Sampling temperature for decoding (default: 0 = argmax). "
                             "Argmax is deterministic and collapses on this data: the "
                             "reference residue is the mode almost everywhere, so many "
                             "different latents decode to the same few sequences. T=0.7 "
                             "restores diversity at an unchanged Hamming distance.")
    parser.add_argument("--freeze-positions", default="", metavar="LIST",
                        help="Comma-separated 0-indexed positions never to substitute, "
                             "e.g. '0'. The avGFP wild type here omits the initiator "
                             "methionine, so ESM puts one back at position 0 and that "
                             "substitution otherwise dominates every sample.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, metavar="N",
                        help=f"Training minibatch size (default: {BATCH_SIZE}). The "
                             "default suits the 64-sequence teaching set; at tens of "
                             "thousands of sequences it leaves the GPU mostly idle and "
                             "128-256 trains several times faster per epoch.")
    parser.add_argument("--cfg-weight", type=float, default=2.0, metavar="W",
                        help="Classifier-free guidance weight for the 'cfg' mode "
                             "(default: 2.0, tuned on a 24x320 latent). A 237x320 "
                             "latent has a much larger norm, so this usually needs "
                             "raising before guidance changes the decoded output.")
    parser.add_argument("--reward-eta", type=float, default=1.0, metavar="ETA",
                        help="Reward-gradient strength for the 'single' and 'multi' "
                             "modes (default: 1.0). Same caveat as --cfg-weight.")
    parser.add_argument("--oracle", type=Path, default=None, metavar="NPZ",
                        help="Fitted brightness oracle from embedding_oracle.py. "
                             "Defaults to data/avgfp_metl_oracle.npz when it exists. "
                             "Scores every decoded sample; never touches guidance, so "
                             "it stays an independent judge. Use --no-oracle to skip.")
    parser.add_argument("--no-oracle", action="store_true",
                        help="Do not score samples, even if an oracle file is present.")
    parser.add_argument("--cache-dir",  type=Path, default=None,
                        help="Shared HuggingFace weight cache. Default: $ESM2_CACHE, "
                             "else project1_eval/cache/ (one copy per model, reused by all runs).")
    parser.add_argument("--plot",        action="store_true",
                        help="Generate all evaluation plots (including training-loss curves) "
                             "immediately after training, using in-memory results.")
    parser.add_argument("--ablate",      action="store_true",
                        help="Also run guidance-strength ablation plots (CFG-weight sweep and "
                             "reward-eta sweep). Implies --plot.")
    parser.add_argument("--list-models", action="store_true",
                        help="Print all known ESM-2 models and exit.")
    args = parser.parse_args()

    if args.list_models:
        print(list_models())
        return

    model_info  = get_model(args.esm_model)
    dataset_tag = args.dataset_tag or Path(args.dataset).stem
    run_tag     = f"{args.esm_model}_{dataset_tag}"
    out_root    = args.outdir or ROOT / "outputs" / run_tag
    cache_dir   = resolve_cache_dir(args.cache_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\nProject 1 — Experiment Runner")
    print(f"  ESM-2 model : {args.esm_model}  ({model_info['hf_id']})")
    print(f"  Dataset     : {args.dataset}  (tag: {dataset_tag})")
    print(f"  Run tag     : {run_tag}")
    print(f"  Output dir  : {out_root}")
    print(f"  Device      : {DEVICE}")
    print(f"  Guidance    : cfg w={args.cfg_weight}  reward eta={args.reward_eta}")
    print(f"  Batch size  : {args.batch_size}   hidden width: {args.hidden}")

    torch.manual_seed(7)
    if DEVICE.type == "cpu":
        torch.set_num_threads(2)

    # ── Encode ────────────────────────────────────────────────────────────────
    print("\nEncoding sequences with ESM-2...")
    dataset, esm, tokenizer, stats, sequences = load_data(
        args.dataset, model_info["hf_id"], cache_dir, args.max_length
    )
    _, length, dim = dataset.tensors[0].shape

    if not 0 <= args.min_polar <= length:
        parser.error(f"--min-polar must be between 0 and {length}")

    reference = None
    if args.reference is not None or args.mut_budget is not None or args.anchor_strength is not None:
        reference = (args.reference.read_text().strip().upper() if args.reference
                     else consensus(sequences))
        if len(reference) != length:
            parser.error(f"reference has {len(reference)} residues, expected {length}")
        source = "supplied" if args.reference else "consensus"

    frozen = tuple(int(x) for x in args.freeze_positions.replace(",", " ").split())
    if args.mut_budget is None:
        decode_fn = lambda z: decode(z, esm, tokenizer, stats, args.min_polar)
    else:
        rule = ("argmax" if args.decode_temperature <= 0
                else f"sampled at T={args.decode_temperature}")
        rule += ", exactly" if args.exact_mutations else ", at most"
        print(f"  Decoding {rule} {args.mut_budget} substitutions from the "
              f"{source} reference")
        if frozen:
            print(f"  Frozen positions (never substituted): {list(frozen)}")
        decode_fn = lambda z: decode_budget(z, esm, tokenizer, stats, reference,
                                            args.mut_budget, args.decode_temperature,
                                            frozen, args.exact_mutations)

    anchor = None
    if args.anchor_strength is not None:
        if not 0.0 < args.anchor_strength <= 1.0:
            parser.error("--anchor-strength must be in (0, 1]")
        anchor = encode_reference(reference, esm, tokenizer, stats)
        print(f"  Anchoring sampling at strength {args.anchor_strength} "
              f"from the {source} reference")
    anchor_kwargs = {} if anchor is None else {"anchor": anchor,
                                               "strength": args.anchor_strength}

    default_oracle = ROOT / "data" / "avgfp_metl_oracle.npz"
    oracle_path = None if args.no_oracle else (args.oracle or
                  (default_oracle if default_oracle.is_file() else None))
    oracle = load_brightness_oracle(oracle_path)
    if oracle is not None and len(oracle[2]) != length:
        # The oracle is fitted to one protein at one length. Silently scoring a
        # different one would produce confident nonsense.
        print(f"  [skip] oracle was fitted on a {len(oracle[2])}-residue protein but "
              f"these sequences are {length}; not scoring")
        oracle, oracle_path = None, None
    if oracle is None and not args.no_oracle and oracle_path is not None:
        print("  No brightness oracle loaded; samples will not be scored. "
              "Fit one with: python embedding_oracle.py --fit --backend metl")

    run_config = {
        "dataset":         str(args.dataset),
        "esm_model":       args.esm_model,
        "epochs":          args.epochs,
        "samples":         args.samples,
        "max_length":      args.max_length,
        "decode":          "mut_budget" if args.mut_budget is not None else "min_polar",
        "mut_budget":      args.mut_budget,
        "decode_temperature": args.decode_temperature,
        "exact_mutations":    args.exact_mutations,
        "freeze_positions":   list(frozen),
        "min_polar":       args.min_polar,
        "reference":       reference,
        "anchor_strength": args.anchor_strength,
        "batch_size":      args.batch_size,
        "hidden":          args.hidden,
        "cfg_weight":      args.cfg_weight,
        "reward_eta":      args.reward_eta,
        "oracle":          str(oracle_path) if oracle_path else None,
    }

    def report(name, seqs, scores=None, latent=None):
        unique = len(set(seqs))
        print(f"    distinct sequences {unique}/{len(seqs)}"
              + ("   <- collapsed; raise --decode-temperature" if unique < len(seqs) // 5
                 else ""))
        if latent is not None:
            norms = latent.reshape(len(latent), -1).norm(dim=1)
            diverged = int((norms > 3 * norms.median()).sum())
            if diverged:
                print(f"    {diverged} sample(s) numerically diverged "
                      f"(latent norm up to {norms.max():.0f} vs median "
                      f"{norms.median():.0f}); lower --cfg-weight/--reward-eta")
        if scores is not None:
            print(f"    oracle brightness  mean {scores.mean():+.3f}   "
                  f"best {scores.max():+.3f}   worst {scores.min():+.3f}")
        if reference is None:
            proxies = composition_proxies(seqs)
            print(f"  {name}: {seqs[0][:48]}...  "
                  f"polar={sum(a in POLAR_RESIDUES for a in seqs[0])}  "
                  f"r1={proxies[:,0].mean():.3f}  r2={proxies[:,1].mean():.3f}")
        else:
            distances = [sum(a != b for a, b in zip(s, reference)) for s in seqs]
            print(f"  {name}: {seqs[0][:48]}...  "
                  f"mean hamming to reference {sum(distances)/len(distances):.1f} "
                  f"(min {min(distances)}, max {max(distances)})")

    guidance_configs = {
        "cfg":    dict(c=1, w=args.cfg_weight, eta=0.0,            lambdas=(1., 0.)),
        "single": dict(c=1, w=0.0,             eta=args.reward_eta, lambdas=(1., 0.)),
        "multi":  dict(c=1, w=0.0,             eta=args.reward_eta, lambdas=(0.7, 0.3)),
    }

    # ── Flow matching ─────────────────────────────────────────────────────────
    print("\nTraining flow matching model...")
    torch.manual_seed(7)
    flow_model, flow_reward, flow_losses = train_flow(dataset, args.epochs, args.batch_size, args.hidden)

    print("\nSampling (flow)...")
    flow_latents = {}
    for name, cfg in guidance_configs.items():
        z    = sample_flow(flow_model, flow_reward, n=args.samples, **cfg, **anchor_kwargs)
        seqs = decode_fn(z)
        scores = score_with_oracle(seqs, oracle)
        report(name, seqs, scores, z)
        flow_latents[name] = {"latent": z, "sequences": seqs, "oracle": scores}
    save_results(out_root, "flow", flow_latents, flow_model, flow_reward,
                 stats, model_info["hf_id"], length, dim, args.min_polar, flow_losses,
                 run_config)

    # ── Diffusion ─────────────────────────────────────────────────────────────
    print("\nTraining diffusion model...")
    torch.manual_seed(7)
    diff_model, diff_reward, diff_losses, alpha_bars, betas, alphas, post_vars = \
        train_diffusion(dataset, args.epochs, args.batch_size, args.hidden)

    print("\nSampling (diffusion)...")
    diff_latents = {}
    for name, cfg in guidance_configs.items():
        z    = sample_diffusion(diff_model, diff_reward, alpha_bars, betas, alphas, post_vars,
                                n=args.samples, **cfg, **anchor_kwargs)
        seqs = decode_fn(z)
        scores = score_with_oracle(seqs, oracle)
        report(name, seqs, scores, z)
        diff_latents[name] = {"latent": z, "sequences": seqs, "oracle": scores}
    save_results(out_root, "diffusion", diff_latents, diff_model, diff_reward,
                 stats, model_info["hf_id"], length, dim, args.min_polar, diff_losses,
                 run_config)

    print(f"\nDone. Results in {out_root}/")

    # ── Oracle comparison ────────────────────────────────────────────────────
    if oracle is not None:
        print("\nOracle brightness by method and guidance mode "
              "(wild-type centered, higher is better)")
        header = (f"  {'method':<12}{'mode':<9}{'mean':>9}{'best':>9}"
                  f"{'hamming':>9}{'uniq':>7}")
        print(header + "\n  " + "-" * (len(header) - 2))
        for method, latents in (("flow", flow_latents), ("diffusion", diff_latents)):
            for name, latent in latents.items():
                scores = latent["oracle"]
                if reference is None:
                    distance = float("nan")
                else:
                    distance = float(np.mean([sum(a != b for a, b in zip(s, reference))
                                              for s in latent["sequences"]]))
                print(f"  {method:<12}{name:<9}{scores.mean():>9.3f}"
                      f"{scores.max():>9.3f}{distance:>9.1f}"
                      f"{len(set(latent['sequences'])):>4}/{len(latent['sequences'])}")
        identical = all(
            latents[m]["sequences"] == latents[list(latents)[0]]["sequences"]
            for latents in (flow_latents, diff_latents) for m in latents)
        if identical:
            print("\n  Warning: every guidance mode decoded to the same sequences. "
                  "Guidance is not\n  changing the output — raise the CFG weight w and "
                  "the reward weight eta.\n  Both defaults were tuned on a 24x320 latent; "
                  "this one is much larger.")
        print("\n  For the random-variant control and the full metric table, run:")
        print(f"    python gfp_metrics.py --run-dir {out_root} \\")
        print(f"        --embedding-oracle {oracle_path} --baseline-n 50")

    # ── Optional plots ────────────────────────────────────────────────────────
    if args.plot or args.ablate:
        print("\nGenerating plots...")
        import sys
        sys.path.insert(0, str(ROOT))
        from evaluate import (
            plot_training_loss, plot_composition_proxies, plot_latent_pca,
            plot_aa_composition, plot_polar_residue_distribution,
            plot_reward_pareto, plot_positional_entropy,
            plot_sequence_diversity, plot_summary_panel,
            plot_guidance_ablation, save_raw_data,
        )

        # Build the data dicts that evaluate.py plot functions expect —
        # everything is already in memory; no disk reload needed.
        flow_data = {
            "standardized_latents": {k: v["latent"].cpu() for k, v in flow_latents.items()},
            "sequences":            {k: v["sequences"]    for k, v in flow_latents.items()},
            "model":                flow_model.state_dict(),
            "reward_model":         flow_reward.state_dict(),
            "stats":                stats,
            "length":               length,
            "dim":                  dim,
            "min_polar":            args.min_polar,
        }
        diff_data = {
            "standardized_latents": {k: v["latent"].cpu() for k, v in diff_latents.items()},
            "sequences":            {k: v["sequences"]    for k, v in diff_latents.items()},
            "model":                diff_model.state_dict(),
            "reward_model":         diff_reward.state_dict(),
            "stats":                stats,
            "length":               length,
            "dim":                  dim,
            "min_polar":            args.min_polar,
        }

        # Read training sequences from the CSV we used (no re-encoding needed)
        with open(args.dataset, newline="") as f:
            train_seqs = [r["sequence"].strip().upper() for r in csv.DictReader(f)]

        prefix   = run_tag
        plot_dir = ROOT / "plots" / run_tag
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Raw metric CSVs — written before plots so data is always there
        print("  Saving raw metric CSVs...")
        save_raw_data(flow_data, diff_data, train_seqs, plot_dir, prefix,
                      flow_losses=flow_losses, diff_losses=diff_losses)

        # Training-loss curves — already collected epoch-by-epoch above
        plot_training_loss(flow_losses, diff_losses, plot_dir, prefix)

        # Static comparison plots
        plot_composition_proxies(flow_data, diff_data, train_seqs, plot_dir, prefix)
        try:
            plot_latent_pca(flow_data, diff_data, plot_dir, prefix)
        except ImportError:
            print("  [skip] PCA — pip install scikit-learn")
        plot_aa_composition(flow_data, diff_data, train_seqs, plot_dir, prefix)
        plot_polar_residue_distribution(flow_data, diff_data, plot_dir, prefix)
        plot_reward_pareto(flow_data, diff_data, train_seqs, plot_dir, prefix)
        plot_positional_entropy(flow_data, diff_data, train_seqs, plot_dir, prefix)
        plot_sequence_diversity(flow_data, diff_data, train_seqs, plot_dir, prefix)
        plot_summary_panel(flow_data, diff_data, train_seqs, plot_dir, prefix, run_tag)

        # Ablation — uses the already-trained models (no extra training)
        if args.ablate:
            print("  Running guidance-strength ablation (uses in-memory models)...")
            plot_guidance_ablation(
                flow_model, flow_reward, diff_model, diff_reward,
                (betas, alphas, alpha_bars, post_vars),
                stats, stats,   # same stats for both (shared encoding)
                plot_dir, prefix,
            )

        n_png = len(list(plot_dir.glob("*.png")))
        n_csv = len(list(plot_dir.glob("*.csv")))
        print(f"\n{n_png} plots + {n_csv} CSV files saved to {plot_dir}/")
    else:
        print(f"To plot: python evaluate.py --model-name {args.esm_model} --dataset {dataset_tag} "
              f"--flow-outdir {out_root}/flow --diff-outdir {out_root}/diffusion")


if __name__ == "__main__":
    main()
