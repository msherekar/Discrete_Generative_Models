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
    def __init__(self, length, dim, hidden=HIDDEN, arch="mlp"):
        super().__init__()
        self.length, self.dim, self.arch = length, dim, arch
        self.time = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.skip = nn.Linear(32, 1)
        if arch == "transformer":
            self.trunk = TransformerField(length, dim, d_model=hidden)
        else:
            self.condition = nn.Embedding(3, 16)
            self.net = nn.Sequential(
                nn.Linear(length * dim + 48, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, length * dim),
            )

    def forward(self, z, t, c):
        time = self.time(t[:, None])
        if self.arch == "transformer":
            return self.skip(time)[:, :, None] * z + self.trunk(z, t, c)
        inputs = torch.cat([z.flatten(1), time, self.condition(c)], dim=1)
        return self.skip(time)[:, :, None] * z + self.net(inputs).reshape_as(z)


class DiffusionModel(nn.Module):
    """Denoiser for the DDPM chain.

    `predict` selects the parameterization.

    "eps" is the original: the output is
        sqrt(1-abar_k) * z  +  sqrt(abar_k) * net(...)
    At high noise abar_k -> 0, so the prediction collapses to z, which already
    equals the added noise almost exactly -- the network is handed a correct
    answer for free and contributes nothing. Measured on the default schedule
    its weight exceeds 0.5 for only 369 of 1000 steps, and across five seeds the
    training loss fell just 25% (flow's fell 78%).

    "x0" follows Lecture 3.3's AMP-Diffusion recipe -- "The network predicts
    Z0_hat" -- so the network owns the prediction at every noise level. The
    sampler converts back with
        eps_hat = (z_k - sqrt(abar_k) x0_hat) / sqrt(1 - abar_k)
    and reuses the same reverse update.
    """

    def __init__(self, length, dim, alpha_bars, hidden=HIDDEN, arch="mlp",
                 predict="eps"):
        super().__init__()
        self.length, self.dim, self.arch, self.predict = length, dim, arch, predict
        self.alpha_bars = alpha_bars
        self.K          = len(alpha_bars) - 1
        if arch == "transformer":
            self.trunk = TransformerField(length, dim, d_model=hidden)
        else:
            self.time      = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
            self.condition = nn.Embedding(3, 16)
            self.net = nn.Sequential(
                nn.Linear(length * dim + 48, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, length * dim),
            )

    def forward(self, z, t, c):
        if self.arch == "transformer":
            raw = self.trunk(z, t, c)
        else:
            time   = self.time(t[:, None])
            inputs = torch.cat([z.flatten(1), time, self.condition(c)], dim=1)
            raw = self.net(inputs).reshape_as(z)
        if self.predict == "x0":
            return raw                      # the clean-latent estimate itself
        k = (t * self.K).round().long().clamp(0, self.K)
        a = self.alpha_bars[k, None, None]
        return (1 - a).sqrt() * z + a.sqrt() * raw


class TransformerField(nn.Module):
    """Per-residue Transformer trunk, following Lecture 3.3's AMP-Diffusion recipe.

    The MLP trunk flattens [L, D] into one vector, so a position's identity is
    encoded only in where it lands in that vector and nothing is shared between
    positions. On avGFP that is 75,840 inputs compressed through a single hidden
    layer, and 94.7% of the latent variance is positional -- the model fits the
    shared backbone and loses the per-variant signal.

    Lecture 3.3: "A Transformer replaces the U-Net... Adds residue-position
    information because order matters in a protein sequence... We will stack six
    Transformer layers... Add the noisy residue embeddings, the diffusion-step
    embedding shared across positions, and a learned embedding for each token
    position before applying the Transformer."

    Weights are shared across positions, so width no longer scales with sequence
    length: this trunk is roughly 5M parameters at any L, against 235M for the
    flattened MLP at width 1024.
    """

    def __init__(self, length, dim, d_model=256, layers=6, heads=8, dropout=0.0):
        super().__init__()
        self.project_in = nn.Linear(dim, d_model)
        self.position   = nn.Parameter(torch.zeros(1, length, d_model))
        nn.init.normal_(self.position, std=0.02)
        self.time = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(),
                                  nn.Linear(d_model, d_model))
        self.condition = nn.Embedding(3, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.project_out = nn.Linear(d_model, dim)
        nn.init.zeros_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)

    def forward(self, z, t, c=None):
        h = self.project_in(z) + self.position + self.time(t[:, None])[:, None, :]
        if c is not None:
            h = h + self.condition(c)[:, None, :]
        return self.project_out(self.encoder(h))


class RewardModel(nn.Module):
    def __init__(self, length, dim, hidden=HIDDEN, arch="mlp"):
        super().__init__()
        self.arch = arch
        if arch == "transformer":
            # Same trunk, then mean-pool over residues to one value per objective.
            self.trunk = TransformerField(length, dim, d_model=hidden)
            self.head = nn.Linear(dim, 2)
        else:
            self.net = nn.Sequential(
                nn.Linear(length * dim + 1, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2),
            )

    def forward(self, z, t):
        if self.arch == "transformer":
            return self.head(self.trunk(z, t).mean(dim=1))
        return self.net(torch.cat([z.flatten(1), t[:, None]], dim=1))


class EMA:
    """Exponential moving average of the weights, used for sampling.

    Standard practice in diffusion training (DDPM uses decay 0.9999) and absent
    here. The averaged weights are far less sensitive to which minibatch landed
    last, which is what drives the run-to-run spread: diffusion's seed-to-seed
    variation was 5-10x flow's (+/-0.04-0.06 against +/-0.005-0.008).
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        state = model.state_dict()
        for k, v in self.shadow.items():
            state[k].copy_(v.to(state[k].dtype))


def sample_timesteps(n, K, device, stratified=False):
    """Diffusion steps for one minibatch.

    Independent uniform draws leave most of the schedule unvisited in a small
    batch: at K=1000 and batch 16, one batch touches 1.6% of it. Stratified
    sampling puts one draw in each of n equal bins, so every batch spans the
    whole schedule and the gradient carries less variance.
    """
    if not stratified:
        return torch.randint(1, K + 1, (n,), device=device)
    edges = torch.arange(n, device=device, dtype=torch.float32)
    k = ((edges + torch.rand(n, device=device)) / n * K).long().clamp(1, K)
    return k[torch.randperm(n, device=device)]


def make_ddpm_schedule(K=1000):
    # Fewer steps means each minibatch covers more of the schedule.
    betas      = torch.cat([torch.zeros(1), torch.linspace(1e-4, 0.02, K)]).to(DEVICE)
    alphas     = 1.0 - betas
    alpha_bars = alphas.cumprod(0)
    previous   = torch.cat([torch.ones(1, device=DEVICE), alpha_bars[:-1]])
    post_vars  = betas * (1 - previous) / (1 - alpha_bars).clamp_min(1e-20)
    return betas, alphas, alpha_bars, post_vars


# ══════════════════════════════════════════════════════════════════════════════
# Probability paths (interpolants)
# ══════════════════════════════════════════════════════════════════════════════

def interpolate(z0, z1, t, kind="linear"):
    """Intermediate state and its conditional velocity for a chosen interpolant.

    A path is X_t = alpha(t) X0 + beta(t) X1 with alpha(0)=beta(1)=1 and
    alpha(1)=beta(0)=0, so the endpoints are preserved whatever the schedule.
    Lecture 2.2 works these three and notes that changing the interpolant "can
    help when velocities are difficult to learn or sampling requires many
    integration steps, because each choice changes the intermediate
    distributions and velocity targets."

      linear     X_t = (1-t) X0 + t X1              U_t = X1 - X0
                 constant speed; the usual default.
      quadratic  X_t = (1-t^2) X0 + t^2 X1          U_t = 2t (X1 - X0)
                 leaves the prior slowly and accelerates; a quarter of the way
                 through, only 6.25% of the distance is covered.
      trig       X_t = cos(pi t/2) X0 + sin(pi t/2) X1
                 U_t = (pi/2)(-sin(pi t/2) X0 + cos(pi t/2) X1)
                 curved path; for independent centered endpoints with identity
                 covariance it preserves that covariance throughout.
    """
    shape = (-1,) + (1,) * (z1.dim() - 1)
    tt = t.view(shape)
    if kind == "linear":
        return (1 - tt) * z0 + tt * z1, z1 - z0
    if kind == "quadratic":
        return (1 - tt ** 2) * z0 + tt ** 2 * z1, 2 * tt * (z1 - z0)
    if kind == "trig":
        half_pi = torch.pi / 2
        a, b = torch.cos(half_pi * tt), torch.sin(half_pi * tt)
        return a * z0 + b * z1, half_pi * (-b * z0 + a * z1)
    raise ValueError(f"unknown interpolant '{kind}'")


INTERPOLANTS = ("linear", "quadratic", "trig")


def endpoint_from_velocity(z, t, v, kind="linear"):
    """Estimate the clean endpoint X1 from the current state and predicted velocity.

    For a path X_t = a(t) X0 + b(t) X1 the velocity is U_t = a'(t) X0 + b'(t) X1.
    Eliminating X0 between the two gives

        X1 = (a'(t) X_t - a(t) U_t) / (a'(t) b(t) - a(t) b'(t))

    which for the linear path collapses to the familiar X1 = X_t + (1-t) U_t.
    The quadratic path's denominator vanishes at t=0, so it is clamped: at the
    very start of generation the state carries no information about the endpoint
    and the estimate is meaningless anyway.
    """
    shape = (-1,) + (1,) * (z.dim() - 1)
    tt = t.view(shape)
    if kind == "linear":
        return z + (1 - tt) * v
    if kind == "quadratic":
        return z + (1 - tt ** 2) * v / (2 * tt).clamp_min(1e-3)
    if kind == "trig":
        half_pi = torch.pi / 2
        return torch.sin(half_pi * tt) * z + (2 / torch.pi) * torch.cos(half_pi * tt) * v
    raise ValueError(f"unknown interpolant '{kind}'")


def endpoint_from_noise(z, eps, alpha_bar):
    """Clean-sample estimate from a DDPM state and its predicted noise.

    Inverting z_k = sqrt(abar) x0 + sqrt(1-abar) eps for x0.
    """
    return (z - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt().clamp_min(1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_flow(dataset, epochs, batch_size=BATCH_SIZE, hidden=HIDDEN,
               interpolant="linear", arch="mlp"):
    _, length, dim = dataset.tensors[0].shape
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model  = FlowModel(length, dim, hidden, arch).to(DEVICE)
    reward = RewardModel(length, dim, hidden, arch).to(DEVICE)
    opt    = torch.optim.Adam(list(model.parameters()) + list(reward.parameters()), lr=LEARNING_RATE)
    losses = []
    for epoch in range(epochs):
        total = 0.0
        for z1, c, r_tilde in loader:
            z1, c, r_tilde = z1.to(DEVICE), c.to(DEVICE), r_tilde.to(DEVICE)
            z0 = torch.randn_like(z1)
            t  = torch.rand(len(z1), device=DEVICE)
            zt, target = interpolate(z0, z1, t, interpolant)
            dropped = c.masked_fill(torch.rand(len(c), device=DEVICE) < CONDITION_DROP, 2)
            loss = F.mse_loss(model(zt, t, dropped), target) + F.mse_loss(reward(zt, t), r_tilde)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        if (epoch + 1) % max(1, epochs // 4) == 0 or epoch + 1 == epochs:
            print(f"  [flow]      epoch {epoch+1:>4}/{epochs}: loss {avg:.4f}")
    model.eval().requires_grad_(False)
    reward.eval().requires_grad_(False)
    return model, reward, losses


def train_diffusion(dataset, epochs, batch_size=BATCH_SIZE, hidden=HIDDEN, arch="mlp",
                    predict="eps", steps=1000, stratified=False, ema_decay=0.0):
    _, length, dim = dataset.tensors[0].shape
    betas, alphas, alpha_bars, post_vars = make_ddpm_schedule(steps)
    K      = len(betas) - 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model  = DiffusionModel(length, dim, alpha_bars, hidden, arch, predict).to(DEVICE)
    reward = RewardModel(length, dim, hidden, arch).to(DEVICE)
    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    opt    = torch.optim.Adam(list(model.parameters()) + list(reward.parameters()), lr=LEARNING_RATE)
    losses = []
    for epoch in range(epochs):
        total = 0.0
        for z0, c, r_tilde in loader:
            z0, c, r_tilde = z0.to(DEVICE), c.to(DEVICE), r_tilde.to(DEVICE)
            k = sample_timesteps(len(z0), K, DEVICE, stratified)
            t = k.float() / K
            a = alpha_bars[k, None, None]
            eps = torch.randn_like(z0)
            zk  = a.sqrt() * z0 + (1 - a).sqrt() * eps
            dropped = c.masked_fill(torch.rand(len(c), device=DEVICE) < CONDITION_DROP, 2)
            target = z0 if predict == "x0" else eps
            loss = F.mse_loss(model(zk, t, dropped), target) + F.mse_loss(reward(zk, t), r_tilde)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if ema is not None:
                ema.update(model)
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        if (epoch + 1) % max(1, epochs // 4) == 0 or epoch + 1 == epochs:
            print(f"  [diffusion] epoch {epoch+1:>4}/{epochs}: loss {avg:.4f}")
    if ema is not None:
        ema.copy_to(model)          # sample from the averaged weights
    model.eval().requires_grad_(False)
    reward.eval().requires_grad_(False)
    return model, reward, losses, alpha_bars, betas, alphas, post_vars


# ══════════════════════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════════════════════

def reward_gradient(reward_model, z, t, lambdas, clip=0.0, normalize=False,
                    endpoint=None):
    """Gradient of the lambda-weighted reward with respect to the latent.

    Lecture 3.4 lists two precautions this implements.

    `normalize` takes each objective's gradient to unit norm before the weighted
    sum, so lambda sets relative importance rather than being confounded with
    whatever scale each reward head happens to have learned. Measured on the
    avGFP reward model the two objectives differ by 1.6x in gradient norm
    (5.24 vs 3.37), so without this a lambda of (0.7, 0.3) is not the ratio it
    appears to be, and a workable eta does not transfer between runs.

    `clip` bounds the per-sample gradient norm. Guidance adds this term at every
    integration step, so an unbounded gradient compounds: a 1,000-step DDPM chain
    at eta=50 reached latent norms of 4.6e17 and destroyed 32 of 100 samples,
    while the 200-step flow path at the same eta stayed bounded.
    """
    with torch.enable_grad():
        state = z.detach().requires_grad_(True)
        if endpoint is None:
            rewards = reward_model(state, t)
        else:
            # Lecture 3.4: "Some properties have little meaning for an
            # intermediate state. A partially denoised protein latent, for
            # example, may not yet correspond to a valid sequence." So predict
            # the clean endpoint, score THAT, and differentiate back through the
            # endpoint estimator. Measured on the avGFP reward model, accuracy
            # goes from Spearman 0.26 at t=0.1 to 0.82 on a clean latent, so this
            # replaces a nearly uninformative signal early in the trajectory.
            x1 = endpoint(state)
            rewards = reward_model(x1, torch.ones_like(t))
        if normalize:
            # One backward per objective so each can be scaled independently.
            total = torch.zeros_like(state)
            for j in range(rewards.shape[1]):
                if float(lambdas[j]) == 0.0:
                    continue
                g = torch.autograd.grad(rewards[:, j].sum(), state, retain_graph=True)[0]
                norm = g.flatten(1).norm(dim=1).clamp_min(1e-12)
                total = total + lambdas[j] * g / norm.view(-1, *([1] * (g.dim() - 1)))
            grad = total
        else:
            grad = torch.autograd.grad((rewards * lambdas).sum(1).sum(), state)[0]
    grad = grad.detach()
    if clip and clip > 0:
        norm = grad.flatten(1).norm(dim=1)
        scale = (clip / norm.clamp_min(1e-12)).clamp(max=1.0)
        grad = grad * scale.view(-1, *([1] * (grad.dim() - 1)))
    return grad


@torch.no_grad()
def sample_flow(model, reward_model, n=8, c=1, w=0.0, eta=0.0, lambdas=(1., 0.), steps=200,
                anchor=None, strength=1.0, clip=0.0, normalize=False,
                interpolant="linear", endpoint_guidance=False, seed=123):
    """Integrate the velocity field from t=0 to t=1.

    With `anchor` (a standardized reference latent) the trajectory starts from a
    partially noised anchor at t = 1 - strength instead of pure noise at t = 0.
    Every measured avGFP variant sits within 15 substitutions of the wild type,
    a vanishingly small region of a 237x320 latent space, so integrating from
    N(0, I) rarely lands on it. strength=1.0 reproduces the unanchored path.
    """
    lam = torch.tensor(lambdas, dtype=torch.float32, device=DEVICE)
    lam = lam / lam.sum()
    torch.manual_seed(seed)
    start = 0.0
    if anchor is None:
        z = torch.randn(n, model.length, model.dim, device=DEVICE)
    else:
        start = 1.0 - float(strength)
        z1    = anchor.to(DEVICE).expand(n, -1, -1)
        z0    = torch.randn(n, model.length, model.dim, device=DEVICE)
        # Start on the path the model was trained against, not a linear guess.
        z, _  = interpolate(z0, z1, torch.full((n,), start, device=DEVICE), interpolant)
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
            endpoint = None
            if endpoint_guidance:
                # Re-predict the velocity inside the gradient tape so the reward
                # gradient flows back through the endpoint estimate as well.
                def endpoint(state, _t=t, _null=null):
                    return endpoint_from_velocity(state, _t, model(state, _t, _null),
                                                  interpolant)
            v = v + kappa * reward_gradient(reward_model, z, t, lam, clip, normalize,
                                            endpoint)
        z = z + dt * v
    return z


@torch.no_grad()
def sample_diffusion(model, reward_model, alpha_bars, betas, alphas, post_vars,
                     n=8, c=1, w=0.0, eta=0.0, lambdas=(1., 0.),
                     anchor=None, strength=1.0, clip=0.0, normalize=False,
                     endpoint_guidance=False, seed=123):
    """Run the reverse chain from step K down to 1.

    With `anchor` the chain starts at step round(strength * K) from the forward-
    noised anchor (SDEdit), rather than at K from pure noise. strength=1.0
    reproduces the unanchored chain.
    """
    lam = torch.tensor(lambdas, dtype=torch.float32, device=DEVICE)
    lam = lam / lam.sum()
    K    = len(betas) - 1
    torch.manual_seed(seed)
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
        a_k = alpha_bars[k]

        def noise_pred(state, condition):
            out = model(state, t, condition)
            if model.predict != "x0":
                return out
            # Lecture 3.3: "Convert Z0_hat into a noise estimate and reuse the
            # DDPM reverse update."
            return (state - a_k.sqrt() * out) / (1 - a_k).sqrt().clamp_min(1e-4)

        eps = noise_pred(z, null)
        if w:
            eps = eps + w * (noise_pred(z, cond) - noise_pred(z, null))
        sigma = (1 - alpha_bars[k]).sqrt()
        if eta:
            endpoint = None
            if endpoint_guidance:
                def endpoint(state, _t=t, _null=null, _a=alpha_bars[k]):
                    out = model(state, _t, _null)
                    # In x0 mode the network already returns the clean estimate.
                    return out if model.predict == "x0" else \
                        endpoint_from_noise(state, out, _a)
            eps = eps - eta * sigma * reward_gradient(reward_model, z, t, lam,
                                                      clip, normalize, endpoint)
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
def load_support_mask(path, reference, level="substitution"):
    """Which (position, residue) substitutions the assay actually measured.

    Returns a bool array of shape (len(reference), 20), True where a substitution
    is inside the measured support. `None` is never returned: an empty mask is an
    error worth raising, not a silent pass-through.

    Why restrict generation at all. The oracle is fitted on a deep mutational
    scan, and a scan is not exhaustive -- avGFP's covers about 38% of the 4,503
    possible single substitutions. Outside that set an indicator oracle has a
    zero coefficient and quietly returns its intercept, so an unmeasured
    substitution is scored as if it were neutral rather than as unknown. Sampling
    freely and then reporting a mean brightness therefore averages real
    predictions with fallbacks. Constraining the decoder to the support makes
    every sample one the oracle was supervised on, which is the difference
    between a measurement and an extrapolation.

    Two sources, detected by extension:
      .npz  a ridge oracle from gfp_oracle.py -- support is its nonzero
            coefficients, i.e. exactly what that oracle can score. Prefer this
            when the ridge oracle is the judge, since the two then agree by
            construction.
      .csv  a variant table with a `sequence` column -- support is every
            substitution appearing in it, read off against `reference`. Use this
            for a dense oracle (METL, ESM) that has no sparse support of its own
            but was still only ever supervised on measured variants.

    `level` chooses how tight the constraint is, which is the ablation:
      substitution  only measured (position, residue) pairs. 1,551 of 4,503.
      position      any residue at a position carrying >=1 measured substitution.
                    On avGFP that is 233 of 237 positions, so it is a much weaker
                    constraint than it sounds -- included precisely to show that
                    the residue identity, not the position, is what matters.
    """
    path = Path(path)
    length, n_aa = len(reference), len(AMINO_ACIDS)
    mask = np.zeros((length, n_aa), dtype=bool)
    if path.suffix == ".npz":
        saved = np.load(path, allow_pickle=False)
        if "coef" not in saved:
            raise SystemExit(f"{path} has no 'coef'; not a gfp_oracle.py oracle")
        coef = np.asarray(saved["coef"], dtype=np.float64)
        if coef.size != length * n_aa:
            raise SystemExit(f"{path} has {coef.size} coefficients but reference "
                             f"needs {length * n_aa}; oracle and reference disagree")
        mask = np.abs(coef.reshape(length, n_aa)) > 1e-9
    elif path.suffix == ".csv":
        index = {a: i for i, a in enumerate(AMINO_ACIDS)}
        with path.open(newline="") as handle:
            rows = csv.DictReader(handle)
            if "sequence" not in (rows.fieldnames or []):
                raise SystemExit(f"{path} has no 'sequence' column")
            for row in rows:
                seq = row["sequence"]
                if len(seq) != length:
                    continue
                for position, (a, b) in enumerate(zip(reference, seq)):
                    if a != b and b in index:
                        mask[position, index[b]] = True
    else:
        raise SystemExit(f"--restrict-support wants a .npz or .csv, got {path.suffix}")

    # The reference residue is not a substitution; drop it so `mask` counts only
    # real alternatives. decode_budget re-admits it as the no-mutation option.
    mask[np.arange(length), [AMINO_ACIDS.index(a) for a in reference]] = False
    if level == "position":
        mask = np.repeat(mask.any(axis=1, keepdims=True), n_aa, axis=1)
        mask[np.arange(length), [AMINO_ACIDS.index(a) for a in reference]] = False
    elif level != "substitution":
        raise SystemExit(f"unknown --support-level '{level}'")
    if not mask.any():
        raise SystemExit(f"{path} yielded an empty support mask; wrong reference?")
    return mask


def decode_budget(z, esm, tokenizer, stats, reference, budget,
                  temperature=0.0, frozen=(), exact=False, allowed=None):
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

    `allowed` is an optional (length, 20) bool mask from `load_support_mask`
    restricting which substitutions may be produced. It is applied to the logits
    before any choice is made, so the model still ranks the permitted residues by
    its own preference -- the constraint removes options, it does not pick for the
    model. Positions with no permitted alternative are treated as frozen, and
    with `allowed` set every returned sequence is inside the oracle's support by
    construction.
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

    # Out-of-support substitutions are removed from the menu, and a position left
    # with nothing to choose from becomes frozen rather than a source of -inf rows.
    if allowed is not None:
        permitted = torch.as_tensor(allowed, dtype=torch.bool, device=device)
        if permitted.shape != (length, len(AMINO_ACIDS)):
            raise SystemExit(f"support mask is {tuple(permitted.shape)}, expected "
                             f"{(length, len(AMINO_ACIDS))}")
        frozen_mask |= ~permitted.any(dim=-1)
        # Keep the reference residue selectable: it is the no-mutation outcome.
        keepable = permitted.clone()
        keepable[torch.arange(length, device=device), ref_ids] = True
        logits = logits.masked_fill(~keepable, -float("inf"))

    if temperature <= 0:
        # Exclude the reference residue so "best" always means a real substitution.
        masked = logits.scatter(-1, ref_ids.expand(n, -1)[..., None], -float("inf"))
        alt_scores, alt_choices = masked.max(dim=-1)
        ref_scores = logits.gather(-1, ref_ids.expand(n, -1)[..., None]).squeeze(-1)
        margin = (alt_scores - ref_scores).masked_fill(frozen_mask, -float("inf"))
        keep = margin.topk(budget, dim=1).indices
        for i in range(n):
            positions = keep[i] if exact else keep[i][margin[i, keep[i]] > 0]
            # topk still returns `budget` indices when fewer than `budget`
            # positions are eligible; drop the ineligible ones.
            positions = positions[torch.isfinite(margin[i, positions])]
            decoded[i, positions] = alt_choices[i, positions]
    else:
        # A sample whose latent diverged (large --cfg-weight or --reward-eta can do
        # this) produces non-finite logits. Sanitize rather than crash: the run
        # already warns about diverged latents, and one bad sample should not take
        # the whole batch down.
        scaled = torch.nan_to_num(logits / temperature, nan=0.0,
                                  posinf=30.0, neginf=-30.0)
        if allowed is not None:
            # nan_to_num turned the masked -inf into a finite -30, which leaves a
            # tiny but real chance of drawing an out-of-support residue. Re-impose
            # the mask now that the sanitizing pass is done.
            scaled = scaled.masked_fill(~keepable, -float("inf"))
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
            # Positions with no permitted alternative leave an all -inf row, which
            # Categorical cannot normalize. Those positions are frozen and never
            # read, but the distribution is built over every position, so give them
            # a uniform row to keep the draw finite.
            dead = ~torch.isfinite(alt).any(dim=-1, keepdim=True)
            alt = torch.where(dead, torch.zeros_like(alt), alt)
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
    parser.add_argument("--arch", default="mlp", choices=("mlp", "transformer"),
                        help="Velocity/noise network. 'mlp' flattens [L,D] into one "
                             "vector -- Lecture 3.2's recipe for small vectors in R^64. "
                             "'transformer' is Lecture 3.3's AMP-Diffusion trunk: six "
                             "self-attention layers over residues with learned positional "
                             "embeddings, weights shared across positions. With --arch "
                             "transformer, --hidden sets d_model (try 256).")
    parser.add_argument("--interpolant", default="linear", choices=INTERPOLANTS,
                        help="Probability path between prior and data (default: linear). "
                             "quadratic leaves the prior slowly and accelerates; trig "
                             "follows a curved, covariance-preserving path. Lecture 2.2 "
                             "suggests changing this when velocities are hard to learn.")
    parser.add_argument("--guidance-clip", type=float, default=0.0, metavar="C",
                        help="Bound the per-sample reward-gradient norm at C "
                             "(default: 0 = no clipping). Lecture 3.4: 'Clip unusually "
                             "large guidance gradients'. Without it, diffusion diverged "
                             "at eta=50.")
    parser.add_argument("--predict", default="eps", choices=("eps", "x0"),
                        help="What the diffusion network predicts (default: eps). "
                             "'x0' is Lecture 3.3's AMP-Diffusion recipe and makes the "
                             "network responsible at every noise level; with 'eps' its "
                             "weight exceeds 0.5 for only 369 of 1000 steps.")
    parser.add_argument("--ema", type=float, default=0.0, metavar="DECAY",
                        help="Exponential moving average of diffusion weights for "
                             "sampling, e.g. 0.999 (default: 0 = off). Standard in DDPM "
                             "training and the usual cure for run-to-run variance.")
    parser.add_argument("--diffusion-steps", type=int, default=1000, metavar="K",
                        help="Length of the DDPM schedule (default: 1000). Smaller K "
                             "means each minibatch covers more of it.")
    parser.add_argument("--stratified-timesteps", action="store_true",
                        help="Spread each minibatch's diffusion steps evenly over the "
                             "schedule instead of drawing them independently.")
    parser.add_argument("--seed", type=int, default=7, metavar="S",
                        help="Seed for weight initialization and batch order "
                             "(default: 7, the value hardcoded before this flag existed).")
    parser.add_argument("--sample-seed", type=int, default=123, metavar="S",
                        help="Seed for the sampling noise (default: 123, the value "
                             "hardcoded before this flag existed). Kept separate from "
                             "--seed so the defaults reproduce pre-flag runs exactly. "
                             "For independent replicates vary BOTH, e.g. "
                             "--seed 11 --sample-seed 11.")
    parser.add_argument("--endpoint-guidance", action="store_true",
                        help="Score the PREDICTED CLEAN ENDPOINT instead of the noisy "
                             "state, and differentiate back through it. Lecture 3.4: "
                             "'Evaluate rewards on a predicted clean endpoint', because "
                             "'a partially denoised protein latent may not yet "
                             "correspond to a valid sequence'. The avGFP reward model "
                             "scores Spearman 0.26 at t=0.1 but 0.82 on a clean latent. "
                             "Costs one extra network evaluation per guided step.")
    parser.add_argument("--normalize-guidance", action="store_true",
                        help="Scale each objective's gradient to unit norm before the "
                             "lambda-weighted sum. Lecture 3.4: 'Normalize objectives "
                             "before combining them'. Makes lambda a true importance "
                             "ratio and makes eta comparable across runs.")
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
                             "so a budget of 5 yields ~1.5 substitutions and ~18%% of "
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
    parser.add_argument("--restrict-support", type=Path, default=None, metavar="NPZ|CSV",
                        help="Only produce substitutions the assay measured. Takes a "
                             "gfp_oracle.py .npz (support = its nonzero coefficients) "
                             "or a variant .csv with a 'sequence' column (support = "
                             "every substitution in it). avGFP's scan covers ~38%% of "
                             "the 4,503 possible substitutions; outside that set the "
                             "oracle returns its intercept, so unrestricted samples mix "
                             "predictions with fallbacks. With this flag every sample is "
                             "in-domain by construction. Requires --reference.")
    parser.add_argument("--support-level", default="substitution",
                        choices=("substitution", "position"),
                        help="How tight --restrict-support is (default: substitution). "
                             "'substitution' permits only measured (position, residue) "
                             "pairs; 'position' permits any residue at a measured "
                             "position, which on avGFP is 233 of 237 and so barely "
                             "constrains anything -- it is the ablation showing that "
                             "residue identity, not position, is what carries the "
                             "constraint.")
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
    print(f"  Guidance    : cfg w={args.cfg_weight}  reward eta={args.reward_eta}"
          + (f"  clip={args.guidance_clip}" if args.guidance_clip else "")
          + ("  normalized" if args.normalize_guidance else "")
          + ("  endpoint" if args.endpoint_guidance else ""))
    print(f"  Batch size  : {args.batch_size}   hidden width: {args.hidden}")
    print(f"  Interpolant : {args.interpolant}   architecture: {args.arch}")
    print(f"  Seed        : {args.seed} (train)  {args.sample_seed} (sampling)")
    print(f"  Diffusion   : predict={args.predict}  K={args.diffusion_steps}"
          + (f"  ema={args.ema}" if args.ema else "")
          + ("  stratified" if args.stratified_timesteps else ""))

    torch.manual_seed(args.seed)
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
    support = None
    if args.restrict_support is not None:
        if args.mut_budget is None:
            parser.error("--restrict-support only applies to budgeted decoding; "
                         "pass --mut-budget as well")
        support = load_support_mask(args.restrict_support, reference,
                                    args.support_level)
        per_position = support.sum(axis=1)
        print(f"  Support     : {args.restrict_support.name} "
              f"({args.support_level}) — {int(support.sum())} of "
              f"{length * (len(AMINO_ACIDS) - 1)} substitutions "
              f"({100 * support.sum() / (length * (len(AMINO_ACIDS) - 1)):.1f}%), "
              f"{int((per_position > 0).sum())} of {length} positions usable, "
              f"median {int(np.median(per_position))} alternatives each")
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
                                            frozen, args.exact_mutations, support)

    anchor = None
    if args.anchor_strength is not None:
        if not 0.0 < args.anchor_strength <= 1.0:
            parser.error("--anchor-strength must be in (0, 1]")
        anchor = encode_reference(reference, esm, tokenizer, stats)
        print(f"  Anchoring sampling at strength {args.anchor_strength} "
              f"from the {source} reference")
    anchor_kwargs = {} if anchor is None else {"anchor": anchor,
                                               "strength": args.anchor_strength}
    guide_kwargs = {"clip": args.guidance_clip, "normalize": args.normalize_guidance,
                    "endpoint_guidance": args.endpoint_guidance,
                    "seed": args.sample_seed}
    flow_path = {"interpolant": args.interpolant}

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
        "interpolant":     args.interpolant,
        "mut_budget":      args.mut_budget,
        "decode_temperature": args.decode_temperature,
        "exact_mutations":    args.exact_mutations,
        "restrict_support":   (str(args.restrict_support)
                               if args.restrict_support else None),
        "support_level":      args.support_level if args.restrict_support else None,
        "freeze_positions":   list(frozen),
        "min_polar":       args.min_polar,
        "reference":       reference,
        "anchor_strength": args.anchor_strength,
        "batch_size":      args.batch_size,
        "hidden":          args.hidden,
        "arch":            args.arch,
        "predict":         args.predict,
        "ema":             args.ema,
        "diffusion_steps": args.diffusion_steps,
        "stratified":      args.stratified_timesteps,
        "cfg_weight":      args.cfg_weight,
        "reward_eta":      args.reward_eta,
        "guidance_clip":   args.guidance_clip,
        "normalize_guidance": args.normalize_guidance,
        "endpoint_guidance":  args.endpoint_guidance,
        "seed":            args.seed,
        "sample_seed":     args.sample_seed,
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
    torch.manual_seed(args.seed)
    flow_model, flow_reward, flow_losses = train_flow(dataset, args.epochs, args.batch_size, args.hidden,
                                                       args.interpolant, args.arch)

    print("\nSampling (flow)...")
    flow_latents = {}
    for name, cfg in guidance_configs.items():
        z    = sample_flow(flow_model, flow_reward, n=args.samples, **cfg,
                           **anchor_kwargs, **guide_kwargs, **flow_path)
        seqs = decode_fn(z)
        scores = score_with_oracle(seqs, oracle)
        report(name, seqs, scores, z)
        flow_latents[name] = {"latent": z, "sequences": seqs, "oracle": scores}
    save_results(out_root, "flow", flow_latents, flow_model, flow_reward,
                 stats, model_info["hf_id"], length, dim, args.min_polar, flow_losses,
                 run_config)

    # ── Diffusion ─────────────────────────────────────────────────────────────
    print("\nTraining diffusion model...")
    torch.manual_seed(args.seed)
    diff_model, diff_reward, diff_losses, alpha_bars, betas, alphas, post_vars = \
        train_diffusion(dataset, args.epochs, args.batch_size, args.hidden, args.arch,
                        args.predict, args.diffusion_steps, args.stratified_timesteps,
                        args.ema)

    print("\nSampling (diffusion)...")
    diff_latents = {}
    for name, cfg in guidance_configs.items():
        z    = sample_diffusion(diff_model, diff_reward, alpha_bars, betas, alphas, post_vars,
                                n=args.samples, **cfg, **anchor_kwargs, **guide_kwargs)
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
