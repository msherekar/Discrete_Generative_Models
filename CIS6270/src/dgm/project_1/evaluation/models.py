"""Models, re-training and sampling, used only by --retrain and --ablate.

Deliberately a standalone copy of the training recipe: evaluate.py must be
able to read a run produced by an older version of run_experiment.py without
inheriting whatever that version's networks have since become.
"""
import csv

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dgm.common.esm_models import get_model

from .loaders import composition_proxies
from .style import (BATCH_SIZE, CONDITION_DROP, DEVICE, HIDDEN, LECTURE3,
                    LEARNING_RATE)

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
#
# Re-exported from pipeline/, not copied. The NETWORK definitions above are
# deliberately frozen -- that is this module's whole purpose, reading a run
# produced by an older run_experiment.py without inheriting whatever its
# networks have since become -- but a sampler is not a network. It only calls
# model(z, t, c), so the current one drives either generation of network, and it
# is the only one that knows about the probability path and the coupling.
#
# The copy that used to live here took no `interpolant` argument at all, so the
# ablation integrated a straight linear path whatever the run was trained on,
# and silently described a model that did not exist. With the path factored into
# a geometry and a schedule there would now be three such arguments to keep in
# step, which is two more than a duplicate can survive.
from ..pipeline.guidance import reward_gradient                  # noqa: F401,E402
from ..pipeline.sampling import sample_diffusion, sample_flow    # noqa: F401,E402


def predict_reward(reward_model, z, stats):
    """Use reward model at t=1 to predict properties, then unstandardize."""
    z = z.to(DEVICE)
    t = torch.ones(len(z), device=DEVICE)
    with torch.no_grad():
        r_tilde = reward_model(z, t).cpu()
    r_mean = stats["r_mean"]
    r_std  = stats["r_std"]
    return r_tilde * r_std + r_mean
