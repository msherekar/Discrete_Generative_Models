"""Training loops for the flow-matching and diffusion heads.

Each loop trains its generative network and the reward head together on one
optimizer, so the reward head sees exactly the noise levels guidance will later
query it at.
"""
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import BATCH_SIZE, CONDITION_DROP, DEVICE, HIDDEN, LEARNING_RATE
from .nets import EMA, DiffusionModel, FlowModel, RewardModel
from .paths import interpolate, make_ddpm_schedule, sample_timesteps

# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def _loader(dataset, batch_size):
    """A DataLoader that does not leave the GPU waiting on one Python thread.

    Measured on an OSPool A40 (job 15833630): GPU utilization 0.57 with one core
    busy for the whole run, because a single thread gathered 128 random rows out
    of an 11.7 GiB tensor 324 times per epoch, synchronously with the step.

    Worker count comes from the CPUs actually available -- sched_getaffinity
    respects the cgroup a Condor slot imposes, where os.cpu_count() would report
    the whole machine and oversubscribe a two-core slot.

    This does not change what the loop sees: batch order and every random draw
    are identical to num_workers=0, which was verified bit-for-bit on a
    25-epoch run.

    persistent_workers is deliberately NOT set. It would avoid respawning
    workers each epoch, but it draws the worker base seed once instead of once
    per epoch, which shifts the global RNG stream and changes every subsequent
    randn in the training loop -- measured as a different loss curve. Respawning
    two workers 250 times costs seconds against hours of training; losing
    comparability with existing runs is not worth it.
    """
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:                              # not Linux
        available = os.cpu_count() or 1
    workers = max(0, min(4, available - 1))
    extra = {"num_workers": workers, "prefetch_factor": 2} if workers else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      pin_memory=torch.cuda.is_available(), **extra)


def train_flow(dataset, epochs, batch_size=BATCH_SIZE, hidden=HIDDEN,
               interpolant="linear", arch="mlp"):
    _, length, dim = dataset.tensors[0].shape
    n_props = dataset.tensors[2].shape[1]
    loader = _loader(dataset, batch_size)
    model  = FlowModel(length, dim, hidden, arch).to(DEVICE)
    reward = RewardModel(length, dim, hidden, arch, n_props).to(DEVICE)
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
    loader = _loader(dataset, batch_size)
    model  = DiffusionModel(length, dim, alpha_bars, hidden, arch, predict).to(DEVICE)
    reward = RewardModel(length, dim, hidden, arch,
                         dataset.tensors[2].shape[1]).to(DEVICE)
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
