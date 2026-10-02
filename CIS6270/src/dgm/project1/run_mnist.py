#!/usr/bin/env python3
"""Second modality: the same guidance study on MNIST images.

PROJECT.md asks for an innovation shown on one continuous modality and then
transferred to a second. The protein pipeline in run_experiment.py cannot simply
be pointed at images -- it encodes with ESM-2, decodes through a language-model
head, and scores amino-acid composition. What DOES transfer is every piece the
study is actually about, and this script imports those directly rather than
reimplementing them:

    interpolate, endpoint_from_velocity, endpoint_from_noise   probability path
    reward_gradient (clip / normalize / endpoint)              guidance rule
    EMA, sample_timesteps                                      training
    make_ddpm_schedule                                         diffusion

Only the modality-specific parts are new here: a U-Net instead of an MLP or
Transformer, MNIST tensors instead of ESM latents, and image properties instead
of residue counts. Because the shared code is literally the same objects, a
difference between modalities cannot be an implementation difference.

Rewards, both exact functions of the image so there is no oracle error:
  r1  mean pixel intensity -- how much ink the digit uses
  r2  left-right mirror symmetry -- correlation between an image and its flip

Usage:
  dgm-run-mnist --epochs 20 --samples 100
  dgm-run-mnist --epochs 20 --reward-eta 20 --endpoint-guidance
  dgm-run-mnist --epochs 20 --predict x0 --ema 0.99
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dgm.common.paths import SHARED_DATA, project_dir

ROOT = project_dir()
# Every guidance and path mechanism is imported, never re-written.
from .run_experiment import (          # noqa: E402
    EMA, INTERPOLANTS, endpoint_from_noise, endpoint_from_velocity,
    interpolate, make_ddpm_schedule, reward_gradient, sample_timesteps,
)
from .pipeline.paths import (PathSpec, time_grid,   # noqa: E402
                             add_arguments as add_path_arguments)
from .pipeline.images import image_properties, load_mnist   # noqa: E402
from .pipeline.coupling import (add_arguments as add_coupling_arguments,  # noqa: E402
                                columns_from_args, pair, source_for)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONDITION_DROP = 0.2


# ══════════════════════════════════════════════════════════════════════════════
# Networks
# ══════════════════════════════════════════════════════════════════════════════

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.SiLU(),
    )


class UNetField(nn.Module):
    """Lecture 2's MNIST U-Net, extended with a class-conditioning channel.

    `predict` plays the same role as in the protein DiffusionModel: "eps" adds
    the standard reparameterization, "x0" returns the clean-image estimate
    directly (Lecture 3.3's AMP-Diffusion choice).
    """

    def __init__(self, base_channels=32, alpha_bars=None, predict="eps"):
        super().__init__()
        c = base_channels
        self.alpha_bars = alpha_bars
        self.K = None if alpha_bars is None else len(alpha_bars) - 1
        self.predict = predict
        self.condition = nn.Embedding(3, 28 * 28)     # 0, 1, and the null class
        self.encoder1 = conv_block(3, c)              # image + time + condition
        self.encoder2 = conv_block(c, 2 * c)
        self.middle   = conv_block(2 * c, 4 * c)
        self.decoder2 = conv_block(6 * c, 2 * c)
        self.decoder1 = conv_block(3 * c, c)
        self.output   = nn.Conv2d(c, 1, 1)
        self.pool     = nn.MaxPool2d(2)

    def trunk(self, xt, t, c):
        time = t[:, None, None, None].expand_as(xt)
        cond = self.condition(c).view(-1, 1, 28, 28)
        x = torch.cat([xt, time, cond], dim=1)
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool(skip1))
        x = self.middle(self.pool(skip2))
        x = F.interpolate(x, size=skip2.shape[-2:], mode="nearest")
        x = self.decoder2(torch.cat([x, skip2], dim=1))
        x = F.interpolate(x, size=skip1.shape[-2:], mode="nearest")
        x = self.decoder1(torch.cat([x, skip1], dim=1))
        return self.output(x)

    def forward(self, xt, t, c):
        raw = self.trunk(xt, t, c)
        if self.alpha_bars is None or self.predict == "x0":
            return raw
        k = (t * self.K).round().long().clamp(0, self.K)
        a = self.alpha_bars[k, None, None, None]
        return (1 - a).sqrt() * xt + a.sqrt() * raw


class ImageReward(nn.Module):
    """Time-conditioned predictor of the two standardized image properties."""

    def __init__(self, base_channels=32):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            conv_block(2, c), nn.MaxPool2d(2),
            conv_block(c, 2 * c), nn.MaxPool2d(2),
            conv_block(2 * c, 2 * c), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2 * c, 2),
        )

    def forward(self, x, t):
        time = t[:, None, None, None].expand_as(x)
        return self.net(torch.cat([x, time], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_flow(dataset, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model  = UNetField(args.channels).to(DEVICE)
    reward = ImageReward(args.channels).to(DEVICE)
    opt = torch.optim.Adam(list(model.parameters()) + list(reward.parameters()),
                           lr=args.lr)
    ema = EMA(model, args.ema) if args.ema > 0 else None
    path = args.path
    source = source_for(args.coupling, dataset.tensors[0].to(DEVICE),
                        dataset.tensors[2].to(DEVICE))
    columns = columns_from_args(args)
    losses = []
    for epoch in range(args.epochs):
        total = 0.0
        for x1, c, r in loader:
            x1, c, r = x1.to(DEVICE), c.to(DEVICE), r.to(DEVICE)
            x0 = (source.paired(r) if source is not None else
                  pair(torch.randn_like(x1), x1, args.coupling,
                       args.coupling_beta, r, columns))
            t = torch.rand(len(x1), device=DEVICE)
            xt, target = interpolate(x0, x1, t, **path.kwargs)
            dropped = c.masked_fill(torch.rand(len(c), device=DEVICE) < CONDITION_DROP, 2)
            loss = F.mse_loss(model(xt, t, dropped), target) + F.mse_loss(reward(xt, t), r)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if ema is not None:
                ema.update(model)
            total += loss.item()
        losses.append(total / len(loader))
        if (epoch + 1) % max(1, args.epochs // 4) == 0:
            print(f"  [flow]      epoch {epoch+1:>4}/{args.epochs}: loss {losses[-1]:.4f}")
    if ema is not None:
        ema.copy_to(model)
    return (model.eval().requires_grad_(False),
            reward.eval().requires_grad_(False), losses, source)


def train_diffusion(dataset, args):
    betas, alphas, alpha_bars, post_vars = make_ddpm_schedule(args.diffusion_steps)
    K = len(betas) - 1
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model  = UNetField(args.channels, alpha_bars, args.predict).to(DEVICE)
    reward = ImageReward(args.channels).to(DEVICE)
    opt = torch.optim.Adam(list(model.parameters()) + list(reward.parameters()),
                           lr=args.lr)
    ema = EMA(model, args.ema) if args.ema > 0 else None
    losses = []
    for epoch in range(args.epochs):
        total = 0.0
        for x0, c, r in loader:
            x0, c, r = x0.to(DEVICE), c.to(DEVICE), r.to(DEVICE)
            k = sample_timesteps(len(x0), K, DEVICE, args.stratified_timesteps)
            t = k.float() / K
            a = alpha_bars[k, None, None, None]
            eps = torch.randn_like(x0)
            xk = a.sqrt() * x0 + (1 - a).sqrt() * eps
            dropped = c.masked_fill(torch.rand(len(c), device=DEVICE) < CONDITION_DROP, 2)
            target = x0 if args.predict == "x0" else eps
            loss = F.mse_loss(model(xk, t, dropped), target) + F.mse_loss(reward(xk, t), r)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if ema is not None:
                ema.update(model)
            total += loss.item()
        losses.append(total / len(loader))
        if (epoch + 1) % max(1, args.epochs // 4) == 0:
            print(f"  [diffusion] epoch {epoch+1:>4}/{args.epochs}: loss {losses[-1]:.4f}")
    if ema is not None:
        ema.copy_to(model)
    return (model.eval().requires_grad_(False), reward.eval().requires_grad_(False),
            losses, alpha_bars, betas, alphas, post_vars)


# ══════════════════════════════════════════════════════════════════════════════
# Sampling -- same structure as the protein samplers
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_flow(model, reward_model, args, n, c=1, w=0.0, eta=0.0, lambdas=(1., 0.),
                source=None):
    lam = torch.tensor(lambdas, dtype=torch.float32, device=DEVICE)
    lam = lam / lam.sum()
    torch.manual_seed(args.sample_seed)
    # An informed-source field was trained away from N(0, I).
    z = (source.draw(n, DEVICE) if source is not None
         else torch.randn(n, 1, 28, 28, device=DEVICE))
    null = torch.full((n,), 2, dtype=torch.long, device=DEVICE)
    cond = torch.full((n,), c, dtype=torch.long, device=DEVICE)
    path = args.path
    grid, deltas = time_grid(args.steps, 0.0, path.sample_schedule, DEVICE)
    for step in range(args.steps):
        t  = grid[step].expand(n)
        dt = deltas[step]
        v = model(z, t, null)
        if w:
            v = v + w * (model(z, t, cond) - model(z, t, null))
        if eta:
            kappa = eta * 4 * t[:, None, None, None] * (1 - t[:, None, None, None])
            endpoint = None
            if args.endpoint_guidance:
                def endpoint(state, _t=t, _null=null):
                    return endpoint_from_velocity(state, _t,
                                                  model(state, _t, _null),
                                                  **path.kwargs)
            v = v + kappa * reward_gradient(reward_model, z, t, lam,
                                            args.guidance_clip, args.normalize_guidance,
                                            endpoint)
        z = z + dt * v
    return z


@torch.no_grad()
def sample_diffusion(model, reward_model, sched, args, n, c=1, w=0.0, eta=0.0,
                     lambdas=(1., 0.)):
    alpha_bars, betas, alphas, post_vars = sched
    lam = torch.tensor(lambdas, dtype=torch.float32, device=DEVICE)
    lam = lam / lam.sum()
    K = len(betas) - 1
    torch.manual_seed(args.sample_seed)
    z = torch.randn(n, 1, 28, 28, device=DEVICE)
    null = torch.full((n,), 2, dtype=torch.long, device=DEVICE)
    cond = torch.full((n,), c, dtype=torch.long, device=DEVICE)
    for k in range(K, 0, -1):
        t = torch.full((n,), k / K, device=DEVICE)
        a_k = alpha_bars[k]

        def noise_pred(state, condition):
            out = model(state, t, condition)
            if model.predict != "x0":
                return out
            return (state - a_k.sqrt() * out) / (1 - a_k).sqrt().clamp_min(1e-4)

        eps = noise_pred(z, null)
        if w:
            eps = eps + w * (noise_pred(z, cond) - noise_pred(z, null))
        sigma = (1 - a_k).sqrt()
        if eta:
            endpoint = None
            if args.endpoint_guidance:
                def endpoint(state, _t=t, _null=null, _a=a_k):
                    out = model(state, _t, _null)
                    return out if model.predict == "x0" else \
                        endpoint_from_noise(state, out, _a)
            eps = eps - eta * sigma * reward_gradient(
                reward_model, z, t, lam, args.guidance_clip,
                args.normalize_guidance, endpoint)
        mean = (z - betas[k] * eps / sigma) / alphas[k].sqrt()
        z = mean + post_vars[k].sqrt() * torch.randn_like(z) if k > 1 else mean
    return z


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-tag", default="mnist")
    p.add_argument("--data-dir", type=Path, default=SHARED_DATA)
    p.add_argument("--limit", type=int, default=20000, help="Training images")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=100, help="Euler steps for flow")
    add_path_arguments(p)
    add_coupling_arguments(p)
    p.add_argument("--cfg-weight", type=float, default=0.0)
    p.add_argument("--reward-eta", type=float, default=0.0)
    p.add_argument("--endpoint-guidance", action="store_true")
    p.add_argument("--normalize-guidance", action="store_true")
    p.add_argument("--guidance-clip", type=float, default=0.0)
    p.add_argument("--predict", default="eps", choices=("eps", "x0"))
    p.add_argument("--ema", type=float, default=0.0)
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--stratified-timesteps", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sample-seed", type=int, default=123)
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out = args.outdir or ROOT / "outputs" / f"mnist_{args.dataset_tag}"
    out.mkdir(parents=True, exist_ok=True)
    print("\nProject 1 - MNIST (second modality)")
    print(f"  Tag         : {args.dataset_tag}")
    print(f"  Device      : {DEVICE}")
    print(f"  Guidance    : cfg w={args.cfg_weight}  reward eta={args.reward_eta}"
          + ("  endpoint" if args.endpoint_guidance else "")
          + ("  normalized" if args.normalize_guidance else ""))
    print(f"  Channels    : {args.channels}   flow steps: {args.steps}")
    print(f"  Coupling    : {args.coupling}"
          + (f" beta={args.coupling_beta}" if args.coupling == "aux" else ""))
    print(f"  Diffusion   : predict={args.predict}  K={args.diffusion_steps}"
          + (f"  ema={args.ema}" if args.ema else ""))
    print(f"  Seed        : {args.seed} (train)  {args.sample_seed} (sampling)")

    torch.manual_seed(args.seed)
    print(f"\nLoading MNIST (up to {args.limit} images)...")
    dataset, stats = load_mnist(args.limit, args.data_dir)
    print(f"  {len(dataset)} images  |  property means {stats['mean'].tolist()}")
    # Resolved once, here, because 'data-arc' measures its scale from the data.
    args.path = PathSpec.from_args(args, dataset.tensors[0])
    print(f"  Path        : {args.path.describe()}")

    modes = {"cfg":    dict(c=1, w=args.cfg_weight, eta=0.0,             lambdas=(1., 0.)),
             "single": dict(c=1, w=0.0,             eta=args.reward_eta, lambdas=(1., 0.)),
             "multi":  dict(c=1, w=0.0,             eta=args.reward_eta, lambdas=(0.7, 0.3))}

    results = {}
    print("\nTraining flow matching model...")
    torch.manual_seed(args.seed)
    fm, fr, flosses, fsource = train_flow(dataset, args)
    print("\nSampling (flow)...")
    flow_out = {}
    for name, cfg in modes.items():
        x = sample_flow(fm, fr, args, args.samples, **cfg, source=fsource)
        props = image_properties(x).cpu()
        flow_out[name] = {"images": x.cpu(), "properties": props}
        print(f"  {name:<7} ink {props[:,0].mean():+.4f}   symmetry {props[:,1].mean():+.4f}")
    results["flow"] = {"losses": flosses, **flow_out}

    print("\nTraining diffusion model...")
    torch.manual_seed(args.seed)
    dm, dr, dlosses, ab, be, al, pv = train_diffusion(dataset, args)
    print("\nSampling (diffusion)...")
    diff_out = {}
    for name, cfg in modes.items():
        x = sample_diffusion(dm, dr, (ab, be, al, pv), args, args.samples, **cfg)
        props = image_properties(x).cpu()
        diff_out[name] = {"images": x.cpu(), "properties": props}
        print(f"  {name:<7} ink {props[:,0].mean():+.4f}   symmetry {props[:,1].mean():+.4f}")
    results["diffusion"] = {"losses": dlosses, **diff_out}

    torch.save({"results": results, "config": vars(args) | {"data_dir": str(args.data_dir),
                                                            "outdir": str(out)}},
               out / "results.pt")
    print(f"\nSaved {out}/results.pt")

    print("\nProperty by method and guidance mode (higher ink = more stroke)")
    header = f"  {'method':<12}{'mode':<9}{'ink':>10}{'symmetry':>11}"
    print(header + "\n  " + "-" * (len(header) - 2))
    for method in ("flow", "diffusion"):
        for name in modes:
            p = results[method][name]["properties"]
            print(f"  {method:<12}{name:<9}{p[:,0].mean():>10.4f}{p[:,1].mean():>11.4f}")


if __name__ == "__main__":
    main()
