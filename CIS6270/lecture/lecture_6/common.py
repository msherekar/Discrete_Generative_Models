"""Shared networks, data, and optimization for CIS 6270 Lecture 6.

The small MLPs keep time derivatives and all training steps visible. Sequence
networks receive the whole padded sequence, so predictions can depend on other
positions. No pretrained checkpoint or dataset download is required.
"""
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_all(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def mlp(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU(),
                         nn.Linear(width, outputs))


def time_like(t, x):
    """One scalar time per batch member, shape [B,1]."""
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    if t.numel() == 1:
        return t.expand(len(x), 1)
    return t.reshape(len(x), 1)


class MapNet(nn.Module):
    def __init__(self, dim=2, width=64, context_dim=0):
        super().__init__()
        self.net = mlp(dim + 2 + context_dim, dim, width)

    def forward(self, x, s, t, context=None):
        inputs = [x, time_like(s, x), time_like(t, x)]
        if context is not None:
            inputs.append(context)
        return self.net(torch.cat(inputs, -1))


class SequenceNet(nn.Module):
    def __init__(self, length, vocab, width=64):
        super().__init__()
        self.length, self.vocab = length, vocab
        self.net = mlp(length * vocab + 2, length * vocab, width)

    def forward(self, x, s, t):
        inputs = torch.cat([x.flatten(1), time_like(s, x), time_like(t, x)], -1)
        return self.net(inputs).reshape(-1, self.length, self.vocab)


def finite_map(model, x, s, t, context=None):
    """F(s,t,x) = x + (t-s) v(s,t,x), including exact F(s,s,x)=x."""
    s, t = time_like(s, x), time_like(t, x)
    return x + (t - s) * model(x, s, t, context)


def ordered_times(x, ceiling=.98):
    times = ceiling * torch.rand(len(x), 2, device=x.device, dtype=x.dtype)
    s, t = times.sort(-1).values.split(1, -1)
    return s, t, (s + t) / 2


def draw_batch(data, count):
    return data[torch.randint(len(data), (count,), device=data.device)]


def interpolate(data, time, noise=None):
    noise = torch.randn_like(data) if noise is None else noise
    t = time.reshape(len(data), *([1] * (data.ndim - 1)))
    return (1 - t) * noise + t * data, data - noise


def ema_copy(model):
    result = copy.deepcopy(model).eval()
    result.requires_grad_(False)
    return result


@torch.no_grad()
def update_ema(ema, model, decay=.99):
    for p, q in zip(ema.parameters(), model.parameters()):
        p.lerp_(q, 1 - decay)


def optimize(model, objective, steps, lr=1e-3, ema=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logs = []
    for step in range(steps):
        loss, details = objective(step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}: {details}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        if not torch.isfinite(norm):
            raise FloatingPointError(f'Nonfinite gradient at step {step}')
        optimizer.step()
        if ema is not None:
            update_ema(ema, model)
        logs.append({'step': step, 'loss': float(loss.detach()),
                     **{k: float(torch.as_tensor(v).detach()) for k, v in details.items()}})
    return logs


CENTERS = torch.tensor([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]])
DATA_STD = .22


def mixture_data(n, generator=None):
    ids = torch.randint(4, (n,), generator=generator)
    return CENTERS[ids] + DATA_STD * torch.randn(n, 2, generator=generator)


def mixture_posterior(x, t):
    """Exact p(X1 | (1-t)X0+tX1=x) for the four Gaussian mixture.

    Return component probabilities, conditional means, scalar variances.
    This analytic oracle is used for diagnostics and the GLASS example.
    """
    t = time_like(t, x)
    a = 1 - t
    variance = a.square() + (t * DATA_STD).square()
    centers = CENTERS.to(x)
    delta = x[:, None, :] - t[:, None, :] * centers
    logits = -delta.square().sum(-1) / (2 * variance)
    prob = logits.softmax(-1)
    gain = t * DATA_STD**2 / variance
    means = centers + gain[:, None, :] * delta
    posterior_variance = DATA_STD**2 * a.square() / variance
    return prob, means, posterior_variance


def exact_denoiser(x, t):
    p, means, _ = mixture_posterior(x, t)
    return (p[..., None] * means).sum(1)


def exact_velocity(x, t):
    t = time_like(t, x)
    # Direct conditional velocity avoids cancellation at t=1.
    a = 1 - t
    variance = a.square() + (t * DATA_STD).square()
    centers = CENTERS.to(x)
    delta = x[:, None] - t[:, None] * centers
    weights = (-delta.square().sum(-1) / (2 * variance)).softmax(-1)
    component_velocity = centers + ((t * DATA_STD**2 - a) / variance)[:, None] * delta
    return (weights[..., None] * component_velocity).sum(1)


def mixture_metrics(samples):
    x = samples.detach().cpu()
    distances = (x[:, None] - CENTERS).square().sum(-1)
    counts = torch.bincount(distances.argmin(-1), minlength=4).float()
    p = counts / len(x)
    logp = torch.logsumexp(-distances / (2 * DATA_STD**2), -1)
    logp -= math.log(4 * 2 * math.pi * DATA_STD**2)
    return {'finite_samples': bool(torch.isfinite(x).all()),
            'mean_distance_to_center': float(distances.min(-1).values.sqrt().mean()),
            'mode_fractions': p.tolist(), 'mode_entropy': float(-(p * p.clamp_min(1e-12).log()).sum()),
            'mean_target_log_density': float(logp.mean())}


def load_text(path, variable=False):
    lines = [x.strip().split() for x in Path(path).read_text().splitlines() if x.strip()]
    if len(lines) < 2:
        raise ValueError('Text data need at least two nonempty lines.')
    vocab = sorted(set(word for line in lines for word in line))
    lengths = torch.tensor([len(x) for x in lines])
    if not variable and len(set(lengths.tolist())) != 1:
        raise ValueError('Fixed-length methods require equal tokens per line; use expanding otherwise.')
    if lengths.max() > 32:
        raise ValueError('Teaching MLP supports up to 32 positions.')
    ids = {word: i for i, word in enumerate(vocab)}
    encoded = torch.zeros(len(lines), int(lengths.max()), dtype=torch.long)
    for i, line in enumerate(lines):
        encoded[i, :len(line)] = torch.tensor([ids[word] for word in line])
    return encoded, lengths, vocab


def text_metrics(ids, vocab, lengths=None, reference=None):
    lengths = [ids.shape[1]] * len(ids) if lengths is None else lengths.tolist()
    texts = [' '.join(vocab[j] for j in row[:length]) for row, length in zip(ids.tolist(), lengths)]
    flat = [j for row, length in zip(ids.tolist(), lengths) for j in row[:length]]
    counts = torch.bincount(torch.tensor(flat, dtype=torch.long), minlength=len(vocab)).float()
    p = counts / counts.sum().clamp_min(1)
    metrics = {'unique_fraction': len(set(texts)) / len(texts),
               'token_entropy': float(-(p * p.clamp_min(1e-12).log()).sum()),
               'mean_length': sum(lengths) / len(lengths), 'empty_fraction': lengths.count(0) / len(lengths)}
    if reference is not None:
        metrics['training_support_fraction'] = sum(x in reference for x in texts) / len(texts)
    return texts, metrics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
