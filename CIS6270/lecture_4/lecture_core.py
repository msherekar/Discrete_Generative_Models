"""Small teaching implementations; illustrative data, not paper reproductions."""

import math

import numpy as np

import torch

from torch import nn

import torch.nn.functional as F

from scipy.special import betainc, beta

from scipy.stats import rankdata

K, MASK = 4, 4

ALPHABET = 'ACGT'

def encode(strings):
    return torch.tensor([[ALPHABET.index(c) for c in s]
                         for s in strings])

def draw(prob):
    shape = prob.shape[:-1]
    sample = torch.multinomial(prob.reshape(-1, K), 1)
    return sample.reshape(shape)

class DNA(nn.Module):
    def __init__(self, width=32, max_len=64):
        super().__init__()
        self.token = nn.Embedding(K + 1, width)
        self.soft = nn.Linear(K, width)
        self.position = nn.Embedding(max_len, width)
        self.time = nn.Linear(1, width)
        layer = nn.TransformerEncoderLayer(
            width, 4, 2 * width, dropout=0., batch_first=True)
        self.context = nn.TransformerEncoder(layer, 1)
        self.output = nn.Linear(width, K)

    def forward(self, z, t=None):
        h = self.token(z) if z.ndim == 2 else self.soft(z)
        pos = torch.arange(z.shape[1], device=z.device)
        h = h + self.position(pos)[None]
        if t is not None:
            h = h + self.time(t[:, None])[:, None]
        return self.output(self.context(h))

def token_ce(logits, target):
    return F.cross_entropy(logits.transpose(1, 2),
                           target, reduction='none')

def mdlm_loss(model, clean):
    batch, length = clean.shape
    t = torch.rand(batch).clamp_min(1e-4)
    masked = torch.rand(batch, length) < t[:, None]
    noisy = clean.masked_fill(masked, MASK)
    logits = model(noisy)  # optimal predictor needs no t
    ce = token_ce(logits, clean)
    weighted = ce * masked / t[:, None]
    return weighted.sum(1).mean()

def train_step(model, optimizer, clean, loss_fn):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model, clean)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def mdlm_sample(model, batch, length, steps=20):
    model.eval()
    z = torch.full((batch, length), MASK)
    grid = torch.linspace(1., 0., steps + 1)
    for t, s in zip(grid[:-1], grid[1:]):
        prob = model(z).softmax(-1)
        candidate = draw(prob)
        reveal = torch.rand(z.shape) < (t - s) / t
        update = (z == MASK) & reveal
        z = torch.where(update, candidate, z)
    return z

def rate_step(z, rates, h):
    exit_rate = rates.sum(-1)
    assert torch.all(h * exit_rate <= 1. + 1e-6)
    prob = h * rates
    prob.scatter_(-1, z[..., None],
                  (1. - h * exit_rate)[..., None])
    return draw(prob.clamp_min(0.))

def udlm_rates(clean_prob, z, t):
    alpha = 1. - t[:, None, None]
    noisy_prob = alpha * clean_prob + (1. - alpha) / K
    current = noisy_prob.gather(-1, z[..., None])
    rates = noisy_prob / (K * alpha * current)
    return rates.scatter(-1, z[..., None], 0.)

def udlm_loss(model, clean):
    t = .02 + .96 * torch.rand(clean.shape[0])
    random = torch.randint(K, clean.shape)
    z = torch.where(torch.rand(clean.shape) < t[:, None],
                    random, clean)
    exact = F.one_hot(clean, K).float()
    pred = model(z, t).softmax(-1)
    a, b = udlm_rates(exact, z, t), udlm_rates(pred, z, t)
    term = a * (a.clamp_min(1e-12).log()
                - b.clamp_min(1e-12).log()) + b - a
    return .96 * term.sum((1, 2)).mean()

def block_loss(model, clean, start, size):
    prefix, block = clean[:, :start], clean[:, start:start+size]
    t = torch.rand(clean.shape[0]).clamp_min(1e-4)
    mask = torch.rand(block.shape) < t[:, None]
    ctx = torch.cat([prefix, block.masked_fill(mask, MASK)], 1)
    logits = model(ctx)[:, start:]
    loss = token_ce(logits, block) * mask / t[:, None]
    return loss.sum(1).mean()

def geometric_cfg(uncond, cond, strength):
    logits = (1. - strength) * uncond.clamp_min(1e-12).log()
    logits += strength * cond.clamp_min(1e-12).log()
    return logits.softmax(-1)

def guide_rates(base_rates, log_values, current_log_value,
                strength=1.):
    log_ratio = log_values - current_log_value[..., None]
    return base_rates * (strength * log_ratio).exp()

def pareto_filter(sequences, scores):
    # Maximize both objectives; keep equal-score alternatives.
    ge = (scores[:, None] >= scores[None, :]).all(-1)
    gt = (scores[:, None] > scores[None, :]).any(-1)
    dominated = (ge & gt).any(0)
    return sequences[~dominated], scores[~dominated]
