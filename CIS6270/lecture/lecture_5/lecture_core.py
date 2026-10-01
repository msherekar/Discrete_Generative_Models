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

def train_step(model, optimizer, clean, loss_fn):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model, clean)
    loss.backward()
    optimizer.step()
    return loss.item()

def gat_loss(model, target, source=None):
    if source is None:
        source = torch.randint(K, target.shape)
    t = torch.rand(target.shape[0])
    use_target = torch.rand(target.shape) < t[:, None]
    z = torch.where(use_target, target, source)
    logits = model(z, t)
    return token_ce(logits, target).sum(1).mean()

def gat_rates(prob, z, t):
    rates = prob / (1. - t[:, None, None])
    return rates.scatter(-1, z[..., None], 0.)

def rate_step(z, rates, h):
    exit_rate = rates.sum(-1)
    assert torch.all(h * exit_rate <= 1. + 1e-6)
    prob = h * rates
    prob.scatter_(-1, z[..., None],
                  (1. - h * exit_rate)[..., None])
    return draw(prob.clamp_min(0.))

@torch.no_grad()
def gat_sample(model, batch, length, steps=20, source=None):
    model.eval()
    z = (torch.randint(K, (batch, length))
         if source is None else source.clone())
    h = 1. / steps
    for n in range(steps):
        t = torch.full((batch,), n / steps)
        prob = model(z, t).softmax(-1)
        rates = gat_rates(prob, z, t)
        z = rate_step(z, rates, h)
    return z

def dirichlet_loss(model, clean):
    a = 8. * torch.rand(clean.shape[0])
    target = F.one_hot(clean, K).float()
    concentration = 1. + a[:, None, None] * target
    z = torch.distributions.Dirichlet(concentration).sample()
    return token_ce(model(z, a), clean).sum(1).mean()

def dirichlet_velocity(z, posterior, a):
    r = z.detach().double().numpy().clip(1e-7, 1. - 1e-7)
    da = 1e-4
    derivative = (betainc(1+a+da, K-1, r)
                  - betainc(1+a-da, K-1, r)) / (2*da)
    density = r**a * (1-r)**(K-2) / beta(1+a, K-1)
    c = torch.as_tensor(-derivative / ((1-r)*density),
                        dtype=z.dtype)
    weight = posterior * c
    return weight - z * weight.sum(-1, keepdim=True)

def fisher_path(z0, target, t):
    y0, y1 = z0.sqrt(), F.one_hot(target, K).float()
    cosine = (y0 * y1).sum(-1, keepdim=True)
    w = cosine.clamp(-1+1e-6, 1-1e-6).acos()
    s = t[:, None, None]
    y = (((1-s)*w).sin()*y0 + (s*w).sin()*y1) / w.sin()
    velocity = w * (-((1-s)*w).cos()*y0
                        + (s*w).cos()*y1) / w.sin()
    return y, velocity

def fisher_loss(model, clean):
    base = torch.distributions.Dirichlet(torch.ones(K))
    z0 = base.sample(clean.shape)
    t = torch.rand(clean.shape[0])
    y, target = fisher_path(z0, clean, t)
    raw = model(y, t)
    tangent = raw - y * (raw * y).sum(-1, keepdim=True)
    return 4. * (tangent - target).square().sum(-1).mean()

def gumbel_path(target, t, beta_noise=1., tau_max=4., decay=4.):
    uniform = torch.rand(*target.shape, K).clamp(1e-6, 1-1e-6)
    gumbel = -(-uniform.log()).log()
    a = F.one_hot(target, K).float() + gumbel / beta_noise
    tau = tau_max * (-decay*t[:, None, None]).exp()
    z = (a / tau).softmax(-1)
    velocity = (decay/tau) * z * (a - (z*a).sum(-1, keepdim=True))
    return z, velocity

def gumbel_loss(model, clean):
    t = torch.rand(clean.shape[0])
    z, target = gumbel_path(clean, t)
    raw = model(z, t)
    tangent = raw - raw.mean(-1, keepdim=True)
    return (tangent - target).square().sum(-1).mean()

def mog_rates(base, changes, preference, importance,
              strength, angle):
    ranks = rankdata(changes, axis=0, method='average')
    ranks = ranks / len(changes)
    norm = np.linalg.norm(changes, axis=1)
    denom = np.maximum(norm*np.linalg.norm(preference), 1e-12)
    cosine = (changes @ preference) / denom
    direction = changes @ preference
    standardize = lambda x: (x-x.mean()) / max(x.std(), 1e-12)
    score = (standardize((ranks*importance).mean(1))
             + standardize(direction))
    keep = (norm > 0) & (cosine >= np.cos(angle))
    return base * np.exp(strength*score) * keep

def rectified_loss(model, source, target):
    t = torch.rand(source.shape[0])
    z = (1-t[:, None, None])*source + t[:, None, None]*target
    conditional_velocity = target - source
    raw = model(z, t)
    tangent = raw - raw.mean(-1, keepdim=True)
    return (tangent-conditional_velocity).square().sum(-1).mean()

@torch.no_grad()
def redi_pairs(teacher, batch, length, teacher_steps=100):
    source = torch.randint(K, (batch, length))
    target = gat_sample(teacher, batch, length,
                        teacher_steps, source=source)
    return source, target

def maxmin(scores, preference):
    return np.min(scores * preference, axis=-1)

def dna_neighbors(sequence):
    return [sequence[:i] + b + sequence[i+1:]
            for i, old in enumerate(sequence)
            for b in ALPHABET if b != old]

def proposal(sequence, score, eta):
    candidates = dna_neighbors(sequence)
    log_weight = .5 * eta * np.array([score(y)-score(sequence)
                                    for y in candidates])
    weight = np.exp(log_weight - log_weight.max())
    return candidates, weight / weight.sum()

def mh_refine(sequence, score, log_ref, eta, rng):
    candidates, forward = proposal(sequence, score, eta)
    index = rng.choice(len(candidates), p=forward)
    candidate = candidates[index]
    reverse_candidates, reverse = proposal(candidate, score, eta)
    q_reverse = reverse[reverse_candidates.index(sequence)]
    log_ratio = log_ref(candidate) - log_ref(sequence)
    log_ratio += eta * (score(candidate)-score(sequence))
    log_ratio += np.log(q_reverse) - np.log(forward[index])
    accept = np.log(rng.random()) < min(0., log_ratio)
    return candidate if accept else sequence
