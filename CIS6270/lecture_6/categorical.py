"""Simplex-valued mean denoisers for Flow Map Language Models and discrete maps.

The state is Gaussian-noised one-hot text and can be outside the simplex.
Only the softmax endpoint prediction is a probability vector.
"""
import math

import torch
from torch.nn import functional as F

from common import SequenceNet, draw_batch, ema_copy, interpolate, optimize, time_like


def categorical_map(model, x, s, t):
    s, t = time_like(s, x), time_like(t, x)
    probability = model(x, s, t).softmax(-1)
    h = ((t - s) / (1 - s)).unsqueeze(-1)
    return x + h * (probability - x), probability


def composition_target(model, x, s, u, t):
    mid, p1 = categorical_map(model, x, s, u)
    _, p2 = categorical_map(model, mid, u, t)
    gamma = ((1 - t) * (u - s) / ((1 - u) * (t - s))).unsqueeze(-1)
    return gamma * p1 + (1 - gamma) * p2


def probability_kl(logits, target):
    return F.kl_div(logits.log_softmax(-1), target, reduction='none').sum(-1).mean()


class DecodingClock:
    """Inverse of the FMLM decoding-accuracy clock for isotropic Gaussian noise.

    P(correct) = E_Z[Phi(Z + t/(1-t))^(V-1)]. Gaussian quadrature gives a
    deterministic lookup. Endpoint values are exact. No empirical vocabulary
    frequency estimate is mixed into this corruption-only clock.
    """
    def __init__(self, vocab, device='cpu'):
        import numpy as np
        nodes, weights = np.polynomial.hermite.hermgauss(64)
        self.t = torch.linspace(0, .999, 1001, device=device)
        z = torch.as_tensor(nodes * math.sqrt(2), device=device, dtype=torch.float64)
        w = torch.as_tensor(weights / math.sqrt(math.pi), device=device, dtype=torch.float64)
        snr = (self.t / (1 - self.t)).double()
        cdf = .5 * (1 + torch.erf((z[None] + snr[:, None]) / math.sqrt(2)))
        accuracy = (cdf.pow(vocab - 1) * w).sum(-1)
        tau = ((vocab * accuracy - 1) / (vocab - 1)).clamp(0, 1).float()
        self.tau = torch.cummax(tau, 0).values
        self.tau[0], self.tau[-1], self.t[-1] = 0., 1., 1.

    def inverse(self, tau):
        indices = torch.searchsorted(self.tau, tau.contiguous()).clamp(1, len(self.tau)-1)
        low, high = self.tau[indices-1], self.tau[indices]
        alpha = (tau-low) / (high-low).clamp_min(1e-7)
        result = self.t[indices-1] + alpha * (self.t[indices]-self.t[indices-1])
        # Finite precision can saturate the lookup before t=1. Enforce the
        # mathematical endpoint so generation actually reaches the simplex.
        return torch.where(tau>=1,torch.ones_like(result),torch.where(tau<=0,torch.zeros_like(result),result))


def corrected_logit_teacher(model, teacher, x, s, t, kind, floor=.05):
    """Lagrangian/Eulerian logit teachers and explicit domain stabilization.

    At an exact solution the denominators are positive. Unconverged networks
    can violate that domain. We clamp to floor and report the changed fraction.
    These clamped off-solution targets are an explicit numerical approximation.
    """
    if kind == 'discrete-lsd':
        logits, dz = torch.func.jvp(lambda end: model(x, s, end),
                                   (t,), (torch.ones_like(t),))
        p = logits.softmax(-1)
        delta = dz - (p * dz).sum(-1, keepdim=True)
        coefficient = ((t-s) * (1-t) / (1-s)).unsqueeze(-1)
        denominator = 1 + coefficient * delta
        with torch.no_grad():
            y, _ = categorical_map(teacher, x, s, t)
            base_logits = teacher(y, t, t)
    else:
        with torch.no_grad():
            p0 = teacher(x, s, s).softmax(-1)
            velocity = (p0 - x) / (1-s).unsqueeze(-1)
            base_logits = teacher(x, s, s)
        logits, dz = torch.func.jvp(lambda z, start: model(z, start, t),
                                   (x, s), (velocity, torch.ones_like(s)))
        p = logits.softmax(-1)
        delta = dz - (p * dz).sum(-1, keepdim=True)
        coefficient = ((1-s)*(t-s)/(1-t)).unsqueeze(-1)
        denominator = 1 - coefficient * delta
    target = (base_logits - denominator.clamp_min(floor).log()).softmax(-1).detach()
    return logits, target, (denominator.detach() < floor).float().mean()


def train_categorical(method, ids, vocab, args):
    model = SequenceNet(ids.shape[1], len(vocab), args.width).to(ids.device)
    teacher = ema_copy(model)
    clock = DecodingClock(len(vocab), ids.device)

    def objective(step):
        batch = draw_batch(ids, args.batch_size)
        clean = F.one_hot(batch, len(vocab)).float()
        ordered = clock.inverse(torch.rand(len(batch), 2, device=ids.device)).sort(-1).values
        # Stay away from singular coefficients in differential teacher targets.
        s, t = (.97 * ordered).split(1, -1)
        t = torch.maximum(t, s + 1e-4)
        x, _ = interpolate(clean, s)
        diagonal = F.cross_entropy(model(x, s, s).flatten(0, 1), batch.flatten())
        logits = model(x, s, t)
        extra = {}
        if method == 'fmlm':
            with torch.no_grad():
                target = composition_target(teacher, x, s, (s+t)/2, t)
            finite = probability_kl(logits, target)
        elif method == 'categorical':
            y, _ = categorical_map(model, x, s, t)
            with torch.no_grad():
                endpoint = teacher(y.detach(), t, t).softmax(-1)
            ec = probability_kl(logits, endpoint)
            _, dt = torch.func.jvp(lambda end: model(x, s, end).softmax(-1),
                                   (t,), (torch.ones_like(t),))
            eta = ((t-s)/(1-s)).unsqueeze(-1)
            temporal = (eta * dt).square().flatten(1).sum(-1).mean()
            # Released ECLD implementation: CE + sum-of-squares time energy.
            # Detached-target KL has the same student gradient as its CE term.
            finite = ec + temporal
            extra = {'endpoint_kl': ec, 'temporal_derivative': temporal}
        else:
            logits, target, fraction = corrected_logit_teacher(model, teacher, x, s, t, method)
            finite = probability_kl(logits, target)
            extra = {'denominator_clamp_fraction': fraction}
        weight = min(1., (step + 1) / max(1, args.train_steps // 4))
        return diagonal + weight * finite, {'diagonal_ce': diagonal, 'finite': finite, **extra}

    logs = optimize(model, objective, args.train_steps, args.lr, teacher)
    return model, {'model': model.state_dict(), 'length': ids.shape[1], 'vocab': vocab}, logs


@torch.no_grad()
def sample_categorical(model, count, steps, device):
    clock = DecodingClock(model.vocab, device)
    grid = clock.inverse(torch.linspace(0, 1, steps+1, device=device))
    x = torch.randn(count, model.length, model.vocab, device=device)
    for s, t in zip(grid[:-1], grid[1:]):
        x, _ = categorical_map(model, x, s, t)
    return x.argmax(-1)
