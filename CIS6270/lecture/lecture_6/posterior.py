"""Posterior Diamond Maps, data-trained Meta Flow Maps, and reward steering.

There are two independent noise draws and a shared clean endpoint. The inner
time evolves while the outer observation and outer time remain fixed.
"""
import math

import torch
from torch.nn import functional as F

from common import (MapNet, draw_batch, ema_copy, exact_denoiser, exact_velocity,
                    finite_map, interpolate, mixture_posterior, optimize,
                    ordered_times, time_like)
from continuous import semigroup_loss


def glass_denoiser(inner, s, outer, t):
    """Fuse independent Gaussian observations by adding their precisions.

    inner=s*X1+(1-s)*eps, outer=t*X1+(1-t)*eps'. The sufficient
    statistic S has noise variance 1/precision. Convert it to the original
    linear schedule t*=sqrt(precision)/(1+sqrt(precision)), then denoise.
    """
    precision = (s / (1-s)).square() + (t / (1-t)).square()
    safe = precision.clamp_min(1e-12)
    statistic = (s * inner / (1-s).square() + t * outer / (1-t).square()) / safe
    ratio = safe.sqrt()
    effective_time = ratio / (1 + ratio)
    value = exact_denoiser(effective_time * statistic, effective_time)
    return torch.where(precision > 1e-12, value, torch.zeros_like(value))


def glass_velocity(inner, s, outer, t):
    return (glass_denoiser(inner, s, outer, t) - inner) / (1-s)


def posterior_context(outer, t):
    return torch.cat([outer, time_like(t, outer)], -1)


def train_posterior(method, data, args):
    model = MapNet(2, args.width, context_dim=3).to(data)
    teacher = ema_copy(model)

    def objective(step):
        clean = draw_batch(data, args.batch_size)
        outer_t = .94 * torch.rand(len(clean), 1, device=data.device)
        outer, _ = interpolate(clean, outer_t)
        context = posterior_context(outer, outer_t)
        s, t, _ = ordered_times(clean, .97)
        # Independent inner noise; reusing the outer noise changes the posterior.
        inner, displacement = interpolate(clean, s)
        if method == 'diamond':
            diagonal_target = glass_velocity(inner, s, outer, outer_t).detach()
        else:
            diagonal_target = displacement
        diagonal = F.mse_loss(model(inner, s, s, context), diagonal_target)
        if method == 'diamond':
            y, derivative = torch.func.jvp(lambda end: finite_map(model, inner, s, end, context),
                                          (t,), (torch.ones_like(t),))
            target = glass_velocity(y.detach(), t, outer, outer_t).detach()
            finite = F.mse_loss(derivative, target)
        else:
            finite = semigroup_loss(model, teacher, inner, s, t, context)
        weight = min(1., (step+1) / max(1, args.train_steps//4))
        return diagonal + weight * finite, {'diagonal': diagonal, 'finite': finite}

    logs = optimize(model, objective, args.train_steps, args.lr, teacher)
    return model, {'model': model.state_dict(), 'dim': 2}, logs


def posterior_samples(model, outer, t, count, steps=1, noise=None):
    context = posterior_context(outer, t)
    context = context[:, None].expand(-1, count, -1).reshape(-1, 3)
    x = torch.randn(len(outer)*count, 2, device=outer.device) if noise is None else noise.reshape(-1, 2)
    for i in range(steps):
        x = finite_map(model, x, i/steps, (i+1)/steps, context)
    return x.reshape(len(outer), count, 2)


def reward(x, strength=1.):
    """Bounded differentiable preference for positive first coordinates."""
    return strength * torch.tanh(x[..., 0])


def posterior_value(model, x, t, particles=32, strength=1., steps=1):
    samples = posterior_samples(model, x, t, particles, steps)
    log_weights = reward(samples, strength)
    value = torch.logsumexp(log_weights, -1) - math.log(particles)
    weighted_mean = (log_weights.softmax(-1)[..., None] * samples).sum(1)
    return value, weighted_mean, samples


def fine_tune_surrogate(delta, weights, grad_weights, coefficient):
    """Equation 43 of Meta Flow Maps. Both Monte Carlo terms are detached.

    d/d(delta) E[loss] = 2 E[w*delta - coefficient*grad(w)].
    Differentiating (w*delta - coefficient*grad(w))^2 instead adds a wrong w.
    """
    residual = delta + (weights.detach()-1) * delta.detach() - coefficient * grad_weights.detach()
    return residual.square().mean()


def train_reward_drift(posterior, data, args):
    posterior.requires_grad_(False)
    drift = MapNet(2, args.width).to(data)
    def objective(_):
        clean = draw_batch(data, args.batch_size)
        t = .05 + .8 * torch.rand(len(clean), 1, device=data.device)
        x, _ = interpolate(clean, t)
        x.requires_grad_(True)
        endpoint = posterior_samples(posterior, x, t, 1, args.posterior_steps)[:, 0]
        w = reward(endpoint, args.reward_strength).exp().unsqueeze(-1)
        grad_w = torch.autograd.grad(w.sum(), x)[0]
        baseline = exact_velocity(x, t).detach()
        delta = drift(x.detach(), t, t) - baseline
        # Linear interpolant's compatible probability-flow correction g^2/2=(1-t)/t.
        # Use t>=.05 to avoid singular coefficients in this demonstration.
        coefficient = (1-t) / t
        loss = fine_tune_surrogate(delta, w, grad_w, coefficient)
        return loss, {'mean_weight': w.mean()}
    logs = optimize(drift, objective, args.finetune_steps, args.lr)
    return drift, logs


@torch.no_grad()
def guided_samples(model, count, steps, particles, strength, posterior_steps=1):
    """Derivative-free reward-weighted posterior mean in the linear flow drift.

    v*(x,t)=(E_reward[X1|x,t]-x)/(1-t). It equals the compatible
    probability-flow correction for an exact posterior. Finite learned maps
    and self-normalized Monte Carlo introduce approximation and ratio bias.
    """
    x = torch.randn(count, 2, device=next(model.parameters()).device)
    h = 1/steps
    for i in range(steps):
        t = torch.full((count, 1), i/steps, device=x.device)
        _, mean, _ = posterior_value(model, x, t, particles, strength, posterior_steps)
        x = x + h * (mean-x) / (1-t)
    return x


def weighted_diamond_samples(outer, t, particles=128):
    """Exact tractable demonstration of weighted Diamond proposal correction.

    Proposal q=N(0,4I) has evaluable density and full support. Reweight with
    p1(z)*p_t(outer|z)/q(z), then resample. This supplies an independent
    check of posterior recovery without assuming an arbitrary learned map's
    Jacobian density is known. Finite self-normalized importance sampling is biased.
    """
    from common import CENTERS, DATA_STD
    t = time_like(t, outer)
    z = 2 * torch.randn(len(outer), particles, 2, device=outer.device)
    delta = z[:, :, None] - CENTERS.to(z)
    lp = torch.logsumexp(-delta.square().sum(-1)/(2*DATA_STD**2), -1)
    lp -= math.log(4 * 2 * math.pi * DATA_STD**2)
    noise = (outer[:, None]-t[:, None]*z)/(1-t[:, None])
    likelihood = -.5*noise.square().sum(-1) - 2*(1-t).log()
    lq = -z.square().sum(-1)/8 - math.log(8*math.pi)
    weights = (lp + likelihood - lq).softmax(-1)
    mean = (weights[..., None]*z).sum(1)
    indices = torch.multinomial(weights, 1)
    selected = z.gather(1, indices[..., None].expand(-1, -1, 2))[:, 0]
    return selected, mean, 1/weights.square().sum(-1)


@torch.no_grad()
def posterior_diagnostics(model, steps=1):
    device = next(model.parameters()).device
    x = torch.tensor([[0., 0.], [1., -.5], [-1., 1.]], device=device)
    t = torch.tensor([[.15], [.5], [.8]], device=device)
    p, means, var = mixture_posterior(x, t)
    exact_mean = (p[..., None]*means).sum(1)
    exact_variance = (p[..., None]*(var[:, None]+means.square())).sum(1)-exact_mean.square()
    samples = posterior_samples(model, x, t, 512, steps)
    mean, variance = samples.mean(1), samples.var(1)
    return {'posterior_mean_rmse': float((mean-exact_mean).square().mean().sqrt()),
            'posterior_variance_rmse': float((variance-exact_variance).square().mean().sqrt()),
            'posterior_reference_mean': exact_mean.tolist(), 'posterior_estimated_mean': mean.tolist()}
