"""Probability paths and the schedules that walk them.

The interpolants and the DDPM schedule live together because both answer the
same question -- where the state is between prior and data -- and the endpoint
estimators invert them.
"""
import torch

from .config import DEVICE

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
