"""Forward-process noise schedules for the diffusion head, and step sampling.

Split out of paths.py, which was at its line limit and whose remaining job is
the flow-matching interpolants. Axis A of the innovation study lives here.
"""
import math

import torch

from .config import DEVICE

# How the cumulative signal abar_k is laid out across the K steps.
BETA_SCHEDULES = ("linear", "cosine", "sigmoid")

# Nichol & Dhariwal clip beta to avoid a singularity at the very end of the
# cosine schedule, where abar -> 0 and beta -> 1.
MAX_BETA = 0.999

# ══════════════════════════════════════════════════════════════════════════════
# Step sampling
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Beta schedules
# ══════════════════════════════════════════════════════════════════════════════

def _linear_alpha_bars(K):
    """The original DDPM schedule: beta linear on [1e-4, 0.02].

    Ho, Jain & Abbeel (NeurIPS 2020). Note the endpoints were tuned for K=1000
    and are NOT rescaled when K changes, which is the historical behaviour this
    reproduces exactly.
    """
    betas = torch.linspace(1e-4, 0.02, K, dtype=torch.float64)
    return (1.0 - betas).cumprod(0)


def _cosine_alpha_bars(K, offset=0.008):
    """abar_k = cos^2((k/K + s)/(1 + s) * pi/2), normalized to abar_0 = 1.

    Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models"
    (ICML 2021), eq. 17. Their argument against the linear schedule is that it
    "is near-destructive toward the end of the forward noising process": abar
    falls to roughly zero well before k=K, so the last several hundred steps
    operate on pure noise and contribute almost nothing. They measure that the
    first 20% of the reverse process can be skipped under linear beta with
    little FID cost, which is wasted sampling budget.

    That matters more here than in their image setting. The avGFP latents are
    standardized per channel, so the signal they carry is destroyed as soon as
    abar drops below the per-channel variance; a schedule that spends its tail
    on noise spends it on nothing. The small offset keeps beta_1 away from
    zero so the first step is not degenerate.
    """
    k = torch.arange(K + 1, dtype=torch.float64) / K
    f = torch.cos((k + offset) / (1 + offset) * math.pi / 2).pow(2)
    return (f / f[0])[1:]


def _sigmoid_alpha_bars(K, start=-3.0, end=3.0, tau=1.0):
    """Logistic interpolation of abar, the control for "is cosine's shape special?".

    Chen, "On the Importance of Noise Scheduling for Diffusion Models"
    (arXiv:2301.10972), which shows the schedule interacts with input scaling
    and resolution rather than being universally optimal. Included so the Axis
    A result can distinguish "cosine helps" from "anything less destructive
    than linear helps", which the two schedules alone cannot separate.
    """
    k = torch.arange(K + 1, dtype=torch.float64) / K
    v = torch.sigmoid(torch.tensor([start, end], dtype=torch.float64) / tau)
    f = (torch.sigmoid((start + k * (end - start)) / tau) - v[0]) / (v[1] - v[0])
    return (1.0 - f)[1:].clamp(1e-9, 1.0)


def make_ddpm_schedule(K=1000, kind="linear"):
    """Forward-process tensors for a K-step chain, index 0 being the clean end.

    Every schedule is defined through its abar and beta is recovered as
    1 - abar_k/abar_{k-1}, rather than the reverse. Defining beta directly
    makes the thing that matters for sample quality -- how much signal is left
    at each step -- an indirect consequence of a choice made in another unit.

    Returns (betas, alphas, alpha_bars, posterior_variances), each of length
    K+1 with a zero/one pad at index 0 so a step index can be used directly.
    """
    if kind == "linear":
        alpha_bars = _linear_alpha_bars(K)
    elif kind == "cosine":
        alpha_bars = _cosine_alpha_bars(K)
    elif kind == "sigmoid":
        alpha_bars = _sigmoid_alpha_bars(K)
    else:
        raise ValueError(f"unknown beta schedule '{kind}', expected {BETA_SCHEDULES}")

    alpha_bars = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars])
    # beta_k = 1 - alpha_k, with alpha_k the per-step signal retention.
    alphas = alpha_bars[1:] / alpha_bars[:-1]
    betas = torch.cat([torch.zeros(1, dtype=torch.float64),
                       (1.0 - alphas).clamp(0.0, MAX_BETA)])
    alphas = 1.0 - betas
    # Recompute abar from the clipped betas so the two stay mutually consistent.
    alpha_bars = alphas.cumprod(0)
    previous = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])
    # DDPM eq. 7: the true posterior q(z_{k-1} | z_k, z_0) variance.
    post_vars = betas * (1 - previous) / (1 - alpha_bars).clamp_min(1e-20)

    out = [t.to(device=DEVICE, dtype=torch.float32)
           for t in (betas, alphas, alpha_bars, post_vars)]
    return tuple(out)


def snr(alpha_bars):
    """Signal-to-noise ratio abar/(1-abar) at every step, for loss weighting."""
    return alpha_bars / (1 - alpha_bars).clamp_min(1e-8)


def add_arguments(parser):
    """Axis A flag for the diffusion forward process."""
    group = parser.add_argument_group("noise schedule (Axis A)")
    group.add_argument("--beta-schedule", default="linear",
                       choices=BETA_SCHEDULES,
                       help="How the forward process destroys signal "
                            "(default: linear, the original DDPM schedule and "
                            "the only one available before this flag). "
                            "Measured at K=1000: linear reaches abar=0.5 at "
                            "step 260 and spends 17%% of its steps below "
                            "abar=1e-3, i.e. on pure noise, while 'cosine' "
                            "reaches it at 497 and spends 2%%. Nichol & "
                            "Dhariwal (ICML 2021) call linear "
                            "'near-destructive toward the end'. 'sigmoid' is "
                            "the control that separates 'cosine helps' from "
                            "'anything less destructive than linear helps'.")
    return parser


def describe(K=1000, kind="linear"):
    """One line summarizing where a schedule puts its noise, for run logs."""
    _, _, alpha_bars, _ = make_ddpm_schedule(K, kind)
    # The step at which half the signal variance is gone is the single most
    # informative number: linear reaches it far earlier than cosine.
    half = int((alpha_bars < 0.5).nonzero()[0]) if (alpha_bars < 0.5).any() else K
    tail = float((alpha_bars < 1e-3).float().mean())
    return (f"{kind}: abar=0.5 at step {half}/{K}, "
            f"{100 * tail:.0f}% of steps below abar=1e-3")


if __name__ == "__main__":
    # Smoke-test all three beta schedules and step sampling.
    import torch
    for sched in BETA_SCHEDULES:
        betas, alphas, abars, post_vars = make_ddpm_schedule(1000, sched)
        assert abars[0] > 0.99 and abars[-1] < 0.01, f"{sched}: abar range wrong"
        assert (abars[:-1] >= abars[1:]).all(), f"{sched}: abars not monotone"
        print(f"  {sched:8s}: abar[0]={abars[0]:.4f}  abar[-1]={abars[-1]:.6f}  "
              f"abar=0.5 at step {int((abars < 0.5).nonzero()[0])}")
    # Stratified sampling covers the whole range.
    k = sample_timesteps(1000, 1000, torch.device("cpu"), stratified=True)
    assert k.min() >= 1 and k.max() <= 1000
    print(f"  stratified k: min={k.min()}  max={k.max()}  unique={k.unique().numel()}/1000")
    print("schedules.py OK")
