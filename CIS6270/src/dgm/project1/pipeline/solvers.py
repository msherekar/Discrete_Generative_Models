"""Integrators for the flow ODE and the diffusion reverse process.

Axis E of the study, plus the Stage 2 prerequisite: DDIM is what lets the
diffusion chain be subsampled to the same number of network evaluations as the
flow Euler budget, so the two methods can be compared at matched cost.
"""
import torch

# Integrators for the flow ODE, in increasing order of cost per step.
FLOW_SOLVERS = ("euler", "midpoint", "heun")

# Reverse processes for the diffusion chain. "ddpm" is the full ancestral
# chain, which cannot be subsampled; the others can.
DIFFUSION_SOLVERS = ("ddpm", "ddim", "heun")

# Network evaluations each flow solver spends per step, before guidance and
# classifier-free multipliers.
STEP_COST = {"euler": 1, "midpoint": 2, "heun": 2}

# ══════════════════════════════════════════════════════════════════════════════
# Cost accounting
# ══════════════════════════════════════════════════════════════════════════════

class NFE:
    """Counts network forward passes, the study's unit of sampling cost.

    Lecture 2.1's framing is the justification: "When a neural network
    represents v_t, every velocity evaluation requires a neural-network
    evaluation." Every earlier result was reported against `--steps`, which is
    not comparable across methods -- the flow path swept 10 to 200 steps while
    the diffusion path always ran the full K=1000 ancestral chain, so the
    "matched protocol" the project requires did not exist.

    Counting is done by the samplers calling `spend`, rather than by wrapping
    the module in a hook, so that the count is explicit at each call site and a
    reader can see where classifier-free guidance doubles it and endpoint
    guidance triples it.
    """

    def __init__(self):
        self.count = 0

    def spend(self, n=1):
        self.count += int(n)
        return self.count

    def __int__(self):
        return self.count


def expected_nfe(steps, solver="euler", cfg=False, endpoint=False, guided=False):
    """What a sampler configuration will cost, for planning a matched grid.

    Classifier-free guidance evaluates the conditional and unconditional
    branches, so it doubles the per-step trunk cost. Endpoint guidance adds one
    more evaluation inside the gradient tape, and that one is strictly more
    expensive than a forward pass because it also backpropagates -- counted
    here as a single evaluation, which understates it, so NFE-matched
    comparisons involving endpoint guidance are conservative in its favour and
    should be reported with that noted.
    """
    per_step = STEP_COST.get(solver, 1)
    if cfg:
        per_step *= 2
    if guided and endpoint:
        per_step += 1
    return steps * per_step


# ══════════════════════════════════════════════════════════════════════════════
# Flow ODE
# ══════════════════════════════════════════════════════════════════════════════

def flow_step(z, t, dt, field, solver="euler"):
    """One step of the flow ODE, where `field` maps (state, time) to velocity.

    Lecture 2.1 states the limitation this addresses outright: "Euler's method
    assumes that the velocity stays approximately constant during one short
    step." On a learned field that curves -- and Lecture 2.2's crossing-path
    argument says it must, because training paths cross and a deterministic
    field cannot -- the assumption fails first at large step sizes, which is
    exactly the low-NFE regime the matched comparison is run in.

      euler     z + dt v(z, t). First order, one evaluation.
      midpoint  evaluates at the half step. Second order, two evaluations.
      heun      averages the slope at both ends of the step (the trapezoid
                rule, applied to the Euler prediction). Second order, two
                evaluations; this is EDM Algorithm 1's deterministic sampler.

    Second order means the local error falls as dt^2 rather than dt, so a
    20-step Heun run and a 40-step Euler run cost the same two-times-20
    evaluations while Heun's error is smaller -- the whole reason to spend the
    extra evaluation rather than halving the step.
    """
    v = field(z, t)
    if solver == "euler":
        return z + dt * v
    if solver == "midpoint":
        half = t + 0.5 * dt
        return z + dt * field(z + 0.5 * dt * v, half)
    if solver == "heun":
        z_euler = z + dt * v
        t_next = t + dt
        # At the final step t_next can land exactly on the endpoint, where the
        # field is evaluated at a time it never saw in training; the average
        # still only uses it as a correction, so this stays stable.
        return z + 0.5 * dt * (v + field(z_euler, t_next))
    raise ValueError(f"unknown solver '{solver}', expected {FLOW_SOLVERS}")


# Largest corrector displacement per step, as a fraction of the sample norm.
MAX_CORRECTOR_STEP = 0.1


def langevin_correct(z, t, score, steps=1, snr=0.1,
                     max_step=MAX_CORRECTOR_STEP):
    """Annealed Langevin corrector steps at a fixed noise level.

    Song et al., "Score-Based Generative Modeling through Stochastic
    Differential Equations" (ICLR 2021), Alg. 4: the predictor moves the sample
    to the next noise level, the corrector re-equilibrates it at that level via
    `z <- z + e grad log p + sqrt(2e) n`.

    Lecture 3.2 teaches the annealing argument directly -- "Learn scores at
    decreasing noise levels... Large noise smooths barriers between modes.
    Smaller noise levels recover finer structure" -- and it is a data-specific
    argument here rather than a generic one: avGFP brightness is strongly
    bimodal, with 39.3% of variants piled against the assay floor and the rest
    in a functional lobe. A between-mode barrier is precisely what a corrector
    is for, and an ODE-only sampler has no mechanism to cross one.

    The step size follows their signal-to-noise heuristic, `e = 2 (snr |n| /
    |g|)^2`, which is calibrated for g being a true score: its norm grows like
    sqrt(D)/sigma, so e lands near 2 snr^2 sigma^2 and the displacement stays
    proportional to sigma.

    `max_step` exists because that calibration fails when g is NOT a score.
    The displacement is e|g| = 2 snr^2 |n|^2 / |g|, which DIVERGES as |g| goes
    to zero -- and the flow sampler's corrector is driven by a reward gradient
    rather than a score, so a near-flat reward gives exactly that. Measured
    without the bound: latent norm 1.1e14 after five corrector steps. Capping
    the displacement at a fraction of the sample norm keeps the heuristic where
    it is valid and degrades to a small fixed step where it is not.
    """
    for _ in range(steps):
        g = score(z, t)
        noise = torch.randn_like(z)
        g_norm = g.flatten(1).norm(dim=1).mean().clamp_min(1e-12)
        n_norm = noise.flatten(1).norm(dim=1).mean()
        z_norm = z.flatten(1).norm(dim=1).mean().clamp_min(1e-12)
        epsilon = 2 * (snr * n_norm / g_norm) ** 2
        # BOTH terms have to be bounded, not just the drift. Capping only
        # e|g| still leaves the diffusion term sqrt(2e)|n|, which grows as
        # sqrt(e) and so also diverges as |g| -> 0. Measured with the drift
        # bound alone: latent norm 1.18e6 against an uncorrected 20.8.
        drift_cap = max_step * z_norm / g_norm
        noise_cap = 0.5 * (max_step * z_norm / n_norm.clamp_min(1e-12)) ** 2
        epsilon = epsilon.clamp(max=float(torch.minimum(drift_cap, noise_cap)))
        z = z + epsilon * g + (2 * epsilon).sqrt() * noise
    return z


# ══════════════════════════════════════════════════════════════════════════════
# Diffusion reverse process
# ══════════════════════════════════════════════════════════════════════════════

def clip_endpoint(x0, limit=None):
    """Bound the clean-sample estimate, which every reverse step is built on.

    Every DDPM and DDIM reference implementation clips x0 to the data range
    ([-1, 1] for images); this pipeline never did, and the omission is not
    cosmetic. Under eps-prediction the estimate is

        x0 = (z - sqrt(1-abar) eps) / sqrt(abar)

    so at the noisy end, where abar is near zero, the division amplifies any
    error in eps without bound. Measured on the point-mass recovery test in
    project1_eval/smoke_units.py: with x0-prediction the samples land at
    +3.011 against a true +3.0, and with eps-prediction and no clipping they
    land at -264 with sd 2579. The same model and the same solver -- only the
    parameterization differs, and only eps needs the guard.

    This is a second, independent reason to prefer x0 as the default (the
    first being Lecture 3.2's AMP-Diffusion recipe): it is the
    parameterization whose output IS the quantity the sampler needs, so no
    ill-conditioned inversion happens at all.

    The limit is in standardized units, since load_data standardizes every
    latent channel to unit variance; None disables it.
    """
    if limit is None:
        return x0
    return x0.clamp(-float(limit), float(limit))


def ddim_step(z, eps, alpha_bar, alpha_bar_next, stochasticity=0.0,
              generator=None, x0_limit=None):
    """One DDIM step from noise level `alpha_bar` to `alpha_bar_next`.

    Song, Meng & Ermon, "Denoising Diffusion Implicit Models" (ICLR 2021),
    eq. 12. Two things make this the Stage 2 prerequisite:

    The update depends only on the two noise levels it moves between, not on
    their adjacency in the training chain, so a model trained with K=1000 can
    be sampled on any subsequence of those levels. That is what makes a
    20-evaluation diffusion run possible at all, and therefore what makes the
    flow-versus-diffusion comparison matchable on cost.

    At `stochasticity=0` it is a deterministic ODE, which removes sampling
    noise as a confound between arms and makes a seed reproduce a trajectory
    rather than a distribution. `stochasticity=1` recovers the ancestral DDPM
    variance, so the parameter interpolates between the two samplers the
    project compares rather than being a third thing.
    """
    x0 = clip_endpoint((z - (1 - alpha_bar).sqrt() * eps)
                       / alpha_bar.sqrt().clamp_min(1e-8), x0_limit)
    if x0_limit is not None:
        # Keep eps consistent with the clipped endpoint, or the two terms of
        # the update below disagree about where the sample is going.
        eps = ((z - alpha_bar.sqrt() * x0)
               / (1 - alpha_bar).sqrt().clamp_min(1e-8))
    # DDIM's sigma_t: the share of the step's variance reinjected as noise.
    sigma = stochasticity * (
        ((1 - alpha_bar_next) / (1 - alpha_bar).clamp_min(1e-8))
        * (1 - alpha_bar / alpha_bar_next.clamp_min(1e-8))
    ).clamp_min(0.0).sqrt()
    direction = (1 - alpha_bar_next - sigma.pow(2)).clamp_min(0.0).sqrt()
    z_next = alpha_bar_next.sqrt() * x0 + direction * eps
    if stochasticity > 0:
        noise = torch.randn(z.shape, device=z.device, dtype=z.dtype,
                            generator=generator)
        z_next = z_next + sigma * noise
    return z_next, x0


def churn(z, alpha_bar, amount=0.0, noise_scale=1.003, generator=None):
    """Re-noise the sample slightly, then report the level it now sits at.

    Karras et al., EDM (NeurIPS 2022), Alg. 2 and sec. 4: a deterministic
    sampler's errors accumulate monotonically because nothing in the update
    can correct an earlier mistake, while injecting and removing a little noise
    at each step drives the sample back toward the correct marginal. They find
    `S_noise` slightly above 1 compensates for the denoiser's tendency to
    remove marginally too much variance.

    Returns the churned sample and its new alpha_bar, so the caller's next step
    integrates from where the sample actually is.
    """
    if amount <= 0:
        return z, alpha_bar
    # Move to a noisier level by `amount`, bounded so abar stays positive.
    alpha_churned = (alpha_bar * (1 - amount)).clamp(1e-6, 1.0)
    added = (1 - alpha_churned / alpha_bar).clamp_min(0.0)
    noise = torch.randn(z.shape, device=z.device, dtype=z.dtype,
                        generator=generator)
    z = ((alpha_churned / alpha_bar).sqrt() * z
         + noise_scale * added.sqrt() * noise)
    return z, alpha_churned


def add_arguments(parser):
    """Axis E flags: which integrator, how its steps are placed, and churn."""
    group = parser.add_argument_group("sampling (Axis E)")
    group.add_argument("--flow-solver", default="euler", choices=FLOW_SOLVERS,
                       help="Integrator for the flow ODE (default: euler, the "
                            "only one available before this flag). 'heun' and "
                            "'midpoint' are second order and cost two "
                            "evaluations per step, so compare them at matched "
                            "NFE -- 20 Heun steps against 40 Euler steps, not "
                            "against 20. Lecture 2.1: 'Euler's method assumes "
                            "that the velocity stays approximately constant "
                            "during one short step.'")
    group.add_argument("--diffusion-solver", default="ddpm",
                       choices=DIFFUSION_SOLVERS,
                       help="Reverse process for diffusion (default: ddpm, the "
                            "full ancestral chain over all K levels, which is "
                            "what made every earlier comparison unmatched: it "
                            "always cost K=1000 evaluations while flow swept "
                            "10-200). 'ddim' visits only --sample-steps of "
                            "those levels and is the prerequisite for the "
                            "matched protocol.")
    group.add_argument("--sample-steps", type=int, default=None, metavar="N",
                       help="Levels the subsampled diffusion chain visits "
                            "(default: follow --steps, so one flag sets the "
                            "budget for both methods). Ignored by 'ddpm'.")
    group.add_argument("--step-spacing", default="linear",
                       choices=("linear", "quadratic"),
                       help="Where the subsampled levels land (default: "
                            "linear). 'quadratic' concentrates them at the "
                            "low-noise end, where the x0 estimate moves "
                            "fastest and DDIM's error is largest.")
    group.add_argument("--ddim-stochasticity", type=float, default=0.0,
                       metavar="S",
                       help="How much of each step's variance to reinject "
                            "(default: 0 = deterministic ODE). 1.0 recovers "
                            "the ancestral DDPM variance, so this interpolates "
                            "between the two samplers rather than adding a "
                            "third. Deterministic removes sampling noise as a "
                            "confound between arms.")
    group.add_argument("--churn", type=float, default=0.0, metavar="A",
                       help="EDM stochastic churn: re-noise by this fraction "
                            "before each step (default: 0 = off). Karras et "
                            "al. (NeurIPS 2022) sec. 4 -- a deterministic "
                            "sampler cannot correct an earlier error, while "
                            "injecting and removing a little noise drives the "
                            "sample back toward the correct marginal.")
    group.add_argument("--corrector-steps", type=int, default=0, metavar="N",
                       help="Annealed Langevin corrector steps per predictor "
                            "step (default: 0). Lecture 3.2 teaches the "
                            "annealing argument, and it is data-specific here: "
                            "avGFP brightness is bimodal, with 39.3%% of "
                            "variants against the assay floor, and a "
                            "between-mode barrier is what a corrector crosses.")
    group.add_argument("--corrector-snr", type=float, default=0.1, metavar="R",
                       help="Signal-to-noise target for the corrector step "
                            "size (default: 0.1, Song et al.'s value).")
    group.add_argument("--x0-limit", type=float, default=None, metavar="X",
                       help="Clip the clean-sample estimate to +/-X "
                            "standardized units (default: 6 inside the "
                            "samplers; pass 0 to disable). Every DDPM "
                            "reference clips x0 and this pipeline did not: "
                            "under --predict eps the estimate divides by "
                            "sqrt(abar), so at high noise it diverges. "
                            "Measured on the point-mass recovery test, "
                            "eps-prediction landed at -264 (sd 2579) "
                            "unclipped against a true +3.0.")
    return parser


def step_indices(K, steps, spacing="linear"):
    """Which of the K training levels a subsampled chain visits, descending.

    `spacing` is the Axis E step-placement knob. "quadratic" concentrates
    steps at the low-noise end, which is where DDIM's error is largest because
    the x0 estimate changes fastest there; Nichol & Dhariwal use the same
    reasoning for their respacing.
    """
    if spacing == "quadratic":
        fractions = torch.linspace(0, 1, steps + 1) ** 2
    else:
        fractions = torch.linspace(0, 1, steps + 1)
    # Descending from K to 0, deduplicated so a short chain cannot stall.
    indices = (K * (1 - fractions)).round().long().clamp(0, K)
    unique = []
    for value in indices.tolist():
        if not unique or value < unique[-1]:
            unique.append(value)
    if unique[-1] != 0:
        unique.append(0)
    return unique


if __name__ == "__main__":
    import torch
    B, L, D = 4, 8, 16
    z = torch.randn(B, L, D)

    # NFE counter.
    nfe = NFE()
    nfe.spend(2); nfe.spend(3)
    assert int(nfe) == 5
    print(f"  NFE counter: {int(nfe)}")

    # expected_nfe: CFG doubles cost per step.
    assert expected_nfe(10, "euler") == 10
    assert expected_nfe(10, "euler", cfg=True) == 20
    assert expected_nfe(10, "heun") == 20
    print(f"  expected_nfe: euler=10  euler+cfg=20  heun=20")

    # flow_step: Euler and Heun return correct shapes.
    def dummy_field(z, t, c=None, drop=None):
        return torch.zeros_like(z)
    t = torch.full((B,), 0.5)
    for solver in FLOW_SOLVERS:
        z_next = flow_step(z, t, 0.05, dummy_field, solver=solver)
        assert z_next.shape == z.shape and z_next.isfinite().all(), f"{solver} shape/NaN"
        print(f"  flow_step({solver}): shape={z_next.shape}")

    # ddim_step: DDPM step.
    from .schedules import make_ddpm_schedule
    _, _, abars, _ = make_ddpm_schedule(100)
    abars = abars.cpu()
    eps = torch.randn_like(z)
    z_prev, x0_est = ddim_step(z, eps, abars[50], abars[49])
    assert z_prev.shape == z.shape and z_prev.isfinite().all()
    print(f"  ddim_step: z_prev shape={z_prev.shape}  x0_est shape={x0_est.shape}")

    # step_indices: monotone decreasing from K to 0.
    idx = step_indices(100, 10)
    assert idx[0] == 100 and idx[-1] == 0 and len(idx) == 11
    print(f"  step_indices(100, 10): {idx}")

    print("solvers.py OK")
