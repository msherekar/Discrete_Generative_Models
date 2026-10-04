"""Differentiating the objective with respect to the latent, and scaling it.

One gradient function, because every guided sampler needs exactly this: the
lambda-weighted reward gradient, optionally normalized per objective, clipped,
and evaluated on a predicted clean endpoint. Then the scaling rule that makes
one eta mean the same intervention in both samplers.
"""
import torch

from .paths import _geometry, _schedule, resolve_path

# How eta is converted into a perturbation of the sampler's own update.
GUIDANCE_SCALINGS = ("score", "legacy", "constant")

# Largest score-to-velocity factor the flow sampler will apply. The exact
# factor diverges as t -> 0, where a vanishing signal coefficient means a
# finite score change implies an unbounded velocity change; see
# score_to_velocity.
MAX_VELOCITY_SCALE = 10.0


def score_to_velocity(t, path_kwargs=None, cap=MAX_VELOCITY_SCALE):
    """Factor converting a score perturbation into a velocity perturbation.

    This is the headline innovation, and it exists because the two samplers
    were not applying the same intervention for the same eta. The flow sampler
    multiplied the reward gradient by `eta * 4t(1-t)` -- a bump function with
    no derivation behind it, chosen so the guidance switches off at both ends
    -- while the diffusion sampler used `eta * sigma_k`, which does follow from
    the lectures. Same nominal eta, different physical intervention, so no
    flow-versus-diffusion guidance comparison was meaningful and no eta
    transferred between modalities (hence the `eta / sqrt(D)` hack in
    transfer.py).

    The common currency is the score. Lecture 3.4 derives reward tilting as

        p^(R) ∝ p exp(eta R)   =>   s^(R) = s + eta grad R

    so the perturbation each sampler must express is a change to the score by
    `eta grad R`. What differs is only the Jacobian of its own update with
    respect to the score, and that is exactly computable.

    For the interpolant X_t = a(t) X0 + b(t) X1 with X0 ~ N(0, I), the marginal
    score and the conditional expectations are related by

        s_t = -E[X0 | z] / a        E[X1 | z] = (z + a^2 s_t) / b

    and the marginal velocity v_t = a' E[X0|z] + b' E[X1|z] becomes

        v_t = (b'/b) z + a^2 (b'/b - a'/a) s_t

    so a score change ds implies a velocity change

        dv = a(t)^2 (b'(t)/b(t) - a'(t)/a(t)) ds.

    For the straight segment on linear time this is (1-t)/t, which is nothing
    like 4t(1-t): it is largest at t -> 0 where the old rule was smallest. The
    old bump therefore applied the least guidance exactly where the score is
    most informative about which mode the trajectory will land in, which is a
    plausible reason the measured guidance arms were nearly indistinguishable
    (diffusion Hamming 8.2-8.4 across every cfg, eta, lambda and setpoint).

    `cap` bounds the factor near t=0, where it diverges because a -> 0.
    """
    kwargs = dict(path_kwargs or {})
    scale = kwargs.pop("scale", 1.0)
    geometry, schedule = resolve_path(**kwargs)
    shape = (-1,) + (1,) * 2
    s, _ = _schedule(t, schedule)
    a, b, da_ds, db_ds = _geometry(s, geometry, scale)
    # The ds/dt factors cancel in the ratio b'/b - a'/a, so the schedule drops
    # out here and only the geometry matters -- which is the same cancellation
    # endpoint_from_velocity relies on, and the reason this is well conditioned
    # under the quadratic and cosine schedules.
    ratio = db_ds / b.clamp_min(1e-6) - da_ds / a.clamp_min(1e-6)
    return (a.pow(2) * ratio).clamp(-cap, cap).view(shape)


def velocity_guidance_scale(eta, t, scaling="score", path_kwargs=None):
    """Coefficient on the reward gradient inside the flow sampler.

      score     the derived factor above, so eta is a score-space strength.
      legacy    eta * 4t(1-t), the underived bump, kept as the ablation that
                shows the derivation is what matters and not merely the change.
      constant  eta, the mechanism-destroyed control: same average magnitude,
                no time dependence at all.
    """
    if scaling == "legacy":
        return eta * 4 * t.view(-1, 1, 1) * (1 - t.view(-1, 1, 1))
    if scaling == "constant":
        return eta * torch.ones_like(t).view(-1, 1, 1)
    if scaling == "score":
        return eta * score_to_velocity(t, path_kwargs)
    raise ValueError(f"unknown scaling '{scaling}', expected {GUIDANCE_SCALINGS}")


def noise_guidance_scale(eta, sigma, scaling="score"):
    """Coefficient on the reward gradient inside the diffusion sampler.

    Subtracted from the noise estimate, since eps = -sigma * s means a score
    increase is a noise DECREASE. This already followed from Lecture 3.4's
    s = -eps/sigma, so "score" reproduces the existing diffusion behaviour
    exactly -- which is the point: the innovation brings flow into line with
    diffusion rather than changing both.
    """
    if scaling == "constant":
        return eta * torch.ones_like(sigma)
    return eta * sigma


def add_arguments(parser):
    """Axis D / Stage 2.4 flags: how eta becomes an intervention."""
    group = parser.add_argument_group("guidance scaling (Stage 2.4)")
    group.add_argument("--guidance-scaling", default="score",
                       choices=GUIDANCE_SCALINGS,
                       help="How --reward-eta is converted into a "
                            "perturbation of the sampler's own update "
                            "(default: score). 'score' makes eta a "
                            "score-space strength in both samplers, derived "
                            "from Lecture 3.4's reward tilting. 'legacy' is "
                            "the flow sampler's eta*4t(1-t) bump, which "
                            "follows from nothing and applies the LEAST "
                            "guidance where the score is most informative; "
                            "'constant' is the mechanism-destroyed control "
                            "that keeps the magnitude and drops the time "
                            "dependence.")
    group.add_argument("--max-velocity-scale", type=float,
                       default=MAX_VELOCITY_SCALE, metavar="M",
                       help=f"Bound on the score-to-velocity factor near t=0, "
                            f"where it diverges because the signal "
                            f"coefficient vanishes (default: "
                            f"{MAX_VELOCITY_SCALE:g}).")
    return parser


def reward_gradient(reward_model, z, t, lambdas, clip=0.0, normalize=False,
                    endpoint=None, objective=None):
    """Gradient of the lambda-weighted reward with respect to the latent.

    Lecture 3.4 lists two precautions this implements.

    `normalize` takes each objective's gradient to unit norm before the weighted
    sum, so lambda sets relative importance rather than being confounded with
    whatever scale each reward head happens to have learned. Measured on the
    avGFP reward model the two objectives differ by 1.6x in gradient norm
    (5.24 vs 3.37), so without this a lambda of (0.7, 0.3) is not the ratio it
    appears to be, and a workable eta does not transfer between runs.

    `clip` bounds the per-sample gradient norm. Guidance adds this term at every
    integration step, so an unbounded gradient compounds: a 1,000-step DDPM chain
    at eta=50 reached latent norms of 4.6e17 and destroyed 32 of 100 samples,
    while the 200-step flow path at the same eta stayed bounded.
    """
    with torch.enable_grad():
        state = z.detach().requires_grad_(True)
        if endpoint is None:
            rewards = reward_model(state, t)
        else:
            # Lecture 3.4: "Some properties have little meaning for an
            # intermediate state. A partially denoised protein latent, for
            # example, may not yet correspond to a valid sequence." So predict
            # the clean endpoint, score THAT, and differentiate back through the
            # endpoint estimator. Measured on the avGFP reward model, accuracy
            # goes from Spearman 0.26 at t=0.1 to 0.82 on a clean latent, so this
            # replaces a nearly uninformative signal early in the trajectory.
            x1 = endpoint(state)
            rewards = reward_model(x1, torch.ones_like(t))
        # Without an Objective this is the historical behaviour: climb the raw
        # predictions. With one, each column is already sign-corrected and any
        # setpoint transform applied, and the constraint is handled separately
        # so its severity rho stays off the lambda simplex.
        terms = rewards if objective is None else objective.terms(rewards)
        penalty = (objective.penalty(rewards)
                   if objective is not None and objective.constrained else None)

        def unit(g):
            norm = g.flatten(1).norm(dim=1).clamp_min(1e-12)
            return g / norm.view(-1, *([1] * (g.dim() - 1)))

        keep = penalty is not None
        if normalize:
            # One backward per objective so each can be scaled independently.
            # Differentiate the RAW property and reapply the objective's factor
            # afterwards: normalizing the composed term would divide that factor
            # out. See Objective.term_scale.
            total = torch.zeros_like(state)
            for j in range(terms.shape[1]):
                if float(lambdas[j]) == 0.0:
                    continue
                source = rewards[:, j] if objective is not None else terms[:, j]
                g = torch.autograd.grad(source.sum(), state, retain_graph=True)[0]
                g = unit(g)
                if objective is not None:
                    scale = objective.term_scale(rewards, j)
                    g = g * scale.view(-1, *([1] * (g.dim() - 1)))
                total = total + lambdas[j] * g
            grad = total
        else:
            grad = torch.autograd.grad((terms * lambdas).sum(1).sum(), state,
                                       retain_graph=keep)[0]
        if keep:
            # Descend the violation: subtract rho times its gradient.
            g = torch.autograd.grad(penalty.sum(), state)[0]
            grad = grad - objective.rho * (unit(g) if normalize else g)
    grad = grad.detach()
    if clip and clip > 0:
        norm = grad.flatten(1).norm(dim=1)
        scale = (clip / norm.clamp_min(1e-12)).clamp(max=1.0)
        grad = grad * scale.view(-1, *([1] * (grad.dim() - 1)))
    return grad
