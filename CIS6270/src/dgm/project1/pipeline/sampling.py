"""Guided samplers for the flow path and the DDPM chain.

Both take the same guidance arguments and both report the network evaluations
they spent, so an arm defined once in arms.py runs unchanged through either
modality and is comparable at matched cost.
"""
import torch

from .config import DEVICE
from .guidance import (noise_guidance_scale, reward_gradient,
                       velocity_guidance_scale)
from .objective import weight_vector
from .paths import (PathSpec, endpoint_from_noise, endpoint_from_velocity,
                    interpolate, time_grid)
from .solvers import (NFE, churn, clip_endpoint, ddim_step, flow_step,
                      langevin_correct, step_indices)

# Default bound on the clean-sample estimate, in standardized latent units.
# Six standard deviations is far outside anything the data contains while
# still being loose enough never to bind on a well-behaved x0-prediction run;
# it exists to stop eps-prediction's high-noise inversion from diverging. See
# solvers.clip_endpoint.
X0_LIMIT = 6.0

# ══════════════════════════════════════════════════════════════════════════════
# Conditioning helpers
# ══════════════════════════════════════════════════════════════════════════════

def condition_tensors(model, n, c, conditioning=None):
    """The (conditional, null) pair the classifier-free branches are built from.

    Under continuous conditioning `c` is a requested brightness in standardized
    units and the null branch is signalled by a drop mask rather than by a
    reserved class index, so both spellings have to be returned together. This
    is what makes "generate me a variant at +2 standard deviations" expressible
    at all; with the binary channel the only request available was "bright".
    """
    conditioning = conditioning or getattr(
        getattr(model, "trunk", model), "conditioning", "binary")
    if conditioning == "none":
        return None, None, None, None
    drop = torch.ones(n, dtype=torch.bool, device=DEVICE)
    keep = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    if conditioning == "continuous":
        trunk = getattr(model, "trunk", model)
        n_cond = (len(trunk.condition)
                  if isinstance(trunk.condition, torch.nn.ModuleList)
                  else trunk.condition[0].in_features)
        # A scalar request names column 0, brightness. The remaining columns
        # are the Rosetta attributes, and asking for nothing in particular is
        # asking for their mean -- which standardization has already put at
        # zero, so an unspecified attribute is requested at the dataset
        # average rather than at an arbitrary value.
        value = torch.zeros(n, n_cond, device=DEVICE)
        requested = torch.as_tensor(c, dtype=torch.float32, device=DEVICE).flatten()
        value[:, :len(requested)] = requested[None, :len(requested)]
        return value, value, keep, drop
    index = torch.full((n,), int(c), dtype=torch.long, device=DEVICE)
    return index, index, keep, drop


# ══════════════════════════════════════════════════════════════════════════════
# Flow
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_flow(model, reward_model, n=8, length=None, dim=None, c=1, w=0.0,
                eta=0.0, lambdas=(1., 0.), steps=200, anchor=None,
                strength=1.0, clip=0.0, normalize=False, path=None,
                endpoint_guidance=False, seed=123, objective=None, source=None,
                solver="euler", scaling="score", corrector_steps=0,
                corrector_snr=0.1, conditioning=None, return_nfe=False):
    """Integrate the velocity field from t=0 to t=1.

    With `anchor` (a standardized reference latent) the trajectory starts from a
    partially noised anchor at t = 1 - strength instead of pure noise at t = 0.
    Every measured avGFP variant sits within 15 substitutions of the wild type,
    a vanishingly small region of a 237x320 latent space, so integrating from
    N(0, I) rarely lands on it. strength=1.0 reproduces the unanchored path.

    With `source` (an InformedSource from coupling.py) the trajectory starts from
    a property-derived point rather than from noise. The field was trained
    against that source, so sampling from N(0, I) instead would start it off its
    own training distribution.

    `scaling` selects how eta becomes a velocity perturbation; "score" is the
    derived rule and "legacy" the 4t(1-t) bump it replaces. See guidance.py.
    """
    lam = weight_vector(lambdas, objective, reward_model.n_props)
    path = path or PathSpec()
    length = length or model.trunk.length
    dim = dim or getattr(model.trunk, "dim", None) or model.trunk.project_out.out_features
    torch.manual_seed(seed)
    nfe = NFE()
    start = 0.0

    def draw():
        if source is not None:
            return source.draw(n, DEVICE)
        return torch.randn(n, length, dim, device=DEVICE)

    if anchor is None:
        z = draw()
    else:
        start = 1.0 - float(strength)
        z1 = anchor.to(DEVICE).expand(n, -1, -1)
        # Start on the path the model was trained against, not a linear guess.
        z, _ = interpolate(draw(), z1, torch.full((n,), start, device=DEVICE),
                           **path.kwargs)

    cond, _, keep, drop = condition_tensors(model, n, c, conditioning)

    def velocity(state, t):
        """The classifier-free-guided velocity at one state and time."""
        v = model(state, t, cond, drop)
        nfe.spend()
        if w:
            v = v + w * (model(state, t, cond, keep) - v)
            nfe.spend()
        return v

    grid, deltas = time_grid(steps, start, path.sample_schedule, DEVICE)
    for step in range(steps):
        t = grid[step].expand(n)
        dt = deltas[step]
        if eta:
            # Guidance is applied to the base velocity rather than inside the
            # multi-stage solver, so an arm's guidance strength means the same
            # thing whichever integrator it runs under.
            base = velocity(z, t)
            kappa = velocity_guidance_scale(eta, t, scaling, path.kwargs)
            endpoint = None
            if endpoint_guidance:
                # Re-predict the velocity inside the gradient tape so the
                # reward gradient flows back through the endpoint estimate too.
                def endpoint(state, _t=t):
                    return endpoint_from_velocity(state, _t,
                                                  model(state, _t, cond, drop),
                                                  **path.kwargs)
                nfe.spend()
            base = base + kappa * reward_gradient(reward_model, z, t, lam, clip,
                                                  normalize, endpoint, objective)
            z = z + dt * base
        else:
            z = flow_step(z, t, dt, velocity, solver)
        if corrector_steps:
            # The flow field is not a score, so the corrector is driven by the
            # reward gradient alone here; the diffusion sampler has the true
            # score and uses it. Reported separately for that reason.
            z = langevin_correct(
                z, t, lambda s, u: reward_gradient(reward_model, s, u, lam,
                                                   clip, normalize, None,
                                                   objective),
                corrector_steps, corrector_snr)
    return (z, int(nfe)) if return_nfe else z


# ══════════════════════════════════════════════════════════════════════════════
# Diffusion
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_diffusion(model, reward_model, alpha_bars, betas, alphas, post_vars,
                     n=8, length=None, dim=None, c=1, w=0.0, eta=0.0,
                     lambdas=(1., 0.), anchor=None, strength=1.0, clip=0.0,
                     normalize=False, endpoint_guidance=False, seed=123,
                     objective=None, solver="ddpm", steps=None,
                     stochasticity=0.0, spacing="linear", churn_amount=0.0,
                     corrector_steps=0, corrector_snr=0.1, scaling="score",
                     conditioning=None, x0_limit=X0_LIMIT, return_nfe=False):
    """Run the reverse chain from the noisiest level down to the clean one.

    `solver="ddpm"` is the full ancestral chain over every one of the K
    training levels and ignores `steps`, which is the historical behaviour and
    the reason no earlier result was NFE-matched: it always cost K=1000
    evaluations while the flow path was swept from 10 to 200.

    `solver="ddim"` or `"heun"` visits only `steps` of those levels, so the
    cost becomes a free parameter and the matched comparison the project
    requires becomes possible. See solvers.step_indices for the placement.

    With `anchor` the chain starts at the level round(strength * K) from the
    forward-noised anchor (SDEdit), rather than at K from pure noise.
    """
    lam = weight_vector(lambdas, objective, reward_model.n_props)
    K = len(betas) - 1
    length = length or model.trunk.length
    dim = dim or getattr(model.trunk, "dim", None) or model.trunk.project_out.out_features
    torch.manual_seed(seed)
    nfe = NFE()

    start = K if anchor is None else max(1, min(K, int(round(float(strength) * K))))
    if anchor is None:
        z = torch.randn(n, length, dim, device=DEVICE)
    else:
        a = alpha_bars[start]
        z1 = anchor.to(DEVICE).expand(n, -1, -1)
        z = a.sqrt() * z1 + (1 - a).sqrt() * torch.randn(n, length, dim,
                                                         device=DEVICE)

    cond, _, keep, drop = condition_tensors(model, n, c, conditioning)

    def noise_at(state, t, alpha_bar):
        """Guided noise estimate, in eps units whatever the parameterization."""
        eps = model.to_noise(model(state, t, cond, drop), state, t)
        nfe.spend()
        if w:
            conditional = model.to_noise(model(state, t, cond, keep), state, t)
            eps = eps + w * (conditional - eps)
            nfe.spend()
        if eta:
            endpoint = None
            if endpoint_guidance:
                def endpoint(inner, _t=t, _a=alpha_bar):
                    out = model(inner, _t, cond, drop)
                    # In x0 mode the network already returns the clean estimate.
                    return out if model.predict == "x0" else \
                        endpoint_from_noise(inner, model.to_noise(out, inner, _t), _a)
                nfe.spend()
            sigma = (1 - alpha_bar).sqrt()
            # Lecture 3.4: s = -eps/sigma, so raising the reward lowers eps.
            eps = eps - noise_guidance_scale(eta, sigma, scaling) * reward_gradient(
                reward_model, state, t, lam, clip, normalize, endpoint, objective)
        return eps

    if solver == "ddpm":
        for k in range(start, 0, -1):
            t = torch.full((n,), k / K, device=DEVICE)
            eps = noise_at(z, t, alpha_bars[k])
            sigma = (1 - alpha_bars[k]).sqrt()
            if x0_limit is not None:
                x0 = clip_endpoint(
                    (z - sigma * eps) / alpha_bars[k].sqrt().clamp_min(1e-8),
                    x0_limit)
                eps = (z - alpha_bars[k].sqrt() * x0) / sigma.clamp_min(1e-8)
            mean = (z - betas[k] * eps / sigma) / alphas[k].sqrt()
            z = mean + post_vars[k].sqrt() * torch.randn_like(z) if k > 1 else mean
        return (z, int(nfe)) if return_nfe else z

    indices = [i for i in step_indices(start, steps or 50, spacing)]
    for position, k in enumerate(indices[:-1]):
        k_next = indices[position + 1]
        alpha_bar = alpha_bars[k]
        if churn_amount:
            z, alpha_bar = churn(z, alpha_bar, churn_amount)
        t = torch.full((n,), k / K, device=DEVICE)
        eps = noise_at(z, t, alpha_bar)
        z_next, _ = ddim_step(z, eps, alpha_bar, alpha_bars[k_next],
                              stochasticity, x0_limit=x0_limit)
        if solver == "heun" and k_next > 0:
            # Second-order correction: average the noise estimate at both ends
            # of the step, as in EDM Alg. 1. Costs one extra evaluation.
            t_next = torch.full((n,), k_next / K, device=DEVICE)
            eps_next = noise_at(z_next, t_next, alpha_bars[k_next])
            z_next, _ = ddim_step(z, 0.5 * (eps + eps_next), alpha_bar,
                                  alpha_bars[k_next], stochasticity,
                                  x0_limit=x0_limit)
        z = z_next
        if corrector_steps and k_next > 0:
            t_next = torch.full((n,), k_next / K, device=DEVICE)
            sigma_next = (1 - alpha_bars[k_next]).sqrt().clamp_min(1e-8)

            def score(state, u, _s=sigma_next, _a=alpha_bars[k_next]):
                return -noise_at(state, u, _a) / _s

            z = langevin_correct(z, t_next, score, corrector_steps, corrector_snr)
    return (z, int(nfe)) if return_nfe else z
