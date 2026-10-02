"""Guided samplers for the flow path and the DDPM chain.

Both take the same guidance arguments, so an arm defined once in arms.py runs
unchanged through either modality.
"""
import torch

from .config import DEVICE
from .guidance import reward_gradient
from .objective import weight_vector
from .paths import (PathSpec, endpoint_from_noise,
                    endpoint_from_velocity, interpolate, time_grid)

@torch.no_grad()
def sample_flow(model, reward_model, n=8, c=1, w=0.0, eta=0.0, lambdas=(1., 0.), steps=200,
                anchor=None, strength=1.0, clip=0.0, normalize=False,
                path=None, endpoint_guidance=False, seed=123,
                objective=None, source=None):
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
    """
    lam = weight_vector(lambdas, objective)
    path = path or PathSpec()
    torch.manual_seed(seed)
    start = 0.0

    def draw():
        if source is not None:
            return source.draw(n, DEVICE)
        return torch.randn(n, model.length, model.dim, device=DEVICE)

    if anchor is None:
        z = draw()
    else:
        start = 1.0 - float(strength)
        z1    = anchor.to(DEVICE).expand(n, -1, -1)
        z0    = draw()
        # Start on the path the model was trained against, not a linear guess.
        z, _  = interpolate(z0, z1, torch.full((n,), start, device=DEVICE),
                            **path.kwargs)
    null = torch.full((n,), 2, dtype=torch.long, device=DEVICE)
    cond = torch.full((n,), c, dtype=torch.long, device=DEVICE)
    # Where the steps land is its own choice; see paths.time_grid.
    grid, deltas = time_grid(steps, start, path.sample_schedule, DEVICE)
    for step in range(steps):
        t  = grid[step].expand(n)
        dt = deltas[step]
        v = model(z, t, null)
        if w:
            v = v + w * (model(z, t, cond) - model(z, t, null))
        if eta:
            kappa = eta * 4 * t[:, None, None] * (1 - t[:, None, None])
            endpoint = None
            if endpoint_guidance:
                # Re-predict the velocity inside the gradient tape so the reward
                # gradient flows back through the endpoint estimate as well.
                def endpoint(state, _t=t, _null=null):
                    return endpoint_from_velocity(state, _t,
                                                  model(state, _t, _null),
                                                  **path.kwargs)
            v = v + kappa * reward_gradient(reward_model, z, t, lam, clip, normalize,
                                            endpoint, objective)
        z = z + dt * v
    return z


@torch.no_grad()
def sample_diffusion(model, reward_model, alpha_bars, betas, alphas, post_vars,
                     n=8, c=1, w=0.0, eta=0.0, lambdas=(1., 0.),
                     anchor=None, strength=1.0, clip=0.0, normalize=False,
                     endpoint_guidance=False, seed=123, objective=None):
    """Run the reverse chain from step K down to 1.

    With `anchor` the chain starts at step round(strength * K) from the forward-
    noised anchor (SDEdit), rather than at K from pure noise. strength=1.0
    reproduces the unanchored chain.
    """
    lam = weight_vector(lambdas, objective)
    K    = len(betas) - 1
    torch.manual_seed(seed)
    start = K
    if anchor is None:
        z = torch.randn(n, model.length, model.dim, device=DEVICE)
    else:
        start = max(1, min(K, int(round(float(strength) * K))))
        a     = alpha_bars[start]
        z1    = anchor.to(DEVICE).expand(n, -1, -1)
        z     = a.sqrt() * z1 + (1 - a).sqrt() * torch.randn(n, model.length, model.dim,
                                                             device=DEVICE)
    null = torch.full((n,), 2, dtype=torch.long, device=DEVICE)
    cond = torch.full((n,), c, dtype=torch.long, device=DEVICE)
    for k in range(start, 0, -1):
        t   = torch.full((n,), k / K, device=DEVICE)
        a_k = alpha_bars[k]

        def noise_pred(state, condition):
            out = model(state, t, condition)
            if model.predict != "x0":
                return out
            # Lecture 3.3: "Convert Z0_hat into a noise estimate and reuse the
            # DDPM reverse update."
            return (state - a_k.sqrt() * out) / (1 - a_k).sqrt().clamp_min(1e-4)

        eps = noise_pred(z, null)
        if w:
            eps = eps + w * (noise_pred(z, cond) - noise_pred(z, null))
        sigma = (1 - alpha_bars[k]).sqrt()
        if eta:
            endpoint = None
            if endpoint_guidance:
                def endpoint(state, _t=t, _null=null, _a=alpha_bars[k]):
                    out = model(state, _t, _null)
                    # In x0 mode the network already returns the clean estimate.
                    return out if model.predict == "x0" else \
                        endpoint_from_noise(state, out, _a)
            eps = eps - eta * sigma * reward_gradient(reward_model, z, t, lam,
                                                      clip, normalize, endpoint,
                                                      objective)
        mean = (z - betas[k] * eps / sigma) / alphas[k].sqrt()
        z    = mean + post_vars[k].sqrt() * torch.randn_like(z) if k > 1 else mean
    return z
