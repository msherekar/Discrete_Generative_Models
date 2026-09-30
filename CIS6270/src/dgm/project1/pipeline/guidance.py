"""Differentiating the objective with respect to the latent.

One function, because every guided sampler needs exactly this: the
lambda-weighted reward gradient, optionally normalized per objective, clipped,
and evaluated on a predicted clean endpoint.
"""
import torch

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
