"""Full teaching samplers for simplex, rectification, and guided jump flows."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from lecture_core import (K, draw, gat_rates, rate_step, gat_sample,
                          dirichlet_velocity, dna_neighbors, mh_refine)
from common import project_simplex, objectives, soft_objectives, decode


@torch.no_grad()
def simplex_sample(model, method, batch, length, steps=100, a_max=8.,
                   beta=1., tau_max=4., guidance=0.):
    """Finite Euler discretization with explicit simplex/orthant projection.

    Projection is a numerical safeguard, not the exact continuous dynamics.
    Gumbel has an approximate finite-temperature base and endpoint.
    """
    base = torch.distributions.Dirichlet(torch.ones(K))
    z = base.sample((batch, length))
    corrections = 0
    if method == 'gumbel':
        uniform = torch.rand(batch, length, K).clamp(1e-6, 1-1e-6)
        g = -(-uniform.log()).log()
        z = (g / (beta * tau_max)).softmax(-1)
    y = z.sqrt() if method == 'fisher' else z
    end = a_max if method == 'dirichlet' else 1.
    dt = end / steps
    for i in range(steps):
        time = i * dt
        t = torch.full((batch,), time)
        if method == 'dirichlet':
            posterior = model(y, t).softmax(-1)
            velocity = dirichlet_velocity(y, posterior, time)
        elif method == 'fisher':
            raw = model(y, t)
            velocity = raw - y * (raw * y).sum(-1, keepdim=True)
        else:
            raw = model(y, t)
            velocity = raw - raw.mean(-1, keepdim=True)
        if guidance:
            # A differentiable toy objective on probabilities, not a trained
            # biochemical predictor. Choose a direction toward higher GC.
            with torch.enable_grad():
                state = y.detach().requires_grad_(True)
                probability = state.square() if method == 'fisher' else state
                score = soft_objectives(probability)[:, 0].sum()
                gradient = torch.autograd.grad(score, state)[0]
            if method == 'fisher':
                gradient -= y * (gradient * y).sum(-1, keepdim=True)
            else:
                gradient -= gradient.mean(-1, keepdim=True)
            velocity += guidance * gradient
        candidate = y + dt * velocity
        if not torch.isfinite(candidate).all():
            raise FloatingPointError('Nonfinite simplex integration state.')
        corrections += int((candidate < 0).any(-1).sum())
        if method == 'fisher':
            candidate = candidate.clamp_min(1e-6)
            y = candidate / candidate.norm(dim=-1, keepdim=True)
        else:
            y = project_simplex(candidate)
    prob = y.square() if method == 'fisher' else y
    result = prob.argmax(-1)
    return result, {'negative_coordinate_corrections': corrections,
                    'max_probability_sum_error': float((prob.sum(-1)-1).abs().max()),
                    'finite_endpoint': True, 'decode': 'argmax'}


def rectified_training_loss(model, clean):
    source = torch.distributions.Dirichlet(torch.ones(K)).sample(clean.shape)
    target = F.one_hot(clean, K).float()
    t = torch.rand(len(clean))
    z = (1-t[:, None, None])*source + t[:, None, None]*target
    raw = model(z, t)
    prediction = raw - raw.mean(-1, keepdim=True)
    return (prediction-(target-source)).square().sum(-1).mean()


@torch.no_grad()
def make_teacher_pairs(teacher, count, length, steps=60, chunk=64):
    sources, targets = [], []
    for start in range(0, count, chunk):
        batch = min(chunk, count-start)
        source = torch.randint(K, (batch, length))
        target = gat_sample(teacher, batch, length, steps, source=source)
        sources.append(source); targets.append(target)
    return torch.cat(sources), torch.cat(targets)


def mog_multiplier(changes, preference, strength=1., angle=math.pi/3):
    """Rank/direction score and acute-cone test; all objectives maximize."""
    from scipy.stats import rankdata
    ranks = rankdata(changes, axis=0, method='average') / len(changes)
    rank_score = ranks.mean(1)
    norm = np.linalg.norm(changes, axis=1)
    cosine = (changes @ preference) / np.maximum(norm*np.linalg.norm(preference), 1e-12)
    direction = changes @ preference
    # Formula matches the lecture: z-standardized rank and direction terms.
    def standardize(x):
        return (x-x.mean()) / max(float(x.std()), 1e-12)
    score = standardize(rank_score) + standardize(direction)
    keep = (norm > 0) & (cosine >= np.cos(angle))
    return strength * np.exp(np.clip(score, -20, 20)) * keep


@torch.no_grad()
def mog_sample(model, batch, length, steps=50, preference=(.7, .3), strength=1.):
    """Evaluate all single-base edits, reweight rates, then use adaptive Euler.

    All-position scoring differs from the paper's random-position loop.
    Rank normalization is per position. The adaptive cone stays acute;
    no outside-cone fallback is used, so the local theorem remains valid.
    """
    z = torch.randint(K, (batch, length))
    t = 0.
    rejected = 0
    angle = math.pi/3
    ema = .5
    count = 0
    while t < 1-1e-6:
        tb = torch.full((batch,), t)
        rates = gat_rates(model(z, tb).softmax(-1), z, tb)
        rejected_step = 0
        for b in range(batch):
            for i in range(length):
                candidates, destinations = [], []
                for a in range(K):
                    if a == int(z[b, i]):
                        continue
                    y = z[b].clone(); y[i] = a
                    candidates.append(y); destinations.append(a)
                changes = (objectives(torch.stack(candidates)) - objectives(z[b:b+1])).numpy()
                multiplier = mog_multiplier(changes, np.array(preference), strength, angle)
                rejected_step += int(np.sum(multiplier == 0))
                for a, value in zip(destinations, multiplier):
                    rates[b, i, a] *= float(value)
        rejected += rejected_step
        rejection = rejected_step / (batch * length * (K-1))
        ema = .9 * ema + .1 * rejection
        angle = float(np.clip(angle * np.exp(.02 * (ema-.5)),
                              np.deg2rad(10), np.deg2rad(89)))
        max_exit = float(rates.sum(-1).max())
        h = min(1./steps, 1-t, .5/max(max_exit, 1e-8))
        z = rate_step(z, rates, h)
        t += h
        count += 1
        if count > 50000:
            raise RuntimeError('MOG integration failed to advance.')
    return z, {'filtered_candidates': rejected, 'adaptive_steps': count,
               'preference': list(preference), 'final_cone_angle_degrees': float(np.rad2deg(angle)),
               'guarantee': 'local weighted progress under the acute-cone assumptions'}


def reference_model(data, pseudocount=.5):
    """A tractable positive product reference for exact MH ratios."""
    counts = F.one_hot(data, K).float().sum(0) + pseudocount
    probability = counts / counts.sum(-1, keepdim=True)
    def log_probability(sequence):
        idx = torch.tensor(['ACGT'.index(c) for c in sequence])
        return float(probability.log()[torch.arange(len(idx)), idx].sum())
    return probability, log_probability


def refine_areuredi(tokens, data, steps=100, preference=(.5,.5), eta_max=8., seed=7):
    """ReDi endpoint refinement with normalized proposals and exact MH ratios.

    The reference is explicitly fitted and evaluable; it is not a denoiser
    joint density. Annealing is finite, so no global-optimum claim is made.
    """
    from lecture_core import encode
    _, log_ref = reference_model(data)
    weight = np.array(preference)
    def score(sequence):
        values = objectives(encode([sequence]))[0].numpy()
        return float(np.min(weight * values))
    rng = np.random.default_rng(seed)
    strings = decode(tokens)
    changes = 0
    for step in range(steps):
        eta = eta_max * (step+1) / steps
        for i, sequence in enumerate(strings):
            new = mh_refine(sequence, score, log_ref, eta, rng)
            changes += int(new != sequence)
            strings[i] = new
    return encode(strings), {'accepted_edits': changes, 'eta_final': eta_max,
                            'reference': 'smoothed position-wise empirical DNA frequencies',
                            'scope': 'finite annealing demonstration, not an equilibrium certificate'}
