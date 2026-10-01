"""Complete samplers and guidance extensions around the lecture code modules."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from lecture_core import (K, MASK, draw, token_ce, udlm_rates,
                          rate_step, geometric_cfg, pareto_filter)
from common import objectives


@torch.no_grad()
def uniform_sample(model, batch, length, steps=100, epsilon=.02):
    """Adaptive reverse Euler on the same truncated interval as udlm_loss.

    Uniform initialization at t=1-epsilon and stopping at t=epsilon
    are endpoint approximations. The returned sequence retains residual
    corruption; UDLM output probabilities are not treated as a clean posterior.
    """
    z = torch.randint(K, (batch, length))
    time = 1. - epsilon
    count = 0
    while time > epsilon + 1e-8:
        t = torch.full((batch,), time)
        rate = udlm_rates(model(z, t).softmax(-1), z, t)
        max_exit = float(rate.sum(-1).max())
        h = min(1. / steps, time - epsilon, .5 / max(max_exit, 1e-8))
        z = rate_step(z, rate, h)
        time -= h
        count += 1
        if count > 100000:
            raise RuntimeError('Adaptive sampler failed to advance.')
    return z


@torch.no_grad()
def block_sample(model, batch, length, size=2, steps=20):
    """Generate one block completely before appending the next one."""
    prefix = torch.empty((batch, 0), dtype=torch.long)
    for start in range(0, length, size):
        width = min(size, length - start)
        block = torch.full((batch, width), MASK)
        grid = torch.linspace(1., 0., steps + 1)
        for t, s in zip(grid[:-1], grid[1:]):
            context = torch.cat((prefix, block), 1)
            p = model(context)[:, start:].softmax(-1)
            candidate = draw(p)
            reveal = (block == MASK) & (torch.rand(block.shape) < (t-s)/t)
            block = torch.where(reveal, candidate, block)
        prefix = torch.cat((prefix, block), 1)
    return prefix


def conditional_loss(model, clean, labels, drop=.15):
    t = torch.rand(len(clean)).clamp_min(1e-4)
    mask = torch.rand(clean.shape) < t[:, None]
    context = clean.masked_fill(mask, MASK)
    condition = labels.clone()
    condition[torch.rand(len(clean)) < drop] = 2
    ce = token_ce(model(context, label=condition), clean)
    return (ce * mask / t[:, None]).sum(1).mean()


@torch.no_grad()
def cfg_sample(model, batch, length, strength=2., label=1, steps=20):
    z = torch.full((batch, length), MASK)
    grid = torch.linspace(1., 0., steps + 1)
    labels = torch.full((batch,), label, dtype=torch.long)
    for t, s in zip(grid[:-1], grid[1:]):
        uncond = model(z).softmax(-1)
        cond = model(z, label=labels).softmax(-1)
        probability = geometric_cfg(uncond, cond, strength)
        candidate = draw(probability)
        reveal = (z == MASK) & (torch.rand(z.shape) < (t-s)/t)
        z = torch.where(reveal, candidate, z)
    return z


class NoisyClassifier(nn.Module):
    """Predict high-GC class from a masked sequence and its noise level."""
    def __init__(self, length):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(length * 5 + 1, 64),
                                 nn.SiLU(), nn.Linear(64, 1))

    def forward(self, z, t):
        features = F.one_hot(z, 5).float() if z.ndim == 2 else z
        return self.net(torch.cat((features.flatten(1), t[:, None]), 1)).squeeze(-1)


def fit_classifier(model, data, labels, steps=200, batch=32):
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    losses = []
    for _ in range(steps):
        idx = torch.randint(len(data), (batch,))
        t = torch.rand(batch)
        noisy = data[idx].masked_fill(torch.rand(batch, data.shape[1]) < t[:, None], MASK)
        loss = F.binary_cross_entropy_with_logits(model(noisy, t), labels[idx].float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    return losses


def log_success(classifier, z, t, label):
    logit = classifier(z, t)
    return F.logsigmoid(logit if label == 1 else -logit)


def guidance_changes(classifier, z, time, label=1, gradient=False):
    """Log h(candidate)-log h(current), exact or first-order one-hot."""
    batch, length = z.shape
    t = torch.full((batch,), float(time))
    if gradient:
        with torch.enable_grad():
            soft = F.one_hot(z, 5).float().requires_grad_(True)
            value = log_success(classifier, soft, t, label)
            grad = torch.autograd.grad(value.sum(), soft)[0]
        current = grad.gather(-1, z[..., None])
        return grad[..., :K] - current
    with torch.no_grad():
        current = log_success(classifier, z, t, label)
        delta = torch.empty((batch, length, K))
        for i in range(length):
            for a in range(K):
                edited = z.clone()
                edited[:, i] = a
                delta[:, i, a] = log_success(classifier, edited, t, label) - current
        return delta


@torch.no_grad()
def classifier_sample(model, classifier, batch, length, strength=1.,
                      label=1, steps=40, gradient=False, epsilon=.005):
    """Rate guidance with adaptive Euler and an explicit endpoint closure."""
    z = torch.full((batch, length), MASK)
    time = 1.
    iterations = 0
    while time > epsilon + 1e-8:
        p = model(z).softmax(-1)
        delta = guidance_changes(classifier, z, time, label, gradient)
        rates = p / time * (strength * delta).clamp(-20, 20).exp()
        rates *= (z == MASK)[..., None]
        exit_rate = rates.sum(-1)
        h = min(1. / steps, time-epsilon, .5 / max(float(exit_rate.max()), 1e-8))
        change = torch.rand(z.shape) < h * exit_rate
        candidate = draw(rates / exit_rate.clamp_min(1e-12)[..., None] + 1e-12)
        z = torch.where(change & (z == MASK), candidate, z)
        time -= h
        iterations += 1
        if iterations > 100000:
            raise RuntimeError('Guided rate integration failed to advance.')
    z = torch.where(z == MASK, draw(model(z).softmax(-1)), z)
    return z


class SearchNode:
    def __init__(self, tokens, parent=None, prior=1.):
        self.tokens, self.parent, self.prior = tokens, parent, prior
        self.children = []
        self.visits = 0
        self.reward = np.zeros(2)


@torch.no_grad()
def peptune_search(model, length=8, iterations=100, branching=4):
    """DNA MCTS: selection, expansion, completion, Pareto rewards, backup.

    All DNA strings are valid. Peptide chemistry, bond-dependent masks,
    RoFormer training, and the PepTune invalid-SMILES penalty are not used.
    """
    root = SearchNode(torch.full((length,), MASK))
    archive = torch.empty((0, length), dtype=torch.long)
    archive_scores = torch.empty((0, 2))
    trace = []
    for iteration in range(iterations):
        node = root
        while node.children:
            def selection(child):
                mean = child.reward.mean() / max(child.visits, 1)
                bonus = 1.5 * child.prior * math.sqrt(node.visits + 1) / (child.visits + 1)
                return mean + bonus
            node = max(node.children, key=selection)
        if (node.tokens == MASK).any():
            p = model(node.tokens[None]).softmax(-1)[0]
            positions = torch.where(node.tokens == MASK)[0]
            seen = set()
            for _ in range(branching):
                pos = int(positions[torch.randint(len(positions), ())])
                token = int(torch.multinomial(p[pos], 1))
                candidate = node.tokens.clone()
                candidate[pos] = token
                key = tuple(candidate.tolist())
                if key not in seen:
                    node.children.append(SearchNode(candidate, node, float(p[pos, token])))
                    seen.add(key)
            node = node.children[0]
        rollout = node.tokens.clone()
        while (rollout == MASK).any():
            prob = model(rollout[None]).softmax(-1)
            position = int(torch.where(rollout == MASK)[0][0])
            rollout[position] = draw(prob)[0, position]
        score = objectives(rollout[None])[0]
        reward = ((score >= archive_scores).float().mean(0).numpy()
                  if len(archive) else np.ones(2))
        archive = torch.cat((archive, rollout[None]))
        archive_scores = torch.cat((archive_scores, score[None]))
        archive, archive_scores = pareto_filter(archive, archive_scores)
        # Equal-score alternatives remain; remove exact repeated sequences only.
        unique = []; seen = set()
        for i, row in enumerate(archive.tolist()):
            key = tuple(row)
            if key not in seen:
                unique.append(i); seen.add(key)
        archive, archive_scores = archive[unique], archive_scores[unique]
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent
        trace.append(dict(iteration=iteration, archive_size=len(archive), score=score.tolist()))
    return archive, archive_scores, trace
