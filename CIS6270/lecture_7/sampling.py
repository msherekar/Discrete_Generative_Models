"""Seeded samplers for the saved Lecture 7 teaching models."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from scipy.linalg import expm
from learned_bridges import mlp, MaskedModel, CoupledCone, token_reward
from discrete_learning import rate_model


def generate(state, samples=256, seed=107, sample_steps=None):
    if samples < 1 or (sample_steps is not None and sample_steps < 1):
        raise ValueError('Sample counts and step counts must be positive.')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    method = state['method']
    steps = sample_steps or state.get('steps', 100)
    traces = []
    report = {'method': method, 'sampling_seed': seed, 'samples': samples}
    result = state.get('result', {})

    if method in ['dsb', 'dsbm']:
        coeffs = np.asarray(state['forward'])
        if steps != len(coeffs):
            raise ValueError('This saved affine model uses its fixed training grid; omit --sample-steps.')
        dt = 1/steps
        x = rng.normal(size=samples)
        start = x.copy()
        traces.append(x[:16].tolist())
        for slope, intercept in coeffs:
            drift = slope*x + intercept
            x = drift if method == 'dsb' else x + dt*drift
            x = x + math.sqrt(state['epsilon']*dt)*rng.normal(size=samples)
            traces.append(x[:16].tolist())
        values = x[:, None]
        report.update(mean=float(x.mean()), variance=float(x.var()),
                      endpoint_covariance=float(np.mean((start-start.mean())*(x-x.mean()))))

    elif method == 'sf2m':
        model = mlp(2, 2)
        model.load_state_dict(state['model'])
        model.eval()
        x = torch.randn(samples, 1)
        dt = 1/steps
        traces.append(x[:16, 0].tolist())
        with torch.no_grad():
            for k in range(steps):
                t = torch.full_like(x, (k+.5)*dt)
                velocity, score = model(torch.cat([t, x], 1)).chunk(2, 1)
                x = x + (velocity + .5*score)*dt + math.sqrt(dt)*torch.randn_like(x)
                traces.append(x[:16, 0].tolist())
        values = x.numpy()
        report.update(mean=float(x.mean()), variance=float(x.var(unbiased=False)),
                      approximation='Euler SDE; boundary times extrapolate beyond the training interval [0.02,0.98].')

    elif method == 'tr2d2':
        length, vocab = state['length'], state['vocab']
        if steps != length:
            raise ValueError('The abstract-token example reveals one of four positions per step.')
        model = MaskedModel(length, vocab)
        model.load_state_dict(state['model'])
        model.eval()
        x = torch.full((samples, length), vocab, dtype=torch.long)
        with torch.no_grad():
            for k in range(length):
                x[:, k] = torch.multinomial(model(x)[:, k].softmax(-1), 1).squeeze(1)
                traces.append(x[:16].tolist())
        values = x.numpy()
        report.update(mean_reward=float(token_reward(x).mean()),
                      vocabulary=['circle', 'square', 'triangle'],
                      sampler='Learned denoiser with fixed left-to-right reveal order; no search at inference.')

    elif method == 'branch':
        model = mlp(6, 2)
        model.load_state_dict(state['model'])
        model.eval()
        x0 = .12*torch.randn(samples, 2)
        x = x0[:, None, :].expand(-1, 3, -1).clone().reshape(-1, 2)
        branch = F.one_hot(torch.arange(3), 3).float().repeat(samples, 1)
        dt = 1/steps
        traces.append(x.reshape(samples, 3, 2)[:16].tolist())
        with torch.no_grad():
            for k in range(steps):
                # Midpoint integration of the learned branch-specific velocity.
                t = torch.full((len(x), 1), k*dt)
                v = model(torch.cat([t, branch, x], 1))
                vm = model(torch.cat([t+dt/2, branch, x+dt*v/2], 1))
                x = x + dt*vm
                traces.append(x.reshape(samples, 3, 2)[:16].tolist())
        values = x.reshape(samples, 3, 2).numpy()
        weights = np.array([1., 0., 0.]) + state['growth'].numpy()
        report.update(terminal_branch_weights=weights.tolist(), total_mass=float(weights.sum()),
                      minimum_weight=float(weights.min()), branch_means=values.mean(0).tolist(),
                      approximation='Learned velocity integration; soft growth constraints. Raw branch weights are retained, including negative residuals.')

    elif method == 'entangled':
        model = CoupledCone()
        model.load_state_dict(state['model'])
        model.eval()
        sigma = state['sigma']
        dt = 1/steps
        x = .3*torch.randn(samples, 3, 2)
        alignment = float('inf')
        traces.append(x[:16].tolist())
        with torch.no_grad():
            for k in range(steps):
                control = model(k*dt, x)
                alignment = min(alignment, float((control*(1-x)).sum(-1).min()))
                reference = -.12*x - .35*(x-x.mean(1, keepdim=True))
                x = x + (reference+sigma*control)*dt + sigma*math.sqrt(dt)*torch.randn_like(x)
                traces.append(x[:16].tolist())
        values = x.numpy()
        report.update(mean=float(x.mean()), minimum_bias_alignment=alignment,
                      terminal_potential_mean=float(torch.exp(-.5*((x-1)**2).sum((1,2))/1.5).mean()),
                      approximation='Overdamped interacting-particle Euler chain; terminal potential, not an enforced target marginal.')

    elif method == 'ddsbm':
        model = rate_model()
        model.load_state_dict(state['model'])
        model.eval()
        x = rng.integers(0, 2, samples)
        traces.append(x[:16].tolist())
        with torch.no_grad():
            for k in range(steps):
                t = (k+.5)/steps
                q = model(torch.tensor([[t, 1., 0.], [t, 0., 1.]])).ravel().numpy()
                total = float(q.sum())
                factor = -math.expm1(-total/steps)/total
                flip = factor*q[x]
                x = np.where(rng.random(samples) < flip, 1-x, x)
                traces.append(x[:16].tolist())
        values = x[:, None]
        report.update(terminal_frequencies=np.bincount(x, minlength=2).tolist(),
                      terminal_probability_one=float(x.mean()), target_probability_one=.8,
                      approximation='Exact two-state transitions for rates frozen at each grid midpoint; finite-time approximation to the learned inhomogeneous CTMC.')

    elif method in ['csbm', 'finite-sb']:
        Q = np.asarray(state['transitions'] if method == 'csbm' else result['Q'])
        if sample_steps is not None and sample_steps != len(Q):
            raise ValueError('This finite-chain checkpoint has two transitions.')
        x = rng.integers(0, 2, samples)
        paths = [x.copy()]
        for q in Q:
            x = (rng.random(samples) < q[x, 1]).astype(int)
            paths.append(x.copy())
        values = np.stack(paths, 1)
        report.update(terminal_probability_one=float(x.mean()), target_probability_one=.8)

    elif method in ['ot', 'sinkhorn']:
        name = 'plan' if method == 'ot' else 'entropic_plan'
        pi = np.asarray(result[name])
        pair = rng.choice(4, samples, p=pi.ravel()/pi.sum())
        values = np.stack([pair//2, pair%2], 1)
        cost = np.asarray(result['cost'])
        report.update(coupling=name, exact_expected_cost=float((pi*cost).sum()),
                      sampled_average_cost=float(cost[values[:, 0], values[:, 1]].mean()))

    elif method == 'discrete-imf':
        prob = np.asarray(result['final'])
        values = np.asarray(result['paths'])[rng.choice(len(prob), samples, p=prob/prob.sum())]
        report.update(terminal_probability_one=float(values[:, -1].mean()),
                      path_l1_to_exact_bridge=result['final_path_l1'])

    elif method == 'ctmc-sb':
        G = np.asarray(result['generator'])
        g = np.asarray(result['values'][-1]['h'])
        x = rng.integers(0, 2, samples)
        paths = [x.copy()]
        K = expm(G/steps)
        for k in range(steps):
            hs = expm((1-k/steps)*G)@g
            ht = expm((1-(k+1)/steps)*G)@g
            Q = K*ht[None, :]/hs[:, None]
            if not np.allclose(Q.sum(1), 1):
                raise ArithmeticError('The Doob transition failed to normalize.')
            x = (rng.random(samples) < Q[x, 1]).astype(int)
            paths.append(x.copy())
        values = np.stack(paths, 1)
        report.update(terminal_probability_one=float(x.mean()),
                      sampler='Exact reference-bridge transitions between observation times, up to matrix-exponential precision.')

    elif method == 'gaussian-sb':
        c = result['covariance']
        t = np.linspace(0, 1, steps+1)
        x0 = rng.normal(size=(samples, 1))
        x1 = 2+c*x0+math.sqrt(1-c*c)*rng.normal(size=(samples, 1))
        W = np.c_[np.zeros(samples), np.cumsum(rng.normal(size=(samples, steps))/math.sqrt(steps), axis=1)]
        paths = (1-t)*x0+t*x1+W-t*W[:, -1, None]
        values = paths[:, -1, None]
        traces = paths[:16].T.tolist()
        report.update(mean=float(values.mean()), variance=float(values.var()),
                      sampler='Exact Gaussian endpoint coupling plus Brownian conditional bridges on the grid.')

    elif method == 'reward-tilt':
        values = rng.choice(3, samples, p=result['target'])[:, None]
        report.update(empirical_probabilities=(np.bincount(values[:, 0], minlength=3)/samples).tolist(),
                      target_probabilities=result['target'])

    elif method == 'branch-mass':
        k = rng.choice(3, samples, p=result['weights'][-1])
        x = np.asarray(result['positions'])[-1, k]
        values = np.stack([k, x], 1)
        report.update(exact_terminal_weights=result['weights'][-1],
                      interpretation='Draw endpoints from the explicitly specified analytic branch mixture.')

    elif method == 'cone-geometry':
        d = np.asarray(result['direction'])
        bias = np.asarray(result['bias'])
        values = np.stack([d, bias, d-result['dt']*bias])
        report.update(samples=3, row_meanings=['target displacement', 'bias', 'displacement after the step'],
                      interpretation='A deterministic geometry calculation; these rows are not random samples.')
    else:
        raise ValueError('Unsupported checkpoint method: '+method)

    if not np.isfinite(values).all():
        raise FloatingPointError('Generated values are not finite.')
    report['finite_values'] = True
    return {'values': values.tolist(), 'trajectory_prefix': traces, 'report': report}
