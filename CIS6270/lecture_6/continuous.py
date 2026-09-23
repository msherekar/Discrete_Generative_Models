"""Flow matching, finite-map identities, Shortcut, MeanFlow, and latent maps.

# %% 1. Learn local motion before learning an interval.
All times run noise -> data except MeanFlow, whose original backward clock
r <= t runs data at zero -> noise at one during training.
"""
import torch
from torch import nn
from torch.nn import functional as F

from common import (MapNet, draw_batch, ema_copy, finite_map, interpolate, mlp,
                    optimize, ordered_times, time_like)


def diagonal_loss(model, data):
    t = torch.rand(len(data), 1, device=data.device)
    x, displacement = interpolate(data, t)
    return F.mse_loss(model(x, t, t), displacement)


def lagrangian_residual(model, velocity, x, s, t):
    # d_t F(s,t,x) = b(t,F(s,t,x)). JVP differentiates only arrival time.
    y, dt = torch.func.jvp(lambda end: finite_map(model, x, s, end),
                         (t,), (torch.ones_like(t),))
    return dt - velocity(y.detach(), t).detach()


def eulerian_residual(model, velocity, x, s, t):
    # (d_s + b_s dot grad_x)F = 0. No full Jacobian is materialized.
    direction = velocity(x, s).detach()
    _, residual = torch.func.jvp(lambda z, start: finite_map(model, z, start, t),
                               (x, s), (direction, torch.ones_like(s)))
    return residual


def semigroup_loss(model, teacher, x, s, t, context=None):
    u = (s + t) / 2
    with torch.no_grad():
        mid = finite_map(teacher, x, s, u, context)
        target = finite_map(teacher, mid, u, t, context)
    prediction = finite_map(model, x, s, t, context)
    return ((prediction - target) / (t - s).clamp_min(.05)).square().mean()


def meanflow_loss(model, data):
    # %% 2. Average backward velocity satisfies u = v - (t-r) D_t u.
    noise = torch.randn_like(data)
    r, t, _ = ordered_times(data, 1.)
    # Include diagonal examples with nonzero probability, as in the paper.
    r = torch.where(torch.rand_like(r) < .75, t, r)
    z = (1 - t) * data + t * noise
    v = noise - data
    average, derivative = torch.func.jvp(model, (z, r, t),
        (v, torch.zeros_like(r), torch.ones_like(t)))
    target = (v - (t - r) * derivative).detach()
    residual = (average - target).square().mean(-1)
    # Detached adaptive weighting controls large self-distillation residuals.
    weight = (residual.detach() + .01).pow(-.5)
    return (weight * residual).mean(), {'mse': residual.mean()}


def shortcut_loss(model, teacher, data):
    t = torch.rand(len(data), 1, device=data.device)
    x, v = interpolate(data, t)
    diag = F.mse_loss(model(x, t, t), v)
    # d is the half-step. Learn the 2d shortcut from two d shortcuts.
    powers = torch.randint(1, 6, (len(data), 1), device=data.device)
    d = 2. ** (-powers)
    start = torch.rand_like(d) * (1 - 2 * d)
    x, _ = interpolate(data, start)
    with torch.no_grad():
        first = teacher(x, start, start + d)
        second = teacher(x + d * first, start + d, start + 2 * d)
        target = .5 * (first + second)
    finite = F.mse_loss(model(x, start, start + 2 * d), target)
    return diag + finite, {'diagonal': diag, 'shortcut': finite}


def train_continuous(method, data, args):
    model = MapNet(data.shape[1], args.width).to(data)
    teacher_logs = []
    teacher = None
    if method in {'fmm-lagrangian', 'fmm-eulerian', 'consistency'}:
        teacher = MapNet(data.shape[1], args.width).to(data)
        teacher_logs = optimize(teacher, lambda _: (diagonal_loss(teacher, draw_batch(data, args.batch_size)), {}),
                                args.teacher_steps, args.lr)
        teacher.requires_grad_(False)
        model.load_state_dict(teacher.state_dict())
    ema = ema_copy(model)

    def objective(step):
        batch = draw_batch(data, args.batch_size)
        if method == 'meanflow':
            return meanflow_loss(model, batch)
        if method == 'shortcut':
            return shortcut_loss(model, ema, batch)
        diag = diagonal_loss(model, batch)
        if method == 'flow-matching':
            return diag, {'diagonal': diag}
        s, t, _ = ordered_times(batch)
        x, _ = interpolate(batch, s)
        if method == 'fmm-lagrangian':
            finite = lagrangian_residual(model, lambda z, a: teacher(z, a, a), x, s, t).square().mean()
        elif method == 'fmm-eulerian':
            finite = eulerian_residual(model, lambda z, a: teacher(z, a, a), x, s, t).square().mean()
        elif method == 'consistency':
            # Endpoint consistency distillation on teacher-solved short intervals.
            # At t=1 the residual endpoint parametrization is exactly identity.
            t = (s + .1).clamp_max(1.)
            with torch.no_grad():
                y = integrate_velocity(lambda z, a: teacher(z, a, a), x, s, t, 4)
                target = finite_map(ema, y, t, torch.ones_like(t))
            finite = F.mse_loss(finite_map(model, x, s, torch.ones_like(s)), target)
        else:
            finite = semigroup_loss(model, ema, x, s, t)
        weight = min(1., (step + 1) / max(1, args.train_steps // 5))
        return diag + weight * finite, {'diagonal': diag, 'finite': finite}

    logs = optimize(model, objective, args.train_steps, args.lr, ema)
    # Use the trained student; EMA supplies fixed bootstrap targets during training.
    state = {'model': model.state_dict(), 'dim': data.shape[1]}
    if teacher is not None:
        state['teacher'] = teacher.state_dict()
    return model, state, logs, teacher_logs


@torch.no_grad()
def integrate_velocity(velocity, x, s, t, steps):
    """Heun integration of a velocity, also supports batch-specific time bounds."""
    s, t = time_like(s, x), time_like(t, x)
    h = (t - s) / steps
    for i in range(steps):
        a = s + i * h
        k1 = velocity(x, a)
        k2 = velocity(x + h * k1, a + h)
        x = x + h * .5 * (k1 + k2)
    return x


@torch.no_grad()
def sample_continuous(model, method, noise, steps):
    if method == 'flow-matching':
        return integrate_velocity(lambda z, a: model(z, a, a), noise, 0., 1., steps)
    if method == 'consistency':
        return finite_map(model, noise, 0., 1.)
    x = noise
    for i in range(steps):
        if method == 'meanflow':
            t, r = 1 - i / steps, 1 - (i + 1) / steps
            x = x - (t - r) * model(x, r, t)
        else:
            x = finite_map(model, x, i / steps, (i + 1) / steps)
    return x


class Autoencoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = mlp(3, 2, width)
        self.decoder = mlp(2, 3, width)


def embed_surface(x):
    return torch.cat([x, .3 * (x[:, :1].square() - x[:, 1:].square())], -1)


def train_latent(data, args):
    # %% 3. First fit the representation, then freeze it while fitting its flow.
    surface = embed_surface(data)
    ae = Autoencoder(args.width).to(data)
    def objective(_):
        batch = draw_batch(surface, args.batch_size)
        reconstruction = ae.decoder(ae.encoder(batch))
        return F.mse_loss(reconstruction, batch), {}
    ae_logs = optimize(ae, objective, args.teacher_steps, args.lr)
    ae.requires_grad_(False)
    with torch.no_grad():
        z = ae.encoder(surface)
        mean, std = z.mean(0), z.std(0).clamp_min(.05)
        z = (z - mean) / std
    model, state, logs, _ = train_continuous('self-distill', z, args)
    state.update({'autoencoder': ae.state_dict(), 'latent_mean': mean, 'latent_std': std})
    return model, ae, state, logs, ae_logs
