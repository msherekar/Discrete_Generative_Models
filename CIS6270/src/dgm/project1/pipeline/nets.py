"""Velocity, noise and reward networks, plus the weight averager.

Both generative heads share one trunk choice -- a flattened MLP or the
per-residue Transformer from trunks.py -- so architecture is a flag rather than
a fork in the pipeline.
"""
import torch
from torch import nn

from .config import HIDDEN
from .embeddings import TimeEmbedding
from .losses import eps_from_velocity
from .trunks import TransformerField

# Width of the time and condition features in the legacy MLP trunk.
MLP_TIME, MLP_COND = 32, 16

# ══════════════════════════════════════════════════════════════════════════════
# The MLP trunk, kept as the architecture control
# ══════════════════════════════════════════════════════════════════════════════

class MLPField(nn.Module):
    """Flattens [L, D] into one vector and regresses the field from it.

    Wrong for a protein and kept only as the Stage 1 control: on avGFP it
    presses 237 x 320 = 75,840 inputs through a single hidden layer with no
    weight sharing across residues, so a position's identity is encoded purely
    in where it lands in the flattened vector. Still the right default for the
    64-sequence toy peptide, where the transformer has nothing to share.
    """

    def __init__(self, length, dim, hidden=HIDDEN, n_cond=1, conditioning="binary"):
        super().__init__()
        self.length, self.dim, self.conditioning = length, dim, conditioning
        self.time = TimeEmbedding(MLP_TIME)
        if conditioning == "continuous":
            self.condition = nn.Sequential(nn.Linear(n_cond, MLP_COND), nn.SiLU())
        elif conditioning == "binary":
            self.condition = nn.Embedding(3, MLP_COND)
        else:
            self.condition = None
        width = length * dim + MLP_TIME + (MLP_COND if self.condition else 0)
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, length * dim),
        )

    def forward(self, z, t, c=None, drop=None):
        parts = [z.flatten(1), self.time(t)]
        if self.condition is not None and c is not None:
            if self.conditioning == "continuous":
                values = (c[:, None] if c.dim() == 1 else c).float()
                # Dropped rows get the mean of the standardized conditioning
                # value, which is zero -- the MLP control has no null token.
                if drop is not None:
                    values = values.masked_fill(drop[:, None], 0.0)
                parts.append(self.condition(values))
            else:
                c = c.long()
                parts.append(self.condition(
                    c.masked_fill(drop, 2) if drop is not None else c))
        return self.net(torch.cat(parts, dim=1)).reshape_as(z)


def make_trunk(length, dim, hidden, arch, conditioning, modulation, rope, n_cond):
    """One constructor for both heads, so a flag never means two things."""
    if arch == "transformer":
        return TransformerField(length, dim, d_model=hidden,
                                conditioning=conditioning, modulation=modulation,
                                rope=rope, n_cond=n_cond)
    return MLPField(length, dim, hidden, n_cond, conditioning)


# ══════════════════════════════════════════════════════════════════════════════
# Generative heads
# ══════════════════════════════════════════════════════════════════════════════

class FlowModel(nn.Module):
    """Predicts the velocity field dX/dt at a state and time.

    The `skip` term adds a time-dependent multiple of the input to the output.
    It is kept because the conditional velocity target for the straight path is
    X1 - X0, which contains a component along X_t itself, and letting the
    network express that through one scalar per time is cheaper than learning
    it in the trunk.
    """

    def __init__(self, length, dim, hidden=HIDDEN, arch="mlp",
                 conditioning="binary", modulation="adaln", rope=True, n_cond=1):
        super().__init__()
        self.arch = arch
        self.time = TimeEmbedding(MLP_TIME)
        self.skip = nn.Linear(MLP_TIME, 1)
        self.trunk = make_trunk(length, dim, hidden, arch, conditioning,
                                modulation, rope, n_cond)

    def forward(self, z, t, c=None, drop=None):
        return (self.skip(self.time(t))[:, :, None] * z
                + self.trunk(z, t, c, drop))


class DiffusionModel(nn.Module):
    """Denoiser for the DDPM chain, in whichever parameterization is selected.

    `predict` selects what the trunk's raw output means. Every option is
    converted back to a noise estimate by `to_noise`, so the samplers never
    branch on it.

    "eps" is the original and the reason this docstring exists: the output is
        sqrt(1-abar_k) * z  +  sqrt(abar_k) * net(...)
    At high noise abar_k -> 0, so the prediction collapses to z, which already
    equals the added noise almost exactly -- the network is handed a correct
    answer for free and contributes nothing. Measured on the default schedule
    its weight exceeds 0.5 for only 369 of 1000 steps, and across five seeds the
    training loss fell just 25% (flow's fell 78%). The 2026-10-03 H100 run is
    the end state: 500 epochs moved the loss from 0.1819 to 0.1799.

    "x0" follows Lecture 3.2's AMP-Diffusion recipe -- "directly predict
    denoised latent and train using MSE loss" -- so the network owns the
    prediction at every noise level. This is now the default.

    "v" is the Salimans & Ho velocity parameterization, which is well
    conditioned at BOTH ends of the schedule rather than one; see losses.py.
    """

    def __init__(self, length, dim, alpha_bars, hidden=HIDDEN, arch="mlp",
                 predict="x0", conditioning="binary", modulation="adaln",
                 rope=True, n_cond=1):
        super().__init__()
        self.arch, self.predict = arch, predict
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)
        self.K = len(alpha_bars) - 1
        self.trunk = make_trunk(length, dim, hidden, arch, conditioning,
                                modulation, rope, n_cond)

    def forward(self, z, t, c=None, drop=None):
        """The raw prediction, in this model's own parameterization."""
        return self.trunk(z, t, c, drop)

    def alpha_bar_at(self, t):
        """abar for a batch of normalized times, shaped to broadcast."""
        k = (t * self.K).round().long().clamp(0, self.K)
        return self.alpha_bars[k].view((-1,) + (1,) * 2)

    def to_noise(self, prediction, z, t):
        """Convert any parameterization to the eps the reverse update needs."""
        a = self.alpha_bar_at(t)
        if self.predict == "eps":
            return prediction
        if self.predict == "v":
            return eps_from_velocity(z, prediction, a)
        # x0: invert z = sqrt(abar) x0 + sqrt(1-abar) eps.
        return (z - a.sqrt() * prediction) / (1 - a).sqrt().clamp_min(1e-8)

    def noise(self, z, t, c=None, drop=None):
        """One call for samplers that only ever want a noise estimate."""
        return self.to_noise(self(z, t, c, drop), z, t)


class RewardModel(nn.Module):
    """Predicts the standardized property vector from a noisy state.

    Trained at the same noise levels guidance queries it at, which is what
    makes the reward gradient meaningful mid-trajectory rather than only at
    t=1. Takes no condition: it is a measurement, not a control.

    `zero_init_out=False` on the shared trunk is load-bearing. Guidance is
    driven entirely by d(reward)/dz, and a zero-initialized output projection
    makes the trunk's output -- and therefore the reward -- constant in z, so
    that derivative is exactly zero and every guided arm reduces to the
    unguided one. The weights do eventually move, because the projection's own
    gradient is nonzero, but nothing below it receives any gradient until they
    do, and the arms are indistinguishable in the meantime.
    """

    def __init__(self, length, dim, hidden=HIDDEN, arch="mlp", n_props=2):
        super().__init__()
        self.arch, self.n_props = arch, n_props
        if arch == "transformer":
            # Same trunk, then mean-pool over residues to one value per objective.
            self.trunk = TransformerField(length, dim, d_model=hidden,
                                          conditioning="none",
                                          zero_init_out=False)
            self.head = nn.Linear(dim, n_props)
        else:
            self.net = nn.Sequential(
                nn.Linear(length * dim + 1, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_props),
            )

    def forward(self, z, t):
        if self.arch == "transformer":
            return self.head(self.trunk(z, t).mean(dim=1))
        return self.net(torch.cat([z.flatten(1), t[:, None]], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# Weight averaging
# ══════════════════════════════════════════════════════════════════════════════

class EMA:
    """Exponential moving average of the weights, used for sampling.

    Standard practice in diffusion training (DDPM uses decay 0.9999) and absent
    from the flow path entirely until Stage 1 -- `--ema` was accepted and
    silently ignored there. The averaged weights are far less sensitive to
    which minibatch landed last, which is what drives the run-to-run spread:
    diffusion's seed-to-seed variation was 5-10x flow's (+/-0.04-0.06 against
    +/-0.005-0.008), and flow's training loss ticked UP on the final epoch of
    both long runs (0.4090 -> 0.4117 at 125, 0.3094 -> 0.3156 at 250).

    *Ref: Karras et al., EDM (NeurIPS 2022); Karras et al., post-hoc EMA
    (EDM2, CVPR 2024).*
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        state = model.state_dict()
        for k, v in self.shadow.items():
            state[k].copy_(v.to(state[k].dtype))
