"""Velocity, noise and reward networks, plus the weight averager.

Both generative heads share one trunk choice -- a flattened MLP or the
per-residue Transformer -- so architecture is a flag rather than a fork in the
pipeline.
"""
import torch
from torch import nn

from .config import HIDDEN

# ══════════════════════════════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════════════════════════════

class FlowModel(nn.Module):
    def __init__(self, length, dim, hidden=HIDDEN, arch="mlp"):
        super().__init__()
        self.length, self.dim, self.arch = length, dim, arch
        self.time = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.skip = nn.Linear(32, 1)
        if arch == "transformer":
            self.trunk = TransformerField(length, dim, d_model=hidden)
        else:
            self.condition = nn.Embedding(3, 16)
            self.net = nn.Sequential(
                nn.Linear(length * dim + 48, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, length * dim),
            )

    def forward(self, z, t, c):
        time = self.time(t[:, None])
        if self.arch == "transformer":
            return self.skip(time)[:, :, None] * z + self.trunk(z, t, c)
        inputs = torch.cat([z.flatten(1), time, self.condition(c)], dim=1)
        return self.skip(time)[:, :, None] * z + self.net(inputs).reshape_as(z)


class DiffusionModel(nn.Module):
    """Denoiser for the DDPM chain.

    `predict` selects the parameterization.

    "eps" is the original: the output is
        sqrt(1-abar_k) * z  +  sqrt(abar_k) * net(...)
    At high noise abar_k -> 0, so the prediction collapses to z, which already
    equals the added noise almost exactly -- the network is handed a correct
    answer for free and contributes nothing. Measured on the default schedule
    its weight exceeds 0.5 for only 369 of 1000 steps, and across five seeds the
    training loss fell just 25% (flow's fell 78%).

    "x0" follows Lecture 3.3's AMP-Diffusion recipe -- "The network predicts
    Z0_hat" -- so the network owns the prediction at every noise level. The
    sampler converts back with
        eps_hat = (z_k - sqrt(abar_k) x0_hat) / sqrt(1 - abar_k)
    and reuses the same reverse update.
    """

    def __init__(self, length, dim, alpha_bars, hidden=HIDDEN, arch="mlp",
                 predict="eps"):
        super().__init__()
        self.length, self.dim, self.arch, self.predict = length, dim, arch, predict
        self.alpha_bars = alpha_bars
        self.K          = len(alpha_bars) - 1
        if arch == "transformer":
            self.trunk = TransformerField(length, dim, d_model=hidden)
        else:
            self.time      = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
            self.condition = nn.Embedding(3, 16)
            self.net = nn.Sequential(
                nn.Linear(length * dim + 48, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, length * dim),
            )

    def forward(self, z, t, c):
        if self.arch == "transformer":
            raw = self.trunk(z, t, c)
        else:
            time   = self.time(t[:, None])
            inputs = torch.cat([z.flatten(1), time, self.condition(c)], dim=1)
            raw = self.net(inputs).reshape_as(z)
        if self.predict == "x0":
            return raw                      # the clean-latent estimate itself
        k = (t * self.K).round().long().clamp(0, self.K)
        a = self.alpha_bars[k, None, None]
        return (1 - a).sqrt() * z + a.sqrt() * raw


class TransformerField(nn.Module):
    """Per-residue Transformer trunk, following Lecture 3.3's AMP-Diffusion recipe.

    The MLP trunk flattens [L, D] into one vector, so a position's identity is
    encoded only in where it lands in that vector and nothing is shared between
    positions. On avGFP that is 75,840 inputs compressed through a single hidden
    layer, and 94.7% of the latent variance is positional -- the model fits the
    shared backbone and loses the per-variant signal.

    Lecture 3.3: "A Transformer replaces the U-Net... Adds residue-position
    information because order matters in a protein sequence... We will stack six
    Transformer layers... Add the noisy residue embeddings, the diffusion-step
    embedding shared across positions, and a learned embedding for each token
    position before applying the Transformer."

    Weights are shared across positions, so width no longer scales with sequence
    length: this trunk is roughly 5M parameters at any L, against 235M for the
    flattened MLP at width 1024.
    """

    def __init__(self, length, dim, d_model=256, layers=6, heads=8, dropout=0.0):
        super().__init__()
        self.project_in = nn.Linear(dim, d_model)
        self.position   = nn.Parameter(torch.zeros(1, length, d_model))
        nn.init.normal_(self.position, std=0.02)
        self.time = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(),
                                  nn.Linear(d_model, d_model))
        self.condition = nn.Embedding(3, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.project_out = nn.Linear(d_model, dim)
        nn.init.zeros_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)

    def forward(self, z, t, c=None):
        h = self.project_in(z) + self.position + self.time(t[:, None])[:, None, :]
        if c is not None:
            h = h + self.condition(c)[:, None, :]
        return self.project_out(self.encoder(h))


class RewardModel(nn.Module):
    def __init__(self, length, dim, hidden=HIDDEN, arch="mlp", n_props=2):
        super().__init__()
        self.arch = arch
        self.n_props = n_props
        if arch == "transformer":
            # Same trunk, then mean-pool over residues to one value per objective.
            self.trunk = TransformerField(length, dim, d_model=hidden)
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


class EMA:
    """Exponential moving average of the weights, used for sampling.

    Standard practice in diffusion training (DDPM uses decay 0.9999) and absent
    here. The averaged weights are far less sensitive to which minibatch landed
    last, which is what drives the run-to-run spread: diffusion's seed-to-seed
    variation was 5-10x flow's (+/-0.04-0.06 against +/-0.005-0.008).
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
