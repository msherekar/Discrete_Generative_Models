"""Time, property and position embeddings for the velocity/score trunks.

Split out of nets.py so the trunk file stays under its line limit and so the
conditioning channel -- which Stage 0.2 widens from one bit to a continuous
brightness value -- can be ablated as a unit.
"""
import math

import torch
from torch import nn

# Largest period in the sinusoidal time basis, the DDPM/Transformer default.
MAX_PERIOD = 10_000.0

# Spread of the random Fourier frequencies used for continuous conditioning.
FOURIER_SCALE = 16.0

# Index reserved for the dropped (null) condition under classifier-free
# guidance. Matches the nn.Embedding(3, .) the binary control path uses.
NULL_CLASS = 2

# ══════════════════════════════════════════════════════════════════════════════
# Time
# ══════════════════════════════════════════════════════════════════════════════

def sinusoidal(t, dim, max_period=MAX_PERIOD):
    """Sinusoidal features for a batch of scalar times, shape [n, dim].

    Lecture 3.3's diffusion U-Net and DDPM itself (Ho, Jain & Abbeel, NeurIPS
    2020, sec. 3.2: "the Transformer sinusoidal position embedding") specify
    this basis for the step index, following Vaswani et al. (2017).

    A plain Linear(1 -> d) on raw t -- what nets.py used -- can only produce a
    direction in feature space whose magnitude is affine in t. Two nearby noise
    levels are then nearly identical inputs, so the network has to spend
    capacity separating them before it can condition on them. The sinusoidal
    basis makes well-separated times nearly orthogonal at the input, which is
    the whole reason it is used for positions as well.

    `t` is expected in [0, 1] here, not as a raw step index, so it is scaled by
    1000 to land in the same numeric range the 10,000 max period was chosen for.
    """
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device,
                                             dtype=torch.float32) / half)
    angles = (1000.0 * t.float())[:, None] * frequencies[None, :]
    out = torch.cat([angles.cos(), angles.sin()], dim=1)
    if dim % 2:                                  # odd width: pad the last slot
        out = torch.cat([out, torch.zeros_like(out[:, :1])], dim=1)
    return out


class TimeEmbedding(nn.Module):
    """Sinusoidal time features followed by the usual two-layer MLP."""

    def __init__(self, dim, hidden=None):
        super().__init__()
        self.dim = dim
        hidden = hidden or dim
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, dim))

    def forward(self, t):
        return self.mlp(sinusoidal(t, self.dim))


# ══════════════════════════════════════════════════════════════════════════════
# Conditioning
# ══════════════════════════════════════════════════════════════════════════════

class FourierCondition(nn.Module):
    """Continuous conditioning on standardized brightness, with a null token.

    Stage 0.2's fix. The pipeline conditioned on c = int(score > -1.0) through
    nn.Embedding(3, .), so the field was told only bright-or-not: a one-bit
    channel cannot support a claim about CONTROLLABLE brightness, because there
    is no input at which to request an intermediate value. The only continuous
    handle was the sampling-time reward setpoint, which is guidance rather than
    conditioning and so cannot be measured against it.

    Random Fourier features are used rather than the sinusoidal basis above
    because the conditioning value is a standardized real number with no
    natural period: Tancik et al., "Fourier Features Let Networks Learn High
    Frequency Functions in Low Dimensional Domains" (NeurIPS 2020). The
    frequencies are fixed at init (registered as a buffer, not a parameter) so
    the basis cannot drift and so a checkpoint reloads to the same function.

    Dropped rows get a learned null embedding instead of a value, which is what
    makes the unconditional branch of classifier-free guidance available from
    the same network (Ho & Salimans, arXiv:2207.12598).
    """

    def __init__(self, dim, scale=FOURIER_SCALE):
        super().__init__()
        half = max(1, dim // 2)
        self.register_buffer("frequencies", torch.randn(half) * scale)
        self.null = nn.Parameter(torch.zeros(dim))
        self.mlp = nn.Sequential(nn.Linear(2 * half, dim), nn.SiLU(),
                                 nn.Linear(dim, dim))
        nn.init.normal_(self.null, std=0.02)

    def forward(self, y, drop=None):
        """`y` is [n] standardized brightness; `drop` is a [n] bool mask."""
        angles = 2 * math.pi * y.float()[:, None] * self.frequencies[None, :]
        out = self.mlp(torch.cat([angles.cos(), angles.sin()], dim=1))
        if drop is not None:
            out = torch.where(drop[:, None], self.null[None, :].expand_as(out), out)
        return out


class ClassCondition(nn.Module):
    """The binary-bucket conditioning, kept as the Stage 0.2 ablation control.

    Wrapped in a module with the same call signature as FourierCondition so
    the trunk does not branch on which one it holds, and so `--conditioning
    binary` measures only the channel width and not an incidental code path.
    """

    def __init__(self, dim, classes=3):
        super().__init__()
        self.embedding = nn.Embedding(classes, dim)

    def forward(self, c, drop=None):
        c = c.long()
        if drop is not None:
            c = c.masked_fill(drop, NULL_CLASS)
        return self.embedding(c)


# ══════════════════════════════════════════════════════════════════════════════
# Modulation and position
# ══════════════════════════════════════════════════════════════════════════════

class AdaLNZero(nn.Module):
    """Per-block scale/shift/gate produced from the conditioning vector.

    Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT, ICCV
    2023), sec. 3.2, which finds adaptive LayerNorm with a zero-initialized
    gate to be the best of the four conditioning mechanisms they compare, and
    better than in-context tokens at equal Gflops.

    The trunk previously ADDED time and class as a bias token to every
    position. That competes with the residue content for the same additive
    budget and gives the condition no multiplicative reach, so a block cannot
    change how it processes a residue as a function of the noise level -- only
    what it adds. Modulation gives it that reach.

    Zero-initializing the final projection makes every gate start at zero, so
    the block begins as the identity and the network starts out as the residual
    stream alone. That is the same reasoning as the zero-init output
    projection already in nets.py, applied per block.
    """

    def __init__(self, dim, cond_dim, blocks=2):
        super().__init__()
        # Two modulated sublayers per transformer block (attention, MLP),
        # three parameters each.
        self.project = nn.Linear(cond_dim, 3 * blocks * dim)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.blocks, self.dim = blocks, dim

    def forward(self, cond):
        out = self.project(nn.functional.silu(cond))
        # [n, blocks, 3, dim] -> one (shift, scale, gate) triple per sublayer,
        # each element [n, dim]. The inner unbind is over the triple, not the
        # batch, which is the whole reason the view carries both axes.
        out = out.view(len(cond), self.blocks, 3, self.dim)
        return tuple(block.unbind(1) for block in out.unbind(1))


def modulate(h, shift, scale):
    """Apply a DiT-style (shift, scale) to a normalized [n, L, D] activation.

    scale is an offset from one so that the zero-initialized projection yields
    the identity transform rather than annihilating the activation.
    """
    return h * (1 + scale[:, None, :]) + shift[:, None, :]


def rope_frequencies(length, head_dim, device=None, base=MAX_PERIOD):
    """(cos, sin) tables for rotary position embedding, shape [1, 1, L, head_dim].

    Su et al., RoFormer (arXiv:2104.09864). Replaces the learned absolute
    nn.Parameter(1, length, d_model) in nets.py, which allocates a vector per
    position and therefore hard-codes L=237: a checkpoint trained on GFP could
    not be evaluated on any other length, and the MNIST transfer could not
    share the trunk at all. RoPE is a function of the index, so nothing is
    stored per position and the relative offset between two residues is what
    the attention logit actually sees.
    """
    half = head_dim // 2
    inverse = 1.0 / (base ** (torch.arange(half, device=device,
                                           dtype=torch.float32) / half))
    angles = torch.arange(length, device=device, dtype=torch.float32)[:, None] * inverse
    # Each angle covers a pair of channels, so duplicate to full head width.
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos()[None, None], angles.sin()[None, None]


def apply_rope(x, cos, sin):
    """Rotate [n, heads, L, head_dim] queries or keys by the position angles."""
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


if __name__ == "__main__":
    import torch
    from torch import nn
    B, D = 8, 64

    # sinusoidal: orthogonality at distinct times.
    t = torch.linspace(0, 1, B)
    feats = sinusoidal(t, D)
    assert feats.shape == (B, D), f"sinusoidal shape {feats.shape}"
    assert feats.isfinite().all()
    # nearby times should differ; distant times should be near-orthogonal.
    dot = (feats[0] * feats[-1]).sum() / (feats[0].norm() * feats[-1].norm())
    print(f"  sinusoidal: shape={feats.shape}  t0·t1 cosine={dot:.3f}")

    # TimeEmbedding forward.
    te = TimeEmbedding(D)
    out = te(t)
    assert out.shape == (B, D) and out.isfinite().all()
    print(f"  TimeEmbedding: output shape={out.shape}")

    # FourierCondition forward (continuous) and null token.
    fc = FourierCondition(D)
    y = torch.randn(B)          # shape [B], not [B, 1]
    cond = fc(y)
    assert cond.shape == (B, D) and cond.isfinite().all()
    null_mask = torch.ones(B, dtype=torch.bool)
    null_out = fc(y, null_mask)
    assert null_out.shape == (B, D)
    print(f"  FourierCondition: cond shape={cond.shape}  null OK")

    # ClassCondition forward.
    cc = ClassCondition(D)
    c = torch.randint(0, 2, (B,))
    out_c = cc(c)
    assert out_c.shape == (B, D)
    print(f"  ClassCondition: output shape={out_c.shape}")

    print("embeddings.py OK")
