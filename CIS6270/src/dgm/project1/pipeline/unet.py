"""The MNIST U-Net, as the transfer modality's velocity/score network.

Lecture 2.3's flow U-Net upgraded to Lecture 3.3's diffusion U-Net: group
normalization, a sinusoidal step embedding, and self-attention at the
bottleneck resolution.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .embeddings import TimeEmbedding, sinusoidal

# Groups for GroupNorm. 8 divides every channel count the widths below produce
# and is the value Ho et al. use.
NORM_GROUPS = 8

# Width of the time/condition embedding carried through the residual blocks.
EMBED_DIM = 128

# ══════════════════════════════════════════════════════════════════════════════
# Blocks
# ══════════════════════════════════════════════════════════════════════════════

def conv_block(in_channels, out_channels):
    """Lecture 2.3's plain two-convolution block, kept as the control.

    No normalization and no time input of its own -- time arrives as an extra
    constant image channel. This is what run_mnist.py used, and it is a
    faithful copy of the lecture it came from rather than an oversight, which
    is why it stays available under --unet plain.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.SiLU(),
    )


class ResBlock(nn.Module):
    """GroupNorm residual block with the step embedding added per channel.

    Three changes from conv_block, all from Lecture 3.3's diffusion U-Net:

    GroupNorm rather than nothing. The network's input distribution changes
    completely across the noise schedule -- near-clean images at one end,
    unit Gaussian noise at the other -- so activation statistics drift by
    orders of magnitude over the very range the network has to handle in one
    set of weights. Normalization per group of channels is what keeps the
    effective learning rate comparable across noise levels. GroupNorm and not
    BatchNorm because the batch statistics would mix samples drawn at
    different noise levels, which is precisely the information the model needs
    kept separate.

    The step embedding is projected to a per-channel bias and ADDED inside the
    block, instead of being concatenated once as a constant image channel. A
    constant channel has to survive every downsampling and convolution to
    still be legible at the bottleneck; a per-block bias reaches every depth
    directly.

    A residual connection, so depth does not degrade the signal path.
    """

    def __init__(self, in_channels, out_channels, embed_dim=EMBED_DIM):
        super().__init__()
        groups_in = min(NORM_GROUPS, in_channels)
        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.embed = nn.Linear(embed_dim, out_channels)
        self.norm2 = nn.GroupNorm(min(NORM_GROUPS, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # Zero-init the second convolution so the block starts as the identity,
        # matching the zero-init output projections elsewhere in the pipeline.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x, embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.embed(F.silu(embedding))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Single-head self-attention over spatial positions.

    Lecture 3.3's U-Net places attention at the coarse resolutions. The reason
    it matters for the transfer test specifically: convolution is local, so
    without attention nothing in the network can relate opposite sides of a
    digit, and the guided objective here is mean ink -- a global quantity. A
    purely local network can only raise it by thickening strokes everywhere,
    which is exactly the degenerate saturation the MNIST metrics watch for.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(NORM_GROUPS, channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(n, 3, c, h * w).unbind(1)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return x + self.out(attended.transpose(1, 2).reshape(n, c, h, w))


# ══════════════════════════════════════════════════════════════════════════════
# The network
# ══════════════════════════════════════════════════════════════════════════════

class UNetField(nn.Module):
    """Velocity or score field over 28x28 images.

    `predict` plays the same role as in the protein DiffusionModel: "x0"
    returns the clean-image estimate directly (Lecture 3.2's AMP-Diffusion
    choice and the pipeline default), "v" the velocity target, "eps" the noise.

    `conditioning` mirrors the protein trunk: "continuous" takes the requested
    standardized property, "binary" the class bucket. Keeping the two modalities
    on one conditioning interface is what lets the transfer test change the
    modality and nothing else.
    """

    def __init__(self, base_channels=32, alpha_bars=None, predict="x0",
                 conditioning="binary", attention=True, embed_dim=EMBED_DIM):
        super().__init__()
        c = base_channels
        self.predict, self.conditioning = predict, conditioning
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)
        self.K = None if alpha_bars is None else len(alpha_bars) - 1
        self.time = TimeEmbedding(embed_dim)
        if conditioning == "continuous":
            # Fourier features of the requested value, as in embeddings.py,
            # plus a learned null vector for the dropped (unconditional) rows.
            self.frequencies = nn.Parameter(torch.randn(embed_dim // 2) * 16.0,
                                            requires_grad=False)
            self.cond_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))
            self.null = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.cond_embed = nn.Embedding(3, embed_dim)

        self.stem = nn.Conv2d(1, c, 3, padding=1)
        self.down1 = ResBlock(c, c, embed_dim)
        self.down2 = ResBlock(c, 2 * c, embed_dim)
        self.middle1 = ResBlock(2 * c, 4 * c, embed_dim)
        # 28x28 halves to 14x14 and again to 7x7; attention goes at 7x7, the
        # coarsest resolution, where it is cheapest and its receptive-field
        # argument is strongest.
        self.attention = SelfAttention2d(4 * c) if attention else nn.Identity()
        self.middle2 = ResBlock(4 * c, 4 * c, embed_dim)
        self.up2 = ResBlock(4 * c + 2 * c, 2 * c, embed_dim)
        self.up1 = ResBlock(2 * c + c, c, embed_dim)
        self.norm_out = nn.GroupNorm(min(NORM_GROUPS, c), c)
        self.output = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.pool = nn.AvgPool2d(2)

    def embedding(self, t, c=None, drop=None):
        """Step embedding plus conditioning, the per-block modulation vector.

        AvgPool rather than the MaxPool of Lecture 2.3: max-pooling discards
        the magnitude information that a noise-level-dependent network needs,
        keeping only the extremes, and the extremes of a noisy image are noise.
        """
        embedding = self.time(t)
        if c is None:
            return embedding
        if self.conditioning == "continuous":
            value = c.float()[:, None] if c.dim() == 1 else c.float()[:, :1]
            angles = 2 * torch.pi * value * self.frequencies[None, :]
            cond = self.cond_mlp(torch.cat([angles.cos(), angles.sin()], dim=1))
            if drop is not None:
                cond = torch.where(drop[:, None],
                                   self.null[None, :].expand_as(cond), cond)
        else:
            index = c.long()
            cond = self.cond_embed(index.masked_fill(drop, 2) if drop is not None
                                   else index)
        return embedding + cond

    def trunk(self, xt, t, c=None, drop=None):
        e = self.embedding(t, c, drop)
        skip1 = self.down1(self.stem(xt), e)
        skip2 = self.down2(self.pool(skip1), e)
        h = self.middle1(self.pool(skip2), e)
        h = self.middle2(self.attention(h), e)
        h = F.interpolate(h, size=skip2.shape[-2:], mode="nearest")
        h = self.up2(torch.cat([h, skip2], dim=1), e)
        h = F.interpolate(h, size=skip1.shape[-2:], mode="nearest")
        h = self.up1(torch.cat([h, skip1], dim=1), e)
        return self.output(F.silu(self.norm_out(h)))

    def forward(self, xt, t, c=None, drop=None):
        """The raw prediction, in this model's own parameterization."""
        return self.trunk(xt, t, c, drop)

    def alpha_bar_at(self, t):
        k = (t * self.K).round().long().clamp(0, self.K)
        return self.alpha_bars[k, None, None, None]

    def to_noise(self, prediction, z, t):
        """Convert any parameterization to eps, as DiffusionModel does."""
        from .losses import eps_from_velocity
        if self.predict == "eps" or self.alpha_bars is None:
            return prediction
        a = self.alpha_bar_at(t)
        if self.predict == "v":
            return eps_from_velocity(z, prediction, a)
        return (z - a.sqrt() * prediction) / (1 - a).sqrt().clamp_min(1e-8)


class ImageReward(nn.Module):
    """Time-conditioned predictor of the standardized image properties."""

    def __init__(self, base_channels=32, n_props=2):
        super().__init__()
        c = base_channels
        self.n_props = n_props
        self.net = nn.Sequential(
            conv_block(2, c), nn.AvgPool2d(2),
            conv_block(c, 2 * c), nn.AvgPool2d(2),
            conv_block(2 * c, 2 * c), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2 * c, n_props),
        )

    def forward(self, x, t):
        # Time as a constant channel is adequate here: the reward head is a
        # regressor, not a generator, and its own ablation is not in scope.
        time = t[:, None, None, None].expand_as(x)
        return self.net(torch.cat([x, time], dim=1))


def add_arguments(parser):
    """Stage 1 flag for the transfer modality's trunk."""
    group = parser.add_argument_group("MNIST architecture")
    group.add_argument("--unet", default="modern", choices=("modern", "plain"),
                       help="Image trunk (default: modern). 'plain' is "
                            "Lecture 2.3's flow U-Net exactly: MaxPool, no "
                            "normalization, time as a concatenated constant "
                            "channel. 'modern' is Lecture 3.3's diffusion "
                            "U-Net: GroupNorm, sinusoidal step embedding added "
                            "per residual block, and 7x7 self-attention. "
                            "Without the upgrade the transfer modality is too "
                            "weak to show an effect either way.")
    group.add_argument("--no-attention", dest="attention",
                       action="store_false",
                       help="Drop the bottleneck self-attention, isolating it "
                            "from the rest of the 2.3-to-3.3 upgrade. Mean ink "
                            "is a global objective, so a purely local network "
                            "can only raise it by thickening strokes "
                            "everywhere -- the degenerate mode the saturation "
                            "check watches for.")
    parser.set_defaults(attention=True)
    return parser
