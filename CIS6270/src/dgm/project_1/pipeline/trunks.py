"""The per-residue Transformer trunk shared by the velocity, score and reward heads.

Stage 1 of the study: the Lecture 3.3 trunk upgraded to DiT-style AdaLN-Zero
conditioning and rotary positions, with the original additive-token form kept
as the ablation control.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .embeddings import (AdaLNZero, ClassCondition, FourierCondition,
                         TimeEmbedding, apply_rope, modulate, rope_frequencies)

# Conditioning channels the trunk accepts: a continuous property value, the
# one-bit bucket the pipeline used before Stage 0.2, or nothing at all.
CONDITIONINGS = ("continuous", "binary", "none")

# How the condition reaches the residue stream.
MODULATIONS = ("adaln", "token")

# ══════════════════════════════════════════════════════════════════════════════
# Blocks
# ══════════════════════════════════════════════════════════════════════════════

class Attention(nn.Module):
    """Multi-head self-attention with optional rotary positions.

    Written out rather than using nn.TransformerEncoderLayer because RoPE has
    to be applied to the queries and keys after projection, which that module
    does not expose. scaled_dot_product_attention still dispatches to the same
    fused kernels.
    """

    def __init__(self, d_model, heads, dropout=0.0):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, h, cos=None, sin=None):
        n, length, d_model = h.shape
        head_dim = d_model // self.heads
        q, k, v = (self.qkv(h)
                   .view(n, length, 3, self.heads, head_dim)
                   .permute(2, 0, 3, 1, 4))
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(attended.transpose(1, 2).reshape(n, length, d_model))


class DiTBlock(nn.Module):
    """Pre-norm attention + MLP, each modulated and gated by the condition.

    Peebles & Xie (DiT, ICCV 2023), fig. 3 right. The two LayerNorms carry no
    learnable affine of their own (elementwise_affine=False) because the
    modulation supplies the scale and shift; leaving both in place gives the
    network two redundant ways to express the same transform and makes the
    zero-init gate no longer a true identity at step zero.
    """

    def __init__(self, d_model, heads, cond_dim, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(d_model, heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model), nn.Dropout(dropout))
        self.modulation = AdaLNZero(d_model, cond_dim, blocks=2)

    def forward(self, h, cond, cos=None, sin=None):
        attention_mod, mlp_mod = self.modulation(cond)
        shift, scale, gate = attention_mod
        h = h + gate[:, None, :] * self.attention(
            modulate(self.norm1(h), shift, scale), cos, sin)
        shift, scale, gate = mlp_mod
        h = h + gate[:, None, :] * self.mlp(
            modulate(self.norm2(h), shift, scale))
        return h


class TokenBlock(nn.Module):
    """The pre-Stage-1 block: standard pre-norm encoder layer, no modulation.

    Retained so `--modulation token` reproduces the original trunk exactly,
    which is what lets the AdaLN-Zero result be attributed to the conditioning
    mechanism rather than to any of the other changes landing alongside it.
    """

    def __init__(self, d_model, heads, cond_dim, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=mlp_ratio * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)

    def forward(self, h, cond, cos=None, sin=None):
        return self.layer(h)


# ══════════════════════════════════════════════════════════════════════════════
# Trunk
# ══════════════════════════════════════════════════════════════════════════════

def add_arguments(parser):
    """Stage 1 flags: the conditioning channel and how it reaches the trunk."""
    group = parser.add_argument_group("architecture (Stage 1)")
    group.add_argument("--conditioning", default="binary",
                       choices=CONDITIONINGS,
                       help="What the field is told about the target property "
                            "(default: binary, the pre-Stage-0.2 control). "
                            "'continuous' passes the standardized value "
                            "itself; the binary bucket c = int(score > -1.0) "
                            "is a one-bit channel, and a claim about "
                            "CONTROLLABLE brightness cannot be made through "
                            "it because there is no input at which to request "
                            "an intermediate value.")
    group.add_argument("--n-cond", type=int, default=1, metavar="N",
                       help="How many property columns to condition on under "
                            "--conditioning continuous (default: 1, "
                            "brightness only). Capped at the number the "
                            "dataset carries.")
    group.add_argument("--modulation", default="adaln", choices=MODULATIONS,
                       help="How the condition reaches the residue stream "
                            "(default: adaln). 'token' adds it as a bias to "
                            "every position, the pre-Stage-1 behaviour, which "
                            "competes with residue content for the same "
                            "additive budget and gives the condition no "
                            "multiplicative reach. 'adaln' is DiT's "
                            "zero-gated adaptive LayerNorm (Peebles & Xie, "
                            "ICCV 2023).")
    group.add_argument("--no-rope", dest="rope", action="store_false",
                       help="Use a learned absolute position embedding instead "
                            "of RoPE. The learned table allocates a vector per "
                            "position and so hard-codes L=237, which is why a "
                            "GFP checkpoint could not transfer to MNIST at "
                            "all (Su et al., arXiv:2104.09864).")
    parser.set_defaults(rope=True)
    return parser


class TransformerField(nn.Module):
    """Maps a noisy latent [n, L, D] and a time to a field of the same shape.

    Lecture 3.3's recipe is the starting point -- "We will stack six
    Transformer layers... Add the noisy residue embeddings, the diffusion-step
    embedding shared across positions, and a learned embedding for each token
    position" -- and three things change from it, each independently switchable:

      sinusoidal time    was Linear(1 -> d_model) on raw t. See embeddings.py.
      rotary positions   was nn.Parameter(1, L, d_model), which hard-codes
                         L=237 and cannot transfer across modalities.
      AdaLN-Zero         was an added bias token, which has no multiplicative
                         reach into the residue stream.

    Weights are shared across positions, so size does not scale with L: ~5M
    parameters at d_model=256 against 235M for the flattened MLP at width 1024,
    of which 94.7% of the latent variance it fits is merely positional.
    """

    def __init__(self, length, dim, d_model=256, layers=6, heads=8, dropout=0.0,
                 conditioning="continuous", modulation="adaln", rope=True,
                 n_cond=1, zero_init_out=True):
        super().__init__()
        self.length, self.dim, self.d_model, self.heads = length, dim, d_model, heads
        self.conditioning, self.use_rope = conditioning, rope
        self.project_in = nn.Linear(dim, d_model)
        self.time = TimeEmbedding(d_model)
        if conditioning == "continuous":
            # One Fourier channel per conditioned property, summed. Brightness
            # is column 0; the Rosetta attribute columns come along when the
            # objective asks for them.
            self.condition = nn.ModuleList(
                [FourierCondition(d_model) for _ in range(n_cond)])
        elif conditioning == "binary":
            self.condition = ClassCondition(d_model)
        else:
            self.condition = None
        # Absolute positions are still needed when RoPE is switched off.
        self.position = None if rope else nn.Parameter(
            torch.zeros(1, length, d_model))
        if self.position is not None:
            nn.init.normal_(self.position, std=0.02)
        block = DiTBlock if modulation == "adaln" else TokenBlock
        self.blocks = nn.ModuleList(
            [block(d_model, heads, d_model, dropout=dropout) for _ in range(layers)])
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final = AdaLNZero(d_model, d_model, blocks=1)
        self.project_out = nn.Linear(d_model, dim)
        if zero_init_out:
            # Start as the zero field, so training begins from the identity
            # flow rather than from a random velocity. Right for a generative
            # head and WRONG for the reward regressor: a zero output projection
            # makes the output constant in z, so d(reward)/dz is identically
            # zero and guidance has no gradient to follow at all. Measured on a
            # fresh transformer RewardModel the input-gradient norm was exactly
            # 0.000e+00 against 9.8e-02 for the MLP, which is a candidate cause
            # of the 2026-10-03 run's indistinguishable guidance arms -- that
            # run used --arch transformer. See nets.RewardModel.
            nn.init.zeros_(self.project_out.weight)
            nn.init.zeros_(self.project_out.bias)
        self._rope_cache = {}

    def _rope(self, length, device, dtype):
        """Cache the rotary tables per (length, device); they never change."""
        if not self.use_rope:
            return None, None
        key = (length, str(device))
        if key not in self._rope_cache:
            cos, sin = rope_frequencies(length, self.d_model // self.heads, device)
            self._rope_cache[key] = (cos, sin)
        cos, sin = self._rope_cache[key]
        return cos.to(dtype), sin.to(dtype)

    def conditioning_vector(self, t, c=None, drop=None):
        """Time plus condition, the single vector every block is modulated by."""
        cond = self.time(t)
        if self.condition is None or c is None:
            return cond
        if self.conditioning == "continuous":
            # c is [n] or [n, n_cond] of standardized property values.
            values = c[:, None] if c.dim() == 1 else c
            for index, embed in enumerate(self.condition):
                cond = cond + embed(values[:, index], drop)
        else:
            cond = cond + self.condition(c, drop)
        return cond

    def forward(self, z, t, c=None, drop=None):
        h = self.project_in(z)
        if self.position is not None:
            h = h + self.position
        cond = self.conditioning_vector(t, c, drop)
        cos, sin = self._rope(h.shape[1], h.device, h.dtype)
        for block in self.blocks:
            h = block(h, cond, cos, sin)
        (shift, scale, _), = self.final(cond)
        return self.project_out(modulate(self.norm_out(h), shift, scale))


if __name__ == "__main__":
    import torch
    B, L, D, H = 4, 16, 32, 64
    z = torch.randn(B, L, D)
    t = torch.rand(B)
    c = torch.randint(0, 2, (B,))

    for cond in CONDITIONINGS:
        cond_val = c if cond == "binary" else (torch.randn(B, 1) if cond == "continuous" else None)
        for mod in MODULATIONS:
            trunk = TransformerField(L, D, H, conditioning=cond, modulation=mod,
                                     rope=True, n_cond=1)
            out = trunk(z, t, cond_val)
            assert out.shape == (B, L, D) and out.isfinite().all(), \
                f"{cond}/{mod}: shape {out.shape} or NaN"
            print(f"  TransformerField({cond}/{mod}): shape={out.shape}")

    print("trunks.py OK")
