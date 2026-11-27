"""
Track B: Lagrangian Self-Distillation (LSD) Simplex Flow Map for 1-to-4-Step IDP Generation.

Uses explicit math self-attention (avoiding fused FlashAttention kernels that lack
forward-mode AD support) so `torch.func.jvp` computes exact output derivatives
d/dt X_{s,t}(x_s) natively on NVIDIA Blackwell (GB10) GPUs.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgm.project_final.config import SamplerConfig
from dgm.project_final.decoding.coupled_finalizer import coupled_decode_single
from dgm.project_final.models.dplm_sampler import SinusoidalProgressEmbed
from dgm.project_final.physics.descriptors import PAD_ID, VOCAB_SIZE, BiophysicalEngine


class JVPCompatibleBlock(nn.Module):
    """
    Pre-Norm Transformer block using explicit matrix-multiply attention and zero
    RNG dropout so `torch.func.jvp` differentiates smoothly wrt target progress `t`.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ada_cond = nn.Linear(d_model, 2 * d_model)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, L, D = h.shape
        shift, gate = self.ada_cond(cond).unsqueeze(1).chunk(2, dim=-1)

        x = self.norm1(h) * (1.0 + 0.1 * gate) + 0.1 * shift
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, D)
        h = h + self.proj(out)
        h = h + self.mlp(self.norm2(h))
        return h


class LSDSimplexFlowMap(nn.Module):
    """
    Continuous-state Flow Map X_{s,t}(x) = x + (t - s) * F_theta(x, s, t)
    operating on 21-dimensional token simplex embeddings [B, L_max, 21].
    Satisfies the Identity Axiom X_{t,t}(x) = x identically for all theta.
    """

    def __init__(
        self, L_max: int = 128, d_model: int = 384, n_heads: int = 6, n_layers: int = 8
    ) -> None:
        super().__init__()
        self.L_max = L_max
        self.in_proj = nn.Linear(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, L_max, d_model) * 0.02)
        self.s_emb = SinusoidalProgressEmbed(d_model)
        self.t_emb = SinusoidalProgressEmbed(d_model)
        self.cond_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([JVPCompatibleBlock(d_model, n_heads) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, VOCAB_SIZE)

    def velocity_net(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict average interval velocity F_theta(x, s, t) of shape [B, L_max, 21]."""
        B, L, _ = x.shape
        cond = self.cond_mlp(torch.cat([self.s_emb(s), self.t_emb(t)], dim=-1))
        h = self.in_proj(x) + self.pos_emb[:, :L, :]
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_proj(self.out_norm(h))

    def forward(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate shortcut map X_{s -> t}(x) from start % s to target % t."""
        dt = (t - s).view(-1, 1, 1)
        return x + dt * self.velocity_net(x, s, t)


def make_ema_copy(model: LSDSimplexFlowMap) -> LSDSimplexFlowMap:
    """Create a parameter-frozen exponential moving average shadow network."""
    ema = copy.deepcopy(model)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema: LSDSimplexFlowMap, model: LSDSimplexFlowMap, decay: float = 0.999) -> None:
    """Polyak EMA update for stable self-distillation velocity targets."""
    for p_ema, p_mod in zip(ema.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p_mod.data, alpha=1.0 - decay)


def lsd_train_step(
    model: LSDSimplexFlowMap,
    ema_model: LSDSimplexFlowMap,
    clean_tokens: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    lsd_weight: float = 0.5,
    ema_decay: float = 0.999,
) -> Dict[str, float]:
    """
    Single Lagrangian Self-Distillation (LSD) optimization step:
      - Half batch (s == t): Diagonal Flow Matching anchor L_b = ||F_theta(x_s, s, s) - (x_1 - x_0)||^2.
      - Half batch (s < t):  Off-diagonal LSD residual L_D = ||d/dt X_{s,t}(x_s) - F_ema(X_{s,t}(x_s), t, t)||^2
                             using exact product-rule JVP: dX/dt = F_theta + (t - s) * sg(dF_theta/dt).
    """
    model.train()
    B, L = clean_tokens.shape
    device = clean_tokens.device

    x_1 = 2.0 * F.one_hot(clean_tokens, num_classes=VOCAB_SIZE).float() - (1.0 / VOCAB_SIZE)
    x_0 = torch.randn_like(x_1)

    n_diag = max(B // 2, 1)
    s_diag = torch.rand(n_diag, device=device)
    x_s_diag = (1.0 - s_diag).view(-1, 1, 1) * x_0[:n_diag] + s_diag.view(-1, 1, 1) * x_1[:n_diag]
    v_pred = model.velocity_net(x_s_diag, s_diag, s_diag)
    loss_anchor = F.mse_loss(v_pred, x_1[:n_diag] - x_0[:n_diag])

    n_off = B - n_diag
    if n_off > 0:
        u1 = torch.rand(n_off, device=device)
        u2 = torch.rand(n_off, device=device)
        s_off = torch.minimum(u1, u2).clamp(max=0.98)
        t_off = torch.maximum(torch.maximum(u1, u2).clamp(max=1.0), s_off + 1e-2)
        x_s_off = (1.0 - s_off).view(-1, 1, 1) * x_0[n_diag:] + s_off.view(-1, 1, 1) * x_1[n_diag:]

        dt_off = (t_off - s_off).view(-1, 1, 1)
        F_val = model.velocity_net(x_s_off, s_off, t_off)
        X_st = x_s_off + dt_off * F_val

        with torch.no_grad():
            _, dF_dt = torch.func.jvp(
                lambda t_var: model.velocity_net(x_s_off, s_off, t_var),
                (t_off,),
                (torch.ones_like(t_off),),
            )
            v_target = ema_model.velocity_net(X_st.detach(), t_off, t_off)

        dX_dt = F_val + dt_off * dF_dt
        loss_lsd = F.mse_loss(dX_dt, v_target)
    else:
        loss_lsd = torch.zeros_like(loss_anchor)

    total_loss = loss_anchor + lsd_weight * loss_lsd
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
    optimizer.step()
    update_ema(ema_model, model, decay=ema_decay)

    return {
        "loss_total": float(total_loss.item()),
        "loss_anchor": float(loss_anchor.item()),
        "loss_lsd": float(loss_lsd.item()),
        "grad_norm": grad_norm,
    }


@torch.no_grad()
def sample_flow_map(
    flow_map: LSDSimplexFlowMap,
    batch_size: int,
    nfe: int = 1,
    temperature: float = 0.25,
    x_0: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Jump from 0% -> 100% unmasked in `nfe` Flow Map evaluations (e.g., 1, 2, 4, 16, 64),
    returning terminal soft simplex probabilities P_final of shape [B, L_max, 21].
    """
    flow_map.eval()
    dev = device or next(flow_map.parameters()).device
    x = x_0.clone() if x_0 is not None else torch.randn((batch_size, flow_map.L_max, VOCAB_SIZE), device=dev)
    grid = torch.linspace(0.0, 1.0, max(nfe, 1) + 1, device=dev)
    for i in range(max(nfe, 1)):
        s_vec = grid[i].expand(x.size(0))
        t_vec = grid[i + 1].expand(x.size(0))
        x = flow_map(x, s_vec, t_vec)
    return torch.softmax(x / max(temperature, 1e-3), dim=-1)


@torch.no_grad()
def decode_flow_map_batch(
    P_final: torch.Tensor,
    engine: BiophysicalEngine,
    cfg: SamplerConfig,
    decoder_mode: str = "1B_full",
    fixed_length: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Decode terminal Flow Map probabilities P_final [B, L_max, 21] using:
      - "argmax":         Naive independent per-position argmax
      - "temp_sample":    Independent categorical sampling
      - "1B_stage2_only": Part 1B Sinkhorn + A* beam without Stage-3 pairwise swaps
      - "1B_full":        Full Part 1B (Sinkhorn + beam + composition-preserving swaps)
    """
    B, L_max, _ = P_final.shape
    out: List[torch.Tensor] = []
    for b in range(B):
        Pb = P_final[b]
        if decoder_mode in ("argmax", "temp_sample"):
            toks = Pb.argmax(dim=-1) if decoder_mode == "argmax" else torch.multinomial(Pb, 1).squeeze(-1)
            if fixed_length is not None:
                aa_only = Pb[:fixed_length, :20].argmax(dim=-1)
                row = torch.full((L_max,), PAD_ID, device=Pb.device, dtype=torch.long)
                row[:fixed_length] = aa_only
                out.append(row)
            else:
                aa_fallback = Pb[:, :20].argmax(dim=-1)
                pad_hits = torch.nonzero(toks[engine.task.L_min :] == PAD_ID, as_tuple=False)
                L_cut = (engine.task.L_min + int(pad_hits[0].item())) if pad_hits.numel() > 0 else engine.task.L_max
                row = torch.full((L_max,), PAD_ID, device=Pb.device, dtype=torch.long)
                row[:L_cut] = torch.where(toks[:L_cut] == PAD_ID, aa_fallback[:L_cut], toks[:L_cut])
                out.append(row)
        else:
            run_swaps = decoder_mode == "1B_full"
            seq_b, _, _ = coupled_decode_single(Pb, engine, cfg, run_swaps=run_swaps, fixed_length=fixed_length)
            out.append(seq_b)
    return out


@torch.no_grad()
def invert_and_redesign_disprot(
    flow_map: LSDSimplexFlowMap,
    natural_tokens: torch.Tensor,
    engine: BiophysicalEngine,
    cfg: SamplerConfig,
    nfe: int = 4,
    noise_scale: float = 0.15,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Reverse Flow Map (100% -> 0% unmasked) on natural DisProt IDPs to recover their
    latent noise blueprint x_0, perturb slightly, and map forward (0% -> 100%) into Part 1B.
    """
    flow_map.eval()
    dev = next(flow_map.parameters()).device
    x = 2.0 * F.one_hot(natural_tokens.to(dev), num_classes=VOCAB_SIZE).float() - (1.0 / VOCAB_SIZE)
    rev_grid = torch.linspace(1.0, 0.0, max(nfe, 1) + 1, device=dev)
    for i in range(max(nfe, 1)):
        x = flow_map(x, rev_grid[i].expand(x.size(0)), rev_grid[i + 1].expand(x.size(0)))
    x_0_edited = x + noise_scale * torch.randn_like(x)
    P_redesign = sample_flow_map(flow_map, x_0_edited.size(0), nfe=nfe, x_0=x_0_edited, device=dev)
    redesigned = decode_flow_map_batch(P_redesign, engine, cfg, decoder_mode="1B_full")
    return redesigned, x_0_edited