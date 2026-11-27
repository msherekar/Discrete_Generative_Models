"""
Track A: Multi-Step Discrete Masked Diffusion / Flow Matching Sampler.

Supports both pretrained HuggingFace protein diffusion/masked models (DPLM-150M,
DPLM-650M, ESM-2) mapped onto our 21-token vocabulary (20 AAs + <PAD>) and a
self-contained LocalMaskedIDPDenoiser trainable directly on pan-organism MobiDB IDPs.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgm.project_final.decoding.coupled_finalizer import coupled_decode_single
from dgm.project_final.config import SamplerConfig
from dgm.project_final.guidance.pairwise_hypercone import (
    apply_doob_rate_tilt,
    compute_exact_and_tag_gains,
    compute_hypercone_log_tilt,
)
from dgm.project_final.physics.descriptors import AA20, PAD_ID, VOCAB_SIZE, BiophysicalEngine

MASK_ID: int = 21  # Input-only absorbing [MASK] state (indices 0..19 = AAs, 20 = <PAD>)


class SinusoidalProgressEmbed(nn.Module):
    """Embed continuous % unmasked alpha in [0, 1] via sinusoidal features."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=alpha.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = alpha.float().unsqueeze(-1) * freqs.unsqueeze(0) * 100.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class LocalMaskedIDPDenoiser(nn.Module):
    """
    Bidirectional Transformer MDLM/DFM denoiser taking tokens in {0..20, MASK=21}
    and % unmasked alpha in [0, 1], outputting clean logits over 21 tokens [B, L, 21].
    """

    def __init__(
        self, L_max: int = 128, d_model: int = 256, n_heads: int = 8, n_layers: int = 6
    ) -> None:
        super().__init__()
        self.L_max = L_max
        self.tok_emb = nn.Embedding(VOCAB_SIZE + 1, d_model)
        self.pos_emb = nn.Embedding(L_max, d_model)
        self.prog_emb = SinusoidalProgressEmbed(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.05, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, VOCAB_SIZE))

    def forward(self, x_tokens: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        B, L = x_tokens.shape
        pos = torch.arange(L, device=x_tokens.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x_tokens) + self.pos_emb(pos) + self.prog_emb(alpha).unsqueeze(1)
        return self.head(self.encoder(h))


def mdlm_train_step(
    denoiser: nn.Module, clean_tokens: torch.Tensor, optimizer: torch.optim.Optimizer
) -> float:
    """
    Single MDLM / DFM weighted cross-entropy step over pan-organism IDPs (including <PAD>).
    Samples progress alpha ~ U(0.02, 0.98) and weights masked loss by 1 / (1 - alpha).
    """
    denoiser.train()
    B, L = clean_tokens.shape
    device = clean_tokens.device
    # Low-discrepancy stratified progress sampling across batch
    u = (torch.arange(B, device=device, dtype=torch.float32) + torch.rand(B, device=device)) / float(B)
    alpha = 0.02 + 0.96 * u  # % unmasked in [0.02, 0.98]
    keep_mask = torch.rand((B, L), device=device) < alpha.unsqueeze(1)
    x_noisy = torch.where(keep_mask, clean_tokens, torch.full_like(clean_tokens, MASK_ID))

    logits = denoiser(x_noisy, alpha)  # [B, L, 21]
    masked_pos = ~keep_mask
    per_tok_ce = F.cross_entropy(logits.transpose(1, 2), clean_tokens, reduction="none")
    weights = (1.0 / (1.0 - alpha).clamp(min=0.05)).unsqueeze(1)
    loss = (weights * per_tok_ce * masked_pos.float()).sum() / masked_pos.float().sum().clamp(min=1.0)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


class HuggingFaceDPLMWrapper(nn.Module):
    """
    Wraps HuggingFace `airkingbd/dplm_150m`, `airkingbd/dplm_650m`, or `facebook/esm2_*`
    masked protein language models, mapping our 22-state tokens into HF IDs and projecting
    output logits back onto our canonical 21-token vocabulary (ACDEFGHIKLMNPQRSTVWY + <PAD>).
    """

    def __init__(self, model_name_or_path: str = "airkingbd/dplm_150m") -> None:
        super().__init__()
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.backbone = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        vocab_map = self.tokenizer.get_vocab()
        pad_hf = self.tokenizer.pad_token_id or vocab_map.get("<pad>", 1)
        mask_hf = self.tokenizer.mask_token_id or vocab_map.get("<mask>", 32)

        aa_hf_ids = [vocab_map[ch] for ch in AA20]
        # in_map: maps our {0..19, PAD=20, MASK=21} -> HF input_ids
        self.register_buffer("in_map", torch.tensor(aa_hf_ids + [pad_hf, mask_hf], dtype=torch.long))
        # out_map: extracts our 21 clean tokens {0..19, PAD=20} from HF vocabulary logits
        self.register_buffer("out_map", torch.tensor(aa_hf_ids + [pad_hf], dtype=torch.long))

    def forward(self, x_tokens: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        hf_ids = self.in_map[x_tokens]
        attn_mask = (x_tokens != PAD_ID).long()
        # Keep attention active on at least first token to avoid NaN softmax on all-PAD rows
        attn_mask[:, 0] = 1
        outputs = self.backbone(input_ids=hf_ids, attention_mask=attn_mask)
        logits_21 = outputs.logits[:, :, self.out_map].float()
        return logits_21


@torch.no_grad()
def sample_track_a(
    denoiser: nn.Module,
    engine: BiophysicalEngine,
    cfg: SamplerConfig,
    batch_size: int = 32,
    mode: str = "1A",
    use_part_1b: bool = False,
    run_swaps_1b: bool = True,
    fixed_length: Optional[int] = None,
    schedule: str = "linear",
    record_tag_diag: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Generate contiguous variable-length responsive IDPs from 0% -> 100% unmasked.
    Preserves the marginal unmasking rate da / (1 - a) while tilting destination
    amino acid probabilities via Doob's h-transform (preventing Step-0 overshoot).
    """
    denoiser.eval()
    device = engine.device
    L_max = engine.L_max
    num_steps = max(int(cfg.num_steps), 1)

    # Sample contiguous active chain length L_b in [L_min, L_max] per sequence
    if fixed_length is not None:
        lengths = torch.full((batch_size,), fixed_length, device=device, dtype=torch.long)
    else:
        lengths = torch.randint(
            engine.task.L_min, engine.task.L_max + 1, (batch_size,), device=device
        )

    pos_idx = torch.arange(L_max, device=device).unsqueeze(0)
    active_chain = pos_idx < lengths.unsqueeze(1)
    x = torch.where(
        active_chain,
        torch.full((batch_size, L_max), MASK_ID, device=device, dtype=torch.long),
        torch.full((batch_size, L_max), PAD_ID, device=device, dtype=torch.long),
    )

    u_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    alphas = u_grid.pow(2) if schedule == "quadratic" else u_grid

    diag_true_dmu: List[torch.Tensor] = []
    diag_tag_g: List[torch.Tensor] = []
    diag_exact_g: List[torch.Tensor] = []

    for step in range(num_steps):
        a_cur = alphas[step].expand(batch_size)
        a_scalar = float(alphas[step].item())
        da = float((alphas[step + 1] - alphas[step]).item())

        logits = denoiser(x, a_cur) / max(cfg.temperature, 1e-3)
        # Enforce contiguous active prefix (no mid-chain PAD) and PAD suffix
        logits[:, :, PAD_ID] = torch.where(active_chain, torch.full_like(logits[:, :, 0], -1e9), 10.0)
        logits[:, :, :20] = torch.where(active_chain.unsqueeze(-1), logits[:, :, :20], -1e9)

        P = torch.softmax(logits, dim=-1)
        is_masked = (x == MASK_ID) & active_chain
        clamped_oh = F.one_hot(x.clamp(max=PAD_ID), num_classes=VOCAB_SIZE).to(P.dtype)
        P_eff = torch.where(is_masked.unsqueeze(-1), P, clamped_oh)

        if mode in ("unguided", "1B_only"):
            dest_probs = P
        else:
            # Relax cone slightly early on when all objectives improve broadly, tighten later
            eff_cone = max(engine.task.cone_angle_deg, 55.0 - 25.0 * a_scalar)
            log_tilt, _ = compute_hypercone_log_tilt(
                P_eff, is_masked, engine,
                cone_angle_deg=eff_cone if mode == "1A" else engine.task.cone_angle_deg,
                gamma=engine.task.gamma,
                mode=mode,
            )
            tilted = apply_doob_rate_tilt(P, log_tilt)
            tilted[:, :, PAD_ID] = torch.where(active_chain, 0.0, 1.0)
            dest_probs = tilted / (tilted.sum(dim=-1, keepdim=True) + 1e-12)

        if record_tag_diag and step == num_steps // 2:
            g_ex, g_tg, dmu = compute_exact_and_tag_gains(P_eff, engine)
            diag_true_dmu.append(dmu.detach().cpu().flatten()[:512])
            diag_tag_g.append(g_tg.detach().cpu().flatten()[:512])
            diag_exact_g.append(g_ex.detach().cpu().flatten()[:512])

        # Exact schedule-matched conditional unmasking probability da / (1 - alpha)
        p_unmask = min(1.0, (da * (1.0 + cfg.remask_eta * a_scalar)) / max(1.0 - a_scalar, 1e-4))
        jump_now = is_masked & (torch.rand((batch_size, L_max), device=device) < p_unmask)
        if step == num_steps - 1:
            jump_now = is_masked

        safe_probs = dest_probs.clamp(min=1e-12)
        safe_probs = safe_probs / safe_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(safe_probs.view(-1, VOCAB_SIZE), 1).view(batch_size, L_max)
        x = torch.where(jump_now, sampled, x)

        if cfg.remask_eta > 0.0 and step < num_steps - 1 and a_scalar > 0.05:
            p_remask = 1.0 - math.exp(-da * cfg.remask_eta)
            remask_now = (~is_masked) & active_chain & (torch.rand((batch_size, L_max), device=device) < p_remask)
            x = torch.where(remask_now, torch.full_like(x, MASK_ID), x)

    final_logits = denoiser(x, torch.ones(batch_size, device=device)) / max(cfg.temperature, 1e-3)
    P_final = torch.softmax(final_logits, dim=-1)
    realized_oh = F.one_hot(x.clamp(max=PAD_ID), num_classes=VOCAB_SIZE).to(P_final.dtype)
    P_handoff = 0.65 * P_final + 0.35 * realized_oh

    if use_part_1b:
        decoded_rows = []
        for b in range(batch_size):
            fix_L_b = fixed_length if fixed_length is not None else None
            seq_b, _, _ = coupled_decode_single(
                P_handoff[b], engine, cfg, run_swaps=run_swaps_1b, fixed_length=fix_L_b
            )
            decoded_rows.append(seq_b)
        x = torch.stack(decoded_rows, dim=0)

    diag_out: Dict[str, torch.Tensor] = {}
    if diag_exact_g:
        diag_out["dmu"] = torch.cat(diag_true_dmu)
        diag_out["tag_gain"] = torch.cat(diag_tag_g)
        diag_out["exact_gain"] = torch.cat(diag_exact_g)
    return x, P_handoff, diag_out