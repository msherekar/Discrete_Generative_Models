"""
Part 1A: Exact Pairwise Posterior-Expectation + Hypercone Multi-Objective Guidance.

Computes exact analytical posterior expectations E_{x_1 ~ P}[R_k(x_1)] and exact
single-site transition gains across all B x L_max x 21 candidate jumps in O(L_max^2)
matrix operations, eliminating first-order Taylor-Approximated Guidance (TAG) error.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from dgm.project_final.physics.descriptors import BiophysicalEngine


def compute_exact_and_tag_gains(
    P: torch.Tensor,
    engine: BiophysicalEngine,
    one_body_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute exact squared-residual gains and first-order TAG baseline gains.

    Returns:
        gains_exact: [B, K, L_max, 21] exact -Delta(0.5 * mu_k^2)
        gains_tag:   [B, K, L_max, 21] 1st-order TAG approximation (-mu_k * dmu_k)
        dmu_all:     [B, K, L_max, 21] exact single-site residual shift Delta mu_k(i, v)
    """
    if P.dim() == 2:
        P = P.unsqueeze(0)

    exact_list, tag_list, dmu_list = [], [], []
    for obj in engine.objectives:
        a_bar = torch.matmul(P, obj.a_tilde)

        if one_body_only or obj.kind == "1body":
            mu = a_bar.sum(dim=-1)
            dmu = obj.a_tilde.view(1, 1, -1) - a_bar.unsqueeze(-1)
        else:
            phi_bar = torch.matmul(P, obj.phi)
            s_field = torch.matmul(phi_bar, obj.W.T)
            mu = a_bar.sum(dim=-1) + 0.5 * (phi_bar * s_field).sum(dim=-1)
            dmu = (obj.a_tilde.view(1, 1, -1) - a_bar.unsqueeze(-1)) + (
                obj.phi.view(1, 1, -1) - phi_bar.unsqueeze(-1)
            ) * s_field.unsqueeze(-1)

        mu_exp = mu.view(-1, 1, 1)
        g_tag = -mu_exp * dmu
        g_exact = g_tag - 0.5 * (dmu ** 2)

        exact_list.append(g_exact)
        tag_list.append(g_tag)
        dmu_list.append(dmu)

    return (
        torch.stack(exact_list, dim=1),
        torch.stack(tag_list, dim=1),
        torch.stack(dmu_list, dim=1),
    )


def positive_percentile_ranks(values_1d: torch.Tensor) -> torch.Tensor:
    """
    Map strictly improving moves (gain > 0) to percentile ranks in (0, 1] while
    assigning 0.0 to non-improving / overshooting moves (gain <= 0).
    """
    pos_mask = values_1d > 1e-8
    out = torch.zeros_like(values_1d)
    n_pos = int(pos_mask.sum().item())
    if n_pos == 0:
        return out
    if n_pos == 1:
        out[pos_mask] = 1.0
        return out
    pos_vals = values_1d[pos_mask]
    lt = (pos_vals.unsqueeze(0) < pos_vals.unsqueeze(1)).to(values_1d.dtype).sum(dim=1)
    le = (pos_vals.unsqueeze(0) <= pos_vals.unsqueeze(1)).to(values_1d.dtype).sum(dim=1)
    # Ranks in (0, 1] scaled smoothly by tanh of gain magnitude to prevent near-zero chatter
    raw_rank = 0.5 * (lt + le) / float(n_pos)
    mag_gate = torch.tanh(pos_vals * 2.0)
    out[pos_mask] = raw_rank * mag_gate
    return out


def compute_hypercone_log_tilt(
    P: torch.Tensor,
    active_slot_mask: torch.Tensor,
    engine: BiophysicalEngine,
    cone_angle_deg: float = 30.0,
    gamma: float = 4.0,
    mode: str = "1A",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute multi-objective Doob h-transform log-rate tilt across candidate jumps.
    Guarantees `log_tilt[gate == 0] == 0` and stops boosting once targets are hit.
    """
    one_body_only = mode == "1A_1body"
    use_tag = mode in ("TAG", "MOG_TAG")
    use_cone = mode in ("1A", "1A_1body", "MOG_TAG")

    g_exact, g_tag, _ = compute_exact_and_tag_gains(P, engine, one_body_only=one_body_only)
    gains = g_tag if use_tag else g_exact
    B, K, L, V = gains.shape
    omega = engine.get_omega()

    if mode == "TAG":
        raw_dir = torch.einsum("bklv,k->blv", gains, omega).clamp(-4.0, 4.0)
        gate = active_slot_mask.unsqueeze(-1).to(gains.dtype).expand_as(raw_dir)
        return gamma * gate * raw_dir, gate

    ranks = torch.zeros_like(gains)
    for b in range(B):
        slot_idx = torch.nonzero(active_slot_mask[b], as_tuple=False).squeeze(-1)
        if slot_idx.numel() == 0:
            continue
        for k in range(K):
            flat_vals = gains[b, k, slot_idx, :].reshape(-1)
            ranks[b, k, slot_idx, :] = positive_percentile_ranks(flat_vals).view(-1, V)

    dir_score = torch.einsum("bklv,k->blv", ranks, omega)
    has_pos = dir_score > 1e-8
    if not use_cone:
        gate = (has_pos & active_slot_mask.unsqueeze(-1)).to(gains.dtype)
        return gamma * gate * dir_score, gate

    r_norm = ranks.norm(dim=1) + 1e-12
    cos_alpha = (dir_score / r_norm).clamp(-1.0, 1.0)
    cos_thresh = math.cos(math.radians(cone_angle_deg))
    gate = (has_pos & (cos_alpha >= cos_thresh - 1e-7) & active_slot_mask.unsqueeze(-1)).to(gains.dtype)
    return gamma * gate * dir_score, gate


def apply_doob_rate_tilt(
    u_base: torch.Tensor,
    log_tilt: torch.Tensor,
    max_log_tilt: float = 12.0,
) -> torch.Tensor:
    """Tilt off-diagonal CTMC jump rates u_base >= 0 via Doob's h-transform."""
    return u_base * torch.exp(log_tilt.clamp(-max_log_tilt, max_log_tilt))