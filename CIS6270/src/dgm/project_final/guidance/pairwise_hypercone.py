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


def signed_normalized_gains(values_1d: torch.Tensor) -> torch.Tensor:
    """
    Scale-normalize gains into [-1.5, 1.0] per objective while preserving sign:
      > 0 improves objective, == 0 chemically orthogonal, < 0 harms/overshoots objective.
    """
    scale = values_1d.abs().quantile(0.95).clamp(min=1e-5)
    return (values_1d / scale).clamp(-1.5, 1.0)


def compute_hypercone_log_tilt(
    P: torch.Tensor,
    active_slot_mask: torch.Tensor,
    engine: BiophysicalEngine,
    cone_angle_deg: float = 45.0,
    gamma: float = 4.0,
    mode: str = "1A",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute multi-objective Doob h-transform log-rate tilt across candidate jumps.
    Preserves symmetric negative suppression for conflict/overshoot moves while gating
    positive boosts via smooth Pareto hypercone alignment over active/harmed objectives.
    """
    one_body_only = mode == "1A_1body"
    use_tag = mode in ("TAG", "MOG_TAG")
    use_cone = mode in ("1A", "1A_1body", "MOG_TAG")

    g_exact, g_tag, _ = compute_exact_and_tag_gains(P, engine, one_body_only=one_body_only)
    gains = g_tag if use_tag else g_exact  # [B, K, L_max, 21]
    gains = torch.nan_to_num(gains, nan=0.0, posinf=1.0, neginf=-1.0)
    B, K, L, V = gains.shape
    omega = engine.get_omega()  # [K]

    norm_g = torch.zeros_like(gains)
    for b in range(B):
        slot_idx = torch.nonzero(active_slot_mask[b], as_tuple=False).squeeze(-1)
        if slot_idx.numel() == 0:
            continue
        for k in range(K):
            flat_vals = gains[b, k, slot_idx, :].reshape(-1)
            norm_g[b, k, slot_idx, :] = signed_normalized_gains(flat_vals).view(-1, V)
    norm_g = torch.nan_to_num(norm_g, nan=0.0, posinf=1.0, neginf=-1.0)

    slot_gate = active_slot_mask.unsqueeze(-1).to(gains.dtype)
    dir_score = torch.einsum("bklv,k->blv", norm_g, omega)

    if mode == "TAG" or not use_cone or cone_angle_deg >= 89.0:
        log_tilt = (gamma * slot_gate * dir_score).clamp(-6.0, 6.0)
        return log_tilt, slot_gate.expand_as(dir_score)

    # Dimension-aware cone floor so high-K tasks (e.g. T4 with K=5) do not over-prune valid moves
    eff_angle = max(cone_angle_deg, 52.0 if K >= 4 else 42.0)
    cos_thresh = math.cos(math.radians(eff_angle))

    # Active/harmed subspace: include objective k if move touches it (|g_k| > 0.05) OR harms it (g_k < -0.02)
    touched = ((norm_g.abs() > 0.05) | (norm_g < -0.02)).to(gains.dtype)
    omega_sub = touched * omega.view(1, K, 1, 1)
    sub_dot = (norm_g * omega_sub).sum(dim=1)
    g_sub_norm = (norm_g * touched).norm(dim=1) + 1e-8
    w_sub_norm = omega_sub.norm(dim=1) + 1e-8
    cos_alpha = (sub_dot / (g_sub_norm * w_sub_norm)).clamp(-1.0, 1.0)

    # Smooth cone weight in [0, 1] for positive gains + hard veto if any weighted objective is badly harmed
    worst_harm = (norm_g * (omega.view(1, K, 1, 1) > 0).to(gains.dtype)).amin(dim=1)
    no_severe_conflict = (worst_harm >= -0.35).to(gains.dtype)
    cone_weight = ((cos_alpha - cos_thresh) / 0.18).clamp(0.0, 1.0) * no_severe_conflict

    pos_part = dir_score.clamp(min=0.0) * cone_weight
    neg_part = dir_score.clamp(max=0.0)
    steered_score = pos_part + neg_part

    log_tilt = (gamma * slot_gate * steered_score).clamp(-6.0, 6.0)
    active_cone_gate = (slot_gate * (cone_weight > 0.0).to(gains.dtype))
    return log_tilt, active_cone_gate


def apply_doob_rate_tilt(
    u_base: torch.Tensor,
    log_tilt: torch.Tensor,
    max_log_tilt: float = 12.0,
) -> torch.Tensor:
    """Tilt off-diagonal CTMC jump rates u_base >= 0 via Doob's h-transform."""
    return u_base * torch.exp(log_tilt.clamp(-max_log_tilt, max_log_tilt))
