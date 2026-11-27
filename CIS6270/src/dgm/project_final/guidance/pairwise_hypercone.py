"""
Part 1A: Exact Pairwise Posterior-Expectation + Hypercone Multi-Objective Guidance.

Computes exact analytical posterior expectations E_{x_1 ~ P}[R_k(x_1)] and exact
single-site transition gains across all B x L_max x 21 candidate jumps in O(L_max^2)
matrix operations, eliminating first-order Taylor-Approximated Guidance (TAG) error
on quadratic IDP functionals (SCD, SHD, sticker spacing).
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

    Args:
        P: Clean factorized marginal probabilities of shape [B, L_max, 21].
        engine: Compiled BiophysicalEngine holding K objectives.
        one_body_only: Ablation flag; if True, ignores pairwise W^(k) terms.

    Returns:
        gains_exact: [B, K, L_max, 21] exact -Delta(0.5 * mu_k^2)
        gains_tag:   [B, K, L_max, 21] 1st-order TAG approximation (-mu_k * dmu_k)
        dmu_all:     [B, K, L_max, 21] exact single-site residual shift Delta mu_k(i, v)
    """
    if P.dim() == 2:
        P = P.unsqueeze(0)

    exact_list, tag_list, dmu_list = [], [], []
    for obj in engine.objectives:
        # Expected 1-body per-position values: [B, L_max]
        a_bar = torch.matmul(P, obj.a_tilde)

        if one_body_only or obj.kind == "1body":
            s_field = torch.zeros_like(a_bar)
            phi_bar = torch.zeros_like(a_bar)
            mu = a_bar.sum(dim=-1)  # [B]
            dmu = obj.a_tilde.view(1, 1, -1) - a_bar.unsqueeze(-1)
        else:
            # Expected 2-body residue feature & O(L_max^2) interaction field s_k = W^(k) @ phi_bar
            phi_bar = torch.matmul(P, obj.phi)  # [B, L_max]
            s_field = torch.matmul(phi_bar, obj.W.T)  # [B, L_max] (W is symmetric)
            mu = a_bar.sum(dim=-1) + 0.5 * (phi_bar * s_field).sum(dim=-1)  # [B]
            dmu = (obj.a_tilde.view(1, 1, -1) - a_bar.unsqueeze(-1)) + (
                obj.phi.view(1, 1, -1) - phi_bar.unsqueeze(-1)
            ) * s_field.unsqueeze(-1)

        # Squared residual energy J_k(P) = 0.5 * mu_k(P)^2 -> reduction is -Delta J_k
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


def average_percentile_ranks(values_1d: torch.Tensor) -> torch.Tensor:
    """
    Tie-aware percentile rank normalization mapping an N-element 1D tensor into [0, 1].
    Handles ties symmetrically via average of strict-less and less-or-equal counts.
    """
    n = values_1d.numel()
    if n <= 1:
        return torch.ones_like(values_1d)
    diff_lt = (values_1d.unsqueeze(0) < values_1d.unsqueeze(1)).to(values_1d.dtype).sum(dim=1)
    diff_le = (values_1d.unsqueeze(0) <= values_1d.unsqueeze(1)).to(values_1d.dtype).sum(dim=1) - 1.0
    return 0.5 * (diff_lt + diff_le) / float(n - 1)


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

    Supported ablation modes:
        - "1A":          Exact 1-body + 2-body gains + rank scaling + hypercone gate (Ours)
        - "1A_no_cone":  Exact gains + rank scaling without hypercone gate
        - "1A_1body":    1-body gains only + rank scaling + hypercone gate
        - "MOG_TAG":     1st-order TAG gains + rank scaling + hypercone gate
        - "TAG":         1st-order TAG gains + scalarized sum (no cone)

    Returns:
        log_tilt: [B, L_max, 21] guidance exponent added to log base jump rates
        gate:     [B, L_max, 21] binary hypercone indicator in {0.0, 1.0}
    """
    one_body_only = mode == "1A_1body"
    use_tag = mode in ("TAG", "MOG_TAG")
    use_cone = mode in ("1A", "1A_1body", "MOG_TAG")

    g_exact, g_tag, _ = compute_exact_and_tag_gains(P, engine, one_body_only=one_body_only)
    gains = g_tag if use_tag else g_exact  # [B, K, L_max, 21]
    B, K, L, V = gains.shape

    omega = engine.get_omega()  # [K], unit norm

    if mode == "TAG":
        # Raw unranked linear scalarization baseline
        raw_dir = torch.einsum("bklv,k->blv", gains, omega)
        active_3d = active_slot_mask.unsqueeze(-1).to(gains.dtype)
        return gamma * raw_dir * active_3d, torch.ones_like(raw_dir)

    # MOG-DFM percentile rank normalization across active masked slots per sequence
    ranks = torch.zeros_like(gains)
    for b in range(B):
        slot_idx = torch.nonzero(active_slot_mask[b], as_tuple=False).squeeze(-1)
        if slot_idx.numel() == 0:
            continue
        for k in range(K):
            flat_vals = gains[b, k, slot_idx, :].reshape(-1)
            ranks[b, k, slot_idx, :] = average_percentile_ranks(flat_vals).view(-1, V)

    dir_score = torch.einsum("bklv,k->blv", ranks, omega)  # [B, L_max, 21]
    if not use_cone:
        gate = active_slot_mask.unsqueeze(-1).to(gains.dtype).expand_as(dir_score)
        return gamma * gate * dir_score, gate

    # Hypercone angular filter: alpha_{i,v} <= Phi
    r_norm = ranks.norm(dim=1) + 1e-12
    cos_alpha = (dir_score / r_norm).clamp(-1.0, 1.0)
    cos_thresh = math.cos(math.radians(cone_angle_deg))
    gate = ((cos_alpha >= cos_thresh - 1e-7) & active_slot_mask.unsqueeze(-1)).to(gains.dtype)
    return gamma * gate * dir_score, gate


def apply_doob_rate_tilt(
    u_base: torch.Tensor,
    log_tilt: torch.Tensor,
    max_log_tilt: float = 12.0,
) -> torch.Tensor:
    """
    Tilt unguided off-diagonal CTMC jump rates u_base >= 0 via Doob's h-transform:
        u_tilted = u_base * exp(clamp(log_tilt, -max_log_tilt, max_log_tilt)).
    Guarantees strict rate nonnegativity (Theorem 2).
    """
    return u_base * torch.exp(log_tilt.clamp(-max_log_tilt, max_log_tilt))