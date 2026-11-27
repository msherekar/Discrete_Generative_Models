"""
Part 1B: Coupled Constraint-Preserving Discretization Operator.

Replaces independent per-position argmax on terminal probability matrices P [L_max, 21]
with a three-stage coupled projection:
  Stage 1: Contiguous variable-length decomposition over L in [L_min, L_max].
  Stage 2: KL-minimal Sinkhorn/dual exponential tilting + suffix-feasible beam rounding.
  Stage 3: Vectorized O(1)-delta composition-preserving pairwise swap refinement.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from dgm.project_final.config import SamplerConfig
from dgm.project_final.physics.descriptors import PAD_ID, BiophysicalEngine


def sinkhorn_dual_tilt(
    log_P_aa: torch.Tensor,
    G: torch.Tensor,
    lo_sum: torch.Tensor,
    hi_sum: torch.Tensor,
    steps: int = 150,
    lr: float = 0.25,
) -> torch.Tensor:
    """
    Stage 2a: Solve min_Q sum_i KL(Q_i || P_aa_i) subject to E_Q[sum_i g_m(x_i)] in [lo_sum, hi_sum]
    via dual Bregman/Sinkhorn exponential tilting Q*_{i,v} proportional to P_{i,v} exp(-lam^T g(v)).
    """
    if G.numel() == 0:
        return torch.log_softmax(log_P_aa, dim=-1)

    lam = torch.zeros(G.size(0), device=log_P_aa.device, dtype=log_P_aa.dtype)
    L = float(log_P_aa.size(0))
    step_lr = lr / max(L * 0.05, 1.0)

    for _ in range(steps):
        log_Q = torch.log_softmax(log_P_aa - torch.matmul(lam, G).unsqueeze(0), dim=-1)
        Q = log_Q.exp()
        expected_sums = torch.matmul(Q, G.T).sum(dim=0)  # [M]
        over = (expected_sums - hi_sum).clamp(min=0.0)
        under = (lo_sum - expected_sums).clamp(min=0.0)
        grad = over - under
        if grad.abs().max().item() < 1e-3:
            break
        lam = (lam + step_lr * grad).clamp(-25.0, 25.0)

    return torch.log_softmax(log_P_aa - torch.matmul(lam, G).unsqueeze(0), dim=-1)


def feasible_beam_round(
    log_Q: torch.Tensor,
    G: torch.Tensor,
    lo_sum: torch.Tensor,
    hi_sum: torch.Tensor,
    beam_width: int = 64,
) -> Tuple[Optional[torch.Tensor], bool]:
    """
    Stage 2b: Discretize tilted marginals log_Q [L, 20] into a hard sequence x in {0..19}^L
    maximizing sum_i log Q_{i, x_i} while strictly satisfying all M box constraints via
    look-ahead suffix min/max pruning.
    """
    L, V_aa = log_Q.shape
    if G.numel() == 0:
        return log_Q.argmax(dim=-1), True

    M = G.size(0)
    g_min = G.min(dim=-1).values  # [M]
    g_max = G.max(dim=-1).values  # [M]
    tol = 1e-4

    # Represent beam states compactly on CPU/GPU tensors: scores [B], sums [B, M], seqs [B, i]
    scores = torch.zeros((1,), device=log_Q.device, dtype=log_Q.dtype)
    sums = torch.zeros((1, M), device=log_Q.device, dtype=log_Q.dtype)
    seqs = torch.empty((1, 0), device=log_Q.device, dtype=torch.long)

    for i in range(L):
        rem = float(L - 1 - i)
        # Expand all B current states by V_aa = 20 candidate amino acids -> [B, 20, M]
        cand_sums = sums.unsqueeze(1) + G.T.unsqueeze(0)  # [B, 20, M]
        cand_scores = scores.unsqueeze(1) + log_Q[i].unsqueeze(0)  # [B, 20]

        can_reach_lo = (cand_sums + rem * g_max.view(1, 1, M) >= lo_sum.view(1, 1, M) - tol).all(dim=-1)
        can_reach_hi = (cand_sums + rem * g_min.view(1, 1, M) <= hi_sum.view(1, 1, M) + tol).all(dim=-1)
        valid = can_reach_lo & can_reach_hi  # [B, 20]

        flat_valid = torch.nonzero(valid, as_tuple=False)
        if flat_valid.numel() == 0:
            return None, False

        b_idx, v_idx = flat_valid[:, 0], flat_valid[:, 1]
        valid_scores = cand_scores[b_idx, v_idx]
        k = min(beam_width, valid_scores.numel())
        top_pos = torch.topk(valid_scores, k=k, largest=True).indices

        sel_b, sel_v = b_idx[top_pos], v_idx[top_pos]
        scores = valid_scores[top_pos]
        sums = cand_sums[sel_b, sel_v]
        seqs = torch.cat([seqs[sel_b], sel_v.unsqueeze(1)], dim=1)

    return seqs[0], True


def refine_pairwise_swaps(
    seq: torch.Tensor,
    log_P_aa: torch.Tensor,
    engine: BiophysicalEngine,
    max_swaps: int = 150,
) -> Tuple[torch.Tensor, float]:
    """
    Stage 3: Vectorized composition-preserving pairwise swap climb over U(x):
        U(x) = sum_{i=1}^L log P_aa[i, x_i] - sum_{k in 2body} beta_k * R_k(x)^2.
    Evaluates ALL L*(L-1)/2 candidate swaps in O(L^2) tensor ops per iteration
    (O(1) amortized per pair) while leaving 1-body counts 100% invariant (Theorem 3).
    """
    seq = seq.clone()
    L = seq.numel()
    idx_L = torch.arange(L, device=seq.device)
    cur_logp = log_P_aa[idx_L, seq].sum().item()

    pair_objs = [o for o in engine.objectives if o.kind == "2body" and o.swap_weight > 0.0]
    if not pair_objs or max_swaps <= 0 or L <= 1:
        # Compute initial U even if swaps are disabled (Stage-2-only ablation)
        pen = 0.0
        for o in pair_objs:
            phi_v = o.phi[seq]
            W_sub = o.W[:L, :L]
            r_val = o.a_tilde[seq].sum() + 0.5 * (phi_v @ (W_sub @ phi_v))
            pen += o.swap_weight * float(r_val.item() ** 2)
        return seq, cur_logp - pen

    # Cache W_sub, phi_vec, s_vec = W_sub @ phi_vec, and scalar R_k for each 2-body objective
    W_list = [o.W[:L, :L] for o in pair_objs]
    phi_vecs = [o.phi[seq] for o in pair_objs]
    s_vecs = [W_sub @ pv for W_sub, pv in zip(W_list, phi_vecs)]
    R_vals = [
        (o.a_tilde[seq].sum() + 0.5 * (pv @ sv)).item()
        for o, pv, sv in zip(pair_objs, phi_vecs, s_vecs)
    ]
    betas = [o.swap_weight for o in pair_objs]
    upper_mask = torch.triu(torch.ones((L, L), device=seq.device, dtype=torch.bool), diagonal=1)

    for _ in range(max_swaps):
        # 1. Exact log-likelihood change matrix for swapping slot i <-> slot j: [L, L]
        lp_i_at_j = log_P_aa[:, seq]  # entry [i, j] is log_P_aa[i, seq[j]]
        cur_lp_i = log_P_aa[idx_L, seq]
        d_logp = (lp_i_at_j + lp_i_at_j.T) - (cur_lp_i.unsqueeze(1) + cur_lp_i.unsqueeze(0))

        # 2. Exact pairwise penalty change across all (i, j) pairs simultaneously
        d_pen = torch.zeros_like(d_logp)
        dR_mats = []
        for W_sub, pv, sv, r_cur, beta in zip(W_list, phi_vecs, s_vecs, R_vals, betas):
            # delta_ij = phi[seq[j]] - phi[seq[i]]
            delta_ij = pv.unsqueeze(0) - pv.unsqueeze(1)
            dR = delta_ij * (sv.unsqueeze(1) - sv.unsqueeze(0)) - (delta_ij ** 2) * W_sub
            dR_mats.append(dR)
            d_pen = d_pen + beta * ((r_cur + dR) ** 2 - (r_cur ** 2))

        same_tok = seq.unsqueeze(1) == seq.unsqueeze(0)
        delta_U = torch.where(upper_mask & (~same_tok), d_logp - d_pen, torch.full_like(d_logp, -1e9))

        best_gain = delta_U.max().item()
        if best_gain <= 1e-7:
            break

        flat_idx = int(delta_U.argmax().item())
        i_sw, j_sw = flat_idx // L, flat_idx % L
        cur_logp += float(d_logp[i_sw, j_sw].item())

        # Update cached fields in O(L) and swap tokens in-place
        for idx_k, (o, W_sub) in enumerate(zip(pair_objs, W_list)):
            d_phi = float((phi_vecs[idx_k][j_sw] - phi_vecs[idx_k][i_sw]).item())
            R_vals[idx_k] += float(dR_mats[idx_k][i_sw, j_sw].item())
            s_vecs[idx_k] = s_vecs[idx_k] + d_phi * (W_sub[:, i_sw] - W_sub[:, j_sw])
            phi_vecs[idx_k][i_sw], phi_vecs[idx_k][j_sw] = (
                phi_vecs[idx_k][j_sw].clone(),
                phi_vecs[idx_k][i_sw].clone(),
            )
        seq[i_sw], seq[j_sw] = seq[j_sw].clone(), seq[i_sw].clone()

    total_pen = sum(b * (r ** 2) for b, r in zip(betas, R_vals))
    return seq, cur_logp - total_pen


def coupled_decode_single(
    P_final: torch.Tensor,
    engine: BiophysicalEngine,
    cfg: SamplerConfig,
    run_swaps: bool = True,
    fixed_length: Optional[int] = None,
) -> Tuple[torch.Tensor, int, Dict[str, float]]:
    """
    Run full Part 1B decoding on a single [L_max, 21] probability matrix.
    Returns right-padded [L_max] integer tensor, optimal length L*, and diagnostics.
    """
    P_clamped = P_final.clamp(min=1e-12)
    P_clamped = P_clamped / P_clamped.sum(dim=-1, keepdim=True)
    log_P = P_clamped.log()
    L_max = P_clamped.size(0)

    G, lo_per_res, hi_per_res = engine.get_linear_constraint_tensors()

    # Candidate contiguous lengths L in [L_min, L_max]
    if fixed_length is not None:
        cand_lengths = [fixed_length]
    else:
        all_L = list(range(engine.task.L_min, min(engine.task.L_max, L_max) + 1))
        log_non_pad = (1.0 - P_clamped[:, PAD_ID]).clamp(min=1e-12).log()
        log_pad = log_P[:, PAD_ID]
        len_priors = [
            float(log_non_pad[:L].sum().item() + log_pad[L:].sum().item()) for L in all_L
        ]
        ranked = sorted(zip(len_priors, all_L), reverse=True)[: cfg.top_k_lengths]
        cand_lengths = [L for _, L in ranked]

    best_total = -float("inf")
    best_aa_seq: Optional[torch.Tensor] = None
    best_L = cand_lengths[0]

    for L in cand_lengths:
        P_aa = P_clamped[:L, :20]
        P_aa = P_aa / P_aa.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        log_P_aa = P_aa.clamp(min=1e-12).log()

        lo_sum, hi_sum = float(L) * lo_per_res, float(L) * hi_per_res
        log_Q = sinkhorn_dual_tilt(
            log_P_aa, G, lo_sum, hi_sum, steps=cfg.sinkhorn_steps, lr=cfg.sinkhorn_lr
        )
        seq_dp, ok = feasible_beam_round(log_Q, G, lo_sum, hi_sum, beam_width=cfg.beam_width)
        if not ok or seq_dp is None:
            continue

        max_sw = cfg.max_swaps if run_swaps else 0
        seq_ref, u_score = refine_pairwise_swaps(seq_dp, log_P_aa, engine, max_swaps=max_sw)
        tail_pad_lp = float(log_P[L:, PAD_ID].sum().item()) if L < L_max else 0.0
        # Normalize length objective slightly by active/tail log-prob sum
        non_pad_mass = float((1.0 - P_clamped[:L, PAD_ID]).clamp(min=1e-12).log().sum().item())
        total_score = u_score + non_pad_mass + tail_pad_lp

        if total_score > best_total:
            best_total = total_score
            best_aa_seq = seq_ref
            best_L = L

    # Graceful fallback to tilted argmax at midpoint length if strict box was empty
    if best_aa_seq is None:
        best_L = fixed_length or ((engine.task.L_min + engine.task.L_max) // 2)
        P_aa = P_clamped[:best_L, :20]
        P_aa = P_aa / P_aa.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        log_Q = sinkhorn_dual_tilt(
            P_aa.clamp(min=1e-12).log(), G, best_L * lo_per_res, best_L * hi_per_res
        )
        best_aa_seq, best_total = refine_pairwise_swaps(
            log_Q.argmax(dim=-1), P_aa.clamp(min=1e-12).log(), engine, max_swaps=cfg.max_swaps
        )

    padded_out = torch.full((L_max,), PAD_ID, device=P_final.device, dtype=torch.long)
    padded_out[:best_L] = best_aa_seq
    diag = engine.evaluate_discrete_tokens(padded_out.tolist())
    diag["objective_score"] = best_total
    return padded_out, best_L, diag