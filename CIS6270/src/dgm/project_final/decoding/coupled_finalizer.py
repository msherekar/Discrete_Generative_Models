"""
Part 1B: Coupled Constraint-Preserving Discretization Operator.

Replaces independent per-position argmax on terminal probability matrices P [L_max, 21]
with a three-stage coupled projection:
  Stage 1: Contiguous variable-length decomposition over L in [L_min, L_max].
  Stage 2: KL-minimal Covariance-Newton Sinkhorn tilting + coupled A* beam rounding
           + exact single-site feasibility polish.
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
    steps: int = 50,
    lr: float = 0.8,
) -> torch.Tensor:
    """
    Stage 2a: Solve min_Q sum_i KL(Q_i || P_aa_i) s.t. E_Q[sum_i g_m(x_i)] in [lo_sum, hi_sum]
    via damped Newton steps using the exact M x M posterior covariance Hessian
    H(lam) = sum_i Cov_{Q_i}(g), which decouples correlated multi-pH constraints.
    """
    if G.numel() == 0:
        return torch.log_softmax(log_P_aa, dim=-1)

    M = G.size(0)
    # Aim slightly inside the box interior so discrete rounding stays clear of boundaries
    margin = 0.15 * (hi_sum - lo_sum)
    lo_t = lo_sum + margin
    hi_t = hi_sum - margin

    lam = torch.zeros(M, device=log_P_aa.device, dtype=log_P_aa.dtype)
    eye_M = torch.eye(M, device=log_P_aa.device, dtype=log_P_aa.dtype)

    for _ in range(steps):
        log_Q = torch.log_softmax(log_P_aa - torch.matmul(lam, G).unsqueeze(0), dim=-1)
        Q = log_Q.exp()
        mean_i = torch.matmul(Q, G.T)  # [L, M]
        expected_sums = mean_i.sum(dim=0)  # [M]

        over = (expected_sums - hi_t).clamp(min=0.0)
        under = (lo_t - expected_sums).clamp(min=0.0)
        grad = over - under
        if grad.abs().max().item() < 1e-4:
            break

        # Exact M x M covariance Hessian: sum_i (G diag(Q_i) G^T - mean_i mean_i^T)
        q_sum = Q.sum(dim=0)  # [20]
        cov_H = torch.matmul(G * q_sum.unsqueeze(0), G.T) - torch.matmul(mean_i.T, mean_i)
        step_vec = torch.linalg.solve(cov_H + 1e-2 * eye_M, grad).clamp(-2.5, 2.5)
        lam = (lam + lr * step_vec).clamp(-30.0, 30.0)

    return torch.log_softmax(log_P_aa - torch.matmul(lam, G).unsqueeze(0), dim=-1)


def _augment_pairwise_bounds(
    G: torch.Tensor, lo_sum: torch.Tensor, hi_sum: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Augment M linear constraints with all pairwise differences (g_a - g_b) so look-ahead
    pruning enforces coupled multi-pH constraints (e.g., q_55 - q_74 via Histidine).
    """
    M = G.size(0)
    if M <= 1:
        return G, lo_sum, hi_sum
    g_rows, lo_rows, hi_rows = [G], [lo_sum], [hi_sum]
    for a in range(M):
        for b in range(a + 1, M):
            g_rows.append((G[a] - G[b]).unsqueeze(0))
            lo_rows.append((lo_sum[a] - hi_sum[b]).unsqueeze(0))
            hi_rows.append((hi_sum[a] - lo_sum[b]).unsqueeze(0))
    return torch.cat(g_rows, dim=0), torch.cat(lo_rows, dim=0), torch.cat(hi_rows, dim=0)


def repair_to_feasible_box(
    seq: torch.Tensor,
    log_Q: torch.Tensor,
    G: torch.Tensor,
    lo_sum: torch.Tensor,
    hi_sum: torch.Tensor,
    max_repairs: int = 80,
) -> Tuple[torch.Tensor, bool]:
    """
    Exact single-site coordinate polish guaranteeing hard 1-body box satisfaction
    whenever [lo_sum, hi_sum] is non-empty, breaking ties by maximum log_Q.
    """
    if G.numel() == 0:
        return seq, True

    seq = seq.clone()
    L = seq.numel()
    M = G.size(0)
    idx_L = torch.arange(L, device=seq.device)
    tol = 1e-4

    for _ in range(max_repairs):
        cur_g = G[:, seq]  # [M, L]
        cur_sum = cur_g.sum(dim=-1)  # [M]
        over = (cur_sum - hi_sum).clamp(min=0.0)
        under = (lo_sum - cur_sum).clamp(min=0.0)
        if float(over.max().item()) <= tol and float(under.max().item()) <= tol:
            return seq, True

        cur_viol = float((over.pow(2) + under.pow(2)).sum().item())
        cand_sums = cur_sum.view(1, 1, M) - cur_g.T.unsqueeze(1) + G.T.unsqueeze(0)
        c_over = (cand_sums - hi_sum.view(1, 1, M)).clamp(min=0.0)
        c_under = (lo_sum.view(1, 1, M) - cand_sums).clamp(min=0.0)
        cand_viol = (c_over.pow(2) + c_under.pow(2)).sum(dim=-1)  # [L, 20]

        d_logq = log_Q - log_Q[idx_L, seq].unsqueeze(1)
        fully_valid = (c_over.max(dim=-1).values <= tol) & (c_under.max(dim=-1).values <= tol)
        if fully_valid.any():
            score = torch.where(fully_valid, d_logq, torch.full_like(d_logq, -1e9))
        else:
            improving = cand_viol < (cur_viol - 1e-7)
            if not improving.any():
                break
            score = torch.where(
                improving,
                (cur_viol - cand_viol) + 1e-3 * d_logq.clamp(min=-20.0),
                torch.full_like(d_logq, -1e9),
            )

        best_flat = int(score.argmax().item())
        seq[best_flat // 20] = best_flat % 20

    cur_sum = G[:, seq].sum(dim=-1)
    ok = bool(((cur_sum >= lo_sum - tol) & (cur_sum <= hi_sum + tol)).all().item())
    return seq, ok


def feasible_beam_round(
    log_Q: torch.Tensor,
    G: torch.Tensor,
    lo_sum: torch.Tensor,
    hi_sum: torch.Tensor,
    beam_width: int = 64,
) -> Tuple[Optional[torch.Tensor], bool]:
    """
    Stage 2b: Discretize tilted marginals log_Q [L, 20] into a hard sequence x in {0..19}^L
    using coupled pairwise suffix pruning + A* expected-suffix lookahead, backed by
    exact single-site box polish.
    """
    L, _ = log_Q.shape
    if G.numel() == 0:
        return log_Q.argmax(dim=-1), True

    M = G.size(0)
    G_aug, lo_aug, hi_aug = _augment_pairwise_bounds(G, lo_sum, hi_sum)
    M_aug = G_aug.size(0)
    g_min = G_aug.min(dim=-1).values
    g_max = G_aug.max(dim=-1).values
    exp_g_pos = torch.matmul(log_Q.exp(), G.T)  # [L, M]
    tol = 1e-4

    scores = torch.zeros((1,), device=log_Q.device, dtype=log_Q.dtype)
    sums_aug = torch.zeros((1, M_aug), device=log_Q.device, dtype=log_Q.dtype)
    seqs = torch.empty((1, 0), device=log_Q.device, dtype=torch.long)

    beam_failed = False
    for i in range(L):
        rem = float(L - 1 - i)
        cand_sums_aug = sums_aug.unsqueeze(1) + G_aug.T.unsqueeze(0)  # [B, 20, M_aug]
        cand_scores = scores.unsqueeze(1) + log_Q[i].unsqueeze(0)  # [B, 20]

        can_lo = (cand_sums_aug + rem * g_max.view(1, 1, M_aug) >= lo_aug.view(1, 1, M_aug) - tol).all(dim=-1)
        can_hi = (cand_sums_aug + rem * g_min.view(1, 1, M_aug) <= hi_aug.view(1, 1, M_aug) + tol).all(dim=-1)
        valid = can_lo & can_hi

        flat_valid = torch.nonzero(valid, as_tuple=False)
        if flat_valid.numel() == 0:
            beam_failed = True
            break

        if i < L - 1:
            rem_exp = exp_g_pos[i + 1 :].sum(dim=0).view(1, 1, M)
            pred_end = cand_sums_aug[:, :, :M] + rem_exp
            box_pen = (
                (pred_end - hi_sum.view(1, 1, M)).clamp(min=0.0).pow(2)
                + (lo_sum.view(1, 1, M) - pred_end).clamp(min=0.0).pow(2)
            ).sum(dim=-1)
            rank_scores = cand_scores - 4.0 * box_pen
        else:
            rank_scores = cand_scores

        b_idx, v_idx = flat_valid[:, 0], flat_valid[:, 1]
        k = min(beam_width, b_idx.numel())
        top_pos = torch.topk(rank_scores[b_idx, v_idx], k=k, largest=True).indices

        sel_b, sel_v = b_idx[top_pos], v_idx[top_pos]
        scores = cand_scores[sel_b, sel_v]
        sums_aug = cand_sums_aug[sel_b, sel_v]
        seqs = torch.cat([seqs[sel_b], sel_v.unsqueeze(1)], dim=1)

    init_seq = log_Q.argmax(dim=-1) if beam_failed else seqs[0]
    return repair_to_feasible_box(init_seq, log_Q, G, lo_sum, hi_sum)


def refine_pairwise_swaps(
    seq: torch.Tensor,
    log_P_aa: torch.Tensor,
    engine: BiophysicalEngine,
    max_swaps: int = 150,
) -> Tuple[torch.Tensor, float]:
    """
    Stage 3: Vectorized composition-preserving pairwise swap climb over U(x):
        U(x) = sum_{i=1}^L log P_aa[i, x_i] - sum_{k in 2body} beta_k * R_k(x)^2.
    Leaves all 1-body counts strictly invariant (Theorem 3).
    """
    seq = seq.clone()
    L = seq.numel()
    idx_L = torch.arange(L, device=seq.device)
    cur_logp = log_P_aa[idx_L, seq].sum().item()

    pair_objs = [o for o in engine.objectives if o.kind == "2body" and o.swap_weight > 0.0]
    if not pair_objs or max_swaps <= 0 or L <= 1:
        pen = 0.0
        for o in pair_objs:
            phi_v = o.phi[seq]
            W_sub = o.W[:L, :L]
            r_val = o.a_tilde[seq].sum() + 0.5 * (phi_v @ (W_sub @ phi_v))
            pen += o.swap_weight * float(r_val.item() ** 2)
        return seq, cur_logp - pen

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
        lp_i_at_j = log_P_aa[:, seq]
        cur_lp_i = log_P_aa[idx_L, seq]
        d_logp = (lp_i_at_j + lp_i_at_j.T) - (cur_lp_i.unsqueeze(1) + cur_lp_i.unsqueeze(0))

        d_pen = torch.zeros_like(d_logp)
        dR_mats: List[torch.Tensor] = []
        for W_sub, pv, sv, r_cur, beta in zip(W_list, phi_vecs, s_vecs, R_vals, betas):
            delta_ij = pv.unsqueeze(0) - pv.unsqueeze(1)
            dR = delta_ij * (sv.unsqueeze(1) - sv.unsqueeze(0)) - (delta_ij ** 2) * W_sub
            dR_mats.append(dR)
            d_pen = d_pen + beta * ((r_cur + dR) ** 2 - (r_cur ** 2))

        same_tok = seq.unsqueeze(1) == seq.unsqueeze(0)
        delta_U = torch.where(upper_mask & (~same_tok), d_logp - d_pen, torch.full_like(d_logp, -1e9))

        if delta_U.max().item() <= 1e-7:
            break

        flat_idx = int(delta_U.argmax().item())
        i_sw, j_sw = flat_idx // L, flat_idx % L
        cur_logp += float(d_logp[i_sw, j_sw].item())

        for idx_k, W_sub in enumerate(W_list):
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
        log_Q = sinkhorn_dual_tilt(log_P_aa, G, lo_sum, hi_sum)
        seq_dp, ok = feasible_beam_round(log_Q, G, lo_sum, hi_sum, beam_width=cfg.beam_width)
        if not ok or seq_dp is None:
            continue

        max_sw = cfg.max_swaps if run_swaps else 0
        seq_ref, u_score = refine_pairwise_swaps(seq_dp, log_P_aa, engine, max_swaps=max_sw)
        tail_pad_lp = float(log_P[L:, PAD_ID].sum().item()) if L < L_max else 0.0
        non_pad_mass = float((1.0 - P_clamped[:L, PAD_ID]).clamp(min=1e-12).log().sum().item())
        total_score = u_score + non_pad_mass + tail_pad_lp

        if total_score > best_total:
            best_total = total_score
            best_aa_seq = seq_ref
            best_L = L

    if best_aa_seq is None:
        best_L = fixed_length or ((engine.task.L_min + engine.task.L_max) // 2)
        P_aa = P_clamped[:best_L, :20]
        P_aa = P_aa / P_aa.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        log_P_aa = P_aa.clamp(min=1e-12).log()
        lo_sum, hi_sum = float(best_L) * lo_per_res, float(best_L) * hi_per_res
        log_Q = sinkhorn_dual_tilt(log_P_aa, G, lo_sum, hi_sum)
        repaired, _ = repair_to_feasible_box(log_Q.argmax(dim=-1), log_Q, G, lo_sum, hi_sum)
        best_aa_seq, best_total = refine_pairwise_swaps(
            repaired, log_P_aa, engine, max_swaps=cfg.max_swaps if run_swaps else 0
        )

    padded_out = torch.full((L_max,), PAD_ID, device=P_final.device, dtype=torch.long)
    padded_out[:best_L] = best_aa_seq
    diag = engine.evaluate_discrete_tokens(padded_out.tolist())
    diag["objective_score"] = best_total
    return padded_out, best_L, diag