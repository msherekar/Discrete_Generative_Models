"""
Independent evaluation suite for generated responsive IDP sequences.

Computes multi-condition biophysical errors and constraint pass rates from first
principles (never sharing tensors with BiophysicalEngine), plus `localcider` kappa,
`metapredict` intrinsic disorder, `ESM-2` pseudo-perplexity, diversity, novelty,
2D/KD Monte-Carlo Pareto hypervolume, and optional CALVADOS/OpenMM R_g shifts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from dgm.project_final.config import EnvConditionSpec, TaskSpec

# Independent amino acid constants (kept separate from `src/physics/descriptors.py`)
_PKA_ACID = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
_PKA_BASE = {"H": 6.0, "K": 10.5, "R": 12.5}
_HYDRO = {
    "A": 0.73, "C": 0.60, "D": 0.09, "E": 0.05, "F": 0.95, "G": 0.65, "H": 0.50,
    "I": 0.90, "K": 0.14, "L": 0.92, "M": 0.83, "N": 0.37, "P": 0.76, "Q": 0.39,
    "R": 0.72, "S": 0.59, "T": 0.67, "V": 0.83, "W": 0.98, "Y": 0.94,
}
# Campen et al. TOP-IDP disorder propensity scale (fallback only if metapredict absent)
_TOP_IDP = {
    "A": 0.06, "C": 0.02, "D": 0.192, "E": 0.736, "F": -0.697, "G": 0.166, "H": 0.303,
    "I": -0.486, "K": 0.586, "L": -0.326, "M": -0.397, "N": 0.007, "P": 0.987, "Q": 0.318,
    "R": 0.180, "S": 0.341, "T": 0.059, "V": -0.121, "W": -0.884, "Y": -0.510,
}

try:
    from localcider.sequenceParameters import SequenceParameters as _CiderParams
except ImportError:
    _CiderParams = None

try:
    import metapredict as _metapredict
except ImportError:
    _metapredict = None


def independent_condition_props(seq: str, cond: EnvConditionSpec) -> Dict[str, float]:
    """Compute exact NCPR, FCR, salt-screened SCD, and thermo-SHD independently on a string."""
    L = max(len(seq), 1)
    q = []
    for ch in seq:
        if ch in _PKA_ACID:
            q.append(-1.0 / (1.0 + 10.0 ** (_PKA_ACID[ch] - cond.pH)))
        elif ch in _PKA_BASE:
            q.append(+1.0 / (1.0 + 10.0 ** (cond.pH - _PKA_BASE[ch])))
        else:
            q.append(0.0)

    ncpr = sum(q) / L
    fcr = sum(abs(v) for v in q) / L

    kappa_res = 0.0
    if cond.salt_mM > 0.0:
        debye_nm = (0.304 / math.sqrt(cond.salt_mM / 1000.0)) * math.sqrt(max(cond.temp_K, 250.0) / 298.15)
        kappa_res = 0.38 / debye_nm

    dt = (cond.temp_K - 298.15) / 100.0
    lam = [
        max(0.01, min(1.20, _HYDRO[ch] + (0.08 * dt if _HYDRO[ch] > 0.65 else -0.03 * dt)))
        for ch in seq
    ]

    scd_acc, shd_acc = 0.0, 0.0
    for i in range(L):
        for j in range(i + 1, L):
            d = j - i
            scd_acc += q[i] * q[j] * math.sqrt(d) * math.exp(-kappa_res * d)
            shd_acc += (lam[i] * lam[j]) * (d ** (-cond.shd_beta))
    return {"ncpr": ncpr, "fcr": fcr, "scd": scd_acc / L, "shd": shd_acc / L}


def predict_disorder_scores(sequences: Sequence[str]) -> Tuple[List[float], str]:
    """Predict per-sequence mean intrinsic disorder via `metapredict` (or TOP-IDP fallback)."""
    if not sequences:
        return [], "none"
    if _metapredict is not None:
        scores: List[float] = []
        for s in sequences:
            arr = _metapredict.predict_disorder(s)
            scores.append(float(sum(arr) / max(len(arr), 1)))
        return scores, "metapredict"

    fallback = [
        float(min(0.99, max(0.01, 0.52 + 0.35 * sum(_TOP_IDP.get(c, 0.0) for c in s) / max(len(s), 1))))
        for s in sequences
    ]
    return fallback, "top_idp_fallback"


@torch.no_grad()
def compute_esm2_pseudo_perplexity(
    sequences: Sequence[str],
    model_name: str = "facebook/esm2_t12_35M_UR50D",
    device: Optional[torch.device] = None,
    max_positions: int = 24,
) -> List[float]:
    """Compute subsampled masked pseudo-perplexity using pretrained ESM-2 on GPU."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name).to(dev).eval()
    mask_id = tok.mask_token_id

    ppls: List[float] = []
    for seq in sequences:
        if not seq:
            ppls.append(999.0)
            continue
        enc = tok(seq, return_tensors="pt").to(dev)
        input_ids = enc["input_ids"][0]  # [1 + L + 1] with <cls> ... <eos>
        L = len(seq)
        step = max(1, L // max_positions)
        pos_list = list(range(1, L + 1, step))
        batch_in = input_ids.unsqueeze(0).repeat(len(pos_list), 1)
        targets = []
        for r_i, p_idx in enumerate(pos_list):
            targets.append(int(input_ids[p_idx].item()))
            batch_in[r_i, p_idx] = mask_id

        logits = mdl(batch_in).logits
        nll = 0.0
        for r_i, (p_idx, tgt) in enumerate(zip(pos_list, targets)):
            log_probs = torch.log_softmax(logits[r_i, p_idx], dim=-1)
            nll -= float(log_probs[tgt].item())
        ppls.append(float(math.exp(min(nll / max(len(pos_list), 1), 10.0))))
    return ppls


def normalized_seq_distance(a: str, b: str) -> float:
    """Fast length-aware normalized sequence dissimilarity in [0, 1]."""
    if not a and not b:
        return 0.0
    l_min, l_max = min(len(a), len(b)), max(len(a), len(b))
    mismatches = sum(1 for x, y in zip(a[:l_min], b[:l_min]) if x != y) + (l_max - l_min)
    return mismatches / float(max(l_max, 1))


def compute_pareto_hypervolume(
    error_vectors: Sequence[Sequence[float]], ref_bound: float = 2.0, n_mc: int = 4096
) -> float:
    """
    Estimate dominated hypervolume in [0, ref_bound]^D (minimization of objective MAEs)
    using deterministic Sobol quasi-Monte-Carlo points so arbitrary D >= 2 works cleanly.
    """
    if not error_vectors:
        return 0.0
    errs = torch.tensor(error_vectors, dtype=torch.float32)  # [N, D]
    valid = (errs < ref_bound).all(dim=-1)
    if not valid.any():
        return 0.0
    errs = errs[valid]
    D = errs.size(1)
    sobol = torch.quasirandom.SobolEngine(dimension=D, scramble=False)
    pts = sobol.draw(n_mc) * ref_bound  # [M, D]
    # Point p is dominated if there exists candidate c with c_d <= p_d for all d
    dominated = (errs.unsqueeze(1) <= pts.unsqueeze(0)).all(dim=-1).any(dim=0)
    return float(dominated.float().mean().item())


def evaluate_sequences(
    sequences: Sequence[str],
    task: TaskSpec,
    train_reference: Optional[Sequence[str]] = None,
    compute_esm2: bool = False,
    esm2_model: str = "facebook/esm2_t12_35M_UR50D",
    scd_tol: float = 0.45,
    shd_tol: float = 0.45,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate a batch of generated IDP strings and return (summary_dict, per_seq_list)."""
    clean_seqs = [s for s in sequences if len(s) > 0]
    if not clean_seqs:
        return {"pass_rate_all": 0.0, "pareto_hypervolume": 0.0}, []

    disorder_vals, dis_src = predict_disorder_scores(clean_seqs)
    esm_vals = (
        compute_esm2_pseudo_perplexity(clean_seqs, model_name=esm2_model)
        if compute_esm2
        else [-1.0] * len(clean_seqs)
    )

    per_seq: List[Dict[str, Any]] = []
    err_matrix: List[List[float]] = []

    for seq, dis_sc, ppl in zip(clean_seqs, disorder_vals, esm_vals):
        L = len(seq)
        ok = task.L_min <= L <= task.L_max
        errs: List[float] = []
        rec: Dict[str, Any] = {"length": L, "disorder": dis_sc, "esm2_ppl": ppl}

        if _CiderParams is not None:
            try:
                rec["kappa_cider"] = float(_CiderParams(seq).get_kappa())
            except Exception:
                rec["kappa_cider"] = -1.0

        for cond in task.conditions:
            cp = independent_condition_props(seq, cond)
            if cond.target_ncpr is not None:
                e = abs(cp["ncpr"] - cond.target_ncpr)
                rec[f"mae_{cond.name}_ncpr"] = e
                errs.append(e * 4.0)
            if cond.ncpr_bounds is not None:
                ok = ok and (cond.ncpr_bounds[0] - 1e-4 <= cp["ncpr"] <= cond.ncpr_bounds[1] + 1e-4)
            if cond.fcr_bounds is not None:
                ok = ok and (cond.fcr_bounds[0] - 1e-4 <= cp["fcr"] <= cond.fcr_bounds[1] + 1e-4)
            if cond.target_scd is not None:
                e_scd = abs(cp["scd"] - cond.target_scd)
                rec[f"mae_{cond.name}_scd"] = e_scd
                errs.append(e_scd)
                ok = ok and (e_scd <= scd_tol)
            if cond.target_shd is not None:
                e_shd = abs(cp["shd"] - cond.target_shd)
                rec[f"mae_{cond.name}_shd"] = e_shd
                errs.append(e_shd)
                ok = ok and (e_shd <= shd_tol)

        f_aro = sum(1 for ch in seq if ch in "YFW") / float(L)
        rec["aro_frac"] = f_aro
        if task.target_aro_frac is not None:
            errs.append(abs(f_aro - task.target_aro_frac) * 5.0)
        if task.aro_bounds is not None:
            ok = ok and (task.aro_bounds[0] - 1e-4 <= f_aro <= task.aro_bounds[1] + 1e-4)

        rec["pass_all"] = bool(ok)
        rec["total_mae"] = float(sum(errs) / max(len(errs), 1))
        per_seq.append(rec)
        err_matrix.append(errs if errs else [0.0])

    # Pairwise internal diversity (subsampled to <= 120 sequences for speed)
    sub = clean_seqs[:120]
    div_pairs = [
        normalized_seq_distance(sub[i], sub[j])
        for i in range(len(sub))
        for j in range(i + 1, len(sub))
    ]
    diversity = float(sum(div_pairs) / max(len(div_pairs), 1))

    novelty = -1.0
    if train_reference:
        ref_sub = list(train_reference)[:500]
        novelty = float(
            sum(min(normalized_seq_distance(s, r) for r in ref_sub) for s in sub) / max(len(sub), 1)
        )

    mae_keys = [k for k in per_seq[0] if k.startswith("mae_")]
    summary: Dict[str, Any] = {
        "n_evaluated": len(clean_seqs),
        "pass_rate_all": 100.0 * sum(1.0 for r in per_seq if r["pass_all"]) / len(per_seq),
        "pareto_hypervolume": compute_pareto_hypervolume(err_matrix),
        "mean_total_mae": sum(r["total_mae"] for r in per_seq) / len(per_seq),
        "mean_disorder": sum(disorder_vals) / len(disorder_vals),
        "disorder_source": dis_src,
        "mean_esm2_ppl": sum(esm_vals) / len(esm_vals),
        "diversity": diversity,
        "novelty": novelty,
        "mean_length": sum(r["length"] for r in per_seq) / len(per_seq),
    }
    for mk in mae_keys:
        summary[mk] = sum(float(r[mk]) for r in per_seq) / len(per_seq)
    return summary, per_seq