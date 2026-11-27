"""
Vectorized 21-token biophysical sequence descriptors and quadratic interaction kernels.

All 1-body and 2-body environmental objectives are represented in the unified form:
    R_k(x) = sum_i a_tilde_k(x_i) + 0.5 * sum_{i != j} W_ij^(k) * phi_k(x_i) * phi_k(x_j)
with strict zero weight on the padding token (<PAD>, index 20) so variable-length
sequences on a canvas of size L_max are evaluated without ratio-expectation bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from dgm.project_final.config import TaskSpec


# ==============================================================================
# 1. Vocabulary & Canonical Amino Acid Lookup Constants
# ==============================================================================
AA20: str = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN: str = "<PAD>"
VOCAB: List[str] = list(AA20) + [PAD_TOKEN]
AA_TO_ID: Dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
PAD_ID: int = 20
VOCAB_SIZE: int = 21

# Standard side-chain pKa values for Henderson-Hasselbalch ionization
ACIDIC_PKA: Dict[str, float] = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
BASIC_PKA: Dict[str, float] = {"H": 6.0, "K": 10.5, "R": 12.5}

# CALVADOS / Urry normalized hydropathy scale (lambda in [0, 1]) at 298.15 K
CALVADOS_HYDRO_20: Dict[str, float] = {
    "A": 0.73, "C": 0.60, "D": 0.09, "E": 0.05, "F": 0.95,
    "G": 0.65, "H": 0.50, "I": 0.90, "K": 0.14, "L": 0.92,
    "M": 0.83, "N": 0.37, "P": 0.76, "Q": 0.39, "R": 0.72,
    "S": 0.59, "T": 0.67, "V": 0.83, "W": 0.98, "Y": 0.94,
}


# ==============================================================================
# 2. User-Parameterized Physical Property Builders (Any pH, Salt, Temperature)
# ==============================================================================
def henderson_hasselbalch_charges(
    pH: float, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Compute exact side-chain charge q(v; pH) for all 20 amino acids + <PAD>=0
    at ANY user-specified pH using the Henderson-Hasselbalch equation.
    """
    charges: List[float] = []
    for aa in AA20:
        if aa in ACIDIC_PKA:
            q = -1.0 / (1.0 + 10.0 ** (ACIDIC_PKA[aa] - pH))
        elif aa in BASIC_PKA:
            q = +1.0 / (1.0 + 10.0 ** (pH - BASIC_PKA[aa]))
        else:
            q = 0.0
        charges.append(q)
    charges.append(0.0)  # <PAD> carries strictly zero charge
    return torch.tensor(charges, device=device, dtype=dtype)


def thermo_hydropathy(
    temp_K: float, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Compute temperature-modulated hydropathy lambda(v; T) for 20 AAs + <PAD>=0.
    Captures strengthening of hydrophobic solvation penalty with temperature
    around T0 = 298.15 K while clamping <PAD> strictly to 0.0.
    """
    delta_t = (temp_K - 298.15) / 100.0
    vals: List[float] = []
    for aa in AA20:
        base = CALVADOS_HYDRO_20[aa]
        # Non-polar/aromatic side chains exhibit mild positive thermal shift
        shift = 0.08 * delta_t if base > 0.65 else -0.03 * delta_t
        vals.append(max(0.01, min(1.20, base + shift)))
    vals.append(0.0)  # <PAD> = 0
    return torch.tensor(vals, device=device, dtype=dtype)


def debye_scd_kernel(
    L_max: int,
    salt_mM: float,
    temp_K: float = 298.15,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build symmetric zero-diagonal L_max x L_max Debye-Huckel salt-screened SCD matrix:
        W_ij = |j - i|^(1/2) * exp(-kappa_D(salt_mM, T) * |j - i|)  for i != j.
    At salt_mM <= 0, kappa_D = 0, recovering unscreened Sawle & Ghosh SCD.
    """
    idx = torch.arange(L_max, device=device, dtype=dtype)
    dist = (idx[:, None] - idx[None, :]).abs()
    if salt_mM <= 0.0:
        kappa_per_res = 0.0
    else:
        ionic_M = salt_mM / 1000.0
        # Debye screening length in nm (~0.304 / sqrt(I) at 298 K) vs C_alpha step 0.38 nm
        debye_nm = (0.304 / math.sqrt(ionic_M)) * math.sqrt(max(temp_K, 250.0) / 298.15)
        kappa_per_res = 0.38 / debye_nm
    W = torch.where(dist > 0, dist.sqrt() * torch.exp(-kappa_per_res * dist), torch.zeros_like(dist))
    return W


def shd_kernel(
    L_max: int,
    beta: float = 1.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build symmetric zero-diagonal L_max x L_max SHD matrix W_ij = |j - i|^(-beta)."""
    idx = torch.arange(L_max, device=device, dtype=dtype)
    dist = (idx[:, None] - idx[None, :]).abs()
    return torch.where(dist > 0, dist.clamp(min=1.0).pow(-beta), torch.zeros_like(dist))


def sticker_clumping_kernel(
    L_max: int,
    min_sep: int = 5,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Penalize aromatic stickers (Y, F, W) placed closer than `min_sep` residues."""
    idx = torch.arange(L_max, device=device, dtype=dtype)
    dist = (idx[:, None] - idx[None, :]).abs()
    mask = (dist > 0) & (dist < float(min_sep))
    return torch.where(mask, (float(min_sep) - dist) / float(max(min_sep, 1)), torch.zeros_like(dist))


# ==============================================================================
# 3. Unified Objective & Linear Constraint Containers
# ==============================================================================
@dataclass
class ObjectiveTerm:
    """Single 1-body + 2-body residual R_k(x) used in Part 1A & Part 1B Stage 3."""

    name: str
    kind: str  # "1body" or "2body"
    a_tilde: torch.Tensor  # [21]
    phi: torch.Tensor      # [21]
    W: torch.Tensor        # [L_max, L_max], symmetric with zero diagonal
    target: float
    swap_weight: float = 1.0


@dataclass
class LinearConstraintSpec:
    """Hard per-residue 1-body box constraint sum_i g(x_i) in [L * lo, L * hi] for Part 1B."""

    name: str
    g_aa20: torch.Tensor  # [20] values over canonical amino acids
    lo_per_res: float
    hi_per_res: float


class BiophysicalEngine:
    """
    Compiles any user TaskSpec into GPU-ready 21-token vectors and L_max x L_max
    matrices for Part 1A trajectory guidance and Part 1B coupled decoding.
    """

    def __init__(
        self,
        task: TaskSpec,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.task = task
        self.L_max = task.L_max
        self.device = device
        self.dtype = dtype

        self.active_m = torch.tensor([1.0] * 20 + [0.0], device=device, dtype=dtype)
        self.aro_vec = torch.tensor(
            [1.0 if aa in "YFW" else 0.0 for aa in AA20] + [0.0], device=device, dtype=dtype
        )
        self.zero_21 = torch.zeros(VOCAB_SIZE, device=device, dtype=dtype)
        self.zero_W = torch.zeros((self.L_max, self.L_max), device=device, dtype=dtype)

        self.objectives: List[ObjectiveTerm] = []
        self.linear_constraints: List[LinearConstraintSpec] = []
        self._compile_task()

    def _compile_task(self) -> None:
        """Build all 1-body and 2-body objectives dynamically from user conditions."""
        for cond in self.task.conditions:
            q_vec = henderson_hasselbalch_charges(cond.pH, self.device, self.dtype)
            fcr_vec = q_vec.abs()

            if cond.target_ncpr is not None:
                self.objectives.append(
                    ObjectiveTerm(
                        name=f"{cond.name}_ncpr",
                        kind="1body",
                        a_tilde=q_vec - cond.target_ncpr * self.active_m,
                        phi=self.zero_21,
                        W=self.zero_W,
                        target=cond.target_ncpr,
                        swap_weight=0.0,
                    )
                )
            if cond.ncpr_bounds is not None:
                self.linear_constraints.append(
                    LinearConstraintSpec(
                        name=f"{cond.name}_ncpr_box",
                        g_aa20=q_vec[:20],
                        lo_per_res=cond.ncpr_bounds[0],
                        hi_per_res=cond.ncpr_bounds[1],
                    )
                )

            if cond.target_fcr is not None:
                self.objectives.append(
                    ObjectiveTerm(
                        name=f"{cond.name}_fcr",
                        kind="1body",
                        a_tilde=fcr_vec - cond.target_fcr * self.active_m,
                        phi=self.zero_21,
                        W=self.zero_W,
                        target=cond.target_fcr,
                        swap_weight=0.0,
                    )
                )
            if cond.fcr_bounds is not None:
                self.linear_constraints.append(
                    LinearConstraintSpec(
                        name=f"{cond.name}_fcr_box",
                        g_aa20=fcr_vec[:20],
                        lo_per_res=cond.fcr_bounds[0],
                        hi_per_res=cond.fcr_bounds[1],
                    )
                )

            if cond.target_scd is not None:
                W_scd = debye_scd_kernel(self.L_max, cond.salt_mM, cond.temp_K, self.device, self.dtype)
                self.objectives.append(
                    ObjectiveTerm(
                        name=f"{cond.name}_scd",
                        kind="2body",
                        a_tilde=-cond.target_scd * self.active_m,
                        phi=q_vec,
                        W=W_scd,
                        target=cond.target_scd,
                        swap_weight=self.task.swap_beta,
                    )
                )

            if cond.target_shd is not None:
                lam_vec = thermo_hydropathy(cond.temp_K, self.device, self.dtype)
                W_shd = shd_kernel(self.L_max, cond.shd_beta, self.device, self.dtype)
                self.objectives.append(
                    ObjectiveTerm(
                        name=f"{cond.name}_shd",
                        kind="2body",
                        a_tilde=-cond.target_shd * self.active_m,
                        phi=lam_vec,
                        W=W_shd,
                        target=cond.target_shd,
                        swap_weight=self.task.swap_beta,
                    )
                )

        if self.task.target_aro_frac is not None:
            W_stk = (
                sticker_clumping_kernel(self.L_max, self.task.min_sticker_sep, self.device, self.dtype)
                if self.task.penalize_sticker_clumps
                else self.zero_W
            )
            kind = "2body" if self.task.penalize_sticker_clumps else "1body"
            self.objectives.append(
                ObjectiveTerm(
                    name="aromatic_stickers",
                    kind=kind,
                    a_tilde=self.aro_vec - self.task.target_aro_frac * self.active_m,
                    phi=self.aro_vec if self.task.penalize_sticker_clumps else self.zero_21,
                    W=W_stk,
                    target=self.task.target_aro_frac,
                    swap_weight=self.task.swap_beta if self.task.penalize_sticker_clumps else 0.0,
                )
            )

        if self.task.aro_bounds is not None:
            self.linear_constraints.append(
                LinearConstraintSpec(
                    name="aro_frac_box",
                    g_aa20=self.aro_vec[:20],
                    lo_per_res=self.task.aro_bounds[0],
                    hi_per_res=self.task.aro_bounds[1],
                )
            )

    def get_linear_constraint_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stacked (G [M, 20], lo [M], hi [M]) for Part 1B Stage 2."""
        if not self.linear_constraints:
            empty = torch.zeros((0, 20), device=self.device, dtype=self.dtype)
            b = torch.zeros((0,), device=self.device, dtype=self.dtype)
            return empty, b, b
        G = torch.stack([c.g_aa20 for c in self.linear_constraints], dim=0)
        lo = torch.tensor([c.lo_per_res for c in self.linear_constraints], device=self.device, dtype=self.dtype)
        hi = torch.tensor([c.hi_per_res for c in self.linear_constraints], device=self.device, dtype=self.dtype)
        return G, lo, hi

    def get_omega(self) -> torch.Tensor:
        """Return unit-normalized Pareto weight vector omega [K] over active objectives."""
        K = max(len(self.objectives), 1)
        if self.task.omega is not None:
            w = torch.tensor(self.task.omega, device=self.device, dtype=self.dtype)
        else:
            w = torch.ones(K, device=self.device, dtype=self.dtype) / float(K)
        return w / (w.norm() + 1e-12)

    def evaluate_discrete_tokens(self, seq_ids: Sequence[int]) -> Dict[str, float]:
        """Compute exact intensive property values and residuals on an integer token list."""
        aa_ids = [t for t in seq_ids if t != PAD_ID]
        L = max(len(aa_ids), 1)
        padded = torch.tensor(
            aa_ids + [PAD_ID] * (self.L_max - len(aa_ids)), device=self.device, dtype=torch.long
        )
        out: Dict[str, float] = {"length": float(len(aa_ids))}
        for obj in self.objectives:
            a_sum = obj.a_tilde[padded].sum()
            phi_v = obj.phi[padded]
            pair_sum = 0.5 * (phi_v @ (obj.W @ phi_v))
            res = (a_sum + pair_sum).item()
            out[f"residual_{obj.name}"] = res
            out[f"value_{obj.name}"] = obj.target + (res / float(L))
        return out


def tokens_to_seq(seq_ids: Sequence[int]) -> str:
    """Convert integer token sequence to clean 1-letter amino acid string (stripping <PAD>)."""
    return "".join(AA20[t] for t in seq_ids if 0 <= t < 20)


def seq_to_tokens(seq: str, L_max: int) -> torch.Tensor:
    """Encode clean amino acid string into right-padded [L_max] integer tensor."""
    trimmed = seq[:L_max]
    ids = [AA_TO_ID[ch] for ch in trimmed] + [PAD_ID] * (L_max - len(trimmed))
    return torch.tensor(ids, dtype=torch.long)