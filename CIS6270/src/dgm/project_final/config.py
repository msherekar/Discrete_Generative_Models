"""
Central configuration, filesystem paths, and user-defined environmental task specs
for responsive intrinsically disordered protein (IDP) discrete generation.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# 1. Canonical Project Paths (with environment-variable override support)
# ==============================================================================
BASE_CIS_DIR = Path("/home/mukulsherekar/Discrete_Generative_Models/CIS6270")

CODE_DIR = Path(os.environ.get("IDP_CODE_DIR", BASE_CIS_DIR / "src/dgm/project_final"))
EVAL_DIR = Path(os.environ.get("IDP_EVAL_DIR", BASE_CIS_DIR / "eval_project_final"))
DATA_DIR = Path(os.environ.get("IDP_DATA_DIR", BASE_CIS_DIR / "data/project_final"))
WANDB_DIR = Path(os.environ.get("IDP_WANDB_DIR", BASE_CIS_DIR / "wandb"))


def ensure_project_dirs() -> None:
    """Create evaluation, data, and WandB output directories if missing."""
    for folder in (
        EVAL_DIR / "runs",
        EVAL_DIR / "figures",
        EVAL_DIR / "tables",
        DATA_DIR / "raw",
        DATA_DIR / "processed",
        WANDB_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. User-Defined Environmental Condition & Multi-Objective Task Specifications
# ==============================================================================
@dataclass
class EnvConditionSpec:
    """
    Defines a single user-chosen physical environment (pH, salt, temperature)
    and optional biophysical sequence targets under that environment.

    Users may pass ANY number of EnvConditionSpec objects to a TaskSpec
    (e.g., 1, 2, 3, or more pH/salt/temperature states).
    """

    name: str
    pH: float = 7.4
    salt_mM: float = 150.0
    temp_K: float = 310.0

    # Intensive targets (per active residue) for Part 1A & Stage-3 swaps
    target_ncpr: Optional[float] = None
    target_scd: Optional[float] = None
    target_shd: Optional[float] = None
    target_fcr: Optional[float] = None

    # Optional hard per-residue [min, max] windows enforced in Part 1B Stage 2
    ncpr_bounds: Optional[Tuple[float, float]] = None
    fcr_bounds: Optional[Tuple[float, float]] = None

    # SHD distance-decay exponent |j - i|^(-shd_beta)
    shd_beta: float = 1.0


@dataclass
class TaskSpec:
    """
    Complete multi-condition responsive-IDP design specification.
    Supports variable chain length in [L_min, L_max] up to canvas L_max.
    """

    name: str
    conditions: List[EnvConditionSpec]
    L_min: int = 40
    L_max: int = 128

    # Optional sequence-wide aromatic sticker controls (Y, F, W)
    target_aro_frac: Optional[float] = None
    aro_bounds: Optional[Tuple[float, float]] = None
    min_sticker_sep: int = 5
    penalize_sticker_clumps: bool = False

    # Part 1A multi-objective hypercone & Doob h-transform hyperparameters
    omega: Optional[List[float]] = None  # Defaults to uniform Pareto weights
    cone_angle_deg: float = 30.0
    gamma: float = 4.0

    # Part 1B Stage-3 quadratic penalty weight on 2-body residuals
    swap_beta: float = 1.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task specification cleanly for JSON and WandB logging."""
        return asdict(self)


@dataclass
class SamplerConfig:
    """
    Inference and ablation configuration. Note that `num_steps` is fully
    user-configurable (not locked to 64) for step-budget sweeps.
    """

    num_steps: int = 64
    track_a_step_sweep: Tuple[int, ...] = (32, 64, 128)
    flow_map_nfe_grid: Tuple[int, ...] = (1, 2, 4, 16, 64)

    # Stochastic zero-flux remasking corrector rate (0.0 = pure unmasking)
    remask_eta: float = 0.0
    temperature: float = 1.0

    # Part 1B coupled decoding controls
    beam_width: int = 64
    sinkhorn_steps: int = 150
    sinkhorn_lr: float = 0.25
    max_swaps: int = 150
    top_k_lengths: int = 16

    # Multi-seed ICML evaluation settings
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    samples_per_seed: int = 500


# ==============================================================================
# 3. Preset Benchmark Tasks (T1-T4) & Custom User Task Builder
# ==============================================================================
def make_user_task(
    name: str,
    conditions: List[EnvConditionSpec],
    L_min: int = 40,
    L_max: int = 128,
    target_aro_frac: Optional[float] = None,
    aro_bounds: Optional[Tuple[float, float]] = None,
    cone_angle_deg: float = 30.0,
    gamma: float = 4.0,
) -> TaskSpec:
    """Helper to build a custom user-defined (pH, salt, temp) design task."""
    return TaskSpec(
        name=name,
        conditions=conditions,
        L_min=L_min,
        L_max=L_max,
        target_aro_frac=target_aro_frac,
        aro_bounds=aro_bounds,
        cone_angle_deg=cone_angle_deg,
        gamma=gamma,
    )


def get_preset_tasks(L_max: int = 128) -> Dict[str, TaskSpec]:
    """Return physically calibrated ICML responsive-material benchmark tasks (T1-T4)."""
    t1 = TaskSpec(
        name="T1_dual_pH",
        L_min=40,
        L_max=L_max,
        target_aro_frac=0.10,
        aro_bounds=(0.06, 0.14),
        conditions=[
            EnvConditionSpec(
                name="phys_pH74",
                pH=7.4,
                salt_mM=150.0,
                temp_K=310.0,
                target_ncpr=-0.10,
                ncpr_bounds=(-0.14, -0.06),
            ),
            EnvConditionSpec(
                name="acidic_pH55",
                pH=5.5,
                salt_mM=150.0,
                temp_K=310.0,
                target_ncpr=0.00,
                ncpr_bounds=(-0.04, 0.04),
            ),
        ],
    )

    t2 = TaskSpec(
        name="T2_salt_switch",
        L_min=40,
        L_max=L_max,
        conditions=[
            EnvConditionSpec(
                name="low_salt_10mM",
                pH=7.4,
                salt_mM=10.0,
                temp_K=298.15,
                target_ncpr=0.00,
                ncpr_bounds=(-0.05, 0.05),
                target_fcr=0.36,
                fcr_bounds=(0.28, 0.45),
                target_scd=-0.55,
            ),
            EnvConditionSpec(
                name="high_salt_500mM",
                pH=7.4,
                salt_mM=500.0,
                temp_K=298.15,
                target_scd=0.06,
            ),
        ],
    )

    t3 = TaskSpec(
        name="T3_thermo_assembly",
        L_min=40,
        L_max=L_max,
        target_aro_frac=0.11,
        aro_bounds=(0.07, 0.15),
        penalize_sticker_clumps=True,
        min_sticker_sep=5,
        conditions=[
            EnvConditionSpec(
                name="cool_285K",
                pH=7.0,
                salt_mM=150.0,
                temp_K=285.0,
                target_ncpr=0.00,
                ncpr_bounds=(-0.05, 0.05),
                target_shd=1.08,
            ),
            EnvConditionSpec(
                name="warm_330K",
                pH=7.0,
                salt_mM=150.0,
                temp_K=330.0,
                target_shd=1.20,
            ),
        ],
    )

    t4 = TaskSpec(
        name="T4_triple_stimulus",
        L_min=40,
        L_max=L_max,
        target_aro_frac=0.10,
        aro_bounds=(0.06, 0.14),
        penalize_sticker_clumps=True,
        min_sticker_sep=5,
        conditions=[
            EnvConditionSpec(
                name="soluble_pH74_highSalt",
                pH=7.4,
                salt_mM=400.0,
                temp_K=310.0,
                target_ncpr=-0.10,
                ncpr_bounds=(-0.14, -0.06),
                target_scd=0.05,
            ),
            EnvConditionSpec(
                name="assembly_pH55_lowSalt",
                pH=5.5,
                salt_mM=15.0,
                temp_K=290.0,
                target_ncpr=0.00,
                ncpr_bounds=(-0.04, 0.04),
                target_scd=-0.40,
                target_shd=1.12,
            ),
        ],
    )
    return {t.name: t for t in (t1, t2, t3, t4)}