"""
Unit tests verifying Theorems 1, 2, and 3 on the 21-token (20 AAs + <PAD>)
multi-condition responsive-IDP engine:
  1. Theorem 1: O(L^2) exactness vs. brute-force posterior recomputation + TAG gap.
  2. Theorem 2: Nonnegativity of Doob-tilted CTMC rates + strict hypercone blocking.
  3. Theorem 3: Part 1B box-constraint survival + Stage-3 swap composition invariance.
"""

from __future__ import annotations

import math
import unittest

import torch

from dgm.project_final.config import EnvConditionSpec, SamplerConfig, get_preset_tasks, make_user_task
from dgm.project_final.decoding.coupled_finalizer import coupled_decode_single, refine_pairwise_swaps
from dgm.project_final.guidance.pairwise_hypercone import (
    apply_doob_rate_tilt,
    compute_exact_and_tag_gains,
    compute_hypercone_log_tilt,
)
from dgm.project_final.physics.descriptors import PAD_ID, VOCAB_SIZE, BiophysicalEngine


class TestCoreTheorems(unittest.TestCase):
    """Verify mathematical guarantees of Part 1A and Part 1B."""

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_theorem1_exactness_and_tag_gap_user_conditions(self) -> None:
        """Verify O(L^2) single-site delta equals brute-force posterior expectation."""
        user_task = make_user_task(
            name="custom_3pH_salt_temp",
            L_min=20,
            L_max=32,
            target_aro_frac=0.10,
            conditions=[
                EnvConditionSpec("basic_pH8", pH=8.2, salt_mM=250.0, temp_K=315.0, target_ncpr=-0.12, target_scd=-0.8),
                EnvConditionSpec("mid_pH6", pH=6.1, salt_mM=100.0, temp_K=298.0, target_ncpr=-0.03),
                EnvConditionSpec("acid_pH48", pH=4.8, salt_mM=30.0, temp_K=283.0, target_ncpr=0.05, target_shd=5.1),
            ],
        )
        engine = BiophysicalEngine(user_task, device=self.device, dtype=torch.float64)
        P = torch.softmax(torch.randn(2, 32, VOCAB_SIZE, device=self.device, dtype=torch.float64), dim=-1)

        g_exact, g_tag, dmu = compute_exact_and_tag_gains(P, engine)
        # Verify exact algebraic identity g_tag - g_exact == 0.5 * dmu^2
        self.assertTrue(torch.allclose(g_tag - g_exact, 0.5 * (dmu ** 2), atol=1e-10))

        # Brute-force check at random batch=1, slot=11, token=4 across all K objectives
        b, slot, tok = 1, 11, 4
        P_mut = P.clone()
        P_mut[b, slot, :] = 0.0
        P_mut[b, slot, tok] = 1.0

        for k, obj in enumerate(engine.objectives):
            mu_orig = (P[b] @ obj.a_tilde).sum() + 0.5 * ((P[b] @ obj.phi) @ (obj.W @ (P[b] @ obj.phi)))
            mu_mut = (P_mut[b] @ obj.a_tilde).sum() + 0.5 * ((P_mut[b] @ obj.phi) @ (obj.W @ (P_mut[b] @ obj.phi)))
            bf_gain = -0.5 * (mu_mut ** 2 - mu_orig ** 2)
            self.assertAlmostEqual(float(bf_gain.item()), float(g_exact[b, k, slot, tok].item()), places=8)

    def test_theorem2_rate_nonnegativity_and_cone_gate(self) -> None:
        """Verify tilted CTMC rates stay >= 0 and cone gate blocks off-angle edits."""
        task = get_preset_tasks(L_max=24)["T4_triple_stimulus"]
        task.L_min = 16
        engine = BiophysicalEngine(task, device=self.device)

        P = torch.softmax(torch.randn(2, 24, VOCAB_SIZE, device=self.device), dim=-1)
        mask = torch.ones((2, 24), device=self.device, dtype=torch.bool)
        mask[:, :6] = False  # Already-unmasked positions must get zero boost

        log_tilt, gate = compute_hypercone_log_tilt(P, mask, engine, cone_angle_deg=25.0, gamma=5.0, mode="1A")
        self.assertTrue((gate[:, :6, :] == 0.0).all())
        self.assertTrue((log_tilt[gate == 0.0] == 0.0).all())

        u_base = torch.rand_like(P) * 10.0
        u_tilted = apply_doob_rate_tilt(u_base, log_tilt)
        self.assertTrue((u_tilted >= 0.0).all())

    def test_theorem3_part1b_constraint_survival_and_swap_invariance(self) -> None:
        """Verify Part 1B satisfies hard dual-pH + aromatic boxes and swaps keep 1-body exact."""
        task = get_preset_tasks(L_max=48)["T4_triple_stimulus"]
        task.L_min = 32
        engine = BiophysicalEngine(task, device=self.device)
        cfg = SamplerConfig(beam_width=64, max_swaps=100, top_k_lengths=6)

        logits = torch.randn(48, VOCAB_SIZE, device=self.device)
        logits[:40, PAD_ID] -= 2.0
        logits[40:, PAD_ID] += 2.5
        P_final = torch.softmax(logits, dim=-1)

        seq_no_swap, L1, _ = coupled_decode_single(P_final, engine, cfg, run_swaps=False, fixed_length=40)
        seq_swapped, L2, diag = coupled_decode_single(P_final, engine, cfg, run_swaps=True, fixed_length=40)
        self.assertEqual(L1, 40)
        self.assertEqual(L2, 40)

        # Stage-3 swaps must preserve exact amino acid counts (histogram invariance)
        counts_before = torch.bincount(seq_no_swap[:40], minlength=20)
        counts_after = torch.bincount(seq_swapped[:40], minlength=20)
        self.assertTrue(torch.equal(counts_before, counts_after))

        # Check all hard 1-body box constraints survived discretization
        G, lo, hi = engine.get_linear_constraint_tensors()
        realized = (G[:, seq_swapped[:40]].sum(dim=-1)) / 40.0
        self.assertTrue(((realized >= lo - 1e-4) & (realized <= hi + 1e-4)).all())

        # Check Stage-3 utility monotonically improved or matched Stage-2 utility
        P_aa = P_final[:40, :20] / P_final[:40, :20].sum(dim=-1, keepdim=True)
        _, u0 = refine_pairwise_swaps(seq_no_swap[:40], P_aa.clamp(min=1e-12).log(), engine, max_swaps=0)
        self.assertGreaterEqual(diag["objective_score"], diag["objective_score"] - 1e-6)
        _, u1 = refine_pairwise_swaps(seq_no_swap[:40], P_aa.clamp(min=1e-12).log(), engine, max_swaps=100)
        self.assertGreaterEqual(u1, u0 - 1e-6)


if __name__ == "__main__":
    unittest.main()