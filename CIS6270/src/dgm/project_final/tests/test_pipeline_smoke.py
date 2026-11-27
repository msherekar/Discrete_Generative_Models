"""
End-to-end GPU smoke test verifying the full pipeline across Modules 1-8:
  - Pan-organism dataset builder + 30% clustering (`prepare_data.py`)
  - Track A MDLM training + Part 1A + Part 1B guided variable-length sampling
  - Track B LSD Simplex Flow Map Identity Axiom + `torch.func.jvp` step + latent redesign
  - Independent sequence evaluation + multi-seed figure & LaTeX table generation
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from dgm.project_final.config import SamplerConfig, get_preset_tasks
from dgm.project_final.data.prepare_data import create_synthetic_pan_organism_splits
from dgm.project_final.eval.evaluate import evaluate_sequences
from dgm.project_final.models.dplm_sampler import (
    LocalMaskedIDPDenoiser,
    mdlm_train_step,
    sample_track_a,
)
from dgm.project_final.models.lsd_flow_map import (
    LSDSimplexFlowMap,
    decode_flow_map_batch,
    invert_and_redesign_disprot,
    lsd_train_step,
    make_ema_copy,
    sample_flow_map,
)
from dgm.project_final.physics.descriptors import BiophysicalEngine, tokens_to_seq
from dgm.project_final.scripts.make_plots import generate_all_outputs


class TestEndToEndPipelineSmoke(unittest.TestCase):
    """Verify full data -> training -> guided sampling -> eval -> plotting pipeline."""

    def setUp(self) -> None:
        torch.manual_seed(11)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_full_pipeline_on_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proc_dir = tmp_path / "data/processed"
            counts = create_synthetic_pan_organism_splits(
                out_dir=proc_dir, n_seqs=80, L_min=24, L_max=40, seed=11
            )
            self.assertGreater(counts["train"], 20)

            train_pt = torch.load(proc_dir / "train.pt", weights_only=False)
            disprot_pt = torch.load(proc_dir / "disprot.pt", weights_only=False)
            train_tokens = train_pt["tokens"][:16].to(self.device)

            task = get_preset_tasks(L_max=40)["T1_dual_pH"]
            task.L_min = 24
            engine = BiophysicalEngine(task, device=self.device)
            cfg = SamplerConfig(num_steps=8, beam_width=24, max_swaps=30, top_k_lengths=4)

            # 1. Track A: 1 training step + 1A+1B guided variable-length sampling
            denoiser = LocalMaskedIDPDenoiser(L_max=40, d_model=64, n_heads=4, n_layers=2).to(self.device)
            opt_a = torch.optim.AdamW(denoiser.parameters(), lr=1e-3)
            loss_a = mdlm_train_step(denoiser, train_tokens, opt_a)
            self.assertTrue(loss_a > 0.0)

            toks_a, _, diag = sample_track_a(
                denoiser, engine, cfg, batch_size=4, mode="1A",
                use_part_1b=True, record_tag_diag=True,
            )
            seqs_a = [tokens_to_seq(r.tolist()) for r in toks_a]
            summary_a, _ = evaluate_sequences(seqs_a, task, train_pt["sequences"])
            self.assertEqual(summary_a["pass_rate_all"], 100.0)
            self.assertIn("exact_gain", diag)

            # 2. Track B: Identity Axiom + `torch.func.jvp` LSD step + 1-step & DisProt redesign
            fmap = LSDSimplexFlowMap(L_max=40, d_model=64, n_heads=4, n_layers=2).to(self.device)
            ema = make_ema_copy(fmap)
            x_rand = torch.randn(3, 40, 21, device=self.device)
            s_same = torch.full((3,), 0.42, device=self.device)
            self.assertTrue(torch.allclose(fmap(x_rand, s_same, s_same), x_rand, atol=1e-6))

            opt_b = torch.optim.AdamW(fmap.parameters(), lr=1e-3)
            stats_b = lsd_train_step(fmap, ema, train_tokens, opt_b)
            self.assertTrue(stats_b["loss_total"] > 0.0)

            P_1step = sample_flow_map(ema, batch_size=4, nfe=1, device=self.device)
            rows_b = decode_flow_map_batch(P_1step, engine, cfg, decoder_mode="1B_full")
            seqs_b = [tokens_to_seq(r.tolist()) for r in rows_b]
            summary_b, _ = evaluate_sequences(seqs_b, task)
            self.assertEqual(summary_b["pass_rate_all"], 100.0)

            if disprot_pt["tokens"].size(0) > 0:
                redesigned, _ = invert_and_redesign_disprot(
                    ema, disprot_pt["tokens"][:2], engine, cfg, nfe=2
                )
                self.assertEqual(len(redesigned), 2)

            # 3. Verify plotting + LaTeX table exporter
            eval_tmp = tmp_path / "eval"
            runs_dir = eval_tmp / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            torch.save(diag, runs_dir / "T1_dual_pH_tag_vs_1a_diag.pt")
            for idx, (m_name, trk, ab, ang, nfe) in enumerate([
                ("unguided", "trackA", "main", 30.0, 8),
                ("1A+1B", "trackA", "main", 30.0, 8),
                ("1A+1B_fixedL", "trackA", "length_ablation", 30.0, 8),
                ("1A_cone30", "trackA", "cone_ablation", 30.0, 8),
                ("argmax", "trackB", "flowmap_nfe", 30.0, 1),
                ("1B_full", "trackB", "flowmap_nfe", 30.0, 1),
            ]):
                rec = {
                    **summary_a, "task": "T1_dual_pH", "track": trk, "method": m_name,
                    "ablation": ab, "cone_angle_deg": ang, "nfe": nfe, "seed": 0,
                }
                (runs_dir / f"r{idx}_summary.json").write_text(json.dumps(rec), encoding="utf-8")

            artifacts = generate_all_outputs(eval_tmp)
            self.assertGreaterEqual(len(artifacts), 5)


if __name__ == "__main__":
    unittest.main()