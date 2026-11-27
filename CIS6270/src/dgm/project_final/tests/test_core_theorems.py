"""
Experiment tracking wrapper unifying Weights & Biases (`wandb`) and deterministic
local JSONL/JSON persistence for multi-seed ICML aggregation and offline plotting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dgm.project_final.config import EVAL_DIR, WANDB_DIR, ensure_project_dirs

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False


class ExperimentTracker:
    """
    Manages WandB run lifecycle + local JSONL step logs and summary JSON exports.
    Automatically points WANDB_DIR to /home/mukulsherekar/.../CIS6270/wandb.
    """

    def __init__(
        self,
        run_name: str,
        group: str,
        job_type: str,
        config: Dict[str, Any],
        tags: Optional[Sequence[str]] = None,
        project: str = "cis6270-responsive-idp-dgm",
        entity: Optional[str] = None,
        mode: str = "online",  # "online", "offline", or "disabled"
    ) -> None:
        ensure_project_dirs()
        os.environ["WANDB_DIR"] = str(WANDB_DIR)

        self.run_name = run_name
        self.local_dir = EVAL_DIR / "runs" / group
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.step_log_path = self.local_dir / f"{run_name}_steps.jsonl"
        self.summary_path = self.local_dir / f"{run_name}_summary.json"

        self.enabled = HAS_WANDB and (mode != "disabled")
        self.run = None
        if self.enabled:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                group=group,
                job_type=job_type,
                tags=list(tags) if tags else [],
                config=config,
                dir=str(WANDB_DIR),
                mode=mode,
                reinit=True,
            )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log step metrics both to local JSONL and to WandB."""
        record = {"step": step, **metrics}
        with self.step_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if self.enabled and self.run is not None:
            wandb.log(metrics, step=step)

    def log_sequence_table(
        self,
        sequences: Sequence[str],
        per_seq_metrics: Sequence[Dict[str, Any]],
        top_k: int = 25,
    ) -> None:
        """Upload top generated IDP candidates and their biophysical scores to WandB."""
        paired = list(zip(sequences, per_seq_metrics))
        paired.sort(key=lambda item: (not item[1].get("pass_all", False), item[1].get("total_mae", 999.0)))
        top_items = paired[:top_k]

        table_path = self.local_dir / f"{self.run_name}_top_sequences.json"
        with table_path.open("w", encoding="utf-8") as f:
            json.dump([{"sequence": s, **m} for s, m in top_items], f, indent=2)

        if self.enabled and self.run is not None and top_items:
            cols = ["rank", "sequence", "length", "pass_all", "disorder", "esm2_ppl"]
            rows: List[List[Any]] = []
            for r_idx, (seq, m) in enumerate(top_items):
                rows.append([
                    r_idx + 1,
                    seq,
                    len(seq),
                    bool(m.get("pass_all", False)),
                    float(m.get("disorder", -1.0)),
                    float(m.get("esm2_ppl", -1.0)),
                ])
            wandb.log({"candidates/top_sequences": wandb.Table(columns=cols, data=rows)})

    def log_figure(self, key: str, fig_path: Path) -> None:
        """Upload a saved PNG plot to WandB."""
        if self.enabled and self.run is not None and fig_path.exists():
            wandb.log({key: wandb.Image(str(fig_path))})

    def finish(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Persist final run summary to disk and close WandB run cleanly."""
        if summary is not None:
            with self.summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            if self.enabled and self.run is not None:
                for k, v in summary.items():
                    self.run.summary[k] = v
        if self.enabled and self.run is not None:
            wandb.finish()