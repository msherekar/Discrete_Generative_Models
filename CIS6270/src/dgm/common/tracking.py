"""Weights & Biases logging, kept entirely outside the experiment code.

Nothing in pipeline/ or project1/ knows this module exists. Everything logged
here is read back out of the results.pt that save_results already writes, which
has two consequences worth knowing:

  - Loss curves come out at full resolution. run_experiment prints only four or
    five epoch lines, but it stores every epoch in results["losses"], so a
    250-epoch run logs 250 points.
  - Logging works after the fact, from anywhere with network access. That is
    what makes OSG runs loggable: the worker node needs no credentials and no
    wandb in its image.

Defaults are this group's: entity "proterial", project "CIS6270". Both are
overridable with WANDB_ENTITY and WANDB_PROJECT.
"""
import os
import platform
from pathlib import Path

DEFAULT_ENTITY = "proterial"
DEFAULT_PROJECT = "CIS6270"
METHODS = ("flow", "diffusion")


def environment_info() -> dict:
    """Host and accelerator facts, so a run records what produced it."""
    info = {
        "host": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    # OSPool sets this on the worker; absent locally.
    for key in ("GLIDEIN_ResourceName", "GLIDEIN_Site"):
        if os.environ.get(key):
            info[key.lower()] = os.environ[key]
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = ".".join(
                str(x) for x in torch.cuda.get_device_capability(0))
    except Exception:                                   # noqa: BLE001
        pass
    return info


def load_results(run_dir: Path) -> dict:
    """Both modalities' results.pt from a run directory, keyed by method.

    A modality whose results.pt is missing is simply absent, so a half-finished
    run still logs the half that exists.
    """
    import torch
    found = {}
    for method in METHODS:
        path = run_dir / method / "results.pt"
        if path.is_file() and path.stat().st_size > 0:
            found[method] = torch.load(path, weights_only=False,
                                       map_location="cpu")
    return found


def _hamming(sequences, reference) -> list:
    return [sum(a != b for a, b in zip(s, reference)) for s in sequences]


def arm_metrics(results: dict, reference=None) -> dict:
    """Per-arm summary for one modality: brightness, divergence, diversity."""
    rows = {}
    for arm, seqs in results.get("sequences", {}).items():
        row = {
            "n": len(seqs),
            "unique": len(set(seqs)),
            "unique_frac": len(set(seqs)) / max(1, len(seqs)),
        }
        scores = (results.get("oracle_brightness") or {}).get(arm)
        if scores is not None and len(scores):
            row["oracle_mean"] = float(scores.mean())
            row["oracle_best"] = float(scores.max())
            row["oracle_worst"] = float(scores.min())
        if reference:
            distances = _hamming(seqs, reference)
            row["hamming_mean"] = sum(distances) / len(distances)
            row["hamming_min"] = min(distances)
            row["hamming_max"] = max(distances)
        rows[arm] = row
    return rows


def build_config(found: dict, extra: dict = None) -> dict:
    """The W&B config: the run's own saved configuration plus the environment.

    run_config is identical across modalities -- save_results writes the same
    dict into both -- so either copy serves.
    """
    config = {}
    for results in found.values():
        config.update(results.get("config") or {})
        break
    config.update({f"env/{k}": v for k, v in environment_info().items()})
    for method, results in found.items():
        config[f"{method}/epochs_recorded"] = len(results.get("losses") or [])
    if extra:
        config.update(extra)
    return config


def start_run(name, group=None, tags=None, config=None, entity=None,
              project=None, mode=None, notes=None):
    """Open a W&B run before the experiment starts, so it shows as running."""
    import wandb
    return wandb.init(
        entity=entity or os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY),
        project=project or os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
        name=name, group=group, job_type="experiment",
        config=config or {"env/" + k: v for k, v in environment_info().items()},
        tags=list(tags or []), notes=notes,
        mode=mode or os.environ.get("WANDB_MODE", "online"), reinit=True,
    )


def log_run(run_dir: Path, name=None, group=None, tags=None, extra=None,
            entity=None, project=None, mode=None, artifacts=True,
            notes=None, run=None) -> str:
    """Log one finished run directory to W&B and return its URL.

    Reads results.pt rather than instrumenting the training loop, so this can
    run on the machine that produced the run, on an access point, or on a
    laptop weeks later. Pass `run` to fill in a run already opened by
    start_run, which is how the live wrapper reuses one.
    """
    import wandb

    run_dir = Path(run_dir).expanduser().resolve()
    found = load_results(run_dir)
    if not found:
        raise SystemExit(
            f"no results.pt under {run_dir}/{{flow,diffusion}}; "
            f"nothing to log. For an OSG run, copy them from OSDF first.")

    config = build_config(found, extra)
    reference = config.get("reference")

    owns_run = run is None
    if owns_run:
        run = start_run(name or run_dir.name, group=group, tags=tags,
                        config=config, entity=entity, project=project,
                        mode=mode, notes=notes)
    else:
        # The wrapper opened the run before the experiment, with only the
        # environment known; the saved configuration arrives now.
        run.config.update(config, allow_val_change=True)

    # Loss curves, one series per modality, at the epoch resolution the run
    # actually recorded rather than the four lines it printed.
    for method, results in found.items():
        wandb.define_metric(f"{method}/loss", step_metric="epoch")
    longest = max((len(r.get("losses") or []) for r in found.values()),
                  default=0)
    for epoch in range(longest):
        row = {"epoch": epoch + 1}
        for method, results in found.items():
            losses = results.get("losses") or []
            if epoch < len(losses):
                row[f"{method}/loss"] = losses[epoch]
        wandb.log(row)

    # Per-arm outcomes: a table for comparing arms, plus flat summary keys so
    # they are sortable and sweepable in the W&B UI.
    columns = ["method", "arm", "n", "unique", "unique_frac", "oracle_mean",
               "oracle_best", "oracle_worst", "hamming_mean", "hamming_min",
               "hamming_max"]
    table = wandb.Table(columns=columns)
    for method, results in found.items():
        for arm, row in arm_metrics(results, reference).items():
            table.add_data(method, arm, *(row.get(c) for c in columns[2:]))
            for key, value in row.items():
                run.summary[f"{method}/{arm}/{key}"] = value
        losses = results.get("losses") or []
        if losses:
            run.summary[f"{method}/final_loss"] = losses[-1]
            run.summary[f"{method}/best_loss"] = min(losses)
    wandb.log({"arms": table})

    if artifacts:
        _log_artifacts(run, run_dir)

    url = run.url
    if owns_run:
        run.finish()
    return url


def _log_artifacts(run, run_dir: Path) -> None:
    """Attach the small, human-readable outputs; never the multi-hundred-MB
    results.pt, which belongs in OSDF rather than in W&B storage."""
    import wandb

    artifact = wandb.Artifact(f"{run_dir.name}-samples", type="samples")
    added = 0
    for pattern in ("*/*.fasta", "*.txt", "*.json"):
        for path in sorted(run_dir.glob(pattern)):
            artifact.add_file(str(path), name=str(path.relative_to(run_dir)))
            added += 1
    plots = run_dir / "plots"
    if plots.is_dir():
        for path in sorted(plots.rglob("*")):
            if path.is_file() and path.suffix in (".png", ".csv"):
                artifact.add_file(str(path),
                                  name=str(path.relative_to(run_dir)))
                added += 1
    if added:
        run.log_artifact(artifact)
