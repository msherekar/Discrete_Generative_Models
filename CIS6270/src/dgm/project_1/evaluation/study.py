"""Figures the project's claims need but the standard plot set does not cover.

Three per-run figures, each answering a question the printed tables leave as
numbers:

  setpoint calibration   achieved brightness against requested, with a fitted
                         slope. cli.py asks for exactly this -- "plot achieved
                         against requested and report the slope" -- because
                         steering GFP dimmer is a null task and hitting a
                         requested value is the stronger claim.
  reward vs oracle       the reward head's own prediction against an
                         independent oracle, per guidance strength. Guidance
                         climbs the reward model; if that model and the oracle
                         disagree more as eta rises, the gain is reward hacking
                         rather than brightness. analyze_sweep has this figure
                         for images only.
  lambda Pareto front    achieved brightness against achieved substitution
                         count across --reward-lambda. reporting.py prints this
                         table and notes a front needs both columns to move.

All three read a finished run directory, so none of them touches the
experiment code or needs the training process.
"""
import csv
import re

import numpy as np

from .output import _save
from .style import METHOD_COLORS, METHOD_MARKERS, plt

METHODS = ("flow", "diffusion")
# "sp50", "spp99", "spp50@100" -- optional percentile p, optional @eta suffix.
SETPOINT_ARM = re.compile(r"^spp?(?P<value>[\d.]+)(?:@(?P<eta>[\d.]+))?$")
LAMBDA_ARM = re.compile(r"^lam(?P<value>[\d.]+)(?:@(?P<eta>[\d.]+))?$")
ETA_ARM = re.compile(r"^(?P<mode>single|multi)(?:@(?P<eta>[\d.]+))?$")


def _write(rows, columns, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _hamming(sequences, reference):
    return np.array([sum(a != b for a, b in zip(s, reference)) for s in sequences])


def setpoint_calibration(runs, scores, outdir, prefix):
    """Achieved brightness against the brightness that was asked for.

    A slope near 1 through the identity line means the controller tracks the
    request; a flat line means every setpoint collapsed to the same output,
    which is what saturation looks like.
    """
    rows = []
    for method, results in runs.items():
        config = results.get("config") or {}
        given = config.get("setpoints_given") or []
        raw = config.get("setpoints_raw") or []
        if not raw:
            continue
        requested = dict(zip((f"{g:g}" for g in given), raw))
        for arm, achieved in scores.get(method, {}).items():
            match = SETPOINT_ARM.match(arm)
            if not match or achieved is None or not len(achieved):
                continue
            target = requested.get(match["value"])
            if target is None:
                continue
            rows.append({"method": method, "arm": arm,
                         "requested_given": float(match["value"]),
                         "requested_raw": float(target),
                         "achieved_mean": float(np.mean(achieved)),
                         "achieved_sd": float(np.std(achieved)),
                         "n": len(achieved),
                         "eta": float(match["eta"]) if match["eta"] else None})
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    span = [min(r["requested_raw"] for r in rows), max(r["requested_raw"] for r in rows)]
    pad = 0.1 * (span[1] - span[0] or 1.0)
    line = np.array([span[0] - pad, span[1] + pad])
    ax.plot(line, line, color="k", lw=0.9, ls=":", label="perfect calibration")

    for method in METHODS:
        block = [r for r in rows if r["method"] == method]
        if not block:
            continue
        x = np.array([r["requested_raw"] for r in block])
        y = np.array([r["achieved_mean"] for r in block])
        err = np.array([r["achieved_sd"] for r in block])
        ax.errorbar(x, y, yerr=err, marker=METHOD_MARKERS[method], capsize=4,
                    ls="none", color=METHOD_COLORS[method], label=method)
        if len(set(x)) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            ax.plot(line, slope * line + intercept, lw=1.4, alpha=0.8,
                    color=METHOD_COLORS[method],
                    label=f"{method} fit: slope {slope:+.2f}")
            for r in block:
                r["fitted_slope"] = float(slope)

    ax.set_xlabel("requested brightness (raw r1 units)")
    ax.set_ylabel("achieved brightness (oracle, mean of samples)")
    ax.set_title("Setpoint calibration: does the request control the outcome?")
    ax.legend(fontsize=9)
    _save(fig, outdir, f"{prefix}_11_setpoint_calibration")
    _write(rows, ["method", "arm", "requested_given", "requested_raw",
                  "achieved_mean", "achieved_sd", "n", "eta", "fitted_slope"],
           outdir / f"{prefix}_setpoint_calibration.csv")
    return rows


def _reward_head(results):
    """Rebuild the run's reward head from its saved weights, or None.

    The output width comes from the saved tensor rather than the config, so a
    two- or three-property run both load.
    """
    from ..pipeline.nets import RewardModel
    state = results.get("reward_model")
    if not state:
        return None
    config = results.get("config") or {}
    final = [v for k, v in state.items() if k.endswith("bias")][-1]
    model = RewardModel(results["length"], results["dim"],
                        config.get("hidden", 128), config.get("arch", "mlp"),
                        n_props=final.numel())
    model.load_state_dict(state)
    return model.eval().requires_grad_(False)


def reward_vs_oracle(runs, scores, outdir, prefix):
    """The reward head's own prediction against an independent oracle.

    Guidance climbs the reward model, so agreement with the oracle is what
    separates a real brightness gain from reward hacking. Plotted as rank
    correlation against guidance strength: a correlation that falls as eta
    rises means the two measures are diverging exactly where the gain is
    claimed.
    """
    from scipy.stats import spearmanr

    rows = []
    for method, results in runs.items():
        head = _reward_head(results)
        latents = results.get("standardized_latents") or {}
        if head is None or not latents:
            continue
        for arm, achieved in scores.get(method, {}).items():
            z = latents.get(arm)
            if z is None or achieved is None or len(achieved) < 3:
                continue
            import torch
            with torch.no_grad():
                predicted = head(z.float().cpu(), torch.ones(len(z)))[:, 0].numpy()
            if not np.isfinite(predicted).all() or not np.isfinite(achieved).all():
                continue
            rho = spearmanr(predicted, achieved).statistic
            match = ETA_ARM.match(arm)
            rows.append({
                "method": method, "arm": arm,
                "eta": float(match["eta"]) if match and match["eta"] else None,
                "spearman": float(rho) if np.isfinite(rho) else None,
                "reward_mean": float(predicted.mean()),
                "oracle_mean": float(np.mean(achieved)), "n": len(achieved)})
    if not rows:
        return None

    scored = [r for r in rows if r["spearman"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # Left: every arm's agreement, as a bar per arm.
    ax = axes[0]
    width = 0.38
    for offset, method in enumerate(METHODS):
        block = [r for r in scored if r["method"] == method]
        if not block:
            continue
        names = [r["arm"] for r in block]
        index = np.arange(len(names)) + offset * width
        ax.bar(index, [r["spearman"] for r in block], width,
               color=METHOD_COLORS[method], label=method)
        ax.set_xticks(np.arange(len(names)) + width / 2)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Spearman, reward head vs oracle")
    ax.set_title("Agreement per arm")
    ax.legend(fontsize=9)

    # Right: the same against guidance strength, which is the hacking signal.
    ax = axes[1]
    for method in METHODS:
        block = sorted((r for r in scored
                        if r["method"] == method and r["eta"] is not None),
                       key=lambda r: r["eta"])
        if len(block) < 2:
            continue
        ax.plot([r["eta"] for r in block], [r["spearman"] for r in block],
                marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                label=method)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("guidance strength eta")
    ax.set_ylabel("Spearman, reward head vs oracle")
    ax.set_title("Falling with eta is reward hacking")
    if any(r["eta"] for r in scored):
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "needs more than one --reward-eta",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)

    fig.suptitle("Is the gain real brightness, or the reward model's opinion?",
                 fontsize=13)
    _save(fig, outdir, f"{prefix}_12_reward_vs_oracle")
    _write(rows, ["method", "arm", "eta", "spearman", "reward_mean",
                  "oracle_mean", "n"],
           outdir / f"{prefix}_reward_vs_oracle.csv")
    return rows


def lambda_pareto(runs, scores, outdir, prefix):
    """Achieved brightness against achieved substitution count, across lambda.

    reporting.py prints these columns and says what to look for: "A front needs
    both columns to move. A flat mutation column means parsimony had no room to
    act." Plotted, that is immediate.
    """
    rows = []
    for method, results in runs.items():
        config = results.get("config") or {}
        reference = config.get("reference")
        if not reference:
            continue
        for arm, seqs in (results.get("sequences") or {}).items():
            match = LAMBDA_ARM.match(arm)
            achieved = scores.get(method, {}).get(arm)
            if not match or achieved is None or not len(achieved):
                continue
            distances = _hamming(seqs, reference)
            rows.append({"method": method, "arm": arm,
                         "lambda": float(match["value"]),
                         "eta": float(match["eta"]) if match["eta"] else None,
                         "mutations_mean": float(distances.mean()),
                         "brightness_mean": float(np.mean(achieved)),
                         "brightness_best": float(np.max(achieved)),
                         "unique": len(set(seqs)), "n": len(seqs)})
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    for method in METHODS:
        block = sorted((r for r in rows if r["method"] == method),
                       key=lambda r: r["lambda"])
        if not block:
            continue
        x = [r["mutations_mean"] for r in block]
        y = [r["brightness_mean"] for r in block]
        ax.plot(x, y, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                label=method, alpha=0.9)
        for r in block:
            ax.annotate(f"$\\lambda$={r['lambda']:g}",
                        (r["mutations_mean"], r["brightness_mean"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("achieved substitutions from reference (mean)")
    ax.set_ylabel("achieved brightness (oracle, mean)")
    ax.set_title("Objective trade-off: a front needs both axes to move")
    ax.legend(fontsize=9)
    _save(fig, outdir, f"{prefix}_13_lambda_pareto")
    _write(rows, ["method", "arm", "lambda", "eta", "mutations_mean",
                  "brightness_mean", "brightness_best", "unique", "n"],
           outdir / f"{prefix}_lambda_pareto.csv")
    return rows
