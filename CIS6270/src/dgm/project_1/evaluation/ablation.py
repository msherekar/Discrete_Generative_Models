"""Plot 9: the guidance-strength ablation.

Re-samples the already-trained models across a CFG-weight and reward-eta grid,
so the sweep costs sampling passes rather than training runs.
"""
from pathlib import Path

import numpy as np

from .models import predict_reward, sample_diffusion, sample_flow
from .output import _save, _write_csv
from .style import METHOD_COLORS, plt

def plot_guidance_ablation(flow_model, flow_reward, diff_model, diff_reward,
                            diff_schedule, flow_stats, diff_stats, out_dir,
                            prefix="", path=None, steps=200):
    """Vary CFG weight w and reward eta; plot + save predicted r1, r2.

    `path` is the run's PathSpec. It has to be passed in rather than defaulted:
    re-sampling a model along a different path than it was trained on produces
    numbers for a model that does not exist, which is what this figure did
    before the sampler was shared with pipeline/.
    """
    betas, alphas, alpha_bars, post_vars = diff_schedule
    out_dir  = Path(out_dir)

    w_vals   = [0.0, 0.5, 1.0, 2.0, 3.0]
    eta_vals = [0.0, 0.5, 1.0, 2.0]

    def get_pred(z, reward_model, stats):
        return predict_reward(reward_model, z, stats).numpy()

    # Collect raw numbers for both sweeps so we can write CSVs
    cfg_rows, eta_rows = [], []

    # --- CFG sweep (eta=0) ---
    cfg_results = {}   # method -> {"means_r1": [...], "stds_r1": [...], ...}
    for method, model, reward, stats, sampler in [
        ("flow",      flow_model,  flow_reward,  flow_stats,
         lambda m, r, w: sample_flow(m, r, n=8, c=1, w=w, eta=0,
                                     path=path, steps=steps)),
        ("diffusion", diff_model, diff_reward, diff_stats,
         lambda m, r, w: sample_diffusion(m, r, alpha_bars, betas, alphas, post_vars,
                                          n=8, c=1, w=w, eta=0)),
    ]:
        mr1, sr1, mr2, sr2 = [], [], [], []
        for w in w_vals:
            pred = get_pred(sampler(model, reward, w), reward, stats)
            mr1.append(pred[:, 0].mean()); sr1.append(pred[:, 0].std())
            mr2.append(pred[:, 1].mean()); sr2.append(pred[:, 1].std())
            cfg_rows.append({
                "method": method, "cfg_weight_w": w,
                "mean_r1": round(float(pred[:, 0].mean()), 6),
                "std_r1":  round(float(pred[:, 0].std()),  6),
                "mean_r2": round(float(pred[:, 1].mean()), 6),
                "std_r2":  round(float(pred[:, 1].std()),  6),
            })
        cfg_results[method] = {
            "mr1": np.array(mr1), "sr1": np.array(sr1),
            "mr2": np.array(mr2), "sr2": np.array(sr2),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key_m, key_s, ylabel in zip(
        axes,
        ["mr1", "mr2"], ["sr1", "sr2"],
        ["Predicted r₁ (charge proxy)", "Predicted r₂ (polar fraction)"],
    ):
        for method in ("flow", "diffusion"):
            r = cfg_results[method]
            ax.plot(w_vals, r[key_m], color=METHOD_COLORS[method], marker="o",
                    lw=2, label=method.capitalize())
            ax.fill_between(w_vals, r[key_m] - r[key_s], r[key_m] + r[key_s],
                            alpha=0.15, color=METHOD_COLORS[method])
        ax.set_xlabel("CFG weight w"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs CFG weight (η=0)")
        ax.legend(); ax.axhline(0, color="gray", lw=0.7, ls=":")
    fig.suptitle("Ablation: CFG Guidance Strength (predicted rewards)", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir, "09a_ablation_cfg_weight.png", prefix)
    _write_csv(out_dir / f"{prefix}_ablation_cfg_weight.csv", cfg_rows)

    # --- Reward eta sweep (w=0, lambda=(1,0)) ---
    eta_results = {}
    for method, model, reward, stats, sampler in [
        ("flow",      flow_model,  flow_reward,  flow_stats,
         lambda m, r, e: sample_flow(m, r, n=8, c=1, w=0, eta=e,
                                     lambdas=(1., 0.), path=path,
                                     steps=steps)),
        ("diffusion", diff_model, diff_reward, diff_stats,
         lambda m, r, e: sample_diffusion(m, r, alpha_bars, betas, alphas, post_vars,
                                          n=8, c=1, w=0, eta=e, lambdas=(1., 0.))),
    ]:
        mr1, sr1, mr2, sr2 = [], [], [], []
        for eta in eta_vals:
            pred = get_pred(sampler(model, reward, eta), reward, stats)
            mr1.append(pred[:, 0].mean()); sr1.append(pred[:, 0].std())
            mr2.append(pred[:, 1].mean()); sr2.append(pred[:, 1].std())
            eta_rows.append({
                "method": method, "reward_eta": eta,
                "mean_r1": round(float(pred[:, 0].mean()), 6),
                "std_r1":  round(float(pred[:, 0].std()),  6),
                "mean_r2": round(float(pred[:, 1].mean()), 6),
                "std_r2":  round(float(pred[:, 1].std()),  6),
            })
        eta_results[method] = {
            "mr1": np.array(mr1), "sr1": np.array(sr1),
            "mr2": np.array(mr2), "sr2": np.array(sr2),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key_m, key_s, ylabel in zip(
        axes,
        ["mr1", "mr2"], ["sr1", "sr2"],
        ["Predicted r₁ (charge proxy)", "Predicted r₂ (polar fraction)"],
    ):
        for method in ("flow", "diffusion"):
            r = eta_results[method]
            ax.plot(eta_vals, r[key_m], color=METHOD_COLORS[method], marker="^",
                    lw=2, linestyle="--", label=method.capitalize())
            ax.fill_between(eta_vals, r[key_m] - r[key_s], r[key_m] + r[key_s],
                            alpha=0.15, color=METHOD_COLORS[method])
        ax.set_xlabel("Reward steering strength η"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs reward strength (w=0, λ=(1,0))")
        ax.legend(); ax.axhline(0, color="gray", lw=0.7, ls=":")
    fig.suptitle("Ablation: Reward Steering Strength (predicted rewards)", fontsize=14)
    fig.tight_layout()
    _save(fig, out_dir, "09b_ablation_reward_eta.png", prefix)
    _write_csv(out_dir / f"{prefix}_ablation_reward_eta.csv", eta_rows)
