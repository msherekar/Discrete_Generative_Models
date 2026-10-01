"""Optional plotting, driven from the in-memory results.

Nothing is reloaded from disk: the latents, sequences and loss curves are all
still in memory when a run finishes, so --plot costs no re-encoding and the
figures are guaranteed to describe the run that just happened.
"""
import csv

from dgm.common.paths import plots_dir


def _plot_payload(latents, model, reward, setup, min_polar) -> dict:
    """The dict shape evaluate.py's plot functions expect."""
    return {
        "standardized_latents": {k: v["latent"].cpu() for k, v in latents.items()},
        "sequences":            {k: v["sequences"]    for k, v in latents.items()},
        "model":                model.state_dict(),
        "reward_model":         reward.state_dict(),
        "stats":                setup.stats,
        "length":               setup.length,
        "dim":                  setup.dim,
        "min_polar":            min_polar,
    }


def generate_plots(args, setup, flow, diff, run_tag):
    """Write every evaluation figure and raw metric CSV for a finished run.

    `flow` and `diff` are (latents, model, reward, losses) per modality; `diff`
    carries the DDPM schedule as a fifth element for the ablation.
    """
    print("\nGenerating plots...")
    # Imported here, not at module scope: evaluate imports matplotlib and the
    # whole figure stack, which a run without --plot should not pay for.
    from ..evaluate import (
        plot_training_loss, plot_composition_proxies, plot_latent_pca,
        plot_aa_composition, plot_polar_residue_distribution,
        plot_reward_pareto, plot_positional_entropy,
        plot_sequence_diversity, plot_summary_panel,
        plot_guidance_ablation, save_raw_data,
    )

    flow_latents, flow_model, flow_reward, flow_losses = flow
    diff_latents, diff_model, diff_reward, diff_losses, schedule = diff
    flow_data = _plot_payload(flow_latents, flow_model, flow_reward,
                              setup, args.min_polar)
    diff_data = _plot_payload(diff_latents, diff_model, diff_reward,
                              setup, args.min_polar)

    # Read training sequences from the CSV we used (no re-encoding needed).
    with open(args.dataset, newline="") as f:
        train_seqs = [r["sequence"].strip().upper() for r in csv.DictReader(f)]

    prefix   = run_tag
    plot_dir = plots_dir() / run_tag
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Raw metric CSVs — written before plots so data is always there.
    print("  Saving raw metric CSVs...")
    save_raw_data(flow_data, diff_data, train_seqs, plot_dir, prefix,
                  flow_losses=flow_losses, diff_losses=diff_losses)

    # Training-loss curves — already collected epoch-by-epoch during training.
    plot_training_loss(flow_losses, diff_losses, plot_dir, prefix)

    plot_composition_proxies(flow_data, diff_data, train_seqs, plot_dir, prefix)
    try:
        plot_latent_pca(flow_data, diff_data, plot_dir, prefix)
    except ImportError:
        print("  [skip] PCA — pip install scikit-learn")
    plot_aa_composition(flow_data, diff_data, train_seqs, plot_dir, prefix)
    plot_polar_residue_distribution(flow_data, diff_data, plot_dir, prefix)
    plot_reward_pareto(flow_data, diff_data, train_seqs, plot_dir, prefix)
    plot_positional_entropy(flow_data, diff_data, train_seqs, plot_dir, prefix)
    plot_sequence_diversity(flow_data, diff_data, train_seqs, plot_dir, prefix)
    plot_summary_panel(flow_data, diff_data, train_seqs, plot_dir, prefix, run_tag)

    # Ablation — uses the already-trained models (no extra training).
    if args.ablate:
        print("  Running guidance-strength ablation (uses in-memory models)...")
        plot_guidance_ablation(
            flow_model, flow_reward, diff_model, diff_reward, schedule,
            setup.stats, setup.stats,   # same stats for both (shared encoding)
            plot_dir, prefix,
        )

    n_png = len(list(plot_dir.glob("*.png")))
    n_csv = len(list(plot_dir.glob("*.csv")))
    print(f"\n{n_png} plots + {n_csv} CSV files saved to {plot_dir}/")
