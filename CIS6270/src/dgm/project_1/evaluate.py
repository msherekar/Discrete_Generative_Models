#!/usr/bin/env python3
"""Project 1 Evaluation: Flow Matching vs Diffusion on ESM-2 Protein Embeddings.

Loads saved results and produces comparison plots.  Results can come from the
lecture_3 scripts (default) or from run_experiment.py (recommended for new models).

Usage:
  # lecture_3 baseline (esm2_8m, toy dataset)
  dgm-evaluate

  # results produced by run_experiment.py
  dgm-evaluate --model-name esm2_35m --dataset toy64 \\
      --flow-outdir outputs/esm2_35m_toy64/flow \\
      --diff-outdir outputs/esm2_35m_toy64/diffusion

  # optional extras
  dgm-evaluate --retrain 100      # re-train with loss tracking
  dgm-evaluate --ablate           # guidance-strength sweep
  dgm-evaluate --list-models      # show all known ESM-2 variants

The figures and the metric tables live in evaluation/; this file is the command
line over them. It also re-exports every plot function, because
run_experiment.py --plot imports them from here.
"""
import argparse
from pathlib import Path

import torch

from dgm.common.esm_models import list_models
from .evaluation.ablation import plot_guidance_ablation
from .evaluation.loaders import (composition_proxies, load_fasta, load_results,
                                load_training_sequences)
from .evaluation.models import (DiffusionModel, FlowModel, RewardModel,
                               load_dataset_from_csv, make_ddpm_schedule,
                               predict_reward, sample_diffusion, sample_flow,
                               train_diffusion, train_flow)
from .evaluation.plots_basic import (plot_aa_composition,
                                    plot_composition_proxies,
                                    plot_latent_pca,
                                    plot_polar_residue_distribution,
                                    plot_reward_pareto, plot_training_loss)
from .evaluation.plots_sequence import (plot_positional_entropy,
                                       plot_sequence_diversity)
from .evaluation.rawdata import save_raw_data
from .evaluation.style import DEVICE, LECTURE3, ROOT
from .evaluation.summary import plot_summary_panel

__all__ = [
    "DEVICE", "DiffusionModel", "FlowModel", "LECTURE3", "ROOT",
    "RewardModel", "composition_proxies", "load_dataset_from_csv",
    "load_fasta", "load_results", "load_training_sequences",
    "make_ddpm_schedule", "plot_aa_composition", "plot_composition_proxies",
    "plot_guidance_ablation", "plot_latent_pca",
    "plot_polar_residue_distribution", "plot_positional_entropy",
    "plot_reward_pareto", "plot_sequence_diversity", "plot_summary_panel",
    "plot_training_loss", "predict_reward", "sample_diffusion", "sample_flow",
    "save_raw_data", "train_diffusion", "train_flow",
]


def _build_parser():
    """The evaluation command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ── Naming ────────────────────────────────────────────────────────────────
    parser.add_argument("--model-name",   default="esm2_8m",
                        help="ESM-2 model tag (default: esm2_8m). Use --list-models to see all.")
    parser.add_argument("--dataset",      default="toy64",
                        help="Short label for the dataset (default: toy64). "
                             "Used in folder name and every filename.")
    # ── Result sources ────────────────────────────────────────────────────────
    parser.add_argument("--flow-outdir",  type=Path, default=None,
                        help="Directory with flow results.pt + FASTA files. "
                             "Default: lecture/lecture_3/esm2_flow_outputs/.")
    parser.add_argument("--diff-outdir",  type=Path, default=None,
                        help="Directory with diffusion results.pt + FASTA files. "
                             "Default: lecture/lecture_3/esm2_diffusion_outputs/.")
    # ── Optional extras ───────────────────────────────────────────────────────
    parser.add_argument("--retrain",      type=int, metavar="EPOCHS",
                        help="Re-train both models for N epochs to capture loss curves.")
    parser.add_argument("--ablate",       action="store_true",
                        help="Run guidance-strength ablation (loads saved model weights).")
    parser.add_argument("--outdir",       type=Path, default=None,
                        help="Override plot output directory. "
                             "Default: plots/<model_name>_<dataset>/")
    parser.add_argument("--list-models",  action="store_true",
                        help="Print all known ESM-2 model tags and exit.")
    return parser


def _retrain(args, flow_data, diff_data, train_seqs, outdir, prefix):
    """Re-train both models to capture the loss curves the saved runs lack."""
    print(f"\nRe-training for {args.retrain} epochs "
          f"(model: {args.model_name}, requires ESM-2 weights)...")
    csv_path  = LECTURE3 / "esm2_example.csv"
    cache_dir = (args.flow_outdir / ".." / "cache") if args.flow_outdir else None
    dataset, esm, tokenizer, stats = load_dataset_from_csv(
        csv_path, esm_model_tag=args.model_name, cache_dir=cache_dir
    )
    torch.manual_seed(7)
    _, _, flow_losses = train_flow(dataset, args.retrain)
    torch.manual_seed(7)
    _, _, diff_losses, *_ = train_diffusion(dataset, args.retrain)
    plot_training_loss(flow_losses, diff_losses, outdir, prefix)
    save_raw_data(flow_data, diff_data, train_seqs, outdir, prefix,
                  flow_losses=flow_losses, diff_losses=diff_losses)


def _reload_models(data, factory, length, dim, alpha_bars=None):
    """Rebuild one modality's trained networks from its saved state dicts."""
    model = (factory(length, dim) if alpha_bars is None
             else factory(length, dim, alpha_bars)).to(DEVICE)
    reward = RewardModel(length, dim).to(DEVICE)
    model.load_state_dict(data["model"])
    reward.load_state_dict(data["reward_model"])
    model.eval().requires_grad_(False)
    reward.eval().requires_grad_(False)
    return model, reward


def _ablate(flow_data, diff_data, outdir, prefix):
    """The guidance-strength sweep, from the saved weights."""
    print("\nRunning guidance ablation (loading saved model weights)...")
    length, dim = flow_data["length"], flow_data["dim"]
    betas, alphas, alpha_bars, post_vars = make_ddpm_schedule(1000, device=DEVICE)
    flow_model, flow_reward = _reload_models(flow_data, FlowModel, length, dim)
    diff_model, diff_reward = _reload_models(diff_data, DiffusionModel,
                                             length, dim, alpha_bars)
    plot_guidance_ablation(
        flow_model, flow_reward, diff_model, diff_reward,
        (betas, alphas, alpha_bars, post_vars),
        flow_data["stats"], diff_data["stats"],
        outdir, prefix,
    )


def _static_plots(flow_data, diff_data, train_seqs, outdir, prefix, run_tag):
    """Every figure that needs nothing but the saved results."""
    print("\nGenerating static plots...")
    plot_composition_proxies(flow_data, diff_data, train_seqs, outdir, prefix)
    try:
        plot_latent_pca(flow_data, diff_data, outdir, prefix)
    except ImportError:
        print("  [skip] latent PCA — install scikit-learn: pip install scikit-learn")
    plot_aa_composition(flow_data, diff_data, train_seqs, outdir, prefix)
    plot_polar_residue_distribution(flow_data, diff_data, outdir, prefix)
    plot_reward_pareto(flow_data, diff_data, train_seqs, outdir, prefix)
    plot_positional_entropy(flow_data, diff_data, train_seqs, outdir, prefix)
    plot_sequence_diversity(flow_data, diff_data, train_seqs, outdir, prefix)
    plot_summary_panel(flow_data, diff_data, train_seqs, outdir, prefix, run_tag)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_models:
        print(list_models())
        return

    run_tag = f"{args.model_name}_{args.dataset}"
    prefix  = run_tag
    outdir  = args.outdir if args.outdir else ROOT / "plots" / run_tag
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Run tag       : {run_tag}")
    print(f"Output dir    : {outdir}")
    print(f"File prefix   : {prefix}_<plotname>.png")
    if args.flow_outdir:
        print(f"Flow results  : {args.flow_outdir}")
    if args.diff_outdir:
        print(f"Diff results  : {args.diff_outdir}")

    print("\nLoading results...")
    flow_data = load_results("flow", args.flow_outdir)
    diff_data = load_results("diffusion", args.diff_outdir)
    train_seqs, train_classes = load_training_sequences()
    print(f"  Flow guidance modes : {list(flow_data['sequences'].keys())}")
    print(f"  Diff guidance modes : {list(diff_data['sequences'].keys())}")
    print(f"  Training sequences  : {len(train_seqs)}")

    print("\nSaving raw metric CSVs...")
    save_raw_data(flow_data, diff_data, train_seqs, outdir, prefix)

    _static_plots(flow_data, diff_data, train_seqs, outdir, prefix, run_tag)

    if args.retrain:
        _retrain(args, flow_data, diff_data, train_seqs, outdir, prefix)
    if args.ablate:
        _ablate(flow_data, diff_data, outdir, prefix)

    n_png = len(list(outdir.glob("*.png")))
    n_csv = len(list(outdir.glob("*.csv")))
    print(f"\nDone. {n_png} plots + {n_csv} CSV files saved to {outdir}/")


if __name__ == "__main__":
    main()
