#!/usr/bin/env python3
"""Run the full Project 1 pipeline for any ESM-2 model variant.

Encodes sequences with the chosen ESM-2 model, trains flow-matching and
diffusion models, samples under three guidance modes, decodes to amino-acid
sequences, and saves results.pt + FASTA files in a structured output directory
ready for evaluate.py.

Usage:
  dgm-run-experiment --esm-model esm2_8m  --dataset ../lecture/lecture_3/esm2_example.csv
  dgm-run-experiment --esm-model esm2_35m --dataset /path/to/my_data.csv --epochs 300
  dgm-run-experiment --list-models

Output layout:
  outputs/<esm_model>_<dataset_tag>/
    flow/       results.pt, cfg.fasta, single.fasta, multi.fasta
    diffusion/  results.pt, cfg.fasta, single.fasta, multi.fasta

  cache/        HuggingFace weights, shared by every model and every run

The stages live in pipeline/; this file only orchestrates them. It also
re-exports the names other scripts in this directory import (run_mnist.py,
resample_cfg.py, diagnostics/*), so `import run_experiment as R` keeps working.
"""
from pathlib import Path

import torch

from dgm.common.esm_models import get_model, list_models
from dgm.common.paths import outputs_dir
from .pipeline.arms import build_arms, build_objective
from .pipeline.cli import build_parser
from .pipeline.config import (AMINO_ACIDS, BATCH_SIZE, CONDITION_DROP, DEVICE,
                             HIDDEN, LEARNING_RATE, POLAR_RESIDUES,
                             resolve_cache_dir)
from .pipeline.data import (composition_proxies, consensus, encode_reference,
                           load_data)
from .pipeline.decoding import (decode, decode_budget, load_support_mask)
from .pipeline.guidance import reward_gradient
from .pipeline.nets import (EMA, DiffusionModel, FlowModel, RewardModel,
                           TransformerField)
from .pipeline.objective import Objective, weight_vector
from .pipeline.oracle import (decode_all_budgets, load_brightness_oracle,
                             resolve_oracle, score_with_oracle)
from .pipeline.paths import (INTERPOLANTS, PathSpec, endpoint_from_noise,
                            endpoint_from_velocity, interpolate,
                            make_ddpm_schedule, sample_timesteps)
from .pipeline.plots import generate_plots
from .pipeline.prepare import describe_run, prepare_run
from .pipeline.reporting import oracle_summary, report_arm
from .pipeline.results import save_results
from .pipeline.runconfig import build_run_config
from .pipeline.sampling import sample_diffusion, sample_flow
from .pipeline.training import train_diffusion, train_flow
from .pipeline.wiring import (describe_innovations, diffusion_sample_kwargs,
                              diffusion_train_kwargs, flow_sample_kwargs,
                              flow_train_kwargs, latent_kwargs, sample_kwargs)

__all__ = [
    "AMINO_ACIDS", "BATCH_SIZE", "CONDITION_DROP", "DEVICE", "HIDDEN",
    "INTERPOLANTS", "LEARNING_RATE", "POLAR_RESIDUES",
    "DiffusionModel", "EMA", "FlowModel", "Objective", "RewardModel",
    "TransformerField", "composition_proxies", "consensus", "decode",
    "decode_all_budgets", "decode_budget", "encode_reference",
    "endpoint_from_noise", "endpoint_from_velocity", "interpolate",
    "load_brightness_oracle", "load_data", "load_support_mask",
    "make_ddpm_schedule", "resolve_cache_dir", "reward_gradient",
    "sample_diffusion", "sample_flow", "sample_timesteps", "save_results",
    "score_with_oracle", "train_diffusion", "train_flow", "weight_vector",
]


def _sample_all(sampler, arms, setup, oracle, samples, guide_kwargs,
                extra=None):
    """Sample, decode and report every arm from one trained field."""
    latents = {}
    for name, cfg in arms.items():
        z = sampler(n=samples, **cfg, **setup.anchor_kwargs, **guide_kwargs,
                    **(extra or {}))
        entry = decode_all_budgets(z, setup.decode_fns, setup.primary, oracle)
        report_arm(name, entry["sequences"], setup.reference, entry["oracle"], z)
        latents[name] = entry
    return latents


def main():
    parser = build_parser(__doc__)
    args = parser.parse_args()

    if args.list_models:
        print(list_models())
        return

    model_info  = get_model(args.esm_model)
    dataset_tag = args.dataset_tag or Path(args.dataset).stem
    run_tag     = f"{args.esm_model}_{dataset_tag}"
    out_root    = args.outdir or outputs_dir() / run_tag
    cache_dir   = resolve_cache_dir(args.cache_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    describe_run(args, model_info, dataset_tag, run_tag, out_root)
    torch.manual_seed(args.seed)
    if DEVICE.type == "cpu":
        torch.set_num_threads(2)

    setup = prepare_run(args, parser, model_info, cache_dir)
    spec  = build_objective(args, parser, setup)
    print(describe_innovations(args))

    # The validation split is encoded with the same ESM-2 weights and THEN
    # re-standardized with the training split's statistics, so the two losses
    # are on one scale and the epoch budget can be read off the val curve
    # rather than guessed; see Stage 0.5.3.
    # load_data standardizes each split with its own per-CSV statistics; we must
    # undo the val-CSV standardization and apply the training statistics instead.
    val_dataset = None
    if args.val_dataset is not None:
        print(f"\nEncoding validation split {args.val_dataset.name}...")
        val_ds, _, _, val_stats, _ = load_data(
            args.val_dataset, model_info["hf_id"], cache_dir,
            args.max_length, **latent_kwargs(args))
        # Re-standardize latents and properties using training statistics.
        import torch as _torch
        from torch.utils.data import TensorDataset as _TDS
        zv, cv, rv = val_ds.tensors
        s = setup.stats
        n_val_props   = rv.shape[1]
        n_train_props = s["r_mean"].shape[0]
        # Undo val-CSV standardization, apply training standardization.
        zv_rescaled = ((zv.float() * val_stats["z_std"].to(zv.device)
                        + val_stats["z_mean"].to(zv.device))
                       - s["z_mean"].to(zv.device)) / s["z_std"].to(zv.device)
        rv_rescaled = ((rv.float() * val_stats["r_std"].to(rv.device)
                        + val_stats["r_mean"].to(rv.device))
                       - s["r_mean"][:n_val_props].to(rv.device)
                       ) / s["r_std"][:n_val_props].to(rv.device)
        # Val CSV may have fewer property columns than train (e.g. r3 from
        # add_properties not yet run on it). Pad missing columns with 0
        # (the training-standardized mean) so tensor shapes match throughout.
        if n_val_props < n_train_props:
            pad = n_train_props - n_val_props
            rv_rescaled = _torch.cat(
                [rv_rescaled,
                 _torch.zeros(rv.shape[0], pad,
                              dtype=rv_rescaled.dtype, device=rv.device)], dim=1)
            cv = _torch.cat(
                [cv,
                 _torch.zeros(cv.shape[0], pad,
                              dtype=cv.dtype, device=cv.device)], dim=1)
        val_dataset = _TDS(zv_rescaled.to(zv.dtype), cv, rv_rescaled)

    guide_kwargs = sample_kwargs(args)
    oracle, oracle_path = resolve_oracle(args, setup.length)
    arms       = build_arms(args, parser, spec)
    path       = PathSpec.from_args(args, setup.dataset.tensors[0])
    print(f"  Path        : {path.describe()}")
    run_config = build_run_config(args, setup, spec, oracle_path, path)
    save_args  = (setup.stats, model_info["hf_id"], setup.length, setup.dim,
                  args.min_polar)

    # ── Flow matching ─────────────────────────────────────────────────────────
    flow_latents = flow_model = flow_reward = flow_source = None
    flow_losses = []
    if args.only in (None, "flow"):
      print("\nTraining flow matching model...")
      torch.manual_seed(args.seed)
      columns = ([c - 1 for c in args.coupling_columns]
                 if args.coupling_columns else None)
      flow_model, flow_reward, flow_losses, flow_source = train_flow(
          setup.dataset, args.epochs,
          **flow_train_kwargs(args, setup.stats, val_dataset, path, columns))

      print("\nSampling (flow)...")
      flow_latents = _sample_all(
          lambda **kw: sample_flow(flow_model, flow_reward, **kw),
          arms, setup, oracle, args.samples, guide_kwargs,
          extra=flow_sample_kwargs(args, path, flow_source))
      save_results(out_root, "flow", flow_latents, flow_model, flow_reward,
                   *save_args, flow_losses, run_config)


    # ── Diffusion ─────────────────────────────────────────────────────────────
    diff_latents = diff_model = diff_reward = None
    diff_losses, alpha_bars, betas, alphas, post_vars = [], None, None, None, None
    if args.only in (None, "diffusion"):
      print("\nTraining diffusion model...")
      torch.manual_seed(args.seed)
      diff_model, diff_reward, diff_losses, alpha_bars, betas, alphas, post_vars = \
          train_diffusion(setup.dataset, args.epochs,
                          **diffusion_train_kwargs(args, setup.stats, val_dataset))

      print("\nSampling (diffusion)...")
      diff_latents = _sample_all(
          lambda **kw: sample_diffusion(diff_model, diff_reward, alpha_bars,
                                        betas, alphas, post_vars, **kw),
          arms, setup, oracle, args.samples, guide_kwargs,
          extra=diffusion_sample_kwargs(args))
      save_results(out_root, "diffusion", diff_latents, diff_model, diff_reward,
                   *save_args, diff_losses, run_config)


    print(f"\nDone. Results in {out_root}/")

    if oracle is not None:
        oracle_summary(args, setup, arms, flow_latents, diff_latents,
                       oracle_path, out_root)

    if (args.plot or args.ablate) and args.only is None:
        generate_plots(
            args, setup,
            (flow_latents, flow_model, flow_reward, flow_losses),
            (diff_latents, diff_model, diff_reward, diff_losses,
             (betas, alphas, alpha_bars, post_vars)),
            run_tag, path)
    else:
        print(f"To plot: dgm-evaluate --model-name {args.esm_model} "
              f"--dataset {dataset_tag} "
              f"--flow-outdir {out_root}/flow --diff-outdir {out_root}/diffusion")


if __name__ == "__main__":
    main()
