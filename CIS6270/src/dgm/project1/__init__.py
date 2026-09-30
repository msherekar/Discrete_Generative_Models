"""Project 1: flow matching against diffusion on ESM-2 protein latents.

Runnable modules (each has a console script and works with `python -m`):

  run_experiment      train both models, sample every guidance arm, save
  evaluate            figures and metric tables for a finished run
  embedding_oracle    fit or score the independent brightness oracle
  compare_models      cross-model scaling analysis
  gfp_metrics         avGFP-specific evaluation with a random-variant control
  resample_cfg        re-sample a finished run at new guidance weights
  run_mnist           the same guidance machinery on images
  prepare_gfp, split_gfp, add_properties, gfp_oracle, oracle_sweep,
  esm_compare, analyze_sweep, compute_fid, plot_guidance_sweep

Supporting packages:

  pipeline    the experiment stages, from encoding to saved results
  evaluation  figures and metric tables
  oracles     the embedding-based brightness oracles
  comparison  cross-model scaling figures
  diagnostics standalone checks on the objective, rewards and conditioning
"""
