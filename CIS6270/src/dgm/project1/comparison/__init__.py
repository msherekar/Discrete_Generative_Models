"""Cross-model scaling analysis: how each metric moves with ESM-2 size.

  style           the model registry, parameter counts, palette, plot helpers
  loading         reading the per-model metric CSVs
  scaling_plots   the four single-metric scaling figures
  trend_plots     loss convergence and guidance sensitivity
  summary         the summary panel and the summary CSV

compare_models.py at the project root is the command line over these.
"""
