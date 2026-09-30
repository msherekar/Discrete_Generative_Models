"""Evaluation figures and metric tables for a finished Project 1 run.

  style           constants, the colour palette, the matplotlib style
  loaders         reading results.pt, FASTA and the training CSV back
  metrics         sequence statistics shared by tables and figures
  output          the two writers every CSV and figure goes out through
  models          networks, re-training and sampling for --retrain/--ablate
  rawdata         one CSV per metric
  plots_basic     plots 1-6
  plots_sequence  plots 7-8
  ablation        plot 9, the guidance-strength sweep
  summary         plot 10, the one-page panel

evaluate.py at the project root is the command line over these, and re-exports
the plot functions that run_experiment.py --plot imports.
"""
