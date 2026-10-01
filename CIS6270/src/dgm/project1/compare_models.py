"""Cross-model scaling analysis for Project 1.

Reads all esm2_*_<dataset> CSV outputs and produces scaling plots with
x-axis = parameter count (log scale).

Usage:
    dgm-compare-models
    dgm-compare-models --dataset toy64 --outdir scaling_plots/

The figures live in comparison/; this file is the command line over them.
"""
import argparse
import sys
from pathlib import Path

from .comparison.loading import load_all
from .comparison.scaling_plots import (plot_composition_scaling,
                                      plot_entropy_scaling,
                                      plot_hamming_scaling,
                                      plot_polar_count_scaling)
from .comparison.summary import plot_summary_scaling, save_summary_csv
from .comparison.trend_plots import (plot_ablation_sensitivity_scaling,
                                    plot_loss_convergence_scaling)

PLOTS = (plot_composition_scaling, plot_polar_count_scaling,
         plot_hamming_scaling, plot_entropy_scaling,
         plot_loss_convergence_scaling, plot_ablation_sensitivity_scaling,
         plot_summary_scaling)


def main():
    parser = argparse.ArgumentParser(description="Cross-model scaling analysis")
    parser.add_argument("--plots-dir", default=None,
                        help="Directory containing esm2_*_<dataset> sub-folders "
                             "(default: project1_eval/plots/ relative to script)")
    parser.add_argument("--dataset", default="toy64",
                        help="Dataset tag (default: toy64)")
    parser.add_argument("--outdir", default=None,
                        help="Output directory for scaling plots "
                             "(default: <plots-dir>/scaling_<dataset>/)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    plots_dir = Path(args.plots_dir) if args.plots_dir else script_dir / "plots"
    out_dir = Path(args.outdir) if args.outdir else plots_dir / f"scaling_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSVs from: {plots_dir}")
    data = load_all(plots_dir, args.dataset)

    if not data:
        print(f"No data found for dataset '{args.dataset}' in {plots_dir}.", file=sys.stderr)
        print("Expected folders like: esm2_8m_toy64/, esm2_35m_toy64/, ...", file=sys.stderr)
        sys.exit(1)

    found = list(data.keys())
    print(f"Found {len(found)} model(s): {', '.join(found)}")
    print(f"Writing plots to: {out_dir}\n")

    for plot in PLOTS:
        plot(data, out_dir)
    save_summary_csv(data, out_dir, args.dataset)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} plots + 1 CSV saved to {out_dir}")


if __name__ == "__main__":
    main()
