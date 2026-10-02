"""The command line.

Separated from the run itself because the help text is the documentation for
every knob in the study, and it is far longer than the code that acts on it.
"""
import argparse
from pathlib import Path

from dgm.common.paths import lecture_dir

from .config import BATCH_SIZE, HIDDEN
from .coupling import add_arguments as add_coupling_arguments
from .paths import add_arguments as add_path_arguments

DEFAULT_DATASET = lecture_dir(3) / "esm2_example.csv"


def build_parser(description=None):
    """The full experiment parser, ready for parse_args()."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--esm-model",  default="esm2_8m",
                        help=f"ESM-2 model tag. (default: esm2_8m)")
    parser.add_argument("--dataset",    type=Path,
                        default=DEFAULT_DATASET,
                        help="Path to input CSV (sequence,c[,r1,r2]). (default: lecture/lecture_3/esm2_example.csv)")
    parser.add_argument("--dataset-tag", default=None,
                        help="Short label for the dataset used in the output dir name. "
                             "Defaults to the CSV filename stem.")
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--samples",    type=int, default=8)
    parser.add_argument("--min-polar",  type=int, default=12)
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum (and shared) sequence length. Raise to 237 for avGFP.")
    parser.add_argument("--mut-budget", type=int, nargs="+", default=None,
                        metavar="K",
                        help="Decode as variants of a reference with at most this many "
                             "substitutions (replaces the min-polar constraint). "
                             "The reference defaults to the training-set consensus.")
    parser.add_argument("--reference",  type=Path, default=None,
                        help="File holding the reference sequence for --mut-budget "
                             "(e.g. data/avgfp_wt.txt). Default: training consensus.")
    parser.add_argument("--outdir",     type=Path, default=None,
                        help="Override output root. Default: outputs/<esm_model>_<dataset_tag>/")
    parser.add_argument("--anchor-strength", type=float, default=None, metavar="S",
                        help="Start sampling from the partially noised reference instead of "
                             "pure noise. S in (0,1]: 0.3 keeps the reference largely intact, "
                             "1.0 is the unanchored baseline. Requires --reference or "
                             "--mut-budget (which supplies the consensus reference).")
    parser.add_argument("--arch", default="mlp", choices=("mlp", "transformer"),
                        help="Velocity/noise network. 'mlp' flattens [L,D] into one "
                             "vector -- Lecture 3.2's recipe for small vectors in R^64. "
                             "'transformer' is Lecture 3.3's AMP-Diffusion trunk: six "
                             "self-attention layers over residues with learned positional "
                             "embeddings, weights shared across positions. With --arch "
                             "transformer, --hidden sets d_model (try 256).")
    add_path_arguments(parser)
    parser.add_argument("--steps", type=int, default=200, metavar="N",
                        help="Euler steps for the flow sampler (default: 200, the "
                             "value hardcoded before this flag existed). This is the "
                             "axis a coupling claim is made on: an independent "
                             "coupling and an OT coupling reach the same marginal "
                             "given enough steps, and differ in how few steps they "
                             "need. Sweep 200 50 20 10 and report quality against "
                             "step count, not quality at 200.")
    add_coupling_arguments(parser)
    parser.add_argument("--guidance-clip", type=float, default=0.0, metavar="C",
                        help="Bound the per-sample reward-gradient norm at C "
                             "(default: 0 = no clipping). Lecture 3.4: 'Clip unusually "
                             "large guidance gradients'. Without it, diffusion diverged "
                             "at eta=50.")
    parser.add_argument("--predict", default="eps", choices=("eps", "x0"),
                        help="What the diffusion network predicts (default: eps). "
                             "'x0' is Lecture 3.3's AMP-Diffusion recipe and makes the "
                             "network responsible at every noise level; with 'eps' its "
                             "weight exceeds 0.5 for only 369 of 1000 steps.")
    parser.add_argument("--ema", type=float, default=0.0, metavar="DECAY",
                        help="Exponential moving average of diffusion weights for "
                             "sampling, e.g. 0.999 (default: 0 = off). Standard in DDPM "
                             "training and the usual cure for run-to-run variance.")
    parser.add_argument("--diffusion-steps", type=int, default=1000, metavar="K",
                        help="Length of the DDPM schedule (default: 1000). Smaller K "
                             "means each minibatch covers more of it.")
    parser.add_argument("--stratified-timesteps", action="store_true",
                        help="Spread each minibatch's diffusion steps evenly over the "
                             "schedule instead of drawing them independently.")
    parser.add_argument("--seed", type=int, default=7, metavar="S",
                        help="Seed for weight initialization and batch order "
                             "(default: 7, the value hardcoded before this flag existed).")
    parser.add_argument("--sample-seed", type=int, default=123, metavar="S",
                        help="Seed for the sampling noise (default: 123, the value "
                             "hardcoded before this flag existed). Kept separate from "
                             "--seed so the defaults reproduce pre-flag runs exactly. "
                             "For independent replicates vary BOTH, e.g. "
                             "--seed 11 --sample-seed 11.")
    parser.add_argument("--endpoint-guidance", action="store_true",
                        help="Score the PREDICTED CLEAN ENDPOINT instead of the noisy "
                             "state, and differentiate back through it. Lecture 3.4: "
                             "'Evaluate rewards on a predicted clean endpoint', because "
                             "'a partially denoised protein latent may not yet "
                             "correspond to a valid sequence'. The avGFP reward model "
                             "scores Spearman 0.26 at t=0.1 but 0.82 on a clean latent. "
                             "Costs one extra network evaluation per guided step.")
    parser.add_argument("--normalize-guidance", action="store_true",
                        help="Scale each objective's gradient to unit norm before the "
                             "lambda-weighted sum. Lecture 3.4: 'Normalize objectives "
                             "before combining them'. Makes lambda a true importance "
                             "ratio and makes eta comparable across runs.")
    parser.add_argument("--hidden", type=int, default=HIDDEN, metavar="H",
                        help=f"Width of the velocity/noise network (default: {HIDDEN}). "
                             "The network maps length*dim -> H -> H -> length*dim, so H "
                             "caps the rank of the learned field. At 237x320 the default "
                             "compresses 75,840 dimensions to 128, which may be what "
                             "limits GFP; the 64-sequence teaching set it was chosen for "
                             "is only 7,680.")
    parser.add_argument("--exact-mutations", action="store_true",
                        help="Give every sample exactly --mut-budget substitutions "
                             "instead of at most that many. Without it the budget is a "
                             "ceiling and the reference residue usually wins the draw, "
                             "so a budget of 5 yields ~1.5 substitutions and ~18%% of "
                             "samples are the unmutated reference. Fixing the count "
                             "makes mutational distance a controlled variable.")
    parser.add_argument("--decode-temperature", type=float, default=0.0, metavar="T",
                        help="Sampling temperature for decoding (default: 0 = argmax). "
                             "Argmax is deterministic and collapses on this data: the "
                             "reference residue is the mode almost everywhere, so many "
                             "different latents decode to the same few sequences. T=0.7 "
                             "restores diversity at an unchanged Hamming distance.")
    parser.add_argument("--freeze-positions", default="", metavar="LIST",
                        help="Comma-separated 0-indexed positions never to substitute, "
                             "e.g. '0'. The avGFP wild type here omits the initiator "
                             "methionine, so ESM puts one back at position 0 and that "
                             "substitution otherwise dominates every sample.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, metavar="N",
                        help=f"Training minibatch size (default: {BATCH_SIZE}). The "
                             "default suits the 64-sequence teaching set; at tens of "
                             "thousands of sequences it leaves the GPU mostly idle and "
                             "128-256 trains several times faster per epoch.")
    parser.add_argument("--cfg-weight", type=float, nargs="+", default=[2.0],
                        metavar="W",
                        help="Classifier-free guidance weights, adding one 'cfg' arm "
                             "per value (default: 2.0, tuned on a 24x320 latent; a "
                             "237x320 latent has a much larger norm). Like "
                             "--reward-eta this is a sampling-time knob, so several "
                             "values cost one extra sampling pass each rather than a "
                             "retraining apiece, and every arm reads the same trained "
                             "field. The first value keeps the bare name 'cfg'.\n"
                             "Pass 0 to get the UNCONDITIONAL control: Lecture 3.4's "
                             "field is v_uncond + w(v_cond - v_uncond), so w=0 is the "
                             "unconditional field, w=1 the ordinary conditional field, "
                             "and w>1 extrapolates beyond it. Without a w=0 arm a run "
                             "cannot separate what the conditioning contributes from "
                             "what the model and the decoder would have produced "
                             "anyway, so '0 1 2 4' is the sweep that actually "
                             "demonstrates conditional generation.")
    parser.add_argument("--reward-lambda", type=float, nargs="+", default=None,
                        metavar="L1",
                        help="Sweep the objective mix, adding an arm 'lamL1' per value "
                             "with lambdas=(L1, 1-L1): 1.0 is brightness alone, 0.0 is "
                             "parsimony alone. Only informative when the mutation count "
                             "can actually vary -- with --exact-mutations every sample "
                             "carries the same number of substitutions, so r2 is "
                             "constant in the output and the arms come out identical. "
                             "Pair this with --mut-budget as a ceiling and no "
                             "--exact-mutations to trace a real Pareto front.")
    parser.add_argument("--setpoint", type=float, nargs="+", default=None,
                        metavar="Y",
                        help="Target brightness values, in the same raw units as "
                             "the r1 column. Replaces 'maximize r1' with the "
                             "calibration objective -(r1 - y)^2, adding one arm "
                             "'sp<y>' per value. Steering GFP dimmer is a null "
                             "task -- nearly any substitution does it -- so "
                             "hitting a requested value is the stronger claim: "
                             "plot achieved against requested and report the "
                             "slope. Use --setpoint-percentile to give these as "
                             "percentiles of the training brightness instead.")
    parser.add_argument("--setpoint-percentile", action="store_true",
                        help="Read --setpoint values as percentiles (0-100) of "
                             "the training r1 distribution rather than as raw "
                             "scores. Multiples of the wild type are NOT offered "
                             "because prepare_gfp.py centres r1 on the wild type, "
                             "so WT is exactly 0 and every multiple of it "
                             "collapses to the same target. The avgfp_train "
                             "distribution is also strongly bimodal -- a dark "
                             "cluster near -2.42 and a functional cluster near "
                             "-0.1 -- so evenly spaced raw targets are not "
                             "evenly populated, and percentiles are the safer "
                             "default. p50 = -0.43, p90 = -0.00, p99 = +0.15.")
    parser.add_argument("--setpoint-saturation", type=float, default=4.0,
                        metavar="C",
                        help="Largest multiplier the setpoint controller may "
                             "apply, which caps guidance at C (plus the "
                             "constraint's rho) and so keeps the step bounded. "
                             "The factor is 2*(target - predicted) in "
                             "STANDARDIZED units, so the controller saturates "
                             "once it is C/2 standard deviations from target and "
                             "below that decelerates toward it. Set this too low "
                             "and distant setpoints become indistinguishable: "
                             "every one of them saturates, and the whole sweep "
                             "collapses onto plain maximize. avgfp_train spans "
                             "about 2.7 standard deviations of r1, so C=6 never "
                             "saturates anywhere in the observed range; the "
                             "default 4.0 trades a little of that for a smaller "
                             "step.")
    parser.add_argument("--minimize", type=int, nargs="+", default=None,
                        metavar="IDX",
                        help="1-based property columns to minimize rather than "
                             "maximize, so larger normalized values always mean "
                             "more desirable (Lecture 3.4). With the CSV that "
                             "add_properties.py writes, pass 2: r2 is "
                             "exposed_hydrophobics, an aggregation proxy.")
    parser.add_argument("--constraint-property", type=int, default=None,
                        metavar="IDX",
                        help="1-based property column held as a CONSTRAINT "
                             "instead of a scalarized objective. Columns from "
                             "here on are excluded from the lambda simplex. Pass "
                             "3 for the add_properties.py layout, where r3 is "
                             "Rosetta total_score: measured Spearman -0.67 with "
                             "brightness on avgfp_train, the most "
                             "brightness-redundant of METL's 55 attributes, so "
                             "weighting it against brightness would trace a line "
                             "rather than a Pareto front. As a constraint that "
                             "redundancy is harmless.")
    parser.add_argument("--constraint-delta", type=float, default=0.5,
                        metavar="D",
                        help="Allowed slack above the reference sequence's own "
                             "value of the constrained property, in raw units "
                             "(default: 0.5). The penalty is zero at or below "
                             "reference + D.")
    parser.add_argument("--constraint-rho", type=float, default=0.0,
                        metavar="RHO",
                        help="Severity of the one-sided squared constraint "
                             "penalty (default: 0, off). Deliberately not on the "
                             "lambda simplex, so raising it cannot silently eat "
                             "the objective budget.")
    parser.add_argument("--reward-eta", type=float, nargs="+", default=[1.0],
                        metavar="ETA",
                        help="Reward-gradient strength for the 'single' and 'multi' "
                             "modes (default: 1.0). Several values sample the same "
                             "trained model once per eta, adding arms named "
                             "'single@ETA'. Eta does not transfer between modalities "
                             "under --normalize-guidance: the gradient is a unit "
                             "vector, so its step is the same absolute size at any "
                             "dimension while the latent norm grows as sqrt(D). A "
                             "237x320 protein latent has norm ~275 against MNIST's "
                             "~28, so the same eta moves a protein ~10x less.")
    parser.add_argument("--oracle", type=Path, default=None, metavar="NPZ",
                        help="Fitted brightness oracle from embedding_oracle.py. "
                             "Defaults to data/avgfp_metl_oracle.npz when it exists. "
                             "Scores every decoded sample; never touches guidance, so "
                             "it stays an independent judge. Use --no-oracle to skip.")
    parser.add_argument("--no-oracle", action="store_true",
                        help="Do not score samples, even if an oracle file is present.")
    parser.add_argument("--restrict-support", type=Path, default=None, metavar="NPZ|CSV",
                        help="Only produce substitutions the assay measured. Takes a "
                             "gfp_oracle.py .npz (support = its nonzero coefficients) "
                             "or a variant .csv with a 'sequence' column (support = "
                             "every substitution in it). avGFP's scan covers ~38%% of "
                             "the 4,503 possible substitutions; outside that set the "
                             "oracle returns its intercept, so unrestricted samples mix "
                             "predictions with fallbacks. With this flag every sample is "
                             "in-domain by construction. Requires --reference.")
    parser.add_argument("--support-level", default="substitution",
                        choices=("substitution", "position"),
                        help="How tight --restrict-support is (default: substitution). "
                             "'substitution' permits only measured (position, residue) "
                             "pairs; 'position' permits any residue at a measured "
                             "position, which on avGFP is 233 of 237 and so barely "
                             "constrains anything -- it is the ablation showing that "
                             "residue identity, not position, is what carries the "
                             "constraint.")
    parser.add_argument("--cache-dir",  type=Path, default=None,
                        help="Shared HuggingFace weight cache. Default: $ESM2_CACHE, "
                             "else project1_eval/cache/ (one copy per model, reused by all runs).")
    parser.add_argument("--plot",        action="store_true",
                        help="Generate all evaluation plots (including training-loss curves) "
                             "immediately after training, using in-memory results.")
    parser.add_argument("--ablate",      action="store_true",
                        help="Also run guidance-strength ablation plots (CFG-weight sweep and "
                             "reward-eta sweep). Implies --plot.")
    parser.add_argument("--list-models", action="store_true",
                        help="Print all known ESM-2 models and exit.")
    return parser
