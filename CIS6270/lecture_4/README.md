# CIS 6270 - Lecture 4 - Discrete Diffusion

Course hub: [ChatterjeeLab/CIS6270 on Hugging Face](https://huggingface.co/ChatterjeeLab/CIS6270).

Complete CPU training and generation examples for the code walkthroughs in the [Discrete Generation lecture](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit). This folder ends with PepTune. Classic discrete flow matching and the later methods are in [Lecture 5](../lecture_5/README.md).

These are small, inspectable implementations of the mathematical mechanisms. The DNA data and two objective functions are synthetic. They are not benchmark reproductions or biological property predictors.

## Start with the complete MDLM example

Python 3.11 or later, CPU. No downloaded data or pretrained checkpoint is required.

From the course repository root, enter this lecture folder.

```bash
cd lecture_4
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py --method mdlm --data data/dna_train.tsv --out outputs/mdlm
```

This command loads the DNA examples, reserves the final 20% for validation, trains a transformer denoiser, writes a checkpoint, and generates new sequences. Each output folder contains `config.json`, `losses.json`, `report.json`, `samples.txt`, and `checkpoint.pt`.

Inspect `data/dna_train.tsv` before training. `label=1` means GC fraction at least 0.6. It is a deterministic toy label, not measured activity. Omitting `--data` generates synthetic examples at `--length`.

```bash
# Continue with the saved weights for generation only.
python run.py --method mdlm --mode sample --out outputs/mdlm
# An entire small experiment using the four-base vocabulary.
python examples/mdlm.py --length 4 --train-steps 160 --samples 8
# Run every method, or a short CPU integration check.
python run_all.py
python run_all.py --quick
python -m unittest discover -s tests -v
python numerical_examples.py
```

For `--mode sample`, use the same `--method`, `--width`, and `--length` as the checkpoint. Checkpoints contain model weights and metadata; they do not contain optimizer state for resuming optimization.

## One shared backbone; small method changes

`lecture_core.py` contains the functions shown in the slides. `common.py` contains data, optimization, and output utilities. `diffusion.py` adds the complete samplers and guidance/search loops that make those modules runnable. `run.py` connects training to generation. Each file under `examples/` is a complete command-line entry point for one method and accepts the same options.

| Example | Training target | Generation procedure |
| --- | --- | --- |
| `mdlm` | Weighted cross-entropy at masked positions | Schedule-based reveals; preserve visible bases |
| `udlm` | Reverse-rate KL including the staying term | Adaptive reverse Euler; visible bases may be revised |
| `block` | Conditional masked loss on a sampled block, scaled by block count | Finish each block before extending the sequence |
| `cfg` / `classifier-free` | Label-conditioned MDLM with 15% label dropout | Geometric conditional/unconditional prediction blend |
| `classifier-exact` | MDLM plus a classifier trained on noisy sequences | Evaluate every replacement's log-value change and multiply rates |
| `classifier-gradient` | Same denoiser and noisy classifier | Approximate replacement log-value changes with one-hot gradients |
| `peptune` | MDLM backbone | MCTS selection, expansion, completion, Pareto rewards, and reward backup |

```bash
python examples/udlm.py --out outputs/udlm
python examples/block.py --block-size 2
python examples/cfg.py --label 1 --strength 2
python examples/classifier_exact.py --classifier-steps 300
python examples/classifier_gradient.py --classifier-steps 300
python examples/peptune.py --search-steps 100
```

MDLM and block losses sum over positions, then average over sequences. The uniform method uses the slide's linear schedule and integrates its training loss over `t in [0.02, 0.98]`; uniform initialization at 0.98 and stopping at 0.02 approximate the ideal endpoints. The output retains residual corruption. We do not reinterpret the UDLM parameter vector as a clean-token posterior for a final resampling step. The classifier-guided sampler uses adaptive steps to keep outgoing probabilities valid, then closes any residual masks at `t=0.005`. These finite endpoint conventions are reported in the outputs.

CFG here is a categorical prediction blend. It is not asserted to equal geometric mixing of every possible rate parameterization. Classifier guidance is approximate because the classifier is learned; the gradient version also approximates finite edit differences. `strength=0` recovers the unguided rate multiplier in the classifier routines.

The block implementation omits future blocks from the input entirely. This is a transparent teaching alternative to a full block-causal attention mask and KV-cache implementation. Training prefixes are clean; generation prefixes are sampled.

The PepTune example uses the search mechanism on DNA. Every A/C/G/T sequence is valid, so there is no invalid-SMILES penalty. Peptide tokenization, bond-dependent schedules, chemical validity, RoFormer pretraining, and the paper's biological predictors are outside this toy example. Its archive is the non-dominated subset of evaluated DNA candidates; it is not a global Pareto certificate.

## Read the math next to the implementation

The tests verify the masked reverse KL cancellation, the small-step UDLM KL limit, the numerical CFG example, and the classifier directional derivative. `numerical_examples.py` prints corruption, training-loss, and generation-step calculations from the slides. `SLIDE_CODE_MAP.md` maps every Lecture 4 code unit to its function.

The seeded example runs in `verified_examples/` record actual CPU results. Losses from different objectives are not directly comparable; one Monte Carlo validation draw is not a perplexity estimate. Sampling quality should be assessed with more seeds and longer training if extending these examples.

## Sources

- [MDLM - Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- [UDLM - Simple Guidance Mechanisms for Discrete Diffusion Models](https://arxiv.org/abs/2412.10193)
- [Block Diffusion](https://arxiv.org/abs/2503.09573)
- [PepTune](https://arxiv.org/abs/2412.17780)

The source papers define the full research methods and experiments. Comments identify the classroom simplifications.
