# ESM-2 flow matching and diffusion guidance

Two independent teaching scripts, each with encoding, normalization, a small
conditional generator, a time-conditioned reward predictor, training, three
guidance examples, and constrained sequence decoding. Neither imports the other.

## Run

Install Python 3.10+ and the dependencies in your own environment:

```sh
pip install torch transformers==4.57.6
python esm2_flow_guidance.py --epochs 200 --samples 8
python esm2_diffusion_guidance.py --epochs 200 --samples 8
```

Keep both scripts and `esm2_example.csv` together. Their default paths are
relative to the scripts, so the commands also work with absolute script paths.
The first run downloads the public `facebook/esm2_t6_8M_UR50D` checkpoint into
`.esm2_cache`. CUDA is used when available; otherwise the scripts use CPU.
These examples were executed with PyTorch 2.9.1 and Transformers 4.57.6.

## Motivation

We will generate protein-sequence representations with a flow and a diffusion
model, then guide generation toward specified properties. Using frozen ESM-2
residue embeddings, we can compare classifier-free conditioning with
single-objective and weighted multi-objective reward steering in the same latent
space. After generation, we decode the latents into amino acids and enforce an
explicit residue-count constraint. This separates learning the sequence
distribution, expressing preferences, and guaranteeing a discrete output rule.

## Where the rewards come from

The bundled dataset contains 64 synthetic sequences of length 24. These are
teaching examples, not natural proteins or experimentally validated peptides.

For sequence s of length L, `composition_proxies()` computes:

- r1 = (number of K and R minus number of D and E) / L.
  This is a side-chain charge-count proxy, not a complete pH-dependent charge model.
- r2 = number of residues in `DEHKNQRST` / L.
  This is a defined polar/charged composition fraction, not measured solubility.
- The bundled class label c is 1 when r1 > 0, and 0 otherwise.

CSV columns are `sequence,c,r1,r2`. Both classes 0 and 1 must be present. If r1
and r2 are omitted, the scripts compute the composition proxies directly.
Supplied r1/r2 columns override the proxy calculation, allowing measured
objectives or external predictor outputs. Orient each objective so larger is
better before supplying it. For an undesirable quantity, negate it first.

All sequences in one run must have the same length, between 1 and 128, using
the 20 canonical amino acids. The default hard minimum is 12, so shorter
sequences require a smaller `--min-polar` value.

## Normalization and reward prediction

- Preserve one ESM-2 embedding per residue, giving [N, L, 320]. Remove BOS/EOS.
- Standardize each latent feature using its mean and standard deviation over
  training sequences and residue positions.
- Standardize each property separately: r_tilde = (r - mean) / std.
- Save these statistics. Do not refit them on generated, validation, or test data.
- Train a small predictor on intermediate latents and their clean-sequence
  standardized property labels. Its two outputs estimate expected endpoint
  rewards from the current state and time.

The examples use all 64 sequences for training and do not claim held-out quality.
For research, split and cluster data before fitting statistics or networks, then
assess prediction and sequence reconstruction on held-out sequences.

## Three sampling modes in each script

1. CFG: `w=2`, `eta=0`. Combine unconditional and class-conditioned predictions
   as F_uncond + w * (F_cond - F_uncond). Here w=0 is unconditional and w=1 is
   ordinary conditional sampling. Drop the class to null index 2 in 20% of
   training examples. CFG does not use the external reward predictor.
2. Single objective: `w=0`, `eta=1`, `lambdas=(1,0)`.
3. Multiple objectives: `w=0`, `eta=1`, `lambdas=(0.7,0.3)`.

The weights are normalized to sum to one. Lambda controls the relative
tradeoff in standardized property units; eta controls overall steering strength.
Set both w and eta positive to combine CFG and reward steering. Edit the three
calls in `main()` to change the demonstration settings.

Flow training uses Z_0=noise, Z_1=data, a straight conditional path, and target
velocity Z_1-Z_0. Sampling integrates from t=0 to t=1 with Euler steps. Reward
steering adds kappa(t) times the reward gradient to the velocity, with the chosen
schedule kappa(t)=4*eta*t*(1-t). This is heuristic velocity steering, not a claim
of exact sampling from a reward-tilted density.

DDPM training uses Z_0=data and predicts the Gaussian noise used to construct
Z_k. Sampling runs k=1000 down to 1. The small noise predictor includes the
Gaussian-reference skip sqrt(1-alpha_bar_k)*Z_k and learns an additive correction
scaled by sqrt(alpha_bar_k). This is a parameterization of the noise predictor;
the target and DDPM equations remain noise-prediction equations. It lets the
small network pass through high-dimensional noise without reconstructing every
coordinate through its narrow hidden layer.

For DDPM steering, epsilon_guided = epsilon_CFG - eta *
sqrt(1-alpha_bar_k) * gradient(R_lambda). This follows the score-to-noise
conversion s = -epsilon / sqrt(1-alpha_bar_k). The posterior standard deviation
used for the reverse random increment is a different quantity.

In both scripts, gradient ascent uses a learned expected-reward predictor.
It is not the exact log conditional likelihood or log exponential-reward
expectation needed for exact conditional or reward-tilted sampling. The
generator and reward weights are frozen during sampling; gradients are enabled
only for the current latent. No gradients through a discrete sequence scorer
are required. A research implementation must validate surrogate quality and
recheck the true objectives after decoding.

## One decoder at the end

The identical `decode()` function appears in both files to keep each script
standalone. Both samplers use this same final decoding operation.

1. Undo latent standardization.
2. Apply ESM-2's frozen language-model head and retain the 20 amino-acid logits.
3. Take the highest-logit residue at each position.
4. If fewer than M positions contain a residue in `DEHKNQRST`, replace exactly
   the missing number using the lowest logit-cost changes to that set.

For the default M=12, every decoded 24-residue output contains at least 12
members of the selected set. `--min-polar 0` removes the constraint. The procedure
maximizes the sum of the fixed per-position logits subject to this minimum
count. It does not enforce the constraint along the latent trajectory, preserve
an exact conditioned generative distribution, or guarantee physical solubility.

The language-model head is a simple available decoder, not a mathematically
exact inverse of ESM-2. Generated latents can be off the encoding manifold.
For serious protein generation, validate decoding and consider training a
dedicated sequence decoder. Property improvements in the surrogate need not
survive discretization or the constrained substitutions.

## Outputs and checks

Each script writes `cfg.fasta`, `single.fasta`, `multi.fasta`, and `results.pt`
to its own output directory. The tensor file contains standardized generated
latents, both networks' state dictionaries, normalization statistics, ESM model
name, latent shape, and constraint settings. The terminal prints example
sequences and re-evaluated mean composition proxies.

Both examples have been executed through actual ESM-2 encoding, 200 training
epochs, all three guidance modes, and final decoding. Checks included finite
outputs, reward input gradients, normalization, schedule indexing, and the
hard constraint. For small test cases, constrained decoding was compared with
exhaustive search over polar/nonpolar assignments. These checks establish code
mechanics, not biological validity or reliable property optimization.

## Primary references

- ESM model: https://huggingface.co/facebook/esm2_t6_8M_UR50D
- ESM implementation: https://huggingface.co/docs/transformers/model_doc/esm
- Flow Matching: https://arxiv.org/abs/2210.02747
- DDPM: https://arxiv.org/abs/2006.11239
- Classifier-Free Diffusion Guidance: https://arxiv.org/abs/2207.12598
