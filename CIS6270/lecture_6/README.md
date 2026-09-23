# CIS 6270 - Lecture 6 - Flow Maps

Course hub: [ChatterjeeLab/CIS6270](https://huggingface.co/ChatterjeeLab/CIS6270).
Slides: [Lecture 6 - Flow Maps](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit).
Previous lecture: [Discrete Flow Matching](../lecture_5/README.md).

Lecture 5 learned how a distribution moves locally. Here we learn the motion over
an entire time interval. The examples progress from continuous flow maps to
latent representations, categorical text, posterior maps, expanding states,
and maps driven by a shared Brownian path.

This folder contains **16 complete training and generation examples**, with
checkpoint loading, data, mathematical checks, and saved execution results.
The code uses small PyTorch networks and inspectable datasets. Each method's
objective and sampler are implemented here; the large image and language
benchmark runs from the papers require their original architectures, datasets,
and training budgets. [SOURCES.md](SOURCES.md) records the exact papers, inspected
author-code revisions, and differences from those implementations.

## Train a velocity, then learn a finite map

Use Python 3.11 or later. The minimal requirements retain the course's PyTorch
2.9.1 baseline. All examples run on CPU; `--device cuda` selects an available GPU.
The data require no downloads.

From the course repository root:

```bash
cd lecture_6
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python run.py --method flow-matching --sample-steps 32 --out outputs/flow-matching
python run.py --method self-distill --out outputs/self-distill
python run.py --mode sample --out outputs/self-distill --sample-steps 1
```

The first command fits the conditional-mean velocity and integrates it with
Heun's method. The second trains a two-time map using a diagonal velocity loss
and a detached composition target. The third loads its checkpoint and generates
in one map evaluation. The baseline uses two velocity evaluations per Heun step;
finite maps use one evaluation per step.

Training writes `checkpoint.pt`, `config.json`, `losses.json`, `report.json`, and
`samples.txt`. A teacher or autoencoder stage also writes `teacher_losses.json`.
Generation-only mode reads model architecture and vocabulary from the checkpoint
and writes `resampled.txt` and `sample_report.json`, preserving training records.
The checkpoint contains the inference state, not optimizer state for resuming
training. `--mode train` saves training records without generating samples.

## Run every method in the same structure as Lectures 4 and 5

```bash
python run_all.py --quick
python run_all.py
python -m unittest discover -s tests -v
python numerical_examples.py --output outputs/numerical
```

`--quick` performs 20 optimization steps per method and verifies execution and
checkpoint reloading. The regular command uses 1,000 steps per training stage.
Both run every method and require seeded generated samples to agree exactly
after checkpoint reload. The quick run measures execution correctness; inspect
the longer run's losses and sample metrics when discussing learned behavior.

| Method | Training | Generation |
| --- | --- | --- |
| `flow-matching` | Conditional displacement regression on a linear interpolant | Heun integration of the learned diagonal velocity |
| `fmm-lagrangian` | Train a velocity teacher, then match the map's arrival-time derivative to the teacher | Residual finite-time maps |
| `fmm-eulerian` | Train a velocity teacher, then match the start-time directional derivative to zero | Residual finite-time maps |
| `self-distill` | Diagonal flow matching and EMA two-subinterval composition targets | One or more direct map updates |
| `consistency` | Endpoint consistency along short Heun-solved teacher intervals, with the diagonal anchor | One endpoint prediction |
| `shortcut` | Conditional flow matching and two-half-step average-velocity targets on dyadic intervals | One or more shortcuts |
| `meanflow` | Backward average-velocity target with a directional JVP and detached adaptive weights | Backward updates from noise at time one to data at zero |
| `latent` | Train a 3D-to-2D autoencoder, freeze it, standardize its encodings, then train a flow map | Latent map followed by the saved decoder |
| `fmlm` | Diagonal token CE and weighted progressive denoiser consistency | Warped-time affine categorical maps, followed by argmax |
| `categorical` | Diagonal CE and endpoint consistency plus temporal-derivative energy | The same categorical map parameterization |
| `discrete-lsd` | Diagonal CE and Lagrangian logit-corrected KL teacher | Categorical finite-time maps |
| `discrete-esd` | Diagonal CE and Eulerian directional-JVP logit teacher | Categorical finite-time maps |
| `diamond` | Distill a GLASS posterior velocity built from an analytic Gaussian-mixture denoiser | Conditional posterior samples, differentiable value estimates, and reward steering |
| `meta` | Independent inner and outer noise with a shared data endpoint, diagonal regression, and conditional composition | Posterior samples and reward-weighted posterior-mean steering |
| `expanding` | Local-clock token CE, compatible-canvas composition, and remaining/interval gap-count losses | Learned binomial gap insertion, budget capping, and local-clock transport |
| `ssfm` | OU drift anchoring, small-step stochastic matching, and same-noise composition | Strong maps using consistently aggregated Brownian coefficients |

Each method also has its own entry point:

```bash
python examples/fmm_lagrangian.py
python examples/fmm_eulerian.py
python examples/consistency.py
python examples/shortcut.py
python examples/meanflow.py --sample-steps 1
python examples/latent.py --teacher-steps 2000
python examples/fmlm.py --sample-steps 1
python examples/categorical.py
python examples/discrete_lsd.py
python examples/discrete_esd.py
python examples/diamond.py --posterior-steps 4 --particles 64
python examples/meta.py --finetune-steps 300
python examples/expanding.py --sample-steps 4
python examples/ssfm.py --sample-steps 4
python examples/ssfm.py --ssfm-target paper --lr 0.0001 --train-steps 2000 --out outputs/ssfm-paper
```

The common options are `--train-steps`, `--teacher-steps`, `--batch-size`,
`--width`, `--lr`, `--seed`, `--samples`, `--sample-steps`, and `--out`.
`--teacher-steps` controls the teacher or autoencoder stage where one exists.
Run `python run.py --help` for the remaining options. The `consistency` example
always makes one endpoint prediction. SSFM requires a power-of-two sampling
step count so its finite Brownian tree supports every requested partition.

## Change one component at a time

`lecture_core.py` exposes the functions used in the slide walkthroughs.
`common.py` contains networks, data handling, optimization, and reference
distributions. Each method family has a separate implementation file.

| File | Responsibility |
| --- | --- |
| `continuous.py` | Diagonal, Lagrangian, Eulerian, semigroup, Shortcut, MeanFlow, and latent training |
| `categorical.py` | Affine probability-valued maps, inverse decoding clock, ECLD, and logit teachers |
| `posterior.py` | GLASS fusion, conditional map training, posterior values, importance correction, and reward fine-tuning |
| `expanding.py` | Birth times, compaction, gap labels, count losses, and expansion/transport sampling |
| `stochastic.py` | Brownian integrals, Chen composition, OU training, and strong-error evaluation |
| `numerical_examples.py` | The lecture's numerical examples, including the trained exponential-flow example |
| `SLIDE_CODE_MAP.md` | Stable links from every code slide and major method to the implementation |
| `MATHEMATICS.md` | Equations, variable definitions, and the meaning of each loss |
| `verified_examples/` | Actual seeded training logs, generated samples, and verification reports |

The small sequence MLP receives the entire sequence. Its per-position output
can therefore depend on every input position. The variable-length network also
receives padding masks and birth-time metadata. Replacing these backbones with
transformers does not require changing the displayed map identities.

## Data and evaluation

The continuous examples use four 2D Gaussians with known centers and variance.
Reports include nearest-center distance, mode fractions, mode entropy, and exact
target log density. These metrics describe different aspects of the samples;
high mode entropy alone does not establish sample fidelity.

`data/phrases.txt` contains correlated four-token phrases such as `red circle
moves left`. `data/variable_text.txt` contains two-, four-, and five-token
versions. Reports record token entropy, unique fraction, training-support
fraction, and lengths. These tiny-data statistics are not LM1B/OWT perplexity or
large-language-model evaluations. The split is a seeded 80/20 split of examples;
the small grammar intentionally appears in both splits.

Custom fixed-length text can be supplied with `--data my_phrases.txt`.
The expanding example also accepts variable-length lines, up to 32 tokens.
Continuous methods accept `--data points.csv`, with two finite numeric columns
and no header. Diamond, Meta, and SSFM retain their specified reference
distributions so their analytic diagnostics remain valid. See [data/README.md](data/README.md).

## Numerical and theorem boundaries

- **Continuous maps.** The diagonal is exactly the identity as a map, while its
  predicted average velocity is trained by flow matching. Interpolant samples
  are not individual ODE trajectories. Diagonal regression has an irreducible
  conditional-variance floor, so raw training loss need not approach zero.
- **Teacher distillation.** The finite-map teachers are learned in a separate
  first stage. This retains teacher approximation error. The fixed MLP, bounded
  interval sampling, and loss weights are classroom choices.
- **Consistency and Shortcut.** Endpoint consistency is a compact
  flow-matching-clock teaching implementation. Shortcut uses dyadic interval
  lengths, a continuous start-time sample, EMA targets, and a separate diagonal
  loss. The authors' large-image code has additional schedules and clipping.
- **MeanFlow.** Data are at time zero and noise at one. The directional JVP uses
  the conditional training velocity, and the entire corrected target is
  detached. Seventy-five percent of training examples use equal times. Uniform
  time sampling replaces the author's default logit-normal schedule.
- **Latent maps.** This section demonstrates a learned representation and
  decoder. It does not claim a separate canonical paper named Latent Flow Maps.
  Reconstruction error and latent transport error are reported separately.
- **Categorical maps.** Only the predicted denoiser lies on the simplex during
  the trajectory. The final step reaches time one exactly; argmax is an explicit
  decoding approximation for an imperfect model. We use Gaussian quadrature
  for the decoding clock. Categorical ECLD uses the released code's bounded
  endpoint-plus-time-energy objective, with an EMA endpoint target. Detached KL
  and CE have the same student gradient; they log different scalar values.
- **Discrete logit teachers.** The logarithmic correction requires positive
  denominators. The implementation clamps them at 0.05 and logs the affected
  fraction. This changes the teacher away from the exact solution; softmax
  alone cannot repair an invalid logarithm. Differential objectives stay below
  terminal time 0.97. No unreported gradient surgery is applied.
- **Diamond and Meta.** The two noises are independent and share a clean
  endpoint. Diamond uses an exact, evaluable mixture denoiser as its GLASS
  teacher; Meta learns directly from samples. Posterior diagnostics compare
  both means and variances. Finite importance estimates and reward-weighted
  posterior means have self-normalization bias. The weighted-Diamond companion
  uses an explicit full-support Gaussian proposal with a known density.
- **Meta fine-tuning.** `--finetune-steps` runs and saves the detached surrogate
  from Equation 43. It uses a bounded coordinate reward and the analytic base
  mixture drift. Training is restricted to outer times 0.05 through 0.85;
  extending the learned drift outside that interval is extrapolation.
- **Expanding maps.** The data use a linear birth CDF. The network predicts
  remaining gap means and interval means through the conditional insertion
  factor. Binomial proposals and global budget capping approximate a joint count
  law; a mean alone does not identify that law. A finite sampling jump introduces
  new coordinates at its start, then transports them. Training compares routes
  on a shared destination canvas with compatible noise and local clocks.
  These are explicit finite-step choices, not a proof of exact conditional-law
  recovery by the fitted network. Empty generated sequences remain visible in
  the reports; the sampler does not silently replace them.
- **Strong stochastic maps.** The example is an additive-noise OU SDE with a
  known diffusion coefficient. It uses the first two Legendre integrals and a
  learned average drift. Default `official-code` targets detach an EMA split
  prediction; `--ssfm-target paper` detaches the direct prediction, as in the
  lecture's pseudocode. Both reuse exactly the same Brownian history. The
  reference is a fine Euler-Maruyama solve on that history, so its residual
  discretization error is retained. No claim is made for multiplicative noise
  or exact pathwise recovery with only two coefficients. The optional paper
  orientation remained inaccurate in the recorded experiments, including the
  lower-learning-rate recipe above; see the explicit sensitivity results in
  `verified_examples/README.md`. Use the default released-code orientation for
  the successful OU demonstration.

## Read the saved results with the implementation

[verified_examples/README.md](verified_examples/README.md) records the executed
environment, commands, and metrics. Checkpoints are generated locally and are
excluded from this folder's Git history, matching Lectures 4 and 5. The saved
text/JSON records let students inspect a complete run before training their own
models. The [slide code map](SLIDE_CODE_MAP.md) connects that run to the lecture.
