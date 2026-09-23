# Lecture 7 · Optimal Transport and Schrödinger Bridges

[Native lecture](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit) · [Slide code map](SLIDE_CODE_MAP.md) · [Equation readings](EQUATION_READINGS.md) · [Teaching notes](lecture7_ot_sbm_notes.md) · [Primary sources](SOURCES.md)

We first solve OT and Sinkhorn problems, then construct continuous and discrete Schrödinger bridges and train the matching methods. The final lecture closes the course by connecting local dynamics, finite maps, endpoint couplings, and reference path laws.

## Run, save, and generate

From the repository root:

```bash
python -m pip install -r lecture_7/requirements.txt
python lecture_7/run.py --method sf2m --out lecture_7/outputs/sf2m
python lecture_7/run.py --mode sample --out lecture_7/outputs/sf2m
python lecture_7/run_all.py --quick
python -m unittest discover -s lecture_7/tests -v
```

The default `train-sample` mode solves or trains, saves `checkpoint.pt`, and generates samples. `--mode train` only solves/trains and saves. `--mode sample --out RUN` reloads the checkpoint and generates again. Runs write `config.json`, `report.json`, `losses.json`, `samples.json`, `samples.txt`, and `generation_report.json`; reloads write `resampled.json`, `resampled.txt`, and `sample_report.json`. Exact methods save their solved probability objects; learned methods save their fitted parameters.

## All method commands

```bash
python lecture_7/run.py --method ot --out lecture_7/outputs/ot
python lecture_7/run.py --method sinkhorn --out lecture_7/outputs/sinkhorn
python lecture_7/run.py --method finite-sb --out lecture_7/outputs/finite-sb
python lecture_7/run.py --method discrete-imf --out lecture_7/outputs/discrete-imf
python lecture_7/run.py --method ctmc-sb --out lecture_7/outputs/ctmc-sb
python lecture_7/run.py --method gaussian-sb --out lecture_7/outputs/gaussian-sb
python lecture_7/run.py --method reward-tilt --out lecture_7/outputs/reward-tilt
python lecture_7/run.py --method branch-mass --out lecture_7/outputs/branch-mass
python lecture_7/run.py --method cone-geometry --out lecture_7/outputs/cone-geometry
python lecture_7/run.py --method dsb --out lecture_7/outputs/dsb
python lecture_7/run.py --method dsbm --out lecture_7/outputs/dsbm
python lecture_7/run.py --method sf2m --out lecture_7/outputs/sf2m
python lecture_7/run.py --method tr2d2 --out lecture_7/outputs/tr2d2
python lecture_7/run.py --method branch --out lecture_7/outputs/branch
python lecture_7/run.py --method entangled --out lecture_7/outputs/entangled
python lecture_7/run.py --method ddsbm --out lecture_7/outputs/ddsbm
python lecture_7/run.py --method csbm --out lecture_7/outputs/csbm
```

The 17 numbered files in `examples/` are independent entry points for those same methods. For example, `python lecture_7/examples/12_sf2m.py --quick` uses the common runner. `lecture7_examples.ipynb` provides an interactive walkthrough. All examples use synthetic data and run on CPU. They require no network access after installing dependencies.

## Training and sampling settings

Use `--samples N` for the number of generated samples and `--seed S` for reproducibility. Sampling uses seed S plus 100. `--train-steps` applies to SF2M, Branch, DDSBM, and CSBM; `--rounds` to DSB and DSBM; `--epochs` to TR2-D2 and Entangled; and `--searches` to TR2-D2 rollouts. `--batch-size` applies only where the selected training function accepts it. Inapplicable flags produce a clear error.

`--sample-steps` changes the grid for samplers that support it. DSB and DSBM checkpoints retain their fitted 80-step grid; finite CSBM and the fixed-reveal TR2-D2 model also retain their trained discrete grids. The exact finite-state CTMC sampler uses matrix-exponential Doob transitions; learned DDSBM uses a finite midpoint grid. `--quick` checks execution with shorter training and is not a quality result.

Run `python lecture_7/run_all.py` to reproduce full training and verify seeded checkpoint reloads for every method. [Verified examples](verified_examples/README.md) contain the actual run records. The mathematical test runs 24 independent checks, including duality, endpoint marginals, path conditioning, master equations, score/flow factors, importance weights, and finite-step cone geometry.

## What each implementation demonstrates

| File or function | Computation | Scope |
|---|---|---|
| `transport_example` | LP solution, dual certificate, stable Sinkhorn | Exact 2×2 teaching problem |
| `finite_bridge_example` | Schrödinger potentials, Doob transitions, all paths | Exact two-state finite chain |
| `discrete_imf_example` | Markov and reciprocal projections | Enumerates all sixteen paths |
| `ctmc_bridge_example` | Matrix exponentials and Doob rates | Exact finite-state continuous-time bridge |
| `gaussian_example` | Analytic covariance, mean, variance, drift | Brownian Gaussian bridge |
| `dsb` | Original reverse-regression targets and endpoint resets | Affine models; finite Gaussian reverse kernels |
| `dsbm` | Conditional Brownian drift regression and alternating directions | Affine fields; Monte Carlo and Euler approximation |
| `sf2m` | Joint conditional velocity and score matching | Small MLP; exact Gaussian endpoint coupling |
| `fit_ddsbm` | Conditional jump-rate divergence loss | Neural Markov projection at a known small SB coupling |
| `fit_csbm` | Categorical transition cross-entropy | Full transition tables; known finite path weights |
| `tr2d2` | Tree selection, expansion, rollout, backup, replay, WDCE | Abstract tokens; recorded full-support search proposal |
| `branch` | Interpolant energy, velocity matching, growth, joint refinement | Three branches; fixed coupling; soft mass penalties |
| `entangled` | Coupled cone bias, Euler path likelihood, weighted CE | Interacting abstract particles in normalized units |

These implementations preserve the displayed objectives and training operations while using small model classes and tractable examples. They do not reproduce the papers’ benchmark architectures, datasets, or reported results.

## Distinctions to retain

- Sinkhorn finds an endpoint coupling. A path law additionally needs conditional reference dynamics.
- DSB and DSBM use different projection constructions. The examples explicitly implement both directions and reset the appropriate endpoint.
- DDSBM uses continuous time and discrete states. CSBM here uses discrete time and discrete states. The exact outer IMF iteration and its learned inner projections are separated so they can be checked independently.
- The TR2-D2 teaching search keeps all rollouts, uses full-support softmax selection, and records its actual proposal likelihood. This makes importance weighting inspectable. It differs from heuristic curation of a paper-scale replay buffer. A pure model likelihood ratio alone does not undo search selection.
- Branching uses a fixed endpoint pairing in the toy interpolant. Its soft penalties leave small measurable mass errors, recorded in `verified_examples`. A conditional path optimization is not a proof of global optimality over all couplings.
- Entangled control is expressed in noise-normalized units. The geometric guarantee applies to the bias contribution. The synthetic CE example uses a terminal potential and a fixed sampled initial law; it does not claim to enforce an arbitrary pair of endpoint densities.
- Original paper figures and benchmark values in the lecture are attributed results. Course-generated plots come from the included computations.

## Lecture map

| Section | Slides |
|---|---|
| Choosing an optimal transport | 2–70 |
| Transport along a probability path | 71–84 |
| Computing transport with Sinkhorn | 85–120 |
| From a coupling to a Schrödinger bridge | 121–186 |
| Continuous bridges and stochastic control | 187–241 |
| Diffusion Schrödinger Bridge | 242–271 |
| Diffusion Schrödinger Bridge Matching | 272–305 |
| Simulation-free score and flow matching | 306–328 |
| Bridges on discrete spaces | 329–382 |
| Categorical Schrödinger Bridge Matching | 383–411 |
| Reward-guided discrete bridges with TR2-D2 | 412–475 |
| Branching Schrödinger Bridge Matching | 476–535 |
| Entangled Schrödinger Bridge Matching | 536–616 |
| Putting the formulations together | 617–629 |
| Course wrap-up | 630–651 |

`lecture7_ot_sbm_notes.md` includes every derivation sequence, proof note, source, and direct link to its completed native slide. `EXERCISES.md` includes short extensions and answers. `verified_examples/` contains the runs used to validate this delivery.
