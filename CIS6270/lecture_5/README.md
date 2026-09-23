# CIS 6270 - Lecture 5 - Discrete Flow Matching

Course hub: [ChatterjeeLab/CIS6270 on Hugging Face](https://huggingface.co/ChatterjeeLab/CIS6270).

Complete CPU examples for the flow-matching code in the [Discrete Generation lecture](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit). Begin with **Gat et al.'s classic discrete flow matching**, then change the path, loss, or guidance module. Lecture 4's diffusion examples are in [lecture_4/](../lecture_4/README.md). The slide deck remains combined.

The examples use synthetic DNA and small transformers to make training and generation inspectable. They are teaching implementations, not replications of large-scale paper experiments.

## Run classic DFM first

Python 3.11 or later, CPU. Data generation and training require no external downloads beyond the dependencies.

From the course repository root, enter this lecture folder.

```bash
cd lecture_5
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py --method gat --data data/dna_train.tsv --out outputs/gat
python run.py --method gat --mode sample --out outputs/gat
```

The first command trains on independent source/target pairs, validates on held-out synthetic DNA, saves a checkpoint, and generates sequences using the learned jump rates. Outputs include `config.json`, `losses.json`, `report.json`, `samples.txt`, and `checkpoint.pt`. Generation-only commands must use the checkpoint's method, width, and sequence length. Optimizer state is not saved for resumed training.

```bash
python examples/gat.py --length 4 --train-steps 160 --samples 8
python run_all.py
python run_all.py --quick
python -m unittest discover -s tests -v
python numerical_examples.py
```

## Change only the relevant module

`lecture_core.py` contains the code modules from the slides. `common.py` contains the backbone-independent data and training utilities. `flows.py` implements complete integration, teacher recoupling, guidance, and refinement loops. `run.py` selects the loss and sampler. Each `examples/*.py` file is a runnable method-specific entry point.

| Example | Training | Generation |
| --- | --- | --- |
| `gat` | Endpoint-token cross-entropy on a categorical mixture path | Euler updates from posterior-derived jump rates |
| `dirichlet` | Endpoint prediction from Dirichlet samples | Posterior average of Beta-CDF-derived conditional fields |
| `fisher` | Tangent velocity regression on square-root geodesics | Tangent integration, positive-orthant correction, spherical normalization |
| `gumbel` | Regression to the full noisy path derivative | Integrate from an approximate finite-temperature base and decode |
| `rectified` | Straight source-to-one-hot probability-path regression | Integrate the learned probability-vector field |
| `redi` | Train a Gat teacher, retain source/endpoints, train a new student | Generate from the recoupled student's rates |
| `mog-dfm` | Train a Gat backbone | Rank/direction rate reweighting and an adaptive acute hypercone |
| `areuredi` | Gat teacher followed by ReDi student training | Student generation, then annealed locally balanced MH refinement |

```bash
python examples/dirichlet.py
python examples/fisher.py --guidance 0.1
python examples/gumbel.py
python examples/rectified.py
python examples/redi.py --teacher-steps 300 --pairs 256
python examples/mog_dfm.py --preference 0.7 0.3 --strength 1
python examples/areuredi.py --preference 0.5 0.5 --refine-steps 100
```

All objectives in `common.objectives` are maximized: GC fraction and agreement with a repeating ATAT motif. They deliberately conflict. `--guidance` adds the differentiable toy GC objective to a simplex field; it is not a learned biological classifier.

## Numerical and theorem boundaries

- **Gat:** the linear schedule uses `t=n/steps`, so the singular rate at `t=1` is never evaluated. Its Euler update is valid at each position because `h <= 1-t`. Parallel token updates have finite-step joint approximation error.
- **Dirichlet:** concentration increment ranges from 0 to 8. The Beta-CDF concentration derivative is evaluated with a centered finite difference. A finite concentration endpoint is not a vertex distribution; the final output uses argmax.
- **Fisher:** the metric has the factor four under the square-root map. Spherical tangency does not ensure positive coordinates under a finite neural Euler step. The sampler records negative-coordinate corrections before projection and normalization.
- **Gumbel:** the slide construction fixes the full perturbed logits and cools temperature. Defaults are `beta=1`, `tau_max=4`, decay 4. A finite starting temperature does not remove endpoint dependence exactly; the sampler's data-independent noise base is an approximation. The finite noisy endpoint need not select the training letter. The code does not claim exact endpoint matching. `numerical_examples.py` shows why noise scale and temperature play different roles.
- **Continuous rectification:** this example trains one rectified field. The convex-cost theorem concerns the exact conditional-mean field and preserved marginals; a finite learned projected solver is approximate.
- **ReDi:** the code performs one teacher-recoupling round and stores `teacher_pairs.pt`. It does not assume that recoupling automatically decreases conditional total correlation. The lecture states the extra projection-compatibility condition needed for the data-processing argument.
- **MOG-DFM:** candidate ranks are normalized within each position. The code evaluates all positions before a parallel Euler step, rather than using the paper's random-position loop. The cone adapts within 10 to 89 degrees and does not use an outside-cone fallback, so the acute-cone local progress condition is retained. The reported guarantee is local weighted progress, not improvement in every property or global Pareto optimality.
- **AReUReDi:** the MH demonstration uses an explicitly evaluable, smoothed position-wise reference fit to the DNA training data. A denoiser alone is not treated as a joint density. The full forward/reverse proposal and reference ratios remain in acceptance. Finite annealing is not a proof of equilibrium concentration or global optimization.

The simplex solver reports how often a proposed update needed a nonnegativity correction. Increase `--sample-steps` when studying discretization; projection itself modifies the numerical dynamics.

## Numerical checks and slide mapping

The tests verify the exact master equation on all 16 two-base states, Fisher midpoint/tangency, Gumbel's path derivative with a fixed noise realization, and MH detailed balance on the complete two-base state space. `SLIDE_CODE_MAP.md` links all Lecture 5 code units to the matching functions. `verified_examples/` contains actual seeded CPU training and generation results for every method.

## Sources

- [Gat et al. - Discrete Flow Matching](https://arxiv.org/abs/2407.15595)
- [Dirichlet Flow Matching](https://arxiv.org/abs/2402.05841)
- [Fisher Flow Matching](https://arxiv.org/abs/2405.14664)
- [Gumbel-Softmax Flow Matching](https://arxiv.org/abs/2503.17361)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [ReDi](https://arxiv.org/abs/2507.15897)
- [MOG-DFM](https://arxiv.org/abs/2505.07086)
- [AReUReDi](https://arxiv.org/abs/2510.00352)
