---
license: mit
language:
  - en
tags:
  - education
  - pytorch
  - flow-matching
  - diffusion
  - esm2
  - protein-generation
  - guidance
  - mnist
  - image-generation
---

# CIS 6270 Course Code 
# Taught by Prof.Pranam Chatterjee
# My reproduction of code and corresponding projects


![cis6270-pic](https://cdn-uploads.huggingface.co/production/uploads/64cd5b3f0494187a9e8b7c69/xt1nYYJTgQPVqtQEhC2Me.png)

This repository contains code examples for CIS 6270, organized by lecture.
We will develop the implementations alongside the course material, which connect
the mathematical definitions, training objectives, and sampling equations to
the executable PyTorch code herein. Each lecture directory contains its examples, data,
and implementation notes.

## Lecture index

| Lecture | Topics | Code and documentation |
| --- | --- | --- |
| 2 | Unconditional MNIST image generation with flow matching and a simple U-Net | [Guide](lecture_2/README.md) · [Script](lecture_2/flow_matching_unet_lecture.py) · [Saved checkpoint](https://huggingface.co/ChatterjeeLab/CIS6270/resolve/main/lecture_2/flow_unet_mnist.pt?download=true) |
| 3 | Flow matching, diffusion, and guidance for ESM-2 residue embeddings | [Guide](lecture_3/README.md) · [Flow matching](lecture_3/esm2_flow_guidance.py) · [Diffusion](lecture_3/esm2_diffusion_guidance.py) |
| 4 | Discrete diffusion, masked and uniform corruption, block generation, and guidance | [Guide](lecture_4/README.md) · [Training and generation](lecture_4/run.py) · [Slide code map](lecture_4/SLIDE_CODE_MAP.md) |
| 5 | Discrete flow matching, Dirichlet and Fisher paths, Gumbel-Softmax, rectification, and multi-objective generation | [Guide](lecture_5/README.md) · [Training and generation](lecture_5/run.py) · [Slide code map](lecture_5/SLIDE_CODE_MAP.md) |
| 6 | Continuous, latent, categorical, posterior, expanding, and strong stochastic flow maps | [Guide](lecture_6/README.md) · [Training and generation](lecture_6/run.py) · [Slide code map](lecture_6/SLIDE_CODE_MAP.md) |
| 7 | Optimal transport, Sinkhorn, continuous and discrete Schrödinger bridges, matching, reward tilting, branching, and interacting control | [Guide](lecture_7/README.md) · [Training and generation](lecture_7/run.py) · [Slide code map](lecture_7/SLIDE_CODE_MAP.md) |

## Installation

Use **Python 3.11** (compatible range: `>=3.10,<3.13`). Dependencies are declared in
`pyproject.toml` (preferred) and mirrored in `requirements.txt`. Shared pins:

| Package | Version |
| --- | --- |
| PyTorch | 2.9.1 |
| TorchVision | 0.24.1 |
| Transformers | 4.57.6 |
| NumPy | ≥1.24 |
| SciPy | ≥1.10 |

Lecture 2 uses PyTorch and TorchVision; Lecture 3 also uses Transformers.
Lectures 4–5 and 7 use PyTorch, NumPy, and SciPy; Lecture 6 uses PyTorch and
NumPy. Those lecture folders also ship minimal `requirements.txt` files.

### NVIDIA DGX Spark (this machine)

This checkout is set up for an **NVIDIA DGX Spark** with **GB10** (compute
capability **12.1** / `sm_121`), **aarch64**, driver **580.x**, and system
**CUDA 13.0**. PyTorch is installed from the official **cu130** wheel index
(`torch==2.9.1+cu130`). Official builds list `sm_120`; that is binary-compatible
with `sm_121` — a capability warning on import is expected and safe to ignore.

### Recommended: [uv](https://docs.astral.sh/uv/) + `pyproject.toml`

```bash
git clone https://huggingface.co/ChatterjeeLab/CIS6270
cd CIS6270
git lfs pull

# Creates .venv with Python 3.11 and installs deps (torch/vision from cu130 on Linux)
uv sync --python 3.11

source .venv/bin/activate
# or run without activating:
# uv run python lecture_2/flow_matching_unet_lecture.py
```

Verify GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# Expected on DGX Spark: 2.9.1+cu130  True  NVIDIA GB10  (12, 1)
```

Re-sync after pulling dependency changes: `uv sync`.

### Pip fallback (no uv)

On DGX Spark / CUDA 13 aarch64, pull torch from the cu130 index (plain PyPI
wheels are often CPU-only on this platform):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

On other Linux/Windows/macOS machines without Blackwell/CUDA 13, the course
`requirements.txt` / default PyPI install is usually enough:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The saved checkpoint uses Git LFS. If Git LFS is unavailable, download
[flow_unet_mnist.pt](https://huggingface.co/ChatterjeeLab/CIS6270/resolve/main/lecture_2/flow_unet_mnist.pt?download=true)
directly and place it in `lecture_2/`. Training from scratch does not require
the saved checkpoint.

## Lecture 2: MNIST flow matching

Lecture 2 contains one self-contained, commented script. It downloads MNIST,
loads batches of normalized images, defines a small U-Net velocity network,
trains it with conditional flow matching, and generates images from Gaussian
noise using Euler integration.

From the repository root:

```bash
python lecture_2/flow_matching_unet_lecture.py
```

The default run trains for 20 epochs and writes `samples.png`, `trajectory.png`,
and a new `flow_unet_mnist.pt` to `flow_matching_outputs/`. The script selects
CUDA, Apple MPS, or CPU according to availability. Its numbered `# %%` sections
match the lecture walkthrough.

The bundled [checkpoint](lecture_2/flow_unet_mnist.pt) has **5 completed epochs**
on all 60,000 MNIST training images. The [Lecture 2 guide](lecture_2/README.md)
includes a short example for loading it and generating images without training,
along with selected generated digits and their noise-to-image trajectories.

## Lecture 3: Flow and diffusion guidance

From the repository root:

```bash
python lecture_3/esm2_flow_guidance.py --epochs 200 --samples 8
python lecture_3/esm2_diffusion_guidance.py --epochs 200 --samples 8
```

Both scripts train on ESM-2 residue embeddings and compare classifier-free
guidance, single-objective reward steering, and scalarized multi-objective
steering. We use the same property definitions, normalization, and final
constrained decoder to compare the flow and diffusion implementations.

The first run downloads the public ESM-2 checkpoint. Each script then writes
generated sequences and model parameters to its own output directory.
The [lecture guide](lecture_3/README.md) describes the data format, training
and sampling settings, property calculations, normalization, and residue-count
constraint, with commands for using a custom dataset.

## Lecture 4 - Discrete diffusion

Train small DNA denoisers and generate sequences with MDLM, UDLM, block diffusion,
classifier-free guidance, exact and gradient-based classifier guidance, and a
PepTune-style search. The [guide](lecture_4/README.md) includes each method's
command and mathematical assumptions. The [code map](lecture_4/SLIDE_CODE_MAP.md)
links the slide walkthroughs to their functions.

From the repository root, run the complete MDLM example.

```bash
python lecture_4/run.py --method mdlm --data lecture_4/data/dna_train.tsv --out lecture_4/outputs/mdlm
python lecture_4/run.py --method mdlm --mode sample --out lecture_4/outputs/mdlm
```

The script trains, saves a checkpoint, and writes generated DNA and loss logs.
The bundled data are synthetic, and the guidance objectives are explicit toy
properties. No pretrained model or external dataset is required.

## Lecture 5 - Discrete flow matching

Start with Gat et al.'s discrete flow matching, then run Dirichlet, Fisher,
Gumbel-Softmax, rectified flow, ReDi, MOG-DFM, and AReUReDi examples. Each method
has a complete training and generation command in the [guide](lecture_5/README.md).

```bash
python lecture_5/run.py --method gat --data lecture_5/data/dna_train.tsv --out lecture_5/outputs/gat
python lecture_5/run.py --method gat --mode sample --out lecture_5/outputs/gat
```

Both folders include numerical examples, mathematical tests, and saved results
from seeded CPU runs. The guides explain finite endpoint approximations and
classroom simplifications for each method.

## Lecture 6 - Flow Maps

Learn finite-time motion after the local flows from Lecture 5. The new folder
contains 16 complete examples covering flow-map matching and self-distillation,
consistency, Shortcut, MeanFlow, learned latent representations, Flow Map
Language Models, Categorical and Discrete Flow Maps, Diamond Maps, Meta Flow
Maps, Expanding Flow Maps, and Strong Stochastic Flow Maps.

```bash
python lecture_6/run.py --method self-distill --out lecture_6/outputs/self-distill
python lecture_6/run.py --mode sample --out lecture_6/outputs/self-distill --sample-steps 1
python lecture_6/run_all.py --quick
```

The [Lecture 6 guide](lecture_6/README.md) includes all method commands,
training objectives, data formats, and numerical assumptions. The
[slide code map](lecture_6/SLIDE_CODE_MAP.md) links implementations to the
[Flow Maps presentation](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit).
[Source notes](lecture_6/SOURCES.md) identify the exact papers and inspected
author-code revisions, including differences between paper pseudocode and
released implementations. [Verified examples](lecture_6/verified_examples/README.md)
contain actual training logs, generated samples, and checkpoint-reload checks.

## Lecture 7 - Optimal Transport and Schrödinger Bridges

The final lecture includes 17 complete numerical and learned examples: OT,
Sinkhorn, finite and continuous-time bridges, DSB, DSBM, SF2M, DDSBM, CSBM,
TR2-D2, BranchSBM, and EntangledSBM. Each runner saves its solved distribution
or learned parameters and can reload them to generate samples.

```bash
python lecture_7/run.py --method sf2m --out lecture_7/outputs/sf2m
python lecture_7/run.py --mode sample --out lecture_7/outputs/sf2m
python lecture_7/run_all.py --quick
```

The [guide](lecture_7/README.md), [mathematical notes](lecture_7/MATHEMATICS.md),
[slide code map](lecture_7/SLIDE_CODE_MAP.md), and [verified examples](lecture_7/verified_examples/README.md)
follow the earlier lecture structure. The synthetic demonstrations distinguish
exact calculations, finite discretizations, and learned approximations.
Lectures [6](lecture_6/EQUATION_READINGS.md) and [7](lecture_7/EQUATION_READINGS.md)
also include spoken readings of their equations before the intuition.

## Project code (`src/dgm`)

Project code is an installable package under a `src/` layout, separate from the
lecture scripts. `uv sync` installs it editable into `.venv`, which puts every
runnable module on the path as a console script.

```
src/dgm/
  common/              shared by every project
    paths.py           where the repository, artifacts and vendored METL live
    esm_models.py      the ESM-2 model registry
  project1/            flow matching vs diffusion on ESM-2 protein latents
    run_experiment.py  train, sample every guidance arm, save
    evaluate.py        figures and metric tables
    embedding_oracle.py  the independent brightness oracle
    ...
    pipeline/          experiment stages: encode, train, guide, sample, decode
    evaluation/        figures and metric tables
    oracles/           METL and ESM-2 embedding oracles
    comparison/        cross-model scaling figures
    diagnostics/       standalone checks on the objective and conditioning
```

Run a command either way; both work from any directory:

```bash
uv sync                                     # once, installs dgm editable
dgm-run-experiment --esm-model esm2_8m --dataset data/avgfp_train_4000.csv
python -m dgm.project1.run_experiment --list-models
```

Every module listed above has a console script named after it
(`dgm-evaluate`, `dgm-gfp-metrics`, `dgm-embedding-oracle`, and so on).

### Experiment tracking

Runs log to Weights & Biases at `proterial/CIS6270`. Nothing in the experiment
code knows about it — `dgm-wandb` reads the `results.pt` that every run already
writes, which means loss curves come out at full per-epoch resolution and any
past run can be backfilled.

```bash
# wrap a run: opens the W&B run first, streams output, logs when it finishes
dgm-wandb run --name long250_20261001 --tags local gb10 -- \
    --esm-model esm2_8m --dataset data/avgfp_train_props.csv --epochs 250

# log a run that already exists, including old ones
dgm-wandb sync --run-dir project1_eval/outputs/esm2_8m_long250
```

What gets logged: both modalities' per-epoch loss curves; per-arm oracle
brightness, Hamming distance and uniqueness as both a sortable table and flat
summary keys; the full 45-key run configuration; host and GPU; and the FASTA
files and plots as an artifact. `results.pt` itself is not uploaded — it is
hundreds of megabytes and belongs in OSDF.

Override the destination with `--entity`/`--project` or `WANDB_ENTITY`/
`WANDB_PROJECT`. For OSG, see [`osg/README.md`](osg/README.md#weights--biases).

### Where results go

Code no longer sits next to its data, so paths are resolved in
`dgm.common.paths` rather than from each module's own location. Project 1 reads
and writes `project1_eval/`:

| Directory | Contents |
| --- | --- |
| `project1_eval/data/` | prepared CSVs, splits, wild-type sequences, fitted oracles |
| `project1_eval/outputs/` | one directory per run: `results.pt` and FASTA files |
| `project1_eval/plots/` | figures and the metric CSVs behind them |
| `project1_eval/cache/` | HuggingFace weights, shared by every model and run |

Set `DGM_ROOT` to point the package at a different checkout, or `METL_ROOT` to
a METL clone elsewhere. A later project adds itself to `_PROJECT_DIRS` in
`dgm/common/paths.py` and gets the same helpers.

## Repository organization

| Location | Contents |
| --- | --- |
| Repository root | Course index, installation requirements, and license |
| `src/dgm/` | Installable project code: `common/` plus one package per project |
| `project1_eval/` | Project 1 artifacts: data, outputs, plots, weight cache, sweep drivers |
| [`lecture_2/`](https://huggingface.co/ChatterjeeLab/CIS6270/tree/main/lecture_2) | One MNIST flow-matching script, guide, trained checkpoint, and selected example images |
| [`lecture_3/`](https://huggingface.co/ChatterjeeLab/CIS6270/tree/main/lecture_3) | ESM-2 flow and diffusion guidance scripts, sequence data, guide, and mathematical notes |
| [`lecture_4/`](https://huggingface.co/ChatterjeeLab/CIS6270/tree/main/lecture_4) | Seven discrete diffusion and guidance examples, synthetic DNA, slide code map, and verified outputs |
| [`lecture_5/`](https://huggingface.co/ChatterjeeLab/CIS6270/tree/main/lecture_5) | Eight discrete and simplex flow examples, synthetic DNA, slide code map, and verified outputs |
| [`lecture_6/`](https://huggingface.co/ChatterjeeLab/CIS6270/tree/main/lecture_6) | Sixteen flow-map implementations, text data, mathematical notes, slide links, and verified outputs |
| [`lecture_7/`](https://huggingface.co/ChatterjeeLab/CIS6270/tree/main/lecture_7) | Seventeen OT and bridge examples, derivations, equation readings, slide links, tests, and verified outputs |

Installation instructions and the lecture index are maintained at the
repository root. Lecture-specific commands, mathematical explanations, and
references accompany the corresponding code.

## License

The repository code is distributed under the
[MIT License](https://huggingface.co/ChatterjeeLab/CIS6270/blob/main/LICENSE).
ESM-2 weights are downloaded from their original repository under the
corresponding distribution terms.
