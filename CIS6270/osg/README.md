# Running Project 1 on the OSPool

Submitting GPU jobs to OSG via HTCondor. Nothing here changes the existing
code: `src/dgm` is transferred and put on `PYTHONPATH`, and the two hooks it
already honours — `DGM_ROOT` and `ESM2_CACHE` — are what point it at the data
on the worker node.

```
osg/
  env/
    dgm-gpu.def            Apptainer image on OSG's PyTorch base (default)
    dgm-gpu-cuda126.def    fallback: OSG Rocky 9 + CUDA 12.6, torch pinned here
  bin/
    stage.sh               build the three input tarballs
    prep_run.sh            create a run's OSDF output folder before submitting
    fetch_esm_weights.py   populate a weight cache for staging
  jobs/
    run_experiment.sub     one run
    sweep.sub              one job per line of sweep_params.txt
    sweep_params.txt       tag, seed, eta per line
    run_experiment.sh      what actually runs on the node
    logs/                  Condor out/err/log land here
```

## Why not just use the local pins

`pyproject.toml` pins `torch==2.9.1+cu130` for this laptop's GB10 (sm_121).
Those wheels need NVIDIA driver >= 580, which most OSPool nodes do not have, so
jobs would either not match or fail at CUDA init. The image instead builds on
OSG's own PyTorch base (torch 2.3.1 / CUDA 11.8), whose wheels cover sm_70
through sm_90 — V100, A100, A40, L40S. Results will not be bit-identical to
local runs, which is true of any different GPU.

## One-time setup

### 1. Clone and install on the access point

```bash
git clone <your-repo-url> ~/Discrete_Generative_Models
cd ~/Discrete_Generative_Models/CIS6270
```

No `uv sync` is needed on the access point for submitting — the jobs carry
their own environment. Install it only if you also want to run `dgm-gfp-metrics`
there to score results.

### 2. Get the inputs onto the access point

`project1_eval/data/` and `project1_eval/cache/` are gitignored, so a fresh
clone has neither. Both are small for `esm2_8m`:

```bash
# data: scp from your laptop (13 MB), or rebuild with dgm-prepare-gfp
mkdir -p project1_eval/data
scp you@laptop:.../project1_eval/data/avgfp_{train_props.csv,wt.txt,oracle_v2.npz} \
    project1_eval/data/

# weights: fetch once (30 MB for esm2_8m)
PYTHONPATH=$PWD/src python3 osg/bin/fetch_esm_weights.py --model esm2_8m
```

### 3. Build the image

The `TMPDIR` exports and `--ignore-proot` are OSPool policy, not optional —
building without them strains shared storage.

```bash
mkdir -p $HOME/tmp
export TMPDIR=$HOME/tmp APPTAINER_TMPDIR=$HOME/tmp APPTAINER_CACHEDIR=$HOME/tmp
cd osg/env
apptainer build --ignore-proot dgm-gpu-v1.sif dgm-gpu.def
```

The definition's `%test` block imports every dependency, so an incompatible pin
fails the build rather than every job. If it fails on `transformers` against
torch 2.3.1, build `dgm-gpu-cuda126.def` instead — it pins torch itself — and
set `IMAGE_VERSION` accordingly in the submit files.

A PyTorch `.sif` is several GB, so stage it in OSDF rather than `/home`:

```bash
cp dgm-gpu-v1.sif /ospool/$AP/data/$USER/
```

**Bump the version on every rebuild.** OSDF caches by file name, so a rebuilt
`dgm-gpu-v1.sif` can keep serving the old image.

### 4. Fill in your identity

In both `jobs/run_experiment.sub` and `jobs/sweep.sub`:

```
OSG_USER = mukul.sherekar
OSG_AP   = ap40

OSDF_IMAGE = osdf:///ospool/$(OSG_AP)/data/$(OSG_USER)/containers
OSDF_RUNS  = osdf:///ospool/$(OSG_AP)/data/$(OSG_USER)/DGM/runs
OSDF_DATA  = osdf:///ospool/$(OSG_AP)/data/$(OSG_USER)/DGM/data
```

Three locations, not one. `OSDF_IMAGE` is read by every job; `OSDF_RUNS`
receives each run's heavy outputs; `OSDF_DATA` is only for weight caches too
big to transfer. Collapsing these into a single variable is what sent the first
smoke run's `results.pt` to `containers/runs/`.

## Running

```bash
cd ~/Discrete_Generative_Models/CIS6270/project1_eval
bash ../osg/bin/stage.sh --model esm2_8m \
    --data data/avgfp_train_props.csv data/avgfp_wt.txt data/avgfp_oracle_v2.npz

cd ../osg/jobs
OSG_AP=ap40 OSG_USER=$USER bash ../bin/prep_run.sh long250_20261001
condor_submit run_experiment.sub -append 'TAG = long250_20261001'
condor_q
```

Staging produces about 21 MB in total — 128 KB of code, 1.8 MB of inputs,
19 MB of weights — so everything goes through `transfer_input_files`. Re-run
`stage.sh` whenever the code or the input files change.

`run_experiment.sub` carries the full `long250` flag set as its default
`RUN_ARGS`, character-for-character what you run locally. Edit it, or override:

```bash
condor_submit run_experiment.sub -append 'TAG = k5_t09'
```

For a sweep, the varying values live in `sweep_params.txt` and the shared flags
in `BASE_ARGS` inside `sweep.sub`:

```bash
condor_submit sweep.sub
```

That split is forced by HTCondor: `queue ... from` splits each line on commas,
so a value containing a comma — `--freeze-positions 63,64,65` — cannot live in
the params file.

## Resource requests, measured

Measured locally on the GB10, then **confirmed against a real OSPool run** on
an L40 at NCSU-OSG-CE1 (job 15833608).

| Request | Value | Why |
| --- | --- | --- |
| `request_cpus` | 2 | Confirmed usage 1.99 of 2 on the L40 — saturated. The job is **CPU**-bound, not GPU-bound: `GPUs usage` was 0.06. Try 4 and compare `TimeExecute`. |
| `request_memory` | 32GB | Confirmed 24,861 MB used. The standardized latent tensor alone is 41372x237x320x4 = 11.7 GiB, and encoding holds working copies. 16GB would have been killed. |
| `gpus_minimum_memory` | 12G | Peak 6.9 GiB reserved during training at batch 128. Condor reported only 1,282 MB for the smoke run — it samples periodically and caught the encode phase. Do not trust that figure. |
| `request_disk` | 16GB | Confirmed 9.0 GiB used, essentially all of it the ~8.8 GiB `.sif`. **The image is the disk request**; data is noise beside it. |

Note the GB10 has **unified** memory — `nvidia-smi` reports `memory.total` as
`[N/A]` because GPU and host share one 119 GB pool. "It fits locally" therefore
says nothing about a discrete GPU's VRAM, which is why the GPU figure above
comes from `torch.cuda.max_memory_reserved()` rather than from `nvidia-smi`.

### Runtime: 250 epochs does not fit

The full 250-epoch run took **over 21 hours** on the GB10 and was still
sampling. OSPool offers `"Medium"` (10 h) and `"Long"` (20 h), so this config
cannot complete as a single job.

The loss curve shows the second half buys little:

| | epoch 124 | epoch 250 |
| --- | --- | --- |
| flow | 0.3733 | 0.3156 |
| diffusion | 0.1808 | 0.1663 |

`--epochs 125` lands near 10.5 h, inside `"Long"` with margin. Both submit
files are set to `"Long"` and carry this warning; reduce `--epochs` before
submitting, or implement checkpointing if you need all 250.

## What comes back, and where

Two separate OSDF locations, set at the top of each submit file. Keeping them
apart matters: the first smoke run wrote its `results.pt` under
`containers/runs/` because one variable was serving both purposes.

```
/ospool/ap40/data/mukul.sherekar/
  containers/
    dgm-gpu-v1.sif              <- OSDF_IMAGE, read by every job
  DGM/
    data/                       <- OSDF_DATA, for weight caches over ~1 GB
    runs/                       <- OSDF_RUNS, one folder per run
      long250_20261001/
        flow_results.pt
        diffusion_results.pt
        manifest.txt
```

One folder per `TAG`, so `results.pt` keeps its plain name and the folder
identifies the run. `manifest.txt` travels with them so a folder is readable on
its own months later:

```
tag          long250_20261001
finished     2026-10-01T20:55:41Z
host         vcledch0201.hpc.ncsu.edu
glidein_site NCSU-OSG-CE1
gpu          NVIDIA L40, 570.158.01, 46068 MiB
torch        2.3.1
capability   (8, 9)
flags        --esm-model esm2_8m --dataset data/avgfp_train_props.csv ...

files
  flow_results.pt  386M
  diffusion_results.pt  389M
```

| Artifact | Size | Destination |
| --- | --- | --- |
| `<tag>.tar.gz` — FASTA, run config, plots, log | ~1 MB | access point, `jobs/` |
| `flow_results.pt`, `diffusion_results.pt` | 386 MB each | `DGM/runs/<tag>/` |
| `manifest.txt` | <1 KB | `DGM/runs/<tag>/` |

### Create the folder before submitting

`transfer_output_remaps` writes files into a path but does not reliably create
the intermediate directories, and `/ospool` is mounted on the access point, so:

```bash
OSG_AP=ap40 OSG_USER=mukul.sherekar bash ../bin/prep_run.sh long250_20261001
```

It also refuses a tag that already holds results, because **OSDF caches by
path** — reusing a tag can serve the previous run's file to a later read. Date-
stamp or version every tag.

### Unpacking a finished run

```bash
tar -xzf long250_20261001.tar.gz        # -> results/long250_20261001/{flow,diffusion}
R=/ospool/ap40/data/$USER/DGM/runs/long250_20261001
cp $R/flow_results.pt      results/long250_20261001/flow/results.pt
cp $R/diffusion_results.pt results/long250_20261001/diffusion/results.pt
```

`results.pt` holds the latents and weights, which is what makes
`dgm-resample-cfg` and re-decoding at another mutation budget possible later —
worth keeping, too big to pile up in `/home`.

## Weights & Biases

Logging reads `results.pt` rather than instrumenting the training loop, so no
experiment code changed and loss curves arrive at **full per-epoch
resolution** — `run_experiment` prints only four or five epoch lines but stores
every one. Default destination is `proterial/CIS6270`.

Two routes for OSG. Both end with the run in W&B; pick by whether you want to
rebuild the image.

### Route A — sync after recovery (no image rebuild)

Works with the image you already built. Recover the run as usual, then:

```bash
dgm-wandb sync --run-dir results/long250_20261001 --tags osg l40 --group long250
```

Needs the 386 MB `results.pt` files present, so it costs an OSDF download per
run. Fine for one run; wasteful across a sweep.

### Route B — offline on the node, sync on the access point

Set `OSG_WANDB=1` in the job environment. The node writes an **offline** W&B
run into the results directory, the tarball carries it home (about 4 KB), and
you sync it on the access point — **no `results.pt` download at all**:

```bash
tar -xzf long250_20261001.tar.gz
wandb sync results/long250_20261001/wandb/wandb/offline-run-*
```

Add to the submit file:

```
environment = "OSG_WANDB=1"
```

This needs `wandb` in the image. Both `.def` files now install it, so rebuild
and **bump `IMAGE_VERSION`**. The worker needs no W&B credentials — offline
mode writes locally and the upload happens from the access point.

If `OSG_WANDB=1` is set on an image without wandb, the job logs a warning and
carries on; the run itself is unaffected.

## Scoring

Jobs run with `--no-oracle`: METL needs `pytorch-lightning` plus five more
packages and a 64 MB checkout, which would be carried by every job to compute
something that is pure post-processing. Score afterwards, locally or on the
access point:

```bash
dgm-gfp-metrics --run-dir results/long250 \
    --embedding-oracle project1_eval/data/avgfp_metl_oracle_v2.npz \
    --train project1_eval/data/avgfp_train_props.csv --baseline-n 50
```

To score on the node instead, add METL to the image, stage the checkout, set
`METL_ROOT`, and submit with `OSG_WITH_ORACLE=1` in the job environment.

## Bigger models

`esm2_8m` weights are 30 MB. Above roughly a gigabyte OSG asks you to use OSDF
rather than `transfer_input_files`:

| Model | Cache size | How to stage |
| --- | --- | --- |
| `esm2_8m` | 30 MB | tarball (default) |
| `esm2_35m` | 130 MB | tarball |
| `esm2_150m` | 568 MB | tarball |
| `esm2_650m` | 2.5 GB | OSDF |
| `esm2_3b` | 11 GB | OSDF |

For the OSDF route, copy the cache directory up and swap the input line:

```bash
cp -r project1_eval/cache /ospool/$AP/data/$USER/esm-cache
```

```
transfer_input_files = dgm-src.tar.gz, inputs.tar.gz, $(OSDF)/esm-cache/?recursive
```

The `?recursive` is required for directories. `run_experiment.sh` already
detects a staged `cache/` directory and points `ESM2_CACHE` at it.

## How the node is set up

`run_experiment.sh` does this, in order:

1. Unpacks the tarballs into the Condor scratch directory.
2. Sets `DGM_ROOT` to the unpacked `CIS6270/`. `dgm.common.paths` validates it
   contains `pyproject.toml` and `src/dgm`, so a bad tarball fails immediately
   instead of writing results somewhere surprising.
3. Puts `$DGM_ROOT/src` on `PYTHONPATH` — **no `pip install` at job time**, so
   jobs need no network to start and a code change needs no image rebuild.
4. Creates `project1_eval/{data,outputs,plots,cache,logs}`, which the clone
   lacks because they are gitignored, and copies the staged inputs into `data/`.
5. Points `ESM2_CACHE` at the staged weights and sets `HF_HUB_OFFLINE=1`, so a
   missing weight fails loudly rather than every job pulling from HuggingFace.
6. `cd`s into `project1_eval` so relative `--dataset data/...` flags resolve
   exactly as they do locally.
7. Runs `python -m dgm.project1.run_experiment`, then splits the outputs into
   the small tarball and the per-modality `results.pt`.

## Troubleshooting

**Held jobs.** `condor_q -hold` and read the reason. Running past
`+JobDurationCategory = "Medium"` (10 h) is the likely one for 250 epochs on a
slower card; raise it to `"Long"` (20 h max).

**Nothing matches.** Check `condor_q -better-analyze`. `require_gpus
(Capability >= 7.0)` plus `gpus_minimum_memory = 16G` is restrictive; lower the
memory first.

**Image serves stale code or deps.** You rebuilt without bumping
`IMAGE_VERSION`. OSDF caches by name.

**`no staged ESM weights` warning in the log.** The weight tarball did not
arrive, and the job then tried to download with `HF_HUB_OFFLINE=1` set and
failed. Re-run `stage.sh`.

**`results.pt` is not in OSDF.** Check `OSDF_RUNS` in the submit file actually
points where you think, and that the run folder exists — `prep_run.sh` creates
it. Look under `containers/runs/` too: that is where it lands if `OSDF_IMAGE`
and `OSDF_RUNS` have been collapsed into one variable.

**GPU utilization near zero.** Expected on short jobs — encoding is CPU-bound
and dominates them. On a 250-epoch run training should dominate; if `GPUs usage`
is still low there, the bottleneck is `load_data`'s batch-16 encode loop, and
the fix is in the code, not the submit file.

**A zero-byte `results.pt`.** That modality did not finish. Both file names are
always created because Condor fails the entire transfer — losing the small
tarball with it — if a file named in `transfer_output_files` is missing. Read
the `.err` log for the real failure.

**Calibrating runtime.** Time a short run before committing to a sweep:

```bash
condor_submit run_experiment.sub -append 'TAG = cal10' \
    -append 'RUN_ARGS = --esm-model esm2_8m --dataset data/avgfp_train_props.csv --max-length 237 --epochs 10 --samples 10 --batch-size 128 --arch transformer --hidden 256'
```

250 epochs costs roughly 25x the training portion of that.
