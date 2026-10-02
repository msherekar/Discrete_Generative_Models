#!/bin/bash
#
# Runs one Project 1 experiment on an OSPool worker node.
#
#   run_experiment.sh <tag> [run_experiment flags ...]
#
# Everything arrives as tarballs in the Condor scratch directory. This script
# rebuilds the directory layout dgm.common.paths expects, points the package at
# it with DGM_ROOT, and runs the experiment with the flags it was handed -- so
# the flags are character-for-character the ones used locally.
#
# Contract with run_experiment.sub. On success this leaves, at scratch root:
#
#   <tag>.tar.gz                   FASTA, run config, plots, this log  -> access point
#   <tag>_flow_results.pt          latents + weights                   -> OSDF
#   <tag>_diffusion_results.pt     latents + weights                   -> OSDF
#
# results.pt is 200-400 MB per modality, which is why it is split out and
# remapped to OSDF rather than returned to the access point.
#
# A heartbeat runs alongside the experiment, printing elapsed time and GPU
# utilization every OSG_HEARTBEAT_SECONDS (default 300). With stream_output in
# the submit file, that is what makes the .out file worth tailing: the
# experiment itself prints only every epochs//4 epochs.
#
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <tag> [run_experiment flags ...]" >&2
    exit 2
fi
TAG="$1"; shift

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$SCRATCH"

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# ── what we landed on ────────────────────────────────────────────────────────
say "host $(hostname)"
say "scratch $SCRATCH"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "  [warn] no nvidia-smi; this node may have no GPU" >&2
fi

# ── unpack ───────────────────────────────────────────────────────────────────
# dgm-src.tar.gz contains CIS6270/{pyproject.toml,src/dgm/...}: both are needed
# because paths.py identifies the course root by finding them together.
for archive in dgm-src.tar.gz inputs.tar.gz; do
    [ -f "$archive" ] || { echo "missing input: $archive" >&2; exit 1; }
    say "unpacking $archive"
    tar -xzf "$archive"
done

# The weight cache is optional here: a large model is staged from OSDF as a
# directory instead, and ESM2_CACHE is pointed straight at it.
if [ -f esm-cache.tar.gz ]; then
    say "unpacking esm-cache.tar.gz"
    tar -xzf esm-cache.tar.gz
fi

export DGM_ROOT="$SCRATCH/CIS6270"
[ -f "$DGM_ROOT/pyproject.toml" ] && [ -d "$DGM_ROOT/src/dgm" ] || {
    echo "dgm-src.tar.gz did not unpack to CIS6270/{pyproject.toml,src/dgm}" >&2
    echo "contents:" >&2; ls -la "$SCRATCH" >&2
    exit 1
}

# No pip install: the source is on PYTHONPATH, so a code change needs no image
# rebuild and the job does no network I/O to start.
export PYTHONPATH="$DGM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# project1_eval holds only gitignored artifacts, so a fresh clone has none of
# it. Create the tree paths.py will reach for.
mkdir -p "$DGM_ROOT/project1_eval"/{data,outputs,plots,cache,logs}

# Staged inputs become the data/ directory the flags refer to relatively.
if [ -d inputs ] && [ -n "$(ls -A inputs 2>/dev/null)" ]; then
    cp -v inputs/* "$DGM_ROOT/project1_eval/data/"
fi

# Weights: prefer an unpacked tarball, else an OSDF-staged directory.
if [ -d esm-cache ]; then
    export ESM2_CACHE="$SCRATCH/esm-cache"
elif [ -d cache ]; then
    export ESM2_CACHE="$SCRATCH/cache"
else
    export ESM2_CACHE="$DGM_ROOT/project1_eval/cache"
    echo "  [warn] no staged ESM weights; the run will try to download them" >&2
fi
say "ESM2_CACHE $ESM2_CACHE"
ls "$ESM2_CACHE" 2>/dev/null | sed 's/^/    /' || true

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
say "interpreter $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
"$PY" - <<'PY'
import torch
print(f"    torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    device {torch.cuda.get_device_name(0)}  capability {torch.cuda.get_device_capability(0)}")
PY

# ── run ──────────────────────────────────────────────────────────────────────
OUT="$SCRATCH/results/$TAG"
mkdir -p "$OUT"

# cd into the artifact directory so relative --dataset data/... flags resolve
# exactly as they do locally.
cd "$DGM_ROOT/project1_eval"

# METL is not staged, so scoring is off by default and done afterwards with
# dgm-gfp-metrics. Set OSG_WITH_ORACLE=1 only once METL is in the image.
ORACLE_FLAG=(--no-oracle)
if [ "${OSG_WITH_ORACLE:-0}" = "1" ]; then
    ORACLE_FLAG=()
fi

# run_experiment prints one epoch line every epochs//4 -- on a 250-epoch run
# that is once every ~62 epochs, which can be hours apart. With stream_output
# on, that would make a nearly silent .out file. This heartbeat gives the log
# something to say while training is between those prints.
heartbeat() {
    local interval="${OSG_HEARTBEAT_SECONDS:-300}" start=$SECONDS
    while sleep "$interval"; do
        local gpu=""
        if command -v nvidia-smi >/dev/null 2>&1; then
            # memory.used reads [N/A] on unified-memory parts such as the
            # GB10, so it is only printed when it is actually a number.
            gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
                    --format=csv,noheader,nounits 2>/dev/null | head -1 \
                  | awk -F', *' '{printf "  gpu %s%%", $1;
                                  if ($2 ~ /^[0-9]+$/) printf "  vram %sMiB", $2}')
        fi
        echo "[$(date -u +%H:%M:%S)]   alive $(( (SECONDS - start) / 60 ))m$gpu"
    done
}
heartbeat &
HEARTBEAT=$!
# Stop it however the script leaves, so a failure does not orphan the loop.
trap 'kill "$HEARTBEAT" 2>/dev/null || true' EXIT

say "running: $* --outdir $OUT ${ORACLE_FLAG[*]:-}"
"$PY" -u -m dgm.project1.run_experiment "$@" \
    --outdir "$OUT" "${ORACLE_FLAG[@]:-}"

kill "$HEARTBEAT" 2>/dev/null || true
trap - EXIT
say "run finished"

# ── post-run: the avGFP metric table ─────────────────────────────────────────
# Automates the dgm-gfp-metrics step that otherwise has to be run by hand after
# every run. Only the indicator oracle runs here: the METL embedding oracle
# needs pytorch-lightning and a 64 MB checkout this image does not carry, so
# re-run locally with --embedding-oracle to add that column.
#
# dgm-evaluate is deliberately NOT used. It reads the lecture_3 training CSV,
# which is not staged on the node -- whereas run_experiment --plot reads the
# run's own --dataset and produces the same figures.
if [ "${OSG_METRICS:-1}" = "1" ]; then
    ORACLE_NPZ=""
    for candidate in "$DGM_ROOT/project1_eval/data"/avgfp_oracle*.npz; do
        [ -f "$candidate" ] && ORACLE_NPZ="$candidate"
    done
    # The --dataset the run was given, so the novelty and reference-cloud
    # columns compare against the right training set.
    DATASET=""; prev=""
    for a in "$@"; do [ "$prev" = "--dataset" ] && DATASET="$a"; prev="$a"; done
    if [ -z "$ORACLE_NPZ" ]; then
        say "[skip] no avgfp_oracle*.npz staged; not scoring"
    else
        say "scoring with $(basename "$ORACLE_NPZ")"
        TRAIN_FLAG=()
        [ -n "$DATASET" ] && TRAIN_FLAG=(--train "$DATASET")
        "$PY" -u -m dgm.project1.gfp_metrics --run-dir "$OUT" \
            --oracle "$ORACLE_NPZ" "${TRAIN_FLAG[@]}" \
            --baseline-n "${OSG_BASELINE_N:-50}" --outdir "$OUT/metrics" \
            || say "[warn] metrics failed; the run itself is unaffected"

        # The three project-specific figures: setpoint calibration, reward head
        # against the oracle, and the lambda Pareto front. Each skips itself
        # with a reason when the run lacks the arms it needs, so this is safe
        # to run after every job.
        say "study figures"
        "$PY" -u -m dgm.project1.study_plots --run-dir "$OUT" \
            --oracle "$ORACLE_NPZ" --outdir "$OUT/plots/study" \
            --prefix "$TAG" \
            || say "[warn] study figures failed; the run itself is unaffected"
    fi
fi

# ── optional: W&B, offline ───────────────────────────────────────────────────
# Opt-in, because the node holds no W&B credentials and an older image may not
# carry wandb at all. When enabled this writes an offline run into the results
# directory, which the tarball carries home -- so `wandb sync` on the access
# point registers the run without downloading the 386 MB results.pt just to log
# it. Must happen here, before packaging moves results.pt out of $OUT.
if [ "${OSG_WANDB:-0}" = "1" ]; then
    if "$PY" -c "import wandb" >/dev/null 2>&1; then
        export WANDB_MODE=offline
        export WANDB_DIR="$OUT/wandb"
        mkdir -p "$WANDB_DIR"
        say "logging to W&B (offline) in $WANDB_DIR"
        "$PY" -m dgm.project1.wandb_cli sync --run-dir "$OUT" --name "$TAG" \
            --mode offline --tags osg "${GLIDEIN_ResourceName:-unknown}" \
            || say "[warn] offline W&B logging failed; the run itself is fine"
    else
        say "[warn] OSG_WANDB=1 but wandb is not in this image; skipping"
    fi
fi

# ── package outputs ──────────────────────────────────────────────────────────
cd "$SCRATCH"

# results.pt out to its own uniquely named file per modality: OSDF caches by
# name, so the tag must make it unique.
# Both names are always created, even if empty: run_experiment.sub declares
# them under transfer_output_files, and Condor fails the whole transfer --
# losing the tarball too -- if a declared file is absent. A zero-byte
# results.pt therefore means that modality did not finish; check the .err log.
for method in flow diffusion; do
    target="${TAG}_${method}_results.pt"
    if [ -f "results/$TAG/$method/results.pt" ]; then
        mv "results/$TAG/$method/results.pt" "$target"
        say "staged $target ($(du -h "$target" | cut -f1))"
    else
        : > "$target"
        say "[warn] no results.pt for $method; staging an empty placeholder"
    fi
done

# A manifest beside the results in OSDF, so a run folder months from now says
# what it is without needing the tarball: which node, which GPU, which flags.
MANIFEST="${TAG}_manifest.txt"
{
    echo "tag          $TAG"
    echo "finished     $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "host         $(hostname)"
    # ClusterId.ProcId, read out of the job ad Condor drops in the sandbox.
    if [ -n "${_CONDOR_JOB_AD:-}" ] && [ -f "${_CONDOR_JOB_AD}" ]; then
        echo "condor_job   $(awk -F' = ' '/^ClusterId/{c=$2} /^ProcId/{p=$2} END{print c"."p}' \
                             "${_CONDOR_JOB_AD}")"
    fi
    echo "glidein_site ${GLIDEIN_ResourceName:-unknown}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "gpu          $(nvidia-smi --query-gpu=name,driver_version,memory.total \
                             --format=csv,noheader | head -1)"
    fi
    # Versions only: a peak-memory figure would read 0 here, since this is a
    # fresh interpreter and not the process that did the training.
    "$PY" - <<'PY' 2>/dev/null || true
import torch
print(f"torch        {torch.__version__}")
if torch.cuda.is_available():
    print(f"capability   {torch.cuda.get_device_capability(0)}")
PY
    echo "flags        $*"
    echo
    echo "files"
    for method in flow diffusion; do
        f="${TAG}_${method}_results.pt"
        [ -s "$f" ] && echo "  ${method}_results.pt  $(du -h "$f" | cut -f1)" \
                    || echo "  ${method}_results.pt  MISSING (that modality did not finish)"
    done
} > "$MANIFEST"
say "staged $MANIFEST"

# Everything small: FASTA files, the metric table written above, and any
# --plot figures, which land under project1_eval/plots rather than --outdir.
PLOTS_REL=""
if [ -d "$DGM_ROOT/project1_eval/plots" ] && \
   [ -n "$(ls -A "$DGM_ROOT/project1_eval/plots" 2>/dev/null)" ]; then
    # Copy the contents, not the directory: study_plots has already created
    # results/$TAG/plots, and `cp -r src dest` with an existing dest would
    # nest it as plots/plots.
    mkdir -p "results/$TAG/plots"
    cp -r "$DGM_ROOT/project1_eval/plots/." "results/$TAG/plots/"
    PLOTS_REL="(with plots)"
fi
tar -czf "${TAG}.tar.gz" -C "$SCRATCH" "results/$TAG"
say "staged ${TAG}.tar.gz $PLOTS_REL ($(du -h "${TAG}.tar.gz" | cut -f1))"

say "done"
