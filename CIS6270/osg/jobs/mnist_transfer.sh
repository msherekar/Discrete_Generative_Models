#!/bin/bash
#
# Runs one MNIST transfer job (Stage 6) on an OSPool worker node.
#
#   mnist_transfer.sh <tag> [run_mnist flags ...]
#
# Much simpler than gfp_sweep.sh: MNIST needs no ESM weights, no latent cache
# and no METL checkpoint, so the only staged input besides the source is the
# dataset itself. It is staged rather than downloaded because worker nodes
# generally have no outbound network, and because 40 jobs each pulling the same
# 11 MB from the same mirror is rude.
#
# On success this leaves, at scratch root:
#   <tag>.tar.gz        sample grids, config, log   -> access point
#   <tag>_results.pt    samples + weights          -> OSDF
#
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <tag> [run_mnist flags ...]" >&2
    exit 2
fi
TAG="$1"; shift

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$SCRATCH"
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

say "host $(hostname)   tag $TAG"
command -v nvidia-smi >/dev/null 2>&1 && \
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || \
    echo "  [warn] no nvidia-smi; this node may have no GPU" >&2

[ -f dgm-src.tar.gz ] || { echo "missing input: dgm-src.tar.gz" >&2; exit 1; }
say "unpacking dgm-src.tar.gz"
tar -xzf dgm-src.tar.gz
[ -f mnist-data.tar.gz ] && { say "unpacking mnist-data.tar.gz"; tar -xzf mnist-data.tar.gz; }

export DGM_ROOT="$SCRATCH/CIS6270"
[ -f "$DGM_ROOT/pyproject.toml" ] && [ -d "$DGM_ROOT/src/dgm" ] || {
    echo "dgm-src.tar.gz did not unpack to CIS6270/{pyproject.toml,src/dgm}" >&2
    ls -la "$SCRATCH" >&2; exit 1
}
export PYTHONPATH="$DGM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$DGM_ROOT/project1_eval"/{data,outputs,plots,logs}

# torchvision looks for MNIST/raw under --data-dir; accept either layout the
# staging tarball might have been built with.
DATA_DIR="$DGM_ROOT/project1_eval/data"
if [ -d mnist-data ]; then
    cp -r mnist-data/* "$DATA_DIR/" && say "staged MNIST from mnist-data/"
elif [ -d MNIST ]; then
    cp -r MNIST "$DATA_DIR/" && say "staged MNIST from MNIST/"
else
    say "[warn] no staged MNIST; the run will try to download it"
fi

export MPLBACKEND=Agg
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
"$PY" - <<'PY'
import torch
print(f"    torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    device {torch.cuda.get_device_name(0)}")
PY

OUT="$SCRATCH/results/$TAG"
mkdir -p "$OUT"
cd "$DGM_ROOT/project1_eval"

heartbeat() {
    local interval="${OSG_HEARTBEAT_SECONDS:-300}" start=$SECONDS
    while sleep "$interval"; do
        echo "[$(date -u +%H:%M:%S)]   alive $(( (SECONDS - start) / 60 ))m"
    done
}
heartbeat & HEARTBEAT=$!
trap 'kill "$HEARTBEAT" 2>/dev/null || true' EXIT

say "running: $*"
"$PY" -u -m dgm.project1.run_mnist "$@" --outdir "$OUT" --data-dir "$DATA_DIR"

kill "$HEARTBEAT" 2>/dev/null || true
trap - EXIT
say "run finished"

# Heavy artifact first, for the same reason as gfp_sweep.sh: an eviction during
# packaging must not find transfer_output_files naming a file that is absent.
cd "$SCRATCH"
TARGET="${TAG}_results.pt"
if [ -f "results/$TAG/results.pt" ]; then
    mv "results/$TAG/results.pt" "$TARGET"
    say "staged $TARGET ($(du -h "$TARGET" | cut -f1))"
else
    : > "$TARGET"
    say "[warn] no results.pt; staging an empty placeholder"
fi

tar -czf "${TAG}.tar.gz" "results/$TAG" 2>/dev/null || : > "${TAG}.tar.gz"
say "wrote ${TAG}.tar.gz"
