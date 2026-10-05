#!/bin/bash
#
# Runs ONE method (flow or diffusion) of one avGFP experiment on an OSPool node.
#
#   gfp_sweep.sh <tag> <method> [run_experiment flags ...]
#
# Differs from run_experiment.sh in three ways:
#
#   1. One method per job. Job 15833935 trained both, finished both, then was
#      held for exceeding request_memory during plotting and lost all of it.
#      Splitting them halves the exposure and turns one ten-hour "Long" job into
#      two ~30-minute "Medium" ones, which match against a far larger pool.
#
#   2. No --plot. Plotting ran in-process after the science was done and is what
#      pushed the job over its memory limit. Plot locally from the staged
#      results.pt with dgm-postprocess.
#
#   3. Unpacks latent-cache.tar.gz when present, so the job skips re-encoding
#      41,372 sequences with ESM-2 -- work every job was repeating.
#
# On success this leaves, at scratch root:
#   <tag>.tar.gz                FASTA, run config, logs      -> access point
#   <tag>_<method>_results.pt   latents + weights            -> OSDF
#   <tag>_manifest.txt          what was produced            -> OSDF
#
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <tag> <flow|diffusion> [run_experiment flags ...]" >&2
    exit 2
fi
TAG="$1"; METHOD="$2"; shift 2
case "$METHOD" in
    flow|diffusion) ;;
    *) echo "method must be 'flow' or 'diffusion', got '$METHOD'" >&2; exit 2 ;;
esac

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$SCRATCH"
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

say "host $(hostname)   tag $TAG   method $METHOD"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "  [warn] no nvidia-smi; this node may have no GPU" >&2
fi

# ── unpack ───────────────────────────────────────────────────────────────────
for archive in dgm-src.tar.gz inputs.tar.gz; do
    [ -f "$archive" ] || { echo "missing input: $archive" >&2; exit 1; }
    say "unpacking $archive"
    tar -xzf "$archive"
done
for archive in esm-cache.tar.gz latent-cache.tar.gz; do
    [ -f "$archive" ] && { say "unpacking $archive"; tar -xzf "$archive"; }
done

export DGM_ROOT="$SCRATCH/CIS6270"
[ -f "$DGM_ROOT/pyproject.toml" ] && [ -d "$DGM_ROOT/src/dgm" ] || {
    echo "dgm-src.tar.gz did not unpack to CIS6270/{pyproject.toml,src/dgm}" >&2
    ls -la "$SCRATCH" >&2; exit 1
}
export PYTHONPATH="$DGM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$DGM_ROOT/project1_eval"/{data,outputs,plots,cache,logs}

if [ -d inputs ] && [ -n "$(ls -A inputs 2>/dev/null)" ]; then
    cp inputs/* "$DGM_ROOT/project1_eval/data/" && say "staged $(ls inputs | wc -l) input files"
fi

# Weights and the precomputed latents share the cache directory, which is also
# where the METL checkpoint is resolved from; see oracles/common.py.
# An inherited ESM2_CACHE wins. Without this branch the variable was exported
# unconditionally, so a caller could not point the job at a cache it had
# already populated -- which is exactly what a local dry run of this script
# needs to do, and the reason the first such run died on an offline HF lookup.
if [ -n "${ESM2_CACHE:-}" ] && [ -d "${ESM2_CACHE}" ]; then
    export ESM2_CACHE
elif [ -d esm-cache ]; then
    export ESM2_CACHE="$SCRATCH/esm-cache"
elif [ -d cache ]; then
    export ESM2_CACHE="$SCRATCH/cache"
else
    export ESM2_CACHE="$DGM_ROOT/project1_eval/cache"
    echo "  [warn] no staged ESM weights; the run will try to download them" >&2
fi
# Latents, if staged, land beside the weights so load_data finds them.
# latent-cache/ = unpacked tarball; latents/ = OSDF ?recursive of DGM/latents.
if [ -d latent-cache ]; then
    cp latent-cache/*.pt "$ESM2_CACHE/" 2>/dev/null || true
    say "latent cache: $(ls latent-cache | tr '\n' ' ')"
elif [ -d latents ]; then
    cp latents/*.pt "$ESM2_CACHE/" 2>/dev/null || true
    say "latent cache: $(ls latents | tr '\n' ' ')"
fi
# The METL checkpoint rides in whichever cache was staged; point the resolver
# at it explicitly rather than relying on the search order.
if [ -f "$ESM2_CACHE/Hr4GNHws.pt" ]; then
    export METL_CKPT="$ESM2_CACHE/Hr4GNHws.pt"
    say "METL_CKPT $METL_CKPT"
fi
# metl-pretrained downloads through torch.hub; keep it inside the cache so a
# node with no outbound network still finds the finetuned target model.
export TORCH_HOME="$ESM2_CACHE/torch_hub"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
"$PY" - <<'PY'
import torch
print(f"    torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    device {torch.cuda.get_device_name(0)}  "
          f"capability {torch.cuda.get_device_capability(0)}  "
          f"vram {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
PY

# ── run ──────────────────────────────────────────────────────────────────────
OUT="$SCRATCH/results/$TAG"
mkdir -p "$OUT"
cd "$DGM_ROOT/project1_eval"

# On-node brightness scoring is viable now: metl-pretrained resolves with no
# new dependencies beyond torch, unlike the full metl repo. Still opt-in.
ORACLE_FLAG=(--no-oracle)
if [ "${OSG_WITH_ORACLE:-0}" = "1" ]; then
    if "$PY" -c "import metl" 2>/dev/null; then
        ORACLE_FLAG=()
        # Preference order, not a glob, and the order depends on what the node
        # can actually run.
        #
        # Two different kinds of .npz match avgfp*oracle*. The EMBEDDING
        # oracles carry a `backend` key and are what load_brightness_oracle
        # reads; avgfp_oracle*.npz are the older indicator oracles with no
        # `backend` and a different loader. Globbing picked the latter and the
        # run died with KeyError: 'backend is not a file in the archive'.
        #
        # Among the embedding oracles, backend matters more than recency. A
        # `metl` backend re-encodes through the full METL REPO -- a 64 MB
        # checkout plus pytorch-lightning, and metl.py chdir's into it -- which
        # this image does not carry, so it fails with FileNotFoundError on
        # $SCRATCH/metl. The esm2_8m_pca64 backend re-encodes with the ESM
        # weights that are already staged, so it is the one that works here.
        # METL brightness still gets reported, from metl-pretrained, when
        # dgm-postprocess runs on the staged results.pt.
        ORACLE_DATA="$DGM_ROOT/project1_eval/data"
        ORACLE_ORDER=(avgfp_esm2_8m_pca64_oracle.npz
                      avgfp_metl_oracle_v2.npz avgfp_metl_oracle.npz)
        if [ -d "$SCRATCH/metl" ]; then
            ORACLE_ORDER=(avgfp_metl_oracle_v2.npz avgfp_metl_oracle.npz
                          avgfp_esm2_8m_pca64_oracle.npz)
        fi
        for candidate in "${ORACLE_ORDER[@]}"; do
            if [ -f "$ORACLE_DATA/$candidate" ]; then
                ORACLE_FLAG=(--oracle "$ORACLE_DATA/$candidate")
                break
            fi
        done
        if [ "${#ORACLE_FLAG[@]}" -eq 0 ]; then
            say "[warn] METL importable but no avgfp*oracle*.npz staged;"
            say "       samples will not be scored on the node"
        else
            say "on-node scoring with $(basename "${ORACLE_FLAG[1]}")"
        fi
    else
        say "[warn] OSG_WITH_ORACLE=1 but metl is not importable; scoring deferred"
    fi
fi

# --only-<method> trains just the one head this job owns.
heartbeat() {
    local interval="${OSG_HEARTBEAT_SECONDS:-300}" start=$SECONDS
    while sleep "$interval"; do
        local gpu=""
        if command -v nvidia-smi >/dev/null 2>&1; then
            gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
                    --format=csv,noheader,nounits 2>/dev/null | head -1 \
                  | awk -F', *' '{printf "  gpu %s%%", $1;
                                  if ($2 ~ /^[0-9]+$/) printf "  vram %sMiB", $2}')
        fi
        echo "[$(date -u +%H:%M:%S)]   alive $(( (SECONDS - start) / 60 ))m$gpu"
    done
}
heartbeat & HEARTBEAT=$!
trap 'kill "$HEARTBEAT" 2>/dev/null || true' EXIT

say "running $METHOD"
# ${arr[@]+"${arr[@]}"} and NOT "${arr[@]:-}". Under set -u the latter is the
# obvious-looking guard, but `:-` substitutes its default when the array is
# EMPTY, so an empty ORACLE_FLAG expanded to one empty-string argument and
# argparse rejected it with `unrecognized arguments:`. The former expands to
# nothing at all, which is what is wanted. This only ever fires when
# OSG_WITH_ORACLE=1 makes the array empty, which is why it survived until the
# first on-node oracle run.
"$PY" -u -m dgm.project1.run_experiment "$@" \
    --outdir "$OUT" --only "$METHOD" ${ORACLE_FLAG[@]+"${ORACLE_FLAG[@]}"}

kill "$HEARTBEAT" 2>/dev/null || true
trap - EXIT
say "run finished"

# ── stage the heavy artifact FIRST ───────────────────────────────────────────
# Before anything else allocates. transfer_output_files names a file that must
# already exist if the job is evicted during a later step.
cd "$SCRATCH"
TARGET="${TAG}_${METHOD}_results.pt"
if [ -f "results/$TAG/$METHOD/results.pt" ]; then
    mv "results/$TAG/$METHOD/results.pt" "$TARGET"
    say "staged $TARGET ($(du -h "$TARGET" | cut -f1))"
else
    : > "$TARGET"
    say "[warn] no results.pt for $METHOD; staging an empty placeholder"
fi

# ── arms sanity check, on the node, while the data is here ───────────────────
# Cheap, and it answers the question that decides whether the run is worth
# analyzing at all: did the guidance arms actually differ? Non-fatal, because a
# failed check is a finding rather than a broken job.
if [ "${OSG_CHECK_ARMS:-1}" = "1" ] && [ -s "$TARGET" ]; then
    mkdir -p "check/$METHOD" && cp "$TARGET" "check/$METHOD/results.pt"
    "$PY" -u -m dgm.project1.diagnostics.check_arms "check/$METHOD" \
        2>&1 | sed 's/^/    /' || say "[note] arms check reported a problem"
    rm -rf check
fi

# ── optional: metrics on the node ────────────────────────────────────────────
if [ "${OSG_METRICS:-1}" = "1" ]; then
    ORACLE_NPZ=""
    for candidate in "$DGM_ROOT/project1_eval/data"/avgfp_oracle*.npz; do
        [ -f "$candidate" ] && ORACLE_NPZ="$candidate"
    done
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
        say "study figures"
        "$PY" -u -m dgm.project1.study_plots --run-dir "$OUT" \
            --oracle "$ORACLE_NPZ" --outdir "$OUT/plots/study" \
            --prefix "$TAG" \
            || say "[warn] study figures failed; the run itself is unaffected"
    fi
fi

# ── manifest and the light tarball ───────────────────────────────────────────
{
    echo "tag       $TAG"
    echo "method    $METHOD"
    echo "host      $(hostname)"
    echo "finished  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "args      $*"
    echo "results   $(du -h "$TARGET" 2>/dev/null | cut -f1)"
} > "${TAG}_manifest.txt"

tar -czf "${TAG}.tar.gz" -C "$SCRATCH" "results/$TAG" || {
    echo "[warn] results directory missing or unreadable; packaging manifest only" >&2
    tar -czf "${TAG}.tar.gz" "${TAG}_manifest.txt"
}
say "wrote ${TAG}.tar.gz and ${TAG}_manifest.txt"
