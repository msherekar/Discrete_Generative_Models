#!/bin/bash
#
# Builds the input tarballs a job needs, into osg/jobs/.
#
#   stage.sh --model esm2_8m \
#            --data data/avgfp_train_props.csv data/avgfp_val.csv \
#                   data/avgfp_wt.txt data/avgfp_oracle_v2.npz \
#            --latents avgfp_train_props avgfp_val \
#            --metl --mnist
#
# Run this on the OSG access point after cloning the repo, or locally and scp
# the tarballs up. It produces, beside the submit files:
#
#   dgm-src.tar.gz       CIS6270/{pyproject.toml,src/}   the package, ~1 MB
#   inputs.tar.gz        inputs/<the --data files>       this run's inputs
#   esm-cache.tar.gz     esm-cache/models--facebook--... weights, + METL
#   latent-cache.tar.gz  precomputed ESM latents         --latents
#   mnist-data.tar.gz    MNIST raw                       --mnist
#
# The last three are opt-in. --metl adds ~250 MB and is only needed by jobs run
# with OSG_WITH_ORACLE=1; --latents removes the ESM encode from every job's
# critical path; --mnist is only for the Stage 6 transfer submission.
#
# Note that project1_eval/data and project1_eval/cache are gitignored, so a
# fresh clone on the access point has neither. Either scp them up, or build the
# data with dgm-prepare-gfp and fetch the weights with fetch_esm_weights.py.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COURSE_ROOT="$(cd "$HERE/../.." && pwd)"     # .../CIS6270
JOBS="$HERE/../jobs"

MODEL=""
DATA=()
LATENTS=()
CACHE_ROOT="$COURSE_ROOT/project1_eval/cache"
WITH_METL=0
WITH_MNIST=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --cache-root) CACHE_ROOT="$2"; shift 2 ;;
        --data)       shift; while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do DATA+=("$1"); shift; done ;;
        --latents)    shift; while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do LATENTS+=("$1"); shift; done ;;
        --metl)       WITH_METL=1; shift ;;
        --mnist)      WITH_MNIST=1; shift ;;
        -h|--help)    sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)            echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -n "$MODEL" ] || { echo "--model is required (e.g. --model esm2_8m)" >&2; exit 2; }
[ "${#DATA[@]}" -gt 0 ] || { echo "--data needs at least one file" >&2; exit 2; }

mkdir -p "$JOBS"
# -s matters: without it, du prints a line per subdirectory and a directory
# argument spills across the output it was meant to annotate.
size() { du -sh "$1" | cut -f1; }

warn_if_large() {
    local file="$1" bytes
    bytes=$(stat -c %s "$file")
    if [ "$bytes" -gt 1073741824 ]; then
        echo "  [warn] $(basename "$file") is over 1 GB. OSG asks that you stage"
        echo "         files this large through OSDF instead of transfer_input_files:"
        echo "           cp $file /ospool/<ap>/data/<user>/"
        echo "         then reference it as osdf:///ospool/<ap>/data/<user>/$(basename "$file")"
    fi
}

# ── the package ──────────────────────────────────────────────────────────────
# The CIS6270/ prefix matters: dgm.common.paths identifies the course root by
# finding pyproject.toml and src/dgm together, and the job script sets DGM_ROOT
# to the unpacked CIS6270/.
echo "staging code"
tar -czf "$JOBS/dgm-src.tar.gz" \
    -C "$(dirname "$COURSE_ROOT")" \
    --exclude='__pycache__' --exclude='*.pyc' \
    "$(basename "$COURSE_ROOT")/pyproject.toml" \
    "$(basename "$COURSE_ROOT")/src"
echo "  dgm-src.tar.gz    $(size "$JOBS/dgm-src.tar.gz")"

# ── this run's inputs ────────────────────────────────────────────────────────
echo "staging inputs"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/inputs"
for f in "${DATA[@]}"; do
    path="$f"
    [ -f "$path" ] || path="$COURSE_ROOT/project1_eval/$f"
    [ -f "$path" ] || { echo "  no such data file: $f" >&2; exit 1; }
    cp "$path" "$STAGE/inputs/"
    echo "  + $(basename "$path")  $(size "$path")"
    # add_properties.py writes <csv>.wt_attributes.json beside its output. It
    # holds the wild-type METL attributes, which is what sets the constraint
    # reference -- 3 KB that saves staging METL itself, and without it a worker
    # falls back to the training median and constrains against a different
    # reference than a local run.
    sidecar="${path%.csv}.wt_attributes.json"
    if [ "$path" != "$sidecar" ] && [ -f "$sidecar" ]; then
        cp "$sidecar" "$STAGE/inputs/"
        echo "  + $(basename "$sidecar")  $(size "$sidecar")   (constraint reference)"
    fi
done
tar -czf "$JOBS/inputs.tar.gz" -C "$STAGE" inputs
echo "  inputs.tar.gz     $(size "$JOBS/inputs.tar.gz")"
warn_if_large "$JOBS/inputs.tar.gz"

# ── the model weights ────────────────────────────────────────────────────────
echo "staging weights for $MODEL"
HF_ID="$(PYTHONPATH="$COURSE_ROOT/src" python3 -c "
from dgm.common.esm_models import get_model
print(get_model('$MODEL')['hf_id'])
")"
# HuggingFace namespaces its cache as models--<org>--<name>.
CACHE_SUBDIR="models--${HF_ID//\//--}"
if [ ! -d "$CACHE_ROOT/$CACHE_SUBDIR" ]; then
    echo "  no cached weights at $CACHE_ROOT/$CACHE_SUBDIR" >&2
    echo "  fetch them first:" >&2
    echo "    PYTHONPATH=$COURSE_ROOT/src python3 $HERE/fetch_esm_weights.py --model $MODEL" >&2
    exit 1
fi
mkdir -p "$STAGE/esm-cache"
cp -r "$CACHE_ROOT/$CACHE_SUBDIR" "$STAGE/esm-cache/"
tar -czf "$JOBS/esm-cache.tar.gz" -C "$STAGE" esm-cache
echo "  esm-cache.tar.gz  $(size "$JOBS/esm-cache.tar.gz")"
warn_if_large "$JOBS/esm-cache.tar.gz"

# ── the METL checkpoint ──────────────────────────────────────────────────────
# Rides inside esm-cache.tar.gz rather than its own archive, because
# gfp_sweep.sh exports METL_CKPT from whichever cache directory it finds and
# oracles/common.py resolves it from there. ~250 MB, so it is opt-in: only the
# arms that score brightness on the node need it, and OSG_WITH_ORACLE=0 jobs
# would otherwise pay the transfer for nothing.
if [ "$WITH_METL" = "1" ]; then
    echo "staging METL checkpoint"
    METL_SRC="${METL_CKPT:-$CACHE_ROOT/Hr4GNHws.pt}"
    if [ -f "$METL_SRC" ]; then
        cp "$METL_SRC" "$STAGE/esm-cache/"
        echo "  + $(basename "$METL_SRC")  $(size "$METL_SRC")"
        # metl-pretrained pulls the finetuned target model through torch.hub,
        # which a worker node cannot reach. Ship whatever has been cached.
        if [ -d "$CACHE_ROOT/torch_hub" ]; then
            cp -r "$CACHE_ROOT/torch_hub" "$STAGE/esm-cache/"
            echo "  + torch_hub/  $(size "$CACHE_ROOT/torch_hub")   (metl-pretrained cache)"
        else
            echo "  [warn] no torch_hub cache; on-node METL scoring will fail to"
            echo "         download its target model. Populate it locally first:"
            echo "           dgm-postprocess --metl-target ft-1d <any run dir>"
        fi
        # Rebuild the archive now that the checkpoint is in it.
        tar -czf "$JOBS/esm-cache.tar.gz" -C "$STAGE" esm-cache
        echo "  esm-cache.tar.gz  $(size "$JOBS/esm-cache.tar.gz")  (with METL)"
        warn_if_large "$JOBS/esm-cache.tar.gz"
    else
        echo "  [warn] no METL checkpoint at $METL_SRC; skipping" >&2
        echo "         set METL_CKPT or pass --cache-root" >&2
    fi
fi

# ── precomputed ESM latents ──────────────────────────────────────────────────
# Every job was re-encoding the same 41,372 sequences with ESM-2 before
# training anything -- identical work, repeated once per job, on a GPU rented
# by the hour. Encoding once and shipping the tensor removes it from the
# critical path of all 395 sweep jobs.
if [ "${#LATENTS[@]}" -gt 0 ]; then
    echo "staging latent cache"
    mkdir -p "$STAGE/latent-cache"
    for tag in "${LATENTS[@]}"; do
        # Accept either a bare dataset tag or a full path to the .pt.
        if [ -f "$tag" ]; then
            found="$tag"
        else
            found=""
            for candidate in "$CACHE_ROOT"/latents_"$tag"_*.pt; do
                [ -f "$candidate" ] && found="$candidate"
            done
        fi
        if [ -z "$found" ]; then
            echo "  [warn] no cached latents for '$tag'; build them first:" >&2
            echo "           dgm-cache-latents --dataset data/$tag.csv --model $MODEL" >&2
            continue
        fi
        cp "$found" "$STAGE/latent-cache/"
        echo "  + $(basename "$found")  $(size "$found")"
    done
    if [ -n "$(ls -A "$STAGE/latent-cache" 2>/dev/null)" ]; then
        tar -czf "$JOBS/latent-cache.tar.gz" -C "$STAGE" latent-cache
        echo "  latent-cache.tar.gz  $(size "$JOBS/latent-cache.tar.gz")"
        warn_if_large "$JOBS/latent-cache.tar.gz"
    else
        echo "  [warn] nothing cached; not writing latent-cache.tar.gz" >&2
    fi
fi

# ── MNIST, for the Stage 6 transfer jobs ─────────────────────────────────────
# Staged rather than downloaded: worker nodes generally have no outbound
# network, and 40 jobs pulling the same 11 MB from the same mirror is rude.
if [ "$WITH_MNIST" = "1" ]; then
    echo "staging MNIST"
    MNIST_SRC="$COURSE_ROOT/project1_eval/data/mnist"
    if [ -d "$MNIST_SRC/MNIST" ]; then
        tar -czf "$JOBS/mnist-data.tar.gz" \
            --transform 's|^mnist|mnist-data|' \
            -C "$(dirname "$MNIST_SRC")" mnist
        echo "  mnist-data.tar.gz  $(size "$JOBS/mnist-data.tar.gz")"
    else
        echo "  [warn] no MNIST at $MNIST_SRC/MNIST; fetch it locally first:" >&2
        echo "           dgm-run-mnist --epochs 1 --limit 100" >&2
    fi
fi

echo
echo "staged into $JOBS"
ls -1 "$JOBS"/*.tar.gz 2>/dev/null | sed 's|.*/|  |'
echo "next: edit OSG_USER and OSG_AP in the .sub files, then"
echo "      cd $JOBS && condor_submit gfp_sweep.sub"
