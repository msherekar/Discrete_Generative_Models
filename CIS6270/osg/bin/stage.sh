#!/bin/bash
#
# Builds the three input tarballs a job needs, into osg/jobs/.
#
#   stage.sh --model esm2_8m --data data/avgfp_train_props.csv data/avgfp_wt.txt \
#            data/avgfp_oracle_v2.npz
#
# Run this on the OSG access point after cloning the repo, or locally and scp
# the tarballs up. It produces, beside the submit files:
#
#   dgm-src.tar.gz    CIS6270/{pyproject.toml,src/}   the package, ~1 MB
#   inputs.tar.gz     inputs/<the --data files>        this run's inputs
#   esm-cache.tar.gz  esm-cache/models--facebook--...  one model's weights
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
CACHE_ROOT="$COURSE_ROOT/project1_eval/cache"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --cache-root) CACHE_ROOT="$2"; shift 2 ;;
        --data)       shift; while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do DATA+=("$1"); shift; done ;;
        -h|--help)    sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)            echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -n "$MODEL" ] || { echo "--model is required (e.g. --model esm2_8m)" >&2; exit 2; }
[ "${#DATA[@]}" -gt 0 ] || { echo "--data needs at least one file" >&2; exit 2; }

mkdir -p "$JOBS"
size() { du -h "$1" | cut -f1; }

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

echo
echo "staged into $JOBS"
echo "next: edit OSG_USER and OSG_AP in jobs/run_experiment.sub, then"
echo "      cd $JOBS && condor_submit run_experiment.sub"
