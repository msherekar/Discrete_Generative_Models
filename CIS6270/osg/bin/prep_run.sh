#!/bin/bash
#
# Creates a run's OSDF output folder before submitting.
#
#   prep_run.sh <tag> [<tag> ...]
#
# transfer_output_remaps writes files INTO a path; it does not reliably create
# the intermediate directories. /ospool is mounted on the access point, so the
# folder is just an mkdir here. Run this for every tag you are about to submit.
#
# It also refuses a tag that already has results, because OSDF caches by path:
# reusing a tag can serve the previous run's file to a later read.
#
set -euo pipefail

OSG_USER="${OSG_USER:-$USER}"
OSG_AP="${OSG_AP:-ap40}"
RUNS="/ospool/${OSG_AP}/data/${OSG_USER}/DGM/runs"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <tag> [<tag> ...]" >&2
    echo "  OSG_USER and OSG_AP may be set in the environment" >&2
    echo "  current: $RUNS" >&2
    exit 2
fi

[ -d "$(dirname "$RUNS")" ] || {
    echo "no such OSDF area: $(dirname "$RUNS")" >&2
    echo "check OSG_AP ($OSG_AP) and OSG_USER ($OSG_USER)" >&2
    exit 1
}

fail=0
for tag in "$@"; do
    dir="$RUNS/$tag"
    if [ -e "$dir/flow_results.pt" ] || [ -e "$dir/diffusion_results.pt" ]; then
        echo "  REFUSING $tag: $dir already holds results"
        echo "    OSDF caches by path, so reuse can serve the old file."
        echo "    Use a new tag, e.g. ${tag}_$(date +%Y%m%d)"
        fail=1
        continue
    fi
    mkdir -p "$dir"
    echo "  ready $dir"
done

[ "$fail" -eq 0 ] || exit 1
echo
echo "now submit with a matching TAG, e.g."
echo "  condor_submit run_experiment.sub -append 'TAG = $1'"
