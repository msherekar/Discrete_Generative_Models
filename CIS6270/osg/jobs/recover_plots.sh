#!/bin/bash
# Recovery script for held jobs that generated plots but didn't transfer them
# Usage: ./recover_plots.sh <cluster_id>

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <cluster_id>" >&2
    echo "Example: $0 15833935" >&2
    exit 1
fi

CLUSTER_ID="$1"
TAG="${2:-GFP_H100}"  # Default tag, override as needed

echo "Attempting to recover plots from job ${CLUSTER_ID}..."
echo "This requires the job still be held (not removed)"

# Check if job exists
if ! condor_q -l "${CLUSTER_ID}" >/dev/null 2>&1; then
    echo "ERROR: Job ${CLUSTER_ID} not found in queue" >&2
    echo "If the job was removed, plots are lost on the remote node" >&2
    exit 1
fi

# Release the job with updated memory
echo "Option 1: Release job with higher memory allocation"
echo "  condor_qedit ${CLUSTER_ID} RequestMemory 72000"
echo "  condor_release ${CLUSTER_ID}"
echo ""
echo "Option 2: If plots are critical and job is held:"
echo "  Contact OSG support to retrieve /srv/scratch/CIS6270/project1_eval/plots/"
echo "  from the worker node before it's cleaned up"
echo ""
echo "Option 3: Resubmit with fixed memory:"
echo "  condor_rm ${CLUSTER_ID}"
echo "  condor_submit sweep.sub  # or run_experiment.sub"
