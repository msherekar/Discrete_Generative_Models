#!/usr/bin/env bash
# One-epoch end-to-end smoke test of the whole Project 1 pipeline.
#
# Exercises every innovation flag on a small slice of real avGFP data, then the
# MNIST transfer modality, then the post-training METL scoring pass. The point
# is to catch a signature or shape break in minutes rather than ten hours into
# an OSG job -- not to produce any scientific result, so every budget here is
# deliberately far too small to learn anything.
#
# Usage:
#   bash project1_eval/smoke_pipeline.sh            # everything
#   bash project1_eval/smoke_pipeline.sh units      # fast unit checks only
#   bash project1_eval/smoke_pipeline.sh gfp        # avGFP arms only
set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/python"
DATA="project1_eval/data"
OUT="project1_eval/outputs/smoke"
STAGE="${1:-all}"

# A slice small enough to encode in seconds, at the real avGFP length of 237.
SUBSET="$DATA/avgfp_smoke_train.csv"
VALSET="$DATA/avgfp_smoke_val.csv"
SUBSET_ROWS=400
VALSET_ROWS=200

# Shared budgets. One epoch, few steps, few samples: just enough to touch every
# code path once.
COMMON="--epochs 1 --samples 8 --steps 5 --batch-size 32 --hidden 64
        --max-length 237 --diffusion-steps 100 --sample-steps 5
        --arch transformer --no-oracle"

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

make_subset() {
  # head -n keeps the header row plus N data rows; the splits are already
  # shuffled by split_gfp.py, so a prefix is a valid random subset.
  [ -f "$SUBSET" ] || head -n $((SUBSET_ROWS + 1)) "$DATA/avgfp_train_props.csv" > "$SUBSET"
  [ -f "$VALSET" ] || head -n $((VALSET_ROWS + 1)) "$DATA/avgfp_val.csv" > "$VALSET"
  echo "  train slice: $(($(wc -l < "$SUBSET") - 1)) rows   val slice: $(($(wc -l < "$VALSET") - 1)) rows"
}

run_gfp() {
  local tag="$1"; shift
  banner "avGFP: $tag"
  # shellcheck disable=SC2086
  $PY -m dgm.project1.run_experiment \
    --dataset "$SUBSET" --dataset-tag "smoke_$tag" --outdir "$OUT/$tag" \
    $COMMON "$@" 2>&1 | grep -Ev 'Warning|warn|Found GPU|Minimum and|\(8\.0\)|enable_nested' \
    | grep -E 'Innovations|Latents|epoch|Path|objectives|distinct|Saved|Error|Traceback' || true
}

# ══════════════════════════════════════════════════════════════════════════════
# Stages
# ══════════════════════════════════════════════════════════════════════════════

if [ "$STAGE" = "all" ] || [ "$STAGE" = "units" ]; then
  banner "Unit checks (trunks, schedules, solvers, point-mass recovery)"
  $PY project1_eval/smoke_units.py --stage schedules
  $PY project1_eval/smoke_units.py --stage train 2>&1 | grep -Ev 'epoch|Warning|warn|Found GPU|Minimum and|\(8\.0\)|enable_nested'
  $PY project1_eval/smoke_units.py --stage sample 2>&1 | grep -Ev 'epoch|Warning|warn|Found GPU|Minimum and|\(8\.0\)|enable_nested'
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "gfp" ]; then
  banner "Preparing avGFP slices"
  make_subset

  # Stage 3 baseline: the control every innovation is read against.
  run_gfp baseline \
    --conditioning binary --predict x0 --beta-schedule linear \
    --flow-solver euler --diffusion-solver ddpm

  # Stage 0.1 + 0.2: censored reward loss and continuous conditioning.
  run_gfp data \
    --conditioning continuous --n-cond 3 --censor-floor -2.418182 \
    --val-dataset "$VALSET"

  # Stage 1: the architecture upgrade and its controls.
  run_gfp arch --conditioning continuous --modulation adaln --ema 0.999 \
    --warmup 5 --lr 1e-3 --reward-lr 3e-4
  run_gfp arch_control --conditioning continuous --modulation token --no-rope

  # Stage 2: matched NFE via DDIM.
  run_gfp matched --diffusion-solver ddim --sample-steps 5 --flow-solver euler

  # Axis A: probability path and noise schedule.
  run_gfp axis_a_cosine --beta-schedule cosine --time-schedule cosine
  run_gfp axis_a_hermite --beta-schedule sigmoid --time-schedule hermite
  run_gfp axis_a_arc --path-geometry arc

  # Axis B: coupling.
  run_gfp axis_b_ot --coupling ot
  run_gfp axis_b_informed --coupling informed

  # Axis C: objective.
  run_gfp axis_c_v --predict v --loss-weighting min-snr
  run_gfp axis_c_eps --predict eps --loss-weighting none

  # Axis D: guidance scaling, with both ablation controls.
  run_gfp axis_d_score --guidance-scaling score --reward-eta 10 --guidance-clip 2
  run_gfp axis_d_legacy --guidance-scaling legacy --reward-eta 10 --guidance-clip 2
  run_gfp axis_d_constant --guidance-scaling constant --reward-eta 10 --guidance-clip 2
  run_gfp axis_d_endpoint --endpoint-guidance --normalize-guidance --reward-eta 10 \
    --guidance-clip 2

  # Axis E: samplers.
  run_gfp axis_e_heun --flow-solver heun --diffusion-solver heun --sample-steps 5
  run_gfp axis_e_churn --diffusion-solver ddim --churn 0.05 --step-spacing quadratic
  run_gfp axis_e_corrector --diffusion-solver ddim --corrector-steps 1 \
    --reward-eta 5 --guidance-clip 2
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "mnist" ]; then
  banner "MNIST transfer modality"
  for unet in modern plain; do
    $PY -m dgm.project1.run_mnist --epochs 1 --limit 512 --samples 8 --steps 5 \
      --batch-size 64 --channels 16 --diffusion-steps 50 --predict x0 \
      --beta-schedule cosine --loss-weighting min-snr --unet "$unet" \
      --cfg-weight 2.0 --reward-eta 5.0 --outdir "$OUT/mnist_$unet" 2>&1 \
      | grep -Ev 'Warning|warn|Found GPU|Minimum and|\(8\.0\)' \
      | grep -E 'epoch|ink|Saved|Error|Traceback' || true
  done
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "post" ]; then
  banner "Post-training METL scoring"
  # --no-stability because the stability axis needs the full metl repo and the
  # PDB file; brightness needs only metl-pretrained.
  $PY -m dgm.project1.postprocess \
    "$OUT/baseline/flow" "$OUT/baseline/diffusion" \
    --no-stability --json "$OUT/postprocess.json" 2>&1 \
    | grep -Ev '%\|' || true
fi

banner "Smoke test complete"
