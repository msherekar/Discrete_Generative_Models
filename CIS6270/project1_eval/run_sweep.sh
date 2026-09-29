#!/usr/bin/env bash
#
# One driver for the whole guidance study: sweep, analyze, plot, save.
#
# Everything that changes between experiments is in the CONFIG block below or
# overridable from the environment, so a new dataset or a new parameter grid
# needs no edits to the loop itself.
#
#   bash run_sweep.sh                                  # peptides, the default grid
#   MODALITY=image PREFIX=mn bash run_sweep.sh         # MNIST instead
#   ETAS="1 5" SEEDS="11 12" bash run_sweep.sh         # a smaller grid
#   VARIANTS="ep" EXTRA="--predict x0 --ema 0.99" bash run_sweep.sh
#   DRY_RUN=1 bash run_sweep.sh                        # print commands only
#
# Adding a third modality means writing one runner script that accepts
# --dataset-tag/--seed/--sample-seed/--reward-eta/--outdir and adding a case to
# build_cmd() below. Nothing else changes.
#
set -u -o pipefail
cd "$(dirname "$0")" || exit 1

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODALITY="${MODALITY:-protein}"          # protein | image
PREFIX="${PREFIX:-ep}"                   # run-name prefix; keep distinct per study
PY="${PY:-../.venv/bin/python}"
SEEDS="${SEEDS:-11 12 13 14 15}"
ETAS="${ETAS:-1 5 20 50}"
VARIANTS="${VARIANTS:-st ep}"            # st = state-based, ep = endpoint guidance
EPOCHS="${EPOCHS:-200}"
SAMPLES="${SAMPLES:-100}"
CFG_WEIGHT="${CFG_WEIGHT:-0}"
EXTRA="${EXTRA:-}"                       # any extra flags, applied to every run
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"      # do not recompute finished runs

# Modality-specific defaults.
case "$MODALITY" in
  protein) RUNNER="run_experiment.py"
           DATA_ARGS="${DATA_ARGS:---esm-model esm2_8m --dataset ../lecture_3/esm2_example.csv}" ;;
  image)   RUNNER="run_mnist.py"
           DATA_ARGS="${DATA_ARGS:---limit 20000}"
           EPOCHS="${EPOCHS_IMAGE:-$EPOCHS}" ;;
  *) echo "Unknown MODALITY '$MODALITY' (expected protein or image)" >&2; exit 1 ;;
esac

LOGDIR="logs"; mkdir -p "$LOGDIR" outputs
LOG="$LOGDIR/${PREFIX}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
say() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

say "sweep starting"
echo "  modality : $MODALITY  (runner: $RUNNER)"
echo "  prefix   : $PREFIX"
echo "  seeds    : $SEEDS"
echo "  etas     : $ETAS"
echo "  variants : $VARIANTS"
echo "  epochs   : $EPOCHS   samples: $SAMPLES"
echo "  extra    : ${EXTRA:-<none>}"
echo "  log      : $LOG"

# ── preflight ────────────────────────────────────────────────────────────────
fail=0
[ -x "$PY" ] || { echo "  MISSING interpreter: $PY"; fail=1; }
[ -f "$RUNNER" ] || { echo "  MISSING runner: $RUNNER"; fail=1; }
[ -f analyze_sweep.py ] || { echo "  MISSING analyze_sweep.py"; fail=1; }
[ $fail -eq 0 ] || { say "preflight failed, nothing run"; exit 1; }
n_runs=$(( $(wc -w <<<"$SEEDS") * $(wc -w <<<"$ETAS") * $(wc -w <<<"$VARIANTS") ))
echo "  preflight OK -- $n_runs run(s) planned"

# A variant is just a flag set; add cases here to extend the study.
variant_flags() {
  case "$1" in
    st)   echo "" ;;                       # state-based guidance (the default)
    ep)   echo "--endpoint-guidance" ;;
    *)    echo "" ;;
  esac
}

build_cmd() {
  local variant="$1" eta="$2" seed="$3" dir="$4"
  printf '%s %s %s --dataset-tag %s --seed %s --sample-seed %s --reward-eta %s --cfg-weight %s --epochs %s --samples %s --outdir %s %s %s' \
    "$PY" "$RUNNER" "$DATA_ARGS" "$(basename "$dir")" "$seed" "$seed" "$eta" \
    "$CFG_WEIGHT" "$EPOCHS" "$SAMPLES" "$dir" "$(variant_flags "$variant")" "$EXTRA"
}

# ── sweep ────────────────────────────────────────────────────────────────────
declare -a FAILED=(); done_n=0; skipped=0
for seed in $SEEDS; do
  for eta in $ETAS; do
    for variant in $VARIANTS; do
      dir="outputs/${PREFIX}_${variant}_${eta}_s${seed}"
      # A finished run has results.pt somewhere under its directory.
      if [ "$SKIP_EXISTING" = "1" ] && compgen -G "$dir/**/results.pt" >/dev/null 2>&1 \
         || { [ "$SKIP_EXISTING" = "1" ] && [ -f "$dir/results.pt" ]; }; then
        skipped=$((skipped+1)); continue
      fi
      cmd="$(build_cmd "$variant" "$eta" "$seed" "$dir")"
      if [ "$DRY_RUN" = "1" ]; then echo "  $cmd"; continue; fi
      started=$SECONDS
      if eval "$cmd" >/dev/null 2>&1; then
        done_n=$((done_n+1))
        printf '  %-42s %4ds\n' "${PREFIX}_${variant}_${eta}_s${seed}" $((SECONDS-started))
      else
        FAILED+=("${PREFIX}_${variant}_${eta}_s${seed}")
        say "FAILED: ${PREFIX}_${variant}_${eta}_s${seed} -- continuing"
      fi
    done
  done
done
[ "$DRY_RUN" = "1" ] && { say "dry run complete ($n_runs command(s))"; exit 0; }
say "ran $done_n, skipped $skipped already-complete, failed ${#FAILED[@]}"

# ── analyze + plot ───────────────────────────────────────────────────────────
say "analyzing"
IMAGES_FLAG=""; [ "$MODALITY" = "image" ] && IMAGES_FLAG="--images"
"$PY" analyze_sweep.py --prefix "$PREFIX" --variants $VARIANTS \
     --etas $ETAS --seeds $SEEDS $IMAGES_FLAG || say "analysis FAILED"

say "finished"
[ ${#FAILED[@]} -eq 0 ] || echo "  failed runs: ${FAILED[*]}"
echo "  results : outputs/${PREFIX}_*"
echo "  figures : plots/${PREFIX}_sweep/"
echo "  log     : $LOG"
