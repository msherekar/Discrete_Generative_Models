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
PY="${PY:-../.venv/bin/python}"          # dgm is installed here by `uv sync`
SEEDS="${SEEDS:-11 12 13 14 15}"
ETAS="${ETAS:-1 5 20 50}"
AXIS="${AXIS:-eta}"                     # eta | steps
STEPS="${STEPS:-10 20 50 200}"           # levels when AXIS=steps
FIXED_ETA="${FIXED_ETA:-0}"              # guidance strength held fixed then
BETA="${BETA:-1.0}"                      # --coupling-beta for the c2/d variants
VARIANTS="${VARIANTS:-st ep}"            # st = state-based, ep = endpoint guidance
EPOCHS="${EPOCHS:-200}"
SAMPLES="${SAMPLES:-100}"
CFG_WEIGHT="${CFG_WEIGHT:-0}"
EXTRA="${EXTRA:-}"                       # any extra flags, applied to every run
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"      # do not recompute finished runs

# Modality-specific defaults.
case "$MODALITY" in
  protein) RUNNER="dgm.project1.run_experiment"
           DATA_ARGS="${DATA_ARGS:---esm-model esm2_8m --dataset ../lecture/lecture_3/esm2_example.csv}" ;;
  image)   RUNNER="dgm.project1.run_mnist"
           DATA_ARGS="${DATA_ARGS:---limit 20000}"
           EPOCHS="${EPOCHS_IMAGE:-$EPOCHS}" ;;
  *) echo "Unknown MODALITY '$MODALITY' (expected protein or image)" >&2; exit 1 ;;
esac

LOGDIR="logs"; mkdir -p "$LOGDIR" outputs
# Per-run logs live under the prefix, so one study's logs stay together and a
# rerun overwrites only its own.
RUNLOGDIR="$LOGDIR/$PREFIX"; mkdir -p "$RUNLOGDIR"
LOG="$LOGDIR/${PREFIX}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
say() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

say "sweep starting"
echo "  modality : $MODALITY  (runner: $RUNNER)"
echo "  prefix   : $PREFIX"
echo "  seeds    : $SEEDS"
echo "  axis     : $AXIS  (levels: $([ "$AXIS" = steps ] && echo "$STEPS" || echo "$ETAS"))"
echo "  variants : $VARIANTS"
echo "  epochs   : $EPOCHS   samples: $SAMPLES"
echo "  extra    : ${EXTRA:-<none>}"
echo "  log      : $LOG"
echo "  run logs : $RUNLOGDIR/<run>.log  (one per run, kept on success too)"

# ── preflight ────────────────────────────────────────────────────────────────
fail=0
[ -x "$PY" ] || { echo "  MISSING interpreter: $PY"; fail=1; }
"$PY" -c "import $RUNNER" 2>/dev/null \
  || { echo "  MISSING runner module: $RUNNER (run 'uv sync' in ..)"; fail=1; }
"$PY" -c "import dgm.project1.analyze_sweep" 2>/dev/null \
  || { echo "  MISSING dgm.project1.analyze_sweep"; fail=1; }
[ $fail -eq 0 ] || { say "preflight failed, nothing run"; exit 1; }
[ "$AXIS" = "steps" ] && LEVELS="$STEPS" || LEVELS="$ETAS"
n_runs=$(( $(wc -w <<<"$SEEDS") * $(wc -w <<<"$LEVELS") * $(wc -w <<<"$VARIANTS") ))
echo "  preflight OK -- $n_runs run(s) planned"

# A variant is just a flag set; add cases here to extend the study.
# base/ot/c1/c2/d are the coupling study: see src/dgm/project1/pipeline/coupling.py.
variant_flags() {
  case "$1" in
    st)   echo "" ;;                       # state-based guidance (the default)
    ep)   echo "--endpoint-guidance" ;;
    base) echo "--coupling independent" ;; # the control every coupling is read against
    ot)   echo "--coupling ot" ;;                                  # option B
    c1)   echo "--coupling informed" ;;                            # option C1
    # The two controls C1 needs. shuf keeps the map, the residual scale and the
    # source marginal and destroys only which property belongs to which sample,
    # so it isolates the INFORMATION. scaled drops the map entirely and keeps
    # only the width, so it isolates the SCALE -- which matters because N(0,I)
    # is 63% wider than MNIST as trained (data sd 0.614) while peptide latents
    # are standardized to 1.000, and the informed source matches each.
    shuf)   echo "--coupling informed-shuffled" ;;
    scaled) echo "--coupling scaled" ;;
    c2)   echo "--coupling aux --coupling-beta ${BETA:-1.0}" ;;     # option C2
    d)    echo "--coupling aux --coupling-beta ${BETA:-1.0} --coupling-columns 1" ;;
    # The interpolant study: a geometry, a training schedule and a sampling grid
    # are three independent choices, so they get three independent variants.
    seg)   echo "--path-geometry segment" ;;          # the straight-line control
    arc)   echo "--path-geometry arc" ;;              # variance-preserving at sd 1
    darc)  echo "--path-geometry data-arc" ;;         # corrected for the data's sd
    tquad) echo "--time-schedule quadratic" ;;        # training weighting only
    tcos)  echo "--time-schedule cosine" ;;
    gquad) echo "--sample-schedule quadratic" ;;      # step placement only
    gcos)  echo "--sample-schedule cosine" ;;
    arcq)  echo "--path-geometry arc --time-schedule quadratic" ;;  # was unreachable
    *)    echo "" ;;
  esac
}

# AXIS=eta sweeps guidance strength (the original study); AXIS=steps sweeps
# integration steps at a fixed eta, which is the axis a coupling claim needs.
# Either way the level lands in the run-directory name, so analyze_sweep.py
# reads both layouts with --x-axis.
level_flag() {
  if [ "$AXIS" = "steps" ]; then
    echo "--steps $1 --reward-eta ${FIXED_ETA:-0}"
  else
    echo "--reward-eta $1"
  fi
}

build_cmd() {
  local variant="$1" level="$2" seed="$3" dir="$4"
  # -u is not cosmetic: with stdout block-buffered into a log, the whole run's
  # output flushes at exit AFTER any stderr traceback, so the tail of a failed
  # run's log shows its header instead of the error that killed it. Unbuffered
  # also makes `tail -f` on a run log work while it is still going.
  printf '%s -u -m %s %s --dataset-tag %s --seed %s --sample-seed %s %s --cfg-weight %s --epochs %s --samples %s --outdir %s %s %s' \
    "$PY" "$RUNNER" "$DATA_ARGS" "$(basename "$dir")" "$seed" "$seed" \
    "$(level_flag "$level")" \
    "$CFG_WEIGHT" "$EPOCHS" "$SAMPLES" "$dir" "$(variant_flags "$variant")" "$EXTRA"
}

# ── sweep ────────────────────────────────────────────────────────────────────
declare -a FAILED=(); done_n=0; skipped=0
for seed in $SEEDS; do
  for eta in $LEVELS; do
    for variant in $VARIANTS; do
      dir="outputs/${PREFIX}_${variant}_${eta}_s${seed}"
      # A finished run has results.pt somewhere under its directory.
      if [ "$SKIP_EXISTING" = "1" ] && compgen -G "$dir/**/results.pt" >/dev/null 2>&1 \
         || { [ "$SKIP_EXISTING" = "1" ] && [ -f "$dir/results.pt" ]; }; then
        skipped=$((skipped+1)); continue
      fi
      cmd="$(build_cmd "$variant" "$eta" "$seed" "$dir")"
      if [ "$DRY_RUN" = "1" ]; then echo "  $cmd"; continue; fi
      run="${PREFIX}_${variant}_${eta}_s${seed}"
      # One log per run, kept whether it passed or failed. This used to go to
      # /dev/null, which made every failure unreadable: a sweep would report 46
      # failed runs and there was nothing to look at. The command itself goes in
      # at the top, so the log is enough to reproduce the run by hand.
      runlog="$RUNLOGDIR/${run}.log"
      { echo "# $(date '+%F %T')"; echo "# $cmd"; echo; } > "$runlog"
      started=$SECONDS
      if eval "$cmd" >>"$runlog" 2>&1; then
        done_n=$((done_n+1))
        printf '  %-42s %4ds\n' "$run" $((SECONDS-started))
      else
        status=$?
        FAILED+=("$run")
        say "FAILED (exit $status): $run -- continuing"
        # The tail is almost always the traceback, which is what you need to see
        # without opening the file.
        sed -n '4,$p' "$runlog" | tail -n "${FAIL_TAIL:-8}" | sed 's/^/      | /'
        echo "      full log: $runlog"
      fi
    done
  done
done
[ "$DRY_RUN" = "1" ] && { say "dry run complete ($n_runs command(s))"; exit 0; }
say "ran $done_n, skipped $skipped already-complete, failed ${#FAILED[@]}"

# ── analyze + plot ───────────────────────────────────────────────────────────
say "analyzing"
IMAGES_FLAG=""; [ "$MODALITY" = "image" ] && IMAGES_FLAG="--images"
if [ "$AXIS" = "steps" ]; then
  "$PY" -m dgm.project1.analyze_sweep --prefix "$PREFIX" --variants $VARIANTS \
       --x-axis steps --steps $STEPS --seeds $SEEDS $IMAGES_FLAG \
       || say "analysis FAILED"
else
  "$PY" -m dgm.project1.analyze_sweep --prefix "$PREFIX" --variants $VARIANTS \
       --etas $ETAS --seeds $SEEDS $IMAGES_FLAG || say "analysis FAILED"
fi

say "finished"
if [ ${#FAILED[@]} -ne 0 ]; then
  echo "  failed runs: ${FAILED[*]}"
  echo "  inspect:     tail -40 $RUNLOGDIR/${FAILED[0]}.log"
fi
echo "  results : outputs/${PREFIX}_*"
echo "  figures : plots/${PREFIX}_sweep/"
echo "  log     : $LOG"
echo "  run logs: $RUNLOGDIR/"
