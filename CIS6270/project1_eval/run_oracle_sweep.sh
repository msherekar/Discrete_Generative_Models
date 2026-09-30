#!/usr/bin/env bash
#
# Full-data oracle comparison, safe to start in tmux and walk away.
#
# Fits the same ridge regression on six representations across three tasks of
# increasing difficulty, then fits the METL oracle used by gfp_metrics.py.
#
#   in-support   37,232 train / 4,655 test   ordinary interpolation
#   unseen-subs  21,670 train /   791 test   test substitutions absent from train
#   unseen-pos   22,975 train /   655 test   test positions absent from train
#
# Every model appends to data/oracle_sweep.json, so re-running skips nothing but
# never loses finished work either: kill it and start it again and the earlier
# rows are still there. One model failing does not stop the rest.
#
#   bash run_oracle_sweep.sh                 # everything
#   bash run_oracle_sweep.sh esm2_150m       # just these models
#   SKIP_ORACLE_FIT=1 bash run_oracle_sweep.sh
#
set -u -o pipefail

cd "$(dirname "$0")" || exit 1
PY=../.venv/bin/python
OUT=data/oracle_sweep.json
LOG="data/oracle_sweep_$(date +%Y%m%d_%H%M%S).log"
# 650M's feature matrix is ~45 GB at full data. If it fails, retry this size.
BIG_FALLBACK_N=20000

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(onehot metl esm2_8m esm2_35m esm2_150m esm2_650m)
fi

mkdir -p data
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

say "oracle sweep starting"
echo "  models : ${MODELS[*]}"
echo "  output : $OUT"
echo "  log    : $LOG"
echo "  memory : $(free -g | awk '/^Mem:/{print $7" GiB available of "$2" GiB"}')"

# ── preflight ────────────────────────────────────────────────────────────────
fail=0
for f in "$PY" data/avgfp_wt.txt; do
    [ -e "$f" ] || { echo "  MISSING: $f"; fail=1; }
done
"$PY" -c "import dgm.project1.oracle_sweep, dgm.project1.embedding_oracle" 2>/dev/null \
  || { echo "  MISSING dgm package (run 'uv sync' in ..)"; fail=1; }
for f in ../../metl/pretrained_models/Hr4GNHws.pt \
         ../../Data_GFP/data/dms_data/avgfp/avgfp.tsv; do
    [ -e "$f" ] || { echo "  MISSING: $f"; fail=1; }
done
[ $fail -eq 0 ] || { say "preflight failed, nothing run"; exit 1; }
echo "  preflight OK"

# True when the last run recorded an error row for this model, which is how
# oracle_sweep.py reports a task that ran out of memory rather than crashing.
model_errored() {
    "$PY" - "$OUT" "$1" <<'PYEOF'
import json, sys
from pathlib import Path
path, tag = Path(sys.argv[1]), sys.argv[2]
if not path.is_file():
    sys.exit(1)
rows = json.loads(path.read_text())
sys.exit(0 if any(r.get("model", "").startswith(tag) and "error" in r for r in rows) else 1)
PYEOF
}

# ── sweep ────────────────────────────────────────────────────────────────────
declare -a FAILED=()
for model in "${MODELS[@]}"; do
    say "=== $model ==="
    started=$SECONDS
    "$PY" -m dgm.project1.oracle_sweep --models "$model" --append --out "$OUT"
    status=$?

    if [ "$model" = "esm2_650m" ] && { [ $status -ne 0 ] || model_errored esm2_650m; }; then
        say "650M did not complete at full data; retrying with --train-n $BIG_FALLBACK_N"
        "$PY" -m dgm.project1.oracle_sweep --models "$model" --train-n "$BIG_FALLBACK_N" \
              --append --out "$OUT"
        status=$?
    fi

    if [ $status -ne 0 ]; then
        FAILED+=("$model")
        say "$model FAILED (exit $status) after $((SECONDS-started))s -- continuing"
    else
        say "$model done in $((SECONDS-started))s"
    fi
    echo "  memory now: $(free -g | awk '/^Mem:/{print $7" GiB available"}')"
done

# ── fit the METL oracle that gfp_metrics.py consumes ─────────────────────────
if [ "${SKIP_ORACLE_FIT:-0}" != "1" ]; then
    say "=== fitting METL oracle on the full oracle split ==="
    if [ -e data/avgfp_oracle.csv ]; then
        "$PY" -m dgm.project1.embedding_oracle --fit --backend metl \
            || say "oracle fit FAILED -- the sweep results above are unaffected"
    else
        say "data/avgfp_oracle.csv not found; run dgm-prepare-gfp first. Skipping."
    fi
fi

# ── summary ──────────────────────────────────────────────────────────────────
say "=== summary ==="
"$PY" - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    print("no results file"); raise SystemExit
rows = json.loads(path.read_text())
tasks, models = [], []
for r in rows:
    if r["task"] not in tasks: tasks.append(r["task"])
    if r["model"] not in models: models.append(r["model"])
cell = {}
for r in rows:
    if "error" in r:                       cell[(r["model"], r["task"])] = "ERROR"
    elif r.get("constant"):                cell[(r["model"], r["task"])] = "CONST"
    elif r.get("spearman") is not None:    cell[(r["model"], r["task"])] = f"{r['spearman']:+.4f}"
feat = {r["model"]: r.get("features") for r in rows if r.get("features")}
w = max(len(m) for m in models) + 2
print(f"\nSpearman on held-out test (higher is better; CONST = no ranking ability)\n")
print(f"{'model':<{w}}{'features':>10}" + "".join(f"{t:>16}" for t in tasks))
print("-" * (w + 10 + 16 * len(tasks)))
for m in models:
    print(f"{m:<{w}}{feat.get(m, 0):>10}"
          + "".join(f"{cell.get((m, t), '-'):>16}" for t in tasks))
print("\nCONST means every prediction was identical: the representation carries no")
print("information about substitutions the training split never contained.")
PYEOF

say "finished"
[ ${#FAILED[@]} -eq 0 ] || echo "  models that failed: ${FAILED[*]}"
echo "  results : $OUT"
echo "  log     : $LOG"
