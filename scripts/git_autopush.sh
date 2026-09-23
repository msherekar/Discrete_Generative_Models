#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/git_autopush.sh [options]

Watches the working tree and automatically commits + pushes changes once
they have settled (no further edits for DEBOUNCE seconds).

Options:
  -p, --path PATH        Watch only this path (repeatable). Default: whole repo
  -i, --interval SECS    How often to check for changes. Default: 5
  -d, --debounce SECS    Wait this long after the last change before committing.
                         Default: 20
  -m, --message PREFIX   Commit message prefix. Default: "auto"
  -1, --once             Do one check-and-commit pass, then exit (for cron/hooks)
  -n, --dry-run          Show what would be committed; do not commit or push
      --no-push          Commit locally but never push
  -l, --log FILE         Also append output to FILE
  -h, --help             Show this help

Examples:
  # Watch everything, background it, log to a file
  nohup ./scripts/git_autopush.sh -l /tmp/autopush.log >/dev/null 2>&1 &

  # Only watch src/ and diagnostics/, commit 60s after edits stop
  ./scripts/git_autopush.sh -p src -p diagnostics -d 60

  # See what it would do without touching git
  ./scripts/git_autopush.sh --once --dry-run

Notes:
  * __pycache__ and *.pyc are always excluded.
  * Respects .gitignore (it uses `git status` as its change signal).
  * Only one instance can run per repo (lock file in .git/).
  * On a rejected push it tries `git pull --rebase --autostash` once, then
    retries. If that fails it logs and keeps waiting rather than forcing.
EOF
}

# ---------------------------------------------------------------- config -----

INTERVAL=5
DEBOUNCE=20
PREFIX="auto"
DRY_RUN=0
ONCE=0
DO_PUSH=1
LOG_FILE=""
WATCH_PATHS=()

# Never auto-commit Python bytecode, whatever .gitignore says.
EXCLUDES=(
  ':(exclude)**/__pycache__/**'
  ':(exclude)**/*.pyc'
)

# ------------------------------------------------------------------ utils ----

log() {
  local line
  line="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$line"
  [[ -n "$LOG_FILE" ]] && echo "$line" >>"$LOG_FILE"
  return 0
}

die() {
  log "ERROR: $*"
  exit 1
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)     usage; exit 0 ;;
      -p|--path)     WATCH_PATHS+=("${2:?--path needs a value}"); shift 2 ;;
      -i|--interval) INTERVAL="${2:?--interval needs a value}"; shift 2 ;;
      -d|--debounce) DEBOUNCE="${2:?--debounce needs a value}"; shift 2 ;;
      -m|--message)  PREFIX="${2:?--message needs a value}"; shift 2 ;;
      -l|--log)      LOG_FILE="${2:?--log needs a value}"; shift 2 ;;
      -1|--once)     ONCE=1; shift ;;
      -n|--dry-run)  DRY_RUN=1; shift ;;
      --no-push)     DO_PUSH=0; shift ;;
      *)             usage; die "unknown option: $1" ;;
    esac
  done

  [[ "$INTERVAL" =~ ^[0-9]+$ && "$INTERVAL" -ge 1 ]] || die "--interval must be a positive integer"
  [[ "$DEBOUNCE" =~ ^[0-9]+$ ]] || die "--debounce must be a non-negative integer"

  if [[ ${#WATCH_PATHS[@]} -eq 0 ]]; then
    WATCH_PATHS=('.')
  fi
  return 0
}

require_git_repo() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "not inside a git repository."
  cd "$(git rev-parse --show-toplevel)"
}

# Don't commit mid-merge/mid-rebase: staging then would corrupt the operation.
repo_is_busy() {
  local gitdir
  gitdir="$(git rev-parse --git-dir)"
  [[ -e "$gitdir/MERGE_HEAD" ]] && { echo "merge in progress"; return 0; }
  [[ -d "$gitdir/rebase-merge" || -d "$gitdir/rebase-apply" ]] && { echo "rebase in progress"; return 0; }
  [[ -e "$gitdir/CHERRY_PICK_HEAD" ]] && { echo "cherry-pick in progress"; return 0; }
  [[ -e "$gitdir/BISECT_LOG" ]] && { echo "bisect in progress"; return 0; }
  return 1
}

current_branch() {
  git branch --show-current
}

# Porcelain status, scoped to the watched paths, minus the hard excludes.
# This is our change signal: empty output means nothing to do.
# -uall so untracked directories are listed file-by-file rather than collapsed
# to "dir/" — gives accurate commit messages and a finer change signal.
snapshot() {
  git status --porcelain -uall -- "${WATCH_PATHS[@]}" "${EXCLUDES[@]}"
}

# Turn porcelain lines into a list of paths (handling `R old -> new` renames).
changed_files() {
  local line path
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    path="${line:3}"
    if [[ "$path" == *" -> "* ]]; then
      path="${path##* -> }"
    fi
    path="${path%\"}"
    path="${path#\"}"
    echo "$path"
  done
}

build_message() {
  local files=("$@")
  local count=${#files[@]}
  local stamp
  stamp="$(date '+%Y-%m-%d %H:%M')"

  if [[ $count -eq 1 ]]; then
    echo "$PREFIX: update ${files[0]} [$stamp]"
  elif [[ $count -le 3 ]]; then
    local joined
    printf -v joined '%s, ' "${files[@]}"
    echo "$PREFIX: update ${joined%, } [$stamp]"
  else
    echo "$PREFIX: update ${files[0]}, ${files[1]} and $((count - 2)) more [$stamp]"
  fi
}

# Run a git command, log each output line, and preserve its exit status.
run_logged() {
  local out status=0
  out="$("$@" 2>&1)" || status=$?
  while IFS= read -r l; do
    [[ -n "$l" ]] && log "  $l"
  done <<<"$out"
  return $status
}

push_current_branch() {
  local branch="$1"

  if git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' >/dev/null 2>&1; then
    run_logged git push origin "$branch" && return 0
  else
    log "  no upstream set; pushing with -u"
    run_logged git push -u origin "$branch" && return 0
  fi

  log "  push rejected; trying pull --rebase --autostash"
  if run_logged git pull --rebase --autostash; then
    run_logged git push origin "$branch" && return 0
  fi

  log "  WARNING: push still failing. Commit is saved locally; resolve by hand."
  return 1
}

commit_and_push() {
  local snap="$1"
  local files=()
  mapfile -t files < <(printf '%s\n' "$snap" | changed_files)
  [[ ${#files[@]} -eq 0 ]] && return 0

  local message
  message="$(build_message "${files[@]}")"

  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY RUN — would commit ${#files[@]} file(s): $message"
    printf '%s\n' "${files[@]}" | sed 's/^/    /'
    return 0
  fi

  log "Staging ${#files[@]} change(s)"
  git add -A -- "${WATCH_PATHS[@]}" "${EXCLUDES[@]}"

  if git diff --cached --quiet; then
    log "Nothing staged after exclusions; skipping."
    return 0
  fi

  log "Committing: $message"
  git commit -q -m "$message"

  if [[ $DO_PUSH -eq 0 ]]; then
    log "--no-push set; commit kept local."
    return 0
  fi

  local branch
  branch="$(current_branch)"
  log "Pushing to origin/$branch"
  push_current_branch "$branch" || true
}

# ------------------------------------------------------------------- main ----

main() {
  parse_args "$@"
  require_git_repo

  [[ -n "$LOG_FILE" ]] && touch "$LOG_FILE"

  local branch
  branch="$(current_branch)"
  [[ -z "$branch" ]] && die "detached HEAD — refusing to auto-commit."

  # One watcher per repo, or two instances race on the index.
  local lock="$(git rev-parse --git-dir)/autopush.lock"
  exec 9>"$lock"
  flock -n 9 || die "another git_autopush.sh is already running for this repo."

  if [[ $ONCE -eq 1 ]]; then
    local busy
    if busy="$(repo_is_busy)"; then
      log "Skipping: $busy"
      exit 0
    fi
    local snap
    snap="$(snapshot)"
    if [[ -z "$snap" ]]; then
      log "No changes."
      exit 0
    fi
    commit_and_push "$snap"
    exit 0
  fi

  log "Watching ${WATCH_PATHS[*]} on branch '$branch'"
  log "  poll every ${INTERVAL}s, commit after ${DEBOUNCE}s of quiet$([[ $DRY_RUN -eq 1 ]] && echo ' (dry run)')"
  log "  Ctrl-C to stop."

  trap 'log "Stopped."; exit 0' INT TERM

  local last_snap="" stable=0
  while true; do
    local busy snap
    if busy="$(repo_is_busy)"; then
      log "Paused: $busy"
      sleep "$INTERVAL"
      continue
    fi

    snap="$(snapshot)"

    if [[ -z "$snap" ]]; then
      last_snap=""
      stable=0
      sleep "$INTERVAL"
      continue
    fi

    if [[ "$snap" == "$last_snap" ]]; then
      stable=$((stable + INTERVAL))
    else
      local n
      n="$(printf '%s\n' "$snap" | grep -c . || true)"
      log "Detected $n change(s); waiting for edits to settle"
      last_snap="$snap"
      stable=0
    fi

    if [[ $stable -ge $DEBOUNCE ]]; then
      commit_and_push "$snap"
      last_snap=""
      stable=0
    fi

    sleep "$INTERVAL"
  done
}

main "$@"
