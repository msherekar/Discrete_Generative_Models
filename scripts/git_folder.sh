#!/usr/bin/env bash
# Stage and push a single folder (not the whole repo).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/git_folder.sh <folder_path> "your commit message"

Examples:
  ./scripts/git_folder.sh scripts "chore: update scripts"
  ./scripts/git_folder.sh src/of3 "feat: add of3 package"

What it does:
  1) Shows git status
  2) Adds only the given folder
  3) Commits with your message
  4) Pushes current branch to origin
EOF
}

require_git_repo() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Error: not inside a git repository."
    exit 1
  }
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
    usage
    exit 0
  fi

  local folder_path="$1"
  local commit_message="$2"
  require_git_repo

  if [[ ! -d "$folder_path" ]]; then
    echo "Error: folder does not exist -> $folder_path"
    exit 1
  fi

  # Normalize trailing slash so git paths are consistent
  folder_path="${folder_path%/}"

  # Nested repos become unclickable gitlinks (mode 160000) on GitHub.
  if [[ -e "$folder_path/.git" ]]; then
    echo "Error: '$folder_path' has its own .git (nested repository)."
    echo "Git would record a submodule-style pointer, not the files."
    echo "Fix: move/remove '$folder_path/.git', then re-run this script,"
    echo "  or add it properly with: git submodule add <url> $folder_path"
    exit 1
  fi

  echo "==> Current status"
  git status --short
  echo

  echo "==> Staging folder: $folder_path"
  git add -- "$folder_path"

  if git diff --cached --quiet; then
    echo "Nothing staged under $folder_path. Aborting."
    exit 0
  fi

  echo "==> Creating commit"
  git commit -m "$commit_message"

  echo "==> Pushing to origin"
  local branch
  branch="$(git branch --show-current)"
  git push -u origin "$branch"

  echo "Done: folder committed and pushed."
}

main "$@"
