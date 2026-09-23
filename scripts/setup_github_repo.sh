#!/usr/bin/env bash
set -euo pipefail

# Initializes a local git repo, creates README/.gitignore, and creates/links GitHub.
# Usage: ./scripts/setup_github_repo.sh <repo-name> [--public|--private] [--no-push]
#        ./scripts/setup_github_repo.sh <repo-name> <remote-url> [--no-push]
# With only <repo-name>: creates github.com/<you>/<repo-name> via `gh` and pushes.
# --private (default) and --public only apply when creating the repo with gh.

usage() {
  echo "Usage: $0 <repo-name> [--public|--private] [--no-push]" >&2
  echo "       $0 <repo-name> <https-or-git-remote-url> [--no-push]" >&2
  echo "  First form creates the repository on GitHub (requires: gh auth login)." >&2
  echo "  Visibility defaults to private; use --public for a public repository." >&2
  exit 1
}

REPO_NAME="${1:-}"
[ -n "${REPO_NAME}" ] || usage
REMOTE_URL=""
NO_PUSH=0
VISIBILITY=""
VISIBILITY_EXPLICIT=0
shift || true
while [ $# -gt 0 ]; do
  case "$1" in
    --no-push) NO_PUSH=1 ;;
    --public)
      if [ "${VISIBILITY_EXPLICIT}" -eq 1 ] && [ "${VISIBILITY}" != "public" ]; then
        echo "Use only one of --public or --private." >&2
        exit 1
      fi
      VISIBILITY="public"
      VISIBILITY_EXPLICIT=1
      ;;
    --private)
      if [ "${VISIBILITY_EXPLICIT}" -eq 1 ] && [ "${VISIBILITY}" != "private" ]; then
        echo "Use only one of --public or --private." >&2
        exit 1
      fi
      VISIBILITY="private"
      VISIBILITY_EXPLICIT=1
      ;;
    http://*|https://*|git@*) REMOTE_URL="$1" ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
  shift
done

VISIBILITY="${VISIBILITY:-private}"

DEFAULT_BRANCH="main"

if [ ! -d .git ]; then
  git init
fi

if [ ! -f README.md ]; then
  echo "# ${REPO_NAME}" > README.md
elif ! grep -q "^# ${REPO_NAME}$" README.md; then
  echo "# ${REPO_NAME}" >> README.md
fi

if [ ! -f .gitignore ]; then
  cat <<'EOF' > .gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# OS / Editor
.DS_Store
.vscode/

# Logs / Env
*.log
.env
EOF
fi

git add README.md .gitignore

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  git commit -m "first commit"
fi

git branch -M "${DEFAULT_BRANCH}"

if [ -n "${REMOTE_URL}" ]; then
  if [ "${VISIBILITY_EXPLICIT}" -eq 1 ]; then
    echo "Note: --public/--private are ignored when a remote URL is given." >&2
  fi
  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "${REMOTE_URL}"
  else
    git remote add origin "${REMOTE_URL}"
  fi
  echo "Remote origin is set to: $(git remote get-url origin)"
  if [ "${NO_PUSH}" -eq 0 ]; then
    git push -u origin "${DEFAULT_BRANCH}"
  else
    echo "Skipping push. To push later, run: git push -u origin ${DEFAULT_BRANCH}"
  fi
else
  command -v gh >/dev/null 2>&1 || {
    echo "GitHub CLI (gh) is required to create the repo. Install: https://cli.github.com" >&2
    exit 1
  }
  gh auth status >/dev/null 2>&1 || {
    echo "Not logged in to GitHub. Run: gh auth login" >&2
    exit 1
  }
  if git remote get-url origin >/dev/null 2>&1; then
    git remote remove origin
  fi
  GH_VISIBILITY=(--private)
  if [ "${VISIBILITY}" = "public" ]; then
    GH_VISIBILITY=(--public)
  fi
  if [ "${NO_PUSH}" -eq 0 ]; then
    gh repo create "${REPO_NAME}" "${GH_VISIBILITY[@]}" --source=. --remote=origin --push
  else
    gh repo create "${REPO_NAME}" "${GH_VISIBILITY[@]}" --source=. --remote=origin
    echo "Skipping push. To push later, run: git push -u origin ${DEFAULT_BRANCH}"
  fi
  echo "Remote origin is set to: $(git remote get-url origin)"
fi
