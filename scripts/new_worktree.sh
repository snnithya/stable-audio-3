#!/usr/bin/env bash
# Create an isolated worktree + branch so an agent can work while you keep
# editing the main checkout. See "Branch & worktree policy" in AGENTS.md.
#
#   scripts/new_worktree.sh <task-name> [base-ref]
#
# Worktrees live outside the repo (default: <repo-parent>/sa3-wt/<task-name>)
# and share the main checkout's .git and .venv.
set -euo pipefail

usage() { sed -n '2,8p' "$0" | sed 's/^# \?//'; exit 1; }
[[ $# -ge 1 && $1 != -h && $1 != --help ]] || usage

name=$1
[[ $name =~ ^[a-z0-9][a-z0-9._-]*$ ]] || { echo "error: task name must be kebab-case: $name" >&2; exit 1; }

# Resolve the main checkout even when invoked from inside another worktree.
main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
base=${2:-$(git rev-parse --abbrev-ref HEAD)}
branch=claude/$name
wt_root=${SA3_WT_ROOT:-$(dirname "$main")/sa3-wt}
wt=$wt_root/$name

git show-ref --verify --quiet "refs/heads/$branch" && { echo "error: branch $branch already exists" >&2; exit 1; }
[[ -e $wt ]] && { echo "error: $wt already exists" >&2; exit 1; }

git worktree add -b "$branch" "$wt" "$base"

# Gitignored bits the worktree needs. Symlink .env so secrets live in one place.
ln -s "$main/.env" "$wt/.env"

# The venv's editable install hardcodes $main, so imports silently resolve to
# the main checkout unless PYTHONPATH points here. Also expose the shared
# interpreter: `uv run`/`uv sync` must NOT be used from a worktree.
mkdir -p "$wt/.claude"
cat > "$wt/.claude/settings.local.json" <<JSON
{
  "env": {
    "PYTHONPATH": "$wt",
    "SA3_PY": "$main/.venv/bin/python"
  }
}
JSON

# Editor side of the same problem: there is no .venv here, and launch.json
# resolves the interpreter from the window's selection. Point both at the
# shared venv and let Pylance see this worktree's sources.
mkdir -p "$wt/.vscode"
cat > "$wt/.vscode/settings.json" <<JSON
{
  "python.defaultInterpreterPath": "$main/.venv/bin/python",
  "python.analysis.extraPaths": ["."]
}
JSON

cat <<MSG

Worktree ready:
  branch   $branch  (based on $base)
  path     $wt

  cd $wt && claude
  code $wt          # or: File > Add Folder to Workspace, to see both trees at once

Run things with \$SA3_PY (never 'uv run' / 'uv sync' here — they would build a
second 4.4G venv or repoint the shared editable install at this worktree):
  \$SA3_PY -m pytest tests/...

Review and merge from $main:
  git log --oneline $base..$branch
  git diff $base...$branch
  git merge $branch          # or: git cherry-pick <sha>

Tear down when merged:
  git worktree remove $wt && git branch -d $branch
MSG
