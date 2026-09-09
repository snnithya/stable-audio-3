# Parallel work with git worktrees

Agents work on their own branch in their own directory, so you can keep editing
and running jobs in the main checkout at the same time. A branch alone is not
enough for this: one directory holds one checked-out branch, so an agent that
ran `git checkout -b` in the main checkout would pull the files out from under
you. Each task therefore gets a **branch and a worktree**.

- **Main checkout** — `/data/hai-res/snnithya/stable-audio-3`, your `v/r` work. Agents never touch it.
- **Agent worktrees** — `../sa3-wt/<task-name>` on branch `claude/<task-name>`, sharing the main checkout's `.git` and `.venv`.

Because the `.git` directory is shared, an agent's commits are visible to you
immediately — no remote, no fetch.

## Creating one

From the main checkout:

```bash
scripts/new_worktree.sh <task-name> [base-ref]   # base-ref defaults to the current branch
```

This creates branch `claude/<task-name>`, adds the worktree, and fills in the
three gitignored things a worktree needs but does not inherit:

| File | Why |
|------|-----|
| `.env` (symlink to the main checkout) | Secrets stay in one place and cannot drift |
| `.claude/settings.local.json` | Sets `PYTHONPATH` to the worktree root and `SA3_PY` to the shared interpreter |
| `.vscode/settings.json` | Points the Python extension at the shared venv and adds the worktree to Pylance's search path |

Override the worktree location with `SA3_WT_ROOT` if you want them somewhere else.

## Running code in a worktree

The shared `.venv` contains an **editable install whose path is hardcoded to the
main checkout** (`.venv/.../_editable_impl_stable_audio_3.pth` literally
contains `/data/hai-res/snnithya/stable-audio-3`). So an import from a worktree
can silently resolve to the main checkout's source — no error, just the wrong
code. Observed from a worktree with a sentinel added to its
`stable_audio_3/__init__.py`:

```
$SA3_PY scripts/probe.py                    → main checkout   ✗   (sys.path[0] is scripts/)
PYTHONPATH=<worktree> $SA3_PY scripts/probe.py → worktree     ✓
$SA3_PY -m scripts.probe                    → worktree        ✓   (cwd wins)
```

`PYTHONPATH` and the current directory both beat the `.pth` entry, which is why
`.claude/settings.local.json` pins `PYTHONPATH`. Rules that follow:

- Run things with `$SA3_PY`, e.g. `$SA3_PY -m pytest tests/...`.
- **Do not run `uv run` or `uv sync` from a worktree.** They would either build a
  second ~4.4G venv or repoint the shared editable install at the worktree and
  break imports in the main checkout. The `uv run` guidance elsewhere in the
  docs applies to the main checkout only.
- `sbatch/*.sbatch` hardcode `REPO=/data/hai-res/snnithya/stable-audio-3`, so a
  job submitted from a worktree runs the **main checkout's** code on the compute
  node. Check that before submitting from anywhere but the main checkout.

## Looking at an agent's branch

To read the changes you do not need to open the worktree at all:

```bash
git diff v/r...claude/<task-name>                       # full diff
git show claude/<task-name>:path/to/file.py             # one file at that branch
git difftool -d v/r...claude/<task-name>                # side-by-side
```

GitLens can also browse a branch's tree read-only. To edit in the worktree,
either open a second window (`code ../sa3-wt/<task-name>`) or — usually nicer —
add the folder to your current window (File > Add Folder to Workspace) so both
trees appear with separate Source Control sections.

`launch.json` is shared between the main checkout and every worktree, so its
debug configs resolve the interpreter with `${command:python.interpreterPath}`
rather than `${workspaceFolder}/.venv/bin/python` (which does not exist in a
worktree). If a worktree window shows unresolved imports, check that the
selected interpreter is the main checkout's `.venv`.

## Merging and cleanup

Agents commit on their branch and stop; merging is yours.

```bash
git merge claude/<task-name>          # or: git cherry-pick <sha>
git worktree remove ../sa3-wt/<task-name>
git branch -d claude/<task-name>
```

A worktree branch is based on the last **commit** of your branch, so
uncommitted work in the main checkout is invisible to the agent. Commit before
starting a task that builds on it.
