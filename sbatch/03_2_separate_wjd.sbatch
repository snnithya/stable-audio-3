#!/bin/bash
#SBATCH --job-name=wjd-sep
#SBATCH --partition=hai-res-l40s
#SBATCH --account=hai-res
#SBATCH --qos=hai-res-low
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/03_2_separate_wjd_%j.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/03_2_separate_wjd_%j.out

# Experiment 3.2: source-separate the aligned WJD recordings.
# Plan: experiments/03-wjd-jazz-stems/02-source-separation.md.
#
# One model per job. Default is BS-Roformer-SW (6 stems: drums, bass, other, vocals,
# piano, guitar) over every `status: pass` track in audio/manifest.csv. Output:
#   /data/hai-res/shared/snnithya/sao-3/data/wjd/stems/<model_name>/<Track>/{stem}.flac + done.json
#
# Preemptible + restart-safe. `hai-res-low` runs on idle lab GPUs and is bumped by any
# hai-res-main/-extended/-debug job (PreemptType=preempt/qos, PreemptMode=REQUEUE,
# GraceTime=0, KillWait=10 s -- i.e. SIGTERM then SIGKILL 10 s later, no time to finish a
# track). `--requeue` puts the job back in the queue with the same job id; `--open-mode=append`
# keeps one log across restarts. The runner tolerates a kill at any instant: a track is
# only ever marked done by an atomic rename of done.json, a half-written stem dir is wiped
# and redone on the next start, and an `_inprogress.json` lease keeps two workers (e.g. this
# job plus one in an interactive shell, or a requeued job restarting while the old lease is
# still fresh) off the same track. Use `--qos=hai-res-main` for a non-preemptible run.
#
# Resumable (tracks with done.json are skipped), so re-submitting after a time-out or
# a failed track just fills the gaps. Knobs:
#   SEP_MODEL=BS-Roformer-SW.ckpt      audio-separator model filename (must be in models.json,
#                                      or already present in $MODEL_DIR with its yaml)
#   SEP_ARGS="--limit 3"               anything separate.py accepts, e.g. "--status pass check",
#                                      "--tracks A B", "--overlap 4", "--force"
# Smoke test:  SEP_ARGS="--limit 1" sbatch sbatch/03_2_separate_wjd.sbatch
#
# Env: the `sep` conda env (audio-separator[gpu]==0.47.0), NOT the project venv --
# see the experiment file for how it was made. The script is standalone and runs straight
# from the checkout below (uncommitted edits ARE what runs; deliberate, like 03_1).

set -euo pipefail

REPO=/data/hai-res/snnithya/sa3-tap/stable-audio-3
PY=/data/hai-res/snnithya/miniconda3/envs/sep/bin/python
WJD_ROOT=/data/hai-res/shared/snnithya/sao-3/data/wjd
MODEL_DIR=/data/hai-res/shared/snnithya/sao-3/models/sep

# Keep every cache off the over-quota AFS home.
export XDG_CACHE_HOME=/data/hai-res/snnithya/.cache
export TORCH_HOME=/data/hai-res/snnithya/.cache/torch
export HF_HOME=/data/hai-res/snnithya/.cache/huggingface
export AUDIO_SEPARATOR_MODEL_DIR=$MODEL_DIR

mkdir -p /data/scratch-fast/snnithya/sao-3/logs "$MODEL_DIR"
echo "=== job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) start=$(date -Is) commit=$(git -C "$REPO" rev-parse --short HEAD) model=${SEP_MODEL:-BS-Roformer-SW.ckpt} qos=${SLURM_JOB_QOS:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXTRA=()
[ -n "${SEP_ARGS:-}" ] && EXTRA+=($SEP_ARGS)

# Forward SIGTERM (preemption) to the runner so it can log and release its lease before
# SIGKILL lands; `wait` returns the runner's exit status either way.
"$PY" "$REPO/scripts/wjd/separate.py" \
    --wjd_root "$WJD_ROOT" \
    --model_file_dir "$MODEL_DIR" \
    --model "${SEP_MODEL:-BS-Roformer-SW.ckpt}" \
    "${EXTRA[@]}" &
RUNNER=$!
trap 'echo "SIGTERM at $(date -Is) (preempted or cancelled), forwarding to runner"; kill -TERM $RUNNER 2>/dev/null' TERM
wait $RUNNER
STATUS=$?

echo "=== runner exit=$STATUS finished=$(date -Is)"
exit $STATUS
