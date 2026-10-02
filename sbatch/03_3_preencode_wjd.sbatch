#!/bin/bash
#SBATCH --job-name=wjd-preencode
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --account=hai-res
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/03_3_preencode_wjd_%j.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/03_3_preencode_wjd_%j.out

# Experiment 3.3: pre-encode the WJD stem mirror with the drum RMS envelope as the control.
# Plan: experiments/03-wjd-jazz-stems/03-dataset-and-wiring.md.
#
#   sbatch sbatch/03_3_preencode_wjd.sbatch                 # train split
#   SPLIT=validation sbatch sbatch/03_3_preencode_wjd.sbatch
#
# Whole tracks (131-901 s): --sample_size 16760832 is the 380 s cap used for whole-track Slakh
# encodes, --batch_size 1 and no --pad so each latent is stored at its own length and the
# training config crops it (the script refuses batch_size > 1 without --pad). ~500 items x
# ~5 min at ~10x realtime is a couple of GPU-hours single-GPU, so the work is sharded over
# 4 GPUs: --gres allocates 4 to the job, --ntasks=4 sizes the CPU allocation (cpus-per-task
# is per task), and `srun --ntasks=4 --gpus-per-task=1` starts one process per GPU. Each
# process reads its shard from the srun step (SLURM_STEP_NUM_TASKS / SLURM_PROCID) and takes
# every 4th global batch, so the output is byte-for-byte what one GPU would have written.
# Per-task logs go to <jobid>_<task>.out next to this job's log.
#
# WJD_CONTROL_MODE picks what custom_md_wjd.py returns: rms (default, --features drums_rms),
# audio (--controls drums_audio, the VAE-latent condition as in Slakh), or both.

set -euo pipefail

export HF_HOME=/data/hai-res/snnithya/.cache/huggingface

SPLIT=${SPLIT:-train}
MODEL=${MODEL:-same-s}
export WJD_CONTROL_MODE=${WJD_CONTROL_MODE:-rms}

# This experiment lives on tap-dance-expt in the sa3-tap checkout, not on v/r.
REPO=/data/hai-res/snnithya/sa3-tap/stable-audio-3
branch=tap-dance-expt
mkdir -p /data/scratch-fast/snnithya/sao-3/logs

SAO_REF=${SAO_REF:-$(git -C "$REPO" rev-parse HEAD)}
if ! git -C "$REPO" diff-index --quiet HEAD --; then
    echo "WARNING: ${REPO} has uncommitted changes; this job runs ${SAO_REF} without them."
fi

WORKDIR=/tmp/home/snnithya/slurm-${SLURM_JOB_ID:-$$}
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT
git clone -q -b "$branch" "$REPO/.git" "$WORKDIR/stable-audio-3"
git -C "$WORKDIR/stable-audio-3" checkout -q --detach "$SAO_REF"
cd "$WORKDIR/stable-audio-3"

# Borrow the checkout's venv; PYTHONPATH puts the clone ahead of its editable install.
PYTHON="$REPO/.venv/bin/python"
export PYTHONPATH="$PWD"

case "$WJD_CONTROL_MODE" in
  rms)   CTRL=(--features drums_rms) ;;
  audio) CTRL=(--controls drums_audio) ;;
  both)  CTRL=(--controls drums_audio --features drums_rms) ;;
  *) echo "bad WJD_CONTROL_MODE=$WJD_CONTROL_MODE"; exit 2 ;;
esac

srun --ntasks=4 --gpus-per-task=1 \
     --output=/data/scratch-fast/snnithya/sao-3/logs/03_3_preencode_wjd_%j_%t.out \
     "$PYTHON" scripts/pre_encode_dataset.py \
    --dataset_config "stable_audio_3/configs/dataset_configs/dataset2preencoding/wjd_stems_${SPLIT}.json" \
    --model "$MODEL" \
    --batch_size 1 \
    --sample_size 16760832 \
    "${CTRL[@]}"

echo "finished=$(date -Is)"
