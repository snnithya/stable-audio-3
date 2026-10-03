#!/bin/bash
#SBATCH --job-name=wjd-3-3-ft
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --account=hai-res
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/03_3_finetune_wjd_%A_%a.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/03_3_finetune_wjd_%A_%a.out
#
# Experiment 3.3 -- first WJD finetune: drum RMS envelope as the control, prompt picks the stem.
# Plan: experiments/03-wjd-jazz-stems/03-dataset-and-wiring.md.
#
#   task 0  rms   -- small_music_wjd_drums_rms.json: 1-ch causal drum RMS as a modular local cond
#   task 1  base  -- small_music_baseline.json: same model with no local cond (text + causal
#                    inpainting only), the matched baseline as in 1.2. Not run by default.
#
#   sbatch sbatch/03_3_finetune_wjd.sbatch                # the rms run
#   sbatch --array=0-1 sbatch/03_3_finetune_wjd.sbatch    # rms + baseline pair
#
# Mirrors 01_2_finetune.sbatch: same pretrained small-music, same causal inpainting task,
# same optimizer, same seed. Items are whole-track latents cropped to 144 frames (13.3 s)
# at a random offset each epoch; each item's prompt is its stem's instrument name(s)
# ("upright bass", "piano", "trumpet, tenor saxophone", ...). The text conditioner is
# frozen: the prompts vary here, unlike Slakh's constant "drums", but T5Gemma already
# separates instrument names and the DiT's cross-attention is what has to learn to use
# them. Drop --freeze_conditioner to test the alternative.
#
# Prerequisite: 03_3_preencode_wjd.sbatch has finished for the train split.

set -euo pipefail

export HF_HOME=/data/hai-res/snnithya/.cache/huggingface

# This experiment lives on tap-dance-expt in the sa3-tap checkout, not on v/r.
REPO=/data/hai-res/snnithya/sa3-tap/stable-audio-3
branch=tap-dance-expt
SAVE_ROOT=/data/scratch-fast/snnithya/sao-3/ft_checkpoints/03-3-wjd-drums-rms
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

ARMS=(rms base)
CONFIGS=(
    stable_audio_3/configs/model_configs/small_music_wjd_drums_rms.json
    stable_audio_3/configs/model_configs/small_music_baseline.json
)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
MODEL_CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "host=$(hostname) arm=${ARM} config=${MODEL_CONFIG} ref=${SAO_REF} started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PYTHON" scripts/train_finetune.py \
    --model small-music \
    --model_config "${MODEL_CONFIG}" \
    --dataset_config stable_audio_3/configs/dataset_configs/preencoded/wjd_stems_train_preencoded.json \
    --steps 20000 \
    --batch_size 8 \
    --accum_batches 2 \
    --lr 1e-5 \
    --seed 42 \
    --freeze_conditioner \
    --num_workers 12 \
    --checkpoint_every 5000 \
    --demo_every 2000 \
    --log_every 50 \
    --export_safetensors \
    --logger wandb \
    --project sao-3 \
    --group 03-3-wjd-drums-rms \
    --name "wjd-${ARM}" \
    --save_dir "${SAVE_ROOT}/${ARM}"

echo "finished=$(date -Is)"
