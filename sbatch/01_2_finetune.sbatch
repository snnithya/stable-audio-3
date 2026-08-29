#!/bin/bash
#SBATCH --job-name=sg-1-2-ft
#SBATCH --partition=hai-res-l40s
#SBATCH --account=hai-res
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/01_2_finetune_%A_%a.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/01_2_finetune_%A_%a.out
#
# Experiment 01, sub-experiment 1.2 -- step 2 of 3.
#
# The matched pair the whole sub-experiment rests on:
#   task 0  cond  -- small_music_streamgen.json, accompaniment as a modular local cond
#   task 1  base  -- small_music_baseline.json, the same config with those conds removed
#
# Everything else is held identical on purpose: same pretrained checkpoint, same drum
# target, same causal inpainting task, same optimizer, same seed, same data order. The
# only difference between the two runs is whether the accompaniment latent reaches the
# model, which is what makes the difference in held-out loss attributable to it.
#
# The text conditioner is frozen in both arms. Every item's prompt is the constant string
# "drums", so there is nothing for the text encoder to learn, and training it would only
# add a second moving part that differs between runs.
#
# Prerequisite: 01_2_preencode_slakh.sbatch has finished and the alignment check passed.
#
#   sbatch sbatch/01_2_finetune.sbatch
#   sbatch --array=0 sbatch/01_2_finetune.sbatch   # just the conditioned arm

set -euo pipefail

# The Stable Audio 3 repos are gated; the compute nodes reach them only through this
# already-populated cache, so set it before anything imports huggingface_hub.
export HF_HOME=/data/hai-res/snnithya/.cache/huggingface

REPO=/data/hai-res/snnithya/stable-audio-3
SAVE_ROOT=/data/scratch-fast/snnithya/sao-3/ft_checkpoints/01-2-streamgen-baseline
mkdir -p /data/scratch-fast/snnithya/sao-3/logs
cd "$REPO"

ARMS=(cond base)
CONFIGS=(
    stable_audio_3/configs/model_configs/small_music_streamgen.json
    stable_audio_3/configs/model_configs/small_music_baseline.json
)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
MODEL_CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "host=$(hostname) arm=${ARM} config=${MODEL_CONFIG} started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD

uv run python scripts/train_finetune.py \
    --model small-music \
    --model_config "${MODEL_CONFIG}" \
    --dataset_config stable_audio_3/configs/dataset_configs/preencoded/slakh_streamgen_train_preencoded.json \
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
    --group 01-2-streamgen-baseline \
    --name "streamgen-${ARM}" \
    --save_dir "${SAVE_ROOT}/${ARM}"

echo "finished=$(date -Is)"
