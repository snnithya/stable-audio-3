#!/bin/bash
#SBATCH --job-name=sg-1-2-eval
#SBATCH --partition=hai-res-l40s
#SBATCH --account=hai-res
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/01_2_eval_%j.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/01_2_eval_%j.out
#
# Experiment 01, sub-experiment 1.2 -- step 3 of 3.
#
# Scores both finetuned arms on the held-out Slakh validation split: paired denoising
# loss at fixed timesteps and fixed noise, the zero/shuffle accompaniment ablations, and
# onset alignment of generated drums against the accompaniment they were given.
#
# Point CKPT_COND / CKPT_BASE at the checkpoints you want to compare -- pass them in the
# environment to score an intermediate step rather than the last one:
#   sbatch --export=ALL,CKPT_COND=...,CKPT_BASE=... sbatch/01_2_eval.sbatch

set -euo pipefail

# The Stable Audio 3 repos are gated; the compute nodes reach them only through this
# already-populated cache, so set it before anything imports huggingface_hub.
export HF_HOME=/data/hai-res/snnithya/.cache/huggingface

REPO=/data/hai-res/snnithya/stable-audio-3
SAVE_ROOT=/data/scratch-fast/snnithya/sao-3/ft_checkpoints/01-2-streamgen-baseline
RESULTS=/data/hai-res/snnithya/stable-audio-3/experiments/01-streamgen-conditioning/results
mkdir -p /data/scratch-fast/snnithya/sao-3/logs "$RESULTS"
cd "$REPO"

# With the wandb logger the checkpoints land under <save_dir>/<project>/<run_id>/checkpoints/,
# so search rather than guess the run id.
latest_ckpt() {
    find "$1" -name last.ckpt -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
}
CKPT_COND=${CKPT_COND:-$(latest_ckpt "${SAVE_ROOT}/cond")}
CKPT_BASE=${CKPT_BASE:-$(latest_ckpt "${SAVE_ROOT}/base")}

if [[ -z "${CKPT_COND}" || -z "${CKPT_BASE}" ]]; then
    echo "Could not locate both checkpoints under ${SAVE_ROOT}; pass CKPT_COND and CKPT_BASE explicitly." >&2
    exit 1
fi

echo "cond=${CKPT_COND}"
echo "base=${CKPT_BASE}"

uv run python scripts/eval_streamgen.py \
    --arm cond stable_audio_3/configs/model_configs/small_music_streamgen.json "${CKPT_COND}" \
    --arm base stable_audio_3/configs/model_configs/small_music_baseline.json "${CKPT_BASE}" \
    --dataset_config stable_audio_3/configs/dataset_configs/preencoded/slakh_streamgen_validation_preencoded.json \
    --max_items 64 \
    --batch_size 4 \
    --context_seconds 8.0 \
    --lookahead_seconds 0.0 \
    --num_generate 12 \
    --gen_steps 50 \
    --cfg_scale 1.0 \
    --out "${RESULTS}/1_2.json" \
    --audio_dir "${RESULTS}/audio"

echo "finished=$(date -Is)"
