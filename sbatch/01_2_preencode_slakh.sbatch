#!/bin/bash
#SBATCH --job-name=sg-preencode
#SBATCH --partition=hai-res-l40s
#SBATCH --account=hai-res
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/01_2_preencode_%A_%a.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/01_2_preencode_%A_%a.out
#
# Experiment 01, sub-experiment 1.2 -- step 1 of 3.
#
# Pre-encodes the Slakh2100 train and validation splits into drum latents plus the
# accompaniment control sidecar. Array task 0 does train (1289 tracks, ~89h of audio),
# task 1 does validation (270 tracks).
#
# Runs at batch_size 1 with no --pad, so each track keeps its natural length. This is
# also what keeps the target and its control on the same crop: pre_encode_dataset.py
# builds the dataset with random_crop=False for exactly that reason.
#
#   sbatch sbatch/01_2_preencode_slakh.sbatch
#
# After it finishes, ALWAYS verify alignment before training on the result:
#   uv run python scripts/check_streamgen_alignment.py \
#       --config stable_audio_3/configs/dataset_configs/preencoded/slakh_streamgen_validation_preencoded.json

set -euo pipefail

# The Stable Audio 3 repos are gated; the compute nodes reach them only through this
# already-populated cache, so set it before anything imports huggingface_hub.
export HF_HOME=/data/hai-res/snnithya/.cache/huggingface

REPO=/data/hai-res/snnithya/stable-audio-3
mkdir -p /data/scratch-fast/snnithya/sao-3/logs
cd "$REPO"

SPLITS=(train validation)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

echo "host=$(hostname) split=${SPLIT} started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --sample_size 16760832 = 4092 latent frames = 380s, which covers the 95th percentile
# Slakh track; longer tracks are truncated. It has to be passed here rather than set in the
# config: merge_config_into_args treats any argparse default that is not None as
# "CLI-supplied", so a sample_size key in the JSON would be silently ignored.
uv run python scripts/pre_encode_dataset.py \
    --dataset_config "stable_audio_3/configs/dataset_configs/dataset2preencoding/slakh_streamgen_${SPLIT}.json" \
    --model same-l \
    --batch_size 1 \
    --sample_size 16760832

echo "finished=$(date -Is)"
