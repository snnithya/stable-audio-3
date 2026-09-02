#!/bin/bash
#SBATCH --job-name=sg-preencode-val
#SBATCH --partition=hai-res-l40s
#SBATCH --account=hai-res
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/01_2_preencode_validation.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/01_2_preencode_validation.out


set -euo pipefail

# The Stable Audio 3 repos are gated; the compute nodes reach them only through this
# already-populated cache, so set it before anything imports huggingface_hub.
export HF_HOME=/data/hai-res/snnithya/.cache/huggingface


REPO=/data/hai-res/snnithya/stable-audio-3
branch=v/r
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

# The 4.4G venv is not worth rebuilding per job, so the clone borrows the one in $REPO.
# The project is installed into it *editable*, pointing back at $REPO, so without the
# PYTHONPATH below `import stable_audio_3` would resolve to the live tree and undo the
# whole point of the clone -- PYTHONPATH is searched ahead of that .pth entry. --no-sync
# keeps uv from rewriting the shared venv to point at this throwaway directory instead.
export UV_PROJECT_ENVIRONMENT="$REPO/.venv"
export PYTHONPATH="$PWD"


uv run --no-sync --frozen python scripts/pre_encode_dataset.py \
    --dataset_config "stable_audio_3/configs/dataset_configs/dataset2preencoding/slakh_streamgen_validation.json" \
    --model same-s \
    --batch_size 8 \
    --sample_size 587853

echo "finished=$(date -Is)"
