#!/bin/bash
#SBATCH --job-name=sg-census-val
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --account=hai-res
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/01_2_preencode_validation_nofilter_%j.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/01_2_preencode_validation_nofilter_%j.err

# Level census of the Slakh validation split: the same encode as
# 01_2_preencode_slakh_validation.sbatch but with --no_silence_filter, so every item that
# reaches the level check gets a sidecar and its `levels` block can be read.
#
# The point is that a filtered pass cannot tell you where to put the filter. Levels are
# computed for every item but the sidecar is only written for survivors
# (pre_encode_dataset.py:478), so the 56 items the 2026-09-08 encode kept all sit below the
# 0.30 cutoff by construction -- max target silence_fraction 0.297 -- and say nothing about
# the 214 that went. This job produces the uncensored distribution to choose from.
#
# DIAGNOSTIC OUTPUT. It writes latents into its own root and nothing should train on them.
#
# One GPU, not two: pre_encode_dataset.py runs a single autoencoder on a single device
# (`next(ae.autoencoder.parameters()).device`), so the second L40S the filtered validation
# job asked for was idle. 12h was likewise inherited; that job encoded 56 of 270 items in
# 3m45s and this one encodes ~213, so 2h is generous.

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
# whole point of the clone -- PYTHONPATH is searched ahead of that .pth entry.
# uv itself lives only in the AFS home, which batch jobs cannot reach, so call the venv
# interpreter directly instead of going through `uv run`; with --no-sync --frozen that
# was all uv was doing anyway.
PYTHON="$REPO/.venv/bin/python"
export PYTHONPATH="$PWD"

CENSUS=/data/hai-res/shared/snnithya/sao-3/data/slakh-streamgen-nofilter-census-same-s/validation

# --sample_size / --pad / --batch_size must match the real encode or the census does not
# transfer: 587853 samples is 13.3 s / 144 latent frames from the START of every track
# (the pre-encode path sets random_crop=False), and a silence_fraction cutoff read at that
# length means something quite different at 380 s.
"$PYTHON" scripts/pre_encode_dataset.py \
    --dataset_config "stable_audio_3/configs/dataset_configs/dataset2preencoding/slakh_streamgen_validation_nofilter.json" \
    --model same-s \
    --batch_size 8 \
    --sample_size 587853 \
    --pad \
    --no_silence_filter

# The census is only useful read, so read it here -- the numbers land in this job's log
# next to the encode that produced them.
"$PYTHON" scripts/analyze_silence_levels.py --dir "$CENSUS" \
    --json "$CENSUS/_levels_census.json"

echo "finished=$(date -Is)"
