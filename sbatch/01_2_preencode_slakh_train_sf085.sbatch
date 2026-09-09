#!/bin/bash
#SBATCH --job-name=sg-preencode-train-sf085
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --account=hai-res
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/01_2_preencode_train_sf085_%j.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/01_2_preencode_train_sf085_%j.err

# Slakh train split at the chosen silence cutoff, --max_silence_fraction 0.85.
# Expect ~784 items per variant, 2 variants, from 1289 tracks.
#
# THE CUTOFF IS IN THE CONFIG, NOT HERE. slakh_streamgen_train_sf085.json carries
# "max_silence_fraction": 0.85, and this script deliberately passes no such flag: a CLI flag
# always wins over a config value, so having both would mean two sources of truth and a silent
# override the next time one of them is edited. The encode prints its resolved filter at
# startup ("Silence filter: RMS >= -50 dBFS and silent frames <= 85% ..."), so the job log
# still records exactly what ran. Change the cutoff in the config, and change it for BOTH
# splits or they stop being comparable.
#
# One GPU: pre_encode_dataset.py runs a single autoencoder on a single device
# (`next(ae.autoencoder.parameters()).device`), so the 2 and 4 GPUs the 2026-09-08 jobs asked
# for were idle beyond the first. Those jobs took 3m45s (validation, 270 items) and 27m
# (train, 1289 x 2 variants) of their 12h and 16h requests.
#
# Writes to a NEW root whose name states the cutoff -- see docs/data/slakh-streamgen.md. Do
# not point this at an existing root: this dataset family has already been silently redefined
# twice by encodes landing on a name that no longer described them.

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

OUT=/data/hai-res/shared/snnithya/sao-3/data/slakh-streamgen-preencoded-same-s-sf085/train

# --sample_size must be on the CLI: merge_config_into_args treats any argparse default that is
# not None as CLI-supplied, so a "sample_size" config key is silently ignored. 587853 samples
# is 13.3s / 144 latent frames from the START of every track (random_crop=False).
"$PYTHON" scripts/pre_encode_dataset.py \
    --dataset_config "stable_audio_3/configs/dataset_configs/dataset2preencoding/slakh_streamgen_train_sf085.json" \
    --model same-s \
    --batch_size 8 \
    --sample_size 587853 \
    --pad

# What the cutoff actually kept, in this job's log next to the encode that produced it.
cat "$OUT/_skipped.json"
"$PYTHON" scripts/analyze_silence_levels.py --dir "$OUT" --variant v0 \
    --max_silence_fraction 0.85 --json "$OUT/_levels.json"

echo "finished=$(date -Is)"
