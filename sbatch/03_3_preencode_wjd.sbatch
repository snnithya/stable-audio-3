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
# Window (decided 2026-10-02): CHUNKS. Every track is cut into CHUNK_SECONDS windows (default
# 12 s, which the script rounds up to whole latent frames: 130 frames = 12.07 s) every
# CHUNK_HOP_RATIO of a window (default 0.5, 50 % overlap); one item per window, silence-gated
# per window, nothing capped. Output goes to .../wjd/preencoded-chunked/<split>/ and the
# matching training config is preencoded/wjd_stems_<split>_chunked_preencoded.json, whose
# latent_crop_length must equal the window in frames (130 for 12 s) -- change both together.
# Both knobs are plain script flags (--chunk_seconds, --chunk_hop_ratio); the env vars here
# only set them:  CHUNK_SECONDS=20 CHUNK_HOP_RATIO=0.25 sbatch sbatch/03_3_preencode_wjd.sbatch
# CHUNK_SECONDS=0 restores the
# earlier whole-track encode (first 380 s of each track, --sample_size 16760832, --batch_size 1,
# no --pad; output .../wjd/preencoded/<split>/, config wjd_stems_<split>_preencoded.json).
#
# The work is sharded over 4 GPUs: --gres allocates 4 to the job, --ntasks=4 sizes the CPU
# allocation (cpus-per-task is per task), and `srun --ntasks=4 --gpus-per-task=1` starts one
# process per GPU. Each process reads its shard from the srun step (SLURM_STEP_NUM_TASKS /
# SLURM_PROCID) and takes every 4th global batch, so the output is byte-for-byte what one GPU
# would have written. Per-task logs go to <jobid>_<task>.out next to this job's log.
#
# The feature controls are sliced from per-song files (tracks/drums/<Track>/drums_features.npz,
# scripts/wjd/compute_drum_features.py, run below on the job's CPUs before the encode; existing
# files are kept). WJD_DRUM_FEATURES=audio computes them from the window's drum audio instead.
#
# WJD_CONTROL_MODE picks what custom_md_wjd.py returns, as a comma-separated set: rms
# (--features drums_rms), tria_fixed / tria_ema (--features drums_tria_*, experiment 3.4's
# causal TRIA controls, which need configs/dataset_configs/features/wjd_drums_tria_stats.json),
# audio (--controls drums_audio, the VAE-latent condition as in Slakh); 'both' = rms,audio.
# Default: rms,tria_fixed,tria_ema, the layout preencoded/wjd_stems_*_preencoded.json expects.

set -euo pipefail

export HF_HOME=/data/hai-res/snnithya/.cache/huggingface

SPLIT=${SPLIT:-train}
MODEL=${MODEL:-same-s}
CHUNK_SECONDS=${CHUNK_SECONDS:-12}
CHUNK_HOP_RATIO=${CHUNK_HOP_RATIO:-0.5}
export WJD_CONTROL_MODE=${WJD_CONTROL_MODE:-rms,tria_fixed,tria_ema}
export WJD_DRUM_FEATURES=${WJD_DRUM_FEATURES:-precomputed}
MIRROR=/data/hai-res/shared/snnithya/sao-3/data/wjd/wjd-stem-mirror

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

# Map the mode set onto the pre-encode flags: audio controls first, then features, which is
# also the channel order of the fused sidecar.
AUDIO_CTRLS=()
FEATS=()
IFS=',' read -ra MODES <<< "$WJD_CONTROL_MODE"
for m in "${MODES[@]}"; do
  case "$m" in
    audio) AUDIO_CTRLS+=(drums_audio) ;;
    both)  AUDIO_CTRLS+=(drums_audio); FEATS+=(drums_rms) ;;
    rms|tria_fixed|tria_ema) FEATS+=("drums_${m}") ;;
    *) echo "bad WJD_CONTROL_MODE entry '$m' in '$WJD_CONTROL_MODE'"; exit 2 ;;
  esac
done
CTRL=()
[ ${#AUDIO_CTRLS[@]} -gt 0 ] && CTRL+=(--controls "${AUDIO_CTRLS[@]}")
[ ${#FEATS[@]} -gt 0 ] && CTRL+=(--features "${FEATS[@]}")
echo "WJD_CONTROL_MODE=${WJD_CONTROL_MODE} -> ${CTRL[*]}"

if [ "$WJD_DRUM_FEATURES" = "precomputed" ]; then
    "$PYTHON" scripts/wjd/compute_drum_features.py --mirror "$MIRROR" --split "$SPLIT" --workers 8
fi

if [ "$CHUNK_SECONDS" != "0" ]; then
    WINDOW=(--chunk_seconds "$CHUNK_SECONDS" --chunk_hop_ratio "$CHUNK_HOP_RATIO" --pad --batch_size 8
            --output_path "/data/hai-res/shared/snnithya/sao-3/data/wjd/preencoded-chunked/${SPLIT}/")
else
    WINDOW=(--batch_size 1 --sample_size 16760832)
fi
echo "window: ${WINDOW[*]}"

srun --ntasks=4 --gpus-per-task=1 \
     --output=/data/scratch-fast/snnithya/sao-3/logs/03_3_preencode_wjd_%j_%t.out \
     "$PYTHON" scripts/pre_encode_dataset.py \
    --dataset_config "stable_audio_3/configs/dataset_configs/dataset2preencoding/wjd_stems_${SPLIT}.json" \
    --model "$MODEL" \
    "${WINDOW[@]}" \
    "${CTRL[@]}"

echo "finished=$(date -Is)"
