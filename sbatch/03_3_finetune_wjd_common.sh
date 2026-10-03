#!/bin/bash
# Shared body of the WJD finetune jobs (experiments 3.3 / 3.4). Not submitted directly:
# sbatch/03_3_finetune_wjd_<arm>.sbatch sets ARM, MODEL_CONFIG and GROUP and sources this
# file by its absolute path (Slurm runs a copy of the batch script from its spool dir, so
# $0 / BASH_SOURCE would not point here). One script per arm so each can carry its own
# #SBATCH --qos / partition / time; everything the arms share lives here once.
#
# Mirrors 01_2_finetune.sbatch: same pretrained small-music, same causal inpainting task,
# same optimizer, same seed. Items are 12 s chunks (130 frames) from the chunked encode
# (DATASET_CONFIG overrides; .../wjd_stems_train_preencoded.json is the earlier whole-track
# one, cropped to 144 frames at a random offset). Each item's prompt is its stem's
# instrument name(s) ("upright bass", "piano", "trumpet, tenor saxophone", ...). The text
# conditioner is frozen: the prompts vary here, unlike Slakh's constant "drums", but
# T5Gemma already separates instrument names and the DiT's cross-attention is what has to
# learn to use them. Drop --freeze_conditioner to test the alternative.
#
# All arms read the same pre-encoded items; the sidecar carries every control (encode with
# the sbatch default WJD_CONTROL_MODE=audio,rms,tria_fixed,tria_ema) and the model config
# names the one its arm uses. Every control is gated by the teacher-forcing mask exactly as
# the Slakh streamgen latent is (visible in the context and up to the lookahead, hidden
# beyond), see DiffusionCondTrainingWrapper._add_streamgen_conditioning.
#
# Prerequisite: 03_3_preencode_wjd.sbatch has finished for the train split.

set -euo pipefail

export HF_HOME=/data/hai-res/snnithya/.cache/huggingface

# This experiment lives on tap-dance-expt in the sa3-tap checkout, not on v/r.
REPO=/data/hai-res/snnithya/sa3-tap/stable-audio-3
branch=tap-dance-expt
SAVE_BASE=/data/scratch-fast/snnithya/sao-3/ft_checkpoints
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

# Chunked encode by default (2026-10-02); DATASET_CONFIG=.../wjd_stems_train_preencoded.json
# selects the earlier whole-track one.
DATASET_CONFIG=${DATASET_CONFIG:-stable_audio_3/configs/dataset_configs/preencoded/wjd_stems_train_chunked_preencoded.json}

: "${ARM:?set by the per-arm sbatch script}" "${MODEL_CONFIG:?}" "${GROUP:?}"
SAVE_ROOT=${SAVE_BASE}/${GROUP}

echo "host=$(hostname) arm=${ARM} config=${MODEL_CONFIG} ref=${SAO_REF} started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PYTHON" scripts/train_finetune.py \
    --model small-music \
    --model_config "${MODEL_CONFIG}" \
    --dataset_config "${DATASET_CONFIG}" \
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
    --group "${GROUP}" \
    --name "wjd-${ARM}" \
    --save_dir "${SAVE_ROOT}/${ARM}"

echo "finished=$(date -Is)"
