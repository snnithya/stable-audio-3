#!/bin/bash
# Bootstrap of the WJD finetune jobs (experiments 3.3 / 3.4). Not submitted directly:
# sbatch/03_3_finetune_wjd_<arm>.sbatch sets ARM, MODEL_CONFIG and GROUP and sources this
# file by its absolute path (Slurm runs a copy of the batch script from its spool dir, so
# $0 / BASH_SOURCE would not point here). One script per arm so each can carry its own
# #SBATCH --qos / partition / time.
#
# This file is read from the live checkout because it is what picks the run to continue,
# the git ref to pin, and makes the clone; keep it to that. The training flags live in
# sbatch/03_3_finetune_wjd_train.sh, which is sourced from the pinned clone, so edits to
# the flags take effect only once committed, like the rest of the code.
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

# Fail before cloning anything if a per-arm script forgot to set these.
: "${ARM:?set by the per-arm sbatch script}" "${MODEL_CONFIG:?}" "${GROUP:?}"

# This experiment lives on tap-dance-expt in the sa3-tap checkout, not on v/r.
REPO=/data/hai-res/snnithya/sa3-tap/stable-audio-3
branch=tap-dance-expt

# Secrets and wandb dirs (WANDB_API_KEY, WANDB_DIR, ...) live in the checkout's gitignored
# .env, as 03_1_fetch_wjd_youtube.sbatch does. A batch job has no interactive login and
# ~/.netrc sits on AFS, which the compute nodes do not read, so without this wandb.init
# dies with "No API key configured" (jobs 2534778-80, 2026-10-03). set -a exports every
# assignment so the Python process inherits them.
set -a
source "$REPO/.env"
set +a
export HF_HOME=/data/hai-res/snnithya/.cache/huggingface
SAVE_BASE=/data/scratch-fast/snnithya/sao-3/ft_checkpoints
mkdir -p /data/scratch-fast/snnithya/sao-3/logs

# Restart on preemption. The partition preempts by QoS with PreemptMode=REQUEUE and
# GraceTime=0: a preempted job gets SIGTERM, SIGKILL 10 s later, and Slurm requeues it
# under the SAME job id (the arm headers set --requeue and --open-mode=append). Nothing can
# be checkpointed in those 10 s, so the restart resumes from the last.ckpt that
# train_finetune.py refreshes every --resume_every steps. Three things make that work:
#  * the wandb run id is derived from the job id, so the requeued job resumes the same
#    wandb run and lands in the same checkpoint dir (<save_dir>/<project>/<run id>/checkpoints);
#  * the git ref is recorded next to the run on first start and reused, so a requeue does
#    not pick up commits made in the meantime;
#  * --resume_ckpt is passed whenever that dir already holds a last.ckpt.
# A plain resubmit (e.g. after the 24 h MaxWall of hai-res-main) continues too: with no
# WANDB_RUN_ID given, the arm's most recent run that already has a last.ckpt is reused, so
# nothing starts over by accident (jobs 2539647-49, 2026-10-03, did exactly that). To pick
# a specific run:  WANDB_RUN_ID=wjd-rms-2534900 sbatch sbatch/03_3_finetune_wjd_rms.sbatch
# To start a new run next to existing ones:  FRESH=1 sbatch sbatch/03_3_finetune_wjd_rms.sbatch
WANDB_PROJECT_NAME=sao-3
ARM_DIR=${SAVE_BASE}/${GROUP}/${ARM}/${WANDB_PROJECT_NAME}
if [ -z "${WANDB_RUN_ID:-}" ] && [ "${FRESH:-0}" != 1 ]; then
    # `|| true`: with no match ls fails and pipefail + set -e would abort the job here.
    latest_ckpt=$(ls -t "$ARM_DIR"/*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)
    if [ -n "$latest_ckpt" ]; then
        WANDB_RUN_ID=$(basename "$(dirname "$(dirname "$latest_ckpt")")")
        echo "no WANDB_RUN_ID given: continuing the arm's latest run ${WANDB_RUN_ID} (FRESH=1 to start over)"
    fi
fi
export WANDB_RUN_ID=${WANDB_RUN_ID:-wjd-${ARM}-${SLURM_JOB_ID:-local}}
export WANDB_RESUME=allow
RUN_DIR=${ARM_DIR}/${WANDB_RUN_ID}
mkdir -p "$RUN_DIR"
REF_FILE=$RUN_DIR/sao_ref
if [ -f "$REF_FILE" ]; then
    SAO_REF=$(cat "$REF_FILE")
    echo "requeue/continuation of ${WANDB_RUN_ID}: pinned to ${SAO_REF} from ${REF_FILE}"
else
    SAO_REF=${SAO_REF:-$(git -C "$REPO" rev-parse HEAD)}
    echo "$SAO_REF" > "$REF_FILE"
fi
RESUME=()
if [ -f "$RUN_DIR/checkpoints/last.ckpt" ]; then
    RESUME=(--resume_ckpt "$RUN_DIR/checkpoints/last.ckpt")
    echo "resuming from $RUN_DIR/checkpoints/last.ckpt ($(date -r "$RUN_DIR/checkpoints/last.ckpt" -Is))"
fi
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

# From here on, run the sbatch logic of the pinned commit, not of the live checkout: the
# training flags live in sbatch/03_3_finetune_wjd_train.sh inside the clone. Runs pinned to
# a commit older than that file fall back to the checkout's copy.
TRAIN_SH=$PWD/sbatch/03_3_finetune_wjd_train.sh
if [ ! -f "$TRAIN_SH" ]; then
    TRAIN_SH=$REPO/sbatch/03_3_finetune_wjd_train.sh
    echo "NOTE: ${SAO_REF} predates sbatch/03_3_finetune_wjd_train.sh; using the checkout's copy."
fi
source "$TRAIN_SH"
