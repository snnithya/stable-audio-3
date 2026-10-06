#!/bin/bash
# Training body of the WJD finetune jobs (experiments 3.3 / 3.4). Sourced by
# sbatch/03_3_finetune_wjd_common.sh from the PINNED CLONE (cwd), after it has resolved the
# run to continue, recorded/reused the git ref, cloned it and set PYTHON / PYTHONPATH. So the
# training flags here are versioned with the code they run, and a requeue or continuation
# keeps the flags of the commit the run started on. Expects: ARM, MODEL_CONFIG, GROUP,
# SAVE_BASE, WANDB_PROJECT_NAME, SAO_REF, RESUME (array), PYTHON.

# Chunked encode by default (2026-10-02); DATASET_CONFIG=.../wjd_stems_train_preencoded.json
# selects the earlier whole-track one.
DATASET_CONFIG=${DATASET_CONFIG:-stable_audio_3/configs/dataset_configs/preencoded/wjd_stems_train_chunked_preencoded.json}

SAVE_ROOT=${SAVE_BASE}/${GROUP}

# Effective batch (samples per optimizer step) is fixed at EFFECTIVE_BATCH; the per-GPU
# batch defaults to EFFECTIVE_BATCH / NGPU with no gradient accumulation, i.e. the whole
# step is one forward/backward spread over the GPUs the arm's header requested (items are
# 130 latent frames, so a large per-GPU batch fits). --batch_size is PER GPU under Lightning
# DDP, which is why it is derived here rather than written once for every GPU count.
# If a per-GPU batch of EFFECTIVE_BATCH / NGPU runs out of memory, force a smaller one and
# the difference is made up by accumulation:  MICRO_BATCH=32 sbatch sbatch/03_3_finetune_wjd_rms.sbatch
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-768}
NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
if (( EFFECTIVE_BATCH % NGPU != 0 )); then
    echo "EFFECTIVE_BATCH=${EFFECTIVE_BATCH} is not a multiple of NGPU=${NGPU}" >&2
    exit 2
fi
MICRO_BATCH=${MICRO_BATCH:-$(( EFFECTIVE_BATCH / NGPU ))}
if (( EFFECTIVE_BATCH % (MICRO_BATCH * NGPU) != 0 )); then
    echo "EFFECTIVE_BATCH=${EFFECTIVE_BATCH} is not a multiple of MICRO_BATCH x NGPU = ${MICRO_BATCH} x ${NGPU}" >&2
    exit 2
fi
ACCUM=$(( EFFECTIVE_BATCH / (MICRO_BATCH * NGPU) ))

echo "host=$(hostname) arm=${ARM} config=${MODEL_CONFIG} ref=${SAO_REF} started=$(date -Is)"
echo "gpus=${NGPU} micro_batch=${MICRO_BATCH} accum=${ACCUM} -> effective batch $(( MICRO_BATCH * NGPU * ACCUM )) per optimizer step"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PYTHON" scripts/train_finetune.py \
    --model small-music-base \
    --model_config "${MODEL_CONFIG}" \
    --dataset_config "${DATASET_CONFIG}" \
    --steps 100000 \
    --batch_size "${MICRO_BATCH}" \
    --accum_batches "${ACCUM}" \
    --lr 1e-5 \
    --lr_schedule inverse --lr_inv_gamma 1000000 --lr_power 0.5 --lr_warmup_decay 0.995 \
    --seed 42 \
    --freeze_conditioner \
    --num_workers 12 \
    --checkpoint_every 5000 \
    --demo_every 1000 \
    --log_every 50 \
    --logger wandb \
    --project "${WANDB_PROJECT_NAME}" \
    --resume_every 1000 \
    "${RESUME[@]}" \
    --group "${GROUP}" \
    --name "wjd-${ARM}-smbase" \
    --save_dir "${SAVE_ROOT}/${ARM}"

echo "finished=$(date -Is)"
