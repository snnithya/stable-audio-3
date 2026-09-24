#!/bin/bash
#SBATCH --job-name=df-test
#SBATCH --partition=hai-res-l40s
#SBATCH --account=hai-res
#SBATCH --qos=hai-res-main
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/%j.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/%j.err

set -euo pipefail

# The Stable Audio 3 repos are gated; the compute nodes reach them only through this
# already-populated cache, so set it before anything imports huggingface_hub.
set -a
source /data/hai-res/snnithya/stable-audio-3/.env
set +a

export SLURM_TMPDIR=/tmp/slurm-$SLURM_JOB_ID
mkdir -p $SLURM_TMPDIR
cd $SLURM_TMPDIR
git clone -b nithya/df /data/hai-res/snnithya/stable-audio-3/.git ./stable-audio-3
cd stable-audio-3
export PYTHONPATH=$PWD

source /data/hai-res/snnithya/stable-audio-3/.venv/bin/activate 
srun python scripts/train_finetune.py \
    --model small-music-base \
    --model_config  stable_audio_3/configs/model_configs/small_music_base_df.json\
    --dataset_config stable_audio_3/configs/dataset_configs/preencoded/slakh_streamgen_train_preencoded.json \
    --val_dataset_config stable_audio_3/configs/dataset_configs/preencoded/slakh_streamgen_validation_preencoded.json \
    --val_every 1000 \
    --steps 1000000 \
    --batch_size 128 \
    --seed 42 \
    --freeze_conditioner \
    --checkpoint_every 5000 \
    --demo_every 10000 \
    --log_every 100 \
    --logger wandb \
    --project sao-3 \
    --group df-tests \
    --name df-slakh \
    --save_dir /data/scratch-fast/snnithya/sao-3/ft_checkpoints/ \
    --resume_ckpt /data/scratch-fast/snnithya/sao-3/ft_checkpoints/sao-3/7tvd98n0/checkpoints/last.ckpt \
    --wandb_id 7tvd98n0


echo "finished=$(date -Is)"
