#!/bin/bash
#SBATCH --job-name=wjd-ft-base
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --account=hai-res
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/03_3_finetune_wjd_base_%j.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/03_3_finetune_wjd_base_%j.out
#
# WJD finetune, arm "base": text-only baseline: same model, no local control (text + causal inpainting only), the matched pair for every control arm as in 1.2.
#
#   sbatch sbatch/03_3_finetune_wjd_base.sbatch
#   sbatch --qos=<other> sbatch/03_3_finetune_wjd_base.sbatch   # or edit the --qos line above
#
# One file per arm so each can run under its own QoS. Everything else is shared:
# sbatch/03_3_finetune_wjd_common.sh (training flags, checkout clone, save dir).

ARM=base
MODEL_CONFIG=stable_audio_3/configs/model_configs/small_music_baseline.json
GROUP=03-3-wjd-drums-rms
source /data/hai-res/snnithya/sa3-tap/stable-audio-3/sbatch/03_3_finetune_wjd_common.sh
