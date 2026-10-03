#!/bin/bash
#SBATCH --job-name=wjd-ft-rms
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-low
#SBATCH --account=hai-res
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/03_3_finetune_wjd_rms_%j.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/03_3_finetune_wjd_rms_%j.out
#
# WJD finetune, arm "rms": 1-ch causal drum RMS envelope as the control (3.3).
#
#   sbatch sbatch/03_3_finetune_wjd_rms.sbatch
#   sbatch --qos=<other> sbatch/03_3_finetune_wjd_rms.sbatch   # or edit the --qos line above
#
# One file per arm so each can run under its own QoS. Everything else is shared:
# sbatch/03_3_finetune_wjd_common.sh (training flags, checkout clone, save dir).

ARM=rms
MODEL_CONFIG=stable_audio_3/configs/model_configs/small_music_wjd_drums_rms.json
GROUP=03-3-wjd-drums-rms
source /data/hai-res/snnithya/sa3-tap/stable-audio-3/sbatch/03_3_finetune_wjd_common.sh
