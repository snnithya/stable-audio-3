#!/bin/bash
#SBATCH --job-name=wjd-ft-tria-fixed
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --account=hai-res
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/03_3_finetune_wjd_tria_fixed_%j.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/03_3_finetune_wjd_tria_fixed_%j.out
#
# WJD finetune, arm "tria_fixed": 2-ch causal TRIA bands, dataset-statistics normalisation (3.4, T1-fixed).
#
#   sbatch sbatch/03_3_finetune_wjd_tria_fixed.sbatch
#   sbatch --qos=<other> sbatch/03_3_finetune_wjd_tria_fixed.sbatch   # or edit the --qos line above
#
# One file per arm so each can run under its own QoS. Everything else is shared:
# sbatch/03_3_finetune_wjd_common.sh (training flags, checkout clone, save dir).

ARM=tria_fixed
MODEL_CONFIG=stable_audio_3/configs/model_configs/small_music_wjd_drums_tria_fixed.json
GROUP=03-4-wjd-tria
source /data/hai-res/snnithya/sa3-tap/stable-audio-3/sbatch/03_3_finetune_wjd_common.sh
