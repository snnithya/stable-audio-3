#!/bin/bash
#SBATCH --job-name=wjd-ft-audio
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --account=hai-res
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/03_3_finetune_wjd_audio_%j.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/03_3_finetune_wjd_audio_%j.out
#
# WJD finetune, arm "audio": the drum stem's 256-ch VAE latent as 'streamgen_latent', the Slakh-style condition (3.3's original plan). Needs the 'audio' block in the sidecar.
#
#   sbatch sbatch/03_3_finetune_wjd_audio.sbatch
#   sbatch --qos=<other> sbatch/03_3_finetune_wjd_audio.sbatch   # or edit the --qos line above
#
# One file per arm so each can run under its own QoS. Everything else is shared:
# sbatch/03_3_finetune_wjd_common.sh (training flags, checkout clone, save dir).

ARM=audio
MODEL_CONFIG=stable_audio_3/configs/model_configs/small_music_streamgen.json
GROUP=03-3-wjd-drums-audio
source /data/hai-res/snnithya/sa3-tap/stable-audio-3/sbatch/03_3_finetune_wjd_common.sh
