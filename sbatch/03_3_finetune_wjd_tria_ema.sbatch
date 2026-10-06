#!/bin/bash
#SBATCH --job-name=wjd-ft-tria-ema
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --account=hai-res
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/03_3_finetune_wjd_tria_ema_%j.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/03_3_finetune_wjd_tria_ema_%j.out
#
# WJD finetune, arm "tria_ema": 2-ch causal TRIA bands, running (EMA, tau = 30 s) normalisation (3.4, T1-ema).
#
#   sbatch sbatch/03_3_finetune_wjd_tria_ema.sbatch
#   sbatch --qos=<other> sbatch/03_3_finetune_wjd_tria_ema.sbatch   # or edit the --qos line above
#
# One file per arm so each can run under its own QoS. Everything else is shared:
# sbatch/03_3_finetune_wjd_common.sh (run to continue, git ref pin, clone; read from the
# checkout) and sbatch/03_3_finetune_wjd_train.sh (training flags; read from the pinned clone).

ARM=tria_ema
MODEL_CONFIG=stable_audio_3/configs/model_configs/small_music_wjd_drums_tria_ema.json
GROUP=03-4-wjd-tria-smbase
source /data/hai-res/snnithya/sa3-tap/stable-audio-3/sbatch/03_3_finetune_wjd_common.sh
