#!/bin/bash
#SBATCH --job-name=wjd-fetch-yt
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-extended
#SBATCH --account=hai-res
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/data/scratch-fast/snnithya/sao-3/logs/03_1_fetch_wjd_youtube_%j.out
#SBATCH --error=/data/scratch-fast/snnithya/sao-3/logs/03_1_fetch_wjd_youtube_%j.err

# Experiment 3.1, route C: fetch the WJD recordings from YouTube via the JazzTube link table.
# Plan and time-reference conventions: experiments/03-wjd-jazz-stems/01-audio-acquisition.md.
#
# Selection is done by the script (drum tracks with a JSD entry and a live linked video,
# ~220 tracks); one video per track, best available audio stream, transcoded to 44.1 kHz
# stereo FLAC. Writes under /data/hai-res/shared/snnithya/sao-3/data/wjd/audio/:
#   raw/<Track>/<id>.<ext>, aligned/<Track>.flac, aligned/<Track>.alignment.json,
#   manifest.csv, _failed.json
#
# CPU only, no GPU. Needs outbound internet from the node -- if the log shows every track
# failing at the probe step with a network error, run it from the login node instead
# (`nohup ... &`); the script is identical.
#
# Resumable: tracks with an alignment.json are skipped, so re-submitting fills the gaps
# after link rot or rate limiting. If YouTube starts answering "Sign in to confirm you're
# not a bot" from the cluster's IP, export a Netscape-format cookies file exported from a
# logged-in browser as YTDLP_COOKIES before submitting; the script passes it to yt-dlp.
#
# Smoke test first:  FETCH_LIMIT=3 sbatch sbatch/03_1_fetch_wjd_youtube.sbatch
# Probe only:        FETCH_ARGS=--dry_run sbatch sbatch/03_1_fetch_wjd_youtube.sbatch
#
# No repo clone here: the script is standalone (stdlib only) and is run straight from the
# checkout below, so uncommitted edits to it ARE what runs. That is the opposite of the
# training sbatch scripts, which pin a commit; it is deliberate for a download job.

set -euo pipefail

REPO=/data/hai-res/snnithya/sa3-tap/stable-audio-3
PY=$REPO/.venv/bin/python           # this checkout's venv; yt-dlp + static-ffmpeg installed with `uv pip install`
YTDLP=$REPO/.venv/bin/yt-dlp
FFMPEG=$REPO/.venv/bin/ffmpeg        # symlink to the static-ffmpeg binary (ffprobe alongside it)
WJD_ROOT=/data/hai-res/shared/snnithya/sao-3/data/wjd

set -a
source "$REPO/.env"
set +a

mkdir -p /data/scratch-fast/snnithya/sao-3/logs
echo "host=$(hostname) start=$(date -Is) commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "network: $(curl -sI --max-time 20 https://www.youtube.com | head -1 || echo 'no reply')"

# Keep yt-dlp's cache off the over-quota AFS home.
export XDG_CACHE_HOME=/data/hai-res/snnithya/.cache
# Recreate the tools if the venv is rebuilt:
#   uv pip install --python .venv/bin/python yt-dlp static-ffmpeg && .venv/bin/static_ffmpeg_paths
#   then symlink the printed ffmpeg/ffprobe into .venv/bin (see experiments/03-wjd-jazz-stems/01-audio-acquisition.md)

EXTRA=()
[ -n "${FETCH_LIMIT:-}" ] && EXTRA+=(--limit "$FETCH_LIMIT")
[ -n "${FETCH_ARGS:-}" ] && EXTRA+=($FETCH_ARGS)

"$PY" "$REPO/scripts/wjd/fetch_youtube.py" \
    --wjd_root "$WJD_ROOT" \
    --yt_dlp "$YTDLP" \
    --ffmpeg "$FFMPEG" \
    --duration_tol 15 \
    --max_candidates 4 \
    --sleep 3 8 \
    "${EXTRA[@]}"

echo "finished=$(date -Is)"
