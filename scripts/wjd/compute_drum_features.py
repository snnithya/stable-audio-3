"""Compute the drum feature controls once per song, so a chunked pre-encode slices them.

For every ``tracks/drums/<Track>/drums.flac`` of a split this writes
``tracks/drums/<Track>/drums_features.npz`` holding, at the latent rate (one frame per
``LATENT_HOP`` = 4096 samples), the controls ``custom_md_wjd.py`` can return:

    drums_rms          [1, T]  causal RMS envelope in [0, 1]           (rms_envelope_control)
    drums_tria_fixed   [2, T]  causal TRIA bands, dataset z-score      (tria_control, "fixed")
    drums_tria_ema     [2, T]  causal TRIA bands, running z-score      (tria_control, "ema")
    rms_db             [1, T]  the per-frame RMS in dBFS the module's silence gate reads
    meta               JSON: sample_rate, hop, n_samples, n_frames, the TriaStats used, source

``custom_md_wjd.py`` (``WJD_DRUM_FEATURES=precomputed``, the default) then takes the frames
``[chunk_offset / hop, ...)`` of these for each item instead of recomputing from audio. For
``drums_rms`` that is the same thing frame for frame (the audio is zero-padded to whole
frames here exactly as a window is). For the TRIA controls the song-level computation is the
*right* one, not just the faster one: the crossover is an IIR filter, so computed per window
it starts from rest at every chunk boundary and the chunk's first frame carries a start-up
transient (``drums_tria_fixed`` differs there, agrees from the second frame on); and the
running statistics of ``drums_tria_ema`` here carry the song's actual preceding seconds into
the chunk, as a live stream would, instead of restarting from the dataset prior per chunk.
The speed-up is on top: the drum stem is processed once instead of once per window per target
stem, about 150 times at a 50 % hop with three targets.

Usage (the pre-encode sbatch runs this first; existing files are skipped unless --overwrite):

    uv run python scripts/wjd/compute_drum_features.py \\
        --mirror /data/hai-res/shared/snnithya/sao-3/data/wjd/wjd-stem-mirror --split train
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_audio_3.data.features import (  # noqa: E402
    LATENT_HOP,
    TriaStats,
    block_rms_db,
    frames_for,
    rms_envelope_control,
    tria_control,
)

FEATURES_FILENAME = "drums_features.npz"
FEATURE_KEYS = ("drums_rms", "drums_tria_fixed", "drums_tria_ema")


def features_path_for(drums_path) -> Path:
    return Path(drums_path).parent / FEATURES_FILENAME


def compute_track_features(drums_path, stats: TriaStats, sample_rate: int = 44100, hop: int = LATENT_HOP):
    """All feature controls of one drum stem over the whole song: ``({key: [C, T]}, meta)``.

    The audio is zero-padded to whole frames before anything is computed, so the song's last
    partial frame is diluted exactly as it is when a window's padding covers it; slicing these
    arrays therefore reproduces the per-window RMS control bit for bit.
    """
    audio, sr = torchaudio.load(str(drums_path))
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    n = audio.shape[-1]
    n_frames = frames_for(n, hop)
    audio = torch.nn.functional.pad(audio, (0, n_frames * hop - n))
    feats = {
        "drums_rms": rms_envelope_control(audio, hop=hop),
        "drums_tria_fixed": tria_control(audio, stats, norm="fixed", hop=hop),
        "drums_tria_ema": tria_control(audio, stats, norm="ema", hop=hop),
        "rms_db": block_rms_db(audio, hop),
    }
    meta = {
        "source": str(drums_path),
        "sample_rate": sample_rate,
        "hop": hop,
        "n_samples": n,
        "n_frames": n_frames,
        "keys": list(FEATURE_KEYS),
        "stats": asdict(stats),
        "script": "scripts/wjd/compute_drum_features.py",
    }
    return feats, meta


def save_features(path, feats, meta):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, meta=json.dumps(meta), **{k: v.numpy().astype(np.float32) for k, v in feats.items()})
    tmp.replace(path)


def load_features(path):
    """``({key: float32 tensor [C, T]}, meta)`` as written by `save_features`."""
    with np.load(path) as npz:
        meta = json.loads(str(npz["meta"]))
        feats = {k: torch.from_numpy(npz[k]) for k in npz.files if k != "meta"}
    return feats, meta


def _worker(args):
    drums_path, stats_path, overwrite = args
    out = features_path_for(drums_path)
    if out.exists() and not overwrite:
        return str(out), "exists"
    torch.set_num_threads(1)
    feats, meta = compute_track_features(drums_path, TriaStats.load(stats_path))
    save_features(out, feats, meta)
    return str(out), f"{meta['n_frames']} frames"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mirror", required=True, help="wjd-stem-mirror root")
    ap.add_argument("--split", default="train")
    ap.add_argument("--stats", default=None, help="TriaStats JSON (default: the committed WJD one)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    drums = sorted((Path(args.mirror) / args.split / "tracks" / "drums").glob("*/drums.flac"))
    if not drums:
        sys.exit(f"no drum stems under {args.mirror}/{args.split}/tracks/drums")
    stats_path = args.stats  # None -> TriaStats.load picks the committed default
    print(f"{len(drums)} drum stems; stats {stats_path or 'default (configs/dataset_configs/features/wjd_drums_tria_stats.json)'}")
    done = skipped = 0
    jobs = [(p, stats_path, args.overwrite) for p in drums]
    if args.workers > 1:
        pool = ProcessPoolExecutor(args.workers)
        results = pool.map(_worker, jobs)
    else:
        pool = None
        results = map(_worker, jobs)  # in-process: no pickling, easier to debug
    try:
        for i, (out, status) in enumerate(results, 1):
            if status == "exists":
                skipped += 1
            else:
                done += 1
            if i % 25 == 0 or i == len(drums):
                print(f"  {i}/{len(drums)}  (written {done}, skipped existing {skipped})")
    finally:
        if pool is not None:
            pool.shutdown()
    print(f"done: wrote {done}, skipped {skipped} existing {FEATURES_FILENAME} files")


if __name__ == "__main__":
    main()
