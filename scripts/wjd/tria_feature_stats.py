"""Dataset constants for the causal TRIA drum control (experiment 3.4).

TRIA's rhythm features split the spectrum at the clip's equal-energy frequency and z-score
each band against the clip's own mean and std. Neither is causal, so the control in
``stable_audio_3/data/features.py`` (``tria_control``) uses constants of the training corpus
instead: one split frequency and one mean/std per band. This script measures them on the drum
stems of the WJD stem mirror and writes ``TriaStats`` JSON.

Two passes over the drums of one split, over each whole track (or its first
``--sample_size`` samples, when the pre-encode caps the tracks):

  1. Equal-energy frequency per track: mean power spectrum over 4096-sample Hann blocks,
     cumulated over frequency, read where it crosses 50 %. The corpus value is the median
     over tracks (``--split_hz`` overrides it, and skips nothing: pass 1 still runs for the
     report).
  2. With that split, ``band_rms_db`` per track; mean/std/percentiles per band over all valid
     frames, plus the fraction of frames at the floor. The mean/std written to the stats file
     are taken over the frames *above* the floor unless ``--include_floor_frames``: a frame at
     -80 dBFS is a separator dropout or a tacet, maps to 0 whatever the statistics are, and
     would otherwise only inflate the std and compress the range the real dynamics get.

Usage:

    uv run python scripts/wjd/tria_feature_stats.py \\
        --mirror /data/hai-res/shared/snnithya/sao-3/data/wjd/wjd-stem-mirror --split train \\
        --out stable_audio_3/configs/dataset_configs/features/wjd_drums_tria_stats.json \\
        --report /data/hai-res/shared/snnithya/sao-3/data/wjd/wjd-stem-mirror/tria_stats_report.json

The report JSON carries the per-track numbers so the choice can be revisited without
re-reading 11 hours of audio.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_audio_3.data.features import (  # noqa: E402
    LATENT_HOP,
    RMS_FLOOR_DB,
    TriaStats,
    band_rms_db,
)

SPECTRUM_BLOCK = 4096


def _init_worker():
    torch.set_num_threads(1)


def load_mono(path, sample_size, sample_rate):
    audio, sr = torchaudio.load(str(path), num_frames=-1 if sample_size is None else sample_size)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio.float().mean(dim=0)


def energy_quantile_hz(mono, sample_rate, quantiles=(0.25, 0.5, 0.75), block=SPECTRUM_BLOCK):
    """Frequencies below which the given fractions of the clip's energy lie (linear interpolation)."""
    n = mono.shape[-1] // block * block
    if n == 0:
        return {q: float("nan") for q in quantiles}
    frames = mono[:n].reshape(-1, block).double() * torch.hann_window(block, dtype=torch.float64)
    power = torch.fft.rfft(frames, dim=-1).abs().pow(2).mean(dim=0)
    cum = torch.cumsum(power, dim=0)
    cum = cum / cum[-1]
    freqs = torch.fft.rfftfreq(block, 1 / sample_rate)
    out = {}
    for q in quantiles:
        idx = int(torch.searchsorted(cum, torch.tensor(q, dtype=cum.dtype)).item())
        idx = min(max(idx, 1), len(cum) - 1)
        c0, c1 = cum[idx - 1].item(), cum[idx].item()
        frac = 0.0 if c1 == c0 else (q - c0) / (c1 - c0)
        out[q] = float(freqs[idx - 1] + frac * (freqs[idx] - freqs[idx - 1]))
    return out


def pass1(args):
    path, sample_size, sample_rate = args
    mono = load_mono(path, sample_size, sample_rate)
    qs = energy_quantile_hz(mono, sample_rate)
    rms = 20 * np.log10(max(float(mono.pow(2).mean().sqrt()), 1e-12))
    return {
        "track": Path(path).parent.name,
        "seconds": mono.shape[-1] / sample_rate,
        "rms_dbfs": rms,
        "hz_25": qs[0.25],
        "hz_50": qs[0.5],
        "hz_75": qs[0.75],
    }


def pass2(args):
    path, sample_size, sample_rate, split_hz, floor_db = args
    mono = load_mono(path, sample_size, sample_rate)
    db = band_rms_db(mono.unsqueeze(0), split_hz, sample_rate, LATENT_HOP, floor_db)
    return Path(path).parent.name, db.numpy()


def summarize(db, floor_db):
    pct = np.percentile(db, [1, 5, 25, 50, 75, 95, 99])
    return {
        "mean": float(db.mean()),
        "std": float(db.std()),
        "percentiles": {str(p): float(v) for p, v in zip([1, 5, 25, 50, 75, 95, 99], pct)},
        "fraction_at_floor": float((db <= floor_db + 1e-6).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mirror", required=True, help="wjd-stem-mirror root")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample_size", type=int, default=None,
                    help="only the first N samples of each track (a pre-encode cap); default: whole track")
    ap.add_argument("--sample_rate", type=int, default=44100)
    ap.add_argument("--split_hz", type=float, default=None, help="override the measured median")
    ap.add_argument("--floor_db", type=float, default=RMS_FLOOR_DB)
    ap.add_argument("--ema_tau_seconds", type=float, default=30.0)
    ap.add_argument("--out", required=True, help="TriaStats JSON to write")
    ap.add_argument("--report", default=None, help="per-track report JSON")
    ap.add_argument("--include_floor_frames", action="store_true",
                    help="take mean/std over all valid frames, floor frames included")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N tracks only (smoke test)")
    args = ap.parse_args()

    drums_dir = Path(args.mirror) / args.split / "tracks" / "drums"
    paths = sorted(drums_dir.glob("*/drums.flac"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        sys.exit(f"no drums under {drums_dir}")
    print(f"{len(paths)} drum stems under {drums_dir}")

    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
        rows = list(pool.map(pass1, [(p, args.sample_size, args.sample_rate) for p in paths]))
    hz50 = np.array([r["hz_50"] for r in rows])
    print("equal-energy frequency over tracks (Hz): "
          + ", ".join(f"p{p}={v:.0f}" for p, v in zip([5, 25, 50, 75, 95], np.percentile(hz50, [5, 25, 50, 75, 95]))))
    split_hz = args.split_hz if args.split_hz is not None else float(np.median(hz50))
    print(f"split_hz = {split_hz:.1f} ({'given' if args.split_hz is not None else 'median'})")

    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
        feats = list(pool.map(
            pass2, [(p, args.sample_size, args.sample_rate, split_hz, args.floor_db) for p in paths]
        ))
    all_db = np.concatenate([db for _, db in feats], axis=1)  # [2, frames]
    bands = {"low": summarize(all_db[0], args.floor_db), "high": summarize(all_db[1], args.floor_db)}
    above = {}
    for i, name in enumerate(("low", "high")):
        sel = all_db[i] > args.floor_db + 1e-6
        above[name] = {"mean": float(all_db[i][sel].mean()), "std": float(all_db[i][sel].std())}
        s = bands[name]
        print(f"{name:>4} band: mean {s['mean']:.1f} dB, std {s['std']:.1f} dB "
              f"(above floor: mean {above[name]['mean']:.1f}, std {above[name]['std']:.1f}), "
              f"p5 {s['percentiles']['5']:.1f}, p50 {s['percentiles']['50']:.1f}, p95 {s['percentiles']['95']:.1f}, "
              f"at floor {s['fraction_at_floor']:.2%}")
        bands[name]["above_floor"] = above[name]
    chosen = bands if args.include_floor_frames else above

    stats = TriaStats(
        split_hz=round(split_hz, 1),
        band_mean_db=[round(chosen["low"]["mean"], 2), round(chosen["high"]["mean"], 2)],
        band_std_db=[round(chosen["low"]["std"], 2), round(chosen["high"]["std"], 2)],
        sample_rate=args.sample_rate,
        floor_db=args.floor_db,
        ema_tau_seconds=args.ema_tau_seconds,
        source=f"{drums_dir} ({len(paths)} tracks, "
               f"{'whole tracks' if args.sample_size is None else f'first {args.sample_size} samples each'}, "
               f"mean/std over frames {'incl.' if args.include_floor_frames else 'above'} the floor), "
               f"scripts/wjd/tria_feature_stats.py",
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stats.save(args.out)
    print(f"wrote {args.out}")

    if args.report:
        per_track = {r["track"]: dict(r) for r in rows}
        for track, db in feats:
            per_track[track]["band_mean_db"] = [float(db[0].mean()), float(db[1].mean())]
            per_track[track]["band_std_db"] = [float(db[0].std()), float(db[1].std())]
            per_track[track]["frames"] = int(db.shape[1])
        report = {"split_hz": split_hz, "bands": bands, "n_frames": int(all_db.shape[1]), "tracks": per_track}
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=1))
        npz = Path(args.report).with_suffix(".npz")
        np.savez_compressed(npz, low=all_db[0], high=all_db[1],
                            track_frames=np.array([db.shape[1] for _, db in feats]),
                            tracks=np.array([t for t, _ in feats]))
        print(f"wrote {args.report} and {npz} (per-frame band levels, for re-deriving statistics)")


if __name__ == "__main__":
    main()
