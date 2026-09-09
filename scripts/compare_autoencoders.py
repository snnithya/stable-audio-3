"""A/B the SAME-S and SAME-L autoencoders on reconstruction quality.

Cuts fixed-length excerpts out of a directory of stems, round-trips each one through
every requested autoencoder (encode -> decode), and writes source + one reconstruction
per model into a flat directory, named for scripts/make_listening_page.py to pick up:

    <stem>-<track>-t<offset>_source.wav
    <stem>-<track>-t<offset>_same-s.wav
    <stem>-<track>-t<offset>_same-l.wav

Chunking is deliberately off. Chunked encode/decode splices independently-run windows
with a hard cut at each seam (autoencoders.py:550), so any seam artifact would land in
the A/B as if it were an autoencoder difference. Excerpts are sized to fit one pass.

Excerpt length is snapped down to a whole number of latent frames so the encoder's
pad-to-multiple-of-downsampling-ratio step is a no-op: without that, a 12s excerpt is
padded with ~74ms of silence and the reconstruction comes back longer than its source.

The source is written as the model sees it -- resampled and channel-converted -- so all
three rows are sample-aligned and the metrics compare like with like. The `other` stems
in streamgen-drum-mirror are mono against a stereo autoencoder, so this is not academic.

Wavs are large -- 48 excerpts x 3 streams is ~570MB -- so write --out to scratch, not into
the repo. `outputs/` under the repo root is not gitignored beyond `*.wav`, which would leave
the mel PNGs and json to be picked up by a `git add -A`.

Usage:
  uv run python scripts/compare_autoencoders.py \
      --data_dir /data/hai-res/shared/snnithya/sat-zenon-data/bryan-data-1/streamgen-drum-mirror/train/tracks \
      --out /data/scratch-fast/snnithya/sao-3/outputs/ae_compare -n 4 --seconds 12
  uv run python scripts/make_listening_page.py --dir /data/scratch-fast/snnithya/sao-3/outputs/ae_compare
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torchaudio

from stable_audio_3 import AutoencoderModel
from stable_audio_3.data.utils import is_silent, rms_dbfs, silence_fraction
from stable_audio_3.inference.audio_utils import prepare_audio


def find_audio(data_dir, stems, pattern):
    """Discover stems laid out as <data_dir>/<stem>/<track>/<file>.wav.

    Falls back to a plain recursive glob when the directory is not in that layout, so the
    script is not welded to streamgen-drum-mirror.
    """
    data_dir = Path(data_dir)
    found = []
    for path in sorted(data_dir.glob(pattern)):
        rel = path.relative_to(data_dir).parts
        if len(rel) >= 3:
            stem, track = rel[0], rel[1]
        elif len(rel) == 2:
            stem, track = rel[0], path.stem
        else:
            stem, track = "audio", path.stem
        if stems and stem not in stems:
            continue
        found.append((f"{stem}-{track}", path))
    return found


def pick_excerpts(path, n, num_samples, rng, args):
    """Random non-silent offsets into one file, as (start_sample, info) pairs.

    Rejects windows that are empty or mostly empty. A drum stem's rest bars are a real
    part of these tracks, and reconstruction metrics on near-silence are meaningless --
    every model scores well on nothing.
    """
    info = torchaudio.info(str(path))
    total = info.num_frames
    if total < num_samples:
        print(f"  skip (shorter than one excerpt: {total / info.sample_rate:.1f}s)")
        return []

    picks, seen, attempts = [], set(), 0
    budget = max(args.max_attempts, n * args.max_attempts)
    while len(picks) < n and attempts < budget:
        attempts += 1
        start = rng.randrange(0, total - num_samples + 1)
        # Quantise so two picks a few samples apart cannot both survive as "distinct".
        bucket = start // num_samples
        if bucket in seen:
            continue
        audio, sr = torchaudio.load(str(path), frame_offset=start, num_frames=num_samples)
        level = rms_dbfs(audio)
        frac = silence_fraction(audio, sr)
        if is_silent(audio, args.silence_threshold_db) or frac > args.max_silence_fraction:
            continue
        seen.add(bucket)
        picks.append((start, {"rms_dbfs": round(level, 2), "silence_fraction": round(frac, 3)}))

    if len(picks) < n:
        print(f"  only {len(picks)}/{n} non-silent excerpts after {attempts} attempts")
    return sorted(picks)


def si_sdr(est, ref, eps=1e-8):
    """Scale-invariant SDR in dB, over the flattened clip.

    Scale-invariant because a reconstruction that is right but half a dB quieter is not
    the failure plain SNR would call it.
    """
    est, ref = est.flatten().float(), ref.flatten().float()
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = torch.dot(est, ref) / (torch.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return float(10 * torch.log10((target.pow(2).sum() + eps) / (noise.pow(2).sum() + eps)))


def log_mel_l1(est, ref, sample_rate, n_mels=128):
    """Mean L1 between log-mel spectrograms -- a perceptual companion to SI-SDR.

    SI-SDR is waveform-aligned and punishes any phase drift; a decoder can be perceptually
    close while scoring badly on it. Comparing magnitudes as well keeps that visible.
    """
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=2048, hop_length=512, n_mels=n_mels, power=2.0
    )
    a = torch.log10(mel(est.float().mean(0, keepdim=True)).clamp(min=1e-10))
    b = torch.log10(mel(ref.float().mean(0, keepdim=True)).clamp(min=1e-10))
    return float((a - b).abs().mean())


def roundtrip(ae, audio, sample_rate):
    """Encode then decode one excerpt, returning (audio, timings, latent shape).

    Timed with a CUDA sync on each side so the split between encode and decode is real
    rather than an artifact of async kernel launches.
    """
    def sync():
        if ae.device == "cuda" or (isinstance(ae.device, torch.device) and ae.device.type == "cuda"):
            torch.cuda.synchronize()

    sync()
    t0 = time.perf_counter()
    latents = ae.encode(audio, sample_rate, chunked=False)
    sync()
    t1 = time.perf_counter()
    decoded = ae.decode(latents, chunked=False)
    sync()
    t2 = time.perf_counter()

    return decoded[0].cpu(), {"encode_s": round(t1 - t0, 3), "decode_s": round(t2 - t1, 3)}, tuple(latents.shape)


def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    files = find_audio(args.data_dir, set(args.stems) if args.stems else None, args.pattern)
    if not files:
        raise SystemExit(f"No audio found under {args.data_dir} matching {args.pattern!r}")
    print(f"{len(files)} files under {args.data_dir}")

    # Load each autoencoder once and drive every excerpt through it. Loading SAME-L per
    # excerpt would dominate the runtime and make the timings meaningless.
    print(f"\nLoading {', '.join(args.models)} …")
    aes = {}
    for name in args.models:
        t0 = time.perf_counter()
        aes[name] = AutoencoderModel.from_pretrained(name, device=args.device)
        print(f"  {name}: {time.perf_counter() - t0:.1f}s on {aes[name].device}")

    sample_rate = next(iter(aes.values())).sample_rate
    if any(ae.sample_rate != sample_rate for ae in aes.values()):
        raise SystemExit("Autoencoders disagree on sample rate; cannot share one source wav")

    # Snap to a whole number of latent frames so the encoder pads nothing (see module docstring).
    ratio = int(next(iter(aes.values())).autoencoder.downsampling_ratio)
    num_samples = max(ratio, (int(args.seconds * sample_rate) // ratio) * ratio)
    channels = next(iter(aes.values())).autoencoder.in_channels
    print(f"\nExcerpt: {num_samples} samples = {num_samples / sample_rate:.3f}s "
          f"= {num_samples // ratio} latent frames, {channels}ch @ {sample_rate} Hz, chunking off")

    rows = []
    for sample_id, path in files:
        print(f"\n{sample_id}  ({path.name})")
        for start, stats in pick_excerpts(path, args.n, num_samples, rng, args):
            raw, sr = torchaudio.load(str(path), frame_offset=start, num_frames=num_samples)
            # Resample + channel-convert once, up front: this is the model's actual input,
            # and writing it as `source` is what makes the three rows comparable.
            source = prepare_audio(
                raw, in_sr=sr, target_sr=sample_rate, target_length=num_samples,
                target_channels=channels, device="cpu",
            )[0]

            excerpt_id = f"{sample_id}-t{start // sr:04d}"
            torchaudio.save(str(out_dir / f"{excerpt_id}_source.wav"), source, sample_rate)

            meta = {
                "track": sample_id,
                "path": str(path),
                "start_seconds": round(start / sr, 2),
                "seconds": round(num_samples / sample_rate, 3),
                "source_channels": raw.shape[0],
                **stats,
                "streams": {},
            }

            line = {"id": excerpt_id, **{k: meta[k] for k in ("track", "start_seconds", "rms_dbfs")}}
            for name, ae in aes.items():
                decoded, timings, latent_shape = roundtrip(ae, source, sample_rate)
                decoded = decoded[..., : source.shape[-1]]
                torchaudio.save(str(out_dir / f"{excerpt_id}_{name}.wav"), decoded, sample_rate)

                sdr = si_sdr(decoded, source)
                mel_l1 = log_mel_l1(decoded, source, sample_rate)
                meta["streams"][name] = (
                    f"SI-SDR {sdr:.2f} dB · mel L1 {mel_l1:.4f} · "
                    f"enc {timings['encode_s']:.2f}s dec {timings['decode_s']:.2f}s · "
                    f"latent {latent_shape[1]}×{latent_shape[2]}"
                )
                line[name] = {"si_sdr_db": round(sdr, 3), "log_mel_l1": round(mel_l1, 5),
                              "latent_shape": list(latent_shape), **timings}
                print(f"    {name:7s} SI-SDR {sdr:7.2f} dB   mel L1 {mel_l1:.4f}   "
                      f"enc {timings['encode_s']:.2f}s  dec {timings['decode_s']:.2f}s")

            (out_dir / f"{excerpt_id}.json").write_text(json.dumps(meta, indent=2))
            rows.append(line)

    summary = {
        "data_dir": str(args.data_dir),
        "models": list(args.models),
        "seconds": num_samples / sample_rate,
        "num_samples": num_samples,
        "chunked": False,
        "seed": args.seed,
        "excerpts": rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'':42s} " + "  ".join(f"{n:>22s}" for n in args.models))
    print(f"{'':42s} " + "  ".join(f"{'SI-SDR dB   mel L1':>22s}" for _ in args.models))
    for line in rows:
        cells = "  ".join(f"{line[n]['si_sdr_db']:11.2f} {line[n]['log_mel_l1']:10.4f}" for n in args.models)
        print(f"{line['id']:42s} {cells}")
    if rows:
        print(f"{'MEAN':42s} " + "  ".join(
            f"{sum(r[n]['si_sdr_db'] for r in rows) / len(rows):11.2f} "
            f"{sum(r[n]['log_mel_l1'] for r in rows) / len(rows):10.4f}" for n in args.models))

    print(f"\nWrote {len(rows)} excerpts × {len(args.models)} models to {out_dir.resolve()}")
    print(f"Next:  uv run python scripts/make_listening_page.py --dir {out_dir.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="Root of the stems to sample from")
    p.add_argument("--out", required=True, help="Directory to write wavs, per-excerpt json and metrics.json")
    p.add_argument("--models", nargs="+", default=["same-s", "same-l"], help="Autoencoders to compare")
    p.add_argument("-n", type=int, default=4, help="Excerpts per file")
    p.add_argument("--seconds", type=float, default=12.0,
                   help="Excerpt length; snapped down to a whole latent frame")
    p.add_argument("--pattern", default="*/*/*.wav", help="Glob, relative to --data_dir")
    p.add_argument("--stems", nargs="*", default=None, help="Restrict to these top-level stem dirs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="Defaults to cuda when available")
    p.add_argument("--silence_threshold_db", type=float, default=-50.0,
                   help="Reject excerpts whose RMS is below this")
    p.add_argument("--max_silence_fraction", type=float, default=0.5,
                   help="Reject excerpts more than this fraction silent")
    p.add_argument("--max_attempts", type=int, default=20, help="Offset draws per wanted excerpt")
    main(p.parse_args())
