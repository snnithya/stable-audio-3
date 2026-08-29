"""Verify a streamgen control sidecar is time-aligned with its audio latents.

Decodes a cropped drum latent and its paired streamgen control back to audio, then
cross-correlates each against the source audio at the offset implied by latent_crop_start.
Both peaks must land at lag 0.

This is the check to run after (re-)pre-encoding a dataset: a crop-offset desync between
the target and its control produces a model that trains happily on misaligned conditioning.

Usage:
  uv run python scripts/check_streamgen_alignment.py \
      --config stable_audio_3/configs/dataset_configs/preencoded/local_babyslakh_streamgen_preencoded.json
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from stable_audio_3 import AutoencoderModel
from stable_audio_3.data.utils import build_dataset_from_config

HOP = 512


def envelope(x, hop=HOP):
    e = x.mean(0).abs()
    return torch.nn.functional.avg_pool1d(e[None, None], hop, hop)[0, 0]


def best_lag(a, b):
    """Lag (in envelope frames) maximizing normalized cross-correlation of a against b."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    # conv1d is cross-correlation in torch (no kernel flip), which is what we want here.
    padded = torch.nn.functional.pad(a[None, None], (n // 2, n // 2))
    xc = torch.nn.functional.conv1d(padded, b[None, None])[0, 0]
    peak = int(xc.argmax())
    return peak - n // 2, float(xc[peak] / n)


def load_resampled(path, target_sr):
    audio, sr = torchaudio.load(str(path))
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio


def main(args):
    ae = AutoencoderModel.from_pretrained(args.ae_model, device="cuda")
    sr = ae.sample_rate
    ds = build_dataset_from_config(
        args.config, sr, args.ds_ratio, args.latent_length * args.ds_ratio / sr
    )

    failures = 0
    for n in range(args.num_samples):
        latents, info = ds[n]
        control = info.get("controls", {}).get(args.control)
        if control is None:
            raise KeyError(
                f"Control '{args.control}' not in info['controls']. "
                f"Check 'controls'/'controls_dim' in {args.config}."
            )

        start = info["latent_crop_start"] * args.ds_ratio
        length = latents.shape[-1] * args.ds_ratio

        with torch.no_grad():
            target = ae.decode(latents.unsqueeze(0).cuda()).squeeze(0).float().cpu()
            accomp = ae.decode(control.unsqueeze(0).cuda()).squeeze(0).float().cpu()

        drum_path = Path(info["path"])
        src_target = load_resampled(drum_path, sr)[:, start : start + length]

        other_dir = drum_path.parent.parent.parent / "other" / drum_path.parent.name
        src_other = None
        for stem in sorted(other_dir.glob("*.wav")):
            a = load_resampled(stem, sr)[:, start : start + length]
            src_other = a if src_other is None else src_other + a

        lag_t, corr_t = best_lag(envelope(target), envelope(src_target))
        print(f"[{n}] {info.get('track_id')} crop@{info['latent_crop_start']}")
        print(f"      target vs source        : lag {lag_t:+d} ({lag_t * HOP / sr * 1000:+.0f} ms), corr {corr_t:.3f}")

        if src_other is None:
            print("      no source accompaniment found; skipping control check")
            continue

        lag_c, corr_c = best_lag(envelope(accomp), envelope(src_other))
        # The control correlates below 1.0 by design: it is a random, re-leveled subset of
        # the stems, not the full sum. Only the LAG matters for alignment.
        print(f"      control vs source stems : lag {lag_c:+d} ({lag_c * HOP / sr * 1000:+.0f} ms), corr {corr_c:.3f}")

        if abs(lag_t) > args.tolerance or abs(lag_c) > args.tolerance:
            failures += 1
            print("      MISALIGNED")

    print()
    print("ALIGNMENT:", "OK" if failures == 0 else f"FAILED for {failures}/{args.num_samples} samples")
    return 1 if failures else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Pre-encoded dataset config to check")
    p.add_argument("--control", default="streamgen_latent", help="Control name to verify")
    p.add_argument("--ae_model", default="same-l", help="Autoencoder used for pre-encoding")
    p.add_argument("--ds_ratio", type=int, default=4096, help="Autoencoder downsampling ratio")
    p.add_argument("--latent_length", type=int, default=256, help="Latent crop length")
    p.add_argument("-n", "--num_samples", type=int, default=3)
    p.add_argument("--tolerance", type=int, default=1, help="Allowed lag in envelope frames")
    raise SystemExit(main(p.parse_args()))
