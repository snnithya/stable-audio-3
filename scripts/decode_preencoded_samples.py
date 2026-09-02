"""Decode pre-encoded latents back to audio and dump them next to the source audio.

Listening check for a pre-encoded dataset: for each sample it writes the decoded
latent, the matching slice of the original file, and every decoded control stream,
so you can A/B them in any audio player.

Accepts either config used in the pre-encoding workflow:
  * a dataset2preencoding/ config (the one you ran pre_encode_dataset.py with) —
    latents are read from its "output_path", model/controls come from the config
  * a preencoded/ training config, or a bare latents directory

pre_encode_dataset.py --sanity_check_samples N does the same thing inline while
encoding; this script is for checking a dataset that is already on disk.
Complements check_streamgen_alignment.py, which measures crop alignment
numerically rather than letting you hear it.

On a dataset pre-encoded with --augment_variants, -n counts *tracks*, not files: every
augmentation variant of each selected track is decoded, so -n 5 over a 4-variant dataset
writes 20 samples. Comparing variants only means anything when they are variants of the
same track, and picking N files off disk would have given an arbitrary mix instead.

Usage:
  uv run python scripts/decode_preencoded_samples.py \
      --config stable_audio_3/configs/dataset_configs/dataset2preencoding/local_babyslakh_streamgen.json \
      --out /tmp/preencode_check -n 5
"""

import argparse
import json
import re
from pathlib import Path

import torch
import torchaudio

from stable_audio_3 import AutoencoderModel
from stable_audio_3.data.augmentation import pitch_shift_and_time_stretch
from stable_audio_3.data.utils import build_dataset_from_config


def load_config(path):
    path = Path(path)
    if path.is_dir():
        return None
    with open(path) as f:
        return json.load(f)


def is_preencoding_config(cfg):
    """True for a dataset2preencoding config (raw audio in, latents out)."""
    return cfg is not None and cfg.get("dataset_type", "pre_encoded") != "pre_encoded"


def latents_dirs_from_preencoding_config(cfg):
    """Mirror pre_encode_dataset.py's output layout: one dir, or output_path/<id> per dataset."""
    base = cfg.get("output_path")
    if not base:
        raise ValueError("Pre-encoding config has no 'output_path' to read latents from")
    entries = cfg.get("datasets", [])
    if len(entries) > 1:
        return [str(Path(base) / e.get("id", "dataset")) for e in entries]
    return [str(base)]


def controls_dim_from_metadata(latents_dir, controls):
    """Read the per-control latent widths that pre-encoding recorded in the metadata."""
    for md_path in sorted(Path(latents_dir).glob("*.json")):
        with open(md_path) as f:
            dims = json.load(f).get("controls_dim")
        if dims:
            return dims
        break
    raise ValueError(
        f"No 'controls_dim' in the metadata under {latents_dir}; "
        f"cannot split the fused sidecar for controls {controls}"
    )


def build_dataset(cfg, args, sample_rate, ds_ratio, duration):
    if not is_preencoding_config(cfg):
        return build_dataset_from_config(args.config, sample_rate, ds_ratio, duration)

    from stable_audio_3.data.dataset import LatentDatasetConfig, PreEncodedDataset

    dirs = latents_dirs_from_preencoding_config(cfg)
    print(f"Pre-encoding config: reading latents from {', '.join(dirs)}")

    controls = cfg.get("controls") or None
    controls_dim = controls_dim_from_metadata(dirs[0], controls) if controls else None

    latent_crop_length = None
    if duration is not None:
        latent_crop_length = int(duration * sample_rate) // ds_ratio

    return PreEncodedDataset(
        [LatentDatasetConfig(id=Path(d).name, path=d) for d in dirs],
        latent_crop_length=latent_crop_length,
        random_crop=False,
        controls=controls,
        controls_dim=controls_dim,
    )


VARIANT_RE = re.compile(r"_v(\d+)$")


def select_indices(ds, num_samples):
    """Dataset indices for the first ``num_samples`` tracks, every variant of each.

    Pre-encoding with --augment_variants writes each track once per variant as
    ``<id>_v<k>``, so slicing the first N dataset indices would return a mix of variants
    of unrelated tracks — and in scandir order, a different mix on every machine. Grouping
    by base id and sorting makes the choice reproducible and makes the output what a
    listening check actually wants: the same tracks, once per variant.

    Returns ``[(stem, index), ...]``, variants of a track adjacent and in variant order.
    """
    if not hasattr(ds, "filenames"):
        return [(f"{n:04d}", n) for n in range(min(num_samples, len(ds)))]

    groups = {}
    for idx, entry in enumerate(ds.filenames):
        stem = Path(entry[0]).stem
        groups.setdefault(VARIANT_RE.sub("", stem), []).append((stem, idx))

    selected = []
    for base in sorted(groups)[:num_samples]:
        selected.extend(sorted(groups[base]))
    return selected


def load_resampled(path, target_sr, aug=None):
    """Load a source file at the model's sample rate, replaying any augmentation it got.

    An augmented item's latent is a transposed, re-paced rendering of the file on disk, so
    comparing it against the raw file would show a drift that is not a defect. ``aug`` is
    the ``augmentation`` block pre_encode_dataset.py recorded on the item; replaying it puts
    the ground truth back on the same timeline and in the same key. ``pitch_scope`` is
    honoured: under ``controls`` the target was never transposed, only the accompaniment.
    """
    audio, sr = torchaudio.load(str(path))
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    if aug:
        rate = float(aug.get("time_stretch_rate", 1.0))
        semitones = float(aug.get("pitch_semitones", 0.0))
        if aug.get("pitch_scope") == "controls":
            semitones = 0.0
        if rate != 1.0 or semitones:
            audio = pitch_shift_and_time_stretch(audio, semitones=semitones, rate=rate)
    return audio


def save(path, audio, sr):
    torchaudio.save(str(path), audio.clamp(-1, 1), sr)
    print(f"      wrote {path.name}  ({audio.shape[-1] / sr:.2f}s, {tuple(audio.shape)})")


def main(args):
    cfg = load_config(args.config)
    model = args.ae_model or (cfg or {}).get("model") or "same-l"

    ae = AutoencoderModel.from_pretrained(model, device=args.device)
    sr = ae.sample_rate
    ds_ratio = int(ae.autoencoder.downsampling_ratio)
    print(f"Autoencoder: {model} (sample_rate={sr}, ds_ratio={ds_ratio})")

    duration = None if args.full else args.latent_length * ds_ratio / sr
    ds = build_dataset(cfg, args, sr, ds_ratio, duration)
    if args.full:
        # Decode whatever was stored, uncropped and unpadded.
        ds.latent_crop_length = None
    ds.random_crop = args.random_crop

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.indices:
        selected = [(Path(ds.filenames[n][0]).stem if hasattr(ds, "filenames") else f"{n:04d}", n)
                    for n in args.indices]
    else:
        selected = select_indices(ds, args.num_samples)
        tracks = len({VARIANT_RE.sub("", stem) for stem, _ in selected})
        if len(selected) != tracks:
            print(f"{len(selected)} samples = every augmentation variant of {tracks} tracks")

    # Output stems carry the source latent id, variant suffix included, so
    # make_listening_page.py gives each variant its own card.
    for stem, n in selected:
        latents, info = ds[n]
        aug = info.get("augmentation") or {}
        aug_note = (f"  aug v{aug.get('variant')} rate {aug.get('time_stretch_rate', 1.0):.3f} "
                    f"{aug.get('pitch_semitones', 0.0):+.2f}st") if aug else ""
        print(f"[{n}] {Path(info.get('latent_filename', '?')).name}  "
              f"crop@{info.get('latent_crop_start', 0)}  "
              f"prompt={str(info.get('prompt', ''))[:60]!r}{aug_note}")

        with torch.no_grad():
            decoded = ae.decode(latents.unsqueeze(0).to(args.device)).squeeze(0).float().cpu()
        save(out_dir / f"{stem}_decoded.wav", decoded, sr)

        # Source slice. Pre-encoding runs with random_crop=False, so a stored latent
        # always starts at sample 0 of its file; only the dataset's own crop offsets it.
        src_path = info.get("path")
        if src_path and Path(src_path).exists():
            start = info.get("latent_crop_start", 0) * ds_ratio
            # Augment first, then slice: latent_crop_start indexes the augmented timeline.
            src = load_resampled(src_path, sr, aug)[:, start : start + decoded.shape[-1]]
            save(out_dir / f"{stem}_source.wav", src, sr)
        else:
            print(f"      no source audio at {src_path!r}; skipping ground truth")

        for name, control in (info.get("controls") or {}).items():
            with torch.no_grad():
                ctrl = ae.decode(control.unsqueeze(0).to(args.device)).squeeze(0).float().cpu()
            save(out_dir / f"{stem}_control_{name}.wav", ctrl, sr)

    print(f"\nWrote {len(selected)} samples to {out_dir.resolve()}")
    print(f"Listen to them: uv run python scripts/make_listening_page.py --dir {out_dir.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="Pre-encoding config, pre-encoded training config, or a latents directory")
    p.add_argument("--out", required=True, help="Directory to write wavs into")
    p.add_argument("--ae_model", default=None,
                   help="Autoencoder used for pre-encoding (default: config 'model', else same-l)")
    p.add_argument("--latent_length", type=int, default=256, help="Latent crop length (ignored with --full)")
    p.add_argument("--full", action="store_true", help="Decode the whole stored latent instead of a crop")
    p.add_argument("--random_crop", action="store_true", help="Randomize the crop offset (default: deterministic)")
    p.add_argument("-n", "--num_samples", type=int, default=5,
                   help="Number of tracks to decode; every augmentation variant of each is written")
    p.add_argument("--indices", type=int, nargs="+", default=None, help="Specific dataset indices instead of the first n")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
