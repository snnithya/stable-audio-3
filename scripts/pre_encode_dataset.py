"""
Pre-encode a dataset of audio clips into latents using Stable Audio 3, saving the latents and metadata to disk.

Dataset layout:
  data_dir/
    clip1.wav   (or .flac, .mp3, .ogg)
    clip1.txt   ← text prompt for clip1
    clip2.wav
    clip2.txt
    ...

Saves .npy files for latents and .json files for metadata, compatible with train_lora.py --encoded_dir.

Usage (CLI args):
  uv run python scripts/pre_encode_dataset.py --model same-s --data_dir ./my_data --output_path ./latents_out
  uv run python scripts/pre_encode_dataset.py --model same-l --data_dir ./my_data --output_path ./latents_out --batch_size 4

Usage (dataset config):
  uv run python scripts/pre_encode_dataset.py --dataset_config stable_audio_3/configs/dataset_configs/dataset2preencoding/local_babyslakh.json

Sanity check — decode the first few encoded items back to audio (source/decoded pairs,
plus every control stream) into <output_path>/_sanity_check/ so you can listen to them:
  uv run python scripts/pre_encode_dataset.py --dataset_config ... --sanity_check_samples 3

Config format (mirrors existing dataset JSON configs):
  {
    "dataset_type": "audio_dir",
    "output_path": "/path/to/output/latents/",   ← required if --output_path not given
    "datasets": [
      {
        "id": "my_dataset",
        "path": "/path/to/audio/files/",          ← source audio; overridden by --data_dir
        "custom_metadata_module": "path/to/custom_md.py"   ← optional; replaces default caption fn
      }
    ],
    "model": "same-l",           ← optional overrides for any CLI arg
    "sanity_check_samples": 3,   ← decode this many items back to audio for a listening check
    "batch_size": 1,
    "sample_size": 12582912,
    "model_half": false,
    "pad": false
  }

  When multiple datasets are listed, each is encoded into output_path/<dataset_id>/.
  CLI args always take precedence over config file values.

Augmentation — write several differently transposed / differently paced copies of each track,
so a frozen-latent dataset still shows the model a range of keys and tempos:
  uv run python scripts/pre_encode_dataset.py --dataset_config ... --augment_variants 4

  Variant 0 is the unaugmented pass; variants 1..N-1 each draw a time-stretch rate from
  U(0.9, 1.1) and a pitch shift from U(-2, +2) semitones (see --augment_* flags, all of which
  are also accepted as config keys). Both are applied to the target AND to every control, with
  the same values, so a variant is a transposition of the whole arrangement and the streams
  stay frame-aligned. --augment_pitch_scope controls holds the target at its original pitch
  while still transposing the controls. Each variant is a fresh pass over the dataset, so
  anything stochastic in the custom_metadata_module (the streamgen stem submix, for one) is
  re-rolled per variant as well.
"""

import argparse
import gc
import importlib.util
import json
import os
import random
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F

from stable_audio_3 import AutoencoderModel
from stable_audio_3.model_configs import ae_models
from stable_audio_3.data.augmentation import (
    augment_padded_clip,
    sample_augmentation_params,
)
from stable_audio_3.data.dataset import (
    LocalDatasetConfig,
    SampleDataset,
    collation_fn,
)


def caption_metadata_fn(info, _audio):
    """Default metadata fn: reads a .txt sidecar file as the prompt."""
    txt = Path(info["path"]).with_suffix(".txt")
    if not txt.exists():
        return {"__reject__": True}
    return {"prompt": txt.read_text().strip()}


def load_custom_metadata_module(module_path: str) -> ModuleType:
    """Dynamically import a custom_metadata module from a file path."""
    spec = importlib.util.spec_from_file_location("_custom_metadata", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_sanity_wavs(ae, out_dir, latent_id, source_audio, latent, control_audio, control_latents):
    """Round-trip one encoded item back to audio so it can be listened to.

    Writes the exact tensor that went into the encoder next to the decode of the
    latent that came out, plus the same pair for every control stream. This is
    the listening counterpart to scripts/check_streamgen_alignment.py.
    """
    os.makedirs(out_dir, exist_ok=True)
    sr = ae.sample_rate

    def save(suffix, audio, length=None):
        audio = audio.detach().float().cpu()
        if length is not None:
            audio = audio[:, :length]
        torchaudio.save(os.path.join(out_dir, f"{latent_id}_{suffix}.wav"), audio.clamp(-1, 1), sr)

    with torch.no_grad():
        decoded = ae.decode(latent.unsqueeze(0)).squeeze(0)
    save("decoded", decoded)
    # Source is padded to sample_size; trim it to what the (possibly truncated) latent covers.
    save("source", source_audio, length=decoded.shape[-1])

    for key, ctrl_latent in control_latents.items():
        with torch.no_grad():
            ctrl_decoded = ae.decode(ctrl_latent.unsqueeze(0)).squeeze(0)
        save(f"control_{key}_decoded", ctrl_decoded)
        save(f"control_{key}_source", control_audio[key], length=ctrl_decoded.shape[-1])


def latent_id_for(nb, i, variant, variants):
    """Name one encoded item. Variant suffixes only appear once there is more than one."""
    base = f"{nb:06d}{i:04d}"
    return f"{base}_v{variant}" if variants > 1 else base


def augment_item(audio, md, control_keys, params, pitch_controls_only):
    """Apply one augmentation roll to a sample's target audio and all of its controls.

    The same time-stretch rate goes to every stream, which is what keeps them frame-aligned,
    and by default the same pitch shift does too — a variant is a transposition of the whole
    arrangement. ``pitch_controls_only`` holds the target at its original pitch instead; see
    the note on --augment_pitch_scope for when that is the one you want.

    Controls are stretched over the *target's* valid region rather than their own. Their
    latents are cropped to the target's length when they are written anyway, so measuring
    from the target is what guarantees the two land on the same frame grid.

    Mutates ``md`` in place (control audio, padding mask, ``seconds_total``) and returns the
    augmented target audio, padded back to its original width.
    """
    total_length = audio.shape[-1]
    valid_length = int(md["padding_mask"][0].sum().item())

    audio, new_valid = augment_padded_clip(
        audio,
        valid_length,
        params,
        apply_pitch=not pitch_controls_only,
        total_length=total_length,
    )
    for key in control_keys:
        md[key], _ = augment_padded_clip(
            md[key], valid_length, params, apply_pitch=True, total_length=total_length
        )

    mask = torch.zeros_like(md["padding_mask"][0])
    mask[:new_valid] = 1
    md["padding_mask"] = [mask]
    md["seconds_total"] = md["seconds_total"] / params.rate
    md["augmentation"] = {
        **params.as_dict(),
        "pitch_scope": "controls" if pitch_controls_only else "all",
    }
    return audio


def encode_dataset(ae, data_dir, output_path, custom_metadata_fn, args):
    """Encode all audio in data_dir and write latents to output_path."""
    dataset = SampleDataset(
        [
            LocalDatasetConfig(
                id="train", # dead id, shouldn't make a difference downstream
                path=data_dir,
                custom_metadata_fn=custom_metadata_fn,
            )
        ],
        sample_size=args.sample_size,
        sample_rate=ae.sample_rate,
        force_channels="stereo",
        # PadCrop_Normalized_T draws a fresh offset on every call, and extra `__audio__`
        # tensors (e.g. the streamgen accompaniment) are cropped in a separate call from
        # the main audio. A deterministic offset is what keeps them time-aligned.
        random_crop=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        drop_last=False,
        collate_fn=collation_fn,
    )

    os.makedirs(output_path, exist_ok=True)
    device = next(ae.autoencoder.parameters()).device

    sanity_dir = args.sanity_check_dir or os.path.join(output_path, "_sanity_check")

    control_keys = list(args.controls or [])
    variants = max(1, args.augment_variants or 1)
    rate_range = tuple(args.augment_time_stretch)

    if variants > 1:
        pitched = "target + controls" if not args.augment_pitch_controls_only else "controls only"
        print(
            f"Augmentation: {variants} variants (0 unaugmented), "
            f"rate U{rate_range}, pitch U(-{args.augment_pitch_semitones}, "
            f"+{args.augment_pitch_semitones}) st applied to {pitched}, seed {args.augment_seed}"
        )

    silence_path = os.path.join(output_path, "silence.npy")
    if not os.path.exists(silence_path):
        print("Saving silence latent")
        silence_audio = torch.zeros(
            1, ae.autoencoder.io_channels, args.sample_size, device=device
        )
        if args.model_half:
            silence_audio = silence_audio.half()
        with torch.no_grad():
            silence_latent = ae.encode(silence_audio, ae.sample_rate)
        np.save(silence_path, silence_latent.cpu().numpy())

    # Each variant is a full pass over the dataset writing a differently pitched/paced copy.
    # The pass also re-runs the custom metadata fn, so anything stochastic in there (the
    # streamgen submix picks a random stem subset and levels) is re-rolled per variant too.
    for variant in range(variants):
        if variants > 1:
            print(f"=== Augmentation variant {variant}/{variants - 1}"
                  f"{' (unaugmented)' if variant == 0 else ''} ===")
        # Reset per variant so a listening check covers every variant, not just the first.
        sanity_remaining = args.sanity_check_samples or 0

        for nb, (audio, metadata) in enumerate(loader):
            print(f"Processing batch {nb}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            audio = audio.to(device)

            for key in control_keys:
                missing = [i for i, md in enumerate(metadata) if key not in md]
                if missing:
                    raise KeyError(
                        f"Control '{key}' missing from metadata for {len(missing)} item(s) in batch {nb}. "
                        f"The custom_metadata_module must return it under '__audio__'."
                    )
                for md in metadata:
                    md[key] = md[key].to(device)

            latent_ids = [latent_id_for(nb, i, variant, variants) for i in range(audio.shape[0])]

            if variant > 0:
                for i, md in enumerate(metadata):
                    # Seeded per item rather than from a running stream, so re-encoding a
                    # subset of the data reproduces the same rolls it got the first time.
                    params = sample_augmentation_params(
                        variant,
                        random.Random(f"{args.augment_seed}:{data_dir}:{latent_ids[i]}"),
                        max_semitones=args.augment_pitch_semitones,
                        rate_range=rate_range,
                    )
                    audio[i] = augment_item(
                        audio[i], md, control_keys, params, args.augment_pitch_controls_only
                    )
                    print(f"  [{latent_ids[i]}] rate {params.rate:.3f}, {params.semitones:+.2f} st")

            if args.model_half:
                audio = audio.half()

            latents = ae.encode(audio, ae.sample_rate)

            # Control signals are encoded with the same autoencoder, so they land on the same
            # frame grid as the latents and can be cropped in lockstep at training time.
            control_latents = None
            if control_keys:
                control_parts = []
                for key in control_keys:
                    control_audio = torch.stack([md[key] for md in metadata], dim=0)
                    if args.model_half:
                        control_audio = control_audio.half()
                    control_parts.append(ae.encode(control_audio, ae.sample_rate))
                control_dims = [p.shape[1] for p in control_parts]
                # Fused along the channel axis; the dataset config's controls/controls_dim
                # split it back out in this same order.
                control_latents = torch.cat(control_parts, dim=1)

            for i, latent in enumerate(latents):
                latent_np = latent.cpu().numpy()
                latent_id = latent_ids[i]

                md = dict(metadata[i])

                # Control audio is multi-minute raw waveform; it has already been encoded into
                # the sidecar and must not reach the JSON metadata dump below.
                for key in control_keys:
                    md.pop(key, None)
                padding_mask = (
                    F.interpolate(
                        md["padding_mask"][0].unsqueeze(0).unsqueeze(1).float(),
                        size=latent_np.shape[-1],
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .int()
                )
                if not args.pad:
                    padding_np = padding_mask.cpu().numpy()
                    valid_indices = np.where(padding_np == 1)[0]
                    if len(valid_indices) > 0:
                        valid_length = valid_indices[-1] + 1
                        latent_np = latent_np[:, :valid_length]
                        padding_mask = padding_mask[:valid_length]

                np.save(os.path.join(output_path, f"{latent_id}.npy"), latent_np)

                if control_latents is not None:
                    control_np = control_latents[i].cpu().numpy()[:, : latent_np.shape[-1]]
                    np.save(os.path.join(output_path, f"{latent_id}_controls.npy"), control_np)
                    md["controls_dim"] = [
                        control_latents.shape[1] // len(control_keys)
                    ] * len(control_keys)

                md["padding_mask"] = padding_mask.cpu().numpy().tolist()
                for k, v in md.items():
                    if isinstance(v, torch.Tensor):
                        md[k] = v.cpu().numpy().tolist()

                with open(os.path.join(output_path, f"{latent_id}.json"), "w") as f:
                    json.dump(md, f)

                if sanity_remaining > 0:
                    sanity_controls = {}
                    if control_latents is not None:
                        ind = 0
                        for key, dim in zip(control_keys, control_dims):
                            sanity_controls[key] = control_latents[i][ind : ind + dim, : latent_np.shape[-1]]
                            ind += dim
                    write_sanity_wavs(
                        ae,
                        sanity_dir,
                        latent_id,
                        audio[i],
                        torch.from_numpy(latent_np).to(device),
                        {key: metadata[i][key] for key in sanity_controls},
                        sanity_controls,
                    )
                    sanity_remaining -= 1
                    if sanity_remaining == 0:
                        print(f"Wrote sanity-check wavs to {sanity_dir}")
                        print(f"  Listen to them: uv run python scripts/make_listening_page.py --dir {sanity_dir}")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def merge_config_into_args(args, cfg: dict, parser: argparse.ArgumentParser):
    """
    Populate args from cfg for any value the user did not explicitly set on the CLI.
    Top-level scalar keys in the config map directly to arg names.
    CLI-supplied values always win.
    """
    cli_supplied = {a for a in vars(args) if getattr(args, a) is not None and getattr(args, a) is not False}

    scalar_keys = (
        "model",
        "batch_size",
        "sample_size",
        "model_half",
        "pad",
        "output_path",
        "controls",
        "sanity_check_samples",
        "sanity_check_dir",
        "augment_variants",
        "augment_pitch_semitones",
        "augment_time_stretch",
        "augment_pitch_scope",
        "augment_seed",
    )
    for key in scalar_keys:
        if key in cfg and key not in cli_supplied:
            setattr(args, key, cfg[key])


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = AutoencoderModel.from_pretrained(args.model, device=str(device))
    if args.model_half:
        ae.autoencoder = ae.autoencoder.half()
    cfg = load_config(args.dataset_config) if args.dataset_config else None

    if cfg is not None:
        datasets_cfg = cfg.get("datasets", [])
        if not datasets_cfg:
            raise ValueError("dataset_config has no 'datasets' entries")

        multiple = len(datasets_cfg) > 1

        for ds_entry in datasets_cfg:
            ds_id = ds_entry.get("id", "dataset")
            data_dir = args.data_dir or ds_entry.get("path")
            if not data_dir:
                raise ValueError(f"No source path for dataset '{ds_id}'; set 'path' in config or pass --data_dir")

            base_out = args.output_path or cfg.get("output_path")
            if not base_out:
                raise ValueError("No output_path; set it in the config or pass --output_path")
            output_path = os.path.join(base_out, ds_id) if multiple else base_out

            module_path = ds_entry.get("custom_metadata_module")
            if module_path:
                mod = load_custom_metadata_module(module_path)
                metadata_fn = mod.get_custom_metadata
                print(f"Loaded custom metadata fn from {module_path}")
            else:
                metadata_fn = caption_metadata_fn

            print(f"Encoding dataset '{ds_id}': {data_dir} → {output_path}")
            encode_dataset(ae, data_dir, output_path, metadata_fn, args)
    else:
        if not args.data_dir or not args.output_path:
            raise ValueError("--data_dir and --output_path are required when --dataset_config is not provided")
        encode_dataset(ae, args.data_dir, args.output_path, caption_metadata_fn, args)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-encode audio dataset to latents")
    parser.add_argument(
        "--dataset_config",
        help="Path to a dataset JSON config (same format as dataset_configs/). "
             "Supplies data_dir, output_path, custom_metadata_module, and optional scalar overrides. "
             "CLI args take precedence.",
    )
    parser.add_argument("--model", choices=list(ae_models), default="same-l")
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Folder with audio files (overrides datasets[i].path from config)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Folder to write .npy/.json latent pairs (overrides output_path from config)",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=12582912,  # 380s at 44.1kHz, 2 channels
        help="Audio samples to pad/crop to (default ~380s at 44.1kHz)",
    )
    parser.add_argument(
        "--model_half", action="store_true", help="Run autoencoder in fp16"
    )
    parser.add_argument(
        "--pad", action="store_true", help="Pad audio samples to --sample_size"
    )
    parser.add_argument(
        "--controls",
        nargs="*",
        default=None,
        help=(
            "Metadata keys holding extra audio (returned by the custom_metadata_module under "
            "'__audio__') to encode into a fused {id}_controls.npy sidecar, e.g. streamgen_audio. "
            "The order given here is the channel order in the sidecar and must match "
            "'controls'/'controls_dim' in the training dataset config."
        ),
    )
    parser.add_argument(
        "--sanity_check_samples",
        type=int,
        default=None,
        help=(
            "Decode this many encoded items back to audio and write them alongside the "
            "source audio that was encoded, for a listening check. 0 disables."
        ),
    )
    parser.add_argument(
        "--sanity_check_dir",
        default=None,
        help="Where to write sanity-check wavs (default: <output_path>/_sanity_check)",
    )
    parser.add_argument(
        "--augment_variants",
        type=int,
        default=None,
        help=(
            "Number of copies of the dataset to write, each with its own pitch/tempo roll. "
            "Variant 0 is always unaugmented, so N=1 (the default) means no augmentation and "
            "N=4 means the original plus 3 augmented copies. Latent ids gain a _v<N> suffix "
            "as soon as N > 1."
        ),
    )
    parser.add_argument(
        "--augment_pitch_semitones",
        type=float,
        default=None,
        help="Pitch shift is drawn from U(-x, +x) semitones (default 2.0).",
    )
    parser.add_argument(
        "--augment_time_stretch",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=(
            "Time-stretch rate is drawn from U(MIN, MAX); >1 is faster (default 0.9 1.1). "
            "The same rate is applied to the target and every control, or they stop being "
            "time-aligned."
        ),
    )
    parser.add_argument(
        "--augment_pitch_scope",
        choices=("controls", "all"),
        default=None,
        help=(
            "Which streams get the pitch shift. 'all' (default) transposes the target and the "
            "controls by the same interval, so a variant is the whole arrangement in a new key. "
            "'controls' transposes the controls only, holding the target at its original pitch: "
            "that decouples the target's tuning from the accompaniment's key, which matters if "
            "the model starts reading one off the other. Time stretch always applies to "
            "everything; it has to, or the streams desync."
        ),
    )
    parser.add_argument(
        "--augment_seed",
        type=int,
        default=None,
        help="Seed for the per-item pitch/tempo rolls (default 0).",
    )
    args = parser.parse_args()

    if args.dataset_config:
        cfg = load_config(args.dataset_config)
        merge_config_into_args(args, cfg, parser)

    # Resolved after the config merge so a JSON config can supply them; `None` until here
    # is how merge_config_into_args tells "unset" from "explicitly asked for".
    if args.augment_variants is None:
        args.augment_variants = 1
    if args.augment_pitch_semitones is None:
        args.augment_pitch_semitones = 2.0
    if args.augment_time_stretch is None:
        args.augment_time_stretch = [0.9, 1.1]
    if args.augment_pitch_scope is None:
        args.augment_pitch_scope = "all"
    args.augment_pitch_controls_only = args.augment_pitch_scope != "all"
    if args.augment_seed is None:
        args.augment_seed = 0

    if not args.pad and args.batch_size > 1:
        parser.error(
            "padding is required for batch_size > 1; pass --pad or use --batch_size 1"
        )

    main(args)
