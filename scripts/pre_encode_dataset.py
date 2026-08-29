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
"""

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F

from stable_audio_3 import AutoencoderModel
from stable_audio_3.model_configs import ae_models
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
    sanity_remaining = args.sanity_check_samples or 0

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

    for nb, (audio, metadata) in enumerate(loader):
        print(f"Processing batch {nb}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        audio = audio.to(device)
        if args.model_half:
            audio = audio.half()

        latents = ae.encode(audio, ae.sample_rate)

        # Control signals are encoded with the same autoencoder, so they land on the same
        # frame grid as the latents and can be cropped in lockstep at training time.
        control_latents = None
        if args.controls:
            control_parts = []
            for key in args.controls:
                missing = [i for i, md in enumerate(metadata) if key not in md]
                if missing:
                    raise KeyError(
                        f"Control '{key}' missing from metadata for {len(missing)} item(s) in batch {nb}. "
                        f"The custom_metadata_module must return it under '__audio__'."
                    )
                control_audio = torch.stack([md[key] for md in metadata], dim=0).to(device)
                if args.model_half:
                    control_audio = control_audio.half()
                control_parts.append(ae.encode(control_audio, ae.sample_rate))
            control_dims = [p.shape[1] for p in control_parts]
            # Fused along the channel axis; the dataset config's controls/controls_dim
            # split it back out in this same order.
            control_latents = torch.cat(control_parts, dim=1)

        for i, latent in enumerate(latents):
            latent_np = latent.cpu().numpy()
            latent_id = f"{nb:06d}{i:04d}"

            md = dict(metadata[i])

            # Control audio is multi-minute raw waveform; it has already been encoded into
            # the sidecar and must not reach the JSON metadata dump below.
            for key in args.controls or []:
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
                    control_latents.shape[1] // len(args.controls)
                ] * len(args.controls)

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
                    for key, dim in zip(args.controls, control_dims):
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
    args = parser.parse_args()

    if args.dataset_config:
        cfg = load_config(args.dataset_config)
        merge_config_into_args(args, cfg, parser)

    if not args.pad and args.batch_size > 1:
        parser.error(
            "padding is required for batch_size > 1; pass --pad or use --batch_size 1"
        )

    main(args)
