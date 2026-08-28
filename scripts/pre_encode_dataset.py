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


def encode_dataset(ae, data_dir, output_path, custom_metadata_fn, args):
    """Encode all audio in data_dir and write latents to output_path."""
    dataset = SampleDataset(
        [
            LocalDatasetConfig(
                id="train",
                path=data_dir,
                custom_metadata_fn=custom_metadata_fn,
            )
        ],
        sample_size=args.sample_size,
        sample_rate=ae.sample_rate,
        force_channels="stereo",
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

        for i, latent in enumerate(latents):
            latent_np = latent.cpu().numpy()
            latent_id = f"{nb:06d}{i:04d}"

            md = dict(metadata[i])
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

            md["padding_mask"] = padding_mask.cpu().numpy().tolist()
            for k, v in md.items():
                if isinstance(v, torch.Tensor):
                    md[k] = v.cpu().numpy().tolist()

            with open(os.path.join(output_path, f"{latent_id}.json"), "w") as f:
                json.dump(md, f)


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

    scalar_keys = ("model", "batch_size", "sample_size", "model_half", "pad", "output_path")
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
    args = parser.parse_args()

    if args.dataset_config:
        cfg = load_config(args.dataset_config)
        merge_config_into_args(args, cfg, parser)

    if not args.pad and args.batch_size > 1:
        parser.error(
            "padding is required for batch_size > 1; pass --pad or use --batch_size 1"
        )

    main(args)
