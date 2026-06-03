"""
Quick dataset sanity-check script.

Accepts the same JSON config format used by the training scripts.  Instantiates
PreEncodedDataset or SampleDataset, iterates through a few samples and prints
what each one yields — useful for verifying latent loading and
custom_metadata_fn logic before starting a full training run.

Config format (pre-encoded latents):
  {
    "dataset_type": "pre_encoded",
    "datasets": [
      {
        "id": "my-dataset",
        "path": "/path/to/latents",
        "custom_metadata_module": "path/to/my_metadata.py"   // optional
      }
    ],
    "latent_crop_length": 256,   // optional — derived from duration if omitted
    "random_crop": true
  }

Config format (raw audio):
  {
    "dataset_type": "sample",
    "datasets": [
      {
        "id": "my-dataset",
        "path": "/path/to/audio",
        "custom_metadata_module": "path/to/my_metadata.py"   // optional
      }
    ],
    "sample_rate": 44100,
    "sample_size": 131072
  }

Usage:
  uv run python scripts/debug_dataset.py --config my_dataset.json
  uv run python scripts/debug_dataset.py --config my_dataset.json -n 20 --show_latent_stats
"""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import json
from pathlib import Path

import torch

from stable_audio_3.data.utils import build_dataset_from_config


def fmt_value(v):
    if isinstance(v, torch.Tensor):
        return f"Tensor{list(v.shape)} dtype={v.dtype}"
    if isinstance(v, list) and len(v) > 8:
        return f"[{v[0]}, {v[1]}, ..., {v[-1]}] (len={len(v)})"
    if isinstance(v, str) and len(v) > 120:
        return v[:120] + "…"
    return repr(v)


def main():
    p = argparse.ArgumentParser(description="Dataset loading sanity check")
    p.add_argument(
        "--config",
        required=True,
        help="Path to a dataset JSON config file",
    )
    p.add_argument(
        "-n", "--num_samples",
        type=int,
        default=5,
        help="Number of samples to inspect (default: 5)",
    )
    p.add_argument(
        "--show_latent_stats",
        action="store_true",
        help="Print min/max/mean of loaded latents/audio tensors",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Derive latent_crop_length / sample_size from this duration (seconds) "
             "if not set in the config",
    )
    p.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Sample rate used to derive crop lengths when not in the config (default: 44100)",
    )
    p.add_argument(
        "--ds_ratio",
        type=int,
        default=2048,
        help="Pretransform downsampling ratio used to derive latent_crop_length (default: 2048)",
    )
    args = p.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        p.error(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    print(f"Config: {config_path}")
    print(f"  dataset_type : {cfg.get('dataset_type', 'pre_encoded')}")
    print(f"  datasets     : {[d['id'] for d in cfg.get('datasets', [])]}")

    dataset = build_dataset_from_config(
        cfg,
        sample_rate=args.sample_rate,
        ds_ratio=args.ds_ratio,
        duration=args.duration,
    )

    print(f"\nDataset size: {len(dataset)} samples")
    print("=" * 60)

    n = min(args.num_samples, len(dataset))
    rejected = 0
    errors = 0

    for i in range(n):
        print(f"\n--- Sample {i} ---")
        try:
            data, info = dataset[i]

            if args.show_latent_stats and isinstance(data, torch.Tensor):
                print(f"  data shape : {list(data.shape)}")
                print(f"  data min   : {data.min().item():.4f}")
                print(f"  data max   : {data.max().item():.4f}")
                print(f"  data mean  : {data.mean().item():.4f}")

            for k, v in info.items():
                print(f"  {k:30s}: {fmt_value(v)}")

            if info.get("__reject__"):
                print("  *** REJECTED by custom_metadata_fn ***")
                rejected += 1

        except Exception as e:
            import traceback
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            errors += 1

    print("\n" + "=" * 60)
    print(f"Inspected {n} samples — {rejected} rejected, {errors} errors")


if __name__ == "__main__":
    main()
