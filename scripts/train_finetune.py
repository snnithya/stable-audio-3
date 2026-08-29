"""
Full fine-tuning for Stable Audio 3.

All diffusion model weights are trained (no LoRA adapters).  The pretransform
(autoencoder) is frozen by default; the conditioner can optionally be frozen too.

Dataset is specified via a JSON config file (recommended) or inline CLI flags.

JSON config format (pre-encoded latents):
  {
    "dataset_type": "pre_encoded",
    "datasets": [
      {
        "id": "my-dataset",
        "path": "/path/to/latents",
        "custom_metadata_module": "path/to/my_metadata.py"   // optional
      }
    ],
    "latent_crop_length": 256,   // optional — derived from --duration if omitted
    "random_crop": true
  }

JSON config format (raw audio):
  {
    "dataset_type": "sample",
    "datasets": [
      {
        "id": "my-dataset",
        "path": "/path/to/audio",
        "custom_metadata_module": "path/to/my_metadata.py"   // optional
      }
    ],
    "sample_rate": 44100,   // optional — taken from model if omitted
    "sample_size": 131072   // optional — derived from --duration if omitted
  }

Saves PyTorch Lightning checkpoints.  To also export the diffusion model as a
.safetensors file for inference, use the --export_safetensors flag.

Usage:
  uv run python scripts/train_finetune.py --model medium-base --dataset_config my_dataset.json --save_dir ./ft_out
  uv run python scripts/train_finetune.py --model medium-base --data_dir ./my_data --save_dir ./ft_out
  uv run python scripts/train_finetune.py --model medium-base --encoded_dir ./latents_out --save_dir ./ft_out
  uv run python scripts/train_finetune.py --model medium-base --dataset_config my_dataset.json --resume_ckpt ./ft_out/last.ckpt
"""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import itertools
import json
from pathlib import Path
import torch
import pytorch_lightning as pl

from stable_audio_3.data.dataset import collation_fn
from safetensors.torch import load_file, save_file
from stable_audio_3.loading_utils import copy_state_dict
from stable_audio_3.model_configs import models
from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3.training.diffusion import (
    DiffusionCondTrainingWrapper,
    DiffusionCondInpaintDemoCallback,
)
from stable_audio_3.data.utils import (
    build_dataset_from_config,
    build_pre_encoded_dataset,
    build_sample_dataset,
)


def load_model(model_name: str, device: torch.device, model_config_path: str = None):
    """Build the model from `model_name`'s pretrained weights.

    `model_config_path` overrides the architecture config shipped with the checkpoint,
    which is how extra conditioning (e.g. streamgen's modular local conds) gets added to
    a pretrained model. Weights are copied with copy_state_dict, which skips keys the
    checkpoint doesn't have, so newly added modules keep their initialization.
    """
    if model_name not in models:
        raise ValueError(
            f"Unknown model '{model_name}'. Valid choices: {list(models)}"
        )
    model_cfg = models[model_name]
    local_config, local_ckpt = model_cfg.resolve()
    config_path = model_config_path or local_config
    if model_config_path:
        print(f"Using model config override: {model_config_path}")
    with open(config_path) as f:
        model_config = json.load(f)
    model = create_diffusion_cond_from_config(model_config)
    copy_state_dict(model, load_file(local_ckpt))
    model.to(device=device, dtype=torch.bfloat16)
    return model, model_config



class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


class SafetensorsExportCallback(pl.Callback):
    """Export the diffusion model weights as a .safetensors file after each checkpoint."""

    def __init__(self, export_dir: str):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        step = trainer.global_step
        out_path = os.path.join(self.export_dir, f"model_step_{step:08d}.safetensors")
        state_dict = {
            k: v.cpu()
            for k, v in pl_module.diffusion.state_dict().items()
        }
        save_file(state_dict, out_path)
        print(f"Exported model weights to {out_path}")


def train(args):
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision("high")

    seed = args.seed
    pl.seed_everything(seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model(args.model, device, args.model_config)

    sample_rate = model.sample_rate
    ds_ratio = model.pretransform.downsampling_ratio
    sample_size = (int(args.duration * sample_rate) // ds_ratio) * ds_ratio

    if args.dataset_config:
        print(f"Building dataset from config: {args.dataset_config}")
        dataset = build_dataset_from_config(args.dataset_config, sample_rate, ds_ratio, args.duration)
    elif args.encoded_dir:
        dataset = build_pre_encoded_dataset(
            args.encoded_dir,
            latent_crop_length=sample_size // ds_ratio,
            random_crop=True,
            custom_metadata_module=args.custom_metadata_module,
            sample_rate=sample_rate,
            ds_ratio=ds_ratio,
        )
    else:
        dataset = build_sample_dataset(
            args.data_dir,
            sample_size=sample_size,
            sample_rate=sample_rate,
            ds_ratio=ds_ratio,
            custom_metadata_module=args.custom_metadata_module,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collation_fn,
        worker_init_fn=lambda worker_id: torch.manual_seed(seed + worker_id),
    )

    optimizer_config = {
        "diffusion": {
            "optimizer": {
                "type": "AdamW",
                "config": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "betas": [0.9, 0.95],
                },
            }
        }
    }

    # Optionally freeze sub-components before handing to the training wrapper
    if args.freeze_pretransform and model.pretransform is not None:
        model.pretransform.requires_grad_(False)
        model.pretransform.enable_grad = False

    if args.freeze_conditioner and hasattr(model, "conditioner"):
        model.conditioner.requires_grad_(False)

    # Inpainting settings come from the model config when it supplies them, so that
    # mask_type_probabilities / future_visibility can be set per-experiment alongside
    # the conditioning they belong to.
    inpainting_config = model_config.get("training", {}).get(
        "inpainting", {"mask_kwargs": {"mask_type_probabilities": [0.1, 0.8, 0.1]}}
    )
    print(f"Inpainting config: {inpainting_config}")

    training_wrapper = DiffusionCondTrainingWrapper(
        model,
        mask_loss_weight=1.0,
        mask_padding_attention=True,
        silence_extension_scale_seconds=4.0,
        pre_encoded=True,
        use_ema=args.use_ema,
        log_loss_info=False,
        optimizer_configs=optimizer_config,
        timestep_sampler="trunc_logit_normal",
        timestep_sampler_options={},
        inpainting_config=inpainting_config,
        use_effective_length_for_schedule=True,
        sample_rate=model_config.get("sample_rate", 44100),
        sample_size=model_config.get("sample_size"),
        lora_config=None,
        log_every_n_steps=args.log_every,
        ot_coupling=True,
    )

    exc_callback = ExceptionCallback()

    if args.logger == "wandb":
        logger = pl.loggers.WandbLogger(project=args.project, name=args.name, group=args.group)
        logger.watch(training_wrapper)
        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = os.path.join(
                args.save_dir,
                logger.experiment.project,
                logger.experiment.id,
                "checkpoints",
            )
        else:
            checkpoint_dir = None
    elif args.logger == "comet":
        logger = pl.loggers.CometLogger(project=args.name)
        if args.save_dir and isinstance(logger.version, str):
            checkpoint_dir = os.path.join(
                args.save_dir, logger.name, logger.version, "checkpoints"
            )
        else:
            checkpoint_dir = args.save_dir if args.save_dir else None
    elif args.logger == "csv":
        logger = pl.loggers.CSVLogger(args.save_dir)
        checkpoint_dir = args.save_dir if args.save_dir else None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every,
        dirpath=checkpoint_dir,
        save_top_k=-1,
        save_last=True,
    )

    demo_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=collation_fn,
    )

    demo_batch = next(iter(demo_dl))
    _, metadata = demo_batch
    for j in range(min(4, len(metadata))):
        md = metadata[j]
        print(
            f"Demo sample {j}: prompt={md.get('prompt', '')} seconds_total={md.get('seconds_total', '')}"
        )
    demo_dl = itertools.cycle([demo_batch])

    demo_callback = DiffusionCondInpaintDemoCallback(
        demo_every=args.demo_every,
        sample_size=model_config.get("sample_size"),
        sample_rate=model_config.get("sample_rate"),
        demo_steps=50,
        num_demos=4,
        demo_cfg_scales=[2, 4, 7],
        demo_dl=demo_dl,
    )

    callbacks = [ckpt_callback, exc_callback, demo_callback]

    if args.export_safetensors:
        export_dir = os.path.join(args.save_dir, "safetensors_exports")
        callbacks.append(SafetensorsExportCallback(export_dir))

    args_dict = vars(args)
    args_dict.update({"model_config": model_config})

    if args.logger == "comet":
        logger.log_hyperparams(args_dict)

    gradient_clip_val = args.gradient_clip_val if args.gradient_clip_val > 0 else None

    summary = pl.callbacks.ModelSummary(max_depth=2)
    callbacks.append(summary)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy="auto",
        precision="bf16-mixed",
        accumulate_grad_batches=args.accum_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        max_steps=args.steps,
        default_root_dir=args.save_dir,
        gradient_clip_val=gradient_clip_val,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        training_wrapper,
        dataloader,
        ckpt_path=args.resume_ckpt if args.resume_ckpt else None,
    )


def main():
    p = argparse.ArgumentParser(
        description="Full fine-tuning for Stable Audio 3"
    )
    p.add_argument(
        "--model",
        choices=list(models),
        default="medium-base",
        help="Pretrained model to start from",
    )
    p.add_argument(
        "--model_config",
        default=None,
        help=(
            "Path to a model config JSON that overrides the one bundled with the checkpoint. "
            "Use to add conditioning to a pretrained model, e.g. "
            "stable_audio_3/configs/model_configs/small_music_streamgen.json"
        ),
    )
    p.add_argument(
        "--dataset_config",
        default=None,
        help=(
            "Path to a dataset JSON config file (recommended). "
            "Supports multiple datasets, weights, and per-dataset custom_metadata_module. "
            "See script docstring for the config schema."
        ),
    )
    p.add_argument(
        "--data_dir",
        default=None,
        help="(Alternative to --dataset_config) Folder with audio files and matching .txt captions",
    )
    p.add_argument(
        "--encoded_dir",
        default=None,
        help="(Alternative to --dataset_config) Pre-encoded latent directory (.npy/.json pairs)",
    )
    p.add_argument(
        "--custom_metadata_module",
        default=None,
        help=(
            "(Used with --data_dir / --encoded_dir) Path to a Python file defining "
            "get_custom_metadata(info, audio_or_latents) -> dict."
        ),
    )
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    p.add_argument(
        "--weight_decay", type=float, default=0.01, help="AdamW weight decay"
    )
    p.add_argument("--steps", type=int, default=10_000, help="Total training steps")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--accum_batches", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument(
        "--duration",
        type=float,
        default=380.0,
        help="Maximum clip duration in seconds (default 380)",
    )
    p.add_argument(
        "--freeze_pretransform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the pretransform (autoencoder) weights (default: True)",
    )
    p.add_argument(
        "--freeze_conditioner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the text conditioner weights (default: False)",
    )
    p.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Maintain an exponential moving average of model weights",
    )
    p.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Gradient clipping value (0 to disable)",
    )
    p.add_argument(
        "--export_safetensors",
        action="store_true",
        default=False,
        help="Also export diffusion model weights as .safetensors at each checkpoint",
    )
    p.add_argument(
        "--resume_ckpt",
        default=None,
        help="Path to a PyTorch Lightning .ckpt to resume training from",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logger", choices=["wandb", "comet", "csv", "none"], default="wandb")
    p.add_argument("--project", type=str, default="sao-3")
    p.add_argument("--name", type=str, default="test")
    p.add_argument("--group", type=str, default="debug")
    p.add_argument("--save_dir", type=str, default="/data/scratch-fast/snnithya/sao-3/ft_checkpoints/debug")
    p.add_argument("--checkpoint_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--demo_every", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()
    if not args.dataset_config and not args.encoded_dir and not args.data_dir:
        p.error("one of --dataset_config, --data_dir, or --encoded_dir is required")
    train(args)


if __name__ == "__main__":
    main()
