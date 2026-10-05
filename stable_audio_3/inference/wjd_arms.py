"""Finding and loading the WJD finetune arms (experiments 3.3 / 3.4) for inference.

An *arm* is one finetune recipe: ``rms``, ``tria_fixed``, ``tria_ema``, ``audio``, ``base``.
Its model config and checkpoint group are read from ``sbatch/03_3_finetune_wjd_<arm>.sbatch``
so the table lives in one place, and its checkpoint is found the way the sbatch continues a
run: the most recent ``last.ckpt`` under ``<save_base>/<group>/<arm>/sao-3/*/checkpoints/``.

Shared by ``scripts/wjd/listen_wjd_arms.py`` and the Gradio interface
(``stable_audio_3/interface/wjd_control.py``).
"""

from __future__ import annotations

import json
import re
import time
import zipfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARMS = ["audio", "rms", "tria_fixed", "tria_ema"]
DEFAULT_SAVE_BASE = "/data/scratch-fast/snnithya/sao-3/ft_checkpoints"
WANDB_PROJECT = "sao-3"


def read_sbatch_arm(name, repo=REPO):
    """``(MODEL_CONFIG, GROUP)`` of ``sbatch/03_3_finetune_wjd_<name>.sbatch``."""
    path = Path(repo) / "sbatch" / f"03_3_finetune_wjd_{name}.sbatch"
    if not path.exists():
        raise FileNotFoundError(f"No sbatch file for arm {name!r}: {path}")
    text = path.read_text()
    fields = {}
    for key in ("MODEL_CONFIG", "GROUP"):
        m = re.search(rf"^{key}=(\S+)", text, re.MULTILINE)
        if not m:
            raise ValueError(f"{path} does not set {key}")
        fields[key] = m.group(1)
    return fields["MODEL_CONFIG"], fields["GROUP"]


def complete_safetensors(export_dir):
    """Newest export that is not still being written: a half-written file shows up with a
    fraction of its siblings' size (job 2539812 left a 714 MB one next to 1.4 GB ones)."""
    files = sorted(Path(export_dir).glob("model_step_*.safetensors"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    full = max(p.stat().st_size for p in files)
    for p in reversed(files):
        if p.stat().st_size >= 0.98 * full:
            return p
    return None


def is_complete_ckpt(path) -> bool:
    """True when a Lightning .ckpt can be opened as a zip archive.

    A running job rewrites last.ckpt in place (3.4 GB, about 10 s); torch.load on a file caught
    mid-write fails with "failed finding central directory", which is exactly what this check
    reads (the directory is the last thing written), so it costs one seek rather than a full read.
    """
    try:
        with zipfile.ZipFile(path):
            return True
    except (zipfile.BadZipFile, OSError):
        return False


def discover_checkpoint(name, group, save_base=DEFAULT_SAVE_BASE, prefer="ckpt"):
    """Latest checkpoint of an arm, following sbatch/03_3_finetune_wjd_common.sh's rule for
    which run a resubmit continues (the newest last.ckpt by mtime).

    Only complete files count: while the job is writing last.ckpt, the newest complete
    checkpoint of the same run is returned instead (last-v1.ckpt, Lightning's second copy
    written seconds later, or an epoch=...-step=... file), so a click in the UI never lands on
    a half-written file. Otherwise last.ckpt itself is returned, as the sbatch rule says, even
    though last-v1.ckpt usually carries the later mtime.
    """
    arm_dir = Path(save_base) / group / name
    last = sorted(arm_dir.glob(f"{WANDB_PROJECT}/*/checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime)
    newest = None
    if last:
        if is_complete_ckpt(last[-1]):
            newest = last[-1]
        else:
            run_dir = last[-1].parent
            in_run = sorted(run_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            newest = next((p for p in in_run if is_complete_ckpt(p)), None)
    export = complete_safetensors(arm_dir / "safetensors_exports")
    candidates = {"ckpt": newest, "safetensors": export}
    order = [prefer] + [k for k in ("ckpt", "safetensors") if k != prefer]
    for kind in order:
        if candidates[kind] is not None:
            return str(candidates[kind])
    raise FileNotFoundError(f"No checkpoint for arm {name!r} under {arm_dir}")


def resolve_arm(name, save_base=DEFAULT_SAVE_BASE, prefer="ckpt", repo=REPO):
    """``(model_config_path, ckpt_path)`` of an arm by discovery."""
    model_config, group = read_sbatch_arm(name, repo)
    return model_config, discover_checkpoint(name, group, save_base, prefer)


def load_arm(model_config_path, ckpt_path, device, use_ema=True, log=print, info=None):
    """Pretrained small-music, then the arm's finetuned weights on top.

    Starting from the pretrained weights (as scripts/train_finetune.py does) means the
    pretransform and the frozen conditioner are right even if the checkpoint were to lack
    them. A Lightning .ckpt carries the training wrapper's keys ('diffusion.' prefix) and,
    with use_ema, the EMA shadow of diffusion.model replaces the raw weights when the
    checkpoint has one. Returns (model, model_config, step). The DiT
    is bf16 on ``device``; the autoencoder stays fp32 (bf16 costs ~11 dB above 10 kHz).

    A checkpoint written without EMA (``train_finetune.py`` without ``--use_ema``, as the WJD
    jobs run) holds only the raw weights; those are loaded and ``info["weights"]`` says so
    (``"ema"`` or ``"raw"``) when a dict is passed.
    """
    from safetensors.torch import load_file

    from stable_audio_3.factory import create_diffusion_cond_from_config
    from stable_audio_3.loading_utils import copy_state_dict
    from stable_audio_3.model_configs import models

    with open(model_config_path) as f:
        model_config = json.load(f)
    model = create_diffusion_cond_from_config(model_config)

    _, pretrained_ckpt = models["small-music"].resolve()
    copy_state_dict(model, load_file(pretrained_ckpt))

    step = None
    if str(ckpt_path).endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
        m = re.search(r"model_step_(\d+)", Path(ckpt_path).name)
        step = int(m.group(1)) if m else None
        if use_ema:
            log("  note: a .safetensors export holds the raw (non-EMA) weights")
    else:
        ckpt = _torch_load_retry(ckpt_path, log)
        step = ckpt.get("global_step")
        raw = ckpt.get("state_dict", ckpt)
        ema_prefix = "diffusion_ema.ema_model."
        has_ema = any(k.startswith(ema_prefix) for k in raw)
        if use_ema and not has_ema:
            log(f"  note: {ckpt_path} holds no EMA weights; using the raw ones")
            use_ema = False
        state_dict = {}
        for k, v in raw.items():
            if k.startswith(ema_prefix):
                if use_ema:
                    state_dict["model." + k[len(ema_prefix):]] = v
            elif k.startswith("diffusion_ema."):
                continue
            elif k.startswith("diffusion."):
                key = k[len("diffusion."):]
                if use_ema and key.startswith("model."):
                    continue
                state_dict[key] = v
        del ckpt, raw
    copy_state_dict(model, state_dict)
    del state_dict

    model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    if model.pretransform is not None:
        model.pretransform.to(torch.float32)
    if info is not None:
        info["weights"] = "ema" if use_ema else "raw"
        info["step"] = step
    return model, model_config, step


def _torch_load_retry(ckpt_path, log=print, attempts=4, wait=15.0):
    """torch.load that waits out a checkpoint being rewritten under it (see is_complete_ckpt)."""
    for i in range(attempts):
        try:
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except RuntimeError as e:
            if "central directory" not in str(e) or i == attempts - 1:
                raise RuntimeError(
                    f"{ckpt_path} could not be read ({e}); if the training job is writing it, "
                    f"try again in a moment or pass an explicit --arm checkpoint"
                ) from e
            log(f"  {ckpt_path} is being written; retrying in {wait:.0f}s ({i + 1}/{attempts - 1})")
            time.sleep(wait)


def share_frozen_parts(model, donor):
    """Point ``model`` at ``donor``'s autoencoder and text conditioner and drop its own copies.

    Both are frozen in every WJD finetune (``--freeze_conditioner``; the pretransform never
    trains), so the arms hold identical weights there: one copy on the GPU is enough when
    several arms are resident at once.
    """
    if donor is model:
        return model
    model.pretransform = donor.pretransform
    model.conditioner = donor.conditioner
    return model
