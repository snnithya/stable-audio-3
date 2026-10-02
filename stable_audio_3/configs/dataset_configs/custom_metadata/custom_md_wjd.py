"""Custom metadata for the WJD stem mirror: prompt from the instrument, drums as the control.

Called on a *target* stem in the tree ``scripts/wjd/make_stem_mirror.py`` writes:

    <split>/tracks/bass/<Track>/bass.flac       <- what this fn is called on (one item each)
    <split>/tracks/other/<Track>/other.flac
    <split>/tracks/piano/<Track>/piano.flac
    <split>/tracks/drums/<Track>/drums.flac     <- the condition, found from the target's path
    <split>/meta/<Track>.json                   <- prompts, lineup, JSD segments

Returns the prompt the mirror script decided on for that stem (short instrument names, e.g.
``"upright bass"`` or ``"trumpet, tenor saxophone"`` for ``other``) plus the drum control in
one of two forms, chosen by the ``WJD_CONTROL_MODE`` environment variable:

    rms    (default)  ``__features__: {"drums_rms": [1, T_frames]}`` — the causal per-frame RMS
                      envelope of the drum stem at the latent rate, in [0, 1]. Pre-encode with
                      ``--features drums_rms``; train with ``controls: ["drums_rms"],
                      controls_dim: [1]``.  (stable_audio_3/data/features.py)
    audio             ``__audio__: {"drums_audio": [2, T]}`` — the drum waveform, VAE-encoded by
                      the pre-encode script into a 256-ch latent, as the Slakh streamgen
                      accompaniment is. Pre-encode with ``--controls drums_audio``.
    both              both of the above; pass both flags.

An environment variable rather than a config key because the module is loaded by path and
has no other channel from the dataset config; set it in the sbatch script next to the flags.

Rejections (``__reject__`` + reason, so they land in ``_skipped.json`` at pre-encode time):
no ``meta/`` entry, no drums file, or drums silent (RMS floor) over the target's valid
window. The target's own level is the pre-encode script's job.

Alignment: the drums are read from sample 0 and cut to the same window as the target, so
this only holds with ``random_crop=False`` (the pre-encode script's setting), exactly as for
``custom_md_slakh_streamgen.py``.
"""

import json
import os
from pathlib import Path

import torch
import torchaudio

from stable_audio_3.data.features import LATENT_HOP, rms_envelope_control
from stable_audio_3.data.utils import DEFAULT_SILENCE_THRESHOLD_DB, is_silent

CONDITION = "drums"
CONTROL_MODE = os.environ.get("WJD_CONTROL_MODE", "rms")
if CONTROL_MODE not in ("rms", "audio", "both"):
    raise ValueError(f"WJD_CONTROL_MODE must be rms | audio | both, got {CONTROL_MODE!r}")

FEATURE_KEY = "drums_rms"
AUDIO_KEY = "drums_audio"
SILENCE_THRESHOLD_DB = DEFAULT_SILENCE_THRESHOLD_DB


def locate(target_path):
    """(track, stem, drums path, meta path) for a target stem path in the mirror tree."""
    target_path = Path(target_path)
    stem = target_path.stem
    track = target_path.parent.name
    tracks_root = target_path.parent.parent.parent  # <split>/tracks
    drums = tracks_root / CONDITION / track / f"{CONDITION}.flac"
    meta = tracks_root.parent / "meta" / f"{track}.json"
    return track, stem, drums, meta


def _to_stereo(audio):
    if audio.shape[0] == 1:
        return audio.repeat(2, 1)
    if audio.shape[0] > 2:
        return audio[:2]
    return audio


def load_drums(drums_path, sample_rate, total_length):
    """Drum stem as stereo at ``sample_rate``, cut/zero-padded to ``total_length`` samples from 0."""
    audio, sr = torchaudio.load(str(drums_path))
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    audio = _to_stereo(audio)
    if audio.shape[1] > total_length:
        audio = audio[:, :total_length]
    elif audio.shape[1] < total_length:
        audio = torch.nn.functional.pad(audio, (0, total_length - audio.shape[1]))
    return audio


def get_custom_metadata(info, audio):
    track, stem, drums_path, meta_path = locate(info["path"])

    if not meta_path.exists():
        return {"__reject__": True, "__reject_reason__": "no meta json for track"}
    meta = json.loads(meta_path.read_text())

    prompt = meta.get("prompts", {}).get(stem)
    if not prompt:
        return {"__reject__": True, "__reject_reason__": f"no prompt for stem {stem}"}

    if not drums_path.exists():
        return {"__reject__": True, "__reject_reason__": "drums stem missing"}

    mask = info.get("padding_mask")
    total_length = int(mask[0].shape[-1]) if mask else audio.shape[-1]
    valid_length = int(mask[0].sum().item()) if mask else audio.shape[-1]

    drums = load_drums(drums_path, info["sample_rate"], total_length)
    if is_silent(drums[:, :valid_length], SILENCE_THRESHOLD_DB):
        return {"__reject__": True, "__reject_reason__": "drums silent over the encoded window"}

    out = {
        "prompt": prompt,
        "track": track,
        "stem": stem,
        "is_drum": False,
        "decade": meta.get("decade"),
        "lineup": meta.get("lineup"),
        "front_line": meta.get("front_line"),
        "control_mode": CONTROL_MODE,
    }
    if CONTROL_MODE in ("rms", "both"):
        out["__features__"] = {
            FEATURE_KEY: rms_envelope_control(drums, hop=LATENT_HOP, valid_samples=valid_length)
        }
    if CONTROL_MODE in ("audio", "both"):
        out["__audio__"] = {AUDIO_KEY: drums}
    return out
