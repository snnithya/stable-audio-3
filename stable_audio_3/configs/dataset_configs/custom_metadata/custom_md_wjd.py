"""Custom metadata for the WJD stem mirror: prompt from the instrument, drums as the control.

Called on a *target* stem in the tree ``scripts/wjd/make_stem_mirror.py`` writes:

    <split>/tracks/bass/<Track>/bass.flac       <- what this fn is called on (one item each)
    <split>/tracks/other/<Track>/other.flac
    <split>/tracks/piano/<Track>/piano.flac
    <split>/tracks/drums/<Track>/drums.flac     <- the condition, found from the target's path
    <split>/meta/<Track>.json                   <- prompts, lineup, JSD segments

Returns the prompt the mirror script decided on for that stem (short instrument names, e.g.
``"upright bass"`` or ``"trumpet, tenor saxophone"`` for ``other``) plus the drum control in
one or more forms, chosen by the ``WJD_CONTROL_MODE`` environment variable, a comma-separated
set of:

    rms    (default)  ``__features__: {"drums_rms": [1, T_frames]}`` — the causal per-frame RMS
                      envelope of the drum stem at the latent rate, in [0, 1]. Pre-encode with
                      ``--features drums_rms``; train with ``controls: ["drums_rms"],
                      controls_dim: [1]``.  (stable_audio_3/data/features.py)
    tria_fixed        ``__features__: {"drums_tria_fixed": [2, T_frames]}`` — the causal TRIA
                      two-band features, z-scored against the dataset statistics in
                      ``WJD_TRIA_STATS`` (default: configs/dataset_configs/features/
                      wjd_drums_tria_stats.json). Pre-encode with ``--features drums_tria_fixed``;
                      ``controls_dim`` 2.  (experiment 3.4)
    tria_ema          ``__features__: {"drums_tria_ema": [2, T_frames]}`` — the same bands,
                      z-scored against causal running statistics (``ema_standardize``).
    audio             ``__audio__: {"drums_audio": [2, T]}`` — the drum waveform, VAE-encoded by
                      the pre-encode script into a 256-ch latent, as the Slakh streamgen
                      accompaniment is. Pre-encode with ``--controls drums_audio``.
    both              alias for ``rms,audio``.

E.g. ``WJD_CONTROL_MODE=rms,tria_fixed,tria_ema`` returns all three feature controls; the
pre-encode script fuses them into one sidecar in ``--features`` order and the training config
names the ones a model uses. An environment variable rather than a config key because the
module is loaded by path and has no other channel from the dataset config; set it in the
sbatch script next to the flags.

Where the feature controls come from (``WJD_DRUM_FEATURES``):

    precomputed (default)  sliced out of ``tracks/drums/<Track>/drums_features.npz``, written
                           once per song by ``scripts/wjd/compute_drum_features.py``: the frames
                           from ``chunk_offset / 4096`` on, zero past the target's valid length.
                           The silence gate reads the per-frame RMS stored there (mono mixdown).
                           No drum audio is touched unless ``audio`` is among the modes. rms is
                           identical to the audio computation frame for frame; the TRIA
                           controls are the song-level ones, i.e. without the crossover's
                           start-up transient in a chunk's first frame, and with tria_ema's
                           running statistics carrying the song's history into the chunk
                           instead of restarting from the prior at every chunk.
    audio                  computed from the drum waveform of the window, as before.

Rejections (``__reject__`` + reason, so they land in ``_skipped.json`` at pre-encode time):
no ``meta/`` entry, no drums file, or drums silent (RMS floor) over the target's valid
window. The target's own level is the pre-encode script's job.

Alignment: the drums are read from the target's window start (``info["chunk_offset"]`` under
``--chunk_seconds``, else sample 0) and cut to the same length, so this only holds with
``random_crop=False`` (the pre-encode script's setting), exactly as for
``custom_md_slakh_streamgen.py``.
"""

import functools
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

from stable_audio_3.data.features import LATENT_HOP, TriaStats, frames_for, rms_envelope_control, tria_control
from stable_audio_3.data.utils import DEFAULT_SILENCE_THRESHOLD_DB, is_silent

CONDITION = "drums"
CONTROL_MODES = ("rms", "tria_fixed", "tria_ema", "audio")


def parse_control_mode(value):
    """``"rms,tria_ema"`` -> ``("rms", "tria_ema")`` in canonical order; ``"both"`` is ``rms,audio``."""
    modes = set()
    for part in value.split(","):
        part = part.strip()
        if part == "both":
            modes.update(("rms", "audio"))
        elif part in CONTROL_MODES:
            modes.add(part)
        elif part:
            raise ValueError(
                f"WJD_CONTROL_MODE must be a comma-separated set of {' | '.join(CONTROL_MODES)} "
                f"(or 'both'), got {value!r}"
            )
    if not modes:
        raise ValueError(f"WJD_CONTROL_MODE names no control: {value!r}")
    return tuple(m for m in CONTROL_MODES if m in modes)


CONTROL_MODE = os.environ.get("WJD_CONTROL_MODE", "rms")
MODES = parse_control_mode(CONTROL_MODE)

FEATURE_KEY = "drums_rms"
TRIA_KEYS = {"tria_fixed": "drums_tria_fixed", "tria_ema": "drums_tria_ema"}
FEATURE_KEYS = {"rms": FEATURE_KEY, **TRIA_KEYS}  # mode -> sidecar key
AUDIO_KEY = "drums_audio"
FEATURES_FILENAME = "drums_features.npz"  # as scripts/wjd/compute_drum_features.py writes it

FEATURE_SOURCE = os.environ.get("WJD_DRUM_FEATURES", "precomputed")
if FEATURE_SOURCE not in ("precomputed", "audio"):
    raise ValueError(f"WJD_DRUM_FEATURES must be precomputed | audio, got {FEATURE_SOURCE!r}")
TRIA_STATS_PATH = os.environ.get("WJD_TRIA_STATS") or None
_tria_stats = None


def tria_stats():
    """The TRIA dataset constants, loaded on first use (so rms/audio runs need no stats file)."""
    global _tria_stats
    if _tria_stats is None:
        _tria_stats = TriaStats.load(TRIA_STATS_PATH)
    return _tria_stats
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


def load_drums(drums_path, sample_rate, total_length, offset=0):
    """Drum stem as stereo at ``sample_rate``, the ``total_length`` samples from ``offset``, zero-padded.

    ``offset`` is the chunk start when the pre-encode runs chunked (``info["chunk_offset"]``),
    0 for a whole-file encode. Only the window is read when no resampling is needed.
    """
    meta = torchaudio.info(str(drums_path))
    if meta.sample_rate == sample_rate:
        audio, sr = torchaudio.load(str(drums_path), frame_offset=offset, num_frames=total_length)
    else:
        audio, sr = torchaudio.load(str(drums_path))
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
        audio = audio[:, offset:]
    audio = _to_stereo(audio)
    if audio.shape[1] > total_length:
        audio = audio[:, :total_length]
    elif audio.shape[1] < total_length:
        audio = torch.nn.functional.pad(audio, (0, total_length - audio.shape[1]))
    return audio


@functools.lru_cache(maxsize=256)
def load_precomputed(features_path):
    """``{key: float32 tensor [C, T]}`` from a song's ``drums_features.npz`` (cached per process)."""
    with np.load(features_path) as npz:
        return {k: torch.from_numpy(npz[k]) for k in npz.files if k != "meta"}


def window_frames(feat, first_frame, n_frames, valid_frames):
    """Frames ``[first_frame, first_frame + n_frames)`` of ``feat`` ``[C, T]``, zero-padded past the
    song's end and zeroed from ``valid_frames`` on (the target's padding)."""
    out = torch.zeros(feat.shape[0], n_frames, dtype=feat.dtype)
    avail = feat[:, first_frame : first_frame + n_frames]
    out[:, : avail.shape[1]] = avail
    out[:, valid_frames:] = 0
    return out


def window_rms_dbfs(rms_db, first_frame, valid_frames):
    """RMS level over frames ``[first_frame, first_frame + valid_frames)`` from per-frame dB levels."""
    db = rms_db[0, first_frame : first_frame + valid_frames]
    if db.numel() == 0:
        return float("-inf")
    energy = (10.0 ** (db.double() / 10.0)).mean().item()
    return 10 * math.log10(energy) if energy > 0 else float("-inf")


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
    offset = int(info.get("chunk_offset", 0))
    feature_modes = [m for m in MODES if m in FEATURE_KEYS]

    out = {
        "prompt": prompt,
        "track": track,
        "stem": stem,
        "is_drum": False,
        "decade": meta.get("decade"),
        "lineup": meta.get("lineup"),
        "front_line": meta.get("front_line"),
        "control_mode": CONTROL_MODE,
        "feature_source": FEATURE_SOURCE,
    }

    drums = None
    if FEATURE_SOURCE == "precomputed":
        if offset % LATENT_HOP:
            raise ValueError(
                f"precomputed drum features need a frame-aligned window start, got offset {offset} "
                f"(not a multiple of {LATENT_HOP}); use WJD_DRUM_FEATURES=audio for this window"
            )
        features_path = drums_path.parent / FEATURES_FILENAME
        if not features_path.exists():
            raise FileNotFoundError(
                f"{features_path} missing: run scripts/wjd/compute_drum_features.py on this split "
                f"first, or set WJD_DRUM_FEATURES=audio"
            )
        song = load_precomputed(str(features_path))
        f0 = offset // LATENT_HOP
        n_frames = frames_for(total_length)
        valid_frames = frames_for(valid_length)
        if window_rms_dbfs(song["rms_db"], f0, valid_frames) < SILENCE_THRESHOLD_DB:
            return {"__reject__": True, "__reject_reason__": "drums silent over the encoded window"}
        features = {
            FEATURE_KEYS[m]: window_frames(song[FEATURE_KEYS[m]], f0, n_frames, valid_frames) for m in feature_modes
        }
    else:
        drums = load_drums(drums_path, info["sample_rate"], total_length, offset=offset)
        if is_silent(drums[:, :valid_length], SILENCE_THRESHOLD_DB):
            return {"__reject__": True, "__reject_reason__": "drums silent over the encoded window"}
        features = {}
        if "rms" in MODES:
            features[FEATURE_KEY] = rms_envelope_control(drums, hop=LATENT_HOP, valid_samples=valid_length)
        for mode, key in TRIA_KEYS.items():
            if mode in MODES:
                features[key] = tria_control(
                    drums, tria_stats(), norm=mode.split("_")[1], hop=LATENT_HOP, valid_samples=valid_length
                )

    if features:
        out["__features__"] = features
    if "audio" in MODES:
        if drums is None:
            drums = load_drums(drums_path, info["sample_rate"], total_length, offset=offset)
        out["__audio__"] = {AUDIO_KEY: drums}
    return out
