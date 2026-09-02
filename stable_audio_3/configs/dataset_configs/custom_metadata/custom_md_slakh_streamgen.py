"""Custom metadata extractor for Slakh/BabySlakh with streamgen accompaniment mixing.

Extends custom_md_slakh.py: as well as tagging the drum stem, it builds the
"streamgen" accompaniment — a submix of the track's non-drum stems — and hands it back
via the `__audio__` hook so the pre-encode script can VAE-encode it into a control
sidecar aligned with the drum latents.

Expected path layout (the "streamgen-drum-mirror" tree):
    .../tracks/drums/Track00001/Drums.wav      <- target, what this fn is called on
    .../tracks/other/Track00001/Guitar.wav     <- accompaniment stems
    .../tracks/other/Track00001/Piano.wav
    ...

The submix is stochastic: a random subset of the available stems, each normalized to a
random loudness. This mirrors sat-zenon's load_and_mix_stems so the model sees a range of
arrangement densities and balances rather than one fixed full mix.

NOTE: because pre-encoding caches the result, the randomness is rolled once per track *per
pre-encode pass* and then frozen. `--augment_variants N` makes N passes, so an N-variant
dataset holds N different submixes per track — but that is still N rolls, not a fresh roll
per epoch, which would require moving the mix to train time.
"""

import random
import re
from pathlib import Path

import torch
import torchaudio

# Submix sampling parameters (mirrors sat-zenon's defaults).
MIN_STEMS = 1
MAX_STEMS = None  # None = all available stems are eligible
LUFS_RANGE = (-30.0, -15.0)
SILENCE_ENERGY_THRESHOLD = 1e-6
PEAK_CEILING = 0.95

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")


def lufs_normalize(audio, sample_rate, target_lufs, energy_threshold=SILENCE_ENERGY_THRESHOLD):
    """Scale audio to a target integrated loudness, declipping if it overshoots.

    Near-silent stems are returned untouched: their measured loudness is meaningless and
    normalizing them would amplify noise to full scale.
    """
    energy = torch.mean(audio**2)
    if energy < energy_threshold:
        return audio

    loudness_meter = torchaudio.transforms.Loudness(sample_rate)
    input_loudness = loudness_meter(audio)

    if not torch.isfinite(input_loudness):
        return audio

    gain = torch.pow(10.0, (target_lufs - input_loudness) / 20.0)
    output = gain * audio

    max_val = torch.max(torch.abs(output))
    if max_val > 1.0:
        output = output / max_val * PEAK_CEILING

    return output


def _to_stereo(audio):
    if audio.shape[0] == 1:
        return audio.repeat(2, 1)
    if audio.shape[0] > 2:
        return audio[:2]
    return audio


def find_other_stems(drum_path):
    """Map a drum stem path to its sibling accompaniment stems."""
    drum_path = Path(drum_path)
    track_dir = drum_path.parent           # .../tracks/drums/Track00001
    tracks_root = track_dir.parent.parent  # .../tracks
    other_dir = tracks_root / "other" / track_dir.name

    if not other_dir.is_dir():
        return []

    return sorted(p for p in other_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)


def load_and_mix_stems(stem_paths, sample_rate, target_length=None):
    """Load, level, and sum a random subset of the accompaniment stems.

    Returns (mix, selected_names). `mix` is a stereo tensor at `sample_rate`, or None if no
    usable (non-silent) stem was found.
    """
    loaded = []
    for stem_path in stem_paths:
        try:
            audio, sr = torchaudio.load(str(stem_path))
        except Exception:
            continue

        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)

        audio = _to_stereo(audio)

        if target_length is not None:
            if audio.shape[1] > target_length:
                audio = audio[:, :target_length]
            elif audio.shape[1] < target_length:
                audio = torch.nn.functional.pad(audio, (0, target_length - audio.shape[1]))

        # Drop silent stems before sampling, so the subset is drawn from stems that
        # actually contribute something.
        if torch.mean(audio**2) < SILENCE_ENERGY_THRESHOLD:
            continue

        loaded.append((stem_path.stem, audio))

    if not loaded:
        return None, []

    max_stems = len(loaded) if MAX_STEMS is None else min(MAX_STEMS, len(loaded))
    min_stems = min(MIN_STEMS, len(loaded))
    n_stems = random.randint(min_stems, max_stems)
    selected = random.sample(loaded, n_stems)

    mix = None
    for _, audio in selected:
        audio = lufs_normalize(audio, sample_rate, random.uniform(*LUFS_RANGE))
        mix = audio if mix is None else mix + audio

    peak = torch.max(torch.abs(mix))
    if peak > 1.0:
        mix = mix / peak * PEAK_CEILING

    return mix, [name for name, _ in selected]


def get_custom_metadata(info, audio):
    """Tag the drum stem and attach the accompaniment submix as `streamgen_audio`.

    The mix is returned at its natural length via `__audio__`, so SampleDataset applies the
    same pad/crop and channel handling it applied to the drums. That shared treatment is
    what keeps the two time-aligned, and it only holds when the dataset is built with
    random_crop=False (otherwise each call draws its own crop offset).
    """
    filepath = info["path"]

    track_match = re.search(r"(Track\d+)", filepath)
    track_id = track_match.group(1) if track_match else None

    metadata = {
        "prompt": "drums",
        "is_drum": True,
        "track_id": track_id,
    }

    stem_paths = find_other_stems(filepath)
    if not stem_paths:
        # No accompaniment means no streamgen condition, so the sample is useless here.
        return {"__reject__": True}

    mix, selected_names = load_and_mix_stems(stem_paths, info["sample_rate"])
    if mix is None:
        return {"__reject__": True}

    metadata["streamgen_stems"] = selected_names
    metadata["__audio__"] = {"streamgen_audio": mix}
    return metadata
