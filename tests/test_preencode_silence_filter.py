"""Tests for the RMS silence filter applied at pre-encode time.

Ported from sat-zenon's pre_encode.py, where the same check gates both the target and the
accompaniment stems. The properties that matter, and that the peak-based `is_silence` in
dataset.py does not have, are pinned below: RMS not peak, measured over the valid region,
and a reject that *drops* rather than substitutes.
"""

import importlib.util

import pytest
import torch

from stable_audio_3.data.utils import (
    DEFAULT_SILENCE_THRESHOLD_DB,
    is_silent,
    rms_dbfs,
    silence_fraction,
)
from stable_audio_3.data.dataset import is_silence

MODULE_PATH = (
    "stable_audio_3/configs/dataset_configs/custom_metadata/custom_md_slakh_streamgen.py"
)


@pytest.fixture(scope="module")
def md():
    spec = importlib.util.spec_from_file_location("custom_md_slakh_streamgen", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rms_dbfs_matches_known_level():
    sr = 44100
    # A full-scale sine has RMS 1/sqrt(2) ≈ -3.01 dBFS.
    t = torch.arange(sr) / sr
    sine = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
    assert abs(rms_dbfs(sine) - (-3.01)) < 0.1


def test_rms_dbfs_of_digital_silence_is_minus_inf():
    assert rms_dbfs(torch.zeros(2, 1000)) == float("-inf")


def test_rms_catches_a_dead_stem_that_the_peak_check_waves_through():
    """The reason for the port: one stray sample satisfies a peak test forever.

    This is the realistic case — a stem that is noise floor for its whole length with a
    single click in it. `is_silence` sees the click and calls the stem content.
    """
    sr = 44100
    clip = torch.randn(2, sr * 10) * 0.0005  # ≈ -66 dBFS noise floor
    clip[:, 0] = 0.5  # one click

    assert not is_silence(clip), "peak check sees the click and calls it content"
    assert is_silent(clip), "RMS check sees a dead stem"


def test_rms_alone_does_not_measure_percentage_of_silence():
    """Pinning the limit of the RMS check, so nobody reads it as a silence-fraction test."""
    sr = 44100
    clip = torch.zeros(2, sr * 10)
    clip[:, : sr // 100] = 0.9  # one 10ms hit in ten seconds: 99.9% silence

    assert silence_fraction(clip, sr) > 0.99
    assert not is_silent(clip), "an average cannot see that this window is 99.9% empty"
    assert -35 < rms_dbfs(clip) < -25


def test_silence_fraction_tracks_the_silent_proportion():
    sr = 44100
    for content in (0.5, 0.1, 0.01):
        clip = torch.zeros(2, sr * 10)
        clip[:, : int(sr * 10 * content)] = 0.9
        assert abs(silence_fraction(clip, sr) - (1 - content)) < 0.01


def test_silence_fraction_endpoints():
    sr = 44100
    assert silence_fraction(torch.zeros(2, sr), sr) == 1.0
    assert silence_fraction(torch.randn(2, sr) * 0.1, sr) == 0.0


def test_busy_clip_passes_both():
    audio = torch.randn(2, 44100) * 0.1
    assert not is_silence(audio)
    assert not is_silent(audio)


def test_threshold_is_respected():
    audio = torch.randn(2, 44100) * 0.01  # ≈ -40 dBFS
    assert not is_silent(audio, -50.0)
    assert is_silent(audio, -30.0)


def test_padding_must_not_be_measured_a_short_track_is_short_not_silent():
    """80s of music in a 380s window is 79% "silent" if you measure the whole window.

    This is what makes measuring over the valid region non-negotiable for the fraction
    check — sample_size is set from a high percentile of track length, so most items are
    mostly padding. RMS is diluted by padding too, just far more gently (a 10*log10 of the
    fill ratio), which is why it survives the same mistake and the fraction check does not.
    """
    sr = 44100
    valid = sr * 80
    padded = torch.zeros(2, sr * 380)
    padded[:, :valid] = torch.randn(2, valid) * 0.1

    assert silence_fraction(padded, sr) > 0.75, "the padded window looks mostly empty"
    assert silence_fraction(padded[:, :valid], sr) == 0.0, "the track itself is not empty"

    # Same direction, weaker effect: ~7 dB of dilution here, not enough to trip -50.
    assert rms_dbfs(padded) < rms_dbfs(padded[:, :valid]) - 5


def test_stem_silent_over_the_window_is_dropped_even_if_the_file_is_loud(md, tmp_path):
    """The failure this exists to catch: accompaniment that only enters later."""
    import torchaudio

    sr = 44100
    stem = torch.zeros(2, sr * 60)
    stem[:, sr * 30 :] = torch.randn(2, sr * 30) * 0.1  # enters at 0:30
    path = tmp_path / "Guitar.wav"
    torchaudio.save(str(path), stem, sr)

    # Whole file: plenty of content.
    mix, names = md.load_and_mix_stems([path], sr)
    assert mix is not None and names == ["Guitar"]

    # First ten seconds: nothing at all.
    mix, names = md.load_and_mix_stems([path], sr, target_length=sr * 10)
    assert mix is None and names == []


def test_get_custom_metadata_measures_the_target_window(md, tmp_path):
    import torchaudio

    sr = 44100
    track = tmp_path / "tracks"
    (track / "drums" / "Track00001").mkdir(parents=True)
    (track / "other" / "Track00001").mkdir(parents=True)

    drum_path = track / "drums" / "Track00001" / "Drums.wav"
    torchaudio.save(str(drum_path), torch.randn(2, sr * 10) * 0.1, sr)

    stem = torch.zeros(2, sr * 60)
    stem[:, sr * 30 :] = torch.randn(2, sr * 30) * 0.1
    torchaudio.save(str(track / "other" / "Track00001" / "Guitar.wav"), stem, sr)

    # padding_mask says the target only occupies its first 10s of a 60s window.
    mask = torch.zeros(sr * 60)
    mask[: sr * 10] = 1
    info = {"path": str(drum_path), "sample_rate": sr, "padding_mask": [mask]}

    out = md.get_custom_metadata(info, torch.zeros(2, sr * 60))
    assert out.get("__reject__") is True
    assert "silent" in out["__reject_reason__"]


def test_reject_reasons_are_reported(md, tmp_path):
    fake = tmp_path / "tracks" / "drums" / "Track99999" / "Drums.wav"
    fake.parent.mkdir(parents=True)
    fake.touch()
    out = md.get_custom_metadata({"path": str(fake), "sample_rate": 44100}, None)
    assert out["__reject__"] is True
    assert out["__reject_reason__"] == "no accompaniment stems"


def test_default_threshold_is_the_shared_one(md):
    assert md.SILENCE_THRESHOLD_DB == DEFAULT_SILENCE_THRESHOLD_DB == -50.0
