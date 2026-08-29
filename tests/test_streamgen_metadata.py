"""Tests for the streamgen accompaniment submix used as a frame-level condition."""

import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    "stable_audio_3/configs/dataset_configs/custom_metadata/custom_md_slakh_streamgen.py"
)
BABYSLAKH = Path(
    "/data/scratch-fast/snnithya/sat-zenon/data/babyslakh/streamgen-drum-mirror/tracks/drums"
)


@pytest.fixture(scope="module")
def md():
    spec = importlib.util.spec_from_file_location("custom_md_slakh_streamgen", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def drum_path():
    if not BABYSLAKH.is_dir():
        pytest.skip(f"BabySlakh not available at {BABYSLAKH}")
    tracks = sorted(BABYSLAKH.glob("*/Drums.wav"))
    if not tracks:
        pytest.skip("No drum stems found")
    return tracks[0]


def test_lufs_normalize_hits_target(md):
    sr = 44100
    audio = torch.randn(2, sr * 2) * 0.1
    out = md.lufs_normalize(audio, sr, -20.0)
    import torchaudio

    measured = torchaudio.transforms.Loudness(sr)(out)
    assert abs(float(measured) - (-20.0)) < 1.0


def test_lufs_normalize_leaves_silence_alone(md):
    """Normalizing a silent stem would amplify noise to full scale."""
    sr = 44100
    silence = torch.zeros(2, sr)
    assert torch.equal(md.lufs_normalize(silence, sr, -20.0), silence)


def test_find_other_stems_locates_accompaniment(md, drum_path):
    stems = md.find_other_stems(drum_path)
    assert stems, "no accompaniment stems found for a drum track"
    assert all(s.parent.parent.name == "other" for s in stems)
    assert all(s.parent.name == drum_path.parent.name for s in stems)


def test_find_other_stems_returns_empty_when_missing(md, tmp_path):
    fake = tmp_path / "tracks" / "drums" / "Track99999" / "Drums.wav"
    fake.parent.mkdir(parents=True)
    fake.touch()
    assert md.find_other_stems(fake) == []


def test_mix_is_stereo_and_within_range(md, drum_path):
    stems = md.find_other_stems(drum_path)
    mix, names = md.load_and_mix_stems(stems, 44100)
    assert mix is not None and names
    assert mix.shape[0] == 2
    assert float(mix.abs().max()) <= 1.0
    assert set(names) <= {s.stem for s in stems}


def test_mix_length_honours_target(md, drum_path):
    stems = md.find_other_stems(drum_path)
    target = 44100 * 5
    mix, _ = md.load_and_mix_stems(stems, 44100, target_length=target)
    assert mix.shape[-1] == target


def test_get_custom_metadata_returns_accompaniment_audio(md, drum_path):
    info = {"path": str(drum_path), "sample_rate": 44100}
    out = md.get_custom_metadata(info, None)

    assert out["prompt"] == "drums"
    assert out["track_id"] == drum_path.parent.name
    # The accompaniment goes through __audio__ so SampleDataset applies the same
    # pad/crop and channel handling it applies to the drums; that shared treatment is
    # what keeps the two time-aligned.
    assert "streamgen_audio" in out["__audio__"]
    assert out["__audio__"]["streamgen_audio"].shape[0] == 2


def test_rejects_tracks_without_accompaniment(md, tmp_path):
    fake = tmp_path / "tracks" / "drums" / "Track99999" / "Drums.wav"
    fake.parent.mkdir(parents=True)
    fake.touch()
    out = md.get_custom_metadata({"path": str(fake), "sample_rate": 44100}, None)
    assert out.get("__reject__") is True
