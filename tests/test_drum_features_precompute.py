"""Per-song drum features (scripts/wjd/compute_drum_features.py) and the metadata module
slicing them per window (WJD_DRUM_FEATURES=precomputed, the default) versus computing from
the window's audio (WJD_DRUM_FEATURES=audio)."""

import json

import numpy as np
import pytest
import torch

from stable_audio_3.data.features import LATENT_HOP, frames_for
from test_wjd_metadata import (  # noqa: F401
    DRUM_FEATURES,
    MD_MODULE,
    SR,
    _info,
    _tone,
    _write,
    fixture_stats,
    load,
    md,
    mirror_tree,
)


@pytest.fixture(scope="module")
def df():
    return load(DRUM_FEATURES, "compute_drum_features_test")


@pytest.fixture
def audio_md(monkeypatch, tmp_path):
    monkeypatch.setenv("WJD_DRUM_FEATURES", "audio")
    monkeypatch.setenv("WJD_CONTROL_MODE", "rms,tria_fixed,tria_ema")
    stats = tmp_path / "stats.json"
    fixture_stats().save(stats)
    monkeypatch.setenv("WJD_TRIA_STATS", str(stats))
    return load(MD_MODULE, "custom_md_wjd_audio_source")


@pytest.fixture
def pre_md(monkeypatch):
    monkeypatch.setenv("WJD_DRUM_FEATURES", "precomputed")
    monkeypatch.setenv("WJD_CONTROL_MODE", "rms,tria_fixed,tria_ema")
    monkeypatch.setenv("WJD_TRIA_STATS", "/nonexistent/never-read.json")  # slicing needs no stats
    return load(MD_MODULE, "custom_md_wjd_precomputed_source")


def test_song_features_file_layout(df, mirror_tree):
    path = df.features_path_for(mirror_tree / "tracks" / "drums" / "A" / "drums.flac")
    feats, meta = df.load_features(path)
    n = frames_for(3 * SR)
    assert set(feats) == {"drums_rms", "drums_tria_fixed", "drums_tria_ema", "rms_db"}
    assert feats["drums_rms"].shape == (1, n) and feats["rms_db"].shape == (1, n)
    assert feats["drums_tria_fixed"].shape == (2, n) and feats["drums_tria_ema"].shape == (2, n)
    assert all(v.dtype == torch.float32 for v in feats.values())
    assert meta["n_frames"] == n and meta["hop"] == LATENT_HOP and meta["stats"]["split_hz"] == 200.0


def test_rms_matches_the_audio_computation_and_tria_fixed_after_the_filter_settles(audio_md, pre_md, mirror_tree):
    # drums_rms is a per-block statistic: identical. drums_tria_fixed goes through the IIR
    # crossover, which in audio mode starts from rest at the chunk boundary, so a chunk's
    # first frame carries a start-up transient the song-level computation does not have;
    # from the second frame on the two agree. (The song's first chunk has no such difference.)
    target = mirror_tree / "tracks" / "bass" / "A" / "bass.flac"
    window = 11 * LATENT_HOP
    for k in range(4):
        info = {**_info(target, window, window), "chunk_offset": k * 6 * LATENT_HOP}
        a = audio_md.get_custom_metadata(info, None)["__features__"]
        p = pre_md.get_custom_metadata(info, None)["__features__"]
        assert torch.equal(a["drums_rms"], p["drums_rms"])
        if k == 0:
            assert torch.equal(a["drums_tria_fixed"], p["drums_tria_fixed"])
        assert torch.allclose(a["drums_tria_fixed"][:, 1:], p["drums_tria_fixed"][:, 1:], atol=1 / 32)


def test_ema_control_carries_the_song_history(audio_md, pre_md, mirror_tree):
    target = mirror_tree / "tracks" / "bass" / "A" / "bass.flac"
    window = 11 * LATENT_HOP
    first = {**_info(target, window, window), "chunk_offset": 0}
    later = {**_info(target, window, window), "chunk_offset": 12 * LATENT_HOP}
    a0 = audio_md.get_custom_metadata(first, None)["__features__"]["drums_tria_ema"]
    p0 = pre_md.get_custom_metadata(first, None)["__features__"]["drums_tria_ema"]
    assert torch.equal(a0, p0)  # the song's first window has no history either way
    a1 = audio_md.get_custom_metadata(later, None)["__features__"]["drums_tria_ema"]
    p1 = pre_md.get_custom_metadata(later, None)["__features__"]["drums_tria_ema"]
    assert not torch.equal(a1, p1)  # audio mode restarts from the prior; precomputed keeps running
    # Steady tone: the song-history EMA has settled on the signal (z ~ 0 -> 0.5), while the
    # restarted one is still being pulled from the prior.
    assert (p1 - 0.5).abs().max() < (a1 - 0.5).abs().max()


def test_padding_and_song_end_are_zero(pre_md, mirror_tree):
    target = mirror_tree / "tracks" / "bass" / "A" / "bass.flac"
    total, valid = 4 * SR, 3 * SR  # whole-file window longer than the 3 s song
    feats = pre_md.get_custom_metadata(_info(target, total, valid), None)["__features__"]
    first_pad = frames_for(valid)
    for v in feats.values():
        assert v.shape[1] == frames_for(total)
        assert torch.all(v[:, first_pad:] == 0) and v[:, :first_pad].gt(0).any()


def test_silent_drums_are_rejected_from_the_stored_levels(pre_md, mirror_tree):
    target = mirror_tree / "tracks" / "bass" / "B" / "bass.flac"
    out = pre_md.get_custom_metadata(_info(target, 3 * SR, 3 * SR), None)
    assert out["__reject__"] and "silent" in out["__reject_reason__"]


def test_missing_song_features_is_a_clear_error(pre_md, mirror_tree):
    (mirror_tree / "tracks" / "drums" / "A" / "drums_features.npz").unlink()
    with pytest.raises(FileNotFoundError, match="compute_drum_features"):
        pre_md.get_custom_metadata(_info(mirror_tree / "tracks" / "bass" / "A" / "bass.flac", SR, SR), None)


def test_unaligned_offset_is_refused(pre_md, mirror_tree):
    info = {**_info(mirror_tree / "tracks" / "bass" / "A" / "bass.flac", SR, SR), "chunk_offset": 1000}
    with pytest.raises(ValueError, match="frame-aligned"):
        pre_md.get_custom_metadata(info, None)


def test_audio_mode_still_returns_the_waveform(pre_md, mirror_tree, monkeypatch):
    monkeypatch.setenv("WJD_CONTROL_MODE", "audio,rms")
    mod = load(MD_MODULE, "custom_md_wjd_precomputed_plus_audio")
    window = 11 * LATENT_HOP
    info = {**_info(mirror_tree / "tracks" / "bass" / "A" / "bass.flac", window, window), "chunk_offset": 6 * LATENT_HOP}
    out = mod.get_custom_metadata(info, None)
    assert out["__audio__"]["drums_audio"].shape == (2, window)
    assert out["__features__"]["drums_rms"].shape == (1, 11)


def test_script_cli_skips_existing_files(df, mirror_tree, tmp_path, capsys):
    import sys
    stats = tmp_path / "stats.json"
    fixture_stats().save(stats)
    mirror = mirror_tree.parent  # <tmp>/train is the split dir
    argv = ["x", "--mirror", str(mirror), "--split", "train", "--stats", str(stats), "--workers", "1"]
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sys, "argv", argv)
    try:
        df.main()
    finally:
        monkey.undo()
    assert "skipped 2 existing" in capsys.readouterr().out
