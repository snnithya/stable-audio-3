"""Tests for the WJD stem mirror: prompt mapping, the drum RMS feature control, and the
`__features__` path through the pre-encode script, all on a hand-made fixture (no data,
no GPU, no autoencoder).
"""

import importlib.util
import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

from stable_audio_3.data.features import (
    LATENT_HOP,
    RMS_FLOOR_DB,
    block_rms_db,
    frames_for,
    rms_envelope_control,
)

REPO = Path(__file__).resolve().parents[1]
MIRROR_SCRIPT = REPO / "scripts/wjd/make_stem_mirror.py"
MD_MODULE = REPO / "stable_audio_3/configs/dataset_configs/custom_metadata/custom_md_wjd.py"
PRE_ENCODE = REPO / "scripts/pre_encode_dataset.py"

SR = 44100


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mirror():
    return load(MIRROR_SCRIPT, "make_stem_mirror")


@pytest.fixture(scope="module")
def md():
    os.environ.pop("WJD_CONTROL_MODE", None)
    return load(MD_MODULE, "custom_md_wjd")


# ---------------------------------------------------------------------------
# Lineup parsing and prompts (make_stem_mirror.py)
# ---------------------------------------------------------------------------

LINEUP = "Freddie Hubbard (tp); Wayne Shorter (ts); Cedar Walton (p); Jymie Merritt (b); Art Blakey (dr); Curtis Fuller (tb)"


def test_parse_lineup_expands_abbreviations(mirror):
    players = mirror.parse_lineup(LINEUP)
    assert [p["performer"] for p in players][:2] == ["Freddie Hubbard", "Wayne Shorter"]
    assert players[1]["instruments"] == ["tenor saxophone"]
    assert players[4]["abbr"] == ["dr"]


def test_parse_lineup_splits_hyphenated_doubling(mirror):
    players = mirror.parse_lineup("Someone (p-tp); Other (ts, cl)")
    assert players[0]["instruments"] == ["piano", "trumpet"]
    assert players[1]["instruments"] == ["tenor saxophone", "clarinet"]


def test_other_prompt_is_the_front_line_without_rhythm_section(mirror):
    players = mirror.parse_lineup(LINEUP)
    assert mirror.prompt_for("other", players) == "trumpet, tenor saxophone, trombone"


def test_other_prompt_is_none_for_a_trio(mirror):
    """A piano trio has nothing melodic left in `other`; the stem must not be mirrored."""
    players = mirror.parse_lineup("Bill Evans (p); Scott LaFaro (b); Paul Motian (dr)")
    assert mirror.prompt_for("other", players) is None


def test_vocals_in_lineup_do_not_name_the_other_stem(mirror):
    """The separator has its own vocals stem, so `voc` is not what `other` sounds like."""
    players = mirror.parse_lineup("Singer (voc); Horn (ts); P (p); B (b); D (dr)")
    assert mirror.prompt_for("other", players) == "tenor saxophone"


def test_fixed_stem_prompts(mirror):
    players = mirror.parse_lineup(LINEUP)
    assert mirror.prompt_for("bass", players) == "upright bass"
    assert mirror.prompt_for("piano", players) == "piano"


def test_split_is_by_track_and_reproducible(mirror):
    names = [f"T{i}" for i in range(50)]
    a = mirror.assign_splits(names, 0.12, seed=3)
    b = mirror.assign_splits(names, 0.12, seed=3)
    assert a == b
    assert sum(v == "validation" for v in a.values()) == 6
    assert mirror.assign_splits(names, 0.12, seed=4) != a


def test_jsd_segments_carry_both_clocks(mirror, tmp_path):
    csv = tmp_path / "X.csv"
    csv.write_text(
        "segment_start;segment_end;label;instrument\n"
        "10.0;20.0;solo_01_01;s_ts,b_p,b_b,b_dr\n"
        "20.0;30.0;theme_01_01;ts,tp,p,b,dr\n"
    )
    segs = mirror.load_jsd_segments(csv, offset_sec=-0.5)
    assert segs[0]["soloists"] == ["ts"]
    assert segs[0]["backing"] == ["p", "b", "dr"]
    # track_time = file_time + offset  =>  file_time = track_time - offset
    assert segs[0]["file_start"] == pytest.approx(10.5)
    assert segs[1]["instruments"] == ["ts", "tp", "p", "b", "dr"]


# ---------------------------------------------------------------------------
# RMS feature (stable_audio_3/data/features.py)
# ---------------------------------------------------------------------------

def test_block_rms_frames_match_latent_grid():
    for n in (LATENT_HOP, LATENT_HOP + 1, 10 * LATENT_HOP - 1, 123456):
        assert block_rms_db(torch.zeros(2, n)).shape == (1, frames_for(n))
        assert frames_for(n) == math.ceil(n / LATENT_HOP)


def test_block_rms_level_of_a_sine_is_minus_3_db():
    t = torch.arange(10 * LATENT_HOP) / SR
    sine = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).repeat(2, 1)
    db = block_rms_db(sine)
    assert db.shape == (1, 10)
    assert torch.all((db - (-3.01)).abs() < 0.3)


def test_block_rms_silence_sits_on_the_floor():
    db = block_rms_db(torch.zeros(2, 3 * LATENT_HOP))
    assert torch.all(db == RMS_FLOOR_DB)


def test_block_rms_is_causal():
    """Frame t must not change when anything after frame t's last sample changes."""
    torch.manual_seed(0)
    a = torch.randn(2, 8 * LATENT_HOP) * 0.1
    b = a.clone()
    b[:, 5 * LATENT_HOP:] = 0.9  # rewrite the future from frame 5 on
    fa, fb = block_rms_db(a), block_rms_db(b)
    assert torch.equal(fa[:, :5], fb[:, :5])
    assert not torch.equal(fa[:, 5:], fb[:, 5:])


def test_block_rms_partial_last_frame_is_not_diluted():
    """A hot signal that ends mid-frame should still read hot in that frame."""
    sig = torch.full((1, 2 * LATENT_HOP + LATENT_HOP // 4), 0.5)
    db = block_rms_db(sig)
    assert db[0, -1] == pytest.approx(db[0, 0], abs=0.01)


def test_rms_envelope_control_is_normalised_and_masks_padding():
    sig = torch.full((2, 6 * LATENT_HOP), 0.5)
    ctrl = rms_envelope_control(sig, valid_samples=4 * LATENT_HOP)
    assert ctrl.shape == (1, 6)
    assert torch.all((ctrl >= 0) & (ctrl <= 1))
    assert torch.all(ctrl[0, :4] > 0.9)  # -6 dBFS -> (80-6)/80
    assert torch.all(ctrl[0, 4:] == 0)   # padding region forced to the floor


# ---------------------------------------------------------------------------
# Fixture tree: a two-track mirror with meta json, used by the module + pre-encode tests
# ---------------------------------------------------------------------------

def _tone(freq, seconds, amp=0.3):
    t = torch.arange(int(seconds * SR)) / SR
    return (amp * torch.sin(2 * torch.pi * freq * t)).unsqueeze(0).repeat(2, 1)


def _write(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio, SR)


@pytest.fixture
def mirror_tree(tmp_path):
    split = tmp_path / "train"
    tracks = split / "tracks"
    # Track A: drums present, bass + other targets. Track B: silent drums.
    _write(tracks / "drums" / "A" / "drums.flac", _tone(100, 3.0))
    _write(tracks / "bass" / "A" / "bass.flac", _tone(55, 3.0))
    _write(tracks / "other" / "A" / "other.flac", _tone(440, 3.0))
    _write(tracks / "drums" / "B" / "drums.flac", torch.zeros(2, 3 * SR))
    _write(tracks / "bass" / "B" / "bass.flac", _tone(55, 3.0))
    (split / "meta").mkdir()
    for name in ("A", "B"):
        (split / "meta" / f"{name}.json").write_text(json.dumps({
            "track": name, "prompts": {"bass": "upright bass", "other": "tenor saxophone"},
            "decade": "1960s", "lineup": LINEUP, "front_line": ["tenor saxophone"],
        }))
    return split


def _info(path, total_samples, valid_samples):
    mask = torch.zeros(total_samples)
    mask[:valid_samples] = 1
    return {"path": str(path), "sample_rate": SR, "padding_mask": [mask]}


def test_module_prompt_and_rms_feature(md, mirror_tree):
    target = mirror_tree / "tracks" / "bass" / "A" / "bass.flac"
    total, valid = 4 * SR, 3 * SR
    out = md.get_custom_metadata(_info(target, total, valid), torch.zeros(2, total))
    assert out["prompt"] == "upright bass"
    assert out["track"] == "A" and out["stem"] == "bass"
    feat = out["__features__"]["drums_rms"]
    assert feat.shape == (1, frames_for(total))
    assert torch.all(feat[0, : frames_for(valid) - 1] > 0.5)
    assert torch.all(feat[0, frames_for(valid):] == 0)
    assert "__audio__" not in out


def test_module_other_stem_gets_front_line_prompt(md, mirror_tree):
    target = mirror_tree / "tracks" / "other" / "A" / "other.flac"
    out = md.get_custom_metadata(_info(target, 3 * SR, 3 * SR), torch.zeros(2, 3 * SR))
    assert out["prompt"] == "tenor saxophone"


def test_module_rejects_silent_drums(md, mirror_tree):
    target = mirror_tree / "tracks" / "bass" / "B" / "bass.flac"
    out = md.get_custom_metadata(_info(target, 3 * SR, 3 * SR), torch.zeros(2, 3 * SR))
    assert out["__reject__"] and "silent" in out["__reject_reason__"]


def test_module_rejects_missing_meta(md, mirror_tree):
    _write(mirror_tree / "tracks" / "bass" / "C" / "bass.flac", _tone(55, 1.0))
    _write(mirror_tree / "tracks" / "drums" / "C" / "drums.flac", _tone(100, 1.0))
    out = md.get_custom_metadata(_info(mirror_tree / "tracks/bass/C/bass.flac", SR, SR), torch.zeros(2, SR))
    assert out["__reject__"]


def test_module_audio_mode_returns_drum_waveform(mirror_tree, monkeypatch):
    monkeypatch.setenv("WJD_CONTROL_MODE", "both")
    mod = load(MD_MODULE, "custom_md_wjd_both")
    target = mirror_tree / "tracks" / "bass" / "A" / "bass.flac"
    total = 4 * SR
    out = mod.get_custom_metadata(_info(target, total, 3 * SR), torch.zeros(2, total))
    assert out["__audio__"]["drums_audio"].shape == (2, total)
    assert out["__features__"]["drums_rms"].shape == (1, frames_for(total))


# ---------------------------------------------------------------------------
# `--features` through pre_encode_dataset.encode_dataset, with a fake autoencoder
# ---------------------------------------------------------------------------

class FakeAE:
    """Stands in for AutoencoderModel: 4-channel 'latent' by average pooling at LATENT_HOP."""

    sample_rate = SR
    LATENT_DIM = 4

    def __init__(self):
        self.autoencoder = torch.nn.Conv1d(2, 2, 1)  # gives .parameters() a device and io_channels
        self.autoencoder.io_channels = 2

    def encode(self, audio, sr):
        pooled = torch.nn.functional.avg_pool1d(audio.abs(), LATENT_HOP, ceil_mode=True)  # [B, 2, T]
        return torch.cat([pooled, pooled], dim=1)  # [B, 4, T]

    def decode(self, latent):
        return torch.nn.functional.interpolate(latent[:, :2], scale_factor=LATENT_HOP, mode="nearest")


def _preencode_args(**over):
    base = dict(
        sample_size=4 * SR, batch_size=1, pad=True, model_half=False, controls=None, features=None,
        sanity_check_samples=1, sanity_check_dir=None, augment_variants=1, augment_pitch_semitones=2.0,
        augment_time_stretch=[0.9, 1.1], augment_pitch_controls_only=False, augment_seed=0,
        silence_threshold_db=-50.0, max_silence_fraction=None, silence_frame_threshold_db=-60.0,
        no_silence_filter=False,
    )
    base.update(over)
    return Namespace(**base)


def test_preencode_fuses_rms_feature_into_controls_sidecar(md, mirror_tree, tmp_path):
    pe = load(PRE_ENCODE, "pre_encode_dataset")
    out = tmp_path / "latents"
    pe.encode_dataset(
        FakeAE(), str(mirror_tree / "tracks" / "bass"), str(out), md.get_custom_metadata,
        _preencode_args(features=["drums_rms"]),
    )
    latents = sorted(p for p in out.glob("*.npy") if "_controls" not in p.name and p.name != "silence.npy")
    assert len(latents) == 1, "track B has silent drums and must be dropped by the module"
    latent = np.load(latents[0])
    controls = np.load(out / f"{latents[0].stem}_controls.npy")
    meta = json.loads((out / f"{latents[0].stem}.json").read_text())

    assert controls.shape == (1, latent.shape[-1])
    assert meta["control_keys"] == ["drums_rms"] and meta["controls_dim"] == [1]
    assert meta["prompt"] == "upright bass"
    assert "drums_rms" not in meta, "the feature tensor must not be dumped into the JSON"
    # 3 s of signal in a 4 s window: the control is on for the valid frames and off after.
    n_valid = frames_for(3 * SR)
    assert np.all(controls[0, : n_valid - 1] > 0.5)
    assert np.all(controls[0, n_valid:] == 0)
    assert (out / "_sanity_check" / f"{latents[0].stem}_feature_drums_rms.npy").exists()

    skipped = json.loads((out / "_skipped.json").read_text())
    assert skipped["v0"]["written"] == 1
    assert "drums silent over the encoded window" in skipped["v0"]["skipped"]


def test_preencode_orders_audio_controls_before_features(mirror_tree, tmp_path, monkeypatch):
    monkeypatch.setenv("WJD_CONTROL_MODE", "both")
    mod = load(MD_MODULE, "custom_md_wjd_both2")
    pe = load(PRE_ENCODE, "pre_encode_dataset2")
    out = tmp_path / "latents"
    pe.encode_dataset(
        FakeAE(), str(mirror_tree / "tracks" / "bass"), str(out), mod.get_custom_metadata,
        _preencode_args(controls=["drums_audio"], features=["drums_rms"], sanity_check_samples=0),
    )
    meta_path = next(p for p in out.glob("*.json") if not p.name.startswith("_skipped"))
    meta = json.loads(meta_path.read_text())
    controls = np.load(out / f"{meta_path.stem}_controls.npy")
    assert meta["control_keys"] == ["drums_audio", "drums_rms"]
    assert meta["controls_dim"] == [FakeAE.LATENT_DIM, 1]
    assert controls.shape[0] == FakeAE.LATENT_DIM + 1
    assert "levels" in meta and "drums_audio" in meta["levels"], "audio controls are level-checked"


def test_preencode_refuses_features_with_augmentation(md, mirror_tree, tmp_path):
    pe = load(PRE_ENCODE, "pre_encode_dataset3")
    with pytest.raises(ValueError, match="augment"):
        pe.encode_dataset(
            FakeAE(), str(mirror_tree / "tracks" / "bass"), str(tmp_path / "x"), md.get_custom_metadata,
            _preencode_args(features=["drums_rms"], augment_variants=2),
        )


# ---------------------------------------------------------------------------
# Sharding: N processes must reproduce exactly the files of one process
# ---------------------------------------------------------------------------

@pytest.fixture
def four_track_tree(mirror_tree):
    """Add tracks C and D (drums present) so the bass dir has 4 items: A, B(silent drums), C, D."""
    split = mirror_tree
    for name in ("C", "D"):
        _write(split / "tracks" / "drums" / name / "drums.flac", _tone(90, 2.0))
        _write(split / "tracks" / "bass" / name / "bass.flac", _tone(60, 2.0))
        (split / "meta" / f"{name}.json").write_text(json.dumps({
            "track": name, "prompts": {"bass": "upright bass"}, "decade": "1950s",
            "lineup": LINEUP, "front_line": ["tenor saxophone"],
        }))
    return split


def _listing(out):
    return sorted(p.name for p in out.iterdir() if p.name.endswith((".npy", ".json")) and not p.name.startswith("_skipped"))


def test_sharded_run_matches_single_run(md, four_track_tree, tmp_path):
    pe = load(PRE_ENCODE, "pre_encode_dataset_shard")
    data = str(four_track_tree / "tracks" / "bass")
    single = tmp_path / "single"
    pe.encode_dataset(FakeAE(), data, str(single), md.get_custom_metadata,
                      _preencode_args(features=["drums_rms"], sanity_check_samples=0))

    sharded = tmp_path / "sharded"
    for k in range(3):
        pe.encode_dataset(FakeAE(), data, str(sharded), md.get_custom_metadata,
                          _preencode_args(features=["drums_rms"], sanity_check_samples=0,
                                          num_shards=3, shard_index=k))

    assert _listing(single) == _listing(sharded)
    for name in _listing(single):
        if name.endswith(".npy"):
            assert np.array_equal(np.load(single / name), np.load(sharded / name)), name
        else:
            a = json.loads((single / name).read_text())
            b = json.loads((sharded / name).read_text())
            assert a["prompt"] == b["prompt"] and a["track"] == b["track"], name

    merged = json.loads((sharded / "_skipped.json").read_text())["v0"]
    assert merged["written"] == 3 and merged["shards"] == "3/3"
    assert merged["skipped"] == {"drums silent over the encoded window": 1}
    assert sorted(p.name for p in sharded.glob("_skipped.shard*")) == [
        f"_skipped.shard{k}of3.json" for k in range(3)
    ]


def test_shard_index_out_of_range_is_refused(md, mirror_tree, tmp_path):
    pe = load(PRE_ENCODE, "pre_encode_dataset_shard2")
    with pytest.raises(ValueError, match="shard_index"):
        pe.encode_dataset(FakeAE(), str(mirror_tree / "tracks" / "bass"), str(tmp_path / "x"),
                          md.get_custom_metadata, _preencode_args(num_shards=2, shard_index=2))
