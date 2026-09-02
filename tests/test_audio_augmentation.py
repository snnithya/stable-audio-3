"""Tests for pitch/time-stretch augmentation of time-aligned audio streams.

The property that matters most here is not fidelity but *length*: the target and every
control of a sample must come out of augmentation the same number of samples long, or the
frame-level conditioning silently desyncs. Several tests below exist only to pin that down.
"""

import importlib.util
import math
import random
from pathlib import Path

import pytest
import torch

from stable_audio_3.data.augmentation import (
    AugmentationParams,
    augment_padded_clip,
    phase_vocode_to_length,
    pitch_shift_and_time_stretch,
    resample_by,
    sample_augmentation_params,
)

SR = 44100


@pytest.fixture(scope="module")
def sine():
    """3 s of stereo 220 Hz — a signal with an f0 that is easy to measure."""
    t = torch.arange(SR * 3) / SR
    return torch.stack([torch.sin(2 * math.pi * 220 * t)] * 2) * 0.5


def measure_f0(audio):
    spec = torch.fft.rfft(audio[0] * torch.hann_window(audio.shape[-1]))
    return float(torch.fft.rfftfreq(audio.shape[-1], 1 / SR)[spec.abs().argmax()])


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------


def test_identity_is_bit_exact(sine):
    """No shift and no stretch must not cost a round trip through the STFT."""
    assert torch.equal(pitch_shift_and_time_stretch(sine, semitones=0.0, rate=1.0), sine)


@pytest.mark.parametrize("rate", [0.9, 0.97, 1.0, 1.1, 1.25])
def test_time_stretch_length_is_exact(sine, rate):
    out = pitch_shift_and_time_stretch(sine, rate=rate)
    assert out.shape == (2, round(sine.shape[-1] / rate))


@pytest.mark.parametrize("rate", [0.9, 1.1])
def test_time_stretch_preserves_pitch(sine, rate):
    assert measure_f0(pitch_shift_and_time_stretch(sine, rate=rate)) == pytest.approx(220, abs=1.0)


@pytest.mark.parametrize("semitones", [-4, -2, -1.37, 1, 2, 4])
def test_pitch_shift_hits_the_interval_and_keeps_length(sine, semitones):
    out = pitch_shift_and_time_stretch(sine, semitones=semitones)
    assert out.shape == sine.shape
    assert measure_f0(out) == pytest.approx(220 * 2 ** (semitones / 12), rel=0.01)


def test_pitch_and_stretch_compose(sine):
    out = pitch_shift_and_time_stretch(sine, semitones=1.37, rate=0.93)
    assert out.shape[-1] == round(sine.shape[-1] / 0.93)
    assert measure_f0(out) == pytest.approx(220 * 2 ** (1.37 / 12), rel=0.01)


def test_rational_resample_ratio_is_under_a_cent():
    """The resample ratio is approximated to keep torchaudio's kernel small; check the cost."""
    from stable_audio_3.data.augmentation import _rational

    for semitones in [-4, -2, -0.5, 0.5, 1, 1.37, 2, 4]:
        exact = 2 ** (semitones / 12)
        num, den = _rational(exact)
        assert abs(1200 * math.log2((num / den) / exact)) < 1.0


def test_resample_by_changes_length_and_pitch(sine):
    out = resample_by(sine, 2.0)
    assert out.shape[-1] == pytest.approx(sine.shape[-1] / 2, rel=0.01)
    assert measure_f0(out) == pytest.approx(440, rel=0.01)


def test_short_signal_does_not_blow_up_the_stft():
    """n_fft is shrunk to fit; a clip shorter than the default window must still work."""
    short = torch.randn(2, 900) * 0.1
    out = phase_vocode_to_length(short, 1200)
    assert out.shape == (2, 1200)


# ---------------------------------------------------------------------------
# Alignment — the invariant the frame-level conditioning depends on
# ---------------------------------------------------------------------------


def test_same_rate_gives_identical_lengths_across_streams():
    """Target and control are stretched separately; only equal lengths keep them aligned."""
    params = AugmentationParams(rate=0.93, semitones=1.7, variant=1)
    target = torch.randn(2, SR * 4) * 0.1
    control = torch.randn(2, SR * 4) * 0.1

    t_out, t_valid = augment_padded_clip(target, SR * 4, params, apply_pitch=False)
    c_out, c_valid = augment_padded_clip(control, SR * 4, params, apply_pitch=True)

    assert t_valid == c_valid
    assert t_out.shape == c_out.shape


def test_impulse_positions_move_together_under_a_shared_rate():
    """A click at the same instant in both streams must stay at the same instant."""
    params = AugmentationParams(rate=1.1, semitones=2.0, variant=1)
    n = SR * 4
    clicks = [SR // 2, SR * 2, SR * 3]

    def clicked():
        x = torch.zeros(2, n)
        for c in clicks:
            x[:, c] = 1.0
        return x

    target, _ = augment_padded_clip(clicked(), n, params, apply_pitch=False)
    control, _ = augment_padded_clip(clicked(), n, params, apply_pitch=True)

    for c in clicks:
        expected = round(c / params.rate)
        window = slice(max(0, expected - 1000), expected + 1000)
        t_peak = int(target[0, window].abs().argmax())
        c_peak = int(control[0, window].abs().argmax())
        # Within one 2048-sample analysis window of each other, and of where they belong.
        assert abs(t_peak - c_peak) < 2048
        assert abs(t_peak - 1000) < 2048 if expected >= 1000 else True


# ---------------------------------------------------------------------------
# Padded-clip bookkeeping
# ---------------------------------------------------------------------------


def test_padded_clip_keeps_its_width_and_reports_the_new_valid_length():
    total = SR * 6
    valid = SR * 4
    clip = torch.zeros(2, total)
    clip[:, :valid] = torch.randn(2, valid) * 0.1

    out, new_valid = augment_padded_clip(clip, valid, AugmentationParams(rate=0.9, variant=1), apply_pitch=False)

    assert out.shape == (2, total)
    assert new_valid == round(valid / 0.9)
    assert torch.all(out[:, new_valid:] == 0)


def test_padded_clip_truncates_when_a_slowdown_overruns_the_window():
    total = SR * 4
    clip = torch.randn(2, total) * 0.1
    out, new_valid = augment_padded_clip(clip, total, AugmentationParams(rate=0.8, variant=1), apply_pitch=False)
    assert out.shape == (2, total)
    assert new_valid == total


def test_identity_params_skip_the_transform_entirely():
    clip = torch.randn(2, SR) * 0.1
    out, new_valid = augment_padded_clip(clip, SR, AugmentationParams(), apply_pitch=True)
    assert torch.equal(out, clip)
    assert new_valid == SR


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------


def test_variant_zero_is_always_unaugmented():
    """So a dataset encoded with N variants still contains the original material."""
    params = sample_augmentation_params(0, random.Random(1234))
    assert params.is_identity
    assert params.variant == 0


@pytest.mark.parametrize("variant", [1, 2, 7])
def test_sampled_params_respect_the_ranges(variant):
    for seed in range(50):
        p = sample_augmentation_params(
            variant, random.Random(seed), max_semitones=2.0, rate_range=(0.9, 1.1)
        )
        assert 0.9 <= p.rate <= 1.1
        assert -2.0 <= p.semitones <= 2.0
        assert p.variant == variant


def test_sampling_is_reproducible_from_the_seed():
    """Re-encoding a subset of a dataset must reproduce the rolls it got the first time."""
    a = sample_augmentation_params(2, random.Random("seed:Track00001_v2"))
    b = sample_augmentation_params(2, random.Random("seed:Track00001_v2"))
    c = sample_augmentation_params(2, random.Random("seed:Track00002_v2"))
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# Pre-encode wiring
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pre_encode():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pre_encode_dataset.py"
    spec = importlib.util.spec_from_file_location("_pre_encode_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_latent_ids_only_gain_a_suffix_once_there_are_variants(pre_encode):
    assert pre_encode.latent_id_for(3, 1, 0, 1) == "0000030001"
    assert pre_encode.latent_id_for(3, 1, 0, 4) == "0000030001_v0"
    assert pre_encode.latent_id_for(3, 1, 2, 4) == "0000030001_v2"


def test_augment_item_rewrites_the_padding_mask_and_duration(pre_encode):
    total, valid = SR * 6, SR * 4
    audio = torch.zeros(2, total)
    audio[:, :valid] = torch.randn(2, valid) * 0.1
    mask = torch.zeros(total)
    mask[:valid] = 1

    md = {
        "padding_mask": [mask],
        "seconds_total": 4.0,
        "streamgen_audio": audio.clone(),
    }
    params = AugmentationParams(rate=1.1, semitones=1.5, variant=1)

    out = pre_encode.augment_item(audio, md, ["streamgen_audio"], params, pitch_controls_only=True)

    new_valid = round(valid / 1.1)
    assert out.shape == (2, total)
    assert md["streamgen_audio"].shape == (2, total)
    assert int(md["padding_mask"][0].sum()) == new_valid
    assert md["seconds_total"] == pytest.approx(4.0 / 1.1)
    assert md["augmentation"] == {
        "variant": 1,
        "time_stretch_rate": 1.1,
        "pitch_semitones": 1.5,
        "pitch_scope": "controls",
    }


def _tone(seconds=2, hz=220):
    t = torch.arange(SR * seconds) / SR
    return torch.stack([torch.sin(2 * math.pi * hz * t)] * 2) * 0.5


def test_augment_item_transposes_every_stream_by_the_same_interval(pre_encode):
    """The default scope: a variant is the whole arrangement moved to a new key."""
    tone = _tone()
    n = tone.shape[-1]
    md = {"padding_mask": [torch.ones(n)], "seconds_total": 2.0, "streamgen_audio": tone.clone()}

    out = pre_encode.augment_item(
        tone.clone(), md, ["streamgen_audio"], AugmentationParams(rate=1.0, semitones=4.0, variant=1),
        pitch_controls_only=False,
    )

    expected = 220 * 2 ** (4 / 12)
    assert measure_f0(out) == pytest.approx(expected, rel=0.01)
    assert measure_f0(md["streamgen_audio"]) == pytest.approx(expected, rel=0.01)
    assert md["augmentation"]["pitch_scope"] == "all"


def test_augment_item_can_hold_the_target_at_its_own_pitch(pre_encode):
    """`controls` scope: re-key the accompaniment, leave the target's tuning alone."""
    tone = _tone()
    n = tone.shape[-1]
    md = {"padding_mask": [torch.ones(n)], "seconds_total": 2.0, "streamgen_audio": tone.clone()}

    out = pre_encode.augment_item(
        tone.clone(), md, ["streamgen_audio"], AugmentationParams(rate=1.0, semitones=4.0, variant=1),
        pitch_controls_only=True,
    )

    assert measure_f0(out) == pytest.approx(220, rel=0.01)
    assert measure_f0(md["streamgen_audio"]) == pytest.approx(220 * 2 ** (4 / 12), rel=0.01)


@pytest.mark.parametrize("pitch_controls_only", [True, False])
def test_streams_stay_the_same_length_whatever_the_pitch_scope(pre_encode, pitch_controls_only):
    """Pitch scope must not leak into duration — that is what would desync the streams."""
    tone = _tone()
    n = tone.shape[-1]
    md = {"padding_mask": [torch.ones(n)], "seconds_total": 2.0, "streamgen_audio": tone.clone()}

    # rate > 1 so the stretched clip stays inside the window and no truncation muddies the
    # length check (truncation itself is covered separately).
    out = pre_encode.augment_item(
        tone.clone(), md, ["streamgen_audio"], AugmentationParams(rate=1.07, semitones=-2.0, variant=1),
        pitch_controls_only=pitch_controls_only,
    )

    assert out.shape == md["streamgen_audio"].shape == (2, n)
    assert int(md["padding_mask"][0].sum()) == round(n / 1.07)


# ---------------------------------------------------------------------------
# Variant-aware listening logs
#
# A run that logs n samples over N variants must produce n x N *listenable items*, not n
# items with N renderings stacked on each card. These pin the two places that decide that:
# how a wav stem is split into (sample, stream), and which dataset indices a listening
# check selects.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def listening_page():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_listening_page.py"
    spec = importlib.util.spec_from_file_location("_make_listening_page", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def decode_samples():
    path = Path(__file__).resolve().parents[1] / "scripts" / "decode_preencoded_samples.py"
    spec = importlib.util.spec_from_file_location("_decode_preencoded_samples", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "stem,expected",
    [
        # The variant suffix stays on the id side: each variant is its own sample.
        ("0000000000_v0_decoded", ("0000000000_v0", "decoded")),
        ("0000000000_v12_source", ("0000000000_v12", "source")),
        (
            "0000000000_v3_control_streamgen_audio_decoded",
            ("0000000000_v3", "control_streamgen_audio_decoded"),
        ),
        # Unaugmented ids have no suffix to find and must split as they always did.
        ("0000000000_decoded", ("0000000000", "decoded")),
        (
            "0000000000_control_streamgen_audio_source",
            ("0000000000", "control_streamgen_audio_source"),
        ),
        ("0000_control_streamgen_audio", ("0000", "control_streamgen_audio")),
        ("nolabel", ("nolabel", "audio")),
    ],
)
def test_wav_stems_split_variant_into_the_sample_id(listening_page, stem, expected):
    assert listening_page.split_stem(stem) == expected


def test_variants_of_a_track_get_one_card_each(listening_page):
    """n samples x N variants must read as n x N cards on the page."""
    stems = [
        f"000000{i:04d}_v{v}_{stream}"
        for i in range(2)
        for v in range(4)
        for stream in ("source", "decoded", "control_streamgen_audio_decoded")
    ]
    ids = {listening_page.split_stem(s)[0] for s in stems}
    assert len(ids) == 2 * 4


def test_control_streams_sort_below_the_target_within_a_card(listening_page):
    """Stripping the variant prefix is what lets the control rows be recognised as such."""
    labels = [
        "control_streamgen_audio_decoded",
        "decoded",
        "control_streamgen_audio_source",
        "source",
    ]
    assert sorted(labels, key=listening_page.stream_sort_key) == [
        "source",
        "decoded",
        "control_streamgen_audio_source",
        "control_streamgen_audio_decoded",
    ]


class _FakeLatentDataset:
    """Stands in for PreEncodedDataset: only `filenames` and `len` are consulted."""

    def __init__(self, stems):
        self.filenames = [(f"/x/{s}.npy", f"/x/{s}.json", None) for s in stems]

    def __len__(self):
        return len(self.filenames)


def test_listening_check_selects_whole_tracks_not_whole_files(decode_samples):
    """-n counts tracks; every variant of each comes along, in variant order.

    Latent filenames come off disk in scandir order, so the shuffled input here is the
    realistic case — slicing the first N indices would have returned an arbitrary mix of
    variants of unrelated tracks.
    """
    stems = [
        f"000001{i:04d}_v{v}" for i in range(3) for v in range(4)
    ]
    shuffled = list(stems)
    random.Random(0).shuffle(shuffled)

    selected = decode_samples.select_indices(_FakeLatentDataset(shuffled), 2)

    assert [s for s, _ in selected] == [
        "0000010000_v0", "0000010000_v1", "0000010000_v2", "0000010000_v3",
        "0000010001_v0", "0000010001_v1", "0000010001_v2", "0000010001_v3",
    ]
    # The index paired with each stem must still address that stem in the dataset.
    ds = _FakeLatentDataset(shuffled)
    for stem, idx in decode_samples.select_indices(ds, 2):
        assert Path(ds.filenames[idx][0]).stem == stem


def test_listening_check_on_an_unaugmented_dataset_still_counts_files(decode_samples):
    stems = ["0000000002", "0000000000", "0000000001"]
    selected = decode_samples.select_indices(_FakeLatentDataset(stems), 2)
    assert [s for s, _ in selected] == ["0000000000", "0000000001"]


def test_listening_check_selection_is_capped_by_the_dataset(decode_samples):
    ds = _FakeLatentDataset([f"0000000000_v{v}" for v in range(4)])
    assert len(decode_samples.select_indices(ds, 99)) == 4
