"""The causal TRIA drum control (stable_audio_3/data/features.py, experiment 3.4): band
split, causality, normalisation modes, quantisation, padding. Synthetic signals only.
"""

import math

import pytest
import torch

from stable_audio_3.data.features import (
    LATENT_HOP,
    RMS_FLOOR_DB,
    TRIA_LEVELS,
    TriaStats,
    band_rms_db,
    block_rms_db,
    crossover,
    ema_standardize,
    fixed_standardize,
    frames_for,
    quantize_unit,
    tria_control,
)

SR = 44100


@pytest.fixture
def stats():
    return TriaStats(split_hz=250.0, band_mean_db=[-30.0, -36.0], band_std_db=[8.0, 6.0])


def sine(hz, seconds, amp=1.0):
    t = torch.arange(int(SR * seconds)) / SR
    return (amp * torch.sin(2 * math.pi * hz * t)).unsqueeze(0)


def noise(seconds, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 0.1 * torch.randn(1, int(SR * seconds), generator=g)


# ---------------------------------------------------------------------------
# Band split
# ---------------------------------------------------------------------------


def test_band_rms_frames_match_latent_grid(stats):
    x = noise(2.0)
    db = band_rms_db(x, stats.split_hz, SR)
    assert db.shape == (2, frames_for(x.shape[-1]))


def test_low_sine_lands_in_the_low_band(stats):
    db = band_rms_db(sine(60, 2.0), stats.split_hz, SR)
    low, high = db[0, -1].item(), db[1, -1].item()  # last frame: filters settled
    assert low == pytest.approx(-3.0, abs=0.3)  # full-scale sine passes the low band
    assert high < low - 40


def test_high_sine_lands_in_the_high_band(stats):
    db = band_rms_db(sine(5000, 2.0), stats.split_hz, SR)
    low, high = db[0, -1].item(), db[1, -1].item()
    assert high == pytest.approx(-3.0, abs=0.3)
    assert low < high - 40


def test_crossover_bands_sum_back_to_the_input(stats):
    # 4th-order Linkwitz-Riley: low + high is an allpass, so the summed bands have the
    # input's level frame for frame.
    x = noise(2.0)
    low, high = crossover(x, stats.split_hz, SR)
    summed = block_rms_db((low + high).float())
    assert torch.allclose(summed, block_rms_db(x), atol=0.05)


def test_crossover_rejects_odd_order(stats):
    with pytest.raises(ValueError):
        crossover(noise(0.1), stats.split_hz, SR, order=3)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_fixed_standardize_is_a_per_channel_z_score():
    x = torch.tensor([[-30.0, -22.0], [-36.0, -42.0]])
    z = fixed_standardize(x, mean=[-30.0, -36.0], std=[8.0, 6.0])
    assert torch.allclose(z, torch.tensor([[0.0, 1.0], [0.0, -1.0]]))


def test_fixed_standardize_floors_a_tiny_std():
    z = fixed_standardize(torch.tensor([[1.0]]), mean=[0.0], std=[1e-6])
    assert z.item() == pytest.approx(1.0)  # divided by the 1 dB floor, not 1e-6


def test_ema_starts_from_the_prior():
    # First frame equal to the prior mean: z = 0 whatever the window does afterwards.
    x = torch.full((1, 1), -30.0)
    z = ema_standardize(x, prior_mean=[-30.0], prior_std=[8.0], tau_frames=40)
    assert z.item() == pytest.approx(0.0, abs=1e-6)


def test_ema_adapts_to_a_level_step():
    tau = 40
    x = torch.cat([torch.full((1, 400), -40.0), torch.full((1, 400), -20.0)], dim=1)
    z = ema_standardize(x, prior_mean=[-40.0], prior_std=[8.0], tau_frames=tau)
    assert z[0, 399].item() == pytest.approx(0.0, abs=1e-3)  # settled on the first level
    assert z[0, 400].item() > 3.0  # the step registers as a strong onset
    assert abs(z[0, -1].item()) < 0.5  # ... and is normalised away after ~10 tau


def test_ema_is_a_unit_z_score_on_stationary_noise():
    g = torch.Generator().manual_seed(1)
    x = -30.0 + 5.0 * torch.randn(2, 4000, generator=g)
    z = ema_standardize(x, prior_mean=[-30.0, -30.0], prior_std=[5.0, 5.0], tau_frames=43)
    tail = z[:, 1000:]
    assert tail.mean().abs().item() < 0.1
    assert tail.std(dim=1).sub(1.0).abs().max().item() < 0.15


def test_ema_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ema_standardize(torch.zeros(3), [0.0], [1.0], tau_frames=10)
    with pytest.raises(ValueError):
        ema_standardize(torch.zeros(1, 3), [0.0], [1.0], tau_frames=0)


# ---------------------------------------------------------------------------
# The control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["fixed", "ema"])
def test_control_is_quantised_in_unit_range(stats, norm):
    feat = tria_control(noise(3.0), stats, norm=norm)
    assert feat.shape == (2, frames_for(int(SR * 3.0)))
    assert feat.min() >= 0 and feat.max() <= 1
    steps = feat * (TRIA_LEVELS - 1)
    assert torch.allclose(steps, steps.round(), atol=1e-5)


def test_quantize_unit_grid():
    q = quantize_unit(torch.tensor([0.0, 0.01, 0.5, 0.999, 1.2]), n_levels=33)
    assert torch.allclose(q, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


@pytest.mark.parametrize("norm", ["fixed", "ema"])
def test_control_is_causal(stats, norm):
    # Two clips identical for the first 6 frames and different after: the control for
    # those 6 frames must be bit-identical (IIR crossover, block RMS, running stats).
    n_shared = 6 * LATENT_HOP
    a = noise(1.0, seed=1)
    b = a.clone()
    b[:, n_shared:] = noise(1.0, seed=2)[:, n_shared:]
    fa = tria_control(a, stats, norm=norm)
    fb = tria_control(b, stats, norm=norm)
    assert torch.equal(fa[:, :6], fb[:, :6])
    assert not torch.equal(fa[:, 6:], fb[:, 6:])


def test_fixed_control_maps_the_dataset_mean_to_half(stats):
    # Flat spectrum noise scaled so each band sits at its dataset mean: sigmoid(0) = 0.5.
    x = noise(2.0)
    db = band_rms_db(x, stats.split_hz, SR)[:, -1]
    stats_here = TriaStats(stats.split_hz, band_mean_db=db.tolist(), band_std_db=[8.0, 6.0])
    feat = tria_control(x, stats_here, norm="fixed")
    assert feat[0, -1].item() == pytest.approx(0.5) and feat[1, -1].item() == pytest.approx(0.5)


@pytest.mark.parametrize("norm", ["fixed", "ema"])
def test_control_masks_padding(stats, norm):
    # With the EMA the stats would adapt to the silence and drift back toward 0.5; the
    # padding must be forced to 0 regardless.
    x = noise(2.0)
    valid = int(0.8 * x.shape[-1])
    feat = tria_control(x, stats, norm=norm, valid_samples=valid)
    first_pad = frames_for(valid)
    assert torch.all(feat[:, first_pad:] == 0)
    assert feat[:, : first_pad - 1].gt(0).any()


def test_control_rejects_unknown_norm(stats):
    with pytest.raises(ValueError):
        tria_control(noise(0.5), stats, norm="clip")


# ---------------------------------------------------------------------------
# Stats file
# ---------------------------------------------------------------------------


def test_stats_round_trip(tmp_path, stats):
    path = tmp_path / "stats.json"
    stats.save(path)
    loaded = TriaStats.load(path)
    assert loaded == stats
    assert loaded.ema_tau_frames() == pytest.approx(4.0 * SR / LATENT_HOP)


def test_stats_validate_shape():
    with pytest.raises(ValueError):
        TriaStats(split_hz=250.0, band_mean_db=[-30.0], band_std_db=[8.0, 6.0])
    with pytest.raises(ValueError):
        TriaStats(split_hz=30000.0, band_mean_db=[-30.0, -36.0], band_std_db=[8.0, 6.0])
