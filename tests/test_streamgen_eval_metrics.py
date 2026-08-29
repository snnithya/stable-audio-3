"""Tests for the alignment metrics used by experiment 1.2 (scripts/eval_streamgen.py).

These are the numbers the sub-experiment's conclusion rests on, and they are easy to get
subtly wrong (a sign flip in the cross-correlation, an off-by-one in the beat grid) in a way
that still produces plausible-looking output. The synthetic pairs here pin down the three
cases that matter: drums locked to the accompaniment, drums on the off-beat, and drums with
no relationship to it at all.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("eval_streamgen", SCRIPTS / "eval_streamgen.py")
eval_streamgen = importlib.util.module_from_spec(spec)
sys.modules["eval_streamgen"] = eval_streamgen
spec.loader.exec_module(eval_streamgen)

SAMPLE_RATE = 44100
FPS = SAMPLE_RATE / eval_streamgen.ONSET_HOP


def click_track(bpm, duration=10.0, offset=0.0, seed=0, sample_rate=SAMPLE_RATE):
    """A percussive click train: noise bursts with a fast exponential decay."""
    generator = torch.Generator().manual_seed(seed)
    n = int(duration * sample_rate)
    x = torch.zeros(n)
    burst_len = int(0.02 * sample_rate)
    envelope = torch.exp(-torch.arange(burst_len) / (0.004 * sample_rate))

    t = offset
    while t < duration:
        i = int(t * sample_rate)
        burst = torch.randn(burst_len, generator=generator) * envelope
        x[i : i + burst_len] += burst[: max(0, min(burst_len, n - i))]
        t += 60.0 / bpm

    return x.unsqueeze(0).repeat(2, 1)


def test_estimate_pulse_recovers_tempo():
    env = eval_streamgen.onset_envelope(click_track(120, seed=1))
    period, phase = eval_streamgen.estimate_pulse(env, FPS)
    assert period is not None
    assert 115 < 60.0 * FPS / period < 125
    assert 0 <= phase < period


def test_aligned_drums_score_high():
    accompaniment = click_track(120, seed=1)
    drums = click_track(120, seed=2)
    m = eval_streamgen.alignment_metrics(drums, accompaniment, SAMPLE_RATE)

    assert m["xcorr_at_zero"] > 0.8
    assert m["xcorr_lag_seconds"] == pytest.approx(0.0, abs=1e-6)
    assert m["beat_hit_rate"] > 0.9


def test_offbeat_drums_score_low_at_zero_lag():
    """Same tempo, wrong phase: the metric must not reward it just for matching tempo."""
    accompaniment = click_track(120, seed=1)
    drums = click_track(120, offset=0.5 * 60.0 / 120.0, seed=3)
    m = eval_streamgen.alignment_metrics(drums, accompaniment, SAMPLE_RATE)

    assert m["xcorr_at_zero"] < 0.2
    assert m["beat_hit_rate"] < 0.2


def test_unrelated_drums_score_near_zero():
    accompaniment = click_track(120, seed=1)
    drums = click_track(97, offset=0.13, seed=4)
    m = eval_streamgen.alignment_metrics(drums, accompaniment, SAMPLE_RATE)

    assert abs(m["xcorr_at_zero"]) < 0.2
    aligned = eval_streamgen.alignment_metrics(click_track(120, seed=2), accompaniment, SAMPLE_RATE)
    assert m["beat_hit_rate"] < aligned["beat_hit_rate"]


def test_xcorr_lag_sign_is_reported_in_seconds():
    """Drums running late must report a positive lag of the right magnitude."""
    accompaniment = click_track(120, seed=1)
    delay_seconds = 0.1
    delayed = torch.roll(accompaniment, shifts=int(delay_seconds * SAMPLE_RATE), dims=-1)

    m = eval_streamgen.alignment_metrics(delayed, accompaniment, SAMPLE_RATE)
    assert m["xcorr_lag_seconds"] == pytest.approx(delay_seconds, abs=2.0 / FPS)
    assert m["xcorr_peak"] > 0.8


def test_silent_input_returns_none_rather_than_nan():
    accompaniment = click_track(120, seed=1)
    silence = torch.zeros_like(accompaniment)
    m = eval_streamgen.alignment_metrics(silence, accompaniment, SAMPLE_RATE)
    assert m["xcorr_peak"] is None
    assert m["beat_hit_rate"] is None


def test_summaries_are_paired_over_items():
    """paired_delta must difference items by id, not by position in the list."""
    a = eval_streamgen.summarize_loss(
        [{"item": 0, "timestep": 0.5, "loss": 1.0}, {"item": 1, "timestep": 0.5, "loss": 2.0}]
    )
    b = eval_streamgen.summarize_loss(
        [{"item": 1, "timestep": 0.5, "loss": 3.0}, {"item": 0, "timestep": 0.5, "loss": 1.5}]
    )
    delta = eval_streamgen.paired_delta(a, b, "a", "b")

    assert delta["mean_delta"] == pytest.approx(-0.75)
    assert delta["n_items"] == 2
    assert delta["fraction_items_lower"] == 1.0
