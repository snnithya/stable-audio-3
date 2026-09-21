"""Per-frame inference schedules and the sampler that walks them (experiment 3.3).

CPU, fp32, a stub model. Nothing here needs a checkpoint.

The properties that matter are the ones the driver script relies on without re-checking:
committed frames really are frozen, a staged run equals an unstaged one, and the zero-delay
case is still the ordinary sampler.
"""

import pytest
import torch

from stable_audio_3.inference.forcing import (
    SCHEDULES,
    build_frame_schedule,
    chunk_bounds,
    emission_steps,
    frame_delays,
    sample_euler_per_frame,
)
from stable_audio_3.inference.sampling import sample_discrete_euler

N_FRAMES = 12
CHUNK = 4
STEPS = 8


class StubModel(torch.nn.Module):
    """v = -x * t. Rank-agnostic in t, so it accepts both (B,) and (B, N)."""

    def forward(self, x, t, **kwargs):
        t_b = t[:, None, None] if t.ndim == 1 else t[:, None, :]
        return -x * t_b


class CouplingStub(torch.nn.Module):
    """v mixes each frame with its left neighbour, so frames interact.

    StubModel is separable per frame, which makes its endpoint independent of the order frames
    are denoised in -- a schedule comparison against it passes trivially. Real coupling between
    frames (attention) is the whole premise of 3.3, so the tripwire needs a stub that has some.
    """

    def forward(self, x, t, **kwargs):
        t_b = t[:, None, None] if t.ndim == 1 else t[:, None, :]
        return -(x + 0.5 * torch.roll(x, 1, dims=-1)) * t_b


class NaNAtZero(StubModel):
    """Returns NaN wherever t == 0, mimicking a model asked for an untrained timestep."""

    def forward(self, x, t, **kwargs):
        v = super().forward(x, t, **kwargs)
        t_b = t[:, None, None] if t.ndim == 1 else t[:, None, :]
        return torch.where(t_b == 0, torch.full_like(v, float("nan")), v)


@pytest.fixture
def base():
    return torch.linspace(1, 0, STEPS + 1)


@pytest.fixture
def x0():
    torch.manual_seed(0)
    return torch.randn(2, 4, N_FRAMES)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_chunk_bounds_tiles_the_generated_region():
    assert chunk_bounds(12, 4, 0) == [(0, 4), (4, 8), (8, 12)]
    assert chunk_bounds(12, 4, 4) == [(4, 8), (8, 12)]


def test_chunk_bounds_rejects_a_layout_that_does_not_tile():
    # A short final chunk would get the same step budget as a full one.
    with pytest.raises(ValueError, match="do not tile"):
        chunk_bounds(144, 40, 0)
    assert chunk_bounds(144, 40, 0, allow_partial=True)[-1] == (120, 144)


def test_real_geometry_auto_prefix_tiles():
    # 144 frames, 43-frame (4s) chunks: the remainder is what the prefix absorbs.
    assert chunk_bounds(144, 43, 144 % 43) == [(15, 58), (58, 101), (101, 144)]


# ---------------------------------------------------------------------------
# The schedule matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_every_column_is_a_valid_trajectory(schedule, base):
    delays = frame_delays(schedule, N_FRAMES, CHUNK, 4, STEPS)
    sigmas = build_frame_schedule(base, delays)

    assert (sigmas[:, :4] == 0).all(), "prefix frames must be clean from step 0"
    assert (sigmas[0, 4:] == 1).all(), "generated frames must start at t = 1"
    assert (sigmas[-1] == 0).all(), "every frame must finish at t = 0"
    assert (sigmas[1:] - sigmas[:-1] <= 1e-6).all(), "columns must be non-increasing"


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_every_frame_gets_the_same_number_of_denoising_steps(schedule, base):
    """Schedules differ in *when* a frame is denoised, never in *how much*."""
    delays = frame_delays(schedule, N_FRAMES, CHUNK, 4, STEPS)
    sigmas = build_frame_schedule(base, delays)
    moving = (sigmas[1:] - sigmas[:-1]).abs() > 0
    assert moving[:, 4:].sum(dim=0).unique().tolist() == [STEPS]
    assert moving[:, :4].sum() == 0


def test_block_is_a_step_function_and_linear_is_a_ramp(base):
    block = frame_delays("block", N_FRAMES, CHUNK, 0, STEPS)
    assert block.tolist() == [0] * 4 + [STEPS] * 4 + [2 * STEPS] * 4

    linear = frame_delays("linear", N_FRAMES, CHUNK, 0, STEPS)
    steps_per_frame = STEPS / CHUNK
    assert linear.tolist() == [round(f * steps_per_frame) for f in range(N_FRAMES)]
    # Strictly staggered inside a chunk, which is what a step function is not.
    assert (linear[1:] - linear[:-1] > 0).all()


def test_pyramid_stagger_controls_how_many_chunks_are_in_flight():
    half = frame_delays("pyramid", N_FRAMES, CHUNK, 0, STEPS, pyramid_stagger=0.5)
    assert half.tolist() == [0] * 4 + [STEPS // 2] * 4 + [STEPS] * 4
    # At stagger 1.0 pyramid degenerates to block.
    full = frame_delays("pyramid", N_FRAMES, CHUNK, 0, STEPS, pyramid_stagger=1.0)
    assert torch.equal(full, frame_delays("block", N_FRAMES, CHUNK, 0, STEPS))


def test_joint_reproduces_the_base_schedule_in_every_column(base):
    delays = frame_delays("joint", N_FRAMES, CHUNK, 0, STEPS)
    sigmas = build_frame_schedule(base, delays)
    assert sigmas.shape == (STEPS + 1, N_FRAMES)
    assert torch.equal(sigmas, base[:, None].expand(-1, N_FRAMES))


def test_emission_cadence_matches_the_documented_latency():
    """block and linear emit every `steps`; linear pays one extra fill stage."""
    bounds = chunk_bounds(144, 43, 15)
    steps = 50

    block = emission_steps(frame_delays("block", 144, 43, 15, steps), steps, bounds)
    linear = emission_steps(frame_delays("linear", 144, 43, 15, steps), steps, bounds)
    pyramid = emission_steps(frame_delays("pyramid", 144, 43, 15, steps), steps, bounds)

    assert block == [50, 100, 150]
    assert linear == [99, 149, 199]
    assert pyramid == [50, 75, 100]

    gaps = lambda e: [b - a for a, b in zip(e, e[1:])]  # noqa: E731
    assert gaps(block) == gaps(linear) == [steps, steps], "cadence must match for the comparison"
    assert gaps(pyramid) == [steps // 2] * 2, "pyramid at 0.5 is not cadence-matched"


def test_linear_ramp_spans_the_requested_number_of_chunks():
    one = frame_delays("linear", 144, 43, 15, 50, ramp=1.0)
    two = frame_delays("linear", 144, 43, 15, 50, ramp=2.0)
    assert int(one.max()) == 2 * int(two.max()) or int(one.max()) - 1 == 2 * int(two.max())
    assert int(two.max()) < int(one.max()), "a longer ramp is shallower"


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="unknown schedule"):
        frame_delays("rolling", N_FRAMES, CHUNK, 0, STEPS)


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------


def test_zero_delays_match_the_ordinary_euler_sampler(base, x0):
    """The default sampling path is unchanged; this one only generalises the t shape."""
    delays = frame_delays("joint", N_FRAMES, CHUNK, 0, STEPS)
    per_frame = sample_euler_per_frame(
        StubModel(), x0.clone(), build_frame_schedule(base, delays), disable_tqdm=True
    )
    global_t = sample_discrete_euler(StubModel(), x0.clone(), sigmas=base, disable_tqdm=True)
    assert torch.equal(per_frame, global_t)


def test_committed_frames_are_frozen_even_when_the_model_returns_nan(base, x0):
    """The update is gated on dt != 0, so `0 * nan` cannot poison a committed frame."""
    delays = frame_delays("block", N_FRAMES, CHUNK, 4, STEPS)
    out = sample_euler_per_frame(
        NaNAtZero(), x0.clone(), build_frame_schedule(base, delays), disable_tqdm=True
    )
    assert torch.equal(out[:, :, :4], x0[:, :, :4]), "prefix must be bit-identical"
    assert torch.isfinite(out).all()


def test_frames_that_have_not_started_still_hold_their_noise(base, x0):
    delays = frame_delays("linear", N_FRAMES, CHUNK, 4, STEPS)
    partial = sample_euler_per_frame(
        StubModel(),
        x0.clone(),
        build_frame_schedule(base, delays),
        step_range=(0, 3),
        disable_tqdm=True,
    )
    not_started = delays >= 3
    assert not_started.any()
    assert torch.equal(partial[:, :, not_started], x0[:, :, not_started])


def test_running_in_stages_equals_running_straight_through(base, x0):
    """The driver rebuilds conditioning at every emission; staging must cost nothing."""
    delays = frame_delays("block", N_FRAMES, CHUNK, 4, STEPS)
    sigmas = build_frame_schedule(base, delays)
    emits = emission_steps(delays, STEPS, chunk_bounds(N_FRAMES, CHUNK, 4))

    staged = x0.clone()
    for lo, hi in zip([0] + emits[:-1], emits):
        staged = sample_euler_per_frame(
            StubModel(), staged, sigmas, step_range=(lo, hi), disable_tqdm=True
        )
    whole = sample_euler_per_frame(StubModel(), x0.clone(), sigmas, disable_tqdm=True)
    assert torch.equal(staged, whole)


def test_varying_the_schedule_changes_the_output(base, x0):
    """Tripwire: without this, every comparison in 3.3 could pass trivially.

    Note it needs CouplingStub. Under a frame-separable model every frame runs the same
    trajectory regardless of when it starts, so all four schedules land on the same answer and
    the whole experiment would be vacuous. The schedule can only matter to the extent that
    frames see each other.
    """
    outputs = {}
    for schedule in ("block", "linear", "pyramid"):
        delays = frame_delays(schedule, N_FRAMES, CHUNK, 4, STEPS)
        outputs[schedule] = sample_euler_per_frame(
            CouplingStub(), x0.clone(), build_frame_schedule(base, delays), disable_tqdm=True
        )
    assert not torch.allclose(outputs["block"], outputs["linear"], atol=1e-6)
    assert not torch.allclose(outputs["block"], outputs["pyramid"], atol=1e-6)
    assert not torch.allclose(outputs["linear"], outputs["pyramid"], atol=1e-6)


def test_a_separable_model_is_schedule_invariant(base, x0):
    """The flip side, pinned so nobody later "fixes" the tripwire back to StubModel.

    With no coupling between frames the schedule cannot change the result at all.
    """
    outputs = [
        sample_euler_per_frame(
            StubModel(),
            x0.clone(),
            build_frame_schedule(base, frame_delays(s, N_FRAMES, CHUNK, 4, STEPS)),
            disable_tqdm=True,
        )
        for s in ("block", "linear", "pyramid")
    ]
    assert torch.allclose(outputs[0], outputs[1], atol=1e-6)
    assert torch.allclose(outputs[0], outputs[2], atol=1e-6)


def test_step_range_and_shape_errors_are_caught(base, x0):
    sigmas = build_frame_schedule(base, frame_delays("joint", N_FRAMES, CHUNK, 0, STEPS))
    with pytest.raises(ValueError, match="out of bounds"):
        sample_euler_per_frame(StubModel(), x0, sigmas, step_range=(0, 999), disable_tqdm=True)
    with pytest.raises(ValueError, match="frames"):
        sample_euler_per_frame(StubModel(), x0[:, :, :-1], sigmas, disable_tqdm=True)
