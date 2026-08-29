"""Tests for the lookahead (`future_visibility`) extension to random_inpaint_mask."""

import torch

from stable_audio_3.models.inpainting import MaskType, random_inpaint_mask


def _make_batch(b=4, c=8, n=64):
    sequence = torch.randn(b, c, n)
    padding_masks = torch.ones(b, n)
    return sequence, padding_masks


def test_returns_three_values_and_tf_defaults_to_inpaint_mask():
    """Without future_visibility the third mask must be identical to the second,
    so existing callers see no behavior change."""
    sequence, padding_masks = _make_batch()
    for mask_type in MaskType:
        masked, mask, tf_mask = random_inpaint_mask(
            sequence, padding_masks=padding_masks, force_mask_type=mask_type
        )
        assert masked.shape == sequence.shape
        assert mask.shape == tf_mask.shape == (sequence.shape[0], 1, sequence.shape[-1])
        assert torch.equal(mask, tf_mask), f"tf mask diverged for {mask_type}"


def test_masked_sequence_matches_mask():
    sequence, padding_masks = _make_batch()
    masked, mask, _ = random_inpaint_mask(
        sequence, padding_masks=padding_masks, force_mask_type=MaskType.CAUSAL_MASK
    )
    assert torch.equal(masked, sequence * mask)


def test_causal_horizon_is_prefix_plus_future_visibility():
    """tf_inpaint_mask is a prefix mask whose length is the causal cursor shifted by
    the lookahead. The cursor is where inpaint_mask first turns 0."""
    sequence, padding_masks = _make_batch(b=32, n=64)
    fv = 5
    _, mask, tf_mask = random_inpaint_mask(
        sequence,
        padding_masks=padding_masks,
        force_mask_type=MaskType.CAUSAL_MASK,
        future_visibility=fv,
    )

    for i in range(mask.shape[0]):
        row = mask[i, 0]
        tf_row = tf_mask[i, 0]

        # Causal masks keep a prefix of ones; the cursor is the first zero (or the
        # full length when nothing was masked).
        zeros = (row == 0).nonzero()
        prefix = int(zeros[0].item()) if len(zeros) else row.shape[0]

        expected = max(0, min(row.shape[0], prefix + fv))
        assert torch.all(tf_row[:expected] == 1)
        assert torch.all(tf_row[expected:] == 0)


def test_negative_future_visibility_hides_before_the_cursor():
    """A negative horizon must reveal strictly less than the causal mask does."""
    sequence, padding_masks = _make_batch(b=32, n=64)
    _, mask, tf_mask = random_inpaint_mask(
        sequence,
        padding_masks=padding_masks,
        force_mask_type=MaskType.CAUSAL_MASK,
        future_visibility=-4,
    )
    assert tf_mask.sum() <= mask.sum()


def test_tuple_future_visibility_stays_in_range():
    sequence, padding_masks = _make_batch(b=64, n=64)
    lo, hi = -4, 4
    _, mask, tf_mask = random_inpaint_mask(
        sequence,
        padding_masks=padding_masks,
        force_mask_type=MaskType.CAUSAL_MASK,
        future_visibility=(lo, hi),
    )

    for i in range(mask.shape[0]):
        row, tf_row = mask[i, 0], tf_mask[i, 0]
        zeros = (row == 0).nonzero()
        prefix = int(zeros[0].item()) if len(zeros) else row.shape[0]
        horizon = int(tf_row.sum().item())

        assert max(0, min(row.shape[0], prefix + lo)) <= horizon <= max(0, min(row.shape[0], prefix + hi))
        # It must be a contiguous prefix, not scattered ones.
        assert torch.equal(tf_row, torch.cat([torch.ones(horizon), torch.zeros(row.shape[0] - horizon)]))


def test_padding_is_excluded_from_the_horizon():
    """With mask_padding, the lookahead must not extend into the padding region."""
    b, n, real = 8, 64, 40
    sequence = torch.randn(b, 8, n)
    padding_masks = torch.zeros(b, n)
    padding_masks[:, :real] = 1

    _, _, tf_mask = random_inpaint_mask(
        sequence,
        padding_masks=padding_masks,
        force_mask_type=MaskType.CAUSAL_MASK,
        mask_padding=True,
        future_visibility=(0, 32),
    )
    assert torch.all(tf_mask[:, :, real:] == 0)
