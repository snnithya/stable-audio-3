"""The training wrapper attaches every sidecar control the model config names, not just
`streamgen_latent`. Exercised on the unbound method with a stand-in for `self`, so no model
is built: what is under test is the id handling and the tf-mask gating.
"""

from types import SimpleNamespace

import pytest
import torch

from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper as W


def fake_module(cond_ids, null_values=None):
    m = SimpleNamespace(
        diffusion=SimpleNamespace(modular_local_cond_ids=list(cond_ids),
                                  modular_local_cond_null_values=dict(null_values or {})),
        device="cpu",
    )
    m.INTERNAL_LOCAL_CONDS = W.INTERNAL_LOCAL_CONDS
    m.control_cond_ids = lambda: W.control_cond_ids(m)
    return m


def batch(controls_per_item):
    return [{"controls": c} for c in controls_per_item]


def test_control_ids_exclude_internal_masks():
    m = fake_module(["drums_rms", "tf_inpaint_mask", "streamgen_latent", "inpaint_mask"])
    assert m.control_cond_ids() == ["drums_rms", "streamgen_latent"]


def test_drums_rms_is_stacked_and_gated():
    m = fake_module(["drums_rms", "tf_inpaint_mask"])
    md = batch([{"drums_rms": torch.ones(1, 6)}, {"drums_rms": torch.full((1, 6), 2.0)}])
    tf = torch.tensor([[[1, 1, 1, 0, 0, 0]], [[1, 1, 1, 1, 0, 0]]], dtype=torch.float32)
    cond = {}
    W._add_streamgen_conditioning(m, cond, md, tf)
    out = cond["drums_rms"][0]
    assert out.shape == (2, 1, 6)
    assert torch.equal(out[0, 0], torch.tensor([1., 1., 1., 0., 0., 0.]))
    assert torch.equal(out[1, 0], torch.tensor([2., 2., 2., 2., 0., 0.]))
    assert "tf_inpaint_mask" not in cond, "the mask is the caller's to add"


def test_streamgen_latent_still_works():
    m = fake_module(["streamgen_latent", "tf_inpaint_mask"])
    md = batch([{"streamgen_latent": torch.randn(256, 4)}])
    cond = {}
    W._add_streamgen_conditioning(m, cond, md, None)
    assert cond["streamgen_latent"][0].shape == (1, 256, 4)


def test_missing_control_is_an_error_naming_the_id():
    m = fake_module(["drums_rms"])
    md = batch([{"drums_rms": torch.ones(1, 4)}, {}])
    with pytest.raises(ValueError, match="drums_rms .* 1/2 items"):
        W._add_streamgen_conditioning(m, {}, md, None)


def test_no_controls_configured_is_a_no_op():
    m = fake_module(["tf_inpaint_mask"])
    cond = {}
    W._add_streamgen_conditioning(m, cond, batch([{}]), None)
    assert cond == {}


# --- control CFG dropout -------------------------------------------------------------

from stable_audio_3.training.diffusion import CONTROL_NULL_VALUE_DEFAULT  # noqa: E402


def dropout_fixture(null_values=None):
    m = fake_module(["drums_rms", "drums_tria_fixed", "tf_inpaint_mask"], null_values)
    md = batch([{"drums_rms": torch.full((1, 6), 0.5), "drums_tria_fixed": torch.full((2, 6), 0.5)} for _ in range(64)])
    tf = torch.ones(64, 1, 6)
    tf[:, :, 4:] = 0
    return m, md, tf


def test_dropout_zero_changes_nothing():
    m, md, tf = dropout_fixture()
    cond = {"tf_inpaint_mask": [tf]}
    W._add_streamgen_conditioning(m, cond, md, tf, dropout_prob=0.0)
    assert torch.equal(cond["tf_inpaint_mask"][0], tf)
    assert torch.equal(cond["drums_rms"][0], 0.5 * tf)


def test_dropout_one_puts_the_null_token_under_an_untouched_mask():
    m, md, tf = dropout_fixture()
    cond = {"tf_inpaint_mask": [tf]}
    W._add_streamgen_conditioning(m, cond, md, tf, dropout_prob=1.0)
    assert torch.equal(cond["tf_inpaint_mask"][0], tf), "the mask must still say where the control would be visible"
    assert CONTROL_NULL_VALUE_DEFAULT > 1.0, "the token has to sit outside the [0, 1] feature range"
    for cid, ch in (("drums_rms", 1), ("drums_tria_fixed", 2)):
        out = cond[cid][0]
        # Token on the visible frames, hidden frames still 0.
        assert torch.equal(out, CONTROL_NULL_VALUE_DEFAULT * tf.expand(-1, ch, -1))


def test_dropout_uses_the_configured_null_value():
    m, md, tf = dropout_fixture({"drums_rms": 7.0})
    cond = {"tf_inpaint_mask": [tf]}
    W._add_streamgen_conditioning(m, cond, md, tf, dropout_prob=1.0)
    assert torch.equal(cond["drums_rms"][0], 7.0 * tf)
    assert torch.equal(cond["drums_tria_fixed"][0], CONTROL_NULL_VALUE_DEFAULT * tf.expand(-1, 2, -1))


def test_dropout_is_per_item_and_shared_across_controls():
    torch.manual_seed(0)
    m, md, tf = dropout_fixture()
    cond = {"tf_inpaint_mask": [tf]}
    W._add_streamgen_conditioning(m, cond, md, tf, dropout_prob=0.5)
    dropped = cond["drums_rms"][0][:, 0, 0] == CONTROL_NULL_VALUE_DEFAULT
    assert 0 < int(dropped.sum()) < 64, "p=0.5 over 64 items should drop some and keep some"
    for cid, ch in (("drums_rms", 1), ("drums_tria_fixed", 2)):
        out = cond[cid][0]
        assert torch.equal(out[dropped], CONTROL_NULL_VALUE_DEFAULT * tf[dropped].expand(-1, ch, -1))
        assert torch.equal(out[~dropped], 0.5 * tf[~dropped].expand(-1, ch, -1))


def test_dropout_without_tf_mask_fills_every_frame():
    m, md, _ = dropout_fixture()
    cond = {}
    W._add_streamgen_conditioning(m, cond, md, None, dropout_prob=1.0)
    assert "tf_inpaint_mask" not in cond
    assert torch.equal(cond["drums_rms"][0], torch.full((64, 1, 6), CONTROL_NULL_VALUE_DEFAULT))


# --- inpaint context CFG dropout -----------------------------------------------------

from stable_audio_3.training.diffusion import apply_inpaint_dropout  # noqa: E402


def inpaint_fixture():
    mask = torch.ones(64, 1, 6)
    mask[:, :, 3:] = 0                      # causal: context on the first 3 frames
    latents = torch.randn(64, 4, 6)
    return latents * mask, mask, latents


def test_inpaint_dropout_zero_changes_nothing():
    mi, m, _ = inpaint_fixture()
    out_mi, out_m = apply_inpaint_dropout(mi, m, 0.0)
    assert torch.equal(out_mi, mi) and torch.equal(out_m, m)


def test_inpaint_dropout_one_nulls_the_values_and_keeps_the_mask():
    mi, m, _ = inpaint_fixture()
    out_mi, out_m = apply_inpaint_dropout(mi, m, 1.0)
    assert torch.equal(out_m, m), "the mask must still say where the context would be"
    assert not out_mi.any(), "no context values given"


def test_inpaint_dropout_is_per_item():
    torch.manual_seed(0)
    mi, m, _ = inpaint_fixture()
    out_mi, out_m = apply_inpaint_dropout(mi, m, 0.5)
    assert torch.equal(out_m, m)
    dropped = ~out_mi.flatten(1).any(dim=1)
    assert 0 < int(dropped.sum()) < 64
    assert torch.equal(out_mi[~dropped], mi[~dropped])
