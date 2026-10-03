"""The training wrapper attaches every sidecar control the model config names, not just
`streamgen_latent`. Exercised on the unbound method with a stand-in for `self`, so no model
is built: what is under test is the id handling and the tf-mask gating.
"""

from types import SimpleNamespace

import pytest
import torch

from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper as W


def fake_module(cond_ids):
    m = SimpleNamespace(diffusion=SimpleNamespace(modular_local_cond_ids=list(cond_ids)), device="cpu")
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
