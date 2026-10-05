"""Nested three-axis CFG (stable_audio_3/inference/multi_cfg.py).

A stand-in DiT whose output encodes which conditions it was given, wrapped in the real
``ConditionedDiffusionModelWrapper`` so the branches go through the real
``get_conditioning_inputs``. What is under test: the three nulls match the training dropouts,
the branch coefficients implement the nested formula, zero-coefficient branches are skipped,
and the sampler-facing callable batches and un-batches correctly.
"""

import itertools

import pytest
import torch
from torch import nn

from stable_audio_3.inference.multi_cfg import (
    AXES,
    CONTROL_NULL_VALUE_DEFAULT,
    branch_inputs,
    control_ids,
    describe_branches,
    make_multi_cfg_denoiser,
    nested_branches,
    null_context,
    null_control,
    null_prompt,
)
from stable_audio_3.models.diffusion import ConditionedDiffusionModelWrapper

B, C, T = 2, 4, 6
NULL = 1.03125


class FakeDiT(nn.Module):
    """Returns a constant per item that identifies the conditions present: 1 for the prompt,
    10 for a non-zero masked input, 100 for a control that is not the null token."""

    def __init__(self, control="drums_rms"):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.control = control
        self.calls = []

    def forward(self, x, t, cross_attn_cond=None, local_add_cond=None, modular_local_cond=None,
                padding_mask=None, cfg_scale=1.0, **kw):
        assert cfg_scale == 1.0, "the composed denoiser must pin the DiT's own CFG to 1"
        self.calls.append({"batch": x.shape[0], "padding_mask": padding_mask, "kw": kw})
        code = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        if cross_attn_cond is not None:
            code += (cross_attn_cond.abs().flatten(1).sum(1) > 0).float()
        if local_add_cond is not None:
            code += 10 * (local_add_cond[:, 1:].abs().flatten(1).sum(1) > 0).float()
        if modular_local_cond is not None and self.control in modular_local_cond:
            ctrl = modular_local_cond[self.control]
            tf = modular_local_cond["tf_inpaint_mask"].expand_as(ctrl) > 0
            for i in range(x.shape[0]):  # per item: a batch mixes null and real branches
                visible = ctrl[i][tf[i]]
                is_null = visible.numel() > 0 and bool(((visible - NULL).abs() < 1e-6).all())
                code[i] += 100 * float(not is_null)
        return code.view(-1, 1, 1).expand_as(x).clone()


def make_model(control="drums_rms", with_control=True):
    ids = ["tf_inpaint_mask"] + ([control] if with_control else [])
    dit = FakeDiT(control)
    return ConditionedDiffusionModelWrapper(
        model=dit, conditioner=None, io_channels=C, sample_rate=44100, min_input_length=1,
        diffusion_objective="rf_denoiser",
        cross_attn_cond_ids=["prompt", "seconds_total"], global_cond_ids=["seconds_total"],
        local_add_cond_ids=["inpaint_mask", "inpaint_masked_input"],
        modular_local_cond_ids=ids, modular_local_cond_null_values={control: NULL},
    ), dit


def make_cond(control="drums_rms", with_control=True, cursor=3, horizon=4):
    inpaint = torch.zeros(B, 1, T)
    inpaint[:, :, :cursor] = 1
    tf = torch.zeros(B, 1, T)
    tf[:, :, :horizon] = 1
    cond = {
        "prompt": (torch.ones(B, 5, 8), torch.ones(B, 5)),
        "seconds_total": (torch.ones(B, 8), torch.ones(B)),
        "inpaint_mask": [inpaint],
        "inpaint_masked_input": [torch.randn(B, C, T) * inpaint],
        "tf_inpaint_mask": [tf],
    }
    if with_control:
        cond[control] = [torch.full((B, 1, T), 0.5) * tf]
    return cond


# --- nulls -------------------------------------------------------------------------------


def test_null_context_zeros_values_and_keeps_mask():
    cond = make_cond()
    out = null_context(cond)
    assert torch.equal(out["inpaint_masked_input"][0], torch.zeros(B, C, T))
    assert out["inpaint_mask"][0] is cond["inpaint_mask"][0]
    assert cond["inpaint_masked_input"][0][:, :, :3].abs().sum() > 0, "input untouched"


def test_null_control_puts_token_under_tf_mask_only():
    cond = make_cond(horizon=4)
    out = null_control(cond, ["drums_rms"], {"drums_rms": NULL})
    ctrl = out["drums_rms"][0]
    assert torch.equal(ctrl[:, :, :4], torch.full((B, 1, 4), NULL))
    assert torch.equal(ctrl[:, :, 4:], torch.zeros(B, 1, 2)), "hidden frames stay 0"
    assert out["tf_inpaint_mask"][0] is cond["tf_inpaint_mask"][0]


def test_null_control_default_token_and_missing_ids():
    cond = make_cond()
    out = null_control(cond, ["drums_rms", "not_there"], {})
    assert out["drums_rms"][0][0, 0, 0].item() == pytest.approx(CONTROL_NULL_VALUE_DEFAULT)
    assert "not_there" not in out


def test_null_prompt_zeros_tokens_keeps_global_and_masks():
    inputs = {"cross_attn_cond": torch.ones(B, 3, 8), "cross_attn_mask": torch.ones(B, 3),
              "global_cond": torch.ones(B, 8), "prepend_cond": None}
    out = null_prompt(inputs)
    assert torch.equal(out["cross_attn_cond"], torch.zeros(B, 3, 8))
    assert out["cross_attn_mask"] is inputs["cross_attn_mask"]
    assert out["global_cond"] is inputs["global_cond"]
    assert out["prepend_cond"] is None


def test_control_ids_excludes_internal_masks():
    model, _ = make_model()
    assert control_ids(model) == ["drums_rms"]
    model, _ = make_model(with_control=False)
    assert control_ids(model) == []


# --- branches ----------------------------------------------------------------------------


def test_nested_branches_all_ones_is_one_branch():
    assert nested_branches({"prompt": 1, "context": 1, "control": 1}) == [(frozenset(AXES), 1.0)]


def test_nested_branches_standard_prompt_cfg():
    br = nested_branches({"prompt": 4.0, "context": 1.0, "control": 1.0}, order=("context", "control", "prompt"))
    assert br == [(frozenset({"context", "control"}), -3.0), (frozenset(AXES), 4.0)]


def test_nested_branches_coefficients_follow_the_order():
    br = nested_branches({"prompt": 2.0, "context": 3.0, "control": 5.0}, order=("prompt", "context", "control"))
    assert br == [
        (frozenset(), 1.0 - 2.0),
        (frozenset({"prompt"}), 2.0 - 3.0),
        (frozenset({"prompt", "context"}), 3.0 - 5.0),
        (frozenset(AXES), 5.0),
    ]
    assert sum(c for _, c in br) == pytest.approx(1.0), "coefficients sum to 1 for every scale setting"


def test_nested_branches_rejects_bad_order():
    with pytest.raises(ValueError):
        nested_branches({}, order=("prompt", "prompt", "control"))


@pytest.mark.parametrize("order", list(itertools.permutations(AXES)))
def test_branch_inputs_null_the_right_conditions(order):
    model, _ = make_model()
    cond = make_cond()
    full = branch_inputs(model, cond, set(AXES))
    for k in range(4):
        on = set(order[:k])
        inputs = branch_inputs(model, cond, on)
        assert (inputs["cross_attn_cond"].abs().sum() > 0) == ("prompt" in on)
        assert (inputs["local_add_cond"][:, 1:].abs().sum() > 0) == ("context" in on)
        assert torch.equal(inputs["local_add_cond"][:, :1], full["local_add_cond"][:, :1]), "inpaint_mask kept"
        ctrl = inputs["modular_local_cond"]["drums_rms"]
        assert torch.equal(inputs["modular_local_cond"]["tf_inpaint_mask"], cond["tf_inpaint_mask"][0])
        if "control" in on:
            assert torch.equal(ctrl, cond["drums_rms"][0])
        else:
            assert torch.equal(ctrl, NULL * cond["tf_inpaint_mask"][0])


# --- the denoiser ------------------------------------------------------------------------


def codes(on):
    return (1 if "prompt" in on else 0) + (10 if "context" in on else 0) + (100 if "control" in on else 0)


@pytest.mark.parametrize("order", list(itertools.permutations(AXES)))
def test_denoiser_matches_nested_formula(order):
    model, dit = make_model()
    cond = make_cond()
    scales = {"prompt": 2.0, "context": 0.5, "control": 3.0}
    den, branches = make_multi_cfg_denoiser(model, cond, scales, order, dtype=torch.float32)
    x = torch.zeros(B, C, T)
    out = den(x, torch.full((B,), 0.5), cfg_scale=7.0, batch_cfg=True, rescale_cfg=True, apg_scale=1.0,
              padding_mask=torch.ones(B, T, dtype=torch.bool))
    expected = sum(c * codes(on) for on, c in branches)
    # Same thing written as the nested sum.
    s = [scales[a] for a in order]
    v = [codes(set(order[:k])) for k in range(4)]
    nested = v[0] + s[0] * (v[1] - v[0]) + s[1] * (v[2] - v[1]) + s[2] * (v[3] - v[2])
    assert expected == pytest.approx(nested)
    assert torch.allclose(out, torch.full_like(out, expected))
    call = dit.calls[-1]
    assert call["batch"] == len(branches) * B
    assert call["padding_mask"].shape == (len(branches) * B, T)
    assert "apg_scale" not in call["kw"] and "rescale_cfg" not in call["kw"]


def test_denoiser_all_ones_is_a_single_plain_forward():
    model, dit = make_model()
    den, branches = make_multi_cfg_denoiser(model, make_cond(), {"prompt": 1, "context": 1, "control": 1},
                                            dtype=torch.float32)
    out = den(torch.zeros(B, C, T), torch.full((B,), 0.5))
    assert len(branches) == 1 and dit.calls[-1]["batch"] == B
    assert torch.allclose(out, torch.full_like(out, 111.0))


def test_denoiser_reproduces_standard_prompt_cfg_on_v():
    """order (context, control, prompt), context = control = 1: v_full + (s-1)(v_full - v_noprompt),
    the DiT's own CFG on the velocity (before its APG / rescale options)."""
    model, dit = make_model()
    den, branches = make_multi_cfg_denoiser(model, make_cond(), {"prompt": 4.0}, ("context", "control", "prompt"),
                                            dtype=torch.float32)
    out = den(torch.zeros(B, C, T), torch.full((B,), 0.5))
    assert len(branches) == 2 and dit.calls[-1]["batch"] == 2 * B
    assert torch.allclose(out, torch.full_like(out, 111.0 + 3.0 * (111.0 - 110.0)))


def test_control_axis_collapses_without_a_control():
    model, dit = make_model(with_control=False)
    den, branches = make_multi_cfg_denoiser(model, make_cond(with_control=False),
                                            {"prompt": 1.0, "context": 1.0, "control": 5.0}, dtype=torch.float32)
    assert len(branches) == 1
    out = den(torch.zeros(B, C, T), torch.full((B,), 0.5))
    assert torch.allclose(out, torch.full_like(out, 11.0))


def test_denoiser_casts_conditioning_to_requested_dtype():
    model, dit = make_model()
    den, _ = make_multi_cfg_denoiser(model, make_cond(), {"prompt": 2.0}, dtype=torch.bfloat16)
    out = den(torch.zeros(B, C, T, dtype=torch.bfloat16), torch.full((B,), 0.5, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16
    # default order: v(none) + 2 (v(p) - v(none)) + (v(p,ctx) - v(p)) + (v(all) - v(p,ctx)) = 0 + 2 + 10 + 100
    assert torch.allclose(out.float(), torch.full((B, C, T), 112.0))


def test_describe_branches_lists_each_branch():
    scales = {"prompt": 2.0, "context": 1.0, "control": 1.0}
    br = nested_branches(scales)
    text = describe_branches(br, scales, AXES)
    assert "prompt -> context -> control" in text
    assert "v(none)" in text and "v(prompt, context, control)" in text
