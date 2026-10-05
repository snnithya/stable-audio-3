"""Nested classifier-free guidance over three conditions: prompt, inpainting context, control.

The WJD arms (experiments 3.3 / 3.4) are trained with an independent CFG dropout on each of
the three conditions, so each has a null the model has seen:

    prompt   -> zeroed cross-attention / prepend tokens (the DiT's own ``cfg_dropout_prob``)
    context  -> ``inpaint_masked_input`` set to the zero latent, ``inpaint_mask`` kept
                (``training.diffusion.apply_inpaint_dropout``)
    control  -> every sidecar control (``drums_rms``, ``drums_tria_*``, ``streamgen_latent``)
                set to its ``null_value`` on the frames ``tf_inpaint_mask`` marks visible, 0
                elsewhere, the mask kept (``DiffusionCondTrainingWrapper._add_streamgen_conditioning``)

This module composes the three guidances InstructPix2Pix-style, as sat-zenon's
``make_multicfg_denoiser`` did for two axes.  For an ``order`` ``(a, b, c)`` of the axes and
scales ``s_a, s_b, s_c``:

    v = v(none) + s_a (v(a) - v(none)) + s_b (v(a,b) - v(a)) + s_c (v(a,b,c) - v(a,b))

i.e. each condition is added on top of the ones before it in the order.  Rearranged by branch,

    v = (1 - s_a) v(none) + (s_a - s_b) v(a) + (s_b - s_c) v(a,b) + s_c v(a,b,c)

so a branch whose coefficient is 0 is not computed: all scales 1 is one forward pass, and the
order ``(context, control, prompt)`` with ``s_context = s_control = 1`` is exactly the DiT's
standard prompt CFG (two passes: with and without the prompt, context and control on in both),
which is what the training demos and ``scripts/wjd/listen_wjd_arms.py`` use.

The branches run as one batch through the DiT with its own ``cfg_scale`` pinned to 1, so this
sits beside ``sample_diffusion`` rather than inside it: build the denoiser with
``make_multi_cfg_denoiser`` and pass it as ``model`` with ``cond_inputs={}``.  No APG / rescale
on the composed estimate; this is vanilla CFG on each axis.
"""

from __future__ import annotations

import itertools
import typing as tp

import torch

AXES = ("prompt", "context", "control")
DEFAULT_ORDER = AXES
ORDERS = tuple(itertools.permutations(AXES))

# Ids the training step builds itself; everything else under modular_local_cond_ids is a
# sidecar control (DiffusionCondTrainingWrapper.INTERNAL_LOCAL_CONDS).
INTERNAL_LOCAL_CONDS = ("tf_inpaint_mask", "inpaint_mask", "inpaint_masked_input")
# Same fallback as training.diffusion.CONTROL_NULL_VALUE_DEFAULT (not imported: that module
# pulls in Lightning).
CONTROL_NULL_VALUE_DEFAULT = 1.0 + 1.0 / 32


def control_ids(model) -> tp.List[str]:
    """Sidecar control ids of a ConditionedDiffusionModelWrapper (``[]`` for a text-only arm)."""
    return [c for c in getattr(model, "modular_local_cond_ids", []) if c not in INTERNAL_LOCAL_CONDS]


# ---------------------------------------------------------------------------
# Nulls, one per axis
# ---------------------------------------------------------------------------


def null_context(cond: dict) -> dict:
    """Context null: the masked input becomes the zero latent, ``inpaint_mask`` is untouched."""
    out = dict(cond)
    if "inpaint_masked_input" in out:
        x = out["inpaint_masked_input"][0]
        out["inpaint_masked_input"] = [torch.zeros_like(x)]
    return out


def null_control(cond: dict, ids: tp.Sequence[str], null_values: tp.Mapping[str, float]) -> dict:
    """Control null: each control in ``ids`` becomes its ``null_value`` under the tf mask.

    The token sits on the frames ``tf_inpaint_mask`` marks visible and the hidden frames stay 0,
    exactly as the training dropout writes it; without a tf mask the whole control is the token.
    """
    out = dict(cond)
    tf = out["tf_inpaint_mask"][0] if "tf_inpaint_mask" in out else None
    for cid in ids:
        if cid not in out:
            continue
        x = out[cid][0]
        null = torch.full_like(x, float(null_values.get(cid, CONTROL_NULL_VALUE_DEFAULT)))
        if tf is not None:
            null = null * tf.to(null.dtype)
        out[cid] = [null]
    return out


def null_prompt(inputs: dict) -> dict:
    """Prompt null on the DiT inputs: zero the cross-attention and prepend tokens, as the DiT's
    own CFG branch and its training dropout do. Masks and the global (duration) embedding stay."""
    out = dict(inputs)
    for key in ("cross_attn_cond", "prepend_cond"):
        if out.get(key) is not None:
            out[key] = torch.zeros_like(out[key])
    return out


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------


def nested_branches(scales: tp.Mapping[str, float], order: tp.Sequence[str] = DEFAULT_ORDER):
    """``[(frozenset of axes on, coefficient)]`` for the nested composition, zero coefficients dropped.

    ``scales`` maps axis -> guidance scale (missing axes count as 1, i.e. always on and never
    contrasted). The first entry is the most unconditional branch.
    """
    order = tuple(order)
    if sorted(order) != sorted(AXES):
        raise ValueError(f"order must be a permutation of {AXES}, got {order}")
    s = [float(scales.get(a, 1.0)) for a in order]
    # v = v0 + s0 (v1 - v0) + s1 (v2 - v1) + s2 (v3 - v2); collect per branch.
    coeffs = [1.0 - s[0], s[0] - s[1], s[1] - s[2], s[2]]
    branches = []
    for k, c in enumerate(coeffs):
        if c != 0.0:
            branches.append((frozenset(order[:k]), c))
    return branches


def branch_inputs(model, cond: dict, on: tp.Collection[str], ids: tp.Sequence[str] = None) -> dict:
    """DiT inputs for one branch: the conditions in ``on`` as given, the others nulled."""
    if ids is None:
        ids = control_ids(model)
    c = cond
    if "context" not in on:
        c = null_context(c)
    if "control" not in on and ids:
        c = null_control(c, ids, getattr(model, "modular_local_cond_null_values", {}) or {})
    inputs = model.get_conditioning_inputs(c)
    if "prompt" not in on:
        inputs = null_prompt(inputs)
    return inputs


def _cat(values):
    """Concatenate matching DiT inputs over the batch; None stays None, dicts are joined per key."""
    first = values[0]
    if first is None:
        return None
    if isinstance(first, dict):
        return {k: torch.cat([v[k] for v in values], dim=0) for k in first}
    return torch.cat(list(values), dim=0)


def make_multi_cfg_denoiser(
    model,
    cond: dict,
    scales: tp.Mapping[str, float],
    order: tp.Sequence[str] = DEFAULT_ORDER,
    dtype: torch.dtype = None,
):
    """A ``model(x, t, **extra)`` callable for ``sample_diffusion`` that applies the nested CFG.

    Args:
        model: a ``ConditionedDiffusionModelWrapper`` (``model.model`` is the DiT wrapper).
        cond: the full conditioning dict as ``model.conditioner`` returns it, with
            ``inpaint_mask``, ``inpaint_masked_input``, ``tf_inpaint_mask`` and every control
            already attached and gated (what ``get_conditioning_inputs`` expects).
        scales: ``{"prompt": s, "context": s, "control": s}``; a missing axis is 1.
        order: which condition is added first, second, third (see module docstring).
        dtype: dtype for the conditioning tensors; defaults to the DiT's parameter dtype.

    Returns ``(denoiser, branches)``; ``branches`` is ``nested_branches(scales, order)`` after
    the collapse for a model without the axis (a text-only arm has no control to contrast).
    The denoiser repeats ``x``, ``t`` and ``padding_mask`` once per branch, runs one batched
    forward with the DiT's own ``cfg_scale`` pinned to 1, and returns the weighted sum. Other
    sampler kwargs that only steer the DiT's internal CFG (``apg_scale``, ``rescale_cfg``, ...)
    are dropped.
    """
    ids = control_ids(model)
    scales = dict(scales)
    if not ids:
        scales["control"] = 1.0  # nothing to null; the axis collapses
    if "inpaint_masked_input" not in cond:
        scales["context"] = 1.0
    branches = nested_branches(scales, order)

    if dtype is None:
        dtype = next(model.model.parameters()).dtype

    per_branch = [branch_inputs(model, cond, on, ids) for on, _ in branches]
    keys = per_branch[0].keys()
    batched = {k: _cat([b[k] for b in per_branch]) for k in keys}

    def to_dtype(v):
        if isinstance(v, dict):
            return {k: to_dtype(x) for k, x in v.items()}
        if torch.is_tensor(v) and v.is_floating_point():
            return v.to(dtype)
        return v

    batched = {k: to_dtype(v) for k, v in batched.items()}
    coeffs = [c for _, c in branches]
    n = len(branches)
    # Kwargs sample_diffusion forwards that would re-enter the DiT's own CFG or do not apply.
    drop = {"cfg_scale", "batch_cfg", "rescale_cfg", "apg_scale", "scale_phi", "cfg_interval",
            "cfg_norm_threshold", "negative_cross_attn_cond", "negative_cross_attn_mask",
            "negative_global_cond", "negative_input_concat_cond"}

    def denoiser(x, t, padding_mask=None, **extra):
        extra = {k: v for k, v in extra.items() if k not in drop and k not in batched}
        if padding_mask is not None:
            padding_mask = padding_mask.repeat(n, *([1] * (padding_mask.dim() - 1)))
        out = model.model(
            x.repeat(n, 1, 1), t.repeat(n), **batched,
            cfg_scale=1.0, batch_cfg=True, padding_mask=padding_mask, **extra,
        )
        if n == 1:
            return out
        parts = out.float().chunk(n, dim=0)
        v = sum(c * p for c, p in zip(coeffs, parts))
        return v.to(out.dtype)

    return denoiser, branches


def describe_branches(branches, scales: tp.Mapping[str, float], order: tp.Sequence[str]) -> str:
    """One line per branch, for logs and the UI."""
    lines = [f"order {' -> '.join(order)}; scales " + ", ".join(f"{a} {float(scales.get(a, 1.0)):g}" for a in order)]
    for on, c in branches:
        name = ", ".join(a for a in AXES if a in on) or "none"
        lines.append(f"  {c:+.3g} x v({name})")
    return "\n".join(lines)
