"""Tests for per-frame (diffusion forcing) timestep conditioning in the DiT.

Under diffusion forcing `t` is (b, n), one noise level per latent frame. The tokens the
transformer prepends (64 memory tokens by default) are not latent frames and are conditioned
at a fixed t instead. Two things need verifying independently of whether diffusion forcing
trains well:

  * value plumbing -- when every frame sits at the fixed t, the per-frame path reproduces the
    global-t path exactly (test_matches_global_t_at_the_register_t);
  * alignment -- frame i's noise level modulates token num_memory_tokens + i and nothing
    else (test_frame_i_modulates_token_i). A uniform-t comparison cannot see this, because
    every row of the modulation tensor is identical.

See experiments/03-diffusion-forcing/.
"""

import warnings

import pytest
import torch

import stable_audio_3.models.transformer as transformer_module
from stable_audio_3.models.dit import DiffusionTransformer

B, C, N = 2, 16, 24
GLOBAL_COND_DIM = 32
REGISTER_T = 0.537


@pytest.fixture(autouse=True)
def sdpa_attention(monkeypatch):
    """Force the SDPA path. flash-attn is CUDA-only and casts to fp16, which would swamp
    the exact-equivalence comparisons these tests rely on."""
    monkeypatch.setattr(transformer_module, "flash_attn_func", None)
    monkeypatch.setattr(transformer_module, "flash_attn_varlen_func", None)


def _make_model(seed=0, **kwargs):
    torch.manual_seed(seed)
    kwargs.setdefault("num_memory_tokens", 64)
    kwargs.setdefault("global_cond_type", "adaLN")
    kwargs.setdefault("depth", 2)
    kwargs.setdefault("df_register_cond_t", REGISTER_T)
    kwargs.setdefault("diffusion_objective", "rectified_flow")
    return DiffusionTransformer(
        io_channels=C,
        embed_dim=128,
        num_heads=2,
        global_cond_dim=GLOBAL_COND_DIM,
        timestep_features_type="expo",
        norm_type="rms_norm",
        # Without this every residual branch outputs exactly zero, so the conditioning
        # cannot reach the output and the comparisons below would pass vacuously.
        zero_init_branch_outputs=False,
        **kwargs,
    ).eval()


def _inputs(seed=1):
    torch.manual_seed(seed)
    return torch.randn(B, C, N), torch.randn(B, GLOBAL_COND_DIM)


def _uniform(value):
    """(b, n) with the same noise level on every frame."""
    return torch.full((B, N), value)


# --------------------------------------------------------------------------------------
# value plumbing
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_kwargs",
    [{}, {"num_memory_tokens": 0}, {"timestep_features_logsnr": True}],
    ids=["memory_tokens", "no_memory_tokens", "logsnr"],
)
def test_matches_global_t_at_the_register_t(model_kwargs):
    """With every frame at the register t, latents and registers all get the same modulation
    as under a global t of that value, so the outputs must agree to float32 roundoff."""
    model = _make_model(**model_kwargs)
    x, global_embed = _inputs()
    t_global = torch.full((B,), REGISTER_T)

    with torch.no_grad():
        out_global = model(x, t_global, global_embed=global_embed)
        out_per_frame = model(x, _uniform(REGISTER_T), global_embed=global_embed)

    assert out_global.shape == out_per_frame.shape == (B, C, N)
    torch.testing.assert_close(out_global, out_per_frame, rtol=0, atol=1e-5)


def test_matches_global_t_without_global_embed():
    """With no global conditioning the timestep is the only global signal -- a different
    branch in _forward."""
    model = _make_model()
    x, _ = _inputs()

    with torch.no_grad():
        out_global = model(x, torch.full((B,), REGISTER_T))
        out_per_frame = model(x, _uniform(REGISTER_T))

    torch.testing.assert_close(out_global, out_per_frame, rtol=0, atol=1e-5)


def test_matches_global_t_with_padding_mask():
    model = _make_model()
    x, global_embed = _inputs()
    padding_mask = torch.ones(B, N, dtype=torch.bool)
    padding_mask[:, -6:] = False

    with torch.no_grad():
        out_global = model(x, torch.full((B,), REGISTER_T), global_embed=global_embed, padding_mask=padding_mask)
        out_per_frame = model(x, _uniform(REGISTER_T), global_embed=global_embed, padding_mask=padding_mask)

    torch.testing.assert_close(out_global, out_per_frame, rtol=0, atol=1e-5)


def test_differs_from_global_t_away_from_the_register_t():
    """Documents the design: at any other uniform t the registers sit at the fixed t while the
    latents do not, so the per-frame path is deliberately *not* the global path."""
    model = _make_model()
    x, global_embed = _inputs()

    with torch.no_grad():
        out_global = model(x, torch.full((B,), 0.9), global_embed=global_embed)
        out_per_frame = model(x, _uniform(0.9), global_embed=global_embed)

    assert (out_global - out_per_frame).abs().max() > 1e-3


def test_varying_per_frame_t_changes_the_output():
    """Guards against the equivalence tests passing because the timestep conditioning is
    not reaching the output at all."""
    model = _make_model()
    x, global_embed = _inputs()
    torch.manual_seed(7)

    with torch.no_grad():
        out_uniform = model(x, _uniform(REGISTER_T), global_embed=global_embed)
        out_varying = model(x, torch.rand(B, N), global_embed=global_embed)

    assert (out_uniform - out_varying).abs().max() > 1e-3


# --------------------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------------------

def _positionwise_model(**kwargs):
    """One block with the attention branch silenced, so each token's hidden state depends
    only on its own input and its own adaLN modulation row. That makes the first layer's
    hidden states a direct readout of which modulation row landed on which token."""
    model = _make_model(depth=1, **kwargs)
    with torch.no_grad():
        model.transformer.layers[0].self_attn.to_out.weight.zero_()
    return model


def _first_layer_delta(model, x, global_embed, t_a, t_b):
    with torch.no_grad():
        _, info_a = model(x, t_a, global_embed=global_embed, return_info=True)
        _, info_b = model(x, t_b, global_embed=global_embed, return_info=True)
    # (b, num_memory_tokens + n) -- per-token magnitude of the change
    return (info_a["hidden_states"][0] - info_b["hidden_states"][0]).abs().amax(dim=-1)


@pytest.mark.parametrize("frame", [0, 7, N - 1])
def test_frame_i_modulates_token_i(frame):
    """Perturbing frame i's noise level changes token num_memory_tokens + i and no other."""
    model = _positionwise_model()
    n_mem = model.transformer.num_memory_tokens
    x, global_embed = _inputs()

    t_a = _uniform(REGISTER_T)
    t_b = t_a.clone()
    t_b[:, frame] = 0.9

    delta = _first_layer_delta(model, x, global_embed, t_a, t_b)
    changed = delta > 1e-6
    expected = torch.zeros_like(changed)
    expected[:, n_mem + frame] = True
    assert torch.equal(changed, expected), (
        f"frame {frame} should map to token {n_mem + frame}; "
        f"changed tokens: {changed[0].nonzero().flatten().tolist()}"
    )


def test_register_t_modulates_only_the_memory_tokens():
    """The complement: changing the fixed register t moves every memory token and no latent."""
    x, global_embed = _inputs()
    model_a = _positionwise_model(df_register_cond_t=REGISTER_T)
    model_b = _positionwise_model(df_register_cond_t=0.9)
    n_mem = model_a.transformer.num_memory_tokens

    t = torch.rand(B, N)
    with torch.no_grad():
        _, info_a = model_a(x, t, global_embed=global_embed, return_info=True)
        _, info_b = model_b(x, t, global_embed=global_embed, return_info=True)
    delta = (info_a["hidden_states"][0] - info_b["hidden_states"][0]).abs().amax(dim=-1)

    assert (delta[:, :n_mem] > 1e-6).all(), "every memory token should move"
    assert (delta[:, n_mem:] <= 1e-6).all(), "no latent frame should move"


def test_registers_are_independent_of_the_schedule():
    """The point of a fixed register t: however the per-frame noise levels are distributed,
    the memory tokens see the same modulation."""
    model = _positionwise_model()
    n_mem = model.transformer.num_memory_tokens
    x, global_embed = _inputs()
    torch.manual_seed(3)

    delta = _first_layer_delta(model, x, global_embed, torch.rand(B, N) * 0.2, torch.rand(B, N) * 0.2 + 0.8)
    assert (delta[:, :n_mem] <= 1e-6).all()
    assert (delta[:, n_mem:] > 1e-6).any()


# --------------------------------------------------------------------------------------
# guards and gradients
# --------------------------------------------------------------------------------------

def test_per_frame_t_requires_adaln():
    model = _make_model(global_cond_type="prepend")
    x, global_embed = _inputs()
    with pytest.raises(AssertionError, match="adaLN"):
        model(x, _uniform(REGISTER_T), global_embed=global_embed)


def test_per_frame_t_requires_patch_size_one():
    model = _make_model(patch_size=2)
    x, global_embed = _inputs()
    with pytest.raises(AssertionError, match="patch_size"):
        model(x, _uniform(REGISTER_T), global_embed=global_embed)


def test_gradients_reach_the_timestep_embedding():
    model = _make_model()
    x, global_embed = _inputs()
    torch.manual_seed(5)

    model.zero_grad()
    model(x, torch.rand(B, N), global_embed=global_embed).sum().backward()

    grad = model.to_timestep_embed[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0


# --------------------------------------------------------------------------------------
# forward(): the CFG / inference path (step 3)
# --------------------------------------------------------------------------------------

COND_TOKEN_DIM = 24
N_COND_TOKENS = 5


def _cfg_model(seed=0, **kwargs):
    kwargs.setdefault("cond_token_dim", COND_TOKEN_DIM)
    return _make_model(seed=seed, **kwargs)


def _cfg_inputs(seed=1):
    x, global_embed = _inputs(seed)
    torch.manual_seed(seed + 100)
    cross_attn_cond = torch.randn(B, N_COND_TOKENS, COND_TOKEN_DIM)
    return x, global_embed, cross_attn_cond


def _forward(model, x, t, global_embed, cross_attn_cond, **kwargs):
    with torch.no_grad():
        return model(x, t, cross_attn_cond=cross_attn_cond, global_embed=global_embed, **kwargs)


@pytest.mark.parametrize("objective", ["rectified_flow", "v"])
def test_cfg_matches_global_t_at_the_register_t(objective):
    """cfg_scale > 1 enters the batched cond/uncond branch of forward(), which reconstructs
    the denoised estimate from x, sigma (and alpha for the v objective) -- every broadcast
    of sigma against x must also accept (b, n) t."""
    model = _cfg_model(diffusion_objective=objective)
    x, global_embed, cross_attn_cond = _cfg_inputs()

    out_global = _forward(model, x, torch.full((B,), REGISTER_T), global_embed, cross_attn_cond, cfg_scale=3.0)
    out_per_frame = _forward(model, x, _uniform(REGISTER_T), global_embed, cross_attn_cond, cfg_scale=3.0)

    assert out_global.shape == out_per_frame.shape == (B, C, N)
    torch.testing.assert_close(out_global, out_per_frame, rtol=0, atol=1e-4)


def test_cfg_changes_the_output():
    """Tripwire: if cross-attention conditioning did not reach the model, every CFG test
    below would pass with cond == uncond."""
    model = _cfg_model()
    x, global_embed, cross_attn_cond = _cfg_inputs()
    t = torch.full((B, N), 0.7)

    out_cfg = _forward(model, x, t, global_embed, cross_attn_cond, cfg_scale=3.0)
    out_no_cfg = _forward(model, x, t, global_embed, cross_attn_cond, cfg_scale=1.0)

    assert (out_cfg - out_no_cfg).abs().max() > 1e-3


def test_cfg_interval_is_per_frame():
    """Under diffusion forcing "is sigma in the CFG window" is a per-frame question. Frames
    outside the window must come out as the plain conditional estimate; frames inside as the
    guided one."""
    model = _cfg_model()
    x, global_embed, cross_attn_cond = _cfg_inputs()
    half = N // 2
    t = torch.empty(B, N)
    t[:, :half] = 0.2   # outside (0.5, 1.0)
    t[:, half:] = 0.8   # inside

    out = _forward(model, x, t, global_embed, cross_attn_cond, cfg_scale=3.0, cfg_interval=(0.5, 1.0))
    out_no_cfg = _forward(model, x, t, global_embed, cross_attn_cond, cfg_scale=1.0)
    out_full_cfg = _forward(model, x, t, global_embed, cross_attn_cond, cfg_scale=3.0, cfg_interval=(0.0, 1.0))

    torch.testing.assert_close(out[..., :half], out_no_cfg[..., :half], rtol=0, atol=1e-5)
    torch.testing.assert_close(out[..., half:], out_full_cfg[..., half:], rtol=0, atol=1e-5)
    assert (out[..., half:] - out_no_cfg[..., half:]).abs().max() > 1e-3


def test_cfg_interval_is_per_item_for_global_t():
    """The per-element-schedule branch of sample_discrete_euler hands items different noise
    levels; previously the whole batch was gated on item 0's sigma."""
    model = _cfg_model()
    x, global_embed, cross_attn_cond = _cfg_inputs()
    t = torch.tensor([0.2, 0.8])   # item 0 outside (0.5, 1.0), item 1 inside

    out = _forward(model, x, t, global_embed, cross_attn_cond, cfg_scale=3.0, cfg_interval=(0.5, 1.0))
    out_no_cfg = _forward(model, x, t, global_embed, cross_attn_cond, cfg_scale=1.0)
    out_full_cfg = _forward(model, x, t, global_embed, cross_attn_cond, cfg_scale=3.0, cfg_interval=(0.0, 1.0))

    torch.testing.assert_close(out[0], out_no_cfg[0], rtol=0, atol=1e-5)
    torch.testing.assert_close(out[1], out_full_cfg[1], rtol=0, atol=1e-5)
    assert (out[1] - out_no_cfg[1]).abs().max() > 1e-3


def test_cfg_skipped_when_no_frame_is_in_the_window():
    """With every frame outside the window the batched pass is not run at all, so the result
    is bit-identical to cfg_scale=1."""
    model = _cfg_model()
    x, global_embed, cross_attn_cond = _cfg_inputs()
    t = torch.full((B, N), 0.2)

    out = _forward(model, x, t, global_embed, cross_attn_cond, cfg_scale=3.0, cfg_interval=(0.5, 1.0))
    out_no_cfg = _forward(model, x, t, global_embed, cross_attn_cond, cfg_scale=1.0)

    assert torch.equal(out, out_no_cfg)


def test_lora_interval_warns_with_per_frame_t():
    """Enabling an adapter is module-wide, so the LoRA interval has no per-frame version. A
    non-default interval with per-frame t warns and gates on sigma[0, 0]; the default
    (0, 1) interval is silent."""
    from stable_audio_3.models.lora.model import add_lora

    model = _make_model()
    add_lora(model)
    model.eval()
    x, global_embed = _inputs()
    t = torch.rand(B, N)

    with pytest.warns(UserWarning, match="lora_interval"):
        with torch.no_grad():
            out = model(x, t, global_embed=global_embed, lora_interval=(0.3, 1.0))
    assert out.shape == (B, C, N) and torch.isfinite(out).all()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with torch.no_grad():
            model(x, t, global_embed=global_embed)


@pytest.mark.parametrize(
    "extra",
    [{}, {"cfg_norm_threshold": 1.0}, {"apg_scale": 1.0}, {"apg_scale": 0.5}],
    ids=["plain", "norm_threshold", "apg_full", "apg_blend"],
)
def test_cfg_with_padding_mask_matches_global_t_at_the_register_t(extra):
    """The last cell of {1-D, 2-D} x {cfg_scale} x {padding_mask}: the CFG-norm and APG
    branches consume padding_mask as (b, 1, n) alongside the now-(b, 1, n) sigma."""
    model = _cfg_model()
    x, global_embed, cross_attn_cond = _cfg_inputs()
    padding_mask = torch.ones(B, N, dtype=torch.bool)
    padding_mask[:, -6:] = False

    out_global = _forward(model, x, torch.full((B,), REGISTER_T), global_embed, cross_attn_cond,
                          cfg_scale=3.0, padding_mask=padding_mask, **extra)
    out_per_frame = _forward(model, x, _uniform(REGISTER_T), global_embed, cross_attn_cond,
                             cfg_scale=3.0, padding_mask=padding_mask, **extra)

    assert torch.isfinite(out_per_frame).all()
    torch.testing.assert_close(out_global, out_per_frame, rtol=0, atol=1e-4)


# --------------------------------------------------------------------------------------
# config plumbing
# --------------------------------------------------------------------------------------

def test_df_register_cond_t_reaches_the_model_through_dit_wrapper():
    """factory.create_diffusion_cond_from_config does DiTWrapper(**diffusion.config), so a
    df_register_cond_t key in that block must land on the DiffusionTransformer."""
    from stable_audio_3.models.diffusion import DiTWrapper

    config = {"io_channels": C, "embed_dim": 64, "depth": 1, "num_heads": 2,
              "global_cond_type": "adaLN", "df_register_cond_t": 0.42}
    wrapper = DiTWrapper(diffusion_objective="rectified_flow", **config)
    assert wrapper.model.df_register_cond_t == 0.42

    default = DiTWrapper(diffusion_objective="rectified_flow", **{k: v for k, v in config.items() if k != "df_register_cond_t"})
    assert default.model.df_register_cond_t == REGISTER_T
