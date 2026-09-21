"""Trainer-side tests for diffusion forcing (step 4 of
experiments/03-diffusion-forcing/01-local-timestep-conditioning.md).

Covers what the DiT module tests cannot: that DiffusionCondTrainingWrapper samples t of shape
(b, n) under df_training, mixes in shared-t items with df_p_global, survives the distribution
shift with per-item sequence lengths, and runs a full training_step -- noising, forward, masked
loss, backward -- with per-frame t. All at toy scale on CPU/fp32/SDPA.
"""

import types
import weakref

import pytest
import torch

import stable_audio_3.models.transformer as transformer_module
from stable_audio_3.inference.distribution_shift import (
    DistributionShift, FluxDistributionShift, LogSNRShift,
)
from stable_audio_3.models.conditioners import MultiConditioner, NumberConditioner
from stable_audio_3.models.diffusion import ConditionedDiffusionModelWrapper, DiTWrapper
from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper

B, C, N = 4, 8, 24
GLOBAL_COND_DIM = 32
SAMPLERS = ["uniform", "logit_normal", "trunc_logit_normal", "log_snr", "log_snr_uniform"]


@pytest.fixture(autouse=True)
def sdpa_attention(monkeypatch):
    monkeypatch.setattr(transformer_module, "flash_attn_func", None)
    monkeypatch.setattr(transformer_module, "flash_attn_varlen_func", None)


# --------------------------------------------------------------------------------------
# _sample_timesteps
# --------------------------------------------------------------------------------------

def _bare_wrapper(sampler, df_training, df_p_global=0.0, seed=0):
    """Just enough of the wrapper to call _sample_timesteps, without a model."""
    torch.manual_seed(seed)
    w = object.__new__(DiffusionCondTrainingWrapper)
    w.timestep_sampler = sampler
    w.rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=seed)
    w.mean_logsnr, w.std_logsnr = -1.2, 2.0
    w.min_logsnr, w.max_logsnr = -6.0, 5.0
    w.df_training = df_training
    w.df_p_global = df_p_global
    return w


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_global_t_shape_is_unchanged(sampler):
    t = _bare_wrapper(sampler, df_training=False)._sample_timesteps(B, N, "cpu")
    assert t.shape == (B,)
    assert ((t >= 0) & (t <= 1)).all()


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_per_frame_t_shape(sampler):
    """Every sampler must accept a (b, n) draw -- SobolEngine.draw in particular takes a
    count, not a shape."""
    t = _bare_wrapper(sampler, df_training=True)._sample_timesteps(B, N, "cpu")
    assert t.shape == (B, N)
    assert ((t >= 0) & (t <= 1)).all()
    # frames genuinely differ
    assert (t.amax(dim=1) - t.amin(dim=1) > 1e-3).all()


def test_df_p_global_one_collapses_every_item():
    t = _bare_wrapper("trunc_logit_normal", df_training=True, df_p_global=1.0)._sample_timesteps(B, N, "cpu")
    assert t.shape == (B, N)
    torch.testing.assert_close(t, t[:, :1].expand(B, N))


def test_df_p_global_mixes_shared_and_per_frame_items():
    w = _bare_wrapper("uniform", df_training=True, df_p_global=0.5)
    t = torch.cat([w._sample_timesteps(B, N, "cpu") for _ in range(16)], dim=0)  # (64, N)
    shared = (t.amax(dim=1) - t.amin(dim=1)) < 1e-6
    assert 10 < shared.sum() < 54, f"{shared.sum()} of 64 items shared a t; expected about half"


def test_uniform_sobol_stratifies_over_frames():
    """(b*n) Sobol points reshaped to (b, n): a quasirandom sequence fills [0, 1] much more
    evenly than iid uniform would, which is easy to check on a coarse histogram."""
    t = _bare_wrapper("uniform", df_training=True)._sample_timesteps(16, 64, "cpu")  # 1024 points
    counts = torch.histc(t.flatten(), bins=16, min=0, max=1)
    assert (counts == 64).all(), counts


# --------------------------------------------------------------------------------------
# distribution shift with per-item sequence lengths
# --------------------------------------------------------------------------------------

SHIFTS = [
    pytest.param(FluxDistributionShift(min_length=16, max_length=512, alpha_min=1.0, alpha_max=4.0), id="flux"),
    pytest.param(DistributionShift(min_length=16, max_length=512), id="full"),
    pytest.param(LogSNRShift(anchor_length=64), id="logsnr"),
]


@pytest.mark.parametrize("shift", SHIFTS)
@pytest.mark.parametrize("n", [N, B], ids=["n!=b", "n==b"])
def test_per_frame_t_shift_matches_per_item_shift(shift, n):
    """Shifting (b, n) t with (b,) sequence lengths must equal shifting each item's (n,) row
    by its own scalar length. n == b is the case that used to broadcast silently and wrongly."""
    torch.manual_seed(0)
    t = torch.rand(B, n) * 0.98 + 0.01
    seq_len = torch.tensor([20, 64, 200, 500])

    shifted = shift.shift(t, seq_len)
    expected = torch.stack([shift.shift(t[i], int(seq_len[i])) for i in range(B)])

    assert shifted.shape == (B, n)
    torch.testing.assert_close(shifted, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shift", SHIFTS)
def test_schedule_broadcast_branch_is_unchanged(shift):
    """The pre-existing (steps,) x (batch,) -> (batch, steps) branch used by the samplers."""
    steps = torch.linspace(0.05, 0.95, 7)
    seq_len = torch.tensor([20, 64, 200, 500])
    out = shift.shift(steps, seq_len)
    assert out.shape == (B, 7)
    torch.testing.assert_close(out[2], shift.shift(steps, 200), rtol=1e-5, atol=1e-6)


def test_shift_rejects_mismatched_batch():
    with pytest.raises(AssertionError, match="per-item shift parameter"):
        DistributionShift(min_length=16, max_length=512).shift(torch.rand(B, N), torch.tensor([20, 64]))


# --------------------------------------------------------------------------------------
# training_step end to end
# --------------------------------------------------------------------------------------

def _tiny_trainer(df_training, df_p_global=0.0, dist_shift=None, seed=0, **wrapper_kwargs):
    torch.manual_seed(seed)
    dit = DiTWrapper(
        diffusion_objective="rectified_flow",
        io_channels=C,
        embed_dim=64,
        depth=2,
        num_heads=2,
        global_cond_dim=GLOBAL_COND_DIM,
        global_cond_type="adaLN",
        timestep_features_type="expo",
        norm_type="rms_norm",
        num_memory_tokens=8,
        zero_init_branch_outputs=False,
    )
    conditioner = MultiConditioner({"seconds_total": NumberConditioner(GLOBAL_COND_DIM, 0, 60)})
    model = ConditionedDiffusionModelWrapper(
        dit, conditioner,
        io_channels=C, sample_rate=1000, min_input_length=1,
        diffusion_objective="rectified_flow",
        global_cond_ids=["seconds_total"],
        mask_padding_attention=True,
        use_effective_length_for_schedule=dist_shift is not None,
        distribution_shift_options=dist_shift,
    )
    wrapper = DiffusionCondTrainingWrapper(
        model,
        optimizer_configs={"diffusion": {"optimizer": {"type": "AdamW", "config": {"lr": 1e-4}}}},
        pre_encoded=True,
        timestep_sampler="trunc_logit_normal",
        silence_extension_scale_seconds=0.0,
        inpainting_config=None,
        use_ema=False,
        ot_coupling=False,
        sample_rate=1000,
        df_training=df_training,
        df_p_global=df_p_global,
        **wrapper_kwargs,
    )
    # training_step reads the lr off the attached Trainer; stand one in.
    fake_trainer = types.SimpleNamespace(optimizers=[wrapper.configure_optimizers()[0]])
    wrapper._fake_trainer = fake_trainer          # keep it alive behind the weakref
    wrapper.trainer = fake_trainer
    return wrapper


def _batch(seed=1):
    torch.manual_seed(seed)
    reals = torch.randn(B, C, N)
    padding = torch.ones(B, N, dtype=torch.bool)
    padding[0, -6:] = False
    metadata = [
        {"seconds_total": float(12 + 3 * i), "padding_mask": [padding[i]]} for i in range(B)
    ]
    return reals, metadata


@pytest.mark.parametrize("df_training", [False, True], ids=["global_t", "per_frame_t"])
def test_training_step_runs_and_backprops(df_training):
    trainer = _tiny_trainer(df_training)
    loss = trainer.training_step(_batch(), 0)

    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grad = trainer.diffusion.model.model.to_timestep_embed[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0

    if df_training:
        assert trainer._last_t.shape == (B * N,)
        assert trainer._last_per_elem_loss.shape == (B * N,)
    else:
        assert trainer._last_t.shape == (B,)
        assert trainer._last_per_elem_loss.shape == (B,)


def test_training_step_with_per_item_schedule_shift():
    """use_effective_length_for_schedule=True hands the shift a (b,) seq_len; with (b, n) t
    that is the case that raised (or silently broadcast wrongly) before."""
    trainer = _tiny_trainer(True, dist_shift={"type": "full", "min_length": 8, "max_length": 256})
    loss = trainer.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    assert trainer._last_t.shape == (B * N,)


def test_df_p_global_one_gives_shared_t_per_item_end_to_end():
    trainer = _tiny_trainer(True, df_p_global=1.0)
    loss = trainer.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    t = trainer._last_t.reshape(B, N)
    torch.testing.assert_close(t, t[:, :1].expand(B, N))


def test_one_shot_is_per_item_not_per_frame():
    trainer = _tiny_trainer(True, p_one_shot=1.0)
    trainer.training_step(_batch(), 0)
    assert torch.equal(trainer._last_t, torch.ones(B * N))
    # and with a partial probability, an item is either all ones or none
    trainer = _tiny_trainer(True, p_one_shot=0.5, seed=3)
    trainer.training_step(_batch(), 0)
    t = trainer._last_t.reshape(B, N)
    is_one = t == 1.0
    assert (is_one.all(dim=1) | ~is_one.any(dim=1)).all()


def test_log_loss_info_bucketing_accepts_per_frame_t():
    trainer = _tiny_trainer(True, log_loss_info=True)
    logged = {}
    trainer.log_dict = lambda d, *a, **k: logged.update(d)
    trainer.all_gather = lambda x, *a, **k: x.unsqueeze(0)   # world_size 1 -> (1, b, ...)
    loss = trainer.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    assert logged and all(k.startswith("model/loss_all_") for k in logged)
