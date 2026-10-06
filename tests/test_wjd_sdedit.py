"""SDEdit in the WJD Gradio interface (stable_audio_3/interface/wjd_control.py): with a noise
level set, sampling starts from the drum or stem latent noised to that level; without it, from noise."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_audio_3.interface import wjd_control
from stable_audio_3.interface.wjd_control import LoadedArm, generate_continuation

DS = 4  # toy downsampling ratio
CH = 3  # toy latent channels


def toy_encode(model, audio, device):
    """A deterministic toy latent: the first sample of each hop (channel mean), repeated over CH channels."""
    x = audio.mean(0).reshape(-1, DS)[:, 0]
    return x[None, None].repeat(1, CH, 1).to(device)


def make_arm(controls=()):
    model = SimpleNamespace(
        model=nn.Linear(1, 1),
        pretransform=SimpleNamespace(downsampling_ratio=DS),
        io_channels=CH,
        conditioner=lambda meta, device: {},
        modular_local_cond_ids=[],
        diffusion_objective="rectified_flow",
        mask_padding_attention=False,
        use_effective_length_for_schedule=False,
        sampling_dist_shift=None,
    )
    return LoadedArm(name="toy", model_config_path="", ckpt_path="", ckpt_mtime=0.0, step=0, weights="raw",
                     model=model, model_config={"sample_rate": 40}, controls=list(controls), loaded_at=0.0)


@pytest.fixture
def calls(monkeypatch):
    seen = {}

    def fake_sample(**kw):
        seen.update(kw)
        return kw["noise"].float()

    monkeypatch.setattr(wjd_control, "encode", toy_encode)
    monkeypatch.setattr(wjd_control, "decode", lambda model, z: z.repeat_interleave(DS, -1)[:, :2])
    monkeypatch.setattr(wjd_control, "make_multi_cfg_denoiser", lambda model, cond, scales, order: (None, [((), 1.0)]))
    monkeypatch.setattr(wjd_control, "sample_diffusion", fake_sample)
    return seen


def run(arm, drums, level, stem=None, source="drums"):
    return generate_continuation(arm, "piano", drums, stem, {}, 0.0, 0.0,
                                 {"prompt": 1.0, "context": 1.0, "control": 1.0},
                                 ("prompt", "context", "control"), 4, "euler", 0,
                                 sdedit_noise_level=level, sdedit_source=source)


def test_sdedit_passes_drum_latent_and_level(calls):
    drums = torch.linspace(-1, 1, 2 * 8 * DS).reshape(2, -1)
    _, info = run(make_arm(), drums, 0.6)
    assert calls["init_noise_level"] == 0.6
    torch.testing.assert_close(calls["init_data"].float(), toy_encode(None, drums, "cpu").to(torch.bfloat16).float())
    assert calls["init_data"].dtype == calls["noise"].dtype
    assert info["sdedit_noise_level"] == 0.6


def test_no_sdedit_starts_from_noise(calls):
    drums = torch.randn(2, 8 * DS)
    _, info = run(make_arm(), drums, None)
    assert "init_data" not in calls and "init_noise_level" not in calls
    assert info["sdedit_noise_level"] is None


def test_sdedit_needs_drums_and_a_valid_level(calls):
    stem = torch.randn(2, 8 * DS)
    with pytest.raises(ValueError, match="drum audio"):
        run(make_arm(), None, 0.5, stem=stem)
    with pytest.raises(ValueError, match="stem to continue"):
        run(make_arm(), torch.randn(2, 8 * DS), 0.5, source="stem")
    with pytest.raises(ValueError, match="source"):
        run(make_arm(), torch.randn(2, 8 * DS), 0.5, stem=stem, source="mix")
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="noise level"):
            run(make_arm(), torch.randn(2, 8 * DS), bad)


def test_sdedit_from_stem_passes_stem_latent(calls):
    drums = torch.randn(2, 8 * DS)
    stem = torch.linspace(-1, 1, 2 * 8 * DS).reshape(2, -1)
    _, info = run(make_arm(), drums, 0.4, stem=stem, source="stem")
    assert calls["init_noise_level"] == 0.4
    torch.testing.assert_close(calls["init_data"].float(), toy_encode(None, stem, "cpu").to(torch.bfloat16).float())
    assert info["sdedit_source"] == "stem"
