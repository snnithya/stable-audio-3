"""LR schedules the WJD finetune can select with --lr_schedule (stepped per optimizer step)."""

import math

import pytest
import torch

from stable_audio_3.training.utils import InverseLR, WarmupCosineLR, create_scheduler_from_config


def _opt(lr=1e-5):
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def _run(sched, opt, n):
    lrs = []
    for _ in range(n):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs


def test_warmup_cosine_shape():
    opt = _opt(1e-5)
    sched = WarmupCosineLR(opt, total_steps=1000, warmup_steps=100, final_frac=0.1)
    lrs = _run(sched, opt, 1001)
    assert lrs[0] == pytest.approx(1e-5 / 100)           # first step of the linear warmup
    assert lrs[99] == pytest.approx(1e-5)                 # warmup ends at the base lr
    assert lrs[550] == pytest.approx(1e-5 * (0.1 + 0.9 * 0.5), rel=1e-2)  # midway: half the drop
    assert lrs[1000] == pytest.approx(1e-6)               # floor at total_steps
    assert all(a >= b for a, b in zip(lrs[99:], lrs[100:]))  # monotone after warmup


def test_warmup_cosine_holds_floor_past_total_steps():
    opt = _opt(1e-5)
    sched = WarmupCosineLR(opt, total_steps=10, warmup_steps=0, final_frac=0.2)
    lrs = _run(sched, opt, 20)
    assert lrs[-1] == pytest.approx(2e-6)


def test_inverse_lr_matches_closed_form():
    opt = _opt(1e-5)
    sched = InverseLR(opt, inv_gamma=1e6, power=0.5, warmup=0.995)
    lrs = _run(sched, opt, 1001)
    t = 1000
    expected = 1e-5 * (1 - 0.995 ** (t + 1)) * (1 + t / 1e6) ** -0.5
    assert lrs[t] == pytest.approx(expected)


def test_factory_builds_both_custom_types():
    for cfg in (
        {"type": "WarmupCosineLR", "config": {"total_steps": 10, "warmup_steps": 2, "final_frac": 0.1}},
        {"type": "InverseLR", "config": {"inv_gamma": 10, "power": 0.5, "warmup": 0.9}},
        {"type": "ExponentialLR", "config": {"gamma": 0.9}},
    ):
        sched = create_scheduler_from_config(cfg, _opt())
        assert isinstance(sched, torch.optim.lr_scheduler.LRScheduler)
