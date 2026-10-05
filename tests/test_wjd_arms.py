"""Checkpoint discovery for the WJD arms (stable_audio_3/inference/wjd_arms.py): a last.ckpt
that the training job is rewriting must not be picked up."""

import os

import torch

from stable_audio_3.inference.wjd_arms import WANDB_PROJECT, discover_checkpoint, is_complete_ckpt


def write_ckpt(path, step, age=0):
    """A tiny checkpoint whose mtime is ``age`` seconds in the past (the real files are written
    seconds apart; in a test they would otherwise tie)."""
    torch.save({"global_step": step, "state_dict": {}}, path)
    t = 1_700_000_000 - age
    os.utime(path, (t, t))


def test_is_complete_ckpt_detects_truncation(tmp_path):
    good = tmp_path / "good.ckpt"
    write_ckpt(good, 1)
    assert is_complete_ckpt(good)
    bad = tmp_path / "bad.ckpt"
    bad.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    assert not is_complete_ckpt(bad)
    assert not is_complete_ckpt(tmp_path / "missing.ckpt")


def test_discover_skips_half_written_last_ckpt(tmp_path):
    ck = tmp_path / "grp" / "rms" / WANDB_PROJECT / "wjd-rms-1" / "checkpoints"
    ck.mkdir(parents=True)
    write_ckpt(ck / "epoch=1-step=5000.ckpt", 5000, age=3600)
    write_ckpt(ck / "last-v1.ckpt", 5990, age=10)
    complete = ck / "last.ckpt"
    write_ckpt(complete, 6000, age=0)
    assert discover_checkpoint("rms", "grp", tmp_path) == str(complete)
    # Now the job is mid-write: last.ckpt is a truncated zip (and still the newest file).
    data = complete.read_bytes()
    complete.write_bytes(data[: len(data) // 2])
    assert discover_checkpoint("rms", "grp", tmp_path) == str(ck / "last-v1.ckpt")
    # And if that one were gone too, the newest step checkpoint.
    (ck / "last-v1.ckpt").unlink()
    assert discover_checkpoint("rms", "grp", tmp_path) == str(ck / "epoch=1-step=5000.ckpt")
    # Written back fully: the newest complete file wins again.
    complete.write_bytes(data)
    assert discover_checkpoint("rms", "grp", tmp_path) == str(complete)
