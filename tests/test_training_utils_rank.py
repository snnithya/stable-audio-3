"""get_rank: the process group wins; SLURM_PROCID is only a last-resort fallback.

Regression for the 4-GPU WJD finetune (job 2534818): a single-task sbatch gives every
Lightning DDP subprocess SLURM_PROCID=0, so every rank wrote and deleted the same demo wav.
"""

import pytest
import torch

from stable_audio_3.training import utils


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


def test_no_launcher_is_rank_zero(clean_env):
    assert utils.get_rank() == 0


def test_process_group_beats_slurm_procid(clean_env, monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    assert utils.get_rank() == 3


def test_lightning_local_rank_beats_slurm_procid(clean_env, monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert utils.get_rank() == 2


def test_torchrun_rank_beats_local_rank(clean_env, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "5")
    assert utils.get_rank() == 5


def test_slurm_procid_still_works_alone(clean_env, monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "1")
    assert utils.get_rank() == 1
