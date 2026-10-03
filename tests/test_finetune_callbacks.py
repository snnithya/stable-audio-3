"""The finetune's callback list must be identical on every DDP rank.

Each ModelCheckpoint broadcasts its resolved dirpath in setup(); a callback present on rank 0
only desynchronises the ranks' collectives and the job hangs before the first step (WJD jobs
2535097-99, 2026-10-03). checkpoint_dir is None on non-zero ranks because it is derived from
the wandb run id, which only rank 0 has, so the decision must not depend on it.
"""

import importlib.util
import os

import pytorch_lightning as pl

HERE = os.path.dirname(__file__)
SPEC = importlib.util.spec_from_file_location("train_finetune", os.path.join(HERE, "..", "scripts", "train_finetune.py"))
tf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tf)


def test_resume_callback_present_regardless_of_checkpoint_dir():
    rank0 = tf.build_resume_checkpoint_callback(1000, "/some/run/checkpoints")
    other = tf.build_resume_checkpoint_callback(1000, None)
    assert isinstance(rank0, pl.callbacks.ModelCheckpoint)
    assert isinstance(other, pl.callbacks.ModelCheckpoint)
    for cb in (rank0, other):
        assert cb._every_n_train_steps == 1000
        assert cb.save_top_k == 0
        assert cb.save_last is True


def test_resume_callback_disabled_the_same_way_everywhere():
    assert tf.build_resume_checkpoint_callback(0, "/x") is None
    assert tf.build_resume_checkpoint_callback(0, None) is None
    assert tf.build_resume_checkpoint_callback(None, None) is None
