# 3.2 — Training with independent per-frame `t`

**Status:** configured, not launched. `sbatch/df_test.sh` is this run.
**Parent:** [Experiment 03](README.md)
**Predecessor:** [3.1 — Local timestep conditioning](01-local-timestep-conditioning.md)
**Branch:** `nithya/df`
**Date:** 2026-09-21

---

## Question

3.1 established that per-frame `t` reaches adaLN correctly aligned and that the global path is
unchanged. That is a statement about shapes. This is the first statement about *training*:

**Does finetuning `small-music-base` with independent per-frame `t` converge, and does the model
keep the full-sequence generation quality the checkpoint already has?**

Two things have to be true, and they fail in different ways:

1. **It converges.** `train/mse_loss` is finite and trending down. The loss *level* is not
   comparable to a global-`t` run — per-frame `t` is a different task, and 3.1 step 5 says so
   explicitly — so the only criterion here is stability.
2. **Full-sequence quality survives.** This is the one that needs a held-out number rather than a
   demo clip, and it is why validation was wired into `train_finetune.py` for this run (below).

## Run configuration

`sbatch/df_test.sh`, 2× L40S, `train_finetune.py --model small-music-base`.

| | |
|---|---|
| model config | `small_music_base_df.json` — `df_training: true`, `df_p_global: 0.0` |
| train data | `slakh_streamgen_train_preencoded.json` — 1678 items, 144 latent frames (~13.3 s) each |
| val data | `slakh_streamgen_validation_preencoded.json` — 56 items, `random_crop: false` |
| batch | 8 per rank × 2 ranks = **16 effective**; `accum_batches` 1 |
| schedule | 10 000 steps ≈ **104 optimizer steps/epoch ≈ 96 epochs** over the train split |
| optimizer | AdamW, lr 1e-5, wd 0.01, betas (0.9, 0.95), no schedule — **not** what the config says; see *The optimizer block* |
| inpainting | `mask_type_probabilities [0, 0, 1]`, `future_visibility [-4, 0] s` |
| validation | every 1000 training batches |
| demos | every 2000 steps, 8 sampling steps, `num_causal: 4` |

`df_p_global: 0.0` means **every** item is per-frame; no shared-`t` batches are mixed in. That is
the harder setting of the two 3.1 step 5 describes, and it is deliberate — the mixed setting is
only worth measuring if the pure one is unstable.

## Validation

`train_finetune.py` gained `--val_dataset_config` / `--val_every` / `--val_batch_size` (2026-09-21).
The model side already existed and was simply never reached: `DiffusionCondTrainingWrapper.
validation_step` scores a fixed ladder of timesteps — `validation_timesteps`, default
`[0.1, 0.3, 0.5, 0.7, 0.9]` — and `on_validation_epoch_end` logs `val/loss_<t>` per rung plus
`val/avg_loss`. Neither finetune script passed a val dataloader, so none of it ran.

Wiring notes, in case the numbers look wrong:

- **`val_check_interval` counts training *batches*, not optimizer steps.** Identical here
  (`accum_batches: 1`); divide by it otherwise. `check_val_every_n_epoch=None` is what allows the
  interval (1000) to exceed the epoch (104) at all.
- **The split must be read deterministically.** `shuffle=False` in the dataloader, `random_crop:
  false` in the config, and every item is exactly 144 stored frames so nothing crops or
  silence-pads. Any of those three changing makes consecutive validations incomparable for reasons
  that have nothing to do with the model.
- **56 items / 2 ranks / batch 8 → 4 batches per rank.** No rank gets zero batches, which matters
  because `on_validation_epoch_end` divides by the number of collected losses.
- **`num_sanity_val_steps=0`**, inherited from the existing Trainer call. A broken val config
  therefore surfaces at step 1000, not at step 0. Set it to 1 on the first run of a new split.

### The caveat that matters: this is *not* a DF validation loss

`validation_step` noises with a **global** timestep — `torch.full((B,), t)` — at every rung. It is
the full-sequence task, measured on held-out data, on a model being trained on the per-frame task.

That makes it the right instrument for claim (2) and the wrong one for claim (1): it answers "is
full-sequence quality surviving DF training?" and says nothing about whether the per-frame regime
itself is being learned. Read `val/avg_loss` as a **regression guard**, not as a training signal.
Rising `val/avg_loss` against falling `train/mse_loss` is the expected shape of the failure this
experiment is looking for; it does not distinguish DF-induced degradation from ordinary
overfitting, and at ~96 epochs over 1678 clips the second explanation is not a remote one.

A per-frame validation loss — the same ladder, but with `t` drawn the way training draws it — does
not exist yet. It is the obvious addition if `val/avg_loss` turns out to be uninformative.

## The optimizer block

**Open issue, deliberately not fixed (2026-09-21). Nothing here blocks the run.**

`train_finetune.py` ignores `training.optimizer_configs` and always builds plain AdamW from
`--lr` (default 1e-5, **no scheduler**). The training wrapper supports the block — 
`configure_optimizers` reads the optimizer type, does the MuonAdamW parameter grouping, and
attaches a per-step scheduler — but the script never passes it through.

Underneath that sits a second problem: **`MuonAdamW` cannot be constructed in this repo at all.**
`create_optimizer_from_config` (`training/utils.py:86`) imports it from `stable_audio_tools`,
which is not a declared dependency and is not installed. All four model configs ask for it.

So every finetune run here — experiment 01's included — has trained on **AdamW @ 1e-5 with no
schedule**, not the `muon_lr: 1e-3` / `adam_lr: 5e-5` / `InverseLR` its config records. The
optimizer block in `small_music_base_df.json` is decorative, and this run will be no different.

That is not fatal for 3.2: AdamW at a low lr is a reasonable setting for a first DF run, and it
matches what every existing curve in this repo was actually trained with, so comparisons stay
valid. The cost is only that the setting is accidental rather than chosen. **Do not cite the
config's optimizer block when writing up any run, here or in experiment 01.**

Fixing it means deciding between two different experiments — AdamW @ 1e-5 + the existing
`InverseLR` (comparable to what exists), or installing `stable-audio-tools` and running Muon at
100× the effective learning rate (not comparable to anything here). Deferred until 3.2 has a
curve; doing it now would confound DF with an optimizer change.

### Also hardcoded

`train_finetune.py` reads only `df_training`, `df_p_global`, `inpainting` and `demo` from the
`training` block. `timestep_sampler`, `mask_loss_weight`, `pre_encoded`, `log_loss_info`,
`silence_extension_scale_seconds` and `ot_coupling` are hardcoded in the script and happen to
agree with this config's values, so nothing else is being misreported today.

`use_ema` is a different case — it is inert in the **wrapper**, not the script.
`DiffusionCondTrainingWrapper.__init__` sets `self.diffusion_ema = None` unconditionally
(`training/diffusion.py:139`), so neither `--use_ema` nor `training.use_ema: true` builds an EMA.
EMA is not implemented in this fork.

## Known confounds

Carried forward, each already documented elsewhere:

- **`seconds_total` is hard-coded to 12 s** (README note, 2026-09-19). The window is 13.3 s, and
  with `use_effective_length_for_schedule: true` the schedule shift sees 130 frames against 144
  real ones. Applies to train and val alike, so it does not bias the comparison between them.
- **The registers sit at `t̄ = 0.537`**, not at the item's `t` (3.1, step 1a). Against the
  pretrained checkpoint this is a systematic perturbation of 64 of 208 tokens; 3.1 step 5 expects
  the first-step loss slightly above a global-`t` baseline, closing as the registers adapt.
- **Demo conditioning asks for durations the model never sees** — 119 / 100 / 9 / 35 s against a
  13.3 s training window. Demo clips are not evidence about quality until that is fixed.

## Results

Not launched.

