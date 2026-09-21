# Experiment 03 — Diffusion forcing

**Status:** 3.1 landed (model + trainer, 56 unit tests passing); 3.2 configured, not launched
**Started:** 2026-09-09 · **Last updated:** 2026-09-21
**Branch:** `nithya/df`
**Paper:** Chen et al., *Diffusion Forcing: Next-Token Prediction Meets Full-Sequence Diffusion*
([arXiv:2407.01392](https://arxiv.org/pdf/2407.01392))

---

## Question

The DiT denoises a whole latent window at **one** noise level: `t` is a `(B,)` tensor, embedded once
and broadcast to every frame through adaLN. Diffusion forcing gives each frame its **own** noise
level — `t` becomes `(B, T)` — so a single model can hold parts of the window nearly clean while the
rest is still noise.

**Can the streamgen DiT be trained with per-frame noise levels, and does that buy anything the
existing inpainting mechanism does not already give?**

Three sub-claims, in dependency order:

1. **The plumbing is sound.** Per-frame `t` reaches adaLN correctly aligned, and with all frames set
   to the same value the model is bit-equivalent to today's global-`t` path. This is pure
   engineering and is the precondition for everything else.
2. **It trains.** Finetuning from the existing checkpoint with independent per-frame `t` converges,
   and the model does not lose the full-sequence generation quality it already has.
3. **It buys horizon control.** Per-frame noise schedules — pyramid / rolling — give a usable
   latency/quality dial for interactive generation, and stable rollout past the training window
   length.

Only (1) is in scope for the first pass.

## Background

### Why this fits here

`AGENTS.md` lists diffusion forcing as the planned mechanism for "planning and interactive edits".
It overlaps with, and arguably subsumes, the inpainting machinery experiment 01 relies on: a
per-frame `t` of 0 on context frames **is** inpainting with clean context. The two stay independent
until (1) is validated — the loss-masking interaction is subtle and not worth entangling with
bring-up.

### What the paper does vs. what this model is

Chen et al. use a **causal** architecture, so frame *i*'s denoising depends only on frames ≤ *i*.
That is what enables the rolling autoregressive sampling that makes DF interesting for streaming.
This DiT is **bidirectional** (`ContinuousTransformer`, full self-attention, optional symmetric
sliding window).

Per-frame timestep conditioning in a bidirectional model is still a coherent object — it is what
Diffusion Forcing Transformer and the rolling-diffusion variants do — but it is a *different* object
from the paper's: you get flexible per-frame noise levels without the causal-rollout guarantee. This
changes nothing in the plumbing of sub-experiment 3.1, but it is the decision that determines
whether the sliding-window attention already in `transformer.py` should be made causal. **Deferred,
not resolved.**

### Why it is cheap to try

No new parameters. `to_timestep_embed` (`models/dit.py:58`), `global_cond_embedder`
(`models/transformer.py:1132`) and `to_scale_shift_gate` (`models/transformer.py:942`) are all
last-dim operations, so they already accept `(B, T, D)` where they currently see `(B, D)`. Existing
checkpoints load and finetune into DF directly — there is no cold-start projection to warm up, and
no equivalent of the zero-init trick experiment 01 needed.

The cost is entirely in the ~8 places that assume `t` has a batch axis and nothing else.

## Sub-experiments

| # | Name | Question | Status |
|---|------|----------|--------|
| 3.1 | [Local timestep conditioning](01-local-timestep-conditioning.md) | Can per-frame `t` reach adaLN, correctly aligned, without disturbing the global path? | **done at toy scale** — model, CFG path and trainer all landed; 56 tests pass. Production run not done |
| 3.2 | [Training with independent per-frame `t`](02-training-with-per-frame-t.md) | Does it converge from the existing checkpoint, and does full-sequence quality survive? | **configured, not launched** (`sbatch/df_test.sh`) |
| 3.3 | Per-frame inference schedules | Do pyramid / rolling schedules work, and do they roll out past the training window? | not started |
| 3.4 | DF vs. inpainting | Does per-frame `t` subsume `tf_inpaint_mask`, or are they complementary? | not started |
| 3.5 | Causality | Does the rolling regime need causal attention, or is bidirectional-with-per-frame-`t` enough? | not started |

## Results

Nothing trained yet. 3.1 is implemented and unit-tested at toy scale; 3.2 is configured and
specified; 3.3–3.5 are placeholders whose shape will change once 3.2 produces a curve.

## Notes

- **Register conditioning is decided, not measured.** The 64 memory tokens are noise-conditioned
  in the pretrained model, and under DF there is no single noise level to condition them on. 3.1
  fixes them at `t̄ = 0.537` (decision 2026-09-15; reasoning in its design log). The cheap
  measurement that would say how much this matters — conditioning magnitude vs.
  `to_scale_shift_gate` in the pretrained checkpoint — has not been run.
- **Held-out validation exists now (2026-09-21).** `train_finetune.py` takes
  `--val_dataset_config` / `--val_every` / `--val_batch_size`, and logs `val/loss_<t>` for each
  rung of `validation_timesteps` (default `[0.1, 0.3, 0.5, 0.7, 0.9]`) plus `val/avg_loss`. The
  wrapper already had `validation_step` and `on_validation_epoch_end`; neither finetune script ever
  passed a val dataloader, so none of it had run. **It is a global-`t` loss**, so under DF it
  measures whether full-sequence quality is surviving, not whether the per-frame task is being
  learned — see [3.2](02-training-with-per-frame-t.md#the-caveat-that-matters-this-is-not-a-df-validation-loss).
- **Every finetune in this repo so far trained on AdamW @ 1e-5, not the optimizer its config
  records (found 2026-09-21; open, not fixed).** `train_finetune.py` ignores
  `training.optimizer_configs` and builds its own AdamW with no scheduler. Separately, the
  `MuonAdamW` all four configs ask for cannot be constructed here at all —
  `create_optimizer_from_config` imports it from `stable_audio_tools`, which is not a dependency.
  Experiment 01's recorded optimizer settings are therefore not what experiment 01 ran. Harmless
  for comparisons (every run shares the same real optimizer), fatal for write-ups that quote the
  config. Details in [3.2](02-training-with-per-frame-t.md#the-optimizer-block).
- **EMA is not implemented in this fork.** `DiffusionCondTrainingWrapper` sets
  `diffusion_ema = None` unconditionally, so neither `--use_ema` nor `training.use_ema` does
  anything.
- **Scope discipline.** 3.1 is plumbing only. It should land in two commits — shape-polymorphism
  behind `df_training=False` (no behavior change), then the trainer changes plus the equivalence
  test — so that a bad DF training curve later can never be confused with a plumbing bug.
- A work-in-progress diff on this branch already touches `models/dit.py` and
  `training/diffusion.py`. It has three defects, documented in
  [3.1](01-local-timestep-conditioning.md#state-of-the-wip-diff); treat that section as the diff's
  review notes rather than starting over.
- `global_cond_type` must be `adaLN` for any of this to work. `small_music_baseline.json` already
  sets it; the `prepend` path cannot represent a per-frame signal and should assert.
- **`seconds_total` is hard-coded to 12 s for training (hack, 2026-09-19).**
  `custom_md_slakh.py` — the `custom_metadata_module` every `preencoded/*.json` config loads at
  train time — now returns `"seconds_total": 12` for every item. Reason: the pre-encoded sidecars
  carry the value `pad_crop` computed from the *whole* source file (`data/utils.py:56`), so a
  multi-minute Slakh track is tagged with hundreds of seconds although only its first 13.3 s
  window (587853 samples, 144 frames) was encoded. That value reaches the DiT as a global and
  cross-attention condition and, with `use_effective_length_for_schedule: true` in
  `small_music_base_df.json`, sets the effective length for the timestep-distribution shift — so
  training would see lengths inference never asks for. Caveats, in case a DF training curve looks
  off: (a) 12 is not the window length; the window is 13.3 s, i.e. 14 under the `ceil`
  convention used everywhere else, so the schedule shift sees 130 frames against 144 real ones.
  (b) Tracks shorter than the window are padded but still claim 12 s, contradicting their padding
  mask. (c) The demo conditioning in `small_music_base_df.json` still asks for 119 / 100 / 9 / 35 s,
  none of which the model sees in training. (d) It applies at train time only, so the cached JSON
  is untouched and the pre-encode script is unaffected. The first attempt put the override in
  `custom_md_slakh_streamgen.py`, which is the *pre-encode* module and never runs on the
  pre-encoded path; that edit was reverted. The proper fix is to derive the value from the
  padding mask — on the pre-encoded path the metadata fn receives latents and a latent-frame mask,
  so it needs the pretransform downsampling ratio (4096) to get back to seconds — and to make the
  demo durations match the window.
