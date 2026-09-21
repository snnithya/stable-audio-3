# 3.1 — Local timestep conditioning

**Status:** steps 1–4 implemented and unit-tested at toy scale (2026-09-15); 56 tests pass. Step 5 production run **not done** — `sbatch/df_test.sh` is the [3.2](02-training-with-per-frame-t.md) run, not the acceptance run below (2026-09-21)
**Parent:** [Experiment 03](README.md)
**Branch:** `nithya/df`

---

## Question

Timestep conditioning is currently a **global** signal: `t` is `(B,)`, embedded once, added to the
global conditioning, and broadcast across every frame by adaLN. Diffusion forcing needs it to be
**local**: `t` of shape `(B, T_latent)`, one noise level per latent frame, still routed through the
same adaLN modulation.

**Can `t` be made shape-polymorphic — `(B,)` = today, `(B, T)` = diffusion forcing — without
changing behavior in the global case?**

The acceptance criterion is deliberately narrow: with per-frame `t` set to the *same* value on every
frame, training loss must match the global-`t` baseline to numerical noise. If that holds, all the
plumbing is right and 3.2 can be about diffusion forcing rather than about shapes.

## Why shape-polymorphic `t` rather than a new argument

Two options were considered:

- **(A) Overload `t`.** `t.ndim` selects the regime; every consumer branches on it.
- **(B) A separate `t_local` argument** threaded alongside the existing `t`.

**(A) is chosen.** The noising math in the trainer and the ODE steps in the samplers already index
and broadcast `t`; they need the *broadcast shape* changed, not a second tensor to reconcile with
the first. And `to_timestep_embed` is a `nn.Linear` stack over the last dim, so it is already
rank-agnostic — (B) would mean either duplicating it or explaining why two paths produce the same
embedding. The cost of (A) is that ~8 call sites must handle both ranks; they are enumerated below.

Both Fourier feature modules are pure last-dim ops and need no change — verified:

```
FourierFeatures      (4,8,1) -> (4,8,256)      # models/blocks.py:46
ExpoFourierFeatures  (4,8,1) -> (4,8,256)      # models/blocks.py:58
```

## State of the WIP diff

The uncommitted changes on this branch sketch the right idea but have three defects. Review notes,
not a reason to start over:

1. **`models/dit.py:240` — `t_cond.flatten()` is the wrong reshape.** It produces `(B*T,)`, so
   `t_cond[:, None]` gives `(B*T, 1)` and `timestep_embed` comes out `(B*T, embed_dim)`. That then
   breaks at `global_embed + timestep_embed` (`dit.py:247`). The fix is not to flatten at all:
   change `t_cond[:, None]` to `t_cond[..., None]`, which is correct for both ranks.
2. **`models/dit.py:239` — `self.df_training` is never assigned** in
   `DiffusionTransformer.__init__`. First forward raises `AttributeError`.
3. **`training/diffusion.py:341` — `reals.shape[2]` is the wrong length.** `reals` is
   pre-pretransform. It is only right because the configs run `pre_encoded: true`; under
   `pre_encoded: false` it is the audio sample count, not the latent frame count. `t` is sampled
   after the pretransform block, so `diffusion_input.shape[2]` is in scope and is the correct value.

## Plan

### Step 1 — adaLN accepts per-token conditioning

**Done**, `tests/test_diffusion_forcing.py`. Verified at toy scale on CPU/fp32/SDPA only — see step 5.

Two edits, mostly in `models/transformer.py`; 1a also touches `models/dit.py`, since the summary row
has to be built where `t` is still visible.

**1a. Give the prepended tokens their own conditioning row.** `global_cond` for the latents is
`(B, T, 6D)`, but `x` in the transformer is `(B, 64+T, D)`. Something has to occupy those leading
rows.

**What is actually prepended.** The **64 memory tokens** — learned register tokens of width `dim`
(`transformer.py:1120`), not 64 dims and not conditioning. They are input-independent scratch space
that every real token attends to and from, frozen at inference, sliced off before `project_out`.
All three model configs set `num_memory_tokens: 64`. Real prepend conditioning (`prepend_embeds`)
would come from `prepend_cond_ids` or from `global_cond_type == "prepend"`, and **neither fires
here**: no model config sets `prepend_cond_dim`, `prepend_cond_ids` is empty, and the configs use
`adaLN`. So `prepend_length = 0` and the sequence is `[64 registers][T latents]` — but the registers
alone are enough to force the issue, and at `latent_crop_length: 144` they are **64/208 ≈ 31% of the
sequence**, not a rounding error.

**The registers are noise-conditioned today.** This is the fact that constrains everything below.
Their *content* is a constant, but their *modulation* is not: `global_cond` is `(B, 6D)`,
`unsqueeze(1)`-ed, and broadcast over all `64+T` tokens, so the registers receive exactly the same
`f(t)` as every latent frame and have for all of pretraining. Constant token, noise-dependent affine
transform applied to it.

That rules out zero. adaLN is affine (`x * (1 + scale) + shift`, gated), not additive, so a zero row
gives the registers `to_scale_shift_gate` alone — a modulation no pretraining step ever produced.
This is the opposite of the `local_add_cond` case, a plain additive residual where zero genuinely
means "add nothing", which is why `_left_pad_to_match` (`transformer.py:77`) zero-pads and why it
**cannot be reused here**.

**Why not the mean over frames (rejected).** The obvious generalization is to condition the
registers at the window's mean noise level, `timestep_embed(t.mean(dim=1))`. It is exact when all
frames are equal, so it passes the acceptance test — but it does not survive contact with DF, for
two reasons.

First, `mean(t)` **concentrates to a constant** under iid per-frame sampling. Over `T=144` frames:

| sampler | mean | std | range |
|---|---|---|---|
| uniform iid | 0.500 | 0.0240 | [0.406, 0.598] |
| `trunc_logit_normal` iid (the configured sampler) | 0.537 | 0.0186 | [0.466, 0.614] |
| global `t` (today) | 0.534 | 0.2223 | [0.024, 1.000] |

So during DF training the registers would see `t ≈ 0.537 ± 0.019` on essentially every step and
never a noise level near 0 or 1 — a constant in all but name.

Second, and worse, at inference a rolling or pyramid schedule sweeps `mean(t)` across the full
range. That pushes the register modulation into a region seen only in *global-t* batches, where the
frames were all at that same `t`; the combination "mean-t ≈ 0.15, frames spread across [0,1]" occurs
nowhere in training. The register conditioning would become **schedule-dependent**, which defeats
the property iid training exists to buy — that any inference schedule is in-distribution.

**Use a frozen constant, `timestep_embed(t̄)` — and only that.** Take `t̄` as the training
sampler's mean (0.537 for `trunc_logit_normal`; `df_register_cond_t` in the model config). Three
properties at once:

- **In-distribution**, unlike zero — it is a real timestep embedding, and specifically the one DF
  training concentrates on anyway.
- **Schedule-independent** at inference: registers see the same modulation under pyramid, rolling
  or global schedules.
- **No new parameters** — a plain float, so checkpoints load unchanged. A *learned* constant would
  cost `6D` per block and break that.

An earlier draft kept `"mean"` alongside it as a knob, so that an all-frames-equal comparison
against the global path would stay exact. **Dropped, by decision (2026-09-15):** the registers'
conditioning should be a constant, full stop, and a second code path whose only purpose is to make
a test pass is not worth carrying. The cost is stated in step 5: the per-frame path is now
bit-equivalent to the global path only at `t = t̄`, and alignment has to be tested directly.

**Where this lives.** `t` is only visible in `dit._forward`; `ContinuousTransformer` sees the
already-embedded `(B, T, 6D)`. So the register row cannot be computed inside the transformer.
`dit.py` builds two things and passes both:

- the per-frame conditioning `(B, T, D)` for the latents, and
- a summary row `(B, D)` for the prepended positions, from whichever rule is selected.

`ContinuousTransformer` expands the summary to `prepend_length + num_memory_tokens` rows and
concatenates. Slightly more interface than a pad helper, but explicit, and it removes the
`_left_pad_to_match` question entirely.

One simplification: `global_cond_ids: ['seconds_total']`, which is already per-sample. Only the
timestep term differs between the summary row and the per-frame rows.

**Still worth measuring.** Compare `‖global_cond_embedder output‖` against `‖to_scale_shift_gate‖`
per block in the pretrained checkpoint. If the conditioning term dominates the register modulation,
this choice matters and `t̄` is load-bearing; if it is a small correction to the bias, the whole
question is a footnote and any of the three rules would do. Cheap to run, and it would settle 1a
with a number instead of an argument.

**1b. Stop unconditionally adding a broadcast axis.** `TransformerBlock.forward`
(`transformer.py:1020`) currently does:

```python
scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (self.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)
```

Becomes:

```python
gc = global_cond if global_cond.ndim == 3 else global_cond.unsqueeze(1)
scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (self.to_scale_shift_gate + gc).chunk(6, dim=-1)
```

Everything downstream already broadcasts: `x * (1 + scale_self) + shift_self` is correct whether the
modulation is `(B,1,D)` or `(B,T,D)`. Note `forward` carries `@compile` — expect one additional
graph for the 3-D regime.

### Step 2 — `models/dit.py`: build the local embedding

**Done** — landed together with 1a, since it is the same lines.

- `dit.py:241`: drop the flatten, use `t_cond[..., None]`.
- `dit.py:247`: `global_embed.unsqueeze(1) + timestep_embed` when `timestep_embed.ndim == 3`.
- `dit.py:251` (`timestep_cond_type == "input_concat"`): `unsqueeze(2).expand(...)` is wrong for
  3-D; it is `rearrange(t_embed, "b t c -> b c t")`. Unused by current configs, but it should not
  break silently.
- `dit.py:257,261` (`global_cond_type == "prepend"`): **fundamentally incompatible** — a per-frame
  signal cannot be collapsed into one prepended token. Assert that per-frame `t` requires `adaLN`.
- `dit.py:275` (`patch_size > 1`): `x` is rearranged `b (t p) c -> b t (c p)`, so the token count is
  `T/p` and the timestep vector must be pooled to match. Configs default to `patch_size=1`; assert,
  or pool.

### Step 3 — `models/dit.py` `forward()`: the CFG / inference path

**Done** (2026-09-15), tests in the `forward()` section of `tests/test_diffusion_forcing.py`.
Implemented as written below, with two small differences: the compute skip is kept — the batched
cond/uncond pass still does not run when *no* element is in the window (`in_window.any()`), so the
all-out-of-window result is bit-identical to `cfg_scale=1`; and the per-element gate applies to
1-D `t` as well, which is the "pre-existing approximation" fix. When every item shares one `t`, the
1-D behavior is unchanged. `_broadcast_t` is a module-level helper in `dit.py`.

Easy to miss, because training never enters it.

- **`dit.py:518,525,531`** — `sigma[0]` gates the LoRA and CFG intervals. With 2-D `t` it is one
  item's `(T,)` vector, and the chained comparison raises `Boolean value of Tensor with more than
  one value is ambiguous` (verified). An earlier draft said `sigma.flatten()[0]`; that is wrong — it
  silences the error by gating the whole forward on frame 0 of item 0, an arbitrary semantic choice
  dressed as a shape fix. Under DF "is sigma in the window" has no single answer, and the two
  intervals need different treatment:
  - **CFG interval → per-frame.** Run the batched cond/uncond pass as usual, then
    `in_window = (sigma >= lo) & (sigma <= hi)` → `(B, T)`, and
    `cfg_denoised = where(in_window[:, None, :], cfg_denoised, cond_denoised)`. Correct semantics,
    small change; loses only the compute skip when no frame is in-window. This also fixes a
    pre-existing approximation: the per-element-schedule branch of `sample_discrete_euler` already
    hands items different noise levels, and `sigma[0]` gates them all on item 0.
  - **LoRA interval → warn, not implement.** Enabling an adapter is module-wide, so there is no
    per-frame version without splitting the forward. Not going to be used with DF (decision,
    2026-09-15): `warnings.warn` when a non-default interval meets 2-D `t`, gate on
    `sigma.flatten()[0]`, move on.
- **`dit.py:578-583, 613-615`** — `sigma[:, None, None]` yields `(B,T,1,1)` against `x` of
  `(B,C,T)`. Add one broadcast helper (`t.view(-1,1,1)` if 1-D else `t.unsqueeze(1)` → `(B,1,T)`)
  and use it at all six sites, plus the matching `alpha` sites.
- `batch_timestep = torch.cat([t, t], dim=0)` (`dit.py:487`) is already rank-agnostic.

### Step 4 — `training/diffusion.py`: sample and apply per-frame `t`

**Done** (2026-09-15), tests in `tests/test_diffusion_forcing_training.py`. Timestep sampling
moved out of `training_step` into `_draw_timesteps` (the five samplers, shape-agnostic) and
`_sample_timesteps` (the DF / `df_p_global` mixing), so both are testable without a full trainer.
The `dist_shift` fix is a small `_align_per_item` helper applied in all three `shift` methods.
`_broadcast_t` is imported from `models/dit.py` rather than duplicated. Everything below landed
as written.

- **`diffusion.py:341`** — `diffusion_input.shape[2]`, per defect 3 above.
- **`diffusion.py:346` — `self.rng.draw(shape_of_t)` raises.** `SobolEngine.draw` takes an int;
  passing a tuple gives `TypeError: unsupported operand type(s) for -: 'tuple' and 'int'` (verified).
  Needs `self.rng.draw(B*T)[:, 0].reshape(B, T)`. Be aware this changes what the quasirandom
  stratification is *over* — it now stratifies across frames rather than across the batch. The other
  three samplers (`torch.randn(shape)`, `truncated_logistic_normal_rescaled(shape)`,
  `sample_timesteps_logsnr(shape)`) accept tuples unchanged.
- **`diffusion.py:382` — `dist_shift.shift` broadcasts wrong.** With
  `use_effective_length_for_schedule=True`, `effective_seq_len` is `(B,)` so `alpha` is `(B,)`, and
  `alpha * t` against `(B,T)` fails — `RuntimeError: The size of tensor a (4) must match the size of
  tensor b (8)` (verified). Worse, when `B == T` it broadcasts *silently and wrongly*. Fix by
  unsqueezing `alpha` in the 2-D case, in `distribution_shift.py:82` and the sibling `shift`
  methods.
- **`diffusion.py:386` — `p_one_shot`** uses `torch.rand_like(t)`, which with 2-D `t` becomes
  *per-frame* one-shot rather than per-sample. Almost certainly not intended; draw the mask at
  `(B,1)`.
- **`diffusion.py:394-395`** — `alphas[:, None, None]` → `(B,1,T)` via the same broadcast helper.
  Identical fix in `validation_step` at `diffusion.py:629-630`.
- **`diffusion.py:486`** — the `log_loss_info` bucketing `.squeeze()`s gathered sigmas and assumes
  one scalar per sample. Off in current configs; guard it or flatten the `(sigma, loss)` pairs so it
  does not trip whoever turns it on.
- **`diffusion.py:558-559`** — `_last_t` / `_last_per_elem_loss` are stashed for callbacks. Flatten
  both to frame granularity so loss-by-timestep keeps meaning something.
- **Wire the flags.** `df_training` exists as a kwarg but nothing sets it. Pass it and
  `df_p_global` from `model_config["training"]` in `scripts/train_finetune.py:205` and
  `scripts/train_lora.py:171`. `df_register_cond_t` is a *model* setting — it changes
  `dit._forward` — so it belongs in the diffusion `config` block alongside `global_cond_type`, not
  in the training wrapper. Its default (0.537) is tied to `trunc_logit_normal`; recompute if the
  sampler changes.

**Add a `df_p_global` knob** — probability of collapsing `t` to a single shared value across frames.
Mixing in some fraction of shared-`t` batches preserves the ordinary full-sequence generation the
checkpoint is already good at, and gives a continuous path off the existing weights instead of a
distribution shift on step 1. It also makes the acceptance test below a config setting
(`df_p_global=1.0`) rather than a scratch patch.

### Step 5 — Validation

**What const-only registers make checkable, and what they do not.** With the `"mean"` knob gone
(1a, decision 2026-09-15), the per-frame path reproduces the global path **only at `t = t̄`**: at
any other uniform `t` the latents sit at `t` while the registers sit at `t̄`, and the outputs differ
by design. Three consequences for this section:

1. **Module-level equivalence is pinned at `t̄`.** Every "matches global" test uses
   `_uniform(REGISTER_T)`. That is a sharp test — shapes, broadcasts, register-row plumbing, CFG
   reconstruction all have to be right for it to hold to `1e-5` — but it is one point, and
   `test_differs_from_global_t_away_from_the_register_t` documents that it is *only* one point.
2. **Alignment has to be tested directly.** A uniform-`t` comparison cannot see it: when every
   frame carries the same value, every row of the `(B, 64+T, 6D)` modulation tensor is identical
   and a wrong `[registers][latents]` order produces the same tensor. Hence
   `test_frame_i_modulates_token_i`, which gives frames *different* values and reads off where
   they land.
3. **There is no training-level baseline to match.** `df_training=True, df_p_global=1.0` is the
   global-`t` sampler expressed in `(B, T)` form, but it is *not* the global-`t` model: on every
   step the registers are modulated at `t̄` instead of at the item's `t`. For the pretrained
   checkpoint that is a small, systematic perturbation of 64/208 tokens, so the first-step loss is
   expected to sit slightly above the baseline and converge toward it as the registers adapt. The
   acceptance criterion is therefore a **loss-range comparison over the first few hundred steps,
   not equality**, and the size of the initial gap is itself the measurement 1a asked for
   ("is `t̄` load-bearing?"). If it is large, log loss-by-timestep (`_last_t` /
   `_last_per_elem_loss` are stashed at frame granularity for exactly this) and expect the gap to
   concentrate at `t` far from `t̄`.

**Module-level tests** (`tests/test_diffusion_forcing.py`, 24 passing). 2-layer, 128-dim model,
CPU, fp32, flash-attn monkeypatched off:

| property | test | how |
|---|---|---|
| value plumbing | `test_matches_global_t_at_the_register_t` | all frames at `t̄` → matches global `t = t̄` to `1e-5`; parametrized over memory tokens / none / logsnr, plus no-`global_embed` and `padding_mask` variants |
| **alignment** | `test_frame_i_modulates_token_i` | depth-1 model with `self_attn.to_out` zeroed so the block is position-wise; perturbing frame `i` changes hidden-state token `64+i` **and no other**. Frames 0, 7, 23. |
| registers are where claimed | `test_register_t_modulates_only_the_memory_tokens` | changing `df_register_cond_t` moves all 64 memory tokens and no latent |
| schedule independence | `test_registers_are_independent_of_the_schedule` | low-noise vs high-noise per-frame `t`: memory tokens unchanged |
| design, not bug | `test_differs_from_global_t_away_from_the_register_t` | at uniform `t = 0.9` the paths differ, as intended |
| not vacuous | `test_varying_per_frame_t_changes_the_output` | tripwire for `zero_init_branch_outputs=True`, under which every branch is zero and *any* conditioning comparison passes trivially |
| CFG broadcast | `test_cfg_matches_global_t_at_the_register_t` | `cfg_scale=3`, all frames at `t̄` → matches global `t = t̄`; parametrized over `rectified_flow` and `v` (exercises the `alpha` sites) |
| CFG interval per frame | `test_cfg_interval_is_per_frame` | half the frames outside `(0.5, 1)`: those match `cfg_scale=1`, the rest match full-interval CFG, and the two differ |
| CFG interval per item | `test_cfg_interval_is_per_item_for_global_t` | 1-D `t = [0.2, 0.8]`: item 0 unguided, item 1 guided — the old `sigma[0]` gate would have left both unguided |
| compute skip kept | `test_cfg_skipped_when_no_frame_is_in_the_window` | all frames out of window → `torch.equal` to `cfg_scale=1` |
| LoRA interval | `test_lora_interval_warns_with_per_frame_t` | non-default `lora_interval` with 2-D `t` warns and still runs; default `(0, 1)` is silent |
| CFG not vacuous | `test_cfg_changes_the_output` | tripwire: cross-attention conditioning actually reaches the output |
| CFG + `padding_mask` | `test_cfg_with_padding_mask_matches_global_t_at_the_register_t` | the last cell of the shape matrix; parametrized over plain / `cfg_norm_threshold` / `apg_scale` 1.0 and 0.5 |
| config plumbing | `test_df_register_cond_t_reaches_the_model_through_dit_wrapper` | `DiTWrapper(**diffusion.config)` — the call `factory.py` makes — lands `df_register_cond_t` on the model; default is 0.537 |

**Trainer-level tests** (`tests/test_diffusion_forcing_training.py`, 29 passing). A real
`DiffusionCondTrainingWrapper` around a 2-layer, 64-dim `DiTWrapper` with a `NumberConditioner`
for `seconds_total`, `mask_padding_attention=True`, and a stand-in `Trainer` for the lr lookup:

| property | test | how |
|---|---|---|
| shapes | `test_global_t_shape_is_unchanged`, `test_per_frame_t_shape` | all five samplers → `(B,)` / `(B, N)`; frames genuinely differ |
| `df_p_global` | `test_df_p_global_one_collapses_every_item`, `..._mixes_shared_and_per_frame_items` | `1.0` → every row constant; `0.5` → roughly half of 64 items shared |
| Sobol | `test_uniform_sobol_stratifies_over_frames` | 1024 points in 16 bins → exactly 64 each, i.e. stratification is now over frames |
| dist shift | `test_per_frame_t_shift_matches_per_item_shift` | flux / full / logsnr, `(B, N)` with `(B,)` lengths equals per-item scalar shifts, including the `N == B` case |
| dist shift unchanged | `test_schedule_broadcast_branch_is_unchanged` | the `(steps,) × (B,)` sampler branch |
| end to end | `test_training_step_runs_and_backprops` | global and per-frame: finite loss, gradient reaches the timestep embedding, `_last_t` is `(B,)` / `(B·N,)` |
| end to end + shift | `test_training_step_with_per_item_schedule_shift` | `use_effective_length_for_schedule=True` with `(B, N)` t |
| `df_p_global=1.0` end to end | `test_df_p_global_one_gives_shared_t_per_item_end_to_end` | rows of `_last_t` constant |
| one-shot | `test_one_shot_is_per_item_not_per_frame` | an item is all ones or none |
| `log_loss_info` | `test_log_loss_info_bucketing_accepts_per_frame_t` | bucketing runs with `(B, 1, N)` sigmas |

**Which checkpoint (decision, 2026-09-15): the base model, `small-music-base`.** The
`small-music` checkpoint that experiments 01–02 finetune is ARC post-trained: its configs carry
`diffusion_objective: rf_denoiser` and an `arc` block, and its intended sampler is the ping-pong
few-step one. Diffusion forcing on top of that would mean untangling the post-training at the same
time as the per-frame conditioning, and the two would be indistinguishable in a loss curve. The
base model is plain `rectified_flow`, trained with the `trunc_logit_normal` sampler whose mean is
`t̄`, and is the checkpoint the analysis in 1a is actually about. Post-trained variants are a later
question.

Consequences: the config is `small_music_base_df.json` — a text-only twin of
`small_music_base_streamgen.json` (accompaniment conditioning removed, text demo prompts) plus the
two DF flags — and **not** a derivative of `small_music_baseline.json`. There is no recorded
base-model finetune to compare against, so the loss-range check below runs its own twin.

**Remaining: the production run.** Everything above is toy scale. Note as of 2026-09-21 that
`sbatch/df_test.sh` runs (3) below — 10 000 steps at `df_p_global: 0.0`, i.e. the 3.2 run — and
skips (1)–(2). Nothing is wrong with running it first; it just means the loss-range criterion
never gets checked, so a bad curve stays ambiguous between "DF is hard" and "the register `t̄`
choice was expensive". (1)–(2) are two 500-step runs and would remove that ambiguity.

Held-out validation is available now (`--val_dataset_config`, 2026-09-21), but it does **not**
substitute for the twin comparison: `validation_step` noises with a global `t`, so it measures
full-sequence quality, not the per-frame task. Training is 20 layers,
1024-dim, bf16, flash-attn — and with `mask_padding_attention: true` the
`flash_attn_varlen_func` route, which derives `extended_padding_mask` from `x.shape[1]` exactly as
1a derives the register row count. They should agree; "should" has been wrong once already in this
document. One pair of runs settles it and the loss-range criterion in (3) at the same time:

1. `small_music_base_df.json` ships with `df_training: true, df_p_global: 0.0`; set
   `df_p_global` to `1.0` for this run. `df_register_cond_t` does not need setting; the default
   (0.537) is the `trunc_logit_normal` mean and the base model uses that sampler.
   `train_finetune.py --model small-music-base` reads both flags from the `training` block;
   nothing else changes. The "Debug DF training step" entry in `.vscode/launch.json` runs this
   config for 5 steps at batch size 2 with no logger, for stepping through `training_step`
   locally.
2. Run it and a twin with `df_training: false` (same config otherwise, same seed, same data) for
   `--steps 500`, `--log_every 10`. Compare `train/mse_loss` over the same steps. Expected:
   finite, same order of magnitude, a small positive gap at step 0 that closes.
3. Repeat with `"df_p_global": 0.0`. Expected: finite and not diverging. The loss *level* will be
   different — per-frame `t` is a different task — so there is no range to match here, only
   stability. This is the start of 3.2, and its loss curve is 3.2's baseline.

**Still not covered, and not planned here:**

- `interface/diffusion_cond.py:112` reads `sigma[0].item()` for the Gradio UI. It is untouched and
  still assumes 1-D `t`; it belongs to 3.3, together with the samplers.
- `log_loss_info` calls `all_gather` and reshapes with a world dimension that Lightning does not
  add on a single device. The bucketing itself is rank-agnostic now and tested with `all_gather`
  stubbed, but the option looks broken on one GPU independently of this work. Off in every config.

## Out of scope

**Inference schedules.** `build_schedule` (`sampling.py:9`) returns `(steps+1,)` or
`(B, steps+1)`; diffusion forcing needs `(B, steps+1, T)` — a per-frame noise-level trajectory.
`sample_discrete_euler` already branches on `t.dim() == 2` (`sampling.py:157`), so there is a
natural place for a 3-D branch where `t_curr_tensor` is `(B,T)` and `dt_broadcast` is `(B,1,T)`.
The interesting part — which schedules (pyramid, rolling autoregressive) and how far they roll out —
is 3.3, and is a design question in its own right rather than a shape fix.

## Commit order

1. Steps 1–3, gated behind `df_training=False`. No behavior change for shared-`t` batches; the
   one intentional 1-D change is that `cfg_interval` is now gated per item rather than on item 0.
2. Step 4 plus the trainer tests from step 5. `df_training` defaults to `False` and nothing
   sets it, so still no behavior change.
3. Real per-frame `t` sampling and `df_p_global` — the start of 3.2.

## Notes

- No new parameters anywhere in steps 1–4, so existing checkpoints load and finetune straight into
  DF. The register `t̄` is a plain float on the module, not a buffer or `Parameter`, so it does not
  appear in the state dict and old checkpoints load with `strict=True`.
  A *learned* register modulation vector is the obvious next idea and would quietly cost the
  property, at `6D` per block.

## Design log — register conditioning

1a went through three positions before settling, and the reasoning is worth keeping because the
first two look reasonable in isolation:

1. **Zero-pad**, reusing `_left_pad_to_match`. Rejected: adaLN is affine, so zero is not neutral —
   it hands the registers `to_scale_shift_gate` alone, which no pretraining step produced.
2. **Replicate frame 0.** Rejected: arbitrary once frames genuinely differ. There is no reason a
   global register should inherit the leftmost frame's noise level.
3. **Mean over frames.** Held briefly, then rejected on two counts. It is also ambiguous between
   *mean of the embeddings* and *embedding of the mean*; the former is badly wrong, because valid
   timestep embeddings live on a sphere of radius `√128 ≈ 11.31` (`cos² + sin² = 1` per frequency
   pair) and averaging across spread `t` collapses off it — norm 1.67 and cosine 0.167 against the
   true embedding under full DF spread. The latter is well-defined but degenerates: `mean(t)`
   concentrates to `0.537 ± 0.019` under iid sampling, then sweeps the full range at inference under
   a rolling schedule, making the register conditioning schedule-dependent.
4. **Frozen `timestep_embed(t̄)`,** with `"mean"` retained as a knob so an all-frames-equal
   comparison against the global path stays exact.
5. **Frozen `timestep_embed(t̄)` only** (decision, 2026-09-15). The knob existed to make a test
   pass, and that test turned out not to check what it claimed (see step 5). Dropped; alignment is
   tested directly instead.

The through-line: the registers are noise-conditioned in the pretrained model, so the rule cannot be
chosen on conceptual grounds alone — but under iid DF training the honest summary of the window's
noise level *is* a constant, so conditioning them on one is both simpler and closer to what training
would converge to anyway.
- Line numbers are as of the WIP diff on `nithya/df` and will drift.
