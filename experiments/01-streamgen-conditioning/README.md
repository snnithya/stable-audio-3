# Experiment 01 — Streamgen conditioning

**Status:** in progress
**Started:** 2026-08-28
**Branch:** `v/r`
**Plan:** `~/.claude/plans/i-want-to-incorporate-sleepy-spring.md`

---

## Hypothesis

A Stable Audio 3 DiT finetuned to generate **drums** while conditioned on a frame-level VAE latent of
the **accompaniment** (the non-drum stems), where that accompaniment is only visible up to the
generation cursor plus a sampled lookahead horizon, will learn to produce drums that are
rhythmically and stylistically locked to the accompaniment — and will do so under a *causal,
limited-lookahead* regime, which is the precondition for real-time / streaming interactive use.

Two sub-claims:

1. **Conditioning works at all.** Adding the accompaniment latent as a `modular_local_cond` gives a
   measurable improvement in drum-track alignment vs. the text-only baseline (`prompt="drums"`).
2. **Lookahead is the useful knob.** Varying `future_visibility` trades off musicality against
   latency: with more lookahead the model anticipates accompaniment changes (fills, transitions);
   with none it can only react. If this holds, `future_visibility` becomes the dial for the
   latency/quality tradeoff in an interactive system.

## Background

Ported from `sat-zenon` (`/data/hai-res/snnithya/sat-zenon-2/sat-zenon`), where the mechanism is
called "streamgen". That implementation routes the accompaniment latent through `input_add_ids` and
gates it with a `tf_inpaint_mask` whose horizon is set by `future_visibility`.

The port is not mechanical — SA3's plumbing differs:

| | sat-zenon | stable-audio-3 |
|---|---|---|
| Latent | 64-ch @ ~21.5 Hz | **256-ch @ ~10.8 Hz** (ds 4096) |
| Routing | `input_add_ids`, one fused `nn.Linear` at transformer input | **`modular_local_cond`**, zero-init MLP per id, added at *every* block |
| Conditioner | `ExtractedTensorConditioner` (identity) | injected post-conditioner (same pattern as inpaint conds) |
| Lookahead mask | `tf_inpaint_mask` + `future_visibility` | **absent — being added** |

Latents are therefore not transferable; the accompaniment is re-encoded with SA3's own autoencoder.
Choosing `modular_local_cond` matters: its output layer is zero-initialized, so the new conditioning
is a **no-op at step 0** and the pretrained checkpoint is not disturbed at the start of finetuning.

## Method

- **Data:** BabySlakh (20 tracks), `streamgen-drum-mirror` layout — `tracks/drums/TrackXXXXX/Drums.wav`
  with sibling `tracks/other/TrackXXXXX/*.wav`.
- **Target:** drum stem latents (already pre-encoded).
- **Condition:** stochastic submix of the "other" stems — random stem subset, each LUFS-normalized to
  `U[-30, -15]`, summed and declipped — VAE-encoded to a `[256, T]` sidecar aligned frame-for-frame
  with the drum latents.
- **Model:** `small-music` + two `modular_local_cond_configs`: `streamgen_latent` (256) and
  `tf_inpaint_mask` (1).
- **Masking:** `CAUSAL_MASK` with `future_visibility` sampled per item. `streamgen_latent` is
  multiplied by `tf_inpaint_mask` (positive polarity — visible in context + lookahead), unlike the
  inpaint conds which use `(1 - mask)`.

## Sub-experiments

| # | Name | Question | Status |
|---|------|----------|--------|
| 1.1 | [Data + wiring validation](01-data-and-wiring.md) | Does the accompaniment latent reach the model correctly aligned? | **complete** |
| 1.2 | [Baseline finetune](02-baseline-finetune.md) | Does conditioning reduce loss / improve alignment vs. text-only? | **implemented, not run** |
| 1.3 | Lookahead sweep | How does `future_visibility` affect musicality and anticipation? | not started |

## Results

**1.1 (complete).** The mechanism is wired and verified end-to-end. Control is time-aligned
with the target at lag 0, gradient reaches the streamgen projections in all 20 transformer
blocks, and the conditioning is a bit-exact no-op at step 0 so the pretrained checkpoint is
undisturbed. See [1.1](01-data-and-wiring.md) for the full table.

**1.2 (implemented, not run).** The matched pair of finetunes, the held-out evaluation and
the sbatch scripts to launch them are written; no GPU time has been spent. It moves off
BabySlakh onto the full Slakh2100 splits (1289 train / 270 validation tracks), because with
20 tracks there is no held-out set worth the name. See [1.2](02-baseline-finetune.md) for the
design and the run order.

**1.3.** Not yet started.

## Notes

- The data for 1.2 onwards is **Slakh2100** (`streamgen-drum-mirror`, official splits), not
  BabySlakh. BabySlakh remains the smoke-test path for plumbing changes.
- The submix is currently **frozen per cached track** (rolled once at pre-encode time). Intent is to
  move mixing to train/inference time so the stem subset and levels are re-rolled — otherwise every
  epoch sees an identical accompaniment per track, which limits augmentation diversity.
- `modular_local_cond` is duplicated for batched CFG but **not** CFG-dropped
  (`models/dit.py:440-450`), so streamgen is an always-on control. Making it CFG-guidable is a
  deliberate follow-up, not part of the first pass.
