# 1.2 — Baseline finetune

**Status:** implemented, not yet run
**Date:** 2026-08-28

## Question

Does conditioning on the accompaniment actually help? Two readings of "help":

1. **Information.** Does the accompaniment latent lower held-out denoising loss on the drum
   target, relative to a text-only model trained identically?
2. **Musicality.** Do the generated drums line up with the accompaniment they were given —
   and not with a different track's accompaniment?

1.1 established that the control reaches the model, aligned, and that gradient flows to it.
It said nothing about whether the model learns to *use* it. That is this sub-experiment.

## Design

Two arms, trained concurrently as a SLURM array, differing in exactly one thing:

| | `cond` | `base` |
|---|---|---|
| Model config | `small_music_streamgen.json` | `small_music_baseline.json` |
| `modular_local_cond_configs` | `streamgen_latent` (256), `tf_inpaint_mask` (1) | *(removed)* |
| Pretrained checkpoint | `small-music` | `small-music` |
| Target / task | drums, CAUSAL_MASK, `future_visibility` `[-4, 0]s` | identical |
| Optimizer, LR, steps, seed, data order | AdamW 1e-5, 20k steps, seed 42 | identical |

`small_music_baseline.json` was generated *from* the streamgen config by deleting one key, and
a check confirms the two are otherwise identical, so the arms cannot drift apart through an
edit to one of them.

The text conditioner is **frozen** in both arms. Every item's prompt is the constant string
`"drums"`, so there is nothing for the text encoder to learn; training it would only add a
second moving part that differs between runs.

### Why a third comparison matters more than the first two

The cross-arm comparison has an unavoidable confound: two training runs land in different
places, and a loss gap of a few percent could be run-to-run variation rather than the
conditioning. So the evaluation also does a **within-model ablation** — the same `cond`
checkpoint, scored with its accompaniment (a) as trained, (b) zeroed, and (c) swapped with
another item's. There is only one set of weights, so a gap there is unambiguously the model
reading the control. If (a) beats (b) and (c) but `cond` does not beat `base`, the honest
conclusion is that the conditioning is used but the text-only model compensates.

### Data

Slakh2100, `streamgen-drum-mirror` layout, using the **official splits**: 1289 train tracks
(~89h of drums) and 270 validation tracks, held out entirely from training.

BabySlakh's 20 tracks were the right size for 1.1's plumbing checks and are the wrong size
here — with 20 tracks there is no held-out set worth the name, and "does conditioning
generalize" is precisely the question. The full corpus is already on disk in the layout the
streamgen metadata fn expects, so this costs a pre-encode pass and nothing else.

Source audio is 16 kHz mono, upsampled by the loader (same caveat as 1.1): the fidelity
ceiling is low, and audio quality judgements from this dataset should be discounted.

### Metrics

**Held-out loss** — `scripts/eval_streamgen.py`, on the validation split. Every arm is scored
on byte-identical inputs: the same centred crop of each track, the same causal mask (8s of
drum context, no lookahead), the same timestep grid, and noise drawn from a generator seeded
by the item index. Losses are therefore *paired per item*, and the reported quantity is the
mean per-item difference with a bootstrap CI over items, not two separate averages eyeballed
against each other.

**Onset alignment** — each arm continues a held-out drum prefix; the continuation is decoded
and compared against the accompaniment's onset envelope (half-wave-rectified spectral flux).
Two numbers, each reported with a ceiling and a floor:

- `xcorr_at_zero` — normalized cross-correlation of the two onset envelopes at lag 0. This is
  the primary number; the peak value and its lag are also reported, because a model that
  finds the tempo but not the phase scores high on the peak and near zero here.
- `beat_hit_rate` — fraction of the drum onset energy landing within ±60 ms of a beat grid
  estimated from the accompaniment alone.

The ceiling is the **ground-truth drums** against the same accompaniment; the floor is the
generated drums against a **different track's** accompaniment. On BabySlakh the ceiling
measures ≈ 0.25 for `xcorr_at_zero` and ≈ 0.26 for `beat_hit_rate` (floor ≈ 0.0 and ≈ 0.19),
so the usable range is narrow and the ceiling/floor pair is what makes a middle number
readable at all. Report where the arms fall between them, never the raw value alone.

## How to run

Four steps, in order. None of them has been run yet.

```bash
# 1. Pre-encode Slakh train + validation (array task 0 = train, 1 = validation)
sbatch sbatch/01_2_preencode_slakh.sbatch

# 2. Verify the control is still time-aligned. Do not skip this after a re-encode --
#    a crop desync produces conditioning that trains happily and is musically wrong (see 1.1).
uv run python scripts/check_streamgen_alignment.py \
    --config stable_audio_3/configs/dataset_configs/preencoded/slakh_streamgen_validation_preencoded.json

# 3. Both finetunes, concurrently (task 0 = cond, task 1 = base)
sbatch sbatch/01_2_finetune.sbatch

# 4. Paired evaluation of the two checkpoints
sbatch sbatch/01_2_eval.sbatch
```

Results land in `experiments/01-streamgen-conditioning/results/1_2.json`, with listenable
audio in `results/audio/` — per item, the generated drums alone and mixed over the
accompaniment, next to the ground-truth drums and the accompaniment on their own.

The eval script also runs standalone against an intermediate checkpoint:

```bash
uv run python scripts/eval_streamgen.py \
    --arm cond stable_audio_3/configs/model_configs/small_music_streamgen.json <ckpt> \
    --arm base stable_audio_3/configs/model_configs/small_music_baseline.json <ckpt> \
    --dataset_config stable_audio_3/configs/dataset_configs/preencoded/slakh_streamgen_validation_preencoded.json
```

## Reading the result

| Outcome | Reading |
|---|---|
| `cond` < `base` on held-out loss, **and** cond < zero/shuffle within the `cond` arm | Conditioning works. Proceed to 1.3. |
| Ablation gap but no cross-arm gap | The control is being read, but text-only compensates on loss. Lean on the alignment metrics and the audio to decide. |
| No ablation gap | The model ignored the control. Suspect the zero-init projections never left zero (check gradient norms in wandb), the LR, or a data problem — not the hypothesis. |
| Alignment near the mismatched floor despite a loss gap | The control is being used for timbre/density, not for timing. That is a different (and less interesting) result than the hypothesis claims. |

## Notes

- **A gap in the demo callback, fixed here.** `_generate_inpaint_demos` built its
  conditioning without `tf_inpaint_mask` or `streamgen_latent`, so the training demos for a
  streamgen model were generated *as if it were the baseline*. They would have looked fine
  and told us nothing. Fixed in `training/diffusion.py`; the demos now see what training
  sees.

- **The eval forces `random_crop=False`.** Caught during a smoke test: with random cropping
  on, each arm — and each rerun — gets a different window of the same track, which silently
  destroys the pairing the whole comparison rests on. The script now overrides it regardless
  of what the dataset config says. With that fixed, the three ablation modes on a zero-init
  model return bit-identical losses, which re-confirms 1.1's step-0 no-op through the eval
  harness itself.

- **`sample_size` must be passed on the pre-encode CLI, not put in the config.**
  `merge_config_into_args` treats any argparse default that is not `None` as
  "CLI-supplied", so a `sample_size` key in the dataset JSON is silently ignored — which
  would have truncated the ~20% of Slakh tracks longer than the 285s default without any
  warning. The sbatch script passes `--sample_size 16760832` (380s) explicitly.

- **The submix is still frozen per cached track.** Rolled once at pre-encode time, as in 1.1.
  Every epoch sees the same accompaniment for a given track. This limits augmentation
  diversity on the train split; on the validation split it is a feature, since it makes the
  held-out condition fixed and identical for both arms.

- **`seconds_total` is the whole track's duration, not the crop's.** So the schedule shift is
  computed as if generating a 4-minute clip while actually generating 23.8s. This is
  consistent between training, demos and eval, so it does not bias the comparison — but it is
  a quirk to fix before reading much into absolute sample quality.

- **Unverified operational assumptions.** Nothing here has been run at scale. Wall-clock,
  throughput, and peak memory are estimates: `--batch_size 8 --accum_batches 2` is a
  conservative guess for a 46 GB L40S at 256 latent frames, and 20k steps is a guess at
  "enough". Check both after the first few hundred steps and adjust — if memory allows,
  `--batch_size 16 --accum_batches 1` is the same effective batch at roughly twice the step
  rate.

- **EMA is off** (`--use_ema` not passed), to keep memory down and the two arms simple. The
  eval script can load EMA weights (`--use_ema`) if that changes.

- **Optional third arm.** A model with the streamgen pathway trained on *shuffled*
  accompaniment would separate "the extra parameters help" from "the accompaniment content
  helps". The eval-time shuffle ablation covers most of this at zero training cost, which is
  why it is not in the default array.

## Verification commands

```bash
uv run pytest tests/test_streamgen_eval_metrics.py -q     # alignment metrics
uv run pytest tests/ -q                                   # 97 tests, all passing
```
