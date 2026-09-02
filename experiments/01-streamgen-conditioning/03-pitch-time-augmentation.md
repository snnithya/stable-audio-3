# 1.4 — Pitch and time-stretch augmentation

**Status:** implemented, not yet trained on
**Date:** 2026-08-31

## Question

Slakh's drum performances live in a narrow band of tempos and its accompaniments in a narrow
band of keys. Does widening both — by writing several transposed / re-paced copies of each
track at pre-encode time — give the model more to generalize from, without breaking the
frame-level alignment the whole streamgen mechanism rests on?

This entry answers the second half (alignment survives) and sets up the first, which needs
GPU time.

## Why pre-encode time, and what that costs

Training reads latents, not audio, so there is nowhere at train time to pitch-shift anything
short of decoding and re-encoding every step. The augmentation therefore happens once, in
`scripts/pre_encode_dataset.py`, and is **frozen into the cached latents** — the same
tradeoff the stochastic stem submix already makes.

The way to buy diversity back is to write more than one copy. `--augment_variants N` runs N
passes over the dataset:

| variant | pitch | tempo | submix |
|---|---|---|---|
| 0 | — | — | rolled |
| 1..N-1 | `U(-2, +2)` semitones, whole mix | `U(0.9, 1.1)` rate | re-rolled |

Variant 0 is deliberately the identity, so a dataset encoded with N variants still contains
the original material rather than N copies of which none is clean. Each variant is a fresh
pass through the dataloader, which means the custom metadata fn runs again — so the
accompaniment submix (random stem subset, random per-stem loudness) is re-rolled per variant
for free. Encode time and disk both scale with N.

## What gets pitched, and what must not

**Time stretch is shared across every stream of a sample.** Drums and accompaniment are
stretched by the same rate, in the same call, or they stop being time-aligned — the failure
mode 1.1 already flagged as silent and fatal.

**Pitch shift is shared too** (`--augment_pitch_scope all`, the default). A variant is the
whole arrangement moved to a new key at a new tempo: the accompaniment is transposed and the
drums are transposed by the same interval, which reads as the same performance on a
differently tuned kit.

The alternative, `--augment_pitch_scope controls`, transposes the accompaniment while holding
the drums at their original tuning. It exists because the shared shift teaches the model one
thing that is not true of real music: **drummers do not retune when the key changes.** Under
`all`, kit tuning and accompaniment key move together in every augmented item, so a model
conditioned on the accompaniment can in principle learn to read the drum tuning off the key —
an artifact of the augmentation, not a property of the data. Whether that actually happens is
an empirical question nobody has answered here; `controls` is the one-flag experiment if it
looks like it might be.

Either way the shift is per-stream at the API level, so the choice costs nothing: the pitch
shift folds into the same single STFT pass the time stretch already needs.

## Implementation

`stable_audio_3/data/augmentation.py`. A shift and a stretch together cost **one** STFT pass:
the pitch shift is a resample (which moves pitch and duration together) and the phase vocoder
then pulls the duration to whatever the stretch asked for.

Two details that are load-bearing:

- **The vocoder is driven by target length, not by rate.** Two streams given the same rate
  still round independently; two streams given the same target length cannot. Every stream of
  a sample comes out the same number of samples long by construction.
- **The resample ratio is a rational approximation** of `2 ** (semitones / 12)` with the
  denominator capped at 200. Asking torchaudio for the exact irrational ratio builds a
  resampling kernel large enough to make the pass unusable; the approximation costs under a
  cent of tuning error (measured: worst case 0.11 cents over ±4 semitones).

## Listening to the variants

The listening check counts **samples per variant**, not files:
`--sanity_check_samples 3 --augment_variants 4` writes 3 x 4 = 12 items. The dataloader is
unshuffled and crops deterministically, so it is the *same three tracks* in every variant —
which is the only way the comparison means anything. `decode_preencoded_samples.py -n 3`
does the same for a dataset already on disk: `-n` counts tracks, and every variant of each
selected track is decoded (latent filenames come off `scandir` in arbitrary order, so
slicing the first N indices would have returned an unrelated mix instead).

Two fixes were needed to make that visible rather than merely written:

- **`make_listening_page.py` grouped by the id before the first underscore**, which put all
  four variants of a track on one card as sixteen stacked rows — and, because the labels
  then read `v0_control_…`, broke the rule that sorts control streams below the target. The
  `_v<n>` suffix now stays with the sample id, so each variant is its own card and picks up
  its own metadata: the card states the rate and interval it was written with.
- **`decode_preencoded_samples.py` compared an augmented decode against the raw source
  file**, which reads as drift that is not a defect. It now replays the recorded
  augmentation on the source before slicing (honouring `pitch_scope`), the same correction
  `check_streamgen_alignment.py` needed.

## Results

Smoke test: 2 BabySlakh tracks, `same-s`, `--augment_variants 3`.

| Check | Result |
|---|---|
| Latent and control sidecar shapes | identical per item, 6/6 items |
| Frame count tracks the rate | 2050 → 1990 @ rate 1.030, 2601 → 2757 @ rate 0.943 (exact) |
| `seconds_total` rescaled | 191.0 → 185.4 s, 242.0 → 256.5 s |
| Padding mask rebuilt | `sum(mask) == latent length` for every item |
| Submix re-rolled per variant | 8 / 1 / 2 stems for the three variants of one track |
| Sanity wavs, all four streams | same duration to the sample, per variant |
| Listening log is per variant | `--sanity_check_samples 2 --augment_variants 4` → the same 2 tracks (Track00005, Track00016) in each variant |
| Listening page cards | 2 tracks x 2 variants → **4 cards**, one per variant, controls sorted below the target |
| Card states its roll | `rate 1.042, -1.72 st (all)`; durations follow — 247.2 s / 1.042 = 237.3 s, 202.8 s / 0.987 = 205.4 s |
| **Drum target is actually transposed** | log-spectral distance to a pitched reconstruction **0.11** vs **1.42** to a stretch-only one — 13× separation |
| Recorded policy | every augmented item carries `pitch_scope`; the encode log states it once per run |
| **Time alignment, augmented items** | lag **+0 frames**, **6/6 samples OK** with drums pitched |
| Variant 0 is bit-identical to no augmentation | yes — identity params skip the transform |
| Tests | 46 augmentation tests pass (12 of them on the variant-aware logging); 8 pre-existing streamgen metadata tests pass |

## Note on verifying the pitch shift

Confirming the drums were really transposed took three attempts, and the first two were bad
measurements rather than bad results — worth recording so the next person does not repeat
them:

1. **Sample-level comparison against an offline reconstruction fails**, by a wide margin,
   even when the transform is correct. The phase vocoder accumulates phase over tens of
   thousands of frames, so a negligible difference in input framing decorrelates the waveform
   while leaving the spectrum and the amplitude envelope intact. Never diff augmented audio
   sample-by-sample.
2. **Peak-picking a log-spectrum cross-correlation is degenerate on drums.** A *known* +1.92
   st shift of the drum stem measures +0.00 st by that method — the spectrum is too smooth
   for the peak to localise. This produced a confident false negative before it was checked
   against a synthetic control.
3. **What works** is a two-way comparison: build both a pitched and a stretch-only
   reconstruction from the same source and ask which one the written audio is closer to in
   log-spectral distance. That gives 0.11 vs 1.42 — unambiguous, and it needs no assumption
   that the reconstruction is bit-exact.

The lesson generalises: validate the ruler on a known quantity before trusting it on the
unknown one.

## Notes

`scripts/check_streamgen_alignment.py` needed two fixes to stay meaningful here, both of
which were latent problems that augmentation only made easy to hit:

1. **It re-times the source.** An augmented item records the rate it was written with, and
   the source is stretched by that rate before the cross-correlation. Without this every
   augmented item reads as misaligned, which would have trained the habit of ignoring the
   check.
2. **It correlates against the *selected* stems, and reports near-silent windows as
   inconclusive.** The submix is a random subset; one item drew a single Brass stem that is
   digitally silent through most of the track. Correlating it against the sum of all nine
   stems gave `corr 0.000` and a lag peak at wherever the noise was loudest — reported as
   MISALIGNED. Both halves were wrong: the comparison used the wrong source, and a
   correlation of zero is not evidence of anything. It now uses `streamgen_stems` from the
   metadata and skips samples below `--min_corr`.

Open items:

- **Not yet trained on.** Whether the augmentation actually helps is 1.2's comparison run
  again with an augmented train split; nothing here says it will.
- **The validation split stays unaugmented** (`augment_variants: 1` in
  `slakh_streamgen_validation.json`). Scoring against pitched and stretched audio would
  measure the augmentation as much as the model.
- **Phase-vocoder artifacts on drums.** At ±10% the smearing is mild, but BabySlakh/Slakh
  source is 16 kHz mono upsampled to 44.1 kHz, so the fidelity ceiling was already low. If
  transient smearing turns out to matter, the alternative is a resample-based (speed-change)
  stretch, which is artifact-free but transposes the drums along with the tempo.
- **Variant count is a guess.** 4 is a starting point, not a tuned value; it multiplies
  pre-encode time and disk by 4.
- **Key/tuning correlation.** See above: `scope: all` couples drum tuning to accompaniment
  key across the augmented copies. Worth a look during 1.2's within-model ablation — if the
  model's drums shift tuning when the accompaniment is transposed at inference, that is the
  artifact showing up, and `--augment_pitch_scope controls` is the fix.

## Verification commands

```bash
uv run pytest tests/test_audio_augmentation.py tests/test_streamgen_metadata.py -q

uv run python scripts/pre_encode_dataset.py \
  --dataset_config stable_audio_3/configs/dataset_configs/dataset2preencoding/local_babyslakh_streamgen.json

uv run python scripts/check_streamgen_alignment.py \
  --config stable_audio_3/configs/dataset_configs/preencoded/local_babyslakh_streamgen_preencoded.json \
  --ae_model same-s --ds_ratio 4096 -n 8

# listen: augmented target next to its augmented accompaniment
uv run python scripts/make_listening_page.py \
  --dir /data/hai-res/shared/snnithya/sao-3/data/babyslakh-streamgen-preencoded-same-s/_sanity_check
```
