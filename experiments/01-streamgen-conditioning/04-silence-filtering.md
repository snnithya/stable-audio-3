# 1.5 — Silence filtering at pre-encode time

**Status:** implemented; both Slakh splits encoded at RMS + fraction 0.3, and both being
re-encoded RMS-only into the other root (train done, validation in flight 2026-09-09). The
cutoff is still the one read off BabySlakh — the RMS-only encodes are the census that should
set it, and the bands are now decoded for a listening pass.
**Date:** 2026-09-02 (filtered Slakh encode 2026-09-08; census + band listening 2026-09-09)

## Question

Which items should a pre-encoded dataset actually contain, and what happens to the ones it
should not?

Two separate questions hid inside that, and the pipeline was answering both badly:

1. **Is this item worth encoding?** The only test was `is_silence` in `dataset.py` — peak
   below -60 dBFS over the whole clip. One sample above that anywhere in a six-minute file
   makes the file content. It never looked at the control at all.
2. **What happens to a rejected item?** It was replaced, not dropped:
   `return self[random.randrange(len(self))]`. The batch stayed full and a *second copy of
   some other track* was written under the rejected item's id.

The second one is the serious defect: it is invisible downstream. Nothing in the output
distinguishes a duplicate from real data.

## Background

`sat-zenon` (`scripts/zenon/pre_encode/pre_encode.py`) had this right and the port to this
repo lost it. Its check is RMS-based:

```python
def is_silent(self, audio, threshold: float = -50.0) -> bool:
    """RMS dBFS based silence detection"""
    rms = 20 * torch.log10(torch.sqrt(torch.mean(audio ** 2)))
    return rms < threshold
```

applied in two places — `_save_sample:1037` on the target (and it `return`s without writing,
so the item is genuinely absent), and `load_and_mix_stems:789` on each accompaniment stem
*after* it has been seeked to the chunk offset and cropped to `target_length`, so the test
is on the window that gets encoded. If no stem survives, the mix is `None` and the drum
latent is not written either.

Both properties were missing here. `custom_md_slakh_streamgen.py` measured
`mean(x²) < 1e-6` on the **whole stem file** with no crop, so a track whose accompaniment
enters at 2:00 passed the gate and produced a silent control for a 0:00–0:13 window.

## Method

Ported both properties, then added the check the original does not have — see Results for
why it turned out to be needed.

- `stable_audio_3/data/utils.py`: `rms_dbfs`, `is_silent` (RMS floor, default -50 dBFS), and
  `silence_fraction` (fraction of 20ms frames whose peak is below -60 dBFS).
- `stable_audio_3/data/dataset.py`: `SampleDataset(resample_on_reject=...)`. Default `True`
  keeps the training behaviour (a rejected item is swapped so the batch stays full). The
  pre-encode script passes `False`, which hands rejects through flagged with `__reject__`
  and `__reject_reason__` for the caller to drop.
- `custom_md_slakh_streamgen.py`: stems are cropped to the **target's valid region** before
  the silence test, and the summed mix is tested too. Cropping there also stops the
  accompaniment from extending past the drums' padding mask into a region the trainer masks
  out.
- `scripts/pre_encode_dataset.py`: drops rejected and silent items instead of writing them,
  applies the level checks to the target **and every control**, and records a `levels` block
  (`rms_dbfs`, `silence_fraction` per stream) in every sidecar JSON. `_skipped.json` records the
  counts and reasons per variant.

Everything is measured over the item's **valid** region, from the padding mask, and after
augmentation, so what is judged is what is written.

## Results

### The substitution was writing duplicates into datasets already on disk

Directly measured, not inferred:

| Dataset on disk | Items | Unique tracks | Extra copies |
|---|---|---|---|
| `babyslakh-streamgen-preencoded-w-augs-same-s` (per variant) | 20 | 17 | 3 (15%) |
| `slakh-streamgen-preencoded-same-s/validation` | 270 | 215 | **55 (20%)** |

A fifth of the Slakh **held-out** split is duplicated content, and the duplicated tracks are
correspondingly over-weighted. That is the set 1.2 scores on.

The substitution is random, so each augmentation variant duplicated a *different* set of
tracks (v0 duplicated Track00013/00008/00019, v3 duplicated Track00020 three times).

On a 13.3s window, the three BabySlakh rejects and what the old code wrote instead:

| Index | File at that index | What was written |
|---|---|---|
| 1 | Track00016 | a second copy of Track00015 |
| 9 | Track00018 | a second copy of Track00006 |
| 11 | Track00009 | a second copy of Track00019 |

### RMS is a much better test than peak, but it is not a percentage-of-silence test

This is worth stating plainly because it is easy to assume otherwise. RMS is an average, so
at the ported -50 dBFS threshold it only rejects windows that are close to empty:

| Content in a 10s window (at full scale) | RMS |
|---|---|
| 10% | -10.9 dBFS |
| 1% | -20.9 dBFS |
| 0.1% | -30.9 dBFS |
| 0.01% | -40.9 dBFS |
| 0.001% | -51.3 dBFS |

A window that is 99.9% silence with one hit in it measures about -31 dBFS and clears the
floor comfortably. What -50 dBFS *does* catch is the realistic failure — a dead stem, a
noise floor, a track that has not started — which the peak test cannot catch at all.

So `silence_fraction` was added alongside it, and it earns its place immediately. From a
BabySlakh pass (17 items, 13.3s window, `same-s`):

| Track | target RMS | target silent | control RMS | control silent |
|---|---|---|---|---|
| Track00019 | -45.6 | **91%** | -21.6 | 28% |
| Track00006 | -35.4 | **86%** | -21.6 | 29% |
| Track00007 | -27.8 | 68% | -27.6 | 18% |
| Track00003 | -26.9 | 19% | -39.9 | **96%** |
| Track00012 | -30.7 | 18% | -32.6 | 62% |
| Track00015 | -27.7 | 5% | -18.8 | 0% |

Every row here passes the RMS floor. Track00003 is the case the whole exercise is about: a
perfectly good drum target paired with an accompaniment that is silent for 96% of the
window — a training pair whose lesson is that the control carries no information.

Distribution over the 17: target `silence_fraction` median 0.30 (0.05–0.91), control median
0.28 (0.00–0.96). No single cutoff is obviously right, which is why
`--max_silence_fraction` is **off by default** and the stats are recorded on every item
instead: a first pass gives the distribution to choose from.

### Alignment survives

Cropping stems to the target's valid window changes what goes into the encoder, so the
alignment check was re-run on freshly encoded output: 3/3 samples, target lag +0 (corr
0.988–0.996), control lag +0 (corr 0.913–0.989).

### Tests

13 new tests in `tests/test_preencode_silence_filter.py`, including the two that pin the
limits honestly — that RMS catches a dead stem the peak test waves through, and that RMS
alone cannot see a 99.9%-empty window. 67 pass across the streamgen/augmentation suites.

## The first filtered encode of Slakh (2026-09-08)

Both splits were re-encoded from scratch with the filter on, into a **new** root so the old
duplicate-contaminated sets are not overwritten while 1.2 is still pointed at them:

```
/data/hai-res/shared/snnithya/sao-3/data/slakh-streamgen-preencoded-same-s-wo-silence/{train,validation}
```

As encoded (`sbatch/01_2_preencode_slakh.sbatch`, `sbatch/01_2_preencode_slakh_validation.sbatch`):

```
scripts/pre_encode_dataset.py --model same-s --batch_size 8 \
    --sample_size 587853 --pad --max_silence_fraction 0.3
```

so the window is **13.3 s — 144 latent frames — taken from the start of every track**. The
pre-encode path sets `random_crop=False`, so this is the first 13.3 s, not a sampled one.
Train writes 2 augmentation variants, validation 1. The `--silence_threshold_db -50` /
`--silence_frame_threshold_db -60` defaults were left alone.

`--max_silence_fraction 0.3` was read off the BabySlakh distribution in the table above
(target median 0.30, control median 0.28) rather than off a first pass over Slakh itself —
the "run once with the filter off, then choose" recommendation below was skipped. The
BabySlakh numbers were measured at the same 13.3 s window, so the cutoff is at least
calibrated to the right window length.

### Validation split: 56 of 270 items survive

Job 1766413, 2×L40S, 3m45s wall clock, completed 2026-09-08T17:54. From `_skipped.json`:

| First failing check | Items |
|---|---|
| target over the silence-fraction limit | 102 |
| peak below silence threshold (`is_silence`, whole file, in `dataset.py`) | 55 |
| target below the RMS floor | 36 |
| `streamgen_audio` over the silence-fraction limit | 19 |
| accompaniment silent over the encoded window (metadata fn) | 2 |
| **written** | **56** |

Each item records only its *first* failure, tested in the order target RMS → target fraction
→ control RMS → control fraction, so these are not independent counts: an item rejected on
target fraction was never tested against its control. The RMS/peak checks alone would
therefore have kept **at most** 177 items (66%); the silence-fraction limit is what takes it
to 56 (21%).

Levels of the 56 that were written (from their sidecar `levels` blocks):

| Stream | RMS dBFS min / median / max | `silence_fraction` min / median / max |
|---|---|---|
| target | -47.6 / **-27.0** / -22.5 | 0.01 / **0.17** / 0.297 |
| `streamgen_audio` | -32.9 / **-20.1** / -3.4 | 0.00 / **0.17** / 0.296 |

### Why the drop rate is this high

Two things compound, and neither is the filter misbehaving:

- **The window is the first 13.3 s of the track.** Intros. A drum stem that enters at 0:20 is
  100% silent in the encoded window even though the track is full of drums downstream. This
  is the single largest bucket (102 items on the target's fraction alone).
- **30% of 13.3 s is 4 seconds.** At the 380 s window the config comments were originally
  written for, a 30% cutoff means a track with long dead stretches; at 13.3 s it rejects any
  phrase containing a 4 s break. The same number is a much stricter test at this length.

The survivors confirm the filter is cutting into the distribution rather than trimming a
tail: max target `silence_fraction` 0.297 and max control 0.296 against the 0.30 cutoff, with
medians at 0.17. The kept set is pressed right up against the threshold.

### Train split: 252 and 240 items, from 1289 tracks

Job 1766290, 4×L40S, 1289 files × 2 variants, completed 2026-09-08T18:16 (job 1766197 was the
same submission with the flag misspelled as `--silence_fraction`; it died in argparse in 77 s,
which is the cheap failure mode this script's flag surface is meant to give). From
`train/_skipped.json`:

| First failing check | v0 | v1 |
|---|---|---|
| target over the silence-fraction limit | 491 | 495 |
| peak below silence threshold | 269 | 269 |
| target below the RMS floor | 158 | 174 |
| `streamgen_audio` over the silence-fraction limit | 105 | 95 |
| accompaniment silent over the encoded window | 14 | 14 |
| `streamgen_audio` below the RMS floor | — | 2 |
| **written** | **252** | **240** |

492 latents across the two variants out of 2578 attempts — 19%, in line with validation's 21%,
and the "expect 250-300 per variant" guess held.

The v0/v1 differences are not measurement noise on the same items. v1 is the augmented pass, so
a time-stretch of up to ±10% changes how much of the fixed 13.3 s window each track fills, and
the stem submix is re-rolled per pass, which is what moves `streamgen_audio` from 105 to 95 and
adds the 2 mixes that land under the RMS floor. The 269 peak rejections and the 14 accompaniment
rejections are identical across variants because neither depends on the roll.

### A second, RMS-only train encode overwrote the old root (job 1776023)

Separately, at 2026-09-08T23:44, job 1776023 encoded the train split **again** with only the
RMS floor — no `--max_silence_fraction` — into `slakh-streamgen-preencoded-same-s/train/`, the
*old* root. That is what the committed `slakh_streamgen_train.json` and
`sbatch/01_2_preencode_slakh.sbatch` still do: neither was repointed at the `-wo-silence` root
nor given the cutoff, so **re-running the committed train recipe does not reproduce the encode
described above**. Fix that before anything else re-encodes train.

Two consequences, both checked on disk rather than inferred:

- **The duplicate-contaminated train encode no longer exists.** That directory now holds 1678
  items (848 + 830), every file written between 23:23 and 23:44, with nothing surviving from the
  Sep 2 pass. The 20%-duplicates problem is gone from train — but by silent overwrite, into a
  path whose name still says nothing about which filter produced it.
- **The duplicate-contaminated validation encode does still exist**, untouched since Sep 2: 270
  items, and not one sidecar carries a `levels` block. That is the set 1.2 is scored on, and it
  is the one that actually needed replacing.

| Encode | Filter | v0 written | v1 written |
|---|---|---|---|
| `…-same-s-wo-silence/train` (job 1766290) | RMS -50 + fraction 0.3 | 252 | 240 |
| `…-same-s-wo-silence/validation` (job 1766413) | RMS -50 + fraction 0.3 | 56 | — |
| `…-same-s/train` (job 1776023) | RMS -50 only | 848 | 830 |
| `…-same-s/validation` (Sep 2) | peak only, **20% duplicates** | 270 | — |

The RMS-only train encode is an accident worth keeping: it is most of the level census this
experiment recommended and never ran.

### Consequences for 1.2

- **The held-out set is 56 tracks, not 270.** The paired per-item loss comparison and its
  bootstrap CI get roughly √(270/56) ≈ 2.2× noisier. That is the price of removing both the
  20% duplicates and the empty windows, but it is worth deciding deliberately: raising the
  cutoff, or encoding the validation split at a longer window, buys the power back.
- **The validation split is no longer a uniform sample of Slakh validation.** It is now the
  subset whose drums *and* accompaniment both start early and play near-continuously.
  Absolute alignment numbers from it are not comparable to anything scored on the old set —
  only within-set arm-vs-arm comparisons are.
- **The preencoded configs were repointed at this encode** (path, and
  `latent_crop_length: 144` to match the stored length — they were 256 for train and 1024 for
  validation, which against 144-frame latents would have silence-padded every item rather
  than cropping it). `scripts/eval_streamgen.py --eval_frames` still defaults to 256, which
  against a 144-frame item silently returns the whole item; pass `--eval_frames 144`.

## The no-filter level census (staged 2026-09-09, not yet submitted)

`--max_silence_fraction 0.3` was taken from 17 BabySlakh tracks. It is now the single biggest
influence on what both splits contain, and it has never been checked against Slakh's own
distribution. This is the pass that checks it.

### Why a filtered pass cannot choose its own threshold

Levels are computed for every item the level check sees, but the sidecar JSON is only written
for items that **survive** (`pre_encode_dataset.py:478`). So the 56 written validation items
all sit below 0.30 by construction — max target `silence_fraction` 0.297, max control 0.296 —
and carry no information about the 214 that went. The distribution is truncated exactly at the
knob it would be used to set. `--no_silence_filter` removes the truncation: nothing is dropped
by the level check, so every item that reaches it gets a sidecar.

`--no_silence_filter` switches off **only** that check. The two upstream gates still drop items
with no sidecar and no recorded levels:

- `is_silence` in `dataset.py` — peak below -60 dBFS over the whole crop (55 validation items).
- The metadata fn's own per-stem RMS test — `accompaniment silent over the encoded window` (2).

So expect **≈213 sidecars from 270 tracks**. Neither omission biases the choice of cutoff:
both buckets hold items that no cutoff would have kept.

### What the accidental train census already says

Job 1776023 (RMS floor only, no fraction limit) left 848 v0 sidecars with `levels` blocks, so
the train distribution can be read right now with no GPU time at all:

| Stream | `silence_fraction` p05 / p25 / **median** / p75 / p95 |
|---|---|
| target | 0.074 / 0.207 / **0.346** / 0.571 / 0.857 |
| `streamgen_audio` | 0.012 / 0.168 / **0.288** / 0.433 / 0.776 |

**The cutoff sits below the target median.** BabySlakh's 17 tracks gave a target median of 0.30
and Slakh's 848 give 0.346, so `--max_silence_fraction 0.3` rejects more than half the corpus on
the target stream alone before the control is even looked at. That is the calibration error, and
it is what the 19-21% keep rate is.

Survivors at each candidate cutoff, RMS floor held at -50 dBFS (target **and** control must
clear it), out of 848 censused / 1289 in the corpus:

| cutoff | 0.1 | 0.2 | **0.3** | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| items | 25 | 123 | **254** | 386 | 514 | 601 | 678 | 757 | 807 | 848 |
| % of corpus | 2% | 10% | **20%** | 30% | 40% | 47% | 53% | 59% | 63% | 66% |

Two things to read off it. There is no plateau — no natural cutoff where the curve flattens and
the choice stops mattering — so this number has to be argued for on musical grounds, not found
in the data. And 0.5 roughly doubles the held-out set for the price of admitting windows that
are half silent, which on a 13.3 s window is a phrase with a ~6.6 s break in it.

Two limits on this census, both real:

- **It is truncated at -50 dBFS**, because the pass that produced it still applied the RMS
  floor. The floor sweep is consequently flat from -70 to -50 and cannot say what the floor
  costs. Only a genuine `--no_silence_filter` pass can.
- **It is train, not validation.** The held-out split is the one whose size drives 1.2's
  statistical power, and validation ran 21% against train's 19% at the same settings — close,
  but not a substitute.

Replaying the filter over this census predicts 254 written / 491 target-fraction /
103 control-fraction against the filtered run's actual 252 / 491 / 105. The 2-item gap is the
stem submix being re-rolled per pass: target-side numbers reproduce exactly, control-side
numbers only to within the roll.

### Listening to the bands (2026-09-09)

There is no plateau, so the cutoff cannot be read off the survivor curve — it has to answer a
musical question: at what point is a window too empty to be a useful training pair?
`scripts/sample_silence_buckets.py` decodes two items from each 0.1-wide `silence_fraction`
band, target and control, and annotates every card with the band and the measured levels, so
the question can be answered by ear. Picks are spread across each band by `silence_fraction`
(for `-n 2`, roughly its 25th and 75th percentile) rather than drawn at random, so the two
clips bracket the band instead of landing next to each other; there is no seed.

Run over the RMS-only train encode (848 v0 items), which is the only set on disk that has the
upper bands at all:

| target `silence_fraction` | 0.0-0.1 | 0.1-0.2 | **0.2-0.3** | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| items | 55 | 149 | **153** | 132 | 93 | 81 | 55 | 65 | 35 | 30 |

The distribution is unimodal with its **mode at 0.2-0.3**, which means 0.3 is the worst place
to put the threshold if stability matters: it slices at the peak, where the derivative of the
survivor count is highest (~130 items per 0.1 of cutoff around 0.3, against ~65 around 0.7).
Any small change to the window length, the augmentation roll, or the corpus moves a large
number of items across it. That is an argument for choosing a cutoff off the mode independent
of what the clips sound like.

Output: `…/data/_listening/silence_bands_train_v0/` — 40 wavs, 20 cards, `index.html` 16.8 MB
self-contained (mp3-embedded; flac would be ~6× that for no audible gain on this question).
The cards show the decode of the stored latent and of the stored control, which is what the
model trains on; there is no source row, so use `scripts/decode_preencoded_samples.py` for a
source/decoded A/B.

### Files

| File | What it is |
|---|---|
| `sbatch/01_2_preencode_slakh_validation_nofilter.sbatch` | The job. Same encode as the filtered validation job, `--no_silence_filter` in place of `--max_silence_fraction 0.3`, then runs the analysis so the numbers land in the job log. |
| `…/dataset2preencoding/slakh_streamgen_validation_nofilter.json` | Config. Mirrors `slakh_streamgen_validation.json` — same source tree, same metadata module, same control, same `augment_seed` — differing only in `output_path`, `no_silence_filter`, `augment_variants: 1`, `sanity_check_samples: 0`. |
| `scripts/analyze_silence_levels.py` | Reads `levels` out of any pre-encode output: per-stream quantiles, a survivor sweep over both knobs, the RMS/fraction overlap table, and the `_skipped.json` buckets the settings would produce. Warns loudly when pointed at a filtered pass, and reports coverage from `_skipped.json` so percentages are of the corpus, not of the sidecars that happen to exist. |
| `scripts/sample_silence_buckets.py` | Decodes N items per `silence_fraction` band into a directory `make_listening_page.py` can render, each card labelled with its band and levels. For choosing the cutoff by ear. |

Output goes to a **separate root**, `slakh-streamgen-nofilter-census-same-s/validation/`, so
neither existing validation encode is touched. It is diagnostic: ~213 items, ~65 MB, and
nothing should train or score on it. One GPU, not the two the filtered job asked for —
`pre_encode_dataset.py` runs one autoencoder on one device, so the second was idle.

### Plan change 2026-09-09: validation re-encoded RMS-only, which is the census

The Sep 2 duplicate-contaminated `same-s/validation` was deleted and
`slakh_streamgen_validation.json` repointed at that path with `--max_silence_fraction` dropped
from `01_2_preencode_slakh_validation.sbatch`, so validation is being re-encoded RMS-only —
exactly what job 1776023 did to train. This is the right order of operations: it fixes the
contaminated held-out set first, and the resulting encode **is** the validation level census,
because an RMS-only pass writes a sidecar for every item the fraction limit would have dropped.
Expect ~177 items (270 − 55 peak − 36 target RMS − 2 accompaniment), spanning the whole
`silence_fraction` range.

After it lands the two roots are a clean matched pair, which they were not before:

| Root | Filter | train | validation |
|---|---|---|---|
| `slakh-streamgen-preencoded-same-s` | RMS -50 only | 848 / 830 | ~177 (encoding) |
| `slakh-streamgen-preencoded-same-s-wo-silence` | RMS -50 + fraction 0.3 | 252 / 240 | 56 |

That makes the dedicated `--no_silence_filter` job **largely redundant**. Its one remaining use
is the RMS floor itself: an RMS-only pass cannot record levels for the items it drops, so the
floor sweep stays flat from -70 to -50 and the -50 default is still unexamined. Run the census
job only to interrogate the floor; the fraction cutoff can be chosen from the RMS-only encodes.

**Open, and it bites a training run silently:** the preencoded configs are currently a mismatched
pair — `slakh_streamgen_train_preencoded.json` points at `same-s/train` (RMS-only) while
`slakh_streamgen_validation_preencoded.json` still points at `-wo-silence/validation` (RMS +
0.3). A run right now trains on one distribution and scores on a strictly denser subset of
another. Repoint validation at `same-s/validation` once the encode lands, or repoint train at
`-wo-silence/train` — either, but not one of each.

### Decided: `--max_silence_fraction 0.85` (2026-09-09)

Chosen by listening to the band page above, not from the survivor curve — the curve has no
plateau, so there was nothing in the data to pick the number out of. **0.85 for Slakh at the
13.3 s window.**

What it keeps, from the train v0 census (848 items, RMS floor already applied):

| cutoff | 0.75 | 0.80 | **0.85** | 0.90 | 0.95 | off |
|---|---|---|---|---|---|---|
| items kept (both streams) | 704 | 757 | **784** | 807 | 834 | 848 |
| % of the 1289 corpus | 55% | 59% | **61%** | 63% | 65% | 66% |

At 0.85 the fraction limit stops being the dominant knob and becomes a **backstop**: it keeps
92% of what clears the RMS floor, and what it removes is windows with under about 2 s of content
in 13.3 s. That is exactly the case the RMS floor is blind to — 99.9% silence with one full-scale
hit measures about -31 dBFS and clears any sane floor — so the two checks now divide the work the
way the write-up above argued they should. Neither is being asked to judge whether sparse-but-real
drumming is worth training on, which is the judgement 0.3 was silently making.

Three reasons to prefer it to 0.3 beyond the listening:

- **0.3 was below the corpus median** (target median 0.346). It rejected more than half the
  corpus on the target stream before the control was even looked at.
- **0.3 sat on the mode** of the distribution (the 0.2-0.3 band holds 153 of 848 items), the
  point of maximum sensitivity: ~130 items cross the line per 0.1 of cutoff there, against ~40
  per 0.1 at 0.85. The threshold is now stable against changes in window length, augmentation
  roll, or corpus.
- **It buys back most of 1.2's statistical power.** The held-out split goes from 56 to an
  expected ~160-165 of 270, so the paired per-item comparison is ≈1.7× tighter than at 0.3 and
  only ≈1.3× noisier than the full 270, against 2.2× at 0.3.

Nothing on disk uses 0.85 yet. Both splits need re-encoding with it, into a root whose **name
states the cutoff** (`…-same-s-sf085/`) — this dataset family has been silently redefined twice
already. Inventory and naming convention: [docs/data/slakh-streamgen.md](../../docs/data/slakh-streamgen.md).

The alternative not taken: encoding at a longer window, where 30% of 380 s is a genuinely dead
track and the original config comments' intuition would have held. That remains the better fix
if the 13.3 s window is ever revisited for other reasons; it was not worth re-encoding for on
its own.

## What this means for 1.2

**The pre-encoded Slakh validation split must be re-encoded before it is used to score
anything.** 20% duplicates in a held-out set is not a small correction: the duplicated
tracks count twice in every metric, and which tracks got duplicated was decided by
`random.randrange`.

The train split should be re-encoded for the same reason, though duplicates there are a
weaker problem (over-sampling, not a corrupted measurement).

Before re-encoding, consider whether `--max_silence_fraction` should be on. The argument for
it: at a long window most Slakh tracks are shorter than it, and a drum stem that plays for
40s of a 380s window is a mostly-empty target that will still clear the RMS floor. The
argument against: it is a second knob that changes what the held-out set contains, so if 1.2
is meant to be comparable to anything already run, it needs to be set once and kept.
Recommendation was: run one pass with the filter off, read `silence_fraction` off the
`levels` blocks, then pick a cutoff and re-encode both splits with it.

**Resolved 2026-09-08** — both splits were re-encoded with `--max_silence_fraction 0.3` on
from the start, the cutoff taken from the BabySlakh distribution rather than from a Slakh
pass with the filter off. See [The first filtered encode of Slakh](#the-first-filtered-encode-of-slakh-2026-09-08)
above: it keeps 56 of 270 validation tracks, so the knob is now the dominant influence on
what the held-out set contains and is the first thing to revisit if 1.2 comes out
underpowered.

**Reopened and re-resolved 2026-09-09** — 0.3 was the BabySlakh number and the accidental
RMS-only train encode showed it sitting *below* Slakh's own target median (0.346) and on the mode
of the distribution, which is why both splits kept only about a fifth of their tracks. Bands were
decoded and listened to, and the cutoff is now **0.85**; see
[Decided: --max_silence_fraction 0.85](#decided---max_silence_fraction-085-2026-09-09).

Still open, and both block reading any 1.2 result:

1. **Neither split is encoded at 0.85 yet.** Re-encode both, same cutoff, into a root whose name
   states it.
2. **The validation encode 1.2 would score on is the Sep 2, duplicate-contaminated one** — or
   rather it no longer exists, having been deleted 2026-09-09 pending re-encode. Train was
   replaced twice, into two different roots; validation never was. The
   `preencoded/` configs are correspondingly a mismatched pair (train RMS-only, validation
   fraction 0.3), so a run today trains on one distribution and scores on a denser subset of
   another.

Dataset inventory, filter settings, naming convention and the config table now live in
[docs/data/slakh-streamgen.md](../../docs/data/slakh-streamgen.md) rather than being scattered
through this write-up.

## Notes

- **Not fixed, deliberately:** the `except Exception` handler in `SampleDataset.__getitem__`
  still resamples on a load failure, so an unreadable file still becomes a duplicate. It is
  the same class of bug but it needs a different fix (there is no audio to hand back), and a
  hard failure mid-way through a 1289-track Slurm job is worse than the current behaviour.
  Load failures do at least print.
- Dropping leaves **gaps in the latent id sequence** (`{batch:06d}{index:04d}`), by design:
  ids stay tied to a file's position in the list, so re-encoding a subset gives the same ids
  it gave the first time. Nothing downstream enumerates ids — `get_latent_filenames` scans
  for `.npy` — so gaps are harmless.
- `_skipped.json` is written next to the latents rather than only logged, because on a
  multi-hour Slurm run the log is where this information otherwise dies. It holds all
  variants in one file: a per-variant name would end in `_v<N>.json`, which is the glob that
  selects that variant's real items.

## Verification commands

```bash
uv run pytest tests/test_preencode_silence_filter.py tests/test_streamgen_metadata.py \
  tests/test_audio_augmentation.py -q

# Encode, then check what was dropped and why
uv run python scripts/pre_encode_dataset.py \
  --dataset_config stable_audio_3/configs/dataset_configs/dataset2preencoding/local_babyslakh_streamgen.json \
  --batch_size 2 --sample_size 587853 --pad
cat <output_path>/_skipped.json

# Read the level distribution before choosing --max_silence_fraction
uv run python -c "
import json, glob, statistics as st
v = [json.load(open(f))['levels']['target']['silence_fraction']
     for f in glob.glob('<output_path>/0*.json')]
print(f'median {st.median(v):.2f}  max {max(v):.2f}  n={len(v)}')"

# Alignment must still be lag 0 after any re-encode
uv run python scripts/check_streamgen_alignment.py --config <preencoded config> --ae_model same-s

# What the 2026-09-08 filtered Slakh encode kept, and why the rest went
SLAKH=/data/hai-res/shared/snnithya/sao-3/data/slakh-streamgen-preencoded-same-s-wo-silence
cat $SLAKH/validation/_skipped.json
cat $SLAKH/train/_skipped.json
ls $SLAKH/validation/*[0-9].npy | wc -l

# Read the level distribution and the survivor sweep off any encode. Point it at the RMS-only
# train encode for numbers available right now (it warns that the pass was filtered, which is
# the point -- the sweep is truncated at -50 dBFS):
uv run python scripts/analyze_silence_levels.py \
  --dir /data/hai-res/shared/snnithya/sao-3/data/slakh-streamgen-preencoded-same-s/train \
  --variant v0

# The staged census. Nothing has been submitted; this is the whole run.
sbatch sbatch/01_2_preencode_slakh_validation_nofilter.sbatch
CENSUS=/data/hai-res/shared/snnithya/sao-3/data/slakh-streamgen-nofilter-census-same-s/validation
uv run python scripts/analyze_silence_levels.py --dir $CENSUS   # the sbatch also does this
ls $CENSUS/[0-9]*.json | wc -l                                  # expect ~213 of 270

# Choose the cutoff by ear: two items per 0.1-wide silence_fraction band, target + control,
# every card labelled with its band and levels. Needs an encode that HAS the upper bands,
# i.e. RMS-only or no-filter -- not one already cut at 0.3.
BANDS=/data/hai-res/shared/snnithya/sao-3/data/_listening/silence_bands_train_v0
uv run python scripts/sample_silence_buckets.py \
  --dir /data/hai-res/shared/snnithya/sao-3/data/slakh-streamgen-preencoded-same-s/train \
  --variant v0 --model same-s -n 2 --controls streamgen_audio --out $BANDS
uv run python scripts/make_listening_page.py --dir $BANDS --embed mp3
# then open $BANDS/index.html, or scp just that one file
```
