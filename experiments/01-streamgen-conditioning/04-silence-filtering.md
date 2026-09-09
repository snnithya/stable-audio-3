# 1.5 — Silence filtering at pre-encode time

**Status:** implemented; Slakh re-encoded with the filter on — validation done, train running
**Date:** 2026-09-02 (filtered Slakh encode 2026-09-08)

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

### Train split

Job 1766290, 4×L40S, 1289 files × 2 variants, launched 2026-09-08T17:49 and **still running**
at the time of writing (job 1766197 was the same submission with the flag misspelled as
`--silence_fraction`; it died in argparse in 77 s, which is the cheap failure mode this
script's flag surface is meant to give). No per-variant counts yet — read them from
`train/_skipped.json` when it lands. If the validation rate carries over, expect on the order
of 250-300 items per variant out of 1289.

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
cat $SLAKH/train/_skipped.json      # written when the train job finishes
ls $SLAKH/validation/*[0-9].npy | wc -l
```
