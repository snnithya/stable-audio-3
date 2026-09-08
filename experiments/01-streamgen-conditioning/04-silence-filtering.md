# 1.5 — Silence filtering at pre-encode time

**Status:** implemented, datasets need re-encoding
**Date:** 2026-09-02

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

## What this means for 1.2

**The pre-encoded Slakh validation split must be re-encoded before it is used to score
anything.** 20% duplicates in a held-out set is not a small correction: the duplicated
tracks count twice in every metric, and which tracks got duplicated was decided by
`random.randrange`.

The train split should be re-encoded for the same reason, though duplicates there are a
weaker problem (over-sampling, not a corrupted measurement).

Before re-encoding, consider whether `--max_silence_fraction` should be on. The argument for
it: with `--sample_size 16760832` (380s) most Slakh tracks are shorter than the window, and a
drum stem that plays for 40s of a 380s window is a mostly-empty target that will still clear
the RMS floor. The argument against: it is a second knob that changes what the held-out set
contains, so if 1.2 is meant to be comparable to anything already run, it needs to be set
once and kept. Recommendation: run one pass with the filter off, read
`silence_fraction` off the `levels` blocks, then pick a cutoff and re-encode both splits with
it.

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
```
