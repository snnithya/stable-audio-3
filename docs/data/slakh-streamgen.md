# Slakh2100 streamgen — pre-encoded datasets

Canonical reference for the drums-plus-accompaniment latent datasets used by experiment 1.2.
What is on disk, which filter produced it, and which config points at it. Written because the
same root name has meant three different things across three encodes; if you re-encode, update
the inventory here in the same commit.

Experiment write-up: [experiments/01-streamgen-conditioning/04-silence-filtering.md](../../experiments/01-streamgen-conditioning/04-silence-filtering.md).

## Source

```
/data/hai-res/shared/snnithya/sat-zenon-data/slakh2100/streamgen-drum-mirror/
  {train,validation}/tracks/
    drums/Track00001/Drums.flac      <- the target, what the metadata fn is called on
    other/Track00001/Guitar.flac     <- accompaniment stems, submixed into the control
    other/Track00001/Piano.flac
```

1289 train tracks, 270 validation, official Slakh splits. The `other/` stems are mixed into the
`streamgen_audio` control by
`stable_audio_3/configs/dataset_configs/custom_metadata/custom_md_slakh_streamgen.py`: a random
subset of the non-silent stems, each LUFS-normalized to a random level in (-30, -15), summed and
peak-limited to 0.95.

**The submix is re-rolled on every pass.** Pre-encoding freezes one roll per item per pass, so
two encodes of the same split are not bit-identical on the control side and the same item can
land either side of a control-side threshold (this is the ±2-item difference between the
filtered train run and a replay of the census). Target-side numbers reproduce exactly.

## Window

`--sample_size 587853 --pad --batch_size 8` → **13.3 s, 144 latent frames, from sample 0 of
every track.** The pre-encode path sets `random_crop=False`, so it is the first 13.3 s, not a
sampled window — deliberately, because `PadCrop_Normalized_T` draws a fresh offset per call and
the control is cropped in a separate call, so a random offset would desync the two streams.

Consequences worth holding onto:

- Intros dominate the silence statistics. A drum stem that enters at 0:20 is 100% silent in the
  encoded window however busy the track is downstream.
- A silence *fraction* means something completely different at this length than at the 380 s
  window the config comments were originally written for. 30% of 13.3 s is 4 s.
- `--sample_size` must be passed **on the CLI, never as a config key**:
  `merge_config_into_args` treats any argparse default that is not `None` as CLI-supplied, so a
  `sample_size` key in the JSON is silently ignored.

The whole-track alternative is `--sample_size 16760832` (4092 frames, 380 s, the 95th percentile
track length) with `--batch_size 1` and no `--pad`.

## Filtering

Three gates, in the order they fire. Each item is recorded under its **first** failure only, so
the buckets in `_skipped.json` are a partition, not overlapping counts.

| Gate | Where | Test | Recorded? |
|---|---|---|---|
| Peak | `is_silence`, `data/dataset.py:282` | loudest sample in the crop below **-60 dBFS** | dropped before the metadata fn runs; **no sidecar** |
| Accompaniment | `custom_md_slakh_streamgen.py:209` | every non-drum stem silent (RMS < -50 dBFS) after cropping to the target's valid region | **no sidecar** |
| Level | `pre_encode_dataset.py:363` | per stream, RMS floor then silence fraction, over the valid region, after augmentation | `levels` block in the sidecar of every **surviving** item |

The level gate applies to the target **and** every control: a real drum latent paired with an
empty accompaniment is a training pair whose lesson is that the control carries no information.

### Settings

```
--silence_threshold_db      -50    RMS floor, per stream, over the valid region
--silence_frame_threshold_db -60   peak below which a 20ms frame counts as silent
--max_silence_fraction      0.85   <- chosen 2026-09-09, see below
```

`--silence_threshold_db -50` catches the realistic empty window — a dead stem, a noise floor, a
track that has not started. It cannot catch a window that is 99.9% silence with one full-scale
hit in it, which averages about -31 dBFS RMS and clears the floor comfortably. That blind spot
is the entire reason `--max_silence_fraction` exists.

### `--max_silence_fraction 0.85`

**Chosen 2026-09-09 by listening**, from `scripts/sample_silence_buckets.py` output over the
RMS-only train encode — two decoded items per 0.1-wide band, target and control, each card
labelled with its measured levels. The survivor curve has no plateau (2% of the corpus at 0.1
rising smoothly to 66% at 1.0), so there was nothing in the data to pick the number out; it had
to be answered by ear.

At 0.85 the fraction limit is a **backstop, not the dominant knob**. It keeps 784 of the 848
RMS-surviving train items — 92% — and what it removes is windows with roughly under 2 s of
content in 13.3 s, which is precisely the case the RMS floor is blind to. That division of
labour is the intended one: RMS catches dead streams, the fraction limit catches one hit in an
empty window, and neither is asked to make a musical judgement about sparse-but-real drumming.

The 0.3 that the 2026-09-08 encodes used was read off 17 BabySlakh tracks and was, on Slakh,
*below the corpus median* (0.346) and sitting on the mode of the distribution (0.2-0.3) — the
point of maximum sensitivity, where ~130 items per 0.1 of cutoff cross the line. 0.85 is out on
the tail where the derivative is ~40 per 0.1, so the number is stable against changes in window
length, augmentation roll, or corpus.

Train v0 census (848 items with `levels`, RMS floor already applied):

| cutoff | 0.75 | 0.80 | **0.85** | 0.90 | 0.95 | off |
|---|---|---|---|---|---|---|
| items kept (both streams) | 704 | 757 | **784** | 807 | 834 | 848 |
| % of the 1289 corpus | 55% | 59% | **61%** | 63% | 65% | 66% |

## Inventory

| Root | Filter | train v0 / v1 | validation | Encoded |
|---|---|---|---|---|
| `slakh-streamgen-preencoded-same-s` | RMS -50 only | 848 / 830 | *deleted 2026-09-09, re-encode pending* | train job 1776023, 2026-09-08 |
| `slakh-streamgen-preencoded-same-s-wo-silence` | RMS -50 + fraction **0.3** | 252 / 240 | 56 | jobs 1766290 / 1766413, 2026-09-08 |
| `slakh-streamgen-nofilter-census-same-s` | none (level gate off) | — | staged, not submitted | — |

All under `/data/hai-res/shared/snnithya/sao-3/data/`, all `same-s`, all the 13.3 s window,
train written with 2 augmentation variants and validation with 1.

Nothing on disk uses the 0.85 cutoff yet. Expect **~784 / ~780 train** per variant and
**~160-165 validation** (270 − 55 peak − 36 target RMS − 2 accompaniment = 177 RMS-survivors,
of which ~92% clear 0.85 if train's ratio carries over).

Two names to distrust. `-wo-silence` does not mean "silence removed" as against the other root
removing none — both apply the RMS floor, and the only difference is the fraction limit. And
`slakh-streamgen-preencoded-same-s/train` was *overwritten* on 2026-09-08 by an RMS-only encode,
so its Sep 2 contents — the duplicate-contaminated set described in the experiment write-up — no
longer exist.

**Naming convention going forward: put the cutoff in the path**, e.g.
`slakh-streamgen-preencoded-same-s-sf085/`. A root whose name does not state its filter cannot
be audited from a config file, and this dataset family has now been silently redefined twice.

## Configs

| Config | Points at |
|---|---|
| `dataset2preencoding/slakh_streamgen_train.json` | writes `…-same-s/train` (**no** fraction limit in `sbatch/01_2_preencode_slakh.sbatch`) |
| `dataset2preencoding/slakh_streamgen_validation.json` | writes `…-same-s/validation` (**no** fraction limit in `sbatch/01_2_preencode_slakh_validation.sbatch`) |
| `dataset2preencoding/slakh_streamgen_validation_nofilter.json` | writes `…-nofilter-census-same-s/validation`, diagnostic only |
| `preencoded/slakh_streamgen_train_preencoded.json` | reads `…-same-s/train` |
| `preencoded/slakh_streamgen_validation_preencoded.json` | reads `…-same-s-wo-silence/validation` |

⚠️ **The two `preencoded/` configs are a mismatched pair**: train reads the RMS-only encode,
validation reads the 0.3-filtered one. A run today trains on one distribution and scores on a
strictly denser subset of another. Point both at the same filter before reading any result.

`latent_crop_length` must be **144** in both, matching the stored latent length — it was 256 and
1024, which against 144-frame latents silence-pads every item instead of cropping it.
`scripts/eval_streamgen.py --eval_frames` still defaults to 256, which against a 144-frame item
silently returns the whole item; pass `--eval_frames 144`.

## Re-encoding

```bash
# Both splits, same cutoff, or the two are not comparable.
sbatch sbatch/01_2_preencode_slakh.sbatch                  # train, 2 variants
sbatch sbatch/01_2_preencode_slakh_validation.sbatch       # validation, 1 variant

# What went and why, per variant, next to the latents.
cat <root>/{train,validation}/_skipped.json

# Level distribution and survivor sweep for any encode.
uv run python scripts/analyze_silence_levels.py --dir <root>/train --variant v0

# Two decoded items per silence band, for a listening pass. Needs an encode that HAS the
# upper bands -- RMS-only or unfiltered, not one already cut.
uv run python scripts/sample_silence_buckets.py --dir <root>/train --variant v0 \
    --model same-s -n 2 --controls streamgen_audio --out <wavdir>
uv run python scripts/make_listening_page.py --dir <wavdir> --embed mp3

# Alignment must still be lag 0 after any re-encode.
uv run python scripts/check_streamgen_alignment.py --config <preencoded config> --ae_model same-s
```

## Gotchas

- **Dropped items leave gaps in the latent ids** (`{batch:06d}{index:04d}`), by design: an id
  stays tied to a file's position in the list, so re-encoding a subset reproduces the same ids.
  Nothing enumerates ids — `get_latent_filenames` globs `.npy` — so gaps are harmless.
- **`_skipped.json` holds all variants in one file.** A per-variant name would end in
  `_v<N>.json`, which is the glob that selects that variant's real items.
- **`levels` are only on disk for survivors.** Levels are computed for every item that reaches
  the level gate, but the sidecar is written after the filter
  (`pre_encode_dataset.py:478`), so a filtered pass yields a distribution truncated at its own
  thresholds and cannot be used to choose them. Use an RMS-only or `--no_silence_filter` pass.
- **An encode is not a superset of a stricter one on the control side**, because the submix is
  re-rolled per pass. Do not diff two roots item-by-item and expect the control to match.
- **`except Exception` in `SampleDataset.__getitem__` still resamples on a load failure**, so an
  unreadable file still becomes a duplicate of a random other track. Load failures print.
