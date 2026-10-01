# 3.3 — Dataset build, wiring, and the drum-latent finetune

**Status:** planned
**Depends on:** 3.1 (aligned audio), 3.2 (stems). The metadata module and configs can be written and unit-tested on a hand-made fixture before either lands.

## Question

Can the experiment-01 conditioning stack, with **drums as the control and a non-drum stem as
the target**, be fed from the WJD with the *smallest possible change*, and does the resulting
finetune support claims 1 (drum latent is a usable condition) and 2 (the prompt selects the
stem)?

## Design decision: segment-level items, cut before pre-encoding

`pre_encode_dataset.py` reads every file from sample 0 with a fixed `--sample_size` and
`random_crop=False`, and every `__audio__` control is cut identically — that discipline is
what keeps target and control aligned (see 1.1's crop-desync note). WJD solos start
mid-track, so instead of teaching the pre-encoder about offsets, **cut the audio into JSD
segments first** and let each segment be a file. Zero changes to the pre-encode script, and
every item is a musically coherent unit: one soloist, one section.

Segment policy: merge consecutive choruses of the same label (`solo_01_01…04` → `solo_01`) so
items are typically 30–300 s; drop segments < 15 s and `silence`/`intro`/`outro`. Encode
whole items with the existing whole-track setting (`--sample_size 16760832`, `--batch_size 1`,
no `--pad`; 380 s cap, padding mask covers the rest) and crop to `latent_crop_length 144`
with `random_crop: true` at train time, so a 90 s solo yields a different 13.3 s window every
epoch instead of its first 13.3 s.

## Layout: `wjd-stem-mirror`

```
/data/hai-res/shared/snnithya/sao-3/data/wjd/wjd-stem-mirror/<split>/tracks/
  drums/<Track>__<seg>/drums.flac              <- condition   (Slakh's mirror had drums as the target)
  targets/<Track>__<seg>/bass.flac             <- one item each
  targets/<Track>__<seg>/other.flac
  targets/<Track>__<seg>/piano.flac            (if 3.2 keeps 6s stems)
  meta/<Track>__<seg>.json                     {track, label, start, end (track time), soloists, backing,
                                                melids overlapping, style, rhythmfeel, avgtempo, key, decade}
```

The dataset `path` points at `targets/`. `scripts/wjd/make_stem_mirror.py` builds the tree from
`stems/` + `alignment.json` + JSD + `wjazzd.db`, writes the split lists, and runs
`check_streamgen_alignment.py`-style lag checks between each target and its drums.

## `custom_md_wjd.py` (new, mirrors `custom_md_slakh_streamgen.py`)

For a target file, read its `meta/` JSON and return:

- **`prompt`** from stem name × segment instrumentation. Abbreviation map: ts → "tenor
  saxophone", as → "alto saxophone", ss → "soprano saxophone", bs → "baritone saxophone",
  tp → "trumpet", tb → "trombone", cor → "cornet", cl → "clarinet", bcl → "bass clarinet",
  vib → "vibraphone", p → "piano", g → "guitar", b → "upright bass". `other` in a solo segment
  → `"<soloist> solo"`; `other` in a theme → `"<front line instruments>, ensemble"`; `bass` →
  `"upright bass, walking"` (or `"bass solo"` when `s_b`). Optional richer template appends
  `style, rhythmfeel, round(avgtempo) bpm` from `solo_info`; whether that helps is a
  sub-question, so make it a module-level flag, not two datasets.
- **`__audio__: {"drums_audio": <drums stem>}`** cropped to the target's valid length, exactly
  as `load_and_mix_stems` does today (no submixing — a single stem).
- **Rejections** (`__reject__` + reason, so they land in `_skipped.json`): drums file missing or
  silent (20 drumless tracks, drum-tacet passages); target silent (RMS gate); `other` when the
  segment lists no melodic non-rhythm-section instrument; any track whose `alignment.json` is
  not `pass`.
- Passthrough metadata for eval and 3.5: `track`, `seg`, `soloist`, `melids`, `style`, `decade`.

**Wiring detail to verify early:** the pre-encoded-stage dataset re-runs a
`custom_metadata_module` (Slakh's returns the constant `"drums"`). For WJD the prompt differs
per item, so the pre-encoded-stage module must **read the prompt back from the sidecar JSON**,
not recompute it from a path. Check what `PreEncodedDataset` exposes to the metadata fn before
assuming.

## Splits

Not by track: tracks from one session share a band and a room sound. **Split by record
(album), stratified by soloist instrument and decade**; ~12 % of records to validation, and
check every major instrument has validation items. Hold out 6 whole tracks as fixed demo
material. Commit the split as a small JSON; seed recorded.

## Configs and model

- `dataset2preencoding/wjd_stems_{train,validation}.json` — `controls: ["drums_audio"]`,
  `sanity_check_samples`, level gates as in Slakh (`--silence_threshold_db -50`; revisit
  `max_silence_fraction` — bass and horns rest more than drums do).
- `preencoded/wjd_stems_{train,validation}_preencoded.json` — `controls: ["drums_latent"]`,
  `controls_dim: [256]`, `latent_crop_length: 144`, `random_crop: true`.
- `model_configs/small_music_wjd_drums.json` — copy of `small_music_streamgen.json` with
  `streamgen_latent` → `drums_latent`. `tf_inpaint_mask` + `CAUSAL_MASK` / `future_visibility`
  stay; the lookahead question from 1.3 applies unchanged.
- Scripts that hardcode `streamgen_latent` (`check_streamgen_alignment.py`, `eval_streamgen.py`,
  the training wrapper's mask gating) need the control id to be a parameter. Grep before assuming
  a rename is free.

## Wiring validation (same table as 1.1, before any GPU-hours)

| Check | Pass condition |
|---|---|
| target ≠ condition | cosine(target, drums) low; drums ≈ drums stem |
| sidecar shape / alignment | `(256, N)` matching each latent; lag **0 frames** on decoded pairs, 5 items across decades |
| prompt survives pre-encode | sidecar JSON `prompt` equals the module's output; distinct across stems of one segment |
| step-0 no-op | bit-identical output with and without the control |
| gradient reaches the control | all 20 blocks |
| 12-step smoke finetune | runs, demos written |

## The finetune (claims 1 and 2)

Matched pair as in 1.2: `small-music` + `drums_latent` vs. `small-music` text-only on the same
items; same steps, batch, LR, seed; `cfg_dropout_prob 0.1` on the prompt (default) so the
prompt is CFG-guidable.

Evaluation, on held-out drums segments:

1. **Rhythmic lock** — onset times of the generated stem vs. the WJD beat grid of the
   conditioning segment (the same proxy 3.2 uses on real stems, so real-stem numbers are the
   reference point). Text-only baseline should be near chance.
2. **Prompt selectivity** — fixed drums, prompts {tenor saxophone, upright bass, piano}:
   instrument classification of outputs (CLAP zero-shot first; a small classifier on our own
   stems if CLAP is unreliable on separated jazz) → confusion matrix; plus a
   "same-drums-different-prompt" distance to show the outputs are not each other.
3. **Listening grid** — 6 demo drums × 3 prompts, `make_listening_page.py`.

## Storage / compute

Stems already counted in 3.2. Latents: 256 ch × 10.77 Hz × fp16 ≈ 5.5 kB/s → 33 h × ~3 targets
+ drum sidecar ≈ 3 GB. Pre-encode ≈ 1 GPU-hour. Finetune as 1.2.

## Deliverables

- `scripts/wjd/make_stem_mirror.py`, `custom_metadata/custom_md_wjd.py`, the four dataset
  configs, the model config, `sbatch/03_3_*.sbatch`
- `tests/test_wjd_metadata.py` (prompt mapping, rejections, control attachment) on a tiny fixture
- wiring table and finetune results here
