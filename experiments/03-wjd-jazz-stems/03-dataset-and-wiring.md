# 3.3 — Dataset build, wiring, and the drum-latent finetune

**Status:** **mirror built, wiring written and unit-tested** (2026-10-01); whole-track pre-encode run 2026-10-01 (train), superseded by the chunked encode decided 2026-10-02 (see below); finetune not run
**Depends on:** 3.1 (aligned audio), 3.2 (stems — done, 166 tracks, BS-Roformer-SW).

## Decisions 2026-10-01 (Nithya)

Taken before the build; they override the original plan below where the two differ.

| Question | Decision | Consequence |
|---|---|---|
| Items | **Whole tracks**, not JSD segments | One file per target stem per track. JSD segments are written to `meta/` (both clocks) so a segment cut or a segment-aware prompt can be added later without reopening the database. Prompts are track-level. |
| File placement | **Copy** the stems (not link) | 6.6 GB mirror next to the 8.8 GB stems. |
| Targets | **bass, other, piano; guitar only where non-silent**; vocals never | Whole-file RMS floor −50 dBFS from `done.json`; same floor for every target stem. |
| Prompt | **Short instrument names only** | `"upright bass"`, `"piano"`, `"guitar"`, and for `other` the lineup's front line, e.g. `"trumpet, tenor saxophone"` (lineup order). Style / feel / tempo not appended. |
| Splits | **By track**, seeded shuffle, 12 % validation | Not by record as planned. `splits.json` records seed 0. |
| Control | **Causal drum RMS envelope** at the latent rate, 1 channel | The 3.4 "R1" representation, moved up to be the first thing trained; the drum *latent* is kept as an option (`WJD_CONTROL_MODE=audio`) rather than the default. |
| Where | `tap-dance-expt` in the `sa3-tap` checkout, as 3.1/3.2 | |

**What was built** (all on the branch, `tests/test_wjd_metadata.py` covers it, 22 tests):

- `scripts/wjd/make_stem_mirror.py` — the Slakh-mirror analogue: `tracks/<inst>/<Track>/<inst>.flac`, `meta/<Track>.json`, `report.html` per split, `splits.json`, `_skipped.json`.
- `stable_audio_3/configs/dataset_configs/custom_metadata/custom_md_wjd.py` — prompt from `meta/`, drums found from the target path, rejected if silent over the window; returns the RMS feature under `__features__` (or the waveform under `__audio__`).
- `stable_audio_3/data/features.py` — `block_rms_db` / `rms_envelope_control`: per-latent-frame RMS, causal by construction (frame *t* sees nothing after sample `(t+1)·4096`), in [0, 1] from a −80 dBFS floor; padding on the target side forces the control to 0.
- `scripts/pre_encode_dataset.py --features <key>` — the `__features__` hook planned in 3.4: frame-rate tensors fused into the same `{id}_controls.npy` after the `--controls` latents, cropped/padded to the latent length; sidecar JSON gains `control_keys` + `controls_dim`. Refuses to combine with `--augment_variants > 1`. `SampleDataset` unpacks `__features__` like `__audio__` (minus the pad/crop).
- `scripts/pre_encode_dataset.py --num_shards/--shard_index` — one process per GPU, defaulting from the srun step; shards take interleaved global batch indices so ids and files are those of a single-GPU run; per-shard `_skipped.shard<k>of<N>.json` merged into `_skipped.json`. Checked on two real tracks against a single-process run: controls and 3 of 4 latents bit-identical, the 4th differs only by the pre-existing random `PhaseFlipper` (p = 0.5) that `SampleDataset` applies to the target at load time (see Notes).
- Configs `dataset2preencoding/wjd_stems_{train,validation}.json` (four entries, one per target dir → `preencoded/<split>/{bass,other,piano,guitar}/`) and `preencoded/wjd_stems_{train,validation}_preencoded.json` (`controls: ["drums_rms"], controls_dim: [1]`, crop 144, random crop on train). `sbatch/03_3_preencode_wjd.sbatch`.

**Training wiring (added later on 2026-10-01, while the pre-encode ran as job 2516883):**

- `stable_audio_3/training/diffusion.py`: `_add_streamgen_conditioning` now attaches **every** modular local cond the model config names, except the masks the training step builds itself (`control_cond_ids()`); the method keeps its name and `streamgen_latent` behaves as before. The demo callback decodes a control to audio only when its channel count equals the DiT's latent width, so a 1-ch feature control is skipped rather than fed to the autoencoder. `tests/test_control_conditioning.py`.
- `model_configs/small_music_wjd_drums_rms.json` — `small_music_streamgen.json` with `{"id": "drums_rms", "dim": 1}` in place of the 256-ch accompaniment; nothing else changed, so the run is comparable to 1.2.
- `sbatch/03_3_finetune_wjd.sbatch` — task 0 = the rms run (`small-music`, 20k steps, batch 8 × accum 2, lr 1e-5, seed 42, conditioner frozen, causal inpainting task with `future_visibility [-4, 0]` as in 1.2); task 1 = text-only baseline (`small_music_baseline.json`), not run by default.
- Smoke-tested on the real train latents (bass/other/piano written, guitar partial): 4 steps at batch 2 plus one demo round at cfg 4 ran end to end with `drums_rms` in every batch's conditioning (a missing control raises).

**Encode redone as chunks (decided 2026-10-02, Nithya):** the whole-track encode above caps
every track at 380 s (`--sample_size 16760832`, inherited from the whole-track Slakh setting
and equal to the model's maximum length) and samples uniformly per *track* rather than per
second, and its silence gate is per track, not per window. Nithya wants fixed-length
overlapping chunks as in sat-zenon's `ChunkedSampleDataset`, so:

- `pre_encode_dataset.py --chunk_seconds N [--chunk_hop_ratio 0.5]` (new `ChunkedSampleDataset`
  in `stable_audio_3/data/dataset.py`): every file becomes one item per window; the window is
  rounded up to whole latent frames and the hop to whole frames, so chunk starts sit on the
  latent grid and a per-frame feature of a chunk is bit-identical to the same frames of a
  whole-file encode (`tests/test_chunked_preencode.py`). Only the window is read from disk.
  Sidecars carry `chunk_index`, `n_chunks`, `chunk_offset`, `chunk_offset_seconds`,
  `chunk_samples`. `custom_md_wjd.py` and `custom_md_slakh_streamgen.py` read their control
  audio at `chunk_offset`. The silence gate now applies per window. No cap.
- `sbatch/03_3_preencode_wjd.sbatch` defaults to `CHUNK_SECONDS=12` (Nithya's choice; the
  script rounds it up to 130 frames = 12.07 s) with 50 % overlap (hop 65 frames),
  `--pad --batch_size 8`, output `.../wjd/preencoded-chunked/<split>/`; training config
  `preencoded/wjd_stems_<split>_chunked_preencoded.json` (`latent_crop_length 130` = the
  window, so nothing is cropped; it has to be changed together with `CHUNK_SECONDS`).
  `CHUNK_SECONDS=0` reproduces the whole-track encode. The finetune sbatch points at the
  chunked config by default (`DATASET_CONFIG` overrides). Expected size: ~48 windows per
  track at 13 s, so ~21k items / ~3 GB at 12 s; the piano dir alone scanned to 6184 windows.
- The whole-track encode in `.../wjd/preencoded/train/` stays (its sidecars also carry the
  3.4 TRIA controls, appended 2026-10-02 with `scripts/add_features_to_preencoded.py`).
- **Per-song drum features** (`scripts/wjd/compute_drum_features.py`, Nithya's ask
  2026-10-02): `tracks/drums/<Track>/drums_features.npz` holds `drums_rms`, `drums_tria_fixed`,
  `drums_tria_ema` and the per-frame `rms_db` over the whole song; `custom_md_wjd.py`
  (`WJD_DRUM_FEATURES=precomputed`, default) slices frames `[chunk_offset/4096, …)` per item and
  reads the silence gate from `rms_db`, touching no drum audio. `drums_rms` is bit-identical to
  the per-window computation; the TRIA controls are *better* this way: no crossover start-up
  transient in a chunk's first frame, and the EMA statistics carry the song's history into the
  chunk as a live stream would, instead of restarting from the prior per chunk.
  `WJD_DRUM_FEATURES=audio` keeps the per-window computation. The pre-encode sbatch runs the
  script (skipping existing files) before the encode. `tests/test_drum_features_precompute.py`.

**Still to do:** the wiring table below on the real latents (step-0 no-op, gradient reaches the control, alignment by listening), then the finetune and its evaluation. Open knobs: conditioner frozen vs trainable (prompts vary here, unlike Slakh); demos cannot yet play the conditioning drums, since the RMS control is not decodable — a demo-side lookup of `tracks/drums/<track>/drums.flac` at `latent_crop_start` would fix that.

## Question

Can the experiment-01 conditioning stack, with **drums as the control and a non-drum stem as
the target**, be fed from the WJD with the *smallest possible change*, and does the resulting
finetune support claims 1 (drum latent is a usable condition) and 2 (the prompt selects the
stem)?

## Original plan (2026-09-22) — kept for reference; see the decisions table above for what changed

### Design decision: segment-level items, cut before pre-encoding *(superseded: whole tracks)*

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

### Layout: `wjd-stem-mirror` *(as built 2026-10-01)*

```
/data/hai-res/shared/snnithya/sao-3/data/wjd/wjd-stem-mirror/
  splits.json, _skipped.json
  <split>/tracks/drums/<Track>/drums.flac      <- condition   (Slakh's mirror had drums as the target)
  <split>/tracks/bass/<Track>/bass.flac        <- one item each; a dataset entry per instrument dir
  <split>/tracks/other/<Track>/other.flac
  <split>/tracks/piano/<Track>/piano.flac
  <split>/tracks/guitar/<Track>/guitar.flac    (only where the stem is above the floor)
  <split>/meta/<Track>.json                    {prompts per stem, lineup + parsed players, front_line, decade,
                                                solos (melid, instrument, style, feel, tempo, key, solostart_sec),
                                                alignment offset, jsd_segments in track AND file time, stem levels}
  <split>/report.html                          listening page (python -m http.server in the split dir)
```

Per-instrument directories rather than one `targets/` dir, so this is the Slakh layout one to
one and a single instrument can be trained on by pointing at its directory. The planned
segment-level `<Track>__<seg>` items were dropped (whole tracks); the segments live in `meta/`.
Lag checks between target and drums are unnecessary here: every stem comes out of one
separator pass with the mix's exact frame count (asserted in `separate.py`).

### `custom_md_wjd.py` (new, mirrors `custom_md_slakh_streamgen.py`) *(built; prompt rules simplified to the decisions table)*

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

### Splits *(superseded: by track, seed 0, 12 %)*

Not by track: tracks from one session share a band and a room sound. **Split by record
(album), stratified by soloist instrument and decade**; ~12 % of records to validation, and
check every major instrument has validation items. Hold out 6 whole tracks as fixed demo
material. Commit the split as a small JSON; seed recorded.

### Configs and model *(written; control is `drums_rms` dim 1, not `drums_latent` 256 — see decisions)*

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

Unit-level checks done 2026-10-01 (`tests/test_wjd_metadata.py`): prompt per stem and per
lineup; `other` rejected for a trio; drums silent → reject; feature length = latent frame
count, 0 over target padding; causality of the RMS; fused sidecar = `[256-ch latent | 1-ch
RMS]` in `--controls`,`--features` order with a fake autoencoder; features + augmentation
refused. The table below is the data-level pass that still has to run on real latents.

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

## Notes

- **Pre-encoding is not deterministic, by upstream design.** `SampleDataset.__init__` hard-wires
  `self.augs = Sequential(PhaseFlipper())`, a p = 0.5 polarity inversion applied to the *target*
  at load time, and the pre-encode script uses that dataset. So half the stored target latents
  encode `-x` rather than `x`, decided per item per run, and a re-encode reproduces ids and
  controls but not latent values (found 2026-10-01 comparing a single-process run with a sharded
  one: 3 of 4 latents bit-identical, the 4th decoded to the source with correlation −0.99).
  `__audio__` controls are *not* flipped (they only go through pad/crop), so a drum-latent
  control and its target can have opposite polarity; the RMS feature is polarity-blind. The
  Slakh encodes in experiment 01 have the same property. Whether to disable the flipper for
  pre-encoding is an open call — it is a legitimate augmentation, but frozen at one roll per
  item it is just noise in the data rather than an augmentation.
