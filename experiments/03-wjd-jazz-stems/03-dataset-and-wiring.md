# 3.3 — Dataset build, wiring, and the drum-latent finetune

**Status:** **mirror built, wiring written and unit-tested** (2026-10-01); whole-track pre-encode run 2026-10-01 (train), superseded by the **chunked encode, submitted 2026-10-02 (job 2534276, train split)**; finetune not run
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
- `sbatch/03_3_finetune_wjd_<arm>.sbatch` (split into one script per arm on 2026-10-03 so each can run under its own QoS; shared body in `03_3_finetune_wjd_common.sh`; originally one array script) — `rms` = the rms run (`small-music`, 20k steps, batch 8 × accum 2, lr 1e-5, seed 42, conditioner frozen, causal inpainting task with `future_visibility [-4, 0]` as in 1.2); `base` = text-only baseline (`small_music_baseline.json`).
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
- **EMA time constant set to 30 s** (Nithya, 2026-10-02; the stats script's default was 4 s). The
  stats file is the only place it lives; the split frequency and band statistics do not depend
  on it, so `tria_feature_stats.py` was not rerun. Song features for both splits were computed
  with it (136 train + 18 validation `drums_features.npz`, 23:21–23:24). Reasoning and the
  prior-weight caveat in 3.4.

**All four controls in one encode (2026-10-03, Nithya):** the drum-*audio* arm (the original
3.3 plan, drum stem → 256-ch VAE latent as the Slakh `streamgen_latent`) was not covered by the
feature-only encode, and a separate encode for it would put that arm on different latents
(polarity re-roll) and possibly a different item set. So the pre-encode default is now
`WJD_CONTROL_MODE=audio,rms,tria_fixed,tria_ema` (sbatch and module), the sidecar is
`[drums_audio 256 | drums_rms 1 | drums_tria_fixed 2 | drums_tria_ema 2]`, and the chunked
training configs name the first block `streamgen_latent` so `small_music_streamgen.json` and
the streamgen inference / eval scripts work unchanged. Finetune script `03_3_finetune_wjd_audio.sbatch`
(`small_music_streamgen.json`, group `03-3-wjd-drums-audio`). Reminders: the
`dataset2preencoding` JSON's `controls` / `features` keys are read as defaults for the
`--controls` / `--features` flags (CLI wins, and the sbatch always passes the flags), and they
only say which module outputs to *write*; `WJD_CONTROL_MODE` says which the module *computes*,
so the two must agree (the JSONs now carry `controls: [drums_audio]` + the three feature keys,
matching the default). The drum latent is an audio control, not a feature, and
`streamgen_latent` is a training-side name only. The training-side `controls` list must
follow the sidecar order, `--controls` before `--features`. Encodes made before this (jobs 2534276, 2534336, 2534339) have the 5-ch
feature-only sidecar and do not fit the new configs. Known cosmetic issue: the recursive latent
scan also picks up `_sanity_check/*_feature_*.npy` (6 per stem dir); the loader catches the
missing JSON and resamples, so training is unaffected, but `Found N files` is 24 too high.

**Training-job plumbing (2026-10-03):** one sbatch per arm (`sbatch/03_3_finetune_wjd_<arm>.sbatch`,
shared body `03_3_finetune_wjd_common.sh`) so each arm can run under its own QoS. The body
fixes the **effective batch at 128 samples per optimizer step** and derives `--batch_size`
(per GPU under DDP) as 128 / GPUs with no accumulation (`MICRO_BATCH` forces a smaller
per-GPU batch and the rest becomes accumulation). **Preemption:** the partition preempts by
QoS with REQUEUE and GraceTime 0, so the headers set `--requeue --open-mode=append`, the body
derives the wandb run id from the Slurm job id (`wjd-<arm>-<jobid>`, `WANDB_RESUME=allow`),
pins the git ref in `<run dir>/sao_ref` on first start, and passes `--resume_ckpt
<run dir>/checkpoints/last.ckpt` whenever it exists; `train_finetune.py --resume_every 1000`
refreshes that `last.ckpt` independently of the kept 5000-step checkpoints. A requeued job
therefore loses at most 1000 steps and continues the same wandb run. Manual continuation:
`WANDB_RUN_ID=wjd-rms-<jobid> sbatch sbatch/03_3_finetune_wjd_rms.sbatch`. `--lr_schedule
{none,inverse,cosine,exponential}` was added to `train_finetune.py` (default `none`, i.e.
unchanged); the WJD arms use **sat-zenon's schedule** (`saos_streamgen_latent.json`): `InverseLR`, `inv_gamma 1e6`,
`power 0.5`, `warmup 0.995`, i.e. an exponential warmup reaching 50 % of the lr at step 138
and 99 % at step 918, then lr × (1 + t/1e6)^-0.5, which is ×0.95 at 100k steps. Base lr stays
1e-5 (sat-zenon's finetune used 1e-4 with weight decay 1e-3).

**Run log**

| Date | What | Job | Notes |
|---|---|---|---|
| 2026-10-01 | whole-track pre-encode, train, `drums_rms` only | 2516883 | 428 items, 380 s cap; TRIA keys appended 2026-10-02 with `add_features_to_preencoded.py`; superseded |
| 2026-10-02 | song features, train + validation | (login node) | τ = 30 s, training-set stats for both splits |
| 2026-10-02 | **chunked pre-encode, train** (commit `99335ba`) | **2534276** | 12 s → 130-frame windows, 50 % hop, `--pad --batch_size 8`, controls `[drums_rms, drums_tria_fixed, drums_tria_ema]` from the song files, 4 GPUs; output `wjd/preencoded-chunked/train/` |
| | chunked pre-encode, validation | — | `SPLIT=validation sbatch sbatch/03_3_preencode_wjd.sbatch`, not yet run |
| 2026-10-02/03 | chunked pre-encode, train + validation, feature-only | 2534339 / 2534336 | re-runs after the output dir was removed; 5-ch sidecar, superseded by the all-four-controls default below |
| | chunked pre-encode, train + validation, all four controls | — | `sbatch sbatch/03_3_preencode_wjd.sbatch` and `SPLIT=validation sbatch ...` with the new default; same output dirs, so cancel any feature-only run first |
| 2026-10-03 | finetunes rms / tria_ema / tria_fixed, first attempt | 2534778-80 | failed at wandb.init, "No API key configured": the job had no credentials (~/.netrc is on AFS, .env was not sourced). Fixed: `03_3_finetune_wjd_common.sh` sources `$REPO/.env` with `set -a`. |
| 2026-10-03 | tria_fixed on 4 GPUs | 2534818 | hung after ranks 1 and 3 crashed at the first demo: `get_rank()` read `SLURM_PROCID` (0 in every DDP subprocess of a single-task sbatch) ahead of the process group, so all four ranks wrote and deleted the same `demo_cfg_4_*.wav`. Fixed in `training/utils.py` (`tests/test_training_utils_rank.py`). Note: batch_size is per GPU, so N GPUs multiply the effective batch (8 × N × accum 2); keep `batch × gpus × accum` constant across arms for comparability. |
| 2026-10-03 | rms / tria_ema / tria_fixed on 2 / 2 / 4 GPUs, effective batch 256 | 2535099 / 2535098 / 2535097 | **hung in Trainer setup** for 20+ min (every rank's GPU at 100 % util but ~100 W and 0 % memory traffic = NCCL spin-wait; no compile cache written; a 4-rank NCCL probe inside the same allocation passed). Cause: the new `--resume_every` ModelCheckpoint was added only `if checkpoint_dir is not None`, and `checkpoint_dir` comes from the wandb run id that only rank 0 has, so rank 0 had one more ModelCheckpoint than the other ranks; each ModelCheckpoint broadcasts its dirpath in `setup()`, so the ranks' collective sequences diverged and DDP deadlocked before the first step (rank 0 ahead: summary printed; other ranks' model never moved off GPU 0). Fixed: the callback is built on every rank (`build_resume_checkpoint_callback`, `tests/test_finetune_callbacks.py`). Cancelled 02:10. |
| | finetunes (rms / tria_fixed / tria_ema / audio) | — | `sbatch sbatch/03_3_finetune_wjd_<arm>.sbatch`, one per arm (own `--qos` each), once the all-controls encode is done |

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

## Control CFG dropout (2026-10-03)

Listening to the four arms side by side (`scripts/wjd/listen_wjd_arms.py`, pages under
`/data/scratch-fast/snnithya/sao-3/listening/`) raised the question of what the CFG scale acts
on. In this repo the DiT's `cfg_dropout_prob` (0.1) nulls only the cross-attention and prepend
conds, i.e. the prompt; the inpaint conds, `tf_inpaint_mask` and the control (`streamgen_latent`,
`drums_rms`, `drums_tria_*`) were present on every training step. So inference CFG could only
contrast prompt vs no prompt, and "zero the control" was an input the model had never seen.
sat-zenon (`stable_audio_tools/models/dit.py`, Nithya's fork) was different: it dropped each
`input_add` group independently with p = 0.4 in `base-fused-inp-add.json`, which is what made
its two-axis inpaint x streamgen multi-CFG demo work.

Added `control_dropout_prob` (training section of the model config, read by
`scripts/train_finetune.py`, applied in `DiffusionCondTrainingWrapper._add_streamgen_conditioning`):
per item, with that probability, every sidecar control's values are replaced by an
**unconditional token** while `tf_inpaint_mask` is left untouched (Nithya's call, over a first
version that zeroed control and mask together). The token sits on the frames the mask marks
visible; hidden frames stay 0. That keeps three states apart: *unconditional* (token under a
visible mask), *hidden* (0 under a zero mask) and *silent drums* (0 under a visible mask, since
RMS and TRIA map silence to 0 and the latent's zero is a -30 dBFS hiss, not silence). The token
is a constant just above the control's permissible range, `null_value` in the control's
`modular_local_cond_configs` entry so inference can build the same null: 1 + 1/32 = 1.03125 for
`drums_rms` / `drums_tria_*` (one TRIA level above the top of [0, 1]; also the default
`CONTROL_NULL_VALUE_DEFAULT`), and **0 for `streamgen_latent`**, as sat-zenon did (Nithya,
2026-10-03). The latent has no "just above the range": its softnorm values are roughly unit-variance
with |z| up to ~4 on the validation sidecars, so a first draft used a constant 5.0. Zero was chosen
instead to match sat-zenon's null for the same control. The three-state separation still holds
for the latent because its zero is not silence (the autoencoder's silence latent is a different
vector), so 0 under a visible tf mask is already a value no real drum frame produces. The draw is independent of the prompt dropout. Set to 0.1 in the four
controlled arm configs; the baseline has no control and no key. The runs started 2026-10-03
before this change (jobs 2535271/2535278/2535273/2535276 and their requeues) do not have it; a
`FRESH=1` resubmit of each arm picks it up. `train_finetune.py` now also passes the config's
`cfg_dropout_prob` instead of relying on the wrapper default. `mask_loss_weight` in the configs is still
not read (train_finetune.py hardcodes 1.0 for the context-reconstruction term); noted in the config comments.

Also added `inpaint_dropout_prob` (same place in the config and in `train_finetune.py`; applied by
`apply_inpaint_dropout` in the training step): per item, with that probability, the inpainting
context's **values** are nulled, `inpaint_masked_input` to the zero latent, while `inpaint_mask`
is kept (Nithya: "keep the mask, don't null the mask", same shape as the control dropout). The
zero latent is the null here as for `streamgen_latent` and in sat-zenon; it is a value no real
frame produces (the silence latent is not the zero vector), so "context nulled" (0 under a
visible mask) stays distinct from FULL_MASK's "no context" (0 under a zero mask). Because the
mask is unchanged, the loss mask is unchanged: a dropped item is still scored only on the region it
was asked to generate. The tf mask and the control stay, so a dropped item is "continue a stem you
cannot hear, drums still given". The draw is independent of the control dropout and of the prompt
dropout. 0.4 in the four controlled arms (Nithya); `small_music_baseline.json` has 0.1 for prompt and
context, so it is not a matched recipe at the moment. This is the second axis of
a sat-zenon-style multi-CFG (context x control) once the inference side exists.

The inference side is below (2026-10-04). Tests: `tests/test_control_conditioning.py` (dropout section).

## Inference: three-axis CFG and the Gradio interface (2026-10-04)

Nithya asked for a Gradio interface that switches between the arms (rms, tria_fixed, tria_ema,
audio), uses each arm's latest model, and exposes all three CFG scales. Decisions (Nithya,
2026-10-04, from the options offered): **nested composition with a selectable order**, default
prompt -> context -> control; **WJD validation picker plus upload**; **EMA weights, newest
`last.ckpt` auto-discovered** with a reload button; **four arms, lazily loaded and kept resident**
(no base arm).

**Guidance.** `stable_audio_3/inference/multi_cfg.py`. For an order `(a, b, c)` of the three
conditions and scales `s_a, s_b, s_c`:

    v = v(none) + s_a (v(a) - v(none)) + s_b (v(a,b) - v(a)) + s_c (v(a,b,c) - v(a,b))
      = (1 - s_a) v(none) + (s_a - s_b) v(a) + (s_b - s_c) v(a,b) + s_c v(a,b,c)

the InstructPix2Pix form sat-zenon's `make_multicfg_denoiser` used for two axes. The nulls are
the training ones: prompt = zeroed cross-attention/prepend tokens (the DiT's own), context =
zero latent under the kept `inpaint_mask`, control = `null_value` under the kept `tf_inpaint_mask`
(0 on hidden frames). Branches with a zero coefficient are not computed, so all scales 1 is one
forward, and the order **context -> control -> prompt with context = control = 1 is exactly the
standard prompt CFG** the wandb demos and `listen_wjd_arms.py` run (two forwards: with/without
prompt, context and control present in both). The branches run as one batch through the DiT with
its own `cfg_scale` pinned to 1; the callable is passed to `sample_diffusion` as `model` with
`cond_inputs={}`, so the default sampling path is untouched. Vanilla CFG per axis: no APG / rescale
on the composed estimate. An arm without a control (base) collapses the control axis.
`tests/test_multi_cfg.py` (26 tests: nulls, coefficients for every order, batching, dtype).

**Interface.** `run_gradio_wjd.py` -> `stable_audio_3/interface/wjd_control.py`. Arm discovery
and loading moved from `listen_wjd_arms.py` into `stable_audio_3/inference/wjd_arms.py` (the
script imports it). Arms resolve from `sbatch/03_3_finetune_wjd_<arm>.sbatch` + the newest
`last.ckpt` of the group, load on first Generate (DiT bf16, autoencoder fp32) and stay on the
GPU. The loader prefers EMA weights, but **no WJD checkpoint has any**: `03_3_finetune_wjd_train.sh`
never passes `--use_ema` to `train_finetune.py` (default off; the `use_ema: true` in the model
configs' training section is not read by that script), so every `last.ckpt` holds raw weights
only, the wandb demos were made from raw weights too, and the UI's status line says
"raw (the checkpoint holds no EMA)". Resuming the arms with `--use_ema` would start an EMA
from the current weights, not recover one; the autoencoder and T5Gemma are shared between arms (both frozen in every
finetune). "Load / reload latest checkpoint" re-resolves, and the status line says when a newer
checkpoint exists. Inputs: a held-out WJD track (drums + target stem + prompt from `meta/`, the
RMS / TRIA controls sliced from the per-song `drums_features.npz` exactly as training sliced
them) or uploaded drums + optional stem (controls computed from the clip; the TRIA ema then
warm-starts at the clip start, which training never did -- noted in the UI). Window (default 12 s
= 130 frames), start offset (frame-aligned), cursor (context length), control lookahead (training
saw -4..0 s; outside that the UI says it is extrapolating), the three scales, the order, steps /
sampler / seed. Outputs: the stem, the stem mixed with the drums (gain 0.7 as the eval script),
a plot of the three feature controls with the context region, cursor and horizon, the references
(drums, context as given, target), and a notes box listing the branches and coefficients. Wavs
are written under `/data/scratch-fast/snnithya/sao-3/gradio-wjd/` with the settings in the name.

    PYTHONPATH=$PWD .venv/bin/python run_gradio_wjd.py            # then ssh -L 7860:<node>:7860
    PYTHONPATH=$PWD .venv/bin/python run_gradio_wjd.py --arms rms tria_ema --preload --share

Smoke run 2026-10-04 (huang-l40s-2, one L40S), `CannonballAdderley_ThisHere_Orig` piano from 60 s,
12 s window, cursor 6 s, lookahead 0, 8 pingpong steps, seed 0; the picker's npz controls matched
the from-audio ones bit for bit for `drums_rms` and `drums_tria_fixed` (TRIA ema differs by
design: running statistics from the song start vs the clip start, max |diff| 0.41).

| arm | last.ckpt step | load | GPU after load | 1 / 2 branches per step |
|---|---|---|---|---|
| rms | 6794 | 12 s | 1.3 GiB | 1.1 s (first call) / 0.5 s |
| audio | 13702 | 9 s | 2.8 GiB (two arms resident) | 0.5 s / 0.5 s |
| tria_fixed | 9446 | ~10 s | 4.2 GiB peak (three resident) | 0.5 s / 0.5 s |
| tria_ema | 3831 | 10 s | 4.7 GiB (all four resident), 5.1 GiB peak | 0.5 s / 0.5 s |

Settings exercised per arm: all scales 1 (one forward); context -> control -> prompt with prompt 4
(two forwards, the standard CFG); prompt -> context -> control with control 3 (two forwards). Also
the no-stem path (cursor 0, lookahead +6 s). Generated regions sat at -37 to -44 dBFS. The Gradio
app built (52 components, 7 endpoints) and served; listening still to do. `tests/test_multi_cfg.py`
and `tests/test_control_conditioning.py` pass on the node (39 tests). Two fixes from first use (Nithya, 16:00): generated wavs live outside Gradio's cwd/temp dirs, so the launcher passes `--out_dir` as `allowed_paths`; and a click while the job was rewriting `last.ckpt` (3.4 GB, ~10 s, in place) hit a truncated zip, so discovery now returns the newest checkpoint of the run whose zip central directory opens (`is_complete_ckpt`: `last-v1.ckpt` or the newest step file meanwhile) and the loader retries a few times if the file is rewritten under it (`tests/test_wjd_arms.py`).

**SDEdit (2026-10-05).** Nithya asked for an SDEdit checkbox: noise the streamgen input by a slider
amount and denoise from there. The "SDEdit" accordion in the UI (off by default) starts sampling
from the **drum latent** noised to level sigma, `x = (1 - sigma) z_drums + sigma noise`, and runs
the chosen steps over [sigma, 0] under the same conditioning and CFG. It goes through
`sample_diffusion`'s existing `init_data` / `init_noise_level` (`generate_continuation(...,
sdedit_noise_level=...)`), so the default path is unchanged. Slider 0.01..1, default 0.7; `_sde<sigma>` is
appended to the wav name. `tests/test_wjd_sdedit.py` (3 tests). Smoke run on an L40S (rms arm,
old ARC-trained `wjd-rms-2543312` step 15772, which is still on disk because the `-smbase` groups have no
checkpoint yet; CannonballAdderley_ThisHere_Orig piano at 60 s, cursor 6 s, 8 pingpong steps, all
scales 1). Envelope correlation of the generated region with the drums:
sigma 1.0 / 0.9 / 0.7 / 0.5 / 0.3 / 0.1 -> -0.26 / -0.34 / -0.23 / -0.21 / +0.71 / +0.90 (no
SDEdit: -0.35). So the drums only start to show through below about 0.5. With euler, sigma = 1 matches no-SDEdit
to GPU noise. Separately, **pingpong is not reproducible from the seed**: its per-step
re-noising uses the global RNG (`torch.randn_like`), and only the initial noise comes from the
seeded generator, so two runs with the same seed differ.

**SDEdit from the stem (2026-10-06).** Nithya also wanted SDEdit from the stem being continued. A
"SDEdit from" radio (`drums` / `stem`, `generate_continuation(..., sdedit_source=...)`) now picks
which latent gets noised. `stem` uses the whole window, so the target after the cursor leaks into the
starting point; that is the point of the edit. Wav names now carry `_sde-<source><sigma>`;
`tests/test_wjd_sdedit.py` has 4 tests. Same smoke setup as above with euler. Envelope correlation of the generated
region with the target stem: sigma 0.9 / 0.7 / 0.5 / 0.3 / 0.1 -> +0.37 / +0.25 / +0.51 / +0.97 /
+0.99 (no SDEdit: +0.27). As with drums, the source dominates below about 0.3 to 0.5. On a fresh node
the T5Gemma download fails with a 401 unless `HF_HOME=/data/hai-res/snnithya/.cache/huggingface`
(the sbatch value) is set.
