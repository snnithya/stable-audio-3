# 3.4 — Drum onset representation as the condition

**Status:** **TRIA control built and unit-tested (2026-10-02)**; per-song features computed (τ = 30 s); chunked train encode running (job 2534276); finetunes T1-fixed / T1-ema prepared, not submitted. The onset/activation representations further down are still planned.
**Depends on:** 3.3 (the drum-RMS finetune is the reference row)

## Decisions 2026-10-02 (Nithya): TRIA features first, at the latent rate

Before any onset transcription, try the rhythm representation of **TRIA** ("The Rhythm In
Anything", O'Reilly, Flores Garcia, Seetharaman, Pardo, ISMIR 2024 LBD,
[pdf](https://oreillyp.github.io/assets/manuscript/474_lbd.pdf)) as the drum control. TRIA
conditions a masked-token drum generator on a deliberately lossy signal at the DAC frame rate
(hop 512, 86 Hz): an 80-bin mel spectrogram summed into **two equal-energy bands**, each
**standardised**, passed through a **sigmoid** and **quantised to 33 levels** so timbre cannot
leak through; at training time the input audio is randomly noised / band-passed / pitch-shifted
/ EQ'd so beatboxing and table-tapping work at test time. It is a two-band, relatively
normalised, coarsely quantised loudness envelope, i.e. a small generalisation of 3.3's
`drums_rms` (one band, absolute dBFS, unquantised).

| Question | Decision | Consequence |
|---|---|---|
| Frame grid | **The latent grid, 10.77 Hz, one value per band per frame** (S = 1). Sub-frame stacking (S = 8 → 16 ch, TRIA's native 86 Hz, 11.6 ms placement) considered and **not run** | 2 channels; the resolution problem below stays open for this control, as for `drums_rms` |
| Normalisation | **Both**: `tria_fixed` (training-set mean/std per band) and `tria_ema` (causal running statistics) | Two arms, T1-fixed and T1-ema; T2/T3 (sub-frame variants) dropped |
| Split frequency | Fixed corpus constant (median equal-energy frequency of the training drum stems) | Measured by `scripts/wjd/tria_feature_stats.py`, stored in `configs/dataset_configs/features/wjd_drums_tria_stats.json` |
| Augmentations | Not in this round | TRIA's robustness augmentations matter for live beatbox/pad input; phase 2 |
| EMA time constant | **30 s** (Nithya, later on 2026-10-02; the measured-stats default was 4 s) | Exponential window: the first frame of a song weighs 37 % after 30 s, 5 % after 90 s, under 1 % after 2.5 min; a frame near the end of a 5-min track is normalised by its last 2–3 min. The prior weight defaults to one τ, so chunks from a song's first ~30 s are normalised mostly against the training-set constants (close to `fixed`); a separate prior weight is a one-field change if a faster hand-over is wanted. 4 s would track the last ~12 s (one chunk, flattens section dynamics); 10 s the last ~30 s (a chorus). |
| Feature source | **Per song, sliced per chunk** (`compute_drum_features.py`, `WJD_DRUM_FEATURES=precomputed`) | With 12 s chunks the EMA still sees the whole song before the chunk; the crossover has no start-up transient at chunk starts. Both match a stateful real-time extractor that starts at the song's start (a `TriaStream` class + parity test is the planned inference piece). |

### Making TRIA causal

Three of TRIA's steps look at the whole clip. The control in `stable_audio_3/data/features.py`
(`tria_control`) replaces each with a causal equivalent and keeps the rest; frame *t* is a
function of samples before `(t+1)·4096` only, the same guarantee as `block_rms_db`.

| Step | In the paper | Here |
|---|---|---|
| Spectrogram → 2 bands | centred STFT windows (look half a window ahead), mel, adaptive split | **causal IIR crossover** (two cascaded 2nd-order Butterworth biquads per band = 4th-order Linkwitz-Riley, low + high is an allpass) at the fixed split, then the same per-frame block RMS as `drums_rms` per band (`band_rms_db`). With one band this *is* `drums_rms`. |
| Equal-energy split | per clip, from the clip's full energy distribution | one constant for the corpus (median over training tracks of the frequency below which half the drum stem's energy lies) |
| Standardisation | per-clip mean/std | `fixed`: training-set mean/std per band, over frames above the −80 dBFS floor (floor frames are separator dropouts/tacets, map to 0 anyway, and would only inflate the std). `ema`: mean/variance over an exponentially weighted window ending at the current frame, τ = 4 s, warm-started with the dataset statistics carrying one τ of weight so the first bars are standardised against the corpus, not against themselves (`ema_standardize`). |
| Sigmoid, 33 levels | per frame | unchanged (`quantize_unit`) |
| Padding | n/a | as for `drums_rms`: control audio zeroed past the target's valid length, and every frame after the last valid one forced to 0 — necessary for `ema`, whose statistics would otherwise adapt to the silence and drift back to 0.5 |

What the two normalisations trade: `fixed` is deterministic and keeps absolute dynamics
(a quiet brushes passage reads quiet), which is what a drum *stem* from the same mix supports.
`ema` is the paper-faithful one: level-invariant after a few seconds, so a tapped or
beatboxed input at any gain lands in the same range, at the cost of a feature that depends
on the preceding ~4 s (a repeated bar is not an identical feature until the statistics have
settled, and the first bar after a long tacet saturates at 1 until they recover).

Latency is unchanged from the RMS control: the feature for latent frame *t* is complete the
moment that frame's audio has arrived.

### What was built (all unit-tested: `tests/test_tria_features.py`, 22 tests; `tests/test_wjd_metadata.py` +5)

- `stable_audio_3/data/features.py` — `TriaStats`, `crossover`, `band_rms_db`,
  `fixed_standardize`, `ema_standardize`, `quantize_unit`, `tria_control`.
- `custom_md_wjd.py` — `WJD_CONTROL_MODE` is now a comma-separated set
  (`rms,tria_fixed,tria_ema,audio`; `both` still means `rms,audio`); the TRIA modes read the
  stats file from `WJD_TRIA_STATS` (default: the committed one). Feature keys
  `drums_tria_fixed`, `drums_tria_ema`, 2 ch each.
- `scripts/wjd/tria_feature_stats.py` — the two passes that produce the stats file and a
  per-track report (`<mirror>/tria_stats/train_report.{json,npz}`).
- `scripts/wjd/compute_drum_features.py` — all three controls (+ per-frame RMS in dB) per
  **whole song**, next to the drum stem; the metadata module slices them per window
  (`WJD_DRUM_FEATURES=precomputed`, default). For `tria_ema` this is the intended semantics:
  running statistics over the song's actual history, not a restart per chunk; for `tria_fixed`
  it removes the crossover's start-up transient in a chunk's first frame.
- `scripts/add_features_to_preencoded.py` — appends feature keys to an existing pre-encoded
  dataset's sidecars **without re-encoding** (which would re-roll the per-item polarity flip,
  3.3 Notes). It recomputes every key the module returns and compares the ones already in the
  sidecar: `drums_rms` has to come back bit-identical before a new key is trusted.
- Configs: `model_configs/small_music_wjd_drums_tria_{fixed,ema}.json` (the rms config with
  `{"id": "drums_tria_*", "dim": 2}`); `preencoded/wjd_stems_*_preencoded.json` now lists all
  three sidecar keys (`controls_dim [1, 2, 2]`; PreEncodedDataset splits by position, the
  model config names what it uses); `sbatch/03_3_finetune_wjd_tria_{fixed,ema}.sbatch` (one script per arm since 2026-10-03, shared body `03_3_finetune_wjd_common.sh`);
  `sbatch/03_3_preencode_wjd.sbatch` defaults to all three features.

### Runs

Same items, same evaluation as 3.3 (rhythmic lock, prompt selectivity, listening grid):

| Run | Control | Channels | Status |
|---|---|---|---|
| 3.3 rms | causal drum RMS, absolute dBFS | 1 | not yet submitted (`03_3_finetune_wjd_rms.sbatch`) |
| T1-fixed | TRIA bands, dataset normalisation | 2 | prepared (`03_3_finetune_wjd_tria_fixed.sbatch`); waiting on the all-controls train encode |
| T1-ema | TRIA bands, EMA normalisation, τ = 30 s | 2 | prepared (`03_3_finetune_wjd_tria_ema.sbatch`); same |

All three train on the chunked encode (12 s windows, 50 % hop, `wjd_stems_train_chunked_preencoded.json`),
so the whole-track sidecars with TRIA keys appended earlier on 2026-10-02 are not used.

One probe specific to this control once an arm is trained: lower the whole drum stem by 12 dB
and re-generate. `fixed` should follow the level (quieter, sparser output); `ema` should not
notice after the first seconds. That is the operational difference between the two arms.

### Measured constants (training split, 136 whole drum stems; 2026-10-02)

First measured on the first 380 s of each track (the whole-track encode's cap: split 198.9 Hz,
means −47.9 / −40.9, stds 14.3 / 10.6), then re-measured on whole tracks once the encode moved
to chunks; the numbers below are the whole-track ones and are what the stats file holds.

`configs/dataset_configs/features/wjd_drums_tria_stats.json`; per-track numbers in
`<mirror>/tria_stats/train_report.json`, per-frame band levels in `train_report.npz`.

| | |
|---|---|
| Equal-energy frequency per track | p5 87 Hz · p25 155 · **median 198** · p75 338 · p95 4672 Hz |
| → `split_hz` | **198.2 Hz** |
| Tracks whose split sits above 1 kHz | 21 of 136 (old recordings with no low end, ride-dominated kits: Lacy, Hawkins, Getz, Desmond …) — for them the low band reads quiet under `fixed`, which is information, not an error |
| Low band (< 198 Hz), frames above floor | mean **−47.7 dBFS**, std **14.3 dB**; 14.3 % of frames at the −80 dB floor |
| High band (> 198 Hz), frames above floor | mean **−40.5 dBFS**, std **10.7 dB**; 8.4 % of frames at the floor |

Floor frames are far more common than in a mixed recording: the separator outputs near-digital
silence between hits, especially in the low band between kicks. Hence mean/std over the
frames *above* the floor (with all frames: −52.3 / 17.4 and −43.9 / 15.0, i.e. the range the
real dynamics get would shrink by a fifth).

Level usage on real tracks (33 levels): a median track uses 26–29 levels per band under
`fixed` and 30–32 under `ema`; no frame saturates at 1 and none sits at 0 outside padding;
`ema` medians are 0.47–0.50 by construction, `fixed` medians 0.28–0.59 depending on the
track's level and spectrum. Frame-to-frame the pattern of the groove is visible in both
(alternating 0.2 / 0.5–0.8 on the low band = kick on the strong beats).

---

## Original 3.4 plan (2026-09-22): onset / activation representations

## Question

Can the drum **latent** be replaced by a **symbolic drum representation** — onsets, or
per-class activations — without losing rhythmic lock (claim 1) or prompt selectivity (claim 2)?
This is the representation an interactive user can actually supply (tap, pad, MIDI, a
transcribed live drummer); the latent condition in 3.3 is a stepping stone that isolates the
data pipeline from the representation question.

## The resolution problem

The control lives at the latent rate: **44100 / 4096 ≈ 10.77 Hz, 92.9 ms per frame.** Swing
eighths at 200 bpm are 150 ms apart and sixteenths 75 ms, so a per-frame binary onset flag
loses the placement *within* a frame — exactly the microtiming that makes jazz drums feel like
jazz drums. Candidate encodings, cheapest first:

| # | Representation | Channels | Timing precision | Code needed |
|---|---|---|---|---|
| R0 | **Click-track rendering**: synthesize the onsets as short percussive clicks (one timbre per class) into audio and pass it through the existing `__audio__` → VAE path | 256 (latent) | sample-exact, carried by the VAE | **none** — a metadata module that renders audio |
| R1 | class-agnostic onset strength, mean-pooled per frame | 1 | frame | small |
| R2 | per-class activation per frame (kick / snare / hi-hat+cymbal) | 3 | frame | small |
| R3 | R2 with **sub-frame bins**: 4 bins per frame per class | 12 | 23 ms | small |
| R4 | R2 + fractional onset position per class | 6 | continuous | small |

R0 is the first thing to run — it costs nothing and answers "does the model need the *timbre*
of the drums or just the *events*". It is also, non-trivially, a plausible production path: a
tapped rhythm rendered to clicks and encoded live. R2/R3 are the "real" symbolic controls.

## Onset extraction

From the separated drum stem (3.2), per segment:

- **Class-agnostic:** `librosa.onset.onset_strength` + peak picking → R0/R1. The WJD beat grid
  is the sanity check: onsets should sit on beats and their subdivisions.
- **Per-class:** an automatic drum transcription model (ADTOF-trained models, or madmom's
  drum-onset networks) → kick / snare / hi-hat → R0 (three timbres) / R2 / R3. Evaluate on a few
  tracks by ear before trusting it on 1950s drum sounds; fall back to class-agnostic if it is
  noisy.

Everything is precomputed per segment and stored next to the stems (`stems/<Track>/drums_onsets.json`),
so 3.3's mirror script can attach either the drum audio or the rendered clicks or the
activations without recomputing.

## Code path for non-audio controls (R1–R4)

Today `pre_encode_dataset.py` only accepts controls under `__audio__` and VAE-encodes them
(`pre_encode_dataset.py:334`). Add a **`__features__`** hook: a `[C, T_frames]` tensor at the
latent rate, cropped/padded to the target's latent length, fused into the same
`{id}_controls.npy` with its own `controls_dim` entry. Then `modular_local_cond_configs` gets
`{"id": "drum_onsets", "dim": C}`. ~50 lines plus a test that the fused sidecar splits back
into the right channels and that a feature control crops in lockstep with the latent
(`PreEncodedDataset.__getitem__` already crops controls with the latent).

Augmentation interaction: `--augment_variants` time-stretches the target and its `__audio__`
controls together; a `__features__` control would have to be re-timed too, or augmentation
disabled for this dataset. Start with augmentation off.

## Experiment

Same items, same held-out drums, same eval as 3.3 (rhythmic lock, prompt selectivity,
listening grid), one finetune per representation, drum-latent result from 3.3 as the
reference:

| Run | Control |
|---|---|
| 3.3 | drum latent (reference) |
| R0-1 | clicks, one timbre |
| R0-3 | clicks, three timbres |
| R2 | 3-ch activations |
| R3 | 12-ch sub-frame activations (only if R2 shows a timing gap) |

Two extra probes that only make sense with a symbolic control:

- **Degrade the control** — drop the hi-hat channel; quantize onsets to the grid; halve the
  density — and check the generation follows. This is what "interactive" means operationally.
- **Human input** — a tapped rhythm from a MIDI pad rendered through the same encoding; a
  listening test, not a metric.

## Deliverables

- `scripts/wjd/extract_drum_onsets.py`, the `__features__` path in `pre_encode_dataset.py` + test
- `custom_md_wjd.py` control-mode flag: `audio | clicks | onsets`
- results table here, with the 3.3 reference row
