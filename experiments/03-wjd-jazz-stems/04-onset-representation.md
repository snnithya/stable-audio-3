# 3.4 — Drum onset representation as the condition

**Status:** planned, after 3.3
**Depends on:** 3.3 (a working drum-latent finetune to compare against)

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
