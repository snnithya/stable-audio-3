# 1.1 — Data + wiring validation

**Status:** complete
**Date:** 2026-08-28

## Question

Does the accompaniment latent reach the model, correctly time-aligned, and does it actually
participate in training — without disturbing the pretrained checkpoint at step 0?

This is plumbing validation, not a musical result. It matters because every failure mode
here is silent: a misaligned control still trains, a dropped control still lowers loss, and
a cond that is never read looks identical to one that is.

## Method

BabySlakh (20 tracks) pre-encoded with `same-l`, drums as target, accompaniment submix as a
fused `_controls.npy` sidecar. Model is `small-music` plus two `modular_local_cond_configs`
(`streamgen_latent` 256, `tf_inpaint_mask` 1).

## Results

| Check | Result |
|---|---|
| Submix is the accompaniment, not the drums | cosine similarity to drums **0.008** |
| Sidecar shape and alignment | 20/20 tracks, `(256, N)` matching each latent's `N`, 0 mismatches |
| JSON metadata not polluted by control waveform | 8.3 KB per file |
| Control reaches the dataset | `info["controls"]["streamgen_latent"]` → `(256, 256)`, cropped in lockstep |
| **Time alignment (decoded vs. source)** | target lag **+0 frames** (corr 0.999); control lag **+0 frames** across 3 samples |
| `future_visibility` seconds → frames | `[-4, 0]s` → `(-43, 0)` frames (44100/4096 ≈ 10.77 fps) ✓ |
| Zero-init MLPs | 40/40 output layers zero (20 blocks × 2 conds) |
| **Step-0 no-op** | output **bit-identical** with and without streamgen (max abs diff 0) |
| **Gradient reaches the cond** | `streamgen_latent` 20/20 blocks (‖g‖ 54.9), `tf_inpaint_mask` 20/20 (‖g‖ 340.2) |
| End-to-end training | 12 steps, loss ≈ 0.14–0.20, demos generated, no errors |
| Test suite | 14 new tests pass; 53 pre-existing pass |

## Notes

Two things nearly went wrong and are worth remembering:

1. **Crop desync.** `PadCrop_Normalized_T` draws a *fresh* random offset on every call, and
   `__audio__` extras are cropped in a separate call from the main audio. With the default
   `random_crop=True` the drums and the accompaniment would have landed on different windows
   of the same track for anything longer than `sample_size` — producing conditioning that is
   plausible, aligned-looking in shape, and musically wrong. Pre-encoding now passes
   `random_crop=False`. `scripts/check_streamgen_alignment.py` exists to catch this class of
   bug after any re-encode.

2. **Metadata pollution.** The control audio arrives in the metadata dict, and the pre-encode
   script serializes every tensor in that dict to JSON. Left alone, each sidecar JSON would
   have contained a multi-minute waveform as a list of floats. The control keys are now
   popped before the dump.

Also noted: BabySlakh source audio is **16 kHz mono**, upsampled to 44.1 kHz stereo by the
loader. Both target and condition get identical treatment so alignment is unaffected, but
the fidelity ceiling is low — worth keeping in mind before reading much into audio quality
from this dataset.

## Verification commands

```bash
uv run pytest tests/test_inpainting_future_visibility.py tests/test_streamgen_metadata.py -q

uv run python scripts/pre_encode_dataset.py \
  --dataset_config stable_audio_3/configs/dataset_configs/dataset2preencoding/local_babyslakh_streamgen.json

uv run python scripts/check_streamgen_alignment.py \
  --config stable_audio_3/configs/dataset_configs/preencoded/local_babyslakh_streamgen_preencoded.json

uv run python scripts/train_finetune.py --model small-music \
  --model_config stable_audio_3/configs/model_configs/small_music_streamgen.json \
  --dataset_config stable_audio_3/configs/dataset_configs/preencoded/local_babyslakh_streamgen_preencoded.json \
  --steps 12 --batch_size 2 --logger none
```
