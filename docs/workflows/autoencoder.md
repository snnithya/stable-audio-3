# Autoencoder [![arXiv](https://img.shields.io/badge/arXiv-2605.18613-b31b1b.svg)](https://arxiv.org/abs/2605.18613)

Stable Audio 3 uses a 44.1k stereo audio autoencoder known as SAME to compress waveforms into a compact continuous latent representation that the diffusion model operates on. This page covers how to use the autoencoder directly, for encoding individual audio files, decoding latents back to audio, and pre-encoding a dataset for training.

## Encoding audio to latents

```python
import torchaudio
from stable_audio_3 import AutoencoderModel

ae = AutoencoderModel.from_pretrained("same-l")  # "same-s" (small), "same-l" (medium/large)
waveform, sr = torchaudio.load("audio.wav")
latents = ae.encode(waveform, sr)
# → (1, latent_dim, latent_time)
```

Resampling, channel conversion, and padding are handled automatically. The latent time dimension is `samples // downsampling_ratio` (4096 for all current models). At 44.1 kHz, 10 seconds of stereo audio produces 216 latent frames, with a latent dimension of 256.

To encode a batch of clips with different lengths in one call:

```python
latents = ae.encode([waveform_a, waveform_b], sr=[44100, 22050])
# → (2, latent_dim, latent_time)
```

## Decoding latents to audio

```python
import torchaudio
from stable_audio_3 import AutoencoderModel

ae = AutoencoderModel.from_pretrained("same-l")
audio_out = ae.decode(latents)
# → (1, 2, samples)

torchaudio.save("reconstructed.wav", audio_out[0].cpu(), ae.sample_rate)
```

## Chunked processing for long audio

For audio that is too long to encode or decode in a single forward pass, pass `chunked=True`. `chunk_size` and `overlap` are both measured in latent frames (not audio samples).

```python
import torchaudio
from stable_audio_3 import AutoencoderModel

ae = AutoencoderModel.from_pretrained("same-l")
waveform, sr = torchaudio.load("audio.wav")

latents = ae.encode(waveform, sr, chunked=True, chunk_size=128, overlap=32)
audio_out = ae.decode(latents, chunked=True, chunk_size=128, overlap=32)
```

The overlap should be at least as large as the model's receptive field. A value of 32 is a reasonable default.

## Saving and loading latents

```python
import numpy as np
import torch
from stable_audio_3 import AutoencoderModel

ae = AutoencoderModel.from_pretrained("same-l")

# Save
np.save("latents.npy", latents[0].cpu().numpy())  # (latent_dim, latent_time)

# Load and decode
latent_tensor = torch.from_numpy(np.load("latents.npy")).unsqueeze(0).to(ae.device)
audio_out = ae.decode(latent_tensor)
```

## Pre-encoding a dataset

For LoRA training, if you have a large dataset, it is much faster to pre-encode your dataset once and train from the saved latents. Use the provided script:

```bash
uv run python scripts/pre_encode_dataset.py \
  --model same-s \
  --data_dir ./my_data \
  --output_path ./latents_out \
  --batch_size 1
```

The script expects audio files paired with `.txt` caption files:

```
my_data/
  clip1.wav
  clip1.txt
  clip2.wav
  clip2.txt
```

Each encoded clip is written as a `.npy` latent and a `.json` metadata file. When `--pad` is used, the metadata includes a padding mask tracking the valid audio region:

```
latents_out/
  000000000000.npy
  000000000000.json
  000000000001.npy
  000000000001.json
```

Pass the output directory to `train_lora.py` via `--encoded_dir`. See [LoRA training](lora.md) for the full training workflow.

### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `same-l` | Autoencoder variant: `same-s` (small), `same-l` (medium/large) |
| `--data_dir` | — | Folder containing audio + `.txt` pairs |
| `--output_path` | — | Where to write `.npy`/`.json` latent pairs |
| `--batch_size` | `1` | Must be `1` for variable-length latents |
| `--sample_size` | `12582912` | Samples to pad/crop to (default ~380s at 44.1kHz)|
| `--model_half` | off | Run the autoencoder in fp16 to reduce memory |
| `--pad` | off | Pad/crop audio to `--sample_size` (required for `--batch_size > 1`) |
| `--sanity_check_samples` | `0` | Decode this many encoded items back to audio for a listening check |
| `--sanity_check_dir` | `<output_path>/_sanity_check` | Where those wavs are written |

## Sanity-checking a pre-encoded dataset

Encoding is easy to get subtly wrong — a resample that never happened, a control
sidecar cropped out of sync with its target, a caption attached to the wrong clip —
and none of it shows up until a training run has already burned GPU hours. The
cheapest check is to listen.

### During encoding

Pass `--sanity_check_samples N` to decode the first N items straight back to audio
as they are encoded:

```bash
uv run python scripts/pre_encode_dataset.py \
  --dataset_config stable_audio_3/configs/dataset_configs/dataset2preencoding/my_dataset.json \
  --sanity_check_samples 3
```

Wavs land in `<output_path>/_sanity_check/` (override with `--sanity_check_dir`):

```
_sanity_check/
  0000000000_source.wav                        ← exactly what went into the encoder
  0000000000_decoded.wav                       ← the saved latent, decoded back
  0000000000_control_streamgen_audio_source.wav
  0000000000_control_streamgen_audio_decoded.wav
```

The `_source.wav` is the tensor the encoder actually saw — post resample, pad/crop and
channel conversion — not a re-read of the original file, so a mismatch between the two
is a real encoding problem rather than a loading artifact. Each pair is trimmed to the
length the stored latent covers, so without `--pad` you hear the truncated region only.

### After encoding

`scripts/decode_preencoded_samples.py` does the same thing for latents already on disk,
and it takes either config in the workflow — the `dataset2preencoding/` one you encoded
with (it reads latents from that config's `output_path`, and picks up `model` and
`controls` from it), the `preencoded/` training config, or a bare latents directory:

```bash
uv run python scripts/decode_preencoded_samples.py \
  --config stable_audio_3/configs/dataset_configs/dataset2preencoding/my_dataset.json \
  --out /tmp/preencode_check -n 5
```

Here `_source.wav` is the matching slice of the original audio file, offset by the crop
the training dataset applied, which makes this the version to use when you want to check
crop alignment as the model will see it. Add `--full` to decode the whole stored latent
instead of a training-length crop.

### Listening to them

`scripts/make_listening_page.py` turns a directory of those wavs into a local HTML page
with a player and a mel spectrogram per file:

```bash
uv run python scripts/make_listening_page.py --dir ./latents_out/_sanity_check
```

It writes `index.html` and `_mels/*.png` into that directory, referencing the wavs in
place — open it directly, or `python -m http.server -d <dir> 8000` when working over SSH.
Each sample becomes one card with its source stacked above its reconstruction (controls
below), all spectrograms plotted on a single page-wide dB range so level and bandwidth
differences are visible rather than normalized away. Click a spectrogram to seek and
play from that point; only one clip sounds at a time.

For a numerical rather than aural check of control alignment, use
`scripts/check_streamgen_alignment.py`, which cross-correlates a decoded target and its
control against the source audio and fails if either peak lands off lag 0.
