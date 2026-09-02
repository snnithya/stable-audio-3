"""Pitch-shift and time-stretch augmentation for time-aligned audio streams.

Used at pre-encode time (`scripts/pre_encode_dataset.py --augment_variants N`) to write
several differently-transposed / differently-paced copies of each track, so a frozen-latent
dataset still shows the model a range of keys and tempos.

Two rules the callers depend on:

* **Time stretch is shared.** The same `rate` must be applied to the target and to every
  control of a sample. A control stretched independently of its target is still
  well-formed, still trains, and is musically wrong — exactly the failure mode
  `scripts/check_streamgen_alignment.py` exists to catch.
* **Pitch shift is opt-in per stream.** Callers decide which streams get it. Transposing
  every stream by the same interval moves the whole arrangement to a new key, which is the
  usual thing to want. Transposing only some of them is the way to decouple one stream's
  tuning from another's — for streamgen that would mean re-keying the accompaniment while
  the drum target keeps its own tuning, so the model cannot read kit tuning off the key.

A shift and a stretch together cost one STFT pass: the pitch shift is a resample (which
moves pitch and duration together) and the phase vocoder then fixes the duration back to
whatever the stretch asked for. The resample ratio is a rational approximation of
``2 ** (semitones / 12)`` with a bounded denominator — under a cent of error at the default
bound, and it keeps torchaudio's resampling kernel small. Asking for the exact irrational
ratio makes that kernel enormous and the pass unusably slow.
"""

import math
import random

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

import torch
import torchaudio

from torch.nn import functional as F

DEFAULT_N_FFT = 2048
DEFAULT_MAX_DENOMINATOR = 200


@dataclass(frozen=True)
class AugmentationParams:
    """One roll of the augmentation dice, shared across a sample's streams.

    Attributes:
        rate:      Time-stretch rate. >1 speeds up (output is shorter), <1 slows down.
                   Pitch is preserved.
        semitones: Pitch shift in semitones, applied only to the streams the caller opts in.
        variant:   Index of the augmented copy this belongs to; 0 is the unaugmented pass.
    """

    rate: float = 1.0
    semitones: float = 0.0
    variant: int = 0

    @property
    def is_identity(self) -> bool:
        return self.rate == 1.0 and self.semitones == 0.0

    def as_dict(self) -> dict:
        return {
            "variant": self.variant,
            "time_stretch_rate": self.rate,
            "pitch_semitones": self.semitones,
        }


def sample_augmentation_params(
    variant: int,
    rng: Optional[random.Random] = None,
    max_semitones: float = 2.0,
    rate_range: Tuple[float, float] = (0.9, 1.1),
) -> AugmentationParams:
    """Roll the augmentation for one variant of one track.

    Variant 0 is always the identity, so a dataset encoded with ``--augment_variants N``
    contains the original material plus ``N - 1`` augmented copies rather than N copies of
    which none is clean.
    """
    if variant == 0:
        return AugmentationParams(variant=0)

    rng = rng or random
    return AugmentationParams(
        rate=rng.uniform(*rate_range),
        semitones=rng.uniform(-max_semitones, max_semitones),
        variant=variant,
    )


def _rational(factor: float, max_denominator: int = DEFAULT_MAX_DENOMINATOR) -> Tuple[int, int]:
    frac = Fraction(float(factor)).limit_denominator(max_denominator)
    return frac.numerator, frac.denominator


def resample_by(
    waveform: torch.Tensor,
    factor: float,
    max_denominator: int = DEFAULT_MAX_DENOMINATOR,
) -> torch.Tensor:
    """Speed the signal up by ``factor`` by resampling: pitch × factor, length ÷ factor.

    This is the "play the tape faster" transform — it is what makes the pitch shift, and the
    phase vocoder afterwards is what undoes its effect on duration.
    """
    num, den = _rational(factor, max_denominator)
    if num == den:
        return waveform
    return torchaudio.functional.resample(waveform, num, den)


def _usable_n_fft(n_fft: int, *lengths: int) -> int:
    """Shrink ``n_fft`` to a power of two that fits the shortest signal involved."""
    shortest = max(8, min(lengths))
    if n_fft <= shortest:
        return n_fft
    return 1 << int(math.log2(shortest))


def phase_vocode_to_length(
    waveform: torch.Tensor,
    target_length: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: Optional[int] = None,
) -> torch.Tensor:
    """Resample-free duration change to exactly ``target_length`` samples, pitch preserved.

    Driving this by target length rather than by a rate is deliberate: the caller needs the
    target and its controls to come out the same number of samples long, and a rate that is
    merely equal for both still rounds independently.
    """
    n = waveform.shape[-1]
    if target_length == n:
        return waveform

    n_fft = _usable_n_fft(n_fft, n, target_length)
    if hop_length is None:
        hop_length = n_fft // 4

    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    phase_advance = torch.linspace(
        0, math.pi * hop_length, spec.shape[-2], device=spec.device
    )[..., None]
    spec = torchaudio.functional.phase_vocoder(spec, n / target_length, phase_advance)
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        length=target_length,
    )


def pitch_shift_and_time_stretch(
    waveform: torch.Tensor,
    semitones: float = 0.0,
    rate: float = 1.0,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: Optional[int] = None,
    max_denominator: int = DEFAULT_MAX_DENOMINATOR,
) -> torch.Tensor:
    """Transpose by ``semitones`` and change duration by ``1 / rate``, in one STFT pass.

    Args:
        waveform:   ``[channels, samples]`` float tensor.
        semitones:  Pitch shift; positive is up. 0 leaves pitch alone.
        rate:       Time-stretch rate; >1 shortens. 1.0 leaves duration alone.

    Returns:
        A tensor of exactly ``round(samples / rate)`` samples.
    """
    n = waveform.shape[-1]
    target_length = max(1, int(round(n / float(rate))))

    out = waveform
    if semitones:
        out = resample_by(out, 2.0 ** (float(semitones) / 12.0), max_denominator)
    return phase_vocode_to_length(out, target_length, n_fft=n_fft, hop_length=hop_length)


def augment_padded_clip(
    audio: torch.Tensor,
    valid_length: int,
    params: AugmentationParams,
    apply_pitch: bool,
    total_length: Optional[int] = None,
    n_fft: int = DEFAULT_N_FFT,
) -> Tuple[torch.Tensor, int]:
    """Augment the valid region of a zero-padded clip and re-pad it to its original width.

    The dataset hands out clips padded to a fixed ``sample_size``; only the leading
    ``valid_length`` samples are real. Stretching the padding too would be wasted work and
    would smear the silence boundary, so the transform is applied to the valid region alone
    and the result is re-padded (or truncated, when a slow-down overruns the window).

    Returns:
        ``(clip, new_valid_length)`` — the clip is ``total_length`` samples wide.
    """
    if total_length is None:
        total_length = audio.shape[-1]

    if params.is_identity:
        return audio[:, :total_length], min(valid_length, total_length)

    dtype = audio.dtype
    clip = audio[:, :valid_length].float()
    clip = pitch_shift_and_time_stretch(
        clip,
        semitones=params.semitones if apply_pitch else 0.0,
        rate=params.rate,
        n_fft=n_fft,
    )
    clip = clip.clamp(-1, 1).to(dtype)

    new_valid = clip.shape[-1]
    if new_valid >= total_length:
        return clip[:, :total_length], total_length
    return F.pad(clip, (0, total_length - new_valid)), new_valid
