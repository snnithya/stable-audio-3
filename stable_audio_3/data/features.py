"""Frame-rate feature controls: low-dimensional signals computed from audio at the latent rate.

The VAE path (``__audio__`` in a custom metadata module, ``--controls`` at pre-encode time)
hands the model a 256-channel latent of the control audio. A *feature* control is the cheap
alternative for when the model should see an event stream rather than a timbre: one or a few
channels per latent frame, computed directly from the waveform, fused into the same
``{id}_controls.npy`` sidecar and split back out by ``controls``/``controls_dim`` exactly like
a latent. Experiment 3.4 (``experiments/03-wjd-jazz-stems/04-onset-representation.md``) is
where this comes from; the first feature is the drum RMS envelope, the second the causal
two-band TRIA features (``tria_control``) that generalise it.

Frame grid
----------
One feature frame per ``LATENT_HOP`` samples, so frame ``t`` covers samples
``[t*hop, (t+1)*hop)``: the same window latent frame ``t`` of the SAME-S / SAME-L
autoencoders summarises (44.1 kHz / 4096 = 10.77 Hz). A feature tensor for a clip of ``T``
samples therefore has ``ceil(T / hop)`` frames, which is also the latent length the
autoencoder returns for that clip (pre_encode_dataset.py still crops/pads to the exact latent
length when it fuses, so an off-by-one here is harmless, but the grid is the same by design).

Causality
---------
``block_rms_db`` is causal by construction: frame ``t`` is a function of samples strictly
before ``(t+1)*hop`` and nothing after. That is the property an interactive system needs,
where the control arrives in real time and frame ``t`` must be computable the moment latent
frame ``t`` is. A windowed/overlapping RMS with a centred window would look ``hop/2`` into the
future; if a smoother envelope is ever wanted, make the window trail (end at the frame
boundary), not centre.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torchaudio.functional as AF

# Samples per latent frame for the Stable Audio 3 autoencoders used here (patch 256 x stride 16).
LATENT_HOP = 4096

# dB floor for RMS features. Digital silence is -inf; clamping at -80 dBFS (below any real
# noise floor in the data) keeps the feature finite and the normalised range well defined.
RMS_FLOOR_DB = -80.0


def frames_for(n_samples: int, hop: int = LATENT_HOP) -> int:
    """Number of feature frames covering ``n_samples`` samples (last frame may be partial)."""
    return math.ceil(n_samples / hop)


def block_rms_db(audio: torch.Tensor, hop: int = LATENT_HOP, floor_db: float = RMS_FLOOR_DB) -> torch.Tensor:
    """Per-frame RMS level in dBFS of a ``[C, T]`` clip, mixed down to mono. Returns ``[1, ceil(T/hop)]``.

    Frame ``t`` is the RMS of samples ``[t*hop, (t+1)*hop)`` of the mono mixdown, so the
    feature is causal (see module docstring). A trailing partial frame is averaged over the
    samples it actually has, not zero-padded, so a short last frame is not reported quieter
    than it is. Levels below ``floor_db`` are clamped to it.
    """
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError(f"expected [C, T] audio, got shape {tuple(audio.shape)}")

    mono = audio.float().mean(dim=0)
    n = mono.shape[-1]
    n_frames = frames_for(n, hop)
    if n_frames == 0:
        return torch.full((1, 0), floor_db, dtype=torch.float32)

    padded = torch.nn.functional.pad(mono, (0, n_frames * hop - n))
    energy = (padded.reshape(n_frames, hop) ** 2).sum(dim=1)
    counts = torch.full((n_frames,), float(hop), dtype=energy.dtype)
    counts[-1] = n - (n_frames - 1) * hop  # samples the partial last frame really has
    rms = torch.sqrt(energy / counts)
    db = 20 * torch.log10(rms.clamp_min(1e-12))
    return db.clamp_min(floor_db).unsqueeze(0)


def normalize_db(db: torch.Tensor, floor_db: float = RMS_FLOOR_DB, ceil_db: float = 0.0) -> torch.Tensor:
    """Map dBFS levels in ``[floor_db, ceil_db]`` linearly onto ``[0, 1]`` (clamped)."""
    return ((db - floor_db) / (ceil_db - floor_db)).clamp(0.0, 1.0)


def rms_envelope_control(
    audio: torch.Tensor,
    hop: int = LATENT_HOP,
    floor_db: float = RMS_FLOOR_DB,
    valid_samples: int = None,
) -> torch.Tensor:
    """The drum-RMS control: ``[1, ceil(T/hop)]`` in ``[0, 1]``, 0 = at/below the floor, 1 = 0 dBFS.

    ``valid_samples`` marks where the *target's* real audio ends inside a padded window. The
    control is forced to its floor from there on, so padding on the target side is matched by
    "nothing" on the control side, whatever the control audio happened to hold there.
    """
    if valid_samples is not None and valid_samples < audio.shape[-1]:
        audio = audio.clone()
        audio[..., valid_samples:] = 0
    return normalize_db(block_rms_db(audio, hop, floor_db), floor_db)


# ---------------------------------------------------------------------------
# TRIA-style two-band features (experiment 3.4)
# ---------------------------------------------------------------------------
#
# TRIA ("The Rhythm In Anything", O'Reilly et al., ISMIR 2024 LBD) conditions a drum generator
# on a deliberately lossy rhythm signal: per codec frame, an 80-bin mel spectrogram summed
# into two equal-energy bands, standardised, squashed by a sigmoid and quantised to 33 levels
# so that timbre cannot leak through. Three of those steps look at the whole clip -- centred
# STFT windows, the equal-energy split frequency, and the per-clip mean/std -- so the
# version here replaces each with a causal equivalent and keeps the rest:
#
#   mel bands        -> a causal IIR crossover (two cascaded 2nd-order Butterworth biquads per
#                       band, i.e. 4th-order Linkwitz-Riley) and the same block RMS as
#                       `block_rms_db` per band. Frame t sees nothing after sample (t+1)*hop.
#   adaptive split   -> a fixed split frequency, the median equal-energy frequency of the
#                       training drum stems (scripts/wjd/tria_feature_stats.py).
#   per-clip z-score -> `fixed`: dataset mean/std per band, from the same script; or
#                       `ema`: running mean/var over an exponentially weighted window ending
#                       at the current frame, warm-started from the dataset statistics.
#   sigmoid, 33 levels, unchanged.
#
# Two channels per latent frame (low band, high band), so with one band this reduces to the
# RMS envelope above up to the normalisation. The frame grid is the latent grid (S = 1 in
# the 3.4 notes): no sub-frame timing, by decision (2026-10-02).

TRIA_LEVELS = 33
TRIA_DEFAULT_STATS = (
    Path(__file__).resolve().parents[1] / "configs/dataset_configs/features/wjd_drums_tria_stats.json"
)
# Floor on the running / dataset std, in dB: keeps z finite on a constant input.
TRIA_MIN_STD_DB = 1.0


@dataclass
class TriaStats:
    """Dataset constants the TRIA control depends on; written by scripts/wjd/tria_feature_stats.py."""

    split_hz: float
    band_mean_db: list  # [low, high]
    band_std_db: list  # [low, high]
    sample_rate: int = 44100
    floor_db: float = RMS_FLOOR_DB
    ema_tau_seconds: float = 4.0
    source: str = ""

    def __post_init__(self):
        if len(self.band_mean_db) != 2 or len(self.band_std_db) != 2:
            raise ValueError("band_mean_db / band_std_db must have one entry per band (2)")
        if not 0 < self.split_hz < self.sample_rate / 2:
            raise ValueError(f"split_hz {self.split_hz} outside (0, nyquist)")

    @classmethod
    def load(cls, path=None) -> "TriaStats":
        path = Path(path or TRIA_DEFAULT_STATS)
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    def ema_tau_frames(self, hop: int = LATENT_HOP) -> float:
        return self.ema_tau_seconds * self.sample_rate / hop


def crossover(audio: torch.Tensor, split_hz: float, sample_rate: int, order: int = 4):
    """Causal crossover of a ``[C, T]`` signal into ``(low, high)``, both ``[C, T]`` float64.

    ``order // 2`` cascaded 2nd-order Butterworth sections per band (Q = 1/sqrt 2), so the
    default is a 4th-order Linkwitz-Riley pair: each band is -6 dB at ``split_hz`` and the
    two sum to an allpass. IIR, hence causal; run in float64 because a low crossover relative
    to the sample rate makes the biquad coefficients ill-conditioned in float32.
    """
    if order < 2 or order % 2:
        raise ValueError("order must be an even number >= 2")
    x = audio.double()
    q = 1 / math.sqrt(2)
    low, high = x, x
    for _ in range(order // 2):
        low = AF.lowpass_biquad(low, sample_rate, split_hz, q)
        high = AF.highpass_biquad(high, sample_rate, split_hz, q)
    return low, high


def band_rms_db(
    audio: torch.Tensor,
    split_hz: float,
    sample_rate: int = 44100,
    hop: int = LATENT_HOP,
    floor_db: float = RMS_FLOOR_DB,
) -> torch.Tensor:
    """Per-frame RMS level in dBFS of the low and high band of a ``[C, T]`` clip: ``[2, ceil(T/hop)]``.

    Mono mixdown, then `crossover`, then `block_rms_db` per band; the same frame grid and
    causality as the RMS envelope.
    """
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    mono = audio.float().mean(dim=0, keepdim=True)
    low, high = crossover(mono, split_hz, sample_rate)
    return torch.cat([block_rms_db(low, hop, floor_db), block_rms_db(high, hop, floor_db)], dim=0)


def fixed_standardize(x: torch.Tensor, mean, std, min_std: float = TRIA_MIN_STD_DB) -> torch.Tensor:
    """``(x - mean) / std`` per channel of a ``[C, T]`` tensor, with ``std`` floored at ``min_std``."""
    mean = torch.as_tensor(mean, dtype=x.dtype).view(-1, 1)
    std = torch.as_tensor(std, dtype=x.dtype).view(-1, 1).clamp_min(min_std)
    return (x - mean) / std


def ema_standardize(
    x: torch.Tensor,
    prior_mean,
    prior_std,
    tau_frames: float,
    prior_frames: float = None,
    min_std: float = TRIA_MIN_STD_DB,
) -> torch.Tensor:
    """Causal z-score of a ``[C, T]`` tensor against running statistics.

    Frame ``t`` is standardised by the mean and variance of an exponentially weighted window
    ending at ``t`` (weights ``exp(-k / tau_frames)`` for the frame ``k`` steps back), so the
    normalisation adapts to the level of the last few seconds the way TRIA's per-clip
    standardisation adapts to the clip, without seeing the future. The window is warm-started
    with ``prior_mean`` / ``prior_std`` carrying the weight of ``prior_frames`` frames
    (default ``tau_frames``), so the first frames are standardised against the dataset rather
    than against themselves.
    """
    if x.ndim != 2:
        raise ValueError(f"expected [C, T], got {tuple(x.shape)}")
    if tau_frames <= 0:
        raise ValueError("tau_frames must be positive")
    if prior_frames is None:
        prior_frames = tau_frames
    alpha = math.exp(-1.0 / tau_frames)
    x64 = x.double()
    mean0 = torch.as_tensor(prior_mean, dtype=torch.float64).view(-1)
    std0 = torch.as_tensor(prior_std, dtype=torch.float64).view(-1)
    weight = float(prior_frames)
    s1 = weight * mean0
    s2 = weight * (std0**2 + mean0**2)
    out = torch.empty_like(x64)
    for t in range(x64.shape[1]):
        xt = x64[:, t]
        weight = alpha * weight + 1.0
        s1 = alpha * s1 + xt
        s2 = alpha * s2 + xt * xt
        mu = s1 / weight
        var = (s2 / weight - mu * mu).clamp_min(0.0)
        out[:, t] = (xt - mu) / torch.sqrt(var).clamp_min(min_std)
    return out.to(x.dtype)


def quantize_unit(x: torch.Tensor, n_levels: int = TRIA_LEVELS) -> torch.Tensor:
    """Round values in [0, 1] to ``n_levels`` evenly spaced levels (0, 1/(n-1), ..., 1)."""
    steps = n_levels - 1
    return torch.round(x.clamp(0.0, 1.0) * steps) / steps


def tria_control(
    audio: torch.Tensor,
    stats: TriaStats,
    norm: str = "fixed",
    hop: int = LATENT_HOP,
    valid_samples: int = None,
    n_levels: int = TRIA_LEVELS,
) -> torch.Tensor:
    """The causal TRIA control of a ``[C, T]`` clip: ``[2, ceil(T/hop)]`` in [0, 1], quantised.

    Channel 0 is the band below ``stats.split_hz``, channel 1 the band above. ``norm`` is
    ``"fixed"`` (dataset mean/std from ``stats``) or ``"ema"`` (running statistics,
    `ema_standardize`, time constant ``stats.ema_tau_seconds``). As for the RMS control,
    ``valid_samples`` marks where the target's real audio ends: the control audio is zeroed
    from there and every frame after the last valid one is forced to 0, whatever the
    normalisation would have made of it.
    """
    if norm not in ("fixed", "ema"):
        raise ValueError(f"norm must be 'fixed' or 'ema', got {norm!r}")
    if valid_samples is not None and valid_samples < audio.shape[-1]:
        audio = audio.clone()
        audio[..., valid_samples:] = 0
    db = band_rms_db(audio, stats.split_hz, stats.sample_rate, hop, stats.floor_db)
    if norm == "fixed":
        z = fixed_standardize(db, stats.band_mean_db, stats.band_std_db)
    else:
        z = ema_standardize(db, stats.band_mean_db, stats.band_std_db, stats.ema_tau_frames(hop))
    feat = quantize_unit(torch.sigmoid(z), n_levels)
    if valid_samples is not None:
        feat[:, frames_for(valid_samples, hop):] = 0
    return feat.float()
