"""Frame-rate feature controls: low-dimensional signals computed from audio at the latent rate.

The VAE path (``__audio__`` in a custom metadata module, ``--controls`` at pre-encode time)
hands the model a 256-channel latent of the control audio. A *feature* control is the cheap
alternative for when the model should see an event stream rather than a timbre: one or a few
channels per latent frame, computed directly from the waveform, fused into the same
``{id}_controls.npy`` sidecar and split back out by ``controls``/``controls_dim`` exactly like
a latent. Experiment 3.4 (``experiments/03-wjd-jazz-stems/04-onset-representation.md``) is
where this comes from; the first feature is the drum RMS envelope.

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

import math

import torch

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
