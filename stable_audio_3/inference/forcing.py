"""Per-frame ("diffusion forcing") noise schedules and the Euler sampler that walks them.

The DF model takes ``t`` of shape ``(B, N)`` -- one noise level per latent frame -- so at every
sampler step the whole window is denoised in **one** forward pass, with different frames at
different points along their trajectories. A schedule is therefore a *matrix*
``t[step, frame]``, and sampling walks down its rows.

Every schedule here is the same 1-D trajectory started at a different step for each frame::

    s = build_schedule(steps, dist_shift=...)   # (S+1,), s[0] = 1, s[S] = 0, already shifted
    t[i, f] = s[clamp(i - d[f], 0, S)]          # frame f starts its trajectory at step d[f]

so frame ``f`` sits in pure noise for ``i < d[f]``, runs the identical ``S``-step trajectory,
and is clean from ``d[f] + S`` on. Two properties fall out:

* **Frozen frames stay frozen.** A frame clamped at either end has ``dt = 0``. Committed chunks
  and not-yet-started chunks need no masking and no copy-back.
* **Every frame gets exactly ``S`` steps** in every schedule, so schedules differ in *when*
  frames are denoised, never in *how much*.

A prefix frame (ground-truth context) is expressed as ``d = -steps``: ``clamp(i + S, 0, S) = S``
for every ``i``, i.e. clean from step 0 onward. No sentinel value is needed.

Nothing here changes ``sample_diffusion`` or ``sample_discrete_euler``; this is an additional
path, used by ``scripts/sample_df_schedules.py``.

See experiments/03-diffusion-forcing/03-per-frame-inference-schedules.md.
"""

import typing as tp

import torch
from tqdm import tqdm

SCHEDULES = ("joint", "block", "linear", "pyramid")


def chunk_bounds(
    n_frames: int,
    chunk_frames: int,
    prefix_frames: int,
    allow_partial: bool = False,
) -> tp.List[tp.Tuple[int, int]]:
    """Split the generated region into chunk ``(start, end)`` frame ranges.

    The prefix occupies ``[0, prefix_frames)`` and is not a chunk: it is ground-truth context,
    clean from step 0. Chunks tile ``[prefix_frames, n_frames)``.

    Args:
        n_frames: Window length in latent frames.
        chunk_frames: Emission granularity in latent frames.
        prefix_frames: Ground-truth context frames at the head of the window.
        allow_partial: Permit a final chunk shorter than ``chunk_frames``. Off by default,
            because a short last chunk gets the same step budget as a full one and so is not
            comparable across schedules.
    """
    if not 0 <= prefix_frames < n_frames:
        raise ValueError(f"prefix_frames must be in [0, {n_frames}), got {prefix_frames}")
    if chunk_frames <= 0:
        raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")

    generated = n_frames - prefix_frames
    remainder = generated % chunk_frames
    if remainder and not allow_partial:
        raise ValueError(
            f"{generated} generated frames do not tile into chunks of {chunk_frames} "
            f"({remainder} left over). Set prefix_frames={remainder + prefix_frames} to absorb "
            f"the remainder into the context, or pass allow_partial=True."
        )

    bounds = []
    start = prefix_frames
    while start < n_frames:
        end = min(start + chunk_frames, n_frames)
        bounds.append((start, end))
        start = end
    return bounds


def frame_delays(
    schedule: str,
    n_frames: int,
    chunk_frames: int,
    prefix_frames: int,
    steps: int,
    ramp: float = 1.0,
    pyramid_stagger: float = 0.5,
    allow_partial: bool = False,
    device: tp.Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Per-frame start step ``d[f]`` for one of the named schedules.

    Returns ``(n_frames,)`` int64. Prefix frames get ``-steps``, meaning "already finished".

    Schedules, with ``k`` the chunk index, ``f0`` the first generated frame and ``C`` the chunk
    length:

    ``joint``
        ``d = 0`` everywhere. The ordinary global-``t`` sampler written as a ``(B, N)`` vector.
        Not streamable -- the coherence ceiling.
    ``block``
        ``d = k * steps``. One chunk in flight; its neighbours are either clean or pure noise.
        Chunk-by-chunk continuation.
    ``linear``
        ``d = round((f - f0) * steps / (ramp * C))``. A ramp ``ramp`` chunks wide sliding right;
        the chunk after the one being emitted is already mostly denoised.
    ``pyramid``
        ``d = round(k * steps * pyramid_stagger)``. A staircase. At the default stagger of 0.5
        two chunks are in flight and emissions come every ``steps / 2``, i.e. **twice as often
        as block and linear** -- same per-frame budget, half the latency, not cadence-matched.

    Args:
        ramp: ``linear`` only. Ramp length in chunks. 1.0 puts the full trajectory across one
            chunk; larger values make a shallower ramp spanning more chunks.
        pyramid_stagger: ``pyramid`` only. Chunk-to-chunk delay as a fraction of ``steps``.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    bounds = chunk_bounds(n_frames, chunk_frames, prefix_frames, allow_partial=allow_partial)

    # -steps => clamp(i + steps, 0, steps) == steps for every i, i.e. t = 0 from step 0 on.
    delays = torch.full((n_frames,), -steps, dtype=torch.long, device=device)

    for k, (start, end) in enumerate(bounds):
        if schedule == "joint":
            value = torch.zeros(end - start, dtype=torch.long, device=device)
        elif schedule == "block":
            value = torch.full((end - start,), k * steps, dtype=torch.long, device=device)
        elif schedule == "pyramid":
            value = torch.full(
                (end - start,), round(k * steps * pyramid_stagger), dtype=torch.long, device=device
            )
        else:  # linear
            if ramp <= 0:
                raise ValueError(f"ramp must be positive, got {ramp}")
            f = torch.arange(start, end, device=device, dtype=torch.float64)
            value = torch.round((f - prefix_frames) * steps / (ramp * chunk_frames)).long()
        delays[start:end] = value

    return delays


def build_frame_schedule(base: torch.Tensor, delays: torch.Tensor) -> torch.Tensor:
    """Expand a 1-D trajectory and per-frame delays into the full ``(S_total + 1, N)`` matrix.

    Args:
        base: ``(S + 1,)`` schedule from ``build_schedule`` -- already distribution-shifted,
            ``base[0] == sigma_max``, ``base[S] == 0``.
        delays: ``(N,)`` start steps from :func:`frame_delays`. Negative means "already clean".

    Returns:
        ``(S_total + 1, N)`` with ``S_total = S + max(0, delays.max())``. Column ``f`` is
        non-increasing, starts at ``base[0]`` (or 0 for a prefix frame) and ends at 0.
    """
    if base.ndim != 1:
        raise ValueError(f"base schedule must be 1-D, got shape {tuple(base.shape)}")
    steps = base.shape[0] - 1
    delays = delays.to(device=base.device, dtype=torch.long)

    total_steps = steps + int(delays.clamp(min=0).max().item()) if delays.numel() else steps
    i = torch.arange(total_steps + 1, device=base.device).unsqueeze(1)  # (S_total+1, 1)
    index = (i - delays.unsqueeze(0)).clamp(0, steps)                   # (S_total+1, N)
    return base[index]


def emission_steps(
    delays: torch.Tensor, steps: int, bounds: tp.Sequence[tp.Tuple[int, int]]
) -> tp.List[int]:
    """Step at which each chunk becomes fully clean, i.e. when it could be played out.

    A chunk is done when its *last* frame finishes, so this is ``max(d) + steps`` over the
    chunk. Under ``block`` these come every ``steps``; under ``linear`` every ``steps`` after a
    one-stage fill; under ``pyramid`` every ``steps * stagger``.
    """
    return [int(delays[start:end].max().item()) + steps for start, end in bounds]


@torch.no_grad()
def sample_euler_per_frame(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    step_range: tp.Optional[tp.Tuple[int, int]] = None,
    callback=None,
    disable_tqdm: bool = False,
    **extra_args,
) -> torch.Tensor:
    """Euler ODE solve with a per-frame noise level. Rectified-flow objectives only.

    The per-frame twin of ``sample_discrete_euler``. Two deliberate differences:

    * ``t`` is kept in **float32** rather than cast to ``x.dtype``. ``dit.forward`` wants it in
      float32 anyway (see its comment on the logsnr transform), and round-tripping through bf16
      first would throw that away. At fp32 this is bit-identical to ``sample_discrete_euler``.
    * The update is gated on ``dt != 0`` rather than relying on ``x + 0 * v == x``. A frozen
      frame is evaluated at ``t = 0``, where the model is not trained and may return non-finite
      values; ``0 * nan`` would poison a committed frame. The gate makes freezing exact.

    Args:
        model: The DiT backbone (``wrapper.model``, as ``sample_diffusion`` uses it), accepting
            ``t`` of shape ``(B, N)``.
        x: ``(B, C, N)`` current latents. Frames whose schedule starts at ``t = 0`` must already
            hold their committed values; frames starting at ``t = 1`` must hold noise.
        sigmas: ``(S_total + 1, N)`` or ``(B, S_total + 1, N)`` schedule matrix.
        step_range: ``(start, end)`` half-open slice of rows to walk, so a caller can run one
            stage at a time and rebuild conditioning in between. Defaults to the whole matrix.
        extra_args: Forwarded to ``model`` -- conditioning inputs, ``cfg_scale``,
            ``padding_mask`` and so on.
    """
    if sigmas.ndim == 2:
        sigmas = sigmas.unsqueeze(0).expand(x.shape[0], -1, -1)
    elif sigmas.ndim != 3:
        raise ValueError(f"sigmas must be (S+1, N) or (B, S+1, N), got {tuple(sigmas.shape)}")
    if sigmas.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"schedule covers {sigmas.shape[-1]} frames but x has {x.shape[-1]}"
        )

    sigmas = sigmas.to(x.device)
    num_steps = sigmas.shape[1] - 1
    start, end = step_range if step_range is not None else (0, num_steps)
    if not 0 <= start <= end <= num_steps:
        raise ValueError(f"step_range {(start, end)} out of bounds for {num_steps} steps")

    for i in tqdm(range(start, end), disable=disable_tqdm, leave=False):
        t_curr = sigmas[:, i, :].float()                                  # (B, N)
        dt = (sigmas[:, i + 1, :] - sigmas[:, i, :]).to(x.dtype)[:, None, :]  # (B, 1, N)

        v = model(x, t_curr, **extra_args)

        if callback is not None:
            callback({"x": x, "t": t_curr, "sigma": t_curr, "i": i})

        x = torch.where(dt != 0, x + dt * v, x)

    return x
