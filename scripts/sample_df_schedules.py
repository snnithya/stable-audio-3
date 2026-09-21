"""Generate listening samples under per-frame ("diffusion forcing") noise schedules.

Sub-experiment 3.3. Takes a DF checkpoint, continues held-out Slakh drum prefixes in fixed
4 s chunks, and writes one wav per (item, schedule, future-visibility) so the chunk seams can
be compared by ear.

The schedules are `block` (each chunk denoised on its own, the whole chunk at one noise level),
`linear` (a rolling ramp, so the next chunk is already partly denoised when the current one is
emitted) and `pyramid` (a staircase, two chunks in flight). `joint` -- the whole window at one
noise level, not streamable -- is available as the coherence ceiling. See
`stable_audio_3/inference/forcing.py` for how each is built, and
experiments/03-diffusion-forcing/03-per-frame-inference-schedules.md for the reasoning.

`--future_visibility` is the accompaniment lookahead in seconds, measured from the emission
cursor, the same quantity `training.inpainting.future_visibility` sets during training (the DF
checkpoint trained on a per-item draw from [-4, 0] s). 0 means the model hears the accompaniment
up to the point it has generated to; -2 means the accompaniment stops 2 s behind the cursor, so
the model is writing 2 s ahead of what it can hear.

Note there is no Slakh *test* split pre-encoded -- only `train` and `validation`. The default
dataset config below is the held-out validation split (56 items), which is what every other
experiment in this repo evaluates on.

Usage:
  uv run python scripts/sample_df_schedules.py \
      --ckpt /data/scratch-fast/snnithya/sao-3/ft_checkpoints/sao-3/7tvd98n0/checkpoints/last.ckpt \
      --num_items 8 \
      --out_dir /data/scratch-fast/snnithya/sao-3/outputs/3_3

  uv run python scripts/make_listening_page.py --dir /data/scratch-fast/snnithya/sao-3/outputs/3_3
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

# load_arm handles both .safetensors exports and Lightning .ckpt files (whose keys carry a
# "diffusion." prefix). Reused rather than copied so checkpoint loading cannot drift between
# the two scripts.
from eval_streamgen import load_arm  # noqa: E402

from stable_audio_3.data.dataset import collation_fn  # noqa: E402
from stable_audio_3.data.utils import build_dataset_from_config  # noqa: E402
from stable_audio_3.inference.forcing import (  # noqa: E402
    SCHEDULES,
    build_frame_schedule,
    chunk_bounds,
    emission_steps,
    frame_delays,
    sample_euler_per_frame,
)
from stable_audio_3.inference.sampling import build_schedule  # noqa: E402
from stable_audio_3.training.utils import resize_padding_mask  # noqa: E402

DEFAULT_DATASET = (
    "stable_audio_3/configs/dataset_configs/preencoded/"
    "slakh_streamgen_validation_preencoded.json"
)
DEFAULT_MODEL_CONFIG = "stable_audio_3/configs/model_configs/small_music_base_df.json"


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------


def prepare_batch(batch, model, device, seed, item_offset):
    """Latents, padding mask, accompaniment and the fixed noise for one batch.

    Every schedule and every future-visibility value reuses this dict, so they all start from
    byte-identical noise over byte-identical frames and differences between the wavs are
    attributable to the schedule alone.
    """
    reals, metadata = batch
    if reals.ndim == 4 and reals.shape[0] == 1:
        reals = reals[0]

    latents = reals.to(device=device, dtype=torch.float32)

    padding_mask = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(device)
    if padding_mask.shape[-1] != latents.shape[-1]:
        padding_mask = resize_padding_mask(padding_mask, latents.shape[-1])
    padding_mask = padding_mask.to(torch.bool)

    controls = None
    if all("streamgen_latent" in md.get("controls", {}) for md in metadata):
        controls = torch.stack(
            [md["controls"]["streamgen_latent"] for md in metadata], dim=0
        ).to(device=device, dtype=torch.float32)

    # Pre-encoded latents are stored in autoencoder space; the model works in latent space.
    scale = getattr(model.pretransform, "scale", 1.0)
    if scale != 1.0:
        latents = latents / scale
        if controls is not None:
            controls = controls / scale

    batch_size, _, n_frames = latents.shape
    noise = torch.stack(
        [
            torch.randn(
                model.io_channels,
                n_frames,
                generator=torch.Generator().manual_seed(seed + item_offset + i),
            )
            for i in range(batch_size)
        ],
        dim=0,
    ).to(device)

    return {
        "latents": latents,
        "metadata": metadata,
        "padding_mask": padding_mask,
        "controls": controls,
        "noise": noise,
    }


# ---------------------------------------------------------------------------
# Conditioning at one stage of the rollout
# ---------------------------------------------------------------------------


def build_conditioning(model, prepared, x, cursor, fv_frames, context_feed, device):
    """Conditioning for the stage that starts with the cursor at frame `cursor`.

    The cursor is the end of the last fully committed (t = 0) chunk. Two channels carry the
    past and the accompaniment, and both move with it:

    * `inpaint_mask` / `inpaint_masked_input` -- the causal inpainting conditioning the model
      trained under. Polarity follows `models/inpainting.py`: the mask is **1 where context is
      provided**, 0 where the model must generate. The context fed back is the committed part
      of `x` itself -- ground truth over the prefix, the model's own output over chunks it has
      already emitted -- which is what a streaming system would have.
    * `tf_inpaint_mask` / `streamgen_latent` -- the accompaniment, gated to
      `cursor + fv_frames`. Polarity here is the opposite of the inpainting conds: the
      accompaniment is multiplied by `tf_inpaint_mask` itself, so it is visible *up to* the
      horizon and silent beyond it (see `_add_streamgen_conditioning` in training/diffusion.py).

    `context_feed="t0"` zeroes the inpainting channel, leaving the committed frames visible
    only as clean values inside `x`. That is the DF-native way to pass context and is the
    ablation that says whether the inpainting channel still earns its keep. `"both"` keeps both,
    which is closest to training.

    Not expressible here: plain `inpaint` context, where context frames sit in `x` as *noised*
    latents. Under the delay formulation a committed frame is by construction at t = 0, so the
    clean value is always what `x` holds.
    """
    metadata = prepared["metadata"]
    padding_mask = prepared["padding_mask"]
    n_frames = padding_mask.shape[-1]
    real_lengths = padding_mask.sum(dim=-1)  # (B,)
    index = torch.arange(n_frames, device=device).unsqueeze(0)  # (1, N)

    conditioning = model.conditioner(metadata, device)

    committed = (index < cursor) & padding_mask
    inpaint_mask = committed.unsqueeze(1).to(torch.float32)  # (B, 1, N)
    if context_feed == "t0":
        inpaint_mask = torch.zeros_like(inpaint_mask)
    conditioning["inpaint_mask"] = [inpaint_mask]
    conditioning["inpaint_masked_input"] = [x.to(torch.float32) * inpaint_mask]

    if "streamgen_latent" in model.modular_local_cond_ids:
        if prepared["controls"] is None:
            raise ValueError(
                "Model expects streamgen_latent but the dataset supplied no control sidecar. "
                "Check 'controls'/'controls_dim' in the dataset config."
            )
        horizon = torch.clamp(
            torch.full_like(real_lengths, cursor + fv_frames), min=0
        )
        horizon = torch.minimum(horizon, real_lengths)  # (B,)
        tf_mask = ((index < horizon.unsqueeze(1)) & padding_mask).unsqueeze(1).to(torch.float32)
        conditioning["tf_inpaint_mask"] = [tf_mask]
        conditioning["streamgen_latent"] = [prepared["controls"] * tf_mask]

    return model.get_conditioning_inputs(conditioning)


# ---------------------------------------------------------------------------
# The rollout
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_schedule(model, prepared, schedule, fv_frames, base_schedule, args, device):
    """Roll one schedule out over the whole window, rebuilding conditioning at each emission.

    One forward pass per sampler step covers the *entire* window; the schedule only sets how
    far along its trajectory each frame is. Sampling is cut into stages at the steps where a
    chunk becomes clean, because that is where the cursor moves and therefore where the
    conditioning changes. Within a stage the conditioning is constant.

    Committed frames need no copy-back: their schedule rows are flat, so `dt == 0` and
    `sample_euler_per_frame` leaves them untouched.

    Returns the final latents plus the schedule matrix and emission steps, for the heatmap.
    """
    latents = prepared["latents"]
    n_frames = latents.shape[-1]
    model_dtype = next(model.parameters()).dtype

    bounds = chunk_bounds(n_frames, args.chunk_frames, args.prefix_frames)
    delays = frame_delays(
        schedule,
        n_frames,
        args.chunk_frames,
        args.prefix_frames,
        args.steps,
        ramp=args.ramp,
        pyramid_stagger=args.pyramid_stagger,
        device=device,
    )
    sigmas = build_frame_schedule(base_schedule, delays)  # (S_total + 1, N)
    if args.t_min > 0:
        # Committed frames are evaluated at t = 0, where the model is untrained and where the
        # CFG branch divides by sigma. Clamping lifts them off zero; the flat rows stay flat,
        # so freezing is unaffected.
        sigmas = sigmas.clamp(min=args.t_min)
    emits = emission_steps(delays, args.steps, bounds)

    # Start state: noise everywhere, ground truth on the prefix. The prefix is at t = 0 from
    # step 0, so it must already hold its committed value.
    x = prepared["noise"].to(model_dtype).clone()
    x[:, :, : args.prefix_frames] = latents[:, :, : args.prefix_frames].to(model_dtype)

    stage_starts = [0] + emits[:-1]
    for stage, (lo, hi) in enumerate(zip(stage_starts, emits)):
        if hi <= lo:
            continue  # two chunks emitted on the same step (possible at small stagger)
        cursor = args.prefix_frames if stage == 0 else bounds[stage - 1][1]
        cond_inputs = build_conditioning(
            model, prepared, x, cursor, fv_frames, args.context_feed, device
        )
        x = sample_euler_per_frame(
            model.model,
            x,
            sigmas,
            step_range=(lo, hi),
            padding_mask=prepared["padding_mask"],
            cfg_scale=args.cfg_scale,
            batch_cfg=True,
            disable_tqdm=args.quiet,
            **cond_inputs,
        )

    return x, sigmas, emits, bounds


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_schedule_heatmap(sigmas, emits, bounds, path, title, fps):
    """Noise level as steps x frames. The fastest way to see a schedule bug."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    array = sigmas.detach().float().cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    image = ax.imshow(
        array.T, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="magma",
        extent=[0, array.shape[0] - 1, 0, array.shape[1]],
    )
    for step in emits:
        ax.axvline(step, color="white", lw=0.8, ls="--", alpha=0.7)
    for start, _ in bounds:
        ax.axhline(start, color="white", lw=0.6, alpha=0.4)
    ax.set_xlabel("sampler step")
    ax.set_ylabel("latent frame")
    secax = ax.secondary_yaxis("right", functions=(lambda f: f / fps, lambda s: s * fps))
    secax.set_ylabel("seconds")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="noise level t")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_wav(audio, path, sample_rate):
    torchaudio.save(str(path), audio.clamp(-1, 1).float().cpu(), sample_rate)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, model_config = load_arm(args.model_config, args.ckpt, device)
    sample_rate = model_config.get("sample_rate", 44100)
    ds_ratio = model.pretransform.downsampling_ratio
    fps = sample_rate / ds_ratio

    args.chunk_frames = int(round(args.chunk_seconds * fps))
    fv_values = [float(v) for v in args.future_visibility.split(",")]
    if args.cfg_scale != 1.0 and args.t_min == 0.0:
        args.t_min = 1e-3
        print(f"cfg_scale={args.cfg_scale} != 1: raising --t_min to {args.t_min} "
              f"(the CFG branch divides by sigma)")

    dataset = build_dataset_from_config(args.dataset_config, sample_rate, ds_ratio)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collation_fn,
    )

    n_frames = dataset.latent_crop_length
    if args.prefix_frames < 0:
        # Absorb the remainder into the context so the chunks tile the window exactly.
        args.prefix_frames = n_frames % args.chunk_frames
    bounds = chunk_bounds(n_frames, args.chunk_frames, args.prefix_frames)

    print(
        f"{n_frames} frames ({n_frames / fps:.2f}s) = {args.prefix_frames}-frame prefix "
        f"({args.prefix_frames / fps:.2f}s) + {len(bounds)} x {args.chunk_frames}-frame chunks "
        f"({args.chunk_frames / fps:.2f}s each)"
    )
    print(f"seams at frames {[b[0] for b in bounds[1:]]} "
          f"= {[round(b[0] / fps, 2) for b in bounds[1:]]}s")
    print(f"schedules {args.schedules}, future_visibility {fv_values}s "
          f"= {[int(v * fps) for v in fv_values]} frames")

    # One base trajectory, shared by every schedule and every frame. Distribution shift is
    # applied here, once, exactly as sample_diffusion does it.
    base_schedule = build_schedule(
        steps=args.steps,
        sigma_max=1.0,
        dist_shift=model.sampling_dist_shift,
        effective_seq_len=None,
        fallback_seq_len=n_frames,
        include_endpoint=True,
        device=device,
    )

    manifest = {
        "ckpt": args.ckpt,
        "model_config": args.model_config,
        "dataset_config": args.dataset_config,
        "sample_rate": sample_rate,
        "frames_per_second": fps,
        "n_frames": n_frames,
        "prefix_frames": args.prefix_frames,
        "chunk_frames": args.chunk_frames,
        "steps_per_chunk": args.steps,
        "cfg_scale": args.cfg_scale,
        "context_feed": args.context_feed,
        "seams_frames": [b[0] for b in bounds[1:]],
        "future_visibility_seconds": fv_values,
        "schedules": {},
        "items": [],
    }

    heatmaps_done = set()
    item_offset = 0

    for batch in loader:
        if item_offset >= args.num_items:
            break

        prepared = prepare_batch(batch, model, device, args.seed, item_offset)
        batch_size = prepared["latents"].shape[0]

        # References, decoded from the same latents the model is conditioned on so any
        # autoencoder colouration is shared by generation and reference alike.
        reference = model.pretransform.decode(
            prepared["latents"].to(torch.bfloat16)
        ).float().cpu()
        accompaniment = (
            model.pretransform.decode(prepared["controls"].to(torch.bfloat16)).float().cpu()
            if prepared["controls"] is not None
            else None
        )

        for i in range(batch_size):
            index = item_offset + i
            if index >= args.num_items:
                break
            save_wav(reference[i], out_dir / f"{index:03d}_reference-drums.wav", sample_rate)
            if accompaniment is not None:
                save_wav(
                    accompaniment[i], out_dir / f"{index:03d}_accompaniment.wav", sample_rate
                )
                save_wav(
                    0.7 * reference[i] + 0.7 * accompaniment[i],
                    out_dir / f"{index:03d}_reference-mix.wav",
                    sample_rate,
                )

        for fv_seconds in fv_values:
            fv_frames = int(round(fv_seconds * fps))
            for schedule in args.schedules:
                tag = f"{schedule}-fv{fv_seconds:g}"
                print(f"items {item_offset}-{item_offset + batch_size - 1}: {tag}")

                with torch.amp.autocast("cuda", enabled=device == "cuda"):
                    latents, sigmas, emits, bounds = run_schedule(
                        model, prepared, schedule, fv_frames, base_schedule, args, device
                    )

                audio = model.pretransform.decode(latents.to(torch.bfloat16)).float().cpu()

                if schedule not in heatmaps_done:
                    save_schedule_heatmap(
                        sigmas,
                        emits,
                        bounds,
                        out_dir / f"schedule-{schedule}.png",
                        f"{schedule}: {sigmas.shape[0] - 1} steps, emissions at {emits}",
                        fps,
                    )
                    heatmaps_done.add(schedule)
                    manifest["schedules"][schedule] = {
                        "total_steps": int(sigmas.shape[0] - 1),
                        "emission_steps": emits,
                    }

                for i in range(batch_size):
                    index = item_offset + i
                    if index >= args.num_items:
                        break
                    save_wav(audio[i], out_dir / f"{index:03d}_{tag}-drums.wav", sample_rate)
                    if accompaniment is not None:
                        save_wav(
                            0.7 * audio[i] + 0.7 * accompaniment[i, :, : audio.shape[-1]],
                            out_dir / f"{index:03d}_{tag}-mix.wav",
                            sample_rate,
                        )

        for i in range(batch_size):
            index = item_offset + i
            if index >= args.num_items:
                break
            manifest["items"].append(
                {"item": index, "track_id": prepared["metadata"][i].get("track_id")}
            )

        item_offset += batch_size
        del prepared, reference, accompaniment
        torch.cuda.empty_cache()

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote {out_dir}")
    print(f"listen: uv run python scripts/make_listening_page.py --dir {out_dir}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="DF checkpoint (.ckpt or .safetensors)")
    p.add_argument("--model_config", default=DEFAULT_MODEL_CONFIG)
    p.add_argument(
        "--dataset_config",
        default=DEFAULT_DATASET,
        help="Held-out pre-encoded dataset. No Slakh test split exists; this is validation.",
    )
    p.add_argument("--out_dir", required=True, help="Where the wavs, heatmaps and manifest go")

    p.add_argument(
        "--schedules",
        nargs="+",
        default=["block", "linear", "pyramid"],
        choices=list(SCHEDULES),
        help="Add 'joint' for the non-streaming coherence ceiling",
    )
    p.add_argument(
        "--future_visibility",
        default="0,-2",
        help=(
            "Comma-separated accompaniment lookahead in seconds, measured from the emission "
            "cursor. Negative means the accompaniment lags the cursor. Comma-separated rather "
            "than space-separated so leading-minus values are not read as flags."
        ),
    )
    p.add_argument("--chunk_seconds", type=float, default=4.0, help="Emission granularity")
    p.add_argument(
        "--prefix_frames",
        type=int,
        default=-1,
        help="Ground-truth context frames. -1 auto-sizes so the chunks tile the window.",
    )
    p.add_argument("--steps", type=int, default=50, help="Sampler steps per chunk")
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument(
        "--context_feed",
        default="both",
        choices=["both", "t0"],
        help="'both' keeps the inpainting channel; 't0' passes context only as clean x",
    )
    p.add_argument("--ramp", type=float, default=1.0, help="linear: ramp length in chunks")
    p.add_argument(
        "--pyramid_stagger",
        type=float,
        default=0.5,
        help="pyramid: chunk-to-chunk delay as a fraction of --steps",
    )
    p.add_argument(
        "--t_min",
        type=float,
        default=0.0,
        help="Floor on the schedule. Auto-raised to 1e-3 when cfg_scale != 1.",
    )

    p.add_argument("--num_items", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true", help="Hide the per-step progress bar")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
