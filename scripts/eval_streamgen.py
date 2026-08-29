"""Paired evaluation of streamgen-conditioned vs. text-only drum models (experiment 1.2).

Two questions, two stages:

  1. **Does the accompaniment lower the denoising loss?**  Every arm is scored on the same
     held-out latents, with the same causal mask, the same timesteps and the *same noise*,
     so the per-item losses are paired and the difference between arms can be bootstrapped
     over items rather than eyeballed off two noisy training curves.

  2. **Do the generated drums actually follow the accompaniment?**  Each arm continues a
     held-out drum prefix, the result is decoded and compared against the accompaniment's
     onset envelope.  Reported alongside two reference points that make the number readable:
     the ground-truth drums (ceiling) and a mismatched accompaniment from another track
     (chance floor).

A streamgen arm is additionally scored with its accompaniment zeroed and with it swapped
between items.  That ablation is the strongest single piece of evidence available here: it
is a within-model comparison, so unlike the cross-arm one it cannot be explained by two
training runs landing in different places.

Usage:
  uv run python scripts/eval_streamgen.py \
      --arm cond stable_audio_3/configs/model_configs/small_music_streamgen.json /path/to/cond/last.ckpt \
      --arm base stable_audio_3/configs/model_configs/small_music_baseline.json  /path/to/base/last.ckpt \
      --dataset_config stable_audio_3/configs/dataset_configs/preencoded/slakh_streamgen_validation_preencoded.json \
      --out experiments/01-streamgen-conditioning/results/1_2.json \
      --audio_dir experiments/01-streamgen-conditioning/results/audio
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from safetensors.torch import load_file

from stable_audio_3.data.dataset import collation_fn
from stable_audio_3.data.utils import build_dataset_from_config
from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3.inference.sampling import sample_diffusion
from stable_audio_3.loading_utils import copy_state_dict
from stable_audio_3.training.utils import (
    compute_normalized_mse,
    resize_padding_mask,
)

ONSET_HOP = 512
ONSET_NFFT = 2048


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_arm(model_config_path, ckpt_path, device, use_ema=False):
    """Build a model from its config and load finetuned weights.

    Accepts either a `.safetensors` export (SafetensorsExportCallback) or a Lightning
    `.ckpt`, whose keys are the training wrapper's and carry a `diffusion.` prefix.
    """
    with open(model_config_path) as f:
        model_config = json.load(f)

    model = create_diffusion_cond_from_config(model_config)

    if ckpt_path is not None:
        if str(ckpt_path).endswith(".safetensors"):
            state_dict = load_file(ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            raw = ckpt.get("state_dict", ckpt)
            ema_prefix = "diffusion_ema.ema_model."
            has_ema = any(k.startswith(ema_prefix) for k in raw)
            if use_ema and not has_ema:
                raise ValueError(f"--use_ema requested but {ckpt_path} holds no EMA weights")
            state_dict = {}
            for k, v in raw.items():
                if k.startswith(ema_prefix):
                    # EMA shadows diffusion.model only; fold it in under that name.
                    if use_ema:
                        state_dict["model." + k[len(ema_prefix):]] = v
                elif k.startswith("diffusion_ema."):
                    continue
                elif k.startswith("diffusion."):
                    key = k[len("diffusion."):]
                    if use_ema and key.startswith("model."):
                        continue
                    state_dict[key] = v
        copy_state_dict(model, state_dict)

    model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    return model, model_config


# ---------------------------------------------------------------------------
# Batch preparation — identical for every arm
# ---------------------------------------------------------------------------


def prepare_batch(batch, model, device, eval_frames, context_frames, lookahead_frames):
    """Crop, rescale and mask one batch the same way for every arm.

    The crop is the centre of each item's valid region, so the evaluation window is
    deterministic (the same frames for every arm and every rerun) without being the
    track's first few seconds, which on Slakh is often an intro or silence.
    """
    reals, metadata = batch
    if reals.ndim == 4 and reals.shape[0] == 1:
        reals = reals[0]

    latents = reals.to(device=device, dtype=torch.float32)

    padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(device)
    if padding_masks.shape[-1] != latents.shape[-1]:
        padding_masks = resize_padding_mask(padding_masks, latents.shape[-1])

    controls = None
    if all("streamgen_latent" in md.get("controls", {}) for md in metadata):
        controls = torch.stack(
            [md["controls"]["streamgen_latent"] for md in metadata], dim=0
        ).to(device=device, dtype=torch.float32)

    # Centre the eval window on each item's valid region.
    cropped_latents, cropped_controls, cropped_padding = [], [], []
    for i in range(latents.shape[0]):
        valid = int(padding_masks[i].sum().item())
        start = max(0, (valid - eval_frames) // 2)
        end = start + eval_frames
        cropped_latents.append(latents[i, :, start:end])
        cropped_padding.append(padding_masks[i, start:end])
        if controls is not None:
            cropped_controls.append(controls[i, :, start:end])

    latents = torch.stack(cropped_latents, dim=0)
    padding_masks = torch.stack(cropped_padding, dim=0)
    controls = torch.stack(cropped_controls, dim=0) if controls is not None else None

    # Pre-encoded latents are stored in autoencoder space; the model works in latent space.
    scale = getattr(model.pretransform, "scale", 1.0)
    if scale != 1.0:
        latents = latents / scale
        if controls is not None:
            controls = controls / scale

    seq_len = latents.shape[-1]
    inpaint_mask = torch.ones(latents.shape[0], 1, seq_len, device=device)
    inpaint_mask[:, :, context_frames:] = 0
    tf_mask = torch.ones_like(inpaint_mask)
    tf_mask[:, :, min(seq_len, context_frames + lookahead_frames):] = 0

    padding_bool = padding_masks.to(torch.bool)
    inpaint_mask = inpaint_mask * padding_bool.unsqueeze(1)
    tf_mask = tf_mask * padding_bool.unsqueeze(1)

    return {
        "latents": latents,
        "metadata": metadata,
        "padding_mask": padding_masks,
        "controls": controls,
        "inpaint_mask": inpaint_mask,
        "tf_mask": tf_mask,
    }


def build_conditioning(model, prepared, streamgen_mode, device):
    """Conditioning dict matching the training wrapper's, with the accompaniment ablated.

    `streamgen_mode` is one of "cond" (as trained), "zero" (accompaniment silenced),
    "shuffle" (each item gets another item's accompaniment) or "none" (arm has no
    streamgen conds at all).  Polarity matches `_add_streamgen_conditioning`: the
    accompaniment is multiplied by tf_mask, not by (1 - mask).
    """
    metadata = prepared["metadata"]
    conditioning = model.conditioner(metadata, device)
    conditioning["inpaint_mask"] = [prepared["inpaint_mask"]]
    conditioning["inpaint_masked_input"] = [prepared["latents"] * prepared["inpaint_mask"]]

    if "streamgen_latent" not in model.modular_local_cond_ids:
        return conditioning

    conditioning["tf_inpaint_mask"] = [prepared["tf_mask"]]

    controls = prepared["controls"]
    if controls is None:
        raise ValueError(
            "Arm expects streamgen_latent but the dataset supplied no control sidecar. "
            "Check 'controls'/'controls_dim' in the dataset config."
        )
    if streamgen_mode == "zero":
        controls = torch.zeros_like(controls)
    elif streamgen_mode == "shuffle":
        if controls.shape[0] < 2:
            raise ValueError("streamgen_mode='shuffle' needs a batch of at least 2 items")
        controls = torch.roll(controls, shifts=1, dims=0)

    conditioning["streamgen_latent"] = [controls * prepared["tf_mask"]]
    return conditioning


# ---------------------------------------------------------------------------
# Stage 1 — held-out denoising loss
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_loss(model, loader, args, device, streamgen_mode, fps):
    """Per-item denoising loss on the generated region, at fixed timesteps and fixed noise.

    The noise is drawn from a generator seeded by the item's global index, so arm A and
    arm B are scored against byte-identical noise and the per-item differences are paired.
    """
    context_frames = int(round(args.context_seconds * fps))
    lookahead_frames = int(round(args.lookahead_seconds * fps))

    records = []
    item_offset = 0

    for batch in loader:
        prepared = prepare_batch(
            batch, model, device, args.eval_frames, context_frames, lookahead_frames
        )
        conditioning = build_conditioning(model, prepared, streamgen_mode, device)

        latents = prepared["latents"]
        batch_size = latents.shape[0]

        # Loss only where the model is generating: valid frames outside the given context.
        loss_mask = prepared["padding_mask"].to(torch.bool) & ~prepared["inpaint_mask"].squeeze(1).to(torch.bool)

        noise = torch.stack(
            [
                torch.randn(
                    latents.shape[1],
                    latents.shape[2],
                    generator=torch.Generator().manual_seed(args.seed + item_offset + i),
                )
                for i in range(batch_size)
            ],
            dim=0,
        ).to(device)

        for timestep in args.timesteps:
            t = torch.full((batch_size,), timestep, device=device)
            alphas, sigmas = (1 - t)[:, None, None], t[:, None, None]
            noised = latents * alphas + noise * sigmas
            targets = noise - latents

            with torch.amp.autocast("cuda"):
                output = model(
                    noised,
                    t,
                    cond=conditioning,
                    cfg_dropout_prob=0.0,
                    padding_mask=prepared["padding_mask"],
                )

            mse_full = compute_normalized_mse(output.float(), targets, loss_mask, "none")
            # Per-item loss, so items stay paired across arms: mean over the generated region.
            signal = torch.where(loss_mask.unsqueeze(1), mse_full, 0.0)
            per_item = signal.sum(dim=(1, 2)) / (loss_mask.sum(dim=1) * mse_full.shape[1] + 1e-8)

            for i in range(batch_size):
                records.append(
                    {
                        "item": item_offset + i,
                        "track_id": prepared["metadata"][i].get("track_id"),
                        "timestep": float(timestep),
                        "loss": float(per_item[i]),
                    }
                )

        item_offset += batch_size

    return records


# ---------------------------------------------------------------------------
# Onset / alignment metrics
# ---------------------------------------------------------------------------


def onset_envelope(audio, hop=ONSET_HOP, n_fft=ONSET_NFFT):
    """Half-wave-rectified spectral flux of a [C, N] waveform, as a 1-D torch tensor."""
    x = audio.float().mean(0)
    window = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(x, n_fft, hop, window=window, return_complex=True).abs()
    log_spec = torch.log1p(10.0 * spec)
    flux = torch.clamp(log_spec[:, 1:] - log_spec[:, :-1], min=0).sum(0)
    return flux


def normalized_xcorr(a, b, max_lag):
    """Peak of the normalized cross-correlation of a against b within +/- max_lag.

    Returns (peak_value, peak_lag, value_at_zero_lag). Lag is in envelope frames and is
    positive when `a` runs late relative to `b`, matching the convention in
    scripts/check_streamgen_alignment.py.
    """
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)

    max_lag = int(min(max_lag, n - 1))
    padded = F.pad(a[None, None], (max_lag, max_lag))
    xc = F.conv1d(padded, b[None, None])[0, 0] / n
    peak = int(xc.argmax())
    return float(xc[peak]), peak - max_lag, float(xc[max_lag])


def estimate_pulse(env, fps, min_bpm=60.0, max_bpm=200.0):
    """Period (frames) and phase (frames) of the dominant pulse in an onset envelope."""
    e = env - env.mean()
    n = len(e)
    if n < 8 or float(e.abs().sum()) == 0:
        return None, None

    spectrum = torch.fft.rfft(e, n=2 * n)
    ac = torch.fft.irfft(spectrum * spectrum.conj(), n=2 * n)[:n].real

    lo = max(1, int(fps * 60.0 / max_bpm))
    hi = min(n - 1, int(fps * 60.0 / min_bpm))
    if hi <= lo:
        return None, None

    period = lo + int(ac[lo:hi].argmax())
    # Phase: the offset whose pulse train collects the most onset energy.
    scores = torch.stack([e[offset::period].sum() for offset in range(period)])
    return period, int(scores.argmax())


def beat_hit_rate(drum_env, accomp_env, fps, tolerance_seconds=0.06):
    """Fraction of the drum onset energy landing on the accompaniment's beat grid.

    Energy-weighted rather than peak-picked: peak-picking a decoded 16 kHz drum stem is
    brittle, and the weighted version degrades gracefully instead of flipping on a
    threshold. A drum track ignoring the accompaniment scores near the grid's duty cycle,
    which is why the mismatched-pair control is reported next to it.
    """
    period, phase = estimate_pulse(accomp_env, fps)
    if period is None:
        return None

    tolerance = max(1, int(round(tolerance_seconds * fps)))
    idx = torch.arange(len(drum_env), device=drum_env.device)
    distance = (idx - phase) % period
    distance = torch.minimum(distance, period - distance)
    on_grid = (distance <= tolerance).float()

    energy = torch.clamp(drum_env, min=0)
    total = float(energy.sum())
    if total <= 0:
        return None
    return float((energy * on_grid).sum() / total)


def alignment_metrics(drum_audio, accomp_audio, sample_rate):
    """Onset-level agreement between a drum track and an accompaniment."""
    fps = sample_rate / ONSET_HOP
    drum_env = onset_envelope(drum_audio)
    accomp_env = onset_envelope(accomp_audio)

    if float(drum_env.std()) < 1e-6 or float(accomp_env.std()) < 1e-6:
        return {"xcorr_peak": None, "xcorr_lag_seconds": None, "xcorr_at_zero": None, "beat_hit_rate": None}

    peak, lag, at_zero = normalized_xcorr(drum_env, accomp_env, max_lag=int(0.25 * fps))
    return {
        "xcorr_peak": peak,
        "xcorr_lag_seconds": lag / fps,
        "xcorr_at_zero": at_zero,
        "beat_hit_rate": beat_hit_rate(drum_env, accomp_env, fps),
    }


# ---------------------------------------------------------------------------
# Stage 2 — generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_generation(model, model_config, loader, args, device, streamgen_mode, fps, arm_name, audio_dir):
    """Continue held-out drum prefixes and score the continuation against the accompaniment."""
    import torchaudio

    context_frames = int(round(args.context_seconds * fps))
    lookahead_frames = int(round(args.lookahead_seconds * fps))
    sample_rate = model_config.get("sample_rate", 44100)
    ds_ratio = model.pretransform.downsampling_ratio

    records = []
    item_offset = 0

    for batch in loader:
        if item_offset >= args.num_generate:
            break

        prepared = prepare_batch(
            batch, model, device, args.eval_frames, context_frames, lookahead_frames
        )
        conditioning = build_conditioning(model, prepared, streamgen_mode, device)
        cond_inputs = model.get_conditioning_inputs(conditioning)

        latents = prepared["latents"]
        batch_size = latents.shape[0]

        noise = torch.stack(
            [
                torch.randn(
                    model.io_channels,
                    latents.shape[2],
                    generator=torch.Generator().manual_seed(args.seed + 10_000 + item_offset + i),
                )
                for i in range(batch_size)
            ],
            dim=0,
        ).to(device=device, dtype=torch.bfloat16)

        with torch.amp.autocast("cuda"):
            fakes = sample_diffusion(
                model=model.model,
                noise=noise,
                cond_inputs=cond_inputs,
                diffusion_objective=model.diffusion_objective,
                steps=args.gen_steps,
                cfg_scale=args.cfg_scale,
                conditioning=prepared["metadata"],
                sample_rate=sample_rate,
                pretransform=model.pretransform,
                mask_padding_attention=model.mask_padding_attention,
                use_effective_length_for_schedule=model.use_effective_length_for_schedule,
                padding_mask=prepared["padding_mask"],
                dist_shift=model.sampling_dist_shift,
                batch_cfg=True,
                disable_tqdm=True,
                decode=True,
            )

        # References decoded from the same latents the model was conditioned on, so any
        # autoencoder colouration is shared by generation and reference alike.
        real_audio = model.pretransform.decode(latents.to(torch.bfloat16)).float().cpu()
        fakes = fakes.float().cpu()
        accomp_audio = (
            model.pretransform.decode(prepared["controls"].to(torch.bfloat16)).float().cpu()
            if prepared["controls"] is not None
            else None
        )

        gen_start = context_frames * ds_ratio

        for i in range(batch_size):
            if item_offset + i >= args.num_generate:
                break
            if accomp_audio is None:
                continue

            generated = fakes[i, :, gen_start:]
            reference = real_audio[i, :, gen_start:]
            accompaniment = accomp_audio[i, :, gen_start:]
            # Chance floor: the same generated drums against a different track's accompaniment.
            mismatched = accomp_audio[(i + 1) % batch_size, :, gen_start:]

            record = {
                "item": item_offset + i,
                "track_id": prepared["metadata"][i].get("track_id"),
                "generated_vs_accompaniment": alignment_metrics(generated, accompaniment, sample_rate),
                "reference_vs_accompaniment": alignment_metrics(reference, accompaniment, sample_rate),
                "generated_vs_mismatched": alignment_metrics(generated, mismatched, sample_rate),
            }
            records.append(record)

            if audio_dir is not None:
                out = Path(audio_dir)
                out.mkdir(parents=True, exist_ok=True)
                index = item_offset + i
                stem = f"{index:03d}_{arm_name}_{streamgen_mode}"
                # Generated drums over the accompaniment: the thing you actually listen to.
                mix = 0.7 * fakes[i] + 0.7 * accomp_audio[i, :, : fakes.shape[-1]]
                torchaudio.save(str(out / f"{stem}_mix.wav"), mix.clamp(-1, 1), sample_rate)
                torchaudio.save(str(out / f"{stem}_drums.wav"), fakes[i].clamp(-1, 1), sample_rate)

                # The two things every generation is judged against, written once per item
                # so the directory sorts into listenable groups.
                reference_mix = out / f"{index:03d}_reference_mix.wav"
                if not reference_mix.exists():
                    torchaudio.save(
                        str(reference_mix),
                        (0.7 * real_audio[i] + 0.7 * accomp_audio[i]).clamp(-1, 1),
                        sample_rate,
                    )
                    torchaudio.save(
                        str(out / f"{index:03d}_reference_drums.wav"),
                        real_audio[i].clamp(-1, 1),
                        sample_rate,
                    )
                    torchaudio.save(
                        str(out / f"{index:03d}_accompaniment.wav"),
                        accomp_audio[i].clamp(-1, 1),
                        sample_rate,
                    )

        item_offset += batch_size

    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def bootstrap_ci(values, n_boot=10_000, seed=0):
    """Percentile bootstrap 95% CI of the mean, over items."""
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize_loss(records):
    by_item = {}
    by_timestep = {}
    for r in records:
        by_item.setdefault(r["item"], []).append(r["loss"])
        by_timestep.setdefault(r["timestep"], []).append(r["loss"])
    item_means = {k: float(np.mean(v)) for k, v in by_item.items()}
    return {
        "mean": float(np.mean(list(item_means.values()))),
        "ci95": bootstrap_ci(list(item_means.values())),
        "n_items": len(item_means),
        "per_timestep": {str(k): float(np.mean(v)) for k, v in sorted(by_timestep.items())},
        "per_item": item_means,
    }


def paired_delta(a_summary, b_summary, label_a, label_b):
    """Paired per-item difference (a - b), which is the comparison that matters here."""
    shared = sorted(set(a_summary["per_item"]) & set(b_summary["per_item"]))
    deltas = [a_summary["per_item"][k] - b_summary["per_item"][k] for k in shared]
    if not deltas:
        return None
    return {
        "comparison": f"{label_a} - {label_b}",
        "mean_delta": float(np.mean(deltas)),
        "ci95": bootstrap_ci(deltas),
        "n_items": len(deltas),
        "fraction_items_lower": float(np.mean([d < 0 for d in deltas])),
    }


def summarize_alignment(records):
    out = {}
    for key in ("generated_vs_accompaniment", "reference_vs_accompaniment", "generated_vs_mismatched"):
        block = {}
        for metric in ("xcorr_peak", "xcorr_at_zero", "xcorr_lag_seconds", "beat_hit_rate"):
            values = [r[key][metric] for r in records if r[key][metric] is not None]
            if values:
                block[metric] = {"mean": float(np.mean(values)), "ci95": bootstrap_ci(values)}
        out[key] = block
    return out


# ---------------------------------------------------------------------------


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    results = {"config": vars(args), "arms": {}}

    for arm_name, model_config_path, ckpt_path in args.arm:
        print(f"\n=== Arm: {arm_name} ===")
        model, model_config = load_arm(model_config_path, ckpt_path, device, args.use_ema)

        sample_rate = model_config.get("sample_rate", 44100)
        ds_ratio = model.pretransform.downsampling_ratio
        fps = sample_rate / ds_ratio

        dataset = build_dataset_from_config(
            args.dataset_config, sample_rate, ds_ratio, args.eval_frames * ds_ratio / sample_rate
        )
        # A random crop would hand each arm (and each rerun) a different window of the same
        # track, which silently destroys the pairing the whole comparison depends on. Force
        # it off here rather than trusting every eval config to have set it.
        if getattr(dataset, "random_crop", False):
            print("  random_crop was on in the dataset config; forcing it off for evaluation")
            dataset.random_crop = False
        if args.max_items is not None:
            dataset = torch.utils.data.Subset(dataset, range(min(args.max_items, len(dataset))))

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=collation_fn,
        )

        is_streamgen = "streamgen_latent" in model.modular_local_cond_ids
        modes = ["cond", "zero", "shuffle"] if is_streamgen else ["cond"]

        arm_results = {"streamgen": is_streamgen, "checkpoint": ckpt_path, "loss": {}, "alignment": {}}

        for mode in modes:
            print(f"  loss / {mode}")
            arm_results["loss"][mode] = summarize_loss(
                eval_loss(model, loader, args, device, mode, fps)
            )

        if is_streamgen:
            for ablation in ("zero", "shuffle"):
                delta = paired_delta(
                    arm_results["loss"]["cond"], arm_results["loss"][ablation], "cond", ablation
                )
                arm_results.setdefault("ablation", {})[ablation] = delta

        if not args.skip_generation:
            for mode in modes if args.ablate_generation else ["cond"]:
                print(f"  generate / {mode}")
                records = eval_generation(
                    model, model_config, loader, args, device, mode, fps, arm_name, args.audio_dir
                )
                arm_results["alignment"][mode] = summarize_alignment(records)
                arm_results.setdefault("alignment_per_item", {})[mode] = records

        results["arms"][arm_name] = arm_results

        del model
        torch.cuda.empty_cache()

    # Cross-arm paired losses, e.g. does conditioning beat text-only on the same items.
    names = [a[0] for a in args.arm]
    if len(names) > 1:
        results["cross_arm"] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                delta = paired_delta(
                    results["arms"][names[i]]["loss"]["cond"],
                    results["arms"][names[j]]["loss"]["cond"],
                    names[i],
                    names[j],
                )
                results["cross_arm"].append(delta)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")

    print_summary(results)


def print_summary(results):
    print("\n" + "=" * 72)
    print("Held-out denoising loss (lower is better)")
    print("=" * 72)
    for name, arm in results["arms"].items():
        for mode, summary in arm["loss"].items():
            ci = summary["ci95"]
            print(f"  {name:<10} {mode:<8} {summary['mean']:.5f}  95% CI [{ci[0]:.5f}, {ci[1]:.5f}]  n={summary['n_items']}")
        for ablation, delta in (arm.get("ablation") or {}).items():
            if delta:
                print(
                    f"    ablation {delta['comparison']:<16} {delta['mean_delta']:+.5f} "
                    f"95% CI [{delta['ci95'][0]:+.5f}, {delta['ci95'][1]:+.5f}]  "
                    f"{delta['fraction_items_lower']:.0%} of items improved"
                )

    for delta in results.get("cross_arm") or []:
        if delta:
            print(
                f"  cross-arm {delta['comparison']:<20} {delta['mean_delta']:+.5f} "
                f"95% CI [{delta['ci95'][0]:+.5f}, {delta['ci95'][1]:+.5f}]  "
                f"{delta['fraction_items_lower']:.0%} of items improved"
            )

    print("\n" + "=" * 72)
    print("Onset alignment of generated drums to the accompaniment")
    print("=" * 72)
    for name, arm in results["arms"].items():
        for mode, block in arm.get("alignment", {}).items():
            print(f"  {name} / {mode}")
            for pairing, metrics in block.items():
                bits = ", ".join(f"{k}={v['mean']:.3f}" for k, v in metrics.items())
                print(f"    {pairing:<32} {bits}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Paired evaluation of streamgen conditioning (experiment 1.2)")
    p.add_argument(
        "--arm",
        nargs=3,
        action="append",
        metavar=("NAME", "MODEL_CONFIG", "CKPT"),
        required=True,
        help="An arm to evaluate: a name, its model config JSON, and a .ckpt or .safetensors",
    )
    p.add_argument("--dataset_config", required=True, help="Held-out pre-encoded dataset config")
    p.add_argument("--out", default=None, help="Where to write the results JSON")
    p.add_argument("--audio_dir", default=None, help="Where to write demo wavs (drums and drums+accompaniment)")
    p.add_argument("--max_items", type=int, default=64, help="Held-out items to score")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--eval_frames",
        type=int,
        default=256,
        help="Latent frames per evaluation window (256 ~= 23.8s at ds 4096)",
    )
    p.add_argument(
        "--context_seconds",
        type=float,
        default=8.0,
        help="Drum context given before the generation cursor",
    )
    p.add_argument(
        "--lookahead_seconds",
        type=float,
        default=0.0,
        help="Accompaniment lookahead past the cursor. 0 is the causal setting; sweep this in 1.3",
    )
    p.add_argument("--timesteps", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_generate", type=int, default=8, help="Items to generate audio for")
    p.add_argument("--gen_steps", type=int, default=50)
    p.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="CFG scale for generation. Streamgen is never CFG-dropped, so this only guides the text prompt",
    )
    p.add_argument("--use_ema", action="store_true", help="Load EMA weights from a Lightning checkpoint")
    p.add_argument("--skip_generation", action="store_true", help="Loss stage only")
    p.add_argument(
        "--ablate_generation",
        action="store_true",
        help="Also generate with the accompaniment zeroed and shuffled (slow)",
    )
    main(p.parse_args())
