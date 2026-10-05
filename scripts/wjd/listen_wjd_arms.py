"""Listen to the WJD finetune arms side by side (experiments 3.3 / 3.4).

wandb's demo table shows each run's own continuations but not the control it was given
(a 1- or 2-channel feature has no audio to upload), and the runs cannot be lined up
against each other there. This script continues the *same* held-out WJD stem chunks with
every arm's latest checkpoint and writes one self-contained HTML page: per item, the
ground-truth stem, the context prefix the model was given, the drum audio every control
was computed from, a plot of the feature controls (drum RMS, TRIA fixed, TRIA ema) with
the cursor and each lookahead horizon drawn in, and each arm's continuation at each
lookahead, alone and mixed with the drums.

Same items, same cursor, same noise for every arm and lookahead, so differences between
rows are the arm and the horizon, nothing else. The masks are built the way training
builds them (scripts/eval_streamgen.py / DiffusionCondTrainingWrapper): the context is
the prefix up to the cursor, and a control is visible up to cursor + lookahead and zero
beyond, with tf_inpaint_mask telling the model where that cut is.

Arms default to the four controlled sbatch arms (audio, rms, tria_fixed, tria_ema). Each
arm's model config and checkpoint group are read from sbatch/03_3_finetune_wjd_<arm>.sbatch
and its checkpoint is found the way the sbatch continues a run: the most recent last.ckpt
under <save_base>/<group>/<arm>/sao-3/*/checkpoints/ (EMA weights when the checkpoint has
them; the WJD jobs run train_finetune.py without --use_ema, so theirs hold raw weights only
and those are used). --arm NAME CONFIG CKPT overrides any of that; CKPT may be a Lightning .ckpt or a
.safetensors export.

Usage (from the checkout, so PYTHONPATH picks up this tree):
  PYTHONPATH=$PWD .venv/bin/python scripts/wjd/listen_wjd_arms.py \
      --out /data/scratch-fast/snnithya/sao-3/listening/wjd-arms-$(date +%F) -n 4

  # explicit checkpoints / a subset of arms
  ... --arm rms stable_audio_3/configs/model_configs/small_music_wjd_drums_rms.json /path/last.ckpt

  # rebuild only the page from the wavs and manifest already in --out (no GPU needed)
  ... --out <same dir> --page_only
"""

import argparse
import base64
import html
import io
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import soundfile
import torch
import torchaudio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from make_listening_page import AUDIO_MIME, data_uri, encode_audio, mel_db  # noqa: E402

from stable_audio_3.inference.multi_cfg import control_ids  # noqa: E402
from stable_audio_3.inference.wjd_arms import (  # noqa: E402
    DEFAULT_ARMS, DEFAULT_SAVE_BASE, discover_checkpoint, load_arm, read_sbatch_arm,
)

DEFAULT_DATASET = "stable_audio_3/configs/dataset_configs/preencoded/wjd_stems_validation_chunked_preencoded.json"
MIX_GAIN = 0.7  # as scripts/eval_streamgen.py mixes drums under a continuation

# Feature controls to plot, in sidecar order. The 256-ch drum latent is listened to, not
# plotted. Names are the dataset config's 'controls' names (what training sees).
FEATURE_CONTROLS = [("drums_rms", "drum RMS"), ("drums_tria_fixed", "TRIA fixed"), ("drums_tria_ema", "TRIA ema")]


# ---------------------------------------------------------------------------
# Arm discovery (shared code: stable_audio_3/inference/wjd_arms.py)
# ---------------------------------------------------------------------------


def resolve_arms(args):
    """[(name, model_config_path, ckpt_path)] from --arm triples or discovery."""
    if args.arm:
        return [(n, c, k) for n, c, k in args.arm]
    arms = []
    for name in args.arms:
        model_config, group = read_sbatch_arm(name)
        ckpt = discover_checkpoint(name, group, args.save_base, args.prefer)
        arms.append((name, model_config, ckpt))
    return arms


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def select_items(ds, n, seed, max_silence):
    """Dataset indices spread across the stem directories (bass / piano / guitar / other),
    skipping chunks whose target or drums are mostly silent. Seeded, so a rerun with the
    same --seed picks the same chunks."""
    rng = random.Random(seed)
    by_stem = {}
    for idx, entry in enumerate(ds.filenames):
        # get_latent_filenames scans recursively and also picks up the per-item feature
        # .npy files under <stem>/_sanity_check/, which have no metadata json; training
        # hits the same entries and substitutes a random item for them.
        if not Path(entry[1]).exists():
            continue
        by_stem.setdefault(Path(entry[0]).parent.name, []).append(idx)
    for pool in by_stem.values():
        rng.shuffle(pool)

    def acceptable(idx):
        with open(ds.filenames[idx][1]) as f:
            md = json.load(f)
        levels = md.get("levels", {})
        for key in ("target", "drums_audio"):
            frac = levels.get(key, {}).get("silence_fraction")
            if frac is not None and frac > max_silence:
                return False
        return True

    picked = []
    stems = sorted(by_stem)
    while len(picked) < n and any(by_stem[s] for s in stems):
        for stem in stems:
            while by_stem[stem]:
                idx = by_stem[stem].pop()
                if acceptable(idx):
                    picked.append(idx)
                    break
            if len(picked) >= n:
                break
    return picked


def load_batch(ds, indices):
    """Latents [B, C, T] (autoencoder space), padding mask [B, T], controls {id: [B, c, T]},
    and the metadata list the conditioner takes."""
    latents, metadata = [], []
    for idx in indices:
        x, info = ds[idx]
        latents.append(x)
        metadata.append(info)
    latents = torch.stack(latents, 0).float()
    padding = torch.stack([md["padding_mask"][0] for md in metadata], 0).float()
    controls = {}
    for key in metadata[0].get("controls", {}):
        controls[key] = torch.stack([md["controls"][key] for md in metadata], 0).float()
    return latents, padding, controls, metadata


def build_masks(padding, cursor_frames, tf_frames):
    """inpaint_mask (context up to the cursor) and tf_inpaint_mask (control visible up to
    cursor + tf), both clamped to the valid region, as random_inpaint_mask builds them for
    CAUSAL_MASK. Returns (inpaint_mask [B,1,T], tf_mask [B,1,T], cursors, horizons)."""
    B, T = padding.shape
    inpaint = torch.zeros(B, 1, T)
    tf = torch.zeros(B, 1, T)
    cursors, horizons = [], []
    for i in range(B):
        valid = int(padding[i].sum().item())
        cursor = max(0, min(valid, cursor_frames))
        horizon = max(0, min(valid, cursor + tf_frames))
        inpaint[i, :, :cursor] = 1
        tf[i, :, :horizon] = 1
        cursors.append(cursor)
        horizons.append(horizon)
    return inpaint, tf, cursors, horizons


def build_conditioning(model, metadata, latents, controls, inpaint_mask, tf_mask, device):
    """The conditioning dict training builds: inpaint conds, the tf mask, and every control
    the arm uses gated by tf_mask (_add_streamgen_conditioning's polarity)."""
    cond = model.conditioner(metadata, device)
    cond["inpaint_mask"] = [inpaint_mask.to(device)]
    cond["inpaint_masked_input"] = [(latents * inpaint_mask).to(device)]
    if "tf_inpaint_mask" in model.modular_local_cond_ids:
        cond["tf_inpaint_mask"] = [tf_mask.to(device)]
    for cid in control_ids(model):
        if cid not in controls:
            raise ValueError(f"arm needs control {cid!r} but the dataset sidecar has {sorted(controls)}")
        cond[cid] = [(controls[cid] * tf_mask).to(device)]
    return cond


@torch.no_grad()
def decode(pretransform, latents):
    with torch.amp.autocast("cuda", enabled=False):
        return pretransform.decode(latents.to(next(pretransform.parameters()).device, torch.float32)).float().cpu()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def save_wav(path, audio, sr):
    torchaudio.save(str(path), audio.clamp(-1, 1), sr)


def generate(args, arms):
    from stable_audio_3.data.utils import build_dataset_from_config
    from stable_audio_3.inference.sampling import sample_diffusion

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    out = Path(args.out)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    (out / "features").mkdir(exist_ok=True)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset_config": args.dataset_config,
        "settings": {
            "steps": args.steps, "cfg_scale": args.cfg_scale, "seed": args.seed,
            "context_seconds": args.context_seconds, "tf_seconds": args.tf_seconds,
            "weights": "ema" if args.use_ema else "raw",
        },
        "arms": [],
        "items": [],
    }

    sample_rate = fps = ds_ratio = None
    ds = None
    indices = None
    refs_done = False
    noise = None

    for arm_name, model_config_path, ckpt_path in arms:
        t0 = time.time()
        print(f"\n=== arm {arm_name}: {ckpt_path}")
        model, model_config, step = load_arm(model_config_path, ckpt_path, device, args.use_ema)
        print(f"  loaded (step {step}) in {time.time() - t0:.0f}s; controls: {control_ids(model) or 'none'}")
        manifest["arms"].append({"name": arm_name, "model_config": model_config_path,
                                 "checkpoint": ckpt_path, "step": step})

        if ds is None:
            sample_rate = model_config.get("sample_rate", 44100)
            ds_ratio = int(model.pretransform.downsampling_ratio)
            fps = sample_rate / ds_ratio
            with open(args.dataset_config) as f:
                crop = json.load(f).get("latent_crop_length")
            ds = build_dataset_from_config(args.dataset_config, sample_rate, ds_ratio,
                                           None if crop else args.frames * ds_ratio / sample_rate)
            ds.random_crop = False
            indices = args.indices or select_items(ds, args.num_items, args.seed, args.max_silence)
            print(f"  items: {indices}")
            latents, padding, controls, metadata = load_batch(ds, indices)
            cursor_frames = int(round(args.context_seconds * fps))
            tf_frames = [int(v * fps) for v in args.tf_seconds]  # as the training wrapper converts
            masks = {tf: build_masks(padding, cursor_frames, f) for tf, f in zip(args.tf_seconds, tf_frames)}
            # One noise per item, reused by every arm and lookahead.
            noise = torch.stack([
                torch.randn(model.io_channels, latents.shape[-1],
                            generator=torch.Generator().manual_seed(args.seed + 10_000 + i))
                for i in range(len(indices))
            ], 0)
            manifest["sample_rate"] = sample_rate
            manifest["fps"] = fps
            for k, (idx, md) in enumerate(zip(indices, metadata)):
                item = {
                    "k": k, "index": idx,
                    "latent_file": str(md.get("latent_filename", "")),
                    "track": md.get("track"), "stem": md.get("stem"), "prompt": md.get("prompt"),
                    "chunk_index": md.get("chunk_index"), "n_chunks": md.get("n_chunks"),
                    "chunk_offset_seconds": md.get("chunk_offset_seconds"),
                    "seconds_total": float(md.get("seconds_total", 0.0)),
                    "cursor_frames": masks[args.tf_seconds[0]][2][k],
                    "horizon_frames": {str(tf): masks[tf][3][k] for tf in args.tf_seconds},
                    "levels": md.get("levels"),
                    "wavs": {},
                }
                manifest["items"].append(item)

        scale = getattr(model.pretransform, "scale", 1.0)
        latents_m = latents / scale
        drums = controls.get("streamgen_latent")
        drums_m = drums / scale if drums is not None else None

        if not refs_done:
            print("  decoding references")
            target = decode(model.pretransform, latents_m)
            # The zero latent after the cursor does not decode to silence (the autoencoder
            # turns it into a -30 dBFS hiss), so mute the decoded prefix there: the row is
            # the context the model was given, and after the cursor that is nothing.
            prefix = decode(model.pretransform, latents_m * masks[args.tf_seconds[0]][0])
            for k, cursor in enumerate(masks[args.tf_seconds[0]][2]):
                prefix[k, :, cursor * ds_ratio:] = 0
            drums_audio = decode(model.pretransform, drums_m) if drums_m is not None else None
            for k, item in enumerate(manifest["items"]):
                stem = f"item{k:02d}"
                save_wav(wav_dir / f"{stem}_target.wav", target[k], sample_rate)
                item["wavs"]["target"] = f"{stem}_target.wav"
                save_wav(wav_dir / f"{stem}_prefix.wav", prefix[k], sample_rate)
                item["wavs"]["prefix"] = f"{stem}_prefix.wav"
                if drums_audio is not None:
                    save_wav(wav_dir / f"{stem}_drums.wav", drums_audio[k], sample_rate)
                    item["wavs"]["drums"] = f"{stem}_drums.wav"
                    if not args.no_mix:
                        mix = MIX_GAIN * (target[k] + drums_audio[k])
                        save_wav(wav_dir / f"{stem}_target+drums.wav", mix, sample_rate)
                        item["wavs"]["target+drums"] = f"{stem}_target+drums.wav"
                plot_features(out / "features" / f"{stem}.png", controls, k, item, fps)
                item["features_png"] = f"{stem}.png"
            refs_done = True

        for tf in args.tf_seconds:
            inpaint, tfm, _, _ = masks[tf]
            cond = build_conditioning(model, metadata, latents_m, controls, inpaint, tfm, device)
            cond_inputs = model.get_conditioning_inputs(cond)
            print(f"  generating tf={tf:+.1f}s ({len(indices)} items, {args.steps} steps, cfg {args.cfg_scale})")
            with torch.amp.autocast("cuda"):
                fakes = sample_diffusion(
                    model=model.model,
                    noise=noise.to(device, torch.bfloat16),
                    cond_inputs=cond_inputs,
                    diffusion_objective=model.diffusion_objective,
                    steps=args.steps,
                    cfg_scale=args.cfg_scale,
                    conditioning=metadata,
                    sample_rate=sample_rate,
                    pretransform=model.pretransform,
                    mask_padding_attention=model.mask_padding_attention,
                    use_effective_length_for_schedule=model.use_effective_length_for_schedule,
                    padding_mask=padding.to(device),
                    dist_shift=model.sampling_dist_shift,
                    batch_cfg=True,
                    disable_tqdm=True,
                    decode=True,
                )
            fakes = fakes.float().cpu()
            for k, item in enumerate(manifest["items"]):
                label = f"{arm_name}_tf{tf:+g}"
                stem = f"item{k:02d}"
                save_wav(wav_dir / f"{stem}_{label}.wav", fakes[k], sample_rate)
                item["wavs"][label] = f"{stem}_{label}.wav"
                if drums_audio is not None and not args.no_mix:
                    n = min(fakes.shape[-1], drums_audio.shape[-1])
                    mix = MIX_GAIN * (fakes[k, :, :n] + drums_audio[k, :, :n])
                    save_wav(wav_dir / f"{stem}_{label}+drums.wav", mix, sample_rate)
                    item["wavs"][label + "+drums"] = f"{stem}_{label}+drums.wav"
            del cond, cond_inputs, fakes
            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  arm done in {time.time() - t0:.0f}s")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ---------------------------------------------------------------------------
# Feature plot
# ---------------------------------------------------------------------------


def plot_features(path, controls, k, item, fps):
    """The feature controls of one item against time, with the cursor (solid) and each
    lookahead horizon (dashed) marked; the control is zeroed to the right of the horizon
    before the model sees it."""
    panels = [(cid, label) for cid, label in FEATURE_CONTROLS if cid in controls]
    if not panels:
        return
    T = next(iter(controls.values())).shape[-1]
    t = np.arange(T) / fps
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 1.6 * len(panels)), sharex=True,
                             facecolor="#1c1917")
    axes = np.atleast_1d(axes)
    colors = ["#f7a76c", "#8ecae6"]
    cursor = item["cursor_frames"] / fps
    for ax, (cid, label) in zip(axes, panels):
        ax.set_facecolor("#12100f")
        x = controls[cid][k].numpy()
        for c in range(x.shape[0]):
            name = label if x.shape[0] == 1 else f"{label} {'low' if c == 0 else 'high'}"
            ax.step(t, x[c], where="post", color=colors[c % 2], lw=1.1, label=name)
        ax.axvline(cursor, color="#ffffff", lw=1.0)
        for tf, h in item["horizon_frames"].items():
            ax.axvline(h / fps, color="#f06060", lw=1.0, ls="--")
        ax.axvspan(cursor, t[-1] + 1 / fps, color="#ffffff", alpha=0.06, lw=0)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, t[-1] + 1 / fps)
        ax.tick_params(colors="#a29a93", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#2f2a27")
        ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#f0ece8", ncol=2)
    axes[-1].set_xlabel("seconds", color="#a29a93", fontsize=8)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def row_order(manifest):
    """Reference rows first, then per lookahead each arm's output and its drum mix."""
    arms = [a["name"] for a in manifest["arms"]]
    tfs = manifest["settings"]["tf_seconds"]
    refs = ["target", "target+drums", "prefix", "drums"]
    groups = [("references", refs)]
    for tf in tfs:
        labels = []
        for arm in arms:
            labels.append(f"{arm}_tf{tf:+g}")
            labels.append(f"{arm}_tf{tf:+g}+drums")
        groups.append((f"lookahead {tf:+g} s", labels))
    return groups


ROW_NOTES = {
    "target": "ground-truth stem (the whole chunk; the model regenerates everything after the cursor)",
    "target+drums": "ground truth under the drums, for reference",
    "prefix": "context given to every arm: the stem up to the cursor (muted after it; the model sees a zero latent there)",
    "drums": "drum stem every control was computed from (the audio arm hears its latent)",
}


def build_page(args, manifest):
    out = Path(args.out)
    wav_dir = out / "wavs"
    arms = {a["name"]: a for a in manifest["arms"]}
    s = manifest["settings"]
    fps = manifest.get("fps", 44100 / 4096)

    wavs = sorted(wav_dir.glob("*.wav"))
    print(f"mel spectrograms for {len(wavs)} wavs")
    specs = {w.name: mel_db(w, args.n_mels, args.max_width) for w in wavs}
    vmax = max(float(db.max()) for db, _ in specs.values())
    vmin = vmax - args.dynamic_range
    mel_src, audio_src = {}, {}
    print(f"inlining audio as {args.embed}")
    for w in wavs:
        db, _ = specs[w.name]
        buf = io.BytesIO()
        plt.imsave(buf, db, cmap="magma", origin="lower", vmin=vmin, vmax=vmax, format="png")
        mel_src[w.name] = data_uri("image/png", buf.getvalue())
        try:
            payload = encode_audio(w, args.embed)
            mime = AUDIO_MIME[args.embed]
        except Exception as e:  # libsndfile without mp3 -> lossless fallback
            print(f"  {args.embed} failed for {w.name} ({e}); using flac")
            payload = encode_audio(w, "flac")
            mime = AUDIO_MIME["flac"]
        audio_src[w.name] = data_uri(mime, payload)

    cards = []
    for item in manifest["items"]:
        cursor_s = item["cursor_frames"] / fps
        horizons = ", ".join(f"{tf} s → control cut at {h / fps:.1f} s"
                             for tf, h in item["horizon_frames"].items())
        lv = item.get("levels") or {}
        levels = " · ".join(f"{k}: {v.get('rms_dbfs', float('nan')):.0f} dBFS, {v.get('silence_fraction', 0):.0%} silent"
                            for k, v in lv.items())
        meta = (f"prompt: <b>{html.escape(str(item.get('prompt')))}</b> · {html.escape(str(item.get('track')))} · "
                f"chunk {item.get('chunk_index')}/{item.get('n_chunks')} @ {item.get('chunk_offset_seconds', 0):.1f} s · "
                f"dataset index {item['index']}<br>"
                f"cursor at {cursor_s:.1f} s (context = first {cursor_s:.1f} s) · lookahead: {html.escape(horizons)}<br>"
                f"{html.escape(levels)}")

        sections = []
        for title, labels in row_order(manifest):
            rows = []
            for label in labels:
                name = item["wavs"].get(label)
                if not name:
                    continue
                duration = specs[name][1]
                arm = label.split("_tf")[0] if "_tf" in label else None
                note = ROW_NOTES.get(label)
                if arm is not None:
                    a = arms.get(arm, {})
                    note = f"step {a.get('step')} · {Path(str(a.get('checkpoint'))).name}"
                    if label.endswith("+drums"):
                        note = "continuation mixed with the drums"
                note_html = f'<span class="note">{html.escape(note)}</span>' if note else ""
                rows.append(f"""
      <div class="row" data-duration="{duration:.4f}">
        <div class="label">{html.escape(label)}<span class="dur">{duration:.1f}s</span>{note_html}</div>
        <audio preload="none" controls src="{audio_src[name]}"></audio>
        <div class="spec"><img src="{mel_src[name]}" alt="mel spectrogram">
          <div class="cursor" style="left:{100 * cursor_s / duration:.2f}%"></div>
          <div class="playhead"></div></div>
      </div>""")
            if rows:
                sections.append(f'<h3>{html.escape(title)}</h3>{"".join(rows)}')

        feat = ""
        png = item.get("features_png")
        if png and (out / "features" / png).exists():
            feat = f'<img class="features" src="{data_uri("image/png", (out / "features" / png).read_bytes())}" alt="feature controls">'

        cards.append(f"""
    <section class="card">
      <h2>item {item['k']:02d} — {html.escape(str(item.get('stem')))} / {html.escape(str(item.get('track')))}</h2>
      <p class="meta">{meta}</p>
      {feat}
      {''.join(sections)}
    </section>""")

    arm_lines = "".join(
        f"<li><b>{html.escape(a['name'])}</b> step {a.get('step')} · {html.escape(Path(a['model_config']).name)} · "
        f"{html.escape(a['checkpoint'])}</li>" for a in manifest["arms"])
    page = PAGE.format(
        title="WJD arms listening page",
        created=html.escape(manifest.get("created", "")),
        dataset=html.escape(manifest.get("dataset_config", "")),
        settings=html.escape(f"{s['steps']} steps · cfg {s['cfg_scale']} · seed {s['seed']} · "
                             f"context {s['context_seconds']} s · lookahead {s['tf_seconds']} s · {s['weights']} weights · "
                             f"mel {vmin:.0f} to {vmax:.0f} dB"),
        arms=arm_lines,
        cards="".join(cards),
    )
    page_path = out / "index.html"
    page_path.write_text(page)
    print(f"\nWrote {page_path}  ({page_path.stat().st_size / 1e6:.1f} MB)")
    print("Self-contained: open over file://, or scp this one file to your laptop.")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --bg: #12100f; --panel: #1c1917; --line: #2f2a27; --fg: #f0ece8; --dim: #a29a93; --accent: #f7a76c; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  header {{ position: sticky; top: 0; z-index: 10; padding: 12px 20px; background: rgba(18,16,15,.94);
    border-bottom: 1px solid var(--line); backdrop-filter: blur(6px); }}
  h1 {{ margin: 0; font-size: 15px; font-weight: 600; }}
  header p, header li {{ margin: 3px 0 0; color: var(--dim); font-size: 11.5px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-all; }}
  header ul {{ margin: 4px 0 0; padding-left: 18px; }}
  main {{ padding: 20px; max-width: 1400px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }}
  .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px 16px; }}
  h2 {{ margin: 0 0 2px; font-size: 14px; font-family: ui-monospace, Menlo, monospace; color: var(--accent); }}
  h3 {{ margin: 14px 0 2px; font-size: 12px; color: var(--dim); text-transform: uppercase; letter-spacing: .06em; }}
  .meta {{ margin: 0 0 10px; color: var(--dim); font-size: 12px; word-break: break-word; }}
  .meta b {{ color: var(--fg); }}
  .features {{ width: 100%; border-radius: 6px; display: block; margin-bottom: 6px; }}
  .row {{ display: grid; grid-template-columns: 230px 260px 1fr; gap: 12px; align-items: center;
    padding: 6px 0; border-top: 1px solid var(--line); }}
  .label {{ font-family: ui-monospace, Menlo, monospace; font-size: 12px; display: flex; flex-direction: column; }}
  .dur {{ color: var(--dim); font-size: 11px; }}
  .note {{ color: var(--dim); font-size: 10.5px; line-height: 1.35; margin-top: 2px; }}
  audio {{ width: 100%; height: 34px; }}
  .spec {{ position: relative; height: 70px; border-radius: 5px; overflow: hidden; background: #000; cursor: pointer; }}
  .spec img {{ width: 100%; height: 100%; display: block; object-fit: fill; }}
  .cursor {{ position: absolute; top: 0; bottom: 0; width: 0; border-left: 1px dashed rgba(255,255,255,.7); pointer-events: none; }}
  .playhead {{ position: absolute; top: 0; bottom: 0; width: 1px; left: 0; background: #fff; box-shadow: 0 0 6px #fff;
    opacity: 0; pointer-events: none; }}
  .playhead.on {{ opacity: .9; }}
  @media (max-width: 900px) {{ .row {{ grid-template-columns: 1fr; }} .spec {{ height: 100px; }} }}
</style>
</head>
<body>
<header>
  <h1>{title} — generated {created}</h1>
  <p>{dataset}</p>
  <p>{settings} · dashed white line on the spectrograms = cursor · click a spectrogram to seek</p>
  <ul>{arms}</ul>
</header>
<main>
{cards}
</main>
<script>
  document.querySelectorAll('.row').forEach(row => {{
    const audio = row.querySelector('audio');
    const spec = row.querySelector('.spec');
    const head = row.querySelector('.playhead');
    const duration = parseFloat(row.dataset.duration);
    const move = () => {{ const d = audio.duration || duration; head.style.left = (100 * audio.currentTime / d) + '%'; }};
    audio.addEventListener('timeupdate', move);
    audio.addEventListener('play', () => {{ head.classList.add('on'); move(); }});
    audio.addEventListener('seeked', move);
    audio.addEventListener('ended', () => head.classList.remove('on'));
    spec.addEventListener('click', e => {{
      const r = spec.getBoundingClientRect();
      const frac = (e.clientX - r.left) / r.width;
      const seek = () => {{ audio.currentTime = (audio.duration || duration) * frac; head.classList.add('on'); move(); audio.play(); }};
      if (audio.readyState === 0) {{ audio.addEventListener('loadedmetadata', seek, {{ once: true }}); audio.load(); }}
      else {{ seek(); }}
    }});
  }});
  // One clip at a time, so A/B stays honest.
  document.querySelectorAll('audio').forEach(a => {{
    a.addEventListener('play', () => {{ document.querySelectorAll('audio').forEach(o => {{ if (o !== a) o.pause(); }}); }});
  }});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Output directory (wavs/, features/, manifest.json, index.html)")
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                   help="Arms to discover from sbatch/03_3_finetune_wjd_<arm>.sbatch (default: %(default)s)")
    p.add_argument("--arm", nargs=3, action="append", metavar=("NAME", "MODEL_CONFIG", "CKPT"),
                   help="Explicit arm; repeatable. Replaces --arms discovery entirely.")
    p.add_argument("--save_base", default=DEFAULT_SAVE_BASE, help="Checkpoint root the sbatch jobs write to")
    p.add_argument("--prefer", choices=["ckpt", "safetensors"], default="ckpt",
                   help="Which discovered checkpoint kind to use when both exist (ckpt carries the EMA weights)")
    p.add_argument("--raw_weights", dest="use_ema", action="store_false",
                   help="Use the raw weights from a .ckpt instead of the EMA shadow")
    p.add_argument("--dataset_config", default=DEFAULT_DATASET)
    p.add_argument("-n", "--num_items", type=int, default=4, help="Items to continue (spread over the stem dirs)")
    p.add_argument("--indices", type=int, nargs="+", default=None, help="Explicit dataset indices instead of -n")
    p.add_argument("--max_silence", type=float, default=0.3,
                   help="Skip chunks whose target or drums have a larger silence fraction (sidecar 'levels')")
    p.add_argument("--frames", type=int, default=130, help="Latent crop if the dataset config has none")
    p.add_argument("--context_seconds", type=float, default=6.0, help="Cursor position: the context given to the model")
    p.add_argument("--tf_seconds", type=float, nargs="+", default=[-4.0, 0.0],
                   help="Lookahead horizons relative to the cursor (training samples from [-4, 0])")
    p.add_argument("--steps", type=int, default=8, help="Sampling steps (the wandb demos use 8)")
    p.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale (the wandb demos use 4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_mix", action="store_true", help="Skip the +drums mixes")
    p.add_argument("--page_only", action="store_true", help="Rebuild index.html from an existing --out")
    p.add_argument("--embed", choices=["mp3", "flac", "wav"], default="mp3",
                   help="Audio codec inlined in the page (mp3 keeps a page of 4 items x 4 arms around 20 MB)")
    p.add_argument("--n_mels", type=int, default=96)
    p.add_argument("--max_width", type=int, default=1200)
    p.add_argument("--dynamic_range", type=float, default=80.0)
    args = p.parse_args()

    if args.page_only:
        manifest = json.loads((Path(args.out) / "manifest.json").read_text())
    else:
        arms = resolve_arms(args)
        for name, cfg, ckpt in arms:
            print(f"arm {name:<11} {cfg}  {ckpt}")
        manifest = generate(args, arms)
    build_page(args, manifest)


if __name__ == "__main__":
    main()
