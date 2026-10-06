"""Gradio interface for the WJD drum-control arms with three-axis CFG (experiments 3.3 / 3.4).

Pick an arm (``rms``, ``tria_fixed``, ``tria_ema``, ``audio``: the newest ``last.ckpt`` of each,
EMA weights, loaded on first use and kept on the GPU), give it drums and a stem to continue,
and set a guidance scale on each of the three conditions the arms were trained to drop:
the prompt, the inpainting context and the drum control.  The composition is the nested CFG
of ``stable_audio_3/inference/multi_cfg.py``; the conditioning is built exactly as the
training step and ``scripts/wjd/listen_wjd_arms.py`` build it (context = the stem up to the
cursor, control visible up to cursor + lookahead and 0 beyond, ``tf_inpaint_mask`` marking
the cut).

Two input sources:

* **WJD validation track** -- a held-out track from the stem mirror.  Drums come from
  ``tracks/drums/<Track>/drums.flac``, the RMS / TRIA controls are sliced from the per-song
  ``drums_features.npz`` (what training saw, including the TRIA ema's running statistics
  from the song start), the stem from ``tracks/<stem>/<Track>/<stem>.flac`` and the prompt
  from ``meta/<Track>.json``.
* **Upload audio** -- your own drum clip and, optionally, a stem to continue.  The controls
  are computed from the clip (``stable_audio_3/data/features.py``); the TRIA ema statistics
  warm-start from the dataset constants at the clip start rather than from a song's history.

**SDEdit** (optional): instead of starting from pure noise, the sampler starts from the latent
of the streamgen input (the drum audio) or of the stem to continue, noised to a chosen level
``sigma`` -- ``x = (1 - sigma) * z + sigma * noise`` -- and denoises from ``sigma`` to 0 under the
same conditioning (``sample_diffusion``'s ``init_data`` / ``init_noise_level``).

Launch with ``run_gradio_wjd.py``.
"""

from __future__ import annotations

import gc
import io
import json
import re
import threading
import time
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import torch
import torchaudio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from stable_audio_3.data.features import (  # noqa: E402
    LATENT_HOP,
    TriaStats,
    rms_envelope_control,
    tria_control,
)
from stable_audio_3.inference.audio_utils import numpy_audio_to_tensor  # noqa: E402
from stable_audio_3.inference.multi_cfg import (  # noqa: E402
    AXES,
    DEFAULT_ORDER,
    ORDERS,
    control_ids,
    describe_branches,
    make_multi_cfg_denoiser,
)
from stable_audio_3.inference.sampling import sample_diffusion  # noqa: E402
from stable_audio_3.inference.wjd_arms import (  # noqa: E402
    DEFAULT_SAVE_BASE,
    load_arm,
    resolve_arm,
    share_frozen_parts,
)

DEFAULT_MIRROR = "/data/hai-res/shared/snnithya/sao-3/data/wjd/wjd-stem-mirror"
DEFAULT_SPLIT = "validation"
DEFAULT_OUT_DIR = "/data/scratch-fast/snnithya/sao-3/gradio-wjd"
FEATURES_FILENAME = "drums_features.npz"  # scripts/wjd/compute_drum_features.py
MIX_GAIN = 0.7  # as scripts/eval_streamgen.py mixes drums under a continuation
TRAINED_LOOKAHEAD = (-4.0, 0.0)  # inpainting.future_visibility of every arm, seconds
FEATURE_CONTROLS = [("drums_rms", "drum RMS"), ("drums_tria_fixed", "TRIA fixed"), ("drums_tria_ema", "TRIA ema")]
SOURCE_PICKER = "WJD validation track"
SOURCE_UPLOAD = "Upload audio"
SAMPLERS = ["pingpong", "euler", "rk4", "dpmpp"]
SDEDIT_SOURCES = ["drums", "stem"]  # what SDEdit noises: the streamgen input or the stem to continue


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


@dataclass
class LoadedArm:
    name: str
    model_config_path: str
    ckpt_path: str
    ckpt_mtime: float
    step: tp.Optional[int]
    weights: str  # "ema" or "raw": what the checkpoint actually held
    model: tp.Any
    model_config: dict
    controls: tp.List[str]
    loaded_at: float


class ArmRegistry:
    """The arms the UI can switch between: resolved lazily, loaded on first use, kept resident.

    ``specs`` maps an arm name to ``(model_config_path, ckpt_path)``; ``None`` for either means
    "discover from the sbatch file / the newest last.ckpt" at load time, so a reload after the
    training job has refreshed its checkpoint picks the new weights up.
    """

    def __init__(self, specs: tp.Dict[str, tp.Tuple[tp.Optional[str], tp.Optional[str]]], device,
                 use_ema: bool = True, save_base: str = DEFAULT_SAVE_BASE, log=print):
        self.specs = dict(specs)
        self.device = device
        self.use_ema = use_ema
        self.save_base = save_base
        self.log = log
        self.loaded: tp.Dict[str, LoadedArm] = {}
        self._shared = None  # (pretransform, conditioner) of the first arm loaded
        self.lock = threading.Lock()

    def names(self):
        return list(self.specs)

    def resolve(self, name):
        if name not in self.specs:
            raise KeyError(f"unknown arm {name!r}; have {self.names()}")
        config, ckpt = self.specs[name]
        if config is None or ckpt is None:
            d_config, d_ckpt = resolve_arm(name, self.save_base)
            config = config or d_config
            ckpt = ckpt or d_ckpt
        return config, ckpt

    def newer_available(self, name) -> tp.Optional[str]:
        """The newest checkpoint path if it differs from the loaded one (path or mtime), else None."""
        cur = self.loaded.get(name)
        if cur is None:
            return None
        try:
            _, ckpt = self.resolve(name)
        except (FileNotFoundError, KeyError, ValueError):
            return None
        if ckpt != cur.ckpt_path or Path(ckpt).stat().st_mtime != cur.ckpt_mtime:
            return ckpt
        return None

    def get(self, name, reload: bool = False) -> LoadedArm:
        with self.lock:
            config, ckpt = self.resolve(name)
            mtime = Path(ckpt).stat().st_mtime
            cur = self.loaded.get(name)
            if cur is not None and not reload and cur.ckpt_path == ckpt and cur.ckpt_mtime == mtime:
                return cur
            if cur is not None:
                self.log(f"[{name}] dropping the loaded checkpoint ({cur.ckpt_path})")
                del self.loaded[name]
                del cur
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            t0 = time.time()
            self.log(f"[{name}] loading {ckpt} ({'EMA' if self.use_ema else 'raw'} weights)")
            info = {}
            model, model_config, step = load_arm(config, ckpt, self.device, self.use_ema, log=self.log, info=info)
            if self._shared is None:
                self._shared = (model.pretransform, model.conditioner)
            else:
                donor = type("Donor", (), {"pretransform": self._shared[0], "conditioner": self._shared[1]})()
                share_frozen_parts(model, donor)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            arm = LoadedArm(
                name=name, model_config_path=config, ckpt_path=ckpt, ckpt_mtime=mtime, step=step,
                weights=info.get("weights", "raw"), model=model, model_config=model_config, controls=control_ids(model), loaded_at=time.time(),
            )
            self.loaded[name] = arm
            self.log(f"[{name}] loaded step {step} ({arm.weights} weights) in {time.time() - t0:.0f}s; "
                     f"controls: {arm.controls or 'none'}")
            return arm

    def status_markdown(self, name) -> str:
        arm = self.loaded.get(name)
        if arm is None:
            try:
                config, ckpt = self.resolve(name)
            except Exception as e:  # noqa: BLE001 - shown to the user
                return f"**{name}**: not loaded; cannot resolve a checkpoint ({e})"
            return (f"**{name}**: not loaded (loads on the first Generate or on *Load / reload*)  \n"
                    f"`{ckpt}`  \n"
                    f"modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(Path(ckpt).stat().st_mtime))}")
        weights = {"ema": "EMA", "raw": "raw"}[arm.weights]
        if self.use_ema and arm.weights == "raw":
            weights = "raw (the checkpoint holds no EMA)"
        lines = [f"**{name}**: step {arm.step}, {weights} weights, "
                 f"controls: {', '.join(arm.controls) or 'none (text only)'}",
                 f"`{arm.ckpt_path}`",
                 f"checkpoint modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(arm.ckpt_mtime))}, "
                 f"loaded {time.strftime('%H:%M', time.localtime(arm.loaded_at))}"]
        newer = self.newer_available(name)
        if newer:
            lines.append(f"**A newer checkpoint exists** (`{newer}`): press *Load / reload* to use it.")
        return "  \n".join(lines)


# ---------------------------------------------------------------------------
# WJD track catalog
# ---------------------------------------------------------------------------


@dataclass
class Track:
    name: str
    drums_path: Path
    features_path: tp.Optional[Path]
    duration: float
    stems: tp.Dict[str, tp.Tuple[Path, str]]  # stem -> (path, prompt)


def load_catalog(mirror_root=DEFAULT_MIRROR, split=DEFAULT_SPLIT) -> tp.Dict[str, Track]:
    """Tracks of ``<mirror_root>/<split>`` that have drums and at least one target stem on disk."""
    root = Path(mirror_root) / split
    catalog = {}
    for meta_path in sorted((root / "meta").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        track = meta.get("track", meta_path.stem)
        drums = root / "tracks" / "drums" / track / "drums.flac"
        if not drums.exists():
            continue
        stems = {}
        for stem, prompt in (meta.get("prompts") or {}).items():
            path = root / "tracks" / stem / track / f"{stem}.flac"
            if prompt and path.exists():
                stems[stem] = (path, prompt)
        if not stems:
            continue
        feats = drums.parent / FEATURES_FILENAME
        catalog[track] = Track(track, drums, feats if feats.exists() else None,
                               float(meta.get("duration_sec", 0.0)), stems)
    return catalog


# ---------------------------------------------------------------------------
# Audio and controls
# ---------------------------------------------------------------------------


def _to_stereo(audio):
    if audio.shape[0] == 1:
        return audio.repeat(2, 1)
    return audio[:2]


def _fit(audio, offset, n_samples):
    audio = audio[:, offset:offset + n_samples]
    if audio.shape[1] < n_samples:
        audio = torch.nn.functional.pad(audio, (0, n_samples - audio.shape[1]))
    return audio


def load_window(path, sample_rate, offset, n_samples):
    """``[2, n_samples]`` float32 of a file at ``sample_rate`` from sample ``offset``, zero-padded."""
    info = torchaudio.info(str(path))
    if info.sample_rate == sample_rate:
        audio, _ = torchaudio.load(str(path), frame_offset=offset, num_frames=n_samples)
        audio = _fit(audio, 0, n_samples)
    else:
        audio, sr = torchaudio.load(str(path))
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
        audio = _fit(audio, offset, n_samples)
    return _to_stereo(audio.float())


def upload_window(audio_in, sample_rate, offset, n_samples):
    """A Gradio ``(sr, ndarray)`` as ``[2, n_samples]`` float32 at ``sample_rate`` from ``offset``."""
    in_sr, data = audio_in
    audio = numpy_audio_to_tensor(np.asarray(data)) if isinstance(data, np.ndarray) else data
    if in_sr != sample_rate:
        audio = torchaudio.functional.resample(audio.float(), in_sr, sample_rate)
    return _to_stereo(_fit(audio.float(), offset, n_samples))


def controls_from_audio(drums, stats: TriaStats):
    """All three feature controls of a drum clip, ``{id: [C, T]}`` (hop = the latent hop)."""
    return {
        "drums_rms": rms_envelope_control(drums, hop=LATENT_HOP),
        "drums_tria_fixed": tria_control(drums, stats, norm="fixed", hop=LATENT_HOP),
        "drums_tria_ema": tria_control(drums, stats, norm="ema", hop=LATENT_HOP),
    }


def controls_from_npz(features_path, first_frame, n_frames):
    """The feature controls of one window out of a song's ``drums_features.npz``, as
    ``custom_md_wjd.window_frames`` slices them for training (zero past the song's end)."""
    out = {}
    with np.load(features_path) as npz:
        for key, _ in FEATURE_CONTROLS:
            if key not in npz.files:
                continue
            feat = torch.from_numpy(npz[key])
            win = torch.zeros(feat.shape[0], n_frames, dtype=feat.dtype)
            avail = feat[:, first_frame:first_frame + n_frames]
            win[:, :avail.shape[1]] = avail
            out[key] = win.float()
    return out


def build_masks(n_frames, cursor_frames, lookahead_frames):
    """``(inpaint_mask, tf_mask)`` both ``[1, 1, T]``: context up to the cursor, control visible
    up to cursor + lookahead, as ``random_inpaint_mask`` builds them for CAUSAL_MASK."""
    cursor = max(0, min(n_frames, cursor_frames))
    horizon = max(0, min(n_frames, cursor + lookahead_frames))
    inpaint = torch.zeros(1, 1, n_frames)
    tf = torch.zeros(1, 1, n_frames)
    inpaint[:, :, :cursor] = 1
    tf[:, :, :horizon] = 1
    return inpaint, tf, cursor, horizon


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode(model, audio, device):
    pt = model.pretransform
    with torch.amp.autocast("cuda", enabled=False):
        return pt.encode(audio[None].to(device, torch.float32)).float()


@torch.no_grad()
def decode(model, latents):
    pt = model.pretransform
    with torch.amp.autocast("cuda", enabled=False):
        return pt.decode(latents.to(next(pt.parameters()).device, torch.float32)).float().cpu()


@torch.no_grad()
def generate_continuation(
    arm: LoadedArm,
    prompt: str,
    drums: tp.Optional[torch.Tensor],
    stem: tp.Optional[torch.Tensor],
    controls: tp.Dict[str, torch.Tensor],
    cursor_seconds: float,
    lookahead_seconds: float,
    scales: tp.Dict[str, float],
    order: tp.Sequence[str],
    steps: int,
    sampler_type: str,
    seed: int,
    sdedit_noise_level: tp.Optional[float] = None,
    sdedit_source: str = "drums",
):
    """One continuation. ``drums`` / ``stem`` are ``[2, N]`` at the model rate with ``N`` a
    multiple of the latent hop; ``controls`` holds the feature controls ``[C, T]`` the arm may
    use. ``sdedit_noise_level`` (in (0, 1]) starts sampling from the latent of ``sdedit_source``
    (``"drums"`` or ``"stem"``, the whole window) noised to that level instead of from pure noise
    (SDEdit); ``None`` = pure noise.
    Returns ``(audio [2, N], info dict)``."""
    model = arm.model
    device = next(model.model.parameters()).device
    sr = arm.model_config["sample_rate"]
    ds = int(model.pretransform.downsampling_ratio)
    fps = sr / ds
    ref = drums if drums is not None else stem
    if ref is None:
        raise ValueError("need drums or a stem to know the window length")
    n_samples = ref.shape[-1]
    n_frames = n_samples // ds
    seconds_total = n_frames * ds / sr

    cursor_frames = int(round(cursor_seconds * fps))
    lookahead_frames = int(lookahead_seconds * fps)  # as the training wrapper converts future_visibility
    inpaint_mask, tf_mask, cursor, horizon = build_masks(n_frames, cursor_frames, lookahead_frames)
    inpaint_mask, tf_mask = inpaint_mask.to(device), tf_mask.to(device)

    sdedit_drums = sdedit_noise_level is not None and sdedit_source == "drums"
    if sdedit_noise_level is not None:
        if sdedit_source not in SDEDIT_SOURCES:
            raise ValueError(f"SDEdit source must be one of {SDEDIT_SOURCES}, got {sdedit_source!r}")
        if sdedit_source == "drums" and drums is None:
            raise ValueError("SDEdit from drums needs drum audio (the streamgen input is what gets noised)")
        if sdedit_source == "stem" and stem is None:
            raise ValueError("SDEdit from the stem needs a stem to continue")
        if not 0.0 < sdedit_noise_level <= 1.0:
            raise ValueError(f"SDEdit noise level must be in (0, 1], got {sdedit_noise_level}")
    drums_latent = None
    if drums is not None and (sdedit_drums or "streamgen_latent" in arm.controls):
        drums_latent = encode(model, drums, device)[..., :n_frames]

    if stem is not None:
        stem_latent = encode(model, stem, device)[..., :n_frames]
    else:
        stem_latent = torch.zeros(1, model.io_channels, n_frames, device=device)

    cond = model.conditioner([{"prompt": prompt, "seconds_total": seconds_total}], device)
    cond["inpaint_mask"] = [inpaint_mask]
    cond["inpaint_masked_input"] = [stem_latent * inpaint_mask]
    if "tf_inpaint_mask" in model.modular_local_cond_ids:
        cond["tf_inpaint_mask"] = [tf_mask]
    for cid in arm.controls:
        if cid == "streamgen_latent":
            if drums is None:
                raise ValueError("the audio arm needs drum audio (its control is the drum latent)")
            ctrl = drums_latent
        elif cid in controls:
            ctrl = controls[cid][None].to(device)
        else:
            raise ValueError(f"arm {arm.name} needs control {cid!r}; have {sorted(controls)}")
        ctrl = ctrl[..., :n_frames]
        if ctrl.shape[-1] < n_frames:
            ctrl = torch.nn.functional.pad(ctrl, (0, n_frames - ctrl.shape[-1]))
        cond[cid] = [ctrl.float() * tf_mask]

    denoiser, branches = make_multi_cfg_denoiser(model, cond, scales, order)
    padding_mask = torch.ones(1, n_frames, dtype=torch.bool, device=device)

    gen = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(1, model.io_channels, n_frames, generator=gen).to(device, torch.bfloat16)
    init_kwargs = {}
    if sdedit_noise_level is not None:
        init = drums_latent if sdedit_source == "drums" else stem_latent
        init_kwargs = {"init_data": init.to(noise.dtype), "init_noise_level": float(sdedit_noise_level)}
    conditioning = [{"prompt": prompt, "seconds_total": seconds_total}]
    t0 = time.time()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        latents = sample_diffusion(
            model=denoiser,
            noise=noise,
            cond_inputs={},
            diffusion_objective=model.diffusion_objective,
            steps=int(steps),
            cfg_scale=1.0,
            conditioning=conditioning,
            sample_rate=sr,
            pretransform=model.pretransform,
            mask_padding_attention=model.mask_padding_attention,
            use_effective_length_for_schedule=model.use_effective_length_for_schedule,
            padding_mask=padding_mask,
            dist_shift=model.sampling_dist_shift,
            sampler_type=sampler_type,
            batch_cfg=True,
            disable_tqdm=True,
            decode=False,
            **init_kwargs,
        )
    audio = decode(model, latents)[0, :, :n_samples].clamp(-1, 1)
    info = {
        "n_frames": n_frames, "seconds_total": seconds_total, "fps": fps,
        "cursor_frames": cursor, "horizon_frames": horizon,
        "cursor_seconds": cursor / fps, "horizon_seconds": horizon / fps,
        "branches": branches, "sampling_seconds": time.time() - t0,
        "sdedit_noise_level": sdedit_noise_level,
        "sdedit_source": sdedit_source if sdedit_noise_level is not None else None,
    }
    return audio, info


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_controls(controls, used, cursor, horizon, n_frames, fps, arm_name):
    """The feature controls against time with the context region, the cursor (solid) and the
    control horizon (dashed); the frames the model does not see are shaded."""
    panels = [(k, label) for k, label in FEATURE_CONTROLS if k in controls]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(10, 1.0 + 1.6 * len(panels)),
                             sharex=True, gridspec_kw={"height_ratios": [0.5] + [1] * len(panels)})
    axes = np.atleast_1d(axes)
    t = np.arange(n_frames) / fps
    ax = axes[0]
    ax.axvspan(0, cursor / fps, color="#4CAF50", alpha=0.6, label="context (stem given)")
    ax.axvspan(cursor / fps, n_frames / fps, color="#FF5722", alpha=0.5, label="generated")
    ax.set_yticks([])
    ax.set_title(f"arm {arm_name}: context [0, {cursor / fps:.2f} s), control visible to {horizon / fps:.2f} s", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)
    for ax, (key, label) in zip(axes[1:], panels):
        x = controls[key].numpy()
        for ch in range(x.shape[0]):
            ax.plot(t, x[ch], lw=1, label=("low" if ch == 0 else "high") if x.shape[0] == 2 else None)
        ax.axvspan(horizon / fps, n_frames / fps, color="grey", alpha=0.25)
        ax.axvline(cursor / fps, color="black", lw=1)
        ax.axvline(horizon / fps, color="black", lw=1, ls="--")
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel(label + ("\n(used)" if key in used else ""), fontsize=8)
        if x.shape[0] == 2:
            ax.legend(loc="upper right", fontsize=7, frameon=False)
    if "streamgen_latent" in used:
        axes[-1].set_xlabel("time (s) -- this arm hears the drum latent, not a plotted feature", fontsize=8)
    else:
        axes[-1].set_xlabel("time (s)", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def _order_label(order):
    return " -> ".join(order)


def _parse_order(label):
    return tuple(a.strip() for a in label.split("->"))


def _slug(text):
    return re.sub(r"[^\w.+-]+", "_", str(text)).strip("_")[:60] or "_"


def _save(path, audio, sr):
    torchaudio.save(str(path), audio.clamp(-1, 1).float().cpu(), sr)
    return str(path)


def create_wjd_control_ui(registry: ArmRegistry, catalog: tp.Dict[str, Track], out_dir=DEFAULT_OUT_DIR,
                          tria_stats: tp.Optional[TriaStats] = None, gradio_title="WJD drum-control arms"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tria_stats = tria_stats or TriaStats.load()
    sample_rate = 44100
    track_names = list(catalog)
    default_track = track_names[0] if track_names else None
    default_source = SOURCE_PICKER if track_names else SOURCE_UPLOAD
    arm_names = registry.names()

    def stems_of(track):
        return list(catalog[track].stems) if track in catalog else []

    def prompt_of(track, stem):
        if track in catalog and stem in catalog[track].stems:
            return catalog[track].stems[stem][1]
        return ""

    def on_track(track, window):
        stems = stems_of(track)
        stem = stems[0] if stems else None
        dur = catalog[track].duration if track in catalog else 0.0
        return (gr.update(choices=stems, value=stem), prompt_of(track, stem),
                gr.update(maximum=max(0.0, dur - window)))

    def on_stem(track, stem):
        return prompt_of(track, stem)

    def on_window(track, window, cursor):
        dur = catalog[track].duration if track in catalog else 0.0
        return gr.update(maximum=max(0.0, dur - window)), gr.update(maximum=window, value=min(cursor, window))

    def on_source(source):
        return gr.update(visible=source == SOURCE_PICKER), gr.update(visible=source == SOURCE_UPLOAD)

    def on_reload(arm_name):
        try:
            registry.get(arm_name, reload=True)
        except Exception as e:  # noqa: BLE001
            raise gr.Error(f"loading {arm_name} failed: {e}") from e
        return registry.status_markdown(arm_name)

    def on_arm(arm_name):
        return registry.status_markdown(arm_name)

    def generate(arm_name, source, track, stem_name, offset_s, window_s, up_drums, up_stem, prompt,
                 cursor_s, lookahead_s, s_prompt, s_context, s_control, order_label,
                 steps, sampler_type, seed, sdedit_on, sdedit_source, sdedit_level, progress=gr.Progress()):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        n_frames = int(np.ceil(window_s * sample_rate / LATENT_HOP))
        n_samples = n_frames * LATENT_HOP
        offset = int(round(offset_s * sample_rate / LATENT_HOP)) * LATENT_HOP  # frame-aligned
        label = {}

        progress(0.05, desc="loading inputs")
        if source == SOURCE_PICKER:
            if track not in catalog:
                raise gr.Error("pick a WJD track")
            tr = catalog[track]
            if stem_name not in tr.stems:
                raise gr.Error(f"{track} has no stem {stem_name!r}")
            drums = load_window(tr.drums_path, sample_rate, offset, n_samples)
            stem = load_window(tr.stems[stem_name][0], sample_rate, offset, n_samples)
            if tr.features_path is not None:
                controls = controls_from_npz(tr.features_path, offset // LATENT_HOP, n_frames)
            else:
                controls = controls_from_audio(drums, tria_stats)
            label.update(track=track, stem=stem_name)
        else:
            if up_drums is None:
                raise gr.Error("upload drum audio (the control)")
            drums = upload_window(up_drums, sample_rate, offset, n_samples)
            stem = upload_window(up_stem, sample_rate, offset, n_samples) if up_stem is not None else None
            controls = controls_from_audio(drums, tria_stats)
            label.update(track="upload", stem="stem" if stem is not None else "nostem")
        if stem is None:
            cursor_s = 0.0

        progress(0.15, desc=f"loading arm {arm_name}")
        try:
            arm = registry.get(arm_name)
        except Exception as e:  # noqa: BLE001
            raise gr.Error(f"loading {arm_name} failed: {e}") from e

        scales = {"prompt": float(s_prompt), "context": float(s_context), "control": float(s_control)}
        order = _parse_order(order_label)
        seed = int(seed)
        if seed < 0:
            seed = int(np.random.randint(0, 2**31 - 1))

        progress(0.25, desc=f"sampling ({steps} steps)")
        sdedit_noise_level = float(sdedit_level) if sdedit_on else None
        try:
            audio, info = generate_continuation(
                arm, prompt, drums, stem, controls, cursor_s, lookahead_s, scales, order,
                steps, sampler_type, seed, sdedit_noise_level=sdedit_noise_level, sdedit_source=sdedit_source,
            )
        except ValueError as e:
            raise gr.Error(str(e)) from e

        progress(0.9, desc="writing files")
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = (f"{ts}_{arm_name}_{_slug(label['track'])}_{_slug(label['stem'])}_off{offset_s:g}"
                f"_cur{info['cursor_seconds']:.1f}_la{lookahead_s:g}"
                f"_p{s_prompt:g}_c{s_context:g}_k{s_control:g}_{'-'.join(a[0] for a in order)}_s{seed}"
                + (f"_sde-{sdedit_source}{sdedit_noise_level:g}" if sdedit_noise_level is not None else ""))
        paths = {"gen": _save(out_dir / f"{base}.wav", audio, sample_rate)}
        mix = MIX_GAIN * (audio + drums)
        paths["mix"] = _save(out_dir / f"{base}+drums.wav", mix, sample_rate)
        paths["drums"] = _save(out_dir / f"{base}_ref-drums.wav", drums, sample_rate)
        if stem is not None:
            context = stem.clone()
            context[:, info["cursor_frames"] * LATENT_HOP:] = 0
            paths["context"] = _save(out_dir / f"{base}_ref-context.wav", context, sample_rate)
            paths["target"] = _save(out_dir / f"{base}_ref-target.wav", stem, sample_rate)
        else:
            paths["context"] = None
            paths["target"] = None

        img = plot_controls(controls, arm.controls, info["cursor_frames"], info["horizon_frames"],
                            info["n_frames"], info["fps"], arm_name)
        notes = [
            f"arm {arm_name}: step {arm.step}, {arm.ckpt_path}",
            f"prompt {prompt!r}; window {info['n_frames']} frames = {info['seconds_total']:.2f} s"
            + (f"; {label['track']} / {label['stem']} from {offset_s:g} s" if source == SOURCE_PICKER else ""),
            f"context [0, {info['cursor_seconds']:.2f} s) = {info['cursor_frames']} frames; "
            f"control visible to {info['horizon_seconds']:.2f} s = {info['horizon_frames']} frames "
            f"(lookahead {lookahead_s:+g} s; training saw {TRAINED_LOOKAHEAD[0]:+g}..{TRAINED_LOOKAHEAD[1]:+g})",
            describe_branches(info["branches"], scales, order),
            f"{len(info['branches'])} forward pass(es) per step; {steps} steps, {sampler_type}, seed {seed}, "
            f"{info['sampling_seconds']:.1f} s",
            f"written to {out_dir}/{base}*.wav",
        ]
        if sdedit_noise_level is not None:
            what = "drum latent (streamgen input)" if sdedit_source == "drums" else "stem latent (whole window, incl. the target after the cursor)"
            notes.insert(4, f"SDEdit: started from the {what} noised to sigma = {sdedit_noise_level:g} "
                            f"({steps} steps over [{sdedit_noise_level:g}, 0])")
        if stem is None:
            notes.insert(2, "no stem: context empty (cursor 0); the control is visible only through the lookahead")
        if not (TRAINED_LOOKAHEAD[0] <= lookahead_s <= TRAINED_LOOKAHEAD[1]):
            notes.append("NOTE: lookahead outside the training range; the model is extrapolating")
        if source == SOURCE_UPLOAD and "drums_tria_ema" in arm.controls:
            notes.append("NOTE: TRIA ema statistics warm-start at the clip start (training saw the song's history)")
        return (paths["gen"], paths["mix"], img, "\n".join(notes),
                paths["drums"], paths["context"], paths["target"], registry.status_markdown(arm_name))

    with gr.Blocks(theme=gr.themes.Base()) as ui:
        gr.Markdown(f"### {gradio_title}")
        gr.Markdown(
            "Continue a stem over the given drums with one of the WJD finetune arms. "
            "Three guidance scales, one per condition the arms were trained to drop: **prompt**, "
            "**context** (the stem up to the cursor) and **control** (the drums). Scale 1 = plain "
            "conditional; the branches are composed in the chosen order (first added first), e.g. "
            f"*{_order_label(('context', 'control', 'prompt'))}* with context = control = 1 is the plain "
            "prompt CFG the training demos use."
        )
        with gr.Row():
            arm_dropdown = gr.Dropdown(arm_names, value=arm_names[0], label="Arm (model)", scale=2)
            reload_button = gr.Button("Load / reload latest checkpoint", scale=1)
            status_md = gr.Markdown(registry.status_markdown(arm_names[0]))
        with gr.Row():
            prompt_box = gr.Textbox(label="Prompt (the stem's instrument)", scale=6,
                                    value=prompt_of(default_track, stems_of(default_track)[0]) if default_track else "")
            generate_button = gr.Button("Generate", variant="primary", scale=1)

        with gr.Row(equal_height=False):
            with gr.Column():
                source_radio = gr.Radio([SOURCE_PICKER, SOURCE_UPLOAD], value=default_source, label="Source")
                with gr.Group(visible=default_source == SOURCE_PICKER) as picker_group:
                    track_dropdown = gr.Dropdown(track_names, value=default_track, label="WJD track (held-out split)")
                    stem_dropdown = gr.Dropdown(stems_of(default_track) if default_track else [],
                                                value=stems_of(default_track)[0] if default_track else None,
                                                label="Stem to continue (target)")
                with gr.Group(visible=default_source == SOURCE_UPLOAD) as upload_group:
                    drums_upload = gr.Audio(label="Drum audio (the control)", type="numpy")
                    stem_upload = gr.Audio(label="Stem audio to continue (optional context)", type="numpy")
                with gr.Row():
                    window_slider = gr.Slider(4.0, 60.0, value=12.0, step=0.5, label="Window (s)",
                                              info="training items are 12 s")
                    offset_slider = gr.Slider(0.0, max(0.0, (catalog[default_track].duration - 12.0) if default_track else 600.0),
                                              value=0.0, step=0.5, label="Start offset in the track / clip (s)")
                with gr.Accordion("What the model sees", open=True):
                    cursor_slider = gr.Slider(0.0, 12.0, value=6.0, step=0.1, label="Context / cursor (s)",
                                              info="the stem is given up to here and generated after")
                    lookahead_slider = gr.Slider(-6.0, 6.0, value=0.0, step=0.1, label="Control lookahead (s)",
                                                 info="drums visible up to cursor + lookahead; training: -4..0 s")
                with gr.Accordion("Three-axis CFG", open=True):
                    s_prompt = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="prompt scale")
                    s_context = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="context scale")
                    s_control = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="control scale")
                    order_dropdown = gr.Dropdown([_order_label(o) for o in ORDERS], value=_order_label(DEFAULT_ORDER),
                                                 label="Order (first added first)",
                                                 info="v = v(none) + s1 (v(a) - v(none)) + s2 (v(a,b) - v(a)) + s3 (v(a,b,c) - v(a,b))")
                with gr.Accordion("Sampler", open=False):
                    with gr.Row():
                        steps_slider = gr.Slider(1, 200, value=8, step=1, label="Steps")
                        sampler_dropdown = gr.Dropdown(SAMPLERS, value=SAMPLERS[0], label="Sampler")
                        seed_number = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                with gr.Accordion("SDEdit", open=False):
                    sdedit_checkbox = gr.Checkbox(value=False, label="SDEdit",
                                                  info="start from the latent below noised to the chosen level, not from pure noise")
                    sdedit_source_radio = gr.Radio(SDEDIT_SOURCES, value=SDEDIT_SOURCES[0], label="SDEdit from",
                                                   info="drums = the streamgen input; stem = the stem to continue (whole window)")
                    sdedit_slider = gr.Slider(0.01, 1.0, value=0.7, step=0.01, label="SDEdit noise level (sigma)",
                                              info="x = (1 - sigma) z + sigma noise; 1 = pure noise, lower keeps more of the source")
            with gr.Column():
                gen_audio = gr.Audio(label="Generated stem", type="filepath", interactive=False)
                mix_audio = gr.Audio(label="Generated stem + drums", type="filepath", interactive=False)
                controls_image = gr.Image(label="Controls and masks", interactive=False)
                notes_box = gr.Textbox(label="Notes", lines=9, interactive=False)
                with gr.Accordion("References (what went in)", open=False):
                    drums_audio = gr.Audio(label="Drums (control source)", type="filepath", interactive=False)
                    context_audio = gr.Audio(label="Context given (stem, muted after the cursor)", type="filepath", interactive=False)
                    target_audio = gr.Audio(label="Target stem (whole window)", type="filepath", interactive=False)

        track_dropdown.change(on_track, [track_dropdown, window_slider], [stem_dropdown, prompt_box, offset_slider])
        stem_dropdown.change(on_stem, [track_dropdown, stem_dropdown], [prompt_box])
        window_slider.change(on_window, [track_dropdown, window_slider, cursor_slider], [offset_slider, cursor_slider])
        source_radio.change(on_source, [source_radio], [picker_group, upload_group])
        arm_dropdown.change(on_arm, [arm_dropdown], [status_md])
        reload_button.click(on_reload, [arm_dropdown], [status_md])
        generate_button.click(
            generate,
            inputs=[arm_dropdown, source_radio, track_dropdown, stem_dropdown, offset_slider, window_slider,
                    drums_upload, stem_upload, prompt_box, cursor_slider, lookahead_slider,
                    s_prompt, s_context, s_control, order_dropdown, steps_slider, sampler_dropdown, seed_number,
                    sdedit_checkbox, sdedit_source_radio, sdedit_slider],
            outputs=[gen_audio, mix_audio, controls_image, notes_box, drums_audio, context_audio, target_audio, status_md],
            api_name="generate",
        )
    return ui


__all__ = [
    "AXES", "ArmRegistry", "LoadedArm", "Track", "load_catalog", "create_wjd_control_ui",
    "generate_continuation", "build_masks", "controls_from_audio", "controls_from_npz",
    "load_window", "upload_window", "plot_controls",
]
