"""Build a local HTML page for listening to sanity-check wavs with mel spectrograms.

Point it at a directory of wavs written by pre_encode_dataset.py --sanity_check_samples
or by decode_preencoded_samples.py. Files are grouped by sample id, so each sample's
source / decoded / control streams land on one card, stacked for A/B comparison with a
shared time axis.

An augmentation variant is its own sample here, not another stream of one. Ids written
for a dataset encoded with --augment_variants carry a `_v<n>` suffix, and that suffix
stays with the id: n logged samples across N variants give n x N cards, each showing the
pitch/tempo roll it was written with. Grouping the variants together instead would stack
four unrelated renderings of a track on one card and break the source/decoded pairing.

The page is plain HTML referencing the wavs in place, so open it from the same
directory (file:// works; over SSH, `python -m http.server` in that directory).

Usage:
  uv run python scripts/make_listening_page.py --dir /path/to/_sanity_check
"""

import argparse
import html
import json
import math
import re
from pathlib import Path

import matplotlib
import torch
import torchaudio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Order streams source-first so the ground truth sits above its reconstruction.
STREAM_ORDER = ["source", "decoded"]

# `<id>` or `<id>_v<n>`, followed by the stream label. The non-greedy first branch is what
# keeps an augmentation-variant suffix on the id side of the split; the second is the
# unaugmented case, where there is no `_v<n>` to find.
STEM_RE = re.compile(r"^(?P<id>.+?_v\d+|[^_]+)_(?P<label>.+)$")


def split_stem(stem):
    """Split a wav stem into (sample_id, stream_label).

    The `_v<n>` augmentation suffix belongs to the id: `0000000000_v2_decoded` is the
    decode of variant 2, a different sample from variant 0, and gets its own card.
    """
    m = STEM_RE.match(stem)
    if not m:
        return stem, "audio"
    return m.group("id"), m.group("label")


def stream_sort_key(label):
    base = label.split("_")[-1]
    return (0 if not label.startswith("control") else 1,
            STREAM_ORDER.index(base) if base in STREAM_ORDER else len(STREAM_ORDER),
            label)


def mel_db(path, n_mels, max_width):
    """Mel spectrogram in dB, with the hop chosen so the image stays max_width wide."""
    audio, sr = torchaudio.load(str(path))
    audio = audio.mean(0, keepdim=True)
    hop = max(256, int(math.ceil(audio.shape[-1] / max_width)))
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=max(1024, hop * 2), hop_length=hop, n_mels=n_mels, power=2.0
    )(audio)[0]
    db = 10.0 * torch.log10(mel.clamp(min=1e-10))
    return db.numpy(), audio.shape[-1] / sr


def load_metadata(wav_dir, sample_id):
    """Metadata json written by pre_encode_dataset.py, if it sits next to the wavs."""
    for candidate in (wav_dir / f"{sample_id}.json", wav_dir.parent / f"{sample_id}.json"):
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    return {}


def build(args):
    wav_dir = Path(args.dir)
    wavs = sorted(p for p in wav_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No .wav files in {wav_dir}")

    mel_dir = wav_dir / "_mels"
    mel_dir.mkdir(exist_ok=True)

    print(f"Computing {len(wavs)} mel spectrograms…")
    specs = {}
    for wav in wavs:
        db, duration = mel_db(wav, args.n_mels, args.max_width)
        specs[wav.name] = (db, duration)
        print(f"  {wav.name}  {duration:.1f}s  {db.shape[1]} frames")

    # One dB range across the whole page so loudness differences between a source and
    # its reconstruction show up instead of being normalized away.
    vmax = max(float(db.max()) for db, _ in specs.values())
    vmin = vmax - args.dynamic_range

    for wav in wavs:
        db, _ = specs[wav.name]
        plt.imsave(mel_dir / f"{wav.stem}.png", db, cmap="magma", origin="lower", vmin=vmin, vmax=vmax)

    groups = {}
    for wav in wavs:
        sample_id, label = split_stem(wav.stem)
        groups.setdefault(sample_id, []).append((label, wav))

    cards = []
    for sample_id, items in groups.items():
        md = load_metadata(wav_dir, sample_id)
        meta_bits = []
        if md.get("prompt"):
            meta_bits.append(f"prompt: {md['prompt']}")
        if md.get("path"):
            meta_bits.append(f"src: {md['path']}")
        if md.get("seconds_total"):
            meta_bits.append(f"{md['seconds_total']}s total")
        aug = md.get("augmentation")
        if aug:
            # Written per item by pre_encode_dataset.py, so the card says which roll it is
            # rather than leaving you to guess from the sound.
            meta_bits.append(
                f"aug v{aug.get('variant')}: rate {aug.get('time_stretch_rate', 1.0):.3f}, "
                f"{aug.get('pitch_semitones', 0.0):+.2f} st ({aug.get('pitch_scope')})"
            )
        meta = html.escape(" · ".join(meta_bits))

        rows = []
        for label, wav in sorted(items, key=lambda it: stream_sort_key(it[0])):
            duration = specs[wav.name][1]
            rows.append(f"""
      <div class="row" data-duration="{duration:.4f}">
        <div class="label">{html.escape(label)}<span class="dur">{duration:.1f}s</span></div>
        <audio preload="none" controls src="{html.escape(wav.name)}"></audio>
        <div class="spec"><img src="_mels/{html.escape(wav.stem)}.png" alt="mel spectrogram">
          <div class="playhead"></div></div>
      </div>""")

        cards.append(f"""
    <section class="card">
      <h2>{html.escape(sample_id)}</h2>
      {f'<p class="meta">{meta}</p>' if meta else ''}
      {''.join(rows)}
    </section>""")

    page = PAGE.format(
        title=html.escape(wav_dir.name),
        dir=html.escape(str(wav_dir.resolve())),
        count=len(wavs),
        groups=len(groups),
        range_note=f"{vmin:.0f} to {vmax:.0f} dB, {args.n_mels} mel bands",
        cards="".join(cards),
    )

    out = Path(args.out) if args.out else wav_dir / "index.html"
    out.write_text(page)
    print(f"\nWrote {out.resolve()}")
    print(f"Open it directly, or serve the directory:  python -m http.server -d {wav_dir.resolve()} 8000")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — pre-encode listening check</title>
<style>
  :root {{
    --bg: #12100f; --panel: #1c1917; --line: #2f2a27;
    --fg: #f0ece8; --dim: #a29a93; --accent: #f7a76c;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  header {{ position: sticky; top: 0; z-index: 10; padding: 14px 20px;
    background: rgba(18,16,15,.94); border-bottom: 1px solid var(--line);
    backdrop-filter: blur(6px); }}
  h1 {{ margin: 0; font-size: 15px; font-weight: 600; letter-spacing: .01em; }}
  header p {{ margin: 4px 0 0; color: var(--dim); font-size: 12px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-all; }}
  main {{ padding: 20px; max-width: 1400px; margin: 0 auto;
    display: flex; flex-direction: column; gap: 18px; }}
  .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 14px 16px 16px; }}
  h2 {{ margin: 0 0 2px; font-size: 14px; font-family: ui-monospace, Menlo, monospace;
    color: var(--accent); font-weight: 600; }}
  .meta {{ margin: 0 0 12px; color: var(--dim); font-size: 12px; word-break: break-all; }}
  .row {{ display: grid; grid-template-columns: 150px 260px 1fr; gap: 12px;
    align-items: center; padding: 7px 0; border-top: 1px solid var(--line); }}
  .row:first-of-type {{ border-top: 0; }}
  .label {{ font-family: ui-monospace, Menlo, monospace; font-size: 12px;
    display: flex; flex-direction: column; }}
  .dur {{ color: var(--dim); font-size: 11px; }}
  audio {{ width: 100%; height: 34px; }}
  .spec {{ position: relative; height: 74px; border-radius: 5px; overflow: hidden;
    background: #000; cursor: pointer; }}
  .spec img {{ width: 100%; height: 100%; display: block; object-fit: fill;
    image-rendering: auto; }}
  .playhead {{ position: absolute; top: 0; bottom: 0; width: 1px; left: 0;
    background: #fff; box-shadow: 0 0 6px #fff; opacity: 0; pointer-events: none; }}
  .playhead.on {{ opacity: .9; }}
  @media (max-width: 900px) {{
    .row {{ grid-template-columns: 1fr; }}
    .spec {{ height: 100px; }}
  }}
</style>
</head>
<body>
<header>
  <h1>Pre-encode listening check — {groups} samples, {count} files</h1>
  <p>{dir} · mel {range_note} · click a spectrogram to seek</p>
</header>
<main>
{cards}
</main>
<script>
  // Playhead tracking + click-to-seek. The mel image is linear in time and spans the
  // full clip, so x/width maps straight onto currentTime.
  document.querySelectorAll('.row').forEach(row => {{
    const audio = row.querySelector('audio');
    const spec = row.querySelector('.spec');
    const head = row.querySelector('.playhead');
    const duration = parseFloat(row.dataset.duration);

    const move = () => {{
      const d = audio.duration || duration;
      head.style.left = (100 * audio.currentTime / d) + '%';
    }};
    audio.addEventListener('timeupdate', move);
    audio.addEventListener('play', () => {{ head.classList.add('on'); move(); }});
    audio.addEventListener('seeked', move);
    audio.addEventListener('ended', () => head.classList.remove('on'));

    spec.addEventListener('click', e => {{
      const r = spec.getBoundingClientRect();
      const frac = (e.clientX - r.left) / r.width;
      const seek = () => {{
        audio.currentTime = (audio.duration || duration) * frac;
        head.classList.add('on');
        move();
        audio.play();
      }};
      // preload="none" means metadata may not be there yet; seeking early throws.
      if (audio.readyState === 0) {{
        audio.addEventListener('loadedmetadata', seek, {{ once: true }});
        audio.load();
      }} else {{
        seek();
      }}
    }});
  }});

  // Only one clip audible at a time, so A/B stays honest.
  document.querySelectorAll('audio').forEach(a => {{
    a.addEventListener('play', () => {{
      document.querySelectorAll('audio').forEach(o => {{ if (o !== a) o.pause(); }});
    }});
  }});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="Directory of wavs to build the page from")
    p.add_argument("--out", default=None, help="Output HTML path (default: <dir>/index.html)")
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--max_width", type=int, default=2000, help="Max spectrogram width in frames")
    p.add_argument("--dynamic_range", type=float, default=80.0, help="dB below the page-wide peak to plot")
    build(p.parse_args())
