"""
Decode a couple of items from each silence-fraction band of a pre-encoded dataset, so a
`--max_silence_fraction` cutoff can be chosen by ear.

The sweep in scripts/analyze_silence_levels.py has no plateau — survivor count falls smoothly
from 0.1 to 1.0 — so there is no cutoff the data picks out for you. What the number actually
has to answer is musical: at what point is a window too empty to be a useful training example?
This script lays the bands out to be listened to. Point it at a dataset encoded with the level
filter OFF (or with only the RMS floor), or the upper bands will not exist.

  uv run python scripts/sample_silence_buckets.py \
      --dir /data/.../slakh-streamgen-preencoded-same-s/train --variant v0 \
      --out /tmp/silence_bands --model same-s -n 2
  uv run python scripts/make_listening_page.py --dir /tmp/silence_bands --embed mp3

Within a band the picks are spread evenly across it by `silence_fraction` rather than drawn at
random — for -n 2 that is roughly the band's 25th and 75th percentile — so the two clips
bracket what the band contains instead of landing next to each other by chance. No seed: the
selection is a function of the data.

What you hear is the decode of the stored latent and of the stored control, which is exactly
what the model is trained on. There is no source track alongside it; use
scripts/decode_preencoded_samples.py for a source/decoded A/B.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import torchaudio

from stable_audio_3 import AutoencoderModel


def load_items(latents_dir, variant, stream):
    """[(id, silence_fraction, levels, md)] for every sidecar that has levels and a latent."""
    items = []
    for path in sorted(glob.glob(os.path.join(latents_dir, "[0-9]*.json"))):
        stem = os.path.basename(path)[: -len(".json")]
        m = re.search(r"_v(\d+)$", stem)
        v = f"v{m.group(1)}" if m else "v0"
        if variant is not None and v != variant:
            continue
        if not os.path.exists(os.path.join(latents_dir, f"{stem}.npy")):
            continue
        with open(path) as f:
            md = json.load(f)
        levels = md.get("levels") or {}
        if stream not in levels:
            continue
        items.append((stem, levels[stream]["silence_fraction"], levels, md))
    return items


def pick(band_items, n):
    """n items spread evenly across the band by silence_fraction, not clustered."""
    band_items = sorted(band_items, key=lambda it: it[1])
    if len(band_items) <= n:
        return band_items
    # Midpoints of n equal slices: for n=2 the 25th and 75th percentile of the band.
    idx = [int((k + 0.5) * len(band_items) / n) for k in range(n)]
    return [band_items[min(i, len(band_items) - 1)] for i in idx]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = AutoencoderModel.from_pretrained(args.model, device=str(device))
    sr = ae.sample_rate

    items = load_items(args.dir, args.variant, args.stream)
    if not items:
        raise SystemExit(f"No sidecars with a '{args.stream}' levels block under {args.dir}"
                         + (f" for variant {args.variant}" if args.variant else ""))
    print(f"{len(items)} item(s) with levels under {args.dir}"
          + (f" (variant {args.variant})" if args.variant else ""))

    edges = args.bands
    os.makedirs(args.out, exist_ok=True)

    # `_v<n>` stays on the id side of make_listening_page.py's stem split, and a `-`
    # separator keeps the band prefix there too; the prefix also sorts the cards by band.
    print(f"\n{'band':>12}{'in band':>9}{'picked':>8}   ids")
    written = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Half-open, and the top band closes so silence_fraction == 1.0 is not dropped.
        top = hi >= edges[-1]
        band = [it for it in items if lo <= it[1] < hi or (top and it[1] == hi)]
        chosen = pick(band, args.n)
        print(f"  {lo:>4.2f}-{hi:<4.2f}{len(band):>9}{len(chosen):>8}   "
              + ", ".join(f"{s} ({f:.0%})" for s, f, _, _ in chosen))

        for stem, frac, levels, md in chosen:
            tag = f"b{int(round(lo * 100)):03d}-{stem}"
            latent = torch.from_numpy(np.load(os.path.join(args.dir, f"{stem}.npy"))).to(device)

            notes = {}
            with torch.no_grad():
                decoded = ae.decode(latent.unsqueeze(0)).squeeze(0)
            torchaudio.save(os.path.join(args.out, f"{tag}_decoded.wav"),
                            decoded.detach().float().cpu().clamp(-1, 1), sr)
            lv = levels[args.stream]
            notes["decoded"] = (f"{args.stream}: {lv['silence_fraction']:.1%} silent, "
                                f"RMS {lv['rms_dbfs']:.1f} dBFS")
            written += 1

            ctrl_path = os.path.join(args.dir, f"{stem}_controls.npy")
            dims = md.get("controls_dim")
            if os.path.exists(ctrl_path) and dims:
                fused = torch.from_numpy(np.load(ctrl_path)).to(device)
                names = args.controls or [f"control{i}" for i in range(len(dims))]
                at = 0
                for name, dim in zip(names, dims):
                    with torch.no_grad():
                        dec = ae.decode(fused[at:at + dim].unsqueeze(0)).squeeze(0)
                    at += dim
                    label = f"control_{name}_decoded"
                    torchaudio.save(os.path.join(args.out, f"{tag}_{label}.wav"),
                                    dec.detach().float().cpu().clamp(-1, 1), sr)
                    clv = levels.get(name)
                    if clv:
                        notes[label] = (f"{clv['silence_fraction']:.1%} silent, "
                                        f"RMS {clv['rms_dbfs']:.1f} dBFS")
                    written += 1

            # make_listening_page.py reads <sample_id>.json next to the wavs: `prompt` and
            # `rms_dbfs` head the card, `streams` annotates each row. Carrying the band and
            # the measured levels here is the whole point — a card that does not say which
            # band it is cannot be used to choose a cutoff.
            with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
                json.dump({
                    "prompt": f"band {lo:.2f}-{hi:.2f} · {args.stream} {frac:.1%} silent"
                              + (f" · {md['prompt']}" if md.get("prompt") else ""),
                    "path": md.get("path"),
                    "seconds_total": md.get("seconds_total"),
                    "rms_dbfs": round(lv["rms_dbfs"], 1),
                    "streams": notes,
                    "levels": levels,
                    "latent_id": stem,
                    "band": [lo, hi],
                }, f, indent=2)

    print(f"\nWrote {written} wav(s) to {args.out}")
    print(f"  Build the page: uv run python scripts/make_listening_page.py "
          f"--dir {args.out} --embed mp3")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="Pre-encode output_path with .npy + .json")
    p.add_argument("--out", required=True, help="Directory to write the wavs into")
    p.add_argument("--model", default="same-s",
                   help="Autoencoder the latents were written with; a mismatch decodes noise")
    p.add_argument("--variant", default="v0",
                   help="Only this augmentation variant (default v0, the unaugmented pass). "
                        "Pass '' for all of them.")
    p.add_argument("--stream", default="target",
                   help="Which stream's silence_fraction defines the bands (default target)")
    p.add_argument("-n", type=int, default=2, help="Items per band (default 2)")
    p.add_argument("--bands", type=float, nargs="*",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   help="Band edges, matching the sweep in analyze_silence_levels.py")
    p.add_argument("--controls", nargs="*", default=None,
                   help="Names for the fused control streams, in sidecar order "
                        "(e.g. streamgen_audio). Default: control0, control1, ...")
    a = p.parse_args()
    a.variant = a.variant or None
    main(a)
