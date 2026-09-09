"""Stage a streamgen-drum-mirror tree whose `other` is one submix per track.

A drum-mirror tree holds the accompaniment as separate instrument stems
(`tracks/other/Track00001/{Bass,Piano,...}.wav`), but the streamgen pre-encode never
encodes a lone stem: custom_md_slakh_streamgen builds a stochastic submix -- a random
subset of the stems, each LUFS-normalized to a random level -- and encodes that. Tools
that want to look at "the accompaniment the model sees" therefore need the submix, not
the stems.

This writes a parallel tree with the drums symlinked through and one `Other.wav` per
track, mixed by importing the pipeline's own `load_and_mix_stems` so there is no second
copy of the mixing rules. The result has the same shape as the bryan-data-1 mirror
(`tracks/{drums,other}/<track>/<one file>.wav`), which is what scripts/compare_autoencoders.py
expects.

The submix is rolled over the WHOLE file here, where the pre-encode rolls it over just the
window it is about to encode. Same distribution of subsets and levels, but a stem that is
silent only inside some later excerpt can still be selected -- fine for reconstruction
work, not equivalent for anything that needs the exact latents.

Usage:
  uv run python scripts/make_submix_mirror.py \
      --src /data/hai-res/shared/snnithya/sat-zenon-data/babyslakh/streamgen-drum-mirror/tracks \
      --out /data/scratch-fast/snnithya/sao-3/data/babyslakh-submix-mirror/tracks
"""

import argparse
import importlib.util
import json
import random
from pathlib import Path

import torchaudio

_MIX_MODULE = (
    Path(__file__).resolve().parent.parent
    / "stable_audio_3/configs/dataset_configs/custom_metadata/custom_md_slakh_streamgen.py"
)


def _load_mixer():
    spec = importlib.util.spec_from_file_location("custom_md_slakh_streamgen", _MIX_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    mixer = _load_mixer()
    src, out = Path(args.src), Path(args.out)
    drum_dirs = sorted(p for p in (src / "drums").iterdir() if p.is_dir())
    if not drum_dirs:
        raise SystemExit(f"No track dirs under {src / 'drums'}")

    manifest = {}
    for track_dir in drum_dirs:
        drums = sorted(p for p in track_dir.iterdir() if p.suffix.lower() in mixer.AUDIO_EXTENSIONS)
        if not drums:
            print(f"{track_dir.name}: no drum stem, skipping")
            continue
        drum_path = drums[0]

        stems = mixer.find_other_stems(drum_path)
        if not stems:
            print(f"{track_dir.name}: no accompaniment stems, skipping")
            continue

        # Seeded per track so a re-run reproduces the same submix, and so the subset drawn
        # for one track does not depend on how many stems the previous track had.
        random.seed(f"{args.seed}:{track_dir.name}")
        sample_rate = torchaudio.info(str(drum_path)).sample_rate
        mix, selected = mixer.load_and_mix_stems(stems, sample_rate)
        if mix is None:
            print(f"{track_dir.name}: accompaniment silent, skipping")
            continue

        (out / "drums" / track_dir.name).mkdir(parents=True, exist_ok=True)
        (out / "other" / track_dir.name).mkdir(parents=True, exist_ok=True)
        link = out / "drums" / track_dir.name / drum_path.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(drum_path.resolve())
        torchaudio.save(str(out / "other" / track_dir.name / "Other.wav"), mix, sample_rate)

        manifest[track_dir.name] = {
            "drums": str(drum_path),
            "available_stems": [p.stem for p in stems],
            "selected_stems": selected,
            "sample_rate": sample_rate,
        }
        print(f"{track_dir.name}: {len(selected)}/{len(stems)} stems -> {', '.join(selected)}")

    (out / "submix_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} tracks staged under {out.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="A .../streamgen-drum-mirror/tracks directory")
    p.add_argument("--out", required=True, help="Where to write the mirrored tracks/ tree")
    p.add_argument("--seed", default="0", help="Mixed into the per-track submix seed")
    main(p.parse_args())
