"""Launch the Gradio interface for the WJD drum-control arms with three-axis CFG.

    PYTHONPATH=$PWD .venv/bin/python run_gradio_wjd.py                      # four arms, newest ckpts
    PYTHONPATH=$PWD .venv/bin/python run_gradio_wjd.py --arms rms tria_ema  # a subset
    PYTHONPATH=$PWD .venv/bin/python run_gradio_wjd.py \
        --arm rms stable_audio_3/configs/model_configs/small_music_wjd_drums_rms.json /path/last.ckpt

Arms are resolved as scripts/wjd/listen_wjd_arms.py resolves them (model config and group from
sbatch/03_3_finetune_wjd_<arm>.sbatch, newest last.ckpt under the group) and loaded on first use
with their EMA weights; the autoencoder and text conditioner are shared between the arms. The
UI's "Load / reload latest checkpoint" button re-resolves while the training jobs run.

On a compute node the UI is served on --server_port; reach it through an SSH tunnel
(ssh -L 7860:<node>:7860 ...) or pass --share for a public Gradio link.
"""

import argparse
import os
import sys
from pathlib import Path

# Silence Python warnings unless --verbose, as run_gradio.py does.
if "--verbose" not in sys.argv:
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    import warnings

    warnings.filterwarnings("ignore")

import torch  # noqa: E402

from stable_audio_3.data.features import TriaStats  # noqa: E402
from stable_audio_3.inference.wjd_arms import DEFAULT_ARMS, DEFAULT_SAVE_BASE  # noqa: E402
from stable_audio_3.interface.wjd_control import (  # noqa: E402
    DEFAULT_MIRROR,
    DEFAULT_OUT_DIR,
    DEFAULT_SPLIT,
    ArmRegistry,
    create_wjd_control_ui,
    load_catalog,
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    specs = {name: (None, None) for name in args.arms}
    for name, config, ckpt in args.arm or []:
        specs[name] = (config, ckpt)
    registry = ArmRegistry(specs, device, use_ema=args.use_ema, save_base=args.save_base)
    for name in registry.names():
        try:
            config, ckpt = registry.resolve(name)
            print(f"arm {name:<11} {config}  {ckpt}")
        except Exception as e:  # noqa: BLE001
            print(f"arm {name:<11} unresolved: {e}")
    if args.preload:
        for name in registry.names():
            registry.get(name)

    catalog = load_catalog(args.mirror_root, args.split)
    print(f"{len(catalog)} WJD {args.split} tracks with drums and a target stem under {args.mirror_root}")
    tria_stats = TriaStats.load(args.tria_stats) if args.tria_stats else TriaStats.load()

    ui = create_wjd_control_ui(registry, catalog, out_dir=args.out_dir, tria_stats=tria_stats, gradio_title=args.title)
    ui.queue()
    # Generated wavs live outside the cwd / temp dir, so Gradio must be told it may serve them.
    ui.launch(share=args.share, server_name=args.server_name, server_port=args.server_port,
              allowed_paths=[str(Path(args.out_dir).resolve())])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                   help="Arms to offer, each resolved from its sbatch file and the newest last.ckpt")
    p.add_argument("--arm", nargs=3, action="append", metavar=("NAME", "MODEL_CONFIG", "CKPT"),
                   help="Pin an arm to an explicit model config and checkpoint (repeatable)")
    p.add_argument("--save_base", default=DEFAULT_SAVE_BASE, help="Checkpoint root the sbatch jobs write to")
    p.add_argument("--raw_weights", dest="use_ema", action="store_false",
                   help="Load the raw diffusion weights instead of the EMA copy")
    p.add_argument("--preload", action="store_true", help="Load every arm at start instead of on first use")
    p.add_argument("--mirror_root", default=DEFAULT_MIRROR, help="WJD stem mirror root")
    p.add_argument("--split", default=DEFAULT_SPLIT, help="Mirror split to offer in the track picker")
    p.add_argument("--tria_stats", default=None, help="TRIA stats JSON (default: the one training used)")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Where generated wavs are written")
    p.add_argument("--title", default="WJD drum-control arms: three-axis CFG")
    p.add_argument("--share", action="store_true", help="Create a public Gradio link")
    p.add_argument("--server_name", default="0.0.0.0")
    p.add_argument("--server_port", type=int, default=7860)
    p.add_argument("--verbose", action="store_true", help="Keep Python warnings")
    main(p.parse_args())
