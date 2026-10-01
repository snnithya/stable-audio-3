#!/usr/bin/env python
"""Experiment 3.2: separate the aligned WJD recordings into stems.

Standalone: no `stable_audio_3` import. Runs inside the `sep` conda env
(`audio-separator[gpu]`), see experiments/03-wjd-jazz-stems/02-source-separation.md.

Input   <wjd_root>/audio/manifest.csv  (+ the `flac` column -> audio/aligned/<Track>.flac)
Output  <out_root>/<model_name>/<Track>/{drums,bass,other,vocals[,piano,guitar]}.flac
        <out_root>/<model_name>/<Track>/done.json   (or failed.json)

Every stem is written at the mix's sample rate with the same number of frames (asserted to
+-1 frame), so the per-track `aligned/<Track>.alignment.json` time reference
(`track_time = file_time + offset_sec`) applies to the stems unchanged.

Resumable and kill-safe (Slurm preemption = SIGTERM, SIGKILL 10 s later): a track counts as
done only once done.json exists, which is written by an atomic rename; a stem dir without it
is partial output and is wiped and redone. Several workers may run on the same manifest at
once (an sbatch job plus an interactive one, or a requeued job restarting): each takes an
`_inprogress.json` lease on a track before touching it and skips tracks whose lease is
fresher than --lease_minutes, retrying them once at the end. A lease carrying this worker's
own SLURM_JOB_ID with a lower SLURM_RESTART_COUNT is its dead predecessor's and is reclaimed
immediately, so a requeued job resumes its own track; a lease from another process in the
same allocation (same job id, same restart count) is a live sibling and is respected.

Peak normalisation is disabled (audio-separator's default rescales every stem to 0.9 peak,
which would break `sum(stems) ~= mix`); stems are only scaled down if they would clip.

Example (smoke test, one track, GPU):
    python scripts/wjd/separate.py --tracks ArtBlakey_DownUnder_Orig
Full pass over the `pass` tracks:
    python scripts/wjd/separate.py --status pass
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WJD_ROOT = "/data/hai-res/shared/snnithya/sao-3/data/wjd"
MODEL_DIR = "/data/hai-res/shared/snnithya/sao-3/models/sep"

# The order in which we try to run the stems out of a multi-stem model; also fixes the
# output file names. Model stem names are matched case-insensitively.
STEM_ORDER = ["drums", "bass", "other", "vocals", "piano", "guitar"]

LEASE = "_inprogress.json"
WORKER_ID = {
    "host": os.uname().nodename,
    "pid": os.getpid(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    # Bumped by Slurm on every requeue; lets a restarted job tell its own dead predecessor's
    # lease (lower count) from a sibling worker running in the same allocation (same count).
    "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT") or 0),
}


class Terminated(Exception):
    """Raised from the SIGTERM handler so the current track unwinds through `finally`."""


def _on_sigterm(signum, frame):
    raise Terminated(f"signal {signum}")


def lease_status(out_dir, lease_minutes):
    """'free' | 'mine' | 'busy' for the lease file in out_dir (missing / unreadable = free)."""
    p = out_dir / LEASE
    try:
        d = json.loads(p.read_text())
        age_min = (time.time() - p.stat().st_mtime) / 60.0
    except (OSError, ValueError):
        return "free"
    same_proc = d.get("host") == WORKER_ID["host"] and d.get("pid") == WORKER_ID["pid"]
    predecessor = (
        WORKER_ID["slurm_job_id"]
        and d.get("slurm_job_id") == WORKER_ID["slurm_job_id"]
        and int(d.get("slurm_restart_count") or 0) < WORKER_ID["slurm_restart_count"]
    )
    if same_proc or predecessor:
        return "mine"
    # Anything else -- another job, or another process in the same allocation -- is a live
    # sibling while its lease is fresh, even if it shares our job id.
    return "busy" if age_min < lease_minutes else "free"


def wipe_dir(out_dir):
    """Empty out_dir but keep the directory. Never rmdir: on this NFS mount rmtree's final
    rmdir races the attribute cache / silly-renamed .nfs* files and fails with ENOTEMPTY."""
    for p in out_dir.iterdir():
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
        except FileNotFoundError:
            pass  # another worker or NFS already took care of it


def take_lease(out_dir):
    (out_dir / LEASE).write_text(json.dumps({**WORKER_ID, "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}))


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest(path, statuses, tracks, limit):
    rows = list(csv.DictReader(open(path, newline="")))
    sel = []
    for r in rows:
        if r.get("result") != "done" or not r.get("flac"):
            continue
        if tracks and r["track"] not in tracks:
            continue
        if not tracks and statuses and r.get("status") not in statuses:
            continue
        sel.append(r)
    if tracks:
        missing = sorted(set(tracks) - {r["track"] for r in sel})
        if missing:
            logging.warning("tracks not in manifest (or not done): %s", ", ".join(missing))
    if limit:
        sel = sel[:limit]
    return sel


class AudioSeparatorBackend:
    """audio-separator (python-audio-separator) wrapping the MSST RoFormer/MDXC code."""

    def __init__(self, args):
        import torch  # noqa: F401  (imported so the version lands in done.json)
        from audio_separator.separator import Separator

        # Put the `Separator` logger where our logging goes; audio-separator also honours
        # AUDIO_SEPARATOR_MODEL_DIR but we pass the dir explicitly.
        self.separator = Separator(
            log_level=logging.INFO if args.verbose else logging.WARNING,
            model_file_dir=args.model_file_dir,
            output_dir=None,
            output_format="FLAC",
            normalization_threshold=args.normalization,
            amplification_threshold=0.0,
            sample_rate=args.sample_rate,
            use_soundfile=True,  # PCM_16 FLAC via soundfile, no pydub round-trip
            use_autocast=args.autocast,
            mdxc_params={
                "segment_size": 256,
                "override_model_segment_size": False,
                "batch_size": args.batch_size,
                "overlap": args.overlap,
                "pitch_shift": 0,
            },
        )
        t0 = time.time()
        self.separator.load_model(model_filename=args.model)
        self.load_sec = time.time() - t0
        inst = self.separator.model_instance
        cfg = getattr(inst, "model_data_cfgdict", None)
        self.model_stems = list(cfg.training.instruments) if cfg is not None else []
        if not self.model_stems:
            raise RuntimeError(f"{args.model}: could not read the stem list from the model config")
        # Recorded so done.json says what actually ran (model defaults may fill overlap/batch).
        self.settings = {
            "overlap": getattr(inst, "overlap", None),
            "batch_size": getattr(inst, "batch_size", None),
            "segment_size_dim_t": int(cfg.inference.dim_t) if cfg is not None else None,
            "override_model_segment_size": getattr(inst, "override_model_segment_size", None),
            "autocast": args.autocast,
            "normalization_threshold": args.normalization,
            "sample_rate": args.sample_rate,
        }
        self.model_path = Path(args.model_file_dir) / args.model
        self.ckpt_sha256 = sha256(self.model_path) if self.model_path.exists() else None
        self.yaml_path = None
        for cand in Path(args.model_file_dir).glob(Path(args.model).stem + "*.yaml"):
            self.yaml_path = cand
            break

    def versions(self):
        import torch
        from importlib.metadata import version

        return {
            "audio_separator": version("audio-separator"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(self.separator.torch_device),
        }

    def separate(self, mix_path, out_dir):
        """Write one FLAC per model stem into out_dir, named <stem>.flac. Returns {stem: path}."""
        inst = self.separator.model_instance
        self.separator.output_dir = str(out_dir)
        inst.output_dir = str(out_dir)
        names = {s: s.lower() for s in self.model_stems}
        written = self.separator.separate(str(mix_path), custom_output_names=names)
        out = {}
        for p in written:
            p = Path(p)
            if not p.is_absolute():
                p = out_dir / p
            out[p.stem.lower()] = p
        return out


def read_info(path, with_levels=False):
    import soundfile as sf

    i = sf.info(str(path))
    d = {"frames": i.frames, "samplerate": i.samplerate, "channels": i.channels, "subtype": i.subtype}
    if with_levels:
        import numpy as np

        x, _ = sf.read(str(path), dtype="float32", always_2d=True)
        peak = float(np.abs(x).max())
        d["peak"] = round(peak, 4)
        d["rms_dbfs"] = round(20.0 * float(np.log10(np.sqrt(np.mean(x**2)) + 1e-9)), 2)
        # audio-separator scales a stem down to `normalization` if it would exceed it, so a
        # stem sitting exactly at the ceiling was (slightly) attenuated relative to the mix.
        d["peak_limited"] = bool(peak >= 0.9999)
    return d


def residual_db(mix_path, stem_paths, max_seconds=None):
    """RMS(mix - sum(stems)) relative to RMS(mix), in dB. Cheap check that nothing was rescaled."""
    import numpy as np
    import soundfile as sf

    mix, sr = sf.read(str(mix_path), dtype="float32", always_2d=True)
    if max_seconds:
        mix = mix[: int(max_seconds * sr)]
    acc = np.zeros_like(mix)
    for p in stem_paths:
        s, _ = sf.read(str(p), dtype="float32", always_2d=True, frames=len(mix))
        n = min(len(s), len(acc))
        acc[:n] += s[:n]
    num = float(np.sqrt(np.mean((mix - acc) ** 2)) + 1e-12)
    den = float(np.sqrt(np.mean(mix**2)) + 1e-12)
    return 20.0 * float(np.log10(num / den))


def process_track(row, backend, args, model_dir):
    track = row["track"]
    mix_path = Path(row["flac"])
    out_dir = model_dir / track
    done = out_dir / "done.json"
    if done.exists() and not args.force:
        return "skipped"
    if not mix_path.exists():
        raise FileNotFoundError(mix_path)
    if out_dir.exists():
        if lease_status(out_dir, args.lease_minutes) == "busy":
            return "busy"
        wipe_dir(out_dir)  # partial output from an earlier killed / failed attempt
    out_dir.mkdir(parents=True, exist_ok=True)
    take_lease(out_dir)

    try:
        record = _separate_and_check(row, backend, args, mix_path, out_dir, model_dir)
    except BaseException:
        # Killed (SIGTERM -> Terminated), OOM, model error, ...: leave nothing that looks
        # finished. The dir (if any) is redone by the next worker; drop the lease so it
        # doesn't have to wait out --lease_minutes.
        (out_dir / LEASE).unlink(missing_ok=True)
        raise
    tmp = out_dir / "done.json.tmp"
    tmp.write_text(json.dumps(record, indent=1))
    tmp.rename(done)  # atomic: done.json is either absent or complete
    (out_dir / LEASE).unlink(missing_ok=True)
    return record


def _separate_and_check(row, backend, args, mix_path, out_dir, model_dir):
    track = row["track"]
    t0 = time.time()
    written = backend.separate(mix_path, out_dir)
    wall = time.time() - t0

    mix_info = read_info(mix_path, with_levels=True)
    stems = {}
    for stem in STEM_ORDER + sorted(set(written) - set(STEM_ORDER)):
        if stem not in written:
            continue
        p = written[stem]
        target = out_dir / f"{stem}.flac"
        if p != target:
            p.rename(target)
        info = read_info(target, with_levels=True)
        if abs(info["frames"] - mix_info["frames"]) > 1:
            raise RuntimeError(
                f"{track}/{stem}: {info['frames']} frames vs mix {mix_info['frames']} (>1 frame off)"
            )
        if info["samplerate"] != mix_info["samplerate"]:
            raise RuntimeError(f"{track}/{stem}: sr {info['samplerate']} vs mix {mix_info['samplerate']}")
        stems[stem] = {"file": target.name, **info}
    if not stems:
        raise RuntimeError(f"{track}: separator wrote no stems")

    resid = residual_db(mix_path, [out_dir / s["file"] for s in stems.values()], max_seconds=args.residual_seconds)

    record = {
        "track": track,
        "mix": str(mix_path),
        "mix_info": mix_info,
        "backend": args.backend,
        "model": args.model,
        "model_name": model_dir.name,
        "checkpoint": str(backend.model_path),
        "checkpoint_sha256": backend.ckpt_sha256,
        "model_config": str(backend.yaml_path) if backend.yaml_path else None,
        "model_stems": backend.model_stems,
        "settings": backend.settings,
        "versions": backend.versions(),
        "stems": stems,
        "residual_db": resid,  # 20*log10(rms(mix - sum stems) / rms(mix)); ~-inf if perfect
        "wall_sec": round(wall, 2),
        "realtime_factor": round(mix_info["frames"] / mix_info["samplerate"] / max(wall, 1e-6), 2),
        "host": WORKER_ID["host"],
        "slurm_job_id": WORKER_ID["slurm_job_id"],
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT"),
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wjd_root", default=WJD_ROOT)
    ap.add_argument("--manifest", default=None, help="default <wjd_root>/audio/manifest.csv")
    ap.add_argument("--out_root", default=None, help="default <wjd_root>/stems")
    ap.add_argument("--backend", default="audio-separator", choices=["audio-separator"])
    ap.add_argument("--model", default="BS-Roformer-SW.ckpt", help="audio-separator model filename")
    ap.add_argument("--model_name", default=None, help="output subdir; default = model filename stem")
    ap.add_argument("--model_file_dir", default=MODEL_DIR)
    ap.add_argument("--status", nargs="*", default=["pass"], help="manifest statuses to include")
    ap.add_argument("--tracks", nargs="*", default=None, help="explicit track names (overrides --status)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="redo tracks that already have done.json")
    ap.add_argument("--reverse", action="store_true", help="process in reverse manifest order (run two workers that meet in the middle)")
    ap.add_argument("--lease_minutes", type=float, default=30.0, help="another worker's _inprogress.json older than this is treated as dead")
    ap.add_argument("--overlap", type=int, default=None, help="MDXC/RoFormer overlapping windows (default: model config)")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--autocast", action="store_true", help="torch autocast (fp16) during inference")
    ap.add_argument("--normalization", type=float, default=1.0, help="peak above which a stem is scaled down (1.0 = only to avoid clipping)")
    ap.add_argument("--sample_rate", type=int, default=44100)
    ap.add_argument("--residual_seconds", type=float, default=None, help="limit the mix-vs-sum check to the first N s")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout, force=True)
    signal.signal(signal.SIGTERM, _on_sigterm)
    logging.info("worker %s", WORKER_ID)
    wjd_root = Path(args.wjd_root)
    manifest = Path(args.manifest) if args.manifest else wjd_root / "audio" / "manifest.csv"
    out_root = Path(args.out_root) if args.out_root else wjd_root / "stems"
    model_name = args.model_name or Path(args.model).stem
    model_dir = out_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    Path(args.model_file_dir).mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest, set(args.status or []), set(args.tracks or []), args.limit)
    todo = [r for r in rows if args.force or not (model_dir / r["track"] / "done.json").exists()]
    if args.reverse:
        todo.reverse()
    logging.info("%d tracks selected, %d to do, output %s", len(rows), len(todo), model_dir)
    if not todo:
        return

    backend = AudioSeparatorBackend(args)
    logging.info("model %s loaded in %.1fs; stems %s; settings %s", args.model, backend.load_sec, backend.model_stems, backend.settings)

    n_ok = n_fail = n_skip = 0
    busy = []
    t_start = time.time()

    def run_one(r, i, n):
        nonlocal n_ok, n_fail, n_skip
        track = r["track"]
        try:
            rec = process_track(r, backend, args, model_dir)
        except Terminated:
            raise
        except Exception as e:  # noqa: BLE001 - keep the batch going, record the failure
            n_fail += 1
            logging.exception("[%d/%d] %s FAILED: %s", i, n, track, e)
            fdir = model_dir / track
            fdir.mkdir(parents=True, exist_ok=True)
            (fdir / "failed.json").write_text(json.dumps({"track": track, "error": repr(e), **WORKER_ID, "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1))
            return
        if rec == "skipped":
            n_skip += 1
        elif rec == "busy":
            busy.append(r)
            logging.info("[%d/%d] %s leased by another worker, deferring", i, n, track)
        else:
            n_ok += 1
            logging.info("[%d/%d] %s ok  %.0fs (%.1fx realtime)  residual %.1f dB", i, n, track, rec["wall_sec"], rec["realtime_factor"], rec["residual_db"])

    try:
        for i, r in enumerate(todo, 1):
            run_one(r, i, len(todo))
        if busy:
            # One more pass over tracks another worker held: it has finished them (skip),
            # is still on them (still busy -> left for a later run), or died (lease expired).
            retry, busy = busy, []
            logging.info("retrying %d deferred tracks", len(retry))
            for i, r in enumerate(retry, 1):
                run_one(r, i, len(retry))
    except Terminated as e:
        logging.warning("terminated (%s) after %d ok, %d failed; current track left for the next worker", e, n_ok, n_fail)
        sys.exit(143)

    logging.info("done: %d ok, %d failed, %d already done, %d left leased to other workers, %.1f min", n_ok, n_fail, n_skip, len(busy), (time.time() - t_start) / 60)
    sys.exit(1 if n_fail and not n_ok else 0)


if __name__ == "__main__":
    main()
