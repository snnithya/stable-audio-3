"""Add feature controls to an existing pre-encoded dataset without re-encoding the latents.

``pre_encode_dataset.py --features`` computes a ``__features__`` control (a ``[C, T_frames]``
tensor at the latent rate, see stable_audio_3/data/features.py) while it encodes, and fuses it
into ``{id}_controls.npy``. A new feature on an already-encoded dataset does not need the GPU
pass again: the features come from the source audio, which the sidecar JSON still points at.
Re-encoding would also re-roll the per-item polarity flip SampleDataset applies to the target
(experiments/03-wjd-jazz-stems/03-dataset-and-wiring.md, Notes), so the latents would not
even be the same ones.

For every ``{id}.json`` under each ``--preencoded_dir`` this script rebuilds the metadata
call the pre-encode made (``get_custom_metadata(info, audio)`` with a padding mask of
``--sample_size`` samples, valid up to the source length, as SampleDataset builds it with
random_crop off), takes the requested keys from ``__features__``, crops/pads them to the
stored latent length and appends them to the sidecar, updating ``control_keys`` /
``controls_dim`` in the JSON. Keys the module returns that are already in the sidecar are
recomputed and compared, requested or not: that comparison is the check that this script
reproduces what the pre-encode wrote (``drums_rms`` must come back bit-identical before any
new key is trusted). A requested key that already exists and differs is an error unless
``--replace``.

The custom metadata module reads its mode from the environment exactly as at pre-encode
time, e.g.

    WJD_CONTROL_MODE=rms,tria_fixed,tria_ema uv run python scripts/add_features_to_preencoded.py \\
        --preencoded_dir /data/hai-res/shared/snnithya/sao-3/data/wjd/preencoded/train/{bass,other,piano,guitar} \\
        --custom_metadata_module stable_audio_3/configs/dataset_configs/custom_metadata/custom_md_wjd.py \\
        --features drums_tria_fixed drums_tria_ema --sample_size 16760832

``--dry_run`` does everything but write. Writes are atomic (temp file + rename) per file, so
an interrupted run leaves every item either old or new, never half-updated.
"""

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_module(path):
    spec = importlib.util.spec_from_file_location("custom_metadata_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fit(feat, n_frames):
    feat = torch.as_tensor(feat)
    if feat.ndim == 1:
        feat = feat.unsqueeze(0)
    if feat.shape[-1] >= n_frames:
        feat = feat[:, :n_frames]
    else:
        feat = torch.nn.functional.pad(feat, (0, n_frames - feat.shape[-1]))
    return feat.float().numpy()


def _save_npy_atomic(path, arr):
    tmp = f"{path}.{os.getpid()}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _save_json_atomic(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def process_item(json_path, module, sample_size, keys, tolerance=0.0, write=True, replace=False):
    """Update one item's sidecar. Returns a dict with per-key statuses (added / replaced /
    unchanged / verified / mismatch) and the max abs difference for compared keys."""
    json_path = Path(json_path)
    item_id = json_path.stem
    with open(json_path) as f:
        info = json.load(f)
    n_frames = int(np.load(json_path.with_suffix(".npy"), mmap_mode="r").shape[-1])
    if len(info["padding_mask"]) != n_frames:
        raise ValueError(f"{item_id}: padding_mask has {len(info['padding_mask'])} frames, latent {n_frames}")

    # A chunked encode records its window in the sidecar; a whole-file encode needs the
    # pre-encode's --sample_size from the caller.
    offset = int(info.get("chunk_offset", 0))
    window = int(info.get("chunk_samples") or 0) or sample_size
    if not window:
        raise ValueError(f"{item_id}: no chunk_samples in the sidecar; pass --sample_size")
    n_src = torchaudio.info(info["path"]).num_frames
    valid = max(0, min(n_src - offset, window))
    mask = torch.zeros(window)
    mask[:valid] = 1
    md_info = {"path": info["path"], "sample_rate": info["sample_rate"], "padding_mask": [mask]}
    for key in ("chunk_index", "n_chunks", "chunk_offset", "chunk_offset_seconds", "chunk_samples"):
        if key in info:
            md_info[key] = info[key]
    md = module.get_custom_metadata(md_info, None)
    if md.get("__reject__"):
        return {"id": item_id, "status": "rejected", "reason": md.get("__reject_reason__")}
    feats = md.get("__features__", {})
    missing = [k for k in keys if k not in feats]
    if missing:
        raise KeyError(f"{item_id}: module returned no feature {missing}; returned {sorted(feats)}")

    ctrl_path = json_path.parent / f"{item_id}_controls.npy"
    old_keys = list(info.get("control_keys", []))
    old_dims = list(info.get("controls_dim", []))
    if ctrl_path.exists():
        old = np.load(ctrl_path)
        if old.shape != (sum(old_dims), n_frames):
            raise ValueError(f"{item_id}: sidecar {old.shape} vs control_keys {old_keys} dims {old_dims}, {n_frames} frames")
    else:
        old = np.zeros((0, n_frames), dtype=np.float32)
    parts, ind = {}, 0
    for k, d in zip(old_keys, old_dims):
        parts[k] = old[ind : ind + d]
        ind += d

    statuses, diffs, changed = {}, {}, False
    for key, feat in feats.items():
        if key not in keys and key not in parts:
            continue
        new = _fit(feat, n_frames)
        if key in parts:
            diff = float(np.abs(parts[key] - new).max()) if parts[key].shape == new.shape else float("inf")
            diffs[key] = diff
            if diff <= tolerance:
                statuses[key] = "unchanged" if key in keys else "verified"
            elif key in keys and replace:
                parts[key] = new
                statuses[key] = "replaced"
                changed = True
            else:
                statuses[key] = "mismatch"
        else:
            parts[key] = new
            statuses[key] = "added"
            changed = True

    new_keys = old_keys + [k for k in keys if k not in old_keys]
    if changed and write:
        fused = np.concatenate([parts[k] for k in new_keys], axis=0).astype(np.float32)
        _save_npy_atomic(ctrl_path, fused)
        info["control_keys"] = new_keys
        info["controls_dim"] = [int(parts[k].shape[0]) for k in new_keys]
        _save_json_atomic(json_path, info)
    return {"id": item_id, "status": "ok", "keys": statuses, "max_diff": diffs, "written": changed and write}


_MODULE = None


def _init_worker(module_path):
    global _MODULE
    torch.set_num_threads(1)
    _MODULE = load_module(module_path)


def _worker(args):
    json_path, sample_size, keys, tolerance, write, replace = args
    try:
        return process_item(json_path, _MODULE, sample_size, keys, tolerance, write, replace)
    except Exception as e:  # keep the pool going; the summary names the item
        return {"id": Path(json_path).stem, "status": "error", "reason": f"{type(e).__name__}: {e}", "path": str(json_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preencoded_dir", nargs="+", required=True)
    ap.add_argument("--custom_metadata_module", required=True)
    ap.add_argument("--features", nargs="+", required=True, help="keys to add, in sidecar order")
    ap.add_argument("--sample_size", type=int, default=None,
                    help="the pre-encode's --sample_size (whole-file encodes); chunked encodes carry it in the sidecar")
    ap.add_argument("--tolerance", type=float, default=0.0, help="max abs diff for 'unchanged'")
    ap.add_argument("--replace", action="store_true", help="overwrite a requested key that differs")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N items per dir (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    items = []
    for d in args.preencoded_dir:
        found = sorted(p for p in Path(d).glob("*.json") if not p.name.startswith("_"))
        items += found[: args.limit] if args.limit else found
    print(f"{len(items)} items in {len(args.preencoded_dir)} dir(s); keys {args.features}; "
          f"{'DRY RUN' if args.dry_run else 'writing'}")

    jobs = [(str(p), args.sample_size, args.features, args.tolerance, not args.dry_run, args.replace) for p in items]
    results = []
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args.custom_metadata_module,)) as pool:
        for i, r in enumerate(pool.map(_worker, jobs), 1):
            results.append(r)
            if r["status"] != "ok":
                print(f"  [{r['id']}] {r['status'].upper()}: {r.get('reason')} {r.get('path', '')}")
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done")

    counts = {}
    worst = {}
    for r in results:
        if r["status"] != "ok":
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            continue
        for k, s in r["keys"].items():
            counts[f"{k}: {s}"] = counts.get(f"{k}: {s}", 0) + 1
        for k, d in r["max_diff"].items():
            worst[k] = max(worst.get(k, 0.0), d)
    print("summary:")
    for k in sorted(counts):
        print(f"  {counts[k]:>6}  {k}")
    for k, d in sorted(worst.items()):
        print(f"  max abs diff vs stored {k}: {d:g}")
    bad = sum(v for k, v in counts.items() if k.endswith("mismatch") or k in ("error", "rejected"))
    if bad:
        sys.exit(f"{bad} item/key problem(s); nothing to do with them was written")


if __name__ == "__main__":
    main()
