#!/usr/bin/env python
"""Experiment 3.3: lay the separated WJD stems out as a condition/target mirror.

The analogue of sat-zenon's ``download_and_setup_slakh.py`` / ``setup_slakh_instrument.py``
for the Weimar Jazz Database: one directory per instrument, one subdirectory per track, so a
dataset config can point at ``tracks/<instrument>`` and the custom metadata module
(``custom_md_wjd.py``) can find the track's drums next door. Here the roles are inverted
relative to the Slakh streamgen mirror: **drums are the condition**, every other stem is a
**target** selected by the text prompt.

Input   <wjd_root>/stems/<model>/<Track>/{drums,bass,other,vocals,piano,guitar}.flac + done.json
        <wjd_root>/audio/aligned/<Track>.alignment.json      (offset between file and WJD time)
        <wjd_root>/jsd/data/annotations_csv/<Track>.csv      (structure segments, WJD time)
        <wjd_root>/wjazzd.db                                   (lineup, solo_info)
Output  <out>/<split>/tracks/drums/<Track>/drums.flac          <- condition
        <out>/<split>/tracks/<inst>/<Track>/<inst>.flac        <- one target per instrument stem
        <out>/<split>/meta/<Track>.json                        <- lineup, solos, JSD segments (both
                                                                  time references), stem levels, prompts
        <out>/<split>/report.html                              <- listening page, as the Slakh scripts write
        <out>/splits.json                                      <- which track went where, with the seed
        <out>/_skipped.json                                    <- tracks / stems left out, and why

Items are whole tracks (decision 2026-10-01); the JSD segments are written to ``meta/`` so a
later segment-level cut or a segment-aware prompt can be built from the same tree without
touching the database again. Time reference, from the alignment step: ``track_time =
file_time + offset_sec``, where ``track_time`` is the WJD / JSD clock and ``file_time`` is a
position in our audio (and so in every stem). Segments are stored in both.

Stems are *copied* by default (decision 2026-10-01; ``--link hardlink`` gives the same tree
at zero disk cost on one filesystem). A target stem whose whole-file RMS is below
``--min_stem_db`` is not mirrored: the separator writes every stem it knows about, so a
track without a guitarist still gets a ``guitar.flac`` of near-silence, and 132/166 guitar
stems are that. A track whose *drums* fall below the floor is skipped entirely, since the
condition would carry nothing. ``vocals`` is never mirrored.

Splits are by track (decision 2026-10-01), a seeded shuffle; pass ``--split_json`` to reuse
an existing assignment instead of drawing a new one.

Example:
    python scripts/wjd/make_stem_mirror.py --dry_run
    python scripts/wjd/make_stem_mirror.py
"""

import argparse
import csv
import html
import json
import os
import random
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

WJD_ROOT = "/data/hai-res/shared/snnithya/sao-3/data/wjd"
DEFAULT_MODEL = "BS-Roformer-SW"
CONDITION = "drums"
DEFAULT_TARGETS = ["bass", "other", "piano", "guitar"]
NEVER_MIRROR = {"vocals"}

# WJD lineup / JSD abbreviations -> instrument names. The first block is
# jsd/data/instruments.csv verbatim (lowercased); the rest are abbreviations that occur in
# track_info.lineup but not in that table. A hyphenated abbreviation ("p-tp": one player on
# piano and trumpet) is split on the hyphen before lookup.
INSTRUMENT_NAMES = {
    "cl": "clarinet",
    "bcl": "bass clarinet",
    "bc": "bass clarinet",
    "ss": "soprano saxophone",
    "as": "alto saxophone",
    "ts": "tenor saxophone",
    "ts-c": "tenor saxophone",  # C-melody / tenor doubling; one lineup uses it
    "bs": "baritone saxophone",
    "tp": "trumpet",
    "fln": "flugelhorn",
    "flgn": "flugelhorn",
    "cor": "cornet",
    "cn": "cornet",
    "tb": "trombone",
    "p": "piano",
    "key": "keyboard",
    "synth": "synthesizer",
    "vib": "vibraphone",
    "voc": "vocals",
    "fl": "flute",
    "g": "guitar",
    "bjo": "banjo",
    "vc": "cello",
    "vcl": "cello",
    "b": "bass",
    "dr": "drums",
    "perc": "percussion",
    "cga": "congas",
    "hca": "harmonica",
}

# What the rhythm section is, for the purpose of naming what is left in the `other` stem.
# These instruments have their own separator stem (or, for percussion, land in `drums`),
# so they are not what `other` sounds like. `voc` has its own stem too and is never a target.
RHYTHM_SECTION = {"dr", "b", "p", "g", "key", "synth", "perc", "cga", "voc", "bjo"}

# Prompt for each target stem. `other` is built from the lineup's front line instead.
PROMPT_BY_STEM = {
    "bass": "upright bass",
    "piano": "piano",
    "guitar": "guitar",
}


# ---------------------------------------------------------------------------
# WJD lookups
# ---------------------------------------------------------------------------

def parse_lineup(lineup):
    """'Art Pepper (as, cl); Charlie Haden (b)' -> [{'performer', 'abbr': [...], 'instruments': [...]}]."""
    players = []
    for part in (lineup or "").split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", part)
        if not m:
            players.append({"performer": part, "abbr": [], "instruments": []})
            continue
        abbrs = []
        for a in m.group(2).split(","):
            a = a.strip()
            if not a:
                continue
            if a in INSTRUMENT_NAMES:
                abbrs.append(a)
            else:
                abbrs.extend(x for x in a.split("-") if x)
        players.append({
            "performer": m.group(1).strip(),
            "abbr": abbrs,
            "instruments": [instrument_name(a) for a in abbrs],
        })
    return players


def instrument_name(abbr):
    return INSTRUMENT_NAMES.get(abbr, INSTRUMENT_NAMES.get(abbr.lower(), abbr))


def front_line(players):
    """Instrument names in the lineup that are not rhythm section, in lineup order, deduplicated."""
    seen, out = set(), []
    for p in players:
        for a in p["abbr"]:
            if a in RHYTHM_SECTION or a.lower() in RHYTHM_SECTION:
                continue
            name = instrument_name(a)
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def prompt_for(stem, players):
    """Short instrument-name prompt for a target stem, or None if the stem cannot be named."""
    if stem in PROMPT_BY_STEM:
        return PROMPT_BY_STEM[stem]
    if stem == "other":
        names = front_line(players)
        return ", ".join(names) if names else None
    return stem


def decade_of(recordingdate):
    m = re.match(r"(\d{4})", recordingdate or "")
    return f"{int(m.group(1)) // 10 * 10}s" if m else None


def load_db(db_path):
    con = sqlite3.connect(str(db_path))
    tracks = {}
    for trackid, fname, lineup, mbzid, recordingdate, recordid in con.execute(
        "select trackid, filename_track, lineup, mbzid, recordingdate, recordid from track_info"
    ):
        tracks[fname] = {
            "trackid": trackid, "lineup": lineup, "mbzid": mbzid,
            "recordingdate": recordingdate, "recordid": recordid, "solos": [],
        }
    by_id = {v["trackid"]: v for v in tracks.values()}
    for row in con.execute(
        "select s.melid, s.trackid, s.performer, s.instrument, s.style, s.rhythmfeel, s.avgtempo, "
        "s.key, s.signature, s.chorus_count, t.solostart_sec "
        "from solo_info s left join transcription_info t on s.melid = t.melid"
    ):
        melid, trackid, performer, instrument, style, feel, tempo, key, sig, choruses, solostart = row
        if trackid in by_id:
            by_id[trackid]["solos"].append({
                "melid": melid, "performer": performer, "instrument": instrument,
                "instrument_name": instrument_name(instrument or ""),
                "style": style, "rhythmfeel": feel, "avgtempo": tempo, "key": key,
                "signature": sig, "chorus_count": choruses, "solostart_sec": solostart,
            })
    con.close()
    return tracks


def load_jsd_segments(csv_path, offset_sec):
    """JSD segments with both clocks. Returns [] if the track has no JSD file."""
    if not csv_path.exists():
        return []
    segs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f, delimiter=";"):
            start, end = float(r["segment_start"]), float(r["segment_end"])
            inst = [x for x in (r.get("instrument") or "").split(",") if x]
            segs.append({
                "label": r["label"],
                "track_start": start, "track_end": end,
                "file_start": start - offset_sec, "file_end": end - offset_sec,
                "soloists": [x[2:] for x in inst if x.startswith("s_")],
                "backing": [x[2:] for x in inst if x.startswith("b_")],
                "instruments": [x for x in inst if not x.startswith(("s_", "b_"))],
            })
    return segs


# ---------------------------------------------------------------------------
# Mirror
# ---------------------------------------------------------------------------

def place_file(src, dst, mode, dry_run):
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "skip"
        if not dry_run:
            dst.unlink()
    if dry_run:
        return mode
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    return mode


def assign_splits(track_names, val_fraction, seed, holdout=()):
    rng = random.Random(seed)
    names = sorted(track_names)
    rng.shuffle(names)
    n_val = max(1, round(len(names) * val_fraction)) if val_fraction > 0 else 0
    splits = {}
    for i, t in enumerate(names):
        splits[t] = "validation" if i < n_val else "train"
    for t in holdout:
        if t in splits:
            splits[t] = "holdout"
    return splits


def write_report(rows, split_root, title):
    def audio_cell(items):
        parts = []
        for label, rel in items:
            parts.append(
                f'<div class="audio-item"><div class="path">{html.escape(label)} — {html.escape(rel)}</div>'
                f'<audio controls preload="none" src="{html.escape(rel)}"></audio></div>'
            )
        return "".join(parts) or "<em>none</em>"

    trs = []
    for r in rows:
        targets = [(f"{inst}: “{r['prompts'][inst]}”", f"tracks/{inst}/{r['track']}/{inst}.flac") for inst in r["targets"]]
        drums = [("drums", f"tracks/{CONDITION}/{r['track']}/{CONDITION}.flac")]
        trs.append(
            "<tr>"
            f"<td><b>{html.escape(r['track'])}</b><br><small>{html.escape(r['lineup'] or '')}<br>"
            f"{html.escape(r['decade'] or '?')} · {r['duration_sec']:.0f} s · drums {r['levels'][CONDITION]:.1f} dBFS</small></td>"
            f"<td>{audio_cell(drums)}</td><td>{audio_cell(targets)}</td></tr>"
        )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
 body {{ font-family: Arial, sans-serif; padding: 16px; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; }}
 th {{ background: #f5f5f5; }}
 .audio-item {{ margin-bottom: 8px; }}
 .path {{ font-size: 12px; color: #444; margin-bottom: 4px; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>{len(rows)} tracks. Drums are the condition; each target stem is one training item, named by the prompt in quotes.
Open over <code>python -m http.server</code> in this directory.</p>
<table><thead><tr><th>Track</th><th>Drums (condition)</th><th>Targets</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></body></html>
"""
    (split_root / "report.html").write_text(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wjd_root", default=WJD_ROOT)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="subdir of <wjd_root>/stems to mirror")
    ap.add_argument("--out", default=None, help="default <wjd_root>/wjd-stem-mirror")
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS, help="stems to mirror as targets")
    ap.add_argument("--min_stem_db", type=float, default=-50.0,
                    help="whole-file RMS floor (dBFS, from done.json) below which a stem is not mirrored; "
                         "a track whose drums are below it is skipped entirely")
    ap.add_argument("--link", choices=["copy", "hardlink", "symlink"], default="copy")
    ap.add_argument("--val_fraction", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", nargs="*", default=[], help="tracks to keep out of both splits (demo material)")
    ap.add_argument("--split_json", default=None, help="reuse an existing splits.json instead of drawing one")
    ap.add_argument("--tracks", nargs="*", default=None, help="restrict to these tracks")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.wjd_root)
    stems_root = root / "stems" / args.model
    out = Path(args.out) if args.out else root / "wjd-stem-mirror"
    db = load_db(root / "wjazzd.db")

    track_dirs = sorted(p for p in stems_root.iterdir() if (p / "done.json").exists())
    if args.tracks:
        want = set(args.tracks)
        track_dirs = [p for p in track_dirs if p.name in want]
    print(f"[scan] {len(track_dirs)} separated tracks under {stems_root}")

    skipped = {"tracks": {}, "stems": {}}
    usable = []
    for tdir in track_dirs:
        done = json.loads((tdir / "done.json").read_text())
        if CONDITION not in done["stems"]:
            skipped["tracks"][tdir.name] = "no drums stem"
            continue
        if done["stems"][CONDITION]["rms_dbfs"] < args.min_stem_db:
            skipped["tracks"][tdir.name] = f"drums {done['stems'][CONDITION]['rms_dbfs']:.1f} dBFS below floor"
            continue
        if tdir.name not in db:
            skipped["tracks"][tdir.name] = "not in wjazzd.db track_info"
            continue
        usable.append((tdir, done))

    if args.split_json:
        splits = json.loads(Path(args.split_json).read_text())["splits"]
        missing = [t.name for t, _ in usable if t.name not in splits]
        if missing:
            sys.exit(f"{len(missing)} tracks missing from {args.split_json}: {missing[:5]} ...")
    else:
        splits = assign_splits([t.name for t, _ in usable], args.val_fraction, args.seed, args.holdout)

    rows_by_split = {}
    n_files = {"copy": 0, "hardlink": 0, "symlink": 0, "skip": 0}
    for tdir, done in usable:
        name = tdir.name
        split = splits[name]
        split_root = out / split
        info = db[name]
        players = parse_lineup(info["lineup"])
        align_path = root / "audio" / "aligned" / f"{name}.alignment.json"
        align = json.loads(align_path.read_text()) if align_path.exists() else {}
        offset = float(align.get("offset_sec", 0.0))
        segments = load_jsd_segments(root / "jsd" / "data" / "annotations_csv" / f"{name}.csv", offset)

        targets, prompts, levels = [], {}, {CONDITION: done["stems"][CONDITION]["rms_dbfs"]}
        for stem in args.targets:
            if stem in NEVER_MIRROR or stem not in done["stems"]:
                continue
            level = done["stems"][stem]["rms_dbfs"]
            levels[stem] = level
            if level < args.min_stem_db:
                skipped["stems"].setdefault(stem, {})[name] = f"{level:.1f} dBFS below floor"
                continue
            prompt = prompt_for(stem, players)
            if prompt is None:
                skipped["stems"].setdefault(stem, {})[name] = "no front-line instrument in lineup"
                continue
            targets.append(stem)
            prompts[stem] = prompt

        if not targets:
            skipped["tracks"][name] = "no usable target stem"
            continue

        for stem in [CONDITION] + targets:
            mode = place_file(tdir / f"{stem}.flac", split_root / "tracks" / stem / name / f"{stem}.flac",
                              args.link, args.dry_run)
            n_files[mode] += 1

        meta = {
            "track": name,
            "split": split,
            "source_stems_dir": str(tdir),
            "separator": done.get("model_name"),
            "sample_rate": done["mix_info"]["samplerate"],
            "frames": done["mix_info"]["frames"],
            "duration_sec": done["mix_info"]["frames"] / done["mix_info"]["samplerate"],
            "condition": CONDITION,
            "targets": targets,
            "prompts": prompts,
            "stem_levels_dbfs": levels,
            "lineup": info["lineup"],
            "players": players,
            "front_line": front_line(players),
            "recordingdate": info["recordingdate"],
            "decade": decade_of(info["recordingdate"]),
            "mbzid": info["mbzid"],
            "recordid": info["recordid"],
            "solos": info["solos"],
            "alignment": {
                "offset_sec": offset,
                "status": align.get("status"),
                "time_reference": "track_time = file_time + offset_sec; file_time indexes the stems",
            },
            "jsd_segments": segments,
        }
        if not args.dry_run:
            (split_root / "meta").mkdir(parents=True, exist_ok=True)
            (split_root / "meta" / f"{name}.json").write_text(json.dumps(meta, indent=1))
        rows_by_split.setdefault(split, []).append({
            "track": name, "targets": targets, "prompts": prompts, "lineup": info["lineup"],
            "decade": meta["decade"], "duration_sec": meta["duration_sec"], "levels": levels,
        })

    for split, rows in sorted(rows_by_split.items()):
        rows.sort(key=lambda r: r["track"])
        n_items = sum(len(r["targets"]) for r in rows)
        hours = sum(r["duration_sec"] * len(r["targets"]) for r in rows) / 3600
        print(f"[{split}] {len(rows)} tracks, {n_items} target items, {hours:.1f} h of target audio")
        per_stem = {}
        for r in rows:
            for s in r["targets"]:
                per_stem[s] = per_stem.get(s, 0) + 1
        print(f"         per stem: {per_stem}")
        if not args.dry_run:
            write_report(rows, out / split, f"WJD stem mirror — {split} — {args.model}")

    print(f"[files] {n_files}")
    print(f"[skipped] tracks: {len(skipped['tracks'])}, stems: "
          f"{ {k: len(v) for k, v in skipped['stems'].items()} }")
    for t, why in sorted(skipped["tracks"].items()):
        print(f"    {t}: {why}")

    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "splits.json").write_text(json.dumps({
            "seed": args.seed, "val_fraction": args.val_fraction, "by": "track",
            "holdout": args.holdout, "source": str(stems_root), "link": args.link,
            "min_stem_db": args.min_stem_db, "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counts": {s: sum(1 for v in splits.values() if v == s) for s in sorted(set(splits.values()))},
            "splits": splits,
        }, indent=1))
        (out / "_skipped.json").write_text(json.dumps(skipped, indent=1))
        print(f"[done] mirror at {out}")
    else:
        print("[dry run] nothing written")


if __name__ == "__main__":
    main()
