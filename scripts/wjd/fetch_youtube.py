#!/usr/bin/env python
"""Fetch WJD track audio from YouTube using the JazzTube link table (experiment 3.1, route C).

Standalone: stdlib + the `yt-dlp` and `ffmpeg` executables. Nothing here imports
stable_audio_3, so it can run from any Python with yt-dlp installed.

Inputs (all under --wjd_root, see that directory's README):
    wjazzd.db                                   WJD v2.1 SQLite
    jsd/data/track_durations.csv                full-track duration of the edit the annotators used
    jazztube/csv_youtube.csv                    Balke et al. (2018) verified links, 988 rows
    jazztube/liveness_<date>.json               {youtube_id: http_status}, 200 = alive

Selection: tracks whose lineup has a drummer, that have a JSD entry (needed later for segment
cutting and now for the duration check), and at least one live linked video. One video per
track: the candidate covering the most of the track's linked solos, ties broken by the lowest
JazzTube chroma distance (mf_min). Before downloading, each candidate's duration is read from
YouTube metadata and compared with the JSD duration; the first within --duration_tol wins,
otherwise the best-ranked candidate is taken and the track is flagged `check`.

Outputs (under --out, default <wjd_root>/audio):
    raw/<Track>/<youtube_id>.<ext>              the stream as served (best audio, usually opus/m4a)
    aligned/<Track>.flac                        44.1 kHz stereo 16-bit FLAC transcode
    aligned/<Track>.alignment.json              offsets, durations, status, provenance
    manifest.csv                                one row per selected track (rewritten each run)
    _failed.json                                tracks with no usable candidate, with reasons

Time reference written to alignment.json (see experiments/03-wjd-jazz-stems/01-audio-acquisition.md):
    JazzTube:  video_time = excerpt_time + solo_start_sec
    WJD:       track_time = excerpt_time + solostart_sec
    =>         track_time = file_time + offset_sec,  offset_sec = solostart_sec - solo_start_sec
`offset_sec` is the median over the track's linked solos; `offset_spread_sec` is their range,
and a spread over --offset_tol means two solos disagree about where the video starts.

Resumable: a track with an existing alignment.json is skipped unless --redo is given.
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = "/data/hai-res/shared/snnithya/sao-3/data/wjd"
DRUM_RE = re.compile(r"\bdr\b")


# --------------------------------------------------------------------------- loading

def load_wjd(db_path):
    con = sqlite3.connect(str(db_path))
    tracks = {}
    for trackid, name, lineup, mbzid, recdate in con.execute(
        "select trackid, filename_track, lineup, mbzid, recordingdate from track_info"
    ):
        tracks[name] = {
            "trackid": trackid,
            "lineup": lineup or "",
            "has_drums": bool(DRUM_RE.search(lineup or "")),
            "mbzid": mbzid or "",
            "recordingdate": recdate or "",
            "solos": [],
        }
    by_trackid = {t["trackid"]: name for name, t in tracks.items()}
    for melid, trackid, instrument, performer, title, style, solostart in con.execute(
        "select s.melid, s.trackid, s.instrument, s.performer, s.title, s.style, ti.solostart_sec "
        "from solo_info s join transcription_info ti on ti.melid = s.melid"
    ):
        b0, b1 = con.execute("select min(onset), max(onset) from beats where melid=?", (melid,)).fetchone()
        tracks[by_trackid[trackid]]["solos"].append(
            {
                "melid": melid,
                "instrument": instrument,
                "performer": performer,
                "title": title,
                "style": style,
                "solostart_sec": solostart,
                "solo_duration_sec": (b1 - b0) if b0 is not None else None,
            }
        )
    con.close()
    return tracks


def load_jsd_durations(csv_path):
    return {r["track_name"]: float(r["duration"]) for r in csv.DictReader(open(csv_path))}


def jsd_name(wjd_name, jsd_durations):
    """WJD spells five hyphenated titles with '-', the JSD with '='. Exact match first."""
    if wjd_name in jsd_durations:
        return wjd_name
    alt = wjd_name.replace("-", "=")
    return alt if alt in jsd_durations else None


def load_links(csv_path, liveness_path):
    live = json.load(open(liveness_path))
    links = defaultdict(list)  # melid -> rows
    for r in csv.DictReader(open(csv_path)):
        yt = r["youtube_id"]
        links[int(r["melid"])].append(
            {
                "youtube_id": yt,
                "alive": live.get(yt) == 200,
                "http_status": live.get(yt),
                "solo_start_sec": float(r["solo_start_sec"]),
                "solo_end_sec": float(r["solo_end_sec"]),
                "mf_min": float(r["mf_min"]),
                "mf_median": float(r["mf_median"]),
            }
        )
    return links


# --------------------------------------------------------------------------- selection

def rank_candidates(track, links):
    """Live videos linked to any solo of the track, best first."""
    per_video = defaultdict(dict)  # youtube_id -> {melid: link row}
    for solo in track["solos"]:
        for row in links.get(solo["melid"], []):
            if row["alive"]:
                per_video[row["youtube_id"]][solo["melid"]] = row
    ranked = sorted(
        per_video.items(),
        key=lambda kv: (-len(kv[1]), min(r["mf_min"] for r in kv[1].values())),
    )
    return [{"youtube_id": yt, "solos": rows} for yt, rows in ranked]


def select_tracks(tracks, links, jsd_durations, require_drums=True, require_jsd=True):
    selected, skipped = {}, {}
    for name, t in sorted(tracks.items()):
        jname = jsd_name(name, jsd_durations)
        cands = rank_candidates(t, links)
        if require_drums and not t["has_drums"]:
            skipped[name] = "no drummer in lineup"
        elif require_jsd and jname is None:
            skipped[name] = "no JSD entry"
        elif not cands:
            skipped[name] = "no live linked video"
        else:
            selected[name] = {"jsd_name": jname, "jsd_duration": jsd_durations.get(jname), "candidates": cands}
    return selected, skipped


# --------------------------------------------------------------------------- tools

def run(cmd, timeout=None, capture=True):
    return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)


def ytdlp_base(args):
    cmd = [args.yt_dlp, "--no-playlist", "--no-warnings", "--retries", "5", "--fragment-retries", "5",
           "--sleep-requests", "1", "--ffmpeg-location", str(Path(args.ffmpeg).parent)]
    if args.cookies:
        cmd += ["--cookies", args.cookies]
    return cmd


def probe_video(args, youtube_id):
    """Metadata only, no download. Returns dict or None (blocked / removed / error)."""
    cmd = ytdlp_base(args) + ["-J", f"https://www.youtube.com/watch?v={youtube_id}"]
    try:
        p = run(cmd, timeout=180)
    except subprocess.TimeoutExpired:
        return None, "probe timeout"
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ["probe failed"])[-1][:200]
    info = json.loads(p.stdout)
    return {
        "title": info.get("title"),
        "duration": info.get("duration"),
        "uploader": info.get("uploader"),
        "upload_date": info.get("upload_date"),
    }, None


def download_audio(args, youtube_id, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    template = str(raw_dir / "%(id)s.%(ext)s")
    cmd = ytdlp_base(args) + [
        "-f", "bestaudio/best",
        "-o", template,
        "--print", "after_move:filepath",
        "--print", "after_move:%(acodec)s|%(abr)s|%(asr)s|%(ext)s|%(format_id)s",
        "--no-simulate",
        f"https://www.youtube.com/watch?v={youtube_id}",
    ]
    try:
        p = run(cmd, timeout=1800)
    except subprocess.TimeoutExpired:
        return None, None, "download timeout"
    if p.returncode != 0:
        return None, None, (p.stderr.strip().splitlines() or ["download failed"])[-1][:300]
    lines = [l for l in p.stdout.strip().splitlines() if l.strip()]
    filepath = lines[0] if lines else None
    fmt = None
    if len(lines) > 1:
        acodec, abr, asr, ext, fid = (lines[1].split("|") + [None] * 5)[:5]
        fmt = {"acodec": acodec, "abr_kbps": abr, "asr_hz": asr, "ext": ext, "format_id": fid}
    if not filepath or not os.path.exists(filepath):
        # fall back to whatever landed in raw_dir for this id
        found = sorted(raw_dir.glob(f"{youtube_id}.*"))
        filepath = str(found[0]) if found else None
    if not filepath:
        return None, None, "download produced no file"
    return filepath, fmt, None


def transcode_flac(args, src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.flac")
    cmd = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn",
           "-ac", "2", "-ar", "44100", "-sample_fmt", "s16", "-c:a", "flac", str(tmp)]
    p = run(cmd, timeout=1800)
    if p.returncode != 0:
        tmp.unlink(missing_ok=True)
        return p.stderr.strip()[-300:]
    tmp.rename(dst)
    return None


def ffprobe_duration(args, path):
    ffprobe = str(Path(args.ffmpeg).parent / "ffprobe")
    p = run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    try:
        return float(p.stdout.strip())
    except ValueError:
        return None


# --------------------------------------------------------------------------- per track

def alignment_record(name, track, sel, cand, meta, fmt, file_duration, raw_path, args, flags):
    per_solo = []
    for solo in track["solos"]:
        row = cand["solos"].get(solo["melid"])
        entry = {
            "melid": solo["melid"],
            "instrument": solo["instrument"],
            "performer": solo["performer"],
            "wjd_solostart_sec": solo["solostart_sec"],
            "wjd_solo_duration_sec": solo["solo_duration_sec"],
            "linked_to_this_video": row is not None,
        }
        if row:
            entry.update(
                {
                    "jazztube_solo_start_sec": row["solo_start_sec"],
                    "jazztube_solo_end_sec": row["solo_end_sec"],
                    "mf_min": row["mf_min"],
                    "offset_sec": solo["solostart_sec"] - row["solo_start_sec"], # solostart_sec is from wjd and solo_start_sec is from jazztube
                }
            )
        per_solo.append(entry)
    offsets = [e["offset_sec"] for e in per_solo if "offset_sec" in e]
    offset = statistics.median(offsets) if offsets else None
    spread = (max(offsets) - min(offsets)) if offsets else None
    jsd_dur = sel["jsd_duration"]
    delta = (file_duration - jsd_dur) if (file_duration is not None and jsd_dur is not None) else None

    if delta is not None and abs(delta) > args.duration_tol:
        flags.append(f"duration differs from JSD by {delta:+.1f}s")
    if spread is not None and spread > args.offset_tol:
        flags.append(f"solo offsets disagree by {spread:.2f}s")
    unlinked = [e["melid"] for e in per_solo if not e["linked_to_this_video"]]
    if unlinked:
        flags.append(f"solos not linked to this video: {unlinked}")

    return {
        "track": name,
        "jsd_track": sel["jsd_name"],
        "lineup": track["lineup"],
        "mbzid": track["mbzid"],
        "recordingdate": track["recordingdate"],
        "source": {
            "kind": "youtube",
            "youtube_id": cand["youtube_id"],
            "url": f"https://www.youtube.com/watch?v={cand['youtube_id']}",
            "title": meta.get("title") if meta else None,
            "uploader": meta.get("uploader") if meta else None,
            "upload_date": meta.get("upload_date") if meta else None,
            "stream": fmt,
            "raw_file": str(raw_path) if raw_path else None,
        },
        "duration_file_sec": file_duration,
        "duration_jsd_sec": jsd_dur,
        "duration_delta_sec": delta,
        "offset_sec": offset,
        "offset_spread_sec": spread,
        "time_reference": "track_time = file_time + offset_sec",
        "per_solo": per_solo,
        "status": "check" if flags else "pass",
        "flags": flags,
        "checks": {"duration": delta is not None, "offset_consistency": spread is not None, "beat_check": None},
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def process_track(name, track, sel, args, out):
    aligned_dir = out / "aligned"
    json_path = aligned_dir / f"{name}.alignment.json"
    flac_path = aligned_dir / f"{name}.flac"
    if json_path.exists() and flac_path.exists() and not args.redo:
        return "skipped", json.load(open(json_path)).get("status")

    raw_dir = out / "raw" / name
    jsd_dur = sel["jsd_duration"]
    chosen, chosen_meta, fallback, fallback_meta, errors = None, None, None, None, []
    for cand in sel["candidates"][: args.max_candidates]:
        meta, err = probe_video(args, cand["youtube_id"])
        if meta is None:
            errors.append(f"{cand['youtube_id']}: {err}")
            continue
        dur = meta.get("duration")
        if jsd_dur is not None and dur is not None and abs(dur - jsd_dur) <= args.duration_tol:
            chosen, chosen_meta = cand, meta
            break
        if fallback is None:
            fallback, fallback_meta = cand, meta
        errors.append(f"{cand['youtube_id']}: duration {dur} vs JSD {jsd_dur:.0f}")
    flags = []
    if chosen is None:
        if fallback is None:
            return "failed", {"reasons": errors}
        chosen, chosen_meta = fallback, fallback_meta
        flags.append("no candidate matched the JSD duration; took the best-ranked one")

    if args.dry_run:
        return "dry_run", {"youtube_id": chosen["youtube_id"], "duration": chosen_meta.get("duration"), "flags": flags}

    raw_path, fmt, err = download_audio(args, chosen["youtube_id"], raw_dir)
    if err:
        return "failed", {"reasons": errors + [f"{chosen['youtube_id']}: {err}"]}
    err = transcode_flac(args, raw_path, flac_path)
    if err:
        return "failed", {"reasons": errors + [f"ffmpeg: {err}"]}
    file_dur = ffprobe_duration(args, flac_path)
    rec = alignment_record(name, track, sel, chosen, chosen_meta, fmt, file_dur, raw_path, args, flags)
    rec["candidates_tried"] = errors
    json.dump(rec, open(json_path, "w"), indent=2)
    return "done", rec["status"]


# --------------------------------------------------------------------------- main

def write_manifest(path, selected, tracks, results, out):
    fields = ["track", "jsd_track", "status", "result", "youtube_id", "duration_delta_sec", "offset_sec",
              "n_candidates", "n_solos", "solo_instruments", "lineup", "recordingdate", "mbzid", "flac"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, sel in sorted(selected.items()):
            t = tracks[name]
            jp = out / "aligned" / f"{name}.alignment.json"
            rec = json.load(open(jp)) if jp.exists() else {}
            w.writerow(
                {
                    "track": name,
                    "jsd_track": sel["jsd_name"],
                    "status": rec.get("status", ""),
                    "result": results.get(name, ""),
                    "youtube_id": rec.get("source", {}).get("youtube_id", ""),
                    "duration_delta_sec": rec.get("duration_delta_sec", ""),
                    "offset_sec": rec.get("offset_sec", ""),
                    "n_candidates": len(sel["candidates"]),
                    "n_solos": len(t["solos"]),
                    "solo_instruments": " ".join(s["instrument"] for s in t["solos"]),
                    "lineup": t["lineup"],
                    "recordingdate": t["recordingdate"],
                    "mbzid": t["mbzid"],
                    "flac": str(out / "aligned" / f"{name}.flac") if jp.exists() else "",
                }
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wjd_root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=None, help="default <wjd_root>/audio")
    ap.add_argument("--liveness", default=None, help="default: newest jazztube/liveness_*.json")
    ap.add_argument("--yt_dlp", default=shutil.which("yt-dlp") or "yt-dlp")
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    ap.add_argument("--cookies", default=os.environ.get("YTDLP_COOKIES") or None,
                    help="Netscape cookies file, if YouTube starts demanding a sign-in from this IP")
    ap.add_argument("--duration_tol", type=float, default=15.0, help="seconds vs JSD track duration")
    ap.add_argument("--offset_tol", type=float, default=0.5, help="seconds of disagreement between solos")
    ap.add_argument("--max_candidates", type=int, default=4, help="videos to probe per track")
    ap.add_argument("--sleep", type=float, nargs=2, default=(3.0, 8.0), metavar=("MIN", "MAX"),
                    help="random pause between tracks, seconds")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N selected tracks")
    ap.add_argument("--tracks", nargs="*", default=None, help="restrict to these filename_track values")
    ap.add_argument("--no_require_drums", action="store_true")
    ap.add_argument("--no_require_jsd", action="store_true")
    ap.add_argument("--redo", action="store_true", help="re-fetch tracks that already have an alignment.json")
    ap.add_argument("--dry_run", action="store_true", help="select and probe, download nothing")
    args = ap.parse_args()

    root = Path(args.wjd_root)
    out = Path(args.out) if args.out else root / "audio"
    out.mkdir(parents=True, exist_ok=True)
    liveness = Path(args.liveness) if args.liveness else sorted((root / "jazztube").glob("liveness_*.json"))[-1]

    for tool in (args.yt_dlp, args.ffmpeg):
        if shutil.which(tool) is None and not os.path.exists(tool):
            sys.exit(f"tool not found: {tool}")
    print(f"yt-dlp: {run([args.yt_dlp, '--version']).stdout.strip()}  ffmpeg: {args.ffmpeg}")
    print(f"liveness table: {liveness}")

    tracks = load_wjd(root / "wjazzd.db")
    jsd_durations = load_jsd_durations(root / "jsd" / "data" / "track_durations.csv")
    links = load_links(root / "jazztube" / "csv_youtube.csv", liveness)
    selected, skipped = select_tracks(
        tracks, links, jsd_durations, require_drums=not args.no_require_drums, require_jsd=not args.no_require_jsd
    )
    from collections import Counter
    print(f"selected {len(selected)} tracks; skipped {len(skipped)}: {dict(Counter(skipped.values()))}")

    names = sorted(selected)
    if args.tracks:
        names = [n for n in names if n in set(args.tracks)]
    if args.limit:
        names = names[: args.limit]

    results, failed = {}, {}
    t_start = time.time()
    for i, name in enumerate(names, 1):
        try:
            kind, detail = process_track(name, tracks[name], selected[name], args, out)
        except Exception as e:  # keep going; one bad track must not kill a 200-track job
            kind, detail = "failed", {"reasons": [f"exception: {e!r}"]}
        results[name] = kind
        if kind == "failed":
            failed[name] = detail
        print(f"[{i}/{len(names)}] {name}: {kind} {detail if kind != 'done' else ''}".rstrip(), flush=True)
        if kind in ("done", "failed") and i < len(names):
            time.sleep(random.uniform(*args.sleep))

    json.dump(failed, open(out / "_failed.json", "w"), indent=2)
    write_manifest(out / "manifest.csv", selected, tracks, results, out)

    statuses = Counter()
    for jp in (out / "aligned").glob("*.alignment.json"):
        statuses[json.load(open(jp)).get("status")] += 1
    print(f"\nrun results: {dict(Counter(results.values()))}")
    print(f"on disk: {sum(statuses.values())} aligned tracks, by status {dict(statuses)}; "
          f"{len(failed)} failed this run (see {out / '_failed.json'})")
    print(f"manifest: {out / 'manifest.csv'}   elapsed {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
