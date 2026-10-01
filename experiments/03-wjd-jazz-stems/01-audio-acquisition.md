# 3.1 — Audio acquisition + alignment

**Status:** route C chosen (2026-09-22), fetch script + sbatch written and smoke-tested; full download not yet run
**Depends on:** nothing. Everything else in 03 depends on this.

## Question

How do we obtain the 344 (340 unique) commercial recordings the WJD annotates, and how do we
know that a given file is **the same take, at a known time offset**, as the one the
transcribers worked from?

The second half is not optional and is route-independent: whichever way the audio arrives,
remasters shift the start by seconds, 78-rpm transfers run at different speeds, and albums
contain alternate takes of the same tune with different solos. The annotations are only usable
if each file is verified against the transcription itself.

## What the database gives us to find the audio

Per track (`track_info` ⨝ `record_info` ⨝ `composition_info`):

| Field | Use |
|---|---|
| `track_info.mbzid` | **MusicBrainz recording id**, 302 of 344 tracks. A recording id resolves to every release it appears on, ISRCs, and (via URL relationships) store / streaming links. |
| `record_info.artist`, `recordtitle`, `label`, `recordbib` | Album, label and catalogue number for the 184 records — enough to buy or find any of them by hand. |
| `composition_info.title`, `solo_info.performer` | Text queries ("Art Pepper Anthropology"). |
| `transcription_info.solostart_sec`, solo length from `beats` | Where in the track the solo sits — the alignment window. |
| JSD `track_durations.csv` | Full-track duration of the exact edit the annotators had (340 tracks). |

**First deliverable:** `scripts/wjd/build_manifest.py` → `/data/hai-res/shared/snnithya/sao-3/data/wjd/manifest.csv`, one row per
track with all of the above plus `has_drums`, decade, the list of solos (melid, instrument,
start, length) and the JSD duration. Every later script keys on `filename_track`.

## Acquisition routes

| Route | Coverage | Cost | Caveats |
|---|---|---|---|
| **A. Ask the Jazzomat group** (HfM Weimar; Klaus Frieler / Martin Pfleiderer) and **Stefan Balke** (JazzTube author) | Working audio for all 456 solos exists on their side; JazzTube has verified YouTube links for **329 of 456 solos** (988 videos, [Balke et al. 2018](https://www.frontiersin.org/journals/digital-humanities/articles/10.3389/fdigh.2018.00001/full)), ~~though the link table itself was never published~~ **— the table IS published; see the 2026-09-22 finding below**. | one email | Uncertain reply; sharing audio would need a research agreement. The link table alone would remove the search step from route C. |
| **B. Buy digital copies via MusicBrainz ids** | 302 tracks with ids; the remaining 42 by album/catalogue number by hand. Most tracks are on multiple releases; any release of the same *recording* is fine (alignment absorbs remaster offsets). | ~184 albums; many are box sets / compilations. Rough order: US$1–2k if bought as albums, less where single tracks are sold. Slow (manual). | Cleanest legally; this is what the project itself suggests. Some releases are out of print digitally. |
| **C. YouTube via `yt-dlp`** using the Balke et al. two-stage method: text search (soloist+title, artist+title, top 20 each) → candidates → **audio-based verification against the transcription** (below) | 2018: 329/456 solos; availability churns both ways. | Compute only. | Opus ~128 kbps, occasional pitch/speed-shifted uploads (the verifier catches these), live versions and wrong takes (also caught). **Downloading is against YouTube's terms of service**; whether that is acceptable for this research use is Nithya's call, and the plan does not presume it. |
| D. Record from a streaming service | full | subscription | Not proposed. |

**Recommendation:** send the route-A email on day one (it costs nothing and may short-circuit
everything), build the manifest and the verifier regardless (they are needed under every
route), and decide between B and C once the reply is in. Under B, the verifier still runs —
purchases are not immune to wrong takes.

## 2026-09-22 finding: the JazzTube link table is published, with offsets

The claim above that "the link table itself was never published" is **wrong**, and the
consequences are large enough to restructure this experiment.

`jazztube.hfm-weimar.de` is offline, but the site is mirrored at
`mir.audiolabs.uni-erlangen.de/jazztube/`, whose Downloads page serves
**`static/csv_youtube.csv`** — all 988 videos from Balke et al. (2018). Mirrored to
`/data/hai-res/shared/snnithya/sao-3/data/wjd/jazztube/` (see its README) because the original host has already died once.

What the table gives us per row:

| Column | Meaning |
|---|---|
| `melid`, `youtube_id` | the link, 329/456 solos over 778 unique videos (~3 per solo) |
| `solo_start_sec`, `solo_end_sec` | solo bounds **in the YouTube video's timeline** |
| `mf_min`, `mf_median` | chroma matching-function distance; **every row already passes the paper's 0.1 threshold** (median `mf_min` 0.044) |
| `wp_start_idx`, `wp_end_idx` | DTW warping path endpoints |

**This pre-solves steps 1–3 of the verification plan.** The offset we intended to recover by
cross-correlation is `solo_start_sec`, already computed by the authors against the real audio
(a stronger reference than our synthesized piano roll). Verified on melid 10:

    video_time = wjd_excerpt_time + solo_start_sec
    check: solo_end_sec - solo_start_sec = 122.8  vs  WJD solo_duration = 122.876

So the verifier's job shrinks from *search and align* to *confirm the published offset still
holds for the file we actually downloaded* — a single cheap beat-strength check (step 3), since
the one thing the 2018 table cannot vouch for is that today's upload under a given id is the
same edit it was then.

### Coverage, after link rot

Liveness of all 778 ids via the YouTube oEmbed endpoint, 2026-09-22:
562 alive (200), 130 deleted (404), 86 blocked (401/403 — spot-checked, genuinely not
playable, not merely embed-disabled).

| Set | 2018 table | still alive |
|---|---|---|
| Solos | 329 / 456 | **297 / 456** |
| Tracks | 257 / 344 | **231 / 344** |
| JSD tracks | 250 / 340 | **224 / 340** |
| **JSD tracks with drums** | 238 / 316 | **214 / 316** |

Redundancy helps: 72 % of individual videos survive, but ~3 videos per solo lifts per-solo
survival to 90 %.

**Against the success criterion:** 214 drum tracks is well past the "minimum useful set" of
150, but short of the ≥ 280 target. Route C alone therefore caps 3.1 at roughly two-thirds of
the goal; closing the last ~100 tracks still needs route A or B. That argues for sending the
route-A email regardless — and now it has a second purpose, since the Jazzomat group may hold
audio for the 127 solos the link table never covered.

### Revised recommendation

1. Mirror the link table — **done**, `/data/hai-res/shared/snnithya/sao-3/data/wjd/jazztube/`.
2. Build `build_manifest.py`, joining WJD ⨝ JSD ⨝ the link table, with liveness per video.
3. Fetch with `yt-dlp` (not `pytube`, which is unmaintained and breaks on cipher changes),
   best available audio, preferring the lowest-`mf_min` live video per solo.
4. Verify only the beat check + duration, not the full chroma search.
5. Send the route-A email in parallel, scoped to the gap.

Caveat unchanged: downloading is against YouTube's ToS; the table's existence makes route C
cheaper, not more permitted.

## Implementation (2026-09-22): decision taken — route C, drum tracks only

Nithya's decision: download from YouTube, restrict to the drum tracks the link table covers,
best available audio. Implemented as:

- `scripts/wjd/fetch_youtube.py` — standalone (stdlib + `yt-dlp` + `ffmpeg`). Selects tracks
  with a drummer in the lineup, a JSD entry and ≥ 1 live linked video: **220 tracks** (skipped:
  100 with no live video, 20 drumless, 4 without a JSD entry). One video per track, ranked by
  solos covered then `mf_min`; up to 4 candidates are probed for duration and the first within
  15 s of the JSD duration is taken (else best-ranked, flagged `check`). Downloads
  `bestaudio` (typically Opus ~135 kbps in webm), keeps the stream in `raw/`, transcodes to
  44.1 kHz stereo 16-bit FLAC in `aligned/`, and writes `alignment.json` with
  `offset_sec = solostart_sec − solo_start_sec` (median over the track's linked solos),
  the duration delta, and `status: pass|check`. Resumable; `_failed.json` + `manifest.csv`.
- `sbatch/03_1_fetch_wjd_youtube.sbatch` — CPU job on `tig-cpu` (`--qos=tig-main`; the
  `hai-res-*` QoS are rejected there). Compute nodes reach YouTube (checked on `groenig-5`).
  Tools live in this checkout's `.venv`: `uv pip install --python .venv/bin/python yt-dlp
  static-ffmpeg`, then `.venv/bin/static_ffmpeg_paths` downloads ffmpeg/ffprobe 8.0 into
  site-packages and `.venv/bin/ffmpeg` / `ffprobe` are symlinks to them. Not in
  `pyproject.toml`: re-locking currently fails because the prebuilt flash-attn wheel is
  cp310-only while `requires-python = ">=3.10"` has no upper bound — pin it to `<3.11` before
  the next `uv add`. `FETCH_LIMIT=N` / `FETCH_ARGS=--dry_run` for smoke tests;
  `YTDLP_COOKIES=<netscape cookies file>` if YouTube starts demanding a sign-in.

Smoke test on `ArtBlakey_DownUnder_Orig`: 328.1 s file vs 330.5 s JSD, three solos agree on
the offset to within 0.16 s, status `pass`. Dry run over the first three selected tracks
already hit one video that went "unavailable" between the liveness check and now — expect
the final count to land a little under 220.

Still to write: the beat check (step 3 below) as `scripts/wjd/verify_alignment.py`, run over
`aligned/` after the fetch; `alignment.json` reserves `checks.beat_check` for it.

## Verification + alignment (route-independent)

For each candidate file, in order, writing everything to `audio/aligned/<Track>.alignment.json`:

1. **Duration check** against JSD `track_durations.csv`. |Δ| ≤ 3 s pass; ≤ 15 s flag; longer
   means a different edit (fade, applause, hidden track) — not fatal, step 2 decides.
2. **Transcription-to-audio alignment.** Render each solo's `melody` rows (onset, duration,
   MIDI pitch) as a piano roll → chroma at 10 Hz. Compute chroma from the candidate over
   `[solostart_sec − 60 s, solostart_sec + solo_len + 60 s]` (`librosa` CQT chroma; add
   `librosa` to the env). Cross-correlate over offsets to get **offset** and a normalized
   score; repeat over time-stretch factors 0.94–1.06 in 0.5 % steps and ±1 semitone chroma
   rolls to catch speed- and pitch-shifted copies. A track passes if every one of its solos
   agrees on the offset within 0.1 s. Balke et al. used a chroma-distance threshold of 0.1
   against the real audio; ours is against a *synthesized* reference, so the threshold is
   calibrated on the score distribution (expected bimodal) plus listening on ~20 borderline
   cases.
3. **Beat check** (cheap second opinion). Mean onset-strength at the aligned WJD beat times vs.
   at random times; a ratio near 1 means the offset is wrong by half a beat or the take is
   wrong even though the chroma matched (same tune, different solo, similar changes).
4. **Tuning check** (optional). [jazzomat/raw_data](https://github.com/jazzomat/raw_data) has
   the estimated tuning frequency per solo for 299 solos (WJD v1.2). More than ~20 cents
   away on the candidate = pitch-shifted upload; flag.

Output convention: audio stored **untouched** as 44.1 kHz stereo FLAC (mono duplicated), and
`alignment.json` = `{source, offset_sec, stretch, score, duration_delta, per_solo: [...],
status: pass|check|fail}` where `track_time = file_time + offset_sec`. Phase 1 excludes any
file with `stretch ≠ 1 ± 1 %` rather than resampling it; revisit if that costs many tracks.

## Success criterion

≥ 280 of the 324 drum tracks with status `pass` (~28 h). Minimum useful set for 3.3: 150
tracks. Report coverage by decade, style and soloist instrument — the losses will not be
uniform (1920s–40s material is the hardest to find and to verify).

## Deliverables

- `scripts/wjd/build_manifest.py`, `scripts/wjd/verify_alignment.py`
  (+ `scripts/wjd/fetch_youtube.py` only if route C is chosen)
- `/data/hai-res/shared/snnithya/sao-3/data/wjd/manifest.csv`, `/data/hai-res/shared/snnithya/sao-3/data/wjd/audio/raw/`, `/data/hai-res/shared/snnithya/sao-3/data/wjd/audio/aligned/` (~10–12 GB FLAC)
- a results table here: candidates per track, pass/check/fail, coverage by decade
- `tests/test_wjd_alignment.py` on a synthetic case (render a transcription to audio, offset it,
  recover the offset)
