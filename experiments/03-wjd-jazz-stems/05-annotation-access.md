# 3.5 — Annotation access: chords, form, chorus, beats in audio time

**Status:** planned, low priority, no GPU
**Depends on:** 3.1 for the *audio-time* half (needs `alignment.json`); the database half depends on nothing and can be written today

## Goal

One module that answers, for any WJD track and any time or latent frame, **what chord, form
part, chorus and beat position is playing** — in the time base of *our* audio file — so a
later experiment can condition on harmony or structure without re-deriving the database
conventions. Nothing in 03 trains on these; this is the part of the plan that keeps the
option open, as asked.

## What the database has, and where

| Annotation | Best source | Notes |
|---|---|---|
| Beat grid | `beats.onset`, `bar`, `beat`, `signature` | tapped to the audio, per solo; 132 k beats |
| **Chord per beat** | `beats.chord` (non-empty at chord *changes*; forward-fill) | 419 distinct raw labels: `Bb6`, `G-7`, `C7alt`, `Am7b5`, `D79b`, `NC` … Jazzomat's chord syntax is documented on the [melospy annotations page](https://jazzomat.hfm-weimar.de/melospy/annotations.html) |
| Form part per beat | `beats.form` (`A1`, `B1`, `A2`, `I1` intro, `*B1` …; forward-fill) | plus the tune's template in `composition_info.form` (`A8A8B8A8`) |
| Chorus | `beats.chorus_id` (0 = pickup / intro) | |
| Bass note per beat | `beats.bass_pitch` (MIDI) | also the 3.2 bass proxy metric |
| Lead sheet | `solo_info.chord_changes` (text, bar-by-bar with form labels) | |
| Key, tempo, metre, style, feel | `solo_info` | per solo |
| Phrases / ideas | `sections` (type PHRASE / IDEA), `start`/`end` as **melody event indices** | convert via `melody.onset`; not needed for conditioning |
| Whole-track structure | JSD segments (`solo_01`, `theme_01`, instrumentation) | the only annotation that covers time *outside* the transcribed solos |

The `sections` table also carries CHORD / FORM / CHORUS spans, but indexed by melody events;
`beats` has the same information indexed by time and is the one to use.

## Time conversion — two hops

```
solo-excerpt time  ──(+ transcription_info.solostart_sec)──►  track time (annotators' edit)
track time         ──(− alignment.json.offset_sec, ÷ stretch)──►  our file's time
our file's time    ──(× 44100 / 4096)──►  latent frame
```

Hop 1 is verified (419/444 solos land in a JSD solo segment with it, README). Hop 2 is what
3.1 produces. The module should never expose solo-excerpt time to callers.

## Coverage caveat

Chords, form, chorus and beats exist **only inside the 456 transcribed solos — 12.8 h of the
33.4 h of audio.** Themes, intros and other players' un-transcribed solos have JSD segment
labels but no chords or beats from the WJD. Extending coverage (align the lead-sheet
`chord_changes` + form template to a beat tracker across the whole track) is a separate,
non-trivial project; note it, do not do it here.

## Module

`stable_audio_3/data/wjd_annotations.py` — stdlib `sqlite3` + numpy, no torch:

```python
db = WJD("…/wjd/wjazzd.db", alignments_dir="…/wjd/audio/aligned")   # alignments optional
db.tracks()                                    # -> [TrackInfo]  (filename_track, lineup, mbzid, decade, has_drums, …)
db.solos("ArtPepper_Anthropology_Orig")        # -> [Solo(melid, instrument, start_s, end_s, style, feel, tempo, key)]  in track time
db.beats(melid)                                # -> array of Beat(t, bar, beat, chord, form, chorus, bass_pitch)  chords/forms forward-filled
db.frame_labels(track, n_frames, frame_rate=44100/4096, file_time=True)
   # -> dict of per-frame arrays: chord (str), chord_class (int), form (str), chorus (int),
   #    beat_phase (float in [0,1)), is_downbeat (bool), in_solo (bool), soloist (str)   — "NC"/-1 outside solos
db.chord_vocab(reduction="root_quality")       # -> label list
```

Chord parsing: a small grammar for the Jazzomat syntax (root, `-`/`m` minor, `j7`/`maj`,
`7`, `o`/`dim`, `m7b5`/`ø`, `sus`, `alt`, `+`, extensions, slash bass, `NC`) → structured
chord, plus a reduction to **root × {maj, min, dom, dim, hdim, sus} + NC = 73 classes** for
any conditioning that wants a categorical, keeping the raw string alongside. Unit-test the
parser on all 419 observed labels (they are all in the DB; the test can enumerate them).

Also an exporter, `scripts/wjd/export_annotations.py`, that writes
`wjd-stem-mirror/<split>/annotations/<Track>__<seg>.json` (beats + per-beat chord/form/chorus in
file time, clipped to the segment) so the training-time metadata module reads a small JSON per
item and never opens SQLite inside a dataloader worker.

## Tests (no data beyond `wjazzd.db` + JSD needed)

- beat grids monotone, 456/456 solos
- forward-filled chord non-empty for every beat after the first chord symbol
- the 419/444 JSD containment check from the README as a regression test on hop 1
- chord parser round-trips all 419 labels; reduction hits one of 73 classes

## Deliverables

- `stable_audio_3/data/wjd_annotations.py`, `tests/test_wjd_annotations.py`
- `scripts/wjd/export_annotations.py`
- a short `docs/data/wjd.md` inventory (mirroring `docs/data/slakh-streamgen.md`) once 3.1–3.3
  have put files on disk
