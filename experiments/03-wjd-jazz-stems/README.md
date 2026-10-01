# Experiment 03 — Weimar Jazz Database: drum-conditioned stem generation

**Status:** planned — dataset surveyed, no audio acquired, nothing trained
**Started:** 2026-09-22
**Branch:** `tap-dance-expt`
**Data root:** `/data/hai-res/shared/snnithya/sao-3/data/wjd/` (database + JSD annotations staged; see its README)

---

## Hypothesis

A Stable Audio 3 DiT finetuned on **source-separated jazz recordings from the Weimar Jazz
Database (WJD)**, conditioned on the **drum part** through `modular_local_cond` and on a **text
prompt naming the target instrument**, will generate instrument-specific stems (tenor sax, bass,
piano, …) that lock to the drummer's time and feel, and will switch instrument on the prompt
alone with the drum condition held fixed.

This inverts experiment 01: there the drums were the *target* and the accompaniment the
*condition*. Here the drums are the *condition* and each non-drum stem is a *target*, one item
per stem, disambiguated by the prompt. Three sub-claims:

1. **The drum latent is a usable condition.** With the drum stem's VAE latent as the control
   (the existing `controls` path, no new code), generated stems are rhythmically aligned to the
   drums — measurable as beat/onset agreement, audible as "playing with the drummer".
2. **The prompt selects the stem.** Same drum condition, prompt `bass` vs `tenor saxophone` vs
   `piano` gives stems that an instrument classifier separates, and that are not each other.
3. **A symbolic drum representation is enough.** Replacing the drum *latent* with a drum
   *onset* representation (kick/snare/hat activations at the latent frame rate) keeps 1 and 2.
   This is the representation the interactive system will actually have — a tapped or played
   rhythm, not a mixed drum recording — so it is the one that matters, and it is deliberately
   phase 2 because the latent path exists today and isolates the data questions from the
   representation question.

The WJD is chosen over Slakh for this because it carries **musical annotations the model can
later be conditioned on** — chords per beat, form parts (A1/B1…), chorus boundaries, beat grid,
key, style, tempo — on real recordings. Nothing in this experiment trains on them; 3.5 makes
them accessible so a later experiment can.

## Background — what the WJD is, and is not

**What you download** (`wjazzd.db`, SQLite, 42 MB, ODbL, v2.1 / db-version 2.2, dated 2018-01-07,
[download page](https://jazzomat.hfm-weimar.de/download/download.html),
[schema](https://jazzomat.hfm-weimar.de/dbformat/dbformat.html)):

| Table | Rows | What it holds |
|---|---|---|
| `solo_info` | 456 | one row per transcribed solo: `performer`, `title`, `instrument`, `style`, `rhythmfeel`, `avgtempo`, `key`, `signature`, `chord_changes` (lead-sheet string), `chorus_count`, FKs to track/record/composition |
| `transcription_info` | 456 | `solostart_sec` — where the solo excerpt starts in the full track; `solotime` "mm:ss-mm:ss" |
| `track_info` | 344 | `filename_track` (canonical name, e.g. `ArtPepper_Anthropology_Orig`), `lineup` ("Art Pepper (as, cl); Charlie Haden (b); Billy Higgins (dr)"), **`mbzid` = MusicBrainz recording id (302/344 filled)**, `recordingdate` |
| `record_info` | 184 | album, label, catalogue no.; `mbzid` empty for all rows |
| `composition_info` | 302 | tune, composer, `form` template (`A8A8B8A8` ×70, `A12` blues ×60, …), `tonalitytype` |
| `melody` | 200,809 | the transcribed notes: `onset`, `duration` (s), `pitch` (MIDI), bar/beat/tatum, loudness, f0 modulation |
| `beats` | 132,329 | **the beat grid with `chord`, `form`, `bass_pitch`, `chorus_id`, `signature` per beat** |
| `sections` | 58,560 | CHORD / FORM / CHORUS / PHRASE / IDEA spans as *melody event index* ranges |

**Time reference.** `melody.onset` and `beats.onset` are in seconds **from the start of the
solo excerpt**, not the track. Track time = `transcription_info.solostart_sec + onset`. Checked
against the JSD segments below: 419 of 444 solos land inside a JSD `solo_*` segment under this
rule, 390 of them with the same soloist instrument. (The remainder are near boundaries or in
tracks with trading soloists; none suggests a different convention.)

**What you do not download: the audio.** All 344 tracks are commercial recordings and the
project does not distribute them, only the identifiers to buy them. This is the whole
difficulty of the experiment and is [3.1](01-audio-acquisition.md).

**Coverage** (from the database, 2026-09-22):

| | |
|---|---|
| Transcribed solos | 456, total **12.8 h**, median 88 s, mean 101 s (range 16 s – 13.6 min) |
| Solo instruments | ts 157 · tp 102 · as 80 · tb 26 · ss 23 · cor 15 · cl 15 · vib 12 · bs 11 · p 6 · g 6 · bcl 2 |
| Style | postbop 147 · hardbop 76 · swing 66 · bebop 56 · cool 54 · traditional 32 · fusion 20 · free 5 |
| Feel | swing 361 · two-beat 32 · latin 27 · funk 20 · ballad 10 |
| Metre | 4/4 in 435 of 456 |
| Tracks with drums in the lineup | **324 of 344** (the 20 without are duets and the 1920s Hot Five sides with banjo) |
| Tracks with a MusicBrainz recording id | 302 of 344 (282 of those have drums) |
| Recording decades | 1920s 16 · 1930s 15 · 1940s 53 · 1950s 89 · 1960s 51 · 1970s 8 · 1980s 23 · 1990s 39, ~40 unparseable |

The instrument imbalance matters for sub-claim 2: **saxophone and trumpet solos are 79 % of the
transcribed material; piano and guitar are 12 solos between them.** The JSD (next) is what
rescues piano and bass as targets.

### The Jazz Structure Dataset (JSD) — same recordings, whole-track labels

[Balke et al., TISMIR 2022](https://transactions.ismir.net/articles/10.5334/tismir.131),
[github.com/stefan-balke/jsd](https://github.com/stefan-balke/jsd), CC BY 4.0. Staged at
`/data/hai-res/shared/snnithya/sao-3/data/wjd/jsd/`. Structure annotations for **340 of the 344 WJD tracks** (3 Metheny/Brecker
duplicates dropped, 5 hyphenated names spelled differently), covering the **entire recording**,
not just the transcribed solo:

```
segment_start;segment_end;label;instrument
14.889795918;60.669387755;theme_01_01;cl,b,dr
60.669387755;96.848979591;solo_01_01;s_cl,b_b,b_dr      <- s_ soloist, b_ backing
201.404081632;235.036734693;solo_02_01;s_b,b_dr         <- bass solo, drums only behind it
266.775510204;301.779591836;solo_03_01;s_cl,s_dr,b_b    <- trading with the drummer
```

Segments are chorus-level (`solo_01_03` = third chorus of the first solo). Totals over the 340
tracks: **33.4 h of audio**, 22.1 h of it `solo`, 9.1 h `theme`, 1.6 h intro/outro.
Solo minutes by soloist across the *whole* corpus: ts 406 · **p 295** · tp 226 · as 141 · dr 88 ·
**b 87** · g 73 · ss 64 · tb 38 · bcl 25. Backing minutes: dr 1751 · b 1685 · p 1254.

So the JSD gives every segment of every track a **who-is-playing-what label**, which is what
turns a separated `other` stem into "tenor saxophone solo" for 60–97 s and "clarinet + ensemble
theme" for the head — and gives ~5 h of piano solo and ~1.5 h of bass solo that the WJD alone
does not label. `track_durations.csv` (full-track length per file) doubles as a check that an
acquired audio file is the same edit the annotators used.

## Method (overview — details in the sub-experiment files)

```
wjazzd.db + JSD ──► acquisition list (MBID, artist/title, duration)      3.1
                          │
            acquire audio ▼  (route is Nithya's call — see 3.1)
                          │
        verify + align each file against the transcription ──► alignment.json   3.1
                          │
        separate: demucs htdemucs_ft (drums/bass/other/vocals) [+ 6s piano/guitar]   3.2
                          │
        cut per JSD segment ──► wjd-stem-mirror/<split>/{drums,targets}/<Track>__<seg>/   3.3
                          │
        custom_md_wjd.py: prompt from segment instrumentation + solo_info; drums via __audio__
                          │
        pre_encode_dataset.py --controls drums_audio ──► latents + drum-latent sidecar   3.3
                          │
        finetune small-music + modular_local_cond("drums_latent")  ──► claims 1, 2      3.3
                          │
        swap control for onset activations                            ──► claim 3      3.4
```

**Reuse.** The whole conditioning stack from experiment 01 carries over unchanged: the
`__audio__` hook, fused `_controls.npy` sidecars, `modular_local_cond` with zero-init MLPs,
`CAUSAL_MASK` / `future_visibility`, `check_streamgen_alignment.py`, the silence gates. The only
new code in phase 1 is a metadata module and a segment-cutting script. The model config needs a
control id rename (`streamgen_latent` → `drums_latent`, 256) and nothing else.

**Prompts.** `small-music` already has a T5Gemma text conditioner with `cfg_dropout_prob 0.1`, so
the prompt is CFG-guidable out of the box. Phase 1 prompts are short and templated from the
labels — `"tenor saxophone solo"`, `"upright bass, walking"`, `"piano, comping"` — with style /
feel / tempo appended as an optional richer form (`"tenor saxophone solo, hard bop, swing, 220
bpm"`). Which template helps is a sub-question of 3.3, not a decision to make now.

## Sub-experiments

| # | Name | Question | Status |
|---|------|----------|--------|
| 3.1 | [Audio acquisition + alignment](01-audio-acquisition.md) | How do we get the 340 recordings, and how do we know each file is the right take at the right offset? | **planned — needs a decision on the acquisition route** |
| 3.2 | [Source separation](02-source-separation.md) | Which separator gives usable drums (condition) and bass/piano/horn (targets) on 1940s–90s jazz, and how bad is the `other` stem as a horn stem? | planned |
| 3.3 | [Dataset build + wiring + first finetune](03-dataset-and-wiring.md) | Segment-level mirror, prompts, pre-encode, and the drum-*latent* finetune that tests claims 1 and 2 | planned |
| 3.4 | [Drum onset representation](04-onset-representation.md) | Does a symbolic drum control (onset/activation channels) work as well as the drum latent? | planned, after 3.3 |
| 3.5 | [Annotation access](05-annotation-access.md) | One module that gives chords / form / chorus / beats in *audio* time for any track, so a future experiment can condition on them | planned, low priority, no GPU |

## Results

None yet.

## Notes

- **Licensing is split.** The database is ODbL and the JSD is CC BY 4.0 — free to use with
  attribution. The audio is copyrighted commercial recordings; the project's own position is
  that researchers buy them (via the MusicBrainz ids). Which acquisition route is acceptable
  is a call for Nithya, not for this plan — 3.1 lays out the options with their costs and
  caveats and everything downstream is route-agnostic.
- **Old recordings will separate badly.** 84 tracks are pre-1950 mono, many transferred from
  78s. Every separator we can run was trained on modern stereo mixes. 3.2 should measure
  quality *by decade* and expect to drop or down-weight the oldest material; sub-claims 1–2
  do not need it.
- **Jazz has alternate takes.** `ArtPepper_Stardust-1` is a take number. The same tune on the
  same album can exist in two takes with different solos; a duration match is not enough to
  confirm a file, which is why 3.1 aligns the *transcription* to the audio rather than
  trusting metadata.
- **Solo excerpt audio names exist in the DB** (`transcription_info.filename_solo`,
  `filename_sv` = Sonic Visualiser project). Those files were the annotators' working copies and
  are not distributed either, but the Jazzomat group has shared audio with collaborating
  researchers under agreement before. Worth one email before buying anything (3.1).
- Why not the Jazz Trio Database (Cheston et al. 2024, ~45 h piano trios, Zenodo with
  approval, already separated, `mirdata` loader)? It is a real alternative for the *drums →
  bass/piano* half of this experiment and has no horn material and no chord/form labels. Kept
  as a fallback if 3.1 stalls; noted here so it is not rediscovered.
- Nothing here depends on the Slakh streamgen datasets or their re-encode status.
