# 3.2 — Source separation

**Status:** **running** (2026-09-28). 3.1 has 190 aligned tracks on disk (166 `pass`, 24 `check`,
18.2 h, 30 failed). Nithya's decision 2026-09-28: **skip the model comparison, run
BS-Roformer-SW only** — the pilot / Demucs / SCNet rows below are kept for reference, not
scheduled. `sep` env built, smoke test passed, full pass over the 166 `pass` tracks submitted as
Slurm job 2397035 (see Results).
**Depends on:** 3.1 (`audio/aligned/<Track>.flac` + per-track `aligned/<Track>.alignment.json`;
only `status: pass` tracks are separated for now).

## Question

Which separator gives (a) a **drum stem clean enough to be the condition** and (b) **bass /
piano / horn stems clean enough to be targets**, on 1920s–1990s jazz that none of these
models were trained on? And how bad is the `other` stem as a "horn solo" stem?

Separation quality is the ceiling on this whole experiment in the same way the autoencoder was
in 02: the DiT learns to generate what the separator produced, artifacts included.

## Candidates

What is actually available (checked 2026-09-24 against `python-audio-separator` 0.47.0's
`models.json` and ZFTurbo's MSST `docs/pretrained_models.md`). Almost every public RoFormer
checkpoint is a vocals/instrumental 2-stem model; the multistem ones are few:

| Model | Stems | Train data | Reported SDR | Notes |
|---|---|---|---|---|
| **BS-Roformer-SW** (jarredou) — `BS-Roformer-SW.ckpt` in audio-separator | bass, drums, other, vocals, **guitar, piano** | community (not MUSDB-only) | not listed in `models.json`; take from the release notes when the env is up | dim 256 / depth 12, 44.1 kHz stereo, 13.3 s chunks. **Primary.** The only 6-stem RoFormer; piano and guitar come from the same model as drums, so no cross-model stem mixing. |
| BS-RoFormer 4-stem (ZFTurbo, `model_bs_roformer_ep_17_sdr_9.6568.ckpt`) | bass, drums, other, vocals | MUSDB18HQ only (100 songs) | MUSDB test: drums 11.61, bass 8.48, other 7.44; Multisong drums 11.29 | Second RoFormer opinion on drums. Small training set → generalisation to old mono jazz is the open question. Runs via MSST `inference.py` or audio-separator with the config. |
| SCNet-XL IHF (ZFTurbo, `model_scnet_ep_36_sdr_10.0891.ckpt`) | bass, drums, other, vocals | MUSDB18HQ only | MUSDB test: drums **11.81**, bass 9.23, other 7.88 (best 4-stem numbers in the list) | Not a RoFormer, but the strongest published drums stem; cheap to add to the pilot since it runs in the same MSST env. Optional. |
| `htdemucs_ft` (Demucs v4, fine-tuned) | drums, bass, other, vocals | MUSDB18HQ + 800 extra songs | Multisong drums 11.13 (FT drums) | **Baseline.** Already installed (`sat-v2`). |
| `htdemucs_6s` | + guitar, piano | as above | piano stem documented as weak | Baseline for piano/guitar only. |
| DrumSep (jarredou, MDX23C `drumsep_5stems_mdx23c_jarredou.ckpt`) | kick, snare, toms, hh, cymbals **from a drum stem** | DrumSep | kick 16.7, snare 11.5, toms 12.3, hh 4.0, cymbals 6.4 | Not for 3.2. Note for **3.4**: a per-instrument drum decomposition is a candidate onset representation. |
| Horn-specific model | — | — | — | None public. Horns land in `other` (and partly in `vocals` — check). |

Why RoFormer first: on MUSDB the band-split transformers lead Demucs by 0.5–1 dB on drums and
more on `other`, and the SW checkpoint is the only model that gives piano/guitar without falling
back to `htdemucs_6s`. Why keep Demucs: it was trained on ~9× more songs than the MUSDB-only
RoFormer/SCNet checkpoints, and old jazz is far out of distribution for all of them — the SDR
table above says nothing about a 1938 mono Basie side. The pilot decides.

## Infrastructure (as built, 2026-09-28)

- **Env.** `sep` conda env at `/data/hai-res/snnithya/miniconda3/envs/sep` (python 3.11,
  `audio-separator==0.47.0`, torch 2.14.0+cu130, onnxruntime-gpu 1.30), made exactly as planned:
  ```bash
  export PIP_CACHE_DIR=/data/hai-res/snnithya/.cache/pip_cache   # AFS home is at 96 % quota
  conda create -n sep python=3.11 -y
  /data/hai-res/snnithya/miniconda3/envs/sep/bin/pip install "audio-separator[gpu]==0.47.0"
  ```
  Dedicated env, **not** the project `uv` env (torch pins would fight). Demucs would run from
  `sat-v2` (demucs 4.0.1) if the baseline is ever wanted; MSST is not installed.
- **Model files** in `/data/hai-res/shared/snnithya/sao-3/models/sep/`: `BS-Roformer-SW.ckpt`
  (699 MB, sha256 `24e7d35e…5916e`) + `BS-Roformer-SW.yaml`, both from the
  `python-audio-separator` GitHub release `model-configs` (the usual UVR `model_repo` release
  404s for this model). Pre-downloaded with curl; audio-separator only re-fetches if missing,
  but it does pull a 3.5 kB `models.json` on every model load, so compute nodes need outbound
  internet (they have it). The yaml's inference block is what runs: `dim_t 801`, chunk
  588 800 samples = 13.35 s, `num_overlap 2`, `batch_size 1`; stems
  `['bass','drums','other','vocals','guitar','piano']`.
- **Runner:** `scripts/wjd/separate.py` — standalone, no `stable_audio_3` import, runs inside
  the `sep` env and drives audio-separator through its **Python API** (not subprocess: that is
  what lets it set the output names to `<stem>.flac` directly, no renaming). Driven by
  `audio/manifest.csv` (`result == done` rows only); `--status pass` (default), `--tracks …`,
  `--limit N`, `--force`; `--overlap` / `--batch_size` default to the yaml. Resumable: a track
  with `done.json` is skipped; a partial dir is wiped and redone; failures land in
  `failed.json` and the batch continues. **Peak normalisation is turned off**
  (`--normalization 1.0`; audio-separator's default rescales *every* stem to 0.9 peak, which
  would wreck `Σ stems ≈ mix`). A stem is still scaled down if it would clip; `done.json`
  flags that per stem as `peak_limited` (with `peak`, `rms_dbfs`) so the level change is
  visible. Also recorded: mix info, checkpoint sha256, package versions, effective
  overlap/batch/`dim_t`, wall time, realtime factor, and `residual_db` =
  20·log10(rms(mix − Σ stems)/rms(mix)).
- **Output layout:** `/data/hai-res/shared/snnithya/sao-3/data/wjd/stems/<model>/<Track>/{drums,bass,other,vocals,piano,guitar}.flac`
  + `done.json`, 44.1 kHz stereo PCM-16 FLAC (soundfile, input subtype). **Time base is
  unchanged from the aligned FLAC**: `separate.py` asserts frames == mix frames ± 1 and same
  sample rate, so the per-track `aligned/<Track>.alignment.json` (`track_time = file_time +
  offset_sec`, WJD onsets at `file_time = jazztube_solo_start_sec + onset_in_excerpt`) applies
  to stems as is.
- **Compute:** `sbatch/03_2_separate_wjd.sbatch` — `hai-res-l40s`, 1 L40S, 8 CPU, 48 GB, 12 h,
  `SEP_MODEL=…` / `SEP_ARGS="…"` knobs. A QoS is mandatory (the account has no default; without
  one `sbatch` says "Invalid qos specification"). Measured on an L40S: **6.5–11× real time**
  for BS-Roformer-SW at overlap 2, 50 s for a 5.5 min track → the 166 `pass` tracks (~16 h
  audio) are **~2 GPU-hours**, ~25 GB FLAC.
- **Preemptible + restart-safe** (Nithya's ask, 2026-09-28). The cluster preempts by QoS
  (`PreemptType=preempt/qos`, `PreemptMode=REQUEUE`, `GraceTime=0`, `KillWait=10 s`): a
  preempted job gets SIGTERM and SIGKILL 10 s later, so there is never time to finish a track.
  The sbatch runs `--qos=hai-res-low` (priority 20, bumped by any `hai-res-main/-extended/-debug`
  job; also queued *behind* them, so it only starts on idle GPUs — use `--qos=hai-res-main` for
  a run that has to happen), `--requeue` (Slurm's default is `Requeue=0`), `--open-mode=append`
  (one log per job id across restarts, each start prints `restart=$SLURM_RESTART_COUNT`), and
  forwards SIGTERM to the runner. In the runner: `done.json` appears only via atomic rename; a
  stem dir without it is partial output, wiped and redone; each worker takes an
  `_inprogress.json` lease (host, pid, Slurm job id) before touching a track and defers tracks
  whose lease is fresher than `--lease_minutes` (30), retrying them once at the end; a lease
  with the worker's own Slurm job id **and a lower `SLURM_RESTART_COUNT`** is its dead
  predecessor's and is reclaimed at once, so a requeued job resumes its own track, while a
  lease from another process in the same allocation (same job id, same count) is a live
  sibling and is respected — the first version reclaimed on job id alone and two hand-started
  workers in one `srun` shell promptly clobbered each other; SIGTERM raises through the
  current track, drops the lease and exits 143. Tested 2026-09-28 on
  `ZootSims_Undecided_Orig` (kill mid-separation → no `done.json`, no lease; foreign fresh
  lease → deferred, untouched; predecessor lease → reclaimed and redone) plus a unit check of
  the ownership rules. This is also what makes it safe to run a second worker by hand
  (`--reverse` so the two meet in the middle). When stopping a hand-started worker, `kill` the
  python pid, not the wrapping shell — the shell dies and the worker keeps going.
  One NFS lesson from the first restart: `shutil.rmtree` on a partial dir failed with
  `ENOTEMPTY` on the final `rmdir` (attribute-cache / `.nfs*` race on `/data/hai-res/shared`),
  so the runner now empties a partial dir file by file and never removes the directory.

## Stem → role mapping

| Separator stem | Role | Label source |
|---|---|---|
| `drums` | **condition** (3.3 latent, 3.4 onsets) | — |
| `bass` | target `"upright bass"` | JSD `b_b` / `s_b` per segment |
| `piano`, `guitar` (SW; `htdemucs_6s` as fallback) | targets | JSD `p`, `g` per segment |
| `other` | target = **the soloist** in `solo_*` segments where JSD names a horn / vibes / clarinet soloist; the front line in `theme` segments | JSD `s_ts`, `s_tp`, … + `solo_info.instrument` |
| `vocals` | discard, **after** checking horn leakage (7 tracks have voice) | — |

If saxophone bleeds systematically into `vocals` — plausible, it is the most voice-like source —
the horn target becomes `other + vocals`. This is one of the first things to listen for, and it
may differ between RoFormer and Demucs (vocal models are what the RoFormer community optimises
for, so its `vocals` stem may be *more* eager to grab a tenor).

## Evaluation — no ground-truth stems, but the DB gives three free proxies

All proxies are computed inside the WJD solo windows, placed in file time via
`alignment.json`: solo excerpt starts at `jazztube_solo_start_sec`; melody / beat onsets are
relative to that excerpt start (the excerpt has a 2–6 s lead-in before the first solo note, see
3.1 notes 2026-09-24). Only `status: pass` tracks are scored.

| Stem | Proxy metric | Why it is available |
|---|---|---|
| horn (`other`) | pitch-track the stem inside each solo window (`pyin` / CREPE) and score frame-wise agreement with the `melody` table's pitches: **soloist recall**. Also: energy of `other` inside `s_dr`-only segments (drum solos with nobody else playing) = **leakage floor**. | WJD is a *transcription* dataset — the solo's pitches are known at 10 ms resolution. |
| bass | pitch-track the bass stem and compare to `beats.bass_pitch` (MIDI bass note annotated **per beat**). | 132 k beats carry a bass pitch. |
| drums | fraction of aligned `beats.onset` times with a detected drum onset within ±50 ms; and the converse (onsets far from any beat / subdivision) as a bleed indicator. Plus **residual bass energy in the drum stem** below 150 Hz during walking-bass segments — the drum stem is the condition, so bass bleeding into it leaks the target. | The beat grid is tapped to the audio. |
| piano (SW / 6s) | chroma of the piano stem inside solo windows vs. the `melody` pitch of *piano* solos (`solo_info.instrument = p`); leakage = piano-stem energy in trio-less `s_ts` solos where JSD lists no `p`. | WJD has 60+ piano solos; JSD lists the accompanying instruments per segment. |

Plus, always: a listening page (`scripts/make_listening_page.py`) on **12 `pass` tracks stratified
by decade × style × soloist**, all stems side by side with the mix, **one column per model** —
this is the decision-making artifact; the metrics are there to rank the rest of the corpus
without listening to 18 h. Fix the 12 tracks in `experiments/03-wjd-jazz-stems/pilot_tracks.txt`
so every model sees the same set.

And the round trip: run `scripts/compare_autoencoders.py` on a handful of separated stems
(SAME-S, and SAME-L if 02 is still open) so we know separation artifacts + codec together,
which is the actual ceiling.

## Run plan

1. **Env + smoke test** (½ day): `sep` conda env, BS-Roformer-SW on one track, check stem
   count, length parity with the mix, FLAC output; same for MSST BS-RoFormer 4-stem.
2. **Pilot** (1 GPU-hour): 12 pilot tracks × {BS-Roformer-SW, BS-RoFormer 4-stem, SCNet-XL IHF,
   `htdemucs_ft`, `htdemucs_6s`}. Listening page + the four proxies by model. Log results here.
3. **Decide** (see below), then **full pass** with the winner(s) over all `pass` tracks;
   `check` tracks only after 3.1's beat check clears them.
4. Round-trip a dozen stems through SAME-S; note the combined ceiling.

## Decisions this sub-experiment has to produce

1. **RoFormer or Demucs**, per stem — one model for everything (simplest, no cross-model
   phase/level mismatch between stems) or per-stem picks (e.g. SW piano + `htdemucs_ft` drums)?
   Per-stem picks are only acceptable if the stems still sum close to the mix; report
   `|mix − Σ stems|` for the chosen combination.
2. 4-stem only, or 6-stem for piano/guitar? (With SW this is one model either way; the
   question becomes whether its piano stem is usable.)
3. Horn target = `other`, or `other + vocals`?
4. **Which decades to keep.** ~40 of the 190 aligned tracks are pre-1950 mono. Report every
   metric by decade and decide a cutoff (or a down-weighting) rather than assume. Watch for the
   RoFormer models, trained on stereo, doing something odd on dual-mono input.

## Deliverables

- `scripts/wjd/separate.py`, `sbatch/03_2_separate_wjd.sbatch`
- `scripts/wjd/eval_separation.py` (the proxies, by model / decade / instrument)
- `experiments/03-wjd-jazz-stems/pilot_tracks.txt`
- `/data/hai-res/shared/snnithya/sao-3/data/wjd/stems/<model>/<Track>/*.flac`
  (~15 GB FLAC per 4-stem model over 18 h; ~25 GB for 6 stems; pilot only for the losers)
- `/data/hai-res/shared/snnithya/sao-3/models/sep/` — checkpoints + configs (not in git)
- results table + listening page path here

## Results

### 2026-09-28 — smoke test, BS-Roformer-SW on `ArtBlakey_DownUnder_Orig` (1961, tp/ts/tb/p/b/dr)

Ran on huang-l40s-2 inside an interactive allocation. 328.1 s stereo mix (peak 1.00, i.e. the
YouTube transcode is already limited at 0 dBFS). All six stems: 14 468 097 frames = mix,
PCM-16, 44.1 kHz. Wall 50 s cold / 33 s warm.

| stem | rms dBFS | peak | note |
|---|---|---|---|
| mix | −14.8 | 1.00 | |
| drums | −24.4 | **1.00** | `peak_limited`: the raw stem exceeded 0 dBFS and was scaled down to fit PCM-16. Expect this on most tracks (the mix is already at the ceiling). Small global gain change on the *condition* stem; harmless for 3.3/3.4 as long as it is known. |
| bass | −19.4 | 0.80 | |
| other | −20.0 | 0.92 | horns + everything else |
| piano | −22.7 | 0.69 | |
| vocals | −81.1 | 0.002 | effectively empty on an instrumental — no horn leaked into `vocals` here (decision 3, one data point) |
| guitar | −99.8 | 0.0004 | empty, lineup has no guitar — correct |

`residual_db` = **−24.7 dB** (rms of mix − Σ stems is 5.8 % of the mix rms). RoFormer stems are
not constrained to sum to the mix, so this is the model's own residual, not a level bug;
`|mix − Σ stems|` for decision 1 is therefore already answered for the single-model case.
Not yet listened to.

### Full pass

Two workers on the same manifest, 2026-09-28:
- Job **2397042** (`sbatch/03_2_separate_wjd.sbatch`, defaults = SW over the 166 `pass` tracks,
  `hai-res-low`, requeueable), submitted 09:52 while the partition was full (16/16 L40S
  allocated), log `/data/scratch-fast/snnithya/sao-3/logs/03_2_separate_wjd_2397042.out`.
  (2397035, the first submission under `hai-res-main` without `--requeue`, was cancelled
  unstarted.)
- A `--reverse` worker in Nithya's interactive allocation on huang-l40s-2 (job 2396897,
  2 h debug limit), log `…/logs/03_2_separate_wjd_interactive_20260928_0941.out`; 11 tracks done
  by 09:50, restarted on the lease-aware code at 09:53.

Check progress with
`ls /data/hai-res/shared/snnithya/sao-3/data/wjd/stems/BS-Roformer-SW/*/done.json | wc -l`
(and `*/failed.json`). Re-submit the same sbatch to fill gaps if it times out or the
interactive allocation ends first.

Next: listening page on ~12 `pass` tracks stratified by decade (`scripts/make_listening_page.py`),
the proxies in `scripts/wjd/eval_separation.py` (not written yet), decision 3 (`other` vs
`other + vocals`) and decision 4 (decade cutoff) from those.
