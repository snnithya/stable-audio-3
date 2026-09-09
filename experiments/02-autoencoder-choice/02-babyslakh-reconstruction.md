# 2.2 — SAME-S vs SAME-L reconstruction on babySlakh

**Status:** measured; no decision taken yet
**Date:** 2026-09-08
**Branch:** `v/r`
**Scripts:** `scripts/make_submix_mirror.py` (staging), `scripts/compare_autoencoders.py`
**Outputs:** `/data/scratch-fast/snnithya/sao-3/outputs/ae_compare_babyslakh/` (79 excerpts × 3 wavs,
`metrics.json`, `index.html`) — ~857MB, kept on scratch rather than in the repo

## Question

[2.1](01-same-s-vs-same-l-reconstruction.md) measured the SAME-S/SAME-L reconstruction gap on
**bryan-data-1** — six Latin percussion tracks at 44.1 kHz, the material this pipeline is *aimed*
at. It is not the material the pipeline is currently *trained* on. Slakh2100 is
**16 kHz mono** (`slakh2100/streamgen-drum-mirror/train/tracks/drums/Track00001/Drums.flac`),
and everything in `slakh-streamgen-preencoded-same-s-wo-silence/` came from that.

So: does the gap 2.1 found hold on the corpus the DiT is actually being finetuned on?

## Why this is not just 2.1 with a different `--data_dir`

Two structural differences, both of which change what is being measured:

1. **16 kHz source.** Both autoencoders run at 44.1 kHz, so every excerpt is upsampled 16k → 44.1k
   before it is encoded. The top 18 kHz of the band the codec is designed for is *empty in the
   source* — across the excerpts checked, ≥8 kHz holds 4.6e-5 of the source's energy. This is a
   confound for 2.1's numbers but it is not a confound for the pipeline: it is exactly the signal
   the pre-encode sees. It does break one of 2.1's two metrics — see [Metrics](#metrics-the-mel-term-needed-fixing).
2. **`other` is per-instrument stems, not a premix.** bryan-data-1 ships one `Other.wav` per track.
   babySlakh ships 189 separate instrument stems across 20 tracks, and the streamgen pre-encode
   never encodes a lone stem — `custom_md_slakh_streamgen.load_and_mix_stems` builds a **stochastic
   submix** (a random subset of the non-silent stems, each LUFS-normalized into
   `[-30, -15]` and summed) and encodes *that*. Round-tripping `Piano.wav` on its own would measure
   something the model never sees.

`scripts/make_submix_mirror.py` handles (2): it stages a parallel tree with the drums symlinked
through and one `Other.wav` per track, built by importing the pipeline's own `load_and_mix_stems`
so there is no second copy of the mixing rules. Seeded per track, so it reproduces. The 20 submixes
span **1–13 stems** (median 6), which turns out to matter — see
[Density](#density-is-the-variable-that-predicts-difficulty).

The one place it is not equivalent: the submix is rolled over the **whole file** here, where the
pre-encode rolls it over just the window it is about to encode. Same distribution of subsets and
levels; a stem that is silent only inside some later excerpt can still be selected. Fine for
reconstruction work, not equivalent for anything needing the exact latents.

## Method

Identical to 2.1 otherwise, and deliberately so — same seed, same excerpt geometry, same harness:

- **Data:** the staged mirror, `{drums,other}/Track000{01..20}/` — 20 tracks × 2 stems, 241s each,
  16 kHz **mono** (so both streams get duplicated to stereo; the `other` submix is already
  stereo-with-identical-channels out of `_to_stereo`).
- **79 excerpts:** 2 random offsets per file, seed 0, rejecting windows below -50 dBFS RMS or more
  than 50% silent. 79 not 80 — `other-Track00019`'s submix is a single `Reed` stem (1 of 13 drawn)
  and only one window in 40 attempts cleared the silence gate.
- **11.981s each** = 528384 samples = **129 latent frames** exactly; chunking off in both
  directions; source written as the model sees it. Same reasoning as 2.1 for all three.
- **Metrics:** SI-SDR, plus log-mel L1 **split at 8 kHz** (new here, and necessary).

## Results

| Set | n | SAME-S SI-SDR | SAME-L SI-SDR | Δ | SAME-S mel L1 <8k | SAME-L mel L1 <8k |
|---|---|---|---|---|---|---|
| **All** | 79 | 14.60 ± 3.78 dB | 19.07 ± 5.79 dB | **+4.47 dB** | 0.547 | 0.431 |
| `drums` | 40 | 14.46 ± 3.61 dB | 17.98 ± 4.60 dB | **+3.53 dB** | 0.551 | 0.441 |
| `other` | 39 | 14.74 ± 3.95 dB | 20.18 ± 6.62 dB | **+5.43 dB** | 0.542 | 0.421 |

SAME-L wins on SI-SDR on **79/79** excerpts and on sub-8 kHz mel L1 on **79/79**.

**The headline replicates.** +3.53 dB on drums here against +3.16 dB on bryan-data-1; +5.43 dB on
`other` against +6.38 dB. Different corpus, different sample rate, different accompaniment
construction, and the drums margin lands within 0.4 dB. The 2.1 result was not a property of six
Latin percussion tracks.

Per track, SI-SDR (SAME-S → SAME-L):

```
drums-Track00001  14.23 → 16.51  +2.28    other-Track00001  13.84 → 19.39   +5.55
drums-Track00002  15.38 → 19.62  +4.24    other-Track00002   8.59 → 11.02   +2.42
drums-Track00003  11.75 → 14.32  +2.57    other-Track00003  14.23 → 17.72   +3.50
drums-Track00004  14.38 → 18.29  +3.91    other-Track00004  20.76 → 30.05   +9.29
drums-Track00005  13.58 → 17.30  +3.72    other-Track00005   9.61 → 11.76   +2.15
drums-Track00006  15.75 → 19.07  +3.32    other-Track00006  13.80 → 17.82   +4.02
drums-Track00007  15.68 → 19.49  +3.80    other-Track00007  15.50 → 20.71   +5.20
drums-Track00008  16.94 → 19.94  +3.01    other-Track00008  21.80 → 32.34  +10.54
drums-Track00009  19.31 → 24.07  +4.76    other-Track00009  13.68 → 18.42   +4.74
drums-Track00010  13.70 → 17.51  +3.80    other-Track00010  12.04 → 15.45   +3.41
drums-Track00011  12.51 → 15.13  +2.62    other-Track00011  15.40 → 19.31   +3.91
drums-Track00012  14.50 → 17.77  +3.27    other-Track00012  15.71 → 21.00   +5.29
drums-Track00013  11.46 → 14.37  +2.91    other-Track00013  16.37 → 21.79   +5.42
drums-Track00014   7.70 → 10.94  +3.25    other-Track00014  13.29 → 17.77   +4.48
drums-Track00015  14.96 → 17.78  +2.82    other-Track00015  19.00 → 26.68   +7.68
drums-Track00016  21.33 → 29.69  +8.36    other-Track00016  11.37 → 15.23   +3.85
drums-Track00017  16.00 → 19.58  +3.58    other-Track00017  12.42 → 16.73   +4.30
drums-Track00018   8.99 → 11.14  +2.16    other-Track00018   9.47 → 12.65   +3.18
drums-Track00019  19.28 → 23.07  +3.79    other-Track00019  18.31 → 25.29   +6.98  (n=1)
drums-Track00020  11.77 → 14.12  +2.35    other-Track00020  21.47 → 34.97  +13.50
```

### The gap is largest where it is least needed

2.1 noticed this as one anecdote — `el-espiritu-de-la-rumba` was the worst track under SAME-S
(8.3 dB) *and* had the narrowest SAME-L margin (+2.1 dB). With 40 tracks it is a corpus-wide
correlation, not an anecdote:

| | Pearson r |
|---|---|
| SAME-S SI-SDR vs. the SAME-L gap, all 79 excerpts | **+0.72** |
| … `drums` only | +0.60 |
| … `other` tracks only | **+0.91** |

**SAME-L's advantage grows with how well SAME-S already did.** The three worst drum tracks
(Track00014 at 7.70 dB, Track00018 at 8.99, Track00003 at 11.75) get +3.25, +2.16 and +2.57; the
best (Track00016 at 21.33) gets +8.36. On `other` the relationship is nearly linear: the
easy end (Track00020, 21.47 dB) gains +13.50 dB while the hard end (Track00005, 9.61 dB) gains
+2.15.

This is the single most important thing in this experiment, and it cuts *against* the switch. The
argument for SAME-L is that hard material is being reconstructed badly. What the data says is that
SAME-L is mostly buying headroom on the material that was already fine, and that whatever makes a
track hard is not something a bigger autoencoder fixes.

### Density is the variable that predicts difficulty

The staged submixes give a direct handle on arrangement density, and it explains most of the
`other` spread:

| | Pearson r |
|---|---|
| stems in the submix vs. SAME-S SI-SDR | **-0.74** |
| stems in the submix vs. the SAME-L gap | -0.66 |

```
 1 stem    18.31 / 20.76 / 21.47 dB      gap  +6.98 / +9.29 / +13.50
 2 stems   15.40 / 21.80 dB              gap  +3.91 / +10.54
 6 stems    8.59 / 11.37 / 12.42 / 16.37 gap  +2.42 / +3.85 / +4.30 / +5.42
10 stems    9.61 dB                      gap  +2.15
13 stems    9.47 dB                      gap  +3.18
```

A one-stem submix (a lone bass, a lone reed) round-trips at ~20 dB and gains ~10 dB from SAME-L; a
ten-stem arrangement sits under 10 dB and gains ~2. So the `other` numbers in *both* experiments are
substantially a statement about how many things are playing at once, and the "+5.43 dB on
accompaniment" line should not be read as a property of accompaniment. It is the average over a
density distribution that the pre-encode re-rolls anyway.

This also reframes 2.4 (`el-espiritu-de-la-rumba`): dense simultaneous content is a sufficient
explanation for a hard track, and does not require a per-recording story.

### Metrics: the mel term needed fixing

Computed the way 2.1 computed it — over all 128 mel bins — SAME-L wins mel L1 on only **41/79**
excerpts here, against 46/48 on bryan-data-1, and every value is ~60% higher (0.96 vs 0.60 mean).
That is not a real disagreement with SI-SDR. It is the 16 kHz source:

| band | mel bins | SAME-S | SAME-L | SAME-L wins |
|---|---|---|---|---|
| **< 8 kHz** | 93/128 | 0.547 | **0.431** | **79/79** |
| ≥ 8 kHz | 35/128 | 2.057 | 2.331 | 1/79 |

Above 8 kHz the source is empty (4.6e-5 of its energy), so `log10(clamp(·, 1e-10))` puts those bins
on the floor and any tiny nonzero decoder output reads as a huge L1. It is not audible content
being measured: the reconstructions hold 8.1e-5 (SAME-S) and 6.4e-5 (SAME-L) of their energy up
there — SAME-L emits *less* excess energy in absolute terms while scoring worse in the log domain,
which is what a broader, lower noise floor across many clamped bins does to a mean L1.

**So the ≥8 kHz term should be discarded, not interpreted**, and the sub-8 kHz numbers (0.547 →
0.431) line up with bryan-data-1's (0.600 → 0.500) and with SI-SDR's 79/79. Worth knowing before
anyone runs this harness on 16 kHz material again — the whole-band mel L1 in `metrics.json` is not
usable there.

### Timing

Per 12s excerpt on an L40S, median over 79 excerpts, no other process on the GPU: **SAME-S 32ms
encode / 36ms decode, SAME-L 281ms / 299ms**. Still irrelevant at pre-encode scale — all 79
excerpts through both models took a couple of minutes — but both numbers are ~2× and ~4× 2.1's
(16/11ms and 71/74ms) and the S→L ratio is 8.3× here against 4.5× there. Same script, same GPU
model, same excerpt geometry. Unexplained; noted in Open. The within-run spread is tight
(SAME-L encode 271–291ms), so it is not measurement noise.

SAME-S again emits **130** latent frames for a 129-frame input where SAME-L emits 129, as in 2.1.

## Reading

2.1's central number survives the change of corpus: **+3.5 dB on drums**, on the 16 kHz mono
material the streamgen latents are actually built from, on 79/79 excerpts. That is the strongest
version of the reconstruction-ceiling argument available without a training run.

But the two new findings both weaken the case for paying the migration cost:

- The gain concentrates on **easy** material (r = +0.72). The tracks reconstructed worst under
  SAME-S — which are the ones that would motivate a switch — gain the least from SAME-L.
- Most of the `other` spread is **arrangement density**, a variable the pre-encode re-rolls every
  variant. A headline gap measured on a particular density draw is not a stable property of the
  corpus.

Neither says SAME-L is not better; it is, unambiguously and everywhere. They say the ceiling this
measures is being raised mostly where the ceiling was not the binding constraint. Still nothing here
about whether a DiT on 256-ch SAME-L latents learns streamgen conditioning as well or as fast —
that is 2.6, and it remains the experiment that would actually justify re-encoding Slakh.

## Open

- [ ] Listen to `/data/scratch-fast/snnithya/sao-3/outputs/ae_compare_babyslakh/index.html` — is
      +3.5 dB on 16 kHz drums audible at all? The band where a 4096× ratio should hurt most
      (cymbals, stick attacks above 8 kHz) is *absent from the source*, so this material may be an
      easier case for SAME-S than 44.1 kHz drums are, in which case 2.1's bryan-data-1 numbers are
      the ones that describe the eventual target material and these describe the training corpus.
- [ ] Reconcile the timings against 2.1 (32/36 vs 16/11ms for the same model on the same GPU model).
      Cheap to check: re-run 2.1's command and see whether its numbers reproduce.
- [ ] Fold the 8 kHz mel split into `compare_autoencoders.py` rather than computing it after the
      fact, or drop the mel term for band-limited sources. As written, `metrics.json`'s
      `log_mel_l1` is misleading on any 16 kHz corpus — which is all of Slakh.
- [ ] Test the density finding directly: fix a track and sweep the submix from 1 to N stems. If
      -0.74 holds at fixed material, arrangement density is the thing to report alongside every
      reconstruction number in this experiment, and the honest way to state 2.1 and 2.2's `other`
      results is per-density.

## Reproduce

```bash
uv run python scripts/make_submix_mirror.py \
    --src /data/hai-res/shared/snnithya/sat-zenon-data/babyslakh/streamgen-drum-mirror/tracks \
    --out /data/scratch-fast/snnithya/sao-3/data/babyslakh-submix-mirror/tracks
uv run python scripts/compare_autoencoders.py \
    --data_dir /data/scratch-fast/snnithya/sao-3/data/babyslakh-submix-mirror/tracks \
    --out /data/scratch-fast/snnithya/sao-3/outputs/ae_compare_babyslakh -n 2 --seconds 12 --seed 0
uv run python scripts/make_listening_page.py --dir /data/scratch-fast/snnithya/sao-3/outputs/ae_compare_babyslakh
python -m http.server -d /data/scratch-fast/snnithya/sao-3/outputs/ae_compare_babyslakh 8000
```
