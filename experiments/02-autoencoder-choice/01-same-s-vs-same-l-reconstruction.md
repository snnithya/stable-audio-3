# 2.1 — SAME-S vs SAME-L reconstruction on the bryan-data-1 stems

**Status:** measured; no decision taken yet
**Date:** 2026-09-08
**Branch:** `v/r`
**Script:** `scripts/compare_autoencoders.py`
**Outputs:** `/data/scratch-fast/snnithya/sao-3/outputs/ae_compare/` (48 excerpts × 3 wavs, `metrics.json`, `index.html`) — ~570MB, kept on
scratch rather than in the repo

## Question

The streamgen latents are pre-encoded with **SAME-S**
(`.../slakh-streamgen-preencoded-same-s-wo-silence/`). Everything the DiT can ever produce is
bounded above by what the autoencoder can reconstruct, so: how much quality is that choice
costing on *this* kind of material — percussion-forward Latin stems, drums against accompaniment?

## Method

Encode → decode round-trip only. No diffusion, no DiT: this isolates the autoencoder.

- **Data:** `bryan-data-1/streamgen-drum-mirror/train/tracks/{drums,other}/<track>/` — 6 tracks
  × 2 stems, 114–296s each, 44.1 kHz. `drums` is stereo, `other` is **mono**.
- **48 excerpts:** 4 random offsets per file, seed 0, rejecting windows below -50 dBFS RMS or
  more than 50% silent (`is_silent` / `silence_fraction` from `data/utils.py`). Drum stems have
  real rest bars and every model reconstructs nothing perfectly.
- **11.981s each** = 528384 samples = **129 latent frames** exactly. 12s was snapped *down* to a
  whole frame so the encoder's pad-to-multiple-of-4096 step is a no-op; otherwise the round-trip
  returns ~74ms longer than its source and the A/B is misaligned at the tail.
- **Chunking off** in both directions. Chunked encode/decode splices independently-run windows
  with a hard cut at each inner edge (`autoencoders.py:550`), so a seam artifact would show up in
  the comparison as if it were an autoencoder difference. 129 frames fits one pass on an L40S.
- **Source is written as the model sees it** — resampled and channel-converted before saving — so
  all three rows are sample-aligned. This matters here: the mono `other` stems get duplicated to
  stereo for a stereo autoencoder, and comparing the mono original against a stereo
  reconstruction would have measured that conversion rather than the codec.
- **Metrics:** SI-SDR (scale-invariant, so a correct-but-quieter reconstruction is not punished)
  and mean L1 between log-mel magnitudes (SI-SDR is waveform-aligned and punishes phase drift a
  decoder can be perceptually fine about; the pair keeps both visible).

## Results

| Set | n | SAME-S SI-SDR | SAME-L SI-SDR | Δ | SAME-S mel L1 | SAME-L mel L1 |
|---|---|---|---|---|---|---|
| **All** | 48 | 12.76 ± 2.64 dB | 17.53 ± 4.03 dB | **+4.77 dB** | 0.600 | 0.500 |
| `drums` | 24 | 11.22 ± 2.40 dB | 14.38 ± 2.98 dB | **+3.16 dB** | 0.520 | 0.450 |
| `other` | 24 | 14.31 ± 1.86 dB | 20.69 ± 1.87 dB | **+6.38 dB** | 0.690 | 0.550 |

SAME-L wins on SI-SDR on **48/48** excerpts and on mel L1 on 46/48 (the two exceptions,
`other-chango-iv-1-t0134` and `other-sei-seisima-1-t0093`, are the two highest-mel-L1 excerpts
for both models — worth a listen before reading anything into them).

Per track, SI-SDR (SAME-S → SAME-L):

```
drums-chango-iv-1                13.67 → 17.14  +3.47      other-chango-iv-1                13.20 → 19.65  +6.45
drums-chango-iv-2                13.78 → 16.95  +3.17      other-chango-iv-2                14.17 → 20.48  +6.31
drums-el-espiritu-de-la-rumba-1   8.40 → 10.59  +2.19      other-el-espiritu-de-la-rumba-1  15.69 → 22.05  +6.36
drums-el-espiritu-de-la-rumba-2   8.34 → 10.46  +2.12      other-el-espiritu-de-la-rumba-2  14.54 → 20.62  +6.08
drums-sei-seisima-1              10.85 → 14.82  +3.97      other-sei-seisima-1              13.85 → 20.21  +6.36
drums-sei-seisima-2              12.27 → 16.32  +4.05      other-sei-seisima-2              14.41 → 21.12  +6.71
```

Two things the aggregate hides:

1. **Drums are the hard case, and the gap is smallest exactly where absolute quality is worst.**
   `el-espiritu-de-la-rumba` sits at 8.3 dB under SAME-S — 4–5 dB below the other drum tracks —
   and SAME-L only recovers ~2.1 dB of it, its *narrowest* margin anywhere. Whatever is hard about
   that track is not something the bigger autoencoder fixes. Both `el-espiritu` stems behave this
   way, so it is a property of the track, not of one excerpt draw.
2. **The `other` stems are mono duplicated to stereo**, i.e. perfectly correlated channels. That is
   an easy case for a stereo codec and it inflates the `other` numbers for both models. The +6.38 dB
   gap there should not be read as "SAME-L is twice as far ahead on accompaniment" — it is
   the gap on a signal class that is not what real stereo accompaniment looks like.

**Timing** (L40S, per 12s excerpt, warm): SAME-S 16ms encode / 11ms decode, SAME-L 71ms / 74ms.
~4.5× slower, and irrelevant at pre-encode scale — 48 excerpts round-tripped through both models
in well under a minute. Latency only becomes an argument if the decoder ends up in an interactive
loop.

SAME-S emits **130** latent frames for a 129-frame input where SAME-L emits 129 — its chunked
attention pads internally. The extra frame lands at the *end*: FFT cross-correlation puts both
reconstructions at lag 0 against the source, so trimming to the source length is correct and the
metrics above are not measuring an offset.

## Reading

SAME-L is unambiguously the better reconstructor here, by a margin (+3.2 dB on drums) that is
well outside the excerpt-to-excerpt spread. But this measures the *ceiling*, not the finetune:
it says nothing about whether a DiT trained on 256-ch SAME-L latents learns the streamgen
conditioning as well or as fast as one trained on SAME-S latents, and switching means re-encoding
the Slakh dataset and retraining from scratch.

## Open

- [ ] Listen to `/data/scratch-fast/snnithya/sao-3/outputs/ae_compare/index.html` and check whether +3.2 dB on drums is *audible*
      on transients — cymbals and stick attacks are where a 4096× downsampling ratio should hurt,
      and SI-SDR averaged over 12s is not a good detector of that.
- [ ] Work out why `el-espiritu-de-la-rumba` is 4–5 dB worse than every other drum track under
      both models. If it is a property of the recording (level, ambience, dense percussion) it
      predicts which material this pipeline will struggle with generally.
- [ ] Re-run the `other` comparison on genuinely stereo accompaniment before trusting the +6.38 dB.

## Reproduce

```bash
uv run python scripts/compare_autoencoders.py \
    --data_dir /data/hai-res/shared/snnithya/sat-zenon-data/bryan-data-1/streamgen-drum-mirror/train/tracks \
    --out /data/scratch-fast/snnithya/sao-3/outputs/ae_compare -n 4 --seconds 12 --seed 0
uv run python scripts/make_listening_page.py --dir /data/scratch-fast/snnithya/sao-3/outputs/ae_compare
python -m http.server -d /data/scratch-fast/snnithya/sao-3/outputs/ae_compare 8000
```
