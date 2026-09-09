# Experiment 02 — Autoencoder choice

**Status:** in progress — 2.1 and 2.2 measured, no decision taken
**Started:** 2026-09-08
**Branch:** `v/r`

---

## Question

The streamgen latents are pre-encoded with **SAME-S**
(`.../slakh-streamgen-preencoded-same-s-wo-silence/`, see
[1.5](../01-streamgen-conditioning/04-silence-filtering.md)). Nothing the DiT generates can be
better than what that autoencoder can reconstruct, so the ceiling on the whole streamgen line of
work is set before any training starts.

**Is SAME-S costing enough on percussion-forward material to be worth moving off?**

Two sub-claims, and only the first is testable without a training run:

1. **There is a real reconstruction gap on this material.** SAME-S is a distillation of SAME-L
   (266M vs 1.7B, `docs/guides/model-overview.md:57-64`), so a gap is expected; the question is
   its size on drums specifically, where a 4096× downsampling ratio should hurt most.
2. **The gap survives the DiT.** A finetune on SAME-L latents produces audibly better drums than
   the same finetune on SAME-S latents. This does not follow from 1 — a larger reconstruction
   ceiling is not worth much if the DiT cannot reach it, or learns the conditioning more slowly in
   the new latent space.

## Background

Both autoencoders emit **256-channel latents at ds 4096** (~10.8 Hz at 44.1 kHz), so the shapes are
interchangeable and every downstream config — `latent_crop_length`, the `modular_local_cond` widths,
the streamgen sidecars — is unchanged by a swap. The *values* are not interchangeable: they are
different latent spaces, and the pretrained checkpoints are paired with one each
(`stable_audio_3/model_configs.py:95-105`):

| | SAME-S | SAME-L |
|---|---|---|
| Params | 266M | 1.7B |
| Attention | chunked w/ midpoint shift | sliding window |
| Paired DiT | `small-music`, `small-sfx` | `medium` |

So "switch to SAME-L" is not a one-line config change in practice: it means re-encoding Slakh, and
finetuning either `medium` (a 1.4B DiT, ~3× the current one) or `small-music` on latents it was not
pretrained on. That cost is the reason 2.1 exists as a measurement rather than as a decision.

## Sub-experiments

| # | Name | Question | Status |
|---|------|----------|--------|
| 2.1 | [SAME-S vs SAME-L reconstruction](01-same-s-vs-same-l-reconstruction.md) | How much quality does SAME-S cost on the bryan-data-1 stems, round-trip only? | **measured** |
| 2.2 | [Same, on babySlakh](02-babyslakh-reconstruction.md) | Does the 2.1 gap hold on the 16 kHz corpus the DiT is actually finetuned on? | **measured** |
| 2.3 | Listening check | Is +3.2 dB on drums audible on transients — cymbals, stick attacks? | not started |
| 2.4 | `el-espiritu-de-la-rumba` diagnosis | Why is one track 4–5 dB worse than every other under *both* models? | not started |
| 2.5 | Genuinely stereo accompaniment | Does the +6.4 dB `other` gap hold when the stems are not mono-duplicated? | not started |
| 2.6 | Downstream finetune | Does a DiT on SAME-L latents learn streamgen conditioning as well and as fast? | not started |
| 2.7 | Submix density sweep | Does reconstruction quality track arrangement density at fixed material (2.2 found r = -0.74)? | not started |

## Results

**2.1 (measured).** Encode→decode round-trip on 48 excerpts (6 tracks × 2 stems × 4 offsets,
11.981s each = 129 latent frames, chunking off). SAME-L wins on SI-SDR on **48/48** excerpts:
**+3.16 dB on drums** (11.22 → 14.38) and +6.38 dB on `other` (14.31 → 20.69). The drums margin is
well outside the excerpt-to-excerpt spread (±2.4 / ±3.0 dB); the `other` margin is inflated because
those stems are mono duplicated to stereo, which is an easy case for a stereo codec.

Two things that do not show in the aggregate: the gap is *narrowest* (+2.1 dB) exactly on the track
where absolute quality is worst (`el-espiritu-de-la-rumba`, 8.3 dB under SAME-S), so the bigger
autoencoder does not fix whatever is hard there; and timing (16/11ms vs 71/74ms per 12s excerpt on
an L40S) is irrelevant at pre-encode scale — it only becomes an argument if a decoder ends up in an
interactive loop. See [2.1](01-same-s-vs-same-l-reconstruction.md).

**2.2 (measured).** The same round-trip on **babySlakh** — 79 excerpts, 20 tracks × 2 stems × 2
offsets — i.e. on 16 kHz mono material, which is what Slakh2100 and therefore
`slakh-streamgen-preencoded-same-s-wo-silence/` actually are. `other` is staged as the pipeline's
own stochastic stem submix (`scripts/make_submix_mirror.py`), not as individual stems, because a
lone `Piano.wav` is not something the pre-encode ever encodes.

**2.1's headline replicates:** SAME-L wins 79/79 on SI-SDR, **+3.53 dB on drums** (14.46 → 17.98)
against 2.1's +3.16 dB, and +5.43 dB on `other` against +6.38. Different corpus, different sample
rate, different accompaniment construction, drums margin within 0.4 dB.

Two new findings, both of which weaken the case for switching:

1. **The gain concentrates on easy material.** SAME-S's own score predicts SAME-L's margin with
   r = **+0.72** over all 79 excerpts (+0.91 across the `other` tracks). The worst tracks gain
   ~2 dB, the best gain 8–13. 2.1 saw this as one anecdote (`el-espiritu-de-la-rumba`); it is a
   corpus-wide relationship. Whatever makes a track hard, a bigger autoencoder does not fix it.
2. **Most of the `other` spread is arrangement density.** Stems in the submix vs. SI-SDR is
   r = **-0.74**: a one-stem submix round-trips at ~20 dB and gains ~10 dB from SAME-L, a ten-stem
   arrangement sits under 10 dB and gains ~2. Since the pre-encode re-rolls the submix per variant,
   a headline `other` gap is a statement about a density draw, not about accompaniment.

It also found that **the whole-band mel L1 metric is unusable on 16 kHz sources** — above 8 kHz the
source is empty (4.6e-5 of its energy) and the log floor dominates, flipping the metric to 41/79.
Restricted to the 93 bins below 8 kHz, SAME-L wins 79/79 (0.547 → 0.431), consistent with SI-SDR
and with 2.1. See [2.2](02-babyslakh-reconstruction.md).

**No decision taken.** 2.1 and 2.2 measure the ceiling, not the finetune, and the migration cost
above is real. 2.6 is the experiment that would justify paying it — and 2.2's first finding is a
reason to expect less from it than 2.1's headline suggests.

## Notes

- The round-trip harness is `scripts/compare_autoencoders.py`; it takes `--models` and is not welded
  to these two checkpoints or to the `streamgen-drum-mirror` layout.
- Outputs live on scratch, not in the repo, each with an `index.html` from
  `scripts/make_listening_page.py` for A/B listening: `outputs/ae_compare/` (2.1, ~570MB) and
  `outputs/ae_compare_babyslakh/` (2.2, ~857MB) under
  `/data/scratch-fast/snnithya/sao-3/`.
- 2.1 is measured on **bryan-data-1**, six Latin percussion tracks at 44.1 kHz — the material this
  pipeline is aimed at. 2.2 is measured on **babySlakh**, 16 kHz mono — the material it is currently
  trained on. The drums gap agrees to within 0.4 dB, so on that number the two do transfer; the
  band-limited source means 2.2 says nothing about the >8 kHz transient detail that is the whole
  reason to worry about a 4096× ratio on drums.
- `scripts/make_submix_mirror.py` stages a drum-mirror tree whose `other` is one submix per track,
  by importing `custom_md_slakh_streamgen.load_and_mix_stems`. Any tool that wants to look at "the
  accompaniment the model sees" on Slakh needs it; the raw tree has only per-instrument stems.
