# 3.3 — Per-frame inference schedules: does a rolling schedule buy block coherence?

**Status:** plan, nothing implemented (2026-09-21). Edit freely; decision points are marked **[decide]**.
**Parent:** [Experiment 03](README.md)
**Predecessor:** [3.2 — Training with independent per-frame `t`](02-training-with-per-frame-t.md)
**Branch:** `nithya/df`
**Checkpoint under test:** run `df-slakh` / wandb `7tvd98n0`, 10 000 steps at `df_p_global: 0.0` —
`/data/scratch-fast/snnithya/sao-3/ft_checkpoints/sao-3/7tvd98n0/checkpoints/{last,epoch=48-step=5000,epoch=96-step=10000}.ckpt`

---

## Question

The DF model can hold every latent frame at its own noise level. For streaming we want to emit
audio in fixed **4 s chunks**. The noise schedule inside the window is then a free choice, and
the two extremes are:

1. **Block-uniform** (`block` below; "the same `t` for all elements"). Chunk *k* is denoised on its own, every
   frame in it at the same `t`, with the chunks before it clean and the chunks after it untouched
   noise. When it reaches `t = 0` it is emitted and chunk *k+1* starts from scratch. This is
   inpainting-style continuation, one chunk at a time.
2. **Linear / rolling** (`linear` below; "denoised at different levels"). Frame `f` lags frame `f−1` by a fixed
   number of sampler steps, so at any moment the window shows a ramp: clean on the left, noise
   on the right, every intermediate level in between. Chunk *k+1* is already partly denoised —
   jointly with chunk *k* — when chunk *k* is emitted.

**Does the rolling schedule produce more coherent seams between consecutive chunks than the
block-uniform one, at the same per-chunk latency?**

Hypothesis: yes. Under (1) chunk *k+1* only ever sees chunk *k* as fixed context, so the seam is
a hard conditioning boundary; under (2) the frames on both sides of the seam spend most of the
trajectory being denoised together, so tempo, phase and level should carry across. The cost is
pipeline fill: the first chunk arrives one stage later.

Secondary questions, cheap once the script exists:

- How far does either schedule sit from **joint full-window generation** (the coherence
  ceiling, but no streaming)?
- Does **pyramid** (block granularity, chunks overlapping by half a trajectory) get most of the
  rolling benefit? It is the paper's schedule and easier to reason about than the per-frame ramp.

## Setup

### Geometry

| | |
|---|---|
| latent frame rate | 44100 / 4096 = **10.77 frames/s** |
| training window | 144 frames = 13.37 s (`latent_crop_length: 144`) |
| 4 s chunk | 43.07 frames → **`C = 43` frames = 3.99 s** |
| layout **[decide]** | **15-frame ground-truth prefix (1.39 s) + 3 × 43-frame chunks = 144.** Tiles the window exactly, and a non-empty prefix matches the causal-inpaint regime the model trained under (`mask_type_probabilities [0, 0, 1]`, random prefix length). Alternatives: no prefix, 3 chunks + a 15-frame partial 4th; or `C = 48` (4.46 s) × 3 with no prefix. |
| window length | stay at 144. Rolling out *past* 144 is a separate question (README row 3.3 second half) and confounds coherence with extrapolation. |

### What a schedule is

At every sampler step the model does **one forward pass over the whole 144-frame window**, and
`t` is a `(B, 144)` vector — one noise level per frame. A schedule is therefore a *matrix*: the
noise level of every frame at every step, `t[step, frame]`, shape `(S_total + 1, 144)`. Sampling
walks down its rows, one forward pass per row.

That is the crux of the hypothesis. Frames are never denoised in separate forward passes; they
are always denoised *together*, and the schedule only sets how far apart in the trajectory they
are. Two frames either side of a chunk boundary attend to each other at every step of both their
trajectories — the question is whether that helps more when they are at *similar* noise levels
(rolling) than when one is already clean and the other is still pure noise (block).

### Building one: a shared trajectory plus a per-frame delay

Every schedule below is the *same* 1-D trajectory, started at a different step for each frame.
That keeps distribution shift and `sigma_max` handling exactly as `build_schedule` does them
today, and it makes each schedule a single vector of integers.

```
s = build_schedule(S, dist_shift=..., ...)   # (S+1,), s[0] = 1, s[S] = 0, already shifted
t[i, f] = s[clamp(i - d[f], 0, S)]           # frame f starts its trajectory at step d[f]
```

So frame `f`:

- sits at `t = 1`, untouched noise, for steps `i < d[f]`;
- runs the identical `S`-step trajectory over steps `d[f] … d[f] + S`;
- sits at `t = 0`, clean and committed, from then on.

Two properties fall out for free. **Frozen frames stay frozen**: a frame clamped at either end
has `dt = s[j] - s[j] = 0`, so the Euler update leaves it bit-identical — committed chunks and
not-yet-started chunks need no masking and no copy-back. **Every frame gets exactly `S` steps of
denoising in every schedule**, so the schedules differ in *when*, never in *how much*.

Notation: `S` = steps per chunk, `C = 43` = chunk length in frames, `K = 3` = number of chunks,
`f0 = 15` = first generated frame, `k(f)` = chunk index of frame `f` (prefix frames are `k = -1`
and pinned at `t = 0`), `N = 144`.

### The four schedules, drawn

Toy version so the shapes are visible: 12 frames, no prefix, `C = 4` frames per chunk, `K = 3`
chunks, `S = 4` steps per chunk. Each digit is one frame's noise level on a 0–4 scale, where
**4 = pure noise (`t = 1`)** and **0 = clean (`t = 0`)**. `<` marks a chunk being emitted.

```
  block   d = 0,0,0,0, 4,4,4,4, 8,8,8,8       linear   d = 0,1,2,3, 4,5,6,7, 8,9,10,11
  step   c0   c1   c2                          step   c0   c1   c2
    0   4444 4444 4444                           0   4444 4444 4444
    1   3333 4444 4444                           1   3444 4444 4444
    2   2222 4444 4444                           2   2344 4444 4444
    3   1111 4444 4444                           3   1234 4444 4444
    4   0000 4444 4444  < c0                     4   0123 4444 4444
    5   0000 3333 4444                           5   0012 3444 4444
    6   0000 2222 4444                           6   0001 2344 4444
    7   0000 1111 4444                           7   0000 1234 4444  < c0
    8   0000 0000 4444  < c1                     8   0000 0123 4444
    9   0000 0000 3333                           9   0000 0012 3444
   10   0000 0000 2222                          10   0000 0001 2344
   11   0000 0000 1111                          11   0000 0000 1234  < c1
   12   0000 0000 0000  < c2                    12   0000 0000 0123
                                                13   0000 0000 0012
                                                14   0000 0000 0001
                                                15   0000 0000 0000  < c2

  pyramid d = 0,0,0,0, 2,2,2,2, 4,4,4,4       joint    d = 0 everywhere
  step   c0   c1   c2                          step   c0   c1   c2
    0   4444 4444 4444                           0   4444 4444 4444
    1   3333 4444 4444                           1   3333 3333 3333
    2   2222 4444 4444                           2   2222 2222 2222
    3   1111 3333 4444                           3   1111 1111 1111
    4   0000 2222 4444  < c0                     4   0000 0000 0000  < all
    5   0000 1111 3333
    6   0000 0000 2222  < c1
    7   0000 0000 1111
    8   0000 0000 0000  < c2
```

Read off the shape of each row: `block` is a **step function** — one chunk in flight, its
neighbours either clean or pure noise. `linear` is a **ramp** exactly one chunk wide, sliding
right one frame per `S/C` steps. `pyramid` is a **staircase** with two chunks in flight. `joint`
is **flat** — the ordinary global-`t` sampler, just expressed as a `(B, N)` vector.

### Real numbers, and what "same latency" means

At `S = 50`, `C = 43`, `K = 3`, `f0 = 15`:

| schedule | delay `d[f]` | steps each frame gets | first chunk out at step | steps between emissions | total forward passes |
|---|---|---|---|---|---|
| `joint` | `0` | `S` | 50, all three at once | n/a | 50 |
| `block` | `k(f)·S` | `S` | 50 | 50 | `K·S` = 150 |
| `linear` | `round((f − f0)·S/C)` | `S` | ≈ 99 | 50 | ≈ 199 |
| `pyramid` | `k(f)·S/2` | `S` | 50 | 25 | 100 |

- **`block` vs `linear` is the matched comparison** the question asks for: identical `S` per
  frame, identical 50-step gap between emissions in steady state. `linear` pays one extra
  pipeline-fill stage (first chunk at step 99 rather than 50) and ~33% more total forwards.
  Both are fill costs, not a larger quality budget.
- **`pyramid` at the same `S` is *not* cadence-matched** — it emits every `S/2` steps, twice as
  often. Read it as "same per-frame budget at half the latency", or halve `S` for the others if
  you want emissions matched.
- **This run is offline.** Frozen frames cannot change, so the final latents are identical
  whether or not anything is "emitted". Emission timing matters in exactly two places: the
  latency column above, and the step at which the conditioning cursor advances (next section).
- **`linear` could emit much faster than one chunk at a time.** Frame `f0` is clean at step 50
  and each later frame finishes ~1.16 steps after its neighbour, so a real streaming system
  could release one frame (93 ms) at a time once the pipeline is full. Holding emission to 4 s
  chunks keeps the comparison controlled and *understates* the rolling schedule's latency
  advantage.
- **Ramp width** under `linear` is one chunk by construction (`S` steps spread over `C` frames):
  the frame just past a freshly emitted chunk sits at `s[S − S/C]`, nearly clean, while the frame
  one chunk further on is still at `t = 1`. `--ramp r` stretches this over `r` chunks if the
  effect is real but small.
- Save each matrix as a heatmap (`steps × frames`) next to the audio. It is the figure for the
  write-up and the fastest way to see a schedule bug.


### Conditioning — what the model sees at each stage

The DF checkpoint was trained with **both** mechanisms present: per-frame `t` on the noised
input *and* causal inpaint conditioning (`inpaint_mask`, `inpaint_masked_input` as
`local_add_cond`) *and* the accompaniment latent gated to a lookahead horizon
(`streamgen_latent`, `tf_inpaint_mask`, horizon ∈ `[−4, 0]` s of the cursor). So there are two
ways to hand the model the committed past, and the accompaniment has to follow a cursor.

**Cursor** := end of the last *committed* (fully `t = 0`) chunk. Advances once per emission.
Conditioning is rebuilt at every stage boundary (`get_conditioning_inputs` is cheap).

| knob | options | default **[decide]** |
|---|---|---|
| `--context_feed` | `t0` (DF-native: committed frames at `t = 0` in `x`, inpaint mask all zero); `inpaint` (as trained: inpaint mask 1 on committed frames, `x` there still follows the schedule — for `block` that is what today's continuation does); `both` | **`both`** — it is what the model saw most often in training (per-frame iid `t` includes near-zero frames on the prefix, *and* the inpaint cond was always there). Run `t0` as the ablation that says whether the inpaint channel still matters under DF. |
| `--accomp` | `causal` (accompaniment visible up to cursor + `lookahead_seconds`, the streaming-honest setting; chunk *k+1* starts denoising under `linear` *before* it can see its accompaniment); `oracle` (whole window visible to every schedule — isolates the noise-schedule effect but is outside the training horizon) | **`causal`, lookahead 0 s** — matches `sbatch/01_2_eval.sbatch`. Report `oracle` as a secondary row; if `linear` only wins under `oracle`, the streaming version needs lookahead ≥ one chunk to help. |
| `--cfg_scale` | | **1.0**, as in experiment 01's eval. Note `t = 0` frames divide by `sigma` in the CFG branch (`dit.py`, `(x − cfg_denoised) / sigma_b`); with `cfg_scale ≠ 1` clamp committed frames to `t = 1e-3` rather than 0. |
| `S` (steps per chunk) | | **50** (matches `--gen_steps 50`); `8` as a quick mode (the demo setting). |
| `--sampler` | | **euler** only. `dpmpp` per-frame is a later thing; `pingpong` is for `rf_denoiser` and irrelevant to the base model. |

### Data

`slakh_streamgen_validation_preencoded.json` — 56 held-out items, all exactly 144 frames,
`random_crop: false`, so every schedule generates the same frames from the same prefix. Fixed
per-item noise seeded by item index (as `eval_streamgen.py` does), **the same noise tensor for
every schedule**, so per-item comparisons are paired.

## Metrics — "coherence at the seam"

Every metric is computed **at seams** (the two chunk boundaries at frames 58 and 101, i.e.
5.4 s and 9.4 s) and **at matched non-seam positions** in the same clip, and reported as a
seam/non-seam ratio, so a schedule is judged against itself and clip-level differences cancel.
Three references make the numbers readable: `joint` (ceiling), ground truth (what real drums
look like at those positions), and a **splice control** — chunks from two different seeds
glued at the seam, which is what a maximally incoherent seam scores.

| metric | where | what it catches |
|---|---|---|
| **latent jump** — `‖z[f_b] − z[f_b − 1]‖₂` at the boundary frame vs. the median over interior frames | latent, free | a discontinuity the decoder then has to paper over |
| **onset-envelope spike** — spectral-flux energy in ±50 ms around the seam vs. elsewhere (`onset_envelope` from `eval_streamgen.py`) | audio | clicks, level jumps, a doubled or dropped hit |
| **beat continuity** — `estimate_pulse` on chunk *k* and *k+1* separately: period ratio and phase offset at the seam, also against the accompaniment's grid (`beat_hit_rate`) | audio, drums-specific | tempo drift or a phase reset across the seam — the musically meaningful failure |
| **listening** | `make_listening_page.py` over the output dir (may need a `--label` regex for the new stems); mel with vertical lines at seams | the thing the metrics approximate |

Bootstrap CIs over items and paired deltas (`linear − block`) per metric, reusing
`bootstrap_ci` / `paired_delta` from `eval_streamgen.py`. Success = `linear` seam/non-seam ratio
closer to `joint`'s than `block`'s is, with a CI that excludes zero on at least the latent jump
and beat continuity.

## Implementation

Guiding rule from AGENTS.md: core logic behind an explicit flag in `inference/`, default
`sample_diffusion` behaviour untouched, unit tests, one CLI entrypoint.

### 1. `stable_audio_3/inference/forcing.py` (new)

```
build_frame_schedule(base: (S+1,), delays: (N,) long) -> (S_total+1, N)
    # t[i, f] = base[clamp(i - delays[f], 0, S)]; S_total = S + delays.max()
delays_for(schedule: str, N, chunk_frames, prefix_frames, S, ramp=1.0) -> (N,) long
    # 'joint' | 'block' | 'linear' | 'pyramid'; prefix frames get a sentinel meaning "t = 0 always"
sample_euler_per_frame(model, x, sigmas: (B, S_total+1, N) or (S_total+1, N), callback=None,
                       disable_tqdm=False, **extra_args) -> x
    # t_curr = sigmas[:, i, :] (B, N); dt = (sigmas[:, i+1] - sigmas[:, i])[:, None, :]
    # v = model(x, t_curr, **extra_args); x = x + dt * v
    # step range is a slice so the caller can run one stage at a time and rebuild conditioning
```

**[decide]** separate function vs. a 3-D branch inside `sample_discrete_euler` (3.1 "Out of
scope" suggested the branch). Separate is the smaller blast radius and keeps the default sampler
identical byte-for-byte; the loop is ~15 lines either way. `sample_diffusion` is *not* extended —
the driver below calls the pieces directly, the way the demo callback and `eval_streamgen.py`
already do.

### 2. `scripts/sample_df_schedules.py` (new)

Structure mirrors `eval_streamgen.py`, and imports `load_arm`, `prepare_batch`-style cropping,
`onset_envelope`, `estimate_pulse`, `bootstrap_ci`, `paired_delta` from it rather than copying
(`tests/test_streamgen_eval_metrics.py` shows the `importlib` pattern for importing a script).

```
--model_config stable_audio_3/configs/model_configs/small_music_base_df.json
--ckpt /data/scratch-fast/snnithya/sao-3/ft_checkpoints/sao-3/7tvd98n0/checkpoints/last.ckpt
--dataset_config .../slakh_streamgen_validation_preencoded.json
--schedules joint block linear [pyramid]
--chunk_seconds 4.0  --prefix_frames 15  --steps 50  --cfg_scale 1.0
--context_feed both|t0|inpaint   --accomp causal|oracle   --lookahead_seconds 0.0
--ramp 1.0                          # linear: ramp length in chunks
--max_items 56  --batch_size 8  --seed 0
--out experiments/03-diffusion-forcing/results/3_3.json
--audio_dir /data/scratch-fast/snnithya/sao-3/outputs/3_3   # wavs are big; JSON + PNGs go in the repo
```

Per batch, per schedule:

1. crop/scale latents, build fixed noise (seeded per item), put ground truth into the prefix frames;
2. `delays = delays_for(...)`, `sigmas = build_frame_schedule(build_schedule(S, dist_shift=model.sampling_dist_shift, ...), delays)`;
3. loop over stages (every `S` steps): rebuild `inpaint_mask` / `inpaint_masked_input` /
   `tf_inpaint_mask` / gated `streamgen_latent` from the current cursor per `--context_feed` /
   `--accomp`, `cond_inputs = model.get_conditioning_inputs(...)`, run `sample_euler_per_frame`
   over that stage's step slice, advance the cursor over any chunk now at `t = 0`;
4. decode (`pretransform.decode`, bf16 as in the eval), write `{item:03d}_{schedule}_drums.wav`,
   `_mix.wav` (0.7 drums + 0.7 accompaniment), once per item `_reference_drums.wav` /
   `_accompaniment.wav`; save the schedule heatmap once per schedule;
5. metrics at seams and matched interior positions; splice control built from two seeds of the
   same item.

Output JSON: per-item records + per-schedule summaries + paired deltas, same shape as `1_2.json`.

### 3. Tests — `tests/test_diffusion_forcing_sampling.py`

CPU, stub model, no checkpoint:

- `build_frame_schedule`: every column non-increasing, starts at 1 (or 0 for prefix), ends at 0;
  `block` columns constant within a chunk and stepwise across chunks; `linear` strictly
  different between adjacent frames inside a chunk; `joint` with zero delays reproduces the
  base schedule in every column.
- `sample_euler_per_frame` with all-zero delays equals `sample_discrete_euler` on the 1-D
  schedule **bit-for-bit** for a stub `model(x, t) = −x · broadcast(t)` that accepts both ranks.
- Frozen frames: a frame at `t = 0` (prefix) is unchanged after the whole run, `torch.equal`.
- Cursor logic: after stage `k` exactly chunks `≤ k` are at `t = 0` under `block`; under `linear`
  the emitted chunk count lags by one.
- `delays_for` rejects `chunk_frames` that do not fit the window unless a partial last chunk is
  allowed.

### 4. Sbatch — `sbatch/03_3_df_schedules.sbatch`

1× L40S, 1 h is plenty: 7 batches × 3–4 schedules × ~200 forward passes at 144 frames. Copy the
`REPO=` / `latest_ckpt` pattern from `01_2_eval.sbatch`; **mind that `REPO` and `uv run` resolve
to the main checkout — set `PYTHONPATH` to the worktree or run after merging.**

### 5. Write-up

Results table + the four heatmaps + paired deltas into this file; README row 3.3 to "done" or
"negative". Keep the wavs on scratch and link the listening page.

## Confounds and caveats

- **The registers sit at `t̄ = 0.537` regardless of schedule** (3.1 decision). Schedule-
  independent by construction, so it cannot explain a *difference* between schedules.
- **`seconds_total` is hard-coded to 12 s** at train time (README note). Pass the same
  `seconds_total: 12` in the conditioning here, and pass `padding_mask` explicitly (all ones over
  144 frames) rather than letting `sample_diffusion`'s headroom logic derive it — the eval script
  already does this.
- **Distribution shift**: `use_effective_length_for_schedule: true`, so `build_schedule` shifts
  the base schedule by effective length. Applied once to the 1-D base, then delayed per frame —
  every frame follows the same shifted trajectory. (An alternative would shift each frame by its
  own "remaining horizon"; not this experiment.)
- **`t = 1` frames that have not started** are in-distribution for a model trained with iid
  per-frame `t`, but they are 100 frames of pure noise in the first stage of `block`. If `block`
  looks pathological on chunk 0, try masking not-yet-started chunks out with `padding_mask`
  instead (then the window length changes per stage — closer to how continuation runs today).
- **No EMA** in this fork; `last.ckpt` is the raw weights. Also score `epoch=48-step=5000` if
  `last` looks overfit (96 epochs over 1678 clips — 3.2 flagged it).
- **The DF model was trained at `df_p_global: 0.0`**, so `joint` (all frames equal `t`) is the
  *least* trained configuration for this checkpoint. If `joint` is not the coherence ceiling,
  that is a finding about the checkpoint, not a bug in the script — but check with a
  block-uniform run on the pretrained `small-music-base` via ordinary inpainting continuation
  as a sanity anchor (optional, the script can do it with `--schedules block --context_feed
  inpaint` against a non-DF config, since `block` under `inpaint` is 1-D-equivalent per stage).
- `interface/diffusion_cond.py:112` still reads `sigma[0].item()`; irrelevant to the script,
  but any Gradio toggle for these schedules has to fix it first.

## Out of scope

Rollout past 144 frames (sliding the window), causal attention (3.5), DF vs. inpainting as
mechanisms (3.4 — although `--context_feed t0` vs `inpaint` is its first data point), learned or
adaptive schedules, and any training change.
