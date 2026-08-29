# AGENTS.md — Interactive music research on Stable Audio 3

This repo extends [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) for **interactive music making**: finetuning with structured controls, experimenting with CFG variants at inference, and (planned) **diffusion forcing** for planning / horizon control and beyond.

**Default stance for agents:** optimize for **fast research iteration** — small, reviewable diffs; wire experiments behind flags or config; prefer extending existing hooks over new abstractions. Do not refactor upstream-style code unless the task requires it. Ask questions before making big assumptions. 

---

## Project goals (edit as scope evolves)

| Area | Intent |
|------|--------|
| **Finetuning** | Adapt diffusion (+ optionally conditioner) to datasets with **extra controls** (e.g. instrument/stem metadata, track IDs) via `custom_metadata` and training wrappers. |
| **CFG / guidance** | Try **vanilla CFG**, **APG**, **CFG rescale**, **CFG++** samplers, batched vs double-forward, and training-time `cfg_dropout` — keep train/infer paths consistent. |
| **Diffusion forcing** | *(Planned)* Autoregressive or block-wise denoising over latent time for **planning** and interactive edits. Document design decisions here when implemented. |
| **Interactive UX** | Gradio (`run_gradio.py`) and/or custom interfaces; latency and listenable demos matter as much as loss curves. |

Document experiments in experiments/. Each main experiment should have a hypothesis and can contain multiple sub-experiments structured in a hierarchical structure.

---

## Environment

```bash
# Base
uv sync

# Gradio UI
uv sync --extra ui

# Training (LoRA script; full finetune uses same stack)
uv sync --extra lora

# Flash Attention (pre-built wheel; keeps flash-attn across future syncs)
uv sync --extra flash
```

- **Python entrypoints:** `uv run python …` or `uv run stable-audio …`
- **GPU:** Always run on GPU.
- **Default research model:** `small-music` unless the user specifies otherwise (post-trained `medium` uses different default `cfg_scale` / `steps`).

<!-- TODO: fill in your machine(s), CUDA version, typical data roots -->
- **Data roots:**: /data/hai-res/shared/snnithya/sat-zenon-data/babyslakh-preencoded/
- **Checkpoint / log dirs:**: /data/scratch-fast/snnithya/sao-3/

---

## Code map (where to change what)

| Task | Start here |
|------|------------|
| **Training loop** | `stable_audio_3/training/diffusion.py` (`DiffusionCondTrainingWrapper`) |
| **Full finetune CLI** | `scripts/train_finetune.py` |
| **LoRA CLI** | `scripts/train_lora.py`, `docs/workflows/lora.md` |
| **Dataset + metadata** | `stable_audio_3/data/dataset.py`, JSON under `stable_audio_3/configs/dataset_configs/`, `custom_metadata/*.py` |
| **Pre-encode latents** | `scripts/pre_encode_dataset.py` |
| **Sanity-check a pre-encoded dataset** | `scripts/pre_encode_dataset.py --sanity_check_samples N`, `scripts/decode_preencoded_samples.py`, `scripts/make_listening_page.py`, `scripts/check_streamgen_alignment.py` |
| **DiT / CFG in forward** | `stable_audio_3/models/dit.py` (`cfg_scale`, `apg_scale`, `scale_phi`, `cfg_interval`, …) |
| **Sampling / schedulers** | `stable_audio_3/inference/sampling.py` (`sample_diffusion`, sampler types) |
| **High-level infer API** | `stable_audio_3/model.py` (`StableAudioModel.generate`, …) |
| **Gradio controls** | `stable_audio_3/interface/diffusion_cond.py`, launch via `run_gradio.py` |
| **Conditioning** | `stable_audio_3/models/conditioners.py` |
| **Inpainting / masks** | `stable_audio_3/models/inpainting.py`, training `inpainting_config` in finetune script |
| **Upstream docs** | `docs/workflows/inference.md`, `docs/guides/model-overview.md` |

**Research-specific today:** Slakh-style metadata in `stable_audio_3/configs/dataset_configs/custom_metadata/custom_md_slakh.py` and `scripts/train_finetune.py` with `local_babyslakh_preencoded.json` (adjust paths in JSON for your machine).

---

## Iteration workflows

When implementing, agents should:

1. Log hypothesis, notes, method and results in the relavant experiment/sub-experiment file. 
2. Implement core logic in `inference/sampling.py` (or a sibling `forcing.py`) behind an explicit flag — avoid changing default `sample_diffusion` behavior until validated.
3. Write unit tests for each feature.
4. Expose one Gradio or CLI toggle for interactive testing.

### D. Interactive demo loop

```bash
uv run python run_gradio.py --model small-music  # add --lora_ckpt_path if needed
```

After training changes, verify: load weights → generate 5–10s clip → inpaint/continuation if relevant.

---

## Conventions for agents

- **Minimal diff:** one experiment per PR-sized change; use argparse flags or gin-style config over hardcoding.
- **Match existing style:** PyTorch Lightning in training; typed hints where the file already uses them; same naming as `dit.py` / `sampling.py`.
- **Train/infer parity:** if training uses `mask_padding_attention` / `use_effective_length_for_schedule`, inference must pass the same `padding_mask` semantics (see `sample_diffusion` docstring).
- **CFG defaults:** `small-sao` → `cfg_scale` often ~7 at inference; post-trained `small-sao` defaults to `1`. State which checkpoint you assume.
- **Do not commit:** `.env`, API keys, large checkpoints, or personal dataset paths (use JSON configs with local paths).
- **Tests:** run targeted tests when touching core paths: `uv run pytest tests/test_inference.py -q` (add tests only when behavior is stable and user wants them).
- **Commits:** only when the user asks.

---

## Experiment logging (fill in)

- **Naming:** `<!-- e.g. {date}_{idea}_{dataset}_{cfg} -->`
- **WandB / Comet:** `<!-- project, entity -->`
- **What to log every run:** loss, demo audio, `cfg_scale`, `steps`, `sampler_type`, `seed`, dataset id, git sha, config file.