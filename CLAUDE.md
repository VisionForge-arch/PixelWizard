# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PixelWizard is a two-stage video super-resolution pipeline built on Wan2.2-TI2V-5B:

1. **LR generation** — generates 240p video latents (50 sampling steps)
2. **SR upscaling** — upscales to 2K/4K using shortcut distillation (4-5 steps)
3. **Decode** — converts latents to video via `decode.py` (spatial chunking with overlap blending)

All inference stages run in a single `generate.py` script — LR latents are passed in memory to SR without writing intermediates to disk, then SR latents are decoded to videos.

## Key Files

- `generate.py` — Prompt → LR → SR → decode pipeline (saves SR `.pt` files and `.mp4` videos)
- `decode.py` — Batch VAE decode with spatial chunking and cosine-ramp overlap blending
- `dataset_upsample.py` — `UnifiedDataset` for loading latents/videos (used internally by SR stage)
- `wan/` — Core model library:
  - `textimage2video.py` — `WanTI2V` class (LR inference pipeline)
  - `textimage2video_sr_shortcut.py` — `WanTI2V_Upsample_Shortcut` class (SR inference pipeline)
  - `modules/model.py` — Base Wan DiT backbone
  - `modules/model_upsample_shortcut2.py` — SR shortcut model with spatial adapter + dt conditioning
  - `modules/vae2_2.py` — Wan2.2 VAE (encode/decode)
  - `modules/t5.py` — T5 text encoder
  - `configs/` — Model configs (ti2v-5B with resolution mappings)
  - `distributed/` — FSDP + Ulysses sequence parallelism

## Inference Commands

```bash
# LR → SR → decode generation (8 GPUs)
torchrun --nproc_per_node=8 generate.py \
    --ckpt_dir ./Wan2.2-TI2V-5B --sr_ckpt <sr_checkpoint> \
    --prompt_file prompts.txt --save_dir outputs/sr --video_dir outputs/videos \
    --sr_size 2560*1440 --sr_steps 4 \
    --dit_fsdp --t5_fsdp --ulysses_size 8
```

## Architecture Notes

- `generate.py` saves both SR latents and decoded videos
- SR uses **shortcut distillation**: a spatial adapter (3D CNN) injects LR features into the DiT backbone via forward hooks, with dt (step-size) conditioning for variable-step sampling
- `decode.py` splits high-res latents spatially into patches, decodes each independently, and blends overlap regions with cosine-ramp masks to avoid seam artifacts
- Distributed inference uses PyTorch FSDP + DeepSpeed Ulysses sequence parallelism
- All configs use flow matching with UniPC or DPM++ solvers

## Prompt Format

`generate.py` accepts a plain text file (one prompt per line) or JSONL format (`{id, text}` per line).

## Resolutions

| Stage | Resolution | Sampling Steps | Shift |
|-------|-----------|---------------|-------|
| LR base | 448x256 (240p) | 50 | 5.0 |
| SR 2K | 2560x1440 | 4 | 5.5 |
| SR 4K | 3840x2144 | 5 | 5.8 |
