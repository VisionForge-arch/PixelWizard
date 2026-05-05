# PixelWizard

A two-stage video super-resolution pipeline built on [Wan2.2-TI2V-5B](https://github.com/Wan-Video/Wan2.2). Generates low-resolution video latents in stage 1, then upscales to 2K/4K in stage 2 using shortcut distillation (only 4-5 sampling steps).

## Pipeline

```
prompts.txt → generate.py → sr_latents/*.pt + videos/*.mp4
```

## Installation

```bash
# Ensure torch >= 2.4.0
pip install -r requirements.txt

# If flash_attn fails, install other packages first then:
pip install flash-attn --no-build-isolation
```

## Model Download

Download the Wan2.2-TI2V-5B base checkpoint:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

You also need a fine-tuned SR checkpoint (not included in this repo).

## Usage

Prepare a text file with one prompt per line:

```
A cat sitting on a sofa
Sunset over the ocean
```

Run the full pipeline (LR generation + SR upscaling + decode):

```bash
# 2K output
torchrun --nproc_per_node=8 generate.py \
    --ckpt_dir ./Wan2.2-TI2V-5B \
    --lr_ckpt <lr_checkpoint> \
    --sr_ckpt <sr_2k_checkpoint> \
    --prompt_file prompts.txt \
    --save_dir outputs/sr_2k \
    --video_dir outputs/videos_2k \
    --resolution 2k \
    --dit_fsdp --t5_fsdp --ulysses_size 8

# 4K output
torchrun --nproc_per_node=8 generate.py \
    --ckpt_dir ./Wan2.2-TI2V-5B \
    --lr_ckpt <lr_checkpoint> \
    --sr_ckpt <sr_4k_checkpoint> \
    --prompt_file prompts.txt \
    --save_dir outputs/sr_4k \
    --video_dir outputs/videos_4k \
    --resolution 4k \
    --dit_fsdp --t5_fsdp --ulysses_size 8
```

`--resolution 2k` uses fixed LR `448x256`, SR `2560x1440`, 4 SR steps, and shift `5.5`. `--resolution 4k` uses fixed LR `448x256`, SR `3840x2144`, 5 SR steps, and shift `5.8`.

By default, `generate.py` uses `./Wan2.2-TI2V-5B/Wan2.2_VAE.pth` for decode. Override it with `--vae_path` if your VAE checkpoint lives elsewhere.

`--lr_ckpt` is optional. If omitted, LR generation uses the base Wan2.2 weights from `--ckpt_dir`; pass it when you have a fine-tuned LR checkpoint.

`generate.py` processes prompts one by one: LR latent → SR latent → decoded video, then moves to the next prompt. The default `--model_load_mode auto` keeps models resident with CPU offload on single-process runs, and reloads LR/SR per prompt for distributed runs to avoid holding both FSDP model stacks at once. You can force either behavior with `--model_load_mode resident` or `--model_load_mode reload`.

The `--num_patches`, `--patch_dim`, and `--overlap` options control spatial chunking during decode to avoid OOM on high-resolution latents. The overlap region uses cosine-ramp blending to avoid seam artifacts.

## Resolutions

| Stage | Resolution | Sampling Steps | Shift |
|-------|-----------|---------------|-------|
| LR base | 448x256 (240p) | 50 | 5.0 |
| SR 2K | 2560x1440 | 4 | 5.5 |
| SR 4K | 3840x2144 | 5 | 5.8 |

## Acknowledgements

Built on [Wan2.2](https://github.com/Wan-Video/Wan2.2) by the Alibaba Wan Team. Licensed under Apache 2.0.
