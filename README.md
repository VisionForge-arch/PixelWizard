# PixelWizard

**Towards Efficient High-Fidelity Video Generation at Ultra-Large Spatial Resolutions**

Official repository for **PixelWizard**, a framework for efficient high-resolution text-to-video generation at 2K and 4K spatial resolutions. PixelWizard decouples global structure modeling from high-resolution detail synthesis: it first builds a compact spatial-temporal anchor, then uses the anchor to guide high-resolution latent generation with a shortcut-trained Wan DiT backbone.

> Paper, project page, checkpoints, and examples will be released after publication.

## Highlights

- **Ultra-large spatial resolution video generation**: native 2K and 4K video synthesis in latent space.
- **Spatial-Temporal Anchor Modeling**: generates global motion and layout at a compact 448x256 latent regime.
- **Anchor-Guided High-Resolution Synthesis**: injects the anchor into the HR DiT backbone through a lightweight spatial control adapter.
- **Noise-Span Aligned Shortcut Training**: models shortcut step size `dt` for stable few-step high-resolution inference.
- **Efficient inference**: the HR stage uses only 4 sampling steps by default for both 2K and 4K presets.

## Method Overview

PixelWizard addresses the optimization and efficiency bottlenecks of high-resolution video generation with a two-stage pipeline:

1. **Stage I: Spatial-Temporal Anchor Modeling**
   A Wan2.2-TI2V-5B model generates a compact low-resolution latent video at 448x256. This stage focuses on global structure, object layout, and motion consistency.

2. **Stage II: Anchor-Guided High-Resolution Synthesis**
   The LR anchor latent is interpolated to the HR latent grid and passed to a spatial adapter. The adapter uses 3D convolutions plus timestep and step-size conditioning to produce control features, which are injected into the HR DiT blocks.

3. **Shortcut HR Sampling**
   The HR model is conditioned on both diffusion timestep `t` and shortcut step size `dt`, enabling large denoising steps and few-step inference without a heavy teacher-student distillation pipeline.

4. **Chunked VAE Decode**
   HR latents are decoded to video with spatial chunking and cosine-ramp overlap blending to reduce memory usage and avoid patch boundary artifacts.

## Repository Structure

```text
generate.py                 # End-to-end prompt -> anchor -> HR latent -> video pipeline
decode.py                   # Chunked Wan VAE latent decoder
dataset_upsample.py         # Unified latent/video dataset utilities
wan/
  textimage2video.py        # LR Wan TI2V pipeline
  textimage2video_HR.py     # HR anchor-guided shortcut pipeline
  modules/hr_model.py       # HR DiT, spatial adapter, dt conditioning, hooks
  modules/model.py          # Base Wan DiT backbone
  modules/vae2_2.py         # Wan2.2 VAE
  modules/t5.py             # T5 text encoder
  configs/                  # Wan2.2 TI2V configs and resolution mappings
  distributed/              # FSDP and sequence parallel utilities
```

## Installation

```bash
git clone <this-repository-url>
cd PixelWizard

# PyTorch >= 2.4.0 is recommended.
pip install -r requirements.txt

# If flash-attn fails during the main install, install other packages first, then:
pip install flash-attn --no-build-isolation
```

## Checkpoints

Download the Wan2.2-TI2V-5B base checkpoint:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

PixelWizard also requires the HR shortcut checkpoint for the target resolution. The PixelWizard checkpoints are not included in this repository and will be released separately.

Expected checkpoint inputs:

- `--ckpt_dir`: Wan2.2-TI2V-5B base checkpoint directory.
- `--lr_ckpt`: optional LR anchor-model checkpoint. If omitted, the LR stage uses base Wan2.2 weights from `--ckpt_dir`.
- `--hr_ckpt`: required PixelWizard HR shortcut checkpoint.

## Inference

Prepare a prompt file with one prompt per line:

```text
A Samoyed and a Golden Retriever dog are playfully romping through a futuristic neon city at night.
A sunny day, a pure white cat moves through a verdant garden with stately trees and vibrant flowers.
```

Run the full LR anchor -> HR synthesis -> decode pipeline:

```bash
torchrun --nproc_per_node=8 generate.py \
    --ckpt_dir ./Wan2.2-TI2V-5B \
    --lr_ckpt <lr_anchor_checkpoint> \
    --hr_ckpt <pixelwizard_hr_checkpoint> \
    --prompt_file prompts.txt \
    --video_dir outputs/videos_2k \
    --resolution 2k \
    --dit_fsdp --t5_fsdp --ulysses_size 8
```

For 4K generation, use the 4K HR checkpoint and switch the preset:

```bash
torchrun --nproc_per_node=8 generate.py \
    --ckpt_dir ./Wan2.2-TI2V-5B \
    --lr_ckpt <lr_anchor_checkpoint> \
    --hr_ckpt <pixelwizard_hr_4k_checkpoint> \
    --prompt_file prompts.txt \
    --video_dir outputs/videos_4k \
    --resolution 4k \
    --dit_fsdp --t5_fsdp --ulysses_size 8
```

By default, `generate.py` does **not** save HR latent `.pt` files. To keep HR latents for later decoding or debugging, pass `--save_dir`:

```bash
python generate.py \
    --ckpt_dir ./Wan2.2-TI2V-5B \
    --hr_ckpt <pixelwizard_hr_checkpoint> \
    --prompt_file prompts.txt \
    --video_dir outputs/videos \
    --save_dir outputs/hr_latents \
    --resolution 2k
```

## Resolution Presets

| Preset | Anchor Resolution | HR Resolution | HR Steps | Shift | Decode Patches |
|--------|-------------------|---------------|----------|-------|----------------|
| `2k` | 448x256 | 2560x1440 | 4 | 5.5 | 3 |
| `4k` | 448x256 | 3840x2144 | 4 | 5.8 | 4 |

`generate.py` processes prompts one at a time: LR anchor latent -> HR latent -> decoded video, then moves to the next prompt. The default `--model_load_mode auto` keeps models resident with CPU offload in single-process runs and reloads LR/HR models per prompt in distributed runs to reduce peak memory.

Decode options:

- `--num_patches`: number of spatial chunks for HR VAE decode.
- `--patch_dim`: decode split dimension, `w` by default.
- `--overlap`: latent-space overlap between chunks; overlap is blended with a cosine ramp.
- `--vae_path`: optional path to the Wan2.2 VAE checkpoint. If omitted, `generate.py` uses the VAE under `--ckpt_dir`.

## Prompt Format

`generate.py` supports either plain text or JSONL prompt files.

Plain text:

```text
Prompt one
Prompt two
```

JSONL:

```jsonl
{"id": "sample_0001", "text": "Prompt one"}
{"id": "sample_0002", "text": "Prompt two"}
```

## Citation

If PixelWizard is useful for your research, please cite our paper. BibTeX will be added after publication.

```bibtex
@article{pixelwizard2026,
  title   = {PixelWizard: Towards Efficient High-Fidelity Video Generation at Ultra-Large Spatial Resolutions},
  author  = {Anonymous Authors},
  journal = {ACM Transactions on Graphics},
  year    = {2026}
}
```

## Acknowledgements

PixelWizard is built on [Wan2.2](https://github.com/Wan-Video/Wan2.2). We thank the Wan team for releasing their open video generation models and infrastructure.

## License

This project follows the license terms of the released code and the underlying Wan2.2 components. Please also check the license of the base Wan2.2 checkpoint before use.
