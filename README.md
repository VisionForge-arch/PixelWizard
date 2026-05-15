<div align="center">

<img src="./teaser/logo.png" width="48%"/>

# PixelWizard: Towards Efficient High-Fidelity Video Generation at Ultra-Large Spatial Resolutions

<a href="#"><img src="https://img.shields.io/static/v1?label=Paper&message=Coming%20Soon&color=red"></a>
&ensp;
<a href="#"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Coming%20Soon&color=green"></a>
&ensp;
<a href="#"><img src="https://img.shields.io/static/v1?label=Checkpoints&message=Coming%20Soon&color=yellow"></a>
&ensp;
<a href="#"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue"></a>

</div>

PixelWizard is a high-resolution text-to-video generation framework for efficient 2K/4K video synthesis. It decouples global spatial-temporal structure modeling from high-resolution detail generation, then accelerates the expensive high-resolution stage with shortcut step-size conditioning.

> Paper, project page, checkpoints, and visual examples will be released after publication.

<div align="center">
<img src="./teaser/teaser.jpeg" width="95%"/>
</div>

## News

- **[2026.05]** Initial repository for PixelWizard.
- Project page, paper link, checkpoints, and demo videos are coming soon.

## Abstract

High-resolution video generation faces a coupled bottleneck of optimization instability and prohibitive computational cost. As spatial resolution increases, the token sequence expands dramatically, making optimization biased toward local textures while weakening global structural coherence. PixelWizard addresses this by hierarchically decoupling global structure modeling from fine-grained high-resolution synthesis. It first establishes a compact spatial-temporal anchor that captures motion and layout, then uses this anchor to guide high-resolution latent generation through an Anchor-Guided Injector. To reduce inference latency, PixelWizard further introduces Noise-Span Aligned Shortcut Training, enabling robust few-step generation at native 2K/4K resolutions without a memory-heavy teacher-student distillation pipeline.

## Method

PixelWizard consists of three main components:

### Spatial-Temporal Anchor Modeling

The first stage generates a compact low-resolution latent video at **448x256**. This anchor concentrates semantic and structural information in a dense latent grid, allowing the model to capture global layout, object structure, and motion patterns with substantially lower cost.

### Anchor-Guided High-Resolution Synthesis

The low-resolution anchor is interpolated to the target high-resolution latent grid and injected into the HR DiT backbone. The Anchor-Guided Injector uses lightweight 3D convolutions to refine the anchor features, then modulates their influence with timestep and shortcut step-size embeddings before adding them into selected DiT blocks.

### Noise-Span Aligned Shortcut Training

The HR model is conditioned on both the diffusion timestep `t` and the integration step size `dt`. This lets the model learn large-step transitions and run the high-resolution stage in only a few sampling steps, which is critical for practical 2K/4K generation.

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd PixelWizard
```

### 2. Set Up the Environment

```bash
# 1. Create and activate a clean environment.
conda create -n pixelwizard python=3.10
conda activate pixelwizard

# 2. Install PyTorch first. Choose the command matching your CUDA version.
# Example for CUDA 12.1:
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install the remaining Python dependencies.
pip install -r requirements.txt

# 4. Install flash-attn after PyTorch is available.
pip install flash-attn --no-build-isolation
```

### 3. Download Base Models

Download the Wan2.2-TI2V-5B base checkpoint:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

PixelWizard also requires a high-resolution shortcut checkpoint:

- `--ckpt_dir`: Wan2.2-TI2V-5B base checkpoint directory.
- `--lr_ckpt`: optional low-resolution anchor checkpoint. If omitted, the LR stage uses base Wan2.2 weights.
- `--hr_ckpt`: required PixelWizard HR shortcut checkpoint.

PixelWizard checkpoints will be released separately.

### 4. Prepare Prompts

Create a text file with one prompt per line:

```text
A Samoyed and a Golden Retriever dog are playfully romping through a futuristic neon city at night.
A sunny day, a pure white cat moves through a verdant garden with stately trees and vibrant flowers.
```

JSONL is also supported:

```jsonl
{"id": "sample_0001", "text": "A cinematic shot of a white cat walking through a garden."}
{"id": "sample_0002", "text": "Majestic snow-covered rocky mountain peaks tower over a canyon."}
```

### 5. Run Inference

2K generation:

```bash
torchrun --nproc_per_node=8 generate.py \
    --ckpt_dir ./Wan2.2-TI2V-5B \
    --lr_ckpt <lr_anchor_checkpoint> \
    --hr_ckpt <pixelwizard_hr_2k_checkpoint> \
    --prompt_file prompts.txt \
    --video_dir outputs/videos_2k \
    --resolution 2k \
    --dit_fsdp --t5_fsdp --ulysses_size 8
```

4K generation:

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

By default, `generate.py` does **not** save HR latent `.pt` files. To save HR latents for later decoding or debugging, pass `--save_dir`:

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

`generate.py` processes prompts one by one: LR anchor latent -> HR latent -> decoded video, then moves to the next prompt. The default `--model_load_mode auto` keeps models resident with CPU offload for single-process runs and reloads LR/HR models per prompt for distributed runs to reduce peak memory.

Decode options:

- `--num_patches`: number of spatial chunks for HR VAE decode.
- `--patch_dim`: decode split dimension, `w` by default.
- `--overlap`: latent-space overlap between chunks, blended with a cosine ramp.
- `--vae_path`: optional path to the Wan2.2 VAE checkpoint. If omitted, the VAE under `--ckpt_dir` is used.

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

## Citation

If PixelWizard is useful for your research, please cite our paper. BibTeX will be updated after publication.

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
