<div align="center">

<img src="./teaser/logo.png" width="48%"/>

# PixelWizard: Towards Efficient High-Fidelity Video Generation at Ultra-Large Spatial Resolutions

<a href="http://arxiv.org/abs/2605.25801"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red"></a>
&ensp;
<a href="https://wxliii.github.io/pixelwizard/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Page&color=green"></a>
&ensp;
<a href="https://huggingface.co/wxli318/PixelWizard"><img src="https://img.shields.io/static/v1?label=Checkpoints&message=HuggingFace&color=yellow"></a>
&ensp;
<a href="#"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue"></a>

</div>

PixelWizard is a high-resolution text-to-video generation framework for efficient 2K/4K video synthesis. It decouples global spatial-temporal structure modeling from high-resolution detail generation, then accelerates the expensive high-resolution stage with shortcut step-size conditioning.


<div align="center">
<img src="./teaser/teaser.jpeg" width="95%"/>
</div>

## News

- **[2026.05]** Initial repository for PixelWizard.
- Project page, paper link, checkpoints, and demo videos are coming soon.



## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/VisionForge-arch/PixelWizard
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

### 3. Download Weights

Put all model weights under `./weight`:

```text
weight/
  Wan2.2-TI2V-5B/
  PixelWizard/
    lr/model.pt
    2k/model.pt
    4k/model.pt
```

Download the Wan2.2-TI2V-5B base checkpoint:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./weight/Wan2.2-TI2V-5B
```

Download the PixelWizard checkpoints and place them under `./weight/PixelWizard`:

```bash
huggingface-cli download wxli318/PixelWizard --local-dir ./weight/PixelWizard
```

- `--ckpt_dir`: Wan2.2-TI2V-5B base checkpoint directory, for example `./weight/Wan2.2-TI2V-5B`.
- `--lr_ckpt`: optional low-resolution anchor checkpoint, for example `./weight/PixelWizard/lr/model.pt`. If omitted, the LR stage uses base Wan2.2 weights.
- `--hr_ckpt`: required PixelWizard HR shortcut checkpoint, for example `./weight/PixelWizard/4k/model.pt`.

TODO:

- Upload the 2K PixelWizard HR shortcut checkpoint to [wxli318/PixelWizard](https://huggingface.co/wxli318/PixelWizard).

### 4. Run Inference

Single-GPU generation:

```bash
python generate.py \
    --ckpt_dir ./weight/Wan2.2-TI2V-5B \
    --lr_ckpt ./weight/PixelWizard/lr/model.pt \
    --hr_ckpt ./weight/PixelWizard/<resolution>/model.pt \
    --prompt_file prompts.txt \
    --video_dir outputs/videos \
    --resolution <2k_or_4k>
```

For single-GPU inference, expect approximately **52 GB VRAM** for 2K generation and **100 GB VRAM** for 4K generation.

Distributed generation:

```bash
torchrun --standalone --nproc_per_node=<n_gpus> generate.py \
    --ckpt_dir ./weight/Wan2.2-TI2V-5B \
    --lr_ckpt ./weight/PixelWizard/lr/model.pt \
    --hr_ckpt ./weight/PixelWizard/<resolution>/model.pt \
    --prompt_file prompts.txt \
    --video_dir outputs/videos \
    --resolution <2k_or_4k> \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size <n_gpus>
```

Set `<resolution>` to `2k` or `4k`. Distributed inference uses FSDP/Ulysses for multi-GPU memory sharding. Set `<n_gpus>` to the number of GPUs in the job. The pipeline still processes prompts one by one rather than distributing different prompts across GPUs.

By default, `generate.py` does **not** save HR latent `.pt` files. To save HR latents for later decoding or debugging, pass `--save_dir outputs/hr_latents`.

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

## Citation

If PixelWizard is useful for your research, please cite our paper. BibTeX will be updated after publication.

```bibtex
@misc{pixelwizard,
  title   = {PixelWizard: Towards Efficient High-Fidelity Video Generation at Ultra-Large Spatial Resolutions},
  author  = {Li, Wenxue and Ren, Jingjing and Zhang, Peng and Ye, Tian and Zhou, Daiguo and Luan, Jian and Zhu, Lei},
  year    = {2026}
}
```

## Acknowledgements

PixelWizard is built on [Wan2.2](https://github.com/Wan-Video/Wan2.2). We thank the Wan team for releasing their open video generation models and infrastructure.
