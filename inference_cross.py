import argparse
import torch
import os
import json
import logging
from datetime import datetime
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from pipeline_long import (
    CausalInferencePipeline,
)
#from dataset_text import TextDataset_json
from utils_long.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from dataset_upsample import UnifiedDataset

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="./configs/self_forcing_causal.yaml", help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/2k_causal/checkpoint_model_001500/model.pt", help="Path to the checkpoint folder")
#parser.add_argument("--checkpoint_path", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_latent_up_2_time_modulation2/checkpoint_model_000800/model.pt", help="Path to the checkpoint folder")
parser.add_argument("--prompt_file", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/prompt_to_file_240p.json", help="JSON file with prompts/files for upsample inference")
parser.add_argument("--output_folder", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/ar_upsample/pt", help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=31, help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true", help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--height", type=int, default=1440)
parser.add_argument("--width", type=int, default=2560)
parser.add_argument("--sampling_steps", type=int, default=5)
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("./configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
# set SR/causal flags to match training
config.sr_mode = True
config.causal = True
config.kv_cache_height = int(args.height // 32)
config.kv_cache_width = int(args.width // 32)
config.sampling_steps = args.sampling_steps


def _align_noise_frames(
    requested_frames: int,
    frames_per_block: int,
    independent_first_frame: bool,
    has_initial_latent: bool,
) -> int:
    if frames_per_block <= 0:
        raise ValueError(f"frames_per_block must be positive, got {frames_per_block}")
    if requested_frames <= 0:
        return 0
    if independent_first_frame and not has_initial_latent:
        return 1 + ((requested_frames - 1) // frames_per_block) * frames_per_block
    return (requested_frames // frames_per_block) * frames_per_block


# No padding: drop leftover frames to satisfy the causal block constraint.
requested_noise_frames = args.num_output_frames
aligned_noise_frames = _align_noise_frames(
    requested_frames=requested_noise_frames,
    frames_per_block=config.num_frame_per_block,
    independent_first_frame=bool(getattr(config, "independent_first_frame", False)),
    has_initial_latent=bool(args.i2v),
)
if aligned_noise_frames <= 0:
    raise ValueError(
        f"--num_output_frames={requested_noise_frames} cannot satisfy "
        f"num_frame_per_block={config.num_frame_per_block} without padding; "
        "increase --num_output_frames or change num_frame_per_block."
    )
if aligned_noise_frames != requested_noise_frames:
    print(
        f"[no-pad] Truncating requested noise frames {requested_noise_frames} -> {aligned_noise_frames} "
        f"to satisfy num_frame_per_block={config.num_frame_per_block}."
    )

# KV cache needs to cover the (aligned) length; keep +1 for safety/headroom.
config.kv_cache_time = aligned_noise_frames + 1

# recompute seq_len based on cache grid and (aligned) time
config.seq_len = int(args.height // 32) * int(args.width // 32) * aligned_noise_frames

# Initialize pipeline
# Few-step inference
pipeline = CausalInferencePipeline(config, device=device)
pipeline.generator.eval()

# if args.checkpoint_path:
#     state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    
#     generator_state_dict = state_dict['generator' if not args.use_ema else 'generator_ema']
#     corrected_state_dict = {
#         key.replace("model._fsdp_wrapped_module.", "model."): value
#         for key, value in generator_state_dict.items()
#     }
#     pipeline.generator.load_state_dict(corrected_state_dict)


# ----------------------------------------------------
#                      加载模型权重    
# ----------------------------------------------------
state_dict = torch.load(args.checkpoint_path, map_location="cpu")
if args.use_ema:
    print("------- Using EMA Weight ------")
    generator_state_dict = state_dict['generator_ema']
else:
    print("------- Using None EMA Weight ------")
    generator_state_dict = state_dict['generator']
    
# 先去掉 FSDP 产生的中间前缀
def clean_fsdp_keys(d):
    new = {}
    for k, v in d.items():
        k2 = (k.replace("_fsdp_wrapped_module.", "")
                .replace("_checkpoint_wrapped_module.", "")
                .replace("_orig_mod.", ""))
        new[k2] = v
    return new
generator_state_dict = clean_fsdp_keys(generator_state_dict)

# 分离 adapter 的权重
adapter_keys = [k for k in generator_state_dict.keys() if k.startswith('spatial_adapter.') or k.startswith('model.spatial_adapter.')]
adapter_state_dict = {}
for k in adapter_keys:
    raw_key = k[len('model.spatial_adapter.'):] if k.startswith('model.spatial_adapter.') else k[len('spatial_adapter.'):]
    adapter_state_dict[raw_key] = generator_state_dict[k]
print(f"Found {len(adapter_keys)} adapter keys in checkpoint")

# 加载主模型权重
missing_keys, unexpected_keys = pipeline.generator.load_state_dict(generator_state_dict, strict=False)
print(f"Missing keys: {len(missing_keys)}, unexpected: {len(unexpected_keys)}")

# 单独加载 adapter 权重，避免缺失
adapter_missing, adapter_unexpected = pipeline.generator.model.spatial_adapter.load_state_dict(adapter_state_dict, strict=True)
print(f"Adapter missing: {len(adapter_missing)}, unexpected: {len(adapter_unexpected)}")




pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
dataset = UnifiedDataset(
    base_path=None,
    metadata_path=args.prompt_file,
    repeat=1,
    data_file_keys=("file",),
    main_data_operator=UnifiedDataset.default_video_operator(
        base_path=None,
        max_pixels=448*256,
        height=256,
        width=448,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=121,
        time_division_factor=4,
        time_division_remainder=1,
    ),
)
print("len(dataset):", len(dataset))
num_prompts = len(dataset)


if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()




for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    #idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    cond_latent_lr = None
    initial_latent = None
    H = int(config.height // 16)
    W = int(config.width // 16)

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch["prompt"]
        video_input = batch["file"].to(device=device, dtype=torch.bfloat16)
        prompts = prompt

        
        initial_latent = None
        
        print(video_input.shape)

        with torch.no_grad():
            cond_latent_lr = pipeline.vae.encode_to_latent(video_input).to(device=device, dtype=torch.bfloat16)  # [B,T,C,h',w']
            B, T, C, h, w = cond_latent_lr.shape
            
            print("cond_latent_lr.shape:", cond_latent_lr.shape)
            
            #cond_latent_lr = cond_latent_lr.permute(0, 2, 1, 3, 4)  
            cond_latent_lr = cond_latent_lr.reshape(B*T, C, h, w)  # [B*T, C, h, w]
            cond_latent_lr = F.interpolate(cond_latent_lr, size=(H, W), mode='bilinear', align_corners=False)  # 可加 antialias=True（若版本支持）
            cond_latent_lr = cond_latent_lr.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)    # [B, C, T, h, w]
            #print(cond_latent.shape) # [1, 48, 21, 90, 160]
            #cond_latent_lr = cond_latent_lr.to(device=device, dtype=torch.bfloat16)

    
    # No padding: drop leftover frames to satisfy the block constraint.
    target_frames = aligned_noise_frames
    has_initial_latent = initial_latent is not None
    independent_first_frame = bool(getattr(config, "independent_first_frame", False))
    if cond_latent_lr is not None:
        available_frames = int(cond_latent_lr.shape[2])
        target_frames = min(target_frames, available_frames)
        target_frames = _align_noise_frames(
            requested_frames=target_frames,
            frames_per_block=config.num_frame_per_block,
            independent_first_frame=independent_first_frame,
            has_initial_latent=has_initial_latent,
        )
        if target_frames <= 0:
            print(
                f"[no-pad] Skip sample {i}: available_frames={available_frames} cannot satisfy "
                f"num_frame_per_block={config.num_frame_per_block} without padding."
            )
            continue
        cond_latent_lr = cond_latent_lr[:, :, :target_frames].contiguous()

    # 噪声也用 target_frames (already aligned)
    sampled_noise = torch.randn([args.num_samples, target_frames, 48, H, W], device=device, dtype=torch.bfloat16)
    print("sampled_noise.shape:", sampled_noise.shape)
        
    # Generate 81 frames
    latents = pipeline.inference(
        noise=sampled_noise,
        clean_latent_lr=cond_latent_lr,
        text_prompts=prompts,
        return_latents=False,
        initial_latent=initial_latent,
        low_memory=low_memory,
        
    )
    print(latents.shape)

    # No padding: output already matches target_frames (+ any initial_latent frames).
    num_input_frames = 0 if initial_latent is None else initial_latent.shape[1]
    target_total_frames = target_frames + num_input_frames
    latents = latents[:, :target_total_frames].contiguous()

    # Align latent format with Wan2.2_Cross generate_multiple_upsample: list of [C, T, H, W]
    latents_to_save = []
    for sample_idx in range(latents.shape[0]):
        latents_to_save.append(
            latents[sample_idx]
            .permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W]
            .contiguous()
            .to(dtype=torch.float32, device="cpu")
        )

    prompt_text = prompts[0] if isinstance(prompts, (list, tuple)) else prompts
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    formatted_prompt = str(prompt_text).replace(" ", "_").replace("/", "_")[:50]
    file_name = f"latent_{i}_{formatted_prompt}_{formatted_time}.pt"
    output_path = os.path.join(args.output_folder, file_name)

    print(f"Saving latent to {output_path}")
    torch.save(
        {
            "latent": latents_to_save,
            "prompt": prompt_text,
            "seed": args.seed,
            "frame_num": target_total_frames,
            "num_samples": args.num_samples,
            "size": f"{config.height}x{config.width}",
        },
        output_path,
    )
    print("Latent saved successfully.")

    if dist.is_initialized():
        dist.barrier()
