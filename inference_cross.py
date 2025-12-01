import argparse
import torch
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
parser.add_argument("--config_path", type=str, default="./configs/self_forcing_dmd.yaml", help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_latent_up_2/checkpoint_model_009800/model.pt", help="Path to the checkpoint folder")
parser.add_argument("--prompt_file", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompt_to_file.json", help="JSON file with prompts/files for upsample inference")
parser.add_argument("--output_folder", type=str, default="/hpc2hdd/home/htian395/Wenxue/Self-Forcing-Long/outputs/", help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=42, help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", default=True, help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true", help="Whether to save the video using the index or prompt as the filename")
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
# recompute seq_len based on height/width/num_frames from config
time_part = int((config.num_frames - 1) // 4) + 1
config.seq_len = int(config.height // 32) * int(config.width // 32) * time_part

# Initialize pipeline
# Few-step inference
pipeline = CausalInferencePipeline(config, device=device)


# if args.checkpoint_path:
#     state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    
#     generator_state_dict = state_dict['generator' if not args.use_ema else 'generator_ema']
#     corrected_state_dict = {
#         key.replace("model._fsdp_wrapped_module.", "model."): value
#         for key, value in generator_state_dict.items()
#     }
#     pipeline.generator.load_state_dict(corrected_state_dict)

state_dict = torch.load(args.checkpoint_path, map_location="cpu")
generator_state_dict = state_dict['generator']
pipeline.generator.load_state_dict(generator_state_dict)



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
        max_pixels=832*480,
        height=480,
        width=832,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=config.num_frames,
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

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch["prompt"]
        video_input = batch["file"].to(device=device, dtype=torch.bfloat16)
        prompts = prompt
        
        print(prompt)
        print(prompts)
        
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
        
        print(video_input.shape)

        with torch.no_grad():
            cond_latent_lr = pipeline.vae.encode_to_latent(video_input).to(device=device, dtype=torch.bfloat16)  # [B,T,C,h',w']
            B, C, T, h, w = cond_latent_lr.shape
            H = 90
            W = 160
            print("cond_latent_lr.shape:", cond_latent_lr.shape)
            
            #cond_latent_lr = cond_latent_lr.permute(0, 2, 1, 3, 4)  
            cond_latent_lr = cond_latent_lr.reshape(B*T, C, h, w)  # [B*T, C, h, w]
            cond_latent_lr = F.interpolate(cond_latent_lr, size=(H, W), mode='bilinear', align_corners=False)  # 可加 antialias=True（若版本支持）
            cond_latent_lr = cond_latent_lr.reshape(B, T, C, H, W)  # [B*T, C, h, w]
            #print(cond_latent.shape) # [1, 48, 21, 90, 160]
            #cond_latent_lr = cond_latent_lr.to(device=device, dtype=torch.bfloat16)

    
    # Generate 81 frames
    video, latents = pipeline.inference(
        noise=sampled_noise,
        clean_latent_lr=cond_latent_lr,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
        
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()
    
    exit()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)
