import argparse
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline_long import BidirectionalDiffusionInferencePipeline2
from dataset_image import UIEBD_Dataset_Wrapper
from utils_long.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="/hpc2hdd/home/htian395/Wenxue/Underwater/Underwater_Video_UIE/configs/self_forcing_dmd.yaml", help="Path to the config file")
#parser.add_argument("--checkpoint_path", type=str, default="/hpc2hdd/home/htian395/Wenxue/Self-Forcing/checkpoints/self_forcing_dmd.pt", help="Path to the checkpoint folder")
parser.add_argument("--checkpoint_path", type=str, default="/hpc2hdd/home/htian395/Wenxue/Underwater/Underwater_Video_UIE/logs/self_forcing_dmd/checkpoint_model_002000/model.pt", help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, default="/hpc2ssd/JH_DATA/spooler/htian395/C-underwater/0-dataset-split/UIED-no-rename/test/raw-890", help="Path to the dataset")
parser.add_argument("--gt_path", type=str, default="/hpc2ssd/JH_DATA/spooler/htian395/C-underwater/0-dataset-split/UIED-no-rename/test/reference-890", help="Path to the dataset")
parser.add_argument("--output_folder", type=str, default="/hpc2hdd/home/htian395/Wenxue/Underwater/Underwater_Video_UIE/outputs/", help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=1, help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", default=True, help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
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
default_config = OmegaConf.load("/hpc2hdd/home/htian395/Wenxue/Underwater/Underwater_Video_UIE/configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
# Few-step inference
pipeline = BidirectionalDiffusionInferencePipeline2(config, device=device)


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
dataset = UIEBD_Dataset_Wrapper(args.data_path, args.gt_path, 'test')
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")


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


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data[2]

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames


    prompt = ""  # Get caption from batch
    prompts = [prompt] * args.num_samples

    # Process the image
    image = batch[0].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

    # Encode the input image as the first latent
    initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
    initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

    sampled_noise = torch.randn(
        [args.num_samples, args.num_output_frames, 48, 14, 14], device=device, dtype=torch.bfloat16
    )
    
    

    # Generate 81 frames
    video, latents = pipeline.inference(
        input_image=initial_latent,
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt

            # All processes save their videos
            
    name = idx[0].split(".")[0]
    output_path = os.path.join(args.output_folder, f'{name}.mp4')

    #write_video(output_path, video[0], fps=16)
    write_video(output_path, video[0], fps=16)
