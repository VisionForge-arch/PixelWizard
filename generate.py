# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
End-to-end pipeline: generates low-resolution latents (stage 1), immediately
upscales them to 2K/4K (stage 2), and decodes the SR latents to videos.

Usage:
    python generate.py --ckpt_dir /mnt/vision-gen-ks3/ModelZoo/Video_Generation/Wan2.2-TI2V-5B \
        --lr_ckpt /mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_240p_new/checkpoint_model_001800/model.pt \
        --sr_ckpt /mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/2k_shortcut_new2/checkpoint_model_001300/model.pt \
        --save_dir /mnt/nas01-ak/IndividualDirs/wenxueli/test_github/2k_pt \
        --video_dir /mnt/nas01-ak/IndividualDirs/wenxueli/test_github/2k_mp4 \
        --resolution 2k
    
    python generate.py --ckpt_dir /mnt/vision-gen-ks3/ModelZoo/Video_Generation/Wan2.2-TI2V-5B \
        --lr_ckpt /mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_240p_new/checkpoint_model_001800/model.pt \
        --sr_ckpt /mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/4k_shortcut_new2/checkpoint_model_001150/model.pt \
        --save_dir /mnt/nas01-ak/IndividualDirs/wenxueli/test_github/4k_pt \
        --video_dir /mnt/nas01-ak/IndividualDirs/wenxueli/test_github/4k_mp4 \
        --resolution 4k

    torchrun --nproc_per_node=8 generate.py --ckpt_dir ./Wan2.2-TI2V-5B \
        --lr_ckpt <lr_checkpoint> --sr_ckpt <sr_checkpoint> \
        --prompt_file prompts.txt --save_dir outputs/sr \
        --video_dir outputs/videos --resolution 2k
"""
import argparse
import gc
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import random
import torch
import torch.distributed as dist
import torch.nn.functional as F

import wan
from decode import decode_latent_gpu_chunked
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.utils.utils import str2bool


LR_SIZE = "448*256"

RESOLUTION_CONFIGS = {
    "2k": {
        "sr_size": "2560*1440",
        "sr_steps": 4,
        "sr_shift": 5.5,
    },
    "4k": {
        "sr_size": "3840*2144",
        "sr_steps": 5,
        "sr_shift": 5.8,
    },
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Unified LR → SR video generation pipeline")
    # --- shared ---
    parser.add_argument("--task", type=str, default="ti2v-5B",
                        choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to Wan2.2-TI2V-5B checkpoint directory")
    parser.add_argument("--frame_num", type=int, default=121,
                        help="Number of frames (should be 4n+1)")
    parser.add_argument("--prompt_file", type=str, default="prompts/demos.txt",
                        help="Prompt file: .txt (one per line) or .jsonl ({id, text})")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save output SR .pt files")
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Directory to save decoded .mp4 videos (default: sibling videos/ directory)")
    parser.add_argument("--resolution", type=str, default="2k",
                        choices=list(RESOLUTION_CONFIGS.keys()),
                        help="Output resolution preset. 2k = 2560x1440, 4k = 3840x2144")
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true")
    parser.add_argument("--dit_fsdp", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--base_seed", type=int, default=0,
                        help="Random seed (-1 for random)")
    # --- stage I ---
    parser.add_argument("--lr_steps", type=int, default=50,
                        help="LR diffusion sampling steps")
    parser.add_argument("--lr_shift", type=float, default=None,
                        help="LR flow matching shift (None = use config default)")
    parser.add_argument("--lr_guide_scale", type=float, default=None)
    parser.add_argument("--lr_ckpt", type=str, default=None,
                        help="Optional fine-tuned LR checkpoint (.pt). Omit to use ckpt_dir base weights")
    # --- stage II ---
    parser.add_argument("--sr_ckpt", type=str, required=True,
                        help="Path to Stage II   checkpoint (.pt)")
    parser.add_argument("--sr_guide_scale", type=float, default=None)
    parser.add_argument("--sr_solver", type=str, default='unipc',
                        choices=['unipc', 'dpm++'])
    parser.add_argument("--sample_solver", type=str, default='unipc',
                        choices=['unipc', 'dpm++'],
                        help="LR sampling solver")
    # --- decode ---
    parser.add_argument("--vae_path", type=str, default=None,
                        help="Path to Wan2.2 VAE checkpoint (.pth). Defaults to ckpt_dir/config VAE")
    parser.add_argument("--num_patches", type=int, default=3,
                        help="Number of spatial patches for chunked decoding")
    parser.add_argument("--patch_dim", type=str, default="w",
                        choices=['h', 'w'],
                        help="Dimension to split during decode")
    parser.add_argument("--overlap", type=int, default=3,
                        help="Latent-space overlap pixels between decode patches")
    parser.add_argument("--decode_device", type=str, default="cuda",
                        help="Device for VAE decoding")

    args = parser.parse_args()
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    preset = RESOLUTION_CONFIGS[args.resolution]
    args.lr_size = LR_SIZE
    args.sr_size = preset["sr_size"]
    args.sr_steps = preset["sr_steps"]
    args.sr_shift = preset["sr_shift"]
    return args


def _load_prompts(prompt_file):
    """Load prompts from .txt (one per line) or .jsonl ({id, text})."""
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}. "
            "Create a text file with one prompt per line, or pass --prompt_file /path/to/prompts.txt."
        )

    prompts = []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)

        if first_line.startswith('{'):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt_id = obj.get('id', obj.get('prompt_id', ''))
                text = obj.get('text', obj.get('prompt', obj.get('caption', '')))
                prompts.append({'id': str(prompt_id), 'text': text})
        else:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    prompts.append({'id': str(i), 'text': line})
    if not prompts:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return prompts

def _interpolate_cond_latent(latent, H_lr, W_lr):
    """Interpolate a LR latent to the target SR conditioning size.

    Args:
        latent: shape (C, T, h, w) or (B, C, T, h, w)
        H_lr, W_lr: target spatial dimensions in latent space
    Returns:
        Tensor of shape (1, C, T, H_lr, W_lr)
    """
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)  # (1, C, T, h, w)
    B, C, T, h, w = latent.shape
    x = latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, h, w)
    x = F.interpolate(x, size=(H_lr, W_lr), mode='bilinear', align_corners=False)
    x = x.reshape(B, T, C, H_lr, W_lr).permute(0, 2, 1, 3, 4)
    return x


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://",
            rank=rank, world_size=world_size)
    else:
        assert not args.t5_fsdp, "t5_fsdp requires distributed"
        assert not args.dit_fsdp, "dit_fsdp requires distributed"
        assert args.ulysses_size == 1, "ulysses_size > 1 requires distributed"

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    os.makedirs(args.save_dir, exist_ok=True)
    logging.info(
        f"Resolution preset: {args.resolution} "
        f"(LR={args.lr_size}, SR={args.sr_size}, "
        f"SR steps={args.sr_steps}, SR shift={args.sr_shift})")
    logging.info(f"LR checkpoint: {args.lr_ckpt or 'base weights from ckpt_dir'}")
    logging.info(f"SR checkpoint: {args.sr_ckpt}")

    # ================================================================
    # Phase 1: LR latent generation
    # ================================================================
    lr_cfg = WAN_CONFIGS[args.task]
    lr_size = SIZE_CONFIGS[args.lr_size]

    lr_shift = args.lr_shift if args.lr_shift is not None else lr_cfg.sample_shift
    lr_guide_scale = args.lr_guide_scale if args.lr_guide_scale is not None else lr_cfg.sample_guide_scale

    lr_prompts = _load_prompts(args.prompt_file)
    logging.info(f"Loaded {len(lr_prompts)} prompts from {args.prompt_file}")

    logging.info("Phase 1: Loading LR model (WanTI2V)...")
    lr_model = wan.WanTI2V(
        config=lr_cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        wan_ckpt=args.lr_ckpt,
    )

    logging.info("Phase 1: Generating LR latents...")
    lr_latents = []
    for prompt_idx, prompt_obj in enumerate(lr_prompts, 1):
        if rank == 0:
            logging.info(
                f"  LR {prompt_idx}/{len(lr_prompts)}: {prompt_obj['text'][:80]}...")

        current_seed = args.base_seed + prompt_idx * 100
        if dist.is_initialized():
            seed_list = [current_seed] if rank == 0 else [None]
            dist.broadcast_object_list(seed_list, src=0)
            current_seed = seed_list[0]

        latent = lr_model.generate(
            prompt_obj['text'],
            size=lr_size,
            max_area=MAX_AREA_CONFIGS[args.lr_size],
            frame_num=args.frame_num,
            shift=lr_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.lr_steps,
            guide_scale=lr_guide_scale,
            seed=current_seed,
            offload_model=args.offload_model)

        # Every rank keeps the latent so distributed SR executes the same loop.
        if isinstance(latent, (list, tuple)):
            latent = latent[0]
        lr_latents.append((prompt_obj['text'], latent.cpu()))
        del latent

    del lr_model
    torch.cuda.empty_cache()
    gc.collect()
    if dist.is_initialized():
        dist.barrier()

    logging.info(f"Phase 1: Generated {len(lr_latents)} LR latents.")

    # ================================================================
    # Phase 2: SR upscaling
    # ================================================================
    sr_size = SIZE_CONFIGS[args.sr_size]
    W_target, H_target = sr_size
    H_lr = H_target // 32
    W_lr = W_target // 32

    logging.info("Phase 2: Loading SR model (WanTI2V_Upsample_Shortcut)...")
    sr_model = wan.WanTI2V_Upsample_Shortcut(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        wan_ckpt=args.sr_ckpt,
    )

    sr_guide_scale = args.sr_guide_scale if args.sr_guide_scale is not None else cfg.sample_guide_scale

    logging.info("Phase 2: Generating SR latents...")
    saved_sr_paths = []
    for idx, (prompt, lr_latent) in enumerate(lr_latents):
        if rank == 0:
            logging.info(
                f"  SR {idx + 1}/{len(lr_latents)}: {prompt[:80]}...")

        current_seed = args.base_seed + idx * 100 + 10000
        if dist.is_initialized():
            seed_list = [current_seed] if rank == 0 else [None]
            dist.broadcast_object_list(seed_list, src=0)
            current_seed = seed_list[0]

        cond_latent = _interpolate_cond_latent(lr_latent, H_lr, W_lr)
        cond_latent = cond_latent.to(device=sr_model.device, dtype=torch.float32)

        sr_latent = sr_model.generate(
            prompt,
            cond_latent=cond_latent,
            size=sr_size,
            max_area=MAX_AREA_CONFIGS[args.sr_size],
            frame_num=args.frame_num,
            shift=args.sr_shift,
            sample_solver=args.sr_solver,
            sampling_steps=args.sr_steps,
            guide_scale=sr_guide_scale,
            seed=current_seed,
            offload_model=args.offload_model)

        if rank == 0:
            save_path = os.path.join(args.save_dir, f"{idx}.pt")
            torch.save({
                'latent': sr_latent,
                'prompt': prompt,
                'seed': current_seed,
                'size': args.sr_size,
                'frame_num': args.frame_num,
            }, save_path)
            saved_sr_paths.append(save_path)
            logging.info(f"  Saved: {save_path}")

        del sr_latent, cond_latent
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    del sr_model
    torch.cuda.empty_cache()
    gc.collect()

    # ================================================================
    # Phase 3: Decode SR latents to videos
    # ================================================================
    if rank == 0:
        video_dir = args.video_dir
        if video_dir is None:
            save_parent = os.path.dirname(os.path.abspath(args.save_dir))
            video_dir = os.path.join(save_parent, "videos")
        os.makedirs(video_dir, exist_ok=True)

        vae_path = args.vae_path
        if vae_path is None:
            vae_path = os.path.join(args.ckpt_dir, cfg.vae_checkpoint)

        logging.info("Phase 3: Loading VAE for decode...")
        vae = Wan2_2_VAE(vae_pth=vae_path, device=args.decode_device)

        logging.info("Phase 3: Decoding SR latents to videos...")
        for idx, sr_path in enumerate(saved_sr_paths, 1):
            output_filename = os.path.basename(sr_path).replace('.pt', '.mp4')
            output_path = os.path.join(video_dir, output_filename)
            logging.info(
                f"  Decode {idx}/{len(saved_sr_paths)}: {os.path.basename(output_path)}")
            decode_latent_gpu_chunked(
                sr_path,
                output_path,
                vae,
                num_patches=args.num_patches,
                device=args.decode_device,
                patch_dim=args.patch_dim,
                overlap=args.overlap,
            )

        del vae
        torch.cuda.empty_cache()
        gc.collect()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
