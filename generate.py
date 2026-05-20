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
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.utils.utils import str2bool


LR_SIZE = "448*256"

RESOLUTION_CONFIGS = {
    "2k": {
        "hr_size": "2560*1440",
        "hr_steps": 4,
        "hr_shift": 5.5,
        "decode_num_patches": 3,
    },
    "4k": {
        "hr_size": "3840*2144",
        "hr_steps": 4,
        "hr_shift": 5.8,
        "decode_num_patches": 4,
    },
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="PixelWizard Ultra-High Resolution Video Generation Pipeline")
    # --- shared ---
    parser.add_argument("--task", type=str, default="ti2v-5B",
                        choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to Wan2.2-TI2V-5B checkpoint directory")
    parser.add_argument("--frame_num", type=int, default=121,
                        help="Number of frames (should be 4n+1)")
    parser.add_argument("--prompt_file", type=str, default="prompts/demos.txt",
                        help="Prompt file: .txt (one per line) or .jsonl ({id, text})")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Optional directory to save output high resolution .pt latent files")
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Directory to save decoded .mp4 videos (default: sibling videos/ directory when --save_dir is set, otherwise outputs/videos)")
    parser.add_argument("--resolution", type=str, default="2k",
                        choices=list(RESOLUTION_CONFIGS.keys()),
                        help="Output resolution preset. 2k = 2560x1440, 4k = 3840x2144")
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true")
    parser.add_argument("--dit_fsdp", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--model_load_mode", type=str, default="auto",
                        choices=["auto", "resident", "reload"],
                        help="auto: resident+offload on single process, reload per prompt for distributed")
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
    parser.add_argument("--hr_ckpt", type=str, required=True,
                        help="Path to Stage II   checkpoint (.pt)")
    parser.add_argument("--hr_guide_scale", type=float, default=None)
    parser.add_argument("--hr_solver", type=str, default='unipc',
                        choices=['unipc', 'dpm++'])
    parser.add_argument("--sample_solver", type=str, default='unipc',
                        choices=['unipc', 'dpm++'],
                        help="sampling solver")
    # --- decode ---
    parser.add_argument("--vae_path", type=str, default=None,
                        help="Path to Wan2.2 VAE checkpoint (.pth). Defaults to ckpt_dir/config VAE")
    parser.add_argument("--num_patches", type=int, default=None,
                        help="Number of spatial patches for chunked decoding (default: 2k=3, 4k=4)")
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
    args.hr_size = preset["hr_size"]
    args.hr_steps = preset["hr_steps"]
    args.hr_shift = preset["hr_shift"]
    if args.num_patches is None:
        args.num_patches = preset["decode_num_patches"]
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
    """Interpolate a LR latent to the target HR conditioning size.

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


def _latent_to_cpu(latent):
    if isinstance(latent, torch.Tensor):
        return latent.cpu()
    if isinstance(latent, (list, tuple)):
        return [x.cpu() if isinstance(x, torch.Tensor) else x for x in latent]
    return latent


def _clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _offload_unused_t2v_vae(model):
    """The generation path only needs VAE shape metadata, not VAE weights."""
    vae = getattr(model, "vae", None)
    if vae is None:
        return
    if hasattr(vae, "model"):
        vae.model.cpu()
    if hasattr(vae, "scale"):
        vae.scale = [
            x.cpu() if isinstance(x, torch.Tensor) else x
            for x in vae.scale
        ]
    vae.device = torch.device("cpu")
    _clear_cuda()


def _build_lr_model(args, lr_cfg, device, rank):
    model = wan.WanTI2V(
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
    _offload_unused_t2v_vae(model)
    return model


def _build_hr_model(args, cfg, device, rank):
    model = wan.WanTI2V_HR(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        wan_ckpt=args.hr_ckpt,
    )
    _offload_unused_t2v_vae(model)
    return model


def _decode_one(latent_input, output_path, vae_path, args):
    vae = Wan2_2_VAE(vae_pth=vae_path, device=args.decode_device)
    try:
        decode_latent_gpu_chunked(
            latent_input,
            output_path,
            vae,
            num_patches=args.num_patches,
            device=args.decode_device,
            patch_dim=args.patch_dim,
            overlap=args.overlap,
        )
    finally:
        del vae
        _clear_cuda()


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

    lr_cfg = WAN_CONFIGS[args.task]
    lr_size = SIZE_CONFIGS[args.lr_size]
    lr_shift = args.lr_shift if args.lr_shift is not None else lr_cfg.sample_shift
    lr_guide_scale = args.lr_guide_scale if args.lr_guide_scale is not None else lr_cfg.sample_guide_scale
    hr_size = SIZE_CONFIGS[args.hr_size]
    W_target, H_target = hr_size
    H_lr = H_target // 32
    W_lr = W_target // 32
    hr_guide_scale = args.hr_guide_scale if args.hr_guide_scale is not None else cfg.sample_guide_scale

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    if rank == 0:
        video_dir = args.video_dir
        if video_dir is None:
            if args.save_dir is not None:
                save_parent = os.path.dirname(os.path.abspath(args.save_dir))
                video_dir = os.path.join(save_parent, "videos")
            else:
                video_dir = os.path.join("outputs", "videos")
        os.makedirs(video_dir, exist_ok=True)
        vae_path = args.vae_path
        if vae_path is None:
            vae_path = os.path.join(args.ckpt_dir, cfg.vae_checkpoint)
    else:
        video_dir = None
        vae_path = None

    lr_prompts = _load_prompts(args.prompt_file)
    reload_models = (
        args.model_load_mode == "reload"
        or (args.model_load_mode == "auto" and world_size > 1)
    )

    logging.info(
        f"Resolution preset: {args.resolution} "
        f"(LR={args.lr_size}, HR={args.hr_size}, "
        f"HR steps={args.hr_steps}, HR shift={args.hr_shift})")
    logging.info(f"Decode patches: {args.num_patches} along {args.patch_dim}")
    logging.info(f"LR checkpoint: {args.lr_ckpt or 'base weights from ckpt_dir'}")
    logging.info(f"HR checkpoint: {args.hr_ckpt}")
    logging.info(f"HR latent saving: {args.save_dir or 'disabled'}")
    logging.info(f"Loaded {len(lr_prompts)} prompts from {args.prompt_file}")
    logging.info(
        "Pipeline mode: one prompt at a time "
        f"({'reload models per prompt' if reload_models else 'resident models with offload'})")

    if not reload_models and args.offload_model is False:
        logging.warning(
            "resident mode with --offload_model False keeps LR/HR models on GPU during decode; "
            "use --offload_model True if GPU memory is tight.")

    lr_model = None
    hr_model = None
    if not reload_models:
        logging.info("Loading resident LR model (WanTI2V)...")
        lr_model = _build_lr_model(args, lr_cfg, device, rank)
        logging.info("Loading resident HR model (WanTI2V_Upsample_Shortcut)...")
        hr_model = _build_hr_model(args, cfg, device, rank)

    try:
        for prompt_idx, prompt_obj in enumerate(lr_prompts, 1):
            prompt = prompt_obj['text']
            output_idx = prompt_idx - 1
            if rank == 0:
                logging.info("=" * 72)
                logging.info(
                    f"Prompt {prompt_idx}/{len(lr_prompts)}: {prompt[:100]}...")

            if reload_models:
                logging.info("Loading LR model (WanTI2V)...")
                lr_model = _build_lr_model(args, lr_cfg, device, rank)

            lr_seed = args.base_seed + prompt_idx * 100
            if dist.is_initialized():
                seed_list = [lr_seed] if rank == 0 else [None]
                dist.broadcast_object_list(seed_list, src=0)
                lr_seed = seed_list[0]

            logging.info(f"Phase 1/3: generating LR latent for prompt {prompt_idx}")
            lr_latent = lr_model.generate(
                prompt,
                size=lr_size,
                frame_num=args.frame_num,
                shift=lr_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.lr_steps,
                guide_scale=lr_guide_scale,
                seed=lr_seed,
                offload_model=args.offload_model)

            if isinstance(lr_latent, (list, tuple)):
                lr_latent = lr_latent[0]
            lr_latent = lr_latent.cpu()

            if reload_models:
                del lr_model
                lr_model = None
                _clear_cuda()
                if dist.is_initialized():
                    dist.barrier()

            if reload_models:
                logging.info("Loading HR model (WanTI2V_Upsample_Shortcut)...")
                hr_model = _build_hr_model(args, cfg, device, rank)

            hr_seed = args.base_seed + output_idx * 100 + 10000
            if dist.is_initialized():
                seed_list = [hr_seed] if rank == 0 else [None]
                dist.broadcast_object_list(seed_list, src=0)
                hr_seed = seed_list[0]

            logging.info(f"Phase 2/3: generating HR latent for prompt {prompt_idx}")
            cond_latent = _interpolate_cond_latent(lr_latent, H_lr, W_lr)
            cond_latent = cond_latent.to(device=hr_model.device, dtype=torch.float32)

            hr_latent = hr_model.generate(
                prompt,
                cond_latent=cond_latent,
                size=hr_size,
                frame_num=args.frame_num,
                shift=args.hr_shift,
                sample_solver=args.hr_solver,
                sampling_steps=args.hr_steps,
                guide_scale=hr_guide_scale,
                seed=hr_seed,
                offload_model=args.offload_model)

            if rank == 0:
                latent_data = {
                    'latent': _latent_to_cpu(hr_latent),
                    'prompt': prompt,
                    'prompt_id': prompt_obj.get('id', str(output_idx)),
                    'lr_seed': lr_seed,
                    'seed': hr_seed,
                    'size': args.hr_size,
                    'frame_num': args.frame_num,
                }
                if args.save_dir is not None:
                    save_path = os.path.join(args.save_dir, f"{output_idx}.pt")
                    torch.save(latent_data, save_path)
                    logging.info(f"Saved HR latent: {save_path}")
                else:
                    save_path = None
            else:
                latent_data = None
                save_path = None

            del lr_latent, hr_latent, cond_latent
            _clear_cuda()

            if reload_models:
                del hr_model
                hr_model = None
                _clear_cuda()

            if dist.is_initialized():
                dist.barrier()

            if rank == 0:
                if save_path is not None:
                    output_filename = os.path.basename(save_path).replace('.pt', '.mp4')
                    decode_input = save_path
                else:
                    output_filename = f"{output_idx}.mp4"
                    decode_input = latent_data
                output_path = os.path.join(video_dir, output_filename)
                logging.info(f"Phase 3/3: decoding video to {output_path}")
                _decode_one(decode_input, output_path, vae_path, args)

            if dist.is_initialized():
                dist.barrier()
    finally:
        if lr_model is not None:
            del lr_model
        if hr_model is not None:
            del hr_model
        _clear_cuda()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
