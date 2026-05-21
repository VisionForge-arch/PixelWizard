#!/usr/bin/env python
"""
Batch decode latent .pt files to video .mp4 files.
Supports spatial chunking with overlap blending to avoid OOM and seam artifacts.

Usage:
    python decode.py --input_dir <latent_dir> --output_dir <video_dir> --vae_path <vae_checkpoint>
"""
import argparse
import math
import torch
import gc
import os
import glob
from wan.modules.vae2_2 import Wan2_2_VAE, unpatchify


def build_spatial_blend_mask(
    patch_h,
    patch_w,
    *,
    overlap_h=0,
    overlap_w=0,
    is_bound=(False, False, False, False),
    device="cpu",
    dtype=torch.float32,
):
    """Build a 2D cosine-ramp blending mask for smooth overlap transitions."""

    def cosine_ramp(length):
        if length <= 0:
            return torch.empty((0,), device=device, dtype=dtype)
        if length == 1:
            return torch.zeros((1,), device=device, dtype=dtype)
        t = torch.linspace(0, 1, steps=length, device=device, dtype=dtype)
        return 0.5 - 0.5 * torch.cos(math.pi * t)

    top, bottom, left, right = is_bound

    patch_h = int(patch_h)
    patch_w = int(patch_w)
    overlap_h = int(min(max(overlap_h, 0), patch_h))
    overlap_w = int(min(max(overlap_w, 0), patch_w))

    mask_h = torch.ones((patch_h,), device=device, dtype=dtype)
    mask_w = torch.ones((patch_w,), device=device, dtype=dtype)

    if overlap_h > 0:
        ramp = cosine_ramp(overlap_h)
        if not top:
            mask_h[:overlap_h] *= ramp
        if not bottom:
            mask_h[-overlap_h:] *= ramp.flip(0)

    if overlap_w > 0:
        ramp = cosine_ramp(overlap_w)
        if not left:
            mask_w[:overlap_w] *= ramp
        if not right:
            mask_w[-overlap_w:] *= ramp.flip(0)

    mask = mask_h[:, None] * mask_w[None, :]
    return mask.unsqueeze(0).unsqueeze(0)


def decode_latent_gpu_chunked(
    latent_input,
    output_path,
    vae,
    num_patches=2,
    device='cuda',
    patch_dim='h',
    overlap=3,
):
    """Decode a latent to video, splitting spatially into patches with overlap blending.

    Args:
        latent_input: Path to .pt latent file, or an in-memory latent data dict
        output_path: Path to save .mp4 video
        vae: Wan2_2_VAE instance
        num_patches: Number of spatial patches to split into
        device: Device for VAE decoding
        patch_dim: 'h' for height split, 'w' for width split
        overlap: Number of latent-space overlap pixels between patches
    """
    if isinstance(latent_input, (str, os.PathLike)):
        print(f"Loading latent: {latent_input}")
        data = torch.load(latent_input, map_location='cpu')
    elif isinstance(latent_input, dict):
        print("Loading latent: in-memory")
        data = latent_input
    else:
        raise TypeError("latent_input must be a path or latent data dict")

    latent = data['latent']
    prompt = data.get('prompt', 'unknown')
    print(f"Prompt: {prompt}")
    print(f"Latent shape: {latent[0].shape}")

    C_lat, T_lat, H_lat, W_lat = latent[0].shape

    final_video = None
    weight = None
    scale_h = None
    scale_w = None

    if patch_dim == 'h':
        base_size = H_lat // num_patches
    else:
        base_size = W_lat // num_patches

    print(f"Decoding in {num_patches} patches, overlap={overlap} pixels...")

    for i in range(num_patches):
        print(f"Processing patch {i+1}/{num_patches}...")

        if patch_dim == 'h':
            h_start_base = base_size * i
            h_end_base = H_lat if i == num_patches - 1 else base_size * (i + 1)
            h_start = max(0, h_start_base - (overlap if i > 0 else 0))
            h_end = min(H_lat, h_end_base + (overlap if i < num_patches - 1 else 0))

            patch_latents = []
            for l in latent:
                patch_latents.append(l[:, :, h_start:h_end, :].to(device))
        else:
            w_start_base = base_size * i
            w_end_base = W_lat if i == num_patches - 1 else base_size * (i + 1)
            w_start = max(0, w_start_base - (overlap if i > 0 else 0))
            w_end = min(W_lat, w_end_base + (overlap if i < num_patches - 1 else 0))

            patch_latents = []
            for l in latent:
                patch_latents.append(l[:, :, :, w_start:w_end].to(device))

        patch_decoded = vae.decode(patch_latents)
        if isinstance(patch_decoded, (list, tuple)):
            patch_decoded = patch_decoded[0]
        patch_decoded = patch_decoded.detach().cpu()

        if final_video is None:
            C_dec, T_dec, H_dec, W_dec = patch_decoded.shape

            if patch_dim == 'h':
                patch_lat_h = h_end - h_start
                scale_h = H_dec // patch_lat_h
                scale_w = W_dec // W_lat
            else:
                patch_lat_w = w_end - w_start
                scale_w = W_dec // patch_lat_w
                scale_h = H_dec // H_lat

            H_out = H_lat * scale_h
            W_out = W_lat * scale_w

            final_video = torch.zeros((C_dec, T_dec, H_out, W_out), dtype=torch.float32)
            weight = torch.zeros((1, 1, H_out, W_out), dtype=torch.float32)

        if patch_dim == 'h':
            out_h_start = h_start * scale_h
            out_h_end = h_end * scale_h
            ph = patch_decoded.shape[2]
            assert ph == (out_h_end - out_h_start), "patch height mismatch"
            mask = build_spatial_blend_mask(
                ph, patch_decoded.shape[3],
                overlap_h=overlap * scale_h, overlap_w=0,
                is_bound=(i == 0, i == num_patches - 1, True, True),
                device=final_video.device, dtype=final_video.dtype,
            )
            patch_decoded = patch_decoded.to(dtype=final_video.dtype)
            final_video[:, :, out_h_start:out_h_end, :] += patch_decoded * mask
            weight[:, :, out_h_start:out_h_end, :] += mask
        else:
            out_w_start = w_start * scale_w
            out_w_end = w_end * scale_w
            pw = patch_decoded.shape[3]
            assert pw == (out_w_end - out_w_start), "patch width mismatch"
            mask = build_spatial_blend_mask(
                patch_decoded.shape[2], pw,
                overlap_h=0, overlap_w=overlap * scale_w,
                is_bound=(True, True, i == 0, i == num_patches - 1),
                device=final_video.device, dtype=final_video.dtype,
            )
            patch_decoded = patch_decoded.to(dtype=final_video.dtype)
            final_video[:, :, :, out_w_start:out_w_end] += patch_decoded * mask
            weight[:, :, :, out_w_start:out_w_end] += mask

        del patch_latents, patch_decoded
        torch.cuda.empty_cache()
        gc.collect()

    weight = weight.clamp_min(1e-6)
    final_video = final_video / weight

    save_video(final_video, output_path)
    print(f"Video saved to: {output_path}")

    return prompt, final_video.shape


def save_video(video, save_path, fps=24):
    """Save video tensor as mp4 file."""
    import numpy as np
    import imageio

    video = video.permute(1, 2, 3, 0)  # (T, H, W, 3)
    video = ((video + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with imageio.get_writer(save_path, fps=fps, quality=8) as w:
        for f in video:
            w.append_data(np.array(f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch decode latent .pt files to video .mp4 files")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing .pt latent files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save decoded .mp4 videos")
    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path to Wan2.2 VAE checkpoint (.pth)")
    parser.add_argument("--num_patches", type=int, default=3,
                        help="Number of spatial patches for chunked decoding")
    parser.add_argument("--patch_dim", type=str, default="w",
                        choices=['h', 'w'],
                        help="Dimension to split: 'h' for height, 'w' for width")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overlap", type=int, default=3,
                        help="Overlap pixels between patches in latent space")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pt")))
    print(f"Found {len(pt_files)} .pt files")

    print("Initializing VAE...")
    vae = Wan2_2_VAE(vae_pth=args.vae_path, device=args.device)

    for idx, pt_file in enumerate(pt_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing {idx}/{len(pt_files)}: {os.path.basename(pt_file)}")
        print(f"{'='*60}")

        try:
            output_filename = os.path.basename(pt_file).replace('.pt', '.mp4')
            output_path = os.path.join(args.output_dir, output_filename)

            if os.path.exists(output_path):
                print(f"Output exists, skipping: {output_path}")
                continue

            decode_latent_gpu_chunked(
                pt_file, output_path, vae,
                args.num_patches, args.device,
                args.patch_dim, args.overlap,
            )

        except Exception as e:
            print(f"Error processing {pt_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("All files processed!")
    print(f"{'='*60}")
