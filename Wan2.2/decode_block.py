#!/usr/bin/env python
"""
单独decode latent的脚本
用法: python decode_latent.py --latent_path latent_20241016_123456.pt --output output.mp4
"""
import argparse
import torch
import gc
from wan.modules.vae2_2 import Wan2_2_VAE, unpatchify


def decode_latent_gpu_chunked(latent_path, output_path, vae_path="checkpoint/vae", 
                                num_patches=2, device='cuda', patch_dim='h'):
    """在GPU上分patch decode latent
    
    Args:
        num_patches: 分成几个patch，例如2表示分成2块
        patch_dim: 在哪个维度分割，'h'表示高度维度，'w'表示宽度维度
    """
    
    # 加载latent
    print(f"加载latent: {latent_path}")
    data = torch.load(latent_path, map_location='cpu')
    latent = data['latent']
    prompt = data.get('prompt', 'unknown')
    print(f"Prompt: {prompt}")
    print(f"Latent shape: {latent[0].shape}")
    
    # 初始化VAE
    print("初始化VAE...")
    vae = Wan2_2_VAE(
        vae_pth=vae_path,
        device=device
    )
    
    # 分patch decode
    print(f"分成 {num_patches} 个patch进行decode...")
    decoded_patches = []
    
    for i in range(num_patches):
        print(f"处理patch {i+1}/{num_patches}...")
        
        # 分割latent
        if patch_dim == 'h':
            # 在高度维度分割 [C, T, H, W]
            patch_latents = []
            for l in latent:
                h_dim = l.shape[2]
                h_start = h_dim // num_patches * i
                h_end = h_dim // num_patches * (i + 1) if i < num_patches - 1 else h_dim
                patch_latents.append(l[:, :, h_start:h_end, :].to(device))
        else:  # patch_dim == 'w'
            # 在宽度维度分割
            patch_latents = []
            for l in latent:
                w_dim = l.shape[3]
                w_start = w_dim // num_patches * i
                w_end = w_dim // num_patches * (i + 1) if i < num_patches - 1 else w_dim
                patch_latents.append(l[:, :, :, w_start:w_end].to(device))
        
        # decode这个patch
        patch_decoded = vae.decode(patch_latents)
        if patch_decoded is not None:
            if isinstance(patch_decoded, (list, tuple)):
                decoded_patches.append(patch_decoded[0].cpu())
            else:
                decoded_patches.append(patch_decoded.cpu())
        
        # 清理显存
        del patch_latents, patch_decoded
        torch.cuda.empty_cache()
        gc.collect()
    
    # 拼接所有patches
    print("拼接所有patches...")
    if patch_dim == 'h':
        # 在高度维度拼接 [C, T, H, W]
        final_video = torch.cat(decoded_patches, dim=2)
    else:  # patch_dim == 'w'
        # 在宽度维度拼接
        final_video = torch.cat(decoded_patches, dim=3)
    
    # 保存视频
    save_video(final_video, output_path)
    print(f"视频已保存到: {output_path}")

def save_video(video, output_path):
    """保存视频tensor为mp4文件"""
    import numpy as np
    import imageio
    
    # video shape: (3, T, H, W), 范围 [-1, 1]
    video = video.permute(1, 2, 3, 0)  # (T, H, W, 3)
    video = ((video + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
    
    imageio.mimwrite(output_path, video, fps=24, quality=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_path", type=str, default="/hpc2hdd/home/htian395/Wenxue/Wan2.2/ti2v-5B_1920*1056_1_Close-up_of_an_Asian_man_with_a_hopeful_expression_20251023_122841.pt")
    parser.add_argument("--output", type=str, default="/hpc2hdd/home/htian395/Wenxue/Wan2.2/ti2v-5B_1920*1056_1_Close-up_of_an_Asian_man_with_a_hopeful_expression_20251023_122841.mp4")
    parser.add_argument("--vae_path", type=str, default="/hpc2ssd/JH_DATA/spooler/htian395/Wenxue/Weight/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    parser.add_argument("--num_patches", type=int, default=2, help="分成几个patch进行decode，默认2表示分一半一半")
    parser.add_argument("--patch_dim", type=str, default="h", choices=['h', 'w'], help="在哪个维度分割，h=高度，w=宽度")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    decode_latent_gpu_chunked(args.latent_path, args.output, args.vae_path, 
                                args.num_patches, args.device, args.patch_dim)