#!/usr/bin/env python
"""
批量decode latent的脚本
用法: python decode_list.py
"""
import argparse
import torch
import gc
import os
import glob
from wan.modules.vae2_2 import Wan2_2_VAE, unpatchify


def decode_latent_gpu_chunked(latent_path, output_path, vae, 
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
    
    return prompt, final_video.shape

def save_video(video, output_path):
    """保存视频tensor为mp4文件"""
    import numpy as np
    import imageio
    
    # video shape: (3, T, H, W), 范围 [-1, 1]
    video = video.permute(1, 2, 3, 0)  # (T, H, W, 3)
    video = ((video + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
    
    imageio.mimwrite(output_path, video, fps=24, quality=8)

def generate_output_filename(prompt, resolution, timestamp):
    """根据prompt和分辨率生成输出文件名
    
    Args:
        prompt: 提示词
        resolution: (height, width) 分辨率元组
        timestamp: 时间戳
    
    Returns:
        文件名字符串
    """
    # 清理prompt，移除特殊字符
    clean_prompt = prompt.replace(' ', '_').replace('/', '_').replace('\\', '_')
    # 限制prompt长度
    if len(clean_prompt) > 50:
        clean_prompt = clean_prompt[:50]
    
    height, width = resolution
    filename = f"ti2v-5B_{width}x{height}_{clean_prompt}_{timestamp}.mp4"
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/720p_5s")
    parser.add_argument("--output_dir", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/720p_5s/decode_video")
    parser.add_argument("--vae_path", type=str, default="/mnt/vision-gen-ks3/ModelZoo/Video_Generation/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    parser.add_argument("--num_patches", type=int, default=1, help="分成几个patch进行decode，默认4")
    parser.add_argument("--patch_dim", type=str, default="w", choices=['h', 'w'], help="在哪个维度分割，h=高度，w=宽度")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有pt文件
    pt_files = glob.glob(os.path.join(args.input_dir, "*.pt"))
    print(f"找到 {len(pt_files)} 个pt文件")
    
    # 初始化VAE（只需要初始化一次）
    print("初始化VAE...")
    vae = Wan2_2_VAE(
        vae_pth=args.vae_path,
        device=args.device
    )
    
    # 遍历所有pt文件
    for idx, pt_file in enumerate(pt_files, 1):
        print(f"\n{'='*60}")
        print(f"处理 {idx}/{len(pt_files)}: {os.path.basename(pt_file)}")
        print(f"{'='*60}")
        
        try:
            # 直接用原文件名，只改扩展名
            filename = os.path.basename(pt_file)
            output_filename = filename.replace('.pt', '.mp4')
            output_path = os.path.join(args.output_dir, output_filename)
            
            # 如果输出文件已存在，跳过
            if os.path.exists(output_path):
                print(f"输出文件已存在，跳过: {output_path}")
                continue
            
            # Decode
            decode_latent_gpu_chunked(
                pt_file, 
                output_path, 
                vae,
                args.num_patches, 
                args.device, 
                args.patch_dim
            )
            
        except Exception as e:
            print(f"处理文件 {pt_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("所有文件处理完成！")
    print(f"{'='*60}")