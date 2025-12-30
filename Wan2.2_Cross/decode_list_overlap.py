#!/usr/bin/env python
"""
批量decode latent的脚本
用法: python decode_list.py
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
    """
    构造 2D 平滑融合 mask（用于 overlap 区域的平滑过渡）。

    思路：对 patch 的边缘做余弦窗衰减（全图边界处不衰减），避免拼接缝。
    返回 shape: (1, 1, patch_h, patch_w)，可广播到 (C, T, H, W)。
    """

    def cosine_ramp(length):
        if length <= 0:
            return torch.empty((0,), device=device, dtype=dtype)
        if length == 1:
            return torch.zeros((1,), device=device, dtype=dtype)
        t = torch.linspace(0, 1, steps=length, device=device, dtype=dtype)
        return 0.5 - 0.5 * torch.cos(math.pi * t)  # 0 -> 1

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
    latent_path, 
    output_path, 
    vae, 
    num_patches=2, 
    device='cuda', 
    patch_dim='h',
    overlap=32,  # 新增：每个 patch 两侧的重叠像素
):
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
    
    # latent 空间尺寸
    C_lat, T_lat, H_lat, W_lat = latent[0].shape

    # 输出视频和权重在第一次解码后再初始化，以适配 VAE 上采样倍率
    final_video = None
    weight = None
    scale_h = None
    scale_w = None
    
    # 每块的“基础尺寸”（不含重叠），基于 latent 尺寸
    if patch_dim == 'h':
        base_size = H_lat // num_patches
    else:
        base_size = W_lat // num_patches

    print(f"分成 {num_patches} 个patch进行decode，overlap={overlap} 像素...")

    for i in range(num_patches):
        print(f"处理patch {i+1}/{num_patches}...")

        # ------------ 计算当前 patch 的起止位置（含 overlap）------------
        if patch_dim == 'h':
            h_start_base = base_size * i
            h_end_base = H_lat if i == num_patches - 1 else base_size * (i + 1)

            # 向前 / 向后扩一点，做 overlap
            h_start = max(0, h_start_base - (overlap if i > 0 else 0))
            h_end   = min(H_lat, h_end_base + (overlap if i < num_patches - 1 else 0))

            # 切 latent
            patch_latents = []
            for l in latent:
                patch_latents.append(l[:, :, h_start:h_end, :].to(device))
        else:  # patch_dim == 'w'
            w_start_base = base_size * i
            w_end_base = W_lat if i == num_patches - 1 else base_size * (i + 1)

            w_start = max(0, w_start_base - (overlap if i > 0 else 0))
            w_end   = min(W_lat, w_end_base + (overlap if i < num_patches - 1 else 0))

            patch_latents = []
            for l in latent:
                patch_latents.append(l[:, :, :, w_start:w_end].to(device))

        # ------------ decode 这个 patch ------------
        patch_decoded = vae.decode(patch_latents)
        if isinstance(patch_decoded, (list, tuple)):
            patch_decoded = patch_decoded[0]
        # patch_decoded: [C, T, patch_H, patch_W]
        patch_decoded = patch_decoded.detach().cpu()

        # 第一次解码时，根据 patch 尺寸推断放大倍率，并分配输出 tensor
        if final_video is None:
            C_dec, T_dec, H_dec, W_dec = patch_decoded.shape

            if patch_dim == 'h':
                patch_lat_h = h_end - h_start
                scale_h = H_dec // patch_lat_h
                scale_w = W_dec // W_lat
            else:  # patch_dim == 'w'
                patch_lat_w = w_end - w_start
                scale_w = W_dec // patch_lat_w
                scale_h = H_dec // H_lat

            H_out = H_lat * scale_h
            W_out = W_lat * scale_w

            final_video = torch.zeros((C_dec, T_dec, H_out, W_out), dtype=torch.float32)
            weight = torch.zeros((1, 1, H_out, W_out), dtype=torch.float32)

        # ------------ 累加到 final_video + weight（用平滑 mask 融合 overlap）------------
        if patch_dim == 'h':
            out_h_start = h_start * scale_h
            out_h_end   = h_end * scale_h
            ph = patch_decoded.shape[2]
            assert ph == (out_h_end - out_h_start), "patch height mismatch"
            mask = build_spatial_blend_mask(
                ph,
                patch_decoded.shape[3],
                overlap_h=overlap * scale_h,
                overlap_w=0,
                is_bound=(i == 0, i == num_patches - 1, True, True),
                device=final_video.device,
                dtype=final_video.dtype,
            )
            patch_decoded = patch_decoded.to(dtype=final_video.dtype)
            final_video[:, :, out_h_start:out_h_end, :] += patch_decoded * mask
            weight[:, :, out_h_start:out_h_end, :] += mask
        else:  # w
            out_w_start = w_start * scale_w
            out_w_end   = w_end * scale_w
            pw = patch_decoded.shape[3]
            assert pw == (out_w_end - out_w_start), "patch width mismatch"
            mask = build_spatial_blend_mask(
                patch_decoded.shape[2],
                pw,
                overlap_h=0,
                overlap_w=overlap * scale_w,
                is_bound=(True, True, i == 0, i == num_patches - 1),
                device=final_video.device,
                dtype=final_video.dtype,
            )
            patch_decoded = patch_decoded.to(dtype=final_video.dtype)
            final_video[:, :, :, out_w_start:out_w_end] += patch_decoded * mask
            weight[:, :, :, out_w_start:out_w_end] += mask

        # 清理显存
        del patch_latents, patch_decoded
        torch.cuda.empty_cache()
        gc.collect()
    
    # ------------ 归一化：对 overlap 区域做平均 ------------
    # 避免除以0（理论上不会有0，但保险）
    weight = weight.clamp_min(1e-6)
    final_video = final_video / weight

    # 保存视频
    save_video(final_video, output_path)
    print(f"视频已保存到: {output_path}")

    return prompt, final_video.shape
    

def save_video(video, save_path, fps=24):
    """保存视频tensor为mp4文件"""
    import numpy as np
    import imageio
    from tqdm import tqdm
    
    video = video.permute(1, 2, 3, 0)  # (T, H, W, 3)
    video = ((video + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    desc = f"Saving {os.path.basename(save_path)}"
    with imageio.get_writer(save_path, fps=fps, quality=8) as w:
        for f in tqdm(video, desc=desc):
            w.append_data(np.array(f))

# def save_video(video, output_path):
#     """保存视频tensor为mp4文件"""
#     import numpy as np
#     import imageio
    
#     # video shape: (3, T, H, W), 范围 [-1, 1]
#     video = video.permute(1, 2, 3, 0)  # (T, H, W, 3)
#     video = ((video + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
    
#     imageio.mimwrite(output_path, video, fps=24, quality=8)

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
    parser.add_argument("--input_dir", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/720p_upsample_first10")
    parser.add_argument("--output_dir", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/720p_upsample_first10/decode_video")
    parser.add_argument("--vae_path", type=str, default="/mnt/vision-gen-ks3/ModelZoo/Video_Generation/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    parser.add_argument("--num_patches", type=int, default=3, help="分成几个patch进行decode，默认4")
    parser.add_argument("--patch_dim", type=str, default="w", choices=['h', 'w'], help="在哪个维度分割，h=高度，w=宽度")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overlap", type=int, default=3)
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
                args.patch_dim,
                args.overlap,
            )
            
        except Exception as e:
            print(f"处理文件 {pt_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("所有文件处理完成！")
    print(f"{'='*60}")
