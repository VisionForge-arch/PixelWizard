#!/usr/bin/env python
"""
单独decode latent的脚本
用法: python decode_latent.py --latent_path latent_20241016_123456.pt --output output.mp4
"""
import argparse
import torch
import gc
from wan.modules.vae2_2 import Wan2_2_VAE, unpatchify


'''
def decode_latent_gpu_chunked(latent_path, output_path, vae_path="checkpoint/vae", 
                                chunk_size=2, device='cuda'):
    """在GPU上分块decode latent"""
    
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
    
    latent0 = latent[0].to(device)  # [48, 31, 90, 160]
    latent = [l.to(device) for l in latent]
    
    x = vae.decode(latent)
    
    # 保存视频
    save_video(x[0], output_path)
    print(f"视频已保存到: {output_path}")
'''

def decode_latent_gpu_chunked(latent_path, output_path, vae_path="checkpoint/vae", 
                                chunk_size=2, device='cuda'):
    """在GPU上分块decode latent，避免OOM"""
    
    # 加载latent
    print(f"加载latent: {latent_path}")
    data = torch.load(latent_path, map_location='cpu')
    latent_list = data['latent']
    prompt = data.get('prompt', 'unknown')
    print(f"Prompt: {prompt}")
    print(f"Latent列表长度: {len(latent_list)}")
    
    # 初始化VAE
    print("初始化VAE...")
    vae = Wan2_2_VAE(vae_pth=vae_path, device=device)
    
    all_videos = []
    for idx, latent in enumerate(latent_list):
        print(f"\n处理latent {idx+1}/{len(latent_list)}")
        print(f"  Shape: {latent.shape}")
        
        # 手动复现VAE decode，但加入CPU offload
        scale = vae.scale
        
        vae.model.clear_cache()
        latent = latent.to(device)
        
        if isinstance(scale[0], torch.Tensor):
            latent = latent / scale[1].view(1, -1, 1, 1, 1) + scale[0].view(1, -1, 1, 1, 1)
        else:
            latent = latent / scale[1] + scale[0]
        
        
        x = vae.model.conv2(latent)
        del latent
        
        iter_ = x.shape[2]
        cpu_chunks = []
        gpu_out = None

        print(f"  开始分块decode (chunk_size={chunk_size}, 总帧数={iter_})...")
        for i in range(iter_):
            vae.model._conv_idx = [0]
            if i == 0:
                out_frame = vae.model.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=vae.model._feat_map,
                    feat_idx=vae.model._conv_idx,
                    first_chunk=True,
                )
                gpu_out = out_frame
            else:
                out_frame = vae.model.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=vae.model._feat_map,
                    feat_idx=vae.model._conv_idx,
                )
                if gpu_out is None:
                    gpu_out = out_frame
                else:
                    gpu_out = torch.cat([gpu_out, out_frame], 2)
                del out_frame
            
            # 每chunk_size帧offload到CPU
            if (i + 1) % chunk_size == 0 or i == iter_ - 1:
                gpu_out_unpatched = unpatchify(gpu_out, patch_size=2)
                cpu_chunks.append(gpu_out_unpatched.cpu())
                del gpu_out, gpu_out_unpatched
                torch.cuda.empty_cache()
                gc.collect()
                gpu_out = None
                print(f"    已处理 {i+1}/{iter_} 帧，已offload到CPU")
        
        del x
        vae.model.clear_cache()
        
                # 拼接所有chunks
        print("  拼接所有chunks...")
        video = torch.cat(cpu_chunks, dim=2).squeeze(0)
        video = video.float().clamp_(-1, 1)
        del cpu_chunks
        
        print(f"  完成! Video shape: {video.shape}")
        all_videos.append(video)
    
    # 保存
    save_video(all_videos[0], output_path)
    print(f"\n视频已保存到: {output_path}")

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
    parser.add_argument("--latent_path", type=str, default="/hpc2hdd/home/htian395/Wenxue/Wan2.2/ti2v-5B_1920*1056_1_Close-up_of_an_Asian_man_with_a_hopeful_expression_20251023_121111.pt")
    parser.add_argument("--output", type=str, default="/hpc2hdd/home/htian395/Wenxue/Wan2.2/ti2v-5B_1920*1056_1_Close-up_of_an_Asian_man_with_a_hopeful_expression_20251023_121111.mp4")
    parser.add_argument("--vae_path", type=str, default="/hpc2ssd/JH_DATA/spooler/htian395/Wenxue/Weight/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    parser.add_argument("--chunk_size", type=int, default=1, help="GPU分块大小")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    decode_latent_gpu_chunked(args.latent_path, args.output, args.vae_path, 
                                args.chunk_size, args.device)