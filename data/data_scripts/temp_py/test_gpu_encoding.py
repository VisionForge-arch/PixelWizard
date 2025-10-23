#!/usr/bin/env python3
"""
测试 GPU 编码是否正常工作
"""

import subprocess
import sys
import os

def test_nvenc_availability():
    """测试 NVENC 编码器是否可用"""
    print("=" * 60)
    print("测试 1: 检查 ffmpeg 是否支持 h264_nvenc")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'],
            capture_output=True,
            text=True
        )
        
        if 'h264_nvenc' in result.stdout:
            print("✓ ffmpeg 支持 h264_nvenc 编码器")
            return True
        else:
            print("✗ ffmpeg 不支持 h264_nvenc 编码器")
            print("\n请安装支持 NVENC 的 ffmpeg 版本")
            return False
    except Exception as e:
        print(f"✗ 运行 ffmpeg 失败: {e}")
        return False

def test_cuda_availability():
    """测试 CUDA 是否可用"""
    print("\n" + "=" * 60)
    print("测试 2: 检查 CUDA 是否可用")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ CUDA 可用，GPU 信息:")
            print(result.stdout)
            return True
        else:
            print("✗ nvidia-smi 运行失败")
            return False
    except Exception as e:
        print(f"✗ nvidia-smi 未找到: {e}")
        return False

def test_simple_encoding(test_video_path=None):
    """测试简单的 GPU 编码"""
    print("\n" + "=" * 60)
    print("测试 3: 实际 GPU 编码测试")
    print("=" * 60)
    
    if test_video_path and not os.path.exists(test_video_path):
        print(f"✗ 测试视频不存在: {test_video_path}")
        return False
    
    if not test_video_path:
        print("跳过实际编码测试（未提供测试视频）")
        print("\n使用方法: python test_gpu_encoding.py <test_video.mp4>")
        return None
    
    output_path = "/tmp/test_nvenc_output.mp4"
    
    # 方案 1: 简单的 NVENC（无硬件加速解码）
    print("\n方案 1: 基础 NVENC 编码（推荐）")
    command1 = [
        'ffmpeg',
        '-i', test_video_path,
        '-r', '24',
        '-c:v', 'h264_nvenc',
        '-preset', 'medium',
        '-b:v', '5M',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-t', '5',  # 只处理5秒
        output_path,
        '-hide_banner',
        '-y'
    ]
    
    print("命令:", ' '.join(command1))
    result = subprocess.run(command1, capture_output=True, text=True)
    
    if result.returncode == 0 and os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"✓ 方案 1 成功！输出文件: {output_path} ({size} 字节)")
        os.remove(output_path)
        success1 = True
    else:
        print(f"✗ 方案 1 失败")
        print("错误信息:", result.stderr)
        success1 = False
    
    # 方案 2: 带 CUDA 硬件加速解码
    print("\n方案 2: CUDA 硬件加速解码 + NVENC 编码")
    command2 = [
        'ffmpeg',
        '-vsync', '0',
        '-hwaccel', 'cuda',
        '-hwaccel_output_format', 'cuda',
        '-i', test_video_path,
        '-r', '24',
        '-c:v', 'h264_nvenc',
        '-preset', 'medium',
        '-b:v', '5M',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-t', '5',
        output_path,
        '-hide_banner',
        '-y'
    ]
    
    print("命令:", ' '.join(command2))
    result = subprocess.run(command2, capture_output=True, text=True)
    
    if result.returncode == 0 and os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"✓ 方案 2 成功！输出文件: {output_path} ({size} 字节)")
        os.remove(output_path)
        success2 = True
    else:
        print(f"✗ 方案 2 失败")
        print("错误信息:", result.stderr)
        success2 = False
    
    return success1 or success2

def main():
    """主函数"""
    print("开始 GPU 编码诊断测试...\n")
    
    test_video = sys.argv[1] if len(sys.argv) > 1 else None
    
    # 运行测试
    nvenc_ok = test_nvenc_availability()
    cuda_ok = test_cuda_availability()
    encoding_ok = test_simple_encoding(test_video)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"NVENC 支持: {'✓' if nvenc_ok else '✗'}")
    print(f"CUDA 可用: {'✓' if cuda_ok else '✗'}")
    if encoding_ok is not None:
        print(f"编码测试: {'✓' if encoding_ok else '✗'}")
    else:
        print("编码测试: 未运行")
    
    if nvenc_ok and cuda_ok:
        print("\n建议: 您的环境支持 GPU 加速，可以使用 --use_gpu true")
        if encoding_ok == False:
            print("但是编码测试失败了，建议先解决编码问题")
    else:
        print("\n建议: 您的环境不支持 GPU 加速，请使用 CPU 模式（不加 --use_gpu 参数）")

if __name__ == "__main__":
    main()

