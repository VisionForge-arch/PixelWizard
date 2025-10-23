#!/usr/bin/env python3
"""
查看视频分辨率的脚本
"""
import os
import sys

# 方法1: 使用 OpenCV (推荐，速度快)
def get_video_resolution_cv2(video_path):
    """使用OpenCV获取视频分辨率"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, "无法打开视频文件"
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': frame_count / fps if fps > 0 else 0
        }, None
    except ImportError:
        return None, "需要安装 opencv-python: pip install opencv-python"
    except Exception as e:
        return None, f"错误: {str(e)}"


# 方法2: 使用 ffprobe (需要系统安装ffmpeg)
def get_video_resolution_ffprobe(video_path):
    """使用ffprobe获取视频分辨率"""
    try:
        import subprocess
        import json
        
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None, "ffprobe执行失败"
        
        data = json.loads(result.stdout)
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            return None, "未找到视频流"
        
        return {
            'width': video_stream.get('width'),
            'height': video_stream.get('height'),
            'fps': eval(video_stream.get('r_frame_rate', '0/1')),
            'codec': video_stream.get('codec_name'),
            'duration': float(video_stream.get('duration', 0))
        }, None
        
    except FileNotFoundError:
        return None, "需要安装 ffmpeg 和 ffprobe"
    except Exception as e:
        return None, f"错误: {str(e)}"


def main():
    # 视频目录
    video_dir = '/hpc2ssd/JH_DATA/spooler/htian395/Wenxue/UltraVideo/clips_short_1920'
    
    # 获取目录中的第一个视频文件
    try:
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        if not video_files:
            print("目录中没有找到 .mp4 视频文件")
            return
        
        # 选择第一个视频
        video_file = video_files[0]
        video_path = os.path.join(video_dir, video_file)
        
        print("=" * 80)
        print(f"视频文件: {video_file}")
        print("=" * 80)
        
        # 文件大小
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        print(f"\n文件大小: {file_size:.2f} MB")
        
        # 尝试使用 OpenCV
        print("\n【使用 OpenCV 获取信息】")
        info, error = get_video_resolution_cv2(video_path)
        if info:
            print(f"  分辨率: {info['width']} x {info['height']}")
            print(f"  帧率: {info['fps']:.2f} fps")
            print(f"  总帧数: {info['frame_count']}")
            print(f"  时长: {info['duration']:.2f} 秒")
        else:
            print(f"  {error}")
        
        # 尝试使用 ffprobe
        print("\n【使用 ffprobe 获取信息】")
        info, error = get_video_resolution_ffprobe(video_path)
        if info:
            print(f"  分辨率: {info['width']} x {info['height']}")
            print(f"  帧率: {info['fps']:.2f} fps")
            print(f"  编码: {info['codec']}")
            print(f"  时长: {info['duration']:.2f} 秒")
        else:
            print(f"  {error}")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()