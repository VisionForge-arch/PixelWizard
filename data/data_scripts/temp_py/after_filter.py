#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从JSON文件中过滤出clip_id对应文件存在的记录
"""

import json
import os
from pathlib import Path

def filter_existing_clips():
    # 输入JSON文件路径
    input_json = "/mnt/vision-gen-nas02-ak/IndividualDirs/wenxueli/Dataset/UltraVideo/matched_short.json"
    
    # 视频文件存储路径
    video_base_path = "/mnt/vision-gen-nas02-ak/IndividualDirs/wenxueli/Dataset/clips_short_merged_fps24"
    
    # 输出JSON文件路径
    output_json = "/mnt/vision-gen-nas02-ak/IndividualDirs/wenxueli/Dataset/UltraVideo/matched_short_filtered.json"
    
    # 读取JSON文件
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 调试：打印前3个clip_id看看格式
    print("=" * 60)
    print("调试信息：")
    print(f"视频目录: {video_base_path}")
    print(f"目录是否存在: {os.path.exists(video_base_path)}")
    print(f"\n前3个clip_id示例:")
    for i, record in enumerate(data[:3]):
        clip_id = record.get('clip_id', '')
        print(f"  {i+1}. clip_id: {clip_id}")
        print(f"     尝试路径: {os.path.join(video_base_path, f'{clip_id}.mp4')}")
    
    # 获取视频目录下的实际文件
    if os.path.exists(video_base_path):
        video_files = os.listdir(video_base_path)[:5]
        print(f"\n视频目录下前5个文件示例:")
        for f in video_files:
            print(f"  - {f}")
    
    print("=" * 60)
    
    # 过滤存在的文件
    filtered_data = []
    for record in data:
        clip_id = record.get('clip_id', '')
        if clip_id:
            # 检查 .mp4 文件是否存在
            file_path = os.path.join(video_base_path, f"{clip_id}")
            if os.path.exists(file_path):
                filtered_data.append(record)
    
    # 保存过滤后的数据
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print(f"\n原始记录: {len(data)} 条")
    print(f"过滤后: {len(filtered_data)} 条")
    print(f"已保存到: {output_json}")

if __name__ == "__main__":
    filter_existing_clips()

