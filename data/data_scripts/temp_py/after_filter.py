#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从JSON文件中过滤出clip_id对应文件存在的记录
"""

import json
import os

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
    
    # 过滤存在的文件
    filtered_data = []
    for record in data:
        clip_id = record.get('clip_id', '')
        if clip_id:
            # 检查 .mp4 文件是否存在
            file_path = os.path.join(video_base_path, f"{clip_id}.mp4")
            if os.path.exists(file_path):
                filtered_data.append(record)
    
    # 保存过滤后的数据
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print(f"原始记录: {len(data)} 条")
    print(f"过滤后: {len(filtered_data)} 条")
    print(f"已保存到: {output_json}")

if __name__ == "__main__":
    filter_existing_clips()

