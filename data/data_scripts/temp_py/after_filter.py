#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
过滤CSV文件，只保留clip_id能和视频文件名对应的记录
将结果保存为JSON格式
"""

import pandas as pd
import json
import os
from pathlib import Path

def filter_existing_clips(csv_path, video_dir, output_json_path):
    """
    过滤CSV文件，只保留存在对应视频文件的记录
    
    Args:
        csv_path: CSV文件路径
        video_dir: 视频文件所在目录
        output_json_path: 输出JSON文件路径
    """
    try:
        # 读取CSV文件
        print("正在读取CSV文件...")
        df = pd.read_csv(csv_path)
        print(f"成功读取 {len(df)} 条记录")
        
        # 获取视频目录下所有文件
        print(f"正在扫描视频目录: {video_dir}")
        video_dir_path = Path(video_dir)
        
        if not video_dir_path.exists():
            print(f"错误：视频目录不存在: {video_dir}")
            return
        
        # 获取所有视频文件（支持多种格式）
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(video_dir_path.glob(f"*{ext}")))
        
        # 提取文件名（不含扩展名）作为集合，用于快速查找
        video_filenames = set()
        for video_file in video_files:
            # 去除扩展名
            filename_without_ext = video_file.stem
            video_filenames.add(filename_without_ext)
        
        print(f"找到 {len(video_filenames)} 个视频文件")
        
        # 过滤数据：只保留clip_id在视频文件中存在的记录
        print("正在过滤数据...")
        filtered_df = df[df['clip_id'].astype(str).isin(video_filenames)].copy()
        
        print(f"过滤后保留 {len(filtered_df)} 条记录 (原始: {len(df)} 条)")
        print(f"过滤掉了 {len(df) - len(filtered_df)} 条记录")
        
        # 提取需要的字段
        required_fields = ['clip_id', 'Detailed Description', 'Brief Description', 'Summarized Description', 'total_frames']
        
        # 检查字段是否存在
        available_fields = [field for field in required_fields if field in filtered_df.columns]
        missing_fields = [field for field in required_fields if field not in filtered_df.columns]
        
        if missing_fields:
            print(f"警告：以下字段不存在: {missing_fields}")
            print(f"将使用可用字段: {available_fields}")
        
        # 转换为JSON格式
        json_data = []
        for _, row in filtered_df.iterrows():
            record = {}
            
            # 添加可用的字段
            if 'clip_id' in available_fields:
                record['clip_id'] = row['clip_id']
            if 'Detailed Description' in available_fields:
                record['detailed_description'] = row['Detailed Description']
            if 'Brief Description' in available_fields:
                record['brief_description'] = row['Brief Description']
            if 'Summarized Description' in available_fields:
                record['summarized_description'] = row['Summarized Description']
            if 'total_frames' in available_fields:
                record['total_frames'] = row['total_frames']
            
            json_data.append(record)
        
        # 保存到JSON文件
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print("=" * 50)
        print(f"过滤后的JSON数据已保存到: {output_json_path}")
        print(f"总共保存了 {len(json_data)} 条记录")
        
        # 打印一些统计信息
        if len(json_data) > 0:
            print("\n前3条记录示例:")
            for i, record in enumerate(json_data[:3]):
                print(f"\n记录 {i+1}:")
                print(f"  clip_id: {record.get('clip_id', 'N/A')}")
                if 'total_frames' in record:
                    print(f"  total_frames: {record['total_frames']}")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    # 配置路径
    csv_path = "/mnt/vision-gen-nas02-ak/IndividualDirs/wenxueli/Dataset/UltraVideo/matched_short.csv"
    video_dir = "/mnt/vision-gen-nas02-ak/IndividualDirs/wenxueli/Dataset/clips_short_merged"  # 请根据实际情况修改
    output_json_path = "/mnt/vision-gen-nas02-ak/IndividualDirs/wenxueli/Dataset/UltraVideo/matched_short_filtered.json"
    
    print("=" * 50)
    print("CSV过滤工具 - 根据视频文件存在性过滤")
    print("=" * 50)
    print(f"CSV文件: {csv_path}")
    print(f"视频目录: {video_dir}")
    print(f"输出文件: {output_json_path}")
    print("=" * 50)
    print()
    
    filter_existing_clips(csv_path, video_dir, output_json_path)

if __name__ == "__main__":
    main()

