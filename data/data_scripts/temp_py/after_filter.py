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
    
    # 视频文件存储路径（根据实际情况修改）
    video_base_path = "/mnt/vision-gen-nas02-ak/IndividualDirs/wenxueli/Dataset/clips_short_merged_fps24"
    
    # 视频文件扩展名（根据实际情况修改，可以是 .mp4, .avi, .mov 等）
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # 输出JSON文件路径
    output_json = "/mnt/vision-gen-nas02-ak/IndividualDirs/wenxueli/Dataset/UltraVideo/matched_short_filtered.json"
    
    try:
        # 读取JSON文件
        print("正在读取JSON文件...")
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取 {len(data)} 条记录")
        
        # 过滤存在的文件
        filtered_data = []
        not_found = []
        
        print("\n开始检查文件是否存在...")
        for idx, record in enumerate(data):
            if (idx + 1) % 100 == 0:
                print(f"已处理: {idx + 1}/{len(data)}")
            
            clip_id = record.get('clip_id', '')
            if not clip_id:
                not_found.append({'clip_id': 'EMPTY', 'reason': '无clip_id'})
                continue
            
            # 检查文件是否存在（尝试不同的扩展名）
            file_exists = False
            for ext in video_extensions:
                file_path = os.path.join(video_base_path, f"{clip_id}{ext}")
                if os.path.exists(file_path):
                    file_exists = True
                    break
            
            if file_exists:
                filtered_data.append(record)
            else:
                not_found.append({'clip_id': clip_id, 'reason': '文件不存在'})
        
        # 保存过滤后的数据
        print("\n正在保存过滤后的数据...")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        # 输出统计信息
        print("\n" + "=" * 60)
        print(f"原始记录数: {len(data)}")
        print(f"过滤后记录数: {len(filtered_data)}")
        print(f"被过滤掉的记录数: {len(not_found)}")
        print(f"保留比例: {len(filtered_data)/len(data)*100:.2f}%")
        print(f"输出文件: {output_json}")
        print("=" * 60)
        

        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_json}")
        print("请确认文件路径是否正确")
    except json.JSONDecodeError as e:
        print(f"错误：JSON格式解析失败 - {e}")
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("UltraVideo 数据过滤工具")
    print("=" * 60)
    print("功能：根据文件是否存在过滤JSON记录")
    print("=" * 60 + "\n")
    
    filter_existing_clips()

