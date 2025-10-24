#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从CSV文件提取指定字段并转换为JSON格式
提取字段：clip_id, Detailed Description, total_frames
"""

import pandas as pd
import json

def extract_to_json():
    # 读取CSV文件
    csv_file = "/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/UltraVideo/matched_short.csv"
    
    try:
        # 读取CSV文件
        print("正在读取CSV文件...")
        df = pd.read_csv(csv_file)
        print(f"成功读取 {len(df)} 条记录")
        
        # 提取指定字段
        required_fields = ['clip_id', 'Detailed Description', 'Brief Description', 'Summarized Description', 'total_frames']
        
        # 检查字段是否存在
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            print(f"警告：以下字段不存在于CSV文件中: {missing_fields}")
            print(f"可用字段: {list(df.columns)}")
            return
        
        # 提取数据
        extracted_data = df[required_fields].copy()
        
        # 转换为JSON格式
        json_data = []
        for _, row in extracted_data.iterrows():
            record = {
                "clip_id": row['clip_id'],
                "detailed_description": row['Detailed Description'],
                "brief_description": row['Brief Description'],
                "summarized_description": row['Summarized Description'],
                "total_frames": row['total_frames']
            }
            json_data.append(record)
        
        # 输出JSON
        json_output = json.dumps(json_data, ensure_ascii=False, indent=2)
        
        # print("提取的JSON数据：")
        # print("=" * 50)
        # print(json_output)
        
        # 保存到文件
        output_file = "/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/UltraVideo/matched_short.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_output)
        
        print("=" * 50)
        print(f"JSON数据已保存到: {output_file}")
        print(f"总共提取了 {len(json_data)} 条记录")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")

if __name__ == "__main__":
    extract_to_json()