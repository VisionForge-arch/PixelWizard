
import argparse
import os
import pandas as pd


def match_videos_with_csv(video_dir, csv_file):
    """
    第一步：匹配视频文件和CSV记录
    """
    print("=== 第一步：开始匹配视频文件和CSV记录 ===")

    
    # 检查路径是否存在
    if not os.path.exists(video_dir):
        print(f"错误：视频目录不存在: {video_dir}")
        return None, None
    
    if not os.path.exists(csv_file):
        print(f"错误：CSV文件不存在: {csv_file}")
        return None, None
    
    # 读取CSV文件
    print("正在读取CSV文件...")
    try:
        df = pd.read_csv(csv_file)
        print(f"CSV文件读取成功，共有 {len(df)} 条记录")
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return None, None
    
    # 获取视频目录中的所有mp4文件
    print("正在扫描视频目录...")
    video_files = []
    for file in os.listdir(video_dir):
        if file.endswith('.mp4'):
            video_files.append(file)
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 进行匹配
    print("正在进行匹配...")
    matched_data = []
    
    for index, row in df.iterrows():
        clip_id = row['clip_id']
        if clip_id in video_files:
            # 添加视频文件的完整路径
            row_dict = row.to_dict()
            row_dict['video_path'] = os.path.join(video_dir, clip_id)
            matched_data.append(row_dict)
    
    print(f"匹配成功 {len(matched_data)} 条记录")
    
    # 将匹配结果转换为DataFrame
    matched_df = pd.DataFrame(matched_data)
    
    return matched_df, video_files

def save_matched_data(matched_df, output_file):
    """
    第二步：保存匹配的数据
    """
    print(f"\n=== 第二步：保存匹配数据到 {output_file} ===")
    
    try:
        matched_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"匹配数据已保存，共 {len(matched_df)} 条记录")
        return True
    except Exception as e:
        print(f"保存匹配数据时出错: {e}")
        return False
    
def save_all_matched_data(matched_df, output_file=None):
    """
    第三步：保存所有匹配的数据
    """
    print(f"\n=== 第三步：保存所有匹配数据 ===")
    
    print(f"找到 {len(matched_df)} 条匹配的数据")
    
    # 保存所有匹配结果
    if output_file:
        try:
            matched_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"所有匹配数据已保存到: {output_file}")
        except Exception as e:
            print(f"保存匹配数据时出错: {e}")
            return None
    
    return matched_df

def main(args):
    # 第一步：匹配视频文件和CSV记录
    matched_df, video_files = match_videos_with_csv(args.video_dir, args.csv_file)
    if matched_df is None:
        print("匹配过程失败，程序退出")
        return
    
    # 第二步：保存所有匹配的数据
    all_matched_df = save_all_matched_data(matched_df, output_file=args.output_path)
    
    if all_matched_df is not None:
        print(f"✓ 所有匹配结果已保存")
        
        # 显示匹配结果的一些信息
        print(f"\n=== 处理结果汇总 ===")
        print(f"总视频文件数: {len(video_files) if video_files else 0}")
        print(f"CSV记录总数: {len(matched_df) if matched_df is not None else 0}")
        print(f"成功匹配数: {len(matched_df)}")
        print(f"匹配数据文件: {args.output_path}")
        
        # 显示所有匹配的视频文件列表
        print(f"\n=== 所有匹配的视频文件 ({len(all_matched_df)} 个) ===")
        for i, clip_id in enumerate(all_matched_df['clip_id'], 1):
            print(f"{i:3d}. {clip_id}")
            
    else:
        print("✗ 保存匹配数据失败")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="/hpc2ssd/JH_DATA/spooler/htian395/Wenxue/UltraVideo/short.csv")
    parser.add_argument("--video_dir", type=str, default="/hpc2ssd/JH_DATA/spooler/htian395/Wenxue/UltraVideo/clips_short_1920")
    parser.add_argument("--output_path", type=str, default="/hpc2ssd/JH_DATA/spooler/htian395/Wenxue/UltraVideo/matched_short.csv")
    args = parser.parse_args()
    
    
    
    main(args)