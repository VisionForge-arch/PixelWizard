import pandas as pd
from collections import Counter

# 文件路径
short_csv_path = '/hpc2ssd/JH_DATA/spooler/htian395/Wenxue/UltraVideo/short.csv'

print("=" * 80)
print("Short.csv - frame_width 和 frame_height 统计")
print("=" * 80)

# 方法1: 逐块读取，统计 frame_width 和 frame_height
print("\n正在读取文件...(可能需要一些时间)")

width_counter = Counter()
height_counter = Counter()
resolution_counter = Counter()
total_rows = 0

# 使用 chunksize 分块读取大文件
chunksize = 10000
for chunk in pd.read_csv(short_csv_path, chunksize=chunksize):
    total_rows += len(chunk)
    
    # 统计 frame_width
    width_counter.update(chunk['frame_width'].tolist())
    
    # 统计 frame_height
    height_counter.update(chunk['frame_height'].tolist())
    
    # 统计分辨率组合
    resolutions = [f"{w}x{h}" for w, h in zip(chunk['frame_width'], chunk['frame_height'])]
    resolution_counter.update(resolutions)
    
    # 显示进度
    if total_rows % 50000 == 0:
        print(f"  已处理: {total_rows:,} 行...")

print(f"\n✓ 读取完成！总样本数: {total_rows:,} 条\n")

# 输出 frame_width 统计
print("【Frame Width 统计】")
print("-" * 80)
print(f"不同的宽度值数量: {len(width_counter)}")
print("\n宽度分布（按数量排序）:")
for width, count in width_counter.most_common():
    percentage = (count / total_rows) * 100
    print(f"  {width:>5} : {count:>8,} 条 ({percentage:>6.2f}%)")

# 输出 frame_height 统计
print("\n" + "=" * 80)
print("【Frame Height 统计】")
print("-" * 80)
print(f"不同的高度值数量: {len(height_counter)}")
print("\n高度分布（按数量排序）:")
for height, count in height_counter.most_common():
    percentage = (count / total_rows) * 100
    print(f"  {height:>5} : {count:>8,} 条 ({percentage:>6.2f}%)")

# 输出分辨率组合统计
print("\n" + "=" * 80)
print("【分辨率组合统计（Width x Height）】")
print("-" * 80)
print(f"不同的分辨率组合数量: {len(resolution_counter)}")
print("\n分辨率分布（按数量排序，Top 20）:")
for resolution, count in resolution_counter.most_common(20):
    percentage = (count / total_rows) * 100
    print(f"  {resolution:>15} : {count:>8,} 条 ({percentage:>6.2f}%)")

# 基本统计信息
print("\n" + "=" * 80)
print("【基本统计】")
print("-" * 80)
widths = list(width_counter.keys())
heights = list(height_counter.keys())
print(f"最小宽度: {min(widths)}")
print(f"最大宽度: {max(widths)}")
print(f"最小高度: {min(heights)}")
print(f"最大高度: {max(heights)}")

# 计算加权平均
avg_width = sum(w * c for w, c in width_counter.items()) / total_rows
avg_height = sum(h * c for h, c in height_counter.items()) / total_rows
print(f"平均宽度: {avg_width:.2f}")
print(f"平均高度: {avg_height:.2f}")

print("\n" + "=" * 80)