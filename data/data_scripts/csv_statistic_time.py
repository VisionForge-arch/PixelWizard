import pandas as pd
import numpy as np
from collections import Counter

# 文件路径
short_csv_path = '/hpc2ssd/JH_DATA/spooler/htian395/Wenxue/UltraVideo/short.csv'

print("=" * 80)
print("Short.csv - fps、total_frames 和 duration 统计")
print("=" * 80)

# 方法1: 逐块读取，统计 fps、total_frames 和 duration
print("\n正在读取文件...(可能需要一些时间)")

fps_counter = Counter()
total_frames_counter = Counter()
duration_counter = Counter()
total_rows = 0

# 用于计算统计量的列表
fps_values = []
total_frames_values = []
duration_values = []

# 使用 chunksize 分块读取大文件
chunksize = 10000
for chunk in pd.read_csv(short_csv_path, chunksize=chunksize):
    total_rows += len(chunk)
    
    # 统计 fps
    fps_counter.update(chunk['fps'].tolist())
    fps_values.extend(chunk['fps'].tolist())
    
    # 统计 total_frames
    total_frames_counter.update(chunk['total_frames'].tolist())
    total_frames_values.extend(chunk['total_frames'].tolist())
    
    # 统计 duration
    duration_counter.update(chunk['duration'].tolist())
    duration_values.extend(chunk['duration'].tolist())
    
    # 显示进度
    if total_rows % 50000 == 0:
        print(f"  已处理: {total_rows:,} 行...")

print(f"\n✓ 读取完成！总样本数: {total_rows:,} 条\n")

# 输出 fps 统计
print("【FPS 统计】")
print("-" * 80)
print(f"不同的FPS值数量: {len(fps_counter)}")
print("\nFPS分布（按数量排序）:")
for fps, count in fps_counter.most_common():
    percentage = (count / total_rows) * 100
    print(f"  {fps:>5} : {count:>8,} 条 ({percentage:>6.2f}%)")

# 输出 total_frames 统计
print("\n" + "=" * 80)
print("【Total Frames 统计】")
print("-" * 80)
print(f"不同的总帧数值数量: {len(total_frames_counter)}")
print("\n总帧数分布（按数量排序，Top 20）:")
for frames, count in total_frames_counter.most_common(20):
    percentage = (count / total_rows) * 100
    print(f"  {frames:>8} : {count:>8,} 条 ({percentage:>6.2f}%)")

# 输出 duration 统计
print("\n" + "=" * 80)
print("【Duration 统计】")
print("-" * 80)
print(f"不同的时长值数量: {len(duration_counter)}")
print("\n时长分布（按数量排序，Top 20）:")
for duration, count in duration_counter.most_common(20):
    percentage = (count / total_rows) * 100
    print(f"  {duration:>8} : {count:>8,} 条 ({percentage:>6.2f}%)")

# 基本统计信息
print("\n" + "=" * 80)
print("【基本统计信息】")
print("-" * 80)

# FPS 统计
fps_array = np.array(fps_values)
print("FPS 统计:")
print(f"  最小值: {fps_array.min()}")
print(f"  最大值: {fps_array.max()}")
print(f"  平均值: {fps_array.mean():.2f}")
print(f"  中位数: {np.median(fps_array):.2f}")
print(f"  标准差: {fps_array.std():.2f}")

# Total Frames 统计
frames_array = np.array(total_frames_values)
print("\nTotal Frames 统计:")
print(f"  最小值: {frames_array.min()}")
print(f"  最大值: {frames_array.max()}")
print(f"  平均值: {frames_array.mean():.2f}")
print(f"  中位数: {np.median(frames_array):.2f}")
print(f"  标准差: {frames_array.std():.2f}")

# Duration 统计
duration_array = np.array(duration_values)
print("\nDuration 统计:")
print(f"  最小值: {duration_array.min()}")
print(f"  最大值: {duration_array.max()}")
print(f"  平均值: {duration_array.mean():.2f}")
print(f"  中位数: {np.median(duration_array):.2f}")
print(f"  标准差: {duration_array.std():.2f}")

# 分位数统计
print("\n" + "=" * 80)
print("【分位数统计】")
print("-" * 80)

print("FPS 分位数:")
for p in [25, 50, 75, 90, 95, 99]:
    value = np.percentile(fps_array, p)
    print(f"  {p:>2}%: {value:.2f}")

print("\nTotal Frames 分位数:")
for p in [25, 50, 75, 90, 95, 99]:
    value = np.percentile(frames_array, p)
    print(f"  {p:>2}%: {value:.2f}")

print("\nDuration 分位数:")
for p in [25, 50, 75, 90, 95, 99]:
    value = np.percentile(duration_array, p)
    print(f"  {p:>2}%: {value:.2f}")

# 相关性分析
print("\n" + "=" * 80)
print("【相关性分析】")
print("-" * 80)

# 计算相关系数
correlation_fps_frames = np.corrcoef(fps_array, frames_array)[0, 1]
correlation_fps_duration = np.corrcoef(fps_array, duration_array)[0, 1]
correlation_frames_duration = np.corrcoef(frames_array, duration_array)[0, 1]

print(f"FPS 与 Total Frames 相关系数: {correlation_fps_frames:.4f}")
print(f"FPS 与 Duration 相关系数: {correlation_fps_duration:.4f}")
print(f"Total Frames 与 Duration 相关系数: {correlation_frames_duration:.4f}")

# 验证计算
print("\n" + "=" * 80)
print("【数据验证】")
print("-" * 80)

# 检查 fps * duration 是否等于 total_frames（理论上应该相等）
theoretical_frames = fps_array * duration_array
actual_frames = frames_array
difference = np.abs(theoretical_frames - actual_frames)
max_diff = np.max(difference)
mean_diff = np.mean(difference)

print(f"理论总帧数 (fps * duration) 与实际总帧数的差异:")
print(f"  最大差异: {max_diff:.2f}")
print(f"  平均差异: {mean_diff:.2f}")
print(f"  差异为0的记录数: {np.sum(difference == 0):,} / {total_rows:,} ({np.sum(difference == 0)/total_rows*100:.2f}%)")

print("\n" + "=" * 80)
print("统计完成！")
print("=" * 80)