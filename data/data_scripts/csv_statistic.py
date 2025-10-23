import pandas as pd
import os

# 文件路径
long_csv_path = '/hpc2ssd/JH_DATA/spooler/htian395/Wenxue/UltraVideo/long.csv'
short_csv_path = '/hpc2ssd/JH_DATA/spooler/htian395/Wenxue/UltraVideo/short.csv'

print("=" * 80)
print("CSV文件统计信息")
print("=" * 80)

# 统计 long.csv
print("\n【Long.csv 文件信息】")
print("-" * 80)
if os.path.exists(long_csv_path):
    # 读取文件
    df_long = pd.read_csv(long_csv_path)
    
    # 文件大小
    file_size = os.path.getsize(long_csv_path) / (1024 * 1024)  # MB
    print(f"文件大小: {file_size:.2f} MB")
    
    # 样本数量
    print(f"样本数量: {len(df_long):,} 条")
    
    # 列数
    print(f"列数: {len(df_long.columns)} 列")
    
    # 列名
    print("\n列名列表:")
    for i, col in enumerate(df_long.columns, 1):
        print(f"  {i}. {col}")
    
    # 基本统计信息
    print(f"\n数据形状: {df_long.shape}")
    print(f"内存使用: {df_long.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
else:
    print("文件不存在！")

# 统计 short.csv
print("\n" + "=" * 80)
print("【Short.csv 文件信息】")
print("-" * 80)
if os.path.exists(short_csv_path):
    # 读取文件
    df_short = pd.read_csv(short_csv_path)
    
    # 文件大小
    file_size = os.path.getsize(short_csv_path) / (1024 * 1024)  # MB
    print(f"文件大小: {file_size:.2f} MB")
    
    # 样本数量
    print(f"样本数量: {len(df_short):,} 条")
    
    # 列数
    print(f"列数: {len(df_short.columns)} 列")
    
    # 列名
    print("\n列名列表:")
    for i, col in enumerate(df_short.columns, 1):
        print(f"  {i}. {col}")
    
    # 基本统计信息
    print(f"\n数据形状: {df_short.shape}")
    print(f"内存使用: {df_short.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
else:
    print("文件不存在！")

# 对比统计
print("\n" + "=" * 80)
print("【对比统计】")
print("-" * 80)
if os.path.exists(long_csv_path) and os.path.exists(short_csv_path):
    print(f"总样本数: {len(df_long) + len(df_short):,} 条")
    print(f"Long.csv 占比: {len(df_long)/(len(df_long)+len(df_short))*100:.2f}%")
    print(f"Short.csv 占比: {len(df_short)/(len(df_long)+len(df_short))*100:.2f}%")
    
    # 检查列名是否相同
    if list(df_long.columns) == list(df_short.columns):
        print("\n✓ 两个文件的列名完全相同")
    else:
        print("\n✗ 两个文件的列名不同")
        long_only = set(df_long.columns) - set(df_short.columns)
        short_only = set(df_short.columns) - set(df_long.columns)
        if long_only:
            print(f"  仅在 long.csv 中的列: {long_only}")
        if short_only:
            print(f"  仅在 short.csv 中的列: {short_only}")

print("\n" + "=" * 80)