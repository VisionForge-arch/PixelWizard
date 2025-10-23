import pandas as pd

# 读取CSV文件
df = pd.read_csv('/hpc2ssd/JH_DATA/spooler/htian395/Wenxue/Dataset/UltraVideo-Long/long.csv')

# 查看基本信息
print("=== CSV文件基本信息 ===")
print(f"总样本数: {len(df)}")
print(f"列数: {len(df.columns)}")
print("\n=== 列名列表 ===")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n=== 数据类型 ===")
print(df.dtypes)

print("\n=== 前5行数据 ===")
print(df.head())

print("\n=== 数据概要 ===")
print(df.info())