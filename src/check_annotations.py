import pandas as pd
import numpy as np

def clean_column_name(col):
    """清理列名"""
    return col.strip().strip('"').strip("'")

# 读取标注文件
print("读取标注文件...")
with open('data/raw/annotations_final.csv', 'r', encoding='utf-8') as f:
    header = f.readline().strip()
    
# 分割列名
columns = [clean_column_name(col) for col in header.split('\t')]

# 重新读取文件
print("\n使用正确的列名读取...")
df = pd.read_csv('data/raw/annotations_final.csv', 
                 sep='\t',
                 names=columns,
                 skiprows=1)

# 显示前几行
print("\n前5行数据:")
print(df.head())

# 显示列名
print("\n列名:")
print(columns)

# 显示基本信息
print("\n基本信息:")
print(df.info())

# 检查一些关键标签的分布
key_tags = ['rock', 'metal', 'disco', 'pop', 'hip hop', 'rap', 'reggae', 
            'classical', 'jazz', 'blues', 'country', 'folk']

print("\n关键标签的分布:")
for tag in key_tags:
    if tag in df.columns:
        count = (df[tag] == '1').sum()
        print(f"{tag}: {count}首")
        
# 检查mp3_path列
print("\nmp3_path示例:")
print(df['mp3_path'].head()) 