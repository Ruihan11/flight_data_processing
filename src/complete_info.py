#!/usr/bin/env python3
"""
根据 FAA 航空数据库补充飞机数据
"""

import os
import pandas as pd

# 设置路径
base_path = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_path, '../outputs')
features_path = os.path.join(output_dir, 'plane_features.csv')
db_path = os.path.join(base_path, 'aircraftDatabase.csv')
output_path = os.path.join(output_dir, 'plane_features_labeled.csv')

# 加载 aircraft database 并标准化 registration
dataBase = pd.read_csv(db_path)
dataBase['registration'] = dataBase['registration'].astype(str).str.upper().str.strip()

# 加载 plane_features 并标准化 Tail
df = pd.read_csv(features_path)
df['Tail'] = df['Tail'].astype(str).str.upper().str.strip()

# 合并 ICAO 信息
merged = df.merge(dataBase, left_on='Tail', right_on='registration', how='left')
merged['icao24'] = merged['icao24'].fillna('UNKNOWN')
merged['icaoaircrafttype'] = merged['icaoaircrafttype'].fillna('UNKNOWN')
merged.drop(columns=['registration'], inplace=True)

merged["type_code"] = pd.factorize(merged["icaoaircrafttype"])[0] + 1


# 保存结果
merged.to_csv(output_path, index=False)
print(f"✅ ICAO 信息已添加，保存至: {output_path}")
