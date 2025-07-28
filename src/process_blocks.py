#!/usr/bin/env python3
"""
src/process_blocks.py

Segment each flight into sequential sample blocks.
Each block consists of a fixed number of samples (default 500).
For each block, compute time span, average speed, mean change of heading, and 2D distance.
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

# Compute total Euclidean distance between consecutive (X, Y) points
def compute_distance(x_list, y_list):
    return sum(
        euclidean((x1, y1), (x2, y2))
        for (x1, y1), (x2, y2) in zip(zip(x_list, y_list), zip(x_list[1:], y_list[1:]))
    )

# Compute average absolute heading change per step (wrapped to [-180, +180])
def compute_mean_heading_change(heading_series):
    diffs = heading_series.diff().dropna()
    wrapped = ((diffs + 180) % 360) - 180
    return np.abs(wrapped).mean()

# Process one flight file by slicing it into fixed-size sample chunks
def process_single_file(path, sample_size=500):
    try:
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)

        total_rows = df.shape[0]
        block_indices = np.arange(0, total_rows, sample_size)

        block_list = []
        for i, start in enumerate(block_indices):
            end = min(start + sample_size, total_rows)
            block = df.iloc[start:end]

            if block.shape[0] < 2:
                continue  # skip blocks too short for distance calc

            entry_time = block['timestamp'].iloc[0]
            exit_time = block['timestamp'].iloc[-1]
            duration = (exit_time - entry_time).total_seconds()

            mean_speed = block['Speed'].mean() if 'Speed' in block else np.nan
            heading_change = compute_mean_heading_change(block['Heading']) if 'Heading' in block else np.nan
            distance = compute_distance(block['X'].tolist(), block['Y'].tolist())

            ID = block['ID'].iloc[0] if 'ID' in block else 'UNKNOWN'
            Tail = block['Tail'].iloc[0] if 'Tail' in block else 'UNKNOWN'

            block_list.append({
                'ID': ID,
                'Tail': Tail,
                'sample_index': i,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'duration': duration,
                'mean_speed': mean_speed,
                'mean_changeofheading': heading_change,
                'distance': distance
            })

        return pd.DataFrame(block_list)
    except Exception as e:
        print(f"⚠️ 跳过出错文件 {path}: {e}")
        return None

# Process all .csv files in the input directory
def batch_process(input_dir, output_file, sample_size=500):
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    all_results = []

    for f in all_files:
        print(f"📄 正在处理: {f}")
        result = process_single_file(f, sample_size=sample_size)
        if result is not None:
            all_results.append(result)

    if all_results:
        full_df = pd.concat(all_results, ignore_index=True)
        full_df.to_csv(output_file, index=False)
        print(f"✅ 所有特征已保存至: {output_file}")
    else:
        print("❌ 没有成功处理任何文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block-wise process flight CSVs using fixed-size sample count")
    parser.add_argument("--input_dir", required=True, help="输入目录（含多个 .csv 文件）")
    parser.add_argument("--output_file", required=True, help="输出文件路径，例如 plane_features.csv")
    parser.add_argument("--sample_size", type=int, default=100, help="每个 block 的采样点数（默认 500）")
    args = parser.parse_args()

    batch_process(args.input_dir, args.output_file, args.sample_size)
