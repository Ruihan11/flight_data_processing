#!/usr/bin/env python3
"""
src/sort.py

Split raw CSV files by aircraft ID into separate files.
Adds interpolation, deduplication, and minimum row filtering.
Supports global row count tracking and user-defined minimum output row threshold.
"""

import os
import glob
import argparse
import pandas as pd

# Global dictionary to store final row count per aircraft ID
row_counts = {}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split raw CSV files by aircraft ID and write valid files to output directory"
    )
    parser.add_argument(
        "raw_data_path",
        type=str,
        help="Input root directory containing raw CSV files (supports recursion)"
    )
    parser.add_argument(
        "tmp_data_path",
        type=str,
        help="Output directory to save per-ID CSV files"
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1000,
        help="Minimum number of rows required to keep a file (default: 10)"
    )
    return parser.parse_args()

def split_by_id(raw_data_path: str, tmp_data_path: str, min_rows: int):
    global row_counts
    row_counts = {}

    os.makedirs(tmp_data_path, exist_ok=True)

    pattern = os.path.join(raw_data_path, '**', '*.csv')
    csv_files = glob.glob(pattern, recursive=True)
    if not csv_files:
        print(f'>> No CSV files found under {raw_data_path}')
        return

    data_by_id = {}

    for csv_file in csv_files:
        rel_path = os.path.relpath(csv_file, raw_data_path)
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f'>> Warning: Failed to read {rel_path}: {e}')
            continue

        # Skip if input has too few rows
        if df.shape[0] < min_rows:
            print(f'>> Skipped {rel_path} ({df.shape[0]} rows < {min_rows})')
            continue

        # Must have ID and Time columns
        if 'ID' not in df.columns or 'Time' not in df.columns:
            print(f'>> Skipped {rel_path}: missing "ID" or "Time" column')
            continue

        # Interpolate and fill numerical columns
        num_cols = ['Altitude', 'Speed', 'Heading', 'Lat', 'Lon']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                df[col] = df[col].ffill().bfill()
            else:
                df[col] = pd.NA

        # Group by ID
        for id_val, group in df.groupby('ID'):
            data_by_id.setdefault(id_val, []).append(group)

        print(f'>> Processed: {rel_path}')

    written_count = 0
    for id_val, frames in data_by_id.items():
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=['Time'])

        # Fill missing Tail and Metar
        for col in ['Tail', 'Metar']:
            if col in combined.columns:
                non_null = combined[col].dropna()
                fill_val = non_null.iloc[0] if not non_null.empty else 'UNKNOWN'
                combined[col] = combined[col].fillna(fill_val)
            else:
                combined[col] = 'UNKNOWN'

        # Record row count
        row_counts[id_val] = combined.shape[0]

        # Skip if output too small
        if combined.shape[0] < min_rows:
            print(f'>> Skipped writing {id_val}.csv: only {combined.shape[0]} rows < {min_rows}')
            continue

        out_path = os.path.join(tmp_data_path, f'{id_val}.csv')
        combined.to_csv(out_path, index=False)
        written_count += 1

    print(f">> Total {written_count} files written to {tmp_data_path}")
    # print(f">> Row count per ID: {row_counts}")

def main():
    args = parse_args()
    split_by_id(args.raw_data_path, args.tmp_data_path, args.min_rows)

if __name__ == "__main__":
    main()
