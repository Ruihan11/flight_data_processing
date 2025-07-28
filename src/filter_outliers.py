#!/usr/bin/env python3
"""
filter_outliers.py

Remove outlier blocks based on duration:
For each ID, compute the median duration, and remove rows where duration > 1.5 × median.
"""

import argparse
import pandas as pd

def filter_by_duration(input_file, output_file):
    # Load data
    df = pd.read_csv(input_file)

    # Compute median duration per ID
    duration_median = df.groupby('ID')['duration'].median().rename('median_duration')
    df = df.merge(duration_median, on='ID')

    # Filter rows where duration is within 1.5× median
    filtered_df = df[df['duration'] <= 1.5 * df['median_duration']].copy()
    filtered_df.drop(columns=['median_duration'], inplace=True)

    # Save result
    filtered_df.to_csv(output_file, index=False)
    print(f"✅ Saved reduced data to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Remove long-duration outliers based on 1.5× median duration per ID"
    )
    parser.add_argument("input_file", help="Input CSV file (must contain ID and duration columns)")
    parser.add_argument("output_file", help="Output CSV file to save filtered result")
    args = parser.parse_args()

    filter_by_duration(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
