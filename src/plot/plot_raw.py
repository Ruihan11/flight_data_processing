#!/usr/bin/env python3
"""
plot_latlon_with_labels.py

Plot 2D flight trajectory from a CSV file using Lat/Lon columns,
with "START" and "END" markers.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_latlon(csv_path, show_points=True):
    df = pd.read_csv(csv_path)

    if 'Lat' not in df.columns or 'Lon' not in df.columns:
        print("❌ CSV must contain 'Lat' and 'Lon' columns")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['Lon'], df['Lat'], linestyle='-', marker='o' if show_points else None, label='Trajectory')

    # Mark start and end points
    ax.scatter(df['Lon'].iloc[0], df['Lat'].iloc[0], c='green', marker='o', label='START', zorder=5)
    ax.scatter(df['Lon'].iloc[-1], df['Lat'].iloc[-1], c='red', marker='X', label='END', zorder=5)

    # Add text annotations
    ax.text(df['Lon'].iloc[0], df['Lat'].iloc[0], ' START', color='green', fontsize=10, ha='left', va='bottom')
    ax.text(df['Lon'].iloc[-1], df['Lat'].iloc[-1], ' END', color='red', fontsize=10, ha='left', va='top')

    ax.set_title(f"Flight Path: {os.path.basename(csv_path)}")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    ax.legend()

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot flight path with start/end using Lat/Lon from a CSV")
    parser.add_argument("csv_file", help="Path to input CSV file with Lat/Lon columns")
    parser.add_argument("--no-points", action="store_false", dest="show_points", help="Hide point markers")
    args = parser.parse_args()

    plot_latlon(args.csv_file, show_points=args.show_points)

if __name__ == "__main__":
    main()
