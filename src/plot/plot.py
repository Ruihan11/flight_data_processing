#!/usr/bin/env python3
"""
visualize_single_blocked.py

Visualize one flight CSV segmented into sequential blocks.
Each block is plotted in a different color.
Summary stats are displayed in a separate window as a clean table.
Block index is annotated directly on the trajectory line.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

def compute_distance(x_list, y_list):
    return sum(
        euclidean((x1, y1), (x2, y2))
        for (x1, y1), (x2, y2) in zip(zip(x_list, y_list), zip(x_list[1:], y_list[1:]))
    )

def compute_mean_heading_change(heading_series):
    diffs = heading_series.diff().dropna()
    wrapped = ((diffs + 180) % 360) - 180
    return np.abs(wrapped).mean()

def visualize_file(input_file, block_size=100, show_legend=True):
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)

    total_rows = df.shape[0]
    block_indices = np.arange(0, total_rows, block_size)

    cmap = plt.get_cmap("tab20")
    table_data = []

    # Create first figure: trajectory plot
    fig_plot, ax_plot = plt.subplots(figsize=(10, 8))

    for i, start in enumerate(block_indices):
        end = min(start + block_size, total_rows)
        block = df.iloc[start:end]
        if block.shape[0] < 2:
            continue
        x = block['X'].tolist()
        y = block['Y'].tolist()
        t0 = block['timestamp'].iloc[0]
        t1 = block['timestamp'].iloc[-1]
        duration = (t1 - t0).total_seconds()
        speed = block['Speed'].mean() if 'Speed' in block else np.nan
        heading_change = compute_mean_heading_change(block['Heading']) if 'Heading' in block else np.nan
        distance = compute_distance(x, y)

        table_data.append([
            f"{duration:.0f}s",
            f"{speed:.1f}kt",
            f"{heading_change:.1f}°",
            f"{distance:.0f}m"
        ])

        color = cmap(i % 20)
        ax_plot.plot(x, y, marker='.', markersize=4, linewidth=1, linestyle='-', color=color)
        mid_idx = len(x) // 2
        if mid_idx < len(x):
            ax_plot.text(x[mid_idx], y[mid_idx], f"{i}", fontsize=9, color=color,
                         ha='center', va='center',
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

    ax_plot.set_aspect('equal', 'box')
    ax_plot.set_xlabel("ECEF X (m)")
    ax_plot.set_ylabel("ECEF Y (m)")
    ax_plot.set_title(f"Trajectory: {os.path.basename(input_file)}")
    if show_legend:
        ax_plot.legend([], [], frameon=False)

    fig_plot.tight_layout()

    # Create second figure: text-based table
    fig_table, ax_table = plt.subplots(figsize=(10, 1 + 0.4 * len(table_data)))
    ax_table.axis("off")
    table_df = pd.DataFrame(table_data, columns=["Duration", "Speed", "ΔHeading", "Distance"])
    table_text = table_df.to_string(index=True)
    ax_table.text(0, 1, table_text, fontsize=10, fontfamily='monospace', verticalalignment='top')

    fig_table.tight_layout()

    # Show both
    plt.show()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize one flight CSV as sequential blocks with trajectory and separate summary table"
    )
    parser.add_argument("input_file", help="Path to the flight CSV file")
    parser.add_argument("--block_size", type=int, default=100, help="Number of points per block (default 100)")
    parser.add_argument("--no-legend", action="store_false", dest="show_legend", help="Hide legend")
    args = parser.parse_args()

    visualize_file(args.input_file, args.block_size, args.show_legend)

if __name__ == "__main__":
    main()
