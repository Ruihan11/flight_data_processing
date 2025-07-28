import pandas as pd
import numpy as np
import argparse
import os

"""
src/ecef.py

将地理坐标 (经纬度) 转换为 ECEF 坐标系 (地心地固坐标系)。
输入 CSV 文件包含经纬度信息，输出文件将添加 X, Y, Z 三个 ECEF 坐标列。
由于没有高度信息，假设所有数据的高度为 30000 英尺
"""

# 地球参数
a = 6378137.0
e_sq = 6.69437999014e-3
ft_to_m = 0.3048
alt_ft_default = 30000.0

def process_file(input_dir,output_dir):
    try:
        df = pd.read_csv(input_dir)
        lat = df.iloc[:,6]
        lon = df.iloc[:,7]
        alt_ft = np.full(len(df),alt_ft_default) # 30000ft in altitude by default
    except Exception as e:
        print(f"ECEF convert or reading fault in {input_dir} : {e}")
        return

    alt = alt_ft * ft_to_m
    X, Y, Z = geodetic_to_ecef(lat,lon,alt)
    df['X'] = X
    df['Y'] = Y
    df['Z'] = Z

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(input_dir))
    df.to_csv(output_path, index=False)
    print(f">> Converted ECEF written to: {output_path}")


def geodetic_to_ecef(lat,lon,alt):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    N = a / np.sqrt(1 - e_sq * np.sin(lat_rad) ** 2)
    X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = ((1 - e_sq) * N + alt) * np.sin(lat_rad)
    return X, Y, Z



def main():
    parser = argparse.ArgumentParser(description="convert lat/lon to ecef coordinates") 
    parser.add_argument("input_file",help="input csv dir")
    parser.add_argument("output_dir",help="output csv fir")
    args = parser.parse_args()

    process_file(args.input_file,args.output_dir)

if __name__ == '__main__':
    main()
