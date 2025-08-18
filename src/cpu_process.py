#!/usr/bin/env python3
"""
cpu_process.py

简化的航空数据处理脚本，整合了数据读取、坐标转换、特征提取和标签合并的全流程。
避免生成中间文件，提高处理效率。
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, cast
import time
from numpy.typing import ArrayLike

# 地球参数 (WGS84)
EARTH_A = 6378137.0
EARTH_E_SQ = 6.69437999014e-3
FT_TO_M = 0.3048
DEFAULT_ALT_FT = 30000.0

class FlightDataProcessor:
    def __init__(self, min_rows: int = 1000, sample_size: int = 100):
        self.min_rows = min_rows
        self.sample_size = sample_size
        self.aircraft_db = None
        
    def load_aircraft_database(self, db_path: str) -> None:
        """加载航空器数据库"""
        if os.path.exists(db_path):
            self.aircraft_db = pd.read_csv(db_path)
            self.aircraft_db['registration'] = (
                self.aircraft_db['registration']
                .astype(str).str.upper().str.strip()
            )
            print(f"✅ 航空器数据库已加载: {len(self.aircraft_db)} 条记录")
        else:
            print(f"⚠️ 航空器数据库未找到: {db_path}")
    
    def geodetic_to_ecef(self, lat: object, lon: object, 
                        alt: object) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将地理坐标转换为ECEF坐标"""
        lat_arr = np.asarray(lat, dtype=float)
        lon_arr = np.asarray(lon, dtype=float)
        alt_arr = np.asarray(alt, dtype=float)
        lat_rad = np.radians(lat_arr)
        lon_rad = np.radians(lon_arr)
        N = EARTH_A / np.sqrt(1 - EARTH_E_SQ * np.sin(lat_rad) ** 2)
        
        X = (N + alt_arr) * np.cos(lat_rad) * np.cos(lon_rad)
        Y = (N + alt_arr) * np.cos(lat_rad) * np.sin(lon_rad)
        Z = ((1 - EARTH_E_SQ) * N + alt_arr) * np.sin(lat_rad)
        
        return X, Y, Z
    
    def compute_distance(self, x_list: List[float], y_list: List[float]) -> float:
        """计算连续点之间的欧几里得距离总和（使用NumPy实现，避免SciPy依赖）"""
        if len(x_list) < 2:
            return 0.0
        x = np.asarray(x_list, dtype=float)
        y = np.asarray(y_list, dtype=float)
        dx = np.diff(x)
        dy = np.diff(y)
        return float(np.sum(np.hypot(dx, dy)))
    
    def compute_mean_heading_change(self, heading_series: pd.Series) -> float:
        """计算平均航向变化率"""
        diffs = heading_series.diff().dropna()
        if diffs.empty:
            return 0.0
        # 处理角度环绕 [-180, +180]
        wrapped = ((diffs + 180) % 360) - 180
        return np.abs(wrapped).mean()
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """预处理单个数据框"""
        # 检查必要列
        if 'ID' not in df.columns or 'Time' not in df.columns:
            return None
            
        # 过滤行数过少的数据
        if len(df) < self.min_rows:
            return None
        
        # 数据类型转换和插值
        num_cols = ['Altitude', 'Speed', 'Heading', 'Lat', 'Lon']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                df[col] = df[col].ffill().bfill()
            else:
                df[col] = np.nan
        
        # 处理字符串列
        for col in ['Tail', 'Metar']:
            if col in df.columns:
                non_null = df[col].dropna()
                fill_val = non_null.iloc[0] if not non_null.empty else 'UNKNOWN'
                df[col] = df[col].fillna(fill_val)
            else:
                df[col] = 'UNKNOWN'
        
        # 时间戳处理
        df['timestamp'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'], errors='coerce'
        )
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # ECEF坐标转换
        alt_m = np.full(len(df), DEFAULT_ALT_FT * FT_TO_M)
        df['X'], df['Y'], df['Z'] = self.geodetic_to_ecef(
            df['Lat'].to_numpy(dtype=float, copy=False),
            df['Lon'].to_numpy(dtype=float, copy=False),
            alt_m
        )
        
        return df
    
    def extract_features_from_flight(self, df: pd.DataFrame) -> List[Dict]:
        """从单个航班数据中提取块特征"""
        total_rows = len(df)
        blocks = []
        
        for i in range(0, total_rows, self.sample_size):
            end_idx = min(i + self.sample_size, total_rows)
            block = df.iloc[i:end_idx].copy()
            
            if len(block) < 2:
                continue
            
            # 时间特征
            entry_time = block['timestamp'].iloc[0]
            exit_time = block['timestamp'].iloc[-1]
            duration = (exit_time - entry_time).total_seconds()
            
            # 运动特征
            mean_speed = block['Speed'].mean()
            heading_change = self.compute_mean_heading_change(block['Heading'])
            distance = self.compute_distance(
                block['X'].tolist(), block['Y'].tolist()
            )
            
            # 基本信息
            aircraft_id = block['ID'].iloc[0]
            tail = block['Tail'].iloc[0]
            
            blocks.append({
                'ID': aircraft_id,
                'Tail': tail,
                'sample_index': i // self.sample_size,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'duration': duration if not pd.isna(duration) else 0.0,
                'mean_speed': mean_speed if not pd.isna(mean_speed) else 0.0,
                'mean_changeofheading': heading_change,
                'distance': distance
            })
        
        return blocks
    
    def add_aircraft_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加航空器信息"""
        if self.aircraft_db is None:
            df['icao24'] = 'UNKNOWN'
            df['icaoaircrafttype'] = 'UNKNOWN'
            df['type_code'] = 1
            return df
        
        # 标准化Tail列
        df['Tail'] = df['Tail'].astype(str).str.upper().str.strip()
        
        # 合并数据库信息
        merged = df.merge(
            self.aircraft_db, 
            left_on='Tail', 
            right_on='registration', 
            how='left'
        )
        
        # 填充缺失值
        merged['icao24'] = merged['icao24'].fillna('UNKNOWN')
        merged['icaoaircrafttype'] = merged['icaoaircrafttype'].fillna('UNKNOWN')
        merged['type_code'] = pd.factorize(merged['icaoaircrafttype'])[0] + 1
        
        # 清理列
        if 'registration' in merged.columns:
            merged.drop(columns=['registration'], inplace=True)
        
        return merged
    
    def filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤基于持续时间的异常值"""
        # 计算每个ID的持续时间中位数
        duration_median_series = df.groupby('ID')['duration'].median()
        duration_median = duration_median_series.reset_index()
        duration_median.columns = ['ID', 'median_duration']
        df = df.merge(duration_median, on='ID')
        
        # 过滤超过1.5倍中位数的数据
        filtered = cast(pd.DataFrame, df[df['duration'] <= 1.5 * df['median_duration']].copy())
        filtered.drop(columns=['median_duration'], inplace=True)
        
        return filtered
    
    def process_files(self, input_path: str, output_file: str, 
                     aircraft_db_path: Optional[str] = None) -> Dict[str, float]:
        """处理所有文件的主函数，返回性能指标"""
        start_time = time.time()
        
        # 加载航空器数据库
        if aircraft_db_path:
            self.load_aircraft_database(aircraft_db_path)
        
        # 查找所有CSV文件
        pattern = os.path.join(input_path, '**', '*.csv')
        csv_files = glob.glob(pattern, recursive=True)
        
        if not csv_files:
            print(f"❌ 在 {input_path} 下未找到CSV文件")
            return {
                'total_time': 0.0,
                'read_time': 0.0,
                'process_time': 0.0,
                'post_time': 0.0,
                'processed_aircraft': 0,
                'extracted_features': 0,
                'throughput': 0.0,
                'files_processed': 0
            }
        
        print(f"📁 找到 {len(csv_files)} 个CSV文件")
        
        all_features = []
        processed_count = 0
        file_count = 0
        
        # 数据读取阶段
        read_start = time.time()
        data_by_id = {}
        
        for csv_file in csv_files:
            file_count += 1
            try:
                # 设置 low_memory=False 避免混合类型警告
                df = pd.read_csv(csv_file, low_memory=False)
                if 'ID' not in df.columns:
                    continue
                    
                for aircraft_id, group in df.groupby('ID'):
                    if aircraft_id not in data_by_id:
                        data_by_id[aircraft_id] = []
                    data_by_id[aircraft_id].append(group)
                
                # 显示文件处理进度
                if file_count % 20 == 0:
                    print(f"📄 已读取 {file_count}/{len(csv_files)} 个文件...")
                    
            except Exception as e:
                print(f"⚠️ 跳过文件 {os.path.basename(csv_file)}: {e}")
                continue
        
        read_time = time.time() - read_start
        print(f"✅ 文件读取完成，共找到 {len(data_by_id)} 个不同的航空器ID")
        
        # 数据处理阶段
        process_start = time.time()
        
        for aircraft_id, frames in data_by_id.items():
            try:
                # 合并同一ID的所有数据
                combined_df = pd.concat(frames, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['Time'])
                
                # 预处理
                processed_df = self.preprocess_dataframe(combined_df)
                if processed_df is None:
                    continue
                
                # 特征提取
                features = self.extract_features_from_flight(processed_df)
                all_features.extend(features)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"✅ 已处理 {processed_count} 个航空器")
                    
            except Exception as e:
                print(f"⚠️ 处理航空器 {aircraft_id} 时出错: {e}")
                continue
        
        process_time = time.time() - process_start
        
        if not all_features:
            print("❌ 没有提取到任何特征")
            return {
                'total_time': time.time() - start_time,
                'read_time': read_time,
                'process_time': process_time,
                'post_time': 0.0,
                'processed_aircraft': processed_count,
                'extracted_features': 0,
                'throughput': 0.0,
                'files_processed': len([f for f in csv_files if os.path.exists(f)])
            }
        
        # 后处理阶段
        post_start = time.time()
        
        # 转换为DataFrame
        features_df = pd.DataFrame(all_features)
        
        # 添加航空器信息
        features_df = self.add_aircraft_info(features_df)
        
        # 过滤异常值
        features_df = self.filter_outliers(features_df)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存结果
        features_df.to_csv(output_file, index=False)
        
        post_time = time.time() - post_start
        total_time = time.time() - start_time
        
        print(f"🎉 处理完成!")
        print(f"   - 处理了 {processed_count} 个航空器")
        print(f"   - 提取了 {len(features_df)} 个特征块")
        print(f"   - 结果保存至: {output_file}")
        
        # 返回性能指标
        return {
            'total_time': total_time,
            'read_time': read_time,
            'process_time': process_time,
            'post_time': post_time,
            'processed_aircraft': processed_count,
            'extracted_features': len(features_df),
            'throughput': len(features_df) / total_time if total_time > 0 else 0,
            'files_processed': len([f for f in csv_files if os.path.exists(f)])
        }

def main():
    parser = argparse.ArgumentParser(
        description="简化的航空数据处理脚本"
    )
    parser.add_argument(
        "input_path", 
        help="输入数据路径（支持递归搜索CSV文件）"
    )
    parser.add_argument(
        "output_file", 
        help="输出特征文件路径"
    )
    parser.add_argument(
        "--aircraft-db", 
        help="航空器数据库CSV文件路径", 
        default=None
    )
    parser.add_argument(
        "--min-rows", 
        type=int, 
        default=1000, 
        help="最小行数阈值 (默认: 1000)"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=100, 
        help="每个特征块的样本数 (默认: 100)"
    )
    
    args = parser.parse_args()
    
    # 创建处理器并运行
    processor = FlightDataProcessor(
        min_rows=args.min_rows,
        sample_size=args.sample_size
    )
    
    metrics = processor.process_files(
        input_path=args.input_path,
        output_file=args.output_file,
        aircraft_db_path=args.aircraft_db
    )
    
    # 输出性能指标
    if metrics:
        print("\n📊 性能指标:")
        print(f"   总处理时间: {metrics['total_time']:.2f} 秒")
        print(f"   读取时间: {metrics['read_time']:.2f} 秒")
        print(f"   处理时间: {metrics['process_time']:.2f} 秒")
        print(f"   后处理时间: {metrics['post_time']:.2f} 秒")
        print(f"   吞吐量: {metrics['throughput']:.2f} 特征/秒")

if __name__ == "__main__":
    main()