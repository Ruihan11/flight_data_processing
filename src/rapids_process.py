#!/usr/bin/env python3
"""
rapids_process.py

基于RAPIDS的GPU加速航空数据处理脚本
使用cuDF和cuPy进行高性能数据处理
"""

import os
import glob
import argparse
import warnings
from typing import Any, Dict, List, Tuple, Optional, cast
import time

# RAPIDS imports
try:
    import cudf  # type: ignore[import-not-found]
    import cupy as cp  # type: ignore[import-not-found]
    import pandas as pd
    import numpy as np
    RAPIDS_AVAILABLE = True
    print("✅ RAPIDS库已加载")
except ImportError as e:
    print(f"❌ RAPIDS库未找到: {e}")
    print("请安装RAPIDS: conda install -c rapidsai -c nvidia -c conda-forge rapids=23.10")
    RAPIDS_AVAILABLE = False
    # Define placeholders typed as Any to satisfy static analyzers
    cudf = cast(Any, None)
    cp = cast(Any, None)
    # Fallback to pandas/numpy
    import pandas as pd
    import numpy as np

# 地球参数 (WGS84)
EARTH_A = 6378137.0
EARTH_E_SQ = 6.69437999014e-3
FT_TO_M = 0.3048
DEFAULT_ALT_FT = 30000.0

class RapidsFlightDataProcessor:
    def __init__(self, min_rows: int = 1000, sample_size: int = 100, use_gpu: bool = True):
        self.min_rows = min_rows
        self.sample_size = sample_size
        self.use_gpu = use_gpu and RAPIDS_AVAILABLE
        self.aircraft_db = None
        
        if self.use_gpu:
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
                print(f"🚀 使用GPU加速处理 (设备: {device_count} GPU可用)")
            except Exception:
                print("🚀 使用GPU加速处理")
        else:
            print("⚠️ 使用CPU处理 (RAPIDS不可用或被禁用)")
        
    def load_aircraft_database(self, db_path: str) -> None:
        """加载航空器数据库"""
        if os.path.exists(db_path):
            if self.use_gpu:
                self.aircraft_db = cudf.read_csv(db_path)
                self.aircraft_db['registration'] = (
                    self.aircraft_db['registration']
                    .astype(str).str.upper().str.strip()
                )
            else:
                self.aircraft_db = pd.read_csv(db_path)
                self.aircraft_db['registration'] = (
                    self.aircraft_db['registration']
                    .astype(str).str.upper().str.strip()
                )
            print(f"✅ 航空器数据库已加载: {len(self.aircraft_db)} 条记录")
        else:
            print(f"⚠️ 航空器数据库未找到: {db_path}")
    
    def geodetic_to_ecef_gpu(self, lat, lon, alt):
        """GPU加速的地理坐标转ECEF坐标转换"""
        if self.use_gpu:
            # 使用 cupy 直接从 cuDF 列中构造数组，避免中间转换为 numpy
            # lat.values 是 cudf 的列，可以被 cupy.asarray 无拷贝地转换为 GPU 数组
            lat_vals = cp.asarray(lat.values)
            lon_vals = cp.asarray(lon.values)
            lat_rad = cp.radians(lat_vals)
            lon_rad = cp.radians(lon_vals)
            alt_vals = cp.full(len(lat_vals), DEFAULT_ALT_FT * FT_TO_M)
            
            N = EARTH_A / cp.sqrt(1 - EARTH_E_SQ * cp.sin(lat_rad) ** 2)
            
            X = (N + alt_vals) * cp.cos(lat_rad) * cp.cos(lon_rad)
            Y = (N + alt_vals) * cp.cos(lat_rad) * cp.sin(lon_rad)
            Z = ((1 - EARTH_E_SQ) * N + alt_vals) * cp.sin(lat_rad)
            # 返回 cupy 数组，外部在需要时自行转换为 cuDF Series
            return X, Y, Z
        else:
            lat_rad = np.radians(lat.values)
            lon_rad = np.radians(lon.values)
            alt_vals = np.full(len(lat), DEFAULT_ALT_FT * FT_TO_M)
            
            N = EARTH_A / np.sqrt(1 - EARTH_E_SQ * np.sin(lat_rad) ** 2)
            
            X = (N + alt_vals) * np.cos(lat_rad) * np.cos(lon_rad)
            Y = (N + alt_vals) * np.cos(lat_rad) * np.sin(lon_rad)
            Z = ((1 - EARTH_E_SQ) * N + alt_vals) * np.sin(lat_rad)
            
            return X, Y, Z
    
    def compute_distance_gpu(self, x_series, y_series):
        """GPU加速的距离计算"""
        try:
            if len(x_series) < 2:
                return 0.0
                
            if self.use_gpu:
                x_vals = cp.asarray(x_series.values)
                y_vals = cp.asarray(y_series.values)
                
                # 检查是否有有效数据
                valid_mask = ~(cp.isnan(x_vals) | cp.isnan(y_vals))
                if cp.sum(valid_mask) < 2:
                    return 0.0
                
                x_vals = x_vals[valid_mask]
                y_vals = y_vals[valid_mask]
                
                dx = cp.diff(x_vals)
                dy = cp.diff(y_vals)
                distances = cp.sqrt(dx**2 + dy**2)
                
                return float(cp.sum(distances))
            else:
                x_vals = x_series.values
                y_vals = y_series.values
                
                # 检查是否有有效数据
                valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                if np.sum(valid_mask) < 2:
                    return 0.0
                
                x_vals = x_vals[valid_mask]
                y_vals = y_vals[valid_mask]
                
                dx = np.diff(x_vals)
                dy = np.diff(y_vals)
                distances = np.sqrt(dx**2 + dy**2)
                
                return float(np.sum(distances))
        except Exception as e:
            print(f"⚠️ 距离计算失败: {e}")
            return 0.0
    
    def compute_mean_heading_change_gpu(self, heading_series):
        """GPU加速的航向变化计算"""
        try:
            if self.use_gpu and hasattr(heading_series, 'values'):
                # 先检查数据是否有效
                if len(heading_series) < 2:
                    return 0.0
                
                heading_vals = cp.asarray(heading_series.values)
                diffs = cp.diff(heading_vals)
                
                # 处理角度环绕 [-180, +180]
                wrapped = ((diffs + 180) % 360) - 180
                valid_diffs = wrapped[~cp.isnan(wrapped)]
                
                if len(valid_diffs) == 0:
                    return 0.0
                
                return float(cp.mean(cp.abs(valid_diffs)))
            else:
                diffs = heading_series.diff().dropna()
                if diffs.empty:
                    return 0.0
                # 处理角度环绕 [-180, +180]
                wrapped = ((diffs + 180) % 360) - 180
                return float(np.abs(wrapped).mean())
        except Exception as e:
            print(f"⚠️ 航向变化计算失败: {e}")
            return 0.0
    
    def safe_to_datetime(self, datetime_strings, use_cudf=True):
        """安全的日期时间转换"""
        try:
            if use_cudf and self.use_gpu:
                # 尝试cuDF转换
                return cudf.to_datetime(datetime_strings)
            else:
                # 使用pandas转换
                if hasattr(datetime_strings, 'to_pandas'):
                    datetime_strings = datetime_strings.to_pandas()
                return pd.to_datetime(datetime_strings, errors='coerce')
        except Exception as e:
            print(f"⚠️ cuDF时间转换失败，使用pandas: {e}")
            # 回退到pandas
            if hasattr(datetime_strings, 'to_pandas'):
                datetime_strings = datetime_strings.to_pandas()
            return pd.to_datetime(datetime_strings, errors='coerce')
    
    def safe_timedelta_seconds(self, timedelta_obj):
        """安全的时间差秒数计算"""
        try:
            if hasattr(timedelta_obj, 'total_seconds'):
                return timedelta_obj.total_seconds()
            elif isinstance(timedelta_obj, np.timedelta64):
                # 转换numpy.timedelta64到秒
                return float(timedelta_obj / np.timedelta64(1, 's'))
            elif hasattr(timedelta_obj, 'seconds'):
                # 某些cuDF对象可能有seconds属性
                return float(timedelta_obj.seconds)
            else:
                # 尝试转换为pandas Timedelta
                pd_timedelta = pd.Timedelta(timedelta_obj)
                return pd_timedelta.total_seconds()
        except Exception as e:
            print(f"⚠️ 时间差计算失败: {e}")
            return 0.0
    
    def preprocess_dataframe_gpu(self, df):
        """GPU加速的数据预处理"""
        try:
            # 检查必要列
            required_cols = ['ID', 'Time']
            if not all(col in df.columns for col in required_cols):
                return None
                
            # 过滤行数过少的数据
            if len(df) < self.min_rows:
                return None
            
            # 转换为GPU DataFrame - 添加错误处理
            if self.use_gpu and not isinstance(df, cudf.DataFrame):
                try:
                    df = cudf.from_pandas(df)
                except Exception as e:
                    print(f"⚠️ 转换为cuDF失败，使用pandas: {e}")
                    self.use_gpu = False
            
            # 数据类型转换和插值
            num_cols = ['Altitude', 'Speed', 'Heading', 'Lat', 'Lon']
            for col in num_cols:
                if col in df.columns:
                    # 先转换为字符串再转换为数值，避免类型错误
                    if self.use_gpu:
                        try:
                            # 统一为字符串
                            df[col] = df[col].astype('str')
                            # 标记无效字符串（空串/none/nan/true/false）为缺失
                            lower_vals = df[col].str.lower()
                            invalid_mask = lower_vals.isin(['', 'none', 'nan', 'true', 'false'])
                            df[col] = df[col].where(~invalid_mask, None)
                            # 转换为浮点并前后填充
                            df[col] = df[col].astype('float64')
                            df[col] = df[col].ffill().bfill()
                        except Exception as e:
                            print(f"⚠️ cuDF数值转换失败 {col}: {e}")
                            # 回退到pandas
                            temp_series = df[col].to_pandas() if hasattr(df[col], 'to_pandas') else df[col]
                            temp_series = pd.to_numeric(temp_series, errors='coerce')
                            if not isinstance(temp_series, pd.Series):
                                temp_series = pd.Series(temp_series)
                            temp_series = temp_series.ffill().bfill()
                            if self.use_gpu:
                                df[col] = cudf.from_pandas(temp_series)
                            else:
                                df[col] = temp_series
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
                        df[col] = df[col].ffill().bfill()
                else:
                    if self.use_gpu:
                        df[col] = cudf.Series([np.nan] * len(df), dtype='float64')
                    else:
                        df[col] = np.nan
            
            # 处理字符串列
            for col in ['Tail', 'Metar']:
                if col in df.columns:
                    if self.use_gpu:
                        # cuDF字符串处理
                        try:
                            non_null = df[col].dropna()
                            fill_val = non_null.iloc[0] if len(non_null) > 0 else 'UNKNOWN'
                            df[col] = df[col].fillna(fill_val)
                        except Exception as e:
                            print(f"⚠️ cuDF字符串处理失败 {col}: {e}")
                            df[col] = 'UNKNOWN'
                    else:
                        non_null = df[col].dropna()
                        fill_val = non_null.iloc[0] if not non_null.empty else 'UNKNOWN'
                        df[col] = df[col].fillna(fill_val)
                else:
                    df[col] = 'UNKNOWN'
            
            # 时间戳处理 - 使用安全的转换函数
            try:
                if 'Date' in df.columns and 'Time' in df.columns:
                    # 先合并为字符串
                    if self.use_gpu:
                        datetime_str = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
                    else:
                        datetime_str = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
                    
                    # 安全转换
                    df['timestamp'] = self.safe_to_datetime(datetime_str, use_cudf=self.use_gpu)
                    
                    # 如果转换后是pandas系列，需要转回cuDF
                    if self.use_gpu and isinstance(df['timestamp'], pd.Series):
                        df['timestamp'] = cudf.from_pandas(df['timestamp'])
                else:
                    # 如果没有Date和Time列，尝试其他时间列
                    df['timestamp'] = self.safe_to_datetime(df['Time'], use_cudf=self.use_gpu)
                    if self.use_gpu and isinstance(df['timestamp'], pd.Series):
                        df['timestamp'] = cudf.from_pandas(df['timestamp'])
                
                df = df.sort_values('timestamp').reset_index(drop=True)
            except Exception as e:
                print(f"⚠️ 时间处理失败: {e}")
                # 创建一个简单的时间索引
                if self.use_gpu:
                    df['timestamp'] = cudf.Series(range(len(df)), dtype='int64')
                else:
                    df['timestamp'] = pd.Series(range(len(df)), dtype='int64')
            
            # ECEF坐标转换
            try:
                X, Y, Z = self.geodetic_to_ecef_gpu(df['Lat'], df['Lon'], None)
                if self.use_gpu:
                    # 将 cupy 数组转换为 cuDF Series，避免从 GPU 到 CPU 的往返传输
                    df['X'] = cudf.Series(X)
                    df['Y'] = cudf.Series(Y)
                    df['Z'] = cudf.Series(Z)
                else:
                    # 对于 CPU，X/Y/Z 已经是 numpy 数组
                    df['X'] = X
                    df['Y'] = Y
                    df['Z'] = Z
            except Exception as e:
                print(f"⚠️ 坐标转换失败: {e}")
                return None
            
            return df
            
        except Exception as e:
            print(f"⚠️ 数据预处理失败: {e}")
            return None
    
    def extract_features_from_flight_gpu(self, df) -> List[Dict]:
        """GPU加速的特征提取"""
        total_rows = len(df)
        blocks = []
        
        for i in range(0, total_rows, self.sample_size):
            end_idx = min(i + self.sample_size, total_rows)
            
            if self.use_gpu:
                block = df.iloc[i:end_idx]
            else:
                block = df.iloc[i:end_idx].copy()
            
            if len(block) < 2:
                continue
            
            try:
                # 时间特征
                if self.use_gpu:
                    entry_time = block['timestamp'].iloc[0]
                    exit_time = block['timestamp'].iloc[-1]
                else:
                    entry_time = block['timestamp'].iloc[0]
                    exit_time = block['timestamp'].iloc[-1]
                
                # 安全的时间差计算
                time_diff = exit_time - entry_time
                duration = self.safe_timedelta_seconds(time_diff)
                
                # 运动特征
                if self.use_gpu:
                    mean_speed = float(block['Speed'].mean())
                    heading_change = self.compute_mean_heading_change_gpu(block['Heading'])
                    distance = self.compute_distance_gpu(block['X'], block['Y'])
                    
                    aircraft_id = block['ID'].iloc[0]
                    tail = block['Tail'].iloc[0]
                else:
                    mean_speed = float(block['Speed'].mean())
                    heading_change = self.compute_mean_heading_change_gpu(block['Heading'])
                    distance = self.compute_distance_gpu(block['X'], block['Y'])
                    
                    aircraft_id = block['ID'].iloc[0]
                    tail = block['Tail'].iloc[0]
                
                blocks.append({
                    'ID': aircraft_id,
                    'Tail': tail,
                    'sample_index': i // self.sample_size,
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'duration': duration,
                    'mean_speed': mean_speed if not pd.isna(mean_speed) else 0.0,
                    'mean_changeofheading': heading_change,
                    'distance': distance
                })
            except Exception as e:
                print(f"⚠️ 特征提取失败 (块 {i}): {e}")
                continue
        
        return blocks
    
    def add_aircraft_info_gpu(self, df):
        """GPU加速的航空器信息合并"""
        if self.aircraft_db is None:
            df['icao24'] = 'UNKNOWN'
            df['icaoaircrafttype'] = 'UNKNOWN'
            df['type_code'] = 1
            return df
        
        try:
            # 确保数据类型一致
            if self.use_gpu:
                if not isinstance(df, cudf.DataFrame):
                    df = cudf.from_pandas(df)
                df['Tail'] = df['Tail'].astype(str).str.upper().str.strip()
                
                # 确保aircraft_db也是cuDF格式
                if not isinstance(self.aircraft_db, cudf.DataFrame):
                    self.aircraft_db = cudf.from_pandas(self.aircraft_db)
                
                # GPU加速合并
                merged = df.merge(
                    self.aircraft_db, 
                    left_on='Tail', 
                    right_on='registration', 
                    how='left'
                )
            else:
                df['Tail'] = df['Tail'].astype(str).str.upper().str.strip()
                
                # 确保aircraft_db是pandas格式
                aircraft_db_pd = self.aircraft_db.to_pandas() if hasattr(self.aircraft_db, 'to_pandas') else self.aircraft_db
                
                merged = df.merge(
                    aircraft_db_pd, 
                    left_on='Tail', 
                    right_on='registration', 
                    how='left'
                )
            
            # 填充缺失值
            merged['icao24'] = merged['icao24'].fillna('UNKNOWN')
            merged['icaoaircrafttype'] = merged['icaoaircrafttype'].fillna('UNKNOWN')
            
            # 类型编码
            if self.use_gpu:
                unique_types = merged['icaoaircrafttype'].unique()
                if hasattr(unique_types, 'to_pandas'):
                    unique_types_list = unique_types.to_pandas().tolist()
                else:
                    unique_types_list = unique_types.tolist()
                type_mapping = {t: i+1 for i, t in enumerate(unique_types_list)}
                merged['type_code'] = merged['icaoaircrafttype'].map(type_mapping)
                merged['type_code'] = merged['type_code'].fillna(1)
            else:
                merged['type_code'] = pd.factorize(merged['icaoaircrafttype'])[0] + 1
            
            # 清理列
            if 'registration' in merged.columns:
                merged = merged.drop(columns=['registration'])
            
            return merged
        except Exception as e:
            print(f"⚠️ 航空器信息合并失败: {e}")
            # 回退：只添加默认值
            df['icao24'] = 'UNKNOWN'
            df['icaoaircrafttype'] = 'UNKNOWN'
            df['type_code'] = 1
            return df
    
    def filter_outliers_gpu(self, df):
        """GPU加速的异常值过滤"""
        try:
            if self.use_gpu:
                # GPU加速的分组统计
                duration_median = df.groupby('ID')['duration'].median().reset_index()
                duration_median.columns = ['ID', 'median_duration']
                
                df = df.merge(duration_median, on='ID')
                filtered = df[df['duration'] <= 1.5 * df['median_duration']]
                filtered = filtered.drop(columns=['median_duration'])
            else:
                duration_median = (
                    df.groupby('ID')['duration']
                    .median()
                    .reset_index(name='median_duration')
                )
                df = df.merge(duration_median, on='ID')
                filtered = df[df['duration'] <= 1.5 * df['median_duration']].copy()
                filtered.drop(columns=['median_duration'], inplace=True)
            
            return filtered
        except Exception as e:
            print(f"⚠️ 异常值过滤失败: {e}")
            return df
    
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
                if self.use_gpu:
                    # 为确保不同文件读取时每列数据类型一致，强制所有列读取为字符串。
                    # 这样合并时不会因类型不一致而报错，后续再进行数值转换。
                    df = cudf.read_csv(csv_file, dtype=str)
                else:
                    df = pd.read_csv(csv_file, low_memory=False)
                
                if 'ID' not in df.columns:
                    continue
                    
                for aircraft_id, group in df.groupby('ID'):
                    if aircraft_id not in data_by_id:
                        data_by_id[aircraft_id] = []
                    data_by_id[aircraft_id].append(group)
                
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
                if self.use_gpu:
                    # 由于读取时已将所有列作为字符串读入，直接使用 cuDF 合并即可
                    combined_df = cudf.concat(frames, ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['Time'])
                else:
                    combined_df = pd.concat(frames, ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['Time'])
                
                # 预处理
                processed_df = self.preprocess_dataframe_gpu(combined_df)
                if processed_df is None:
                    continue
                
                # 特征提取
                features = self.extract_features_from_flight_gpu(processed_df)
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
        if self.use_gpu:
            features_df = cudf.DataFrame(all_features)
        else:
            features_df = pd.DataFrame(all_features)
        
        # 添加航空器信息
        features_df = self.add_aircraft_info_gpu(features_df)
        
        # 过滤异常值
        features_df = self.filter_outliers_gpu(features_df)
        
        # 如果结果是 GPU DataFrame，直接使用 cuDF 写出，避免中间转换
        if self.use_gpu and isinstance(features_df, cudf.DataFrame):
            post_time = time.time() - post_start
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            features_df.to_csv(output_file, index=False)
        else:
            # 结果为 pandas DataFrame
            post_time = time.time() - post_start
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            features_df.to_csv(output_file, index=False)
        
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
        description="RAPIDS GPU加速航空数据处理脚本"
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
    parser.add_argument(
        "--cpu-only", 
        action="store_true",
        help="强制使用CPU处理"
    )
    
    args = parser.parse_args()
    
    # 创建处理器并运行
    processor = RapidsFlightDataProcessor(
        min_rows=args.min_rows,
        sample_size=args.sample_size,
        use_gpu=not args.cpu_only
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