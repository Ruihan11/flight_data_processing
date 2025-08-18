#!/usr/bin/env python3
"""
cpu_parallel.py

并行加速的航空数据处理脚本，使用 Python 的 multiprocessing
模块在 CPU 上同时处理多个航空器。整体流程与 cpu_process.py
一致，但在特征提取阶段对每个航空器 ID 分配独立的进程以
提升吞吐量。
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import time
from typing import Any, Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count

# 导入现有的 CPU 处理器以复用预处理和特征提取逻辑
try:
    from cpu_process import FlightDataProcessor
except ImportError:
    # 如果模块不存在，抛出异常
    raise ImportError("cpu_process.py 未找到，无法导入 FlightDataProcessor")


def process_single_aircraft(args: Tuple[Any, List[pd.DataFrame], int, int]) -> List[Dict[str, Any]]:
    """在一个独立进程中处理单个航空器的数据，返回特征列表。"""
    aircraft_id, frames, min_rows, sample_size = args
    # 为每个进程创建独立的处理器，避免共享状态
    processor = FlightDataProcessor(min_rows=min_rows, sample_size=sample_size)
    try:
        # 合并同一航空器的所有数据
        combined_df = pd.concat(frames, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['Time'])
        # 预处理
        processed_df = processor.preprocess_dataframe(combined_df)
        if processed_df is None:
            return []
        # 特征提取
        features = processor.extract_features_from_flight(processed_df)
        return features
    except Exception as e:
        # 发生异常时返回空列表，同时打印调试信息
        print(f"⚠️ 并行处理航空器 {aircraft_id} 时出错: {e}")
        return []


class ParallelFlightDataProcessor:
    """并行化的 CPU 数据处理器。"""
    def __init__(self, min_rows: int = 1000, sample_size: int = 100, workers: Optional[int] = None):
        self.min_rows = min_rows
        self.sample_size = sample_size
        self.workers = workers or max(1, cpu_count() - 1)
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

    def add_aircraft_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加航空器信息"""
        if self.aircraft_db is None:
            df['icao24'] = 'UNKNOWN'
            df['icaoaircrafttype'] = 'UNKNOWN'
            df['type_code'] = 1
            return df
        df['Tail'] = df['Tail'].astype(str).str.upper().str.strip()
        merged = df.merge(
            self.aircraft_db,
            left_on='Tail',
            right_on='registration',
            how='left'
        )
        merged['icao24'] = merged['icao24'].fillna('UNKNOWN')
        merged['icaoaircrafttype'] = merged['icaoaircrafttype'].fillna('UNKNOWN')
        merged['type_code'] = pd.factorize(merged['icaoaircrafttype'])[0] + 1
        if 'registration' in merged.columns:
            merged.drop(columns=['registration'], inplace=True)
        return merged

    def filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤基于持续时间的异常值"""
        duration_median = (
            df.groupby('ID')['duration']
            .median()
            .reset_index(name='median_duration')
        )
        df = df.merge(duration_median, on='ID')
        filtered = df[df['duration'] <= 1.5 * df['median_duration']].copy()
        filtered.drop(columns=['median_duration'], inplace=True)
        return filtered

    def process_files(self, input_path: str, output_file: str, aircraft_db_path: Optional[str] = None) -> Dict[str, float]:
        """处理所有文件，返回性能指标"""
        start_time = time.time()
        # 加载数据库
        if aircraft_db_path:
            self.load_aircraft_database(aircraft_db_path)
        # 查找 CSV 文件
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

        read_start = time.time()
        data_by_id: Dict[Any, List[pd.DataFrame]] = {}
        file_count = 0
        # 顺序读取文件并分组
        for csv_file in csv_files:
            file_count += 1
            try:
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

        # 并行处理每个航空器
        process_start = time.time()
        all_features: List[Dict[str, Any]] = []
        processed_count = 0
        # 构造任务列表
        tasks = [
            (aircraft_id, frames, self.min_rows, self.sample_size)
            for aircraft_id, frames in data_by_id.items()
        ]
        # 使用 multiprocessing.Pool 执行任务
        with Pool(processes=self.workers) as pool:
            for features in pool.imap_unordered(process_single_aircraft, tasks):
                if features:
                    all_features.extend(features)
                    processed_count += 1
                    if processed_count % 10 == 0:
                        print(f"✅ 已并行处理 {processed_count} 个航空器")
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
        features_df = pd.DataFrame(all_features)
        # 合并航空器信息
        features_df = self.add_aircraft_info(features_df)
        # 过滤异常值
        features_df = self.filter_outliers(features_df)
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # 保存结果
        features_df.to_csv(output_file, index=False)
        post_time = time.time() - post_start

        total_time = time.time() - start_time
        print(f"🎉 并行处理完成!")
        print(f"   - 处理了 {processed_count} 个航空器")
        print(f"   - 提取了 {len(features_df)} 个特征块")
        print(f"   - 结果保存至: {output_file}")

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="并行加速的航空数据处理脚本"
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
        "--workers",
        type=int,
        default=None,
        help="并行进程数量 (默认: CPU 核心数 - 1)"
    )
    args = parser.parse_args()

    processor = ParallelFlightDataProcessor(
        min_rows=args.min_rows,
        sample_size=args.sample_size,
        workers=args.workers,
    )
    processor.process_files(
        input_path=args.input_path,
        output_file=args.output_file,
        aircraft_db_path=args.aircraft_db
    )


if __name__ == "__main__":
    main()