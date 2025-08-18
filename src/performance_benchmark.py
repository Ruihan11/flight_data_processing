#!/usr/bin/env python3
"""
performance_benchmark.py

性能对比脚本：CPU vs GPU (RAPIDS) 数据处理性能测试
测量 throughput、latency 和资源使用情况
"""

import os
import sys
import time
import json
import argparse
import subprocess
import tempfile
from typing import Dict, List, Tuple
import pandas as pd

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("⚠️ 可视化库不可用，将跳过图表生成")

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

# 尝试导入处理器
try:
    from cpu_process import FlightDataProcessor as CPUProcessor
    CPU_AVAILABLE = True
except ImportError:
    CPU_AVAILABLE = False
    print("⚠️ CPU处理器未找到")

try:
    from rapids_process import RapidsFlightDataProcessor as GPUProcessor
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPU处理器未找到")

class PerformanceBenchmark:
    def __init__(self, dataset_sizes: List[str] = None):
        self.dataset_sizes = dataset_sizes or ['small', 'medium', 'large']
        self.results = {
            'cpu': {},
            'gpu': {}
        }
        
    def prepare_test_datasets(self, base_path: str) -> Dict[str, str]:
        """准备不同大小的测试数据集"""
        datasets = {}
        
        # 查找所有CSV文件
        import glob
        pattern = os.path.join(base_path, '**', '*.csv')
        all_files = glob.glob(pattern, recursive=True)
        
        if not all_files:
            print(f"❌ 在 {base_path} 下未找到CSV文件")
            return datasets
        
        total_files = len(all_files)
        
        # 创建不同大小的数据集
        size_configs = {
            'small': min(10, max(1, total_files // 4)),
            'medium': min(30, max(5, total_files // 2)),
            'large': total_files
        }
        
        for size_name, file_count in size_configs.items():
            if file_count > 0:
                # 创建临时目录
                temp_dir = tempfile.mkdtemp(prefix=f'benchmark_{size_name}_')
                
                # 复制文件到临时目录
                selected_files = all_files[:file_count]
                for i, src_file in enumerate(selected_files):
                    dst_file = os.path.join(temp_dir, f'file_{i:03d}.csv')
                    try:
                        import shutil
                        shutil.copy2(src_file, dst_file)
                    except Exception as e:
                        print(f"复制文件失败: {e}")
                        continue
                
                datasets[size_name] = temp_dir
                print(f"✅ 创建 {size_name} 数据集: {file_count} 个文件 -> {temp_dir}")
        
        return datasets
    
    def run_cpu_benchmark(self, input_path: str, output_file: str, 
                         aircraft_db: str = None, **kwargs) -> Dict:
        """运行CPU基准测试"""
        if not CPU_AVAILABLE:
            return {'error': 'CPU处理器不可用'}
        
        print("🖥️  开始CPU基准测试...")
        start_time = time.time()
        
        try:
            processor = CPUProcessor(
                min_rows=kwargs.get('min_rows', 1000),
                sample_size=kwargs.get('sample_size', 100)
            )
            
            metrics = processor.process_files(
                input_path=input_path,
                output_file=output_file,
                aircraft_db_path=aircraft_db
            )
            
            # 确保metrics不为None
            if metrics is None:
                metrics = {
                    'total_time': 0.0,
                    'throughput': 0.0,
                    'processed_aircraft': 0,
                    'extracted_features': 0
                }
            
            total_time = time.time() - start_time
            
            # 增强指标
            metrics.update({
                'method': 'CPU',
                'wall_time': total_time,
                'memory_efficient': True,
                'gpu_memory_used': 0
            })
            
            return metrics
            
        except Exception as e:
            print(f"CPU测试异常: {e}")
            return {'error': f'CPU测试失败: {e}'}
    
    def run_gpu_benchmark(self, input_path: str, output_file: str, 
                         aircraft_db: str = None, **kwargs) -> Dict:
        """运行GPU基准测试"""
        if not GPU_AVAILABLE:
            return {'error': 'GPU处理器不可用'}
        
        print("🚀 开始GPU基准测试...")
        start_time = time.time()
        
        try:
            processor = GPUProcessor(
                min_rows=kwargs.get('min_rows', 1000),
                sample_size=kwargs.get('sample_size', 100),
                use_gpu=True
            )
            
            metrics = processor.process_files(
                input_path=input_path,
                output_file=output_file,
                aircraft_db_path=aircraft_db
            )
            
            # 确保metrics不为None
            if metrics is None:
                metrics = {
                    'total_time': 0.0,
                    'throughput': 0.0,
                    'processed_aircraft': 0,
                    'extracted_features': 0
                }
            
            total_time = time.time() - start_time
            
            # 增强指标
            metrics.update({
                'method': 'GPU',
                'wall_time': total_time,
                'memory_efficient': False,  # GPU通常使用更多内存
                'gpu_memory_used': self._get_gpu_memory_usage()
            })
            
            return metrics
            
        except Exception as e:
            print(f"GPU测试异常: {e}")
            return {'error': f'GPU测试失败: {e}'}
    
    def _get_gpu_memory_usage(self) -> float:
        """获取GPU内存使用量"""
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            return mempool.used_bytes() / (1024**3)  # GB
        except:
            return 0.0
    
    def run_comparison(self, base_path: str, aircraft_db: str = None, 
                      output_dir: str = "benchmark_results") -> Dict:
        """运行完整的性能对比"""
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查是否有可用的处理器
        if not CPU_AVAILABLE and not GPU_AVAILABLE:
            print("❌ 没有可用的处理器")
            return {}
        
        # 准备测试数据集
        print("📁 准备测试数据集...")
        datasets = self.prepare_test_datasets(base_path)
        
        if not datasets:
            print("❌ 无法创建测试数据集")
            return {}
        
        # 测试参数
        test_params = {
            'min_rows': 500,
            'sample_size': 100
        }
        
        # 运行基准测试
        for size_name, dataset_path in datasets.items():
            print(f"\n{'='*50}")
            print(f"🧪 测试数据集: {size_name.upper()}")
            print(f"{'='*50}")
            
            # CPU测试
            if CPU_AVAILABLE:
                cpu_output = os.path.join(output_dir, f'cpu_{size_name}_features.csv')
                cpu_metrics = self.run_cpu_benchmark(
                    dataset_path, cpu_output, aircraft_db, **test_params
                )
                self.results['cpu'][size_name] = cpu_metrics
            else:
                self.results['cpu'][size_name] = {'error': 'CPU处理器不可用'}
            
            # GPU测试
            if GPU_AVAILABLE:
                gpu_output = os.path.join(output_dir, f'gpu_{size_name}_features.csv')
                gpu_metrics = self.run_gpu_benchmark(
                    dataset_path, gpu_output, aircraft_db, **test_params
                )
                self.results['gpu'][size_name] = gpu_metrics
            else:
                self.results['gpu'][size_name] = {'error': 'GPU处理器不可用'}
            
            # 即时对比
            self._print_comparison(size_name, 
                                 self.results['cpu'][size_name], 
                                 self.results['gpu'][size_name])
        
        # 生成报告
        self._generate_report(output_dir)
        if PLOT_AVAILABLE:
            self._generate_plots(output_dir)
        else:
            print("⚠️ 跳过图表生成（matplotlib/seaborn不可用）")
        
        # 清理临时文件
        self._cleanup_temp_datasets(datasets)
        
        return self.results
    
    def _print_comparison(self, size: str, cpu_metrics: Dict, gpu_metrics: Dict):
        """打印单个测试的对比结果"""
        print(f"\n📊 {size.upper()} 数据集结果:")
        print("-" * 40)
        
        if 'error' in cpu_metrics:
            print(f"CPU: ❌ {cpu_metrics['error']}")
        else:
            print(f"CPU: ✅ {cpu_metrics.get('total_time', 0):.2f}s, "
                  f"{cpu_metrics.get('throughput', 0):.2f} 特征/秒")
        
        if 'error' in gpu_metrics:
            print(f"GPU: ❌ {gpu_metrics['error']}")
        else:
            print(f"GPU: ✅ {gpu_metrics.get('total_time', 0):.2f}s, "
                  f"{gpu_metrics.get('throughput', 0):.2f} 特征/秒")
        
        # 计算加速比
        if ('error' not in cpu_metrics and 'error' not in gpu_metrics and 
            cpu_metrics.get('total_time', 0) > 0 and gpu_metrics.get('total_time', 0) > 0):
            speedup = cpu_metrics['total_time'] / gpu_metrics['total_time']
            print(f"⚡ GPU加速比: {speedup:.2f}x")
    
    def _generate_report(self, output_dir: str):
        """生成详细的性能报告"""
        report_path = os.path.join(output_dir, 'performance_report.json')
        
        # 计算汇总统计
        summary = {
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'datasets_tested': list(self.results['cpu'].keys()),
            'cpu_available': CPU_AVAILABLE,
            'gpu_available': GPU_AVAILABLE,
            'detailed_results': self.results
        }
        
        # 计算平均加速比
        speedups = []
        for size in self.results['cpu'].keys():
            cpu_result = self.results['cpu'][size]
            gpu_result = self.results['gpu'][size]
            
            if ('error' not in cpu_result and 'error' not in gpu_result):
                cpu_time = cpu_result.get('total_time', 0)
                gpu_time = gpu_result.get('total_time', 0)
                if cpu_time > 0 and gpu_time > 0:
                    speedups.append(cpu_time / gpu_time)
        
        if speedups:
            summary['average_speedup'] = sum(speedups) / len(speedups)
            summary['max_speedup'] = max(speedups)
            summary['min_speedup'] = min(speedups)
        else:
            summary['average_speedup'] = None
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📋 详细报告已保存: {report_path}")
        
        # 打印汇总
        print(f"\n{'='*60}")
        print("🏆 性能对比汇总")
        print(f"{'='*60}")
        if speedups:
            print(f"平均GPU加速比: {summary['average_speedup']:.2f}x")
            print(f"最大GPU加速比: {summary['max_speedup']:.2f}x")
            print(f"最小GPU加速比: {summary['min_speedup']:.2f}x")
        else:
            print("无法计算加速比（缺少有效的对比数据）")
    
    def _generate_plots(self, output_dir: str):
        """生成性能对比图表"""
        try:
            # 准备数据
            plot_data = []
            for method in ['cpu', 'gpu']:
                for size, metrics in self.results[method].items():
                    if 'error' not in metrics:
                        plot_data.append({
                            'Method': method.upper(),
                            'Dataset Size': size,
                            'Total Time (s)': metrics.get('total_time', 0),
                            'Throughput (features/s)': metrics.get('throughput', 0),
                            'Features Extracted': metrics.get('extracted_features', 0)
                        })
            
            if not plot_data:
                print("⚠️ 没有足够的数据生成图表")
                return
            
            df = pd.DataFrame(plot_data)
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('CPU vs GPU Performance Comparison', fontsize=16, fontweight='bold')
            
            # 1. 处理时间对比
            sns.barplot(data=df, x='Dataset Size', y='Total Time (s)', 
                       hue='Method', ax=axes[0,0])
            axes[0,0].set_title('Processing Time Comparison')
            axes[0,0].set_ylabel('Time (seconds)')
            
            # 2. 吞吐量对比
            sns.barplot(data=df, x='Dataset Size', y='Throughput (features/s)', 
                       hue='Method', ax=axes[0,1])
            axes[0,1].set_title('Throughput Comparison')
            axes[0,1].set_ylabel('Features per Second')
            
            # 3. 特征提取数量
            sns.barplot(data=df, x='Dataset Size', y='Features Extracted', 
                       hue='Method', ax=axes[1,0])
            axes[1,0].set_title('Features Extracted')
            axes[1,0].set_ylabel('Number of Features')
            
            # 4. 加速比
            speedup_data = []
            for size in df['Dataset Size'].unique():
                cpu_time = df[(df['Method'] == 'CPU') & (df['Dataset Size'] == size)]['Total Time (s)'].values
                gpu_time = df[(df['Method'] == 'GPU') & (df['Dataset Size'] == size)]['Total Time (s)'].values
                
                if len(cpu_time) > 0 and len(gpu_time) > 0 and gpu_time[0] > 0:
                    speedup = cpu_time[0] / gpu_time[0]
                    speedup_data.append({'Dataset Size': size, 'Speedup': speedup})
            
            if speedup_data:
                speedup_df = pd.DataFrame(speedup_data)
                sns.barplot(data=speedup_df, x='Dataset Size', y='Speedup', ax=axes[1,1])
                axes[1,1].set_title('GPU Speedup (CPU Time / GPU Time)')
                axes[1,1].set_ylabel('Speedup Factor')
                axes[1,1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No Speedup')
                axes[1,1].legend()
            else:
                axes[1,1].text(0.5, 0.5, 'No speedup data available', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('GPU Speedup')
            
            plt.tight_layout()
            
            # 保存图表
            plot_path = os.path.join(output_dir, 'performance_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 性能对比图表已保存: {plot_path}")
            
            # 保存数据表
            csv_path = os.path.join(output_dir, 'performance_data.csv')
            df.to_csv(csv_path, index=False)
            print(f"📈 性能数据已保存: {csv_path}")
            
        except Exception as e:
            print(f"⚠️ 生成图表时出错: {e}")
    
    def _cleanup_temp_datasets(self, datasets: Dict[str, str]):
        """清理临时数据集"""
        import shutil
        for size_name, temp_dir in datasets.items():
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 已清理临时目录: {temp_dir}")
            except Exception as e:
                print(f"⚠️ 清理临时目录失败 {temp_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="CPU vs GPU 性能对比基准测试"
    )
    parser.add_argument(
        "input_path",
        help="输入数据路径"
    )
    parser.add_argument(
        "--aircraft-db",
        help="航空器数据库路径",
        default=None
    )
    parser.add_argument(
        "--output-dir",
        help="输出目录",
        default="benchmark_results"
    )
    parser.add_argument(
        "--dataset-sizes",
        nargs='+',
        help="要测试的数据集大小",
        default=['small', 'medium', 'large']
    )
    
    args = parser.parse_args()
    
    # 检查处理器可用性
    if not CPU_AVAILABLE and not GPU_AVAILABLE:
        print("❌ CPU和GPU处理器都不可用，请检查安装")
        return
    
    # 运行基准测试
    benchmark = PerformanceBenchmark(args.dataset_sizes)
    results = benchmark.run_comparison(
        base_path=args.input_path,
        aircraft_db=args.aircraft_db,
        output_dir=args.output_dir
    )
    
    print(f"\n🎉 基准测试完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()