#!/bin/bash
# run_benchmark.sh
# CPU vs CPU并行 性能对比运行脚本（已去掉GPU）

set -e
cd "$(dirname "$0")"

# 默认参数
DATASET_NAME="111_days"
SAMPLE_SIZE=100
MIN_ROWS=1000
MODE="benchmark"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}==========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}==========================================${NC}"
}
print_success(){ echo -e "${GREEN}✅ $1${NC}"; }
print_warning(){ echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error(){ echo -e "${RED}❌ $1${NC}"; }
print_info(){ echo -e "${BLUE}ℹ️  $1${NC}"; }

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset) DATASET_NAME="$2"; shift 2 ;;
    --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
    --min-rows) MIN_ROWS="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    -h|--help)
      echo "用法: $0 [选项]"
      echo ""
      echo "选项:"
      echo "  --dataset NAME        数据集名称 (默认: 111_days)"
      echo "  --sample-size SIZE    每个特征块的样本数 (默认: 100)"
      echo "  --min-rows ROWS       最小行数阈值 (默认: 1000)"
      echo "  --mode MODE           运行模式: cpu|cpu_parallel|benchmark (默认: benchmark)"
      echo "  -h, --help            显示此帮助信息"
      echo ""
      echo "运行模式:"
      echo "  cpu           仅运行单线程CPU版本"
      echo "  cpu_parallel  运行并行CPU版本"
      echo "  benchmark     顺序运行CPU与CPU并行并给出对比"
      echo ""
      echo "示例:"
      echo "  $0 --mode cpu"
      echo "  $0 --mode cpu_parallel"
      echo "  $0 --mode benchmark"
      exit 0
      ;;
    *) print_error "未知选项: $1"; echo "使用 -h 或 --help 查看帮助"; exit 1 ;;
  esac
done

# 路径定义
RAW_DATA_DIR="data/${DATASET_NAME}/raw_data"
OUTPUT_DIR="outputs"
AIRCRAFT_DB="src/aircraftDatabase.csv"
BENCHMARK_DIR="benchmark_results"

print_header "🚀 航空数据处理 - CPU vs CPU并行 性能对比"
echo "数据集: ${DATASET_NAME}"
echo "样本大小: ${SAMPLE_SIZE}"
echo "最小行数: ${MIN_ROWS}"
echo "运行模式: ${MODE}"
echo "原始数据: ${RAW_DATA_DIR}"

# 检查输入目录
if [ ! -d "$RAW_DATA_DIR" ]; then
  print_error "找不到原始数据目录: ${RAW_DATA_DIR}"
  exit 1
fi

# 检查Python脚本存在性
check_script() {
  local script_path="$1"
  local name="$2"
  if [ ! -f "$script_path" ]; then
    print_error "找不到 ${name}: ${script_path}"
    return 1
  fi
  return 0
}

# 数据库参数
DB_PARAM=""
if [ -f "$AIRCRAFT_DB" ]; then
  DB_PARAM="--aircraft-db $AIRCRAFT_DB"
  print_success "找到航空器数据库: ${AIRCRAFT_DB}"
else
  print_warning "未找到航空器数据库: ${AIRCRAFT_DB}"
fi

# 基础依赖（不再检查RAPIDS）
check_dependencies() {
  print_info "检查Python依赖..."
  python3 -c "import pandas, numpy, scipy" 2>/dev/null || {
    print_error "缺少基本依赖: pandas, numpy, scipy"
    echo "安装命令: pip install pandas numpy scipy"
    return 1
  }
  return 0
}

# 单线程CPU
run_cpu() {
  print_header "🖥️  CPU处理模式"
  if ! check_script "src/cpu_process.py" "CPU处理脚本"; then return 1; fi
  mkdir -p "$OUTPUT_DIR"
  local t0=$(date +%s)
  print_info "开始CPU数据处理..."
  python3 src/cpu_process.py \
    "$RAW_DATA_DIR" \
    "$OUTPUT_DIR/cpu_features.csv" \
    $DB_PARAM \
    --sample-size "$SAMPLE_SIZE" \
    --min-rows "$MIN_ROWS"
  local t1=$(date +%s); local dur=$((t1 - t0))
  if [ -f "$OUTPUT_DIR/cpu_features.csv" ]; then
    local rows=$(tail -n +2 "$OUTPUT_DIR/cpu_features.csv" | wc -l)
    local size=$(du -h "$OUTPUT_DIR/cpu_features.csv" | cut -f1)
    print_success "CPU处理完成!"
    echo "   ⏱️  处理时间: ${dur} 秒"
    echo "   📊 特征数量: ${rows} 行"
    echo "   💾 文件大小: ${size}"
    echo "   📁 输出文件: ${OUTPUT_DIR}/cpu_features.csv"
    CPU_TIME=$dur; CPU_ROWS=$rows
    return 0
  else
    print_error "CPU处理失败"; return 1
  fi
}

# CPU并行
run_cpu_parallel() {
  print_header "🖥️  CPU并行处理模式"
  # 优先根目录脚本，若无则回退到 src/
  local par_script="cpu_parallel.py"
  if [ ! -f "$par_script" ]; then par_script="src/cpu_parallel.py"; fi
  if ! check_script "$par_script" "CPU并行处理脚本"; then return 1; fi
  mkdir -p "$OUTPUT_DIR"
  local t0=$(date +%s)
  print_info "开始CPU并行数据处理..."
  python3 "$par_script" \
    "$RAW_DATA_DIR" \
    "$OUTPUT_DIR/cpu_parallel_features.csv" \
    $DB_PARAM \
    --sample-size "$SAMPLE_SIZE" \
    --min-rows "$MIN_ROWS"
  local t1=$(date +%s); local dur=$((t1 - t0))
  if [ -f "$OUTPUT_DIR/cpu_parallel_features.csv" ]; then
    local rows=$(tail -n +2 "$OUTPUT_DIR/cpu_parallel_features.csv" | wc -l)
    local size=$(du -h "$OUTPUT_DIR/cpu_parallel_features.csv" | cut -f1)
    print_success "CPU并行处理完成!"
    echo "   ⏱️  处理时间: ${dur} 秒"
    echo "   📊 特征数量: ${rows} 行"
    echo "   💾 文件大小: ${size}"
    echo "   📁 输出文件: ${OUTPUT_DIR}/cpu_parallel_features.csv"
    PAR_TIME=$dur; PAR_ROWS=$rows
    return 0
  else
    print_error "CPU并行处理失败"; return 1
  fi
}

# 对比模式（顺序执行两者并生成简表）
run_benchmark() {
  print_header "📊 CPU vs CPU并行 性能对比模式"
  mkdir -p "$BENCHMARK_DIR"
  run_cpu || { print_error "CPU阶段失败"; return 1; }
  run_cpu_parallel || { print_error "CPU并行阶段失败"; return 1; }
  # 简要对比汇总
  local speedup="N/A"
  if [ -n "$CPU_TIME" ] && [ -n "$PAR_TIME" ] && [ "$PAR_TIME" -gt 0 ]; then
    speedup=$(python3 - <<PY
cpu=$CPU_TIME
par=$PAR_TIME
print(round(cpu/par, 2))
PY
)
  fi
  echo "mode,time_sec,rows" > "$BENCHMARK_DIR/cpu_vs_parallel.csv"
  echo "cpu,$CPU_TIME,$CPU_ROWS" >> "$BENCHMARK_DIR/cpu_vs_parallel.csv"
  echo "cpu_parallel,$PAR_TIME,$PAR_ROWS" >> "$BENCHMARK_DIR/cpu_vs_parallel.csv"
  print_success "对比完成!"
  echo "   ⏱️  CPU: ${CPU_TIME}s, 并行: ${PAR_TIME}s"
  echo "   🚀  加速比(约): ${speedup}x"
  echo "   📄  明细: ${BENCHMARK_DIR}/cpu_vs_parallel.csv"
  return 0
}

main() {
  if ! check_dependencies; then exit 1; fi
  case "$MODE" in
    cpu) run_cpu ;;
    cpu_parallel) run_cpu_parallel ;;
    benchmark) run_benchmark ;;
    *) print_error "未知运行模式: $MODE"; echo "支持的模式: cpu, cpu_parallel, benchmark"; exit 1 ;;
  esac
  local code=$?
  if [ $code -eq 0 ]; then
    print_header "🎉 所有任务完成!"
    echo "📁 可用输出:"
    for f in "$OUTPUT_DIR"/*.csv "$BENCHMARK_DIR"/*.csv; do
      [ -f "$f" ] && echo "   $(ls -lh "$f" | awk '{print $9, "("$5")"}')"
    done
  else
    print_error "执行过程中出现错误"
  fi
  exit $code
}
main
