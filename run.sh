#!/bin/bash
# run.sh -j${nproc}
# 全流程：原始数据 -> 拆分 -> ECEF转换 -> 特征提取 -> 标签合并

set -e  # 出错时立即退出
cd "$(dirname "$0")"

# 默认参数
DATASET_NAME="7days1"


# 路径定义
RAW_DATA_DIR="data/${DATASET_NAME}/raw_data"
TMP_DIR="data/${DATASET_NAME}/processed_data/tmp"
ECEF_DIR="data/${DATASET_NAME}/processed_data/ecef"
OUTPUT_DIR="outputs"

echo ">> Cleaning previous outputs..."
rm -rf "$TMP_DIR" "$ECEF_DIR" "$OUTPUT_DIR"
mkdir -p "$TMP_DIR" "$ECEF_DIR" "$OUTPUT_DIR"

# 步骤 1：拆分数据
echo "-----------------------------------------------"
echo ">> Splitting raw CSVs into per-ID files"
python3 src/sort.py "$RAW_DATA_DIR" "$TMP_DIR"

# 步骤 2：转换为 ECEF
echo "-----------------------------------------------"
echo ">> Converting per-ID CSVs to ECEF coordinates"
for file in "$TMP_DIR"/*.csv; do
    if [ -f "$file" ]; then
        python3 src/ecef.py "$file" "$ECEF_DIR"
    fi
done

# 步骤 2.5: 推荐可视化路线以及切分原理，确保数据点的持续性，以及切分后可以提取出特征
TEST_FILE_RAW="${TMP_DIR}/11217673.csv"
TEST_FILE="${ECEF_DIR}/11217673.csv"
python3 src/plot/plot_raw.py ${TEST_FILE_RAW}
python3 src/plot/plot.py ${TEST_FILE}

# 步骤 3：块级特征提取（基于 ECEF）
echo "-----------------------------------------------"
echo ">> Extracting block-level features from ECEF data"
python3 src/process_blocks.py --input_dir "$ECEF_DIR" --output_file "$OUTPUT_DIR/plane_features.csv"

# 步骤 4：标签合并（ICAO）
echo "-----------------------------------------------"
echo ">> Merging aircraft database info"
python3 src/complete_info.py --features "$OUTPUT_DIR/plane_features.csv" \
                           --database "data/aircraftDatabase.csv" \
                           --output "$OUTPUT_DIR/plane_features_labeled.csv"

python3 src/filter_outliers.py ${OUTPUT_DIR}/plane_features_labeled.csv ${OUTPUT_DIR}/plane_features_reduced.csv

# 可选清理
rm -rf "$TMP_DIR" "$ECEF_DIR"
echo "-----------------------------------------------"
echo ">> Pipeline completed!"

