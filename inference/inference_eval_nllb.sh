#!/bin/bash

# ================= 配置区域 =================

# 1. NLLB 模型路径 
NLLB_MODEL_PATH="/path/to/your/nllb_model" 
EXTRACT_SCRIPT="inference/eval/extract_to_eval.py"
COUNT_SCRIPT="inference/eval/count_metric_score.py"
COMET_MODEL="path/to/your/xcomet_model"  
COMET_FREE_MODEL="path/to/your/cometkiwi_model"  

# 3. 输入数据目录
INPUT_DIR="data/test/main_test"

# 4. 输出目录
BASE_SAVE_DIR="results/nllb_eval"

# 5. GPU 设置
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPU_IDS="0,1,2,3"

# ===========================================

MODEL_NAME="NLLB-200-3.3B"
SAVE_DIR="${BASE_SAVE_DIR}/${MODEL_NAME}"
mkdir -p "$SAVE_DIR"

echo "==============================================================="
echo "开始运行 NLLB Baseline 评测"
echo "模型: ${MODEL_NAME}"
echo "输出目录: ${SAVE_DIR}"
echo "==============================================================="

# 遍历语言对 (en-zh, zh-en)
all_language_pairs="en-zh zh-en"

for test_pair in $all_language_pairs; do
    src=$(echo "${test_pair}" | cut -d "-" -f 1)
    tgt=$(echo "${test_pair}" | cut -d "-" -f 2)
    
    # 查找输入文件
    INPUT_PATTERN="${INPUT_DIR}/*${src}${tgt}.jsonl"
    INPUT_FILES=( $INPUT_PATTERN )
    
    if [ ${#INPUT_FILES[@]} -eq 0 ]; then
        echo "警告: 未找到匹配 ${INPUT_PATTERN} 的文件"
        continue
    fi
    
    INPUT_PATH="${INPUT_FILES[0]}"
    OUTPUT_DIR="${SAVE_DIR}/${test_pair}"
    # 1. 获取输入文件的文件名 
    input_filename=$(basename "$INPUT_PATH")
    # 2. 把扩展名 .jsonl 换成 .json 
    output_filename="${input_filename%.jsonl}.json"
    # 3. 组合完整路径
    OUTPUT_JSON="${OUTPUT_DIR}/${output_filename}"
    mkdir -p "$OUTPUT_DIR"
    
    # --- 步骤 1: 运行推理 ---
    echo "Processing ${test_pair}..."

    python inference/infer_nllb.py \
        --model-path "$NLLB_MODEL_PATH" \
        --input-file "$INPUT_PATH" \
        --output-file "$OUTPUT_JSON" \
        --gpus "$GPU_IDS" \
        --batch-size 16

    # --- 步骤 2: 评估  ---
    echo "Evaluating ${test_pair}..."
    
    src_dir="${OUTPUT_DIR}/texts"
    mkdir -p "${src_dir}"
    
    src_txt="${src_dir}/all_source.txt"
    tgt_txt="${src_dir}/all_target.txt"
    trans_txt="${src_dir}/translations.txt"
    
    # 提取
    python "$EXTRACT_SCRIPT" "${OUTPUT_JSON}" "${src_txt}" "${trans_txt}" "${tgt_txt}"
    
    if [ ! -s "${trans_txt}" ]; then
        echo "Error: 提取的翻译文件为空!"
        continue
    fi

    # 计算 BLEU
    if [ "${tgt}" = "zh" ]; then TOK="zh"; else TOK="13a"; fi
    SACREBLEU_FORMAT=text sacrebleu -tok "${TOK}" -w 2 "${tgt_txt}" < "${trans_txt}" > "${trans_txt}.bleu"
    
    # 计算 COMET
    comet-score -s "${src_txt}" -t "${trans_txt}" -r "${tgt_txt}" --batch_size 64 --model "${COMET_MODEL}" --gpus 1 > "${trans_txt}.xcomet"
    comet-score -s "${src_txt}" -t "${trans_txt}" --batch_size 64 --model "${COMET_FREE_MODEL}" --gpus 1 > "${trans_txt}.cometkiwi"
    
    # 打印简报
    echo "--- ${test_pair} Results ---"
    cat "${trans_txt}.bleu"
    tail -n 1 "${trans_txt}.xcomet"
    
    # 复制结果文件到上级目录，方便汇总
    cp "${trans_txt}.bleu" "${SAVE_DIR}/${MODEL_NAME}-${src}-${tgt}.bleu"
    cp "${trans_txt}.xcomet" "${SAVE_DIR}/${MODEL_NAME}-${src}-${tgt}.xcomet" 
    cp "${trans_txt}.cometkiwi" "${SAVE_DIR}/${MODEL_NAME}-${src}-${tgt}.cometkiwi"
done

# --- 步骤 3: 汇总所有分数 ---
echo "--- 汇总所有指标 ---"
python "$COUNT_SCRIPT" "${SAVE_DIR}"

echo "NLLB 评测全部完成!"