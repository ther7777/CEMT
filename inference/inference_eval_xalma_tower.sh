#!/bin/bash

# 1. 显卡设置
export CUDA_VISIBLE_DEVICES=0,1

# 2. 模型定义 (数组一一对应)

MODELS=(
    "X-ALMA-13B-Group6"
    "TowerInstruct-7B-v0.2"
)

MODEL_PATHS=(
    "path/to/trans_model/X_ALMA_13B_Group6_local/"
    "path/to/trans_model/UnbabelTowerInstruct-7B-v0.2/"
)

MODEL_TYPES=(
    "xalma"
    "tower"
)

# 3. 评估工具与 Metric 模型路径
EXTRACT_SCRIPT="inference/eval/extract_to_eval.py"
COUNT_SCRIPT="inference/eval/count_metric_score.py"
COMET_MODEL="path/to/xcomet/model"
COMET_FREE_MODEL="path/to/cometkiwi/model"

# 4. 输入数据与输出目录
INPUT_DIR="data/test/main_test"
BASE_SAVE_DIR="results/xalma_tower_eval"



# Python 推理脚本路径
INFER_SCRIPT="inference/infer_xalma_tower.py"


# 遍历所有配置的模型
for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$i]}"
    MODEL_PATH="${MODEL_PATHS[$i]}"
    MODEL_TYPE="${MODEL_TYPES[$i]}"
    
    SAVE_DIR="${BASE_SAVE_DIR}/${MODEL_NAME}"
    
    echo "正在处理模型 [${i}]: ${MODEL_NAME}"
    echo "类型: ${MODEL_TYPE}"
    echo "路径: ${MODEL_PATH}"
    
    mkdir -p "$SAVE_DIR"
    
    # --- 步骤 1: 运行推理 ---
    echo "--- 开始推理... ---"
    
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
        
        for INPUT_PATH in "${INPUT_FILES[@]}"; do
            # 构建输出文件名 (将 .jsonl 替换为 .json)
            input_filename=$(basename "$INPUT_PATH")
            base_name="${input_filename%.jsonl}"
            output_filename="${base_name}.json"
            OUTPUT_JSON="${SAVE_DIR}/${test_pair}/${output_filename}"
            
            echo "正在推理: ${test_pair} -> ${OUTPUT_JSON}"
            
            # 调用 Python 脚本
            python "$INFER_SCRIPT" \
                --model-path "$MODEL_PATH" \
                --model-type "$MODEL_TYPE" \
                --input-file "$INPUT_PATH" \
                --output-file "$OUTPUT_JSON" \
                --batch-size 8 

            # --- 步骤 2: 评估翻译质量 ---
            echo "--- 开始评估: ${test_pair} - ${base_name} ---"
            
            OUTPUT_DIR=$(dirname "$OUTPUT_JSON")
            src_txt="${OUTPUT_DIR}/texts/${base_name}_source.txt"
            tgt_txt="${OUTPUT_DIR}/texts/${base_name}_target.txt"
            trans_txt="${OUTPUT_DIR}/texts/${base_name}_translations.txt"
            mkdir -p "${OUTPUT_DIR}/texts"
            
            # 提取结果 (生成 source_text, reference_translation 等字段)
            python "$EXTRACT_SCRIPT" "${OUTPUT_JSON}" "${src_txt}" "${trans_txt}" "${tgt_txt}"
            
            if [ ! -s "${trans_txt}" ]; then
                echo "错误: 提取的翻译文件为空, 跳过评估"
                continue
            fi

            # 计算 BLEU
            if [ "${tgt}" = "zh" ]; then TOK="zh"; else TOK="13a"; fi
            SACREBLEU_FORMAT=text sacrebleu -tok "${TOK}" -w 2 "${tgt_txt}" < "${trans_txt}" > "${trans_txt}.bleu"
            
            # 计算 COMET (XCOMET)
            echo "运行 COMET (XCOMET) 评分..."
            comet-score -s "${src_txt}" -t "${trans_txt}" -r "${tgt_txt}" --batch_size 32 --model "${COMET_MODEL}" --gpus 1 > "${trans_txt}.xcomet"
            
            # 计算 COMET (Kiwi) 
            echo "运行 COMET (Kiwi) 评分..."
            comet-score -s "${src_txt}" -t "${trans_txt}" --batch_size 32 --model "${COMET_FREE_MODEL}" --gpus 1 > "${trans_txt}.cometkiwi"
            
            # 打印分数预览
            echo "-------------------------------------"
            echo "Result Preview (${test_pair} - ${base_name}):"
            echo "BLEU: $(cat ${trans_txt}.bleu)"
            echo "XCOMET: $(tail -n 1 ${trans_txt}.xcomet)"
            echo "Kiwi: $(tail -n 1 ${trans_txt}.cometkiwi)"
            echo "-------------------------------------"
            
            # 复制结果到模型根目录方便查看 (汇总脚本依赖这些文件)
            cp "${trans_txt}.bleu" "${SAVE_DIR}/${MODEL_NAME}-${src}-${tgt}-${base_name}.bleu"
            cp "${trans_txt}.xcomet" "${SAVE_DIR}/${MODEL_NAME}-${src}-${tgt}-${base_name}.xcomet" 
            cp "${trans_txt}.cometkiwi" "${SAVE_DIR}/${MODEL_NAME}-${src}-${tgt}-${base_name}.cometkiwi"
        done
    done
    
    # --- 步骤 3: 汇总所有指标分数 ---
    echo "--- 汇总 ${MODEL_NAME} 指标分数... ---"
    python "$COUNT_SCRIPT" "${SAVE_DIR}"
    
    echo "模型 ${MODEL_NAME} 处理完成!"
done