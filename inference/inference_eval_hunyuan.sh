#!/bin/bash

INFERENCE_BACKEND="vllm"
TRUST_REMOTE_CODE="true"
TRUST_FLAG="--trust-remote-code"

export CUDA_VISIBLE_DEVICES=0,1

# --- 模型配置 ---
MODELS=( 
    "hunyuan_mt_7b"
)
MODEL_PATHS=(
    "path/to/hunyuan_mt_7b/model"
)

# --- 评估指标模型路径 ---
comet_model_path="path/to/xcomet/model"
comet_free_model_path="path/to/cometkiwi/model"

# --- 推理参数 (参考 Hunyuan 官方推荐) ---
TENSOR_PARALLEL_SIZE=2
TEMPERATURE=0.7
TOP_P=0.6
MAX_TOKENS=4096
BATCH_SIZE=16
repetition_penalty=1

# --- 输入与输出目录 ---
BASE_SAVE_DIR="results/hunyuan_eval"
INPUT_DIR="data/test/main_test"
all_language_pairs="en-zh zh-en"

# ===================================================================================
#                                 脚本主逻辑
# ===================================================================================

INFER_SCRIPT=""
BACKEND_ARGS=()

case "$INFERENCE_BACKEND" in
    vllm)
        INFER_SCRIPT="inference/infer_vllm.py"
        BACKEND_ARGS=(
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
            --gpu-memory-utilization 0.85
            --max-model-len 8192
            --use-chat-template
            $TRUST_FLAG
        )
        ;;
    hf)
        INFER_SCRIPT="inference/infer_hf.py"
        BACKEND_ARGS=(
            --device-map auto
            --dtype bfloat16
            --max-input-tokens 4096
            --use-chat-template
            $TRUST_FLAG
        )
        ;;
    *)
        echo "错误: 无效的 INFERENCE_BACKEND \"$INFERENCE_BACKEND\"。"
        exit 1
        ;;
esac

echo "使用推理后端: $INFERENCE_BACKEND"

# 对每个模型执行推理和评估
for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$i]}"
    MODEL_PATH="${MODEL_PATHS[$i]}"
    SAVE_DIR="${BASE_SAVE_DIR}/${MODEL_NAME}"
    
    echo "==============================================================="
    echo "正在处理模型: ${MODEL_NAME}"
    echo "模型路径: ${MODEL_PATH}"
    
    mkdir -p "$SAVE_DIR"
    
    # 步骤 1: 运行推理
    echo "--- 开始 ${INFERENCE_BACKEND} 推理... ---"
    
    for test_pair in $all_language_pairs; do
        src=$(echo "${test_pair}" | cut -d "-" -f 1)
        tgt=$(echo "${test_pair}" | cut -d "-" -f 2)
        
        INPUT_PATTERN="${INPUT_DIR}/*${src}${tgt}.jsonl"
        INPUT_FILES=( $INPUT_PATTERN )
        
        if [ ${#INPUT_FILES[@]} -eq 0 ]; then
            echo "警告: 未找到匹配 ${INPUT_PATTERN} 的文件"
            continue
        fi
        
        INPUT_PATH="${INPUT_FILES[0]}"
        OUTPUT_DIR="${SAVE_DIR}/${test_pair}"
        mkdir -p "$OUTPUT_DIR"

        if [[ "$src" == "zh" || "$tgt" == "zh" ]]; then
            CURRENT_PROMPT_FILE="prompts/templates/hunyuan_to_zh.txt"
            echo "语言对含中文 ($test_pair)，使用中文模板: $CURRENT_PROMPT_FILE"
        else
            CURRENT_PROMPT_FILE="prompts/templates/hunyuan_to_non_zh.txt"
            echo "语言对不含中文 ($test_pair)，使用英文模板: $CURRENT_PROMPT_FILE"
        fi
        
        echo "正在处理语言对 ${test_pair}, 输入文件: ${INPUT_PATH}"
        
        python "$INFER_SCRIPT" \
            --model "$MODEL_PATH" \
            --temperature "$TEMPERATURE" \
            --top-p "$TOP_P" \
            --max-tokens "$MAX_TOKENS" \
            --input "$INPUT_PATH" \
            --output-dir "$OUTPUT_DIR" \
            --batch-size "$BATCH_SIZE" \
            --prompt-file "$CURRENT_PROMPT_FILE" \
            "${BACKEND_ARGS[@]}"

        echo "推理完成! ${test_pair} 的结果已保存至 ${OUTPUT_DIR}"
    done

    # 步骤 2: 评估翻译质量
    echo "--- 开始翻译质量评估... ---"
    
    for test_pair in $all_language_pairs; do
        src=$(echo "${test_pair}" | cut -d "-" -f 1)
        tgt=$(echo "${test_pair}" | cut -d "-" -f 2)
        
        OUTPUT_DIR="${SAVE_DIR}/${test_pair}"
        json_files=( "$OUTPUT_DIR"/*.json )
        
        if [ ${#json_files[@]} -eq 0 ]; then
            echo "警告: 在 ${OUTPUT_DIR} 中未找到JSON输出文件"
            continue
        fi
        
        json_file="${json_files[0]}"
        src_dir="${SAVE_DIR}/${test_pair}/texts"
        mkdir -p "${src_dir}"
        
        src_path="${src_dir}/all_source.txt"
        tgt_path="${src_dir}/all_target.txt"
        output_path="${src_dir}/translations.txt"
        
        python inference/eval/extract_to_eval.py "${json_file}" "${src_path}" "${output_path}" "${tgt_path}"
        
        if [ ! -s "${output_path}" ]; then
            echo "警告: 提取的翻译文件为空, 跳过评估: ${output_path}"
            continue
        fi

        if [ "${tgt}" = "zh" ]; then TOK="zh"; else TOK="13a"; fi
        
        echo "-------------------- ${test_pair} (${MODEL_NAME}) 的评估结果 --------------------"
        
        SACREBLEU_FORMAT=text sacrebleu -tok "${TOK}" -w 2 "${tgt_path}" < "${output_path}" > "${output_path}.bleu"
        cat "${output_path}.bleu"
        
        comet-score -s "${src_path}" -t "${output_path}" -r "${tgt_path}" --batch_size 64 --model "${comet_model_path}" --gpus 1 > "${output_path}.xcomet"
        comet-score -s "${src_path}" -t "${output_path}" --batch_size 64 --model "${comet_free_model_path}" --gpus 1 > "${output_path}.cometkiwi"
        
        echo "--------------------------- ${src}-${tgt} (${MODEL_NAME}) 最终得分 ---------------------------"
        cat "${output_path}.bleu"
        tail -n 1 "${output_path}.xcomet"
        tail -n 1 "${output_path}.cometkiwi"
        
        # 复制结果到顶层目录
        cp "${output_path}.bleu" "${SAVE_DIR}/${MODEL_NAME}-${src}-${tgt}.bleu"
        cp "${output_path}.xcomet" "${SAVE_DIR}/${MODEL_NAME}-${src}-${tgt}.xcomet" 
        cp "${output_path}.cometkiwi" "${SAVE_DIR}/${MODEL_NAME}-${src}-${tgt}.cometkiwi"
    done
    
    # 步骤 3: 汇总所有指标分数
    echo "--- 汇总所有指标分数... ---"
    python inference/eval/count_metric_score.py "${SAVE_DIR}"

    # 如果使用HF后端，清理CUDA缓存
    if [ "$INFERENCE_BACKEND" = "hf" ]; then
        echo "--- 清理CUDA缓存 (HF后端) ---"
        python - <<'PY'
import gc, torch
try:
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(f"清理缓存时出错: {e}")
PY
    fi
    
    echo "模型 ${MODEL_NAME} 的评估已完成! 结果保存在 ${SAVE_DIR}"
done

echo "==============================================================="
echo "所有模型的评估均已完成!"
