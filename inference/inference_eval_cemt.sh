#!/usr/bin/env bash
# 用途：使用指定推理后端对 CEMT 模型执行推理，并计算 BLEU、XCOMET、CometKiwi 指标。
# 输入：测试集目录 `data/test/main_test`、提示词 `prompts/templates/cot_full_cemt.txt`、模型路径与 COMET 模型路径。
# 输出：`results/cemt_eval/<model_name>` 下的逐语种结果、文本抽取文件与汇总指标。
# 运行示例：bash inference/inference_eval_cemt.sh
set -euo pipefail

# 设置要使用的推理后端，可选值为 "vllm" 或 "hf"。
INFERENCE_BACKEND="vllm"
PROMPT_FILE="prompts/templates/cot_full_cemt.txt"
TRUST_REMOTE_CODE="true"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODELS=(
    "CEMT"
)
MODEL_PATHS=(
    "path/to/cemt/model"
)
COMET_MODEL_PATH="path/to/xcomet/model"
COMET_FREE_MODEL_PATH="path/to/cometkiwi/model"

TENSOR_PARALLEL_SIZE=2
TEMPERATURE=0
TOP_P=0.95
MAX_TOKENS=3000
BATCH_SIZE=16
ALL_LANGUAGE_PAIRS=(en-zh zh-en)

BASE_SAVE_DIR="results/cemt_eval"
INPUT_DIR="data/test/main_test"

TRUST_FLAG=()
if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
    TRUST_FLAG=(--trust-remote-code)
fi

INFER_SCRIPT=""
BACKEND_ARGS=()

case "${INFERENCE_BACKEND}" in
    vllm)
        INFER_SCRIPT="inference/infer_vllm.py"
        BACKEND_ARGS=(
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
            --gpu-memory-utilization 0.85
            --max-model-len 6000
            --use-chat-template
            --prompt-file "${PROMPT_FILE}"
            "${TRUST_FLAG[@]}"
        )
        ;;
    hf)
        INFER_SCRIPT="inference/infer_hf.py"
        BACKEND_ARGS=(
            --device-map auto
            --dtype bfloat16
            --max-input-tokens 4096
            --use-chat-template
            --prompt-file "${PROMPT_FILE}"
            "${TRUST_FLAG[@]}"
        )
        ;;
    *)
        echo "错误：INFERENCE_BACKEND 只能是 \"vllm\" 或 \"hf\"，当前值为 ${INFERENCE_BACKEND}。"
        exit 1
        ;;
esac

echo "使用推理后端: ${INFERENCE_BACKEND}"

for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$i]}"
    MODEL_PATH="${MODEL_PATHS[$i]}"
    SAVE_DIR="${BASE_SAVE_DIR}/${MODEL_NAME}"

    echo "==============================================================="
    echo "正在处理模型: ${MODEL_NAME}"
    echo "模型路径: ${MODEL_PATH}"

    mkdir -p "${SAVE_DIR}"

    # 步骤 1：执行推理。
    echo "--- 开始 ${INFERENCE_BACKEND} 推理 ---"

    for test_pair in "${ALL_LANGUAGE_PAIRS[@]}"; do
        src="${test_pair%-*}"
        tgt="${test_pair#*-}"

        shopt -s nullglob
        input_files=("${INPUT_DIR}"/*"${src}${tgt}".jsonl)
        shopt -u nullglob

        if [[ "${#input_files[@]}" -eq 0 ]]; then
            echo "警告：未找到匹配 ${INPUT_DIR}/*${src}${tgt}.jsonl 的文件。"
            continue
        fi

        INPUT_PATH="${input_files[0]}"
        OUTPUT_DIR="${SAVE_DIR}/${test_pair}"
        mkdir -p "${OUTPUT_DIR}"

        echo "正在处理语言对 ${test_pair}，输入文件: ${INPUT_PATH}"

        python "${INFER_SCRIPT}" \
            --model "${MODEL_PATH}" \
            --temperature "${TEMPERATURE}" \
            --top-p "${TOP_P}" \
            --max-tokens "${MAX_TOKENS}" \
            --input "${INPUT_PATH}" \
            --output-dir "${OUTPUT_DIR}" \
            --batch-size "${BATCH_SIZE}" \
            "${BACKEND_ARGS[@]}"

        echo "推理完成：${test_pair} 结果已保存到 ${OUTPUT_DIR}"
    done

    # 步骤 2：抽取文本并评估翻译质量。
    echo "--- 开始翻译质量评估 ---"

    for test_pair in "${ALL_LANGUAGE_PAIRS[@]}"; do
        src="${test_pair%-*}"
        tgt="${test_pair#*-}"
        OUTPUT_DIR="${SAVE_DIR}/${test_pair}"

        shopt -s nullglob
        json_files=("${OUTPUT_DIR}"/*.json)
        shopt -u nullglob

        if [[ "${#json_files[@]}" -eq 0 ]]; then
            echo "警告：在 ${OUTPUT_DIR} 中未找到 JSON 输出文件。"
            continue
        fi

        json_file="${json_files[0]}"
        text_dir="${OUTPUT_DIR}/texts"
        mkdir -p "${text_dir}"

        src_path="${text_dir}/all_source.txt"
        tgt_path="${text_dir}/all_target.txt"
        output_path="${text_dir}/translations.txt"

        python inference/eval/extract_to_eval.py "${json_file}" "${src_path}" "${output_path}" "${tgt_path}"

        if [[ ! -s "${output_path}" ]]; then
            echo "警告：提取后的翻译文件为空，跳过评估: ${output_path}"
            continue
        fi

        if [[ "${tgt}" == "zh" ]]; then
            TOK="zh"
        else
            TOK="13a"
        fi

        echo "-------------------- ${test_pair} (${MODEL_NAME}) 评估结果 --------------------"

        SACREBLEU_FORMAT=text sacrebleu -tok "${TOK}" -w 2 "${tgt_path}" < "${output_path}" > "${output_path}.bleu"
        cat "${output_path}.bleu"

        comet-score -s "${src_path}" -t "${output_path}" -r "${tgt_path}" --batch_size 64 --model "${COMET_MODEL_PATH}" --gpus 1 > "${output_path}.xcomet"
        comet-score -s "${src_path}" -t "${output_path}" --batch_size 64 --model "${COMET_FREE_MODEL_PATH}" --gpus 1 > "${output_path}.cometkiwi"

        echo "-------------------- ${src}-${tgt} (${MODEL_NAME}) 最终得分 --------------------"
        cat "${output_path}.bleu"
        tail -n 1 "${output_path}.xcomet"
        tail -n 1 "${output_path}.cometkiwi"

        cp "${output_path}.bleu" "${SAVE_DIR}/${MODEL_NAME}-${src}-${tgt}.bleu"
        cp "${output_path}.xcomet" "${SAVE_DIR}/${MODEL_NAME}-${src}-${tgt}.xcomet"
        cp "${output_path}.cometkiwi" "${SAVE_DIR}/${MODEL_NAME}-${src}-${tgt}.cometkiwi"
    done

    # 步骤 3：汇总指标。
    echo "--- 汇总所有指标分数 ---"
    python inference/eval/count_metric_score.py "${SAVE_DIR}"

    if [[ "${INFERENCE_BACKEND}" == "hf" ]]; then
        echo "--- 清理 CUDA 缓存 (HF 后端) ---"
        python - <<'PY'
import gc
import torch

try:
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(f"清理缓存时出错: {e}")
PY
    fi

    echo "模型 ${MODEL_NAME} 的评估已完成，结果保存在 ${SAVE_DIR}"
done

echo "==============================================================="
echo "所有模型的评估均已完成。"
