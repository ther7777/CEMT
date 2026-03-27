import os
import json
import argparse
import torch
import math
from tqdm import tqdm
from torch.multiprocessing import Pool, set_start_method
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

# 语言代码映射 (NLLB 需要特定格式)
LANG_MAP = {
    "zh": "zho_Hans",
    "en": "eng_Latn",
}

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def run_inference_worker(args):
    """
    工作进程：加载模型并处理分配到的数据块
    """
    gpu_id, chunk_data, model_path, batch_size = args
    
    # 1. 配置设备
    device = torch.device(f"cuda:{gpu_id}")
    
    # 2. 加载模型 (每个进程独立加载，3.3B 显存占用很小，完全没问题)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16 
    ).to(device)
    model.eval()
    
    results = []
    
    # 3. Batch 推理
    for i in tqdm(range(0, len(chunk_data), batch_size), desc=f"GPU {gpu_id}", position=gpu_id, leave=False):
        batch = chunk_data[i : i + batch_size]
        
        sources = [item['source'] for item in batch]
        pairs = [item['pair'] for item in batch]
        
        # 解析语言方向 (假设一个 Batch 内的 pair 是一致的，或者以第一个为准)
        # NLLB 需要显式设置 src_lang
        try:
            src_code = pairs[0].split('-')[0]
            tgt_code = pairs[0].split('-')[1]
            nllb_src = LANG_MAP[src_code]
            nllb_tgt = LANG_MAP[tgt_code]
        except KeyError:
            print(f"[Error] 未知的语言代码 in pair: {pairs[0]}")
            continue

        tokenizer.src_lang = nllb_src
        
        # 编码
        inputs = tokenizer(sources, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # 生成
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(nllb_tgt), # 强制指定目标语言
                max_new_tokens=512,
                num_beams=5, # NLLB 推荐 Beam Search = 5
            )
        
        # 解码
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # 写入结果
        for item, pred in zip(batch, decoded_preds):
            item['generated_translation'] = pred
            results.append(item)
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="Available GPU IDs, e.g., 0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # 防止 CUDA 初始化冲突
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    # 1. 准备数据
    data = load_jsonl(args.input_file)
    if not data:
        print(f"输入文件为空: {args.input_file}")
        return

    # 2. 分配任务
    gpu_list = [int(x) for x in args.gpus.split(',')]
    num_gpus = len(gpu_list)
    chunk_size = math.ceil(len(data) / num_gpus)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    worker_args = []
    for i, chunk in enumerate(chunks):
        if i < len(gpu_list):
            worker_args.append((gpu_list[i], chunk, args.model_path, args.batch_size))

    print(f"开始 NLLB 推理: {len(data)} 条样本, 使用 {num_gpus} 张 GPU 并行...")

    # 3. 并行执行
    final_results = []
    with Pool(num_gpus) as pool:
        for chunk_res in pool.map(run_inference_worker, worker_args):
            final_results.extend(chunk_res)

    aligned_results = []
    for item in final_results:
        # 1. 对齐 Source
        if "source_text" not in item:
            # 尝试从常见的 source/src 字段获取
            item["source_text"] = item.get("source", item.get("src", ""))
            
        # 2. 对齐 Reference (目标参考译文)
        if "reference_translation" not in item:
            # 尝试从常见的 target/tgt/reference 字段获取
            item["reference_translation"] = item.get("target", item.get("tgt", item.get("reference", "")))
            
        aligned_results.append(item)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        # 使用 json.dump 保存为列表格式，indent=2 方便阅读
        json.dump(aligned_results, f, ensure_ascii=False, indent=2)
    
    print(f"推理完成，结果已保存至: {args.output_file}")
    print(f"已自动对齐 Keys: source -> source_text, target -> reference_translation")

if __name__ == "__main__":
    main()