import os
import json
import argparse
import torch
from tqdm import tqdm
from typing import Tuple, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. 基础配置与辅助函数 ---

PAIR_TEXT_MAP = {
    "zh-en": "Chinese to English",
    "en-zh": "English to Chinese"
}

CODE_TO_LANG = {
    "zh": "Chinese",
    "en": "English",
    "de": "German",
    "fr": "French",
    "cs": "Czech",
    "is": "Icelandic",
    "ja": "Japanese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "es": "Spanish",
    "hi": "Hindi",
}

CODE_TO_ZH_LANG = {
    "ar": "阿拉伯语",
    "cs": "捷克语",
    "de": "德语",
    "en": "英语",
    "es": "西班牙语",
    "fr": "法语",
    "it": "意大利语",
    "ja": "日语",
    "ko": "韩语",
    "ru": "俄语",
    "zh": "中文",
    "uk": "乌克兰语",
    "is": "冰岛语",
    "hi": "印地语",
}

def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: data.append(json.loads(line))
    return data

def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_prompt_template(file_path: str) -> str:
    """严格读取外部 Prompt 文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[错误] 找不到 Prompt 模板文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# --- 2. 核心 Prompt 构建逻辑 ---
def create_prompt_from_sample(example: dict, template: str) -> str:
    try:
        source_text = example['source']
        pair = example['pair']
        
        pair_text = PAIR_TEXT_MAP.get(pair, "Professional")
        
        src_code, tgt_code = pair.split('-') if '-' in pair else (None, None)
        source_lang = CODE_TO_LANG.get(src_code, src_code) if src_code else "Unknown"
        target_lang = CODE_TO_LANG.get(tgt_code, tgt_code) if tgt_code else "Unknown"

        source_lang_zh = CODE_TO_ZH_LANG.get(src_code, src_code) if src_code else "未知语言"
        target_lang_zh = CODE_TO_ZH_LANG.get(tgt_code, tgt_code) if tgt_code else "未知语言"

        # Treat prompt placeholders as canonical: provide aliases for common variants.
        params = {
            "pair_text": pair_text,
            "source_text": source_text,
            "pair": pair,

            # Language names
            "src_lang": source_lang,
            "tgt_lang": target_lang,
            "source_lang": source_lang,
            "target_lang": target_lang,

            # Language codes
            "src_lang_code": src_code or "",
            "tgt_lang_code": tgt_code or "",

            # Chinese language names (some templates use *_zh, some use src/tgt_*_zh)
            "src_lang_zh": source_lang_zh,
            "tgt_lang_zh": target_lang_zh,
            "source_lang_zh": source_lang_zh,
            "target_lang_zh": target_lang_zh,
        }
        
        if "{feature_analysis_json}" in template:
            if 'feature_report' not in example:
                raise KeyError("Prompt 模板要求 {feature_analysis_json}，但输入数据缺少 'feature_report' 字段")
            feature_report = example['feature_report']
            params["feature_analysis_json"] = json.dumps(feature_report, ensure_ascii=False, indent=2)

        prompt = template.format(**params)
        return prompt

    except KeyError as e:
        raise KeyError(f"[致命错误] 数据字段与 Prompt 模板不匹配: {e}。样本ID: {example.get('id', 'unknown')}")
    except Exception as e:
        raise RuntimeError(f"[致命错误] Prompt 生成失败: {e}")

# --- 3. 输出解析逻辑 ---
def extract_translation_after_think(full_model_output: str) -> Tuple[str, str]:
    if "</think>" in full_model_output:
        parts = full_model_output.split("</think>", 1)
        clean_translation = parts[1].strip()
        return clean_translation, full_model_output.strip()
    else:
        return full_model_output.strip(), full_model_output.strip()

# --- 4. HF 专用批量处理函数 ---
def process_batch_hf(
    batch_data: List[dict], 
    model, 
    tokenizer, 
    template: str, 
    generation_config: dict, 
    use_chat_template: bool, 
    system_prompt: str = None
):
    prompts = []
    valid_indices = [] 
    
    for i, item in enumerate(batch_data):
        try:
            raw_prompt = create_prompt_from_sample(item, template)
            
            final_prompt = raw_prompt
            if use_chat_template:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": raw_prompt})
                final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            prompts.append(final_prompt)
            valid_indices.append(i)
        except Exception as e:
            raise e

    if not prompts:
        return []

    # Tokenization & Padding
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_lengths = inputs.input_ids.shape[1]

    # Generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decoding
    results = []
    generated_tokens = outputs[:, input_lengths:]
    decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    for idx, generated_text in enumerate(decoded_texts):
        original_item_idx = valid_indices[idx]
        item = batch_data[original_item_idx]
        
        clean_translation, full_response = extract_translation_after_think(generated_text)
        
        result = {
            'id': original_item_idx, 
            'source_text': item.get('source', ''),
            'reference_translation': item.get('target', ''),
            'generated_translation': clean_translation,
            'lg': item.get('pair', ''),
            'full_response': full_response,
            'feature_report': item.get('feature_report', {})
        }
        results.append(result)

    return results

# --- 5. 主函数 ---
def main():
    parser = argparse.ArgumentParser(description="[MT-Fighting] HF 推理脚本")
    
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--input", type=str, required=True, help="输入 .jsonl 路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--prompt-file", type=str, required=True, help="外部 Prompt 模板文件路径 (.txt)")
    
    parser.add_argument("--trust-remote-code", action="store_true", help="是否允许执行模型仓库中的远程代码")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    
    parser.add_argument("--max-tokens", type=int, default=1024, help="对应 max_new_tokens") 
    
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=8)
    
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--system-prompt", type=str, default=None)

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading prompt template from: {args.prompt_file}")
    prompt_template = load_prompt_template(args.prompt_file)

    print(f"Loading model from: {args.model}")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        trust_remote_code=args.trust_remote_code,
        padding_side="left"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code
    )
    model.eval()

    # [关键修复] 使用 args.max_tokens 赋值给 max_new_tokens
    generation_config = {
        "max_new_tokens": args.max_tokens, 
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.temperature > 0,
    }

    print(f"\n--- 正在处理文件: {args.input} ---")
    base_name = os.path.basename(args.input)
    output_file = os.path.join(args.output_dir, f"result_{os.path.splitext(base_name)[0]}.json")
    
    data = load_jsonl_file(args.input)
    all_results = []

    for i in tqdm(range(0, len(data), args.batch_size), desc="Processing batches (HF)"):
        batch = data[i:i+args.batch_size]
        
        batch_results = process_batch_hf(
            batch_data=batch,
            model=model,
            tokenizer=tokenizer,
            template=prompt_template,
            generation_config=generation_config,
            use_chat_template=args.use_chat_template,
            system_prompt=args.system_prompt
        )
        
        for idx, res in enumerate(batch_results):
            res['id'] = i + idx
            
        all_results.extend(batch_results)

    save_json_file(all_results, output_file)
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    main()
