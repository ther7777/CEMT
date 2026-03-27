import os
import json
import argparse
import re
import importlib.util
from typing import Optional, Tuple
from vllm import LLM, SamplingParams
from tqdm import tqdm
import glob
import torch
from transformers import AutoTokenizer

# Set environment variables for vLLM sleep mode
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("COMPILE_CUSTOM_KERNELS", "1")

if importlib.util.find_spec("modelscope") is not None:
    os.environ["VLLM_USE_MODELSCOPE"] = "True"

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
    "hi": "Hindi"
}

CODE_TO_ZH_LANG = {
    "ar": "阿拉伯语", "cs": "捷克语", "de": "德语", "en": "英语", "es": "西班牙语",
    "fr": "法语", "it": "意大利语", "ja": "日语", "ko": "韩语", "ru": "俄语", "zh": "中文",
    "uk": "乌克兰语", "is": "冰岛语", "hi": "印地语"
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

def create_prompt_from_sample(example: dict, template: str) -> str:
    try:
        source_text = example['source']
        pair = example['pair']
        
        # 准备所有可能用到的参数
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

def extract_translation_after_think(full_model_output: str) -> Tuple[str, str]:
    content = full_model_output.strip()
    
    # 1. Extract after thought/think
    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    elif "</thought>" in content:
        content = content.split("</thought>", 1)[1].strip()
        
    # 2. Extract inside output/answer if present
    # DRT: <output>...</output>
    # TAT: <answer>...</answer>
    
    # Check for <output> tag
    output_match = re.search(r'<output>(.*?)</output>', content, re.DOTALL | re.IGNORECASE)
    if output_match:
        content = output_match.group(1).strip()
    else:
        # Check for <answer> tag
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
        if answer_match:
            content = answer_match.group(1).strip()
            
    return content, full_model_output.strip()

def build_prompt(prompt: str, tokenizer: Optional[AutoTokenizer], use_chat_template: bool, system_prompt: Optional[str]) -> str:
    if not use_chat_template or tokenizer is None:
        return prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 增加 template 参数传给 create_prompt
def process_data(input_data, llm, sampling_params, template, tokenizer=None, use_chat_template=False, system_prompt=None):
    prompts, original_data = [], []
    
    for item in input_data:
        prompt = create_prompt_from_sample(item, template)
        prompts.append(build_prompt(prompt, tokenizer, use_chat_template, system_prompt))
        original_data.append(item)
    
    if not prompts: return []
    
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for i, output in enumerate(outputs):
        item = original_data[i]
        generated_text = output.outputs[0].text
        
        clean_translation, full_response = extract_translation_after_think(generated_text)

        result = {
            'id': i,
            'source_text': item.get('source', ''),
            'reference_translation': item.get('target', ''),
            'generated_translation': clean_translation,
            'lg': item.get('pair', ''),
            'full_response': full_response,
            'feature_report': item.get('feature_report', {})
        }
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="vllm 推理脚本")
    
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--input", type=str, required=True, help="输入 .jsonl 路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    
    # Prompt 文件路径
    parser.add_argument("--prompt-file", type=str, required=True, help="外部 Prompt 模板文件路径 (.txt)")
    
    # 适配自定义模型代码
    parser.add_argument("--trust-remote-code", action="store_true", help="是否允许执行模型仓库中的远程代码 (TeleChat2 需要)")

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--system-prompt", type=str, default=None)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载 Prompt 模板
    print(f"Loading prompt template from: {args.prompt_file}")
    prompt_template = load_prompt_template(args.prompt_file)

    torch.cuda.empty_cache()
    
    tokenizer = None
    if args.use_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token


    llm = LLM(
        model=args.model, 
        tensor_parallel_size=args.tensor_parallel_size, 
        gpu_memory_utilization=args.gpu_memory_utilization, 
        max_model_len=args.max_model_len, 
        enable_sleep_mode=True,
        trust_remote_code=args.trust_remote_code 
    )
    
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    
    print(f"\n--- 正在处理文件: {args.input} ---")
    base_name = os.path.basename(args.input)
    output_file = os.path.join(args.output_dir, f"result_{os.path.splitext(base_name)[0]}.json")
    
    data = load_jsonl_file(args.input)
    
    all_results = []
    for i in tqdm(range(0, len(data), args.batch_size), desc="Processing batches"):
        batch = data[i:i+args.batch_size]
        batch_results = process_data(batch, llm, sampling_params, template=prompt_template, tokenizer=tokenizer, use_chat_template=args.use_chat_template, system_prompt=args.system_prompt)
        all_results.extend(batch_results)
    
    save_json_file(all_results, output_file)
    print(f"结果已保存至: {output_file}")

    llm.sleep(level=2)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
