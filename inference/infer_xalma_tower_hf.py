import os
import json
import argparse
import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================================================================
# 1. 语言名称映射 (Strictly follows Model Cards)
# =========================================================================
LANG_NAME_MAP = {
    "zh": "Chinese",
    "en": "English",
    "de": "German",
    "fr": "French",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "ko": "Korean",
    "it": "Italian",
    "es": "Spanish",
}

# =========================================================================
# 2. 提示词构建器 (Strictly follows Model Card Examples)
# =========================================================================

def build_xalma_prompt(tokenizer, source_text, pair):
    """
    X-ALMA Prompt Logic:
    User: Translate this from {Source} to {Target}:\n{Source}: {text}\n{Target}:
    """
    src_code, tgt_code = pair.split('-')
    src_full = LANG_NAME_MAP.get(src_code, src_code)
    tgt_full = LANG_NAME_MAP.get(tgt_code, tgt_code)

    user_content = f"Translate this from {src_full} to {tgt_full}:\n{src_full}: {source_text}\n{tgt_full}:"
    
    messages = [{"role": "user", "content": user_content}]
    # X-ALMA requires chat template
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return full_prompt

def build_tower_prompt(tokenizer, source_text, pair):
    """
    TowerInstruct Prompt Logic:
    User: Translate the following text from {Source} into {Target}.\n{Source}: {text}\n{Target}:
    Note: Uses 'into' instead of 'to', and a period '.' instead of colon ':' in the instruction.
    """
    src_code, tgt_code = pair.split('-')
    src_full = LANG_NAME_MAP.get(src_code, src_code)
    tgt_full = LANG_NAME_MAP.get(tgt_code, tgt_code)

    # Strictly following TowerInstruct-7B-v0.2 Model Card example
    user_content = f"Translate the following text from {src_full} into {tgt_full}.\n{src_full}: {source_text}\n{tgt_full}:"
    
    messages = [{"role": "user", "content": user_content}]
    # Tower also uses chat template
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return full_prompt

# =========================================================================
# 3. 数据集处理 (PyTorch Dataset)
# =========================================================================
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, model_type):
        self.data = data
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source = item.get('source', item.get('src', ''))
        pair = item.get('pair', '')
        
        if self.model_type == "xalma":
            prompt = build_xalma_prompt(self.tokenizer, source, pair)
        else:
            prompt = build_tower_prompt(self.tokenizer, source, pair)
            
        return prompt, item

# =========================================================================
# 4. 主推理逻辑
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Run HF Inference for X-ALMA/Tower")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True, choices=["xalma", "tower"])
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # 1. 加载模型与分词器 (Native HF Way)
    logging.info(f"Loading Model: {args.model_path} ({args.model_type})")
    
    # 使用 device_map="auto" 自动分配显卡 (Accelerate)
    # 推荐使用 bfloat16 (Tower 明确推荐)，如果显卡不支持则自动回退
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, padding_side="left",local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto", 
        local_files_only=True,
        torch_dtype=dtype
    )
    
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 准备数据
    raw_data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    
    dataset = TranslationDataset(raw_data, tokenizer, args.model_type)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)

    # 3. 推理循环
    logging.info(f"Starting inference with batch size {args.batch_size}...")
    final_results = []
    
    model.eval()
    for batch in tqdm(dataloader, desc="Inferencing"):
        prompts = [x[0] for x in batch]
        original_items = [x[1] for x in batch]
        
        # 编码
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        # 生成参数设置
        # X-ALMA: num_beams=5, max_new_tokens=20 (example) -> we use reasonable max like 512
        # Tower: max_new_tokens=256, do_sample=False (Greedy)
        
        gen_kwargs = {
            "max_new_tokens": 512,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        if args.model_type == "xalma":
            # X-ALMA 推荐 Beam Search
            gen_kwargs.update({"num_beams": 5, "do_sample": False})
        else:
            # Tower 推荐 Greedy (do_sample=False)
            gen_kwargs.update({"do_sample": False})

        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)
        
        # 解码 (只取生成部分)
        # 截断 Prompt 部分，只保留生成的新 Token
        input_len = inputs.input_ids.shape[1]
        generated_tokens = generated_ids[:, input_len:]
        decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # 收集结果
        for item, trans_text in zip(original_items, decoded_texts):
            # 去除可能残留的系统符
            clean_text = trans_text.strip()
            
            # 保存结果
            item['generated_translation'] = clean_text
            
            # 对齐评估 Key
            if "source_text" not in item:
                item["source_text"] = item.get("source", item.get("src", ""))
            if "reference_translation" not in item:
                item["reference_translation"] = item.get("target", item.get("tgt", item.get("reference", "")))
                
            final_results.append(item)

    # 4. 保存结果
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
        
    logging.info(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()