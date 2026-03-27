"""
用途：为每条源文本抽取特征报告，并写回带 `feature_report` 的 JSONL。
输入：原始样本 JSONL、特征抽取 prompt、兼容 OpenAI 的模型服务配置。
输出：附加 `feature_report` 字段的 JSONL，以及 `logs/` 下的执行日志。
运行示例：python utils/extract_features.py --input_file data/raw/24_en_zh.jsonl --output_file data/train/24_en_zh.features.jsonl --prompt_file prompts/templates/feature_extraction.txt
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import time
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, Optional, Tuple

import openai
from tqdm import tqdm


def load_prompt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def setup_logging(input_filename: str) -> str:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    log_filename = f"extract_features_{base_name}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    fh = logging.FileHandler(log_filepath, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return log_filepath


FEATURE_EXTRACTION_PROMPT_TEMPLATE = """
# 角色设定
你是一位顶尖的语言学特征分析师。你的任务是基于一套严谨的正交特征体系，对输入的源文本进行解构和编码。

# 核心任务
你的任务是分析给定的源文本，并严格按照以下三个相互独立的正交维度进行判断，生成一个包含“特征编码”和“特征片段”的JSON输出。
请你【严格按照以下正交特征体系定义 】，任何元素被提取之前都应该反复对照规则思考，是否满足要求。

[P0] 核心仲裁原则：语用优先
DNT (维度二) 和语用 (维度三) 是互斥的。如果一个片段（如缩略语 "IMHO" 或 "GGWP"）同时看似两者，【语用元素 (维度三)】永远优先。DNT (维度二) 仅用于没有语言翻译价值、必须严格复制的硬性字面量。

# 正交特征体系定义 
维度一：认知复杂度 (Cognitive Complexity) - [三元分级: 2, 1, 0]
编码为 2 (深度分析): 源文本包含多个、通过复杂逻辑关联的子句（如使用“虽然...但是...”、“一旦...就...”等），或包含深度嵌套的语法结构（如多个定语从句层层包裹）。  
编码为 1 (基础分析): 如果源文本是一个结构完整且内容简单的单句（有明确的主谓结构）。
[关键补充规则]: 这也包括那些虽然省略了主语，但意图明确的祈使句或口语化句子（例如“快把文件发给我”）。我们的原则是，只要句子能被理解为一个可执行的、完整的指令或陈述，就应编码为1以进行基础分析。
编码为 0 (识别与定义): 如果源文本是一个结构不完整的片段（例如单个名词、无核心谓语的短语）。

维度二：绝对DNT元素  - [二元分级: 1, 0]
编码为 1: 如果源文本**严格**包含任一满足【硬性DNT清单】的绝对不可翻译元素。必须在feature_fragments.dnt中抽取出这些**完整、原始的片段**,但是不在【硬性DNT定义】规则范围的不要抽取出来。
    【硬性DNT清单 (Inclusion List)】:
        标识符: URLs (https://...), 电子邮件 (support@...)
        代码相关: 代码片段 (// TODO), API端点 (/api/v3), MIME类型 (application/json)
        技术ID: 错误码 (0x80070005), 版本号 (v1.2.1), 序列号 (SN:...), 固件版本 (FW:...), 专利/标准号 (ISO 27001)
        视觉符号: Emojis (😂), 颜文字 ((^▽^)), 商标/法律符号 (®, ™, ©)
    【严格排除 、白名单】: 绝不能提取常见的英文术语 (如 on-call, in-depth)、可翻译的中文专有名词 (如 “一带一路”, “两会”)、机构名、时间日期或任何计量单位 (如 kg, °F, 220V)，以及【任何可能不翻译，但是不在以上硬性DNT清单中】的，都不提取。这些是应由翻译模型自行处理的常规或复杂词汇。
    必须在 feature_fragments.dnt 中抽取出这些片段。
编码为 0: 如果不包含任一满足【硬性DNT定义】的DNT元素。

维度三：语用元素 (Pragmatic Elements) - [二元分级: 1, 0]
编码为 1: 如果源文本包含任何需要意译的俚语、成语、流行语、文化负载表达等。必须在feature_fragments.pragmatic中抽取出这些片段。
    这包括但不限于：俚语、成语 (如 “画蛇添足”)、网络流行语。
    这也包括具有语用功能的缩略语 (如 IMHO, GGWP) 或口语词汇 (如 “谢了！”，其功能是确认而非感谢)。
    必须在 feature_fragments.pragmatic 中抽取出这些片段。
编码为 0: 如果不包含。


# 强制输出格式
你的输出必须是一个单独的、合法的JSON对象，前后不得有任何额外文本。
--- 输出范例 ---
输入:
Contact support@apple.com or visit https://support.apple.com #Help
输出:
{{
  "feature_code": [1, 1, 0],
  "feature_fragments": {{ "dnt": ["support@apple.com", "https://support.apple.com", "#Help"], "pragmatic": [] }}
}}
输入:
NASA uses AI to analyze data from the Mars rover,lol.
输出:
{{
  "feature_code": [1, 0, 1],
  "feature_fragments": {{ "dnt": [], "pragmatic": ["lol"] }}
}}
输入: 请检查 v1.2.1 版的 /api/auth 接口，错误码 0x80070005。© 2024 CEMT™ 
输出:
{{
  "feature_code": [1, 1, 0],
  "feature_fragments": {{ "dnt": ["v1.2.1", "/api/auth", "0x80070005", "©", "™"], "pragmatic": [] }}
}}
}}
输入: “一带一路”倡议需要 220V 电压，这很重要。 
输出:
{{
  "feature_code": [1, 0, 0],
  "feature_fragments": {{ "dnt": [], "pragmatic": [] }}
}}
输入: 虽然他很努力，但他还是把 v1.1.0 搞砸了。 
输出:
{{
  "feature_code": [2, 1, 1],
  "feature_fragments": {{ "dnt": ["v1.1.0"], "pragmatic": ["搞砸了"] }}
}}

--- 范例结束 ---
请你仔细思考，尤其是检查提取出DNT必须是硬性DNT清单中的一定不可翻译的成分
任务开始:
# 任务输入
源文本: {source_text}
"""


def convert_sets_to_lists(obj):
    """递归地遍历一个对象，将所有set转换为list。"""
    if isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(elem) for elem in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj

def _robust_json_parser_for_extraction(raw_text: str, logger: logging.Logger) -> Optional[Dict]:
    if not isinstance(raw_text, str):
        return None

    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_block_pattern, clean_text, re.DOTALL)
    if match:
        json_text = match.group(1).strip()
    else:
        start_brace = clean_text.find('{')
        end_brace = clean_text.rfind('}')
        if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
            json_text = clean_text[start_brace:end_brace+1]
        else:
            json_text = clean_text

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        logger.warning(f"标准JSON解析失败，转入下一层解析。文本片段: '{json_text[:200]}...'")

    try:
        # Step 1: Parse with AST
        result = ast.literal_eval(json_text)
        if isinstance(result, dict):
            # Step 2: Recursively convert any sets to lists before returning
            final_result = convert_sets_to_lists(result)
            logger.info("AST解析并净化成功。")
            return final_result
    except (ValueError, SyntaxError, MemoryError, TypeError):
        logger.warning(f"AST解析失败，启动最终正则救援模式。")

    try:
        report = {}
        code_match = re.search(r'["\']feature_code["\']\s*:\s*(\[.*?\])', json_text, re.DOTALL)
        if code_match:
            report['feature_code'] = list(ast.literal_eval(code_match.group(1)))
        else:
             report['feature_code'] = [-1, -1, -1]

        fragments_match = re.search(r'["\']feature_fragments["\']\s*:\s*(\{.*?\})', json_text, re.DOTALL)
        if fragments_match:
            fragments_str = fragments_match.group(1)
            fragments = {}
            dnt_match = re.search(r'["\']dnt["\']\s*:\s*(\[.*?|{.*?})', fragments_str, re.DOTALL) # Regex can match list or set
            pragmatic_match = re.search(r'["\']pragmatic["\']\s*:\s*(\[.*?|{.*?})', fragments_str, re.DOTALL)
            fragments['dnt'] = list(ast.literal_eval(dnt_match.group(1))) if dnt_match else []
            fragments['pragmatic'] = list(ast.literal_eval(pragmatic_match.group(1))) if pragmatic_match else []
            report['feature_fragments'] = fragments
        else:
            report['feature_fragments'] = {"dnt": [], "pragmatic": []}

        if 'feature_code' in report and report['feature_code'] != [-1, -1, -1]:
            logger.info(f"正则救援成功！重构的报告: {report}")
            return report
            
    except Exception as e:
        logger.error(f"正则救援模式也失败了: {e}", exc_info=False)

    return None

def call_llm_api(
    prompt: str,
    sample_id_for_log: Any,
    api_base: str,
    api_key: str,
    model_name: str,
    inference_params: Dict[str, Any],
    max_retries: int,
    retry_delay: int,
) -> Optional[Dict]:
    """调用LLM API并使用定制的鲁棒解析器返回字典。"""
    client = openai.OpenAI(base_url=api_base, api_key=api_key, timeout=300.0)

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **inference_params,
            )
            raw_content = response.choices[0].message.content

            parsed_result = _robust_json_parser_for_extraction(raw_content, logging.getLogger())
            if parsed_result:
                return parsed_result
            else:
                last_error = (
                    f"样本ID {sample_id_for_log} 在第 {attempt + 1}/{max_retries} 次尝试中解析失败。"
                    f"LLM原始输出: '{raw_content[:200]}...'"
                )
                logging.warning(last_error)

        except Exception as e:
            last_error = f"样本ID {sample_id_for_log} API调用在第 {attempt + 1}/{max_retries} 次尝试中失败。"
            logging.error(last_error, exc_info=True)

        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    logging.error(f"样本ID {sample_id_for_log} 经所有重试后仍失败。最后一次错误: {last_error}")
    return None

# --- 5. 核心任务逻辑 (改造为增强任务) ---
def process_feature_extraction_task(
    sample: Dict[str, Any],
    prompt_template: str,
    api_cfg: Dict[str, Any],
    inference_params: Dict[str, Any],
    max_retries: int,
    retry_delay: int,
) -> Optional[Dict[str, Any]]:
    """Worker: generate feature_report for one sample."""
    sample_id = sample.get('sample_id', 'N/A')
    try:
        source_text = sample.get('source')
        if not source_text:
            logging.warning(f"样本ID {sample_id} 缺少 'source' 字段，跳过。")
            return None

        prompt = prompt_template.format(source_text=source_text)

        feature_report = call_llm_api(
            prompt,
            sample_id,
            api_base=api_cfg["base"],
            api_key=api_cfg["key"],
            model_name=api_cfg["model"],
            inference_params=inference_params,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        
        if feature_report:
            sample['feature_report'] = feature_report
            return sample
        else:
            return None

    except Exception as e:
        logging.error(f"处理样本ID {sample_id} 时发生未知错误: {e}", exc_info=True)
        return None

def load_processed_ids(filepath: str) -> set:
    ids = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        processed_sample = json.loads(line)
                        if 'sample_id' in processed_sample:
                            ids.add(processed_sample['sample_id'])
                    except json.JSONDecodeError:
                        logging.warning(f"无法解析已存在输出文件 '{filepath}' 中的行: {line.strip()}")
        except Exception as e:
            logging.error(f"读取文件 '{filepath}' 以实现断点续跑时出错: {e}", exc_info=True)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Extract feature reports with an OpenAI-compatible model")
    parser.add_argument("--input_file", required=True, help="Path to input JSONL")
    parser.add_argument("--output_file", required=True, help="Path to output JSONL")
    parser.add_argument("--prompt_file", required=True, help="Path to feature-extraction prompt template")
    parser.add_argument("--api_base", default="http://localhost:8080/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model_name", default="llm")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_delay", type=int, default=5)
    parser.add_argument("--num_processes", type=int, default=32)
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    log_filepath = setup_logging(args.input_file)
    
    logging.info("="*50)
    logging.info(f"--- 开始要素提取任务 ---")
    logging.info(f"  [+] 输入文件: {args.input_file}")
    logging.info(f"  [+] 输出文件: {args.output_file}")
    logging.info(f"  [+] 并行进程数: {args.num_processes}")
    logging.info(f"  [+] 日志文件: {log_filepath}")
    logging.info("="*50)

    prompt_template = load_prompt(args.prompt_file)

    processed_ids = set()
    if not args.force_rerun:
        processed_ids = load_processed_ids(args.output_file)
        if processed_ids:
            logging.info(f"Resume mode: skip {len(processed_ids)} processed samples")

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            all_samples = [json.loads(line) for line in f]
        logging.info(f"成功读取 {len(all_samples)} 条总样本。")
    except Exception as e:
        logging.error(f"读取输入文件失败: {e}", exc_info=True)
        return

    for i, s in enumerate(all_samples):
        if 'sample_id' not in s:
            s['sample_id'] = f"auto_id_{i}"

    samples_to_process = [s for s in all_samples if s.get('sample_id') not in processed_ids] if processed_ids else all_samples

    if not samples_to_process:
        logging.info("所有样本均已处理完毕。任务结束。")
        return
    logging.info(f"筛选后，本次需要处理 {len(samples_to_process)} 条新样本。")

    inference_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    api_cfg = {
        "base": args.api_base,
        "key": args.api_key,
        "model": args.model_name,
    }

    success_count = 0
    failed_count = 0
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    worker = partial(
        process_feature_extraction_task,
        prompt_template=prompt_template,
        api_cfg=api_cfg,
        inference_params=inference_params,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    with open(args.output_file, "a", encoding="utf-8") as outfile, \
         Pool(processes=args.num_processes) as pool:

        logging.info("进程池已启动，开始处理样本...")
        with tqdm(total=len(samples_to_process), desc="提取要素报告") as pbar:
            for enhanced_sample in pool.imap_unordered(worker, samples_to_process):
                if enhanced_sample:
                    outfile.write(json.dumps(enhanced_sample, ensure_ascii=False) + "\n")
                    success_count += 1
                else:
                    failed_count += 1
                pbar.update(1)

    logging.info("=" * 50)
    logging.info(f"任务完成。本轮共处理 {len(samples_to_process)} 条样本。")
    logging.info(f"  - 成功: {success_count} 条")
    logging.info(f"  - 失败: {failed_count} 条 (详情请查阅日志)")
    logging.info(f"增强后的样本已追加至: {args.output_file}")
    logging.info("--- 要素提取任务结束 ---")

if __name__ == "__main__":
    main()
