"""LLM-based translation alignment validator with pass/fail splitting."""
import os
import json
import re
import time
import argparse
import logging
from multiprocessing import Pool
from typing import Tuple, Dict, Any, Optional
from tqdm import tqdm
import openai

def setup_logging(input_filename: str):
    """配置日志记录，日志文件名与输入文件关联，并体现脚本功能。"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    log_filename = f"validate_alignment_{base_name}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return log_filepath

API_ENDPOINT = "http://localhost:8080/v1"
API_TOKEN = "<YOUR_API_KEY>"
MODEL_NAME = "llm"

INFERENCE_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 512,
}
MAX_RETRIES = 3
RETRY_DELAY = 5

ALIGNMENT_VALIDATION_PROMPT_TEMPLATE = """
# 角色与目标
你是一位高级数据质量审核专家。你的核心任务是甄别出那些存在“信息不对称”或“DNT违规”的翻译对。

# 核心原则 

## 1. 翻译允许“意译”和“合理补充”
一个合格的翻译（is_aligned: true）应该忠实于原文主旨，并可结合世界知识进行推断。

## 2. 允许“语域转换”
即为了适应不同正式程度或文体风格，对词汇进行具体化或概括化操作（如将“走”译为“scheduled departure time”），只要没有引入**无法证伪的具体事实**。

## 3. 严禁翻译 DNT（Do Not Translate）元素 
以下元素在任何情况下都**必须原封不动保留**，翻译即违规：
- URL 链接（如 https://xxx）
- 电子邮件地址（如 user@domain.com）
- 网络中Hashtag 标签（如 #DIY、#HomeRenovation）
- Emoji 表情符号（如 😊、🚀、🎉）
- 网络用户名、ID、订单号（如 @john、Order#12345）
- 品牌名、产品名、商标（如 iPhone、Tesla、Photoshop）
- 代码、变量、占位符（如 {{name}}、user_id、Ctrl+C）
- 国际通用缩写（如 GDP、NASA —— **除非目标语言有官方译名**）
- 文件路径、命令行、SQL/正则表达式等技术内容
- 标准单位与符号（如 5 km/h、pH=7、℃）

## 4. 拒绝标准（is_aligned: false）包括以下两类：

### A. 信息不对称
目标文本包含了一些在翻译过程中丢失、仅存在于“原始上下文”中的具体、可证伪的事实信息（如人名、时间、地点），或凭空捏造内容。**但是如果能从众所周知的信息得到翻译，是可以允许的**

### B. DNT违规
目标文本错误翻译了上述任何一项 DNT 元素。即使语义“正确”，只要形式被改动，即视为严重违规。**但是缩写被翻译为官方译名是允许的**

#注意赦免类型
1.地名机构名，如果有公认、约定俗成的译名，也是允许的。
2.人名的音译也是允许的
当然人名地名机构名不翻译也是可以的

# 任务
请根据“核心原则”分析给定的 source 和 target，判断其是否对齐。

# 输入
- source: "{source_text}"
- target: "{target_text}"

# 输出格式
请严格按照以下JSON格式提供你的分析，不要包含任何额外说明。布尔值请使用 `true` 或 `false`。

{{
  "is_aligned": <true 或 false>,
  "拒绝类型": "<如果拒绝，填写“信息不对称”或“DNT违规”+简短理由>",
  "目标文本问题片段": "<引用目标文本中违规的具体片段>"
}}

# 示例

## 示例 1 (应通过 - 知识补充)
- source: "奥巴马访问英国"
- target: "US President Obama visits the UK"
- 预期输出:
{{
  "is_aligned": true,
  "拒绝类型": "",
  "目标文本问题片段": ""
}}

## 示例 2 (应通过 - 语域转换)
- source: "咱们啥时候走？"
- target: "What is our scheduled departure time?"
- 预期输出:
{{
  "is_aligned": true,
  "拒绝类型": "",
  "目标文本问题片段": ""
}}

## 示例 3 (应拒绝 - 信息不对称)
- source: "请看附件。"
- target: "Please see the attached Q3 financial report."
- 预期输出:
{{
  "is_aligned": false,
  "拒绝类型": "信息不对称，在原文中找不到对应内容",
  "目标文本问题片段": "Q3 financial report"
}}

## 示例 4 (应拒绝 - DNT违规)
- source: "Check out my new setup! #GamingPC #DIY 😎"
- target: "看看我的新配置！#游戏电脑 #自己动手做 😎"
- 预期输出:
{{
  "is_aligned": false,
  "拒绝类型": "DNT违规，社交平台的tag标签不应该翻译",
  "目标文本问题片段": "#游戏电脑 #自己动手做"
}}

## 示例 5 (应拒绝 - DNT违规)
- source: "Contact me at support@apple.com or visit https://apple.com"
- target: "联系我：support@apple.com 或访问 https://苹果.com"
- 预期输出:
{{
  "is_aligned": false,
  "拒绝类型": "DNT违规，url整体不翻译",
  "目标文本问题片段": "https://苹果.com"
}}

## 示例 6 (应拒绝 - DNT违规)
- source: "Use code SAVE20 at checkout."
- target: "结账时使用优惠码 保存20。"
- 预期输出:
{{
  "is_aligned": false,
  "拒绝类型": "DNT违规，save20是代码，没有具体含义不应该翻译",
  "目标文本问题片段": "保存20"
}}

## 示例 7 (应通过 - 满足DNT)
- source: "Please let me verify your #PRS_ORG# account."
- target:  "我允许我验证您的 #PRS_ORG# 帐户。"
- 预期输出:
{{
  "is_aligned": true,
  "拒绝类型": "",
  "目标文本问题片段": ""
}}
/no_think
"""

def _robust_json_parser(raw_text: str, logger: logging.Logger) -> Optional[Dict]:
    """一个健壮的解析器，用于从LLM可能产生的混乱输出中提取纯净的JSON对象。"""
    if not isinstance(raw_text, str):
        return None
        
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_block_pattern, raw_text, re.DOTALL)
    if match:
        clean_text = match.group(1).strip()
    else:
        # 尝试从文本的第一个 '{' 和最后一个 '}' 提取
        start_brace = raw_text.find('{')
        end_brace = raw_text.rfind('}')
        if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
            clean_text = raw_text[start_brace:end_brace+1]
        else:
            clean_text = raw_text.strip()

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        logger.warning(f"JSON解析失败。清理后的文本: '{clean_text[:500]}...'")
        return None

def call_llm_api(prompt: str, sample_id_for_log: Any) -> Optional[Dict]:
    """调用LLM API并返回解析后的JSON字典。"""
    client = openai.OpenAI(
        base_url=API_ENDPOINT,
        api_key=API_TOKEN,
        timeout=300.0
    )
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                **INFERENCE_PARAMS
            )
            raw_content = response.choices[0].message.content
            return _robust_json_parser(raw_content, logging.getLogger())

        except Exception:
            logging.error(f"样本ID {sample_id_for_log} API调用在第 {attempt + 1}/{MAX_RETRIES} 次尝试中失败。", exc_info=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logging.error(f"样本ID {sample_id_for_log} 经所有重试后仍失败。")
                return None
    return None

def process_validation_task(sample: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    工作进程的核心单元，为单个样本进行对齐验证。
    返回一个元组 (status, data)，其中 status 为 "passed" 或 "failed"。
    """
    sample_id = sample.get('sample_id', 'N/A')
    try:
        source_text = sample['source']
        target_text = sample['target']
        
        prompt = ALIGNMENT_VALIDATION_PROMPT_TEMPLATE.format(
            source_text=source_text,
            target_text=target_text
        )
        
        validation_result = call_llm_api(prompt, sample_id)
        
        if not validation_result or "is_aligned" not in validation_result:
            logging.warning(f"样本ID {sample_id} 的LLM返回格式无效或为空，跳过此样本。")
            return None

        if validation_result["is_aligned"]:
            # 通过的样本，返回原始的、完整的样本字典
            return ("passed", sample)
        else:
            # 不通过的样本，按要求构造新的字典
            failed_record = {
                "sample_id": sample.get("sample_id"),
                "source": source_text,
                "target": target_text,
                "error_analysis": validation_result
            }
            return ("failed", failed_record)

    except Exception:
        logging.error(f"处理样本ID {sample_id} 时发生未知错误。", exc_info=True)
    return None

def load_processed_ids(filepath: str) -> set:
    """从单个输出文件中加载已处理的样本ID。"""
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
    """主函数，负责读取、调度和写入，已集成高效断点续跑能力。"""
    parser = argparse.ArgumentParser(description="并行验证翻译数据对齐质量，并分流到不同文件。")
    parser.add_argument("--input_file", type=str, required=True, help="输入的jsonl文件路径。")
    parser.add_argument("--passed_file", type=str, required=True, help="存放验证通过样本的输出文件路径。")
    parser.add_argument("--failed_file", type=str, required=True, help="存放验证失败样本的输出文件路径。")
    parser.add_argument("--num_processes", type=int, default=32, help="并行进程数。")
    parser.add_argument("--force_rerun", action='store_true', help="忽略已处理的样本，强制全部重新运行。")
    args = parser.parse_args()

    log_filepath = setup_logging(args.input_file)
    
    logging.info("="*50)
    logging.info(f"--- 开始对齐验证任务 ---")
    logging.info(f"  [+] 输入文件: {args.input_file}")
    logging.info(f"  [+] 通过文件: {args.passed_file}")
    logging.info(f"  [+] 失败文件: {args.failed_file}")
    logging.info(f"  [+] 并行进程数: {args.num_processes}")
    logging.info(f"  [+] 日志文件: {log_filepath}")
    logging.info("="*50)

    processed_ids = set()
    if not args.force_rerun:
        processed_ids.update(load_processed_ids(args.passed_file))
        processed_ids.update(load_processed_ids(args.failed_file))
        if processed_ids:
            logging.info(f"检测到输出文件中已有 {len(processed_ids)} 条已处理样本，本次运行将跳过它们。")

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            all_samples = [json.loads(line) for line in f]
        logging.info(f"成功读取 {len(all_samples)} 条总样本。")
    except Exception as e:
        logging.error(f"读取输入文件失败: {e}", exc_info=True)
        return

    samples_to_process = [s for s in all_samples if s.get('sample_id') not in processed_ids] if processed_ids else all_samples

    if not samples_to_process:
        logging.info("所有样本均已处理完毕。任务结束。")
        return
    logging.info(f"筛选后，本次需要处理 {len(samples_to_process)} 条新样本。")

    passed_count = 0
    failed_count = 0
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.passed_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.failed_file), exist_ok=True)

    with open(args.passed_file, 'a', encoding='utf-8') as passed_outfile, \
         open(args.failed_file, 'a', encoding='utf-8') as failed_outfile, \
         Pool(processes=args.num_processes) as pool:
        
        logging.info("进程池已启动，开始处理样本...")
        with tqdm(total=len(samples_to_process), desc=f"验证数据对齐") as pbar:
            for result in pool.imap_unordered(process_validation_task, samples_to_process):
                if result:
                    status, data = result
                    if status == "passed":
                        passed_outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                        passed_count += 1
                    elif status == "failed":
                        failed_outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                        failed_count += 1
                pbar.update(1)

    logging.info("="*50)
    logging.info(f"任务完成。本轮共处理 {len(samples_to_process)} 条样本。")
    logging.info(f"  - 通过: {passed_count} 条")
    logging.info(f"  - 失败: {failed_count} 条")
    logging.info(f"通过的样本已追加至: {args.passed_file}")
    logging.info(f"不通过的样本已追加至: {args.failed_file}")
    logging.info("--- 对齐验证任务结束 ---")

if __name__ == "__main__":
    main()