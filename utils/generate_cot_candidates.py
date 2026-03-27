"""
目的:
    为CEMT项目的SFT阶段生成高质量的CoT（Chain-of-Thought）候选数据。
    脚本读取包含源文、目标译文和特征报告的数据集，为每个样本调用一个
    强大的教师LLM（如Qwen235B），生成N个不同的CoT候选版本。

特性:
    - 输入驱动：CoT的生成逻辑由输入的`feature_report`精确指导。
    - 健壮的解析与重构：不信任模型的<think>标签，而是通过正则独立抽取四大核心
      XML标签块，然后由程序重构成一个完美的<think>包裹结构，确保数据高度一致。
    - 参数化控制：可指定每个样本的候选数量(-n)和生成温度。
    - 高效的断点续跑与多进程并行。

执行命令示例 (为每个样本生成4个候选，温度0.6):
python scripts/wmt17_22/generate_cot_candidates_2.py \
  --input_file data/wmt17_22/wmt_17-22_merged_passed_extract_features_qwen235_HardDNT.jsonl \
  --output_file data/wmt17_22/wmt_17-22_merged_passed_cot_byqwen235ef_en_HardDNT_noalt_1024.jsonl \
  --num_candidates 3 \
  --generation_temperature 1 \
  --num_processes 64
"""

import os
import json
import re
import time
import argparse
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple

import openai
from tqdm import tqdm


# --- 1. 日志与API配置 ---

def setup_logging(log_filename: str):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(); ch.setFormatter(formatter); logger.addHandler(ch)
    fh = logging.FileHandler(log_filepath, mode='a', encoding='utf-8'); fh.setFormatter(formatter); logger.addHandler(fh)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return log_filepath

API_ENDPOINT = "http://localhost:8080/v1"
API_TOKEN = "<YOUR_API_KEY>"
MODEL_NAME = "llm"
MAX_RETRIES = 3
RETRY_DELAY = 5


COT_GENERATION_PROMPT ="""## Role Definition
You are a world-class expert in translation methodology and a master of CoT (Chain of Thought) generation.

## Core Task
Your task is not to perform the translation itself, but to generate an ideal, gold-standard CoT based on a given `source text`, a `target translation`, and a `feature analysis report`.
This CoT must perfectly and proactively demonstrate how an expert would reason from the `source text`, step-by-step, to ultimately deduce and explicitly state the given `target translation`.

## CoT Generation Standard Operating Procedure (SOP)
Your CoT must strictly adhere to the following structure and principles. It is constructed using a "Base Analysis + Incremental Response + Concluding Commitment" approach.

### 0. Guiding Principles
- **Principle of Naturalness & Idiomaticity**: The primary goal of the translation strategy is to achieve a result that is **natural and idiomatic** in the target language. Explicitly avoid strategies that lead to overly literal, verbose, or awkward phrasing ("translationese").
- **Principle of Proportional Reasoning**: The depth of analysis must be proportional to the complexity of the source text. This is a core component of avoiding "Inappropriate Inferential Extension.

### 1. Base Analysis (Determined by the first digit of `feature_code`)
- **If cognitive complexity is 2**: The analysis must be **comprehensive and in-depth**, focusing on the source text's **key logical, structural, or semantic difficulties**.
- **If cognitive complexity is 1**: The analysis must be **concise and to the point**, confirming core elements and main logic.
- **If cognitive complexity is 0**: Perform **"identification and definition" only**. For tags where analysis is not applicable (e.g., `<argument_...>` for an acronym), write "N/A" with a brief justification. For **Hard DNTs** like URLs or code, the CoT must be extremely brief, focusing only on identification and the replication directive.

### 2. Incremental Response (Determined by the last two digits of `feature_code`)

#### 2.1 DNT Element Handling Protocol
- **If DNT elements are present (feature_code[1] == 1)**:
    - You must perform a two-step "diagnosis" within the `<argument_predicate_analysis>` tag (or `<holistic_...>` if argument analysis is N/A):
        1.  **Attribute Classification**: Explicitly determine if the DNT is a "**Hard DNT**" or a "**Soft DNT**".
            - **Hard DNT Definition**: Elements whose function or identity would be destroyed by any form of translation. Includes: URLs, code snippets, IDs, technical strings, emails, emojis, hashtags.
            - **Soft DNT Definition**: Elements that are typically not translated to maintain consistency or brand identity, but could be translated. Includes: personal names, acronyms, brand names, currency/physical units (`$`, `km`).
        2.  **Strategy Recommendation**: Based on the classification, propose a handling strategy. For Hard DNTs, it must be "strict replication." For Soft DNTs, it should be "discretionary replication, with a preference for preserving the original form."
- **Subsequent Linkage**: In the `<translation_strategy_formulation>` tag, you must formulate a clear execution "directive" based on the diagnosis above.

#### 2.2 Pragmatic Element Handling (feature_code[2] == 1)
- **If pragmatic elements are present**:
    1.  In `<holistic_...>`: Add a [Pragmatic Element] analysis, identifying the element and its general nature (e.g., idiom, expletive, slang).
    2.  ** In `<translation_strategy_formulation>`**: When formulating the specific "functional equivalence" strategy, its analysis **must naturally reflect** the following reasoning loop:
        * **Context**: Identify the specific context in which the element appears.
        * **Function**: Analyze the element's true *function* in this context (e.g., is it venting, aggression, or playfulness?).
        * **Equivalence**: Based on this function, select a target language expression that is *functionally equivalent* (not literally equivalent).

### 3. [CRITICAL] Concluding Commitment
- At the end of the `<translation_strategy_formulation>` tag, you must add a concluding section. This section must **summarize the analysis above**, and then begin with the fixed phrase "**In summary, the final translation is determined to be:**" followed immediately by the text that is identical to the provided `target translation`.

---
### Few-Shot Examples 

**Example 1: [Comp: 2, DNT: 1, Prag: 1] (Hard DNT + Idiom)**
* **Source Text**: `老实说，他在文档里的那段示例代码 \`// TODO: Implement logic\` 简直是“画蛇添足”，完全没必要。`
* **Target Text**: `To be honest, that \`// TODO: Implement logic\` code snippet he put in the document is just gilding the lily—completely unnecessary.`
* **Feature Analysis Report**: `{{"feature_code": [2, 1, 1], "feature_fragments": {{"dnt": ["`// TODO: Implement logic`"], "pragmatic": ["画蛇添足"]}}}}`

<think>
<holistic_semantics_pragmatics_analysis>
[Base Analysis] This is a sentence with cognitive complexity of 2, expressing a critical opinion about a technical document. The analysis must be in-depth, focusing on the two key challenges: a code snippet and an idiom.
[Incremental Response - Pragmatic Element] This sentence contains a key pragmatic element: the idiom “画蛇添足” (drawing a snake and adding feet).
</holistic_semantics_pragmatics_analysis>
<argument_predicate_analysis>
[Base Analysis] The core predicate-argument structure is "the code snippet is 'drawing a snake with feet' (superfluous)."
[Incremental Response - DNT Element]
- **Attribute Classification**: The text fragment `` `// TODO: Implement logic` `` is a **Hard DNT** because it is a computer code comment. Translating it would destroy its original technical function and meaning.
- **Strategy Recommendation**: It requires strict, character-for-character replication.
</argument_predicate_analysis>
<syntactic_structure_analysis>
[Base Analysis] This is a complex declarative sentence with multiple modifiers, where "简直是..." serves as an emphatic predicate.
</syntactic_structure_analysis>
<translation_strategy_formulation>
1.  **Base Strategy**: The translation needs to maintain the critical, informal tone ("老实说") and accurately convey the core meaning.
2.  **Incremental Strategy (DNT & Pragmatics)**:
    - **DNT Directive**: Based on the diagnosis, the code comment `` `// TODO: Implement logic` `` must be replicated exactly as is.
    - **Pragmatic Directive**: The strategy for “画蛇添足” must reflect a context-function-equivalence loop.
        - **Context**: The context is a critique of a technical document, implying an addition that is unnecessary and potentially harmful to clarity.
        - **Function**: The *function* of the idiom is to label an action as "superfluous and counterproductive."
        - **Equivalence**: A literal translation ("draw snake add feet") is nonsensical. The English idiom "gilding the lily" perfectly captures this function of redundant, unnecessary addition and is a natural fit.

3.  **Concluding Commitment**:
    Based on the analysis above, the strategy is to combine the literal translation of the main opinion with the precise replication of the Hard DNT and the functional-equivalent translation of the pragmatic idiom.
    In summary, the final translation is determined to be: `To be honest, that \`// TODO: Implement logic\` code snippet he put in the document is just gilding the lily—completely unnecessary.`
</translation_strategy_formulation>
</think>

**Example 2: [Comp: 1, DNT: 1, Prag: 0] (Soft DNT)**
* **Source Text**: `SpaceX 公司计划在下周发射。`
* **Target Text**: `SpaceX plans to launch next week.`
* **Feature Analysis Report**: `{{"feature_code": [1, 1, 0], "feature_fragments": {{"dnt": ["SpaceX"], "pragmatic": []}}}}`

<think>
<holistic_semantics_pragmatics_analysis>
[Base Analysis] This is a sentence with cognitive complexity of 1. The analysis should be concise, confirming the simple factual statement.
</holistic_semantics_pragmatics_analysis>
<argument_predicate_analysis>
[Base Analysis] The core structure is "SpaceX plans to launch."
[Incremental Response - DNT Element]
- **Attribute Classification**: The fragment `SpaceX` is a **Soft DNT**. It is a proper noun (a brand/company name).
- **Strategy Recommendation**: Discretionary replication, with a strong preference for preserving the original form as it is the standard international identifier.
</argument_predicate_analysis>
<syntactic_structure_analysis>
[Base Analysis] A simple Subject-Verb-Object (SVO) declarative sentence. The phrase "公司" (company) is a descriptor that is often omitted in English when the brand name is used as the subject.
</syntactic_structure_analysis>
<translation_strategy_formulation>
1.  **Base Strategy**: A direct, concise translation is required. The descriptor "公司" can be naturally omitted in English for fluency.
2.  **Incremental Strategy (DNT)**:
    - **DNT Directive**: Based on the diagnosis, the Soft DNT `SpaceX` will be preserved in its original form.

3.  **Concluding Commitment**:
    Based on the analysis above, the strategy is a direct translation while preserving the Soft DNT.
    In summary, the final translation is determined to be: `SpaceX plans to launch next week.`
</translation_strategy_formulation>
</think>

**Example 3: [Comp: 0, DNT: 0, Prag: 1] (Acronym)**
* **Source Text**: `GGWP`
* **Target Text**: `Good game, well played.`
* **Feature Analysis Report**: `{{"feature_code": [0, 0, 1], "feature_fragments": {{ "dnt": [], "pragmatic": ["GGWP"]}}}}`

<think>
<holistic_semantics_pragmatics_analysis>
[Base Analysis] This is a fragment with cognitive complexity of 0. It is a formulaic acronym, requiring identification only.
[Incremental Response - Pragmatic Element] "GGWP" is a common acronym in gaming culture.
</holistic_semantics_pragmatics_analysis>
<argument_predicate_analysis>
N/A. As a fixed acronym, it lacks a predicate-argument structure.
</argument_predicate_analysis>
<syntactic_structure_analysis>
N/A. Not a sentence structure.
</syntactic_structure_analysis>
<translation_strategy_formulation>
1.  **Base Strategy**: N/A. The translation is dictated entirely by the pragmatic element.
2.  **Incremental Strategy (Pragmatics)**:
    - **Pragmatic Directive**: The strategy must follow the context-function-equivalence loop.
        - **Context**: The context is gaming, typically used at the end of a match.
        - **Function**: The *function* of "GGWP" is to express sportsmanship and respect to opponents/teammates.
        - **Equivalence**: The standard, universally understood expansion and functional equivalent is "Good game, well played."

3.  **Concluding Commitment**:
    Based on the analysis above, the strategy is to expand the acronym into its full, functional meaning.
    In summary, the final translation is determined to be: `Good game, well played.`
</translation_strategy_formulation>
</think>

---

## Task Input
- **Source Text**: {source_text}
- **Target Text (your reasoning endpoint)**: {target_text}
- **Feature Analysis Report**: {feature_analysis_json}

## Mandatory Output Format
<think>
<holistic_semantics_pragmatics_analysis>
...
</holistic_semantics_pragmatics_analysis>
<argument_predicate_analysis>
...
</argument_predicate_analysis>
<syntactic_structure_analysis>
...
</syntactic_structure_analysis>
<translation_strategy_formulation> 
...
</translation_strategy_formulation>
</think>
...Your Final Translation...
/no_think
"""

# --- 3. 核心功能函数 ---
def _parse_and_reconstruct_cot(raw_text: str) -> Optional[str]:
    """
    根据最终确认的健壮解析方案：
    1. 独立用正则寻找四个核心XML标签块。
    2. 只有四个都找到，才算成功。
    3. 成功后，按顺序拼接并用<think>标签包裹。
    """
    if not isinstance(raw_text, str):
        return None

    # 使用 re.DOTALL 标志来确保 '.' 可以匹配包括换行符在内的任何字符
    tag1_match = re.search(r"<holistic_semantics_pragmatics_analysis>.*?</holistic_semantics_pragmatics_analysis>", raw_text, re.DOTALL)
    tag2_match = re.search(r"<argument_predicate_analysis>.*?</argument_predicate_analysis>", raw_text, re.DOTALL)
    tag3_match = re.search(r"<syntactic_structure_analysis>.*?</syntactic_structure_analysis>", raw_text, re.DOTALL)
    tag4_match = re.search(r"<translation_strategy_formulation>.*?</translation_strategy_formulation>", raw_text, re.DOTALL)

    if tag1_match and tag2_match and tag3_match and tag4_match:
        # 提取每个匹配到的完整标签块
        block1 = tag1_match.group(0)
        block2 = tag2_match.group(0)
        block3 = tag3_match.group(0)
        block4 = tag4_match.group(0)

        # 按顺序拼接，并添加换行符以保持格式清晰
        reconstructed_cot = "\n".join([block1, block2, block3, block4])
        
        # 用程序生成完美的<think>标签包裹
        return f"<think>\n{reconstructed_cot}\n</think>"
    else:
        logging.warning(f"解析重构失败：未能找到全部四个核心标签块。原始输出片段: '{raw_text[:500]}...'")
        return None

def call_llm_api(prompt: str, params: dict, sample_id_for_log: any) -> Optional[str]:
    """调用LLM API，返回原始文本输出。"""
    client = openai.OpenAI(base_url=API_ENDPOINT, api_key=API_TOKEN, timeout=600.0)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            return response.choices[0].message.content
        except Exception:
            logging.error(f"样本ID {sample_id_for_log} 的API调用在第 {attempt + 1}/{MAX_RETRIES} 次尝试中失败。", exc_info=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logging.error(f"样本ID {sample_id_for_log} 经过所有重试后仍然失败。")
                return None
    return None

def process_candidate_generation(task,
                                 num_candidates: int,
                                 generation_temperature: float) -> Optional[dict]:
    """为单个样本生成或补齐所需数量的CoT候选。"""
    sample, existing_candidates = task
    sample_id = sample.get('sample_id', 'N/A')
    try:
        source_text = sample['source']
        target_text = sample['target']
        feature_report = sample['feature_report']
        
        feature_analysis_json = json.dumps(feature_report, ensure_ascii=False, indent=2)

        prompt = COT_GENERATION_PROMPT.format(
            source_text=source_text,
            target_text=target_text,
            feature_analysis_json=feature_analysis_json
        )
        
        inference_params = {"temperature": generation_temperature, "top_p": 0.95, "max_tokens": 3500}

        existing_candidates = list(existing_candidates or [])
        existing_count = len(existing_candidates)
        max_existing_id = max([c.get('candidate_id', 0) for c in existing_candidates], default=0)
        remaining_needed = max(num_candidates - existing_count, 0)

        if remaining_needed <= 0:
            logging.info(f"样本ID {sample_id}: 已有 {existing_count} 个候选，满足目标数量 {num_candidates}。")
            final_record = sample.copy()
            final_record['cot_candidates'] = existing_candidates
            return final_record

        logging.info(
            f"样本ID {sample_id}: 已有 {existing_count} 个候选，需补齐 {remaining_needed} 个以达到目标 {num_candidates}。"
        )

        candidates = []
        next_candidate_id = max_existing_id + 1
        for i in range(remaining_needed):
            raw_output = call_llm_api(prompt, inference_params, f"{sample_id}_cand_{existing_count + i + 1}")

            if raw_output:
                generated_cot = _parse_and_reconstruct_cot(raw_output)
                if generated_cot:
                    candidates.append({
                        "candidate_id": next_candidate_id,
                        "generated_cot": generated_cot
                    })
                    logging.info(
                        f"样本ID {sample_id}: 成功生成并重构候选 {next_candidate_id - max_existing_id}/{remaining_needed} (总计目标 {num_candidates})。"
                    )
                    next_candidate_id += 1
                
        if not candidates:
            logging.warning(f"样本ID {sample_id} 本轮未能生成新的有效候选。")
            if existing_candidates:
                final_record = sample.copy()
                final_record['cot_candidates'] = existing_candidates
                return final_record
            return None

        final_candidates = existing_candidates + candidates
        final_record = sample.copy()
        final_record['cot_candidates'] = final_candidates
        return final_record

    except Exception:
        logging.error(f"处理样本ID {sample_id} 时发生致命错误。", exc_info=True)
        return None

# --- 4. 主函数与并行调度 ---
def main():
    parser = argparse.ArgumentParser(description="为SFT数据并行生成N个CoT候选。")
    parser.add_argument("--input_file", type=str, required=True, help="输入的jsonl文件路径，需包含feature_report。")
    parser.add_argument("--output_file", type=str, required=True, help="存放带有CoT候选样本的输出文件路径。")
    parser.add_argument("--num_candidates", "-n", type=int, default=4, help="为每个样本生成的CoT候选数量。")
    parser.add_argument("--generation_temperature", type=float, default=0.6, help="CoT生成步骤的温度，较高值可增加多样性。")
    parser.add_argument("--num_processes", type=int, default=min(32, cpu_count()))
    parser.add_argument("--force_rerun", action='store_true', help="忽略已处理的样本，强制全部重新运行。")
    args = parser.parse_args()

    log_filename = f"generate_cot_candidates_{os.path.basename(args.input_file).replace('.jsonl', '')}.log"
    log_filepath = setup_logging(log_filename)
    
    logging.info("="*50)
    logging.info(f"--- CoT候选生成任务启动 ---")
    logging.info(f"  [+] 输入文件: {args.input_file}")
    logging.info(f"  [+] 输出文件: {args.output_file}")
    logging.info(f"  [+] 候选数量/样本: {args.num_candidates}")
    logging.info(f"  [+] 生成温度: {args.generation_temperature}")
    logging.info(f"  [+] 并行进程数: {args.num_processes}")
    logging.info(f"  [+] 日志文件: {log_filepath}")
    logging.info("="*50)

    existing_records: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(args.output_file) and not args.force_rerun:
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = record.get('sample_id')
                if not sample_id:
                    continue
                existing_records[sample_id] = record
        if existing_records:
            logging.info(f"检测到 {len(existing_records)} 个已存在的样本记录，将复用其候选数据。")

    with open(args.input_file, 'r', encoding='utf-8') as f:
        all_samples = [json.loads(line) for line in f]

    samples_to_process: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    already_satisfied = 0
    topup_count = 0
    fresh_count = 0

    for sample in all_samples:
        sample_id = sample.get('sample_id')
        if sample_id is None:
            logging.warning("检测到缺少sample_id的样本，将视作新样本处理。")
            fresh_count += 1
            samples_to_process.append((sample, []))
            continue

        if args.force_rerun:
            fresh_count += 1
            samples_to_process.append((sample, []))
            continue

        existing_record = existing_records.get(sample_id)
        if existing_record:
            existing_candidates = existing_record.get('cot_candidates', [])
            if not isinstance(existing_candidates, list):
                existing_candidates = []
            candidate_count = len(existing_candidates)
            if candidate_count >= args.num_candidates:
                already_satisfied += 1
                continue
            if candidate_count > 0:
                topup_count += 1
            else:
                fresh_count += 1
            logging.info(
                f"样本ID {sample_id}: 当前已有 {candidate_count} 个候选，将尝试补齐至 {args.num_candidates}。"
            )
            samples_to_process.append((sample, existing_candidates))
        else:
            fresh_count += 1
            samples_to_process.append((sample, []))

    if not samples_to_process:
        logging.info("所有样本的候选数量均已满足要求。任务结束。")
        return

    logging.info(
        f"总样本数: {len(all_samples)}，本次待处理: {len(samples_to_process)} (新增 {fresh_count}，补齐 {topup_count}，已满足 {already_satisfied})"
    )

    worker_func = partial(process_candidate_generation,
                          num_candidates=args.num_candidates,
                          generation_temperature=args.generation_temperature)

    processed_count = 0
    with Pool(processes=args.num_processes) as pool:
        with tqdm(total=len(samples_to_process), desc="生成CoT候选") as pbar:
            for result in pool.imap_unordered(worker_func, samples_to_process):
                if result:
                    sample_id = result.get('sample_id')
                    if sample_id:
                        existing_records[sample_id] = result
                    processed_count += 1
                pbar.update(1)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    written_count = 0
    missing_samples: List[str] = []
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        written_ids: Set[str] = set()
        for sample in all_samples:
            sample_id = sample.get('sample_id')
            record = existing_records.get(sample_id)
            if record:
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                written_count += 1
                written_ids.add(sample_id)
            else:
                if sample_id is not None:
                    missing_samples.append(sample_id)

        extra_ids = [sid for sid in existing_records.keys() if sid not in written_ids]
        for sid in extra_ids:
            outfile.write(json.dumps(existing_records[sid], ensure_ascii=False) + '\n')
            written_count += 1

    logging.info("--- 任务完成 ---")
    logging.info(
        f"本轮共成功处理 {processed_count} / {len(samples_to_process)} 条样本，输出文件已更新，共写入 {written_count} 条记录。"
    )
    if missing_samples:
        logging.warning(
            f"共有 {len(missing_samples)} 个样本暂未生成有效候选: {', '.join(map(str, missing_samples[:10]))}" +
            (" ..." if len(missing_samples) > 10 else "")
        )
    logging.info(f"输出文件: {args.output_file}")

if __name__ == "__main__":
    main()