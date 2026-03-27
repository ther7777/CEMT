# scripts/re_data/evaluate_cot_candidates.py
"""
目的:
    "CoT论证与提纯流水线" 的第二阶段：CoT候选评估。
    读取包含N个CoT候选的jsonl文件，调用`cot_judge` LLM对每个候选进行打分，
    并将详细分数和最终的R_Process奖励分，增量式地添加回每个候选的数据中。

特性:
    - 集成cot_judge提示词和奖励计算公式，标准统一。
    - 复用健壮的API调用和JSON解析逻辑。
    - 高效的断点续跑能力。
    - 多进程并行，为大规模数据评估设计。

"""
import os
import json
import re
import time
import argparse
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import openai

# --- 1. 日志与API配置 ---

def setup_logging(log_filename: str):
    log_dir = "logs"; os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)
    logger = logging.getLogger(); logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(); ch.setFormatter(formatter); logger.addHandler(ch)
    fh = logging.FileHandler(log_filepath, mode='a', encoding='utf-8'); fh.setFormatter(formatter); logger.addHandler(fh)
    logging.getLogger("httpx").setLevel(logging.WARNING); logging.getLogger("httpcore").setLevel(logging.WARNING)
    return log_filepath

API_ENDPOINT = "http://localhost:8080/v1"
API_TOKEN = "<YOUR_API_KEY>"
MODEL_NAME = "llm"
MAX_RETRIES = 3
RETRY_DELAY = 5
INFERENCE_PARAMS = {"temperature": 0.2, "top_p": 0.95, "max_tokens": 2000} # 裁判打分温度可以低一些

# --- 2. CoT裁判提示词与奖励计算公式 ---
COT_EVALUATION_PROMPT="""## 身份设定 (Role Setting)

你是一位极其严苛、追求极致的翻译评审主席。你的任务是基于一套严谨、客观的评估标准，对一个AI模型生成的"思考链(CoT)分析"和"最终译文"的完整过程进行综合评估。请你对照评分标准做仔细的清单式检查并打分，只有完美的分析才能满分。
请你特别注意，一个高质量的CoT必须在其`<translation_strategy_formulation>`的结尾部分，包含一个以“综上所述，最终译文确定为：”开头的“结论性承诺”。你的核心任务之一就是审计这个承诺是否由前面的分析逻辑地导出，以及最终译文是否遵守了这个承诺。

## [核心审计原则] 增量式分析验证
你将收到一份【特征分析报告】，这是AI工作的唯一指令来源。你必须验证AI的【CoT分析】和【最终译文】是否对报告中指出的“增量式要素”做出了正确的回应。

### DNT元素处理原则 
1.  **硬性DNT (必须精确复制)**: 包括但不限于：代码(code)、URL、ID、技术字符串(technical strings)、电子邮件(emails)、Emoji、颜文字(emoticons)、Hashtag (#)、商标(trademarks)。任何对这些元素的翻译或改动，都构成【严重违规】，应在cot中表达不可翻译或者精确复制等含义。
2.  **软性DNT (可酌情处理)**: 包括但不限于：人名、机构名缩写、品牌名、货币/物理单位（如$、km）等。对此类元素的翻译是允许的，但CoT中应体现出合理的决策。

### 语用元素处理原则 (功能对等)
- **核心要求**: 必须进行【功能对等】的意译，绝不能字面翻译。

## 评估输入 (Evaluation Inputs)

你将收到以下三部分内容：

1.  **【源文本】**: 需要被翻译的原始文本。
2.  **【CoT分析】**: AI模型在翻译前生成的、包含四个分析标签的思考过程。
3.  **【最终译文】**: AI模型最终输出的翻译结果。

## 核心任务与评估标准 (Core Task & Rubric)

请你根据以下五项标准，对AI的整个工作流进行评估。对于每一项，你的【理由】必须是**简短且切中要害的一到两句话**。然后根据下述明确的【评分】标准给出整数分。

---


# ### 语用元素处理原则 (功能对等)
# - **核心要求**: 必须进行【功能对等】的意译，绝不能字面翻译。

# ## 评估输入 (Evaluation Inputs)

# 你将收到以下三部分内容：

# 1.  **【源文本】**: 需要被翻译的原始文本。
# 2.  **【CoT分析】**: AI模型在翻译前生成的、包含四个分析标签的思考过程。
# 3.  **【最终译文】**: AI模型最终输出的翻译结果。

# ## 核心任务与评估标准 (Core Task & Rubric)

# 请你根据以下五项标准，对AI的整个工作流进行评估。对于每一项，你的【理由】必须是**简短且切中要害的一到两句话**。然后根据下述明确的【评分】标准给出整数分。

# ---

# **【标准一：宏观分析质量】(Score 1-5)**

# - **评估要点**: CoT中的`<holistic_...>`标签内容。
# - **评分标准**:
#     - **1分 (差)**: 分析完全错误或缺失。
#     - **2分 (弱)**: 分析过于肤浅或套用模板，与文本关联度低。
#     - **3分 (合格)**: 准确识别了基本信息（如文体/领域），但缺乏深度或依据。
#     - **4分 (良)**: 分析准确，并能从原文中找到一些证据来支撑其结论。
#     - **5分 (优)**: 分析不仅精准，而且能捕捉到普通分析者容易忽略的、对最终译文风格有决定性影响的细微语用特征（如隐含的读者对象、潜在的文化背景等）。

# **【标准二：语义分析质量】(Score 1-5)**

# - **评估要点**: CoT中的`<argument_...>`标签内容。
# - **评分标准**:
#     - **1分 (差)**: 核心语义成分（主谓宾）的拆分存在严重错误。
#     - **2分 (弱)**: 拆分部分正确，但有明显遗漏或错误。
#     - **3分 (合格)**: 准确拆分了句子的主干，但忽略了重要的修饰或隐含信息。
#     - **4分 (良)**: 准确且较完整地拆分了语义成分。
#     - **5分 (优)**: 不仅准确完整，并且成功指出了一个最关键的、一旦处理不当就会导致译文产生“翻译腔”或变得不地道或不自然的语义/句法难点。

# **【标准三：句法分析质量】(Score 1-5)**

# - **评估要点**: CoT中的`<syntactic_...>`标签内容。
# - **评分标准**:
#     - **1分 (差)**: 句法结构判断存在严重错误。
#     - **2分 (弱)**: 识别了基本句型，但对从句、特殊句式等判断有误。
#     - **3分 (合格)**: 准确判断了整体句式，但未能识别出所有重要的语法现象。
#     - **4分 (良)**: 准确且较完整地识别了句法结构与特殊现象。
#     - **5分 (优)**: 不仅准确完整，并且成功指出了一个最关键的、一旦处理不当就会导致译文产生“翻译腔”或变得不地道或不自然的语义/句法难点。

# **【标准四：策略质量与原文锚定】(Score 1-5)**

# - **评估要点**: 完整的CoT分析（标签1-4）与【源文本】的整体关联性。
# - **评分标准**:
#     - **1分 (差)**: 策略与分析脱节，且整个分析过程脱离了原文的核心问题。
#     - **2分 (弱)**: 分析和策略与原文有一定关联，但忽略了原文的关键难点，或内部逻辑脱节。
#     - **3分 (合格)**: 分析基本抓住了原文的要点，策略也基本反映了分析，但不够深入或未能解决模糊性。
#     - **4分 (良)**: 分析准确抓住了原文核心挑战，策略在逻辑上紧密承接了分析，并给出了清晰的指导。
#     - **5分 (优)**: 翻译纲领不仅逻辑严谨、决策清晰，其提出的指导方针更是大师级的（masterful），能够直接引导出一个地道、自然且完全符合语境风格的译文，展现了对目标语言的精湛驾驭能力。

# **【标准五：增量式回应质量】(Score 1-5)**
# - **评估要点**: CoT和译文对【特征分析报告】中DNT及语用要素的回应质量。
# - **评估SOP (必须严格遵循)**:
#     **步骤一：解读指令**
#     首先，检查输入的【特征分析报告】中的`feature_code`。这是一个包含三个数字的列表：`[认知复杂度, DNT存在, 语用存在]`。评分的核心依据是第二位（DNT）和第三位（语用）的数字。

#     **步骤二：判断是否需要回应**
#     - **如果DNT编码为`0`，且语用编码也为`0`**: 这意味着源文本不包含任何需要增量回应的要素。
#         - **评分**: 在这种情况下，此项评估**直接判定为满分 `5` 分**。
#         - **理由**: 请填写固定的理由：“报告未要求增量回应，此项不适用。”

#     **步骤三：评估需要回应的质量**
#     - **如果DNT编码为`1`或语用编码为`1`**: 则**必须**做出增量式回应。请根据以下标准评分：
#         - **5分 (完美回应)**: CoT分析深刻，策略清晰，且最终译文**完美地**执行了所有必需的DNT和语用元素处理原则。并且对“硬性DNT”或者软性DNT”有明确的说明，处理符合对硬性或软性的要求。
#         - **4分 (回应良好)**: 正确处理了所有必需的增量式要素，但CoT中的分析或对语用元素的意译不够精彩或者对DNT的具体类别没有清晰说明。
#         - **3分 (回应基本正确)**: 基本处理了增量式要素，但存在一些瑕疵。例如：CoT中未能明确体现对DNT/语用元素的分析；或对语用元素的意译比较生硬。
#         - **2分 (回应质量差)**: 发生了明显错误，但未完全违规。例如：对语用元素进行了字面翻译；或CoT中制定的DNT策略与原则相悖。
#         - **1分 (严重违规或无回应)**: **需要回应但CoT中完全没有体现**；或者代码、URL、Emoji、商标、ID、Hashtag (`#`)等硬性元素被翻译。

# **【标准六：执行质量与忠诚度审计】(Score: 5, 4, 3, 1, 0)**
# - **评估要点**: 对【最终译文】和【CoT分析】中“结论性承诺”的双重核查。
# - **评估SOP (必须严格遵循)**:
#     **步骤一：检查“形式忠诚度”**
#     - 首先，比对【最终译文】与CoT中“综上所述，最终译文确定为：”之后的文本是否**完全一致**。
#     - **如果不一致**：此为严重违规，**直接判定为 `0` 分**。理由固定为：“最终译文与CoT内部的结论性承诺不符，执行完全失败。”

#     **步骤二：审计“逻辑忠诚度” (若形式上忠诚)**
#     - 其次，评估CoT中的“结论性承诺”是否是其上文分析（标准1-4）和策略的**合理、逻辑自洽的产物**。
            
# - **评分标准**:
#     - **5分 (卓越品质)**: 在满足4分标准的基础上，译文的**语言质量极高**，达到了信、达、雅的统一，其流畅度、地道程度和文体风格均堪比人类专家译者。
#     - **4分 (完美执行与逻辑自洽)**: 形式忠诚度通过，且CoT中的结论性承诺是其上文分析和策略的**高度逻辑自洽**的成果。整个“分析-决策-承诺”链条清晰、合理且高质量。
#     - **3分 (执行正确但逻辑有瑕疵)**: 形式忠诚度通过，但CoT的结论性承诺与其上文分析存在**轻微的逻辑脱节或不一致**。例如，分析中强调了某个要点，但在最终结论中被忽略。
#     - **1分 (执行正确但逻辑严重脱节)**: 形式忠诚度通过，但CoT的结论性承诺与其上文分析**严重矛盾**（即我们讨论过的“策略与执行脱节”问题）。例如，分析要保留语用，结论却是生硬直译。
#     - **0分 (事实性错误 或 形式不忠诚 或 硬性DNT违规)**: **（最高优先级）** 译文包含事实错误，**或者**，未通过“形式忠诚度”检查，**或者**，错误地翻译了任何一个“硬性DNT”元素。

            
# 【特征分析报告】:
# {feature_report}

# ## 待评估数据 (Input Data)
# ### 【源文本】:
# {source_text}
# ### 【CoT分析】:
# {cot_analysis}
# ### 【最终译文】:
# {final_translation}

# ## 强制输出格式 (Mandatory Output Format)
# 你的所有评估结果必须严格按照以下包含**6个评估项**的JSON格式进行组织。
# {{
#   "holistic_evaluation": {{"reasoning": "...", "score": <score_1_to_5>}},
#   "argument_evaluation": {{"reasoning": "...", "score": <score_2_1_to_5>}},
#   "syntactic_evaluation": {{"reasoning": "...", "score": <score_3_1_to_5>}},
#   "strategy_and_grounding_evaluation": {{"reasoning": "...", "score": <score_4_1_to_5>}},
#   "incremental_response_evaluation": {{"reasoning": "...", "score": <score_5_1_to_5>}},
#   "execution_and_fidelity_check": {{"reasoning": "...", "score": <score_6_tiered_0_1_3_4_5>}}
# }}
# """

# file: scripts/re_data/evaluate_cot_candidates.py

def calculate_r_process(scores: dict) -> float:
    """[V3 - 优化] 根据6维度分数和“加法-惩罚模型”计算最终的过程奖励。"""
    try:
        s1 = scores["holistic_evaluation"]["score"]
        s2 = scores["argument_evaluation"]["score"]
        s3 = scores["syntactic_evaluation"]["score"]
        s4 = scores["strategy_and_grounding_evaluation"]["score"]
        s5 = scores["incremental_response_evaluation"]["score"]
        s6 = scores["execution_and_fidelity_check"]["score"]
        
        # 1. 计算 R_plan (综合规划分) 
        # S1-S5共同构成了规划与分析的质量
        plan_scores = [s1, s2, s3, s4, s5]
        # 将1-5分的分数归一化到0.0-1.0的区间
        normalized_plan_scores = [(s - 1) / 4.0 for s in plan_scores]
        # 计算平均值作为综合规划分
        r_plan = sum(normalized_plan_scores) / len(normalized_plan_scores)
        
        # 2. 计算 Penalty_execution (执行惩罚)
        # 惩罚表可以根据实验效果微调，但机制不变
        p_map = {5: 0.0, 4: -0.2, 3: -0.4, 1: -0.8, 0: -1.5}
        p = p_map.get(s6, -1.5) # 对任何未定义的分数施加最大惩罚
        
        # 3. 计算最终过程奖励
        # 基础分是规划分，然后减去执行惩罚，最低为0
        r_process = max(0, r_plan + p)
        return r_process
        
    except (KeyError, TypeError) as e:
        logging.error(f"计算R_Process分数时发生错误: 缺少关键字段 - {e}")
        return 0.0 # 返回最低分

# --- 3. 核心功能函数 ---


def _robust_json_parser(raw_text: str, logger: logging.Logger) -> dict | None:
    """
    一个健壮的解析器，用于从LLM可能产生的混乱输出中提取纯净的JSON对象。
    """
    if not isinstance(raw_text, str):
        return None
        
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_block_pattern, raw_text, re.DOTALL)
    if match:
        clean_text = match.group(1).strip()
    else:
        clean_text = raw_text.strip()

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:

        logger.warning(
            f"JSON解析失败: {e}.\n"
            f"--- 完整的原始输出 (清理后) ---\n"
            f"{clean_text}\n"
            f"---------------------------------"
        )
        return None

def call_qwen235b_api(prompt: str, params: dict, sample_id_for_log: any) -> str | None:
    """调用vLLM API，并正确处理“双重<think>”输出格式。"""
    client = openai.OpenAI(base_url=API_ENDPOINT, api_key=API_TOKEN, timeout=600.0)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            raw_content = response.choices[0].message.content


            if "</think>" in raw_content:

                formal_output = raw_content.split("</think>", 1)[1].strip()
            else:

                formal_output = raw_content.strip()
            
            return formal_output

        except Exception:

            logging.error(f"样本ID {sample_id_for_log} 的API调用在第 {attempt + 1}/{MAX_RETRIES} 次尝试中失败。", exc_info=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logging.error(f"样本ID {sample_id_for_log} 经过所有重试后仍然失败。")
                return None
    return None



def process_evaluation_task(sample: dict) -> dict | None:
    """[修改] 为单个样本的所有CoT候选进行评估和打分，并传入feature_report。"""
    sample_id = sample.get('sample_id', 'N/A')
    try:
        # --- vvvvvvvv 核心修改区域 vvvvvvvv ---
        
        # 1. 从样本顶层提取所需信息
        source_text = sample['source']
        target_text = sample['target']
        feature_report = sample.get('feature_report') # 使用.get()安全提取

        # 2. 检查feature_report是否存在，如果不存在则创建一个默认值
        if not feature_report:
            logging.warning(f"样本ID {sample_id} 缺少 'feature_report' 字段，将使用默认值。")
            feature_report = {"feature_code": [0, 0, 0], "feature_fragments": {"dnt": [], "pragmatic": []}}
        
        # 3. 将feature_report字典序列化为JSON字符串，以便填入prompt
        feature_report_json = json.dumps(feature_report, ensure_ascii=False, indent=2)

        # --- ^^^^^^^^ 核心修改区域 ^^^^^^^^ ---
        
        evaluated_candidates = []
        for cand in sample.get('cot_candidates', []):
            cot_analysis = cand.get('generated_cot', '')
            
            # 4. 更新 .format() 调用，传入新的 feature_report_json
            prompt = COT_EVALUATION_PROMPT.format(
                source_text=source_text,
                cot_analysis=cot_analysis,
                final_translation=target_text,
                feature_report_json=feature_report_json # <-- 新增
            )
            
            cand_id_for_log = f"{sample_id}_cand_{cand.get('candidate_id', 'N/A')}"
            raw_eval_output = call_qwen235b_api(
                prompt=prompt, 
                sample_id_for_log=cand_id_for_log,
                params = {"temperature": 0.1, "top_p": 0.95, "max_tokens": 4096} # 裁判打分温度可以低一些
            )
            
            updated_cand = cand.copy()
            if raw_eval_output:
                scores_dict = _robust_json_parser(raw_eval_output, logging.getLogger())
                
                if scores_dict:
                    updated_cand['evaluation_scores'] = scores_dict
                    # 注意：这里的 calculate_r_process 也需要同步更新 (见步骤3)
                    updated_cand['r_process_score'] = calculate_r_process(scores_dict)
                else:
                    updated_cand['evaluation_scores'] = {"error": "parsing failed"}
                    updated_cand['r_process_score'] = 0.0
            else:
                updated_cand['evaluation_scores'] = {"error": "api call failed"}
                updated_cand['r_process_score'] = 0.0

            evaluated_candidates.append(updated_cand)

        final_record = sample.copy()
        final_record['cot_candidates'] = evaluated_candidates
        return final_record

    except Exception:
        logging.error(f"处理样本ID {sample_id} 时发生致命错误。", exc_info=True)
        return None

def main():
    """
    主函数，负责读取、调度、评估、写入，并集成了高效断点续跑能力。
    """
    parser = argparse.ArgumentParser(description="并行评估CoT候选并增量式写入分数。")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="由generate_cot_candidates.py生成的、包含候选的jsonl文件。")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="用于保存已评分样本的输出jsonl文件。")
    parser.add_argument("--num_processes", type=int, default=min(32, cpu_count()), 
                        help="并行进程数。")
    parser.add_argument("--force_rerun", action='store_true', 
                        help="忽略已处理的样本，强制全部重新运行。")
    args = parser.parse_args()

    # --- 1. 初始化日志 ---
    log_filepath = setup_logging(f"evaluate_candidates_{os.path.basename(args.input_file)}")
    logging.info("="*80)
    logging.info(f"--- CoT候选评估任务启动 (V2) ---")
    logging.info(f"  [+] 输入文件: {args.input_file}")
    logging.info(f"  [>] 输出文件: {args.output_file}")
    logging.info(f"  [+] 并行进程数: {args.num_processes}")
    logging.info(f"  [+] 日志文件: {log_filepath}")
    logging.info("="*80)

    # --- 2. 高效断点续跑逻辑 ---
    processed_ids = set()
    if os.path.exists(args.output_file) and not args.force_rerun:
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f_out_check:
                for line in f_out_check:
                    try:
                        processed_sample = json.loads(line)
                        if 'sample_id' in processed_sample:
                            processed_ids.add(processed_sample['sample_id'])
                    except (json.JSONDecodeError, KeyError):
                        continue
            if processed_ids:
                logging.info(f"检测到输出文件中已有 {len(processed_ids)} 条已处理样本，本次运行将跳过它们。")
        except Exception as e:
            logging.error(f"读取已存在的输出文件以实现断点续跑时出错: {e}", exc_info=True)

    # --- 3. 读取并筛选需要处理的样本 ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f_in:
            all_samples = [json.loads(line) for line in f_in]
        logging.info(f"成功从输入文件读取 {len(all_samples)} 条总样本。")
    except Exception as e:
        logging.error(f"读取输入文件失败: {e}", exc_info=True)
        return

    samples_to_process = [s for s in all_samples if s.get('sample_id') not in processed_ids]
    
    if not samples_to_process:
        logging.info("所有样本均已处理完毕。任务结束。")
        return
        
    logging.info(f"筛选后，本次需要处理 {len(samples_to_process)} 条新样本。")

    # --- 4. 并行处理与实时写入 ---
    processed_count = 0
    with open(args.output_file, 'a', encoding='utf-8') as outfile, \
         Pool(processes=args.num_processes) as pool:
        
        logging.info("进程池已启动，开始评估CoT候选...")
        with tqdm(total=len(samples_to_process), desc="评估CoT候选") as pbar:
            for result_sample in pool.imap_unordered(process_evaluation_task, samples_to_process):
                if result_sample:
                    outfile.write(json.dumps(result_sample, ensure_ascii=False) + '\n')
                    processed_count += 1
                pbar.update(1)

    # --- 5. 最终报告 ---
    logging.info("\n" + "="*80)
    logging.info("--- CoT候选评估任务统计报告 ---")
    logging.info(f"  输入文件总样本数: {len(all_samples)}")
    logging.info(f"  已跳过的样本数 (断点续跑): {len(processed_ids)}")
    logging.info(f"  本次计划处理样本数: {len(samples_to_process)}")
    logging.info(f"  本轮成功处理并写入样本数: {processed_count}")
    logging.info(f"  输出文件: {args.output_file}")
    logging.info("="*80)


if __name__ == "__main__":
    main()