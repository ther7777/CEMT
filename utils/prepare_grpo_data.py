"""Convert best-CoT JSONL samples into verl GRPO-ready Parquet."""
import argparse
import pandas as pd
import json
from tqdm import tqdm


TRANSLATION_PROMPT_TEMPLATE="""
## Role Setting
You are a world-class {pair_text} translation expert. Your primary goal is to produce translations that are **natural and idiomatic**, avoiding "translationese." You adhere to a precise "Analysis-Strategy-Execution" workflow.

## Core Task
Your task is to generate a structured Chain of Thought (CoT) analysis based on the provided "Feature Analysis Report," and then produce the final translation. Your CoT and final translation must be logically consistent.

## Instructions
1.  Your output must begin with a `<think>` block.
2.  Your reasoning must follow a **"Base + Incremental"** logic, guided by the `feature_code` in the Feature Analysis Report.
3.  You must fill the following four sub-tags in order. Your **analysis depth must be proportional to the complexity level (`feature_code[0]`)**:

    * **`<holistic_semantics_pragmatics_analysis>`**
        * **Base Analysis:** Analyze overall context/tone. **Comp 2** must be comprehensive; **Comp 1** must be concise; **Comp 0** should be identification-only.
        * **Incremental Response:** If `feature_code[2] == 1` (Pragmatic element present), add an analysis section specifically identifying this pragmatic element.

    * **`<argument_predicate_analysis>`**
        * **Base Analysis:** Analyze the core predicate-argument structure. **If Comp 2**, this *must* identify key logical/semantic difficulties (even untagged ones). **If Comp 0**, this may be "N/A" (e.g., for fragments).
        * **Incremental Response:** If `feature_code[1] == 1` (DNT element present), add an analysis section to: (1) **Classify** the DNT. (2) **Recommend** a handling strategy.

    * **`<syntactic_structure_analysis>`**
        * **Base Analysis:** Analyze the syntactic structure/sentence type. **If Comp 2**, this must cover key structural challenges. **If Comp 0**, this may be "N/A".
        * **Incremental Response:** (N/A for this tag)

    * **`<translation_strategy_formulation>`**
        * **Base Strategy:** Formulate the overall translation approach based on **all difficulties** identified in the Base and Incremental analyses above.
        * **Incremental Strategy:**
            * **DNT:** Formulate specific **"directives"** for any DNT elements.
            * **Pragmatics:** Pragmatic strategies must follow a **'Context-Function-Equivalence' reasoning loop** to find a natural, functionally equivalent expression.
        * **Concluding Commitment: [CRITICAL]** You must end this tag with the fixed phrase: "**In summary, the final translation is determined to be:**" followed immediately by the complete translation.

4.  After the `</think>` block, output the final translation. This translation must be **identical** to the one stated in your concluding commitment.

## Task Inputs
- **Source Text:**
  {source_text}
- **Feature Analysis Report:**
  {feature_analysis_json}

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
"""
PAIR_TEXT_MAP = {
    "zh-en": "Chinese to English",
    "en-zh": "English to Chinese"
}

def create_grpo_dataset(input_file: str, output_file: str):
    """
    读取SFT JSONL文件并将其转换为GRPO训练所需的Parquet格式。
    """
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"处理GRPO数据 {input_file}"):
            sft_data = json.loads(line)
            
            source_text = sft_data.get('source')
            target_text = sft_data.get('target')
            pair_direction = sft_data.get('pair')
            feature_report = sft_data.get('feature_report')

            if not all([source_text, target_text, pair_direction, isinstance(feature_report, dict)]):
                tqdm.write(f"警告: 跳过缺少关键字段的行 (ID: {sft_data.get('sample_id', 'N/A')})。")
                continue

            # 1. 构建完整的提示词 (prompt)
            pair_text = PAIR_TEXT_MAP.get(pair_direction, "专业")
            feature_analysis_json = json.dumps(feature_report, ensure_ascii=False, indent=2)
            
            full_prompt = TRANSLATION_PROMPT_TEMPLATE.format(
                pair_text=pair_text,
                source_text=source_text,
                feature_analysis_json=feature_analysis_json
            )

            # 2. 构建符合verl GRPO格式的记录
            record = {
                "prompt": [
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                "reward_model": {
                    "style": "custom", 
                    "ground_truth": target_text
                },
                "extra_info": {
                    "source": source_text,
                    "target": target_text,
                    "pair": pair_direction,
                    "feature_report": feature_report  # <-- 按您的要求，打包feature_report
                },
                "data_source": "mt_fighting_translation", # 数据源标识
                "ability": "translation"
            }
            records.append(record)

    df = pd.DataFrame(records)
    df.to_parquet(output_file, index=False)
    print(f"成功创建GRPO数据集，共 {len(df)} 条记录，已保存至: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将CEMT项目的SFT数据转换为GRPO Parquet格式。")
    parser.add_argument("--input_file", type=str, required=True, help="输入的SFT JSONL文件的路径 (由select_best_cot.py生成)。")
    parser.add_argument("--output_file", type=str, required=True, help="输出的Parquet文件的路径。")
    args = parser.parse_args()
    
    create_grpo_dataset(args.input_file, args.output_file)