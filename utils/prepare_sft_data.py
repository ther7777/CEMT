"""
用途：将 JSONL 转换为 verl 所需的 SFT Parquet。
输入：带有翻译样本的 JSONL，以及可选的 prompt 模板文件。
输出：包含 `question` 和 `answer` 两列的 Parquet，可直接供 `train/sft_cemt.sh` 使用。
运行示例：python utils/prepare_sft_data.py --mode cot --input_file data/train/sft_cemt_data.jsonl --output_file data/train/sft_cemt_data.parquet --overwrite

支持两种模式：
- `direct`：使用直接翻译 prompt，将源文本映射到参考译文。
- `cot`：保留原始 CEMT 风格的 CoT 与最终译文。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


TRANSLATION_COT_PROMPT_TEMPLATE = """
## Role Setting
You are a world-class {pair_text} translation expert. Your primary goal is to produce translations that are **natural and idiomatic**, avoiding "translationese." You adhere to a precise "Analysis-Strategy-Execution" workflow.

## Core Task
Your task is to generate a structured Chain of Thought (CoT) analysis based on the provided "Feature Analysis Report," and then produce the final translation. Your CoT and final translation must be logically consistent.

## Instructions
1. Your output must begin with a `<think>` block.
2. Your reasoning must follow a **"Base + Incremental"** logic, guided by the `feature_code` in the Feature Analysis Report.
3. You must fill the following four sub-tags in order. Your **analysis depth must be proportional to the complexity level (`feature_code[0]`)**:

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

4. After the `</think>` block, output the final translation. This translation must be **identical** to the one stated in your concluding commitment.

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
""".lstrip()

PAIR_TEXT_MAP = {
    "zh-en": "Chinese to English",
    "en-zh": "English to Chinese",
}

LANG_CODE_TO_NAME = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "is": "Icelandic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "ko": "Korean",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "tr": "Turkish",
    "pl": "Polish",
    "cs": "Czech",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "uk": "Ukrainian",
}


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _read_text_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_num}, got {type(obj)}")
            yield obj


def _pair_to_langs(pair: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not pair or "-" not in pair:
        return None, None
    src, tgt = pair.split("-", 1)
    return src.strip(), tgt.strip()


def _lang_name(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    return LANG_CODE_TO_NAME.get(code, code)


@dataclass(frozen=True)
class ConvertConfig:
    mode: str
    input_file: str
    output_file: str
    prompt_file: Optional[str]
    overwrite: bool
    source_field: str
    target_field: str
    pair_field: str
    cot_field: str
    feature_report_field: str
    fixed_pair: Optional[str]
    fixed_src_lang: Optional[str]
    fixed_tgt_lang: Optional[str]
    batch_size: int


def _ensure_output_path(output_file: str, overwrite: bool) -> None:
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(f"Output file exists: {output_file} (pass --overwrite to replace)")


def _write_parquet_rows(rows: Iterator[Tuple[str, str]], output_file: str, batch_size: int) -> Dict[str, Any]:
    schema = pa.schema([("question", pa.string()), ("answer", pa.string())])

    total = 0
    first_question_len: Optional[int] = None
    first_answer_len: Optional[int] = None

    writer: Optional[pq.ParquetWriter] = None
    try:
        questions: List[str] = []
        answers: List[str] = []

        for question, answer in rows:
            if first_question_len is None:
                first_question_len = len(question)
                first_answer_len = len(answer)

            questions.append(question)
            answers.append(answer)
            total += 1

            if len(questions) >= batch_size:
                if writer is None:
                    writer = pq.ParquetWriter(output_file, schema=schema, compression="zstd")
                table = pa.Table.from_arrays(
                    [pa.array(questions, type=pa.string()), pa.array(answers, type=pa.string())],
                    schema=schema,
                )
                writer.write_table(table)
                questions.clear()
                answers.clear()

        if questions:
            if writer is None:
                writer = pq.ParquetWriter(output_file, schema=schema, compression="zstd")
            table = pa.Table.from_arrays(
                [pa.array(questions, type=pa.string()), pa.array(answers, type=pa.string())],
                schema=schema,
            )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    return {
        "num_rows": total,
        "first_question_len": first_question_len,
        "first_answer_len": first_answer_len,
    }


def _convert_rows_direct(cfg: ConvertConfig) -> Iterator[Tuple[str, str]]:
    if not cfg.prompt_file:
        raise ValueError("mode=direct requires --prompt_file")
    prompt_template = _read_text_file(cfg.prompt_file)

    for obj in tqdm(_iter_jsonl(cfg.input_file), desc="Generating SFT rows (direct)"):
        source_text = obj.get(cfg.source_field)
        target_text = obj.get(cfg.target_field)
        pair = cfg.fixed_pair or obj.get(cfg.pair_field)

        if not isinstance(source_text, str) or not isinstance(target_text, str) or not source_text or not target_text:
            continue

        src_code, tgt_code = _pair_to_langs(pair if isinstance(pair, str) else None)
        src_lang = cfg.fixed_src_lang or _lang_name(src_code) or ""
        tgt_lang = cfg.fixed_tgt_lang or _lang_name(tgt_code) or ""

        question = prompt_template.format_map(
            _SafeFormatDict(
                source_text=source_text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                pair=pair or "",
                src_lang_code=src_code or "",
                tgt_lang_code=tgt_code or "",
            )
        )
        answer = target_text
        yield question, answer


def _convert_rows_cot(cfg: ConvertConfig) -> Iterator[Tuple[str, str]]:
    for obj in tqdm(_iter_jsonl(cfg.input_file), desc="Generating SFT rows (cot)"):
        source_text = obj.get(cfg.source_field)
        target_text = obj.get(cfg.target_field)
        pair = cfg.fixed_pair or obj.get(cfg.pair_field)
        cot = obj.get(cfg.cot_field)
        feature_report = obj.get(cfg.feature_report_field)

        if not isinstance(source_text, str) or not isinstance(target_text, str) or not source_text or not target_text:
            continue
        if not isinstance(cot, str) or not cot:
            continue
        if not isinstance(feature_report, dict):
            continue

        pair_text = PAIR_TEXT_MAP.get(pair, "Professional")
        feature_analysis_json = json.dumps(feature_report, ensure_ascii=False, indent=2)
        question = TRANSLATION_COT_PROMPT_TEMPLATE.format(
            pair_text=pair_text,
            source_text=source_text,
            feature_analysis_json=feature_analysis_json,
        )
        answer = f"{cot}\n{target_text}"
        yield question, answer


def convert_jsonl_to_sft_parquet(cfg: ConvertConfig) -> None:
    if not os.path.exists(cfg.input_file):
        raise FileNotFoundError(cfg.input_file)

    _ensure_output_path(cfg.output_file, cfg.overwrite)

    if cfg.mode == "direct":
        rows_iter = _convert_rows_direct(cfg)
    elif cfg.mode == "cot":
        rows_iter = _convert_rows_cot(cfg)
    else:
        raise ValueError(f"Unknown --mode: {cfg.mode} (expected: direct|cot)")

    stats = _write_parquet_rows(rows_iter, cfg.output_file, batch_size=cfg.batch_size)
    if stats["num_rows"] <= 0:
        raise RuntimeError("No valid rows were written. Check field names / input JSONL.")

    print(f"Saved: {cfg.output_file}")
    print(f"Rows: {stats['num_rows']}")
    print(f"First row lengths: question={stats['first_question_len']}, answer={stats['first_answer_len']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare verl SFT parquet (JSONL -> Parquet).")
    parser.add_argument("--mode", type=str, default="cot", choices=["direct", "cot"])
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL path.")
    parser.add_argument("--output_file", type=str, required=True, help="Output Parquet path.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Prompt template file path (required for mode=direct).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output parquet if exists.")

    parser.add_argument("--source_field", type=str, default="source")
    parser.add_argument("--target_field", type=str, default="target")
    parser.add_argument("--pair_field", type=str, default="pair")
    parser.add_argument("--cot_field", type=str, default="COT_Inf")
    parser.add_argument("--feature_report_field", type=str, default="feature_report")

    parser.add_argument("--fixed_pair", type=str, default=None, help="Override per-row `pair` (e.g. zh-en).")
    parser.add_argument("--fixed_src_lang", type=str, default=None, help="Override {src_lang} (e.g. Chinese).")
    parser.add_argument("--fixed_tgt_lang", type=str, default=None, help="Override {tgt_lang} (e.g. English).")

    parser.add_argument("--batch_size", type=int, default=2048, help="Rows per parquet write batch.")

    args = parser.parse_args()

    cfg = ConvertConfig(
        mode=args.mode,
        input_file=args.input_file,
        output_file=args.output_file,
        prompt_file=args.prompt_file,
        overwrite=args.overwrite,
        source_field=args.source_field,
        target_field=args.target_field,
        pair_field=args.pair_field,
        cot_field=args.cot_field,
        feature_report_field=args.feature_report_field,
        fixed_pair=args.fixed_pair,
        fixed_src_lang=args.fixed_src_lang,
        fixed_tgt_lang=args.fixed_tgt_lang,
        batch_size=args.batch_size,
    )
    convert_jsonl_to_sft_parquet(cfg)


if __name__ == "__main__":
    main()
