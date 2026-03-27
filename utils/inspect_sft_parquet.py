"""
用途：在终端中查看 verl 风格的 SFT Parquet 样本。
输入：包含 `question`、`answer` 两列的 Parquet 文件。
输出：按指定模式打印样本内容，不修改原文件。
运行示例：python utils/inspect_sft_parquet.py --file data/train/sft_cemt_data_direct.parquet --num 5
"""

from __future__ import annotations

import argparse
import random
from typing import List, Optional

import pandas as pd


def _parse_indices(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


def _maybe_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 12] + "\n...[truncated]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print samples from an SFT parquet (question/answer).")
    parser.add_argument("--file", type=str, required=True, help="Path to parquet file.")
    parser.add_argument("--num", type=int, default=3, help="How many samples to print (ignored if --indices set).")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated row indices to print (e.g. 0,1,2).")
    parser.add_argument(
        "--mode",
        type=str,
        default="head",
        choices=["head", "tail", "random"],
        help="How to pick samples when --indices is not set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (only for mode=random).")
    parser.add_argument(
        "--max_chars",
        type=int,
        default=2000,
        help="Truncate each question/answer to this many chars; set 0 to disable truncation.",
    )

    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    print(f"file: {args.file}")
    print(f"rows: {len(df)}")
    print(f"columns: {df.columns.tolist()}")

    expected = {"question", "answer"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Expected columns {sorted(expected)}, got {df.columns.tolist()}")

    indices = _parse_indices(args.indices)
    if indices is None:
        n = max(0, min(int(args.num), len(df)))
        if n == 0:
            return
        if args.mode == "head":
            indices = list(range(n))
        elif args.mode == "tail":
            indices = list(range(len(df) - n, len(df)))
        else:
            rng = random.Random(args.seed)
            indices = rng.sample(range(len(df)), k=n)

    for idx in indices:
        row = df.iloc[idx]
        q = str(row["question"])
        a = str(row["answer"])

        print("\n" + "=" * 80)
        print(f"idx: {idx}")
        print("- question:")
        print(_maybe_truncate(q, args.max_chars))
        print("- answer:")
        print(_maybe_truncate(a, args.max_chars))


if __name__ == "__main__":
    main()
