#!/usr/bin/env python3
"""第 4 讲：解析 vLLM 统计日志（纯标准库）。

支持两类输入行：
1. vLLM 标准统计行，例如
   Engine 000: Avg prompt throughput: 128.4 tokens/s, ..., GPU KV cache usage: 2.7%,
   Prefix cache hit rate: 75.0%
2. code/README.md 里的内部统计打印
   [prefix] requests=2 queries=64 hits=48

用法：
    python3 parse_stats_log.py --demo
    python3 parse_stats_log.py /path/to/vllm.log

没有 GPU 时用 --demo 查看字段和输出格式；有日志文件时直接传路径。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 字段名 -> 正则。字段允许缺省，缺省时输出 "-"。
STAT_PATTERNS: dict[str, re.Pattern[str]] = {
    "prompt_tok_s": re.compile(r"Avg prompt throughput: ([0-9.]+) tokens/s"),
    "gen_tok_s": re.compile(r"Avg generation throughput: ([0-9.]+) tokens/s"),
    "running": re.compile(r"Running: (\d+) reqs"),
    "waiting": re.compile(r"Waiting: (\d+) reqs"),
    "preemptions": re.compile(r"Preemptions: (\d+)"),
    "kv_usage_pct": re.compile(r"GPU KV cache usage: ([0-9.]+)%"),
    "prefix_hit_pct": re.compile(r"Prefix cache hit rate: ([0-9.]+)%"),
    "external_hit_pct": re.compile(r"External prefix cache hit rate: ([0-9.]+)%"),
}

PREFIX_LINE = re.compile(
    r"\[prefix\] requests=(\d+) queries=(\d+) hits=(\d+)"
)

COLUMNS = [
    "prompt_tok_s",
    "gen_tok_s",
    "running",
    "waiting",
    "preemptions",
    "kv_usage_pct",
    "prefix_hit_pct",
    "external_hit_pct",
    "prefix_requests",
    "prefix_queries",
    "prefix_hits",
]

DEMO_LINES = [
    "Engine 000: Avg prompt throughput: 128.4 tokens/s, "
    "Avg generation throughput: 36.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, "
    "GPU KV cache usage: 2.7%, Prefix cache hit rate: 75.0%",
    "Engine 000: Avg prompt throughput: 0.0 tokens/s, "
    "Avg generation throughput: 41.8 tokens/s, Running: 2 reqs, Waiting: 1 reqs, "
    "Preemptions: 2, GPU KV cache usage: 61.4%, Prefix cache hit rate: 40.0%",
    "[prefix] requests=2 queries=64 hits=48",
]


def parse_line(line: str) -> dict[str, str] | None:
    """解析一行，返回字段字典；不是统计行时返回 None。"""
    row: dict[str, str] = {}
    for name, pattern in STAT_PATTERNS.items():
        match = pattern.search(line)
        if match:
            row[name] = match.group(1)

    prefix = PREFIX_LINE.search(line)
    if prefix:
        requests, queries, hits = (int(value) for value in prefix.groups())
        row["prefix_requests"] = str(requests)
        row["prefix_queries"] = str(queries)
        row["prefix_hits"] = str(hits)
        if queries:
            row["prefix_hit_pct"] = f"{hits / queries * 100:.1f}"

    # 至少要认出 Running 或内部 prefix 行，才算一条统计记录。
    if "running" not in row and "prefix_queries" not in row:
        return None
    return row


def parse_lines(lines: list[str]) -> list[dict[str, str]]:
    rows = []
    for line in lines:
        row = parse_line(line)
        if row is not None:
            rows.append(row)
    return rows


def print_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("没有解析到统计行。检查日志里是否包含 'Running:' 或 '[prefix]'。")
        return

    print("| # | " + " | ".join(COLUMNS) + " |")
    print("| --- | " + " | ".join("---" for _ in COLUMNS) + " |")
    for index, row in enumerate(rows, start=1):
        values = [row.get(column, "-") for column in COLUMNS]
        print(f"| {index} | " + " | ".join(values) + " |")


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"找不到日志文件：{path}")
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(description="解析 vLLM 统计日志")
    parser.add_argument("logfile", nargs="?", help="vLLM 日志文件路径")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="不读文件，解析内置示例行",
    )
    args = parser.parse_args()

    if args.demo or not args.logfile:
        lines = DEMO_LINES
        print("使用内置示例行；数值仅用于说明字段。\n")
    else:
        try:
            lines = read_lines(Path(args.logfile))
        except OSError as exc:
            print(exc, file=sys.stderr)
            return 1

    rows = parse_lines(lines)
    print_table(rows)
    print()
    print("字段说明见同目录 README.md；命中率按 hits / queries 计算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
