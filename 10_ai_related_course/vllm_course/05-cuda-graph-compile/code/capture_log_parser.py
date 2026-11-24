#!/usr/bin/env python3
"""第 5 讲实验脚本:解析 vLLM serve 日志中的 CUDA graph capture 事实。

用法:
    vllm serve Qwen/Qwen3-1.7B 2>&1 | tee /tmp/vllm-capture.log   # 终端 A
    python3 capture_log_parser.py /tmp/vllm-capture.log           # 终端 B

输出:
    - capture 总耗时与显存占用(来自 "Graph capturing finished in ... secs, took ... GiB")
    - 图/capture 相关行归类(capture 触发、编译 warmup、模式降级等)
    - 提示:改 --max-num-seqs 重跑可对照 decode 族图数量变化

只依赖标准库。
"""
from __future__ import annotations

import re
import sys

PATTERNS = [
    (re.compile(r"Graph capturing finished in ([\d.]+) secs, took ([\d.]+) GiB"),
     "capture 总量", "耗时 {0} 秒,显存 {1} GiB"),
    (re.compile(r"Capturing (?:cuda graph|cudagraph)s?.*?(\d+)"),
     "capture 进度", "进度行(捕获到第 {0} 张)"),
    (re.compile(r"Capture cudagraph begin", re.IGNORECASE), "capture 开始", "开始捕获"),
    (re.compile(r"(?:CUDAGraph|cudagraph) mode[:=]\s*(\w+)"),
     "图模式", "运行模式:{0}"),
    (re.compile(r"Using (?:(?:FLASH|TRITON|FLASHINFER)\w*)", re.IGNORECASE),
     "attention 后端", "后端选择行(影响图支持)"),
    (re.compile(r"compil(?:ation|e).*?(?:finished|done|warmup)", re.IGNORECASE),
     "编译", "编译/warmup 完成行"),
    (re.compile(r"VLLM_USE_BREAKABLE_CUDAGRAPH|breakable", re.IGNORECASE),
     "breakable 图", "断点续捕变体相关"),
]


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: capture_log_parser.py <serve日志文件>")
        return 2
    try:
        lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
    except OSError as exc:
        print(f"读不到文件:{exc}")
        return 2

    hits: list[tuple[str, str, str]] = []
    for lineno, line in enumerate(lines, 1):
        for pattern, tag, fmt in PATTERNS:
            m = pattern.search(line)
            if m:
                detail = fmt.format(*m.groups()) if m.groups() else fmt
                hits.append((tag, f"L{lineno}", detail))
                break

    if not hits:
        print("没找到 capture/编译相关日志。确认:未加 --enforce-eager;日志用 tee 落了文件。")
        return 1

    print(f"{'类别':<16} {'日志行':>8}  详情")
    print("-" * 64)
    for tag, lineno, detail in hits:
        print(f"{tag:<16} {lineno:>8}  {detail}")

    totals = [h for h in hits if h[0] == "capture 总量"]
    if totals:
        print("-" * 64)
        for _, _, detail in totals:
            print(f"▸ {detail}")
        print("对照提示:改 --max-num-seqs 重启再解析,decode 族图数量随上限变化"
              "(网格:≤6 全捕,之后 0.65·n^(2/3) 取 2 的幂;prefill 族共 54 尺寸)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
