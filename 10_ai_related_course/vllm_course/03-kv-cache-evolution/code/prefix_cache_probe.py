#!/usr/bin/env python3
"""第 3 讲实验脚本:同前缀请求的冷热命中对照探针。

用法:
    vllm serve Qwen/Qwen3-1.7B --enforce-eager   # 终端 A
    python3 prefix_cache_probe.py                # 终端 B

行为:构造约 1.5K token 的公共前缀(system 段)+ 6 个不同问题;
先串行发 2 个预热请求让公共前缀入缓存,再计时发送 6 个请求,
输出每条 TTFT 与均值。对照 --no-enable-prefix-caching 前后差异。

依赖:openai 客户端(随 vLLM 安装)。
"""
from __future__ import annotations

import argparse
import asyncio
import time

QUESTIONS = [
    "用一句话总结这份文档。",
    "文档里的核心指标有哪几个?",
    "第二段说了什么?",
    "给这份文档起一个标题。",
    "文档适合什么读者?",
    "用表格列出文档要点。",
]

PREFIX_TEMPLATE = (
    "你是一个技术文档分析助手。以下是《推理系统入门》文档全文:\n\n"
    + ("第{n}章:推理系统由调度器、KV cache 管理器与注意力后端组成。"
       "调度器决定每一步把哪些请求放进 batch;KV cache 管理器把上下文的中间状态"
       "切成固定大小的块来存放与共享;注意力后端负责在给定这些状态的情况下完成"
       "本步计算。三者通过一个逐 token 的循环协作:每一步先调度、再分配、再计算、"
       "最后回写状态。章节内容围绕以下主题展开:指标口径、显存布局、复用与驱逐、"
       "以及容量规划。具体约定如下:{filler}\n\n"
       ).format(n=0, filler="") +
    "".join(
        f"第{i}章:本章讨论主题 {i} 的细节约定,包括口径定义、边界情况与常见误用。"
        f"其中要点 {j} 强调:任何性能数字都需要同时给出模型变体、精度与硬件环境,"
        f"否则不可比较。约定 {i}-{j}:所有示例默认使用 Qwen 家族小模型演示。\n"
        for i in range(1, 6)
        for j in range(1, 6)
    )
)


async def run_one(client, model: str, question: str, results: list) -> None:
    start = time.perf_counter()
    first_at = None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PREFIX_TEMPLATE},
                {"role": "user", "content": question},
            ],
            max_tokens=48,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content and first_at is None:
                first_at = time.perf_counter()
                break  # 本探针只关心 TTFT,拿到首 token 即可
    except Exception as exc:  # noqa: BLE001
        results.append({"q": question[:12], "ttft": None, "error": str(exc)})
        return
    results.append({"q": question[:12], "ttft": (first_at - start) if first_at else None, "error": None})


async def main() -> int:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("需要 openai 客户端:在装了 vLLM 的环境里运行,或 pip install openai")
        return 2
    parser = argparse.ArgumentParser(description="同前缀命中 TTFT 探针")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="EMPTY")

    # 预热:串行发 2 个请求,让公共前缀的块进入缓存并提交
    print(f"公共前缀约 {len(PREFIX_TEMPLATE)} 字(≈token 数,中文近似)……")
    warm: list = []
    for q in QUESTIONS[:2]:
        await run_one(client, args.model, q, warm)
    print(f"预热完成:{sum(1 for r in warm if r['ttft'] is not None)} 条成功")

    # 计时:6 个不同问题(同前缀)
    results: list = []
    t0 = time.perf_counter()
    for q in QUESTIONS:
        await run_one(client, args.model, q, results)
    wall = time.perf_counter() - t0

    print(f"\n{'问题':<14} {'TTFT(s)':>10}")
    print("-" * 28)
    ok = [r for r in results if r["ttft"] is not None]
    for r in results:
        ttft = f"{r['ttft']:.2f}" if r["ttft"] is not None else "N/A"
        print(f"{r['q']:<14} {ttft:>10}  {r['error'] or ''}")
    print("-" * 28)
    if ok:
        print(f"命中后均值 {sum(r['ttft'] for r in ok) / len(ok):.2f}s | 总墙钟 {wall:.2f}s")
    print("\n对照提示:用 --no-enable-prefix-caching 重启服务再跑,比较均值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
