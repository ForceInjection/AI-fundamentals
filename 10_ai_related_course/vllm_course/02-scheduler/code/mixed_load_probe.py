#!/usr/bin/env python3
"""第 2 讲实验脚本:长短交错负载的 TTFT 对照探针。

用法:
    vllm serve Qwen/Qwen3-4B        # 终端 A(或带 --long-prefill-token-threshold 变体)
    python3 mixed_load_probe.py     # 终端 B

行为:交错发送 4 条长 prompt(~2K token)与 4 条短 prompt(~40 token),
流式接收,记录每条请求的 TTFT(首 token 延迟)与端到端耗时,
输出按请求类型分组的对照表。

依赖:openai 客户端(随 vLLM 安装);服务地址默认 localhost:8000。
"""
from __future__ import annotations

import argparse
import asyncio
import time

LONG_TEXT = (
    "以下是一段用于撑长 prompt 的技术文档片段。" "推理系统的主要指标包括 TTFT、TPOT 与吞吐。" * 100
)
SHORT_TEXTS = [
    "用一句话介绍 KV cache。",
    "用一句话介绍 PagedAttention。",
    "用一句话介绍 continuous batching。",
    "用一句话介绍投机解码。",
]


async def probe_one(client: AsyncOpenAI, model: str, label: str, prompt: str,
                    results: list[dict]) -> None:
    """发一条流式请求,记录 TTFT 与端到端耗时。"""
    start = time.perf_counter()
    first_token_at: float | None = None
    n_tokens = 0
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                n_tokens += 1
    except Exception as exc:  # noqa: BLE001 实验脚本,失败也要进表
        results.append({"label": label, "ttft": None, "total": None, "error": str(exc)})
        return
    end = time.perf_counter()
    results.append({
        "label": label,
        "ttft": first_token_at - start if first_token_at else None,
        "total": end - start,
        "error": None,
    })


async def main() -> int:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("需要 openai 客户端:在装了 vLLM 的环境里运行,或 pip install openai")
        return 2
    parser = argparse.ArgumentParser(description="长短交错负载 TTFT 探针")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--long-tokens", type=int, default=2000,
                        help="长 prompt 的近似 token 数(按 1 字 ≈ 1 token 估算)")
    args = parser.parse_args()

    long_text = LONG_TEXT[: args.long_tokens * 1]  # 近似:中文 1 字 ≈ 1 token
    jobs = [(f"long-{i}", long_text) for i in range(4)] + [
        (f"short-{i}", text) for i, text in enumerate(SHORT_TEXTS)
    ]

    client = AsyncOpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="EMPTY")
    results: list[dict] = []
    start = time.perf_counter()
    await asyncio.gather(*(probe_one(client, args.model, label, text, results)
                           for label, text in jobs))
    wall = time.perf_counter() - start

    short_ttfts = [r["ttft"] for r in results if r["label"].startswith("short") and r["ttft"]]
    long_ttfts = [r["ttft"] for r in results if r["label"].startswith("long") and r["ttft"]]

    print(f"{'请求':<10} {'TTFT(s)':>10} {'总耗时(s)':>10}  备注")
    print("-" * 56)
    for r in results:
        ttft = f"{r['ttft']:.2f}" if r["ttft"] is not None else "N/A"
        total = f"{r['total']:.2f}" if r["total"] is not None else "N/A"
        print(f"{r['label']:<10} {ttft:>10} {total:>10}  {r['error'] or ''}")
    print("-" * 56)
    if short_ttfts:
        print(f"短请求 TTFT:均值 {sum(short_ttfts) / len(short_ttfts):.2f}s,最大 {max(short_ttfts):.2f}s")
    if long_ttfts:
        print(f"长请求 TTFT:均值 {sum(long_ttfts) / len(long_ttfts):.2f}s,最大 {max(long_ttfts):.2f}s")
    print(f"总墙钟:{wall:.2f}s")
    print("\n对照提示:换服务端 --long-prefill-token-threshold 再跑一次,比较短请求 TTFT 均值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
