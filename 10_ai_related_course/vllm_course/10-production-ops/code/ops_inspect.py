#!/usr/bin/env python3
"""第 10 讲实验脚本:vLLM 服务运维面巡检。

用法:
    vllm serve Qwen/Qwen3-1.7B 2>&1 | tee /tmp/vllm-ops.log   # 终端 A
    python3 ops_inspect.py --api-key test123                   # 终端 B

巡检项:
    1. 端点四件:/health(503=引擎死) /metrics /version /load
    2. /ping 陷阱验证(预期 404——仅 SageMaker 适配层才有)
    3. metrics 按族盘点(调度/吞吐/缓存/延迟/外挂,前缀 vllm:)
    4. 认证验证(带/不带 Authorization 的行为差异)

只依赖标准库(openai 等依赖不进本脚本)。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error

METRIC_FAMILIES = [
    ("调度/引擎", ["num_requests_running", "num_requests_waiting",
                  "kv_cache_usage_perc", "admission_rejections"]),
    ("吞吐", ["prompt_tokens", "generation_tokens", "request_success"]),
    ("缓存", ["prefix_cache_queries", "prefix_cache_hits",
              "cached_by_source", "external_prefix_cache_hits"]),
    ("延迟", ["time_to_first_token_seconds", "inter_token_latency_seconds",
              "time_per_output_token_seconds", "e2e_request_latency_seconds",
              "num_preemptions"]),
    ("外挂", ["spec_decode", "kv_connector", "kv_offload", "simple_kv_offload"]),
]


def fetch(base: str, path: str, token: str | None) -> tuple[int, str]:
    req = urllib.request.Request(base + path)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read().decode("utf-8", "replace")
        except Exception:
            return e.code, ""
    except Exception as exc:
        return -1, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM 运维面巡检")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None, help="与 --api-key 一致的 token")
    args = parser.parse_args()
    base = f"http://{args.host}:{args.port}"
    token = args.api_key

    print(f"巡检目标:{base}\n" + "=" * 64)

    # 1. 端点四件 + /ping 陷阱
    for path, note in [("/health", "探活:200 正常,503=引擎死"),
                       ("/metrics", "Prometheus 全量指标"),
                       ("/version", "版本"),
                       ("/load", "在途负载(需 --enable-server-load-tracking)"),
                       ("/ping", "陷阱:仅 SageMaker 适配层有,预期 404")]:
        code, _ = fetch(base, path, token)
        print(f"{path:<10} -> {code}   ({note})")

    # 2. 认证差异
    code_no_auth, _ = fetch(base, "/version", None)
    code_auth, _ = fetch(base, "/version", token or "wrong-key-test")
    print(f"\n认证差异:/version 无 token -> {code_no_auth},带 token -> {code_auth}"
          f"(配了 --api-key 时无 token 应为 401)")

    # 3. metrics 族盘点
    code, body = fetch(base, "/metrics", token)
    if code != 200:
        print(f"\n/metrics 不可用({code}),跳过指标盘点")
        return 1
    metric_names = {m.split("{")[0].strip() for m in body.splitlines()
                    if m.startswith("vllm:")}
    print(f"\nvllm: 指标总数:{len(metric_names)};按族盘点:")
    for fam, keys in METRIC_FAMILIES:
        found = [k for k in keys if any(k in n for n in metric_names)]
        missing = [k for k in keys if k not in found]
        status = "✓ " + ", ".join(found) if found else "✗ 全缺"
        extra = f"(缺:{', '.join(missing)})" if missing and found else ""
        print(f"  {fam:<8} {status} {extra}")

    print("\n提示:redis 抓取建议 15-30s 间隔;TTFT/TPOT 用直方图分位数而非均值。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
