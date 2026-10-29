#!/usr/bin/env python3
"""容量规划算例（第 8 讲）。

只依赖 Python 标准库。三条轨：
  glm        GLM-5.3-Flash + SGLang：token 池 + KDA state 池 + page 对齐
  deepseek   DeepSeek-V4.1-Flash + vLLM：global token 池 + SWA state 池
  nano-vllm  Qwen3-0.6B + nano-vllm：token / block 池
  homework   作业 3 的参考算例：平均驻留与峰值 64×128K

单位：MB = 10^6 B，GB = 10^9 B，TiB = 2^40 B。

用法：
    python3 capacity_plan.py all
    python3 capacity_plan.py glm --seq 131072
    python3 capacity_plan.py homework
"""

from __future__ import annotations

import argparse
import math

MB = 1_000_000
GB = 1_000_000_000
TIB = 1024**4

# GLM-5.3-Flash, scaled FP8
GLM_BYTES_PER_TOKEN = 7_260
GLM_PAGE = 64
GLM_KDA_SLOT = 147_619_840  # 147.6 MB, fp32
GLM_SLOTS_PER_REQUEST = 5

# DeepSeek-V4.1-Flash
DS_GLOBAL_BYTES_PER_TOKEN = 890
DS_SWA_BYTES = 40 * 128 * 528  # 2,703,360 B

# Qwen3-0.6B, bf16
NANO_BYTES_PER_TOKEN = 2 * 28 * 8 * 128 * 2  # 114,688 B
NANO_BLOCK = 256


def align(length: int, page: int) -> int:
    """向上对齐到 page 的整数倍。"""
    return math.ceil(length / page) * page


def fmt_mb(n: float) -> str:
    return f"{n / MB:,.2f} MB"


def fmt_gb(n: float) -> str:
    return f"{n / GB:,.2f} GB"


def glm_table(seqs: tuple[int, ...] = (8192, 32768, 131072)) -> None:
    print("== GLM-5.3-Flash + SGLang ==")
    kda = GLM_SLOTS_PER_REQUEST * GLM_KDA_SLOT
    print(f"token 账：{GLM_BYTES_PER_TOKEN:,} B/token，page {GLM_PAGE}")
    print(f"KDA 账：{GLM_SLOTS_PER_REQUEST} slots × {GLM_KDA_SLOT / MB:.1f} MB = {fmt_mb(kda)}/request")
    print(f"{'seq':>10}{'aligned':>10}{'token 池':>14}{'KDA 池':>14}{'KDA÷token':>12}")
    for seq in seqs:
        aligned = align(seq, GLM_PAGE)
        token_bytes = aligned * GLM_BYTES_PER_TOKEN
        print(f"{seq:>10,}{aligned:>10,}{fmt_mb(token_bytes):>14}{fmt_mb(kda):>14}{kda / token_bytes:>11.2f}x")
    print()


def deepseek_table(seqs: tuple[int, ...] = (8192, 32768, 131072)) -> None:
    print("== DeepSeek-V4.1-Flash + vLLM ==")
    print(f"global KV：{DS_GLOBAL_BYTES_PER_TOKEN} B/token")
    print(f"SWA：40 × 128 × 528 = {fmt_mb(DS_SWA_BYTES)}/request")
    print(f"{'seq':>10}{'global':>14}{'SWA':>12}{'合计':>14}{'SWA 占比':>10}")
    for seq in seqs:
        global_bytes = DS_GLOBAL_BYTES_PER_TOKEN * seq
        swa = DS_SWA_BYTES if seq >= 128 else 40 * 528 * seq
        total = global_bytes + swa
        print(f"{seq:>10,}{fmt_mb(global_bytes):>14}{fmt_mb(swa):>12}{fmt_mb(total):>14}{swa / total * 100:>9.1f}%")
    print()


def nanovllm_table(seqs: tuple[int, ...] = (2048, 4096, 8192), concurrency: int = 8) -> None:
    print("== Qwen3-0.6B + nano-vllm ==")
    per_block = NANO_BYTES_PER_TOKEN * NANO_BLOCK
    print(f"token 账：{NANO_BYTES_PER_TOKEN:,} B/token，block {NANO_BLOCK}")
    print(f"每 block：{fmt_mb(per_block)}")
    print(f"{'seq':>10}{'blocks':>8}{'每请求':>14}{'总池':>16}（{concurrency} 并发）")
    for seq in seqs:
        blocks = math.ceil(seq / NANO_BLOCK)
        per_req = blocks * per_block
        print(f"{seq:>10,}{blocks:>8}{fmt_mb(per_req):>14}{fmt_mb(per_req * concurrency):>16}")
    print()


def homework() -> None:
    print("== 作业 3 参考算例 ==")
    mix = [(0.1, 131072), (0.4, 32768), (0.5, 8192)]
    avg = sum(frac * seq for frac, seq in mix)
    concurrency = 64
    token_avg = concurrency * avg * GLM_BYTES_PER_TOKEN
    kda = concurrency * GLM_SLOTS_PER_REQUEST * GLM_KDA_SLOT
    peak = concurrency * 131072 * GLM_BYTES_PER_TOKEN

    assert math.isclose(avg, 30310.4, rel_tol=1e-12)
    assert math.isclose(token_avg / GB, 14.083, abs_tol=0.01)
    assert math.isclose(kda / GB, 47.238, abs_tol=0.01)
    assert math.isclose(peak / GB, 60.901, abs_tol=0.01)

    print(f"平均 token = 0.1×128K + 0.4×32K + 0.5×8K = {avg:,.1f}")
    print(f"平均驻留 token 池 = 64 × {avg:,.1f} × 7,260 = {fmt_gb(token_avg)}")
    print(f"KDA 池 = 64 × 738 MB = {fmt_gb(kda)}")
    print("平均驻留：KDA 先爆")
    print(f"峰值 token 池 = 64 × 131,072 × 7,260 = {fmt_gb(peak)}")
    print(f"峰值 KDA 池 = {fmt_gb(kda)}")
    print("峰值 64×128K：token 先爆")
    print()

    print("DP8，每 rank 8 个请求：")
    print(f"  平均：token {fmt_gb(token_avg / 8)}，KDA {fmt_gb(kda / 8)}，合计 {fmt_gb((token_avg + kda) / 8)}")
    print(f"  峰值：token {fmt_gb(peak / 8)}，KDA {fmt_gb(kda / 8)}，合计 {fmt_gb((peak + kda) / 8)}")
    print("  每 rank 只算 KV/state，不含模型权重。")
    print()

    print("DRAM 账（1 TiB/节点，60% 前缀复用，1 或 2 个 KDA checkpoint/请求）：")
    cached_kv = 0.6 * concurrency * avg * GLM_BYTES_PER_TOKEN
    kda_1 = concurrency * GLM_KDA_SLOT
    kda_2 = 2 * kda_1
    print(f"  KV 副本 = 0.6 × 64 × {avg:,.1f} × 7,260 = {fmt_gb(cached_kv)}")
    print(f"  KDA 副本（1 checkpoint/请求）= {fmt_gb(kda_1)}，合计 {fmt_gb(cached_kv + kda_1)}")
    print(f"  KDA 副本（2 checkpoint/请求）= {fmt_gb(kda_2)}，合计 {fmt_gb(cached_kv + kda_2)}")
    print(f"  1 TiB = {fmt_gb(TIB)}；indexer 与 compress tail 不计入 HiCache。")
    print()

    print("page 碎片（len = 1,000）：")
    for page in (16, 64, 256):
        aligned = align(1000, page)
        print(f"  page {page:>3}: {aligned:,} − 1,000 = {aligned - 1000} token")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "track",
        nargs="?",
        default="all",
        choices=["all", "glm", "deepseek", "nano-vllm", "homework"],
    )
    parser.add_argument("--seq", type=int, default=None, help="只算一个序列长度")
    args = parser.parse_args()

    seqs = (args.seq,) if args.seq else None
    if args.track in ("all", "glm"):
        glm_table(seqs or (8192, 32768, 131072))
    if args.track in ("all", "deepseek"):
        deepseek_table(seqs or (8192, 32768, 131072))
    if args.track in ("all", "nano-vllm"):
        nanovllm_table(seqs or (2048, 4096, 8192))
    if args.track in ("all", "homework"):
        homework()


if __name__ == "__main__":
    main()
