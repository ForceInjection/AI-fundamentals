#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash 的 KV 账本（第 7 讲）。

只依赖 Python 标准库。口径：
  - global KV 只算 main KV + indexer K，不含 SWA；
  - 890 B/token 需要 NVFP4 main + MXFP4 indexer + SM100 MegaAttention；
  - SWA 按 40 层 × 128 位置 × 528 B 记，不随序列长度增长。

用法：
    python3 kv_account.py
    python3 kv_account.py --seq 32768 --prefill-tps 20000
    python3 kv_account.py --seq 131072 --replay-scale 0.5
"""

from __future__ import annotations

import argparse

MB = 1_000_000
GB = 1_000_000_000


def global_kv_breakdown() -> dict[str, float | int]:
    """890 B/token 的逐项分解。"""
    entries = 3 * 0.5 + 1 * 1.0
    main = 512 / 2 + 32  # 256 B e2m1 + 32 B E4M3 scale
    indexer = 128 / 2 + 4  # 64 B e2m1 + 4 B UE8M0 scale
    total = entries * (main + indexer)
    assert entries == 2.5
    assert main == 288
    assert indexer == 68
    assert total == 890
    return {
        "entries": entries,
        "main": main,
        "indexer": indexer,
        "total": total,
    }


def variant_breakdown() -> list[tuple[str, int, int, float]]:
    """三条配置口径：main / indexer 每 entry 字节，以及每 token 字节。"""
    rows = [
        ("NVFP4 main + MXFP4 indexer", 288, 68),
        ("NVFP4 main + FP8 indexer", 288, 132),
        ("FP8 main + FP8 indexer", 528, 132),
    ]
    result = []
    for name, main, indexer in rows:
        per_token = 2.5 * (main + indexer)
        result.append((name, main, indexer, per_token))
    assert result[0][3] == 890
    assert result[1][3] == 1050
    assert result[2][3] == 1650
    return result


def swa_bytes_per_request() -> int:
    """40 层 × 128 位置 × 528 B（V4.1 FP8 SWA record）。"""
    total = 40 * 128 * 528
    assert total == 2_703_360
    return total


def total_kv_bytes(seq: int) -> int:
    """global KV + SWA；seq < 128 时 SWA 只覆盖 seq 个位置。"""
    if seq <= 0:
        raise ValueError("seq must be positive")
    global_bytes = 890 * seq
    swa = swa_bytes_per_request() if seq >= 128 else 40 * 528 * seq
    return global_bytes + swa


def replay_plan(seq: int, window: int = 128) -> tuple[int, int]:
    """返回 (replay_start, replay_tokens)。"""
    replay_tokens = min(window, seq)
    return seq - replay_tokens, replay_tokens


def fmt_mb(n: float) -> str:
    return f"{n / MB:,.2f} MB"


def print_breakdown() -> None:
    kv = global_kv_breakdown()
    print("== 890 B/token 的分解 ==")
    print(f"entry/token = 3 × 1/2 + 1 × 1 = {kv['entries']}")
    print(f"main KV     = 512/2 + 32 = {kv['main']:.0f} B")
    print(f"indexer K   = 128/2 + 4 = {kv['indexer']:.0f} B")
    print(f"global KV   = 2.5 × (288 + 68) = {kv['total']:.0f} B/token")
    print("注：indexer K 已含在 890 内，不是第三层。")
    print()


def print_variants() -> None:
    print("== 配置口径 ==")
    print(f"{'路径':<32}{'main':>8}{'indexer':>10}{'B/token':>10}")
    for name, main, indexer, per_token in variant_breakdown():
        print(f"{name:<32}{main:>8}{indexer:>10}{per_token:>10.0f}")
    print("890 只在 NVFP4 main + MXFP4 indexer + SM100 MegaAttention 下成立。")
    print()


def print_total_kv(seq: int) -> None:
    swa = swa_bytes_per_request() if seq >= 128 else 40 * 528 * seq
    total = total_kv_bytes(seq)
    share = swa / total * 100
    print("== 总 KV ==")
    print(f"seq = {seq:,}")
    print(f"global KV = 890 × {seq:,} = {fmt_mb(890 * seq)}")
    print(f"SWA       = {fmt_mb(swa)}（{share:.1f}% 的总量）")
    print(f"总 KV     = {fmt_mb(total)}")
    print()


def print_replay(seq: int, prefill_tps: float, scale: float) -> None:
    start, replay_tokens = replay_plan(seq)
    saved_tokens = seq - replay_tokens
    t_saved = saved_tokens / prefill_tps
    t_replay = replay_tokens * scale / prefill_tps
    ratio = saved_tokens / (replay_tokens * scale)
    print("== replay 算例 ==")
    print(f"H = {seq:,}, W = 128, P = {prefill_tps:,.0f} token/s, k = {scale}")
    print(f"replay_start = {seq:,} − min(128, {seq:,}) = {start:,}")
    print(f"replay 区间  = [{start:,}, {seq:,})")
    print(f"节省 token   = {saved_tokens:,}")
    print(f"付出 token   = {replay_tokens:,} × k")
    print(f"T_saved      = {t_saved * 1000:.1f} ms")
    print(f"T_replay     = {t_replay * 1000:.2f} ms")
    print(f"saved : replay = {ratio:,.0f} : 1")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=32768, help="当前序列长度 H")
    parser.add_argument("--prefill-tps", type=float, default=20000.0, help="假设的 prefill 吞吐")
    parser.add_argument("--replay-scale", type=float, default=1.0, help="replay 每 token 代价相对全量 prefill 的倍数")
    args = parser.parse_args()

    print_breakdown()
    print_variants()
    for seq in (8192, 32768, 131072):
        print_total_kv(seq)
    print_replay(args.seq, args.prefill_tps, args.replay_scale)


if __name__ == "__main__":
    main()
