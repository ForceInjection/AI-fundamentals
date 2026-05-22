"""
FlashAttention 简化版：tiling + online softmax forward pass

核心思想：将 Q@K^T 的 [N,N] 矩阵分块计算，用 online softmax 在
不存储完整注意力矩阵的情况下得到等价结果。

用法:
  ASCEND_RT_VISIBLE_DEVICES=7 python3 flash_attention.py --verify
  ASCEND_RT_VISIBLE_DEVICES=7 python3 flash_attention.py --profile --seq-len 4096
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F
import torch_npu


# ── Standard Attention (baseline) ──

def standard_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
                       ) -> torch.Tensor:
    """标准 scaled dot-product attention: O = softmax(Q@K^T/√d) @ V"""
    d = Q.size(-1)
    S = Q @ K.transpose(-2, -1) / math.sqrt(d)        # [B, N, N]
    P = F.softmax(S, dim=-1)                            # [B, N, N]
    return P @ V                                         # [B, N, d]


# ── FlashAttention (tiled + online softmax) ──

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                            Br: int = 64, Bc: int = 64
                            ) -> torch.Tensor:
    """FlashAttention forward pass: tiling + online softmax

    Q, K, V: [B, N, d]
    Br: Q block size (行方向)
    Bc: K/V block size (列方向)

    外层循环: 遍历 Q blocks
      内层循环: 遍历 K/V blocks，累加 softmax 分子和加权 V

    每个内层循环中，只产生 [Br, Bc] 的临时矩阵，不产生 [N, N] 矩阵。
    """
    B, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # 预分块索引
    Tr = (N + Br - 1) // Br    # Q 分块数
    Tc = (N + Bc - 1) // Bc    # K/V 分块数

    O = torch.zeros_like(Q)    # 输出 [B, N, d]

    for i in range(Tr):
        # ── Q block ──
        q_start = i * Br
        q_end = min(q_start + Br, N)
        Qi = Q[:, q_start:q_end, :]                    # [B, Br, d]

        # 每个 Q block 重置 online softmax 状态
        mi = torch.full((B, q_end - q_start, 1), float("-inf"),
                        device=Q.device, dtype=Q.dtype)  # 运行 max
        li = torch.zeros(B, q_end - q_start, 1,
                         device=Q.device, dtype=Q.dtype)  # 运行 sum(exp)
        Oi = torch.zeros(B, q_end - q_start, d,
                         device=Q.device, dtype=Q.dtype)  # 运行加权 V

        for j in range(Tc):
            # ── K/V block ──
            k_start = j * Bc
            k_end = min(k_start + Bc, N)
            Kj = K[:, k_start:k_end, :]                # [B, Bc, d]
            Vj = V[:, k_start:k_end, :]                # [B, Bc, d]

            # 计算当前 block 的注意力分数
            Sij = (Qi @ Kj.transpose(-2, -1)) * scale  # [B, Br, Bc]

            # ── Online Softmax 更新 ──
            # 1. 更新 max
            mi_new = torch.max(mi, Sij.max(dim=-1, keepdim=True).values)

            # 2. 计算当前 block 的 exp(S - m_new)
            Pij = torch.exp(Sij - mi_new)               # [B, Br, Bc]

            # 3. correction: 如果 m_new > m_old，之前累计需要"打折"
            correction = torch.exp(mi - mi_new)          # ≤ 1

            # 4. 更新累计 sum 和 weighted V
            li = correction * li + Pij.sum(dim=-1, keepdim=True)
            Oi = correction * Oi + Pij @ Vj

            # 5. 更新 max
            mi = mi_new

        # ── 最终归一化 ──
        O[:, q_start:q_end, :] = Oi / li

    return O


# ── 验证与对比 ──

def compare_and_verify(device: str = "npu:0"):
    """验证 FlashAttention 与标准 Attention 的数值一致性"""
    print(f"\n{'='*60}")
    print(f"  精度验证")
    print(f"{'='*60}")

    configs = [
        (1, 256, 64),
        (1, 512, 64),
        (1, 1024, 64),
        (4, 512, 64),
    ]

    for B, N, d in configs:
        torch.manual_seed(42)
        Q = torch.randn(B, N, d, device=device)
        K = torch.randn(B, N, d, device=device)
        V = torch.randn(B, N, d, device=device)

        with torch.no_grad():
            out_std = standard_attention(Q, K, V)
            out_flash = flash_attention_forward(Q, K, V)

        max_diff = (out_std - out_flash).abs().max().item()
        rel_diff = ((out_std - out_flash).abs() /
                    (out_std.abs() + 1e-8)).max().item()

        status = "✓" if max_diff < 1e-3 else "✗"
        print(f"  {status} B={B}, N={N:4d}, d={d}: "
              f"max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")


def profile_memory(device: str = "npu:0", seq_len: int = 4096,
                   d: int = 64, B: int = 1, Br: int = 64, Bc: int = 64):
    """对比标准 Attention 和 FlashAttention 的 HBM 峰值占用"""
    print(f"\n{'='*60}")
    print(f"  显存占用对比 (N={seq_len}, d={d})")
    print(f"{'='*60}")

    Q = torch.randn(B, seq_len, d, device=device)
    K = torch.randn(B, seq_len, d, device=device)
    V = torch.randn(B, seq_len, d, device=device)

    # 标准 Attention
    torch.npu.reset_peak_memory_stats()
    torch.npu.empty_cache()
    mem_before = torch.npu.memory_allocated() / 1024**2

    with torch.no_grad():
        _ = standard_attention(Q, K, V)
    torch.npu.synchronize()

    mem_after = torch.npu.memory_allocated() / 1024**2
    peak_std = torch.npu.max_memory_allocated() / 1024**2
    print(f"  Standard Attention:")
    print(f"    稳态: {mem_after - mem_before:.0f} MB  |  峰值: {peak_std - mem_before:.0f} MB")

    # 理论 N×N 矩阵大小
    theory_nn = B * seq_len * seq_len * 4 / 1024**2  # FP32 = 4 bytes
    print(f"    理论 N×N: {theory_nn:.0f} MB (S) + {theory_nn:.0f} MB (P) = {theory_nn*2:.0f} MB")

    # FlashAttention
    torch.npu.reset_peak_memory_stats()
    torch.npu.empty_cache()
    mem_before = torch.npu.memory_allocated() / 1024**2

    with torch.no_grad():
        _ = flash_attention_forward(Q, K, V, Br=Br, Bc=Bc)
    torch.npu.synchronize()

    mem_after = torch.npu.memory_allocated() / 1024**2
    peak_flash = torch.npu.max_memory_allocated() / 1024**2
    print(f"  FlashAttention (Br={Br}, Bc={Bc}):")
    print(f"    稳态: {mem_after - mem_before:.0f} MB  |  峰值: {peak_flash - mem_before:.0f} MB")

    # 理论 block 大小
    theory_block = B * max(Br, Bc) * max(Br, Bc) * 4 / 1024**2
    print(f"    理论最大 block: {theory_block:.1f} MB (Br×Bc = {Br}×{Bc})")

    if peak_std > 0:
        print(f"\n  HBM 峰值节省: {(1 - peak_flash/peak_std)*100:.0f}%")


def benchmark_speed(device: str = "npu:0", seq_len: int = 2048,
                    d: int = 64, B: int = 1):
    """对比两种 Attention 的执行时间"""
    print(f"\n{'='*60}")
    print(f"  速度对比 (N={seq_len}, d={d})")
    print(f"{'='*60}")

    Q = torch.randn(B, seq_len, d, device=device)
    K = torch.randn(B, seq_len, d, device=device)
    V = torch.randn(B, seq_len, d, device=device)

    # Warmup
    for _ in range(5):
        _ = standard_attention(Q, K, V)
        _ = flash_attention_forward(Q, K, V)
    torch.npu.synchronize()

    # Standard
    t0 = time.time()
    for _ in range(10):
        _ = standard_attention(Q, K, V)
    torch.npu.synchronize()
    time_std = (time.time() - t0) / 10 * 1000

    # Flash
    t0 = time.time()
    for _ in range(10):
        _ = flash_attention_forward(Q, K, V)
    torch.npu.synchronize()
    time_flash = (time.time() - t0) / 10 * 1000

    print(f"  Standard Attention:  {time_std:.1f} ms")
    print(f"  FlashAttention:      {time_flash:.1f} ms")
    if time_flash > 0:
        print(f"  加速比:              {time_std/time_flash:.2f}x")

    # Note: Python-level tiling is not faster — the benefit is memory, not speed.
    # Real FlashAttention gets speedup from SRAM-level kernel fusion.
    if time_flash > time_std:
        print(f"\n  Python 分块导致循环开销 > 节省的访存时间，FlashAttention 的")
        print(f"  真正加速来自 CUDA kernel 级别的 SRAM 管理，Python 无法体现。")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="FlashAttention 简化版: tiling + online softmax"
    )
    parser.add_argument("--verify", action="store_true",
                        help="验证数值精度")
    parser.add_argument("--profile", action="store_true",
                        help="对比显存占用")
    parser.add_argument("--benchmark", action="store_true",
                        help="对比执行速度")
    parser.add_argument("--all", action="store_true",
                        help="运行全部对比")
    parser.add_argument("--seq-len", type=int, default=4096,
                        help="序列长度 (default: 4096)")
    parser.add_argument("--device", default="npu:0")

    args = parser.parse_args()
    run_all = args.all or (not args.verify and not args.profile and not args.benchmark)

    torch.npu.set_device(args.device)

    if args.verify or run_all:
        compare_and_verify(args.device)

    if args.profile or run_all:
        profile_memory(args.device, seq_len=args.seq_len)

    if args.benchmark or run_all:
        benchmark_speed(args.device, seq_len=min(args.seq_len, 2048))


if __name__ == "__main__":
    main()
