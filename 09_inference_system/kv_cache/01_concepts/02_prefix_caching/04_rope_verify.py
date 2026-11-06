#!/usr/bin/env python3
"""
RoPE 与 Prefix Caching 验证脚本。

验证文章 rope_and_prefix_caching.md 的三个核心断言：

  1. 相同 token 内容，不同绝对位置 → K 向量不同（RoPE 的绝对位置副作用）
  2. Hash chaining 隐式编码位置 → 即使内容相同，链长不同则 hash 不同
  3. Rotary Correction: K_{m'} = K_m · R_{m'-m}（从位置 m 的缓存 K 校正到 m'）

用法:
  python3 rope_verify.py              # 运行全部验证
  python3 rope_verify.py -v           # 详细输出每步的向量差值
"""

import argparse
import hashlib
import math

import torch


# ── 1. Minimal RoPE (HuggingFace Llama 兼容风格) ──

def precompute_rotary_freqs(dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
    """
    预计算 cos/sin (max_seq_len, dim//2)。
    每对维度 (2i, 2i+1) 共享同一旋转角 θ_i。
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # (dim//2,)
    positions = torch.arange(max_seq_len).float()                       # (max_seq_len,)
    angles = torch.outer(positions, freqs)                              # (max_seq_len, dim//2)
    return torch.cos(angles), torch.sin(angles)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    RoPE 旋转。对待每对相邻维度 (2i, 2i+1) 施加 2D 旋转。
    x:  (seq_len, n_heads, head_dim)
    cos, sin: (seq_len, head_dim//2)  — 每对维度一组旋转角
    """
    x_rot = x.float()
    head_dim = x_rot.shape[-1]
    # 将维度配对: 偶数维 (d0, d2, ...) 和奇数维 (d1, d3, ...)
    x_even = x_rot[..., 0::2]  # (seq_len, n_heads, head_dim//2)
    x_odd  = x_rot[..., 1::2]
    # broadcast cos/sin over n_heads
    cos = cos.unsqueeze(1)  # (seq_len, 1, head_dim//2)
    sin = sin.unsqueeze(1)
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd  = x_even * sin + x_odd * cos
    # 交替交错还原: [r0_e, r0_o, r1_e, r1_o, ...]
    rotated = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
    return rotated.to(x.dtype)


# ── 2. Hash chaining (模拟 vLLM APC) ──

def hash_block(parent_hash: int, token_ids: tuple) -> int:
    """vLLM 的 hash chaining: hash(parent_hash, block_tokens)。"""
    data = f"{parent_hash}:{token_ids}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'big')


# ── Verification 1: 相同内容, 不同位置 → K 不同 ──

def verify_1_position_sensitivity(verbose: bool):
    """验证断言 1: RoPE 使得相同 token 内容在不同绝对位置产生不同的 K。"""
    print("═══ 验证 1: 相同内容, 不同位置 → K 不同 ═══")

    dim, n_heads = 64, 4
    max_seq = 200
    cos, sin = precompute_rotary_freqs(dim, max_seq)

    torch.manual_seed(42)
    seq_len = 8  # 8 token 的 System Prompt
    k_content = torch.randn(seq_len, n_heads, dim)  # 内容部分

    # 施加 RoPE @ 位置 0
    k_pos0 = apply_rotary_emb(k_content.clone(), cos[:seq_len], sin[:seq_len])

    # 施加 RoPE @ 位置 50 (偏移 50)
    k_pos50 = apply_rotary_emb(k_content.clone(), cos[50:50 + seq_len], sin[50:50 + seq_len])

    diff = (k_pos0 - k_pos50).abs().mean().item()

    print(f"  K @ pos 0   shape: {tuple(k_pos0.shape)}  mean abs: {k_pos0.abs().mean():.4f}")
    print(f"  K @ pos 50  shape: {tuple(k_pos50.shape)}  mean abs: {k_pos50.abs().mean():.4f}")
    print(f"  绝对差值 (mean abs diff): {diff:.6f}")

    if diff > 1e-6:
        print(f"  ✅ 断言成立: 内容相同的 K, 在位置 0 和 50 的数值不同 (diff={diff:.4f})")
    else:
        print(f"  ❌ 断言失败: K 应该不同但 diff≈0")
    print()

    if verbose:
        print(f"  K_pos0 首 token 首 head 的前 4 维:  {k_pos0[0, 0, :4]}")
        print(f"  K_pos50 首 token 首 head 的前 4 维: {k_pos50[0, 0, :4]}")
        print()

    return diff > 1e-6


# ── Verification 2: Hash chaining → 位置隐式编码 ──

def verify_2_hash_chaining(verbose: bool):
    """验证断言 2: hash chaining 让相同 token 序列在不同链位置产生不同 hash。"""
    print("═══ 验证 2: Hash chaining 隐式编码位置 ═══")

    system_prompt_tokens = [101, 202, 303, 404, 505, 606, 707, 808]
    block_size = 4

    # 请求 A: System Prompt 从链 root 开始
    chain_a = 0
    blocks_a = []
    for i in range(0, len(system_prompt_tokens), block_size):
        block_tokens = tuple(system_prompt_tokens[i:i + block_size])
        h = hash_block(chain_a, block_tokens)
        blocks_a.append((chain_a, block_tokens, h))
        chain_a = h

    # 请求 B: System Prompt 前面有 12 个 history block → 链位置不同
    chain_b = 0
    for _ in range(12):
        chain_b = hash_block(chain_b, (999,))
    blocks_b = []
    for i in range(0, len(system_prompt_tokens), block_size):
        block_tokens = tuple(system_prompt_tokens[i:i + block_size])
        h = hash_block(chain_b, block_tokens)
        blocks_b.append((chain_b, block_tokens, h))
        chain_b = h

    print(f"  请求 A block_0 parent_hash: {blocks_a[0][0]:016x}  →  hash: {blocks_a[0][2]:016x}")
    print(f"  请求 B block_0 parent_hash: {blocks_b[0][0]:016x}  →  hash: {blocks_b[0][2]:016x}")

    same_content = blocks_a[0][1] == blocks_b[0][1]
    diff_hash = blocks_a[0][2] != blocks_b[0][2]

    print(f"  token 内容相同: {same_content}")
    print(f"  hash 值相同:    {not diff_hash}")

    if same_content and diff_hash:
        print(f"  ✅ 断言成立: 内容相同但 parent_hash 不同 → hash 不同 → 缓存查找失败")
    else:
        print(f"  ❌ 断言失败")
    print()

    if verbose:
        for j, (ba, bb) in enumerate(zip(blocks_a, blocks_b)):
            print(f"  block {j}: A parent={ba[0]:016x} hash={ba[2]:016x}  "
                  f"|  B parent={bb[0]:016x} hash={bb[2]:016x}")
        print()

    return same_content and diff_hash


# ── Verification 3: Rotary Correction ──

def verify_3_rotary_correction(verbose: bool):
    """验证断言 3: K_{m'} = K_m · R_{m'-m} — 旋转校正公式。"""
    print("═══ 验证 3: Rotary Correction — 从缓存 K_m 校正到 K_{m'} ═══")

    dim, n_heads = 64, 4
    max_seq = 200
    cos, sin = precompute_rotary_freqs(dim, max_seq)

    m, m_prime = 30, 50

    # 用 PyTorch 直接构造旋转矩阵验证 R_{m'-m} · R_m = R_{m'}
    # 取单个频率 θ = 0.5 rad, 手工算 cos/sin 确保零浮点噪声
    theta = torch.tensor(0.5)
    cos_m, sin_m = torch.cos(m * theta), torch.sin(m * theta)
    cos_diff, sin_diff = torch.cos((m_prime - m) * theta), torch.sin((m_prime - m) * theta)
    cos_mp, sin_mp = torch.cos(m_prime * theta), torch.sin(m_prime * theta)

    # 2x2 旋转矩阵
    R_m = torch.tensor([[cos_m, -sin_m], [sin_m, cos_m]])
    R_d = torch.tensor([[cos_diff, -sin_diff], [sin_diff, cos_diff]])
    R_mp = torch.tensor([[cos_mp, -sin_mp], [sin_mp, cos_mp]])

    composed = R_d @ R_m   # R_{m'-m} · R_m
    direct   = R_mp        # R_{m'}

    diff_mat = (composed - direct).abs().max().item()
    d = 20  # m' - m
    print(f"  旋转矩阵可加性验证 (theta=0.5, m=30, diff={d}):")
    print(f"    R_diff · R_m = [{composed[0,0]:.8f}  {composed[0,1]:.8f}]")
    print(f"                   [{composed[1,0]:.8f}  {composed[1,1]:.8f}]")
    print(f"    R_m'         = [{direct[0,0]:.8f}  {direct[0,1]:.8f}]")
    print(f"                   [{direct[1,0]:.8f}  {direct[1,1]:.8f}]")
    print(f"    最大差值: {diff_mat:.2e}")

    if diff_mat < 1e-7:
        print("  ✅ 断言成立: R_diff · R_m = R_m'  (精确)")
    else:
        print("  ❌ 断言失败")
    print()

    # 应用到向量
    x = torch.tensor([1.0, 2.0])
    direct_x = R_mp @ x
    composed_x = R_d @ (R_m @ x)
    diff_x = (direct_x - composed_x).abs().max().item()
    print("  向量旋转验证: R_diff · (R_m · x) = R_m' · x")
    print(f"    直接 R_m'·x:     {direct_x.tolist()}")
    print(f"    合成 R_diff·R_m·x: {composed_x.tolist()}")
    print(f"    最大差值: {diff_x:.2e}")

    if diff_x < 1e-7:
        print("  ✅ 断言成立: Rotary Correction 在数学上精确成立")
        print("     (工业实现用 CUDA kernel 直接计算 R_diff)")
    else:
        print("  ❌ 断言失败")
    print()

    return diff_mat < 1e-7 and diff_x < 1e-7


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="RoPE 与 Prefix Caching 验证脚本 — rope_and_prefix_caching.md")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="打印详细的向量差值")
    args = parser.parse_args()

    results = []
    results.append(("相同内容, 不同位置 → K 不同",
                    verify_1_position_sensitivity(args.verbose)))
    results.append(("Hash chaining 隐式编码位置",
                    verify_2_hash_chaining(args.verbose)))
    results.append(("Rotary Correction K_{m'} = K_m · R_{m'-m}",
                    verify_3_rotary_correction(args.verbose)))

    print("═══ 总结 ═══")
    all_pass = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  全部 3 个断言验证通过。文章结论与数学推导一致。")
    else:
        print("\n  ⚠️ 存在未通过的断言，请检查。")
        exit(1)


if __name__ == "__main__":
    main()
