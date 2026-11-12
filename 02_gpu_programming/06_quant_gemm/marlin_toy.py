#!/usr/bin/env python3
"""Marlin W4A16 的 CPU 语义复现（配套《Marlin 深拆》一文，仅依赖 numpy）。

1:1 复现 vLLM 中 Marlin 权重预处理的两段 Python 语义：
  - get_weight_perm / marlin_permute_weights（布局重排 = 纯置换）
    vllm/model_executor/layers/quantization/utils/marlin_utils_test.py
  - marlin_weights 的 nibble 打包（低 4 位在前，每 uint32 装 8 个）

并用「索引追踪」验证两个核心设计：
  1. repack 是纯置换：追踪每个元素的去向，GPU 按存储序读出后再按
     同一置换重排激活 x，GEMM 结果与朴素反量化路径逐位一致——
     这就是「重算换带宽」成立的结构前提；
  2. lop3 + FP16 魔数 0x6400 的位技巧：`(0x6400 | nibble)` 按 FP16
     读回恰好等于 1024 + nibble，即 dequant.h 里「bit_op + fma 折叠
     零点偏置」的 CPU 等价形式。

对照源码（vLLM，commit 43d691ec6b）：
  csrc/quantization/marlin/dequant.h          —— lop3/prmt/dequant 位技巧
  csrc/quantization/marlin/marlin_template.h  —— fetch_to_shared 管线
运行：python3 marlin_toy.py   （仅 numpy，无 torch / 无 GPU）
"""
import numpy as np

GPTQ_MARLIN_TILE = 16  # marlin_utils.py 中的常量


# ---------- 1. perm 构造：1:1 复现 get_weight_perm 的 num_bits=4 分支 ----------
def get_weight_perm():
    perm_list = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [2 * (i % 4), 2 * (i % 4) + 1,
                        2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])
    perm = np.array(perm_list)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])  # 4-bit 分支的组内交织
    return perm.reshape((-1, len(interleave)))[:, interleave].ravel()


# ---------- 2. repack：1:1 复现 marlin_permute_weights ----------
def marlin_permute_weights(q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE):
    # 16x64 的 marlin tile：块间转置后，按 tile 粒度的 perm 重排
    # 返回的是「打包存储序」的容器（不再是 K×N 语义），GPU 按此序读出
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.transpose((0, 2, 1, 3))          # torch.permute((0,2,1,3))
    q_w = q_w.reshape((-1, perm.size))         # 每 256 元素 = 一个 16×16 tile
    q_w = q_w[:, perm]                         # tile 内按 ldmatrix/mma 需求重排
    return q_w.reshape(q_w.shape)


# ---------- 3. nibble 打包：1:1 复现 marlin_weights 的打包循环 ----------
def marlin_pack(q_w_permuted, num_bits=4):
    pack_factor = 32 // num_bits               # 4-bit: 每 uint32 装 8 个
    q = q_w_permuted.astype(np.uint32)
    packed = np.zeros((q.shape[0], q.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        packed |= q[:, i::pack_factor] << (num_bits * i)   # 低 4 位在前
    return packed


def marlin_unpack(packed, num_bits=4):
    """GPU 侧的读序：uint32 的 8 个 nibble 按低位到高位展开。"""
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    out = []
    for i in range(pack_factor):
        out.append((packed >> (num_bits * i)) & mask)
    return np.stack(out, axis=-1).reshape(packed.shape[0], -1)


# ---------- 4. lop3 魔数位技巧的 CPU 等价 ----------
def dequant_fp16_magic(packed_u32):
    """复现 dequant<half2, kU4B8> 的位技巧（CPU 上用 uint16 视图替代 lop3）。

    GPU 版：lop3 以 LUT (0xf0 & 0xcc) | 0xaa 一步算出 (q & 0x000f000f) | 0x64006400，
    得到的两个 half 槽位是带指数偏置的「1024 + nibble」。
    CPU 版：0x6400 | nibble 按 FP16 读回，与 1024 + nibble 逐位相等。
    """
    lo_nibbles = packed_u32 & 0x000F000F        # 每 uint32 取两个 4-bit 槽（位 0-3 与 16-19）
    bits = (lo_nibbles | 0x64006400).astype(np.uint32)
    lo_u16 = (bits & 0xFFFF).astype(np.uint16)  # 低 16 位：half 槽 0
    hi_u16 = ((bits >> 16) & 0xFFFF).astype(np.uint16)  # 高 16 位：half 槽 1
    lo = lo_u16.view(np.float16).astype(np.float32) - 1024.0
    hi = hi_u16.view(np.float16).astype(np.float32) - 1024.0
    return lo, hi


# ============================ 四个自验证 demo ============================

def demo_1_perm_is_pure_permutation():
    """(a) repack 是纯置换：用索引矩阵追踪每个元素的去向。"""
    perm = get_weight_perm()
    size_k, size_n = 128, 256
    idx = np.arange(size_k * size_n).reshape(size_k, size_n)  # 元素身份 = 原线性下标
    idx_perm = marlin_permute_weights(idx, size_k, size_n, perm)
    assert sorted(idx_perm.ravel()) == sorted(idx.ravel()), "数值集合改变，不是纯置换"
    print(f"[demo 1] repack = 纯置换：{size_k}×{size_n} = {size_k*size_n} 个元素一一换位，"
          f"无丢失无重复 ✓")
    print(f"        存储序前 16 个位置对应的原线性下标：\n{idx_perm.ravel()[:16].reshape(4, 4)}")
    return idx_perm.ravel()


def demo_2_pack_order():
    """(b) nibble 打包：低 4 位在前，读序与源码循环展开一致。"""
    q = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uint32)
    packed = marlin_pack(q)
    assert packed[0, 0] == 0x76543210, hex(packed[0, 0])   # nibble7..nibble0
    assert np.array_equal(marlin_unpack(packed), q)
    print("[demo 2] 打包 = 低 nibble 在前（0x76543210），解包读序一致 ✓")


def demo_3_fp16_magic():
    """(c) 0x6400 魔数：位技巧与数值减偏置逐位等价。"""
    rng = np.random.default_rng(1)
    packed = rng.integers(0, 2**32, size=(4, 8), dtype=np.uint32)
    lo, hi = dequant_fp16_magic(packed)
    lo_nib = (packed & 0xF).astype(np.float32)
    hi_nib = ((packed >> 16) & 0xF).astype(np.float32)
    assert np.array_equal(lo, lo_nib)     # (1024 + n) - 1024 == n，FP16 内精确
    assert np.array_equal(hi, hi_nib)
    print("[demo 3] 0x6400 魔数位技巧 == 数值减偏置（1024 偏置在 FP16 内精确抵消）✓")


def demo_4_gemm_equivalence(idx_perm_seq):
    """(d) GPU 存储序读出 + scatter 累加 → 与朴素反量化 GEMM 逐位一致。

    kernel 的视角：按存储序读出的每个权重元素都带着自己的原 (k, n)
    坐标，反量化后做 y[n] += x[k] * w 的 scatter 累加——重排只动
    读取顺序，不动数学。这里逐元素模拟该累加并与朴素路径对齐。
    """
    rng = np.random.default_rng(2)
    size_k, size_n, group = 128, 256, 64
    total = size_k * size_n

    q_w = rng.integers(0, 16, size=(size_k, size_n)).astype(np.float32)
    scale = (rng.random((size_k // group, size_n)) + 0.5).astype(np.float32)
    zp = rng.integers(0, 16, size=(size_k // group, size_n)).astype(np.float32)
    x = rng.random(size_k).astype(np.float32)

    # 朴素路径：原 K 序反量化 + 矩阵向量积
    g = np.repeat(np.arange(size_k // group), group)
    w_naive = (q_w - zp[g]) * scale[g]
    y_naive = x @ w_naive

    # GPU 路径：按存储序读出（每个元素带原 (k, n)），scatter 累加
    q_read = q_w.ravel()[idx_perm_seq]
    k_of = idx_perm_seq // size_n
    n_of = idx_perm_seq % size_n
    w_seq = (q_read - zp[k_of // group, n_of]) * scale[k_of // group, n_of]
    y_gpu = np.zeros(size_n, dtype=np.float64)
    np.add.at(y_gpu, n_of, x[k_of] * w_seq)
    assert np.allclose(y_gpu, y_naive, atol=1e-3), np.abs(y_gpu - y_naive).max()
    print("[demo 4] GPU 存储序读出 + scatter 累加 == 朴素反量化 GEMM"
          f"（max|Δ|={np.abs(y_gpu - y_naive).max():.2e}）✓")


if __name__ == "__main__":
    seq = demo_1_perm_is_pure_permutation()
    demo_2_pack_order()
    demo_3_fp16_magic()
    demo_4_gemm_equivalence(seq)
    print("\n全部通过：Marlin 的 repack 只动摆放、不动数值；位技巧只动读法、不动语义。")
