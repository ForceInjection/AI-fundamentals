# FlashAttention 简化版：Tiling + Online Softmax

本文手写 FlashAttention 的 forward pass，聚焦两个核心算法思想：**online softmax**（用动态 rescaling 单遍计算 softmax）和 **tiling**（分块避免 O(N²) 显存）。目标是理解算法原理——online softmax 中 `correction = exp(m_old - m_new)` 这一行代码为什么能替代标准 softmax 的两遍遍历，以及分块后显存如何从 130 MB 降到 1 MB。全程用纯 Python/PyTorch 实现，不涉及 CUDA kernel。

阅读建议：先通读 §1 理解 O(N²) 瓶颈的来源，再逐段读 §2 的 online softmax 推导（这是全文最核心的数学），最后对照 §3 的伪代码和 §4 的实测数据验证理解。

## 1. 背景

### 1.1 Attention 的 O(N²) 显存问题

标准 self-attention 的计算流程：

```text
S = Q @ K^T / √d      [N, N]  ← 注意力分数矩阵
P = softmax(S)         [N, N]  ← 注意力概率矩阵
O = P @ V              [N, d]  ← 输出
```

其中 S 和 P 都是 [N, N] 的矩阵。当序列长度 N = 4096、FP32 精度时：

- S 占用：4096² × 4 bytes = **64 MB**
- P 占用：4096² × 4 bytes = **64 MB**
- 合计：**128 MB** 仅为了存储中间结果

对于长序列（N = 16384），两个矩阵各占 1 GB，GPT-3 级别的 2048 序列长度下仅 attention 中间结果就需要 ~16 GB。这就是 attention 的 O(N²) 显存瓶颈。

### 1.2 FlashAttention 的核心思想

FlashAttention (Dao et al., 2022) 提出：**不把 N×N 的矩阵写回 HBM，在 on-chip SRAM 中分块计算、当场消费**。

两个关键技术：

- **Tiling（分块）**：将 Q、K、V 切成小块，每次只加载一块到 SRAM，计算完立刻释放
- **Online Softmax（在线 softmax）**：不存储完整 softmax，用动态更新的 running max 和 running sum 增量计算

结果：显存从 O(N²) 降到 O(N)，同时因为减少 HBM 读写（SRAM 带宽远高于 HBM），实际速度反而更快。

### 1.3 本次实现范围

只做 **forward pass**，用纯 Python/PyTorch 实现 tiling + online softmax 算法。目标是理解算法原理，不追求生产级性能（需要 CUDA kernel 级别的 SRAM 管理）。Backward 涉及重计算 + 分段梯度，复杂度是 forward 的 3-5 倍，超出本次学习范围。

---

## 2. Online Softmax 的数学原理

### 2.1 标准 Softmax 需要两遍

标准的 numerically stable softmax：

```text
第一遍: 找最大值
  m = max(x₁, x₂, ..., xₙ)

第二遍: 计算 exp 并求和
  numerator_i = exp(x_i - m)
  l = Σ numerator_i

输出: softmax(x)_i = numerator_i / l
```

为什么需要两遍？因为 softmax 中的 `exp(x_i)` 很容易溢出（x_i = 100 → exp(100) ≈ 2.7e43）。减去最大值 m 后，所有 x_i - m ≤ 0，exp 值在 (0, 1] 之间，数值稳定。

但这个算法要求**先遍历一遍找到 m，再遍历一遍算 exp**，无法与分块计算兼容。

### 2.2 Online Softmax：增量更新

假设数据分两块到达：block₁ = {x₁, x₂, x₃}，block₂ = {x₄, x₅, x₆}。

**处理 block₁：**

```text
m₁ = max(x₁, x₂, x₃)          # 当前最大值
l₁ = Σ exp(x_i - m₁)           # 当前 sum(exp)
```

**处理 block₂——block₂ 中有更大的值怎么办？**

```text
m₂ = max(m₁, max(x₄, x₅, x₆))  # 新最大值（可能更大）

如果 m₂ > m₁：
  # block₁ 的结果需要"打折"——因为之前减的是 m₁，
  # 现在应该减更大的 m₂
  correction = exp(m₁ - m₂)     # ≤ 1，因为 m₂ ≥ m₁
  l₁_corrected = correction * l₁

# block₂ 的新增部分
l₂_new = Σ exp(x_j - m₂)        # j = 4,5,6

# 合并
l_total = correction * l₁ + l₂_new
```

**核心洞察：** `exp(x_i - m_new) = exp(x_i - m_old) * exp(m_old - m_new)`。这个 `exp(m_old - m_new)` 就是 correction 因子——之前的结果不需要重新计算，只需要乘以这个因子进行 rescale。

### 2.3 应用到 Attention

在 FlashAttention 中，online softmax 需要同时维护三个运行状态：

```text
对于每个 Q block（外层循环）:
  m = -∞    ← 运行 max
  l = 0      ← 运行 sum(exp)
  O = 0      ← 运行 softmax-weighted V

  对于每个 K/V block（内层循环）:
    S_block = Q_block @ K_block^T / √d

    m_new = max(m_old, row_max(S_block))
    correction = exp(m_old - m_new)    # ≤ 1

    P_block = exp(S_block - m_new)     # 当前 block 的 softmax 分子

    l_new = correction * l_old + row_sum(P_block)
    O_new = correction * O_old + P_block @ V_block

    m = m_new; l = l_new; O = O_new

  最终: O_output = O / l    ← 除以 sum(exp) 完成归一化
```

这就是全文最核心的 5 行伪代码。`correction` 项的引入使得我们可以在不知道全局 max 的情况下增量计算 softmax，从而**分块处理、不需要存储 N×N 矩阵**。

---

## 3. Tiling 策略

§2 解决了"怎么增量计算 softmax"，本节解决"怎么分块"——把 Q、K、V 切成小块，确保每块的计算都在 on-chip 内存中完成。

### 3.1 分块伪代码

```text
Q [N, d] → Tr 块，每块 Br 行
K [N, d] → Tc 块，每块 Bc 行
V [N, d] → Tc 块，每块 Bc 行

伪代码:
for i in 0..Tr-1:           (外层: Q blocks)
    Q_block = Q[i*Br : (i+1)*Br, :]
    m, l, O = -∞, 0, 0       (重置 online softmax 状态)

    for j in 0..Tc-1:         (内层: K/V blocks)
        K_block = K[j*Bc : (j+1)*Bc, :]
        V_block = V[j*Bc : (j+1)*Bc, :]

        S_block = Q_block @ K_block^T / √d    [Br, Bc]

        # online softmax 更新 (见 §2.3)
        m_new = max(m, max of S_block)
        correction = exp(m - m_new)
        P_block = exp(S_block - m_new)
        l = correction * l + sum(P_block)
        O = correction * O + P_block @ V_block
        m = m_new

    O[i*Br : (i+1)*Br, :] = O / l
```

两层循环的计算总量与标准 attention 完全相同（每个 Q_block 都会与所有 K/V blocks 计算一次）。区别在于：标准 attention 一次性算出完整 N×N 矩阵再 softmax，FlashAttention 在**每个 block 内完成 S→P→O 的完整计算**，S_block 和 P_block 只在 block 范围内存在，不写回 HBM。

### 3.2 显存分析

| 方法           | 最大单次分配                  | 总中间结果                      |
| -------------- | ----------------------------- | ------------------------------- |
| 标准 Attention | N×N 矩阵 (128 MB @ N=4096)    | ~256 MB (S + P + intermediates) |
| FlashAttention | Br×Bc 矩阵 (0.016 MB @ 64×64) | ~0.03 MB (仅 S_block + P_block) |

标准 attention 的 128 MB 来自两个 N×N 矩阵（S 和 P），且它们是同时存在的——S 算完后 P 覆盖其上，但 FP32 下峰值仍为 128 MB。FlashAttention 的 0.016 MB 来自一个 Br×Bc = 4096 个元素的 FP32 矩阵，配合三个 running states（m、l、O，合计约 Br×d × 3 ≈ 12KB），总峰值 < 1 MB。**矩阵尺寸的节省比 = (N² / (Br×Bc))**，当 N=4096、Br=Bc=64 时理论值为 **4096×**。注意这是纯矩阵元素数量的比值——实测 HBM 峰值节省约 130×（130 MB → 1 MB，见 §4.2），差距来自 running states（m, l, O）和框架内存开销。随着 N 增大，这些固定开销占比下降，实际节省比会趋近理论值。

### 3.3 为什么 Python 实现不加速？

真正的 FlashAttention 把 tiling 写进 CUDA kernel，在 GPU/NPU 的 SRAM 中完成所有计算和中间存储。Python 的 for 循环开销（每个 iteration 都是一次 Python→C++→NPU kernel 调用）远大于节省的 HBM 访存时间。但显存节省是真实的——我们的实现确实避免了分配 N×N 矩阵。

**从 Python 到生产级需要做什么：** 将两层 for 循环融合为一个 CUDA/Triton kernel；手动管理 SRAM 的加载和驱逐（double buffering）；利用 warp-level primitives 做线程间通信。这些优化需要 C++/Triton 级别的编程，超出了本次学习范围，但核心算法逻辑（online softmax + tiling）完全一致。

---

## 4. 测试结果

测试环境：Ascend 910B3, CANN 8.0.1, NPU 7。

### 4.1 精度验证

| 配置 (B, N, d) | max_diff | 结果 |
| -------------- | -------- | ---- |
| (1, 256, 64)   | 1.19e-07 | ✓    |
| (1, 512, 64)   | 1.34e-07 | ✓    |
| (1, 1024, 64)  | 1.79e-07 | ✓    |
| (4, 512, 64)   | 2.09e-07 | ✓    |

所有配置的 max_diff < 1e-6，远低于 1e-3 的目标。online softmax 的数值精度与标准两遍 softmax 完全等价（差异仅来自浮点舍入误差的顺序不同）。

### 4.2 显存对比 (N=4096, d=64)

| 方法           | 峰值 HBM   | 说明                                       |
| -------------- | ---------- | ------------------------------------------ |
| 标准 Attention | **130 MB** | 接近理论值 128 MB（S:64MB + P:64MB）       |
| FlashAttention | **1 MB**   | 仅存 Br×Bc=64×64 的 block + running states |
| **节省**       | **95%**    | O(N²) → O(N) 的显存优势充分体现            |

### 4.3 速度对比

Python 实现的 FlashAttention 比标准 Attention 慢（341ms vs 0.1ms @ N=2048），这是预期行为。真正的 FlashAttention 通过以下工程优化获得加速：

1. **Kernel Fusion**：整个 attention 计算融合为一个 CUDA kernel，消除 kernel launch 开销
2. **SRAM 管理**：手动管理 on-chip SRAM 的加载/驱逐，最大化重用
3. **Warp-level 优化**：利用 GPU 的 warp 调度减少同步开销

这些优化需要在 CUDA C++ 或 Triton 层面实现，Python 无法做到。但 Python 实现的显存节省是真实的，而且算法逻辑（online softmax + tiling）与生产级实现完全一致。

---

## 5. 代码结构

`flash_attention.py` 约 190 行，分为三组：核心算法（standard attention + flash attention forward）、验证工具（精度/显存/速度对比）、CLI 入口。

```text
09_flash_attention/
└── flash_attention.py    # FlashAttention 简化版（~190 行）
    ├── standard_attention()       — 标准 PyTorch attention (baseline)
    ├── flash_attention_forward()  — tiled + online softmax forward
    ├── compare_and_verify()       — 数值精度对比
    ├── profile_memory()           — HBM 峰值对比
    └── benchmark_speed()          — 执行速度对比
```

---

## 6. 与之前 phase 的联系

本 phase 的 FlashAttention 不是孤立的算法实验——它与之前多个 phase 直接关联：

| Phase               | 关联                                                            |
| ------------------- | --------------------------------------------------------------- |
| Phase 8 (Mini-GPT)  | `CausalSelfAttention` 中的标准 attention 是本 phase 的 baseline |
| Phase 7 (Profiling) | profiling 方法用于验证 HBM 占用差异                             |
| Phase 1 (Hello NPU) | Q@K^T 矩阵乘法的性能决定了 attention 的效率                     |

---

## 7. 后续扩展

本实验只实现了 FlashAttention 的 forward pass，以下四个方向是最自然的延伸：

- **Backward pass**：实现重计算 + 分段梯度，理解训练时的显存节省（代码量约为 forward 的 3 倍）
- **Causal Mask 集成**：在 tiling 过程中融入 causal mask，用于 GPT 等 decoder 模型
- **Triton 实现**：用 Triton 语言写 NPU kernel，获得接近原生 CUDA 的性能
- **Multi-Query Attention (MQA) / Grouped-Query Attention (GQA)**：进一步减少 K/V 的显存开销

## 参考链接

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)](https://arxiv.org/abs/2307.08691)
- [Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)
