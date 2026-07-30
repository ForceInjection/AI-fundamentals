# FlashAttention 深度解析：从 IO-Aware Tiling 到 Hopper 异步计算

> 2026-07-30 | 基于 FlashAttention (Dao et al., 2022)、FlashAttention-2 (Dao, 2023)、FlashAttention-3 (Shah et al., 2024) 论文及 vLLM 源码分析

FlashAttention 可能是过去五年对 LLM 推理影响最大的单一算子。今天 vLLM、SGLang、HuggingFace TGI 都用它——但大多数工程师只知道"启用 FlashAttention 会变快"，不知道它**为什么**快。

快的原因不是用了更好的矩阵乘法。FlashAttention 和标准 attention 算的是完全相同的公式——`S = softmax(QK^T/√d) × V`，一行没改，计算量一点没少。区别只有一个：**中间结果存在哪里。** 标准做法把每一步的中间矩阵老老实实写回 HBM，FlashAttention 让它们永不离开 SRAM。这一字不改、仅靠重排计算顺序的设计，是它唯一的创新，也是它快全部的原因。

---

## 一、标准 Attention 为什么慢

### 1.1 GPU 的显存金字塔

GPU 有两级显存，它们之间差了整整一个数量级：

| 显存层级                 | A100 (FA1/FA2 基准)         | H100 (FA3 基准)              | 访问延迟    | 编程模型                         |
| ------------------------ | --------------------------- | ---------------------------- | ----------- | -------------------------------- |
| **HBM**（高带宽显存）    | 40–80 GB，~1.5–2.0 TB/s     | 80 GB，~3.35 TB/s            | ~300 cycles | 全局可见，所有 SM 都能读         |
| **SRAM**（片上共享内存） | 192 KB/SM，108 SM 共 ~20 MB | ~228 KB/SM，132 SM 共 ~30 MB | ~30 cycles  | 仅单个 SM 的 thread block 可访问 |

把 SRAM 想象成你桌面上的便签——伸手就能拿，但只能放几张纸条。HBM 是身后的书架——能装很多东西，但每次取放都要站起来走过去。

标准 attention 的问题：它总是**先写在便签上，然后放回书架，再从书架上拿回来继续算**——中间结果在 HBM 和 SRAM 之间往返了太多趟。

### 1.2 标准 Attention 的 HBM 往返

以单头、序列长度 N、头维度 d 为例，一次 forward pass 的标准实现需要：

```text
Step 1: S = Q × K^T         → 写 S [N×N] 到 HBM         (1 次写)
Step 2: P = softmax(S)      → 读 S [N×N]，写 P [N×N]    (1 次读 + 1 次写)
Step 3: O = P × V           → 读 P [N×N]，写 O [N×d]    (1 次读 + 1 次写)
```

**HBM 访问总量 = O(N²) + O(Nd)。** 当 N 很大时，N² 项主导——一个 128K 序列的单头 attention，仅中间矩阵 S 和 P 就各自是 16B 个元素。BF16 精度下：16B × 2 bytes = 32 GB。一张 H100 只有 80 GB HBM，**单个中间矩阵就占掉了 40% 的显存。**

更致命的是带宽瓶颈：把 32 GB 写入 HBM 再读回，仅数据传输就需要 32 × 2 / 3.35 ≈ 19 ms——这还只是读/写**一次**中间结果的耗时。在多 head（如 64 头）、多层（如 43 层）的实际场景中，读写所有中间矩阵的 HBM 带宽开销是计算本身的数十倍。

这暴露了一个事实：**标准 attention 的瓶颈不是 FLOPS，而是 HBM 带宽。** GPU 的 Tensor Core 在做 GEMM 时可以达到 70%+ 的利用率，但大部分时间它们都在等数据从 HBM 搬过来。

---

## 二、核心洞察：改变计算顺序而非计算公式

FlashAttention 的核心创新不是一个新的数学公式——S = softmax(QK^T/√d) × V 这行公式一字不改。创新在于**怎么算**。用烹饪来类比：

标准做法是把所有食材（Q、K、V）摊在巨大的操作台上（HBM），一步步切配、调味、装盘，每完成一步都要把中间产物放回冰箱再拿出来——因为操作台不够大，放不下所有东西。

FlashAttention 的做法：每次只取一小份食材到案板（SRAM），在这块小案板上完成切配+调味+装盘的全部工序——只把最后做好的菜端走。中间过程全部在案板上完成，不和冰箱打交道。

这就是分块计算（tiling）的核心直觉：

```text
标准做法:  全量材料化 S → 全量 softmax → 全量乘 V → 中间结果全写 HBM
FlashAttention: Q 分块 × K/V 分块 → 每块在 SRAM 中完成 attention → 仅最终 O 写 HBM
```

如果矩阵乘法天然可以分块（`C[i,j] = Σ_k A[i,k] × B[k,j]`——每对 (i,j) 独立），为什么 attention 不能？

### 2.1 唯一的障碍：Softmax 不是块可分解的

矩阵可以分块乘，因为输出元素不依赖全局状态。但 softmax 不是——`softmax(x)_i = exp(x_i) / Σ_j exp(x_j)`，分母依赖**全行**的所有元素。如果按行分块、每块独立做 softmax，每块得到自己的"本地分母"，拼起来不等于全局 softmax。

标准 softmax 需要两遍遍历（先找 max 防 overflow，再求 sum 做归一化），这两遍中间把数据存在 HBM 中——这就是那个 N×N 矩阵被写回 HBM 的根本原因。

### 2.2 Online Softmax：维护 running max 和 running sum

FlashAttention 的解决方案是一次遍历完成 softmax。算法的核心是三个变量的递推更新——只需 running max 和 running sum：

```text
对于每块新数据：
  m_new = max(m_old, m_block)                               ← 更新全局 max
  l_new = l_old × exp(m_old - m_new)                        ← 旧 sum 按旧 max 缩放
        + Σ exp(x_block - m_new)                            ← 新块的 sum（用新 max 缩放）
```

这两个变量之所以足够，是因为 `exp(x_i - m_new) = exp(x_i - m_old) × exp(m_old - m_new)`——之前所有元素的指数值可以"重新缩放"到新 max 下，不需要回退到原始数据，只需要旧 max 和新 max 的差值。遍历一块数据、更新 running max 和 running sum、这块数据就可以丢弃了——永远不需要把所有指数值同时留在内存中。

### 2.3 回到 FlashAttention 的上下文：Online Softmax + Tiling

将 online softmax 应用到 attention 中。以下算法直接对应 FA1 论文 Algorithm 1（注：FA1 在每步内层循环将 O 除以 ℓ 以保持归一化；FA2 将除法延后到循环结束，省去了每步的除法开销——两者数学等价，FA2 更高效）：

```text
外层循环：Q 的每个 block (在 SRAM 中)
  O_i ← 0, m_i ← -∞, ℓ_i ← 0              ← 每个 Q 块有独立的 running 状态

  内层循环：K/V 的每个 block (加载到 SRAM 中)
    S_ij = Q_i × K_j^T                         ← 在 SRAM 中完成 GEMM
    m_new = max(m_i, rowmax(S_ij))             ← online softmax: 更新 running max
    P_ij = exp(S_ij - m_new)                   ← 用新 max 重新缩放，未归一化
    ℓ_new = ℓ_i × exp(m_i - m_new)             ← 旧 sum 按比例缩放
          + rowsum(P_ij)                       ← 新块的 sum
    O_i = diag(ℓ_new)⁻¹ × (                    ← FA1 每步归一化
           diag(ℓ_i) × exp(m_i - m_new) × O_i  ← 旧输出按比例缩放
         + P_ij @ V_j )                        ← 新块的贡献
    m_i = m_new, ℓ_i = ℓ_new

  内层循环结束后：
    O_i 已归一化，直接写回 HBM
```

关键的工程约束：每对 `(Q_i, K_j)` 的矩阵乘法和 softmax 更新**完全在 SRAM 中完成**——中间矩阵 `S_ij`（大小为 Q_block × KV_block）和 `P_ij` 永远不会离开 SRAM。只写最终归一化的 O_i 到 HBM。

---

## 三、IO 复杂度分析：数学上为什么减少

上面的算法描述了"怎么做"——Q 分块、K/V 分块、SRAM 内循环。但它到底省了多少 HBM 访问？需要用 IO 复杂度来定量回答。答案取决于三个变量：序列长度 N（越大越受益）、头维度 d（越小越受益）和 SRAM 大小 M（越大越受益）。

### 3.1 标准 Attention 的 IO 下界

顺序读取权重 Q、K、V 需要 O(Nd) HBM 访问，这无法避免。但写回中间矩阵 S（N×N 个元素→O(N²d) 字节）是额外的——且不是正确性必需的。

- 标准 attention：Θ(Nd + N²) HBM 访问。N² 项来自 S 和 P 的读写。
- FlashAttention：Θ(N²d²/M) HBM 访问，其中 M = SRAM 大小。

### 3.2 为什么 d²/M 这个因子很关键

FA1 的 IO 复杂度公式（定理 2）揭示了 tiling 的收益：FlashAttention forward 的 HBM 访问次数与 `N²d²/M` 成正比，而标准实现与 `N²` 成正比。d²/M 这个因子在典型 GPU 配置下是 0.1–0.2——意味着 FA 的 HBM 访问只有标准实现的 10–20%。

论文的实测数据验证了这一点：FA1 paper §4 中，对于 GPT-2 medium（N=1024, d=64, batch=64），标准 attention 的 HBM 读写量为 40.3 GB，FlashAttention 为 4.4 GB——实际 HBM 访问减少了 9.2×，运行时间从 41.7 ms 降到 7.3 ms（5.7× 加速）。对于更长的序列，N² 项主导，加速比进一步增大。

### 3.3 当 N 越大，差距越悬殊

标准实现的 N² 项随序列长度平方增长。以 d=128 为例：N 从 1K 增长到 128K 时，标准实现的 HBM 访问增长 ~16,000×（受 N² 主导），而 FlashAttention 的增长同样受 N² 主导但每元素的 HBM 访问被 d²/M 折扣。在 1M 上下文场景，N² = 10¹²——如果没有 FA 的 tiling，仅存储每个 head 的中间 attention matrix 就需要 TB 级的显存，根本不可能在单机上运行。这解释了为什么长序列生成中 FlashAttention 是不可或缺的基础设施。

---

## 四、Backward：为什么重算比存储更便宜

上述分析只覆盖了 forward pass。但 LLM 的权重是通过训练得到的——训练需要 backward。既然 FlashAttention 的 forward 不存 `S` 和 `P`，backward 需要它们时怎么办？答案是用算力换显存。

### 4.1 训练 Backward 的额外存储需求

深度学习训练中，backward 需要 forward 的中间结果。标准 attention 在 forward 时已经将 S 和 P 写到 HBM 中——存储不是问题（本来就要写）。但 FlashAttention 的 forward 根本不存 S 和 P——它们只在 SRAM 中出现，计算完就被覆盖了。

不存照片，存拍摄参数。如果你拍了一张照片但存储空间有限，与其保存这张巨大的 JPEG（几 MB），不如保存拍摄时的参数——光圈、快门、ISO、焦距（几十 bytes）——需要时花几毫秒重新冲洗。反正你的相机处理器闲着也是闲着。

这就是 FlashAttention backward 的思路：不存 N×N 的 softmax 结果矩阵 P（2 TB for 128K seq），而是存 forward 的统计量 m（running max）和 l（running sum）——每个 Q 行只需 2 个标量，总共 32 MB。backward 需要 P 时，用存储的 Q、K、V 和 m、l 重新计算——数学上精确到逐 bit 相同。

### 4.2 Recomputation 公式

给定 Q_i、K_j、V_j 和统计量 m_i、l_i（全部从 HBM 读取），在 backward 中可精确重建 P：

```text
S_ij = Q_i × K_j^T
P_ij = exp(S_ij - m_i) / l_i      ← 和 forward 逐 bit 相同
```

以 128K 序列、64 头、128 维 head、BF16 为例：存储全量 P = 2 TB（远超单 GPU HBM），存储 (m, l) = 32 MB——差距 ~64,000 倍。在 per-block 粒度上，SRAM 内重算一次小 GEMM 远便宜于从 HBM 读回对应的 P 分块。

---

## 五、FlashAttention-2：榨干非矩阵乘开销

FA1 的 IO-aware tiling + online softmax + recomputation 已经解决了最大的瓶颈——HBM 带宽。但当 IO 不再是限制因素后，之前被掩盖的短板开始暴露。FA2 论文指出，在 A100 上每个非矩阵乘 FLOP 的代价是矩阵乘 FLOP 的 **16 倍**（312 TFLOPs/s matmul vs 19.5 TFLOPs/s non-matmul FP32）——reshape、permute、warp 间同步等非 GEMM 操作虽然 FLOP 占比不大，却占据了可观的 wall-clock 时间。

### 5.1 更好的并行化策略

FA1 只在 batch 和 head 维度并行——序列维度的所有 Q 块由单个 thread block 串行处理。这意味着处理长序列时，大量 SM 空闲。FA2 的关键洞察：在 forward 阶段，**不同的 Q 块之间没有数据依赖**（各自输出独立的 O 行），可以同时让多个 SM 分别处理不同的 Q 块——将并行度从 batch×head 扩展到 batch×head×sequence。在 A100 上 108 SM 的资源下，这种三维并行让短序列也能用满所有 SM。

### 5.2 减少 warp 同步

FA1 在每个 thread block 内，所有 warp 需要将中间结果写到 shared memory、同步、再加总——多次 warp 间同步强制最快的 warp 等最慢的 warp。FA2 通过重新编排 warp 的工作分配——将 Q 分给各 warp 各自处理、K 和 V 对全部 warp 可见——消除了 forward 中 warp 间的通信需求，不再需要 shared memory 的写-同步-读循环。

### 5.3 性能对比

在 A100 上，FA2 相比 FA1：

- 短序列（512-1K）prefill：1.5-2× 加速
- 长序列（8K+）prefill：~2× 加速
- Decode（N=1）：改进较小（~1.1-1.3×），因为序列长度短，非矩阵乘开销影响小

---

## 六、FlashAttention-3：利用 Hopper 的异步硬件

FA2 在 A100 上达到了 73% 的峰值利用率。但同样的 kernel 原封不动搬到 H100 上，利用率骤降到 35%——不是 kernel 变差了，是 Ampere 的并行策略没有利用 H100 最关键的硬件特性。Ampere 上计算和 load/store 必须串行：load 数据 → 等 → 计算 → 存结果 → 等 → 循环。Hopper 打破了这个限制，引入了两个关键的异步能力。

### 6.1 TMA：独立的异步拷贝引擎

Hopper 引入了 TMA（Tensor Memory Accelerator）——专门从 HBM 拷贝数据到 shared memory 的硬件单元，完全独立于 SM 的计算管线。FA3 的用法：

```text
在 SRAM 中对当前 K/V block 做 GEMM 的同时：
  TMA 异步拷贝下一个 K/V block 从 HBM → shared memory
  （不需要消耗 CUDA core 的时钟周期）

GEMM 完成时：
  下一个 block 的数据已经在 shared memory 中了
  立即开始下一轮 GEMM，TMA 继续拷贝再下一个 block
```

这消除了"等待 load"的空闲——计算和 HBM 搬运完全重叠。`cp.async`（Ampere 的异步拷贝）和 `wgmma`（Hopper 的异步矩阵乘法）让 TMA 拷贝和 MMA 可以在同一时刻分别使用总线和 Tensor Core。

### 6.2 Warpgroup MMA：整组 warp 异步发射矩阵乘法

Ampere/Turing 的 `mma.sync` 指令：一个 warp 发射一条矩阵乘法指令，**必须等结果返回**才能继续执行下一条指令。Hopper 引入 `wgmma`（warpgroup matrix multiply-accumulate）：4 个 warp 组成一个 warpgroup，发射一条异步 MMA 指令——不等待结果——然后**立即继续执行其他 warp 的指令**。结果返回时，发射 warp 可以"异步接收"。

FA3 的调度：warpgroup 发射完 MMA 后，不等结果，转而去处理 softmax 的在线更新（纯标量 + 逐元素操作）——计算和 MMA 在指令级交错，而非顺序等待。

### 6.3 FP8 支持

FA3 原生支持 E4M3 FP8 的 K 和 V 输入（BF16/FP16 的 Q，FP32 累积）。K/V 从 HBM 读取带宽直接减半（FP8: 1 byte/element vs BF16: 2 bytes/element）。量化/反量化在 SRAM 内完成——仍然是 IO-aware 的设计。对于推理场景中 KV cache 以 FP8 存储的配置（如 vLLM 的 `--kv-cache-dtype fp8`），这无需单独的反量化步骤——FA3 可以直接消费 FP8 的 KV cache。

### 6.4 性能对比

在 H100 上，FA3 相比 FA2（source: FA3 论文 §5）：

- FP16 forward：1.5–2.0× 加速，达到 740 TFLOPs/s（75% 利用率）。对比 FA2 在 H100 上仅 35% 利用率
- FP16 backward：1.5–1.75× 加速
- FP8 forward：接近 1.2 PFLOPs/s

### 6.5 硬件兼容性

vLLM 源码（`flash_attn_interface.py:52-68`）根据 GPU 架构自动选择 FA 版本：

```python
# FA2: compute capability >= 8.0 (Ampere A100, A6000, etc.)
if not current_platform.has_device_capability(80):
    return False, "FA2 requires compute capability >= 8"

# FA3: compute capability 9.x (Hopper H100, H800, etc.)
if not current_platform.is_device_capability_family(90):
    return False, "FA3 requires compute capability 9.x"
```

A100 (sm_80) 用 FA2，H100 (sm_90) 用 FA3——框架在 backend selection 中自动处理，用户不需要手动指定版本。vLLM 还引入了 FA4（支持 sm_90/sm_100/sm_110），默认的 `DEFAULT_FA_VERSION = 2` 保证了最广泛的向后兼容性，而较新的 FA3/FA4 版本在后端选择中优先在支持的硬件上激活。

---

## 七、FlashAttention 的变体与推理场景的适配

FA1-4 解决的都是同一个场景：Q、K、V 是 dense tensor，每个 token 关注所有历史 token。但推理场景中这两个假设经常不成立——MLA 的 KV 只在低维压缩空间、PagedAttention 的 KV 分布在物理不连续的 block 中。为此，社区围绕 FA 的核心技术（IO-aware tiling + online softmax）构建了两个重要的专用 kernel。

### 7.1 FlashMLA：MLA 的专用 kernel

MLA（DeepSeek V2/V3）的 KV 在压缩空间中（kv_lora_rank=512 + k_rope=64 = 576 维）。标准 FA 要求 Q、K、V 维度对齐——无法直接处理 MLA 的"KV 是 576 维压缩向量、需要矩阵吸收后再展开"的计算模式。

FlashMLA 是专门为 MLA 设计的 CUDA kernel。它将 MLA 的"矩阵吸收"步骤（将 `W_UK` 吸收进 Q）融合进 attention 计算——在 SRAM 中完成吸收 + 点积，不在 HBM 中物化展开后的 K 和 V。这保留了 FA 的 IO-aware 特性，同时适配了 MLA 的压缩运算。

vLLM 中 FlashMLA 作为一个独立的 attention backend 注册（`FlashMLABackend`），仅对 MLA 架构的模型（DeepSeek V2/V3）自动激活。

### 7.2 FlashInfer：分页、变长、稀疏的专用 kernel

FlashInfer 是另一个围绕 IO-aware tiling 构建的推理库，但专攻 FA 不适配的场景：

- **分页 KV cache**（PagedAttention）：KV cache 不是连续存储，而是分布在物理不连续的 block 中。FlashInfer 的 kernel 接受 block table 作为输入，在 SRAM 中做地址转换 + attention 计算。
- **变长序列**：batch 中不同请求的序列长度不同，用 `varlen` kernel 避免 padding 浪费。
- **GQA/MQA 高效广播**：GQA 中 8 个 KV 头被 64 个 Q 头共享——FlashInfer 的 kernel 在 load KV 时内置 `repeat_kv` 广播，不需要在 SRAM 和 HBM 之间来回拷贝。

### 7.3 三者关系

FlashAttention 是通用基座——"所有 token 两两做 attention"的标准答案。FlashMLA 和 FlashInfer 分别是"压缩 KV cache 的 attention"和"分页/变长/稀疏的 attention"的专用解。三者共享同一个核心洞察——**永远不让中间 attention matrix 离开 SRAM**——但在适配具体场景时各自优化了不同的数据布局和 load 策略。

---

## 八、总结

FlashAttention 的核心创新不是新的数学公式——它用和标准 attention **完全相同**的 S = softmax(QK^T/√d) × V。创新在于**计算顺序的重新编排**：

```text
标准:  算全量 S → 写 HBM → 读 HBM 做 softmax → 写 P → 读 P 乘 V → 写 O
FlashAttention: Q分块 × K/V分块 → SRAM内完成 GEMM+softmax+output → 仅 O 写 HBM
              ↑ 中间矩阵 S_ij 和 P_ij 从未离开 SRAM
```

这个编排之所以有效，根源于 GPU 显存金字塔的物理事实：**计算比搬数据更便宜。** FA1 论文的实测数据直观地展示了这一点——GPT-2 medium（N=1024, d=64）上，FlashAttention 的 FLOPs 比标准实现更多（75.2 vs 66.6 GFLOPs），但因为 HBM 访问从 40.3 GB 骤降到 4.4 GB，运行时间从 41.7 ms 降到 7.3 ms——多算了 13% 的 FLOPs，反而快了 5.7×。当 N 增长到 128K 时，标准实现的 HBM 访问被 N² 项淹没（~32 GB 的中间矩阵每层都要写读一轮），而 FA 的 HBM 访问仅以 O(N²d²/M) 增长——差距从 5.7× 扩大到 10× 以上。

从 FA1（IO-aware tiling + online softmax）到 FA2（消除非矩阵乘瓶颈）到 FA3（Hopper TMA + warpgroup MMA 异步计算），每一代都更深地利用了当前 GPU 架构的硬件特性。演进方向很明确：**计算和 HBM 搬运的重叠越彻底、越异步，kernel 的带宽利用率越高。**

---

## 延伸阅读

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) — FA1 原始论文
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Dao, 2023) — FA2 改进
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision](https://arxiv.org/abs/2407.08608) (Shah et al., 2024) — FA3 Hopper 优化
- [vLLM FlashAttention Backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py) — 工业集成
- [vLLM `flash_attn_interface.py`](https://github.com/vllm-project/vllm/blob/main/vllm/vllm_flash_attn/flash_attn_interface.py) — FA2/FA3/FA4 版本选择
- [PagedAttention 退役的技术原因](pagedattention_retirement.md)
- [MLA 的 TP 切分：为什么 8 张 GPU 存了同一份 KV cache](mla_tp_kv_redundancy.md)
