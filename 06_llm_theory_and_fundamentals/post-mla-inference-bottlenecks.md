# MLA 之后，推理系统的新瓶颈在哪里：DeepSeek-V4 与 Kimi K3 架构演进下的系统层分析

> 2026-08-05 | 基于 DeepSeek-V4（arXiv:2606.19348）与 Kimi K3（arXiv:2607.24653）技术报告
>
> **性质说明**：本文是基于两篇技术报告的推理和分析，非实测结论。文中的量化估算（如 chunk_size、$n_{hc}$、Sinkhorn-Knopp 迭代开销等）为量级示意，实际数据需在生产环境中验证。

**目录**

- [一、旧叙事的终结：为什么 KV Cache 不再是最重要的瓶颈](#一旧叙事的终结为什么-kv-cache-不再是最重要的瓶颈)
- [二、新瓶颈一：压缩-稀疏 Pipeline 的 Kernel 开销](#二新瓶颈一压缩-稀疏-pipeline-的-kernel-开销)
- [三、新瓶颈二：KDA Chunkwise 的串行约束](#三新瓶颈二kda-chunkwise-的串行约束)
- [四、新瓶颈三：MoE All-to-All 在超长上下文下的通信](#四新瓶颈三moe-all-to-all-在超长上下文下的通信)
- [五、新瓶颈四：异构 KV Cache 的层次管理](#五新瓶颈四异构-kv-cache-的层次管理)
- [六、新瓶颈五：mHC 与 AttnRes 的推理开销](#六新瓶颈五mhc-与-attnres-的推理开销)
- [七、总结：新架构下推理系统的新职责](#七总结新架构下推理系统的新职责)

---

过去三年，推理系统的核心叙事围绕一件事展开：**KV Cache 太大，怎么办？**

PagedAttention 把碎片率从 40–60% 降到 4% 以下。RadixAttention 用前缀树复用跨请求的公共前缀。GQA 把 KV 头数从 128 压到 8。MLA 更进一步，把 KV 压缩到 512 维 latent space，显存占用降至标准 MHA 的 7%。LMCache 和 Mooncake 把 KV cache 从 GPU 搬到 CPU，再搬到 NVMe，用多级存储突破了单机显存上限。

这条叙事在 2026 年走到了终点。不是因为问题被完美解决了，而是因为问题本身被消解了——DeepSeek-V4 和 Kimi K3 从架构层面对 attention 动了根本性的手术，**KV Cache 不再是瓶颈了**。

但消灭旧瓶颈的同时，新架构制造了新的瓶颈。本文从系统层的视角，分析这五种新瓶颈的本质、两家的不同应对策略，以及对推理引擎设计的启示。

---

## 一、旧叙事的终结：为什么 KV Cache 不再是最重要的瓶颈

### 1.1 两条路线，一个目标

DeepSeek-V4 和 Kimi K3 都支持 100 万 token 上下文。这在两年前是纯理论——标准 attention 在 1M 长度下的 KV Cache 大到不可运行。它们各自的解法完全不同，但都指向同一个方向：**不产生那么大的 KV Cache**。

**DeepSeek-V4：序列维度的压缩-稀疏两阶段**

V4 设计了两类注意力层，交错配置：

- **CSA（压缩稀疏注意力）**：每 m 个 token 的 KV 先被压缩成 1 个 entry（论文中 CSA 的默认压缩率 m=16）。压缩后的 entries 不再做全量 attention，而是通过 DeepSeek Sparse Attention（DSA）做 top-k 选择——每个 query 只关注 k 个最相关的压缩 entry。同时保留一个小滑动窗口的未压缩 KV，补充局部细粒度依赖。
- **HCA（重度压缩注意力）**：压缩率 m′ 设置为远大于 CSA 的 m（论文仅说明 m′ ≫ m，未给出精确值），但保持 dense attention（不做稀疏选择）。

效果是惊人的：1M 上下文下，V4-Pro 的单 token 推理 FLOPs 是 V3.2 的 27%，KV Cache 是 V3.2 的 10%。V4-Flash 更激进：FLOPs 仅 10%，KV Cache 仅 7%。相对标准 BF16 GQA8 基线，KV 体积降至约 2%。

**Kimi K3：序列维度的线性 recurrent + 深度维度的 attention**

K3 的路线完全不同：

- **KDA（Kimi Delta Attention）**：一种线性注意力机制，复杂度 O(N) 而非 O(N²)。核心是一个带通道级遗忘门的 delta-rule 循环状态——每步用当前 query/key 更新隐状态，不需要存储完整的 KV Cache。位置信息由 recurrent decay 隐式捕捉，完全不需要 RoPE（NoPE）。
- **Gated MLA**：每 4 层插入 1 层 MLA，保留全局内容交互能力，避免线性注意力的表达能力损失。
- **Attention Residuals**：这个更激进——在深度维度上做 softmax attention。每层不是简单地加残差，而是通过学到的伪 query 选择性地关注前面所有层的输出。

K3 在 1M 上下文下的 decode 加速达到全 attention 的 6.3×。

### 1.2 旧优化技术的位置

这两条新路线从根本上改变了优化空间——之前在 KV Cache 上投入的大量工程努力，在新架构下还能复用多少？

| 旧技术                    | V4 架构下的价值                                   | K3 架构下的价值                                                        |
| ------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------- |
| PagedAttention            | 仍然需要（管理压缩后的 KV entry 块）              | MLA 层仍需；KDA 层不需要（无 KV Cache）                                |
| Prefix Caching            | 重要（V4 专门设计了 on-disk shared-prefix reuse） | KDA 层需 checkpoint 循环状态 $S_t$（矩阵形式，非 token 序列 KV Cache） |
| KV Cache 量化（FP8/INT4） | V4 已内置混合精度（RoPE BF16 + 其余 FP8）         | MLA latent 量化；KDA 状态本身已紧凑                                    |
| Cross-Layer 共享          | 基本无意义（压缩后的单层 KV 已经极小）            | 同样无意义                                                             |
| Transform Coding 压缩     | 无意义                                            | 无意义                                                                 |

---

## 二、新瓶颈一：压缩-稀疏 Pipeline 的 Kernel 开销

### 2.1 问题

V4 的 CSA 不是在算一个 attention kernel——它在跑一条 pipeline：

```text
KV tokens → Token-Level Compressor (a/b 两条并行路径)
  → 压缩 KV entries + 压缩权重
  → Lightning Indexer（索引器 query × 索引器 key → 索引分数）
  → Top-k Selector（每个 query 选 k 个压缩 entry）
  → Multi-Query Attention（仅在被选的压缩 entries + sliding window entries 上计算）
```

每一步都是独立 kernel。在中等长度序列下，这些 kernel 的 launch overhead 和中间结果的 memory access 可以忽略。但 1M 上下文的 prefill 会触发这个 pipeline 的大量调用——压缩器需要处理每批 prefill chunk 的 KV token，索引器需要计算对应数量的 query × 压缩后的 key。虽然 prefill 本身是按 chunk 分批进行的，但 pipeline 的每次调用仍然面临大量细粒度 kernel 的调度开销。

V4 论文的 3.2 节专门提到了这个问题——他们引入了 TileLang DSL 来解决"大量细粒度 ATen 算子"导致 CPU 端调度开销过高的问题。通过 IR 层同时生成 device kernel 和 host launcher，将 CPU 校验开销从每次调用 "tens or hundreds of microseconds" 降到 "less than one microsecond"。这个数据反过来说明，在引入 TileLang 之前，kernel launch overhead 是一个真实存在的问题。

### 2.2 V4 的应对

V4 的推理框架做了三层优化：

**Kernel 融合**：3.1 节描述了一种"单个融合 kernel for MoE modules"，将计算、通信和内存访问完全重叠。虽然描述的是 MoE，但同样的设计思路被应用到了 CSA pipeline 中。

**稀疏注意力 kernel 协同设计**：3.5.1 节提到 Sparse Attention Kernel Co-Design——将 top-k 选择器的输出格式与底层 attention kernel 的输入需求对齐，避免显式的稀疏→稠密格式转换。

**TileLang 的 SMT 求解器辅助优化**：3.2 节将 Z3 SMT 求解器集成进 TileLang 的编译系统，用于自动推导布局、检测 bank conflict、分析 boundary condition。编译时间控制在"几秒"，不影响开发迭代。

### 2.3 对推理引擎的启示

现有的推理引擎（vLLM、SGLang）的 attention kernel 是围绕标准 attention（FlashAttention）和 MLA（FlashMLA）设计的。它们假设"一次调用，完成整个 attention 计算"。V4 的 CSA pipeline 需要的是**可组合的 attention kernel 链**——压缩 kernel → 索引 kernel → 选择 kernel → 注意力 kernel，且中间结果可以在 SRAM 中接力，避免写回 HBM。

这对 kernel 库的设计提出了新要求：不是"一个 kernel 解决一切"，而是"一套 kernel 原语，允许引擎按需编排"。

---

## 三、新瓶颈二：KDA Chunkwise 的串行约束

### 3.1 问题

KDA 的数学公式（论文 Eq. 1）看起来简洁：

$$S_t = (I - \beta_t k_t k_t^\top) \text{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top$$

但推理时不能逐 token 算——那会是 1M 次串行循环步。K3 实际采用的是**chunkwise 并行**（Eq. 4）：跨 chunk 做 recurrent，chunk 内做并行矩阵乘法。

```text
Chunk 0: [tok0 ... tok_C]  → 并行计算 → 产出 S[0]（进入 chunk 1）
Chunk 1: [tok_C+1 ... tok_2C] → 并行计算（依赖 S[0]）→ 产出 S[1]
...
Chunk N: ... → 产出最终输出
```

这就是核心 trade-off：chunk 越大，串行步数越少，但 chunk 内计算量越大（O(C²) 的因果掩码矩阵乘法）。chunk 越小，计算越轻，但串行步数越多。以 chunk_size=16K 为例，1M 上下文意味着约 63 步串行（论文未给出 K3 在生产环境中的默认 chunk size，此处仅作量级示意）——每一步都必须等待上一步的 recurrent state，且每一步都需要做一次 chunk 内 attention。

### 3.2 K3 的应对

K3 论文的 §2.1.1 描述了两个关键优化：

**Lower-bounded decay**：KDA 的前身 Kimi Linear 使用负 Softplus 映射，log-decay 无下界，导致对角 tile 需要逐位置计算。K3 改用有下界的 scaled sigmoid（$g_{\min} = -5$），使对角 tile 也能用 Tensor Core 做稠密矩阵乘法。这个改动本身不减少串行步数，但减少了每步内的计算延迟。

**FlashKDA kernel**：论文在基础设施章节（§5 "Systems Co-Design for KDA"）提到专门为 KDA 开发的 fused kernel，支持 KDA Context Parallelism（跨设备切分序列长度以分摊 chunkwise 串行开销）和 state-aware prefix caching（在 prompt 边界和每 32K token 处 checkpoint KDA 循环状态）。核心思路是将 chunk 内的因果注意力、recurrent state 更新和输出投影融合到单个 kernel 中，避免中间结果写回 HBM。

### 3.3 与标准 Prefill 的对比

标准 attention 的 prefill 是完全并行的——所有 token 同时计算，chunked prefill 把长序列切成 chunk 是为了让 decode 能插队，不是为了解决 attention 本身的并行性。

KDA 的 prefill 是**本质串行**的——chunk N 严格依赖 chunk N-1 的 recurrent state。这意味着即使给 KDA 分配更多的 GPU 做 tensor 并行，串行步数不会减少——唯一加速的方式是缩小每步的计算延迟。

这对 PD 分离架构有直接影响：在 Prefill 节点上，标准 MLA 的 prefill 可以通过大 TP 加速（compute-bound），但 KDA 的 prefill 是 latency-bound——串行步数决定了 TTFT 的下限。

---

## 四、新瓶颈三：MoE All-to-All 在超长上下文下的通信

### 4.1 为什么超长上下文加剧了 EP 通信

MoE 的 EP 通信量和 batch size × 序列长度成正比——每个 token 都需要经过 all-to-all dispatch/combine，把 hidden states 发送到负责对应专家的 GPU。在 1M 上下文的场景下，单个 request 就可能包含 1M 个 token。

V4 论文的 3.1 节提供了一个关键数据：V4-Pro 每 token-expert 对需要 6hd 次浮点运算，但只需要 3h 字节通信。这意味着通信与计算的比例是 **C/B ≤ 2d = 6144 FLOPs/Byte**。换句话说，1 GBps 的互联带宽需要对应 6.1 TFLOP/s 的 GPU 算力才能完全隐藏通信。

对于 H100（~1000 TFLOP/s FP8），满足 C/B ≤ 2d 所需的互联带宽约为 164 GBps。NVLink 4.0（900 GB/s 双向）远超此要求，通信可被计算完全隐藏。但跨节点的 25G/100G 以太网就明显不足了——batch 内 token 量被 1M 上下文推高后，跨节点 EP 通信将成为瓶颈。

### 4.2 V4 的应对：细粒度计算-通信流水线

V4 的核心创新是 **wave-level pipeline**：不等到所有专家通信完成才开始计算。把专家分成多个 wave，每个 wave 含少量专家。wave 内通信完成后立即开始计算，同时下一波 token 的传输和已完成专家的结果回传在后台进行。

效果：通用推理加速 1.50–1.73×，延迟敏感的 RL rollout 场景最高 1.96×（3.1 节）。已开源在 DeepGEMM 的 MegaMoE 组件中。

V4 论文还给硬件厂商提了一个反直觉的建议：**在满足 C/B ≤ 2d 之后，再增加互联带宽对 MoE 推理的收益递减**。更有效的是增加 GPU 本身的计算密度，让每 GBps 的带宽能喂饱更多的算力。

### 4.3 K3 的应对：MoonEP + Stable LatentMoE

K3 的 Stable LatentMoE 在架构层面就降低了通信量：routed expert 在 latent space（维度 ℓ < d）中操作，权重和激活的字节数都更少。896 个专家、top-16 激活（稀疏度 56），比 DeepSeek-V3 的 256 专家 top-8 更分散，每个 token 的通信目标更多但单次通信量更小。

MoonEP（论文 §5）从两方向优化 EP 通信：一是静态计算形状——预先确定每个 GPU 上的专家分布和 token batch 大小，消除动态路由带来的通信形状变化和同步开销；二是零拷贝通信——通过 RDMA 直接读写远端 GPU 的专家权重缓冲区，避免中间数据在 CPU 内存中的拷贝。两者叠加确保了 2.8T 参数模型在 EP 模式下的通信效率。

---

## 五、新瓶颈四：异构 KV Cache 的层次管理

### 5.1 问题：KV Cache 不再是一个东西

标准 attention 模式下，KV Cache 是同质的——每层、每个 token 存储格式完全一样（k 和 v 各一个 tensor，形状一致）。管理策略也简单：LRU 淘汰、按需加载。

V4 和 K3 各自产生了一套**异构的状态集合**：

**V4 的 KV 状态**（从论文 2.3.4 节和 3.5.1 节目录拼合）：

- 压缩 KV entries（CSA：每 m token 1 entry；HCA：每 m′ token 1 entry）
- 滑动窗口的未压缩 KV（保留局部依赖）
- 压缩索引器的 keys（用于 top-k 选择）
- KV entry 的精度也是混合的：RoPE 维度存 BF16，其余维度存 FP8
- 还有 State Cache for SWA 和解压缩的 tail tokens

**K3 的 KV 状态**（从论文 §2.1 和 §5 拼合）：

- KDA 层的 recurrent state $S_t$（一个矩阵，大小依赖 head 数和 hidden dim，不是 token 序列）
- MLA 层的 latent $c_t$（K2/K2.5 继承的压缩 KV）
- 前缀缓存的 KDA state checkpoint（每 32K token + prompt 边界存一次）

### 5.2 新的管理挑战

**层次间迁移的粒度不匹配**：V4 的 on-disk shared-prefix reuse 需要知道"哪些压缩 entry 是可复用的前缀"。但 CSA 的压缩边界是固定的（每 m 个 token），前缀匹配是语义级的——一个 system prompt 的 token 数不一定是 m 的整数倍。压缩边界和语义边界不一致时，要么截断导致缓存 miss，要么冗余存储覆盖边界 case。论文要求 key 按 page_size 对齐（类似 PagedAttention），但语义级前缀匹配在压缩 KV 上的精确边界处理策略尚无公开实现细节。

**KDA state 的 checkpoint 策略**：KDA 的 recurrent state 不像 KV Cache 那样可以逐 token 访问——它是一个累积状态。要恢复某个前缀上的 KDA 状态，必须从最近的 checkpoint 重放。checkpoint 间隔越密，恢复越快，但存储开销越大。K3 选择的 32K token 间隔是对读写成本的经验折中，但在不同的前缀复用模式下，这个间隔的最优值会变化。

**精度异构带来的编排复杂度**：V4 的 CSA 索引器注意力在 FP4 精度下计算，而压缩 KV entry 的 RoPE 维度在 BF16 下，其余在 FP8 下。一次 attention 计算涉及三种精度——kernel 需要在内部分段处理不同精度，或者提前做格式转换（增加访存）。

### 5.3 对前缀缓存实现的影响

现有的前缀缓存实现（vLLM APC、SGLang RadixAttention、HiCache）都是为同质 KV Cache 设计的：匹配到多个 token 的共同前缀 → 复用对应的 KV block。在 V4/K3 的异构状态下，前缀缓存需要管理多种不同格式、不同粒度的状态对象，且不同层的状态类型可能不同（V4 的 CSA 层和 HCA 层不产生相同类型的状态）。

这意味着前缀缓存的管理逻辑从"一个 KV Cache 池，一种淘汰策略"变成了"每个 layer group 有自己的 state pool 和自己的生命周期管理"。论文中 V4 将 State Cache for SWA 和主 KV Cache 分开管理的设计已经暗示了这个方向。

---

## 六、新瓶颈五：mHC 与 AttnRes 的推理开销

### 6.1 两个"不在 attention 上"的瓶颈

V4 的 mHC（流形约束超连接）和 K3 的 AttnRes（注意力残差）不在 attention 路径上，不在 FFN 路径上——它们在**层与层之间的连接**上。

传统 Transformer 的残差连接基本免费：一次逐元素加法，不涉及矩阵乘法、不产生额外显存占用。但 V4 和 K3 各自把这个"免费"操作升级成了需要计算的操作。

### 6.2 mHC：每层的 Sinkhorn-Knopp 迭代

V4 的 mHC（论文 §2.2）对每一层的残差连接做了如下升级：

```text
旧: h_{l+1} = h_l + F_l(h_l)                    # 逐元素加法，免费

新: X_{l+1} = B_l · X_l + C_l · F_l(A_l · X_l)  # 矩阵乘法 + 输入输出映射
    B_l 必须约束到 doubly stochastic matrices
    约束方法: Sinkhorn-Knopp 迭代（t_max = 20 步）
```

Sinkhorn-Knopp 算法：对矩阵交替做行归一化和列归一化，直到收敛为双随机矩阵。每步迭代是一个逐行/逐列的 softmax。

这里需要区分 mHC 参数的两种组成（论文 Eq. 3-5）：输入独立的**静态偏置** $S_l$ 和输入依赖的**动态分量**（由当前层输入 $X_l$ 经可学习权重生成）。静态偏置可以预计算并在推理时直接加载，减少每步计算量。但动态分量仍需每层实时计算——因为 $B_l$ 的生成依赖当前层的实际 hidden states。也就是说，$B_l$ 不能完全预计算，Sinkhorn-Knopp 迭代（$t_{\max} = 20$）在推理时无法跳过。

对于 $n_{hc}$ 通常取很小的值（论文没有给出具体数字，但说"much smaller than hidden size"），假设 $n_{hc} = 4$，则每层约 320 次 4×4 操作。计算量确实不大，但对于 61 层（V4-Pro）累积起来就值得关注了——特别是考虑到这些操作不在传统的 attention/FFN 计算路径上，可能无法被现有推理引擎的 kernel fusion 覆盖。

### 6.3 AttnRes：O(L²d) 的深度注意力

K3 的 AttnRes（论文 §2.2）更激进：

```text
对每一层 l，计算一个"深度注意力"：
  q_l = w_l（学到的伪 query）
  k_i = v_i = f_i(h_i)（前面每层的输出）
  α_{i→l} = softmax(q_l · RMSNorm(k_i))
  h_l = Σ α_{i→l} · v_i

完整形式的计算量: O(L²d)，L=96 层，K3 的 d 很大
```

K3 的 Block AttnRes 优化把这个降到了 O(N·d)，N=8 个 block。但推理时每层仍然需要计算跨 block 的 attention weights——这部分计算不在 GPU 的 attention kernel 里（那是处理序列维度的），而是在一个完全不同的维度（深度维度）上。

### 6.4 对推理引擎的影响

这部分的优化在现有推理引擎中基本是空白——因为传统架构的残差连接没有优化空间。对于 V4 的 mHC 和 K3 的 AttnRes：

- **mHC**：Sinkhorn-Knopp 迭代可以预先计算 $B_l$（如果输入独立部分可近似静态化），或者用更少的迭代次数（论文当前是 20，可以降到 5-10 步评估精度损失）
- **AttnRes**：Block 形式的跨 block attention 可以跟层的 forward 做 pipeline——当 block n 计算时，block n+1 的 attention weights 可以预取 block 0,...,n 的表示

这些都是推理引擎需要新增的调度逻辑，不属于传统 attention/FFN 的范畴。

---

## 七、总结：新架构下推理系统的新职责

### 7.1 瓶颈转移全景

| 旧瓶颈               | V4 的解法                                                             | K3 的解法                                            | 新瓶颈                                                |
| -------------------- | --------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------- |
| 标准 attention O(N²) | CSA 压缩-稀疏（序列维度压缩 + top-k 选择）                            | KDA 线性 recurrent（O(N) 复杂度）                    | 压缩 pipeline kernel 编排；chunkwise 串行步数         |
| KV Cache 显存        | 异构 KV + 混合精度（RoPE BF16 / 其余 FP8 / Indexer FP4），降至基线 2% | KDA 无需 KV Cache；MLA latent 压缩；state checkpoint | 异构状态的层次管理；prefix caching 的多粒度匹配       |
| Prefill 计算         | 压缩后 attention 量减少 + sparse 选择                                 | Chunkwise 并行 + Tensor Core 全覆盖                  | KDA prefill 的串行依赖（无法被 TP 加速）              |
| EP all-to-all 通信   | 细粒度 wave pipeline（加速 1.5–2×）                                   | Stable LatentMoE + MoonEP                            | 1M 上下文推高 batch 内 token 量，跨节点带宽仍可能不足 |
| 残差连接             | mHC（Sinkhorn-Knopp 约束，每层计算开销）                              | AttnRes（深度注意力，O(N·d)）                        | 推理时新增的计算不在传统优化范畴内                    |

### 7.2 推理引擎需要的新能力

五个新瓶颈指向推理引擎需要发展的五项新能力：

1. **可组合的 attention kernel 链**：不再假设"一个 kernel 完成 attention"，而是提供压缩、索引、选择、注意力四个原语，允许引擎按模型结构编排
2. **Chunkwise 调度感知**：对于 KDA 这类本质串行的 prefill，调度器需要理解 chunk 的依赖关系，在 chunk 间隙插入 decode step（类似 chunked prefill 的思路，但动机不同）
3. **EP 通信的上下文感知调度**：超长请求应该优先路由到节点内 GPU（利用 NVLink），而非跨节点（挤占有限的外部带宽）
4. **多粒度异构状态缓存**：前缀缓存从"同质 KV block 池"演进为"每层每种状态类型独立管理"，不同格式（BF16/FP8/FP4/recurrent state）有各自的生命周期
5. **层间连接的优化意识**：mHC 和 AttnRes 不再是免费的——推理时需要考虑矩阵映射的预计算、深度注意力的 pipeline 调度

这些不是某一个引擎独有的问题。V4 和 K3 代表了两条不同的架构路线，但它们共同指向了一个事实：**架构创新正在把复杂度从模型内部转移到模型与系统之间的接口上**。推理引擎的下一轮竞争，不在 PagedAttention vs RadixAttention——而在谁能更快地适配这些新的接口。

---

> **参考来源**
>
> - DeepSeek-AI. "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence." arXiv:2606.19348, 2026.
> - Kimi Team. "Kimi K3: Open Frontier Intelligence — Technical Report of Kimi K3." arXiv:2607.24653, 2026.
> - Xie et al. "Manifold-Constrained Hyper-Connections." 2026.（V4 引用的 mHC 原始论文）
> - Kimi Team. "Kimi Delta Attention."（K3 引用的 KDA 原始论文，arXiv:2510.26692）
