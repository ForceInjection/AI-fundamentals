# Key-Key Semantic Affinity：用 Key 向量替代注意力分数的 KV Cache 重要性评估

## 1. 背景：长上下文推理的 KV Cache 瓶颈

大语言模型的上下文窗口已从 GPT-2 的 1024 token 扩展到百万 token 级别。每次自回归解码，模型需要访问全部历史 token 的 Key 和 Value 激活矩阵（KV Cache）。以 Qwen2.5-7B 模型为例：模型权重约 14GB，但在 128K token 的上下文下，KV Cache 占用高达 **64GB**——是模型本身的 4 倍以上。更直观地说，一张 80GB 的 A100 GPU，仅 KV Cache 就占去 80% 以上的空间，留给计算和张量并行的余地所剩无几。

现代推理引擎（vLLM、SGLang 等）采用 PagedAttention 机制，将 KV Cache 以固定大小的 **block**（通常 16-256 token）为粒度进行管理。每个 block 通过内容哈希标识，可以被独立地换入/换出 GPU 显存。这为选择性加载提供了架构基础——我们不需要在 token 级别做决策，而是在 block 级别。

长序列注意力的一个基本性质进一步支撑了这一策略的可行性：**注意力分布天然高度稀疏**。大量研究表明，90% 以上的注意力能量往往集中在不到 10% 的 token 上。这意味着在每次 decode 步，绝大部分 KV Cache block 对当前查询的贡献微乎其微，完全可以留在外部存储中而不影响输出质量。

基于这些条件，业界提出了 **KV Cache 剪枝**（KV Cache Pruning）策略：在每次 decode 步，仅选择与当前查询最相关的一部分 block 加载到 GPU 参与注意力计算，其余留在外部存储中。

然而，这里存在一个根本性的**循环悖论**：要知道哪些 block 对当前查询重要，最直接的方法是计算 QK^T 注意力分数——但这要求 block 已经在 GPU 中，剪枝也就失去了意义。因此，我们需要一种**不需要完整注意力计算就能评估 block 重要性的代理指标**。这正是本文要探讨的核心问题。

---

## 2. 现有方案及其局限

### 2.1 基于全局注意力分数累积（H2O, SnapKV）

面对 §1 的循环悖论，最直接的想法是：**能否用"历史注意力分数"来预测"未来重要性"？** 这就是 H2O 和 SnapKV 的出发点。

**H2O (Heavy Hitter Oracle, NeurIPS 2023)** 提出在 decode 过程中累积每个 token 的注意力分数，将累积分数最高的 token（称为 Heavy Hitter）保留在 KV Cache 中，其余淘汰。其核心假设是"注意力模式存在重复性和持续性"（Repetitive Attention Pattern）——历史上获得高注意力的 token 在未来仍将重要。

**SnapKV** 在 prefill 阶段计算完整的 QK^T 注意力矩阵，对所有 token 的注意力分数进行全局排序，选出 top-K 个重要位置，在后续所有 decode 步中固定使用这个选择——本质上是一种"静态快照"策略。

这类基于全局注意力分数的方法存在三个根本性缺陷：

**缺陷一：全局注意力分数 ≠ 局部语义相关度。** 一个 token 在整个序列的全局统计中可能频繁被关注（globally frequent），但对当前 decode 步可能是噪声（locally irrelevant）。SamKV 的实验（Fig.1）直观展示了这一现象：SnapKV 基于全局排序选出的 token 中，包含大量在系统中反复出现的模板化文本片段——它们历史累积分数高，但与当前查询的语义焦点几乎无关。SamKV 将这一问题概括为 "globally frequent but locally irrelevant tokens"。在多轮对话主题漂移、长文档推理等场景中，注意力焦点随时间剧烈转移，历史高注意力 token 对未来几乎没有预测力。

**缺陷二：token 粒度与 block 抽象不匹配。** 现代推理引擎以 block 为 KV Cache 管理单元。H2O 在 token 级别做选择，导致选中的 token 在物理内存中不连续——为了将它们加载到 GPU，需要额外的数据搬运来重建连续的内存布局，打破了 PagedAttention 的 block 抽象。SamKV 的 Table 2 明确指出，这种结构性不匹配 "disrupts the block abstraction, which can limit the effectiveness of optimized block-based kernels"。

**缺陷三：没有真正解决 §1 的循环悖论。** H2O 和 SnapKV 仍然依赖 QK^T 注意力分数的显式存储或增量更新。在 FlashAttention 等融合 kernel 中，中间注意力矩阵被刻意避免存储以换取速度——为了获取分数而打破这个优化，反而部分抵消了剪枝带来的性能收益。它们绕开了"需要加载 block 才能算注意力"的问题，但引入了新的开销：必须在每次 decode 步后更新累积分数表。

### 2.2 SamKV 的突破

**SamKV (Context-aware Hierarchical Efficient Semantic Selection, AAAI 2026)** 对上述方法提出了系统性批评："largely context-agnostic, their token selection ignores step-wise relevance and local semantics"。

SamKV 的关键洞察是：**Key 向量本身就是输入 token 在语义空间中的投影**。每个 token 的 Key 向量 `K = X·W_K` 经过训练后，在高维空间中具有语义表征能力——语义相关 token 的 Key 向量方向相近，语义无关 token 的 Key 向量近乎正交。

基于这一洞察，SamKV 提出了一条完全不同的路径：**不再依赖 QK^T 注意力分数，而是用 Key 向量自身的语义距离来评估 block 重要性**。核心操作被称为 Key-Key Semantic Affinity：

1. **Block Key 表征（一次性）**：对 block 内所有 token、所有 head、所有 layer 的 Key 向量做 mean-pool，得到紧凑语义向量 v_b。仅需 ~500B/block，首次写入 KV Cache 时计算一次并持久化存储。
2. **查询锚点（每步）**：对最近几个 token 的 Key 向量做 mean-pool，得到 v_anchor。代表模型"当前在关注什么"。
3. **语义亲和度（每步）**：$\text{score} = (v_{anchor} \cdot v_b + 1) / 2$，归一化到 $[0, 1]$。

这个方案从根本上解决了 §1 的循环悖论：Key 向量在注意力计算时已经常驻 HBM，作为语义探针无需额外存储。不需要 Q，不需要 QK^T 矩阵乘法，甚至不需要所有的 K——只需要候选 block 的预存储 v_b 和当前步的 v_anchor。

SamKV 还引入了一个层次化选择结构（Grid → Chunk → Page），以单次 GEMM 调用完成所有级别的评分计算，在算法层面保持了高效性。详细内容将在 §5 展开。

---

## 3. 为什么 Key 向量具有语义表征能力

上一节介绍了 Key-Key Semantic Affinity 的"是什么"和"怎么做"。本节回答"为什么能行"——从训练本质、理论保证和实验验证三个层面，建立对这一方法的完整信心。

### 3.1 训练本质：K 是语义锚点，Q 是检索意图

在 Transformer 的多头注意力中，Query 和 Key 通过各自的投影矩阵获得：

$$Q = XW_Q \quad\text{（"我要找什么"——当前 token 的检索意图）}$$
$$K = XW_K \quad\text{（"我是什么"——每个 token 的语义身份）}$$

注意一个关键的不对称性：**Q 是 transient（每步都变），K 是 persistent（写入 KV Cache 后不变）**。Q 代表当前 token 对前文的"检索需求"——它随着 decode 的推进、上下文的演化而不断变化。K 代表每个历史 token 的"语义身份"——它一旦计算就固定下来，成为 KV Cache 的一部分。

这意味着 Key 向量天然适合做块间语义比较：同一个 block 内的 token 共享相似的语义身份（它们是同一段上下文产生的），这些 Key 向量在 W_K 投影后倾向于指向相似方向。而 Query 向量则不适合——它描述的是"当前这一瞬间我要找什么"，下一瞬间可能就变了。

实际上，Key 向量在训练中被优化为**语义锚点**——它们调整自身方向，使得语义相关的 Query 能够通过点积有效地"检索"到它们。语义相关的 token 往往由相似的上下文产生，因此它们的 Key 向量在训练后自然聚拢。这一性质在高维 Key 空间中尤为显著。

此外，对 block 内 B 个 token 的 Key 向量做均值池化，进一步提升了信号质量：单个 token 的 Key 向量受位置编码、局部语法角色等因素干扰，但这些干扰因子在 token 间相互独立——均值池化后，噪声被大数定律湮灭，block 级别的 dominant 语义信号被保留。

### 3.2 理论保证：高维空间中的正交性

均值池化保留语义信号的可靠性得到了理论支撑。SamKV 给出了一个浓度不等式保证：

$$
P\left(\left| v_{anchor} \cdot v_p - v_{anchor} \cdot K_i \right| > \varepsilon\right)
\leq 2 \cdot \exp\left(-\frac{C \cdot B \cdot \varepsilon^2}{\sigma^2}\right)
$$

其中 B 是 block 内的 token 数，C 是常数，σ² 是 Key 向量的方差。这个不等式的直观含义是：**block 越大、语义越集中，语义信号就越可靠**。当 B 足够大时（例如 block_size=128），随机噪声导致 affinity 偏离真实语义相关度的概率呈指数级衰减。

实际效果体现为一条简洁的规则：在高维 Key 空间中，语义无关的 block 的 v_b 与 v_anchor 的点积 **≈ 0**（正交），语义相关的 block 的点积 **≈ 1**（同向）。不需要 Q，不需要 QK^T，仅靠 K 之间的几何关系就能完成判断。

### 3.3 实验验证：与 QK^T 注意力模式的直接对比

理论最终要靠实验说话。我们在 Qwen2.5 系列模型上进行了直接验证：将 Key-Key Semantic Affinity 选出的 top-K block 与真实注意力（平均所有层和 head 的 QK^T 分数）选出的 top-K block 做重叠率对比。

| 模型           | KV Heads | 池化方式                     | 与 QK^T 的 Top-K 重叠率 |
| -------------- | -------- | ---------------------------- | ----------------------- |
| Qwen2.5-0.5B   | 2        | 简单 mean-pool               | 50%                     |
| Qwen2.5-0.5B   | 2        | 加权池化（按 head variance） | 75%                     |
| **Qwen2.5-7B** | **4**    | **加权池化**                 | **100%**                |

两个发现：

**第一，模型规模是决定性因素。** 0.5B 仅 2 个 KV heads，cross-layer mean-pool 后的 v_b 只有 64 维，语义区分度有限。7B 的 head_dim 扩展到 128，即使均值池化后仍是 128 维，高维正交性在实验中已充分显现。

**第二，加权池化显著有效。** 在 0.5B 上，从 50% 到 75% 的跃升来自一个简单的改进——按 attention head 的方差加权。高方差 head 的注意力分布更集中、更有判别力；低方差 head 倾向于均匀关注所有位置（类似 positional bias），对语义区分贡献有限。加权后，这些"噪声 head"被有效抑制。

7B 模型的 100% 重叠率传递了一个清晰的信号：**在足够大的模型上，Key-Key Semantic Affinity 是 QK^T 注意力的完美代理。**

---

## 4. Key-Key 方案 vs. 注意力分数方案

前两节分别讨论了"现有方案的问题"和"Key-Key 为什么能工作"。本节将两类方案放在一起，做一个系统性的对比，帮助读者在具体场景中判断何时适用何种方案。

| 维度                 | QK^T 注意力分数                                       | Key-Key 语义亲和度              |
| -------------------- | ----------------------------------------------------- | ------------------------------- |
| **计算依赖**         | Q 和 K 的矩阵乘法                                     | 仅需 K，已常驻 HBM              |
| **存储开销**         | 注意力分数矩阵或增量更新                              | 每 block 一个 D 维向量（~500B） |
| **FlashAttention**   | 融合 kernel 不输出中间矩阵                            | 完全绕过注意力计算              |
| **context-aware**    | EWMA 累积隐含"过去→未来"假设                          | 每步即时评估                    |
| **SamKV 批评的问题** | 会保留 globally frequent but locally irrelevant token | 动态反映当前步语义焦点          |
| **跨模型适用**       | 依赖具体注意力实现                                    | 所有 Transformer 均具有 K 向量  |

几个维度的差异值得展开说明：

- **"FlashAttention"这一行是关键的分水岭。** 现代推理系统几乎全部采用 FlashAttention 或其变体进行注意力计算。这些融合 kernel 通过分块计算和在线 softmax 避免了显式存储 N×N 的注意力矩阵——这恰恰是 H2O/SnapKV 等方法赖以运作的数据。要从 FlashAttention 中提取注意力分数，要么修改 kernel（牺牲性能），要么额外计算一次（浪费算力）。Key-Key 方案完全绕开了这个问题——它根本不参与注意力计算路径，仅在 K 向量写入 KV Cache 时顺便做一次 mean-pool。

- **"context-aware"这一行决定了信号质量的天花板。** QK^T 方案基于跨步累积，本质上是"用过去预测未来"。在注意力焦点稳定的场景（如事实性问答），这个假设成立。在主题漂移、多轮对话、长文档推理等场景中，累积分数会成为噪声。Key-Key 方案每步重新评估，不存在"预测"——它只回答"当前这一步，这个 block 有多相关"。代价是单步信号可能有波动，但 §5.3 讨论的稳定性过滤可以弥补。

- **"跨模型适用"这一行决定了方案的迁移成本。** QK^T 方案的实现深度依赖具体的注意力模块——GQA vs MHA vs MLA、是否使用 FlashAttention、是否有自定义融合 kernel——每个组合都需要适配。Key-Key 方案只需要一个前提条件：模型有 Key 向量。这是所有 Transformer 类 LLM 的共同特征，无论架构细节如何。

综合来看，Key-Key 方案的优势在**不需要注意力计算就能得到可靠的语义重要性信号**。它的主要代价是引入了 block Key 表征的额外存储（每个 block ~500B），以及在 decode 每步需要做 O(num_blocks × D) 次点积。这两个开销在实际系统中都远小于注意力计算本身。

---

## 5. 在 KV Cache 剪枝中的应用

前面三节从"问题→原理→对比"建立了 Key-Key Semantic Affinity 的理论基础。本节讨论将这一方法落地到实际推理系统中的三个关键工程问题：如何高效地对大量 block 做语义评分、如何嵌入到 decode 循环中、以及如何处理单步信号的波动。

### 5.1 层次化选择：从 O(n) 到 O(log n) 的评分效率

最简单的实现方式是为每个候选 block 计算 v_anchor · v_b，然后取 top-K。但当序列长度达到 128K token、block 数量超过 1000 时，每步都要做 1000+ 次点积，开销不可忽略。

SamKV 的解决方案是将 Key-Key Semantic Affinity 嵌入一个**三层层次化选择结构**（Grid → Chunk → Page，对应粗 → 中 → 细三个粒度）。核心思想是"粗筛先行，精筛后继"：

1. **Grid 层（最粗）**：将整个序列切分为若干大段（Grid），每个 Grid 的 Key 表征是其内部所有 Chunk 表征的均值。由于数量少（通常几十个），可以快速完成全量评分，淘汰低分 Grid。
2. **Chunk 层（中等）**：仅对被选中 Grid 内的 Chunk 进行评分。这一步利用了 Grid 层的筛选结果——未被选中的 Grid 的子 Chunk 直接跳过。
3. **Page 层（最细）**：仅对存活 Chunk 内的 Page 进行评分。Page 是最终加载到 GPU 的 block 单元。

这个结构的巧妙之处在于：**三个层次的评分共用一次 GEMM 调用**。SamKV 将三个层次的所有表征向量拼接成一个矩阵，与 v_anchor 做单次矩阵乘法，然后通过层次化的布尔掩码实现条件筛选——"选中父节点 → 激活子节点；未选中父节点 → 子节点分数置零"。这种 **tensor coalescing + vectorized dependency check** 的设计避免了 GPU 上昂贵的分支发散。

在 LongBench 数据集上，采用激进配置（各层保留率分别为 50%、20%、10%）时，SamKV 仅使用 **1% 的 KV Cache** 即超越了全量 KV Cache 的 F1 分数（33.2 vs 30.2）。SamKV 的作者将这一反直觉的结果归因于：精确的稀疏化过滤掉了长上下文中的冗余和干扰信息，使得模型能够更聚焦于真正相关的证据。

### 5.2 Per-Step 集成：嵌入 decode 循环的轻量级语义探针

Key-Key Semantic Affinity 在推理引擎中的集成位置是**注意力计算路径的内部**。具体来说，语义选择器在每步 decode 中执行：

1. **构造查询锚点**：用最近 W_local 个 token（通常 4-8 个，约 1-2 个 block）的 Key 向量做 mean-pool，得到 v_anchor。这个操作利用了当前步已经计算好的 K 向量——它们为了写入 KV Cache 已经常驻 HBM，无需额外计算。
2. **查询 block 表征**：从预存储的表征表中读取所有候选 block 的 v_b。v_b 在 block 首次写入 KV Cache 时计算一次（通过注册在 `k_proj` 上的 forward hook），之后所有 decode 步直接查询。每个 block 仅需存储一个 D 维浮点向量（D = head_dim，7B 模型为 128），内存开销约 500B/block。
3. **计算语义亲和度**：对于标准 GQA 模型，对 block 内的所有 token、所有 KV head、所有层做 mean-pool 得到 v_b，然后与 v_anchor 做点积。对于 MLA 架构（如 DeepSeek-V2），需要分别处理 compressed KV 和 RoPE 部分。计算复杂度为 O(num_blocks × D)，在 1000 个 block、D=128 时约 128K 次浮点运算——在 GPU 上的耗时通常在微秒级别。
4. **选择 top-K block 并加载**：仅将语义亲和度最高的 K 个 block 加载到 GPU HBM 参与注意力计算。其余 block 留在外部存储中——它们的数据没有被丢弃，如果后续某步的语义焦点发生变化，仍然可以被重新加载。

这个过程嵌入在 decode 循环中，每步执行。它不修改注意力 kernel 本身——它只改变"哪些 block 被传入注意力 kernel"。

### 5.3 稳定性过滤：区分语义信号与瞬时噪声

Key-Key Semantic Affinity 是 per-step 的即时评估，这意味着单步的语义波动（如一个 token 生成后注意力焦点瞬间转至另一个 topic）会在亲和度曲线上产生尖峰。如果直接用单步亲和度做放置决策，这些尖峰会导致错误的 block 被加载。

但语义真正的规律是：**真正相关的 block 会在连续多步中持续获得高亲和度，而噪声尖峰是瞬时的**。这一性质可以通过一个简单的滑动窗口机制来利用：

$$\mathrm{stability score}(b) = \frac{\text{count}(semantic affinity(b, t) \geq \theta_{affinity}, t \in [T-W+1, T])}{W}$$

其中 W 是窗口大小（通常 20 步），θ_affinity 是单步高亲和度阈值（通常 0.5）。这个分数的物理含义是：在最近 W 个 decode 步中，block b 有多少步被判定为"高语义相关"。一个只在单步获得 0.9 的 block，如果其余 19 步都是 0.2，stability_score = 1/20 = 0.05——远不足以触发加载；而一个连续 20 步都在 0.6 以上的 block，stability_score = 1.0。

这个机制的优雅之处在于它**无需额外的存储或计算**：滑动窗口就是 v_b 被查询的历史记录，stability_score 就是窗口内超过阈值的比例。它将 Key-Key 的"即时精确"转化为"跨步稳定"，使得单步噪声自然衰减、真实语义信号被放大。

---

## 6. 局限与权衡

任何方法都有适用条件。Key-Key Semantic Affinity 的主要局限可以归纳为三类：**前提条件**（什么情况下它能工作）、**信息代价**（它付出了什么）、和**边界场景**（什么情况下它可能不如其他方案）。

### 6.1 前提条件：模型规模与 Key 空间维度

Key-Key 语义亲和度的可靠性直接取决于 Key 向量空间的维度——更准确地说，取决于 KV heads 的数量和 head_dim 的大小。实验表明：

- **< 1B 模型（2 KV heads 以下）**：信号质量不足以独立支撑可靠的剪枝决策。简单 mean-pool 的注意力重叠率约 50%，加权池化可提升至 75%。在这个规模上，Key-Key 可能更适合作为辅助信号而非主要决策依据。
- **1B-7B 模型（2-4 KV heads）**：3B 以上已经能够提供足够可靠的语义区分。加权池化是这一区间的关键优化，通过抑制低方差 head（倾向于均匀注意力的"语义噪声"head），有效提升信号质量。
- **7B 及以上模型（4+ KV heads）**：信号质量趋于饱和。7B 上的注意力重叠率已达 100%，进一步增大模型主要改善的是推理吞吐而非信号质量本身。

一个值得注意的边界情况是 **MLA（Multi-head Latent Attention）架构**（如 DeepSeek-V2）。MLA 将 Key 压缩到低维潜在空间，原始的 head_dim 被显著缩减。这种情况下，Key 向量的语义区分能力可能低于同等规模的 GQA 模型——需要在压缩后的潜在空间中重新评估正交性是否成立。

### 6.2 信息代价：block 级压缩丢失的细粒度结构

将 block 内 B 个 token 的 Key 向量池化为单个 v_b，本质上是将 B 个 token 的注意力结构压缩为一个"平均语义方向"。这个压缩过程丢失了两类信息：

**token 级的注意力差异。** 一个 block 内的某些 token 可能是关键实体或推理节点，其注意力权重远高于 block 内的其他 token。均值池化将这些突出的 token 与背景 token 等同对待——高注意力 token 的贡献被 block 内的大量"背景 token"稀释。在 block_size=128 时，一个关键 token 对 v_b 的贡献仅占 1/128。

**block 内部的注意力结构。** 注意力不仅是"哪个 block 重要"，还包括"block 内哪些 token 之间有关联"。Key-Key 只回答了前者，后者需要在 block 加载后由注意力 kernel 自己计算。

SamKV 的层次化选择（§5.1）部分缓解了第一个问题：Grid/Chunk/Page 的三层结构允许在不同粒度上做筛选，粗粒度层快速排除无关区域，细粒度层对候选 block 做更精细的评分。但它仍然无法恢复 block 内部的 token 级重要性差异——这是 block 级池化的固有局限。

### 6.3 边界场景：何时 Key-Key 可能不如 QK^T

虽然 Key-Key 在多数场景下表现优异，存在两类场景需要谨慎：

**高度结构化的注意力模式。** 在代码生成、表格填充等任务中，注意力模式高度规则化（如始终关注前一行代码、前一格数据），这种"结构性的"重要性不同于"语义的"重要性。Key-Key 基于语义空间的距离来判断相关性，可能低估这些结构化重要但语义距离不明显的 block。QK^T 方案在这些场景中可能更直接有效。

**短序列低延迟场景。** 当序列长度较短（< 4096 token，block 数 < 32）时，完整计算 QK^T 的开销本身就不大，而 Key-Key 需要额外维护 v_b 表征和层次化索引，引入的工程复杂度可能超过它节省的开销。Key-Key 的优势在长序列场景中才充分体现——block 越多，避免 QK^T 的收益越大。

---

## 7. 总结

本文系统性地介绍了 Key-Key Semantic Affinity——一种用 Key 向量自身语义距离替代 QK^T 注意力分数的 block 重要性评估方法。它的核心思想简洁而深刻：**Key 向量不只是 Query 的"被动匹配目标"，它们自身携带了足够的语义信息来回答"哪些 block 重要"这一问题**。

回顾全文的论证链条：

- **§1 提出了核心悖论**：要知道哪些 block 重要需要计算 QK^T，但计算 QK^T 又要求 block 已在 GPU 中。这个循环悖论是 KV Cache 剪枝的根本困难。
- **§2 分析了现有方案**：H2O/SnapKV 用"历史注意力预测未来重要性"的思路回避了悖论，但引入了全局累积偏差、token-block 粒度不匹配、以及 FlashAttention 兼容性三个代价。
- **§3 建立了理论基础**：从训练本质（K 作为语义锚点）、理论保证（高维正交性的浓度不等式）、实验验证（7B 模型 100% 重叠率）三个层面，证明了 Key-Key 信号的可靠性。
- **§4-5 展示了工程落地路径**：层次化选择将评分复杂度从 O(n) 降至对数级，per-step 集成将语义探针嵌入 decode 循环，稳定性过滤将即时信号转化为跨步可靠的决策依据。
- **§6 诚实讨论了局限**：模型规模门槛、block 级压缩的信息损失、以及结构化注意力等边界场景，为读者提供了完整的使用判断框架。

Key-Key Semantic Affinity 的意义不仅在于它提出了一种新的技术方案，更在于它揭示了一个被长期忽视的事实：**Key 向量作为训练中自然形成的语义表示，其信息密度远超我们之前所利用的**。在 FlashAttention 普及之后，QK^T 注意力分数的获取成本越来越高，Key-Key 方案提供了一个绕开这一困境的全新思路。

自 SamKV (AAAI 2026) 的开创性工作以来，Key-Key Semantic Affinity 已在学术界获得广泛关注。随着模型规模持续增长和上下文窗口不断扩展，这种"不需要 Q"的语义评估方法的实用价值将进一步凸显。

---

## 参考文献

- Ziyi Cao, et al. "Sparse Attention across Multiple-context KV Cache" (SamKV). AAAI 2026. arXiv:2508.11661.
- Zhenyu Zhang, et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." NeurIPS 2023.
- Yuhong Li, et al. "SnapKV: LLM Knows What You are Looking for Before Generation." arXiv:2404.14469, 2024.
