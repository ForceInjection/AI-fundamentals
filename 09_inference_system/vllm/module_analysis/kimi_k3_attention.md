# Kimi K3 注意力机制深度解析：KDA、Gated MLA 与 AttnRes 的混合架构

> 2026-07-28 | 基于 Kimi Linear 论文（arXiv:2510.26692）、Attention Residuals 论文（arXiv:2603.15031）、vLLM 官方博客、SGLang/LMSYS Day-0 支持文章及 Kimi K3 官方技术博客。

Kimi K3 是 Moonshot AI 于 2026 年 7 月发布的 2.8 万亿参数开源 MoE 模型。（[交互可视化](./assets/kimi_k3_attention_visual.html)），原生支持 1M token 上下文窗口。它的注意力系统不是单一种类的升级，而是 **三种机制的协同设计**：69 层 KDA（Kimi Delta Attention）负责高效长上下文处理，24 层 Gated MLA 提供全局精确检索锚点，Attention Residuals（AttnRes）则将"注意力"从序列维度扩展到模型深度维度。三者共同构成了一个与标准 Transformer 注意力体系有根本性差异的架构。

---

## 一、为什么标准注意力在 1M 上下文面前失效

标准 softmax 注意力在每个 token 位置计算完整的 $QK^\top$ 矩阵。这个矩阵的大小是 $N \times N$——$N$ 是序列长度。当 $N=1{,}000{,}000$ 时，仅注意力分数矩阵就占 ~2TB（FP16，$10^{12}$ 个条目 × 2 字节），根本无法装入任何现有 GPU 的 HBM。

已有的应对手段各有代价：

| 方案                 | 原理                                                        | 代价                        |
| -------------------- | ----------------------------------------------------------- | --------------------------- |
| 滑动窗口             | 每个 token 只看附近 $W$ 个 token                            | 丢失全局信息                |
| KV cache 压缩（MLA） | 将 K、V 压缩为低秩潜变量                                    | 仅减少存储，不减少计算      |
| 稀疏注意力           | 只计算部分 token 对的注意力分数                             | 需要手工设计稀疏模式        |
| **线性注意力**       | 将 $QK^\top V$ 重排为 $Q(K^\top V)$，用递推状态替代矩阵乘法 | 表达能力弱于 softmax 注意力 |

线性注意力将复杂度从 $O(N^2)$ 降到 $O(N)$，代价是 **"稀释"** ——固定大小的递推状态在每一步被新 token 覆盖时，旧信息不可避免地衰减。这就是 KDA 要解决的核心问题。

---

## 二、KDA：用 Delta 规则让线性注意力"不遗忘"

### 2.1 从线性注意力到 Delta 规则

标准线性注意力的递推公式：

$$S_t = S_{t-1} + k_t v_t^\top$$

$$o_t = S_t^\top q_t$$

$S_t \in \mathbb{R}^{d_k \times d_v}$ 是一个固定大小的矩阵状态，每步累加当前 token 的 key-value 外积。问题在于：累加没有"减法"——一旦旧信息写入 $S_t$，就无法被修正或删除。长序列下，早期重要信息被后续 token 的持续写入逐渐淹没。

KDA 的递推公式引入了 **delta 修正项** 和 **逐通道门控**：

$$S_t = (I - \beta_t k_t k_t^\top) \cdot \mathrm{Diag}(\alpha_t) \cdot S_{t-1} + \beta_t k_t v_t^\top$$

$$o_t = S_t^\top q_t$$

其中：

- $\alpha_t \in (0,1)^{d_k}$ 是**逐通道遗忘门**（channel-wise forget gate）：每个特征维度有独立的遗忘速率，不是整头统一遗忘
- $\beta_t \in [0,1]$ 是**学习率标量**：控制当前 token 对状态的写入强度
- $(I - \beta_t k_t k_t^\top)$ 是 **delta 修正**：一个广义 Householder 变换，在写入新信息前先"删除"状态中与当前 key 方向重叠的旧成分。

### 2.2 DPLR 状态更新：三步无需物化完整矩阵

$(I - \beta_t k_t k_t^\top)$ 看起来需要构建 $d_k \times d_k$ 的矩阵，但 KDA 使用 DPLR（Diagonal-Plus-Low-Rank）分解，将状态更新拆为三步：

```text
S'    = Diag(α_t) · S_{t-1}          ← 逐元素广播（Diagonal 退化）
S''   = S' - β_t · k_t (k_t^\top S')  ← Rank-1 修正（两次 einsum）
S_t   = S'' + β_t · k_t v_t^\top      ← KV 写入（外积）
```

这三步全部通过向量-矩阵运算实现，避免了物化任何 $d_k \times d_k$ 矩阵。DPLR 的关键优化在于：将 diagonal 和 low-rank 变量绑定到同一个 $k_t$ 上，使得第二级 chunk 矩阵计算从通用 DPLR 的四个减少到两个——算子效率提升约 100%。

### 2.3 神经网络参数化

对每个 head $h$，输入通过以下路径投影到 KDA 所需的各分量：

| 分量       | 计算                                       | 维度          | 作用                                     |
| ---------- | ------------------------------------------ | ------------- | ---------------------------------------- |
| $q_t, k_t$ | L2Norm(Swish(ShortConv($W_{q/k} x_t$)))    | $d_k$         | L2 归一化 + 4-token 深度可分离卷积预处理 |
| $v_t$      | Swish(ShortConv($W_v x_t$))                | $d_v$         | 值向量                                   |
| $\alpha_t$ | $f(W_\alpha^\top W_\alpha^\downarrow x_t)$ | $(0,1)^{d_k}$ | 逐通道门控，通过低秩瓶颈降低参数量       |
| $\beta_t$  | Sigmoid($W_\beta x_t$)                     | $[0,1]$       | 标量学习率                               |

Key 在 L2 归一化之前通过 **depthwise causal convolution（kernel size=4）** ——这给每个 key 注入了 4 个 token 的局部上下文。per-head RMSNorm 应用于检索结果后再过输出门，保证数值稳定性。

### 2.4 与标准 KV cache 的本质差异

KDA 的状态是**定长递推缓冲区**，每步原地覆写，不是追加。这带来了根本性的差异：

| 维度           | 标准 MLA KV cache    | KDA 递推状态                                                                  |
| -------------- | -------------------- | ----------------------------------------------------------------------------- |
| 增长模式       | 追加（append-only）  | 原地覆写（in-place overwrite）                                                |
| 每步存储       | ~27 KB（全部 24 层） | ~1 KB 原始输入 + ~54 MB 固定状态（全部 69 层, TP=8）                          |
| 状态量级       | 随序列长度线性增长   | 固定大小——单层状态约等于 MLA 几千 token 的 cache，但不增长                    |
| prefix caching | 天然支持             | 默认关闭；需要显式 `--enable-prefix-caching` + checkpoint + sparse radix tree |
| 投机解码       | 直接 fork 状态       | 需要 ReplaySSM 重放                                                           |

---

## 三、Gated MLA：全局检索的"锚点"

24 层 Gated MLA 与 69 层 KDA 以 **3:1 的比例交替排列**——每四个注意力层中，三个是 KDA，一个是 Gated MLA。

Gated MLA 在标准 MLA（Multi-head Latent Attention）基础上增加了 **输出门控**。这与 LSTM/GRU 的门控机制类似：门控信号根据当前上下文决定哪些信息通过、哪些被抑制。在 Kimi K3 中，KDA 的固定大小递推状态在长序列中难免丢失早期细节——但最近一次 Gated MLA 层保留了从序列起点到该位置的**完整、无损**的 KV cache。MLA 的追加（append-only）特性使得它的 KV 可以精确恢复任意历史位置的注意力计算，不受 KDA 递推状态信息衰减的影响。

**这是一个分工设计**：KDA 负责效率（$O(N)$ 复杂度处理 1M token），Gated MLA 负责精度（在关键层提供 $O(N^2)$ 的完整 softmax 注意力作为"校准信号"）。两者不是替代关系，是互补关系。

---

## 四、Attention Residuals：把注意力旋转 90 度

### 4.1 问题：残差连接在深层模型中的退化

自 ResNet（2015）以来，几乎所有深度网络都使用标准残差连接：

$$h_l = h_{l-1} + f_{l-1}(h_{l-1})$$

每个子层的输出以固定权重 1 累加到残差流。当 Kimi K3 有 93 层时，两个结构性退化出现了：

1. **信息稀释**：第 1 层的信号经过 92 次权重为 1 的加法后，其对第 93 层残差流的相对贡献衰减到几乎可以忽略——被后续 92 个层输出"冲淡"了
2. **隐状态爆炸**：为了在持续膨胀的残差流中保持信号强度，深层必须产生越来越大幅值的激活——破坏数值稳定性，造成不均匀的梯度分布

### 4.2 核心洞察："把注意力旋转 90 度"

AttnRes 的核心思想由论文作者杜宇伦提出：**模型的深度维度与序列维度在结构上是同构的**。在序列建模中，Transformer 用注意力替代了 RNN，让每个位置可以**选择性**地访问任意前置位置。AttnRes 将同样的逻辑应用到深度上：

$$\text{标准残差：} \quad h_l = \sum_{i < l} v_i \quad \text{（每个前置子层输出权重均为 1）}$$

$$\text{AttnRes：} \quad h_l = \sum_{i < l} \alpha_{i \to l} \cdot v_i \quad \text{（权重由可学习的深度注意力决定）}$$

其中 $\alpha_{i \to l} = \text{softmax}(w_l^\top \cdot \phi(v_i))$——$w_l$ 是目标层的**伪查询向量**，$\phi(v_i)$ 是前置子层输出的归一化表示（作为"键"），$v_i$ 本身作为"值"。伪查询向量初始化全零，使训练开始时退化为标准均匀加权，模型逐渐学会选择性路由。

### 4.3 两种变体

| 变体          | 粒度                                                        | 内存                      | 适用场景         |
| ------------- | ----------------------------------------------------------- | ------------------------- | ---------------- |
| Full AttnRes  | 对每个子层输出独立计算注意力                                | $O(L \times d)$ per token | 研究与消融实验   |
| Block AttnRes | 子层分组为 N 个 block（~8），块内标准残差求和，块间 AttnRes | $O(N \times d)$ per token | Kimi K3 实际使用 |

### 4.4 收益

- **参数开销 <0.03%**：每层仅增加一个 RMSNorm 和一个伪查询向量
- **训练加速 ~25%**：Block AttnRes 达到相同 loss 需要的计算量约为基线的 0.8×
- **推理开销 <2%**：在测试负载上几乎无感知。vLLM 使用 fused Triton/CUDA kernel 将 logits 计算、softmax 和隐状态聚合融为单次操作

在 48B Kimi Linear 模型的实验中（1.4T 训练 token），Block AttnRes 在下游 benchmark 上全面超越基线：GPQA-Diamond +7.5 分、Minerva Math +3.6 分、HumanEval +3.1 分。

---

## 五、三种机制如何协同：一个 Transformer Block 的完整数据流

Kimi K3 的单个 transformer block 按以下顺序处理：

```text
Embedding → Router → Linear → Norm → Linear
  → (Shared Expert + Routed Expert，各含 Conv/L2 层)
    → KDA（或 Gated MLA，取决于 block 类型）
      → Norm → Output
```

三种机制的分工：

| 机制          | 操作轴   | 解决的问题                              |
| ------------- | -------- | --------------------------------------- |
| **KDA**       | 序列长度 | $O(N)$ 线性复杂度处理 1M token          |
| **Gated MLA** | 序列长度 | 每 4 层提供一次 $O(N^2)$ 完整注意力锚点 |
| **AttnRes**   | 模型深度 | 替换固定残差连接为可学习的深度注意力    |

**序列注意力（KDA/Gated MLA）和深度注意力（AttnRes）是正交的**——前者决定"当前 token 关注序列中哪些前置 token"，后者决定"当前层从前面哪些层的表示中汲取信息"。两者同时存在，使得信息流动在两个维度上都是可学习的、自适应的。

---

## 六、服务端的工程挑战：管理两种根本不同的状态

### 6.1 双状态管理

KDA 和 Gated MLA 的 KV 状态是两种完全不同的数据结构，推理引擎必须同时管理：

- **KDA 递推状态**：定长，原地覆写，TP 下按 head 切分
- **MLA KV cache**：追加式，天然支持 prefix caching，TP 下按 token 位置切分

SGLang/LMSYS 的 Day-0 适配方案使用 **统一内存池**：KDA 状态从一端分配，MLA KV blocks 从另一端分配，中间是单一的空闲区域——不再需要为两种状态预分配独立池。

KDA 的 prefix caching 采用两种保留策略：**interval-based**——每隔可配置的 token 数（如 32K）创建一次 checkpoint，prompt 结束位置始终保留；**Marconi-style selective**（MLSys '26）——"第二次命中才缓存"，第一次观察到 prefix 存在，第二次确认它确实被共享。vLLM 对 Kimi K3 **默认关闭 prefix caching**（需显式传 `--enable-prefix-caching`），因为混合缓存设计仍在持续演进中。

### 6.2 ReplaySSM：投机解码中的 KDA 状态管理

投机解码时，draft model 每生成一个候选 token，KDA 状态就要更新一次。如果每次更新都保存完整快照（每层每头 64 KB），draft window 的内存消耗不可接受。

ReplaySSM 的方案：**不保存 KDA 快照，而是保存原始输入** $(v_i, k_i, gk_i, \beta_i)$——每步约 1 KB。当 sampler 确定接受的 token 数后，一个 **fused fold kernel** 仅重放被接受的 prefix，从 checkpoint 出发、使用 verify kernel 中存储的门控值逐步推进——产生与递推基线**逐比特相同**的结果。draft window 内存从 512 KB 降至 16 KB，约 32× 的削减。

DSpark 的 draft model（`Inferact/Kimi-K3-DSpark`）是 **MLA-native** 的——draft 结构与 Kimi K3 的注意力布局一致，保证 KV 兼容。单次并行生成 `num_speculative_tokens=7` 个候选 token。在 GB300 NVL72 上（TP16），DSpark 实现 **3.14× 加速**（118 → 370 tok/s）。平均接受 token 数随任务熵变化：coding 等低熵场景约 **4.73 tokens/step**，creative writing 等高熵场景约 **2.61 tokens/step**。

### 6.3 FlashKDA Kernel 与 Decode 优化

**KDA decode kernel** 将 causal conv → recurrent update → gated RMSNorm 三个步骤融合为单个 CUDA kernel launch。在不支持的配置上有 Triton fallback。**KDA prefill** 使用 Moonshot AI 开源的 FlashKDA（基于 CUTLASS），并经过进一步优化（Flash-Flash-KDA），改善 H100/GB300 上的数据搬运效率。

KDA 的 chunkwise 并行算法使用两个核心技术：

- **WY representation**：将一连串 rank-1 更新打包为紧凑表示，减少 I/O
- **UT transform**：减少非矩阵乘 FLOPs，提高 Tensor Core 利用率

在 H20 GPU 上，FlashKDA 相比 flash-linear-attention 基线的 prefill 加速为 **1.72–2.22×**。在 GB300 NVL72 上，TP16 配置的 batch-1 decode 达到 **118 tok/s**（无投机），启用 DSpark 后达到 **370 tok/s**（3.14× 加速）。

vLLM 还为 KDA 构建了专用 **metadata builder**：裁剪未使用的 FLA 元数据路径，将 eager PyTorch 序列替换为 fused Triton kernel。在 batch=1 时 **96% 的延迟削减**（870 μs → 34 μs），端到端 DSpark 场景带来约 6% 的改善。

---

## 七、总结

Kimi K3 的注意力设计有两个层面的创新：

**序列维度**：KDA 用 $O(N)$ 线性注意力 + delta 修正替代 $O(N^2)$ softmax 注意力，Gated MLA 每 4 层提供一次完整注意力锚点防止精度退化。两者以 3:1 交替排列，一个负责效率、一个负责精度。

**深度维度**：AttnRes 用可学习的深度注意力替代固定残差连接，让每层可以选择性地从前置层的表示中汲取信息——"注意力不再只是序列上的操作，也是深度上的操作"。

这三个机制的交叉产生了推理引擎层面前所未有的复杂性：两种状态类型（递推 vs. 追加）、两种切分策略（按 head vs. 按 token）、两种 prefix caching 模式（checkpoint + sparse radix tree vs. 标准 radix tree）。vLLM 0.27.0+ 提供了专用的 `vllm/vllm-openai:kimi-k3` Docker 镜像，SGLang 也在发布当天完成了 Day-0 适配——这种混合架构已经从"研究新奇"进入了"工程可行"的阶段。部署的最低硬件门槛是 8× GB300，生产流量需要多节点。

---

## 延伸阅读

- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) — KDA 与 Kimi Linear 48B 的完整技术报告
- [Attention Residuals](https://arxiv.org/abs/2603.15031) — AttnRes 论文
- [SGLang and Miles Add Day-0 Support for Kimi K3](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support) — SGLang 端适配的工程细节
- [vLLM × Kimi K3: Inference at 3-Trillion-Parameter Scale](https://vllm.ai/blog/2026-07-27-k3) — vLLM 端适配：DSpark 3.14× 加速、prefix caching 策略、FlashKDA kernel 融合、metadata builder 96% 延迟削减
- [Kimi K3 官方技术博客](https://www.kimi.com/it/blog/kimi-k3)
- [Learn Linear Attention From Kimi K3's KDA Mechanism in 20 Lines of Python](https://dev.to/magickong/learn-linear-attention-from-kimi-k3s-kda-mechanism-in-20-lines-of-python-cop)
- [Kimi K3 HuggingFace](https://huggingface.co/moonshotai/Kimi-K3)
- [vLLM Recipe: Kimi K3](https://recipes.vllm.ai/moonshotai/Kimi-K3) — 部署配置：TP8/TEP16、all-to-all 后端选择、FP8 KV cache、硬件要求
