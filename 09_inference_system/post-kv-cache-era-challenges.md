# 当百万 Token KV Cache 从 250GB 降到 5GB：推理系统的新挑战

> 250GB → 标准 BF16 GQA8 配置（8 KV 头 × 128 维 × 60 层）在 1M token 时的 KV Cache 规模（此处以典型 60 层模型为例，实际取决于层数和头维度）；5GB → DeepSeek-V4-Flash 在 1M token 时同等条件下的实际 KV Cache（仅为 GQA8 基线的 ~2%）。
>
> 2026-08-07 | 基于 DeepSeek-V4（arXiv:2606.19348）与 Kimi K3（arXiv:2607.24653）技术报告，对照 vLLM（`c810e5e`）与 SGLang（`4d4f802`）源码交叉验证
>
> **性质说明**：本文是基于两篇技术报告的推理和分析，部分数据已与 vLLM/SGLang 源码对照验证（标注 ✓）。未标注 ✓ 的量化估算（如 Sinkhorn-Knopp 迭代次数、$n_{hc}$ 值等依赖模型 config.json 的参数）为量级示意，实际数据需在使用时根据模型配置确认。

---

过去三年，推理系统的核心叙事围绕一件事展开：**KV Cache 太大，怎么办？**

PagedAttention 把碎片率从 40–60% 降到 4% 以下。RadixAttention 用前缀树复用跨请求的公共前缀。GQA 把 KV 头数从 128 压到 8。MLA 更进一步，把 KV 压缩到 512 维 latent space，显存占用降至标准 MHA 的 7%。LMCache 和 Mooncake 把 KV cache 从 GPU 搬到 CPU，再搬到 NVMe，用多级存储突破了单机显存上限。

这条叙事在 2026 年走到了终点。不是因为问题被完美解决了，而是因为问题本身被消解了——DeepSeek-V4 和 Kimi K3 从架构层面对 attention 动了根本性的手术，**KV Cache 不再是首要矛盾了**。

但旧矛盾消退的同时，新架构带来了新的系统层挑战。这些挑战中，有的是架构直接制造的硬问题，有的已经被工程化解但需随规模持续关注。本文将它们分为两类：**需要解决的**（仍影响 scalability 或用户体验的硬问题）与**需要持续关注的**（已有有效方案，但需要在演进中保持重视的方向）。一共五个方向，逐一分析。

---

## 一、旧叙事的终结：为什么 KV Cache 不再是首要矛盾

### 1.1 两条路线，一个目标

DeepSeek-V4 和 Kimi K3 都支持 100 万 token 上下文。这在两年前是纯理论——标准 attention 在 1M 长度下的 KV Cache 大到不可运行。它们各自的解法完全不同，但都指向同一个方向：**不产生那么大的 KV Cache**。

#### 1.1.1 DeepSeek-V4：序列维度的压缩-稀疏两阶段

V4 设计了两类注意力层，交错配置：

- **CSA（压缩稀疏注意力）**：每 4 个 token 的 KV 被压缩成 1 个 entry（即压缩率 m=4，vLLM 代码中以 `compress_ratio=4` 对应 C4A 层 ✓）。压缩后的 entries 不再做全量 attention，而是通过 DeepSeek Sparse Attention（DSA）做 top-k 选择——每个 query 只关注 k 个最相关的压缩 entry。同时保留一个小滑动窗口的未压缩 KV，补充局部细粒度依赖。
- **HCA（重度压缩注意力）**：压缩率 m′=128（vLLM 代码中以 `compress_ratio=128` 对应 C128A 层 ✓），保持 dense attention（不做稀疏选择）。

效果是惊人的：1M 上下文下，V4-Pro 的单 token 推理 FLOPs 是 V3.2 的 27%，KV Cache 是 V3.2 的 10%。V4-Flash 更激进：FLOPs 仅 10%，KV Cache 仅 7%。相对标准 BF16 GQA8 基线，KV 体积降至约 2%。

#### 1.1.2 Kimi K3：序列维度的线性 recurrent + 深度维度的 attention

K3 的路线完全不同：

- **KDA（Kimi Delta Attention）**：一种线性注意力机制，复杂度 O(N) 而非 O(N²)。核心是一个带通道级遗忘门的 delta-rule 循环状态——每步用当前 query/key 更新隐状态，不需要存储完整的 KV Cache。位置信息由 recurrent decay 隐式捕捉，完全不需要 RoPE（NoPE）。
- **Gated MLA**：在 KDA 层之间穿插 MLA 层（具体分配由 `linear_attn_config.kda_layers` 和 `full_attn_layers` 显式指定 ✓，典型配置下 MLA 占比约 1/4），保留全局内容交互能力，避免线性注意力的表达能力损失。
- **Attention Residuals**：这个更激进——在深度维度上做 softmax attention。每层不是简单地加残差，而是通过学到的伪 query 选择性地关注前面所有层的输出。

K3 在 1M 上下文下的 decode 加速达到全 attention 的 6.3×。

### 1.2 旧优化技术的位置

这两条新路线从根本上改变了优化空间——之前在 KV Cache 上投入的大量工程努力，在新架构下还能复用多少？

| 旧技术                    | V4 架构下的价值                                   | K3 架构下的价值                                                                                    |
| ------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| PagedAttention            | 仍然需要（管理压缩后的 KV entry 块）              | MLA 层仍需；KDA 层不需要（无 KV Cache）                                                            |
| Prefix Caching            | 重要（V4 专门设计了 on-disk shared-prefix reuse） | KDA 层需 checkpoint 循环状态 $S_t$（矩阵形式，非 token 序列 KV Cache，间隔由 `block_size` 决定 ✓） |
| KV Cache 量化（FP8/INT4） | V4 已内置混合精度（RoPE BF16 + 其余 FP8）         | MLA latent 量化；KDA 状态本身已紧凑                                                                |
| Cross-Layer 共享          | 基本无意义（压缩后的单层 KV 已经极小）            | 同样无意义                                                                                         |
| Transform Coding 压缩     | 无意义                                            | 无意义                                                                                             |

---

## 二、需要持续关注：CSA 多流 Kernel 编排

### 2.1 问题

> **分类理由**：CSA 替代了单次 attention kernel 调用，引入了多流编排的复杂性。但 TileLang ✓ 已解决了细粒度 kernel 的 CPU launch 开销问题，vLLM/SGLang 的 3 路 CUDA stream 实现 ✓ 也已成熟。当前这更多是"已被解决的工程挑战，但其解法值得被抽象为通用框架"——归为需要持续关注。

V4 的 CSA 不是在算一个 attention kernel——它需要多个计算步骤协同完成：

```text
主 Stream:   fused_wqa_wkv → wq_b + kv_insert
辅 Stream 1: compressor_kv_score → 完整 Compressor
辅 Stream 2: indexer_weights_proj + indexer_compressor_kv_score → 完整 Indexer
```

vLLM 实际实现 ✓ 通过 3 路 CUDA stream 并行执行这组操作：主 stream 负责 Q/KV 投影和写入，两个辅助 stream 分别跑 Compressor 和 Indexer。各 stream 之间通过 CUDA event 同步——需要同步点，因此对 kernel launch overhead 仍然敏感。

V4 论文的 3.2 节专门提到了这个问题——他们引入了 TileLang DSL 来解决"大量细粒度 ATen 算子"导致 CPU 端调度开销过高的问题。通过 IR 层同时生成 device kernel 和 host launcher，将 CPU 校验开销从每次调用 "tens or hundreds of microseconds" 降到 "less than one microsecond"。vLLM 和 SGLang 的 V4 实现中，TileLang 被用于 mHC 层和 DSA top-k 等关键 kernel 的生成 ✓。这个数据反过来说明，在引入 TileLang 之前，kernel launch overhead 是一个真实存在的问题。

### 2.2 V4 的应对

V4 的推理框架做了三层优化：

**Kernel 融合**：3.1 节描述了一种"单个融合 kernel for MoE modules"，将计算、通信和内存访问完全重叠。虽然描述的是 MoE，但同样的设计思路被应用到了 CSA pipeline 中。

**稀疏注意力 kernel 协同设计**：3.5.1 节提到 Sparse Attention Kernel Co-Design——将 top-k 选择器的输出格式与底层 attention kernel 的输入需求对齐，避免显式的稀疏→稠密格式转换。

**TileLang 的 SMT 求解器辅助优化**：3.2 节将 Z3 SMT 求解器集成进 TileLang 的编译系统，用于自动推导布局、检测 bank conflict、分析 boundary condition。编译时间控制在"几秒"，不影响开发迭代。

### 2.3 对推理引擎的启示

传统推理引擎的 attention kernel 围绕"一次调用，完成整个 attention 计算"设计。V4 的 CSA 打破了这一假设——需要**可编排的多流执行**：Q/KV 投影 + Compressor + Indexer 三路并行，各流之间通过 event 同步。vLLM/SGLang 已通过多 CUDA stream + TileLang ✓ 实现了这一模式。这标志着 attention kernel 的设计范式转换——从"一个 kernel 解决一切"到"一套可组合的 kernel 原语，按模型结构编排"——这一范式已确立，但值得作为通用设计原则被更多引擎采纳。

---

## 三、需要解决：KDA Chunkwise 串行约束

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

这就是核心 trade-off：chunk 越大，串行步数越少，但 chunk 内计算量越大（O(C²) 的因果掩码矩阵乘法）。chunk 越小，计算越轻，但串行步数越多。vLLM 和 SGLang 的 KDA 实现均使用 `FLA_CHUNK_SIZE=64` ✓，1M 上下文意味着约 **15,625 步**串行——每一步都必须等待上一步的 recurrent state，且每一步都需要做一次 chunk 内 attention。这对 prefill 延迟的影响远大于此前估算的 "63 步"。

### 3.2 K3 的应对

K3 论文的 §2.1.1 描述了两个关键优化：

**Lower-bounded decay**：KDA 的前身 Kimi Linear 使用负 Softplus 映射，log-decay 无下界，导致对角 tile 需要逐位置计算。K3 改用有下界的 scaled sigmoid，vLLM 代码中硬编码下界为 `_KDA_GATE_LOGBOUND_MIN = -5.0` ✓，实际使用的 `gate_lower_bound` 由模型 config 指定（取值在 [-5.0, 0) 之间）。这使得对角 tile 也能用 Tensor Core 做稠密矩阵乘法。这个改动本身不减少串行步数，但减少了每步内的计算延迟。

**FlashKDA kernel**：论文在基础设施章节（§5 "Systems Co-Design for KDA"）提到专门为 KDA 开发的 fused kernel，支持 KDA Context Parallelism（跨设备切分序列长度以分摊 chunkwise 串行开销）和 state-aware prefix caching（在 prompt 边界和每 block 结尾处 checkpoint KDA 循环状态）。vLLM 中通过 `mamba_cache_mode` 控制 checkpoint 策略 ✓（支持 `"all"`/`"none"`/selective 三种模式，间隔由 `kv_cache_spec.block_size` 决定）。核心思路是将 chunk 内的因果注意力、recurrent state 更新和输出投影融合到单个 kernel 中，避免中间结果写回 HBM。

### 3.3 与标准 Prefill 的对比

标准 attention 的 prefill 是完全并行的——所有 token 同时计算，chunked prefill 把长序列切成 chunk 是为了让 decode 能插队，不是为了解决 attention 本身的并行性。

KDA 的 prefill 是**本质串行**的——chunk N 严格依赖 chunk N-1 的 recurrent state。这意味着即使给 KDA 分配更多的 GPU 做 tensor 并行，串行步数不会减少——唯一加速的方式是缩小每步的计算延迟。

这对 PD 分离架构有直接影响：在 Prefill 节点上，标准 MLA 的 prefill 可以通过大 TP 加速（compute-bound），但 KDA 的 prefill 是 latency-bound——串行步数决定了 TTFT 的下限。

---

## 四、需要持续关注：MoE All-to-All 在超长上下文下的通信压力

### 4.1 为什么超长上下文加剧了 EP 通信

> **分类理由**：MoE EP all-to-all 是 V2/V3 时代就存在的旧瓶颈，并非 V4/K3 架构制造的新问题。1M 上下文使其在量级上加剧，但两家的应对（wave pipeline、LatentMoE ✓）是缓解方案而非新问题的补救措施——归为需要持续关注。

MoE 的 EP 通信量和 batch size × 序列长度成正比——每个 token 都需要经过 all-to-all dispatch/combine，把 hidden states 发送到负责对应专家的 GPU。在 1M 上下文的场景下，单个 request 就可能包含 1M 个 token。

V4 论文的 3.1 节提供了一个关键数据：V4-Pro 每 token-expert 对需要 6hd 次浮点运算，但只需要 3h 字节通信。这意味着通信与计算的比例是 **C/B ≤ 2d = 6144 FLOPs/Byte**。换句话说，1 GBps 的互联带宽需要对应 6.1 TFLOP/s 的 GPU 算力才能完全隐藏通信。

对于 H100（~1000 TFLOP/s FP8），满足 C/B ≤ 2d 所需的互联带宽约为 164 GBps。NVLink 4.0（900 GB/s 双向）远超此要求，通信可被计算完全隐藏。但跨节点的 25G/100G 以太网就明显不足了——batch 内 token 量被 1M 上下文推高后，跨节点 EP 通信将成为瓶颈。

### 4.2 V4 的应对：细粒度计算-通信流水线

V4 的核心创新是 **wave-level pipeline**：不等到所有专家通信完成才开始计算。把专家分成多个 wave，每个 wave 含少量专家。wave 内通信完成后立即开始计算，同时下一波 token 的传输和已完成专家的结果回传在后台进行。

效果：通用推理加速 1.50–1.73×，延迟敏感的 RL rollout 场景最高 1.96×（3.1 节）。已开源在 DeepGEMM 的 MegaMoE 组件中（该机制实现在编译库中，vLLM/SGLang 通过 `prepare_megamoe_inputs` Python 接口调用，无法从 Python 源码直接验证 wave scheduling 细节）。

V4 论文还给硬件厂商提了一个反直觉的建议：**在满足 C/B ≤ 2d 之后，再增加互联带宽对 MoE 推理的收益递减**。更有效的是增加 GPU 本身的计算密度，让每 GBps 的带宽能喂饱更多的算力。

### 4.3 K3 的应对：MoonEP + Stable LatentMoE

K3 的 Stable LatentMoE 在架构层面就降低了通信量：routed expert 在 latent space（维度 ℓ < d）中操作 ✓（vLLM 中由 `config.routed_expert_hidden_size` 控制，`LatentMoERunner` 实现），权重和激活的字节数都更少。896 个专家、top-16 激活（稀疏度 56），比 DeepSeek-V3 的 256 专家 top-8 更分散，每个 token 的通信目标更多但单次通信量更小。此外，K3 的 MoE 计算直接复用了 V4 的 `DeepseekV4MegaMoEExperts` 组件 ✓。

MoonEP（论文 §5）从两方向优化 EP 通信：一是静态计算形状——预先确定每个 GPU 上的专家分布和 token batch 大小，消除动态路由带来的通信形状变化和同步开销；二是零拷贝通信——通过 RDMA 直接读写远端 GPU 的专家权重缓冲区，避免中间数据在 CPU 内存中的拷贝。两者叠加确保了 2.8T 参数模型在 EP 模式下的通信效率。

---

## 五、需要解决：异构状态下的前缀缓存一致性

### 5.1 问题：一套前缀缓存要同时服务多种 KV 类型

标准 attention 模式下，前缀缓存的工作模型很简单——所有层的 KV 格式相同，匹配到 N 个 token 的公共前缀，就复用了 N 个 token 的 KV block。PagedAttention、RadixAttention、各种 prefix caching 方案都建立在这个前提上。

V4 和 K3 打破了这个前提。同一个请求中，不同类型的状态需要被不同的缓存层管理，而前缀缓存必须同时服务所有这些类型。

具体来说，V4 的一个请求在前缀缓存视角下，至少涉及三种各不相同的状态：

- **C4A 层**（compress_ratio=4）：每 4 个 token 压缩为 1 个 entry，block size=4 ✓，按 584B/token 的 fp8_ds_mla 格式存储 ✓
- **C128A 层**（compress_ratio=128）：每 128 个 token 压缩为 1 个 entry，block size=8 ✓，同样按 584B/token 存储
- **Indexer**：独立的 K cache，可选 MXFP4 精度 ✓，有自己独立的 pool 和 block 大小

K3 的状态更异构——KDA 层根本没有 token 序列 KV，只有 `(num_heads/tp, head_dim, head_dim)` 形状的 recurrent state 矩阵 ✓，加上 short convolution state；MLA 层则是传统的 latent KV。前缀缓存必须同时管理这些完全不同形态的对象。

### 5.2 存储层已解决，前缀缓存层是缺口

两个推理引擎在存储层已经建成了通用方案。

vLLM 通过 `MLAAttentionSpec` 的参数化（`compress_ratio`/`model_version`/`cache_dtype`）适配任意 MLA 变体 ✓，`MambaSpec` 管理 SSM 类 state ✓，`KVCacheSpecRegistry` 支持外部插件注册自定义 spec ✓。SGLang 的 `DeepSeekV4TokenToKVPool` 通过 `compression_ratios: List[int]` 参数化 ✓，`K_PER_BLOCK = {4: 32, 128: 1}` 动态计算物理布局 ✓。存储层的抽象已经建立——如果明天出现 compress_ratio=8 的新架构，配置修改即可，不需要新代码。

**但前缀缓存层没有跟上。** 前缀缓存的核心逻辑是：匹配到多少个 token 的公共前缀，就跳过多少 token 的 attention 计算。当不同层的 KV 存储粒度不同时，"多少个 token" 这个前提本身出了问题。vLLM 代码明确承认了这一限制 ✓：`find_longest_cache_hit` "only supports one attention type or two types of full-attention plus exactly one another type"——通用多类型前缀缓存还没有实现。

这意味着当前在生产中部署 V4 时，前缀缓存的收益被打了折扣。一个 system prompt 的前缀命中需要同时满足 C4A 的 4-token 粒度、C128A 的 128-token 粒度、SWA 的 token 粒度——而当前实现只覆盖其中 1–2 种。

### 5.3 配置的自适应也是一个开放问题

除了前缀缓存这个硬缺口，异构状态还带来了一个次生问题：最优配置依赖数据分布，而当前没有自动调优。

KDA state checkpoint 的最优间隔取决于前缀复用率——复用率高的场景希望密集 checkpoint（恢复更快），复用率低的场景希望稀疏 checkpoint（节省存储）。V4 的 Indexer 是否使用 MXFP4 ✓ 也需要在精度损失和存储节省之间权衡。这些参数目前都需要人工根据业务场景调整，缺乏根据实际请求分布自动选择最优策略的机制。

---

## 六、需要持续关注：mHC 与 AttnRes 的推理开销

### 6.1 两个"不在 attention 上"的新增计算

> **分类理由**：mHC 的 Sinkhorn-Knopp 迭代已通过 TileLang/CUDA/AITER 多后端 fused kernel ✓ 高效处理，计算量本身有限；AttnRes 以 Triton kernel ✓ 实现且默认不启用（`config.attn_res_block_size=None` ✓）。两者均不构成当前的核心性能问题，但它们标志着"层间连接不再免费"这一设计趋势——归为需要持续关注。

V4 的 mHC（流形约束超连接）和 K3 的 AttnRes（注意力残差）不在 attention 路径上，不在 FFN 路径上——它们在**层与层之间的连接**上。

传统 Transformer 的残差连接基本免费：一次逐元素加法，不涉及矩阵乘法、不产生额外显存占用。但 V4 和 K3 各自把这个"免费"操作升级成了需要计算的操作。

### 6.2 mHC：每层的 Sinkhorn-Knopp 迭代

V4 的 mHC（论文 §2.2）对每一层的残差连接做了如下升级。vLLM 代码 ✓ 揭示的实际机制比论文公式更复杂——mHC 操作 `hc_mult` 个残差流，每个流对应一组独立的 hidden states：

```text
旧: h_{l+1} = h_l + F_l(h_l)                             # 逐元素加法，免费

新: 1. 将 hc_mult 个残差流拼成 [num_tokens, hc_mult*hidden_size]
    2. 经 fn 矩阵投影 → pre_mix (sigmoid gate)、post_mix (sigmoid×multiplier)、comb_mix (双随机矩阵)
    3. layer_input = sum(pre_mix[:, i] × residual[:, i, :])  # 加权的残差流之和
    4. layer_output = post_mix × output + comb_mix × residual  # post + combined residual
    comb_mix 通过 Sinkhorn-Knopp 迭代约束为双随机矩阵（迭代次数 = config.hc_sinkhorn_iters ✓）
```

Sinkhorn-Knopp 算法：对矩阵交替做行归一化和列归一化，直到收敛为双随机矩阵。每步迭代是一个逐行/逐列的 softmax。

这里需要区分 mHC 参数的两种组成（论文 Eq. 3-5）：输入独立的**静态偏置**（`hc_base`, `hc_scale`，可预计算）和输入依赖的**动态分量**（`comb_mix` 由当前层 hidden states 经可学习权重 `hc_attn_fn`/`hc_ffn_fn` 生成，vLLM 中 ✓ 由 `MHCPreOp`/`MHCPostOp`/`MHCFusedPostPreOp` 三个 CustomOp 实现）。静态偏置可以预计算并在推理时直接加载。但动态分量仍需每层实时计算——`comb_mix` 的生成依赖当前层的实际 hidden states。也就是说，$B_l$ 不能完全预计算，Sinkhorn-Knopp 迭代在推理时无法跳过。

迭代次数 `config.hc_sinkhorn_iters` 的具体值需从 V4 模型 config 读取（并非固定 20）。`hc_mult`（残差流数量）同样来自模型 config——论文称 "much smaller than hidden size"，vLLM 中通过 `config.hc_mult` 指定 ✓，值较小（如 4）时每层约数百次小矩阵操作，但迭代 61 层（V4-Pro）累积起来就值得关注——特别是考虑到这些操作不在传统的 attention/FFN 计算路径上，可能无法被现有推理引擎的 kernel fusion 覆盖。

### 6.3 AttnRes：O(L²d) 的深度注意力

K3 的 AttnRes（论文 §2.2）更激进。vLLM 中 ✓ AttnRes 由 Triton kernel 实现（`kimi_k3/nvidia/ops/attn_res.py`）：

```text
对每个 block（多个连续层为一组），维护一个 hidden states 快照：
  层 l 的输出转换为"query"（学到的权重）
  前面各 block 的快照转换为"key/value"
  通过 softmax attention 聚合跨 block 信息

完整形式的计算量: O(L²d)，L=96 层，K3 的 d 很大
```

Block 大小由 `config.attn_res_block_size` 指定 ✓（默认 `None` 即未启用）。若 96 层按 block_size=12 分组，则为 N=8 个 block，计算量降至 O(N·d)。推理时每层需要计算跨 block 的 attention weights——这部分计算不在 GPU 的序列维度 attention kernel 里，而是在一个完全不同的维度（深度维度）上。

### 6.4 对推理引擎的影响

vLLM 和 SGLang 均已为 mHC 实现了高效的 fused kernel ✓（`mhc_pre`/`mhc_post`/`mhc_fused_post_pre`，含 TileLang/CUDA/AITER 多种后端），将 pre-norm、RMS-norm、Sinkhorn 迭代和残差混合融合为单个 kernel。但优化空间仍然存在：

- **mHC**：Sinkhorn-Knopp 迭代次数取决于 `config.hc_sinkhorn_iters`，实际值可能小于 20；若配置允许，可尝试降低迭代次数评估精度损失，用 TileLang 自动生成不同迭代次数的 kernel 变体
- **AttnRes**：Block 形式的跨 block attention 可以跟层的 forward 做 pipeline——当 block n 计算时，block n+1 的 attention weights 可以预取 block 0,...,n 的表示

这些都是推理引擎已通过 fused kernel 接入的调度逻辑 ✓，不属于传统 attention/FFN 的范畴，但在模型规模和序列长度持续增长的背景下需要持续关注。

---

## 七、总结

下表展示了五个方向的完整图景。"需要解决"意味着架构不改变就无法消除，当前仍有硬缺口；"需要持续关注"意味着引擎已有成熟方案，但需随规模演进保持重视。

| 旧矛盾               | V4 的解法                                                                | K3 的解法                                            | 新挑战                                                               | 分类                            |
| -------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------- |
| 标准 attention O(N²) | C4A 压缩-稀疏（m=4 ✓）+ C128A 重度压缩（m′=128 ✓）                       | KDA 线性 recurrent（O(N)，chunk=64 ✓）               | 多流 kernel 编排；chunkwise 串行步数（1M 需 15,625 步）              | 多流编排已解决 / 串行步数需解决 |
| KV Cache 显存        | 异构 KV + 混合精度（584B/token ✓，RoPE BF16 / NoPE FP8 / Indexer MXFP4） | KDA 无需 KV Cache（state 为矩阵 ✓）+ MLA latent 压缩 | 存储抽象已通用 ✓，但跨类型 prefix caching 未解决；最优配置需人工调优 | 需要解决                        |
| Prefill 计算         | 压缩后 attention 量减少 + sparse 选择                                    | Chunkwise 并行 + Tensor Core 全覆盖                  | KDA prefill 的串行依赖（无法被 TP 加速）                             | 需要解决                        |
| EP all-to-all 通信   | 细粒度 wave pipeline（DeepGEMM .so）+ C/B ≤ 2d 硬件建议                  | LatentMoE ✓ + MoonEP，复用 V4 MegaMoE ✓              | 旧矛盾在 1M 上下文下加剧；V4/K3 提供了缓解方案而非制造新问题         | 需要持续关注                    |
| 残差连接             | mHC（multi-stream + Sinkhorn-Knopp ✓，fused kernel 已实现）              | AttnRes（Triton kernel ✓，默认不启用）               | 层间连接不再免费——当前开销已被 fused kernel 覆盖，但趋势值得持续关注 | 需要持续关注                    |

### 7.1 对推理引擎的启示

**需要解决：**

1. **Chunkwise 调度感知**：对于 KDA 这类本质串行的 prefill（chunk=64，1M 需 15,625 步），调度器需要理解 chunk 的依赖关系，在 chunk 间隙插入 decode step。TP 无法减少串行步数——这是当前最"硬"的新挑战。
2. **跨类型前缀缓存**：KV cache 存储抽象已是通用的 ✓（vLLM `MLAAttentionSpec` 参数化 ✓ + SGLang `compression_ratios: List[int]` ✓），但前缀缓存层是真正的缺口——`find_longest_cache_hit` 只支持 1–2 种 attention 类型 ✓，无法处理 V4（C4A + C128A + SWA + Indexer）或 K3（KDA + MLA）的多类型混合。

**需要持续关注：**

1. **可编排的多流 attention kernel**：vLLM/SGLang 已用 3 路 CUDA stream 实现 ✓，TileLang 已解决 kernel launch 开销 ✓。这更多是一个"已被解决的工程挑战，但其解法值得被抽象为通用范式"。
2. **EP 通信的上下文感知调度**：V2/V3 时代就存在的旧问题，1M 上下文使之加剧。V4 的 wave pipeline 和 K3 的 LatentMoE ✓ 已提供有效缓解，但极端 batch 下的跨节点带宽仍需持续关注。
3. **层间连接的优化意识**：mHC fused kernel ✓ + AttnRes Triton kernel ✓ 已覆盖当前需求。长期价值在于认识到一个趋势：**架构创新正在把复杂度从模型内部转移到模型与系统之间的接口上**。

---

> **参考来源**
>
> - DeepSeek-AI. "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence." arXiv:2606.19348, 2026.
> - Kimi Team. "Kimi K3: Open Frontier Intelligence — Technical Report of Kimi K3." arXiv:2607.24653, 2026.
> - Xie et al. "Manifold-Constrained Hyper-Connections." 2026.（V4 引用的 mHC 原始论文）
> - Kimi Team. "Kimi Delta Attention."（K3 引用的 KDA 原始论文，arXiv:2510.26692）
