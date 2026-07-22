# SGLang HiSparse：当稀疏注意力从算法变为系统组件

> SGLang 的 HiSparse 与 vLLM 的 Sparse MLA 后端实现了同一套 DSA 稀疏选择技术，但架构选择截然不同：两者都在 attention 之外用 Indexer 做 token 级 top-k 选择，但 vLLM 直接将 top-k 索引传入 attention kernel 内部消费，SGLang 在 Indexer 和 attention 之间插入了一个 coordinator 层做 page 级翻译和选择性加载。这不是孰优孰劣的差异——而是稀疏注意力从「attention 的加速技巧」被拉入「存储系统的一部分」时，自然发生的架构分裂。

---

## 一、同一个 DSA，两种实现

DSA（Dynamic Sparse Attention）做的是同一件事：用 Indexer 扫描全量 KV，选出 top-k 个最重要的 token 位置，只对这些位置做精确注意力。关于 DSA 的完整机制和它在稀疏注意力分类学中的位置，参见 [稀疏注意力分类学](../../kv_cache/01_concepts/basic/sparse_attention_taxonomy.md) §3.4。本节聚焦于一个不同的问题：同样的 DSA，在推理引擎中可以被放在不同的系统层级实现——这个选择改变了它的性质。

### 1.1 vLLM：在 attention backend 内部搞定

vLLM 将 DSA 封装在 attention backend 内部。`FLASHMLA_SPARSE` 后端接收全量 KV 和 Indexer 产出的 top-k 索引，在 attention kernel 内部完成选择——kernel 只看 top-k 指定的 token 位置。上游的调度器和 KV cache manager 完全不知道 attention 做了稀疏选择——它们看到的是「一次标准的 attention 计算」。

这套方案的优势是架构零侵入。`FlashMLASparseBackend` 与标准的 `FlashMLABackend` 是平级的独立后端——稀疏版本不继承自密集版本，而是通过构造函数注入 `topk_indices_buffer` 和 `indexer` 来接收稀疏配置。batch 的稀疏性检测（`index_topk` 配置）发生在模型层，backend 只负责消费已经选好的 top-k 索引。调度、内存管理、page 分配——都不需要感知稀疏性的存在。

代价是：稀疏性止步于 attention 层。全量 KV 仍在显存中，attention kernel 在计算时逐 token 跳过未选中的。top-k 选择的结果——每步 decode 中哪些 token 被选中、哪些 token 被忽略——在 attention 完成后直接丢弃，系统层无法复用。

### 1.2 SGLang：在 Indexer 和 attention 之间插入一个 coordinator

SGLang 的 HiSparse 在 Indexer 和 attention backend 之间插入了一个系统级组件——`hisparse_coordinator`。Indexer 产出的 token 级 top-k 索引不直接传给 attention kernel，而是先经过 coordinator：coordinator 将 token 级索引翻译为 page 级加载列表，只加载包含这些 token 的 KV 页。attention backend 拿到的不是全量 KV + top-k 索引，而是一个已经被 coordinator 过滤过的稀疏 page table。attention 只管算，不需要知道「为什么是这些页」。

这个架构变化改变了稀疏性的作用范围。vLLM 的稀疏选择只影响 attention kernel 内的执行路径——它省的是计算，不省存储。HiSparse 的稀疏选择发生在 KV cache 的加载路径上——它省的是数据搬运：未被选中的 page 被排除在 attention 计算的数据路径之外，从 page table 层面减少了需要传递给 attention kernel 的数据量。

### 1.3 分歧的根源：token 级选择 vs page 级管理

为什么两个框架在同一个问题上做出了不同的选择？根源在粒度。

DSA 的 Indexer 产出的是 **token 级**的 top-k 索引——token A 在 page 3、token B 在 page 7、token C 也在 page 3。vLLM 的处理是：全量 KV 在显存中，kernel 内逐 token 跳过未选中的——不需要关心 token 到 page 的映射。HiSparse 的处理是：先按 top-k 索引找到对应的 **page 集合**，只加载这些 page——需要将 token 级索引翻译为 page 级加载指令。

这个翻译开销是 HiSparse 方案的核心成本。当 page 内选中率高时（top-k 覆盖的 page 数接近总 page 数），翻译开销存在但收益不大——大部分 page 本来就要加载。当 page 内选中率低时（sparsity 真正起作用时），翻译开销被大量未加载 page 的带宽节省所覆盖。HiSparse 的三个设计决策——接下来逐一展开——分别解决这个翻译路径上的不同问题：page 级加载机制本身、规划 kernel 的性能优化、以及 extend/decode 双阶段的模式适配。

---

## 二、三个设计决策

HiSparse 的架构围绕三个独立但关联的设计决策展开。每个决策对应系统管道中的一个环节。

### 2.1 swap_in_selected_pages：page 级选择性加载

HiSparse coordinator 的核心操作：在 decode 阶段，根据 Indexer 产出的每层 top-k 索引，仅加载包含这些 token 的 KV 页，而非全量 KV。操作通过 `swap_in_selected_pages` 实现，接收四个参数——请求池索引列表（`req_pool_indices`）、每请求的序列长度（`seq_lens`）、当前层的 top-k 索引（`topk_indices`）、以及层 ID（`layer.layer_id`）——为每层独立计算需要加载的 page 集合。

`swap_in_selected_pages` 在 `forward_decode` 中被逐层调用。coordinator 根据 `topk_indices` 计算需要加载的 page 集合，返回一个稀疏的 page table 直接传给 decode attention kernel。未被选中的 page 不参与 attention 计算的数据搬运。

这与 vLLM 方案形成了一个清晰的对比：vLLM 全量存储 KV（在 HBM 中）但在 attention 内稀疏读取（SM 内跳过），HiSparse 选择性加载（只加载被选中的 page，其余无需在当前 step 驻留在 GPU 工作集中）且稀疏读取（只对加载的 page 做 attention）。**HiSparse 在「读得少」的同时也减少了当前 step 的 GPU 工作集大小——这是在 [稀疏注意力分类学](../../kv_cache/01_concepts/basic/sparse_attention_taxonomy.md) §5.2 的「读和存的正交性」框架中，压缩近似路线之外的另一种工程路径：不靠压缩 KV 的表示，而是靠选择性加载减少 attention 的输入数据量。**

### 2.2 plan_topk_v2：消除选择与计算之间的间隙

稀疏 attention 执行管道中存在一个性能瓶颈：top-k 选择与后续的索引表构建之间存在一个 GPU kernel launch 和一次 global memory 往返——top-k 结果先写回 global memory，索引构建时再读回。随着 page 数增长，这个往返的开销线性增加。

HiSparse 的 `plan_topk_v2` 是一个 planning kernel：接收 `seqlens_expanded`（1D int32 tensor，每行的序列长度），返回一个 cluster-threshold-based 的 plan tensor。这个 plan 在 decode 阶段跨层复用——所有层共享同一份索引布局规划，避免了每层独立计算。

plan 被后续的融合 top-k kernel 消费。当环境变量 `SGLANG_DSA_FUSE_TOPK` 开启时，top-k 选择与索引表构建合并为单次 kernel 调用（使用 plan_topk_v2 产出的 plan），消除了传统两步方案中间的 global memory 往返。开关的存在本身是一个务实的信号：融合路径在长上下文（page 数多）时收益显著，在短上下文或 PD 分离场景下可以关闭以保持模块化。

### 2.3 RAGGED/PAGED 双模式索引转换

HiSparse 为 extend（prefill）和 decode 两个阶段设计了不同的索引转换策略，因为两个阶段的序列特性完全不同：

**RAGGED 模式（extend 阶段）**：多个请求同时做 prefill，每个请求的前缀长度不同，某些请求可能共享同一段前缀（prefix sharing）。RAGGED 使用 offset-based 索引——每个请求独立标记其 top-k 在连续缓冲区中的位置，通过偏移量定位。适合变长、共享前缀的批处理场景。

**PAGED 模式（decode 阶段）**：所有请求每步只处理一个 query token，序列长度相同但 KV 高度分散在 page table 中——每个请求的 `num_computed_tokens` 不同，相同的 token 位置映射到不同的物理 page。PAGED 使用 page-table-based 索引——通过 `topk_indices` 查找每个选中 token 所属的物理 page，再通过页内偏移定位 token 在 page 中的精确位置。

两个模式由 `TopkTransformMethod` 枚举统一管理，`DSAIndexerMetadata` 在 extend 阶段初始化为 RAGGED，在 decode 阶段由 coordinator 切换为 PAGED。双模式的引入是因为 prefill 和 decode 在 KV 布局上的根本差异——RAGGED 适合连续分配的场景，PAGED 适合离散 page-based 的场景。强行用一种模式覆盖两种场景会导致大量无用的索引计算。

---

## 三、信号中枢：超越 attention 的 HiSparse

HiSparse coordinator 的架构价值不只体现在「DSA 的另一种工程实现」——更关键的是，它在 SGLang 的推理管道中充当了**跨模块的信号中枢**。

HiSparse 每步 decode 产出两类对外信号：**sparse page table**（`swap_in_selected_pages` 的返回值，直接传给 attention backend，只包含被选中的 page）和 **`paged MQA logits metadata`**（描述 top-k 选中的 token 如何分布在 page table 中）。后者被 DeepGEMM（SGLang 的 MoE GEMM 优化模块）消费：DeepGEMM 使用这些元数据来调度 MoE 的 logit 计算——稀疏 page table 隐含了哪些 token 参与当前 step 的计算，DeepGEMM 据此调度对应的 expert，避免对未参与 attention 的 token 做无效的 GEMM dispatch。（`plan_topk_v2` 的 plan tensor 属于内部优化中间件——它在 decode 阶段跨层复用、消除重复的索引布局计算——但对 coordinator 之外不可见。）

对比 vLLM 的方案：同样的 Indexer top-k 信号，vLLM 只在 attention kernel 内部使用一次后即丢弃。SGLang 将它分发给了两条下游管线——KV cache 加载（`swap_in_selected_pages`）和 MoE dispatch（DeepGEMM）。一到多的架构，正是 coordinator 层存在的理由。

---

## 四、取舍

### 4.1 系统复杂性 vs 带宽节省

HiSparse coordinator 是系统层的新组件——它在调度器、attention backend 和 KV cache manager 之间引入了稀疏感知的协调逻辑（page 的选择性加载需要与 attention backend 的 page 布局对齐，page 的加载状态需要与 KV cache manager 的分配状态一致）。这个复杂性在 vLLM 的方案中不存在——vLLM 的 sparse attention 是完全 self-contained 的 backend 实现，其他模块不需要感知稀疏性的存在。

回报是 KV 读取量的直接减少。稀疏度越高（top-k 越小，或 page size 越大），HiSparse 相对 vLLM 方案的带宽优势越显著。极端案例：256 token/page，top-k=512 的 DSA，理论选中 page 数 ≤ 512 / 256 + overhead ≈ 2–4 个 page（假设 token 高度聚集），而全量可能有上千个 page。

### 4.2 融合 kernel 的性能 vs 可维护性

`plan_topk_v2` 和 `SGLANG_DSA_FUSE_TOPK` 体现了 SGLang 团队在「极致性能」和「代码可维护性」之间的务实权衡：提供融合 kernel（性能优先），但保留开关（可维护性优先的逃生通道）。当融合 kernel 出现数值问题时，关闭 fuse 回到两步走路径，不影响功能正确性，仅损失部分性能。这种「双路径 + 开关」的模式是推理系统性能敏感组件中的常见工程实践。

### 4.3 与 vLLM 的互补而非竞争

HiSparse 和 vLLM 的 Sparse MLA 后端不是替代关系——它们在不同条件下的优势自然分开。

**最直接的权衡维度：page size 与稀疏度**。两者的相对性能优势主要由这个维度决定：

| page size × top-k                     |   倾向   | 原因                                                                 |
| ------------------------------------- | :------: | -------------------------------------------------------------------- |
| 小 page（16 token）、大 top-k（2048） |   vLLM   | token 级和 page 级选择的差异微乎其微，coordinator 的翻译开销是净损失 |
| 大 page（256 token）、小 top-k（512） | HiSparse | 未选中的 page 完全不加载，带宽节省远大于翻译开销                     |

**不涉及权衡的特征差异**——它们直接决定能不能用，不取决于参数配置：

- **需要跨模块复用 top-k 信号**（如 MoE dispatch）→ 只能选 HiSparse。vLLM 的信号在 attention 完成后即丢弃，系统层不可见。
- **追求最小系统复杂度、不接受新组件** → 只能选 vLLM。HiSparse 需要 coordinator + 三个模块间的状态一致性维护。

---

## 五、一句话总结

HiSparse 把 DSA 的稀疏选择从 attention kernel 的内部优化变成了跨模块的系统组件——三个设计决策（page 级加载、融合 kernel、双模式索引）分别解决了 page 翻译、间隙消除和阶段适配三个工程问题，而 coordinator 的存在使同一份 top-k 信号同时服务于 KV 加载和 MoE 调度两条下游管线。它证明了稀疏注意力的工程价值不只在于「省了多少计算」，而在于「为系统层的存储和调度决策提供了什么信号」。

---

## 相关阅读

- [稀疏注意力分类学：读什么、不读什么、以及为什么读得少不等于存得少](../../kv_cache/01_concepts/basic/sparse_attention_taxonomy.md) — DSA 的机制原理、分类位置及与 Vegas/H2O/CSA 的对比（本文 §1 的上下文）
- [SGLang 超大规模推理调优案例](sglang_scaling_case_study.md) — SGLang 生产环境中的 KV Cache 竞态与时序缺陷定位
- [SGLang HiCache 深入详解](hicache_deep_dive.md) — SGLang 分层存储架构，HiSparse 与之在 KV 管理上的协同
- [投机解码方法全景](../../vllm/module_analysis/speculative_decoding_landscape.md) — 稀疏注意力自投机与 speculative decoding 的协同
