# KV Cache 技术体系

一个 32K token 的 prompt，80 层 LLaMA-2 70B，batch=8——KV Cache 吃掉 320 GB 显存，是模型权重本身的 2 倍以上。**KV Cache 是 LLM 推理最大的显存消费者，也是几乎所有推理优化的主战场。** 42 篇文章，从"KV Cache 到底是什么"到"如何在数十节点的集群上高效传输和复用"，覆盖了这一技术栈的完整纵深。

> **建议阅读路径**：§1 基础原理 → §2 核心优化技术 → §3 进阶架构与系统 → §4 容量规划。每篇文章独立可读，前后交叉引用。

## 1. 基础原理

在深入优化之前，首先要回答三个基础问题：KV Cache 存了什么、为什么只存 K 和 V 不存 Q、不同注意力架构下存储的形态有何不同。

- **[KV Cache 原理简介](01_concepts/basic/kv_cache_basics.md)** ([配套 PPT](01_concepts/basic/kv_cache_basics.pptx))：详细解析了自回归生成的挑战、KV Cache 的工作机制（Prefill 与 Decode 阶段）以及显存占用分析。
- **[PagedAttention 原理介绍](01_concepts/basic/paged_attention.md)** — OS 分页思想 → GPU 显存管理：block table、按需分配、碎片率从 60-80% 降至 <4%。
- **[KV Cache 为什么叫 KV Cache？——Q 去哪了](01_concepts/basic/why_only_kv.md)** — 检索类比解释 Q 的一次性与 K/V 的持久性，因果掩码的数学约束。
- **[不同注意力类型的 KV Cache 到底长什么样](01_concepts/basic/attention_kv_cache_formats.md)** — MHA / GQA / MQA / MLA / CSA-HCA 五种注意力类型下 KV Cache 的精确形状、显存占用和 vLLM 支持状态。
- **[稀疏注意力分类学：读什么、不读什么、以及为什么读得少不等于存得少](01_concepts/basic/sparse_attention_taxonomy.md)** — 按决策时机将稀疏注意力方法分为三条路线：固定模式（Sliding Window、Longformer）、动态选择（StreamingLLM、Vegas、NSA/DSA）、压缩近似（CSA/HCA、Linformer）。每条路线的设计取舍不同，但共享同一个被普遍忽视的约束——减少 KV 读取量（省计算）不等于减少 KV 存储量（省显存）。
- **[为什么 GPU 生成每个 token 时利用率不到 5%？——Prefill 与 Decode 深度拆解](../prefill_decode/prefill_decode_qkv_calculation.md)**（[交互可视化](../prefill_decode/prefill_decode_visual.html) · [校验脚本](../prefill_decode/prefill_decode_validate.py)）：从一个具体例子出发，逐步标注 Prefill 和 Decode 每一步的矩阵形状与计算量变化，从 compute-bound vs memory-bound 的根本差异出发，推导出 GQA、量化、PagedAttention、Offloading、PD 分离等优化方向的必然性。

## 2. 核心优化技术

知道了 KV Cache 存什么、怎么存之后，下一个问题必然是：**怎么让它存得更少、复用得更多、传输得更快？** 这是推理引擎中最活跃的优化方向——四条主线各自独立、在实践中叠加使用。

### 2.1 Prefix Caching

多轮对话、System Prompt、Few-shot 模板等场景下输入前缀高度重复——Prefix Caching 通过 Hash 或 Radix Tree 索引复用已计算的 KV 块，将命中请求的 Prefill 成本压到接近零，是长对话与 RAG 场景下 TTFT 优化的第一道防线。

- **[RadixAttention 原理与 SGLang 实践及 vLLM APC 对比](01_concepts/prefix_caching/radix_attention.md)** ([配套 PPT](01_concepts/prefix_caching/radix_attention.pptx))：深入剖析基于 Radix Tree 自动复用 KV Cache 的核心原理及其在系统中的调度机制，并与 vLLM 的 APC 方案进行对比。
- **[Prefix Caching 原理与实现](01_concepts/prefix_caching/prefix_caching.md)** ([配套 PPT](01_concepts/prefix_caching/prefix_caching.pptx))：详细介绍了 Prefix Caching 的核心原理、vLLM 的 Automatic Prefix Caching (APC) 实现，以及 LMCache 的多级 Prefix Caching 架构。涵盖哈希算法设计、跨实例共享模式、性能收益分析及最佳实践。
- **[Claude 提示词缓存机制与源码实现深度分析](01_concepts/prefix_caching/claude_prompt_caching.md)**：分析 Claude 如何在终端 Agent 环境下落地 Prompt Caching 机制，通过复用请求的上下文前缀降低大规模任务的处理延迟。
- **[RoPE 与 Prefix Caching 的互作用](01_concepts/prefix_caching/rope_and_prefix_caching.md)**：探讨 Rotary Position Embedding 的位置编码机制如何影响 Prefix Caching 的正确性与实现约束。

### 2.2 调度、传输与执行优化

探讨独立于具体系统的架构级优化——调度策略如何改变 KV Cache 的分配时序、跨节点传输如何隐藏延迟、执行模型如何容纳动态 block table。

#### 2.2.1 调度

- **[vLLM Chunked Prefill 如何改变 KV Cache 管理](01_concepts/scheduling/01_vllm_chunked_prefill.md)**：拆解将长 prompt 切成小块逐步 Prefill 后，KV Cache 分配从"一次性申请全部 block"变为"逐步申请逐批增长"，以及与 Prefix Caching 的交互约束。
- **[投机解码如何与 KV Cache 交互](01_concepts/scheduling/02_vllm_spec_decode.md)**：拆解投机解码引入的三个 KV Cache 操作——Placeholder 预留槽位、`num_computed_tokens` 回退实现猜错撤销、draft/target KV 形状对齐。

#### 2.2.2 PD 分离传输

- **[PD 分离架构下的 KV Cache 传输](01_concepts/pd_transfer/01_disaggregated_prefill_kv_transfer.md)**：从 Push/Pull、Eager/Pipelined/Lazy、完整/增量三个维度，对比 vLLM KV Connector V1、LMCache PD Backend 和 Mooncake 的设计选择。

#### 2.2.3 卸载与预取

- **[KV Offloading 架构对比](01_concepts/offloading/01_kv_offloading.md)**：探讨 vLLM 原生 KV Offloading 与 LMCacheConnector 将 KV Cache 卸载到 CPU/磁盘的策略与性能权衡。
- **[KV Cache 层级流水线并行](01_concepts/offloading/02_layerwise_pipeline.md)**：按层流水线传输技术在 Prefill-Decode 分离架构中的应用，计算与 KV I/O 重叠的机制。
- **[KV Cache Prefetching：三层预取](01_concepts/offloading/03_kv_cache_prefetching.md)**：Kernel 层（L2 prefetch）、系统层（PD 异步预取）、存储层（HiCache）三层如何隐藏 KV 访问延迟。
- **[稀疏注意力 × KV Cache Offloading：跨层联动必须回答的八个问题](01_concepts/offloading/sparse_attention_driven_offloading_problems.md)**：当稀疏注意力的 token 重要性信号用于指导 offloading 决策时，在信号质量、信号翻译、系统副作用和经济性四个层级上的关键设计决策点。问题枚举式结构，不提供设计方案，只提供设计者绕不开的取舍。

#### 2.2.4 执行模型

- **[CUDA Graph 与 KV Cache](01_concepts/execution/01_vllm_cuda_graph.md)**：Full / Piecewise / FULL_AND_PIECEWISE 三种模式如何容纳动态 block table——可变输入缓冲区、多尺寸预录制、re-capture 触发条件。

### 2.3 压缩与量化机制

针对超长上下文带来的显存压力，探索如何通过量化、剪枝等技术压缩 KV Cache 的物理体积。

- **[KV Cache 压缩技术详解：原理、架构与趋势](01_concepts/compression/kv_cache_compression.md)** ([配套 PPT](01_concepts/compression/kv_cache_compression.pptx))：系统解析了通过量化（如 INT8/FP8/INT4）、稀疏化（如 StreamingLLM、H2O）以及注意力机制优化等手段，大幅降低大语言模型长上下文场景下的显存占用与传输带宽需求。
- **[KV Cache 量化深度解析](01_concepts/compression/kv_cache_quantization.md)**：拆解三种量化粒度——Per-Tensor、Per-Token-Head（FP8/INT8）、NVFP4——的精度差异、工程实现与 vLLM 配置，以及量化对 Prefix Caching 和误差传播的影响。

### 2.4 淘汰策略

压缩减小每个 token 的体量，淘汰则直接减少存储的 token 数量——当压缩做到极致后，淘汰是唯一可以继续缩容的手段。从 Attention Sinks 的发现出发，回答"滑动窗口为什么不够"和"哪些 token 的 KV 值得保留"。

- **[KV Cache 淘汰策略：从滑动窗口到注意力引导的精确淘汰](01_concepts/eviction/attention_sinks_and_eviction.md)**：Attention Sinks 作为淘汰必须遵守的硬约束，在此之上 H2O（累积注意力 → Heavy Hitter）、SnapKV（观察窗口投票）、StreamingLLM（Sink + Window）三条 Informed Eviction 路线通过注意力分数区分 token 信息价值，以及 vLLM Preemption 在系统层的配合机制。
- **[Key-Key Semantic Affinity：用 Key 向量替代注意力分数的 KV Cache 重要性评估](01_concepts/eviction/key_key_semantic_affinity.md)**：系统介绍 SamKV (AAAI 2026) 提出的 Key-Key 语义亲和度方法——不再依赖 QK^T 注意力分数，而是用 Key 向量自身的语义距离评估 block 重要性。从循环悖论出发，分析 H2O/SnapKV 的全局累积偏差，详解高维正交性原理与浓度不等式保证，并给出层次化选择、Per-Step 集成和稳定性过滤三项工程落地路径。

---

## 3. 进阶架构与管理系统

§2 的优化都在单个推理实例内进行。但当上下文推到百万级、集群规模到数十节点时，**KV Cache 的管理从"进程内"变成了"分布式系统"问题**——跨节点传输、一致性协议、全局调度。以下是业界五个代表性方案，从中心化到去中心化，各有不同的取舍。

### 3.1 LMCache

LMCache 通过多层级存储架构（GPU/CPU/Disk/Remote）实现跨实例的 KV Cache 重用，支持分布式环境下的状态共享与预填充-解码分离。

- **[LMCache 源码分析指南](02_systems/lmcache/README.md)**：完整的七阶段学习路径与文档索引。
- **[LMCache 架构概览](02_systems/lmcache/lmcache_overview.md)**：L1-L4 四层存储架构（GPU、CPU、磁盘、远程）与本地复用 / 集群共享 / 流水线传输三种核心范式。
- **核心链路**：
  - **[LMCacheConnector 源码分析](02_systems/lmcache/lmcache_connector.md)**：vLLM 集成入口与请求拦截。
  - **[LMCacheEngine 源码分析](02_systems/lmcache/lmcache_engine.md)**：核心控制流与 I/O 编排。
- **分布式控制**：
  - **[LMCache Controller（控制平面）架构剖析](02_systems/lmcache/lmcache_controller.md)**：基于 ZMQ 的集群控制平面与元数据管理。
- **存储子系统**：
  - **[LMCache 分层存储架构与调度机制](02_systems/lmcache/lmcache_storage_overview.md)**：StorageManager 调度器与 Write-All/Waterfall 策略。
  - **后端实现细节**：
    - **[LocalCPUBackend 源码分析](02_systems/lmcache/local_cpu_backend.md)** (L1)：高性能内存管理。
    - **[P2PBackend 源码分析](02_systems/lmcache/p2p_backend.md)** (L2)：基于 RDMA 的去中心化传输。
    - **[PDBackend（预填充-解码分离后端）源码分析](02_systems/lmcache/pd_backend.md)**：预填充-解码分离的主动推送机制。
    - **[LocalDiskBackend 源码分析](02_systems/lmcache/local_disk_backend.md)** (L3)：基于 O_DIRECT 的磁盘缓存。
    - **[GdsBackend 源码分析](02_systems/lmcache/gds_backend.md)** (L3)：利用 GPUDirect Storage 的极致持久化。
    - **[NixlStorageBackend 源码分析](02_systems/lmcache/nixl_backend.md)** (L3/L4)：基于 NIXL 的通用传输与 S3 对接。
    - **[Remote Connector（远程连接器）源码分析](02_systems/lmcache/remote_connector.md)** (L4)：适配 Redis/S3/Mooncake 等远程存储。
- **服务端实现**：
  - **[LMCache Server 源码分析](02_systems/lmcache/lmcache_server.md)**：轻量级中心化存储服务。
- **高级特性**：
  - **[CacheBlend：RAG 场景下的 KV Cache 动态融合机制与源码剖析](02_systems/lmcache/cache_blend.md)**：通过选择性重算解决非前缀复用问题。
  - **[CacheGen：KV Cache 的高效压缩与流式传输](02_systems/lmcache/cachegen.md)**：通过自适应量化与算术编码显著降低网络传输带宽需求。

### 3.2 Tair KVCache

Tair KVCache 依托 Tair 数据库构建中心化元数据与分布式存储架构，通过两阶段写入与滑动窗口匹配，提供企业级的高性能 KV Cache 共享与一致性保障。

- **[Tair KVCache 架构与设计深度分析](02_systems/tair_kvcache/tair-kvcache-architecture-design.md)**：深入分析了 Tair KVCache Manager (KVCM) 的架构。它采用中心化元数据管理 + 分布式存储的模式，支持 KV 匹配、前缀匹配和滑动窗口匹配，并实现了两阶段写入机制以保障数据一致性。

### 3.3 NVIDIA KVBM (KV Block Manager)

KVBM 作为 NVIDIA Dynamo 项目的核心组件，通过统一内存 API 管理异构存储（GPU/CPU/SSD），并结合 NIXL 库（GDS/RDMA）实现高效数据传输，服务于 TensorRT-LLM 等高性能推理框架。

- **[KV Block Manager (KVBM) 深度解析](02_systems/kvbm/KVBM_Analysis.md)** ([配套 PPT](02_systems/kvbm/NVIDIA_Dynamo_KVBM_Architecture.pptx) / [可编辑 PPT](02_systems/kvbm/NVIDIA_Dynamo_KVBM_可编辑.pptx))：剖析了 KVBM 如何通过统一内存 API 管理异构存储（GPU/CPU/SSD），利用 Block 机制和状态机管理内存生命周期，并结合 NIXL 库实现高效的数据传输（如 GDS、RDMA）。

### 3.4 Mooncake 架构

Mooncake 采用以 KV Cache 为中心的分离式推理架构，通过分块管道并行（CPP）与全局调度器（Conductor），实现超长上下文场景下的资源极致利用。

- **[Mooncake 架构概览：以 KV Cache 为中心的高效 LLM 推理系统设计](02_systems/mooncake/mooncake_architecture.md)**：介绍了基于 KVCache 调度的预填充-解码分离架构。通过分块管道并行（CPP）和全局调度器（Conductor），Mooncake 实现了超长上下文场景下的高效推理和资源利用。

### 3.5 SGLang HiCache

HiCache 是 SGLang 自带的分层 KV Cache 架构，将 GPU 显存、宿主机内存与分布式存储后端（如 Mooncake、HF3FS）统一为 L1/L2/L3 三级缓存，突破单节点显存天花板并实现跨实例的前缀共享。

- **[HiCache 深入详解](../sglang/hicache_deep_dive.md)**：系统梳理 HiCache 的演进背景、HiRadixTree 元数据拓扑、三种预取策略（`best_effort` / `wait_complete` / `timeout`）与三种写回策略（`write_through` / `write_through_selective` / `write_back`）、`page_first` 内存布局与 GPU 辅助 I/O 算子、存储后端热插拔控制面，以及根据容量 / 异构 TP / PD 一致性 / 存储成本 四维度展开的架构权衡与启动参数示例。

### 3.6 NIXL 网络传输库

NIXL 是 NVIDIA 开源的高性能网络传输抽象层，为 LMCache、KVBM 等 KV Cache 系统提供统一的 RDMA、GDS 与跨节点数据传输能力。

- **[NIXL 网络存储介绍](02_systems/nixl/nixl_introduction.md)**：高性能网络存储架构、核心抽象与应用场景。

---

## 4. 容量规划与 ROI 分析

掌握了"怎么做"之后，最后一个问题是"值不值得做"。KV Cache 本质是一次「用存储成本换计算成本」的投资——合理的分层容量与命中率假设决定整体 ROI。以下推演以 GLM-5 与 Agent 业务负载为基准。

- **[KV Cache 引入收益评估](01_concepts/capacity_planning/kv_cache_roi.md)**：全面评估在 Agent 业务爆发和长上下文常态化背景下，引入 KV Cache（如 LMCache）技术的整体收益与投资回报。
- **[GLM-5 模型 KV Cache 容量规划报告](01_concepts/capacity_planning/glm5_kv_cache_capacity_planning.md)**：针对 GLM-5 模型的显存与各级存储（CPU 内存、NVMe 固态硬盘）的容量需求进行详细推演。

---

## 5. 核心参考文献

> 以下为各篇文章中反复引用的核心论文，按主题分类。每篇文章自身的完整引用见文内脚注。

### 5.1 基础架构与注意力机制

- **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022.
- **FlashAttention-3**: Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," 2024.
- **PagedAttention**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023.
- **GQA**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," EMNLP 2023.
- **MLA**: DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model," 2024.
- **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021.

### 5.2 KV Cache 淘汰与压缩

- **StreamingLLM & Attention Sinks**: Xiao et al., "Efficient Streaming Language Models with Attention Sinks," ICLR 2024.
- **H₂O**: Zhang et al., "H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models," NeurIPS 2023.
- **SnapKV**: Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation," NeurIPS 2024.
- **SamKV**: Cao et al., "Sparse Attention across Multiple-context KV Cache," AAAI 2026. arXiv:2508.11661.
- **KIVI**: Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," 2024.

### 5.3 系统架构与传输

- **vLLM**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023.
- **SGLang**: Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," NeurIPS 2024.
- **Mooncake**: Qin et al., "Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving," 2024.
- **LMCache**: LMCache Project. <https://github.com/LMCache/LMCache>
- **CacheBlend**: Yao et al., "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion," EuroSys 2025.
- **NIXL**: NVIDIA Inc. <https://github.com/ai-dynamo/nixl>

### 5.4 Prefetch 与执行优化

- **KV Cache Prefetching**: Zhao et al., "Asynchronous KV Cache Prefetching for LLM Inference," arXiv:2504.06319, 2025.
- **Speculative Decoding**: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding," ICML 2023.
- **Eagle**: Li et al., "Eagle: Speculative Decoding Requires Rethinking Feature Uncertainty," 2024.

### 5.5 第三方资源

- marsggbo. easy-kvcache. <https://github.com/marsggbo/easy-kvcache>
- vLLM Project. <https://github.com/vllm-project/vllm>
- SGLang Project. <https://github.com/sgl-project/sglang>
