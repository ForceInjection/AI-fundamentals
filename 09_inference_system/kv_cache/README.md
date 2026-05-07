# KV Cache 技术体系

**KV Cache（键值缓存）** 是现代 LLM 推理系统的核心基础设施——它通过缓存 Attention 的 Key/Value 矩阵把自回归生成的单步计算复杂度从 $O(N^2)$ 降至 $O(N)$，直接决定 TTFT、吞吐与长上下文可行性。围绕这一数据结构，业界已从单机显存缓存演进出涵盖 Prefix Caching、分层存储、跨实例共享、预填充-解码分离的完整技术栈，以下内容按「原理 → 前缀复用 → 分布式系统 → 关键技术 → 容量规划」五条主线组织。

## 1. 基础原理

自回归生成天然存在重复计算——每生成一个 Token 都需对全部历史 Token 重算 Attention。缓存 Key/Value 矩阵后，Prefill 阶段批量构造、Decode 阶段增量追加，整体计算量从 $O(N^3)$ 压缩到 $O(N^2)$，而显存占用则与 `layer × head × d_head × seq_len × dtype` 线性相关，构成后续所有优化的权衡起点。

- **[KV Cache 原理简介](01_concepts/basic/kv_cache_原理简介.md)**：详细解析了自回归生成的挑战、KV Cache 的工作机制（Prefill 与 Decode 阶段）以及显存占用分析。

## 2. Prefix Caching

多轮对话、System Prompt、Few-shot 模板等场景下输入前缀高度重复——Prefix Caching 通过 Hash 或 Radix Tree 索引复用已计算的 KV 块，将命中请求的 Prefill 成本压到接近零，是长对话与 RAG 场景下 TTFT 优化的第一道防线。

- **[RadixAttention 原理与 SGLang 实践及 vLLM APC 对比](01_concepts/prefix_caching/radix_attention.md)** ([配套 PPT](01_concepts/prefix_caching/radix_attention.pptx))：深入剖析基于 Radix Tree 自动复用 KV Cache 的核心原理及其在系统中的调度机制，并与 vLLM 的 APC 方案进行对比。
- **[Prefix Caching 原理与实现](01_concepts/prefix_caching/prefix_caching.md)** ([配套 PPT](01_concepts/prefix_caching/prefix_caching.pptx))：详细介绍了 Prefix Caching 的核心原理、vLLM 的 Automatic Prefix Caching (APC) 实现，以及 LMCache 的多级 Prefix Caching 架构。涵盖哈希算法设计、跨实例共享模式、性能收益分析及最佳实践。

## 3. 进阶架构与管理系统

百万级上下文与分离式推理把 KV Cache 推出了单卡显存——业界沿着「分层存储（GPU/CPU/SSD/远程）+ 跨实例共享 + 元数据一致性」三条主线构建了 LMCache、Tair KVCache、KVBM、Mooncake、HiCache 五套代表性方案，设计取舍主要体现在中心化程度、传输协议（RDMA / NIXL / GDS）与面向的推理拓扑上。

### 3.1 LMCache

LMCache 是一个专为 LLM 推理引擎设计的 KV Cache 管理系统，旨在通过多层级存储架构实现跨实例的 KV Cache 重用。

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

Tair KVCache 是阿里云推出的企业级 KVCache 解决方案，基于 Tair 数据库技术，提供了高性能的分布式 KV Cache 管理能力。

- **[Tair KVCache 架构与设计深度分析](02_systems/tair_kvcache/tair-kvcache-architecture-design.md)**：深入分析了 Tair KVCache Manager (KVCM) 的架构。它采用中心化元数据管理 + 分布式存储的模式，支持 KV 匹配、前缀匹配和滑动窗口匹配，并实现了两阶段写入机制以保障数据一致性。

### 3.3 NVIDIA KVBM (KV Block Manager)

KVBM 是 NVIDIA Dynamo 项目中的核心组件，服务于 vLLM 和 TensorRT-LLM 等高性能推理框架。

- **[KV Block Manager (KVBM) 深度解析](02_systems/kvbm/KVBM_Analysis.md)**：剖析了 KVBM 如何通过统一内存 API 管理异构存储（GPU/CPU/SSD），利用 Block 机制和状态机管理内存生命周期，并结合 NIXL 库实现高效的数据传输（如 GDS、RDMA）。

### 3.4 Mooncake 架构

Mooncake 是 Moonshot AI（Kimi）推出的以 KV Cache 为中心的分离式推理架构。

- **[Mooncake 架构概览：以 KV Cache 为中心的高效 LLM 推理系统设计](02_systems/mooncake/mooncake_architecture.md)**：介绍了基于 KVCache 调度的预填充-解码分离架构。通过分块管道并行（CPP）和全局调度器（Conductor），Mooncake 实现了超长上下文场景下的高效推理和资源利用。

### 3.5 SGLang HiCache

HiCache 是 SGLang 自带的分层 KV Cache 架构，将 GPU 显存、宿主机内存与分布式存储后端（如 Mooncake、HF3FS）统一为 L1/L2/L3 三级缓存，突破单节点显存天花板并实现跨实例的前缀共享。

- **[HiCache 深入详解](02_systems/hicache/hicache_deep_dive.md)**：系统梳理 HiCache 的演进背景、HiRadixTree 元数据拓扑、三种预取策略（`best_effort` / `wait_complete` / `timeout`）与三种写回策略（`write_through` / `write_through_selective` / `write_back`）、`page_first` 内存布局与 GPU 辅助 I/O 算子、存储后端热插拔控制面，以及根据容量 / 异构 TP / PD 一致性 / 存储成本 四维度展开的架构权衡与启动参数示例

## 4. 关键技术分析

独立于某一具体系统、但跨方案共性的技术议题——涵盖 Offloading 策略的吞吐/带宽取舍，以及按层流水并行如何掩盖 PD 分离中的传输开销。

- **[vLLM KV Offloading Connector 与 LMCacheConnector：架构设计与性能深度对比](01_concepts/advanced/kv_offloading_analysis.md)**：探讨了将 KV Cache 卸载到 CPU 或磁盘的策略与性能权衡。
- **[KV Cache 层级流水线并行](01_concepts/advanced/layerwise_pipeline.md)**：分析了按层流水线传输技术在 Prefill-Decode 分离架构中的应用。

## 5. 容量规划与 ROI 分析

KV Cache 本质是一次「用存储成本换计算成本」的投资——合理的分层容量（显存/CPU 内存/NVMe）与命中率假设决定整体 ROI，相关推演以 GLM-5 与 Agent 业务负载为基准。

- **[KV Cache 引入收益评估](01_concepts/capacity_planning/kv_cache_roi.md)**：全面评估在 Agent 业务爆发和长上下文常态化背景下，引入 KV Cache（如 LMCache）技术的整体收益与投资回报。
- **[GLM-5 模型 KV Cache 容量规划报告](01_concepts/capacity_planning/glm5_kv_cache_capacity_planning.md)**：针对 GLM-5 模型的显存与各级存储（CPU 内存、NVMe 固态硬盘）的容量需求进行详细推演。

## 6. 目录结构说明

| 目录/文件      | 说明                                                                                  |
| :------------- | :------------------------------------------------------------------------------------ |
| `01_concepts/` | 核心概念与技术 (包含 basic, compression, prefix_caching, advanced, capacity_planning) |
| `02_systems/`  | 具体系统实现 (包含 lmcache, tair_kvcache, kvbm, mooncake, hicache)                    |
