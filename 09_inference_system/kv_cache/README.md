# KV Cache 技术体系

本章节深入探讨大语言模型（LLM）推理中的关键优化技术 —— **KV Cache（键值缓存）**。作为提升推理性能、降低延迟（特别是 Time-To-First-Token, TTFT）和提高吞吐量的核心手段，KV Cache 及其管理系统已成为现代推理系统的基础设施。本目录涵盖了从基础原理到业界前沿的分布式 KV Cache 管理架构。

## 1. 基础原理

KV Cache 是 LLM 推理加速的基石。在自回归生成过程中，通过缓存 Attention 层的 Key 和 Value 矩阵，避免了对历史 Token 的重复计算，从而实现了推理计算量从 $O(N^3)$ 到 $O(N^2)$ 的显著降低（在单步生成层面是从 $O(N^2)$ 降至 $O(N)$）。

- **[KV Cache 原理简介](basic/kv_cache_原理简介.md)**：详细解析了自回归生成的挑战、KV Cache 的工作机制（Prefill 与 Decode 阶段）以及显存占用分析。

## 2. Prefix Caching

Prefix Caching（前缀缓存）是 KV Cache 优化中的关键技术，通过缓存和复用重复前缀的 KV Cache，可以显著降低 TTFT 并提升系统吞吐量。

- **[RadixAttention 原理与 SGLang 实践及 vLLM APC 对比](prefix_caching/radix_attention.md)**：深入剖析基于 Radix Tree 自动复用 KV Cache 的核心原理及其在系统中的调度机制。
- **[Prefix Caching 原理与实现](prefix_caching/prefix_caching.md)**：详细介绍了 Prefix Caching 的核心原理、vLLM 的 Automatic Prefix Caching (APC) 实现，以及 LMCache 的多级 Prefix Caching 架构。涵盖哈希算法设计、跨实例共享模式、性能收益分析及最佳实践。

## 3. 进阶架构与管理系统

随着上下文长度的增加（Long Context）和分布式推理的需求，简单的显存内 KV Cache 已无法满足需求。业界涌现出多种分层存储和分布式管理方案，将 KV Cache 扩展到 CPU 内存、磁盘甚至远程存储。

### 3.1 LMCache

LMCache 是一个专为 LLM 推理引擎设计的 KV Cache 管理系统，旨在通过多层级存储架构实现跨实例的 KV Cache 重用。

- **[LMCache 架构概览](lmcache/lmcache_overview.md)**：介绍了 LMCache 的 L1-L4 四层存储架构（GPU、CPU、磁盘、远程），以及支持的三种核心范式：本地复用、集群共享和流水线传输。
- **核心链路**：
  - **[LMCacheConnector (vLLM 集成) 代码分析](lmcache/lmcache_connector.md)**：vLLM 集成入口与请求拦截。
  - **[LMCacheEngine 核心引擎代码分析](lmcache/lmcache_engine.md)**：核心控制流与 I/O 编排。
- **分布式控制**：
  - **[LMCache Controller (控制平面) 架构剖析](lmcache/lmcache_controller.md)**：基于 ZMQ 的集群控制平面与元数据管理。
- **存储子系统**：
  - **[LMCache 分层存储架构与调度机制详解](lmcache/lmcache_storage_overview.md)**：StorageManager 调度器与 Write-All/Waterfall 策略。
  - **后端实现细节**：
    - **[LocalCPUBackend 源码分析](lmcache/local_cpu_backend.md)** (L1)：高性能内存管理。
    - **[P2PBackend 源码分析](lmcache/p2p_backend.md)** (L2)：基于 RDMA 的去中心化传输。
    - **[PDBackend (预填充-解码分离后端) 源码分析](lmcache/pd_backend.md)**：预填充-解码分离的主动推送机制。
    - **[LocalDiskBackend 源码分析](lmcache/local_disk_backend.md)** (L3)：基于 O_DIRECT 的磁盘缓存。
    - **[GdsBackend 源码分析](lmcache/gds_backend.md)** (L3)：利用 GPUDirect Storage 的极致持久化。
    - **[NixlStorageBackend 源码分析](lmcache/nixl_backend.md)** (L3/L4)：基于 NIXL 的通用传输与 S3 对接。
    - **[Remote Connector (远程连接器) 源码分析](lmcache/remote_connector.md)** (L4)：适配 Redis/S3/Mooncake 等远程存储。
- **服务端实现**：
  - **[LMCache Server 源码分析](lmcache/lmcache_server.md)**：轻量级中心化存储服务。
- **高级特性**：
  - **[CacheBlend 技术详解：RAG 场景下的 KV Cache 动态融合机制与源码剖析](lmcache/cache_blend.md)**：RAG 场景下的 KV Cache 动态融合机制，通过选择性重算解决非前缀复用问题。
  - **[CacheGen 技术详解：KV Cache 的高效压缩与流式传输](lmcache/cachegen.md)**：KV Cache 的高效压缩与流式传输技术，显著降低网络传输带宽需求。

### 3.2 Tair KVCache

Tair KVCache 是阿里云推出的企业级 KVCache 解决方案，基于 Tair 数据库技术，提供了高性能的分布式 KV Cache 管理能力。

- **[Tair KVCache 架构与设计深度分析](ali_tair_kvcache/tair-kvcache-architecture-design.md)**：深入分析了 Tair KVCache Manager (KVCM) 的架构。它采用中心化元数据管理 + 分布式存储的模式，支持 KV 匹配、前缀匹配和滑动窗口匹配，并实现了两阶段写入机制以保障数据一致性。

### 3.3 NVIDIA KVBM (KV Block Manager)

KVBM 是 NVIDIA Dynamo 项目中的核心组件，服务于 vLLM 和 TensorRT-LLM 等高性能推理框架。

- **[KV Block Manager (KVBM) 深度解析](kvbm/KVBM_Analysis.md)**：剖析了 KVBM 如何通过统一内存 API 管理异构存储（GPU/CPU/SSD），利用 Block 机制和状态机管理内存生命周期，并结合 NIXL 库实现高效的数据传输（如 GDS、RDMA）。

### 3.4 Mooncake 架构

Mooncake 是 Moonshot AI（Kimi）推出的以 KV Cache 为中心的分离式推理架构。

- **[Mooncake 架构概览：以 KV Cache 为中心的高效 LLM 推理系统设计](mooncake/mooncake_architecture.md)**：介绍了其基于 KVCache 调度的预填充-解码分离架构。通过分块管道并行（CPP）和全局调度器（Conductor），Mooncake 实现了超长上下文场景下的高效推理和资源利用。

## 4. 关键技术分析

除了具体的系统架构，本目录还包含对特定技术点的深度分析。

- **[vLLM KV Offloading Connector 与 LMCacheConnector：架构设计与性能深度对比](advanced_techniques/kv_offloading_analysis.md)**：探讨了将 KV Cache 卸载到 CPU 或磁盘的策略与性能权衡。
- **[KV Cache 层级流水线并行](advanced_techniques/layerwise_pipeline.md)**：分析了按层流水线传输技术在 Prefill-Decode 分离架构中的应用。

## 5. 目录结构说明

| 目录/文件              | 说明                                     |
| :--------------------- | :--------------------------------------- |
| `basic/`               | KV Cache 基础原理及图解                  |
| `prefix_caching/`      | Prefix Caching & RadixAttention 相关技术 |
| `advanced_techniques/` | 层级流水线与卸载策略等进阶优化           |
| `mooncake/`            | Mooncake 推理系统架构解析                |
| `lmcache/`             | LMCache 项目相关文档及组件详解           |
| `ali_tair_kvcache/`    | 阿里云 Tair KVCache 架构文档             |
| `kvbm/`                | NVIDIA KV Block Manager 技术文档         |
