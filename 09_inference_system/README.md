# 推理优化技术方案

## 1. 推理系统架构设计

推理系统架构直接决定了系统的性能、可扩展性和资源利用效率。本节介绍现代推理系统的核心架构创新与设计模式。

- [Mooncake 架构详解：以 KV Cache 为中心的高效 LLM 推理系统设计](kv_cache/mooncake/mooncake_architecture.md) - 新一代推理系统的架构创新与性能优化策略

## 2. KV Cache 核心技术

KV Cache 的高效管理是大模型长文本推理和并发优化的关键。本节深度剖析 LMCache 与 Tair 等分布式 KV Cache 系统的架构与实现。

- [KV Cache 技术体系](kv_cache/README.md) - KV Cache 技术体系全景指南
- [KV Cache 原理简介](kv_cache/basic/kv_cache_原理简介.md) - 自回归生成的挑战与 KV Cache 的工作机制
- [Prefix Caching 技术详解](kv_cache/prefix_caching/prefix_caching.md) - 从原理到 vLLM/LMCache 实践的前缀缓存技术
- [RadixAttention 技术详解](kv_cache/prefix_caching/radix_attention.md) - 基于 Radix Tree 自动复用 KV Cache 的核心原理与 SGLang 实践

### 2.1 LMCache 核心架构与后端实现

本小节详细解析 LMCache 的四层存储架构及其在跨实例缓存复用中的技术细节。

**基础与架构概览**：

- [LMCache 源码分析指南](kv_cache/lmcache/README.md) - 完整学习路径与文档索引
- [LMCache 架构概览](kv_cache/lmcache/lmcache_overview.md) - 四层存储架构 (L1-L4)、核心组件交互与典型工作流
- [vLLM KV Offloading 与 LMCache 深度对比](kv_cache/advanced_techniques/kv_offloading_analysis.md) - 架构设计、存储层级及跨实例共享能力上的核心差异与性能权衡

**核心运行时组件**：

- [LMCacheEngine 源码分析](kv_cache/lmcache/lmcache_engine.md) - 核心调度中枢、异步事件管理与层级流水线
- [LMCacheConnector 源码分析](kv_cache/lmcache/lmcache_connector.md) - vLLM 集成适配器、视图转换与流水线加载
- [分层存储架构与调度机制](kv_cache/lmcache/lmcache_storage_overview.md) - StorageManager 调度器、Write-All 策略与 Waterfall 检索

**存储后端实现**：

- [LocalCPUBackend 源码分析](kv_cache/lmcache/local_cpu_backend.md) - 本地 CPU 内存后端与并发控制
- [LocalDiskBackend 源码分析](kv_cache/lmcache/local_disk_backend.md) - O_DIRECT 直通 I/O 与异步优化
- [P2PBackend 源码分析](kv_cache/lmcache/p2p_backend.md) - RDMA 零拷贝与去中心化传输
- [GdsBackend 源码分析](kv_cache/lmcache/gds_backend.md) - GPUDirect Storage 零拷贝
- [NixlStorageBackend 源码分析](kv_cache/lmcache/nixl_backend.md) - 高性能网络存储、S3 对象存储对接
- [Remote Connector 源码分析](kv_cache/lmcache/remote_connector.md) - Redis/S3/Mooncake 多后端适配
- [PDBackend 源码分析](kv_cache/lmcache/pd_backend.md) - 预填充-解码分离、Push-based 主动推送机制

**控制面**：

- [LMCache Controller (控制平面)](kv_cache/lmcache/lmcache_controller.md) - 集群元数据管理、ZMQ 三通道通信与节点协调
- [LMCache Server 源码分析](kv_cache/lmcache/lmcache_server.md) - 轻量级中心化存储服务、自定义 TCP 协议

**高级特性**：

- [CacheBlend 技术详解](kv_cache/lmcache/cache_blend.md) - RAG 场景下的动态融合机制、选择性重算与精度保持
- [CacheGen 技术详解](kv_cache/lmcache/cachegen.md) - KV Cache 压缩与流式传输、自适应量化与算术编码

### 2.2 阿里云 Tair KVCache

本小节介绍阿里云企业级的 KVCache 管理系统架构及大规模部署实践。

- [Tair KVCache 架构与设计深度分析](kv_cache/ali_tair_kvcache/tair-kvcache-architecture-design.md) - 阿里云企业级 KVCache 管理系统架构详解，包含与 LMCache 的全面对比分析、中心化管理模式及大规模部署最佳实践

## 3. 推理优化技术体系

推理优化技术体系是提升大模型推理性能的核心技术集合，包括算法优化、硬件加速、系统调优和架构设计等多个维度。

**vLLM 核心机制分析**：

- [vLLM 推理系统优化与分析](vllm/README.md) - vLLM 底层机制和系统架构的深度解构
- [vLLM 注意力机制演进与支持全景](vllm/module_analysis/vllm_attention_mha_mla_nsa.md) - 从 MHA 到 MLA 与 NSA 的架构解析及 vLLM 支持现状
- [vLLM 内置 KV Cache Offloading 模块解析](vllm/module_analysis/vllm_native_kv_offloading.md) - 原生 KV Cache CPU Offloading 功能原理与实现
- [vLLM Hybrid KV Cache Manager](vllm/module_analysis/vllm_hybrid_kv_cache_manager_deep_dive.md) - vLLM 针对混合注意力架构的显存优化机制
- [vLLM Router 架构解析](vllm/related_module/vllm_router.md) - 高性能、轻量级请求转发系统
- [vLLM Semantic Router](vllm/related_module/vllm_semantic_router_deep_dive.md) - 基于语义的智能路由策略

**显存与缓存优化**：

- [LLM 显存占用分析与计算](memory_calc/memory_analysis.md) - 模型参数、KV Cache 与中间激活值的显存估算方法
- [KV Block Manager 分析](kv_cache/kvbm/KVBM_Analysis.md) - KV Cache 内存管理机制深度解析
- [分层流水线技术](kv_cache/advanced_techniques/layerwise_pipeline.md) - Layer-wise Pipeline 技术原理与性能优化

**网络与模型工具**：

- [NIXL 网络存储介绍](infrastructure/nixl_introduction.md) - 高性能网络存储架构与应用
- [NVIDIA 模型优化器](model_optimization/nvidia_model_optimizer.md) - NVIDIA 模型优化工具链详解

## 4. 推理优化参考设计

本系列文档提供了企业级 LLM 推理系统的完整参考设计，涵盖从规模分析到实施落地的全流程指南。

**基础理论与技术选型**：

- [背景与目标](reference_design/01-背景与目标.md) - 推理优化的背景分析与核心目标
- [集群规模分类与特征分析](reference_design/02-集群规模分类与特征分析.md) - 不同规模集群的特点与需求
- [核心推理优化技术深度解析](reference_design/03-核心推理优化技术深度解析.md) - KV Cache、批处理、量化等核心技术
- [不同集群规模的技术选型策略](reference_design/04-不同集群规模的技术选型策略.md) - 针对性的技术方案选择

**架构设计与评估体系**：

- [推理服务架构设计](reference_design/06-推理服务架构设计.md) - 企业级推理服务架构设计方案
- [面向推理执行图的异构调度系统架构设计](reference_design/面向推理执行图的异构调度系统架构设计.md) - 跨设备、跨阶段、跨模型的精细化调度方案
- [性能评估指标体系](reference_design/05-性能评估指标体系.md) - 推理性能评估指标与方法

**专业领域优化**：

- [多模态推理优化](reference_design/10-多模态推理优化.md) - 多模态模型推理优化策略
- [边缘推理优化](reference_design/11-边缘推理优化.md) - 边缘设备上的推理优化方案
- [安全性与合规性](reference_design/09-安全性与合规性.md) - 推理服务的安全与合规要求

**实施落地与运维**：

- [实施建议与最佳实践](reference_design/07-实施建议与最佳实践.md) - 落地实施的指导建议
- [实施检查清单](reference_design/13-实施检查清单.md) - 推理系统上线检查清单
- [场景问题解答](reference_design/12-场景问题解答.md) - 常见问题与解决方案
- [参考资料与延伸阅读](reference_design/08-参考资料与延伸阅读.md) - 推荐阅读与延伸资料
- [总结与展望](reference_design/14-总结与展望.md) - 推理优化技术发展趋势

## 5. 模型部署与运维实践

本节提供将模型转化为可用服务的部署方案与运维经验，涵盖不同硬件平台与框架的实战部署。

- [DeepSeek-V3 MoE 模型 vLLM 部署](inference_solutions/deepseek_v3_moe_vllm_h20_deployment.md) - H20 硬件上的部署方案与 SLO 验证
- [Qwen2-VL-7B 华为昇腾部署](inference_solutions/qwen2_vl_7b_huawei.md) - 国产硬件平台的部署优化

## 6. DeepSeek 专题

本节聚焦于 DeepSeek 模型的前沿推理优化与硬件适配实践，深度剖析其专有的并行架构设计（如 WideEP），以及在以 Blackwell 为代表的下一代高性能计算平台上的扩展性与部署策略。

- [vLLM WideEP 架构](vllm/hardware_optimization/vllm_deepseek_blackwell_wide_ep.md) - vLLM 宽端点 (Wide Endpoint) 架构解析
- [Scaling DeepSeek on Blackwell](vllm/hardware_optimization/scaling_deepseek_blackwell.pptx) - DeepSeek 在 Blackwell 平台上的扩展性优化
