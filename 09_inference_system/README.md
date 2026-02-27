# 推理优化技术方案

## 1. 概述

本文档集介绍了在不同集群规模下进行推理优化的技术方案，涵盖从理论基础到实践应用的完整技术体系。

## 2. 参考设计 (Reference Design)

本章节按照“理论 → 选型与架构 → 场景优化 → 实施运维”的路径组织关键文档，覆盖从概念与指标定义到工程落地的主要决策点。

### 2.1 基础理论

- **[背景与目标](reference_design/01-背景与目标.md)** - 推理优化的技术背景、研究目标和价值主张
- **[集群规模分类与特征分析](reference_design/02-集群规模分类与特征分析.md)** - 小型、中型、大型集群的特征分析和配置要求
- **[核心推理优化技术深度解析](reference_design/03-核心推理优化技术深度解析.md)** - 模型压缩、并行计算、推测解码、连续批处理与性能基准方法论

### 2.2 技术选型与架构

- **[不同集群规模的技术选型策略](reference_design/04-不同集群规模的技术选型策略.md)** - 针对不同规模集群的技术选型指南和决策框架
- **[推理服务架构设计](reference_design/06-推理服务架构设计.md)** - 推理服务的系统架构设计和组件规划
- **[性能评估指标体系](reference_design/05-性能评估指标体系.md)** - 性能指标定义、基准测试方法和评估体系

### 2.3 专业领域优化

- **[多模态推理优化](reference_design/10-多模态推理优化.md)** - 多模态模型的推理架构和跨模态注意力优化
- **[边缘推理优化](reference_design/11-边缘推理优化.md)** - 边缘设备适配、边缘-云协同与端侧部署运维
- **[安全性与合规性](reference_design/09-安全性与合规性.md)** - 推理服务的安全威胁分析、隐私保护和合规要求

### 2.4 实施与运维

- **[实施建议与最佳实践](reference_design/07-实施建议与最佳实践.md)** - 分阶段实施策略、最佳实践和风险管理
- **[实施检查清单](reference_design/13-实施检查清单.md)** - 分阶段实施的详细检查清单和验收标准
- **[常见问题解答 (FAQ)](reference_design/12-场景问题解答.md)** - 常见技术问题和解决方案
- **[参考资料与延伸阅读](reference_design/08-参考资料与延伸阅读.md)** - 相关技术文档、开源项目和学习资源
- **[总结与展望](reference_design/14-总结与展望.md)** - 技术总结和未来发展趋势分析

---

## 3. KV Cache 系统与优化

在大模型推理中，KV Cache 的管理是影响长上下文性能与显存效率的核心瓶颈。本节汇集了主流 KV Cache 系统的架构解析与优化技术。

### 3.1 系统架构解析

- **[LMCache 源码分析指南](kv_cache/lmcache/README.md)** - LMCache 文档入口与推荐阅读路径
  - **[LMCache 架构概览](kv_cache/lmcache/lmcache_overview.md)**
  - **[LMCacheEngine 源码分析](kv_cache/lmcache/lmcache_engine.md)**
  - **[分层存储后端实现](kv_cache/lmcache/lmcache_storage_overview.md)**
- **[Tair KVCache 架构与设计](kv_cache/ali_tair_kvcache/tair-kvcache-architecture-design.md)** - 阿里云高性能 KVCache 系统深度解析
- **[Mooncake 架构详解](kv_cache/mooncake_architecture.md)** - Kimi 背后的分离式推理架构与 KV Cache 全局调度

### 3.2 深度分析与前沿技术

- **[层级流水线并行 (Layerwise Pipelining)](kv_cache/layerwise_pipeline.md)** - 解析 LMCache 和 DualPath 如何通过计算与 I/O 重叠解决存储瓶颈
- **[vLLM KV Offloading 与 LMCache 对比](kv_cache/kv_offloading_analysis.md)** - 深入剖析 vLLM 原生 KV Offloading 与 LMCache 的架构差异与性能权衡

---

## 4. 推理部署方案 (Inference Solutions)

本章汇集了针对主流开源模型（如 DeepSeek、Qwen、Llama 等）的实战部署方案。

- **[DeepSeek-V3 MoE vLLM 部署方案](inference_solutions/deepseek_v3_moe_vllm_h20_deployment.md)** - 基于 H20 硬件的部署指南
  - **[SLO 计算工具](inference_solutions/slo_calc_v2.py)** - 基于腾讯太极团队数据的 SLO 验证脚本
- **[Qwen2-VL-7B 华为平台部署](inference_solutions/qwen2_vl_7b_huawei.md)** - 华为硬件平台的部署优化指南

---

## 5. 基础设施与加速 (Infrastructure & Acceleration)

本章聚焦于分布式推理系统中的底层通信、存储加速与计算引擎优化。

### 5.1 存储与通信基础设施

- **[GDS P2PDMA 故障排查](infrastructure/gds_p2pdma_troubleshooting.md)** - GPUDirect Storage 在特定环境下的 P2PDMA 生效问题分析
- **[GDS 性能测试报告](infrastructure/gds_performance_report.md)** - GPUDirect Storage 性能基准测试
- **[NIXL (NVIDIA Inference Xfer Library)](infrastructure/nixl_introduction.md)** - 专为大规模分布式 AI 推理设计的点对点通信库

### 5.2 vLLM 引擎优化

- **[vLLM WideEP 架构](vllm/wide_ep.md)** - vLLM 宽端点 (Wide Endpoint) 架构解析
- **[vLLM GB200 性能优化报告](vllm/vllm_gb200_optimization.pptx)**
- **[Scaling DeepSeek on Blackwell](vllm/scaling_deepseek_blackwell.pptx)**

---

## 6. 模型优化与显存分析

### 6.1 模型优化

- **[NVIDIA Model Optimizer 技术详解](model_optimization/nvidia_model_optimizer.md)** - NVIDIA 官方模型量化、稀疏化与蒸馏工具库指南

### 6.2 显存分析

- **[LLM 模型推理显存占用深度分析](memory_calc/memory_analysis.md)** - 理论分析模型权重、KV Cache 及激活值的显存构成
- **[显存分析 PPT](memory_calc/llm_memory_analysis.pptx)**
- **[显存计算脚本](memory_calc/calculate_qwen3_memory.py)** - 自动计算指定模型配置下的显存占用
- **[模型配置示例](memory_calc/qwen3_06b_config.json)**
