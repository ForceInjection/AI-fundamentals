# 云原生 AI 平台

## 1. 概述

从单机跟得上 AI 工作负载到一个让上百个任务排队、弹性扩缩、多人共用的集群，中间缺的那一层就是 **云原生 AI 平台**。这一章要回答的核心问题是：**当 Kubernetes 这套调度体系撞上 GPU 这种异构、昂贵、具备独占性的硬件时，到底要解决哪些问题？**

LLM 时代让这个问题变得更尖锐：模型可能一次要吃掉一整个机柜的 GPU，推理流量可能在几秒钟内翻十倍，而传统静态集群管理方式在资源调度、弹性伸缩和故障恢复上都已经跟不上了。云原生 AI 平台通过声明式 API、自动化控制器和不可变基础设施，为 AI 工作负载提供一个标准化的运行环境和智能化的管理能力。

从架构上看，整个平台一般分四层来构建，每一层解决的问题都不一样：

1. **基础设施层 (Infrastructure)**：物理算力、网络和存储的底座。GPU/NPU/TPU 提供异构算力，InfiniBand/RoCE 支撑高速互联，分布式存储负责数据吞吐。
2. **容器编排层 (Orchestration)**：把异构硬件抽象成统一的资源池。核心是 Kubernetes，配合 Device Plugin 完成设备上报，Operators（如 MPIOperator、PyTorchJob）接管任务的全生命周期。
3. **资源调度与管理层 (Scheduling & Management)**：针对 AI 负载的“大并行 + 贵显存”特点做优化。Volcano / Kueue 处理批作业调度和 Gang Scheduling；HAMi / vGPU 做 GPU 细粒度切分与显存隔离。
4. **AI 平台服务层 (Platform Services)**：最靠近开发者的一层，包括 Notebook 开发环境、分布式训练框架适配、模型推理服务，以及 Kubeflow / Argo 这类 MLOps 流水线。

四层合起来，才能让一切能跑、跑得稳、跑得快。

---

## 2. 目录结构与核心模块

整套内容围绕三个支柱展开：**计算编排**、**资源管理**、**存储加速**。前者解决“GPU 如何被 Kubernetes 纳管”，中间解决“一张卡如何被多个任务安全共用”，后者解决“海量数据和模型检查点如何不成为短板”。

### 2.1 Kubernetes AI 基础设施

Kubernetes 是云原生 AI 平台的操作系统。这一块解析 K8s 在 AI 场景下的核心组件与扩展机制，从底层的容器运行时 GPU 支持，一路讲到上层的分布式作业调度。

- [Kubernetes GPU 管理与 AI 工作负载](k8s/README.md)：云原生 AI 基础设施建设指南与技术导图。
- [NVIDIA Container Toolkit 原理](k8s/01_nvidia_container_toolkit_analysis.md)：容器使用 GPU 的底层机制深度解析。
- [Device Plugin 原理](k8s/02_nvidia_k8s_device_plugin_analysis.md)：Kubernetes 设备插件机制源码分析。
- [Kueue + HAMi 调度方案](k8s/03_kueue_hami_integration.md)：云原生作业队列与细粒度 GPU 共享机制。
- [LWS (Leader Worker Set) 介绍](k8s/04_lws_intro.md)：Kubernetes 原生的大模型分布式训练与推理调度抽象。
- [分布式推理框架](k8s/05_llm_d_intro.md)：基于 Kubernetes 的 LLM 推理架构设计。
- [Containerd 日志分析](k8s/06_containerd_log_analysis.md)：云原生容器运行时的日志排查与分析。

### 2.2 GPU 资源管理与虚拟化

GPU 是整个平台上最昂贵的资源，“如何让它不闲着、不浪费、并且能安全地被多个任务共享”是这一模块的主线。相关内容涵盖虚拟化、切分、远程调用以及池化技术。

**基础系列文档**：

- [第一部分：基础理论篇](gpu_manager/第一部分：基础理论篇.md)：构建技术认知框架，解析传统模式局限性与核心技术体系。
- [第二部分：虚拟化技术篇](gpu_manager/第二部分：虚拟化技术篇.md)：深入剖析硬件级、内核态与用户态虚拟化的核心实现机制。
- [第三部分：资源管理与优化篇](gpu_manager/第三部分：资源管理与优化篇.md)：探讨 GPU 切分、CUDA 流及 MPS 等高效资源调度策略。
- [第四部分：实践应用篇](gpu_manager/第四部分：实践应用篇.md)：涵盖环境部署、监控运维及云平台集成的生产落地指南。

**HAMi 专题**：

- [HAMi 资源管理使用手册](gpu_manager/hami/hmai-gpu-resources-guide.md)：异构算力管理与隔离实战指南。
- [HAMi Prometheus 监控指标](gpu_manager/hami/hami-prometheus-metrics.md)：构建完善的 GPU 虚拟化可观测性体系。
- [KAI vs HAMi 对比分析](gpu_manager/hami/KAI_vs_HAMi_Comparison.md)：深度对比原生 Kubernetes AI 调度器与 HAMi 方案。
- [Flex AI 介绍](gpu_manager/hami/flex_ai_intro.md)：探讨灵活异构算力环境下的前沿实践。

**代码实现与配置**：

- [完整实现代码](gpu_manager/code/)：GPU 调度器、虚拟化拦截与远程调用的参考实现代码。
- [配置文件集合](gpu_manager/configs/)：提供适用于生产环境和多云平台的完整部署与配置参考。

### 2.3 高性能分布式存储

训练一轮次要读上亿张小图片、检查点动辄几十上百 GB、推理要持续命中 KV Cache——这些场景把存储系统推到了一个很尴尬的位置：既要像本地盘一样快，又要像对象存储一样能扩。JuiceFS、DeepSeek 3FS 这类云原生分布式文件系统正是冲着这个缝隙作答的。

- [JuiceFS 分布式文件系统](storage/juicefs/README.md)：数据与元数据分离的架构设计，兼容 POSIX 接口。
- [文件修改机制分析](storage/juicefs/01_juicefs_file_modification_mechanism_analysis.md)：底层数据一致性与写入流程解析。
- [后端存储变更手册](storage/juicefs/02_juicefs_backend_storage_migration_guide.md)：生产环境下的存储运维与数据迁移指南。
- [DeepSeek 3FS 设计笔记](storage/deepseek_3fs/01_deepseek_3fs_design_notes.md)：高性能存储系统架构设计与特性分析。
- [NVIDIA ICMS 架构解析](storage/inference_context_memory_storage/01_icms_architecture.md)：面向推理的 KV Cache 存储层架构深度解析。
