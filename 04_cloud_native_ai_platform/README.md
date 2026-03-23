# 云原生 AI 平台

本目录聚焦于 **云原生 AI 平台 (Cloud Native AI Platform)** 的构建与实践，旨在探讨如何利用 Kubernetes、容器化、微服务等云原生技术栈，构建高效、可扩展、高可用的 AI 基础设施。

---

## 1. 概述

随着 AI 模型规模的不断扩大（如 LLM 大语言模型）和计算需求的爆发式增长，传统的静态集群管理方式已难以满足资源调度、弹性伸缩和故障恢复的需求。云原生 AI 平台通过引入声明式 API、自动化控制器和不可变基础设施等理念，为 AI 工作负载提供了标准化的运行环境和智能化的管理能力。

云原生 AI 平台的技术架构通常包含以下几个层次：

1. **基础设施层 (Infrastructure)**：
   这是整个平台的物理和硬件基础，提供 AI 计算所必需的底层算力、数据吞吐与网络通信能力。本层的主要构成包括：
   - 异构计算资源（GPU, NPU, TPU）
   - 高速互联网络（InfiniBand, RoCE）
   - 分布式存储（对象存储, 并行文件系统）

2. **容器编排层 (Orchestration)**：
   该层将底层异构的基础设施抽象为统一的资源池，通过标准化的接口管理硬件设备，并提供应用运行的隔离环境。其核心组件包含：
   - **Kubernetes**：核心调度引擎。
   - **Device Plugins**：硬件设备发现与上报。
   - **Operators**：AI 任务全生命周期管理（如 MPIOperator, PyTorchJob）。

3. **资源调度与管理层 (Scheduling & Management)**：
   针对 AI 负载的特殊性（如大规模并行、昂贵的显存），该层负责优化计算任务的排队、分配与资源切分，确保集群利用率最大化。典型解决方案有：
   - **Volcano / Kueue**：批处理作业调度，支持 Gang Scheduling、公平调度。
   - **HAMi / vGPU**：GPU 细粒度切分与显存隔离，支持多任务共享 GPU。

4. **AI 平台服务层 (Platform Services)**：
   位于最上层，直接面向 AI 开发者和数据科学家，提供从模型开发、训练到推理部署的端到端工具链和业务服务。常见的平台服务模块包括：
   - **模型开发**：Jupyter Notebooks, VS Code Server。
   - **模型训练**：分布式训练框架适配。
   - **模型推理**：Model Serving, Auto-Scaling。
   - **MLOps**：流水线管理 (Kubeflow, Argo)。

---

## 2. 目录结构与核心模块

本章节包含以下三个核心模块，分别对应计算编排、资源管理和存储加速三大支柱。

### 2.1 Kubernetes AI 基础设施

Kubernetes 是云原生 AI 平台的操作系统。本模块深入解析 Kubernetes 在 AI 场景下的核心组件与扩展机制，涵盖从底层的容器运行时支持到上层的分布式作业调度。

- [Kubernetes GPU 管理与 AI 工作负载](k8s/README.md)：云原生 AI 基础设施建设指南与技术导图。
- [NVIDIA Container Toolkit 原理](k8s/01_nvidia_container_toolkit_analysis.md)：容器使用 GPU 的底层机制深度解析。
- [Device Plugin 原理](k8s/02_nvidia_k8s_device_plugin_analysis.md)：Kubernetes 设备插件机制源码分析。
- [Kueue + HAMi 调度方案](k8s/03_kueue_hami_integration.md)：云原生作业队列与细粒度 GPU 共享机制。
- [LWS (Leader Worker Set) 介绍](k8s/04_lws_intro.md)：Kubernetes 原生的大模型分布式训练与推理调度抽象。
- [分布式推理框架](k8s/05_llm_d_intro.md)：基于 Kubernetes 的 LLM 推理架构设计。
- [Containerd 日志分析](k8s/06_containerd_log_analysis.md)：云原生容器运行时的日志排查与分析。

### 2.2 GPU 资源管理与虚拟化

GPU 是 AI 平台最昂贵的计算资源。本模块专注于 GPU 资源的精细化管理，包括虚拟化、切分、远程调用和池化技术，旨在最大化资源利用率。

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

数据是 AI 的燃料。本模块介绍如何利用 JuiceFS、DeepSeek 3FS 等云原生分布式文件系统，解决 AI 训练中海量小文件读取、模型检查点保存和跨节点数据共享的性能瓶颈。

- [JuiceFS 分布式文件系统](storage/juicefs/README.md)：数据与元数据分离的架构设计，兼容 POSIX 接口。
- [文件修改机制分析](storage/juicefs/01_juicefs_file_modification_mechanism_analysis.md)：底层数据一致性与写入流程解析。
- [后端存储变更手册](storage/juicefs/02_juicefs_backend_storage_migration_guide.md)：生产环境下的存储运维与数据迁移指南。
- [DeepSeek 3FS 设计笔记](storage/deepseek_3fs/01_deepseek_3fs_design_notes.md)：高性能存储系统架构设计与特性分析。
- [NVIDIA ICMS 架构解析](storage/inference_context_memory_storage/01_icms_architecture.md)：面向推理的 KV Cache 存储层架构深度解析。
