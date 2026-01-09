# 云原生 AI 平台

## 1. 概述

本目录聚焦于 **云原生 AI 平台 (Cloud Native AI Platform)** 的构建与实践，旨在探讨如何利用 Kubernetes、容器化、微服务等云原生技术栈，构建高效、可扩展、高可用的 AI 基础设施。

随着 AI 模型规模的不断扩大（如 LLM 大语言模型）和计算需求的爆发式增长，传统的静态集群管理方式已难以满足资源调度、弹性伸缩和故障恢复的需求。云原生 AI 平台通过引入声明式 API、自动化控制器和不可变基础设施等理念，为 AI 工作负载提供了标准化的运行环境和智能化的管理能力。

## 2. 目录结构与核心模块

本章节包含以下三个核心模块，分别对应计算编排、资源管理和存储加速三大支柱：

### 2.1 Kubernetes AI 基础设施 (`k8s`)

Kubernetes 是云原生 AI 平台的操作系统。本模块深入解析 Kubernetes 在 AI 场景下的核心组件与扩展机制，涵盖从底层的容器运行时支持到上层的分布式作业调度。

- **核心内容**：
  - [**Kubernetes GPU 管理与 AI 工作负载**](k8s/README.md)：模块总览与技术导图。
  - [**NVIDIA Container Toolkit 原理**](k8s/Nvidia%20Container%20Toolkit%20原理分析.md)：容器使用 GPU 的底层机制。
  - [**Device Plugin 原理**](k8s/nvidia-k8s-device-plugin-analysis.md)：Kubernetes 设备插件机制源码分析。
  - [**Kueue + HAMi 调度方案**](k8s/Kueue%20+%20HAMi.md)：云原生作业队列与细粒度 GPU 共享。
  - [**分布式推理框架**](k8s/llm-d-intro.md)：基于 Kubernetes 的 LLM 推理架构设计。

### 2.2 GPU 资源管理与虚拟化 (`gpu_manager`)

GPU 是 AI 平台最昂贵的计算资源。本模块专注于 GPU 资源的精细化管理，包括虚拟化、切分、远程调用和池化技术，旨在最大化资源利用率。

- **核心内容**：
  - [**GPU 管理技术深度解析**](gpu_manager/README.md)：虚拟化、切分与远程调用技术全景。
  - [**HAMi 资源管理指南**](gpu_manager/hami/hmai-gpu-resources-guide.md)：异构算力管理与隔离实战。
  - [**虚拟化与切分原理**](gpu_manager/GPU虚拟化与切分技术原理解析.md)：时间片轮转、显存隔离等技术细节。
  - [**代码实现**](gpu_manager/code/)：GPU 调度器、内存管理模块的参考实现。

### 2.3 高性能分布式存储 (`storage`)

数据是 AI 的燃料。本模块介绍如何利用 JuiceFS、DeepSeek 3FS 等云原生分布式文件系统，解决 AI 训练中海量小文件读取、模型检查点保存和跨节点数据共享的性能瓶颈。

- **核心内容**：
  - [**JuiceFS 分布式文件系统**](storage/juicefs/README.md)：架构设计与核心特性。
  - [**文件修改机制分析**](storage/juicefs/JuiceFS%20文件修改机制分析.md)：底层数据一致性与写入流程。
  - [**后端存储变更手册**](storage/juicefs/JuiceFS%20后端存储变更手册.md)：生产环境下的存储运维指南。
  - [**DeepSeek 3FS 设计笔记**](storage/deepseek_3fs_design_notes.zh-CN.md)：高性能存储系统设计分析。

## 3. 核心技术架构

云原生 AI 平台的技术架构通常包含以下几个层次：

1. **基础设施层 (Infrastructure)**：

   - 异构计算资源（GPU, NPU, TPU）
   - 高速互联网络（InfiniBand, RoCE）
   - 分布式存储（对象存储, 并行文件系统）

2. **容器编排层 (Orchestration)**：

   - **Kubernetes**：核心调度引擎。
   - **Device Plugins**：硬件设备发现与上报。
   - **Operators**：AI 任务全生命周期管理（如 MPIOperator, PyTorchJob）。

3. **资源调度与管理层 (Scheduling & Management)**：

   - **Volcano / Kueue**：批处理作业调度，支持 Gang Scheduling、公平调度。
   - **HAMi / vGPU**：GPU 细粒度切分与显存隔离，支持多任务共享 GPU。

4. **AI 平台服务层 (Platform Services)**：
   - **模型开发**：Jupyter Notebooks, VS Code Server。
   - **模型训练**：分布式训练框架适配。
   - **模型推理**：Model Serving, Auto-Scaling (KEDA)。
   - **MLOps**：流水线管理 (Kubeflow, Argo)。
