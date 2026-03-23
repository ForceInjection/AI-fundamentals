# GPU 管理技术深度解析

本目录包含 GPU 虚拟化、切分、远程调用等核心技术的深度解析文档和实现代码，旨在帮助读者全面理解 GPU 资源管理的底层机制与实践方案。

---

## 1. 概述

在现代 AI 计算基础设施中，GPU 管理涉及多个核心技术维度的协同。**GPU 虚拟化技术**通过硬件级（如 SR-IOV、MIG）、内核级（如拦截与转发）以及用户空间（如 API 层虚拟化）等多种手段，将物理 GPU 抽象为多个独立的虚拟资源；在此基础上，**GPU 切分技术**利用时间片轮转、物理资源分割以及时空结合的混合策略，实现了计算资源的高效复用。

为了打破单机物理边界，**远程 GPU 调用**利用高效的网络协议与优化的数据传输机制，实现了低延迟的跨节点资源访问。而在云原生环境中，**容器化 GPU 管理**依托 NVIDIA Container Toolkit、CDI 规范及 OCI 运行时，实现了多租户环境下的设备隔离与安全；配合 **Kubernetes GPU 编排**（如 Device Plugin 框架、MIG 集成与智能调度），最终构建出一个具备健康监控、故障恢复且高可用的全局 GPU 资源池。

---

## 2. GPU 管理技术

本章将详细解析 GPU 管理的核心技术栈，涵盖从基础理论到高级优化的完整文档体系，并提供配套的底层代码实现与集群配置文件。

### 2.1 GPU 管理技术文档

我们梳理了 GPU 管理技术的核心理论文档，分为基础理论、虚拟化技术、资源优化及实践应用四个递进部分。

- [第一部分：基础理论篇](第一部分：基础理论篇.md)：构建技术认知框架，解析传统模式局限性与核心技术体系。
- [第二部分：虚拟化技术篇](第二部分：虚拟化技术篇.md)：深入剖析硬件级、内核态与用户态虚拟化的核心实现机制。
- [第三部分：资源管理与优化篇](第三部分：资源管理与优化篇.md)：探讨 GPU 切分、CUDA 流及 MPS 等高效资源调度策略。
- [第四部分：实践应用篇](第四部分：实践应用篇.md)：涵盖环境部署、监控运维及云平台集成的生产落地指南。

### 2.2 代码实现

GPU 管理模块的 demo 代码，用于展示如何在实际场景中实现虚拟化、切分、远程调用等技术。

#### 2.2.1 [Demo 代码](code/)

提供 GPU 资源管理各个核心技术栈的 C/C++ 及 Go 语言参考实现，展示底层控制机制：

- **内存管理** (`code/memory/`)：
  - [统一内存管理](code/memory/unified_memory_manager.c) - 统一虚拟地址空间 (UVM) 实现。
  - [内存热迁移](code/memory/memory_hot_migration.c) - 跨 NUMA 节点的显存迁移逻辑。
  - [内存超分与压缩](code/memory/memory_overcommit_advanced.c) - GPU 显存超额订阅机制。
- **调度系统** (`code/scheduling/`)：
  - [GPU 调度器](code/scheduling/gpu_scheduler.c) - 基于时间片与优先级的资源调度算法。
- **虚拟化拦截** (`code/virtualization/`)：
  - [CUDA API 拦截](code/virtualization/cuda_api_intercept.c) - 用户态 API 劫持实现。
- **远程调用** (`code/remote/`)：
  - [远程 GPU 协议](code/remote/remote_gpu_protocol.c) - 跨网络 GPU 调用的底层通信协议。
- **HAMi 集成** (`code/hami/`)：
  - [MIG 设备插件](code/hami/mig_device_plugin.go) - Kubernetes Kubelet 设备上报与注册。

#### 2.2.2 [配置文件](configs/)

提供适用于生产环境和多云平台的完整部署与配置参考：

- **容器与编排**：
  - [Docker GPU 配置](configs/docker/docker-gpu-config.yaml) - 容器运行时参数调优。
  - [Kubernetes Pod 模板](configs/kubernetes/gpu-pod-templates.yaml) - AI 负载的标准 YAML 定义。
  - [HAMi 部署配置](configs/hami/hami-deployment.yaml) - 集群级资源隔离与切分配置。
- **监控与网络**：
  - [Prometheus/Grafana 监控](configs/monitoring/grafana-gpu-dashboard.json) - 包含显存利用率、温度、功率等指标。
  - [网络优化](configs/network/network-optimization.yaml) - InfiniBand 与 RoCE 性能调优参数。
- **运维脚本** (`configs/scripts/`)：
  - 提供从环境初始化、基准测试到故障排查的全套自动化 Shell 脚本（如 `gpu-performance-test.sh`）。

---

## 3. HAMi 专题

HAMi (Heterogeneous AI Computing Middleware) 是一款开源的异构 AI 计算中间件，作为 GPU 虚拟化与资源切分的优秀参考实现，它在云原生环境中提供了细粒度的 GPU 资源共享能力。本专题深入探讨了 HAMi 的核心机制、监控指标以及与其他调度方案的对比。

- [HAMi 资源管理使用手册](hami/hmai-gpu-resources-guide.md)：异构算力管理与隔离实战指南。
- [HAMi Prometheus 监控指标](hami/hami-prometheus-metrics.md)：构建完善的 GPU 虚拟化可观测性体系。
- [KAI vs HAMi 对比分析](hami/KAI_vs_HAMi_Comparison.md)：深度对比原生 Kubernetes AI 调度器与 HAMi 方案。
- [Flex AI 介绍](hami/flex_ai_intro.md)：探讨灵活异构算力环境下的前沿实践。

---

## 4. 相关资源

以下 GPU 管理相关的内外扩展阅读资料，供读者参考学习。

### 4.1 核心技术资源

涵盖了支撑 GPU 高效运行与优化的底层核心技术体系，包括通信、推理以及基础运维。

- [NCCL 通信优化](../../03_ai_cluster_ops/03_nccl/README.md)
- [AI 推理优化](../../09_inference_system/README.md)
- [GPU 基础运维](../../03_ai_cluster_ops/01_gpu_ops/README.md)

### 4.2 容器化与编排

聚焦于如何在云原生集群中纳管与编排底层 GPU 资源，并深入其软硬件架构原理。

- [Kubernetes GPU 管理](../k8s/README.md)
- [CUDA 编程基础](../../02_gpu_programming/02_cuda/README.md)
- [GPU 架构原理](../../01_hardware_architecture/README.md)

### 4.3 官方文档

整理了 NVIDIA 与 CNCF 官方提供的核心规范与标准文档，作为技术实现的权威依据。

- [NVIDIA Container Toolkit 官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Kubernetes Device Plugin 规范](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
- [Container Device Interface (CDI) 规范](https://github.com/cncf-tags/container-device-interface)
- [NVIDIA K8s Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)
