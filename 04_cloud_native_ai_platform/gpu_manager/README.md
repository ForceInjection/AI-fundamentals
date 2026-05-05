# GPU 管理技术深度解析

GPU 在 AI 基础设施中几乎是最昂贵的硬件，但它的设计初衷却是"独占使用"。把这种硬件放到多人共用、任务混跑的 AI 平台里，就会冒出一串很具体的问题：一张 80GB 显存的 H100 怎么让十个小推理任务共享？训练任务卡在一台机器上、别的机器 GPU 闲着怎么办？容器里怎么安全地看到 GPU 却又不越权？这一章就是围绕这些问题展开的。

---

## 1. 概述

要把 GPU 纳管起来，通常要同时解决四件事，它们环环相扣：

- **GPU 虚拟化**：把一张物理卡抽象成多张"逻辑卡"。这一步有三条路径——硬件级（SR-IOV、MIG）靠芯片自身能力做隔离，内核级通过驱动层做拦截转发，用户态则在 API 层（如 CUDA API 劫持）做虚拟化。三条路径的隔离强度、性能损耗和部署门槛都不一样。
- **GPU 切分**：虚拟化之后真正要做的是"按需分配"。时间片切分让多个任务轮流占用，物理分割（如 MIG）切出独立的算力与显存分区，而混合切分在时间与空间两个维度上同时切，换取更高的利用率。
- **远程 GPU 调用**：当本机没有 GPU 或本机 GPU 被占满，如何把算力"借"到其他节点？这就需要一层网络协议把 CUDA 调用转发到远端 GPU 上执行，并尽量压低延迟和序列化开销。
- **容器化与 Kubernetes 编排**：前面三件事解决了"一张卡怎么被多个任务共用"，这一层要解决的是"整个集群的 GPU 怎么被统一调度"。NVIDIA Container Toolkit、CDI 规范和 OCI 运行时负责把 GPU 安全地注入容器；Kubernetes Device Plugin 框架、MIG 集成和智能调度器则把这些容器摆到正确的节点、正确的卡上，并配上健康监控与故障恢复。

四件事合起来，才构成一个可以在生产环境稳定服役的 GPU 资源池。

---

## 2. GPU 管理技术

下面按“理论 → 实现 → 配置”三条线组织材料：先把概念和机制讲清楚，再用可运行的代码演示关键路径，最后给出可以直接套用的部署配置。

### 2.1 GPU 管理技术文档

这一部分是整个话题的主干，按从浅到深的顺序分为四篇——先理清传统模式为什么不够用，再分别深入虚拟化、资源切分与生产落地。

- [第一部分：基础理论篇](第一部分：基础理论篇.md)：构建技术认知框架，解析传统模式局限性与核心技术体系。
- [第二部分：虚拟化技术篇](第二部分：虚拟化技术篇.md)：深入剖析硬件级、内核态与用户态虚拟化的核心实现机制。
- [第三部分：资源管理与优化篇](第三部分：资源管理与优化篇.md)：探讨 GPU 切分、CUDA 流及 MPS 等高效资源调度策略。
- [第四部分：实践应用篇](第四部分：实践应用篇.md)：涵盖环境部署、监控运维及云平台集成的生产落地指南。

### 2.2 代码实现

光看文档很容易停留在概念层面，这里配套了一组可读、可编译的 demo 代码，把前面讲到的机制落到具体的数据结构和系统调用上——重点不是做一个完整产品，而是让关键路径可以被追踪和调试。

#### 2.2.1 [Demo 代码](code/)

按模块分组的 C/C++ 与 Go 参考实现，覆盖内存、调度、虚拟化、远程调用和 Kubernetes 集成等核心链路：

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

从单机容器配置到多云 GPU 集群模板，这部分给出可以直接套用的 YAML 与脚本，省去“从零写一份生产配置”的心智负担：

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

如果说前面讲的是“GPU 共享可以怎么做”，HAMi 就是一个把这些想法真正跑在 Kubernetes 集群里、并且已经在开源社区长期演进的参考实现。它在云原生环境中提供细粒度的 GPU 共享能力，这里专门拿出来讲它的核心机制、可观测性指标以及与其他调度方案的横向对比。

- [HAMi 资源管理使用手册](hami/hmai-gpu-resources-guide.md)：异构算力管理与隔离实战指南。
- [HAMi Prometheus 监控指标](hami/hami-prometheus-metrics.md)：构建完善的 GPU 虚拟化可观测性体系。
- [KAI vs HAMi 对比分析](hami/KAI_vs_HAMi_Comparison.md)：深度对比原生 Kubernetes AI 调度器与 HAMi 方案。
- [Flex AI 介绍](hami/flex_ai_intro.md)：探讨灵活异构算力环境下的前沿实践。

---

## 4. 相关资源

GPU 管理不是一个孤立的话题——往下会触到硬件和运维，往上会接到 Kubernetes 调度和推理框架。下面这些链接可以作为继续深挖的入口。

### 4.1 核心技术资源

支撑 GPU 高效运行的基础配套：集群通信、推理系统、底层运维，每一项都是 GPU 真正跑起来之后绕不开的环节。

- [NCCL 通信优化](../../03_ai_cluster_ops/03_nccl/README.md)
- [AI 推理优化](../../09_inference_system/README.md)
- [GPU 基础运维](../../03_ai_cluster_ops/01_gpu_ops/README.md)

### 4.2 容器化与编排

把视角拉回到平台侧：GPU 如何被容器和 Kubernetes 纳管，以及它底下的硬件长什么样。

- [Kubernetes GPU 管理](../k8s/README.md)
- [CUDA 编程基础](../../02_gpu_programming/02_cuda/README.md)
- [GPU 架构原理](../../01_hardware_architecture/README.md)

### 4.3 官方文档

遇到细节争议时，以 NVIDIA 与 CNCF 这几份权威规范为准，避免被二手资料误导。

- [NVIDIA Container Toolkit 官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Kubernetes Device Plugin 规范](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
- [Container Device Interface (CDI) 规范](https://github.com/cncf-tags/container-device-interface)
- [NVIDIA K8s Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)
