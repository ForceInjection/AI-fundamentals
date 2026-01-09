# Kubernetes GPU 管理与 AI 工作负载技术文档集

本目录包含了关于 Kubernetes 环境下 GPU 管理、AI 工作负载调度和分布式推理的深度技术文档。这些文档涵盖了从底层硬件抽象到上层应用部署的完整技术栈，为在 Kubernetes 环境中构建高效的 AI 基础设施提供了全面的技术指导。

## 1. 核心基础设施组件

### 1.1 [NVIDIA Container Toolkit 原理分析](./Nvidia%20Container%20Toolkit%20原理分析.md)

深入解析 NVIDIA Container Toolkit 的核心原理和实现机制，包括：

- **容器化 GPU 计算的技术挑战**：设备隔离、驱动依赖、资源管理等核心问题
- **OCI 运行时集成机制**：与 Docker、containerd、CRI-O 等容器运行时的深度集成
- **CDI (Container Device Interface) 规范实现**：标准化的设备接口规范
- **源码级别的架构分析**：核心组件的实现原理和代码解析
- **性能优化策略**：最小化容器化开销的技术手段

### 1.2 [Nvidia K8s Device Plugin 原理解析和源码分析](./nvidia-k8s-device-plugin-analysis.md)

全面分析 NVIDIA Kubernetes Device Plugin 的实现原理，涵盖：

- **Kubernetes Device Plugin 框架规范**：API 接口定义和通信协议
- **设备发现与注册机制**：GPU 设备的自动发现和向 kubelet 注册的流程
- **资源分配与调度策略**：GPU 资源的分配算法和调度优化
- **健康检查与故障恢复**：设备健康监控和异常处理机制
- **源码深度解析**：关键组件的实现细节和代码分析

## 2. 高级调度与资源管理

### 2.1 [Kueue + HAMi：Kubernetes 原生的 AI 工作负载管理与 GPU 虚拟化解决方案](./Kueue%20+%20HAMi.md)

详细介绍 Kueue 作业队列系统与 HAMi GPU 虚拟化技术的集成方案：

- **Kueue 核心概念与架构**：
  - ClusterQueue、LocalQueue、Workload 等核心资源对象
  - 多租户资源配额管理和层次化资源共享
  - 与 kube-scheduler 和 cluster-autoscaler 的协同工作

- **HAMi GPU 虚拟化技术**：
  - GPU 显存和计算核心的细粒度切分
  - vGPU 实例的创建和管理
  - 多容器间的 GPU 资源隔离

- **生产级部署实践**：
  - 完整的配置示例和部署脚本
  - 性能调优和故障排查指南
  - 监控和可观测性最佳实践

## 3. 分布式推理框架

### 3.1 [vLLM + LWS：Kubernetes 上的多机多卡推理方案](./lws_intro.md)

深入探讨 vLLM 推理引擎与 LeaderWorkerSet (LWS) 控制器的集成方案：

- **LeaderWorkerSet (LWS) 控制器原理**：
  - 分布式角色结构：Leader-Worker 模式
  - 统一生命周期管理和拓扑感知调度
  - 与传统 Kubernetes 控制器的差异化优势

- **vLLM 分布式推理架构**：
  - 多节点 GPU 资源协同
  - 分布式 KV Cache 管理
  - gRPC 通信和参数广播机制

- **实战部署配置**：
  - 完整的 YAML 配置示例
  - 拓扑感知调度配置
  - 弹性伸缩和故障恢复策略

### 3.2 [云原生高性能分布式 LLM 推理框架 llm-d 介绍](./llm-d-intro.md)

全面介绍 llm-d 分布式推理框架的技术架构和核心优势：

- **大规模 LLM 推理挑战分析**：
  - 技术复杂性：多层次优化需求、分布式推理复杂性
  - 运营成本：硬件成本、资源利用率、运维复杂度
  - 性能与扩展性：延迟敏感性、吞吐量瓶颈、弹性扩展

- **llm-d 核心技术优势**：
  - Kubernetes 原生设计和云原生架构
  - 多硬件加速器支持和竞争性性价比
  - 智能调度和资源优化算法

- **架构设计与组件详解**：
  - 分层架构设计和核心组件分析
  - 技术特性解析和性能优化策略
  - 生产环境部署和运维最佳实践
