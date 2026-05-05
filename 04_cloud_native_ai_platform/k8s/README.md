# Kubernetes GPU 管理与 AI 工作负载

Kubernetes 最初是为无状态 Web 服务设计的，它对“一个 Pod 就是一个任务”这种模型很熟练。但 AI 负载不是。一个训练任务可能要几十张卡同时起走、不允许部分就绪，一个推理服务可能要在 Leader-Worker 形态下跨节点协作，而 GPU 这种资源既不像 CPU 那样可以随意超分，也不像内存那样有成熟的 Swap 机制。这一章的内容就围绕一个问题展开：**要让 Kubernetes 真正跑好 AI 负载，从容器运行时到上层调度器，究竟要补哪几块？**

## 1. 核心基础设施组件

要让容器内的进程看到 GPU、让 kubelet 知道一个节点有几张卡可用、让 Pod 挂掉时可以被快速诊断——这三件事背后并不是 Kubernetes 的原生能力，而是一套由 NVIDIA Container Toolkit、Device Plugin 和 containerd 日志组成的“暗管道”。这一节拆解这些基础组件，搞清楚每一层在干什么、出了问题去哪里看。

### 1.1 [NVIDIA Container Toolkit 原理分析](./01_nvidia_container_toolkit_analysis.md)

深入解析 NVIDIA Container Toolkit 的核心原理和实现机制，包括：

- **容器化 GPU 计算的技术挑战**：设备隔离、驱动依赖、资源管理等核心问题
- **OCI 运行时集成机制**：与 Docker、containerd、CRI-O 等容器运行时的深度集成
- **CDI (Container Device Interface) 规范实现**：标准化的设备接口规范
- **源码级别的架构分析**：核心组件的实现原理和代码解析
- **性能优化策略**：最小化容器化开销的技术手段

### 1.2 [Nvidia K8s Device Plugin 原理解析和源码分析](./02_nvidia_k8s_device_plugin_analysis.md)

全面分析 NVIDIA Kubernetes Device Plugin 的实现原理，涵盖：

- **Kubernetes Device Plugin 框架规范**：API 接口定义和通信协议
- **设备发现与注册机制**：GPU 设备的自动发现和向 kubelet 注册的流程
- **资源分配与调度策略**：GPU 资源的分配算法和调度优化
- **健康检查与故障恢复**：设备健康监控和异常处理机制
- **源码深度解析**：关键组件的实现细节和代码分析

### 1.3 [容易被忽略的 containerd 运行时日志](./06_containerd_log_analysis.md)

分析容器运行时日志，深入了解 `containerd` 的异常排查方法：

- **runc 运行时日志位置与作用**：探索 `/run/containerd` 目录下的核心日志
- **容器启动失败的深度定位**：绕过常规标准输出日志寻找关键线索
- **底层异常问题诊断**：提升排查 OCI 运行时相关故障的效率

## 2. 高级调度与资源管理

基础件就位之后，下一个绕不开的问题是：集群里的 GPU 是有限的，任务却是排队来的，谁先跑、谁后跑、一张卡能不能切开给几个人用——这就需要比 kube-scheduler 更懂 AI 负载的队列系统和 GPU 切分机制。

### 2.1 [Kueue + HAMi：Kubernetes 原生的 AI 工作负载管理与 GPU 虚拟化解决方案](./03_kueue_hami_integration.md)

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

训练端的“一份任务、多卡并行”到推理端会变成“一份服务、跨节点展开”：Leader 负责调度和请求编排，Worker 负责真正跑模型，而 KV Cache、参数广播都要跨节点协同。这一节的两篇文章正是围绕这个新范式。

### 3.1 [vLLM + LWS：Kubernetes 上的多机多卡推理方案](./04_lws_intro.md)

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

### 3.2 [云原生高性能分布式 LLM 推理框架 llm-d 介绍](./05_llm_d_intro.md)

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
