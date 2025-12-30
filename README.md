# AI Fundamentals

本仓库是一个全面的人工智能基础设施（`AI Infrastructure`）学习资源集合，涵盖从硬件基础到高级应用的完整技术栈。内容包括 GPU 架构与编程、`CUDA` 开发、大语言模型、`AI` 系统设计、性能优化、企业级部署等核心领域，旨在为 `AI` 工程师、研究人员和技术爱好者提供系统性的学习路径和实践指导。

> **适用人群**：`AI` 工程师、系统架构师、`GPU` 编程开发者、大模型应用开发者、技术研究人员。
> **技术栈**：`CUDA`、`GPU` 架构、`LLM`、`AI` 系统、分布式计算、容器化部署、性能优化。

---

**Star History**:

## [![Star History Chart](https://api.star-history.com/svg?repos=ForceInjection/AI-fundermentals&type=date&legend=top-left)](https://www.star-history.com/#ForceInjection/AI-fundermentals&date&legend=top-left)

---

## 1. 硬件与基础设施

本章节主要构建 AI 系统的物理底座，深入探讨从单机计算芯片（GPU/TPU）到大规模集群互联（InfiniBand/RDMA）的核心技术。我们将从硬件架构原理出发，逐步延伸至分布式存储、高性能网络以及云原生基础设施的管理与运维，为构建高性能、高可用的 AI 平台打下坚实基础。

### 1.1 硬件基础知识

在构建现代 AI 基础设施时，深入理解底层硬件架构是至关重要的。从 AI Infra 的视角来看，我们需要掌握以下核心硬件知识：

1. **计算硬件**： `GPU` 架构设计、`CUDA` 核心原理、`Tensor Core` 加速单元、内存层次结构（寄存器、共享内存、全局内存）以及并行计算模型，这些直接影响模型训练和推理的性能表现。
2. **互连技术**： `PCIe` 总线、`NVLink` 高速互连、`InfiniBand` 网络架构，这些技术决定了多 `GPU` 系统的扩展能力和通信效率，是构建大规模分布式训练集群的基础。
3. **存储系统**：高性能 `SSD`、分布式文件系统、内存池化技术，用于支撑海量训练数据的高效读取和模型检查点的快速保存。
4. **网络基础设施**：高带宽、低延迟的数据中心网络设计，支持模型参数同步、梯度聚合等分布式计算场景的通信需求。

理解这些硬件特性有助于我们在 `AI` 系统设计中做出最优的架构选择，实现计算资源的高效利用和成本控制。

- [**PCIe 知识大全**](https://mp.weixin.qq.com/s/dHvKYcZoa4rcF90LLyo_0A) - 深入理解 PCIe 总线架构、带宽计算和性能优化
- [**NVLink 入门**](https://mp.weixin.qq.com/s/fP69UEgusOa_X4ZKLo30ig) - NVIDIA 高速互连技术的原理与应用场景
- [**NVIDIA DGX SuperPOD ：下一代可扩展的 AI 领导基础设施**](https://mp.weixin.qq.com/s/a64Qb6DuAAZnCTBy8g1p2Q) - 企业级 AI 超算集群的架构设计与部署实践

### 1.2 GPU 与 AI 加速器架构深度解析

在准备在 `GPU` 上运行的应用程序时，了解 `GPU` 硬件设计的主要特性并了解与 `CPU` 的相似之处和不同之处会很有帮助。本路线图适用于那些对 `GPU` 比较陌生或只是想了解更多有关 `GPU` 中计算机技术的人。不需要特定的并行编程经验，练习基于 `CUDA` 工具包中包含的标准 `NVIDIA` 示例程序。

#### 1.2.1 Google TPU 架构深度解析

- [**TPU 101: 深度学习专用加速器架构解析**](tpu/tpu%20101.md) - 深入剖析 Google TPU 的设计哲学、脉动阵列架构及其与 GPU 的本质区别

**核心内容：**

- **设计哲学**：为何不使用 GPU？TPU 的专用性（Domain Specific Architecture）权衡
- **核心架构**：脉动阵列（Systolic Array）的工作原理与数据流优化
- **软硬协同**：XLA 编译器、HLO/StableHLO 中间表示在 TPU 性能释放中的关键作用
- **架构对比**：TPU (以数据为中心) vs GPU (以线程为中心) 的根本差异

#### 1.2.2 GPU 架构和编程模型介绍

- [**GPU Architecture and Programming — An Introduction**](gpu_programming/gpu_programming_introduction.md) - `GPU` 架构与编程模型的全面介绍
- [**GPU 词汇表**](https://modal.com/gpu-glossary/readme) - 全面的 GPU 技术术语词典，涵盖 GPU 架构、CUDA 编程、并行计算等核心概念

**参考资料：**

- [**GPU 特性**](gpu_architecture/gpu_characteristics.md)
- [**GPU 内存**](gpu_architecture/gpu_memory.md)
- [**GPU Example: Tesla V100**](gpu_architecture/tesla_v100.md)
- [**GPUs on Frontera: RTX 5000**](gpu_architecture/rtx_5000.md)
- **练习**：
  - [**Exercise: Device Query**](gpu_architecture/exer_device_query.md)
  - [**Exercise: Device Bandwidth**](gpu_architecture/exer_device_bandwidth.md)

#### 1.2.3 CUDA 核心技术

- [**深入理解 NVIDIA CUDA 核心（vs. Tensor Cores vs. RT Cores）**](cuda/cuda_cores_cn.md)

#### 1.2.4 GPGPU vs NPU 技术对比

- [**GPGPU vs NPU：大模型推理与训练的算力选择指南**](gpu_architecture/GPGPU_vs_NPU_大模型推理训练对比.md) - 全面对比分析 GPGPU 和 NPU 在大模型场景下的技术特点、性能表现和应用选择

**核心内容：**

- **技术架构深度对比**：NVIDIA H100 vs 华为昇腾 910B 架构解析，计算模式差异分析
- **大模型训练场景**：训练任务特点、GPGPU 生态优势、NPU 专用优化、成本效益分析
- **大模型推理场景**：在线推理服务、批量推理处理、边缘推理部署的技术选型
- **实用决策框架**：场景化决策矩阵、技术选型指导原则、混合部署策略
- **发展趋势展望**：技术演进路线、生态建设、标准化进程

### 1.3 AI 基础设施架构

- [**高性能 GPU 服务器硬件拓扑与集群组网**](https://arthurchiao.art/blog/gpu-advanced-notes-1-zh/)
- [**NVIDIA GH200 芯片、服务器及集群组网**](https://arthurchiao.art/blog/gpu-advanced-notes-4-zh/)
- [**深度学习（大模型）中的精度**](https://mp.weixin.qq.com/s/b08gFicrKNCfrwSlpsecmQ)

### 1.4 GPU 管理与虚拟化

**理论与架构：**

- [**GPU 虚拟化与切分技术原理解析**](gpu_manager/GPU虚拟化与切分技术原理解析.md) - 技术原理深入
- [**GPU 管理相关技术深度解析 - 虚拟化、切分及远程调用**](gpu_manager/GPU%20管理相关技术深度解析%20-%20虚拟化、切分及远程调用.md) - 全面的 GPU 管理技术指南
- [**第一部分：基础理论篇**](gpu_manager/第一部分：基础理论篇.md) - GPU 管理基础概念与理论
- [**第二部分：虚拟化技术篇**](gpu_manager/第二部分：虚拟化技术篇.md) - 硬件、内核、用户态虚拟化技术
- [**第三部分：资源管理与优化篇**](gpu_manager/第三部分：资源管理与优化篇.md) - GPU 切分与资源调度算法
- [**第四部分：实践应用篇**](gpu_manager/第四部分：实践应用篇.md) - 部署、运维、性能调优实践

**GPU 虚拟化解决方案：**

- [**HAMi GPU 资源管理完整指南**](gpu_manager/hami/hmai-gpu-resources-guide.md)

### 1.5 GPU 运维工具与实践

- [**GPU 监控与运维工具概述**](ops/README.md) - 企业级 GPU 集群的全方位监控与运维解决方案
- [**nvidia-smi 入门**](ops/nvidia-smi.md)
- [**nvtop 入门**](ops/nvtop.md)
- [**NVIDIA GPU XID 故障码解析**](https://mp.weixin.qq.com/s/ekCnhr3qrhjuX_-CEyx65g)
- [**NVIDIA GPU 卡之 ECC 功能**](https://mp.weixin.qq.com/s/nmZVOQAyfFyesm79HzjUlQ)
- [**查询 GPU 卡详细参数**](ops/DeviceQuery.md)
- [**Understanding NVIDIA GPU Performance: Utilization vs. Saturation (2023)**](https://arthurchiao.art/blog/understanding-gpu-performance/)
- [**GPU 利用率是一个误导性指标**](ops/GPU%20利用率是一个误导性指标.md)

### 1.6 分布式存储系统

分布式存储系统是现代 `AI` 基础设施的核心组件，为大规模机器学习训练和推理提供高性能、高可靠性的数据存储解决方案。在 `AI` 工作负载中，分布式存储系统需要应对海量训练数据集、频繁的模型检查点保存、多节点并发访问等挑战。

从 `AI Infra` 的角度，分布式存储系统的关键技术包括：

1. **高吞吐量数据访问**：支持多 `GPU` 节点并发读取训练数据，提供聚合带宽达到数十 GB/s 的性能表现
2. **元数据管理优化**：针对深度学习场景的小文件密集访问模式，优化元数据缓存和索引机制
3. **数据一致性保证**：在分布式训练过程中确保检查点数据的强一致性，支持故障恢复和断点续训
4. **存储层次化**：结合 `NVMe SSD`、`HDD` 和对象存储，实现冷热数据分层管理和成本优化
5. **网络优化**：利用 `RDMA`、`NVLink` 等高速网络技术，减少存储 I/O 延迟对训练性能的影响

**JuiceFS 分布式文件系统：**

- [**JuiceFS 文件修改机制分析**](juicefs/JuiceFS%20文件修改机制分析.md) - 分布式文件系统的修改机制深度解析
- [**JuiceFS 后端存储变更手册**](juicefs/JuiceFS%20后端存储变更手册.md) - JuiceFS 后端存储迁移和变更操作指南
- [**3FS 分布式文件系统**](deepseek/deepseek_3fs_design_notes.zh-CN.md) - 高性能分布式文件系统的设计理念与技术实现
  - **系统架构**：集群管理器、元数据服务、存储服务、客户端四大组件
  - **核心技术**： RDMA 网络、CRAQ 链式复制、异步零拷贝 API
  - **性能优化**： FUSE 局限性分析、本地客户端设计、io_uring 启发的 API 设计

### 1.7 高性能网络与通信

在 `AI` 基础设施中，高性能网络与通信技术是实现大规模分布式训练和推理的关键基础设施。现代 `AI` 工作负载对网络性能提出了极高要求：超低延迟（亚微秒级）、超高带宽（数百 Gbps）、高可靠性和可扩展性。本章节涵盖 `InfiniBand`、`RDMA`、`NCCL` 等核心技术，这些技术通过硬件加速、零拷贝传输、拓扑感知优化等手段，为 `AI` 集群提供高效的数据传输和同步能力。

#### 1.7.1 InfiniBand 网络技术

`InfiniBand`（IB）是专为高性能计算设计的网络架构，在 `AI` 基础设施中扮演着数据传输高速公路的角色。相比传统以太网， `IB` 提供更低的延迟（<1μs）、更高的带宽（200Gbps+）和更强的可扩展性，特别适合大规模 `GPU` 集群的参数同步、梯度聚合等通信密集型任务。

- [**InfiniBand 网络理论与实践**](InfiniBand/IB%20网络理论与实践.md) - 企业级高性能计算网络的核心技术栈
  - **技术特性**：亚微秒级延迟、200Gbps+ 带宽、RDMA 零拷贝传输
  - **应用场景**：大规模分布式训练、高频金融交易、科学计算集群
  - **架构优势**：硬件级卸载、CPU 旁路、内存直接访问
- [**InfiniBand 健康检查工具**](InfiniBand/health/README.md) - 网络健康状态监控和故障诊断
- [**InfiniBand 带宽监控**](InfiniBand/monitor/README.md) - 实时带宽监控和性能分析

#### 1.7.2 RDMA 远程直接内存访问

`RDMA`（Remote Direct Memory Access）是高性能网络通信的核心技术，允许应用程序直接访问远程主机的内存，绕过操作系统内核和 CPU ，实现真正的零拷贝数据传输。在 `AI` 基础设施中， `RDMA` 技术显著降低了分布式训练中的通信延迟和 `CPU` 开销，提高了整体系统效率。

**核心技术特性：**

- **零拷贝传输**：数据直接在网卡和应用内存间传输，无需 `CPU` 参与
- **内核旁路**：绕过操作系统网络协议栈，减少上下文切换开销
- **硬件卸载**：网络处理完全由硬件完成，释放 `CPU` 资源用于计算
- **低延迟保证**：亚微秒级延迟，满足实时通信需求

**在 AI 场景中的应用：**

- **参数服务器架构**：高效的参数同步和梯度聚合
- **AllReduce 优化**：加速分布式训练中的集合通信操作
- **模型并行**：支持大模型的跨节点内存共享
- **数据流水线**：优化训练数据的预处理和传输流程

#### 1.7.3 NCCL 分布式通信

- [**NCCL 分布式通信测试套件使用指南**](nccl/tutorial.md) - NVIDIA 集合通信库的深度技术解析
  - **核心算法**： AllReduce、AllGather、Broadcast、ReduceScatter 优化实现
  - **性能调优**：网络拓扑感知、带宽聚合、计算通信重叠
  - **生态集成**：与 PyTorch、TensorFlow、MPI 的深度集成方案
- [**NCCL Kubernetes 部署**](nccl/k8s/README.md) - 容器化 NCCL 集群部署方案

**核心特性：**

- **PXN 模式支持**：专为多节点优化的高性能通信解决方案
- **三种优化级别**：保守、平衡、激进模式，满足不同性能需求
- **智能网络检测**：自动选择最佳网络配置和通信路径
- **容器化部署**：支持 Docker 和 Kubernetes 部署
- **多节点测试**：支持大规模分布式训练场景

**测试工具：**

- [**NCCL 性能基准测试**](nccl/nccl_benchmark.sh) - 支持 PXN 模式的性能测试
- [**容器化测试管理**](nccl/nccl_container_manager.sh) - 容器化测试环境管理
- [**多节点测试启动器**](nccl/nccl_multinode_launcher.sh) - 原生多节点测试部署

### 1.8 性能分析与调优

#### 1.8.1 AI 系统性能分析概述

- [**AI 系统性能分析**](profiling/README.md) - 企业级 AI 系统的全栈性能分析与瓶颈诊断

**分析维度：**

- **多维分析**：计算密集度、内存访问模式、网络通信效率、存储 I/O 性能
- **专业工具**： Nsight Systems 系统级分析、Nsight Compute 内核级优化、Intel VTune 性能调优
- **优化方法论**：算子融合策略、内存池化管理、计算通信重叠、数据流水线优化

#### 1.8.2 GPU 性能分析

- [**使用 Nsight Compute Tool 分析 CUDA 矩阵乘法程序**](https://www.yuque.com/u41800946/nquqpa/eo7gykiyhg8xi2gg)
- [**CUDA 内核性能分析指南**](profiling/s9345-cuda-kernel-profiling-using-nvidia-nsight-compute.pdf) - NVIDIA 官方 CUDA 内核性能分析详细指南

**性能分析工具：**

- **NVIDIA Nsight Compute**： CUDA 内核级性能分析器
- **NVIDIA Nsight Systems**：系统级性能分析器
- **nvprof**：传统 CUDA 性能分析工具

**关键指标与优化：**

- **硬件指标**： SM 占用率、内存带宽利用率、L1/L2 缓存命中率、Tensor Core 效率
- **内核优化**： CUDA Kernel 性能调优、内存访问模式优化、线程块和网格配置
- **分析工具**： CUDA Profiler 性能剖析、Nsight Graphics 图形分析、GPU-Z 硬件监控

**性能优化实践：**

- **全局内存访问模式优化**：提升内存访问效率
- **共享内存（Shared Memory）优化**：利用片上高速缓存
- **指令级并行（ILP）优化**：提升计算吞吐量
- **内存带宽利用率分析**：优化数据传输性能

---

## 2. 云原生 AI 基础设施

本章聚焦于云原生技术在 AI 领域的应用，探讨如何利用 Kubernetes 等云原生技术栈构建高效、可扩展的 AI 基础设施。

### 2.1 Kubernetes AI 生态

- [**Kubernetes AI 基础设施概述**](k8s/README.md) - 企业级容器化 AI 工作负载的编排管理平台
- [**Kueue + HAMi 集成方案**](k8s/Kueue%20+%20HAMi.md) - GPU 资源调度与管理的云原生解决方案
- [**NVIDIA Container Toolkit 原理分析**](k8s/Nvidia%20Container%20Toolkit%20原理分析.md) - 容器化 GPU 支持的底层机制
- [**NVIDIA K8s Device Plugin 分析**](k8s/nvidia-k8s-device-plugin-analysis.md) - GPU 设备插件的架构与实现

**核心特性：**

- **智能调度**： GPU 资源共享、NUMA 拓扑感知、多优先级调度策略
- **资源管理**： GPU Operator、Node Feature Discovery、MIG Manager 统一管理
- **可观测性**： Prometheus 指标采集、Grafana 可视化、Jaeger 链路追踪

### 2.2 AI 推理服务

- [**云原生高性能分布式 LLM 推理框架 llm-d 介绍**](k8s/llm-d-intro.md) - 基于 Kubernetes 的大模型推理框架
- [**vLLM + LWS ： Kubernetes 上的多机多卡推理方案**](k8s/lws_intro.md) - LWS 旨在提供一种 **更符合 AI 原生工作负载特点的分布式控制器语义**，填补现有原语在推理部署上的能力空白

**技术架构：**

- **服务治理**： Istio 服务网格、Envoy 代理、智能负载均衡
- **弹性伸缩**： HPA 水平扩展、VPA 垂直扩展、KEDA 事件驱动自动化
- **模型运营**：多版本管理、A/B 测试、金丝雀发布、流量切换

---

## 3. 开发与编程

本部分专注于 `AI` 开发相关的编程技术、工具和实践，涵盖从基础编程到高性能计算的完整技术栈。

### 3.1 AI 编程入门

- [**AI 编程入门完整教程**](ai_coding/AI%20编程入门.md) - 面向初学者的 AI 编程完整学习路径与实践指南
- [**AI 编程入门在线版本**](ai_coding/index.html) - 交互式在线学习体验与动手实践

**学习路径：**

- **理论基础**：机器学习核心概念、深度学习原理、神经网络架构设计
- **编程语言生态**： Python AI 生态、R 统计分析、Julia 高性能计算在 AI 中的应用
- **开发环境搭建**： Jupyter Notebook 交互式开发、PyCharm 专业 IDE、VS Code 轻量级配置

### 3.2 GPU 与 CUDA 编程

本节整合了 GPU 基础架构、CUDA 核心编程概念及丰富的学习资源，为开发者提供从入门到进阶的完整技术路径。

#### 3.2.1 核心概念与开发

- [**CUDA 核心概念详解**](cuda/cuda_cores_cn.md) - CUDA 核心、线程块、网格等基础概念的深度解析
- [**CUDA 流详解**](cuda/cuda_streams.md) - CUDA 流的原理、应用场景与性能优化

**技术特色：**

- **CUDA 核心架构**： SIMT 线程模型、分层内存模型、流式执行模型
- **性能调优实践**：内存访问模式优化、线程同步策略、算法并行化重构
- **高级编程特性**： Unified Memory 统一内存、Multi-GPU 多卡编程、CUDA Streams 异步执行

#### 3.2.2 GPU 编程基础

- [**GPU 编程基础**](gpu_programming/README.md) - GPU 编程入门到进阶的完整技术路径，涵盖 GPU 架构、编程模型和性能优化

**核心内容：**

- **GPU 架构理解**：GPU 与 CPU 的架构差异、并行计算原理、内存层次结构
- **CUDA 编程实践**：线程模型、内存管理、核函数编写、性能优化技巧
- **调试与性能分析**：CUDA 调试工具、性能分析方法、瓶颈识别与优化
- **高级特性应用**：流处理、多 GPU 编程、与深度学习框架的集成

#### 3.2.3 学习资源与进阶

**快速入门**：

- [**并行计算、费林分类法和 CUDA 基本概念**](https://mp.weixin.qq.com/s/NL_Bz8JB-LdAtrQake7EdA)
- [**CUDA 编程模型入门**](https://mp.weixin.qq.com/s/IUYzzgt6DUYhfaDnbxoZuQ)
- [**CUDA 并发编程之 Stream 介绍**](cuda/cuda_streams.md)

**参考资料**：

- [**CUDA Reading Group 相关讲座**](https://mp.weixin.qq.com/s/6sOrNzG0UeVBes8stWSoWA)
- [GPU Mode Reading Group](https://github.com/gpu-mode)
- [**《CUDA C++ Programming Guide 》**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [**《CUDA C 编程权威指南》**](https://mp.weixin.qq.com/s/xJY5Znv3cuQi_UCd_XjJ4A)
- [书中示例代码](https://github.com/Eddie-Wang1120/Professional-CUDA-C-Programming-Code-and-Notes)
- [**Nvidia 官方 CUDA 示例**](https://github.com/NVIDIA/cuda-samples)
- [**《CUDA 编程：基础与实践 by 樊哲勇》**](https://book.douban.com/subject/35252459/)
- [**学习笔记**](https://github.com/QINZHAOYU/CudaSteps)
- [**示例代码**](https://github.com/MAhaitao999/CUDA_Programming)
- [**樊哲勇主页**](https://wlkxyjsxy.bhu.edu.cn/engine2/general/4146630/detail?engineInstanceId=656243&typeId=2986094&pageId=85748&websiteId=63087&currentBranch=1)
- [**《CUDA 编程简介: 基础与实践 by 李瑜》**](./cuda/CUDA%20编程简介%20-%20基础与实践.pdf)
- [**《CUDA 编程入门》** - 本文改编自北京大学超算队 CUDA 教程讲义](https://hpcwiki.io/gpu/cuda/)
- [**Multi GPU Programming Models**](https://github.com/NVIDIA/multi-gpu-programming-models)
- [**CUDA Processing Streams**](https://turing.une.edu.au/~cosc330/lectures/display_lecture.php?lecture=22#1)

**专业进阶**：

本小节面向 `CUDA` 编程的专业开发者，提供深度优化的内核实现和高性能计算技术。通过 200+ 个精心设计的内核示例，涵盖 `Tensor Core`、`CUDA Core` 的极致优化，以及现代 `GPU` 架构的前沿技术应用，帮助开发者达到接近硬件理论峰值的性能表现。

- [**CUDA-Learn-Notes**](https://github.com/xlite-dev/CUDA-Learn-Notes)：📚Modern CUDA Learn Notes: 200+ Tensor/CUDA Cores Kernels🎉, HGEMM, FA2 via MMA and CuTe, 98~100% TFLOPS of cuBLAS/FA2.

### 3.3 Java AI 开发

- [**Java AI 开发指南**](java_ai/README.md) - Java 生态系统中的 AI 开发技术
- [**使用 Spring AI 构建高效 LLM 代理**](java_ai/spring_ai_cn.md) - 基于 Spring AI 框架的企业级 AI 应用开发

**技术特色：**

- **企业级框架**：基于成熟的 Spring 生态系统
- **多提供商支持**：统一 API 集成 OpenAI、Azure OpenAI、Hugging Face 等
- **生产就绪**：提供完整的企业级 AI 应用解决方案
- **Java 原生**：充分利用 Java 生态系统的优势

---

## 4. 机器学习基础

本部分基于 [**动手学机器学习**](https://github.com/ForceInjection/hands-on-ML) 项目，提供系统化的机器学习学习路径。

### 4.1 机器学习学习资源

- [**动手学机器学习**](hands-on-ML/README.md) - 全面的机器学习学习资源库，包含理论讲解、代码实现和实战案例

**核心特色：**

- **理论与实践结合**：从数学原理到代码实现的完整学习路径
- **算法全覆盖**：监督学习、无监督学习、集成学习、深度学习等核心算法
- **项目驱动学习**：通过实际项目掌握机器学习的完整工作流程
- **工程化实践**：特征工程、模型评估、超参数调优等工程技能

### 4.2 基础概念与数学准备

- [**通俗理解机器学习核心概念**](hands-on-ML/nju_software/通俗理解机器学习核心概念.md)
- [**梯度下降算法：从直觉到实践**](hands-on-ML/nju_software/梯度下降算法：从直觉到实践.md)
- [**混淆矩阵评价指标**](hands-on-ML/nju_software/混淆矩阵评价指标.md)
- [**误差 vs. 残差**](hands-on-ML/nju_software/误差%20vs%20残差.md)
- [**线性代数的本质**](https://www.bilibili.com/video/BV1ys411472E) - 3Blue1Brown 可视化教程
- [**MIT 18.06 线性代数**](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) - Gilbert Strang 经典课程
- [**概率论与统计学基础**](https://book.douban.com/subject/35798663/) - 贝叶斯定理、概率分布、最大似然估计

### 4.3 监督学习

监督学习是机器学习的核心分支，通过标注数据训练模型进行预测和分类。本节系统介绍监督学习的基础算法和集成方法，从经典的线性模型到现代的集成技术，涵盖理论原理、算法实现和实际应用，为机器学习实践奠定坚实基础。

#### 4.3.1 基础算法

基础算法是监督学习的核心组成部分，包括线性模型、树模型、概率模型等经典算法。这些算法各有特色，适用于不同的数据类型和问题场景，是机器学习从业者必须掌握的基础技能。

- [**KNN 算法**](hands-on-ML/nju_software/ch-03/动手学机器学习%20KNN%20算法.md) - K 近邻算法理论与实现
- [**线性回归**](hands-on-ML/nju_software/ch-03/动手学机器学习线性回归算法.md) - 最小二乘法、正则化
- [**逻辑回归**](hands-on-ML/nju_software/ch-03/动手学机器学习逻辑回归算法.md) - 分类算法基础
- [**决策树**](hands-on-ML/nju_software/ch-03/动手学机器学习决策树算法.md) - ID3、C4.5、CART 算法
- [**支持向量机**](hands-on-ML/nju_software/ch-03/动手学机器学习支持向量机算法.md) - 核技巧与软间隔
- [**朴素贝叶斯**](hands-on-ML/nju_software/ch-03/动手学机器学习朴素贝叶斯算法.md) - 概率分类器

#### 4.3.2 集成学习

集成学习通过组合多个基学习器来提升模型性能，是现代机器学习竞赛和工业应用的重要技术。本小节深入探讨 Bagging、Boosting、Stacking 等主流集成方法，以及它们在实际项目中的应用策略和优化技巧。

- [**随机森林**](hands-on-ML/nju_software/ch-04/动手学机器学习随机森林算法.md) - Bagging 集成方法
- [**AdaBoost**](hands-on-ML/nju_software/ch-04/Adaboost%20计算示例.md) - Boosting 算法
- [**GBDT**](hands-on-ML/nju_software/ch-04/一文读懂GBDT.md) - 梯度提升决策树
- [**Stacking**](hands-on-ML/nju_software/ch-04/kaggle房价预测中的集成技巧.md) - 模型堆叠技术
- [**集成学习概述**](hands-on-ML/nju_software/ch-04/一文深入了解机器学习之集成学习.md) - 集成学习理论与方法

### 4.4 无监督学习

无监督学习从无标注数据中发现隐藏的模式和结构，是数据挖掘和知识发现的重要手段。本节涵盖聚类、降维、概率建模等核心技术，通过理论讲解和实践案例，帮助读者掌握无监督学习的精髓和应用技巧。

#### 4.4.1 聚类算法

聚类算法是无监督学习的重要分支，用于发现数据中的自然分组结构。本小节介绍基于距离、密度、层次等不同原理的聚类方法，分析各算法的优缺点和适用场景，为数据分析和客户细分等应用提供技术支撑。

- [**K-means 聚类**](hands-on-ML/nju_software/ch-05/动手学机器学习%20Kmeans%20聚类算法.md) - 基础聚类算法
- [**层次聚类**](hands-on-ML/nju_software/ch-05/动手学机器学习层次聚类算法.md) - 凝聚与分裂聚类
- [**DBSCAN**](hands-on-ML/nju_software/ch-05/动手学机器学习：DBSCAN%20密度聚类算法.md) - 密度聚类算法

#### 4.4.2 降维算法

降维算法通过减少数据维度来简化问题复杂度，同时保留数据的主要信息。本小节深入探讨线性和非线性降维方法，包括主成分分析、线性判别分析等经典技术，以及它们在数据可视化和特征提取中的应用。

- [**PCA 主成分分析**](hands-on-ML/nju_software/ch-11/PCA降维算法详解.md) - 线性降维方法
- [**LDA 线性判别分析**](hands-on-ML/nju_software/ch-11/LDA降维算法详解.md) - 监督降维技术
- [**PCA vs LDA 比较**](hands-on-ML/nju_software/ch-11/PCA%20vs.%20LDA%20降维方法比较.md) - 降维方法对比分析

#### 4.4.3 概率模型

概率模型基于统计学原理对数据进行建模，能够处理不确定性和噪声。本小节介绍期望最大化算法、高斯混合模型等重要概率建模技术，以及最大似然估计的理论基础和实际应用。

- [**EM 算法**](hands-on-ML/nju_software/ch-06/一文了解%20EM%20算法.md) - 期望最大化算法
- [**高斯混合模型**](hands-on-ML/nju_software/ch-06/一文了解%20GMM%20算法.md) - GMM 聚类方法
- [**最大似然估计**](hands-on-ML/nju_software/ch-06/最大似然估计（MLE）简介.md) - MLE 理论基础

### 4.5 特征工程与模型优化

特征工程是机器学习项目成功的关键因素，直接影响模型的性能和泛化能力。本节系统介绍特征工程的方法论、模型评估技术和优化策略，通过理论指导和实践案例，帮助读者掌握从数据预处理到模型调优的完整技能体系。

#### 4.5.1 特征工程

特征工程是将原始数据转换为机器学习算法可以有效利用的特征的过程。本小节涵盖数据清洗、特征选择、特征变换等核心技术，以及针对不同数据类型（数值、文本、时间序列）的专门处理方法。

- [**特征工程概述**](hands-on-ML/nju_software/ch-07/特征工程.md) - 数据预处理、特征选择与变换
- [**特征选择方法**](hands-on-ML/nju_software/ch-10/特征选择方法概述.md) - 过滤法、包装法、嵌入法
- [**GBDT 特征提取**](hands-on-ML/nju_software/ch-07/GBDT特征提取.md) - 基于树模型的特征工程
- [**时间序列特征提取**](hands-on-ML/nju_software/ch-07/时间序列数据及特征提取.md) - 时间序列数据处理
- [**词袋模型**](hands-on-ML/nju_software/ch-07/词袋模型介绍.md) - 文本特征工程

#### 4.5.2 模型评估

模型评估是机器学习项目的重要环节，用于客观衡量模型性能和选择最优方案。本小节介绍各种评估指标、交叉验证方法、超参数优化技术，以及处理数据不平衡等实际问题的解决方案。

- [**模型评估方法**](hands-on-ML/nju_software/ch-08/图解机器学习-模型评估方法与准则.md) - 评估指标与交叉验证
- [**混淆矩阵评价指标**](hands-on-ML/nju_software/混淆矩阵评价指标.md) - 分类模型性能评估
- [**GridSearchCV**](hands-on-ML/nju_software/ch-09/gridsearchcv_intro.md) - 超参数优化实践
- [**L1 L2 正则化**](hands-on-ML/nju_software/ch-09/L1_L2_intro.md) - 正则化方法介绍
- [**SMOTE 采样**](hands-on-ML/nju_software/ch-09/SMOTE%20介绍.md) - 不平衡数据处理

### 4.6 推荐系统与概率图模型

推荐系统和概率图模型是机器学习在实际应用中的重要分支。推荐系统通过分析用户行为和物品特征来提供个性化推荐，概率图模型则用于处理复杂的依赖关系和不确定性。本节深入探讨这两个领域的核心技术和实际应用。

#### 4.6.1 推荐系统

推荐系统是现代互联网应用的核心技术之一，广泛应用于电商、内容平台、社交网络等场景。本小节系统介绍协同过滤、内容推荐、矩阵分解等主流推荐算法，以及推荐系统的评估方法和工程实践。

- [**推荐系统入门**](hands-on-ML/nju_software/ch-12/recommendation_intro.md) - 推荐算法概述
- [**协同过滤算法**](hands-on-ML/nju_software/ch-12/协同过滤推荐算法：原理、实现与分析.md) - 用户协同过滤与物品协同过滤
- [**基于内容的推荐**](hands-on-ML/nju_software/ch-12/基于内容的推荐算法：原理与实践.md) - 内容推荐算法
- [**矩阵分解推荐**](hands-on-ML/nju_software/ch-12/基于矩阵分解的推荐算法：原理与实践.md) - SVD 推荐算法
- [**关联规则挖掘**](hands-on-ML/nju_software/ch-12/使用%20Apriori%20算法进行关联分析：原理与示例.md) - Apriori 算法

#### 4.6.2 概率图模型

概率图模型结合了概率论和图论，用于建模变量间的复杂依赖关系。本小节介绍贝叶斯网络、隐马尔可夫模型等重要概率图模型，探讨它们在自然语言处理、计算机视觉等领域的应用。

- [**贝叶斯网络**](hands-on-ML/nju_software/ch-13/一文读懂贝叶斯网络.md) - 概率图模型基础
- [**隐马尔可夫模型**](hands-on-ML/nju_software/ch-13/一文读懂隐马尔可夫模型（HMM）.md) - 序列建模与状态推断
- [**马尔可夫模型**](hands-on-ML/nju_software/ch-13/马尔可夫模型简介.md) - 马尔可夫链基础

### 4.7 深度学习基础

深度学习是现代人工智能的核心技术，通过多层神经网络学习数据的深层表示。本节介绍深度学习的基本概念、神经网络架构和训练方法，为后续的大模型学习奠定基础。

- [**深度学习概述**](hands-on-ML/nju_software/ch-14/深度学习概述.md) - 深度学习理论与实践指南
- [**神经网络基础**](hands-on-ML/nju_software/ch-14/神经网络示例.md) - 感知机、多层感知机、反向传播
- [**什么是深度学习**](hands-on-ML/nju_software/ch-14/什么是深度学习？.md) - 深度学习入门介绍

### 4.8 实战项目

实战项目是将理论知识转化为实际技能的重要途径。本节提供多个完整的机器学习项目案例，涵盖数据预处理、特征工程、模型训练、评估优化等完整流程，帮助读者积累实际项目经验。

- [**泰坦尼克号幸存者预测**](hands-on-ML/nju_software/ch-03/使用决策树对泰坦尼克号幸存者数据进行分类.md) - 特征工程与分类实战
- [**朴素贝叶斯实例**](hands-on-ML/nju_software/ch-03/朴素贝叶斯计算：建筑工人打喷嚏后患感冒的概率.md) - 概率计算实例
- [**RFM 用户分析**](hands-on-ML/nju_software/ch-07/数据探索-根据历史订单信息求RFM值.md) - 用户价值分析
- [**电影推荐系统**](hands-on-ML/nju_software/ch-12/movie-recommendation.ipynb) - 推荐算法实战

### 4.9 学习资源

本节汇总了机器学习领域的优质学习资源，包括经典教材、在线课程、实践平台等，为不同学习阶段的读者提供系统性的学习路径和参考资料。

#### 4.9.1 核心教材

- **《统计学习方法》** - 李航著，算法理论基础
- **《机器学习》** - 周志华著，西瓜书经典
- **《模式识别与机器学习》** - Bishop 著，数学严谨

#### 4.9.2 在线资源

- [**机器学习考试复习提纲**](hands-on-ML/nju_software/机器学习考试复习提纲.md) - 考试重点总结
- [**梯度下降算法详解**](hands-on-ML/nju_software/梯度下降算法：从直觉到实践.md) - 优化算法理解
- [**机器学习核心概念**](hands-on-ML/nju_software/通俗理解机器学习核心概念.md) - 概念通俗解释
- [**Andrew Ng 机器学习课程**](https://www.coursera.org/learn/machine-learning) - Coursera 经典课程
- [**CS229 机器学习**](http://cs229.stanford.edu/) - 斯坦福大学课程

#### 4.9.3 实践平台

- [**Kaggle**](https://www.kaggle.com/) - 数据科学竞赛平台
- [**Google Colab**](https://colab.research.google.com/) - 免费 GPU 环境
- [**scikit-learn**](https://scikit-learn.org/) - Python 机器学习库

---

## 5. 大语言模型基础

本章旨在为读者构建扎实的大语言模型（LLM）理论基础，涵盖从词向量嵌入到模型架构设计的核心知识。我们将深入解析 Token 机制、Transformer 架构、混合专家模型（MoE）等关键技术，并探讨量化、思维链（CoT）等前沿优化方向，帮助开发者建立对 LLM 内部机制的直观理解。

### 5.1 基础理论与概念

大语言模型的基础理论涵盖了从文本处理到模型架构的核心概念。理解这些基础概念是深入学习 `LLM` 技术的前提，包括 Token 化机制、文本编码、模型结构等关键技术。这些基础知识为后续的模型训练、优化和应用奠定了坚实的理论基础。

- [**Andrej Karpathy ： Deep Dive into LLMs like ChatGPT （B 站视频）**](https://www.bilibili.com/video/BV16cNEeXEer) - 深度学习领域权威专家的 LLM 技术解析
- [**大模型基础组件 - Tokenizer**](https://zhuanlan.zhihu.com/p/651430181) - 文本分词与编码的核心技术
- [**解密大语言模型中的 Tokens**](llm/token/llm_token_intro.md) - Token 机制的深度解析与实践应用
  - [**Tiktokenizer 在线版**](https://tiktokenizer.vercel.app/?model=gpt-4o) - 交互式 Token 分析工具

### 5.2 嵌入技术与表示学习

嵌入技术是大语言模型的核心组件之一，负责将离散的文本符号转换为连续的向量表示。这一技术不仅影响模型的理解能力，还直接关系到模型的性能和效率。本节深入探讨文本嵌入的原理、实现方式以及在不同场景下的应用策略。

- [**文本嵌入（Text-Embedding） 技术快速入门**](llm/embedding/text_embeddings_guide.md) - 文本向量化的理论基础与实践
- [**LLM 嵌入技术详解：图文指南**](llm/embedding/LLM%20Embeddings%20Explained%20-%20A%20Visual%20and%20Intuitive%20Guide.zh-CN.md) - 可视化理解嵌入技术
- [**大模型 Embedding 层与独立 Embedding 模型：区别与联系**](llm/embedding/embedding.md) - 嵌入层架构设计与选型策略

### 5.3 高级架构与优化技术

现代大语言模型采用了多种先进的架构设计和优化技术，以提升模型性能、降低计算成本并解决特定问题。本节涵盖混合专家系统、量化技术、思维链推理等前沿技术，这些技术代表了当前 LLM 领域的最新发展方向。

- [**大模型可视化指南**](https://www.maartengrootendorst.com/) - 大模型内部机制的可视化分析
- [**一文读懂思维链（Chain-of-Thought, CoT）**](llm/一文读懂思维链（Chain-of-Thought,%20CoT）.md) - 推理能力增强的核心技术
- [**大模型的幻觉及其应对措施**](llm/大模型的幻觉及其应对措施.md) - 幻觉问题的成因分析与解决方案
- [**大模型文件格式完整指南**](llm/大模型文件格式完整指南.md) - 模型存储与部署的技术规范
- [**混合专家系统（MoE）图解指南**](<llm/A%20Visual%20Guide%20to%20Mixture%20of%20Experts%20(MoE).zh-CN.md>) - 稀疏激活架构的设计原理
- [**量化技术可视化指南**](llm/A%20Visual%20Guide%20to%20Quantization.zh-CN.md) - 模型压缩与加速的核心技术
- [**基于大型语言模型的意图检测**](llm/Intent%20Detection%20using%20LLM.zh-CN.md) - 自然语言理解的实际应用

### 5.4 参考书籍

- [**大模型技术 30 讲**](https://mp.weixin.qq.com/s/bNH2HaN1GJPyHTftg62Erg) - 大模型时代，智能体崛起：从技术解构到工程落地的全栈指南
  - 第三方：[大模型技术 30 讲（英文&中文批注）](https://ningg.top/Machine-Learning-Q-and-AI)
- [**大模型基础**](https://github.com/ZJU-LLMs/Foundations-of-LLMs) <br>
  <img src="https://raw.githubusercontent.com/ZJU-LLMs/Foundations-of-LLMs/main/figure/cover.png" height="300"/>

- [**Hands-On Large Language Models**](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) <br>
  <img src="https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/images/book_cover.png" height="300"/>

- [**从零构建大模型**](https://mp.weixin.qq.com/s/FkBjsQmeXEPlsdFXETYSng) - 从理论到实践，手把手教你打造自己的大语言模型
- [**百面大模型**](https://mp.weixin.qq.com/s/rBJ5an0pr3TgjFbyJXa0WA) - 打通大模型求职与实战的关键一书
- [**图解大模型：生成式 AI 原理与实践**](https://mp.weixin.qq.com/s/tYrHrpMrZySgWKE1ECqTWg) - 超过 300 幅全彩图示 × 实战级项目代码 × 中文独家 DeepSeek-R1 彩蛋内容，入门、进阶、实操、求职一步到位！

---

## 6. 大模型训练

大模型的训练是一个复杂且系统的工程，涉及数据处理、分布式训练、指令微调等多个关键环节。本章将详细介绍从指令微调（SFT）到大规模模型预训练的完整技术路径，结合 70B 参数模型的实战案例，深入探讨训练基础设施的搭建、超参数优化及模型后训练（Post-Training）策略，助力开发者掌握模型训练的核心技能。

### 6.1 指令微调与监督学习

指令微调（Instruction Tuning）和监督微调（Supervised Fine-Tuning, SFT）是大语言模型训练的关键技术，通过在预训练模型基础上使用高质量的指令-响应数据对进行进一步训练，使模型能够更好地理解和执行人类指令。这一技术对于提升模型的实用性和安全性具有重要意义。

- [**Qwen 2 大模型指令微调实战**](llm/fine-tuning/train_qwen2.ipynb) - 基于 Qwen 2 的指令微调 Notebook 实践
- [**Qwen 2 指令微调教程**](https://mp.weixin.qq.com/s/Atf61jocM3FBoGjZ_DZ1UA) - 详细的图文教程
- [**一文入门垂域模型 SFT 微调**](llm/一文入门垂域模型SFT微调.md) - 垂直领域模型的监督微调技术与应用实践

### 6.2 大规模模型训练实践

大规模模型训练是一个复杂的系统工程，涉及数据处理、基础设施搭建、分布式训练、超参数优化等多个方面。本节通过实际的 70B 参数模型训练案例，深入探讨从硬件配置到模型评估的完整训练流程，为大规模模型训练提供实践指导。

- [**Training a 70B model from scratch: open-source tools, evaluation datasets, and learnings**](https://imbue.com/research/70b-intro/) - 70B 参数模型从零训练的完整技术路径与经验总结
- [**Sanitized open-source datasets for natural language and code understanding: how we evaluated our 70B model**](https://imbue.com/research/70b-evals/) - 大规模训练数据集的清洗、评估与质量控制方法
- [**From bare metal to a 70B model: infrastructure set-up and scripts**](https://imbue.com/research/70b-infrastructure/) - 大模型训练基础设施的搭建、配置与自动化脚本
- [**Open-sourcing CARBS: how we used our hyperparameter optimizer to scale up to a 70B-parameter language model**](https://imbue.com/research/70b-carbs/) - 超参数优化器在大规模模型训练中的应用与调优策略

### 6.3 模型后训练与评估

模型后训练（Post-Training）和评估是确保模型在实际应用中表现稳定、可靠的关键步骤。本节涵盖 AIOps 场景下的后训练技术、基于 Kubernetes 的评估框架以及基准测试生成方法。

- [**AIOps 后训练技术**](fine-tuning/aiops_post_training.md) - 面向智能运维场景的模型后训练技术与实践
- [**Kubernetes 模型评估框架**](fine-tuning/kubernetes_model_evaluation_framework.md) - 基于 K8s 的大模型评估框架设计与实现
- [**Kubernetes AIOps 基准测试生成框架**](fine-tuning/kubernetes_aiops_benchmark_generation_framework.md) - 自动化生成 AIOps 基准测试数据集的框架设计

---

## 7. 大模型推理

推理是大模型从实验室走向生产环境的“最后一公里”。本章聚焦于构建高性能、低延迟的推理系统，涵盖推理服务架构设计、KV Cache 优化、模型量化压缩等核心技术。通过深入分析 Mooncake 等先进架构及不同规模集群的部署策略，为企业级大模型服务的落地提供全面的技术指导。

### 7.1 推理系统架构设计

推理系统架构是大模型服务化的核心基础，直接决定了系统的性能、可扩展性和资源利用效率。现代推理系统需要在低延迟、高吞吐量和成本效益之间找到最佳平衡点，同时支持动态批处理、内存优化和多模型并发等高级特性。

- [**Mooncake 架构详解：以 KV 缓存为中心的高效 LLM 推理系统设计**](llm/Mooncake%20架构详解：以%20KV%20缓存为中心的高效%20LLM%20推理系统设计.md) - 新一代推理系统的架构创新与性能优化策略

### 7.2 模型部署与运维实践

模型部署与运维是将训练好的大模型转化为可用服务的关键环节，涉及模型格式转换、环境配置、服务监控和故障处理等多个方面。有效的部署策略能够显著降低运维成本，提高服务稳定性和用户体验。

- [**动手部署 ollama**](llm/ollama/README.md) - 轻量级本地大模型部署的完整实践指南

### 7.3 推理优化技术体系

推理优化技术体系是提升大模型推理性能的核心技术集合，包括算法优化、硬件加速、系统调优和架构设计等多个维度。通过系统性的优化策略，可以在保证模型精度的前提下，大幅提升推理速度、降低资源消耗并改善用户体验。

完整的 AI 推理优化技术文档系列，涵盖从小型到大型集群的推理优化策略：

- [**AI 推理优化技术文档导航**](inference/README.md) - 推理优化技术方案的完整导航，涵盖基础理论、技术选型、专业领域优化和实施运维的系统性指南
- [**背景与目标**](inference/01-背景与目标.md) - 大模型推理优化的技术背景分析，包括推理挑战、应用需求、发展趋势和核心研究目标
- [**集群规模分类与特征分析**](inference/02-集群规模分类与特征分析.md) - 小型、中型、大型集群的详细特征分析，配置要求和规模化部署策略
- [**核心推理优化技术深度解析**](inference/03-核心推理优化技术深度解析.md) - 模型压缩、并行计算、推测解码等核心优化技术的深度技术解析和实现指南
- [**不同集群规模的技术选型策略**](inference/04-不同集群规模的技术选型策略.md) - 针对不同规模集群的技术选型决策框架和最优配置策略
- [**性能评估指标体系**](inference/05-性能评估指标体系.md) - 推理服务性能评估的完整指标体系，包括吞吐量、延迟、资源利用率等关键指标
- [**推理服务架构设计**](inference/06-推理服务架构设计.md) - 推理服务的微服务架构设计，负载均衡、容错机制和高可用性保障
- [**实施建议与最佳实践**](inference/07-实施建议与最佳实践.md) - 分阶段实施策略、风险管理和生产环境部署的最佳实践指南
- [**参考资料与延伸阅读**](inference/08-参考资料与延伸阅读.md) - 推理优化相关的技术文档、开源项目和深度学习资源汇总
- [**安全性与合规性**](inference/09-安全性与合规性.md) - 推理服务的安全威胁分析、隐私保护机制和企业级合规要求
- [**多模态推理优化**](inference/10-多模态推理优化.md) - 文本、图像、音频等多模态融合的推理架构设计和跨模态注意力优化技术
- [**边缘推理优化**](inference/11-边缘推理优化.md) - 边缘设备适配、分布式边缘推理和实时推理优化的完整技术方案
- [**场景问题解答**](inference/12-场景问题解答.md) - 推理优化实施过程中的常见技术问题、故障排查和解决方案集合
- [**实施检查清单**](inference/13-实施检查清单.md) - 推理优化项目的分阶段实施检查清单和验收标准
- [**总结与展望**](inference/14-总结与展望.md) - 推理优化技术的发展总结和未来趋势分析，技术演进路线图

### 7.4 DeepSeek 专题

DeepSeek 是当前开源大模型领域的重要力量，其创新的架构设计和高性能表现备受关注。本节汇总了关于 DeepSeek 模型的部署、对比分析和存储系统设计等核心资料。

- [**DeepSeek-R1 模型对比分析**](deepseek/deepseek-r1-cmp.md) - 1.5b、7b、官网版本的性能对比与评测
- [**Mac 上运行 DeepSeek-R1 模型**](deepseek/mac-deepseek-r1.md) - 使用 Ollama 在 Mac 上本地部署 DeepSeek-R1
- [**DeepSeek 3FS 存储系统**](deepseek/deepseek_3fs_design_notes.zh-CN.md) - DeepSeek 自研的高性能分布式文件系统设计笔记
- [**DeepSeek 技术突破**](ai_infra_course/入门级/讲稿.md) - 在 AI Infra 课程中关于 DeepSeek 技术演进的深度解析

---

## 8. 企业级 AI Agent 开发

本章深入探讨企业级 `AI Agent` 开发的完整技术体系，从理论基础到工程实践，涵盖多智能体系统、上下文工程、记忆系统架构、RAG 技术等核心领域。通过系统性的技术框架和丰富的实践案例，帮助开发者构建高性能、可扩展的企业级智能体应用。

### 8.1 理论基础与开发概述

本节提供企业级 `AI Agent` 开发的全景视图，涵盖从基础理论到架构设计的完整技术体系。通过 BDI 架构、协作机制等核心概念的深入解析，为构建高性能、可扩展的智能体系统奠定坚实基础。

- [**AI Agent 开发与实践**](agent/README.md) - 企业级 AI Agent 开发的完整技术体系与最佳实践
- [**多智能体 AI 系统基础：理论与框架**](agent/Part1-Multi-Agent-AI-Fundamentals.md) - 多智能体系统的理论基础、BDI 架构和协作机制
- [**企业级多智能体 AI 系统构建实战**](agent/Part2-Enterprise-Multi-Agent-System-Implementation.md) - 企业级多智能体系统的架构设计、技术选型和工程实现

### 8.2 上下文工程技术体系

上下文工程是现代 AI Agent 开发的核心技术领域，代表了从传统提示工程向智能化上下文管理的技术演进。本节基于中科院权威研究成果，系统阐述上下文工程的理论基础、技术架构和工程实践，涵盖信息检索、智能选择、动态组装、自适应压缩等核心机制，为构建高效智能的企业级 AI 系统提供技术支撑。

**理论基础与核心原理：**

- [**上下文工程原理**](agent/context/上下文工程原理.md) - 基于中科院权威论文的系统性理论阐述与技术框架

  - **范式转变**：从传统提示工程到现代上下文工程的技术演进
  - **核心机制**：信息检索、智能选择、动态组装、自适应压缩和实时调整
  - **技术架构**：多模态信息融合、分布式状态管理、智能组装引擎
  - **企业应用**：全生命周期上下文管理和系统化自动优化策略

- [**上下文工程原理简介**](agent/context/上下文工程原理简介.md) - 面向开发者的深入浅出技术指南

  - **概念演进**：从简单聊天机器人到复杂智能助手的技术进化路径
  - **核心特征**：系统性方法论、动态优化算法、多模态融合、状态管理、智能组装
  - **技术对比**：与传统提示词工程的本质区别、优势分析和应用场景

- [**基于上下文工程的 LangChain 智能体应用**](agent/context/langchain_with_context_engineering.md) - LangChain 框架的上下文工程实践指南
  - **架构设计**：行为准则定义、信息接入策略、会话记忆管理、工具集成方案、用户画像构建
  - **技术实现**： LangChain 与 LangGraph 的深度集成与上下文工程最佳实践
  - **问题解决**：上下文污染检测、信息干扰过滤、语义混淆处理、冲突解决策略
  - **性能优化**：令牌消耗控制算法、成本效益分析、延迟优化技术

### 8.3 Agent 基础设施架构

Agent 基础设施是位于智能 Agent 之外、用于调节、协调、约束 Agent 与环境交互的一组技术系统与共享协议。本节深入探讨 Agent 基础设施的核心概念、架构设计原理和工程实践，涵盖可靠性、可扩展性、可观测性、成本可控性以及合规治理等关键技术领域。

- [**AI Agent 基础设施的崛起**](agent_infra/the-rise-of-ai-agent-infrastructure.md) - AI Agent 基础设施的发展趋势、技术演进和行业应用前景
- [**AI Agent 基础设施三层架构：工具、数据、编排**](agent_infra/ai-agent-infrastructure-three-layers-tools-data-orchestration.md) - AI Agent 基础设施的三层架构设计，涵盖工具层、数据层和编排层的技术实现

### 8.4 AI 智能体记忆系统架构

记忆系统是 AI 智能体实现长期学习和知识积累的关键技术组件。本节深入探讨智能体记忆系统的架构设计原理、存储策略优化、检索算法实现和多轮对话中的指代消解机制，为构建具备持续学习能力的智能体系统提供完整的技术方案。

#### 8.4.1 记忆系统理论基础

- [**AI 智能体记忆系统：理论与实践**](agent/memory/AI%20智能体记忆系统：理论与实践.md) - 智能体记忆系统的架构设计、存储策略与检索优化技术
  - [**记忆系统代码实现**](agent/memory/code/README.md) - 记忆系统的核心算法实现与工程化实践
- [**论文解读 - 大模型 Agent 记忆系统：理论基础与交互机制**](agent/memory/大模型Agent记忆综述.md) - 本文从理论角度深入探讨了大模型 Agent 记忆系统的定义、概念和交互机制，旨在帮助读者理解记忆系统的产生、存储和使用原理。
- [**如何设计支持多轮指代消解的对话系统**](agent/如何设计支持多轮指代消解的对话系统.md) - 多轮对话中的指代消解机制与上下文理解技术

#### 8.4.2 MemoryOS 智能记忆系统

MemoryOS 是一个智能记忆管理系统，采用模块化架构设计，支持可插拔存储后端和分层记忆管理机制。该系统为 AI 智能体提供了完整的记忆存储、检索和管理解决方案。

- [**MemoryOS 智能记忆系统架构设计与开发指南**](agent/memory/MemoryOS智能记忆系统架构设计与开发指南.md) - MemoryOS 智能记忆管理系统的模块化架构设计、可插拔存储后端和分层记忆管理机制

**核心技术特色：**

- **模块化架构**：采用可插拔的组件设计，支持灵活的存储后端配置
- **分层记忆管理**：实现短期、中期、长期记忆的分层存储和管理机制
- **多存储后端支持**：支持文件存储、向量数据库等多种存储方式
- **智能检索优化**：提供高效的记忆检索和相似性匹配算法

#### 8.4.3 Mem0 智能记忆系统

Mem0（发音为 "mem-zero"）是一个为大型语言模型（LLM）应用设计的自改进记忆层，通过提供持久化、个性化的记忆能力来增强 AI 助手和智能体的表现。该系统同时提供托管平台服务和开源解决方案，让开发者能够为 AI 应用添加上下文记忆功能。

- [**Mem0 快速入门指南**](agent/memory/mem0快速入门.md) - Mem0 记忆系统的完整技术指南，包含架构设计、API 接口、集成示例和性能对比

**核心技术特色：**

- **性能优势**：在 LOCOMO 基准测试中比 OpenAI Memory 准确率提升 26%，响应速度提升 91%，token 使用量降低 90%
- **分层架构**：采用模块化设计，支持向量存储、图数据库和多种 LLM 提供商的灵活配置
- **智能记忆处理**：通过事实提取、相似性检索和智能更新机制，实现自改进的记忆管理
- **企业级部署**：支持云端托管和本地自托管两种部署方式，满足不同安全和隐私需求

**应用场景：**

- **AI 助手**：提供一致、上下文丰富的对话体验
- **客户支持**：回忆历史工单和用户历史记录，提供个性化帮助
- **医疗保健**：跟踪患者偏好和历史记录，提供个性化护理
- **生产力和游戏**：基于用户行为自适应工作流程和环境

#### 8.4.4 记忆系统框架集成

- [**使用 LangChain 实现智能对话机器人的记忆功能**](agent/memory/langchain/langchain_memory.md) - 本文将深入探讨如何使用 LangChain 框架实现智能对话机器人的记忆功能，从 AI Agent 记忆系统的理论基础到 LangChain 的具体实现，再到实际应用案例。
  - [**LangChain 记忆功能演示代码**](agent/memory/langchain/code/README.md) - 完整的 LangChain 记忆功能实现示例，包含四种核心记忆类型演示、智能客服应用和现代 LangGraph 记忆管理：
    - **基础记忆类型演示**：ConversationBufferMemory、ConversationSummaryMemory、ConversationBufferWindowMemory、ConversationSummaryBufferMemory 四种核心记忆类型的完整实现
    - **智能客服机器人**：多用户会话管理、智能记忆类型选择、性能监控统计和会话持久化存储的企业级应用示例
    - **LangGraph 记忆管理**：基于状态图的现代记忆管理方案，支持持久化存储、自动总结和跨会话记忆保持
    - **多 LLM 支持**：兼容 OpenAI API、本地模型（Ollama）和其他 OpenAI 兼容服务的灵活配置
    - **交互式演示**：提供实时聊天界面和完整的配置管理，支持一键运行和调试

### 8.5 工程实践与应用案例

本节通过完整的企业级项目案例和系统性的培训体系，展示多智能体系统从理论设计到工程实现的全过程。涵盖 Docker 容器化部署、自动化测试、性能监控、LangGraph 工作流编排、LangSmith 全链路监控等企业级技术栈，为开发者提供可直接应用的工程实践指南。

#### 8.5.1 多智能体系统工程实践

- [**多智能体系统项目**](agent/multi_agent_system/) - 企业级多智能体系统的完整实现项目，包含 Docker 容器化部署、自动化测试用例和性能监控

#### 8.5.2 多智能体培训

- [**多智能体培训课程**](agent/multi_agent_training/) - 系统性的多智能体训练教程，包含理论基础、LangGraph 框架、LangSmith 监控、企业级架构和应用实践
- [**多智能体 AI 系统培训材料**](agent/multi_agent_training/README.md) - 5 天 40 学时的完整培训体系
  - [**多智能体系统概论**](agent/multi_agent_training/01-理论基础/01-多智能体系统概论.md) - BDI 架构、协作机制、系统优势
  - [**LangGraph 深度应用**](agent/multi_agent_training/02-LangGraph框架/02-LangGraph深度应用.md) - 工作流编排引擎深度应用
  - [**LangSmith 监控平台集成**](agent/multi_agent_training/03-LangSmith监控/03-LangSmith监控平台集成.md) - 全链路追踪、告警、性能优化
  - [**企业级系统架构设计与实现**](agent/multi_agent_training/04-企业级架构/04-企业级系统架构设计与实现.md) - 架构设计、技术实现、代码实践
  - [**应用实践与部署运维**](agent/multi_agent_training/05-应用实践/05-应用实践与部署运维.md) - 智能客服、部署、最佳实践

**培训特色：**

- **理论实践结合**：从抽象理论到具体实现的完整转化路径
- **技术栈全覆盖**： LangGraph 工作流编排 + LangSmith 全链路监控
- **企业级标准**：高可用性架构、安全机制、性能优化、运维最佳实践
- **完整项目案例**：智能客服系统、内容创作平台、金融分析系统

#### 8.5.3 应用案例

本节提供具体的 AI Agent 应用部署案例，从通用的科研助手到垂直领域的订单履约系统，展示企业级智能体应用的实际落地过程和最佳实践。

- [**Coze 部署和配置手册**](agent/Coze部署和配置手册.md) - Coze 平台的部署配置指南
- [**科研助手设计文档**](agent/scenario/科研助手.md) - 智能化科研辅助系统的架构设计与功能实现
- [**多源文献检索与聚合系统**](agent/scenario/多源文献检索与聚合系统设计文档.md) - 针对科研场景的智能文献检索与聚合系统设计
- [**订单履约 Agent 系统设计**](agent/scenario/订单履约Agent系统设计文档.md) - 电商领域订单履约智能体的系统架构与业务流程设计
- [**Building Research Agents for Tech Insights**](agent/scenario/《Building%20Research%20Agents%20for%20Tech%20Insights》深度解读.md) - 构建技术洞察研究智能体的深度解读

#### 8.5.4 数据科学智能体应用

数据科学智能体（Data Science Agent）是 Databricks 推出的革命性 AI 助手，专为数据科学家和分析师设计。通过自然语言交互，智能体能够自动生成和执行代码、探索数据、构建机器学习模型，并提供智能化的错误诊断和修复建议，显著提升数据科学工作流程的效率和质量。

- [**Databricks Data Science Agent 深度解析**](agent/scenario/data%20agent.md) - 数据科学智能体的功能特性、使用场景、产品设计评价与最佳实践指南
  - **核心功能**：数据探索与分析、机器学习开发、错误诊断与修复、结果总结与解释
  - **高级特性**：Planner 模式、Notebook 管理、企业级数据治理与安全
  - **产品创新**：渐进式交互设计、上下文感知能力、透明度与自动化的平衡
  - **企业应用**：工具确认机制、最佳实践建议、未来发展方向

### 8.6 技术架构与工具生态

本节系统介绍 AI Agent 开发的核心技术架构与工具生态。从 RAG 检索增强生成的深度优化，到 LangChain、Spring AI 等主流开发框架，再到 MCP、n8n 等新兴技术平台，为开发者提供从底层技术到上层应用的全栈技术指南。

#### 8.6.1 RAG 检索增强生成

检索增强生成（RAG）是现代 AI Agent 系统的核心技术之一，通过结合外部知识库和生成模型，显著提升智能体的知识获取和推理能力。

- [**RAG 技术概述**](llm/rag/README.md)
- [**从 0 到 1 快速搭建 RAG 应用**](llm/rag/写作%20Agentic%20Agent.md)
  - [**配套代码**](llm/rag/lession2.ipynb)
- [**Evaluating Chunking Strategies for Retrieval 总结**](llm/rag/Evaluating%20Chunking%20Strategies%20for%20Retrieval%20总结.md)
- [**中文 RAG 系统 Embedding 模型选型技术文档**](llm/rag/中文RAG系统Embedding模型选型技术文档.md)
- [**Agentic RAG 架构对比**](llm/rag/RAG%20对比.md) - Agentic RAG 与传统 RAG、Router 模式的深度对比分析

#### 8.6.2 AI Agent 框架与协议

本小节系统介绍 AI Agent 开发的主流框架和工具生态，涵盖 Python 和 Java 两大技术栈，以及 MCP 模型上下文协议。

**Python 生态：**

- [**LangChain + 模型上下文协议（MCP）： AI 智能体 Demo**](llm/agent/README.md)
- [**AI Agents for Beginners 课程之 AI 智能体及使用场景简介**](llm/AI%20Agents%20for%20Beginners%20课程之%20AI%20Agent及使用场景简介.md)
- [**LangGraph 实战：用 Python 打造有状态智能体**](llm/langgraph/langgraph_intro.md)
- [**使用 n8n 构建多智能体系统的实践指南**](llm/n8n_multi_agent_guide.md)
- [**开源大语言模型应用编排平台对比**](llm/开源大模型应用编排平台：Dify、AnythingLLM、Ragflow%20与%20n8n%20的功能与商用许可对比分析.md) - Dify、AnythingLLM、Ragflow 与 n8n 的功能与商用许可对比分析

**Java 生态：**

- [**使用 Spring AI 构建高效 LLM 代理**](java_ai/spring_ai_cn.md) - Spring AI 代理模式实现指南
  - **代理系统架构**：工作流 vs 代理的设计理念对比
  - **五种基本模式**：链式工作流、路由工作流、并行化、编排、评估
  - **企业级实践**：可预测性、一致性、可维护性的平衡
  - **技术实现**： Spring AI 的模型可移植性和结构化输出功能

**模型上下文协议 (MCP)：**

Model Context Protocol (MCP) 是一个开放标准，用于连接 AI 助手与外部数据源和工具。

- [**MCP 深度解析与 AI 工具未来**](llm/mcp/A_Deep_Dive_Into_MCP_and_the_Future_of_AI_Tooling_zh_CN.md)
- [**MCP SSE 客户端与服务端实战**](llm/agent/mcp-sse/README.md) - 基于 SSE 的 MCP 服务端与客户端开发实战
- [**从零构建 MCP 服务实践指南**](https://github.com/ForceInjection/markdown-mcp/blob/main/docs/blog-mcp-integration.md) - 完整的 MCP 服务开发与集成教程

### 8.7 前沿方法论与深度研究

本节聚焦 AI Agent 领域的前沿方法论与学术研究，涵盖 12-Factor Agents 开发原则、工作流技术综述及 DeepResearch 等深度研究智能体的架构解析。

#### 8.7.1 12-Factor Agents 方法论

12-Factor Agents 是一套经过验证的构建可靠 LLM 应用的原则体系，借鉴了著名的 12 Factor Apps 方法论。这套方法论为开发者提供了从理论到实践的完整指导，特别适用于需要达到生产级质量标准的 AI 应用开发。

- [**12-Factor Agents 完整指南**](agent/12-factor-agents-intro.md) - 构建可靠 LLM 应用的 12 个核心原则详解

**核心价值**：

- **渐进式采用**：可以逐步引入，不需要重写整个系统
- **完全控制**：开发者拥有对提示词、上下文、控制流的完全控制权
- **生产就绪**：每个要素都考虑了生产环境的实际需求

#### 8.7.2 前沿论文与深度分析

深度解读 AI Agent 领域的最新学术成果和技术突破。

- [**Agent Workflow 综述**](agent/paper/agent-workflow-survey.md) - AI Agent 工作流技术的全面梳理与发展趋势分析
- [**DeepResearch Agent 解析**](agent/paper/deepresearch-agent.md) - 深度研究智能体的架构设计与核心能力解析
- [**通义 DeepResearch 深度分析**](agent/通义DeepResearch深度分析.md) - 通义实验室 DeepResearch 技术的深度技术剖析
- [**Cursor IDE 架构概览**](llm/rag/cursor-deepsearch.md) - AI 驱动的代码编辑器 Cursor 的技术架构与实现原理
- [**Cursor ReAct Agent 深度分析**](llm/rag/react-agent.md) - Cursor IDE 中 ReAct Agent 的架构设计与核心机制解析

---

## 9. 实践案例

本章通过丰富的实践案例，展示 AI 技术在不同场景下的具体应用和实现方法。从模型部署推理到文档处理工具，再到特定领域的专业应用，为开发者提供可直接参考和复用的技术方案和最佳实践。

### 9.1 模型部署与推理

本节聚焦于大语言模型的实际部署和推理实践，通过具体的部署案例，展示如何在生产环境中高效部署和运行大模型服务，包括环境配置、性能优化、资源管理等关键技术环节。

- [**动手部署 ollama**](llm/ollama/README.md)

### 9.2 文档处理工具

文档处理是 AI 应用的重要场景之一，本节介绍多种先进的 AI 驱动文档处理工具和技术。涵盖 PDF 布局检测、复杂文档解析、多格式转换等核心功能，为构建智能文档处理系统提供完整的技术解决方案和工程实践指导。

- [**深入探索： AI 驱动的 PDF 布局检测引擎源代码解析**](llm/marker.zh-CN.md)
- [**上海人工智能实验室开源工具 MinerU 助力复杂 PDF 高效解析提取**](llm/minerU_intro.md)
- [**Markitdown 入门**](llm/markitdown/README.md)
- [**DeepWiki 使用方法与技术原理深度分析**](llm/DeepWiki%20使用方法与技术原理深度分析.md)

### 9.3 特定领域应用

本节展示 AI 技术在垂直领域的深度应用实践，包括中医古籍分析、法律合同审核、智能对话系统等专业场景。通过具体的应用案例，探讨如何针对特定领域的需求进行模型定制、数据处理和系统优化，为行业 AI 应用提供参考范例。

- [**中医古籍分析**](llm/scenario/traditional-chinese-medicine.md) - 关于 7b 模型阅读分析中医古籍能力的探讨与专项训练建议
- [**合同审核清单**](llm/scenario/中国大陆合同审核要点清单.md) - 中国大陆合同审核要点清单与 AI 辅助审核实践
- [**让用户"说半句"话也能懂： ChatBox 的意图识别与语义理解机制解析**](llm/ChatBox_Intent_Recognition_and_Semantic_Understanding_Half_Sentence.md)

---

## 10. 工具与资源生态

本章汇聚了 AI 领域最核心的工具、资源和技术生态，为开发者和研究者提供全方位的技术支撑体系。从系统性的学习资源到前沿的开源项目，从基础设施课程到实用工具选型，构建了完整的 AI 技术栈知识图谱。这些资源不仅涵盖理论基础，更注重工程实践，帮助读者快速掌握企业级 AI 系统开发的核心技能。

### 10.1 开源项目生态与技术选型

本节精选了 AI 领域最具价值的开源项目和技术方案，为企业级 AI 应用提供可靠的技术选型参考。涵盖大模型推理框架、文档处理工具、基础设施组件等多个技术领域，每个项目都经过实际验证，具备良好的社区支持和企业级应用案例。通过对比分析不同技术方案的优势和适用场景，帮助开发者做出最佳的技术选择。

#### 10.1.1 大模型与推理框架

本小节聚焦于大模型训练、微调和推理的核心框架技术，涵盖高性能中文大模型、高效微调工具和推理优化框架。这些项目代表了当前大模型技术的最新进展，在性能优化、内存效率和推理速度方面都有显著突破，为企业级大模型应用提供了强有力的技术支撑。

- [**DeepSeek**](https://github.com/DeepSeek-AI/) - 基于 Transformer 的高性能中文大模型，具备强大的推理能力与多语言支持
- [**unsloth**](https://github.com/unslothai/unsloth) - 高效大模型微调框架，支持 Llama 3.3、DeepSeek-R1 等模型 2 倍速度提升与 70% 内存节省
- [**ktransformers**](https://github.com/kvcache-ai/ktransformers) - 灵活的大模型推理优化框架，提供前沿的推理加速技术

#### 10.1.2 文档处理与数据预处理

本小节专注于文档智能化处理和数据预处理技术，提供从非结构化数据提取到高质量格式转换的完整解决方案。这些工具在 RAG 系统构建、知识库建设和文档智能化处理场景中发挥关键作用，支持多种文档格式的高精度解析和转换，为 AI 应用的数据准备阶段提供强大的技术支持。

- [**unstructured**](https://github.com/Unstructured-IO/unstructured) - 企业级非结构化数据处理库，支持自定义预处理流水线与机器学习数据准备
- [**MinerU**](https://github.com/opendatalab/MinerU) - 高质量 PDF 转换工具，支持 Markdown 和 JSON 格式输出，适用于文档智能化处理
- [**markitdown**](https://github.com/microsoft/markitdown) - Microsoft 开源的文档转换工具，支持多种办公文档格式到 Markdown 的高质量转换

---

## 11. 课程体系与学习路径

本章汇总了 AI 基础、系统开发、编程实战等全方位的课程体系，为学习者提供清晰的学习路径和进阶指南。

### 11.1 AI System 全栈课程（ZOMI 酱）

[**AISystem**](AISystem/README.md) - ZOMI 酱的 AI 系统全栈课程，涵盖从硬件基础到框架设计的全技术栈内容：

- [**系统介绍**](AISystem/01Introduction/README.md) - AI 系统概述、发展历程与技术演进路径
- [**硬件基础**](AISystem/02Hardware/README.md) - AI 芯片架构、硬件加速器与计算平台深度解析
- [**编译器技术**](AISystem/03Compiler/README.md) - AI 编译器原理、优化技术与工程实践
- [**推理优化**](AISystem/04Inference/README.md) - 模型推理加速技术、性能调优与部署策略
- [**框架设计**](AISystem/05Framework/README.md) - AI 框架架构设计、分布式计算与并行优化

### 11.2 AI Infra 基础课程（入门）

- [**大模型原理与最新进展**](ai_infra_course/入门级/index.html) - 交互式在线课程平台
- [**AI Infra 课程演讲稿**](ai_infra_course/入门级/讲稿.md) - 完整的课程演讲内容、技术要点与实践案例
- **学习目标**：深入理解大模型工作原理、最新技术进展与企业级应用实践
- **核心内容**：
  - **Transformer 架构深度解析**：编码器-解码器结构、多头注意力机制、文本生成过程
  - **训练规模与成本分析**： GPT-3/4、PaLM 等主流模型的参数量、训练成本和资源需求
  - **DeepSeek 技术突破**： V1/V2/R1 三代模型演进、MLA 架构创新、MoE 稀疏化优化
  - **能力涌现现象研究**：规模效应、临界点突破、多模态融合发展趋势
  - **AI 编程工具生态**： GitHub Copilot、Cursor、Trae AI 等工具对比分析与应用实践
  - **GPU 架构与 CUDA 编程**：硬件基础、并行计算原理、性能优化策略
  - **云原生 AI 基础设施**：现代化 AI 基础设施设计、容器化部署与运维实践

### 11.3 Trae 编程实战课程

**系统化的 Trae 编程学习体系：**

- [**Trae 编程实战教程**](trae/README.md) - 从基础入门到高级应用的完整 Trae 编程学习路径

**课程结构：**

- **第一部分：Trae 基础入门**：环境配置、交互模式、HelloWorld 项目实战
- **第二部分：常见编程场景实战**：前端开发、Web 开发、后端 API、数据库设计、安全认证
- **第三部分：高级应用场景**：AI 模型集成、实时通信、数据分析、微服务架构
- **第四部分：团队协作与最佳实践**：代码质量管理、项目管理、性能优化、DevOps 实践
- **第五部分：综合项目实战**：企业级应用开发、核心功能实现、部署运维实战

---

## Buy Me a Coffee!

如果您觉得本项目对您有帮助，欢迎购买我一杯咖啡，支持我继续创作和维护。

| **微信**                                                 | **支付宝**                                            |
| -------------------------------------------------------- | ----------------------------------------------------- |
| <img src="./img/weixinpay.JPG" alt="wechat" width="200"> | <img src="./img/alipay.JPG" alt="alipay" width="200"> |

---
