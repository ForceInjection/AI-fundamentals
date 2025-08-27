# AI Fundamentals

本仓库是一个全面的人工智能基础设施（`AI Infrastructure`）学习资源集合，涵盖从硬件基础到高级应用的完整技术栈。内容包括 GPU 架构与编程、`CUDA` 开发、大语言模型、`AI` 系统设计、性能优化、企业级部署等核心领域，旨在为 `AI` 工程师、研究人员和技术爱好者提供系统性的学习路径和实践指导。

> **适用人群**：`AI` 工程师、系统架构师、GPU 编程开发者、大模型应用开发者、技术研究人员。
> **技术栈**：`CUDA`、`GPU` 架构、`LLM`、`AI` 系统、分布式计算、容器化部署、性能优化。

## 1. 硬件与基础设施

### 1.1 硬件基础知识

- [**PCIe 知识大全**](https://mp.weixin.qq.com/s/dHvKYcZoa4rcF90LLyo_0A) - 深入理解 PCIe 总线架构、带宽计算和性能优化
- [**NVLink 入门**](https://mp.weixin.qq.com/s/fP69UEgusOa_X4ZKLo30ig) - NVIDIA 高速互连技术的原理与应用场景
- [**NVIDIA DGX SuperPOD：下一代可扩展的 AI 领导基础设施**](https://mp.weixin.qq.com/s/a64Qb6DuAAZnCTBy8g1p2Q) - 企业级 AI 超算集群的架构设计与部署实践

### 1.2 GPU 架构深度解析

在准备在 `GPU` 上运行的应用程序时，了解 `GPU` 硬件设计的主要特性并了解与 `CPU` 的相似之处和不同之处会很有帮助。本路线图适用于那些对 `GPU` 比较陌生或只是想了解更多有关 `GPU` 中计算机技术的人。不需要特定的并行编程经验，练习基于 `CUDA` 工具包中包含的标准 `NVIDIA` 示例程序。

- [**GPU 特性**](gpu_architecture/gpu_characteristics.md)
- [**GPU 内存**](gpu_architecture/gpu_memory.md)
- [**GPU Example: Tesla V100**](gpu_architecture/tesla_v100.md)
- [**GPUs on Frontera: RTX 5000**](gpu_architecture/rtx_5000.md)
- **练习**：
  - [**Exercise: Device Query**](gpu_architecture/exer_device_query.md)
  - [**Exercise: Device Bandwidth**](gpu_architecture/exer_device_bandwidth.md)

#### 1.2.1 GPU 架构和编程模型介绍

- [**GPU Architecture and Programming — An Introduction**](gpu_programming/gpu_programming_introduction.md) - `GPU` 架构与编程模型的全面介绍

#### 1.2.2 CUDA 核心技术

- [**深入理解 NVIDIA CUDA 核心（vs. Tensor Cores vs. RT Cores）**](cuda/cuda_cores_cn.md)

### 1.3 AI 基础设施架构

- [**高性能 GPU 服务器硬件拓扑与集群组网**](https://arthurchiao.art/blog/gpu-advanced-notes-1-zh/)
- [**NVIDIA GH200 芯片、服务器及集群组网**](https://arthurchiao.art/blog/gpu-advanced-notes-4-zh/)
- [**深度学习（大模型）中的精度**](https://mp.weixin.qq.com/s/b08gFicrKNCfrwSlpsecmQ)

### 1.4 AI 基础设施课程

**完整的AI基础设施技术课程体系：**

- [**在线课程演示**](ai_infra_course/index.html) - 交互式课程演示（包含37个页面的完整课程内容）

**课程内容概览：**

- **大模型原理与最新进展**：`Transformer` 架构、训练规模、`DeepSeek` 技术突破、能力涌现现象
- **AI 编程技术**：`GitHub Copilot`、`Cursor`、`Trae AI` 等工具对比，实际应用场景和效率数据
- **GPU 架构与 CUDA 编程**：`GPU vs CPU` 对比、`NVIDIA` 架构演进、`CUDA` 编程模型、性能优化
- **云原生与 AI Infra 融合**：推理优化技术、量化技术、`AIBrix` 架构、企业级部署实践
- **技术前沿与职业发展**：行业趋势分析、学习路径规划、职业发展建议

### 1.5 GPU 管理与虚拟化

**理论与架构：**

- [**GPU 虚拟化与切分技术原理解析**](gpu_manager/GPU虚拟化与切分技术原理解析.md) - 技术原理深入
- [**GPU 管理相关技术深度解析 - 虚拟化、切分及远程调用**](gpu_manager/GPU%20管理相关技术深度解析%20-%20虚拟化、切分及远程调用.md) - 全面的 GPU 管理技术指南
- [**第一部分：基础理论篇**](gpu_manager/第一部分：基础理论篇.md) - GPU 管理基础概念与理论
- [**第二部分：虚拟化技术篇**](gpu_manager/第二部分：虚拟化技术篇.md) - 硬件、内核、用户态虚拟化技术
- [**第三部分：资源管理与优化篇**](gpu_manager/第三部分：资源管理与优化篇.md) - GPU 切分与资源调度算法
- [**第四部分：实践应用篇**](gpu_manager/第四部分：实践应用篇.md) - 部署、运维、性能调优实践

**GPU 虚拟化解决方案：**

- [**HAMi GPU 资源管理完整指南**](gpu_manager/hami/hmai-gpu-resources-guide.md)

**运维工具与实践：**

- [**nvidia-smi 入门**](ops/nvidia-smi.md)
- [**nvtop 入门**](ops/nvtop.md)
- [**NVIDIA GPU XID 故障码解析**](https://mp.weixin.qq.com/s/ekCnhr3qrhjuX_-CEyx65g)
- [**NVIDIA GPU 卡之 ECC 功能**](https://mp.weixin.qq.com/s/nmZVOQAyfFyesm79HzjUlQ)
- [**查询 GPU 卡详细参数**](ops/DeviceQuery.md)
- [**Understanding NVIDIA GPU Performance: Utilization vs. Saturation (2023)**](https://arthurchiao.art/blog/understanding-gpu-performance/)
- [**GPU 利用率是一个误导性指标**](ops/GPU%20利用率是一个误导性指标.md)

### 1.6 分布式存储系统

**JuiceFS 分布式文件系统：**

- [**JuiceFS 文件修改机制分析**](juicefs/JuiceFS%20文件修改机制分析.md) - 分布式文件系统的修改机制深度解析
- [**JuiceFS 后端存储变更手册**](juicefs/JuiceFS%20后端存储变更手册.md) - JuiceFS 后端存储迁移和变更操作指南

### 1.7 DeepSeek 技术研究

> 注意：相关内容为 2025 年春节完成，需要审慎参考！

**模型对比与评测：**

- [**DeepSeek-R1 模型对比分析**](deepseek/deepseek-r1-cmp.md) - 1.5b、7b、官网版本的性能对比与评测
- [**Mac 上运行 DeepSeek-R1 模型**](deepseek/mac-deepseek-r1.md) - 使用 Ollama 在 Mac 上本地部署 DeepSeek-R1

**分布式系统设计：**

- [**3FS 分布式文件系统**](deepseek/deepseek_3fs_design_notes.zh-CN.md) - 高性能分布式文件系统的设计理念与技术实现
  - **系统架构**：集群管理器、元数据服务、存储服务、客户端四大组件
  - **核心技术**：RDMA 网络、CRAQ 链式复制、异步零拷贝 API
  - **性能优化**：FUSE 局限性分析、本地客户端设计、io_uring 启发的 API 设计

### 1.8 高性能网络与通信

#### 1.8.1 InfiniBand 网络技术

- [**InfiniBand 网络理论与实践**](InfiniBand/IB%20网络理论与实践.md) - 企业级高性能计算网络的核心技术栈
  - **技术特性**：亚微秒级延迟、200Gbps+ 带宽、RDMA 零拷贝传输
  - **应用场景**：大规模分布式训练、高频金融交易、科学计算集群
  - **架构优势**：硬件级卸载、CPU 旁路、内存直接访问
- [**InfiniBand 健康检查工具**](InfiniBand/health/README.md) - 网络健康状态监控和故障诊断
- [**InfiniBand 带宽监控**](InfiniBand/monitor/README.md) - 实时带宽监控和性能分析

#### 1.8.2 NCCL 分布式通信

- [**NCCL 分布式通信测试套件使用指南**](nccl/tutorial.md) - NVIDIA 集合通信库的深度技术解析
  - **核心算法**：AllReduce、AllGather、Broadcast、ReduceScatter 优化实现
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

### 1.9 云原生 AI 基础设施

#### 1.9.1 Kubernetes AI 生态

- [**Kubernetes AI 基础设施概述**](k8s/README.md) - 企业级容器化 AI 工作负载的编排管理平台
- [**Kueue + HAMi 集成方案**](k8s/Kueue%20+%20HAMi.md) - GPU 资源调度与管理的云原生解决方案
- [**NVIDIA Container Toolkit 原理分析**](k8s/Nvidia%20Container%20Toolkit%20原理分析.md) - 容器化 GPU 支持的底层机制
- [**NVIDIA K8s Device Plugin 分析**](k8s/nvidia-k8s-device-plugin-analysis.md) - GPU 设备插件的架构与实现

**核心特性：**

- **智能调度**：GPU 资源共享、NUMA 拓扑感知、多优先级调度策略
- **资源管理**：GPU Operator、Node Feature Discovery、MIG Manager 统一管理
- **可观测性**：Prometheus 指标采集、Grafana 可视化、Jaeger 链路追踪

#### 1.9.2 AI 推理服务

- [**云原生高性能分布式 LLM 推理框架 llm-d 介绍**](k8s/llm-d-intro.md) - 基于 Kubernetes 的大模型推理框架
- [**vLLM + LWS：Kubernetes 上的多机多卡推理方案**](k8s/lws_intro.md) - LWS 旨在提供一种 **更符合 AI 原生工作负载特点的分布式控制器语义**，填补现有原语在推理部署上的能力空白

**技术架构：**

- **服务治理**：Istio 服务网格、Envoy 代理、智能负载均衡
- **弹性伸缩**：HPA 水平扩展、VPA 垂直扩展、KEDA 事件驱动自动化
- **模型运营**：多版本管理、A/B 测试、金丝雀发布、流量切换

### 1.10 性能分析与调优

#### 1.10.1 AI 系统性能分析概述

- [**AI 系统性能分析**](profiling/README.md) - 企业级 AI 系统的全栈性能分析与瓶颈诊断

**分析维度：**

- **多维分析**：计算密集度、内存访问模式、网络通信效率、存储 I/O 性能
- **专业工具**：Nsight Systems 系统级分析、Nsight Compute 内核级优化、Intel VTune 性能调优
- **优化方法论**：算子融合策略、内存池化管理、计算通信重叠、数据流水线优化

#### 1.10.2 GPU 性能分析

- [**使用 Nsight Compute Tool 分析 CUDA 矩阵乘法程序**](https://www.yuque.com/u41800946/nquqpa/eo7gykiyhg8xi2gg)
- [**CUDA 内核性能分析指南**](profiling/s9345-cuda-kernel-profiling-using-nvidia-nsight-compute.pdf) - NVIDIA 官方 CUDA 内核性能分析详细指南

**性能分析工具：**

- **NVIDIA Nsight Compute**：CUDA 内核级性能分析器
- **NVIDIA Nsight Systems**：系统级性能分析器
- **nvprof**：传统 CUDA 性能分析工具

**关键指标与优化：**

- **硬件指标**：SM 占用率、内存带宽利用率、L1/L2 缓存命中率、Tensor Core 效率
- **内核优化**：CUDA Kernel 性能调优、内存访问模式优化、线程块和网格配置
- **分析工具**：CUDA Profiler 性能剖析、Nsight Graphics 图形分析、GPU-Z 硬件监控

**性能优化实践：**

- **全局内存访问模式优化**：提升内存访问效率
- **共享内存（Shared Memory）优化**：利用片上高速缓存
- **指令级并行（ILP）优化**：提升计算吞吐量
- **内存带宽利用率分析**：优化数据传输性能

### 1.11 GPU 监控与运维工具

#### 1.11.1 GPU 监控工具

- [**GPU 监控与运维工具概述**](ops/README.md) - 企业级 GPU 集群的全方位监控与运维解决方案
- [**nvidia-smi 详解**](ops/nvidia-smi.md) - NVIDIA 系统管理接口工具的深度使用指南与最佳实践
- [**nvtop 使用指南**](ops/nvtop.md) - 实时交互式 GPU 监控工具的高级应用
- [**DeviceQuery 工具**](ops/DeviceQuery.md) - CUDA 设备查询工具的完整功能解析

**核心特性：**

- **实时监控**：GPU 利用率、核心温度、功耗曲线、显存占用、PCIe 带宽
- **智能告警**：多级阈值告警、机器学习异常检测、故障预测与预警
- **数据可视化**：Grafana 多维仪表板、历史趋势分析、性能基线报告
- **运维自动化**：基础设施即代码、配置标准化、智能故障恢复

#### 1.11.2 GPU 性能分析

- [**GPU 利用率是一个误导性指标**](ops/GPU%20利用率是一个误导性指标.md) - 深入理解 GPU 利用率指标的局限性与替代方案

---

## 2. 开发与编程

本部分专注于AI开发相关的编程技术、工具和实践，涵盖从基础编程到高性能计算的完整技术栈。

### 2.1 AI 编程入门

- [**AI 编程入门完整教程**](ai_coding/AI%20编程入门.md) - 面向初学者的 AI 编程完整学习路径与实践指南
- [**AI 编程入门在线版本**](ai_coding/index.html) - 交互式在线学习体验与动手实践

**学习路径：**

- **理论基础**：机器学习核心概念、深度学习原理、神经网络架构设计
- **编程语言生态**：Python AI 生态、R 统计分析、Julia 高性能计算在 AI 中的应用
- **开发环境搭建**：Jupyter Notebook 交互式开发、PyCharm 专业 IDE、VS Code 轻量级配置

### 2.2 CUDA 编程与开发

- [**CUDA 核心概念详解**](cuda/cuda_cores_cn.md) - CUDA 核心、线程块、网格等基础概念的深度解析
- [**CUDA 流详解**](cuda/cuda_streams.md) - CUDA 流的原理、应用场景与性能优化
- [**GPU 编程基础**](gpu_programming/gpu_programming_introduction.md) - GPU 编程入门到进阶的完整技术路径

**技术特色：**

- **CUDA 核心架构**：SIMT 线程模型、分层内存模型、流式执行模型
- **性能调优实践**：内存访问模式优化、线程同步策略、算法并行化重构
- **高级编程特性**：Unified Memory 统一内存、Multi-GPU 多卡编程、CUDA Streams 异步执行

### 2.3 Trae 编程实战课程

**系统化的 Trae 编程学习体系：**

- [《Trae 编程实战》课程提纲](trae/《Trae%20编程实战》课程提纲（对外）.md) - 完整的五部分21章课程规划
  - **基础入门**：环境配置、交互模式、HelloWorld项目实战
  - **场景实战**：前端开发、后端API、数据库设计、安全认证
  - **高级应用**：AI集成、实时通信、数据分析、微服务架构
  - **团队协作**：代码质量、版本控制、CI/CD、性能优化
  - **综合项目**：企业级应用开发、部署运维实战

### 2.4 Java AI 开发

- [**Java AI 开发指南**](java_ai/README.md) - Java 生态系统中的 AI 开发技术
- [**使用 Spring AI 构建高效 LLM 代理**](java_ai/spring_ai_cn.md) - 基于 Spring AI 框架的企业级 AI 应用开发

**技术特色：**

- **企业级框架**：基于成熟的 Spring 生态系统
- **多提供商支持**：统一 API 集成 OpenAI、Azure OpenAI、Hugging Face 等
- **生产就绪**：提供完整的企业级 AI 应用解决方案
- **Java 原生**：充分利用 Java 生态系统的优势

### 2.4 CUDA 学习材料

#### 2.4.1 快速入门

- [**并行计算、费林分类法和 CUDA 基本概念**](https://mp.weixin.qq.com/s/NL_Bz8JB-LdAtrQake7EdA)
- [**CUDA 编程模型入门**](https://mp.weixin.qq.com/s/IUYzzgt6DUYhfaDnbxoZuQ)
- [**CUDA 并发编程之 Stream 介绍**](cuda/cuda_streams.md)

#### 2.4.2 参考资料

- [**CUDA Reading Group 相关讲座**](https://mp.weixin.qq.com/s/6sOrNzG0UeVBes8stWSoWA): [GPU Mode Reading Group](https://github.com/gpu-mode)
- [**《CUDA C++ Programming Guide》**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [**《CUDA C 编程权威指南》**](https://mp.weixin.qq.com/s/xJY5Znv3cuQi_UCd_XjJ4A)：[书中示例代码](https://github.com/Eddie-Wang1120/Professional-CUDA-C-Programming-Code-and-Notes)
- [**Nvidia 官方 CUDA 示例**](https://github.com/NVIDIA/cuda-samples)
- [**《CUDA 编程：基础与实践 by 樊哲勇》**](https://book.douban.com/subject/35252459/)
  - [**学习笔记**](https://github.com/QINZHAOYU/CudaSteps)
  - [**示例代码**](https://github.com/MAhaitao999/CUDA_Programming)
- [**《CUDA 编程简介: 基础与实践 by 李瑜》**](http://www.frankyongtju.cn/ToSeminars/hpc.pdf)
- [**《CUDA 编程入门》** - 本文改编自北京大学超算队 CUDA 教程讲义](https://hpcwiki.io/gpu/cuda/)
- [**Multi GPU Programming Models**](https://github.com/NVIDIA/multi-gpu-programming-models)
- [**CUDA Processing Streams**](https://turing.une.edu.au/~cosc330/lectures/display_lecture.php?lecture=22#1)

#### 2.4.3 专业选手

[**CUDA-Learn-Notes**](https://github.com/xlite-dev/CUDA-Learn-Notes)：📚Modern CUDA Learn Notes: 200+ Tensor/CUDA Cores Kernels🎉, HGEMM, FA2 via MMA and CuTe, 98~100% TFLOPS of cuBLAS/FA2.

---

## 3. 机器学习基础

本部分基于 [**动手学机器学习**](https://github.com/ForceInjection/hands-on-ML) 项目，提供系统化的机器学习学习路径。

### 3.1 机器学习学习资源

- [**动手学机器学习**](hands-on-ML/README.md) - 全面的机器学习学习资源库，包含理论讲解、代码实现和实战案例

**核心特色：**

- **理论与实践结合**：从数学原理到代码实现的完整学习路径
- **算法全覆盖**：监督学习、无监督学习、集成学习、深度学习等核心算法
- **项目驱动学习**：通过实际项目掌握机器学习的完整工作流程
- **工程化实践**：特征工程、模型评估、超参数调优等工程技能

### 3.2 基础概念与数学准备

- [**通俗理解机器学习核心概念**](hands-on-ML/nju_software/通俗理解机器学习核心概念.md)
- [**梯度下降算法：从直觉到实践**](hands-on-ML/nju_software/梯度下降算法：从直觉到实践.md)
- [**混淆矩阵评价指标**](hands-on-ML/nju_software/混淆矩阵评价指标.md)
- [**误差 vs. 残差**](hands-on-ML/nju_software/误差%20vs%20残差.md)
- [**线性代数的本质**](https://www.bilibili.com/video/BV1ys411472E) - 3Blue1Brown可视化教程
- [**MIT 18.06 线性代数**](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) - Gilbert Strang经典课程
- [**概率论与统计学基础**](https://book.douban.com/subject/35798663/) - 贝叶斯定理、概率分布、最大似然估计

### 3.2 监督学习

#### 3.2.1 基础算法

- [**KNN算法**](hands-on-ML/nju_software/ch-03/动手学机器学习%20KNN%20算法.md) - K近邻算法理论与实现
- [**线性回归**](hands-on-ML/nju_software/ch-03/动手学机器学习线性回归算法.md) - 最小二乘法、正则化
- [**逻辑回归**](hands-on-ML/nju_software/ch-03/动手学机器学习逻辑回归算法.md) - 分类算法基础
- [**决策树**](hands-on-ML/nju_software/ch-03/动手学机器学习决策树算法.md) - ID3、C4.5、CART算法
- [**支持向量机**](hands-on-ML/nju_software/ch-03/动手学机器学习支持向量机算法.md) - 核技巧与软间隔
- [**朴素贝叶斯**](hands-on-ML/nju_software/ch-03/动手学机器学习朴素贝叶斯算法.md) - 概率分类器

#### 3.2.2 集成学习

- [**随机森林**](hands-on-ML/nju_software/ch-04/动手学机器学习随机森林算法.md) - Bagging集成方法
- [**AdaBoost**](hands-on-ML/nju_software/ch-04/Adaboost%20计算示例.md) - Boosting算法
- [**GBDT**](hands-on-ML/nju_software/ch-04/一文读懂GBDT.md) - 梯度提升决策树
- [**Stacking**](hands-on-ML/nju_software/ch-04/Kaggle房价预测中的集成技巧.md) - 模型堆叠技术
- [**集成学习概述**](hands-on-ML/nju_software/ch-04/一文深入了解机器学习之集成学习.md) - 集成学习理论与方法

### 3.3 无监督学习

#### 3.3.1 聚类算法

- [**K-means聚类**](hands-on-ML/nju_software/ch-05/动手学机器学习%20Kmeans%20聚类算法.md) - 基础聚类算法
- [**层次聚类**](hands-on-ML/nju_software/ch-05/动手学机器学习层次聚类算法.md) - 凝聚与分裂聚类
- [**DBSCAN**](hands-on-ML/nju_software/ch-05/动手学机器学习：DBSCAN%20密度聚类算法.md) - 密度聚类算法

#### 3.3.2 降维算法

- [**PCA主成分分析**](hands-on-ML/nju_software/ch-11/PCA降维算法详解.md) - 线性降维方法
- [**LDA线性判别分析**](hands-on-ML/nju_software/ch-11/LDA降维算法详解.md) - 监督降维技术
- [**PCA vs LDA比较**](hands-on-ML/nju_software/ch-11/PCA%20vs.%20LDA%20降维方法比较.md) - 降维方法对比分析

#### 3.3.3 概率模型

- [**EM算法**](hands-on-ML/nju_software/ch-06/一文了解%20EM%20算法.md) - 期望最大化算法
- [**高斯混合模型**](hands-on-ML/nju_software/ch-06/一文了解%20GMM%20算法.md) - GMM聚类方法
- [**最大似然估计**](hands-on-ML/nju_software/ch-06/最大似然估计（MLE）简介.md) - MLE理论基础

### 3.4 特征工程与模型优化

#### 3.4.1 特征工程

- [**特征工程概述**](hands-on-ML/nju_software/ch-07/特征工程.md) - 数据预处理、特征选择与变换
- [**特征选择方法**](hands-on-ML/nju_software/ch-10/特征选择方法概述.md) - 过滤法、包装法、嵌入法
- [**GBDT特征提取**](hands-on-ML/nju_software/ch-07/GBDT特征提取.md) - 基于树模型的特征工程
- [**时间序列特征提取**](hands-on-ML/nju_software/ch-07/时间序列数据及特征提取.md) - 时间序列数据处理
- [**词袋模型**](hands-on-ML/nju_software/ch-07/词袋模型介绍.md) - 文本特征工程

#### 3.4.2 模型评估

- [**模型评估方法**](hands-on-ML/nju_software/ch-08/图解机器学习-模型评估方法与准则.md) - 评估指标与交叉验证
- [**混淆矩阵评价指标**](hands-on-ML/nju_software/混淆矩阵评价指标.md) - 分类模型性能评估
- [**GridSearchCV**](hands-on-ML/nju_software/ch-09/gridsearchcv_intro.md) - 超参数优化实践
- [**L1 L2正则化**](hands-on-ML/nju_software/ch-09/L1_L2_intro.md) - 正则化方法介绍
- [**SMOTE采样**](hands-on-ML/nju_software/ch-09/SMOTE%20介绍.md) - 不平衡数据处理

### 3.5 推荐系统与概率图模型

#### 3.5.1 推荐系统

- [**推荐系统入门**](hands-on-ML/nju_software/ch-12/recommendation_intro.md) - 推荐算法概述
- [**协同过滤算法**](hands-on-ML/nju_software/ch-12/协同过滤推荐算法：原理、实现与分析.md) - 用户协同过滤与物品协同过滤
- [**基于内容的推荐**](hands-on-ML/nju_software/ch-12/基于内容的推荐算法：原理与实践.md) - 内容推荐算法
- [**矩阵分解推荐**](hands-on-ML/nju_software/ch-12/基于矩阵分解的推荐算法：原理与实践.md) - SVD推荐算法
- [**关联规则挖掘**](hands-on-ML/nju_software/ch-12/使用%20Apriori%20算法进行关联分析：原理与示例.md) - Apriori算法

#### 3.5.2 概率图模型

- [**贝叶斯网络**](hands-on-ML/nju_software/ch-13/一文读懂贝叶斯网络.md) - 概率图模型基础
- [**隐马尔可夫模型**](hands-on-ML/nju_software/ch-13/一文读懂隐马尔可夫模型（HMM）.md) - 序列建模与状态推断
- [**马尔可夫模型**](hands-on-ML/nju_software/ch-13/马尔可夫模型简介.md) - 马尔可夫链基础

### 3.6 深度学习基础

- [**深度学习概述**](hands-on-ML/nju_software/ch-14/深度学习概述.md) - 深度学习理论与实践指南
- [**神经网络基础**](hands-on-ML/nju_software/ch-14/神经网络示例.md) - 感知机、多层感知机、反向传播
- [**什么是深度学习**](hands-on-ML/nju_software/ch-14/什么是深度学习？.md) - 深度学习入门介绍

### 3.7 实战项目

- [**泰坦尼克号幸存者预测**](hands-on-ML/nju_software/ch-03/使用决策树对泰坦尼克号幸存者数据进行分类.md) - 特征工程与分类实战
- [**朴素贝叶斯实例**](hands-on-ML/nju_software/ch-03/朴素贝叶斯计算：建筑工人打喷嚏后患感冒的概率.md) - 概率计算实例
- [**RFM用户分析**](hands-on-ML/nju_software/ch-07/数据探索-根据历史订单信息求RFM值.md) - 用户价值分析
- [**电影推荐系统**](hands-on-ML/nju_software/ch-12/movie-recommendation.ipynb) - 推荐算法实战

### 3.8 学习资源

#### 3.8.1 核心教材

- **《统计学习方法》** - 李航著，算法理论基础
- **《机器学习》** - 周志华著，西瓜书经典
- **《模式识别与机器学习》** - Bishop著，数学严谨

#### 3.8.2 在线资源

- [**机器学习考试复习提纲**](hands-on-ML/nju_software/机器学习考试复习提纲.md) - 考试重点总结
- [**梯度下降算法详解**](hands-on-ML/nju_software/梯度下降算法：从直觉到实践.md) - 优化算法理解
- [**机器学习核心概念**](hands-on-ML/nju_software/通俗理解机器学习核心概念.md) - 概念通俗解释
- [**Andrew Ng机器学习课程**](https://www.coursera.org/learn/machine-learning) - Coursera经典课程
- [**CS229机器学习**](http://cs229.stanford.edu/) - 斯坦福大学课程

#### 3.8.3 实践平台

- [**Kaggle**](https://www.kaggle.com/) - 数据科学竞赛平台
- [**Google Colab**](https://colab.research.google.com/) - 免费GPU环境
- [**scikit-learn**](https://scikit-learn.org/) - Python机器学习库

---

## 4. 大语言模型基础

### 4.1 核心技术与架构

**基础理论与概念：**

- [**Andrej Karpathy：Deep Dive into LLMs like ChatGPT（B站视频）**](https://www.bilibili.com/video/BV16cNEeXEer) - 深度学习领域权威专家的 LLM 技术解析
- [**大模型基础组件 - Tokenizer**](https://zhuanlan.zhihu.com/p/651430181) - 文本分词与编码的核心技术
- [**解密大语言模型中的 Tokens**](llm/token/llm_token_intro.md) - Token 机制的深度解析与实践应用
  - [**Tiktokenizer 在线版**](https://tiktokenizer.vercel.app/?model=gpt-4o) - 交互式 Token 分析工具

**嵌入技术与表示学习：**

- [**文本嵌入（Text-Embedding） 技术快速入门**](llm/embedding/text_embeddings_guide.md) - 文本向量化的理论基础与实践
- [**LLM 嵌入技术详解：图文指南**](llm/embedding/LLM%20Embeddings%20Explained%20-%20A%20Visual%20and%20Intuitive%20Guide.zh-CN.md) - 可视化理解嵌入技术
- [**大模型 Embedding 层与独立 Embedding 模型：区别与联系**](llm/embedding/embedding.md) - 嵌入层架构设计与选型策略

**高级架构与优化技术：**

- [**大模型可视化指南**](https://www.maartengrootendorst.com/) - 大模型内部机制的可视化分析
- [**一文读懂思维链（Chain-of-Thought, CoT）**](llm/一文读懂思维链（Chain-of-Thought,%20CoT）.md) - 推理能力增强的核心技术
- [**大模型的幻觉及其应对措施**](llm/大模型的幻觉及其应对措施.md) - 幻觉问题的成因分析与解决方案
- [**大模型文件格式完整指南**](llm/大模型文件格式完整指南.md) - 模型存储与部署的技术规范
- [**混合专家系统（MoE）图解指南**](llm/A%20Visual%20Guide%20to%20Mixture%20of%20Experts%20(MoE).zh-CN.md) - 稀疏激活架构的设计原理
- [**量化技术可视化指南**](llm/A%20Visual%20Guide%20to%20Quantization.zh-CN.md) - 模型压缩与加速的核心技术
- [**基于大型语言模型的意图检测**](llm/Intent%20Detection%20using%20LLM.zh-CN.md) - 自然语言理解的实际应用

### 4.2 参考书籍

- [**大模型基础**](https://github.com/ZJU-LLMs/Foundations-of-LLMs) <br>
 <img src="https://raw.githubusercontent.com/ZJU-LLMs/Foundations-of-LLMs/main/figure/cover.png" height="300"/>

- [**Hands-On Large Language Models**](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) <br>
 <img src="https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/images/book_cover.png" height="300"/>

- [**从零构建大模型**](https://mp.weixin.qq.com/s/FkBjsQmeXEPlsdFXETYSng)
- [**百面大模型**](https://mp.weixin.qq.com/s/rBJ5an0pr3TgjFbyJXa0WA)
- [**图解大模型：生成式AI原理与实践**](https://mp.weixin.qq.com/s/tYrHrpMrZySgWKE1ECqTWg)

---

## 5. 大模型训练

### 5.1 微调技术与训练策略

**指令微调与监督学习：**

- [**Qwen 2 大模型指令微调入门实战**](https://mp.weixin.qq.com/s/Atf61jocM3FBoGjZ_DZ1UA) - 基于 Qwen 2 的指令微调完整实践流程
- [**一文入门垂域模型 SFT 微调**](llm/一文入门垂域模型SFT微调.md) - 垂直领域模型的监督微调技术与应用实践

**大规模模型训练实践：**

- [**Training a 70B model from scratch: open-source tools, evaluation datasets, and learnings**](https://imbue.com/research/70b-intro/) - 70B 参数模型从零训练的完整技术路径与经验总结
- [**Sanitized open-source datasets for natural language and code understanding: how we evaluated our 70B model**](https://imbue.com/research/70b-evals/) - 大规模训练数据集的清洗、评估与质量控制方法
- [**From bare metal to a 70B model: infrastructure set-up and scripts**](https://imbue.com/research/70b-infrastructure/) - 大模型训练基础设施的搭建、配置与自动化脚本
- [**Open-sourcing CARBS: how we used our hyperparameter optimizer to scale up to a 70B-parameter language model**](https://imbue.com/research/70b-carbs/) - 超参数优化器在大规模模型训练中的应用与调优策略

---

## 6. 大模型推理

### 6.1 推理系统架构设计

- [**Mooncake 架构详解：以 KV 缓存为中心的高效 LLM 推理系统设计**](llm/Mooncake%20架构详解：以%20KV%20缓存为中心的高效%20LLM%20推理系统设计.md) - 新一代推理系统的架构创新与性能优化策略

### 6.2 模型部署与运维实践

- [**动手部署 ollama**](llm/ollama/README.md) - 轻量级本地大模型部署的完整实践指南

### 6.3 推理优化技术体系

完整的 AI 推理优化技术文档系列，涵盖从小型到大型集群的推理优化策略：

- [**AI 推理优化技术文档导航**](inference/README.md)
- [**背景与目标**](inference/01-背景与目标.md)
- [**集群规模分类与特征分析**](inference/02-集群规模分类与特征分析.md)
- [**核心推理优化技术深度解析**](inference/03-核心推理优化技术深度解析.md)
- [**不同集群规模的技术选型策略**](inference/04-不同集群规模的技术选型策略.md)
- [**性能评估指标体系**](inference/05-性能评估指标体系.md)
- [**推理服务架构设计**](inference/06-推理服务架构设计.md)
- [**实施建议与最佳实践**](inference/07-实施建议与最佳实践.md)
- [**参考资料与延伸阅读**](inference/08-参考资料与延伸阅读.md)
- [**安全性与合规性**](inference/09-安全性与合规性.md)
- [**多模态推理优化**](inference/10-多模态推理优化.md)
- [**边缘推理优化**](inference/11-边缘推理优化.md)
- [**场景问题解答**](inference/12-场景问题解答.md)
- [**实施检查清单**](inference/13-实施检查清单.md)
- [**总结与展望**](inference/14-总结与展望.md)

---

## 7. 企业级 AI Agent 开发

### 7.1 AI Agent 开发概述

- [**AI Agent 开发与实践**](agent/README.md) - 企业级 AI Agent 开发的完整技术体系与最佳实践

### 7.2 基础理论与架构框架

- [**多智能体AI系统基础：理论与框架**](agent/Part1-Multi-Agent-AI-Fundamentals.md) - 多智能体系统的理论基础、BDI 架构和协作机制
- [**企业级多智能体AI系统构建实战**](agent/Part2-Enterprise-Multi-Agent-System-Implementation.md) - 企业级多智能体系统的架构设计、技术选型和工程实现

### 7.3 上下文工程技术体系

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
  - **技术实现**：LangChain 与 LangGraph 的深度集成与上下文工程最佳实践
  - **问题解决**：上下文污染检测、信息干扰过滤、语义混淆处理、冲突解决策略
  - **性能优化**：令牌消耗控制算法、成本效益分析、延迟优化技术

### 7.4 AI 智能体记忆系统架构

- [**AI 智能体记忆系统：理论与实践**](agent/memory/AI%20智能体记忆系统：理论与实践.md) - 智能体记忆系统的架构设计、存储策略与检索优化技术
- [**如何设计支持多轮指代消解的对话系统**](agent/如何设计支持多轮指代消解的对话系统.md) - 多轮对话中的指代消解机制与上下文理解技术
- [**记忆系统代码实现**](agent/memory/code/README.md) - 记忆系统的核心算法实现与工程化实践

### 7.5 工程实践与项目案例

#### 7.5.1 多智能体系统工程实践

- [**多智能体系统项目**](agent/multi_agent_system/) - 企业级多智能体系统的完整实现项目，包含 Docker 容器化部署、自动化测试用例和性能监控

#### 7.5.2 多智能体训练

- [**多智能体训练课程**](agent/multi_agent_training/) - 系统性的多智能体训练教程，包含理论基础、LangGraph 框架、LangSmith 监控、企业级架构和应用实践
- [**多智能体AI系统培训材料**](agent/multi_agent_training/README.md) - 5天40学时的完整培训体系
- [**多智能体系统概论**](agent/multi_agent_training/01-理论基础/01-多智能体系统概论.md) - BDI架构、协作机制、系统优势
- [**LangGraph深度应用**](agent/multi_agent_training/02-LangGraph框架/02-LangGraph深度应用.md) - 工作流编排引擎深度应用
- [**LangSmith监控平台集成**](agent/multi_agent_training/03-LangSmith监控/03-LangSmith监控平台集成.md) - 全链路追踪、告警、性能优化
- [**企业级系统架构设计与实现**](agent/multi_agent_training/04-企业级架构/04-企业级系统架构设计与实现.md) - 架构设计、技术实现、代码实践
- [**应用实践与部署运维**](agent/multi_agent_training/05-应用实践/05-应用实践与部署运维.md) - 智能客服、部署、最佳实践

**培训特色：**

- **理论实践结合**：从抽象理论到具体实现的完整转化路径
- **技术栈全覆盖**：LangGraph工作流编排 + LangSmith全链路监控
- **企业级标准**：高可用性架构、安全机制、性能优化、运维最佳实践
- **完整项目案例**：智能客服系统、内容创作平台、金融分析系统

### 7.6 应用案例

- [**Coze 部署和配置手册**](agent/Coze部署和配置手册.md) - Coze 平台的部署配置指南

### 7.7 RAG 技术

- [**RAG 技术概述**](llm/rag/README.md)
- [**从 0 到 1 快速搭建 RAG 应用**](llm/rag/写作%20Agentic%20Agent.md)
  - [**配套代码**](llm/rag/lession2.ipynb)
- [**Evaluating Chunking Strategies for Retrieval 总结**](llm/rag/Evaluating%20Chunking%20Strategies%20for%20Retrieval%20总结.md)
- [**中文 RAG 系统 Embedding 模型选型技术文档**](llm/rag/中文RAG系统Embedding模型选型技术文档.md)

### 7.8 AI Agent 框架与工具

**Python 生态：**

- [**LangChain + 模型上下文协议（MCP）：AI 智能体 Demo**](llm/agent/README.md)
- [**AI Agents for Beginners 课程之 AI 智能体及使用场景简介**](llm/AI%20Agents%20for%20Beginners%20课程之%20AI%20Agent及使用场景简介.md)
- [**MCP 深度解析与 AI 工具未来**](llm/mcp/A_Deep_Dive_Into_MCP_and_the_Future_of_AI_Tooling_zh_CN.md)
- [**LangGraph 实战：用 Python 打造有状态智能体**](llm/langgraph/langgraph_intro.md)
- [**使用 n8n 构建多智能体系统的实践指南**](llm/n8n_multi_agent_guide.md)
- [**开源大语言模型应用编排平台：Dify、AnythingLLM、Ragflow 与 n8n 的功能与商用许可对比分析**](llm/开源大模型应用编排平台：Dify、AnythingLLM、Ragflow%20与%20n8n%20的功能与商用许可对比分析.md)

**Java 生态：**

- [**使用 Spring AI 构建高效 LLM 代理**](java_ai/spring_ai_cn.md) - Spring AI 代理模式实现指南
  - **代理系统架构**：工作流 vs 代理的设计理念对比
  - **五种基本模式**：链式工作流、路由工作流、并行化、编排、评估
  - **企业级实践**：可预测性、一致性、可维护性的平衡
  - **技术实现**：Spring AI 的模型可移植性和结构化输出功能

### 7.9 模型上下文协议（MCP）

- [**MCP 深度解析与 AI 工具未来**](llm/mcp/A_Deep_Dive_Into_MCP_and_the_Future_of_AI_Tooling_zh_CN.md)

---

## 8. 实践案例

### 8.1 模型部署与推理

- [**动手部署 ollama**](llm/ollama/README.md)

### 8.2 文档处理工具

- [**深入探索：AI 驱动的 PDF 布局检测引擎源代码解析**](llm/marker.zh-CN.md)
- [**上海人工智能实验室开源工具 MinerU 助力复杂 PDF 高效解析提取**](llm/minerU_intro.md)
- [**Markitdown 入门**](llm/markitdown/README.md)
- [**DeepWiki 使用方法与技术原理深度分析**](llm/DeepWiki%20使用方法与技术原理深度分析.md)

### 8.3 特定领域应用

- [**读者来信：请问 7b 阅读分析不同中医古籍的能力怎么样？可以进行专项训练大幅度提高这方面能力么？**](llm/scenario/traditional-chinese-medicine.md)
- [**中国大陆合同审核要点清单**](llm/scenario/中国大陆合同审核要点清单.md)
- [**让用户"说半句"话也能懂：ChatBox 的意图识别与语义理解机制解析**](llm/ChatBox_Intent_Recognition_and_Semantic_Understanding_Half_Sentence.md)

---

## 9. 工具与资源生态

### 9.1 AI 系统学习资源与知识体系

[**AISystem**](AISystem/README.md) - 企业级 AI 系统学习的完整知识体系与技术栈，涵盖：

- [**系统介绍**](AISystem/01Introduction/README.md) - AI 系统概述、发展历程与技术演进路径
- [**硬件基础**](AISystem/02Hardware/README.md) - AI 芯片架构、硬件加速器与计算平台深度解析
- [**编译器技术**](AISystem/03Compiler/README.md) - AI 编译器原理、优化技术与工程实践
- [**推理优化**](AISystem/04Inference/README.md) - 模型推理加速技术、性能调优与部署策略
- [**框架设计**](AISystem/05Framework/README.md) - AI 框架架构设计、分布式计算与并行优化

### 9.2 AI 基础设施专业课程体系

- [**大模型原理与最新进展**](ai_infra_course/index.html) - 交互式在线课程平台
- [**AI Infra 课程演讲稿**](ai_infra_course/讲稿.md) - 完整的课程演讲内容、技术要点与实践案例
- **学习目标**：深入理解大模型工作原理、最新技术进展与企业级应用实践
- **核心内容**：
  - **Transformer 架构深度解析**：编码器-解码器结构、多头注意力机制、文本生成过程
  - **训练规模与成本分析**：GPT-3/4、PaLM 等主流模型的参数量、训练成本和资源需求
  - **DeepSeek 技术突破**：V1/V2/R1 三代模型演进、MLA 架构创新、MoE 稀疏化优化
  - **能力涌现现象研究**：规模效应、临界点突破、多模态融合发展趋势
  - **AI 编程工具生态**：GitHub Copilot、Cursor、Trae AI 等工具对比分析与应用实践
  - **GPU 架构与 CUDA 编程**：硬件基础、并行计算原理、性能优化策略
  - **云原生 AI 基础设施**：现代化 AI 基础设施设计、容器化部署与运维实践

### 9.3 开源项目生态与技术选型

**大模型与推理框架：**

- [**DeepSeek**](https://github.com/DeepSeek-AI/) - 基于 Transformer 的高性能中文大模型，具备强大的推理能力与多语言支持
- [**unsloth**](https://github.com/unslothai/unsloth) - 高效大模型微调框架，支持 Llama 3.3、DeepSeek-R1 等模型 2 倍速度提升与 70% 内存节省
- [**ktransformers**](https://github.com/kvcache-ai/ktransformers) - 灵活的大模型推理优化框架，提供前沿的推理加速技术

**文档处理与数据预处理：**

- [**unstructured**](https://github.com/Unstructured-IO/unstructured) - 企业级非结构化数据处理库，支持自定义预处理流水线与机器学习数据准备
- [**MinerU**](https://github.com/opendatalab/MinerU) - 高质量 PDF 转换工具，支持 Markdown 和 JSON 格式输出，适用于文档智能化处理
- [**markitdown**](https://github.com/microsoft/markitdown) - Microsoft 开源的文档转换工具，支持多种办公文档格式到 Markdown 的高质量转换

---

## Buy Me a Coffee

如果您觉得本项目对您有帮助，欢迎购买我一杯咖啡，支持我继续创作和维护。

|**微信**|**支付宝**|
|---|---|
|<img src="./img/weixinpay.JPG" alt="wechat" width="200">|<img src="./img/alipay.JPG" alt="alipay" width="200">|

---
