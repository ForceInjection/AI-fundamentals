# AI Fundamentals

本仓库是一个全面的人工智能基础设施（AI Infrastructure）学习资源集合，涵盖从硬件基础到高级应用的完整技术栈。内容包括 GPU 架构与编程、CUDA 开发、大语言模型、AI 系统设计、性能优化、企业级部署等核心领域，旨在为 AI 工程师、研究人员和技术爱好者提供系统性的学习路径和实践指导。

> - **适用人群**：AI 工程师、系统架构师、GPU 编程开发者、大模型应用开发者、技术研究人员。
> - **技术栈**：CUDA、GPU 架构、LLM、AI 系统、分布式计算、容器化部署、性能优化。

---

**Star History**:

## ![Star History Chart](https://api.star-history.com/svg?repos=ForceInjection/AI-fundermentals&type=date&legend=top-left)

---

## 1. 硬件架构与互连技术

涵盖单机基础计算芯片（GPU、TPU）设计原理，PCIe、NVLink 高速互连总线协议，GPUDirect 跨节点直通技术，以及 NVIDIA GB300 NVL72 等异构融合超级芯片的系统级架构与延迟金字塔模型。详细内容请访问：**[硬件架构与互连技术](01_hardware_architecture/README.md)**。

- **基础计算芯片架构**
  - [深入理解 GPU 架构](./01_hardware_architecture/nvidia/understand_gpu_architecture/README.md)
  - [TPU 101：深度学习专用加速器架构解析](./01_hardware_architecture/tpu/tpu%20101.md)
  - [GPGPU vs NPU：大模型推理训练对比](./01_hardware_architecture/nvidia/GPGPU_vs_NPU_大模型推理训练对比.md)
- **高速互连与数据传输技术**
  - [PCIe 总线技术大全](./01_hardware_architecture/pcie/01_pcie_comprehensive_guide.md)
  - [Linux PCIe P2PDMA 技术介绍](./01_hardware_architecture/pcie/02_p2pdma_technology.md)
  - [NVLink 技术入门](./01_hardware_architecture/nvlink/nvlink_intro.md)
  - [NVIDIA GPUDirect P2P 技术详解：节点内 GPU 高速互联](./01_hardware_architecture/gpudirect/02_gpudirect_p2p.md)
  - [NVIDIA GPUDirect RDMA 与 Storage 技术详解](./01_hardware_architecture/gpudirect/01_gpudirect_technology.md)
- **异构融合架构与系统性能评估**
  - [NVLink-C2C：芯片级高速互连技术详解](./01_hardware_architecture/superchips/nvlink_c2c.md)
  - [NVIDIA GB300 NVL72：机架级计算系统架构解析](./01_hardware_architecture/superchips/nvidia_gb300.md)
  - [AI 基础设施延迟金字塔](./01_hardware_architecture/performance/ai_latency_pyramid.md)

---

## 2. AI 集群运维与高性能通信

构建高吞吐 AI 计算集群的完整运维体系，涵盖基于 Device Query、nvidia-smi 和 nvtop 的 GPU 状态监控，InfiniBand (IB) 网络架构与健康检查，以及 NCCL 分布式通信库的基准测试与多节点部署实战。详细内容请访问：**[AI 集群运维与通信](03_ai_cluster_ops/README.md)**。

- **GPU 基础运维**
  - [设备查询：Device Query](./03_ai_cluster_ops/01_gpu_ops/01_device_query.md)
  - [误区解读：GPU 利用率指标分析](./03_ai_cluster_ops/01_gpu_ops/02_gpu_utilization_myth.md)
  - [状态监控：nvidia-smi 指南](./03_ai_cluster_ops/01_gpu_ops/03_nvidia_smi_guide.md)
  - [状态监控：nvtop 指南](./03_ai_cluster_ops/01_gpu_ops/04_nvtop_guide.md)
- **InfiniBand 高性能网络**
  - [理论基础：IB 网络架构与协议](./03_ai_cluster_ops/02_infiniband/01_ib_network_theory.md)
  - [网络运维：健康检查与性能监控实战](./03_ai_cluster_ops/02_infiniband/README.md)
- **NCCL 分布式通信测试**
  - [理论基础：NCCL 教程](./03_ai_cluster_ops/03_nccl/01_nccl_tutorial.md)
  - [实战指南：基准测试与多节点部署](./03_ai_cluster_ops/03_nccl/README.md)

---

## 3. 云原生 AI 基础设施

基于 Kubernetes 的 AI 基础设施构建方案，涵盖 NVIDIA Container Toolkit 与 Device Plugin 底层机制、Kueue/HAMi 细粒度 GPU 资源切分与池化、LWS/llm-d 分布式推理调度，以及 JuiceFS、DeepSeek 3FS 等高性能分布式存储系统的架构实践。详细内容请访问：**[云原生 AI 平台](04_cloud_native_ai_platform/README.md)**。

### 3.1 Kubernetes AI 基础设施

解析 Kubernetes AI 场景核心组件，包括容器运行时 GPU 支持底层机制、设备插件源码分析、Kueue 调度整合，以及基于 LWS 的大模型分布式训练与推理架构。

- [Kubernetes GPU 管理与 AI 工作负载](./04_cloud_native_ai_platform/k8s/README.md)：云原生 AI 基础设施建设指南与技术导图
- [NVIDIA Container Toolkit 原理](./04_cloud_native_ai_platform/k8s/01_nvidia_container_toolkit_analysis.md)：容器使用 GPU 的底层机制深度解析
- [Device Plugin 原理](./04_cloud_native_ai_platform/k8s/02_nvidia_k8s_device_plugin_analysis.md)：Kubernetes 设备插件机制源码分析
- [Kueue + HAMi 调度方案](./04_cloud_native_ai_platform/k8s/03_kueue_hami_integration.md)：云原生作业队列与细粒度 GPU 共享机制
- [LWS (Leader Worker Set) 介绍](./04_cloud_native_ai_platform/k8s/04_lws_intro.md)：Kubernetes 原生的大模型分布式训练与推理调度抽象
- [分布式推理框架](./04_cloud_native_ai_platform/k8s/05_llm_d_intro.md)：基于 Kubernetes 的 LLM 推理架构设计
- [Containerd 日志分析](./04_cloud_native_ai_platform/k8s/06_containerd_log_analysis.md)：云原生容器运行时的日志排查与分析

### 3.2 GPU 资源管理与虚拟化

提供异构算力环境下的 GPU 资源精细化管理方案，涵盖硬件级/内核态/用户态虚拟化机制、CUDA 流与 MPS 调度优化，并提供 HAMi 资源隔离与 Flex AI 的生产环境落地配置。

**基础系列文档**：

- [第一部分：基础理论篇](./04_cloud_native_ai_platform/gpu_manager/第一部分：基础理论篇.md)：构建技术认知框架，解析传统模式局限性与核心技术体系
- [第二部分：虚拟化技术篇](./04_cloud_native_ai_platform/gpu_manager/第二部分：虚拟化技术篇.md)：深入剖析硬件级、内核态与用户态虚拟化的核心实现机制
- [第三部分：资源管理与优化篇](./04_cloud_native_ai_platform/gpu_manager/第三部分：资源管理与优化篇.md)：探讨 GPU 切分、CUDA 流及 MPS 等高效资源调度与优化策略
- [第四部分：实践应用篇](./04_cloud_native_ai_platform/gpu_manager/第四部分：实践应用篇.md)：涵盖环境部署、监控运维及云平台集成的生产落地指南

**HAMi 专题**：

- [HAMi 资源管理使用手册](./04_cloud_native_ai_platform/gpu_manager/hami/hmai-gpu-resources-guide.md)：异构算力管理与隔离实战指南
- [HAMi Prometheus 监控指标](./04_cloud_native_ai_platform/gpu_manager/hami/hami-prometheus-metrics.md)：构建完善的 GPU 虚拟化可观测性体系
- [KAI vs HAMi 对比分析](./04_cloud_native_ai_platform/gpu_manager/hami/KAI_vs_HAMi_Comparison.md)：深度对比原生 Kubernetes AI 调度器与 HAMi 方案
- [Flex AI 介绍](./04_cloud_native_ai_platform/gpu_manager/hami/flex_ai_intro.md)：探讨灵活异构算力环境下的前沿实践

**代码实现与配置**：

- [完整实现代码](./04_cloud_native_ai_platform/gpu_manager/code/)：GPU 调度器、虚拟化拦截与远程调用的参考实现代码
- [配置文件集合](./04_cloud_native_ai_platform/gpu_manager/configs/)：提供适用于生产环境和多云平台的完整部署与配置参考

### 3.3 高性能分布式存储

针对 AI 训练中海量小文件读取与跨节点共享的性能瓶颈，解析 JuiceFS 数据与元数据分离架构、DeepSeek 3FS 高性能设计及面向推理的 ICMS (KV Cache) 存储层机制。

- [JuiceFS 分布式文件系统](./04_cloud_native_ai_platform/storage/juicefs/README.md)：数据与元数据分离的架构设计，兼容 POSIX 接口
  - [文件修改机制分析](./04_cloud_native_ai_platform/storage/juicefs/01_juicefs_file_modification_mechanism_analysis.md)：底层数据一致性与写入流程解析
  - [后端存储变更手册](./04_cloud_native_ai_platform/storage/juicefs/02_juicefs_backend_storage_migration_guide.md)：生产环境下的存储运维与数据迁移指南
- [DeepSeek 3FS 设计笔记](./04_cloud_native_ai_platform/storage/deepseek_3fs/01_deepseek_3fs_design_notes.md)：高性能存储系统架构设计与特性分析
- [NVIDIA ICMS 架构解析](./04_cloud_native_ai_platform/storage/inference_context_memory_storage/01_icms_architecture.md)：面向推理的 KV Cache 存储层架构深度解析

---

## 4. 底层计算与异构编程

系统级 AI 底层编程路径，剖析 GPU 并行架构、CUDA 线程/网格与流处理机制、SIMT 与 Tile-Based (TileLang) 编程模型对比，以及基于 DOCA 框架的数据处理单元 (DPU) 核心编程范式。

### 4.1 GPU 与 CUDA 编程

涵盖 NVIDIA 容器镜像构建、CUDA 线程块/网格与流并发机制、SIMT 与 Tile-Based 编程模型对比、TileLang 算子开发，以及 nvbandwidth 显存与 PCIe 带宽调优实战，并链接 200+ Tensor Core/CUDA Core 优化内核的进阶学习资源。详细内容请访问：[GPU 编程基础](02_gpu_programming/README.md)。

**开发环境配置**：

- [NVIDIA 容器环境配置](./02_gpu_programming/01_environment/01_nvidia_container_setup.md)：NVIDIA Container Toolkit 原理与构建指南
- [CUDA 镜像构建分析](./02_gpu_programming/01_environment/02_cuda_image_build_analysis.md)：大模型训练与推理框架的 GPU 镜像构建深度解析

**核心编程范式**：

- [GPU 编程入门指南](./02_gpu_programming/02_cuda/01_gpu_programming_introduction.md)：并行计算基础与 CUDA 编程模型
- [CUDA 核心概念详解](./02_gpu_programming/02_cuda/02_cuda_cores.md)：线程块、网格等基础概念的深度解析
- [CUDA 流详解](./02_gpu_programming/02_cuda/03_cuda_streams.md)：CUDA 并发编程之流处理机制
- [SIMT vs Tile-Based 编程模型对比](./02_gpu_programming/02_cuda/04_simt_vs_tile_based.md)：架构差异与演进分析

**Tile-Based 编程**：

- [TileLang 快速入门](./02_gpu_programming/03_tilelang/01_tilelang_quick_start.md)：语法详解、算子开发实战与性能优化技巧

**性能分析与调优**：

- [nvbandwidth 最佳实践](./02_gpu_programming/04_profiling/01_nvbandwidth_best_practices.md)：显存带宽与 PCIe 传输带宽测量指南

**进阶学习资源**：

- [CUDA-Learn-Notes](https://github.com/xlite-dev/CUDA-Learn-Notes)：涵盖 200+ 个 Tensor Core/CUDA Core 极致优化内核示例 (HGEMM, FA2 via MMA and CuTe)
- [Nvidia 官方 CUDA 示例](https://github.com/NVIDIA/cuda-samples)：官方标准范例库
- [Multi GPU Programming Models](https://github.com/NVIDIA/multi-gpu-programming-models)：多卡编程模型示例

### 4.2 DPU 编程

基于 DOCA 框架的数据处理单元 (DPU) 开发指南，解析架构组件与典型网络加速场景的编程实践。详细内容请访问：[DPU 编程](02_dpu_programming/README.md)。

- **DOCA 框架**
  - [DOCA 编程入门](./02_dpu_programming/doca/01_doca_programming_guide.md)：涵盖架构简介、核心组件及典型场景编程实践

---

## 5. 大语言模型应用开发与编排

探索以自然语言驱动与 Agent 自主决策为核心的 Software 3.0 开发范式，包含 OpenSpec 规范驱动开发、Spring AI 企业级 Java 接入、LangGraph 有状态多智能体图计算模型，以及 Coze/n8n 无代码工作流编排技术。详细的深度探讨可参考 [大模型编程指南](98_llm_programming/README.md)。

### 5.1 AI 时代的软件工程：范式转移与重构

软件工程正向以 Agent First 与自主推理为核心的 **Software 3.0** 时代演进。本节梳理驾驭工程 (Harness Engineering) 体系，并演示基于 OpenSpec 的“意图 -> Spec -> AI -> 代码 & 验证”新一代规范驱动开发工作流。

- [Agent First：软件工程的下一个范式转移](98_llm_programming/Agent_First.md) - 梳理编程范式的演变历史，探讨 Agent First 的核心理念与实战指南。
- [驾驭工程](98_llm_programming/Harness_Engineering.md) - 深度解析如何构建驾驭系统，提升 AI 编程助手的可控性与效能。
- [OpenSpec 实战指南](https://github.com/ForceInjection/OpenSpec-practise/blob/main/README.md) - Spec 驱动开发 (Spec-Driven Development) 的工程实践，演示了“意图 -> Spec -> AI -> 代码 & 验证”的新一代开发工作流。

### 5.2 Java AI 开发

剖析 Java 生态 AI 开发技术栈，通过 Spring AI 工程框架实现企业级 Java 应用对 LLM 能力的接入，并演示基于 Spring AI 构建高效 LLM 代理的工程实践。

- [Java AI 开发指南](98_llm_programming/java_ai/README.md) - Java 生态系统中的 AI 开发技术总览。
- [使用 Spring AI 构建高效 LLM 代理](98_llm_programming/java_ai/spring_ai_cn.md) - 基于 Spring AI 框架的企业级 AI 应用开发实践。

### 5.3 LangGraph 开发

LangGraph 通过图计算模型解决 LLM 应用在循环逻辑与状态持久化上的瓶颈，提供状态机机制以支持多轮推理、自我反思的复杂 Agent 工作流构建（如 AI 客服系统 Notebook 实战）。

- [LangGraph 框架学习资源](98_llm_programming/langgraph/README.md) - LangGraph 框架的学习资源与实践案例总览。
- [LangGraph 简介](98_llm_programming/langgraph/langgraph_intro.md) - LangGraph 的核心概念与入门指南。
- [AI 客服系统实战](98_llm_programming/langgraph/aics.ipynb) - 基于 LangGraph 构建的 AI 客服系统 Notebook 实战。

### 5.4 AI 工作流与编排

无代码/低代码（No-Code/Low-Code）AI 应用落地指南，包含 Coze 私有化部署配置、n8n 多智能体编排实践，以及 Dify、Ragflow 等主流开源编排平台的架构与商业许可对比。

- [Coze 部署和配置手册](06_llm_theory_and_fundamentals/workflow/coze_deployment_and_configuration_guide.md) - Coze 平台的私有化部署与配置指南。
- [n8n 多智能体编排指南](06_llm_theory_and_fundamentals/workflow/n8n_multi_agent_guide.md) - 基于 n8n 构建 Multi-Agent 系统。
- [开源大模型应用编排平台对比](06_llm_theory_and_fundamentals/workflow/open_source_llm_orchestration_platforms_comparison.md) - 主流应用编排平台的深度横评。

---

## 6. 机器学习基础

基于 NJU 与 SJTU 课程资源的系统化学习路径，涵盖线性代数与概率论等数学基础、SVM 与 HMM 等核心算法数学原理（《统计学习方法》《PRML》），并提供心脏病预测与房价评估等项目驱动的代码实战。

### 6.1 动手学机器学习

结合特征工程、模型评估与超参数调优，系统讲解监督/无监督学习、集成学习、推荐系统与概率图模型，配套心脏病预测、鸢尾花分类与房价预测等项目完成从理论推导到工程化实战的完整闭环。

[动手学机器学习](https://github.com/ForceInjection/hands-on-ML/blob/main/README.md) - 全面的机器学习学习资源库，包含理论讲解、代码实现和实战案例。

**核心特色：**

- **理论与实践结合**：以 NJU 课程为主线，辅以 SJTU 配套资源，从数学原理到代码实现的完整学习路径。
- **算法全覆盖**：涵盖监督学习、无监督学习、集成学习、推荐系统、概率图模型及深度学习。
- **项目驱动学习**：提供心脏病预测、鸢尾花分类、房价预测等实战案例。
- **工程化实践**：深入特征工程、模型评估、超参数调优及特征选择。

### 6.2 参考资料

汇集 3Blue1Brown 线性代数可视化、MIT 18.06 线性代数课程、李航《统计学习方法》、周志华《机器学习》与 Bishop《PRML》等经典教材，以及 Andrew Ng Coursera 入门课程、Stanford CS229 进阶课程与 Kaggle 竞赛实战平台资源。

**数学基础：**

- [线性代数的本质](https://www.bilibili.com/video/BV1ys411472E) - 3Blue1Brown 可视化教程，直观理解线性变换与矩阵运算。
- [MIT 18.06 线性代数](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) - Gilbert Strang 经典课程，深入矩阵分解与子空间理论。
- [概率论与统计学基础](https://book.douban.com/subject/35798663/) - 掌握贝叶斯定理、最大似然估计与概率分布。
- [Datawhale 数学基础](https://datawhalechina.github.io/math-for-ai/#/) - Datawhale 开源的 AI 数学基础教程，涵盖微积分、线性代数与概率论。
- [华东师大矩阵计算课程](https://math.ecnu.edu.cn/~jypan/Teaching/MC/index.html) - 华东师范大学潘建瑜教授的《矩阵计算》课程主页与讲义资源。

**经典教材：**

- **《统计学习方法》** - 李航著，系统阐述感知机、SVM、HMM 等核心算法的数学原理。
- **《机器学习》** - 周志华著（西瓜书），全面覆盖机器学习基础理论与范式。
- **《模式识别与机器学习》** - Bishop 著（PRML），贝叶斯视角的机器学习圣经。

**在线课程与实战：**

- [Andrew Ng 机器学习课程](https://www.coursera.org/learn/machine-learning) - Coursera 经典入门，强调直觉理解。
- [CS229 机器学习](http://cs229.stanford.edu/) - 斯坦福进阶课程，深入数学推导。
- [Kaggle](https://www.kaggle.com/) - 全球最大的数据科学竞赛平台，提供真实数据集与 Notebook 环境。

---

## 7. 大语言模型理论与基础

LLM 核心理论与架构基石，深入解析 Tokenizer 分词机制、Embedding 向量表示学习、混合专家模型 (MoE) 与模型压缩量化技术，并前瞻思维链 (CoT) 推理增强、基于 LLM 的意图识别及 Deep Research (深度研究) 前沿应用架构。

> 详细内容请访问：[LLM 理论与基础](06_llm_theory_and_fundamentals/README.md) - 核心文档门户，涵盖基础理论、深度研究与工作流编排。

### 7.1 基础理论与概念

拆解 LLM 底层运作机制，包括 Tiktokenizer 分词编码、大模型文件格式存储规范、Chain-of-Thought (CoT) 逻辑推理增强技术，以及模型幻觉 (Hallucination) 的成因分析与工程化应对策略。

- [基础理论与概念导航](06_llm_theory_and_fundamentals/llm_basic_concepts/README.md) - LLM 核心概念的完整学习路径。
- [Andrej Karpathy ： Deep Dive into LLMs like ChatGPT （B 站视频）](https://www.bilibili.com/video/BV16cNEeXEer) - 深度学习领域权威专家的 LLM 技术解析。
- [大模型基础组件 - Tokenizer](https://zhuanlan.zhihu.com/p/651430181) - 文本分词与编码的核心技术。
- [解密大语言模型中的 Tokens](06_llm_theory_and_fundamentals/llm_basic_concepts/token/README.md) - Token 机制的深度解析与实践应用。
  - [Tiktokenizer 在线版](https://tiktokenizer.vercel.app/?model=gpt-4o) - 交互式 Token 分析工具。
- [一文读懂思维链（Chain-of-Thought, CoT）](06_llm_theory_and_fundamentals/llm_basic_concepts/cot/chain_of_thought_cot_intro.md) - 推理能力增强的核心技术。
- [大模型的幻觉及其应对措施](06_llm_theory_and_fundamentals/llm_basic_concepts/hallucination/llm_hallucination_and_mitigation.md) - 幻觉问题的成因分析与解决方案。
- [大模型文件格式完整指南](06_llm_theory_and_fundamentals/llm_basic_concepts/file_formats/llm_file_formats_complete_guide.md) - 模型存储与部署的技术规范。

### 7.2 嵌入技术与表示学习

离散文本到连续向量的表示学习体系，解析 Text Embeddings 的演变历史、距离度量算法，以及独立 Embedding 模型与 LLM 原生 Embedding 层的架构选型与应用权衡。

- [文本嵌入学习资源](06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/README.md) - 深入探讨文本嵌入原理与应用的综合指南门户。
- [深入了解文本嵌入技术](06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/text_embeddings_comprehensive_guide.md) - 全面解析 Text Embeddings 的演变、距离度量及应用。
- [LLM 嵌入技术详解：图文指南](06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/LLM_embeddings_explained_visual_guide.zh-CN.md) - 可视化直观理解大模型 Embeddings。
- [文本嵌入技术快速入门](06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/text_embeddings_guide.md) - 快速上手文本嵌入技术的实用指南。
- [大模型 Embedding 层与独立 Embedding 模型：区别与联系](06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/embedding.md) - 嵌入层架构设计与选型策略。

### 7.3 高级架构与应用技术

探索提升模型性能与压缩比的关键技术，通过可视化指南解析 MoE 稀疏激活原理与量化加速机制，并结合 ChatBox 实战剖析基于 LLM 的语义理解与意图检测系统设计。

- [大模型可视化指南](https://www.maartengrootendorst.com/) - 大模型内部机制的可视化分析。
- [混合专家模型 (MoE) 可视化指南](06_llm_theory_and_fundamentals/llm_basic_concepts/moe/mixture_of_experts_moe_visual_guide.zh-CN.md) - 深入解析 MoE 架构原理。
- [量化技术可视化指南](06_llm_theory_and_fundamentals/llm_basic_concepts/quantization/01_visual_guide_to_quantization.md) - 模型压缩与加速的核心技术。
- [基于 LLM 的意图检测](06_llm_theory_and_fundamentals/llm_basic_concepts/intent_detection/intent_detection_using_llm.zh-CN.md) - 意图识别系统设计与实现。
  - 参见：[ChatBox 意图识别与语义理解](06_llm_theory_and_fundamentals/llm_basic_concepts/intent_detection/chatbox_intent_recognition_and_semantic_understanding.md) - ChatBox 中意图识别的实际案例分析。

### 7.4 Deep Research 深度研究

多步推理规划在复杂信息检索中的应用，解构 DeepWiki、通义 DeepResearch 与 Cursor DeepSearch 等主流系统架构，并提供面向科研助手与复杂订单履约场景的 Agent 需求分析与架构设计方案。

- [Deep Research 深度研究资源指南](06_llm_theory_and_fundamentals/deep_research/README.md) - Deep Research 相关的技术解析与实践案例总览。
- [《Building Research Agents for Tech Insights》深度解读](06_llm_theory_and_fundamentals/deep_research/research_agents/building_research_agents_for_tech_insights.md) - 技术洞察研究 Agent 构建指南。
- [DeepWiki 使用方法与技术原理](06_llm_theory_and_fundamentals/deep_research/deepwiki/deepwiki_usage_and_technical_analysis.md) - 技术实现细节与使用指南。
- [DeepWiki 深度研究报告](06_llm_theory_and_fundamentals/deep_research/deepwiki/deepwiki_research_report.pdf) - DeepWiki 的研究成果与深度分析报告。
- [通义 DeepResearch 深度分析](06_llm_theory_and_fundamentals/deep_research/research_agents/qwen_deepresearch_analysis.md) - 对通义 DeepResearch 的技术剖析。
- [Cursor DeepSearch 解析](06_llm_theory_and_fundamentals/deep_research/research_agents/cursor-deepsearch.md) - Cursor AI 深度搜索功能技术分析。
- [Databricks Data Agent](06_llm_theory_and_fundamentals/deep_research/research_agents/databricks_data_agent.md) - Databricks 数据 Agent 技术架构与实现。
- [科研助手 Agent 设计](06_llm_theory_and_fundamentals/deep_research/design/research_assistant.md) - 面向研究者全生命周期的智能助手设计方案。
- [订单履约 Agent 需求分析](06_llm_theory_and_fundamentals/deep_research/design/order_fulfillment_agent_requirement_analysis.md) - 复杂业务场景下的 Agent 系统需求分析。
- [订单履约 Agent 系统设计](06_llm_theory_and_fundamentals/deep_research/design/order_fulfillment_agent_system_design.md) - 复杂业务场景下的 Agent 系统架构与实现。

### 7.5 工作流编排与应用平台 (Workflow)

将 LLM 能力转化为自动化业务流，横向评测 Dify、AnythingLLM 等开源应用编排平台功能与商用许可，并提供基于 Coze 与 n8n 构建多智能体系统的私有化部署实践。

- [工作流编排指南](06_llm_theory_and_fundamentals/workflow/README.md) - 大模型应用编排平台与自动化工作流实践总览。
- [开源大模型应用编排平台功能与商用许可对比分析](06_llm_theory_and_fundamentals/workflow/open_source_llm_orchestration_platforms_comparison.md) - Dify、AnythingLLM、Ragflow 与 n8n 的深度横评。
- [使用 n8n 构建多智能体系统的实践指南](06_llm_theory_and_fundamentals/workflow/n8n_multi_agent_guide.md) - 基于 n8n 构建 Multi-Agent 系统。
- [Coze 部署和配置手册](06_llm_theory_and_fundamentals/workflow/coze_deployment_and_configuration_guide.md) - Coze 平台的私有化部署与配置指南。

### 7.6 参考书籍

精选《大模型技术 30 讲》《Hands-On Large Language Models》《百面大模型》等著作，涵盖从 Transformer 理论解构、模型从零预训练到全栈工程落地的系统性阅读指南。

- [大模型技术 30 讲](https://mp.weixin.qq.com/s/bNH2HaN1GJPyHTftg62Erg) - 大模型时代，智能体崛起：从技术解构到工程落地的全栈指南。
  - 第三方：[大模型技术 30 讲（英文 & 中文批注）](https://ningg.top/Machine-Learning-Q-and-AI) - 带有中英文对照及批注的版本。
- [大模型基础](https://github.com/ZJU-LLMs/Foundations-of-LLMs)

  <img src="https://raw.githubusercontent.com/ZJU-LLMs/Foundations-of-LLMs/main/figure/cover.png" height="300"/>

- [Hands-On Large Language Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models)

  <img src="https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/images/book_cover.png" height="300"/>

- [从零构建大模型](https://mp.weixin.qq.com/s/FkBjsQmeXEPlsdFXETYSng) - 从理论到实践，手把手教你打造自己的大语言模型。
- [百面大模型](https://mp.weixin.qq.com/s/rBJ5an0pr3TgjFbyJXa0WA) - 打通大模型求职与实战的关键一书。
- [图解大模型：生成式 AI 原理与实践](https://mp.weixin.qq.com/s/tYrHrpMrZySgWKE1ECqTWg) - 超过 300 幅全彩图示 × 实战级项目代码 × 中文独家 DeepSeek-R1 彩蛋内容。

---

## 8. 大模型训练

涵盖从 SFT 监督微调到大规模预训练的完整工程路径，结合 70B 模型从零训练实战，剖析数据清洗、硬件集群配置、超参数优化 (CARBS) ，以及面向 AIOps 场景的 Kubernetes 模型后训练 (Post-Training) 与评估框架设计。详细指南可参考：[模型训练与微调总览](05_model_training_and_fine_tuning/README.md) 。

### 8.1 指令微调与监督学习

基于高质量指令-响应数据的模型行为对齐技术，包含 Qwen 2 大模型的微调 Notebook 实战，以及垂直领域模型 SFT 的理论指南与最佳实践。

- [SFT 微调实战与指南](05_model_training_and_fine_tuning/sft_example/README.md) - 包含基于 Qwen2 的微调代码实战及垂域模型微调理论指南。
- [Qwen 2 大模型指令微调实战](05_model_training_and_fine_tuning/sft_example/train_qwen2.ipynb) - 基于 Qwen 2 的指令微调 Notebook 实践。
- [Qwen 2 指令微调教程](https://mp.weixin.qq.com/s/Atf61jocM3FBoGjZ_DZ1UA) - 详细的图文教程。
- [一文入门垂域模型 SFT 微调](05_model_training_and_fine_tuning/sft_example/一文入门垂域模型SFT微调.md) - 垂直领域模型的监督微调技术与应用实践。

### 8.2 大规模模型训练实践

复盘 70B 参数模型从零训练全生命周期，深度解析开源数据集清洗与评估策略、裸金属基础设施配置与自动化脚本，以及扩展至超大规模参数的优化器选型策略。

- [Training a 70B model from scratch: open-source tools, evaluation datasets, and learnings](https://imbue.com/research/70b-intro/) - 70B 参数模型从零训练的完整技术路径与经验总结。
- [Sanitized open-source datasets for natural language and code understanding: how we evaluated our 70B model](https://imbue.com/research/70b-evals/) - 大规模训练数据集的清洗、评估与质量控制方法。
- [From bare metal to a 70B model: infrastructure set-up and scripts](https://imbue.com/research/70b-infrastructure/) - 大模型训练基础设施的搭建、配置与自动化脚本。
- [Open-sourcing CARBS: how we used our hyperparameter optimizer to scale up to a 70B-parameter language model](https://imbue.com/research/70b-carbs/) - 超参数优化器在大规模模型训练中的应用与调优策略。

### 8.3 模型后训练与评估

保障模型生产环境表现的评估体系，解析 AIOps 后训练策略、基于 Kubernetes 的自动化模型评估框架构建，以及基准测试数据集的生成方法。

- [AIOps 后训练技术](05_model_training_and_fine_tuning/ai_ops_design/aiops_post_training.md) - 面向智能运维场景的模型后训练技术与实践。
- [Kubernetes 模型评估框架](05_model_training_and_fine_tuning/ai_ops_design/kubernetes_model_evaluation_framework.md) - 基于 K8s 的大模型评估框架设计与实现。
- [Kubernetes AIOps 基准测试生成框架](05_model_training_and_fine_tuning/ai_ops_design/kubernetes_aiops_benchmark_generation_framework.md) - 自动化生成 AIOps 基准测试数据集的框架设计。

---

## 9. 大模型推理

企业级大模型推理系统落地指南，解构 Mooncake 以 KV Cache 为中心的调度架构、vLLM/llm-d 核心推理框架底层机制，深度剖析 LMCache 多层存储体系与 Tair 跨实例缓存共享，并提供 DeepSeek 等前沿模型在多硬件平台的部署调优实践。

### 9.1 推理系统架构设计

剖析现代推理系统的底层架构创新，重点解构 Mooncake 等以 KV Cache 为中心的高效 LLM 调度系统设计模式与性能调优策略。

- [Mooncake 架构详解：以 KV Cache 为中心的高效 LLM 推理系统设计](09_inference_system/kv_cache/02_systems/mooncake/mooncake_architecture.md) - 新一代推理系统的架构创新与性能优化策略

### 9.2 核心框架与平台

云原生推理基础设施全景，涵盖基于 LWS 的 Kubernetes 多机多卡分布式推理调度，以及高性能 llm-d 框架在不同集群规模下的技术选型与最佳实践。

- [推理优化技术方案](09_inference_system/README.md) - 企业级推理优化全景指南，涵盖集群规模分析、核心优化技术及实施路径
- [vLLM + LWS ： Kubernetes 上的多机多卡推理方案](04_cloud_native_ai_platform/k8s/04_lws_intro.md) - 大模型推理在 Kubernetes 上的最佳实践
- [云原生高性能分布式 LLM 推理框架 llm-d 介绍](04_cloud_native_ai_platform/k8s/05_llm_d_intro.md) - 云原生架构下的高性能推理服务栈

### 9.3 KV Cache 核心技术

长文本与高并发推理的核心瓶颈突破，解析自回归生成机制、Prefix Caching 前缀缓存与 RadixTree 自动复用原理，并深度对比 LMCache 分层架构与阿里云 Tair KVCache 的企业级分布式部署方案。

- [KV Cache 技术体系](09_inference_system/kv_cache/README.md) - KV Cache 技术体系全景指南
- [KV Cache 原理简介](09_inference_system/kv_cache/01_concepts/basic/kv_cache_原理简介.md) - 自回归生成的挑战与 KV Cache 的工作机制
- [Prefix Caching 技术详解](09_inference_system/kv_cache/01_concepts/prefix_caching/prefix_caching.md) ([配套 PPT](09_inference_system/kv_cache/01_concepts/prefix_caching/prefix_caching.pptx)) - 从原理到 vLLM/LMCache 实践的前缀缓存技术
- [RadixAttention 技术详解](09_inference_system/kv_cache/01_concepts/prefix_caching/radix_attention.md) ([配套 PPT](09_inference_system/kv_cache/01_concepts/prefix_caching/radix_attention.pptx)) - 基于 Radix Tree 自动复用 KV Cache 的核心原理与 SGLang 实践

#### 9.3.1 LMCache 核心架构与后端实现

本小节详细解析 LMCache 的四层存储架构及其在跨实例缓存复用中的技术细节。

**基础与架构概览**：

- [LMCache 源码分析指南](09_inference_system/kv_cache/02_systems/lmcache/README.md) - 完整学习路径与文档索引
- [LMCache 架构概览](09_inference_system/kv_cache/02_systems/lmcache/lmcache_overview.md) - 四层存储架构 (L1-L4)、核心组件交互与典型工作流
- [vLLM KV Offloading 与 LMCache 深度对比](09_inference_system/kv_cache/01_concepts/advanced/kv_offloading_analysis.md) - 架构设计、存储层级及跨实例共享能力上的核心差异与性能权衡

**核心运行时组件**：

- [LMCacheEngine 源码分析](09_inference_system/kv_cache/02_systems/lmcache/lmcache_engine.md) - 核心调度中枢、异步事件管理与层级流水线
- [LMCacheConnector 源码分析](09_inference_system/kv_cache/02_systems/lmcache/lmcache_connector.md) - vLLM 集成适配器、视图转换与流水线加载
- [分层存储架构与调度机制](09_inference_system/kv_cache/02_systems/lmcache/lmcache_storage_overview.md) - StorageManager 调度器、Write-All 策略与 Waterfall 检索

**存储后端实现**：

- [LocalCPUBackend 源码分析](09_inference_system/kv_cache/02_systems/lmcache/local_cpu_backend.md) - 本地 CPU 内存后端与并发控制
- [LocalDiskBackend 源码分析](09_inference_system/kv_cache/02_systems/lmcache/local_disk_backend.md) - O_DIRECT 直通 I/O 与异步优化
- [P2PBackend 源码分析](09_inference_system/kv_cache/02_systems/lmcache/p2p_backend.md) - RDMA 零拷贝与去中心化传输
- [GdsBackend 源码分析](09_inference_system/kv_cache/02_systems/lmcache/gds_backend.md) - GPUDirect Storage 零拷贝
- [NixlStorageBackend 源码分析](09_inference_system/kv_cache/02_systems/lmcache/nixl_backend.md) - 高性能网络存储、S3 对象存储对接
- [Remote Connector 源码分析](09_inference_system/kv_cache/02_systems/lmcache/remote_connector.md) - Redis/S3/Mooncake 多后端适配
- [PDBackend 源码分析](09_inference_system/kv_cache/02_systems/lmcache/pd_backend.md) - 预填充-解码分离、Push-based 主动推送机制

**控制面**：

- [LMCache Controller (控制平面)](09_inference_system/kv_cache/02_systems/lmcache/lmcache_controller.md) - 集群元数据管理、ZMQ 三通道通信与节点协调
- [LMCache Server 源码分析](09_inference_system/kv_cache/02_systems/lmcache/lmcache_server.md) - 轻量级中心化存储服务、自定义 TCP 协议

**高级特性**：

- [CacheBlend 技术详解](09_inference_system/kv_cache/02_systems/lmcache/cache_blend.md) - RAG 场景下的动态融合机制、选择性重算与精度保持
- [CacheGen 技术详解](09_inference_system/kv_cache/02_systems/lmcache/cachegen.md) - KV Cache 压缩与流式传输、自适应量化与算术编码

#### 9.3.2 阿里云 Tair KVCache

本小节介绍阿里云企业级的 KVCache 管理系统架构及大规模部署实践。

- [Tair KVCache 架构与设计深度分析](09_inference_system/kv_cache/02_systems/tair_kvcache/tair-kvcache-architecture-design.md) - 阿里云企业级 KVCache 管理系统架构详解，包含与 LMCache 的全面对比分析、中心化管理模式及大规模部署最佳实践

#### 9.3.3 SGLang HiCache

- [HiCache 深入详解](09_inference_system/kv_cache/02_systems/hicache/hicache_deep_dive.md) - SGLang 分层 KV Cache 架构解析与源码分析

### 9.4 推理优化技术体系

多维度提升推理吞吐的系统级技术，包含 vLLM 注意力机制演进 (MLA/NSA)、CUDA Graphs 与 Hybrid KV Cache 管理，结合参数显存占用估算、KV Block Manager 内存机制与 Layer-wise 流水线进行深度调优。

**vLLM 核心机制分析**：

- [vLLM 推理系统优化与分析](09_inference_system/vllm/README.md) - vLLM 底层机制和系统架构的深度解构
- [vLLM 注意力机制演进与支持全景](09_inference_system/vllm/module_analysis/vllm_attention_mha_mla_nsa.md) ([配套 PPT](09_inference_system/vllm/module_analysis/vllm_attention_mha_mla_nsa.pptx)) - 从 MHA 到 MLA 与 NSA 的架构解析及 vLLM 支持现状
- [vLLM 内置 KV Cache Offloading 模块解析](09_inference_system/vllm/module_analysis/vllm_native_kv_offloading.md) - 原生 KV Cache CPU Offloading 功能原理与实现
- [vLLM Hybrid KV Cache Manager](09_inference_system/vllm/module_analysis/vllm_hybrid_kv_cache_manager_deep_dive.md) - vLLM 针对混合注意力架构的显存优化机制
- [vLLM CUDA Graphs 深度解析](09_inference_system/vllm/module_analysis/vllm_cuda_graph_deep_dive.md) - 深入探讨 vLLM 解码阶段 CUDA Graphs 的核心机制与实践
- [vLLM Router 架构解析](09_inference_system/vllm/related_module/vllm_router.md) - 高性能、轻量级请求转发系统
- [vLLM Semantic Router](09_inference_system/vllm/related_module/vllm_semantic_router_deep_dive.md) - 基于语义的智能路由策略

**显存与缓存优化**：

- [LLM 显存占用分析与计算](09_inference_system/memory_calc/memory_analysis.md) - 模型参数、KV Cache 与中间激活值的显存估算方法
- [KV Block Manager 分析](09_inference_system/kv_cache/02_systems/kvbm/KVBM_Analysis.md) - KV Cache 内存管理机制深度解析
- [分层流水线技术](09_inference_system/kv_cache/01_concepts/advanced/layerwise_pipeline.md) - Layer-wise Pipeline 技术原理与性能优化

**网络与模型工具**：

- [模型优化导航](09_inference_system/model_optimization/README.md) - 模型优化技术概览与导读
- [NIXL 网络存储介绍](09_inference_system/infrastructure/nixl_introduction.md) - 高性能网络存储架构与应用
- [NVIDIA 模型优化器](09_inference_system/model_optimization/nvidia_model_optimizer.md) - NVIDIA 模型优化工具链详解
- [图解投机解码](09_inference_system/model_optimization/illustrated-speculative-decoding.md) - Speculative Decoding 的核心思想、系统实现与工程调优要点
- [vLLM GB200 优化](09_inference_system/vllm/hardware_optimization/vllm_gb200_optimization.pptx) - vLLM 在 GB200 硬件上的性能优化策略

### 9.5 推理优化参考设计

企业级 LLM 推理服务全流程实施指南，从集群规模特征评估、异构执行图调度架构设计，到边缘设备/多模态模型专项优化，并提供安全合规、指标体系与上线检查清单。

- [推理优化参考设计导航](09_inference_system/reference_design/README.md) - 企业级 LLM 推理服务架构设计全景导读

**基础理论与技术选型**：

- [背景与目标](09_inference_system/reference_design/01-背景与目标.md) - 推理优化的背景分析与核心目标
- [集群规模分类与特征分析](09_inference_system/reference_design/02-集群规模分类与特征分析.md) - 不同规模集群的特点与需求
- [核心推理优化技术深度解析](09_inference_system/reference_design/03-核心推理优化技术深度解析.md) - KV Cache、批处理、量化等核心技术
- [不同集群规模的技术选型策略](09_inference_system/reference_design/04-不同集群规模的技术选型策略.md) - 针对性的技术方案选择

**架构设计与评估体系**：

- [推理服务架构设计](09_inference_system/reference_design/06-推理服务架构设计.md) - 企业级推理服务架构设计方案
- [面向推理执行图的异构调度系统架构设计](09_inference_system/reference_design/面向推理执行图的异构调度系统架构设计.md) - 跨设备、跨阶段、跨模型的精细化调度方案
- [性能评估指标体系](09_inference_system/reference_design/05-性能评估指标体系.md) - 推理性能评估指标与方法

**专业领域优化**：

- [多模态推理优化](09_inference_system/reference_design/10-多模态推理优化.md) - 多模态模型推理优化策略
- [边缘推理优化](09_inference_system/reference_design/11-边缘推理优化.md) - 边缘设备上的推理优化方案
- [安全性与合规性](09_inference_system/reference_design/09-安全性与合规性.md) - 推理服务的安全与合规要求

**实施落地与运维**：

- [实施建议与最佳实践](09_inference_system/reference_design/07-实施建议与最佳实践.md) - 落地实施的指导建议
- [实施检查清单](09_inference_system/reference_design/13-实施检查清单.md) - 推理系统上线检查清单
- [场景问题解答](09_inference_system/reference_design/12-场景问题解答.md) - 常见问题与解决方案
- [参考资料与延伸阅读](09_inference_system/reference_design/08-参考资料与延伸阅读.md) - 推荐阅读与延伸资料
- [总结与展望](09_inference_system/reference_design/14-总结与展望.md) - 推理优化技术发展趋势

### 9.6 模型部署与运维实践

跨硬件平台的模型服务化落地指南，涵盖 Mac 本地 DeepSeek-R1 运行、Ollama 架构原理，以及 DeepSeek-V3 MoE 在 H20 硬件与 Qwen2-VL 在华为昇腾上的专项部署调优。

- [动手跑大模型](99_misc/mac-deepseek-r1.md) - 手把手教你如何跑大模型
- [Ollama 推理框架详解](99_misc/ollama/README.md) - Ollama 的架构原理与进阶配置
- [DeepSeek-V3 MoE 模型 vLLM 部署](09_inference_system/inference_solutions/deepseek_v3_moe_vllm_h20_deployment.md) - H20 硬件上的部署方案与 SLO 验证
- [Qwen2-VL-7B 华为昇腾部署](09_inference_system/inference_solutions/qwen2_vl_7b_huawei.md) - 国产硬件平台的部署优化

### 9.7 DeepSeek 专题

DeepSeek 模型极致性能优化实战，深度解析 vLLM 宽端点 (Wide Endpoint) 专有并行架构，以及在 Blackwell 等下一代高性能计算平台上的可扩展性评估与部署策略。

- [vLLM WideEP 架构](09_inference_system/vllm/hardware_optimization/vllm_deepseek_blackwell_wide_ep.md) - vLLM 宽端点 (Wide Endpoint) 架构解析
- [Scaling DeepSeek on Blackwell](09_inference_system/vllm/hardware_optimization/scaling_deepseek_blackwell.pptx) - DeepSeek 在 Blackwell 平台上的扩展性优化
- [从 MLA 到 CSA + HCA：DeepSeek 注意力架构的进化之路](09_inference_system/vllm/module_analysis/deepseek_attention_evolution_mla_to_csa_hca.md) - 结合 vLLM 推理引擎源码，深度解析 DeepSeek 注意力机制演进

---

## 10. 企业级 AI Agent 开发

构建生产级 AI Agent 的系统化指南，涵盖 BDI 认知理论、ReAct/12-Factor 等架构模式，深度拆解动态上下文工程、MemoryOS 多层记忆架构与 MCP 互操作协议，并提供基于 LangGraph 的企业级多智能体系统与 Kagent 基础设施演进分析。

> 详细内容请访问：[AI Agent 开发与实践](08_agentic_system/README.md) - 核心文档门户，涵盖理论、架构与实战。

### 10.1 核心理论与架构设计

解析多智能体协作机制与企业级落地框架，包含 ReAct 推理机制、复杂写作/指代消解系统设计模式，并深度剖析世界模型认知引擎与 Data Agent 数据智能体的新兴架构范式。

**多智能体系统**：

- [理论基础与框架](08_agentic_system/multi_agent/docs/part1_multi_agent_ai_fundamentals.md) - BDI 架构、多 Agent 协作机制与企业级落地
- [企业级构建实战](08_agentic_system/multi_agent/docs/part2_enterprise_multi_agent_system_implementation.md) - 基于 LangGraph 的架构设计与企业级代码落地

**智能体设计模式**：

- [ReAct Agent 模式详解](08_agentic_system/agent_design/docs/react-agent.md) - 推理与行动深度协同的经典机制
- [写作 Agent 设计](08_agentic_system/agent_design/docs/writing-agentic-agent.md) - 针对复杂长文本内容创作的架构设计
- [指代消解系统设计](08_agentic_system/agent_design/docs/coreference-resolution-dialogue-system.md) - 高级对话状态管理与多轮交互技术
- [12-Factor Agents](08_agentic_system/concepts/12-factor-agents-intro.md) - 构建高可靠、可扩展 LLM 应用的 12 要素原则
- [TradingAgents-CN 设计](08_agentic_system/agent_design/docs/trading-agents-cn.md) - 交易领域的智能体设计与交互分析
- [AI Agent 设计模式](08_agentic_system/agent_design/docs/ai-agent-design-patterns.pptx) - 企业级智能体架构设计模式
- [All Agentic Architectures 深入详解](08_agentic_system/agent_design/docs/all-agentic-architectures-deep-dive.md) - 17 种 Agent 架构深度对比与实现解析

**数据智能体 (Data Agents)**：

- [数据智能体综述](08_agentic_system/data_agent/data-agent-survey.md) ([配套 PPT](08_agentic_system/data_agent/data-agent-survey.pptx)) - Data Agent 新兴范式的核心架构与应用挑战
- [企业级 Data Agent 产品需求文档](08_agentic_system/data_agent/enterprise-data-agent-prd.md) ([配套 PPT](08_agentic_system/data_agent/enterprise-data-agent-prd.pptx)) - 完整的商业级 L2 条件自动化 Data Agent PRD
- [企业级 Data Agent 敏捷落地规划](08_agentic_system/data_agent/data-agent-skill-mvp.md) ([配套 PPT](08_agentic_system/data_agent/data-agent-skill-mvp.pptx)) - 针对 MVP 阶段通过技能挂载盘活存量 API 的落地战术

**认知与基础理论**：

- [世界模型简介](08_agentic_system/concepts/world-model-introduction.md) - 解析智能体理解世界的内部引擎

### 10.2 核心工程组件与基础设施

解构构建高可靠 Agent 的底层支撑体系，涵盖动态上下文组装与压缩工程、Claude/Mem0 记忆架构机制、MCP 工具互操作协议规范，以及 Agent Sandbox 与 Kubernetes 运维智能体基础设施演进。

**上下文与记忆系统**：

- [上下文工程原理](08_agentic_system/context/context-engineering-principles.md) - 动态数据组装、压缩与检索技术
  - [快速入门](08_agentic_system/context/context-engineering-intro.md) | [Anthropic 指南](08_agentic_system/context/anthropic-context-engineering-zh.md) | [LangChain 实践](08_agentic_system/context/langchain-with-context-engineering.md) | [OpenViking 剖析](08_agentic_system/context/openviking-deep-dive.md) | [Claude Code 上下文压缩](08_agentic_system/context/claude-code-context-compression.md)
- [记忆系统架构总览](08_agentic_system/memory/README.md) - 赋予智能体长期记忆与个性化能力的核心机制
  - [理论与实践综述](08_agentic_system/memory/research/theory/ai-agent-memory-theory.md) | [大模型记忆综述](08_agentic_system/memory/research/theory/llm-agent-memory-survey.md) | [记忆系统演进思考](08_agentic_system/memory/research/theory/memory-systems-are-dead.md)
  - [MemoryOS 架构](08_agentic_system/memory/research/systems/memoryos-architecture-guide.md) | [MemMachine 深度解析](08_agentic_system/memory/research/systems/memmachine-deep-dive.md) | [Mem0 快速入门](08_agentic_system/memory/research/systems/mem0-quickstart.md) | [Hermes 内存架构解析](08_agentic_system/memory/research/systems/hermes-agent-memory-management.md)
  - [Claude Code: 记忆机制解析](08_agentic_system/memory/research/case-studies/claude-code-memory-analysis.md) | [Claude-Mem: 系统介绍](08_agentic_system/memory/research/case-studies/claude-mem-system-analysis.md) | [Claude Code: Agent 执行流程解析](08_agentic_system/memory/research/case-studies/claude-code-agent-execution-flow.md) | [SuperMemory: 集成分析](08_agentic_system/memory/research/case-studies/supermemory-agent-integration-analysis.md)

**工具及协议**：

- [Model Context Protocol (MCP)](08_agentic_system/mcp/docs/01_deep_dive_into_mcp_and_the_future_of_ai_tooling.md) - MCP 原理与实战，探讨 AI 工具链的未来

**Agent Skill**：

- [Claude Skills 开发指南](08_agentic_system/agent_skills/docs/claude_skills_guide.md) - 扩展智能体能力的工具定义规范与最佳实践
  - [构建完整指南 (PDF)](08_agentic_system/agent_skills/docs/the_complete_guide_to_building_skill_for_claude.pdf)
- [Agent Skill 开发指南](https://github.com/ForceInjection/awesome-skills) - 由原力注入博主维护的优秀认知技能（Agent Skill）合集，包含深度代码阅读、架构分析、文档评审等自动化工作流。
- [CUDA Code Skill](https://github.com/ForceInjection/cuda-code-skill) - 面向 AI IDE（Claude Code、Trae 等）的 CUDA 知识增强代码生成与性能分析技能库。
- [mmx-cli](https://github.com/MiniMax-AI/cli) - MiniMax AI 平台的 CLI 技能，支持文本、图片、视频、语音、音乐生成与 Web 搜索，遵循 agentskills.io 标准。

**AI Agent Infra**：

- [基础设施技术栈](08_agentic_system/agent_infra/docs/ai-agent-infra-stack.md) - 全面梳理工具层、数据层与编排层
- [基础设施的崛起](08_agentic_system/agent_infra/docs/the-rise-of-ai-agent-infrastructure.md) - 生态演进趋势与未来投资方向
- [OpenHarness 深入浅出：解密开源智能体基础设施](08_agentic_system/agent_infra/docs/openharness-deep-dive.md) ([配套 PPT](08_agentic_system/agent_infra/docs/openharness-deep-dive.pptx)) - 大型语言模型 (LLM) 在推理与生成能力上取得了突破性进展，但它们本身受限于静态的上下文窗口，无法直接与真实世界进行交互。要让模型成为能够自主解决复杂任务的工程化智能体 (Agent) ，必须为其配备执行动作的工具、持久化的记忆以及安全隔离的运行边界。这就是“智能体基础设施” (Agent Harness) 的核心使命。
- [Agent Sandbox 的演进与设计范式](08_agentic_system/agent_infra/docs/agent-sandbox-design.md) ([配套 PPT](08_agentic_system/agent_infra/docs/agent-sandbox-design.pptx)) - 探讨 Agent Sandbox 的核心设计理念，对比 OpenShell、Sandlock 等沙箱方案，揭示从“硬件级隔离”向“策略优先”演进的技术趋势。
- [深度解析 Kagent：以构建 Kubernetes 运维智能体为例](08_agentic_system/agent_infra/docs/deep-dive-kagent-k8s-ops-agent.md) ([配套 PPT](08_agentic_system/agent_infra/docs/deep-dive-kagent-k8s-ops-agent.pptx)) - 深度解析 Kagent 的核心架构与工作机制，并以“构建阿里云 ACK 运维智能体”为实战案例，展示大模型与运维工具的编排。
- [OpenClaw Operator 架构深度解析](08_agentic_system/agent_infra/docs/openclaw-operator-deep-dive.md) - 云原生时代 AI Agent 运行时环境编排机制
- [DeepSeek-TUI 实战](08_agentic_system/agent_infra/docs/deepseek-tui-in-practice.md) - 榨干 DeepSeek V4 长上下文红利的命令行编程 Agent 实战指南
- [Claude Code Sandbox 安全隔离机制解析](08_agentic_system/agent_infra/docs/claude-code-sandbox.md) - Linux 环境下基于 Bubblewrap 的底层隔离架构
- [扩展托管智能体](08_agentic_system/agent_infra/docs/scaling-managed-agents.md) - 让决策与执行解耦，各行其职的 AI 原生基础设施
- [在 Elasticsearch 之上实现虚拟文件系统](08_agentic_system/agent_infra/docs/virtual-filesystem-elasticsearch_zh.md) - Elasticsearch 虚拟文件系统实现解析（[英文原文](08_agentic_system/agent_infra/docs/virtual-filesystem-elasticsearch.md)）

### 10.3 实战代码与演示项目

从理论走向落地的工程实践代码库，包含基于异步通信总线的企业级多智能体系统、多轮指代消解微服务、MCP 客户端/服务端交互 Demo，以及结合 LangChain 的记忆功能与 PDF 智能翻译器实现。

**完整端到端系统**：

- [企业级多智能体系统](08_agentic_system/multi_agent/multi_agent_system/README.md) - 包含异步通信总线、状态监控与容错机制的完整 MAS 实现
- [多轮指代消解对话系统](08_agentic_system/agent_design/coref-dialogue-system/README.md) - 支持实体识别、状态管理与微服务部署的 NLP 实战

**专项工具与演示**：

- [MCP 智能体演示](08_agentic_system/mcp/mcp_demo/README.md) - MCP 服务端与客户端交互的完整示例
- [LangChain 记忆功能集成](08_agentic_system/memory/langchain/code/README.md) - 包含基础记忆类型、智能客服应用和 LangGraph 记忆管理的演示
  - [代码实现示例](08_agentic_system/memory/langchain/langchain_memory.md)
- [Agent Skill：PDF 智能翻译器](08_agentic_system/agent_skills/pdf_translator/README.md) - 结合 OCR 与 LLM 的多模态文档处理工具

### 10.4 前沿学术与行业研究

汇集 24 种主流 Agent Workflow 模式综述、Deep Research 深度研究架构等核心学术论文，以及 2025 年度 LangChain 开发者诉求与 Agent 工程化现状等权威行业报告。

**学术论文**：

- [Agent Workflow 综述](08_agentic_system/papers/agent-workflow-survey.md) - 涵盖 24 种主流 Agent 工作流模式的权威系统性总结
- [Deep Research Agents](08_agentic_system/papers/deep-research-agent.md) - 探讨深度研究智能体的多步推理规划能力与核心架构
- [论文资源库汇总](08_agentic_system/papers/README.md) - AI Agent 领域必读核心论文持续更新索引

**行业报告**：

- [LangChain Agent 工程现状报告](08_agentic_system/reports/langchain-state-of-agent-engineering.md) - 2025 年度 Agent 领域最新技术趋势与开发者诉求

---

## 11. 检索增强生成与文档智能

解构从非结构化数据解析到高可信知识库构建的完整技术栈，涵盖 Naive/Advanced RAG 架构演进、Embedding 选型与 Chunking 策略，并深入 GraphRAG (Neo4j/KAG) 复杂关系推理与 MinerU/Marker 等文档智能解析引擎应用。

> 详细内容请访问：[rag 与工具生态](07_rag_and_tools/README.md) - 核心文档门户，涵盖 `RAG`、`GraphRAG` 与文档智能工具。

### 11.1 检索增强生成基础与进阶

RAG 技术全景演进导航，对比不同检索架构优劣势，系统评估文本分块 (Chunking) 策略，并提供面向中文场景的 Embedding 模型深度评测与选型指南。

- [rag 快速开发实战（从 0 到 1 搭建）](https://mp.weixin.qq.com/s/89-bwZ4aPor4ySj5U3n5zw) - `RAG` 技术全景导航，涵盖基础概念到进阶优化
- [rag 策略对比](07_rag_and_tools/rag_basics/rag_comparison.md) - 不同 `RAG` 架构（`Naive RAG`、`Advanced RAG` 等）的优劣势分析
- [chunking 策略评估总结](07_rag_and_tools/rag_basics/evaluating_chunking_strategies_summary.md) - 检索分块策略的深度总结与最佳实践
- [中文 rag 系统 embedding 选型指南](07_rag_and_tools/rag_basics/chinese_rag_embedding_model_selection.md) - 面向中文场景的 `Embedding` 模型评测与推荐

### 11.2 图检索增强生成与知识图谱

解决复杂多跳关系推理难题的图计算架构，深度解析 GraphRAG 核心概念与 KAG 框架，并提供图数据库 Neo4j 安装配置与 Cypher 查询语言的实战教程。

- [graphrag 学习指南](07_rag_and_tools/graph_rag/graph_rag_learning_guide.md) - `GraphRAG` 的核心概念、架构原理与入门路径
- [kag 框架介绍](07_rag_and_tools/graph_rag/kag_introduction.md) - `Knowledge Augmented Generation`（`KAG`）框架深度解析
- [neo4j 实战指南](07_rag_and_tools/knowledge_graph/neo4j_handson_guide.md) - 图数据库 `Neo4j` 的安装、配置与企业级实战
- [neo4j cypher 教程](07_rag_and_tools/knowledge_graph/neo4j_cypher_tutorial.md) - `Neo4j` 查询语言 `Cypher` 完整教程

### 11.3 大模型与知识图谱协同应用

构建高可信、可解释智能应用的最佳实践，结合完整源码与图谱数据，深入剖析基于 LLM+KG 架构的银行反电诈智能风控系统设计方案。

- [银行反电诈智能系统设计](07_rag_and_tools/synergized_llms_kgs/anti_fraud_design.md) - 基于 `LLM` + `KG` 的金融风控系统设计方案，实战反欺诈场景
- [反欺诈 demo 源码](07_rag_and_tools/synergized_llms_kgs/demo/README.md) - 完整的反欺诈系统演示代码，包含数据生成、图谱构建与智能体推理

### 11.4 文档智能解析

突破 RAG 系统的数据质量瓶颈，解析 MinerU 与 Marker 等基于深度学习的复杂 PDF 布局检测与公式提取引擎，以及 Microsoft MarkItDown 跨格式文档转换工具。

- [mineru 文档解析](07_rag_and_tools/pdf_tools/miner_u_intro.md) - 上海人工智能实验室开源工具，助力复杂 `PDF` 高效解析
- [marker pdf 布局检测](07_rag_and_tools/pdf_tools/marker_zh_cn.md) - 基于深度学习的高精度 `PDF` 解析与布局分析引擎
- [markitdown 入门](07_rag_and_tools/pdf_tools/markitdown/markitdown_intro.md) - Microsoft 开源的文档转换工具，支持多种办公文档格式到 `Markdown` 的高质量转换

---

## 12. 课程体系与学习路径

系统化学习路径与进阶指南集合，包含 ZOMI 酱 AI System 全栈硬件架构、大模型底层原理与演进基础、Trae AI 辅助编程实战，以及基于 LangGraph 的企业级多智能体系统培训资源。

### 12.1 AI System 全栈课程（ZOMI 酱）

ZOMI 酱主导的高分开源 AI 基础设施架构体系，从底层 AI 芯片架构、硬件加速器到 AI 编译器原理、推理性能调优与分布式框架设计进行深度解构。

[AISystem](https://github.com/chenzomi12/AISystem) - AI 系统全栈课程代码与资料库。

- [系统介绍](https://github.com/chenzomi12/AISystem/tree/main/01Introduction) - AI 系统概述、发展历程与技术演进路径。
- [硬件基础](https://github.com/chenzomi12/AISystem/tree/main/02Hardware) - AI 芯片架构、硬件加速器与计算平台深度解析。
- [编译器技术](https://github.com/chenzomi12/AISystem/tree/main/03Compiler) - AI 编译器原理、优化技术与工程实践。
- [推理优化](https://github.com/chenzomi12/AISystem/tree/main/04Inference) - 模型推理加速技术、性能调优与部署策略。
- [框架设计](https://github.com/chenzomi12/AISystem/tree/main/05Framework) - AI 框架架构设计、分布式计算与并行优化。

### 12.2 AI Infra 基础课程（入门）

面向初学者的大模型入门知识体系，涵盖 Transformer 架构原理、GPT-3/4 与 PaLM 等模型的规模与训练成本分析、DeepSeek V1/V2/R1 技术演进（MLA 与 MoE 稀疏化）、能力涌现现象研究，以及 GPU 架构/CUDA 编程与云原生 AI 基础设施运维实践。

- [大模型原理与最新进展](10_ai_related_course/ai_coding/index.html) - 交互式在线课程平台。
- [AI 编程入门](10_ai_related_course/ai_coding/AI%20编程入门.md) - AI 编程基础知识与应用入门。
- [AI Infra 课程演讲稿](10_ai_related_course/ai_infra_course/%E5%85%A5%E9%97%A8%E7%BA%A7/%E8%AE%B2%E7%A8%BF.md) - 完整的课程演讲内容、技术要点与实践案例。
- **学习目标**：深入理解大模型工作原理、最新技术进展与企业级应用实践。
- **核心内容**：
  - **Transformer 架构深度解析**：编码器-解码器结构、多头注意力机制、文本生成过程。
  - **训练规模与成本分析**：GPT-3/4、PaLM 等主流模型的参数量、训练成本和资源需求。
  - **DeepSeek 技术突破**：V1/V2/R1 三代模型演进、MLA 架构创新、MoE 稀疏化优化。
  - **能力涌现现象研究**：规模效应、临界点突破、多模态融合发展趋势。
  - **AI 编程工具生态**：GitHub Copilot、Cursor、Trae AI 等工具对比分析与应用实践。
  - **GPU 架构与 CUDA 编程**：硬件基础、并行计算原理、性能优化策略。
  - **云原生 AI 基础设施**：现代化 AI 基础设施设计、容器化部署与运维实践。

### 12.3 Trae 编程实战课程

系统掌握 AI 辅助开发工作流，从 IDE 环境配置与交互模式，进阶至 Web 开发、数据库设计与 DevOps 微服务架构等复杂场景的实战演练。

- [Trae 编程实战教程](10_ai_related_course/trae/README.md) - 从基础入门到高级应用的完整 Trae 编程学习路径。

**课程结构：**

- **第一部分：Trae 基础入门**：环境配置、交互模式、HelloWorld 项目实战。
- **第二部分：常见编程场景实战**：前端开发、Web 开发、后端 API、数据库设计、安全认证。
- **第三部分：高级应用场景**：AI 模型集成、实时通信、数据分析、微服务架构。
- **第四部分：团队协作与最佳实践**：代码质量管理、项目管理、性能优化、DevOps 实践。
- **第五部分：综合项目实战**：企业级应用开发、核心功能实现、部署运维实战。

### 12.4 多智能体 AI 系统培训

面向企业研发团队的生产级架构指南，深度解析 LangGraph 核心调度机制、LangSmith 监控平台集成以及企业级 Multi-Agent 架构设计模式。

- [多智能体 AI 系统培训材料](10_ai_related_course/multi_agent_system/multi_agent_training/README.md)：涵盖 LangGraph 框架深度解析、LangSmith 监控集成及企业级架构设计。

### 12.5 微软 AI Agents for Beginners 课程

由微软提供的初学者课程，旨在帮助学习者全面了解 AI Agent 的构建与应用。

- [AI Agents for Beginners 课程之 AI Agent及使用场景简介](10_ai_related_course/AI Agents for Beginners 课程之 AI Agent及使用场景简介.md) - 涵盖 AI Agent 基础概念、开发框架、设计模式与应用场景。

---

## Buy Me a Coffee

如果您觉得本项目对您有帮助，欢迎购买我一杯咖啡，支持我继续创作和维护。

| **微信**                                                 | **支付宝**                                            |
| -------------------------------------------------------- | ----------------------------------------------------- |
| <img src="./img/weixinpay.JPG" alt="wechat" width="200"> | <img src="./img/alipay.JPG" alt="alipay" width="200"> |

---
