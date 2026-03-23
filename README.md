# AI Fundamentals

本仓库是一个全面的人工智能基础设施（AI Infrastructure）学习资源集合，涵盖从硬件基础到高级应用的完整技术栈。内容包括 GPU 架构与编程、CUDA 开发、大语言模型、AI 系统设计、性能优化、企业级部署等核心领域，旨在为 AI 工程师、研究人员和技术爱好者提供系统性的学习路径和实践指导。

> - **适用人群**：AI 工程师、系统架构师、GPU 编程开发者、大模型应用开发者、技术研究人员。
> - **技术栈**：CUDA、GPU 架构、LLM、AI 系统、分布式计算、容器化部署、性能优化。

---

**Star History**:

## ![Star History Chart](https://api.star-history.com/svg?repos=ForceInjection/AI-fundermentals&type=date&legend=top-left)

---

## 1. 硬件架构与互连技术

本章深入解析 AI 计算硬件与系统互连架构，内容涵盖从单机基础计算芯片（GPU、TPU）的设计哲学，到系统内高速互连总线（PCIe、NVLink）及高级跨节点直通技术（GPUDirect），最后探讨系统级延迟参考与异构融合超级芯片架构。详细内容请访问：**[硬件架构与互连技术](01_hardware_architecture/README.md)**。

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

本章涵盖从底层网络硬件到上层通信库的完整运维体系，包括高性能网络组网、GPU 基础监控运维以及分布式通信实战，为构建高吞吐的 AI 计算集群提供保障。详细内容请访问：**[AI 集群运维与通信](03_ai_cluster_ops/README.md)**。

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

本章聚焦于云原生技术在 AI 领域的应用，探讨如何利用 Kubernetes、容器化、微服务等云原生技术栈构建高效、可扩展的 AI 基础设施。详细内容请访问：**[云原生 AI 平台](04_cloud_native_ai_platform/README.md)**。

### 3.1 Kubernetes AI 基础设施

Kubernetes 是云原生 AI 平台的操作系统。本模块深入解析 Kubernetes 在 AI 场景下的核心组件与扩展机制，涵盖从底层的容器运行时支持到上层的分布式作业调度。

- [Kubernetes GPU 管理与 AI 工作负载](./04_cloud_native_ai_platform/k8s/README.md)：云原生 AI 基础设施建设指南与技术导图
- [NVIDIA Container Toolkit 原理](./04_cloud_native_ai_platform/k8s/01_nvidia_container_toolkit_analysis.md)：容器使用 GPU 的底层机制深度解析
- [Device Plugin 原理](./04_cloud_native_ai_platform/k8s/02_nvidia_k8s_device_plugin_analysis.md)：Kubernetes 设备插件机制源码分析
- [Kueue + HAMi 调度方案](./04_cloud_native_ai_platform/k8s/03_kueue_hami_integration.md)：云原生作业队列与细粒度 GPU 共享机制
- [LWS (Leader Worker Set) 介绍](./04_cloud_native_ai_platform/k8s/04_lws_intro.md)：Kubernetes 原生的大模型分布式训练与推理调度抽象
- [分布式推理框架](./04_cloud_native_ai_platform/k8s/05_llm_d_intro.md)：基于 Kubernetes 的 LLM 推理架构设计
- [Containerd 日志分析](./04_cloud_native_ai_platform/k8s/06_containerd_log_analysis.md)：云原生容器运行时的日志排查与分析

### 3.2 GPU 资源管理与虚拟化

GPU 是 AI 平台最昂贵的计算资源。本模块专注于 GPU 资源的精细化管理，包括虚拟化、切分、远程调用和池化技术，旨在最大化资源利用率。

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

数据是 AI 的燃料。本模块介绍如何利用 JuiceFS、DeepSeek 3FS 等云原生分布式文件系统，解决 AI 训练中海量小文件读取、模型检查点保存和跨节点数据共享的性能瓶颈。

- [JuiceFS 分布式文件系统](./04_cloud_native_ai_platform/storage/juicefs/README.md)：数据与元数据分离的架构设计，兼容 POSIX 接口
  - [文件修改机制分析](./04_cloud_native_ai_platform/storage/juicefs/01_juicefs_file_modification_mechanism_analysis.md)：底层数据一致性与写入流程解析
  - [后端存储变更手册](./04_cloud_native_ai_platform/storage/juicefs/02_juicefs_backend_storage_migration_guide.md)：生产环境下的存储运维与数据迁移指南
- [DeepSeek 3FS 设计笔记](./04_cloud_native_ai_platform/storage/deepseek_3fs/01_deepseek_3fs_design_notes.md)：高性能存储系统架构设计与特性分析
- [NVIDIA ICMS 架构解析](./04_cloud_native_ai_platform/storage/inference_context_memory_storage/01_icms_architecture.md)：面向推理的 KV Cache 存储层架构深度解析

---

## 4. 底层计算与异构编程

本章专注于 AI 系统的底层编程技术，涵盖 GPU 基础架构、CUDA 核心编程范式以及 DPU 的开发指南，为系统级开发者提供从入门到进阶的完整技术路径。

### 4.1 GPU 与 CUDA 编程

整合了 GPU 基础架构、CUDA 核心编程概念及丰富的学习资源。详细内容请访问：[GPU 编程基础](02_gpu_programming/README.md)。

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

介绍数据处理单元（DPU）在现代计算架构中的应用及编程框架。详细内容请访问：[DPU 编程](02_dpu_programming/README.md)。

- **DOCA 框架**
  - [DOCA 编程入门](./02_dpu_programming/doca/01_doca_programming_guide.md)：涵盖架构简介、核心组件及典型场景编程实践

---

## 5. 大语言模型应用开发与编排

本章探讨在 AI 时代下新兴的应用层开发范式与工作流，重点关注如何利用各种编程语言、框架和编排工具构建复杂的 LLM 应用及 Agent 系统。详细的深度探讨可参考 [大模型编程指南](98_llm_programming/README.md)。

### 5.1 AI 时代的软件工程：范式转移与重构

随着大语言模型能力的爆发式增长，软件工程正从 Software 1.0/2.0 时代迈向以自然语言驱动、Agent 自主决策与推理为核心的 **Software 3.0** 时代。本节探讨了在 AI 辅助下新兴的编程范式与工作流，重点关注如何利用 AI 提升开发效率与代码质量：

- [Agent First：软件工程的下一个范式转移](98_llm_programming/Agent_First.md) - 梳理编程范式的演变历史，探讨 Agent First 的核心理念与实战指南。
- [驾驭工程](98_llm_programming/Harness_Engineering.md) - 深度解析如何构建驾驭系统，提升 AI 编程助手的可控性与效能。
- [OpenSpec 实战指南](https://github.com/ForceInjection/OpenSpec-practise/blob/main/README.md) - Spec 驱动开发 (Spec-Driven Development) 的工程实践，演示了“意图 -> Spec -> AI -> 代码 & 验证”的新一代开发工作流。

### 5.2 Java AI 开发

本节主要介绍在 Java 生态系统中开发大语言模型应用的技术栈。Spring AI 作为官方主推的 AI 工程框架，极大地降低了企业级 Java 应用接入 AI 能力的门槛。

- [Java AI 开发指南](98_llm_programming/java_ai/README.md) - Java 生态系统中的 AI 开发技术总览。
- [使用 Spring AI 构建高效 LLM 代理](98_llm_programming/java_ai/spring_ai_cn.md) - 基于 Spring AI 框架的企业级 AI 应用开发实践。

### 5.3 LangGraph 开发

LangGraph 是一个用于构建有状态、多智能体应用程序的库。它通过引入图计算模型，完美解决了传统 LLM 应用在循环逻辑和状态持久化方面的瓶颈，特别适合构建需要多轮推理和自我反思的复杂 Agent 工作流。

- [LangGraph 框架学习资源](98_llm_programming/langgraph/README.md) - LangGraph 框架的学习资源与实践案例总览。
- [LangGraph 简介](98_llm_programming/langgraph/langgraph_intro.md) - LangGraph 的核心概念与入门指南。
- [AI 客服系统实战](98_llm_programming/langgraph/aics.ipynb) - 基于 LangGraph 构建的 AI 客服系统 Notebook 实战。

### 5.4 AI 工作流与编排

除硬编码框架外，无代码或低代码（No-Code/Low-Code）工具也是 AI 应用落地的重要途径，它们能大幅提升编排效率。

- [Coze 部署和配置手册](06_llm_theory_and_fundamentals/workflow/coze_deployment_and_configuration_guide.md) - Coze 平台的私有化部署与配置指南。
- [n8n 多智能体编排指南](06_llm_theory_and_fundamentals/workflow/n8n_multi_agent_guide.md) - 基于 n8n 构建 Multi-Agent 系统。
- [开源大模型应用编排平台对比](06_llm_theory_and_fundamentals/workflow/open_source_llm_orchestration_platforms_comparison.md) - 主流应用编排平台的深度横评。

---

## 6. 机器学习基础

本部分基于开源项目，提供系统化的机器学习学习路径。涵盖从数学原理到代码实现的完整过程，为深入学习大模型打下坚实基础。

### 6.1 动手学机器学习

本节提供全面的理论讲解与代码实战。

[动手学机器学习](https://github.com/ForceInjection/hands-on-ML/blob/main/README.md) - 全面的机器学习学习资源库，包含理论讲解、代码实现和实战案例。

**核心特色：**

- **理论与实践结合**：以 NJU 课程为主线，辅以 SJTU 配套资源，从数学原理到代码实现的完整学习路径。
- **算法全覆盖**：涵盖监督学习、无监督学习、集成学习、推荐系统、概率图模型及深度学习。
- **项目驱动学习**：提供心脏病预测、鸢尾花分类、房价预测等实战案例。
- **工程化实践**：深入特征工程、模型评估、超参数调优及特征选择。

### 6.2 参考资料

本节精选了数学基础、经典教材与实战平台资源，构建完整的知识图谱。

**数学基础：**

- [线性代数的本质](https://www.bilibili.com/video/BV1ys411472E) - 3Blue1Brown 可视化教程，直观理解线性变换与矩阵运算。
- [MIT 18.06 线性代数](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) - Gilbert Strang 经典课程，深入矩阵分解与子空间理论。
- [概率论与统计学基础](https://book.douban.com/subject/35798663/) - 掌握贝叶斯定理、最大似然估计与概率分布。

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

本章旨在为读者构建扎实的大语言模型（LLM）理论基础，涵盖从词向量嵌入到模型架构设计的核心知识。我们将深入解析 Token 机制、混合专家模型（MoE）等关键技术，并探讨量化、思维链（CoT）等前沿优化方向，同时涵盖深度研究（Deep Research）应用与工作流编排等前沿技术。

> 详细内容请访问：[LLM 理论与基础](06_llm_theory_and_fundamentals/README.md) - 核心文档门户，涵盖基础理论、深度研究与工作流编排。

### 7.1 基础理论与概念

本节介绍大语言模型的基础理论，涵盖从文本处理到模型架构的核心概念。理解这些基础概念是深入学习 LLM 技术的前提。

- [Andrej Karpathy ： Deep Dive into LLMs like ChatGPT （B 站视频）](https://www.bilibili.com/video/BV16cNEeXEer) - 深度学习领域权威专家的 LLM 技术解析。
- [大模型基础组件 - Tokenizer](https://zhuanlan.zhihu.com/p/651430181) - 文本分词与编码的核心技术。
- [解密大语言模型中的 Tokens](06_llm_theory_and_fundamentals/llm_basic_concepts/token/README.md) - Token 机制的深度解析与实践应用。
  - [Tiktokenizer 在线版](https://tiktokenizer.vercel.app/?model=gpt-4o) - 交互式 Token 分析工具。
- [一文读懂思维链（Chain-of-Thought, CoT）](06_llm_theory_and_fundamentals/llm_basic_concepts/cot/chain_of_thought_cot_intro.md) - 推理能力增强的核心技术。
- [大模型的幻觉及其应对措施](06_llm_theory_and_fundamentals/llm_basic_concepts/hallucination/llm_hallucination_and_mitigation.md) - 幻觉问题的成因分析与解决方案。
- [大模型文件格式完整指南](06_llm_theory_and_fundamentals/llm_basic_concepts/file_formats/llm_file_formats_complete_guide.md) - 模型存储与部署的技术规范。

### 7.2 嵌入技术与表示学习

本节深入探讨文本嵌入的原理、实现方式以及在不同场景下的应用策略。嵌入技术是大语言模型的核心组件之一，负责将离散的文本符号转换为连续的向量表示。

- [文本嵌入学习资源](06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/README.md) - 深入探讨文本嵌入原理与应用的综合指南门户。
- [深入了解文本嵌入技术](06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/text_embeddings_comprehensive_guide.md) - 全面解析 Text Embeddings 的演变、距离度量及应用。
- [LLM 嵌入技术详解：图文指南](06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/LLM_embeddings_explained_visual_guide.zh-CN.md) - 可视化直观理解大模型 Embeddings。
- [文本嵌入技术快速入门](06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/text_embeddings_guide.md) - 快速上手文本嵌入技术的实用指南。
- [大模型 Embedding 层与独立 Embedding 模型：区别与联系](06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/embedding.md) - 嵌入层架构设计与选型策略。

### 7.3 高级架构与应用技术

本节涵盖混合专家系统、量化技术、意图检测等前沿架构与应用技术。

- [大模型可视化指南](https://www.maartengrootendorst.com/) - 大模型内部机制的可视化分析。
- [混合专家模型 (MoE) 可视化指南](06_llm_theory_and_fundamentals/llm_basic_concepts/moe/mixture_of_experts_moe_visual_guide.zh-CN.md) - 深入解析 MoE 架构原理。
- [量化技术可视化指南](06_llm_theory_and_fundamentals/llm_basic_concepts/quantization/01_visual_guide_to_quantization.md) - 模型压缩与加速的核心技术。
- [基于 LLM 的意图检测](06_llm_theory_and_fundamentals/llm_basic_concepts/intent_detection/intent_detection_using_llm.zh-CN.md) - 意图识别系统设计与实现。
  - 参见：[ChatBox 意图识别与语义理解](06_llm_theory_and_fundamentals/llm_basic_concepts/intent_detection/chatbox_intent_recognition_and_semantic_understanding.md) - ChatBox 中意图识别的实际案例分析。

### 7.4 Deep Research 深度研究

本节深入探讨利用 AI 进行深度研究的技术与应用，包括 Research Agent 的设计与实现。

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

探讨如何将大模型能力转化为实际业务应用与自动化流程。

- [工作流编排指南](06_llm_theory_and_fundamentals/workflow/README.md) - 大模型应用编排平台与自动化工作流实践总览。
- [开源大模型应用编排平台功能与商用许可对比分析](06_llm_theory_and_fundamentals/workflow/open_source_llm_orchestration_platforms_comparison.md) - Dify、AnythingLLM、Ragflow 与 n8n 的深度横评。
- [使用 n8n 构建多智能体系统的实践指南](06_llm_theory_and_fundamentals/workflow/n8n_multi_agent_guide.md) - 基于 n8n 构建 Multi-Agent 系统。
- [Coze 部署和配置手册](06_llm_theory_and_fundamentals/workflow/coze_deployment_and_configuration_guide.md) - Coze 平台的私有化部署与配置指南。

### 7.6 参考书籍

本节列出了深入学习大语言模型理论的优质书籍和阅读材料。

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

大模型的训练是一个复杂且系统的工程，涉及数据处理、分布式训练、指令微调等多个关键环节。本章将详细介绍从指令微调（SFT）到大规模模型预训练的完整技术路径，结合 70B 参数模型的实战案例，深入探讨训练基础设施的搭建、超参数优化及模型后训练（Post-Training）策略。详细指南可参考：[模型训练与微调总览](05_model_training_and_fine_tuning/README.md) 。

### 8.1 指令微调与监督学习

本节介绍指令微调和监督微调（SFT）技术，通过高质量的指令-响应数据对提升模型执行人类指令的能力。

- [SFT 微调实战与指南](05_model_training_and_fine_tuning/sft_example/README.md) - 包含基于 Qwen2 的微调代码实战及垂域模型微调理论指南。
- [Qwen 2 大模型指令微调实战](05_model_training_and_fine_tuning/sft_example/train_qwen2.ipynb) - 基于 Qwen 2 的指令微调 Notebook 实践。
- [Qwen 2 指令微调教程](https://mp.weixin.qq.com/s/Atf61jocM3FBoGjZ_DZ1UA) - 详细的图文教程。
- [一文入门垂域模型 SFT 微调](05_model_training_and_fine_tuning/sft_example/一文入门垂域模型SFT微调.md) - 垂直领域模型的监督微调技术与应用实践。

### 8.2 大规模模型训练实践

本节通过实际的 70B 参数模型训练案例，深入探讨从硬件配置到模型评估的完整训练流程。

- [Training a 70B model from scratch: open-source tools, evaluation datasets, and learnings](https://imbue.com/research/70b-intro/) - 70B 参数模型从零训练的完整技术路径与经验总结。
- [Sanitized open-source datasets for natural language and code understanding: how we evaluated our 70B model](https://imbue.com/research/70b-evals/) - 大规模训练数据集的清洗、评估与质量控制方法。
- [From bare metal to a 70B model: infrastructure set-up and scripts](https://imbue.com/research/70b-infrastructure/) - 大模型训练基础设施的搭建、配置与自动化脚本。
- [Open-sourcing CARBS: how we used our hyperparameter optimizer to scale up to a 70B-parameter language model](https://imbue.com/research/70b-carbs/) - 超参数优化器在大规模模型训练中的应用与调优策略。

### 8.3 模型后训练与评估

本节涵盖 AIOps 场景下的后训练技术、基于 Kubernetes 的评估框架以及基准测试生成方法，确保模型在实际应用中表现稳定。

- [AIOps 后训练技术](05_model_training_and_fine_tuning/ai_ops_design/aiops_post_training.md) - 面向智能运维场景的模型后训练技术与实践。
- [Kubernetes 模型评估框架](05_model_training_and_fine_tuning/ai_ops_design/kubernetes_model_evaluation_framework.md) - 基于 K8s 的大模型评估框架设计与实现。
- [Kubernetes AIOps 基准测试生成框架](05_model_training_and_fine_tuning/ai_ops_design/kubernetes_aiops_benchmark_generation_framework.md) - 自动化生成 AIOps 基准测试数据集的框架设计。

---

## 9. 大模型推理

推理是大模型从实验室走向生产环境的“最后一公里”。本章聚焦于构建高性能、低延迟的推理系统，涵盖推理服务架构设计、核心框架、KV Cache 优化及模型部署实践。通过深入分析 Mooncake 等先进架构及不同规模集群的部署策略，为企业级大模型服务的落地提供全面的技术指导。

### 9.1 推理系统架构设计

推理系统架构直接决定了系统的性能、可扩展性和资源利用效率。本节介绍现代推理系统的核心架构创新与设计模式。

- [**Mooncake 架构详解：以 KV Cache 为中心的高效 LLM 推理系统设计**](./09_inference_system/kv_cache/mooncake/mooncake_architecture.md) - 新一代推理系统的架构创新与性能优化策略

### 9.2 核心框架与平台

本节介绍业界主流的云原生推理框架与平台方案，探讨大模型推理在集群上的最佳实践。

- [**推理优化技术方案**](09_inference_system/README.md) - 企业级推理优化全景指南，涵盖集群规模分析、核心优化技术及实施路径
- [**vLLM + LWS ： Kubernetes 上的多机多卡推理方案**](04_cloud_native_ai_platform/k8s/04_lws_intro.md) - 大模型推理在 Kubernetes 上的最佳实践
- [**云原生高性能分布式 LLM 推理框架 llm-d 介绍**](04_cloud_native_ai_platform/k8s/05_llm_d_intro.md) - 云原生架构下的高性能推理服务栈

### 9.3 KV Cache 核心技术

KV Cache 的高效管理是大模型长文本推理和并发优化的关键。本节深度剖析 LMCache 与 Tair 等分布式 KV Cache 系统的架构与实现。

#### 9.3.1 LMCache 核心架构与后端实现

本小节详细解析 LMCache 的四层存储架构及其在跨实例缓存复用中的技术细节。

- [LMCache 源码分析指南](09_inference_system/kv_cache/lmcache/README.md) - 完整学习路径与文档索引
- [LMCache 架构概览](09_inference_system/kv_cache/lmcache/lmcache_overview.md) - 四层存储架构 (L1-L4)、核心组件交互与典型工作流
- [vLLM KV Offloading 与 LMCache 深度对比](./09_inference_system/kv_cache/advanced_techniques/kv_offloading_analysis.md) - 架构设计、存储层级及跨实例共享能力上的核心差异与性能权衡
- [LMCacheConnector 源码分析](09_inference_system/kv_cache/lmcache/lmcache_connector.md) - vLLM 集成适配器、视图转换与流水线加载
- [LMCacheEngine 源码分析](09_inference_system/kv_cache/lmcache/lmcache_engine.md) - 核心调度中枢、异步事件管理与层级流水线
- [分层存储架构与调度机制](09_inference_system/kv_cache/lmcache/lmcache_storage_overview.md) - StorageManager 调度器、Write-All 策略与 Waterfall 检索
- [LocalCPUBackend 源码分析](09_inference_system/kv_cache/lmcache/local_cpu_backend.md) - 本地 CPU 内存后端与并发控制
- [PDBackend 源码分析](09_inference_system/kv_cache/lmcache/pd_backend.md) - 预填充-解码分离、Push-based 主动推送机制
- [P2PBackend 源码分析](09_inference_system/kv_cache/lmcache/p2p_backend.md) - RDMA 零拷贝与去中心化传输
- [LocalDiskBackend 源码分析](09_inference_system/kv_cache/lmcache/local_disk_backend.md) - O_DIRECT 直通 I/O 与异步优化
- [GdsBackend 源码分析](09_inference_system/kv_cache/lmcache/gds_backend.md) - GPUDirect Storage 零拷贝
- [NixlStorageBackend 源码分析](09_inference_system/kv_cache/lmcache/nixl_backend.md) - 高性能网络存储、S3 对象存储对接
- [Remote Connector 源码分析](09_inference_system/kv_cache/lmcache/remote_connector.md) - Redis/S3/Mooncake 多后端适配
- [LMCache Controller (控制平面)](09_inference_system/kv_cache/lmcache/lmcache_controller.md) - 集群元数据管理、ZMQ 三通道通信与节点协调
- [LMCache Server 源码分析](09_inference_system/kv_cache/lmcache/lmcache_server.md) - 轻量级中心化存储服务、自定义 TCP 协议
- [CacheBlend 技术详解](09_inference_system/kv_cache/lmcache/cache_blend.md) - RAG 场景下的动态融合机制、选择性重算与精度保持
- [CacheGen 技术详解](09_inference_system/kv_cache/lmcache/cachegen.md) - KV Cache 压缩与流式传输、自适应量化与算术编码

#### 9.3.2 阿里云 Tair KVCache

本小节介绍阿里云企业级的 KVCache 管理系统架构及大规模部署实践。

- **[Tair KVCache 架构与设计深度分析](09_inference_system/kv_cache/ali_tair_kvcache/tair-kvcache-architecture-design.md)** - 阿里云企业级 KVCache 管理系统架构详解，包含与 LMCache 的全面对比分析、中心化管理模式及大规模部署最佳实践

### 9.4 推理优化技术体系

推理优化技术体系是提升大模型推理性能的核心技术集合，包括算法优化、硬件加速、系统调优和架构设计等多个维度。

- [**AI 推理优化技术文档导航**](09_inference_system/README.md) - 涵盖基础理论、技术选型、专业领域优化和实施运维的系统性指南
- [**LLM 显存占用分析与计算**](09_inference_system/memory_calc/memory_analysis.md) - 模型参数、KV Cache 与中间激活值的显存估算方法
- [**KV Block Manager 分析**](./09_inference_system/kv_cache/kvbm/KVBM_Analysis.md) - KV Cache 内存管理机制深度解析
- [**分层流水线技术**](./09_inference_system/kv_cache/advanced_techniques/layerwise_pipeline.md) - Layer-wise Pipeline 技术原理与性能优化
- [**NIXL 网络存储介绍**](09_inference_system/infrastructure/nixl_introduction.md) - 高性能网络存储架构与应用
- [**NVIDIA 模型优化器**](09_inference_system/model_optimization/nvidia_model_optimizer.md) - NVIDIA 模型优化工具链详解
- [**vLLM Hybrid KV Cache Manager**](09_inference_system/vllm/module_analysis/vllm_hybrid_kv_cache_manager_deep_dive.md) - vLLM 针对混合注意力架构的显存优化机制
- [**vLLM Router 架构解析**](09_inference_system/vllm/related_module/vllm_router.md) - 高性能、轻量级请求转发系统
- [**vLLM Semantic Router**](09_inference_system/vllm/related_module/vllm_semantic_router_deep_dive.md) - 基于语义的智能路由策略

### 9.5 推理优化参考设计

本系列文档提供了企业级 LLM 推理系统的完整参考设计，涵盖从规模分析到实施落地的全流程指南。

- [背景与目标](09_inference_system/reference_design/01-背景与目标.md) - 推理优化的背景分析与核心目标
- [集群规模分类与特征分析](09_inference_system/reference_design/02-集群规模分类与特征分析.md) - 不同规模集群的特点与需求
- [核心推理优化技术深度解析](09_inference_system/reference_design/03-核心推理优化技术深度解析.md) - KV Cache、批处理、量化等核心技术
- [不同集群规模的技术选型策略](09_inference_system/reference_design/04-不同集群规模的技术选型策略.md) - 针对性的技术方案选择
- [性能评估指标体系](09_inference_system/reference_design/05-性能评估指标体系.md) - 推理性能评估指标与方法
- [推理服务架构设计](09_inference_system/reference_design/06-推理服务架构设计.md) - 企业级推理服务架构设计方案
- [实施建议与最佳实践](09_inference_system/reference_design/07-实施建议与最佳实践.md) - 落地实施的指导建议
- [参考资料与延伸阅读](09_inference_system/reference_design/08-参考资料与延伸阅读.md) - 推荐阅读与延伸资料
- [安全性与合规性](09_inference_system/reference_design/09-安全性与合规性.md) - 推理服务的安全与合规要求
- [多模态推理优化](09_inference_system/reference_design/10-多模态推理优化.md) - 多模态模型推理优化策略
- [边缘推理优化](09_inference_system/reference_design/11-边缘推理优化.md) - 边缘设备上的推理优化方案
- [场景问题解答](09_inference_system/reference_design/12-场景问题解答.md) - 常见问题与解决方案
- [实施检查清单](09_inference_system/reference_design/13-实施检查清单.md) - 推理系统上线检查清单
- [总结与展望](09_inference_system/reference_design/14-总结与展望.md) - 推理优化技术发展趋势

### 9.6 模型部署与运维实践

本节提供将模型转化为可用服务的部署方案与运维经验，涵盖不同硬件平台与框架的实战部署。

- [**动手跑大模型**](99_misc/mac-deepseek-r1.md) - 手把手教你如何跑大模型
- [**Ollama 推理框架详解**](99_misc/ollama/README.md) - Ollama 的架构原理与进阶配置
- [**DeepSeek-V3 MoE 模型 vLLM 部署**](09_inference_system/inference_solutions/deepseek_v3_moe_vllm_h20_deployment.md) - H20 硬件上的部署方案与 SLO 验证
- [**Qwen2-VL-7B 华为昇腾部署**](09_inference_system/inference_solutions/qwen2_vl_7b_huawei.md) - 国产硬件平台的部署优化

### 9.7 DeepSeek 专题

本节聚焦于 DeepSeek 模型的前沿推理优化与硬件适配实践，深度剖析其专有的并行架构设计（如 WideEP），以及在以 Blackwell 为代表的下一代高性能计算平台上的扩展性与部署策略。

- [**vLLM WideEP 架构**](09_inference_system/vllm/hardware_optimization/vllm_deepseek_blackwell_wide_ep.md) - vLLM 宽端点 (Wide Endpoint) 架构解析
- [**Scaling DeepSeek on Blackwell**](09_inference_system/vllm/hardware_optimization/scaling_deepseek_blackwell.pptx) - DeepSeek 在 Blackwell 平台上的扩展性优化

---

## 10. 企业级 AI Agent 开发

本章深入探讨企业级 AI Agent 开发的完整技术体系。

> 详细内容请访问：[**AI Agent 开发与实践**](08_agentic_system/README.md) - 核心文档门户，涵盖理论、架构与实战。

### 10.1 核心模块导航

本节梳理了多智能体系统、记忆系统、上下文工程及基础设施等核心模块。

- **[多智能体系统](08_agentic_system/multi_agent/Part1-Multi-Agent-AI-Fundamentals.md)**：BDI 架构、多 Agent 协作机制与企业级落地
  - [企业级多智能体系统实现](08_agentic_system/multi_agent/Part2-Enterprise-Multi-Agent-System-Implementation.md) - 企业级多 Agent 系统架构与实现
- **[记忆系统](08_agentic_system/memory/docs/AI%20智能体记忆系统：理论与实践.md)**：MemoryOS 架构、Mem0 实战与 LangChain 记忆集成
  - [大模型 Agent 记忆综述](08_agentic_system/memory/docs/大模型Agent记忆综述.md) - Agent 记忆系统的理论基础与研究进展
  - [Mem0 快速入门](08_agentic_system/memory/docs/mem0快速入门.md) - Mem0 记忆系统的安装与使用指南
  - [MemoryOS 智能记忆系统架构设计](08_agentic_system/memory/docs/MemoryOS智能记忆系统架构设计与开发指南.md) - MemoryOS 系统架构与开发指南
  - [MemMachine 深度解析](08_agentic_system/memory/docs/MemMachine深度解析.md) - MemMachine 记忆系统技术原理
  - [LangChain 记忆集成](08_agentic_system/memory/langchain/langchain_memory.md) - LangChain 记忆模块的使用与最佳实践
- **[上下文工程](08_agentic_system/context/上下文工程原理.md)**：动态组装、自适应压缩与 Anthropic 最佳实践
  - [上下文工程原理简介](08_agentic_system/context/上下文工程原理简介.md) - 上下文工程核心概念快速入门
  - [Anthropic 上下文工程指南](08_agentic_system/context/anthropic_context_engineering_zh.md) - Anthropic 官方上下文工程最佳实践
  - [LangChain 上下文工程实践](08_agentic_system/context/langchain_with_context_engineering.md) - 基于 LangChain 的上下文管理实现
- **[工具与 MCP](08_agentic_system/mcp/01_deep_dive_into_mcp_and_the_future_of_ai_tooling.md)**：Model Context Protocol (MCP) 原理与实战
- **[基础设施](08_agentic_system/agent_infra/ai-agent-infra-stack.md)**：Agent 基础设施技术栈
  - [AI Agent 基础设施的崛起](08_agentic_system/agent_infra/the-rise-of-ai-agent-infrastructure.md) - Agent 基础设施发展趋势与技术栈分析
  - [12-Factor Agents 设计原则](08_agentic_system/concepts/12-factor-agents-intro.md) - 构建可靠 Agent 的 12 条设计原则

### 10.2 设计模式与技能

本节总结了构建高效 Agent 所需的设计模式与核心技能。

- **[Agent 设计模式](08_agentic_system/agent_design/react-agent.md)**：ReAct 范式、[写作 Agent](08_agentic_system/agent_design/写作%20Agentic%20Agent.md) 与 [指代消解](08_agentic_system/agent_design/如何设计支持多轮指代消解的对话系统.md)
- **[Agent Skills](08_agentic_system/agent-skills/claude_skills_guide.md)**：Claude Skills 开发指南与 PDF Translator 实战
- **[世界模型](08_agentic_system/concepts/world_model_introduction.md)**：World Model 核心概念与应用

### 10.3 深度报告与论文

本节精选了业界关于 Agent 的深度研究报告和学术论文。

- **[Agent Workflow Survey](08_agentic_system/paper/agent-workflow-survey.md)**：Agent 工作流综述
- **[Deep Research Agent](08_agentic_system/paper/deepresearch-agent.md)**：深度研究 Agent 的设计与实现
- **[LangChain State of Agent Engineering](08_agentic_system/report/langchain-state-of-agent-engineering.md)**：LangChain 发布的 Agent 工程化现状报告

### 10.4 其他资料

本节包含额外的 Agent 框架深度解析资料。

- **[OpenVikin 深度解析](./08_agentic_system/context/OpenVikin-deep-dive.md)**：OpenVikin Agent 框架深度剖析

---

## 11. 检索增强生成与文档智能

本章聚焦于检索增强生成（`RAG`）与文档智能化处理技术，提供从非结构化数据解析到知识库构建的完整解决方案。

> 详细内容请访问：[rag 与工具生态](07_rag_and_tools/README.md) - 核心文档门户，涵盖 `RAG`、`GraphRAG` 与文档智能工具。

### 11.1 检索增强生成基础与进阶

探索 `RAG` 系统的核心组件、策略对比与模型选型，构建高效的检索增强生成系统。

- [rag 快速开发实战（从 0 到 1 搭建）](https://mp.weixin.qq.com/s/89-bwZ4aPor4ySj5U3n5zw) - `RAG` 技术全景导航，涵盖基础概念到进阶优化
- [rag 策略对比](07_rag_and_tools/rag_basics/rag_comparison.md) - 不同 `RAG` 架构（`Naive RAG`、`Advanced RAG` 等）的优劣势分析
- [chunking 策略评估总结](07_rag_and_tools/rag_basics/evaluating_chunking_strategies_summary.md) - 检索分块策略的深度总结与最佳实践
- [中文 rag 系统 embedding 选型指南](07_rag_and_tools/rag_basics/chinese_rag_embedding_model_selection.md) - 面向中文场景的 `Embedding` 模型评测与推荐

### 11.2 图检索增强生成与知识图谱

结合知识图谱增强 `RAG` 的推理能力，深入 `GraphRAG` 前沿技术，解决复杂关系推理难题。

- [graphrag 学习指南](07_rag_and_tools/graph_rag/graph_rag_learning_guide.md) - `GraphRAG` 的核心概念、架构原理与入门路径
- [kag 框架介绍](07_rag_and_tools/graph_rag/kag_introduction.md) - `Knowledge Augmented Generation`（`KAG`）框架深度解析
- [neo4j 实战指南](07_rag_and_tools/knowledge_graph/neo4j_handson_guide.md) - 图数据库 `Neo4j` 的安装、配置与企业级实战
- [neo4j cypher 教程](07_rag_and_tools/knowledge_graph/neo4j_cypher_tutorial.md) - `Neo4j` 查询语言 `Cypher` 完整教程

### 11.3 大模型与知识图谱协同应用

探索大语言模型（`LLM`）与知识图谱的深度融合，构建高可信、可解释的智能应用。

- [银行反电诈智能系统设计](07_rag_and_tools/synergized_llms_kgs/anti_fraud_design.md) - 基于 `LLM` + `KG` 的金融风控系统设计方案，实战反欺诈场景
- [反欺诈 demo 源码](07_rag_and_tools/synergized_llms_kgs/demo/README.md) - 完整的反欺诈系统演示代码，包含数据生成、图谱构建与智能体推理

### 11.4 文档智能解析

高效处理非结构化文档（`PDF`、`Office` 等），为 `RAG` 系统提供高质量的数据输入，解决“垃圾进，垃圾出”（Garbage In, Garbage Out）问题。

- [mineru 文档解析](07_rag_and_tools/pdf_tools/miner_u_intro.md) - 上海人工智能实验室开源工具，助力复杂 `PDF` 高效解析
- [marker pdf 布局检测](07_rag_and_tools/pdf_tools/marker_zh_cn.md) - 基于深度学习的高精度 `PDF` 解析与布局分析引擎
- [markitdown 入门](07_rag_and_tools/pdf_tools/markitdown/markitdown_intro.md) - Microsoft 开源的文档转换工具，支持多种办公文档格式到 `Markdown` 的高质量转换

---

## 12. 课程体系与学习路径

本章汇总了 AI 基础、系统开发、编程实战等全方位的课程体系，为学习者提供清晰的学习路径和进阶指南。

### 12.1 AI System 全栈课程（ZOMI 酱）

ZOMI 酱（陈佐钘）主导的 AI 系统全栈开源课程，涵盖从底层硬件芯片到上层 AI 框架设计的全技术栈内容。该课程在 GitHub 上广受好评（Star 数超 16.5k），是了解 AI 基础设施架构的绝佳资源。

[AISystem](https://github.com/chenzomi12/AISystem) - AI 系统全栈课程代码与资料库。

- [系统介绍](https://github.com/chenzomi12/AISystem/tree/main/01Introduction) - AI 系统概述、发展历程与技术演进路径。
- [硬件基础](https://github.com/chenzomi12/AISystem/tree/main/02Hardware) - AI 芯片架构、硬件加速器与计算平台深度解析。
- [编译器技术](https://github.com/chenzomi12/AISystem/tree/main/03Compiler) - AI 编译器原理、优化技术与工程实践。
- [推理优化](https://github.com/chenzomi12/AISystem/tree/main/04Inference) - 模型推理加速技术、性能调优与部署策略。
- [框架设计](https://github.com/chenzomi12/AISystem/tree/main/05Framework) - AI 框架架构设计、分布式计算与并行优化。

### 12.2 AI Infra 基础课程（入门）

本节提供面向初学者的 AI 基础设施基础课程，帮助快速建立领域知识体系。

- [大模型原理与最新进展](10_ai_related_course/ai_coding/index.html) - 交互式在线课程平台。
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

本节提供系统化的 Trae 编程学习体系，助力开发者掌握 AI 辅助编程的实战技巧。

- [Trae 编程实战教程](10_ai_related_course/trae/README.md) - 从基础入门到高级应用的完整 Trae 编程学习路径。

**课程结构：**

- **第一部分：Trae 基础入门**：环境配置、交互模式、HelloWorld 项目实战。
- **第二部分：常见编程场景实战**：前端开发、Web 开发、后端 API、数据库设计、安全认证。
- **第三部分：高级应用场景**：AI 模型集成、实时通信、数据分析、微服务架构。
- **第四部分：团队协作与最佳实践**：代码质量管理、项目管理、性能优化、DevOps 实践。
- **第五部分：综合项目实战**：企业级应用开发、核心功能实现、部署运维实战。

### 12.4 多智能体 AI 系统培训

本节面向企业技术团队，提供从理论基础到实战应用的完整多智能体系统构建指南。

- [多智能体 AI 系统培训材料](10_ai_related_course/multi_agent_system/multi_agent_training/README.md)：涵盖 LangGraph 框架深度解析、LangSmith 监控集成及企业级架构设计。

---

## Buy Me a Coffee

如果您觉得本项目对您有帮助，欢迎购买我一杯咖啡，支持我继续创作和维护。

| **微信**                                                 | **支付宝**                                            |
| -------------------------------------------------------- | ----------------------------------------------------- |
| <img src="./img/weixinpay.JPG" alt="wechat" width="200"> | <img src="./img/alipay.JPG" alt="alipay" width="200"> |

---
