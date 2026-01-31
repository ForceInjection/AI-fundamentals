# AI Fundamentals

本仓库是一个全面的人工智能基础设施（`AI Infrastructure`）学习资源集合，涵盖从硬件基础到高级应用的完整技术栈。内容包括 GPU 架构与编程、`CUDA` 开发、大语言模型、`AI` 系统设计、性能优化、企业级部署等核心领域，旨在为 `AI` 工程师、研究人员和技术爱好者提供系统性的学习路径和实践指导。

> **适用人群**：`AI` 工程师、系统架构师、`GPU` 编程开发者、大模型应用开发者、技术研究人员。
> **技术栈**：`CUDA`、`GPU` 架构、`LLM`、`AI` 系统、分布式计算、容器化部署、性能优化。

---

**Star History**:

## [![Star History Chart](https://api.star-history.com/svg?repos=ForceInjection/AI-fundermentals&type=date&legend=top-left)](https://www.star-history.com/#ForceInjection/AI-fundermentals&date&legend=top-left)

---

## 1. 硬件架构

本章节主要构建 AI 系统的物理底座，深入探讨从单机计算芯片（GPU/TPU）到大规模集群互联的核心技术。

> 详细内容请访问：[**硬件与架构**](01_hardware_architecture/README.md) - 核心文档门户，涵盖硬件基础知识与关键技术概览。

**核心模块导航**：

- **[GPU 与 AI 加速器架构](01_hardware_architecture/README.md)**：涵盖 NVIDIA GPU 架构、Google TPU 设计哲学、GPUDirect 核心技术及 GPGPU vs NPU 对比分析。
- **[AI 集群运维与通信](03_ai_cluster_ops/README.md)**：包含 GPU 基础运维、InfiniBand 高性能网络及 NCCL 分布式通信实战。
- **[性能分析与调优](02_gpu_programming/profiling/README.md)**：AI 系统全栈性能分析与瓶颈诊断。

---

## 2. 云原生 AI 基础设施

本章聚焦于云原生技术在 AI 领域的应用，探讨如何利用 Kubernetes 等云原生技术栈构建高效、可扩展的 AI 基础设施。

### 2.1 Kubernetes AI 生态

Kubernetes 已成为云原生 AI 基础设施的事实标准，特别是在推理场景中，它提供了不可替代的弹性调度与资源管理能力。通过 K8s，企业可以构建跨混合云的统一推理平台，实现从 GPU 资源池化到 Serverless 推理的完整闭环，从容应对大模型时代高并发、波动剧烈的流量挑战。

- [**Kubernetes AI 平台实战**](04_cloud_native_ai_platform/k8s/README.md) - 云原生 AI 基础设施建设指南
- [**Kueue + HAMi 集成方案**](04_cloud_native_ai_platform/k8s/Kueue%20+%20HAMi.md) - GPU 资源调度与管理的云原生解决方案
- [**NVIDIA Container Toolkit 原理分析**](04_cloud_native_ai_platform/k8s/Nvidia%20Container%20Toolkit%20原理分析.md) - 容器化 GPU 支持的底层机制
- [**NVIDIA K8s Device Plugin 分析**](04_cloud_native_ai_platform/k8s/nvidia-k8s-device-plugin-analysis.md) - GPU 设备插件的架构与实现

### 2.2 AI 推理系统与服务

本节整合了从云原生推理框架到企业级推理系统优化的完整解决方案，涵盖理论基础、技术选型及实战部署。

- [**推理优化技术方案**](09_inference_system/README.md) - 企业级推理优化全景指南，涵盖集群规模分析、核心优化技术及实施路径
- [**云原生高性能分布式 LLM 推理框架 llm-d 介绍**](04_cloud_native_ai_platform/k8s/llm-d-intro.md) - 基于 Kubernetes 的大模型推理框架
- [**vLLM + LWS ： Kubernetes 上的多机多卡推理方案**](04_cloud_native_ai_platform/k8s/lws_intro.md) - LWS 分布式控制器在推理部署中的应用

**核心技术与方案：**

- **推理系统架构**：[Mooncake 架构详解](09_inference_system/Mooncake%20架构详解：以%20KV%20缓存为中心的高效%20LLM%20推理系统设计.md) - 以 KV 缓存为中心的高效 LLM 推理系统设计
- **KV 缓存优化 (LMCache)**：
  - [LMCache 源码分析指南](09_inference_system/lmcache/README.md) - 文档入口与推荐阅读路径
  - [LMCache 架构概览](09_inference_system/lmcache/lmcache_overview.md) - 系统定位、四层存储架构 (L1-L4) 与组件交互
  - [LMCache Controller (控制平面) 架构剖析](09_inference_system/lmcache/lmcache_controller.md) - 集群元数据管理、节点协调及全局指令下发
  - [LMCacheConnector 源码分析](09_inference_system/lmcache/lmcache_connector.md) - 推理引擎 (如 vLLM) 集成入口与请求拦截
  - [LMCacheEngine 源码分析](09_inference_system/lmcache/lmcache_engine.md) - 核心控制流、I/O 编排与元数据管理
  - [分层存储架构与调度机制](09_inference_system/lmcache/lmcache_storage_overview.md) - StorageManager 调度、Write-All 与 Waterfall 检索
  - [LocalCPUBackend 源码分析](09_inference_system/lmcache/local_cpu_backend.md) - L1 本地 CPU 内存后端与并发控制
  - [P2PBackend 源码分析](09_inference_system/lmcache/p2p_backend.md) - L2 弹性互联层与跨节点传输机制
  - [LocalDiskBackend 源码分析](09_inference_system/lmcache/local_disk_backend.md) - L3 本地磁盘后端与 I/O 优化
  - [Remote Connector (远程连接器) 源码分析](09_inference_system/lmcache/remote_connector.md) - L4 共享存储接口与 Redis/S3/Mooncake 实现
  - [LMCache Server 源码分析](09_inference_system/lmcache/lmcache_server.md) - LMCache Server 服务端架构与协议分析
  - [PDBackend (预填充-解码分离后端) 源码分析](09_inference_system/lmcache/pd_backend.md) - 专为分离架构设计的 KV Cache 主动推送机制
- **部署实战**：
  - [DeepSeek-V3 MoE 模型 vLLM 部署](09_inference_system/inference-solution/DeepSeek-V3-MoE-vLLM-H20-Deployment.md) - H20 硬件上的部署方案与 SLO 验证
  - [Qwen2-VL-7B 华为昇腾部署](09_inference_system/inference-solution/Qwen2-VL-7B_Huawei.md) - 国产硬件平台的部署优化

---

## 3. 开发与编程

本部分专注于 `AI` 开发相关的编程技术、工具和实践，涵盖从基础编程到高性能计算的完整技术栈。

### 3.1 GPU 与 CUDA 编程

本节整合了 GPU 基础架构、CUDA 核心编程概念及丰富的学习资源，为开发者提供从入门到进阶的完整技术路径。

#### 3.1.1 核心概念

- [**CUDA 核心概念详解**](02_gpu_programming/cuda/cuda_cores_cn.md) - CUDA 核心、线程块、网格等基础概念的深度解析
- [**CUDA 流详解**](02_gpu_programming/cuda/cuda_streams.md) - CUDA 流的原理、应用场景与性能优化

**技术特色：**

- **CUDA 核心架构**： SIMT 线程模型、分层内存模型、流式执行模型
- **性能调优实践**：内存访问模式优化、线程同步策略、算法并行化重构
- **高级编程特性**： Unified Memory 统一内存、Multi-GPU 多卡编程、CUDA Streams 异步执行

#### 3.1.2 GPU 编程基础

- [**GPU 编程基础**](02_gpu_programming/README.md) - GPU 编程入门到进阶的完整技术路径，涵盖 GPU 架构、编程模型和性能优化

**核心内容：**

- **GPU 架构理解**：GPU 与 CPU 的架构差异、并行计算原理、内存层次结构
- **CUDA 编程实践**：线程模型、内存管理、核函数编写、性能优化技巧
- **调试与性能分析**：CUDA 调试工具、性能分析方法、瓶颈识别与优化
- **高级特性应用**：流处理、多 GPU 编程、与深度学习框架的集成

### 3.2 DPU 编程

本节介绍 NVIDIA BlueField DPU 及其 DOCA 软件框架的编程指南。

- [**DPU 编程与 DOCA 框架**](02_dpu_programming/README.md) - DPU 编程入门与 DOCA 核心组件解析

### 3.3 Java AI 开发

这里的 Java AI 开发主要用于开发 LLM 应用。

- [**Java AI 开发指南**](98_llm_programming/java_ai/README.md) - Java 生态系统中的 AI 开发技术
- [**使用 Spring AI 构建高效 LLM 代理**](98_llm_programming/java_ai/spring_ai_cn.md) - 基于 Spring AI 框架的企业级 AI 应用开发

### 3.4 AI 编程范式

本节探讨在 AI 时代下新兴的编程范式与工作流，重点关注如何利用 AI 提升开发效率与代码质量。

- [**OpenSpec 实战指南**](https://github.com/ForceInjection/OpenSpec-practise/blob/main/README.md) - Spec 驱动开发 (Spec-Driven Development) 的工程实践，演示了 "意图 -> Spec -> AI -> 代码 & 验证" 的新一代开发工作流。

---

## 4. 机器学习基础

本部分基于 [**动手学机器学习**](https://github.com/ForceInjection/hands-on-ML) 项目，提供系统化的机器学习学习路径。该项目整合了 NJU 软件学院课程、上海交大《动手学机器学习》、《精通特征工程》以及极客时间等优质资源，为学习大模型打下基础。

### 4.1 动手学机器学习

[**动手学机器学习**](https://github.com/ForceInjection/hands-on-ML/blob/main/README.md) - 全面的机器学习学习资源库，包含理论讲解、代码实现和实战案例。

**核心特色：**

- **理论与实践结合**：以 NJU 课程为主线，辅以 SJTU 配套资源，从数学原理到代码实现的完整学习路径
- **算法全覆盖**：涵盖监督学习、无监督学习、集成学习、推荐系统、概率图模型及深度学习
- **项目驱动学习**：提供心脏病预测、鸢尾花分类、房价预测等实战案例
- **工程化实践**：深入特征工程、模型评估、超参数调优及特征选择

### 4.2 参考资料

我们精选了数学基础、经典教材与实战平台资源，构建完整的知识图谱。

**数学基础：**

- [**线性代数的本质**](https://www.bilibili.com/video/BV1ys411472E) - 3Blue1Brown 可视化教程，直观理解线性变换与矩阵运算
- [**MIT 18.06 线性代数**](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) - Gilbert Strang 经典课程，深入矩阵分解与子空间理论
- [**概率论与统计学基础**](https://book.douban.com/subject/35798663/) - 掌握贝叶斯定理、最大似然估计与概率分布

**经典教材：**

- **《统计学习方法》** - 李航著，系统阐述感知机、SVM、HMM 等核心算法的数学原理
- **《机器学习》** - 周志华著（西瓜书），全面覆盖机器学习基础理论与范式
- **《模式识别与机器学习》** - Bishop 著（PRML），贝叶斯视角的机器学习圣经

**在线课程与实战：**

- [**Andrew Ng 机器学习课程**](https://www.coursera.org/learn/machine-learning) - Coursera 经典入门，强调直觉理解
- [**CS229 机器学习**](http://cs229.stanford.edu/) - 斯坦福进阶课程，深入数学推导
- [**Kaggle**](https://www.kaggle.com/) - 全球最大的数据科学竞赛平台，提供真实数据集与 Notebook 环境

---

## 5. 大语言模型基础

本章旨在为读者构建扎实的大语言模型（LLM）理论基础，涵盖从词向量嵌入到模型架构设计的核心知识。我们将深入解析 Token 机制、Transformer 架构、混合专家模型（MoE）等关键技术，并探讨量化、思维链（CoT）等前沿优化方向，帮助开发者建立对 LLM 内部机制的直观理解。

### 5.1 基础理论与概念

大语言模型的基础理论涵盖了从文本处理到模型架构的核心概念。理解这些基础概念是深入学习 `LLM` 技术的前提，包括 Token 化机制、文本编码、模型结构等关键技术。这些基础知识为后续的模型训练、优化和应用奠定了坚实的理论基础。

- [**Andrej Karpathy ： Deep Dive into LLMs like ChatGPT （B 站视频）**](https://www.bilibili.com/video/BV16cNEeXEer) - 深度学习领域权威专家的 LLM 技术解析
- [**大模型基础组件 - Tokenizer**](https://zhuanlan.zhihu.com/p/651430181) - 文本分词与编码的核心技术
- [**解密大语言模型中的 Tokens**](06_llm_theory_and_fundamentals/llm_basic_concepts/token/llm_token_intro.md) - Token 机制的深度解析与实践应用
  - [**Tiktokenizer 在线版**](https://tiktokenizer.vercel.app/?model=gpt-4o) - 交互式 Token 分析工具

### 5.2 嵌入技术与表示学习

嵌入技术是大语言模型的核心组件之一，负责将离散的文本符号转换为连续的向量表示。这一技术不仅影响模型的理解能力，还直接关系到模型的性能和效率。本节深入探讨文本嵌入的原理、实现方式以及在不同场景下的应用策略。

- [**文本嵌入（Text-Embedding） 技术快速入门**](06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/text_embeddings_guide.md) - 文本向量化的理论基础与实践
- [**LLM 嵌入技术详解：图文指南**](06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/LLM%20Embeddings%20Explained%20-%20A%20Visual%20and%20Intuitive%20Guide.zh-CN.md) - 可视化理解嵌入技术
- [**大模型 Embedding 层与独立 Embedding 模型：区别与联系**](06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/embedding.md) - 嵌入层架构设计与选型策略

### 5.3 高级架构与优化技术

现代大语言模型采用了多种先进的架构设计和优化技术，以提升模型性能、降低计算成本并解决特定问题。本节涵盖混合专家系统、量化技术、思维链推理等前沿技术，这些技术代表了当前 LLM 领域的最新发展方向。

- [**大模型可视化指南**](https://www.maartengrootendorst.com/) - 大模型内部机制的可视化分析
- [**一文读懂思维链（Chain-of-Thought, CoT）**](06_llm_theory_and_fundamentals/llm_basic_concepts/一文读懂思维链（Chain-of-Thought%2C%20CoT）.md) - 推理能力增强的核心技术
- [**大模型的幻觉及其应对措施**](06_llm_theory_and_fundamentals/llm_basic_concepts/大模型的幻觉及其应对措施.md) - 幻觉问题的成因分析与解决方案
- [**大模型文件格式完整指南**](06_llm_theory_and_fundamentals/llm_basic_concepts/大模型文件格式完整指南.md) - 模型存储与部署的技术规范
- [**量化技术可视化指南**](06_llm_theory_and_fundamentals/llm_basic_concepts/A%20Visual%20Guide%20to%20Quantization.zh-CN.md) - 模型压缩与加速的核心技术
- [**混合专家模型 (MoE) 可视化指南**](<06_llm_theory_and_fundamentals/llm_basic_concepts/A%20Visual%20Guide%20to%20Mixture%20of%20Experts%20(MoE).zh-CN.md>) - 深入解析 MoE 架构原理
- [**基于大型语言模型的意图检测**](06_llm_theory_and_fundamentals/llm_basic_concepts/Intent%20Detection%20using%20LLM.zh-CN.md) - 自然语言理解的实际应用

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

- [**SFT 微调实战与指南**](05_model_training_and_fine_tuning/sft_example/README.md) - 包含基于 Qwen2 的微调代码实战及垂域模型微调理论指南
- [**Qwen 2 大模型指令微调实战**](05_model_training_and_fine_tuning/sft_example/train_qwen2.ipynb) - 基于 Qwen 2 的指令微调 Notebook 实践
- [**Qwen 2 指令微调教程**](https://mp.weixin.qq.com/s/Atf61jocM3FBoGjZ_DZ1UA) - 详细的图文教程
- [**一文入门垂域模型 SFT 微调**](05_model_training_and_fine_tuning/sft_example/一文入门垂域模型SFT微调.md) - 垂直领域模型的监督微调技术与应用实践

### 6.2 大规模模型训练实践

大规模模型训练是一个复杂的系统工程，涉及数据处理、基础设施搭建、分布式训练、超参数优化等多个方面。本节通过实际的 70B 参数模型训练案例，深入探讨从硬件配置到模型评估的完整训练流程，为大规模模型训练提供实践指导。

- [**Training a 70B model from scratch: open-source tools, evaluation datasets, and learnings**](https://imbue.com/research/70b-intro/) - 70B 参数模型从零训练的完整技术路径与经验总结
- [**Sanitized open-source datasets for natural language and code understanding: how we evaluated our 70B model**](https://imbue.com/research/70b-evals/) - 大规模训练数据集的清洗、评估与质量控制方法
- [**From bare metal to a 70B model: infrastructure set-up and scripts**](https://imbue.com/research/70b-infrastructure/) - 大模型训练基础设施的搭建、配置与自动化脚本
- [**Open-sourcing CARBS: how we used our hyperparameter optimizer to scale up to a 70B-parameter language model**](https://imbue.com/research/70b-carbs/) - 超参数优化器在大规模模型训练中的应用与调优策略

### 6.3 模型后训练与评估

模型后训练（Post-Training）和评估是确保模型在实际应用中表现稳定、可靠的关键步骤。本节涵盖 AIOps 场景下的后训练技术、基于 Kubernetes 的评估框架以及基准测试生成方法。

- [**AIOps 后训练技术**](05_model_training_and_fine_tuning/ai_ops_design/aiops_post_training.md) - 面向智能运维场景的模型后训练技术与实践
- [**Kubernetes 模型评估框架**](05_model_training_and_fine_tuning/ai_ops_design/kubernetes_model_evaluation_framework.md) - 基于 K8s 的大模型评估框架设计与实现
- [**Kubernetes AIOps 基准测试生成框架**](05_model_training_and_fine_tuning/ai_ops_design/kubernetes_aiops_benchmark_generation_framework.md) - 自动化生成 AIOps 基准测试数据集的框架设计

---

## 7. 大模型推理

推理是大模型从实验室走向生产环境的“最后一公里”。本章聚焦于构建高性能、低延迟的推理系统，涵盖推理服务架构设计、KV Cache 优化、模型量化压缩等核心技术。通过深入分析 Mooncake 等先进架构及不同规模集群的部署策略，为企业级大模型服务的落地提供全面的技术指导。

### 7.1 推理系统架构设计

推理系统架构是大模型服务化的核心基础，直接决定了系统的性能、可扩展性和资源利用效率。现代推理系统需要在低延迟、高吞吐量和成本效益之间找到最佳平衡点，同时支持动态批处理、内存优化和多模型并发等高级特性。

- [**Mooncake 架构详解：以 KV 缓存为中心的高效 LLM 推理系统设计**](09_inference_system/Mooncake%20架构详解：以%20KV%20缓存为中心的高效%20LLM%20推理系统设计.md) - 新一代推理系统的架构创新与性能优化策略

### 7.2 模型部署与运维实践

模型部署与运维是将训练好的大模型转化为可用服务的关键环节，涉及模型格式转换、环境配置、服务监控和故障处理等多个方面。有效的部署策略能够显著降低运维成本，提高服务稳定性和用户体验。

- [**动手部署 ollama**](99_misc/deepseek/mac-deepseek-r1.md) - 轻量级本地大模型部署的完整实践指南

### 7.3 推理优化技术体系

推理优化技术体系是提升大模型推理性能的核心技术集合，包括算法优化、硬件加速、系统调优和架构设计等多个维度。

- [**AI 推理优化技术文档导航**](09_inference_system/README.md) - 涵盖基础理论、技术选型、专业领域优化和实施运维的系统性指南

### 7.4 DeepSeek 专题

DeepSeek 是当前开源大模型领域的重要力量，其创新的架构设计和高性能表现备受关注。本节汇总了关于 DeepSeek 模型的部署、对比分析和存储系统设计等核心资料。

- [**DeepSeek 3FS 存储系统**](04_cloud_native_ai_platform/storage/deepseek_3fs_design_notes.zh-CN.md) - DeepSeek 自研的高性能分布式文件系统设计笔记

---

## 8. 企业级 AI Agent 开发

本章深入探讨企业级 `AI Agent` 开发的完整技术体系。详细内容请访问：

- [**AI Agent 开发与实践**](08_agentic_system/README.md) - 核心文档门户，涵盖理论、架构与实战。

**核心模块导航**：

- **[多智能体系统](08_agentic_system/multi_agent/Part1-Multi-Agent-AI-Fundamentals.md)**：BDI 架构、多 Agent 协作机制与企业级落地
- **[记忆系统](08_agentic_system/memory/docs/AI%20智能体记忆系统：理论与实践.md)**：MemoryOS 架构、Mem0 实战与 LangChain 记忆集成
- **[上下文工程](08_agentic_system/context/上下文工程原理.md)**：动态组装、自适应压缩与 Anthropic 最佳实践
- **[工具与 MCP](08_agentic_system/mcp/A_Deep_Dive_Into_MCP_and_the_Future_of_AI_Tooling_zh_CN.md)**：Model Context Protocol (MCP) 原理与实战
- **[基础设施](08_agentic_system/agent_infra/ai-agent-infra-stack.md)**：Agent 基础设施技术栈与 12-Factor Agents 设计原则

---

## 9. RAG 与文档智能

本章聚焦于检索增强生成（RAG）与文档智能化处理技术，提供从非结构化数据解析到知识库构建的完整解决方案。详细内容请访问：

- [**RAG 与工具生态**](07_rag_and_tools/README.md) - 核心文档门户，涵盖 RAG、GraphRAG 与文档智能工具。

**核心模块导航**：

- **[RAG 基础与进阶](07_rag_and_tools/rag/README.md)**：RAG 技术全景、Chunking 策略与 Embedding 选型
- **[GraphRAG 与知识图谱](07_rag_and_tools/GraphRAG/GraphRAG_Learning_Guide.md)**：GraphRAG 原理、Neo4j 实战与 KAG 框架
- **[LLM + KG 协同应用](07_rag_and_tools/Synergized%20LLMs%20+%20KGs/anti_fraud_design.md)**：金融反欺诈系统设计与 Demo 源码
- **[文档智能解析](07_rag_and_tools/pdf/minerU_intro.md)**：MinerU、Marker 与 Markitdown 等高精度解析工具

**深度研究与工具**：

- [**DeepWiki 技术原理**](06_llm_theory_and_fundamentals/deep_research/DeepWiki%20使用方法与技术原理深度分析.md) - DeepWiki 使用方法与技术原理深度分析

**特定领域应用**：

- [**ChatBox 意图识别**](06_llm_theory_and_fundamentals/llm_basic_concepts/ChatBox_Intent_Recognition_and_Semantic_Understanding_Half_Sentence.md) - 意图识别与语义理解机制解析

---

## 10. 开源模型与框架生态

本章汇聚了 AI 领域前沿的开源模型与计算框架，聚焦于大模型训练、微调和推理的核心技术，涵盖高性能中文大模型、高效微调工具和推理优化框架，为开发者提供高性能的技术选型参考。

- [**DeepSeek**](https://github.com/DeepSeek-AI/) - 基于 Transformer 的高性能中文大模型，具备强大的推理能力与多语言支持
- [**unsloth**](https://github.com/unslothai/unsloth) - 高效大模型微调框架，支持 Llama 3.3、DeepSeek-R1 等模型 2 倍速度提升与 70% 内存节省
- [**ktransformers**](https://github.com/kvcache-ai/ktransformers) - 灵活的大模型推理优化框架，提供前沿的推理加速技术

---

## 11. 课程体系与学习路径

本章汇总了 AI 基础、系统开发、编程实战等全方位的课程体系，为学习者提供清晰的学习路径和进阶指南。

### 11.1 AI System 全栈课程（ZOMI 酱）

[**AISystem**](https://github.com/Infrasys-AI/AISystem) - ZOMI 酱的 AI 系统全栈课程，涵盖从硬件基础到框架设计的全技术栈内容：

- [**系统介绍**](https://github.com/Infrasys-AI/AISystem/tree/main/01Introduction) - AI 系统概述、发展历程与技术演进路径
- [**硬件基础**](https://github.com/Infrasys-AI/AISystem/tree/main/02Hardware) - AI 芯片架构、硬件加速器与计算平台深度解析
- [**编译器技术**](https://github.com/Infrasys-AI/AISystem/tree/main/03Compiler) - AI 编译器原理、优化技术与工程实践
- [**推理优化**](https://github.com/Infrasys-AI/AISystem/tree/main/04Inference) - 模型推理加速技术、性能调优与部署策略
- [**框架设计**](https://github.com/Infrasys-AI/AISystem/tree/main/05Framework) - AI 框架架构设计、分布式计算与并行优化

### 11.2 AI Infra 基础课程（入门）

- [**大模型原理与最新进展**](10_ai_related_course/ai_coding/index.html) - 交互式在线课程平台
- [**AI Infra 课程演讲稿**](10_ai_related_course/ai_infra_course/%E5%85%A5%E9%97%A8%E7%BA%A7/%E8%AE%B2%E7%A8%BF.md) - 完整的课程演讲内容、技术要点与实践案例
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

- [**Trae 编程实战教程**](10_ai_related_course/trae/README.md) - 从基础入门到高级应用的完整 Trae 编程学习路径

**课程结构：**

- **第一部分：Trae 基础入门**：环境配置、交互模式、HelloWorld 项目实战
- **第二部分：常见编程场景实战**：前端开发、Web 开发、后端 API、数据库设计、安全认证
- **第三部分：高级应用场景**：AI 模型集成、实时通信、数据分析、微服务架构
- **第四部分：团队协作与最佳实践**：代码质量管理、项目管理、性能优化、DevOps 实践
- **第五部分：综合项目实战**：企业级应用开发、核心功能实现、部署运维实战

---

## Buy Me a Coffee

如果您觉得本项目对您有帮助，欢迎购买我一杯咖啡，支持我继续创作和维护。

| **微信**                                                 | **支付宝**                                            |
| -------------------------------------------------------- | ----------------------------------------------------- |
| <img src="./img/weixinpay.JPG" alt="wechat" width="200"> | <img src="./img/alipay.JPG" alt="alipay" width="200"> |

---
