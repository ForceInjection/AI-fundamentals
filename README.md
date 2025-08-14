# AI Fundamentals

本仓库是一个全面的人工智能基础设施（`AI Infrastructure`）学习资源集合，涵盖从硬件基础到高级应用的完整技术栈。内容包括GPU架构与编程、`CUDA` 开发、大语言模型、`AI` 系统设计、性能优化、企业级部署等核心领域，旨在为`AI` 工程师、研究人员和技术爱好者提供系统性的学习路径和实践指导。

> **适用人群**：`AI` 工程师、系统架构师、GPU编程开发者、大模型应用开发者、技术研究人员。
> **技术栈**：`CUDA`、`GPU` 架构、`LLM`、`AI` 系统、分布式计算、容器化部署、性能优化。

## 1. 硬件与基础设施

### 1.1 硬件基础知识

- [PCIe 知识大全](https://mp.weixin.qq.com/s/dHvKYcZoa4rcF90LLyo_0A)
- [NVLink 入门](https://mp.weixin.qq.com/s/fP69UEgusOa_X4ZKLo30ig)
- [NVIDIA DGX SuperPOD：下一代可扩展的AI领导基础设施](https://mp.weixin.qq.com/s/a64Qb6DuAAZnCTBy8g1p2Q)

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

- [**GPU Architecture and Programming — An Introduction**](gpu_programming/gpu_programming_introduction.md) - GPU架构与编程模型的全面介绍

#### 1.2.2 CUDA 核心技术

- [**深入理解 Nvidia CUDA 核心（vs. Tensor Cores vs. RT Cores）**](cuda/cuda_cores_cn.md)

### 1.3 AI 基础设施架构

- [**高性能 GPU 服务器硬件拓扑与集群组网**](https://arthurchiao.art/blog/gpu-advanced-notes-1-zh/)
- [**NVIDIA GH200 芯片、服务器及集群组网**](https://arthurchiao.art/blog/gpu-advanced-notes-4-zh/)
- [**深度学习（大模型）中的精度**](https://mp.weixin.qq.com/s/b08gFicrKNCfrwSlpsecmQ)

### 1.4 AI 基础设施课程

**完整的AI基础设施技术课程体系：**

- [**在线课程演示**](ai_infra_course/index.html) - 交互式课程演示（包含37个页面的完整课程内容）

**课程内容概览：**

- **大模型原理与最新进展**：`Transformer` 架构、训练规模、`DeepSeek` 技术突破、能力涌现现象
- **AI编程技术**：`GitHub Copilot`、`Cursor`、`Trae AI` 等工具对比，实际应用场景和效率数据
- **GPU架构与CUDA编程**：`GPU vs CPU`对比、`NVIDIA` 架构演进、`CUDA` 编程模型、性能优化
- **云原生与AI Infra融合**：推理优化技术、量化技术、`AIBrix` 架构、企业级部署实践
- **技术前沿与职业发展**：行业趋势分析、学习路径规划、职业发展建议

### 1.5 GPU 管理与虚拟化

**理论与架构：**

- [**GPU虚拟化与切分技术原理解析**](gpu_manager/GPU虚拟化与切分技术原理解析.md) - 技术原理深入
- [**GPU 管理相关技术深度解析 - 虚拟化、切分及远程调用**](gpu_manager/GPU%20管理相关技术深度解析%20-%20虚拟化、切分及远程调用.md) - 全面的GPU管理技术指南
- [**第一部分：基础理论篇**](gpu_manager/第一部分：基础理论篇.md) - GPU管理基础概念与理论
- [**第二部分：虚拟化技术篇**](gpu_manager/第二部分：虚拟化技术篇.md) - 硬件、内核、用户态虚拟化技术
- [**第三部分：资源管理与优化篇**](gpu_manager/第三部分：资源管理与优化篇.md) - GPU切分与资源调度算法
- [**第四部分：实践应用篇**](gpu_manager/第四部分：实践应用篇.md) - 部署、运维、性能调优实践

**GPU 虚拟化解决方案：**

- [**HAMi GPU 资源管理完整指南**](hami/hmai-gpu-resources-guide.md)

**运维工具与实践：**

- [**nvidia-smi 入门**](ops/nvidia-smi.md)
- [**nvtop 入门**](ops/nvtop.md)
- [**Nvidia GPU XID 故障码解析**](https://mp.weixin.qq.com/s/ekCnhr3qrhjuX_-CEyx65g)
- [**Nvidia GPU 卡 之 ECC 功能**](https://mp.weixin.qq.com/s/nmZVOQAyfFyesm79HzjUlQ)
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

- [**InfiniBand 网络理论与实践**](InfiniBand/IB%20网络理论与实践.md) - InfiniBand网络架构、协议栈和性能优化
- [**InfiniBand 健康检查工具**](InfiniBand/health/README.md) - 网络健康状态监控和故障诊断
- [**InfiniBand 带宽监控**](InfiniBand/monitor/README.md) - 实时带宽监控和性能分析

#### 1.8.2 NCCL 分布式通信

- [**NCCL 分布式通信测试套件使用指南**](nccl/tutorial.md) - 单节点、多节点 `NCCL` 环境配置和调优
- [**NCCL Kubernetes 部署**](nccl/k8s/README.md) - 容器化NCCL集群部署方案

### 1.9 云原生 AI 基础设施

- [**云原生高性能分布式 LLM 推理框架 llm-d 介绍**](k8s/llm-d-intro.md) - 基于Kubernetes的大模型推理框架
- [**vLLM + LWS：Kubernetes 上的多机多卡推理方案**](k8s/lws_intro.md) - `LWS` 旨在提供一种 **更符合 AI 原生工作负载特点的分布式控制器语义**，填补现有原语在推理部署上的能力空白

### 1.10 性能分析与调优

- [**使用 Nsight Compute Tool 分析 CUDA 矩阵乘法程序**](https://www.yuque.com/u41800946/nquqpa/eo7gykiyhg8xi2gg)
- [**CUDA Kernel Profiling using Nvidia Nsight Compute**](profiling/s9345-cuda-kernel-profiling-using-nvidia-nsight-compute.pdf)

### 1.11 GPU 监控与运维工具

**GPU 性能监控：**

- [**GPU 利用率是一个误导性指标**](ops/GPU%20利用率是一个误导性指标.md) - 深入理解GPU利用率指标的局限性
- [**nvidia-smi 详解**](ops/nvidia-smi.md) - NVIDIA系统管理接口工具使用指南
- [**nvtop 使用指南**](ops/nvtop.md) - 交互式GPU监控工具
- [**DeviceQuery 工具**](ops/DeviceQuery.md) - CUDA设备查询工具详解

---

## 2. 开发与编程

本部分专注于AI开发相关的编程技术、工具和实践，涵盖从基础编程到高性能计算的完整技术栈。

### 2.1 AI 编程入门

- [**AI 编程入门完整教程**](coding/AI%20编程入门.md) - 从零开始的AI编程学习路径
- [**AI 编程入门在线版本**](coding/index.html) - 交互式在线学习体验

### 2.2 CUDA 编程与开发

- [**CUDA 核心概念详解**](cuda/cuda_cores_cn.md) - CUDA核心、线程块、网格等基础概念
- [**CUDA 流详解**](cuda/cuda_streams.md) - CUDA流的原理和应用
- [**GPU 编程基础**](gpu_programming/gpu_programming_introduction.md) - GPU编程入门和进阶

### 2.3 Trae 编程实战课程

**系统化的 Trae 编程学习体系：**

- [《Trae 编程实战》课程提纲](trae/《Trae%20编程实战》课程提纲（对外）.md) - 完整的五部分21章课程规划
  - **基础入门**：环境配置、交互模式、HelloWorld项目实战
  - **场景实战**：前端开发、后端API、数据库设计、安全认证
  - **高级应用**：AI集成、实时通信、数据分析、微服务架构
  - **团队协作**：代码质量、版本控制、CI/CD、性能优化
  - **综合项目**：企业级应用开发、部署运维实战

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

### 3.1 数学基础

- [**线性代数的本质**](https://www.bilibili.com/video/BV1ys411472E) - 3Blue1Brown可视化教程
- [**MIT 18.06 线性代数**](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) - Gilbert Strang经典课程
- [**概率论与统计学基础**](https://github.com/ForceInjection/hands-on-ML) - 贝叶斯定理、概率分布、最大似然估计

### 3.2 监督学习

#### 3.2.1 基础算法

- [**KNN算法**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/KNN/KNN.md) - K近邻算法理论与实现
- [**线性回归**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/LinearRegression/LinearRegression.md) - 最小二乘法、正则化
- [**逻辑回归**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/LogisticRegression/LogisticRegression.md) - 分类算法基础
- [**决策树**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/DecisionTree/DecisionTree.md) - ID3、C4.5、CART算法
- [**支持向量机**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/SVM/SVM.md) - 核技巧与软间隔
- [**朴素贝叶斯**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/NaiveBayes/NaiveBayes.md) - 概率分类器

#### 3.2.2 集成学习

- [**随机森林**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/RandomForest/RandomForest.md) - Bagging集成方法
- [**AdaBoost**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/Adaboost/Adaboost.md) - Boosting算法
- [**GBDT**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/GBDT/GBDT.md) - 梯度提升决策树
- [**Stacking**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/Stacking/Stacking.md) - 模型堆叠技术

### 3.3 无监督学习

#### 3.3.1 聚类算法

- [**K-means聚类**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/Kmeans/Kmeans.md) - 基础聚类算法
- [**层次聚类**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/HierarchicalClustering/HierarchicalClustering.md) - 凝聚与分裂聚类
- [**DBSCAN**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/DBSCAN/DBSCAN.md) - 密度聚类算法

#### 3.3.2 降维算法

- [**PCA主成分分析**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/PCA/PCA.md) - 线性降维方法
- [**LDA线性判别分析**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/LDA/LDA.md) - 监督降维技术

#### 3.3.3 概率模型

- [**EM算法**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/EM/EM.md) - 期望最大化算法
- [**高斯混合模型**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/GMM/GMM.md) - GMM聚类方法

### 3.4 特征工程与模型优化

#### 3.4.1 特征工程

- [**特征工程概述**](https://github.com/ForceInjection/hands-on-ML/blob/main/feature-engineering/README.md) - 数据预处理、特征选择与变换
- [**特征选择方法**](https://github.com/ForceInjection/hands-on-ML/blob/main/feature-engineering/FeatureSelection/FeatureSelection.md) - 过滤法、包装法、嵌入法
- [**GBDT特征提取**](https://github.com/ForceInjection/hands-on-ML/blob/main/feature-engineering/GBDT-FeatureExtraction/GBDT-FeatureExtraction.md) - 基于树模型的特征工程

#### 3.4.2 模型评估

- [**模型评估方法**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/ModelEvaluation/ModelEvaluation.md) - 评估指标与交叉验证
- [**ROC & AUC**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/ROC-AUC/ROC-AUC.md) - 分类模型性能评估
- [**GridSearchCV**](https://github.com/ForceInjection/hands-on-ML/blob/main/let-us-machine-learning/GridSearchCV/GridSearchCV.md) - 超参数优化实践

### 3.5 推荐系统与概率图模型

- [**推荐系统入门**](https://github.com/ForceInjection/hands-on-ML/blob/main/recommendation-system/README.md) - 协同过滤、内容推荐、矩阵分解
- [**贝叶斯网络**](https://github.com/ForceInjection/hands-on-ML/blob/main/probabilistic-graphical-models/BayesianNetwork/BayesianNetwork.md) - 概率图模型基础
- [**隐马尔可夫模型**](https://github.com/ForceInjection/hands-on-ML/blob/main/probabilistic-graphical-models/HMM/HMM.md) - 序列建模与状态推断

### 3.6 深度学习基础

- [**神经网络基础**](https://github.com/ForceInjection/hands-on-ML/blob/main/deep-learning/NeuralNetwork/NeuralNetwork.md) - 感知机、多层感知机、反向传播
- [**深度学习概述**](https://github.com/ForceInjection/hands-on-ML/blob/main/deep-learning/README.md) - 深度学习理论与实践指南

### 3.7 实战项目

- [**泰坦尼克号幸存者预测**](https://github.com/ForceInjection/hands-on-ML/tree/main/titanic) - 特征工程与分类实战
- [**经典数据集项目**](https://github.com/ForceInjection/hands-on-ML) - 鸢尾花分类、波士顿房价预测
- [**Kaggle竞赛案例**](https://github.com/ForceInjection/hands-on-ML) - 房价预测、用户行为分析

### 3.8 学习资源

#### 3.8.1 核心教材

- **《统计学习方法》** - 李航著，算法理论基础
- **《机器学习》** - 周志华著，西瓜书经典
- **《模式识别与机器学习》** - Bishop著，数学严谨

#### 3.8.2 在线资源

- [**动手学机器学习**](https://github.com/ForceInjection/hands-on-ML) - 完整学习路径与代码实现
- [**上海交大机器学习习题**](https://github.com/ForceInjection/hands-on-ML/tree/main/sjtu-hands-on-ML) - 配套习题解答
- [**Andrew Ng机器学习课程**](https://www.coursera.org/learn/machine-learning) - Coursera经典课程
- [**CS229机器学习**](http://cs229.stanford.edu/) - 斯坦福大学课程

#### 3.8.3 实践平台

- [**Kaggle**](https://www.kaggle.com/) - 数据科学竞赛平台
- [**Google Colab**](https://colab.research.google.com/) - 免费GPU环境
- [**scikit-learn**](https://scikit-learn.org/) - Python机器学习库

---

## 4. 大语言模型基础

### 4.1 核心概念

- [**Andrej Karpathy：Deep Dive into LLMs like ChatGPT（B站视频）**](https://www.bilibili.com/video/BV16cNEeXEer)
- [**大模型基础组件 - Tokenizer**](https://zhuanlan.zhihu.com/p/651430181)
- [**解密大语言模型中的 Tokens**](llm/token/llm_token_intro.md)
  - [**Tiktokenizer 在线版**](https://tiktokenizer.vercel.app/?model=gpt-4o)
- [**文本嵌入（Text-Embedding） 技术快速入门**](llm/embedding/text_embeddings_guide.md)
- [**LLM 嵌入技术详解：图文指南**](llm/embedding/LLM%20Embeddings%20Explained%20-%20A%20Visual%20and%20Intuitive%20Guide.zh-CN.md)
- [**大模型 Embedding 层与独立 Embedding 模型：区别与联系**](llm/embedding/embedding.md)
- [**大模型可视化指南**](https://www.maartengrootendorst.com/)
- [**一文读懂思维链（Chain-of-Thought, CoT）**](llm/一文读懂思维链（Chain-of-Thought,%20CoT）.md)
- [**大模型的幻觉及其应对措施**](llm/大模型的幻觉及其应对措施.md)
- [**大模型文件格式完整指南**](llm/大模型文件格式完整指南.md)
- [**混合专家系统（MoE）图解指南**](llm/A%20Visual%20Guide%20to%20Mixture%20of%20Experts%20(MoE).zh-CN.md)
- [**量化技术可视化指南**](llm/A%20Visual%20Guide%20to%20Quantization.zh-CN.md)
- [**基于大型语言模型的意图检测**](llm/Intent%20Detection%20using%20LLM.zh-CN.md)

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

### 5.1 微调技术

- [**大模型训练与推理**](llm/大模型训练与推理.md)
- [**Qwen 2 大模型指令微调入门实战**](llm/Qwen%202%20大模型指令微调入门实战.md)
- [**一文入门垂域模型SFT微调**](llm/一文入门垂域模型SFT微调.md)

### 5.2 从零开始训练

- [**从零开始训练大模型**](llm/从零开始训练大模型.md)
- [**Training a 70B model from scratch: open-source tools, evaluation datasets, and learnings**](https://imbue.com/research/70b-intro/)
- [**Sanitized open-source datasets for natural language and code understanding: how we evaluated our 70B model**](https://imbue.com/research/70b-evals/)
- [**From bare metal to a 70B model: infrastructure set-up and scripts**](https://imbue.com/research/70b-infrastructure/)
- [**Open-sourcing CARBS: how we used our hyperparameter optimizer to scale up to a 70B-parameter language model**](https://imbue.com/research/70b-carbs/)

---

## 6. 大模型推理

### 6.1 推理系统架构

- [**模型推理**](llm/模型推理.md)
- [**Mooncake 架构详解**](llm/Mooncake%20架构详解.md)
- [**vLLM + LWS**](llm/vLLM%20+%20LWS.md)

### 6.2 模型部署实践

- [**动手部署 ollama**](llm/动手部署%20ollama.md)
- [**DeepSeek 模型部署与对比**](llm/DeepSeek%20模型部署与对比.md)

### 6.3 推理优化技术

完整的AI推理优化技术文档系列，涵盖从小型到大型集群的推理优化策略：

- [**AI推理优化技术文档导航**](inference/README.md)
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

### 7.1 上下文工程

**理论基础与实践应用：**

- [**上下文工程原理**](context/上下文工程原理.md) - 基于中科院权威论文的系统性理论阐述
  - **核心定义**：从提示工程到上下文工程的范式转变
  - **理论框架**：信息检索、选择、组装、压缩和动态调整
  - **技术架构**：多模态融合、状态管理、智能组装
  - **企业应用**：全生命周期管理和系统化自动优化

- [**上下文工程原理简介**](context/上下文工程原理简介.md) - 深入浅出的入门指南
  - **生活化类比**：从聊天机器人到智能助手的进化
  - **核心特征**：系统性方法、动态优化、多模态融合、状态管理、智能组装
  - **技术对比**：与传统提示词工程的区别与联系

- [**基于上下文工程的 LangChain 智能体应用**](context/langchain_with_context_engineering.md) - LangChain 实践指南
  - **框架构建**：行为准则、信息接入、会话记忆、工具集成、用户画像
  - **技术实现**：LangChain 与 LangGraph 的上下文工程实践
  - **问题解决**：上下文污染、干扰、混淆、冲突的处理策略
  - **性能优化**：令牌消耗控制、成本与延迟优化

### 7.2 RAG 技术

- [**RAG 技术概述**](llm/rag/README.md)
- [**从0到1快速搭建RAG应用**](llm/rag/写作%20Agentic%20Agent.md)
  - [**配套代码**](llm/rag/lession2.ipynb)
- [**Evaluating Chunking Strategies for Retrieval 总结**](llm/rag/Evaluating%20Chunking%20Strategies%20for%20Retrieval%20总结.md)
- [**中文RAG系统Embedding模型选型技术文档**](llm/rag/中文RAG系统Embedding模型选型技术文档.md)

### 7.3 AI Agent 框架与工具

**Python 生态：**

- [**LangChain + 模型上下文协议（MCP）：AI 智能体 Demo**](llm/agent/README.md)
- [**AI Agents for Beginners 课程之 AI Agent及使用场景简介**](llm/AI%20Agents%20for%20Beginners%20课程之%20AI%20Agent及使用场景简介.md)
- [**A Deep Dive Into MCP and the Future of AI Tooling**](llm/mcp/A_Deep_Dive_Into_MCP_and_the_Future_of_AI_Tooling_zh_CN.md)
- [**LangGraph 实战：用 Python 打造有状态智能体**](llm/langgraph/langgraph_intro.md)
- [**使用 n8n 构建多智能体系统的实践指南**](llm/n8n_multi_agent_guide.md)
- [**开源大模型应用编排平台：Dify、AnythingLLM、Ragflow 与 n8n 的功能与商用许可对比分析**](llm/开源大模型应用编排平台：Dify、AnythingLLM、Ragflow%20与%20n8n%20的功能与商用许可对比分析.md)

**Java 生态：**

- [**使用 Spring AI 构建高效 LLM 代理**](java_ai/spring_ai_cn.md) - Spring AI 代理模式实现指南
  - **代理系统架构**：工作流 vs 代理的设计理念对比
  - **五种基本模式**：链式工作流、路由工作流、并行化、编排、评估
  - **企业级实践**：可预测性、一致性、可维护性的平衡

### 7.4 企业级多智能体系统

**核心理论文档：**

- [**多智能体AI系统基础：理论与框架**](agent/Part1-Multi-Agent-AI-Fundamentals.md) - 多智能体系统的核心理论、技术框架和LangGraph/LangSmith深度解析
- [**企业级多智能体系统实现指南**](agent/Part2-Enterprise-Multi-Agent-System-Implementation.md) - 完整的企业级多智能体系统架构设计与实现

**企业级培训材料：**

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
  - **技术实现**：Spring AI 的模型可移植性和结构化输出功能

### 7.5 模型上下文协议（MCP）

- [**模型上下文协议（MCP）**](llm/mcp/模型上下文协议（MCP）.md)
- [**MCP 技术指南**](llm/mcp/MCP%20技术指南.md)
- [**MCP 深度解析与 AI 工具未来**](llm/mcp/MCP%20深度解析与%20AI%20工具未来.md)

### 7.6 AI 智能体记忆系统

- [**AI 智能体记忆系统：理论与实践**](memory/AI%20智能体记忆系统：理论与实践.md)
- [**如何设计支持多轮指代消解的对话系统**](memory/如何设计支持多轮指代消解的对话系统.md)
- [**记忆系统代码实现**](memory/code/README.md)

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

- [**读者来信：请问7b阅读分析不同中医古籍的能力怎么样？可以进行专项训练大幅度提高这方面能力么？**](llm/scenario/traditional-chinese-medicine.md)
- [**中国大陆合同审核要点清单**](llm/scenario/中国大陆合同审核要点清单.md)
- [**让用户"说半句"话也能懂：ChatBox 的意图识别与语义理解机制解析**](llm/让用户"说半句"话也能懂：ChatBox%20的意图识别与语义理解机制解析.md)

---

## 9. 工具与资源

### 9.1 AI 系统学习资源

[**AISystem**](AISystem/README.md) - 完整的AI系统学习资源，包含：

- [**系统介绍**](AISystem/01Introduction/README.md) - AI系统概述与发展历程
- [**硬件基础**](AISystem/02Hardware/README.md) - AI芯片与硬件架构
- [**编译器技术**](AISystem/03Compiler/README.md) - AI编译器原理与实践
- [**推理优化**](AISystem/04Inference/README.md) - 模型推理加速技术
- [**框架设计**](AISystem/05Framework/README.md) - AI框架架构与并行计算

### 9.2 AI 基础设施专业课程

- [**大模型原理与最新进展**](ai_infra_course/index.html)
- [**AI Infra 课程演讲稿**](ai_infra_course/讲稿.md) - 完整的课程演讲内容和技术要点
- **学习目标**：深入理解大模型工作原理和最新技术进展
- **核心内容**：
  - **Transformer架构**：编码器-解码器结构、注意力机制、文本生成过程
  - **训练规模数据**：GPT-3/4、PaLM等主流模型的参数量、成本和资源需求
  - **DeepSeek模型演进**：V1/V2/R1三代技术突破、MLA架构创新、MoE优化
  - **能力涌现现象**：规模效应、临界点突破、多模态发展趋势
  - **AI编程工具**：GitHub Copilot、Cursor、Trae AI等工具对比和应用实践
  - **GPU架构与CUDA**：硬件基础、并行计算原理、性能优化策略
  - **云原生AI架构**：现代化AI基础设施设计与部署实践

### 9.3 开源项目推荐

- [**DeepSeek**](https://github.com/DeepSeek-AI/DeepSeek): 一个基于Transformer的中文大模型，由上海人工智能实验室开发。
- [**unstructured**](https://github.com/Unstructured-IO/unstructured): Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.
- [**MinerU**](https://github.com/opendatalab/MinerU): A high-quality tool for convert PDF to Markdown and JSON.
- [**markitdown**](https://github.com/microsoft/markitdown): Python tool for converting files and office documents to Markdown.
- [**unsloth**](https://github.com/unslothai/unsloth): Finetune Llama 3.3, DeepSeek-R1 & Reasoning LLMs 2x faster with 70% less memory!
- [**ktransformers**](https://github.com/kvcache-ai/ktransformers): A Flexible Framework for Experiencing Cutting-edge LLM Inference Optimizations

---

## Buy Me a Coffee

如果您觉得本项目对您有帮助，欢迎购买我一杯咖啡，支持我继续创作和维护。

|**微信**|**支付宝**|
|---|---|
|<img src="./img/weixinpay.JPG" alt="wechat" width="200">|<img src="./img/alipay.JPG" alt="alipay" width="200">|
