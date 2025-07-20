# AI Fundamentals

本仓库是一个全面的人工智能基础设施（AI Infrastructure）学习资源集合，涵盖从硬件基础到高级应用的完整技术栈。内容包括GPU架构与编程、CUDA开发、大语言模型、AI系统设计、性能优化、企业级部署等核心领域，旨在为AI工程师、研究人员和技术爱好者提供系统性的学习路径和实践指导。

> **适用人群**：AI工程师、系统架构师、GPU编程开发者、大模型应用开发者、技术研究人员  
> **技术栈**：CUDA、GPU架构、LLM、AI系统、分布式计算、容器化部署、性能优化

## 第一部分：硬件与基础设施

### 1. 硬件基础知识

- [PCIe 知识大全](https://mp.weixin.qq.com/s/dHvKYcZoa4rcF90LLyo_0A)
- [NVLink 入门](https://mp.weixin.qq.com/s/fP69UEgusOa_X4ZKLo30ig)
- [NVIDIA DGX SuperPOD：下一代可扩展的AI领导基础设施](https://mp.weixin.qq.com/s/a64Qb6DuAAZnCTBy8g1p2Q)

### 2. GPU 架构深度解析

在准备在 GPU 上运行的应用程序时，了解 GPU 硬件设计的主要特性并了解与 CPU 的相似之处和不同之处会很有帮助。本路线图适用于那些对 GPU 比较陌生或只是想了解更多有关 GPU 中计算机技术的人。不需要特定的并行编程经验，练习基于 CUDA 工具包中包含的标准 NVIDIA 示例程序。

- [GPU 特性](gpu_architecture/gpu_characteristics.md)
- [GPU 内存](gpu_architecture/gpu_memory.md)
- [GPU Example: Tesla V100](gpu_architecture/tesla_v100.md)
- [GPUs on Frontera: RTX 5000](gpu_architecture/rtx_5000.md)
- 练习：
  - [Exercise: Device Query](gpu_architecture/exer_device_query.md)
  - [Exercise: Device Bandwidth](gpu_architecture/exer_device_bandwidth.md)

#### 2.1 GPU 架构和编程模型介绍

- [GPU Architecture and Programming — An Introduction](gpu_programming/gpu_programming_introduction.md)

#### 2.2 CUDA 核心技术

- [深入理解 Nvidia CUDA 核心（vs. Tensor Cores vs. RT Cores)](cuda/cuda_cores_cn.md)

### 3. AI 基础设施架构

- [高性能 GPU 服务器硬件拓扑与集群组网](https://arthurchiao.art/blog/gpu-advanced-notes-1-zh/)
- [NVIDIA GH200 芯片、服务器及集群组网](https://arthurchiao.art/blog/gpu-advanced-notes-4-zh/)
- [深度学习（大模型）中的精度](https://mp.weixin.qq.com/s/b08gFicrKNCfrwSlpsecmQ)

### 4. GPU 管理与虚拟化

**原理文档：**

- [GPU虚拟化与切分技术原理解析](gpu_manager/GPU虚拟化与切分技术原理解析.md) - 技术原理深入

**详解文档：**

- [第一部分：基础理论篇](gpu_manager/第一部分：基础理论篇.md) - GPU管理基础概念与理论
- [第二部分：虚拟化技术篇](gpu_manager/第二部分：虚拟化技术篇.md) - 硬件、内核、用户态虚拟化技术
- [第三部分：资源管理与优化篇](gpu_manager/第三部分：资源管理与优化篇.md) - GPU切分与资源调度算法
- [第四部分：实践应用篇](gpu_manager/第四部分：实践应用篇.md) - 部署、运维、性能调优实践

### 5. GPU 资源管理工具

- [HAMi GPU 资源管理指南](hami/hmai-gpu-resources-guide.md) - 基于 HAMi 的 GPU 资源管理与调度

### 6. 分布式存储系统

- [JuiceFS 文件修改机制分析](juicefs/JuiceFS%20文件修改机制分析.md) - 分布式文件系统的修改机制深度解析

### 7. AI 智能体记忆系统

**理论与实践：**

- [AI 智能体记忆系统：理论与实践](memory/AI%20智能体记忆系统：理论与实践.md) - 记忆系统的设计原理与实现
- [如何设计支持多轮指代消解的对话系统](memory/如何设计支持多轮指代消解的对话系统.md) - 对话系统中的指代消解技术

**代码实现：**

- [记忆系统代码实现](memory/code/) - 包含记忆管理、向量存储、对话处理等核心组件

### 8. Trae 编程实战课程

**完整的 Trae 编程学习体系：**

- [《Trae 编程实战》课程提纲](trae/《Trae%20编程实战》课程提纲.md) - 完整课程规划

**第一部分：基础入门篇：**

- [第一章-Trae简介与环境搭建](trae/第一部分-基础入门篇/第一章-Trae简介与环境搭建.md)
- [第二章-Trae核心概念](trae/第一部分-基础入门篇/第二章-Trae核心概念.md)
- [第三章-基础语法与数据类型](trae/第一部分-基础入门篇/第三章-基础语法与数据类型.md)
- [第四章-控制流与函数](trae/第一部分-基础入门篇/第四章-控制流与函数.md)

**第二部分：进阶特性篇：**

- [第五章-面向对象编程](trae/第二部分-进阶特性篇/第五章-面向对象编程.md)
- [第六章-模块与包管理](trae/第二部分-进阶特性篇/第六章-模块与包管理.md)
- [第七章-异常处理与调试](trae/第二部分-进阶特性篇/第七章-异常处理与调试.md)
- [第八章-并发与异步编程](trae/第二部分-进阶特性篇/第八章-并发与异步编程.md)

**第三部分：高级应用篇：**

- [第九章-数据库操作](trae/第三部分-高级应用篇/第九章-数据库操作.md)
- [第十章-网络编程](trae/第三部分-高级应用篇/第十章-网络编程.md)
- [第十一章-Web开发](trae/第三部分-高级应用篇/第十一章-Web开发.md)
- [第十二章-API设计与开发](trae/第三部分-高级应用篇/第十二章-API设计与开发.md)

**第四部分：工程实践篇：**

- [第十三章-测试驱动开发](trae/第四部分-工程实践篇/第十三章-测试驱动开发.md)
- [第十四章-代码质量管理](trae/第四部分-工程实践篇/第十四章-代码质量管理.md)
- [第十五章-性能优化](trae/第四部分-工程实践篇/第十五章-性能优化.md)
- [第十六章-安全编程](trae/第四部分-工程实践篇/第十六章-安全编程.md)

**第五部分：综合项目实战：**

- [第十七章-项目架构设计](trae/第五部分-综合项目实战/第十七章-项目架构设计.md)
- [第十八章-微服务架构](trae/第五部分-综合项目实战/第十八章-微服务架构.md)
- [第十九章-容器化与编排](trae/第五部分-综合项目实战/第十九章-容器化与编排.md)
- [第二十章-部署运维](trae/第五部分-综合项目实战/第二十章-部署运维.md)

### 9. 企业级多智能体系统

- [企业级多智能体系统实现指南](agent/Part2-Enterprise-Multi-Agent-System-Implementation.md) - 完整的企业级多智能体系统架构设计与实现

---

## 第二部分：编程与开发

### 1. CUDA 学习材料

#### 1.1 快速入门

- [并行计算、费林分类法和 CUDA 基本概念](https://mp.weixin.qq.com/s/NL_Bz8JB-LdAtrQake7EdA)
- [CUDA 编程模型入门](https://mp.weixin.qq.com/s/IUYzzgt6DUYhfaDnbxoZuQ)
- [CUDA 并发编程之 Stream 介绍](cuda/cuda_streams.md)

#### 1.2 参考资料

- [CUDA Reading Group 相关讲座](https://mp.weixin.qq.com/s/6sOrNzG0UeVBes8stWSoWA): [GPU Mode Reading Group](https://github.com/gpu-mode)
- [《CUDA C++ Programming Guide》](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [《CUDA C 编程权威指南》](https://mp.weixin.qq.com/s/xJY5Znv3cuQi_UCd_XjJ4A)：[书中示例代码](https://github.com/Eddie-Wang1120/Professional-CUDA-C-Programming-Code-and-Notes)
- [Nvidia 官方 CUDA 示例](https://github.com/NVIDIA/cuda-samples)
- [《CUDA 编程：基础与实践 by 樊哲勇》](https://book.douban.com/subject/35252459/)
  - [学习笔记](https://github.com/QINZHAOYU/CudaSteps)
  - [示例代码](https://github.com/MAhaitao999/CUDA_Programming)
- [《CUDA 编程简介: 基础与实践 by 李瑜》](http://www.frankyongtju.cn/ToSeminars/hpc.pdf)
- [《CUDA 编程入门》 - 本文改编自北京大学超算队 CUDA 教程讲义](https://hpcwiki.io/gpu/cuda/)
- [Multi GPU Programming Models](https://github.com/NVIDIA/multi-gpu-programming-models)
- [CUDA Processing Streams](https://turing.une.edu.au/~cosc330/lectures/display_lecture.php?lecture=22#1)

#### 1.3 专业选手

[**CUDA-Learn-Notes**](https://github.com/xlite-dev/CUDA-Learn-Notes)：📚Modern CUDA Learn Notes: 200+ Tensor/CUDA Cores Kernels🎉, HGEMM, FA2 via MMA and CuTe, 98~100% TFLOPS of cuBLAS/FA2.

### 2. 监控与运维

- [nvidia-smi 入门](ops/nvidia-smi.md)
- [nvtop 入门](ops/nvtop.md)
- [Nvidia GPU XID 故障码解析](https://mp.weixin.qq.com/s/ekCnhr3qrhjuX_-CEyx65g)
- [Nvidia GPU 卡 之 ECC 功能](https://mp.weixin.qq.com/s/nmZVOQAyfFyesm79HzjUlQ)
- [查询 GPU 卡详细参数](ops/DeviceQuery.md)
- [Understanding NVIDIA GPU Performance: Utilization vs. Saturation (2023)](https://arthurchiao.art/blog/understanding-gpu-performance/)
- [GPU 利用率是一个误导性指标](ops/GPU%20利用率是一个误导性指标.md)

### 3. 性能分析与调优

- [使用 Nsight Compute Tool 分析 CUDA 矩阵乘法程序](https://www.yuque.com/u41800946/nquqpa/eo7gykiyhg8xi2gg)
- [CUDA Kernel Profiling using Nvidia Nsight Compute](profiling/s9345-cuda-kernel-profiling-using-nvidia-nsight-compute.pdf)

### 4. AI 编程入门

完整的AI编程入门教程，帮助开发者掌握AI编程工具的使用方法和技巧：

- [**AI 编程入门完整教程**](coding/AI%20编程入门.md) - 从大语言模型基础到实际应用的完整指南
- [**在线演示版本**](coding/index.html) - 交互式课程演示（包含26个页面的完整课程内容）

**课程内容概览：**

- **大语言模型基础认知**：理解AI的工作原理和能力边界
- **AI编程革命**：从传统编程到AI辅助编程的转变
- **主流工具对比**：GitHub Copilot、Cursor、Trae AI等工具的特点和使用场景
- **Prompt Engineering**：编写高效编程提示词的技巧和最佳实践
- **实战案例**：代码生成、调试、重构、文档编写等实际应用
- **最佳实践**：安全性、团队协作、质量保证等注意事项

---

## 第三部分：机器学习基础

### 1. 机器学习理论基础

#### 1.1 核心概念与算法

- [《机器学习系统：设计和实现》](https://openmlsys.github.io/index.html)
- [《动手学深度学习》](https://zh.d2l.ai/)
- [大模型时代为什么需要一本深度学习教科书？揭秘《深度学习：基础与概念》的独特价值](https://mp.weixin.qq.com/s/890mBdrIqzo3Of9RefsMxg)

#### 1.2 数学基础

- **线性代数**：矩阵运算、特征值分解、奇异值分解
- **概率论与统计**：贝叶斯定理、概率分布、统计推断
- **微积分**：梯度、偏导数、链式法则
- **优化理论**：梯度下降、随机梯度下降、Adam优化器

### 2. 深度学习框架与实践

#### 2.1 主流框架对比

- **PyTorch**：动态图、研究友好、灵活性强
- **TensorFlow**：静态图、生产部署、生态完善
- **JAX**：函数式编程、自动微分、高性能计算
- **PaddlePaddle**：国产框架、产业应用、易用性强

#### 2.2 模型训练技巧

- **数据预处理**：归一化、数据增强、特征工程
- **模型设计**：网络架构、激活函数、正则化
- **训练策略**：学习率调度、批量大小、早停机制
- **调试技巧**：梯度检查、可视化、性能分析

### 3. 计算机视觉

#### 3.1 基础算法

- **卷积神经网络（CNN）**：LeNet、AlexNet、VGG、ResNet
- **目标检测**：YOLO、R-CNN系列、SSD
- **图像分割**：FCN、U-Net、Mask R-CNN
- **生成模型**：GAN、VAE、扩散模型

#### 3.2 实际应用

- **图像分类**：ImageNet竞赛、迁移学习
- **人脸识别**：特征提取、相似度计算
- **医学影像**：病灶检测、影像分析
- **自动驾驶**：环境感知、路径规划

### 4. 自然语言处理

#### 4.1 传统方法

- **文本预处理**：分词、词性标注、命名实体识别
- **特征表示**：词袋模型、TF-IDF、Word2Vec
- **序列模型**：RNN、LSTM、GRU
- **注意力机制**：Attention、Self-Attention

#### 4.2 现代方法

- **预训练模型**：BERT、GPT、T5
- **多模态模型**：CLIP、DALL-E、GPT-4V
- **对话系统**：检索式、生成式、混合式
- **知识图谱**：实体关系抽取、知识推理

### 5. 强化学习

#### 5.1 基础理论

- **马尔可夫决策过程**：状态、动作、奖励、策略
- **价值函数**：状态价值、动作价值、贝尔曼方程
- **策略梯度**：REINFORCE、Actor-Critic、PPO
- **深度强化学习**：DQN、DDPG、SAC

#### 5.2 应用场景

- **游戏AI**：围棋、电子竞技、策略游戏
- **机器人控制**：运动规划、操作控制
- **推荐系统**：个性化推荐、广告投放
- **金融交易**：算法交易、风险管理

### 6. 机器学习工程

#### 6.1 MLOps实践

- **实验管理**：MLflow、Weights & Biases、TensorBoard
- **版本控制**：DVC、Git LFS、模型版本管理
- **自动化流水线**：Kubeflow、MLflow、Airflow
- **模型监控**：数据漂移、模型性能、A/B测试

#### 6.2 部署与优化

- **模型压缩**：剪枝、量化、知识蒸馏
- **推理优化**：TensorRT、ONNX、OpenVINO
- **边缘计算**：移动端部署、IoT设备
- **云端部署**：容器化、微服务、弹性扩缩

---

## 第四部分：大语言模型

### 1. LLM 基础理论

#### 1.1 核心概念

- [Andrej Karpathy：Deep Dive into LLMs like ChatGPT（B站视频）](https://www.bilibili.com/video/BV16cNEeXEer)
- [大模型基础组件 - Tokenizer](https://zhuanlan.zhihu.com/p/651430181)
- [解密大语言模型中的 Tokens](llm/token/llm_token_intro.md)
  - [Tiktokenizer 在线版](https://tiktokenizer.vercel.app/?model=gpt-4o)
- [文本嵌入（Text-Embedding） 技术快速入门](llm/embedding/text_embeddings_guide.md)
- [LLM 嵌入技术详解：图文指南](llm/embedding/LLM%20Embeddings%20Explained%20-%20A%20Visual%20and%20Intuitive%20Guide.zh-CN.md)
- [大模型 Embedding 层与独立 Embedding 模型：区别与联系](llm/embedding/embedding.md)
- [大模型可视化指南](https://www.maartengrootendorst.com/)
- [一文读懂思维链（Chain-of-Thought, CoT）](llm/一文读懂思维链（Chain-of-Thought,%20CoT）.md)
- [大模型的幻觉及其应对措施](llm/大模型的幻觉及其应对措施.md)
- [大模型文件格式完整指南](llm/大模型文件格式完整指南.md)
- [混合专家系统（MoE）图解指南](llm/A%20Visual%20Guide%20to%20Mixture%20of%20Experts%20(MoE).zh-CN.md)
- [量化技术可视化指南](llm/A%20Visual%20Guide%20to%20Quantization.zh-CN.md)
- [基于大型语言模型的意图检测](llm/Intent%20Detection%20using%20LLM.zh-CN.md)
- [Mooncake 架构详解：以 KV 缓存为中心的高效 LLM 推理系统设计](llm/Mooncake%20架构详解：以%20KV%20缓存为中心的高效%20LLM%20推理系统设计.md)
- [vLLM + LWS：Kubernetes 上的多机多卡推理方案](llm/lws_intro.md)

#### 1.2 参考书籍

- [大模型基础](https://github.com/ZJU-LLMs/Foundations-of-LLMs) <br>
 <img src="https://raw.githubusercontent.com/ZJU-LLMs/Foundations-of-LLMs/main/figure/cover.png" height="300"/>

- [Hands-On Large Language Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) <br>
 <img src="https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/images/book_cover.png" height="300"/>

- [《从零构建大模型》：从理论到实践，手把手教你打造自己的大语言模型](https://mp.weixin.qq.com/s/FkBjsQmeXEPlsdFXETYSng)
- [《百面大模型》打通大模型求职与实战的关键一书](https://mp.weixin.qq.com/s/rBJ5an0pr3TgjFbyJXa0WA)
- [《图解大模型：生成式AI原理与实践》](https://mp.weixin.qq.com/s/tYrHrpMrZySgWKE1ECqTWg)

### 2. LLM 训练

#### 2.1 微调技术

- [**Qwen 2 大模型指令微调入门实战**](https://mp.weixin.qq.com/s/Atf61jocM3FBoGjZ_DZ1UA)
  - [配套代码](llm/fine-tuning/train_qwen2.ipynb)
- [一文入门垂域模型SFT微调](llm/一文入门垂域模型SFT微调.md)

#### 2.2 从零开始训练大模型

- [Training a 70B model from scratch: open-source tools, evaluation datasets, and learnings](https://imbue.com/research/70b-intro/)
- [Sanitized open-source datasets for natural language and code understanding: how we evaluated our 70B model](https://imbue.com/research/70b-evals/)
- [From bare metal to a 70B model: infrastructure set-up and scripts](https://imbue.com/research/70b-infrastructure/)
- [Open-sourcing CARBS: how we used our hyperparameter optimizer to scale up to a 70B-parameter language model](https://imbue.com/research/70b-carbs/)

### 3. LLM 应用开发

#### 3.1 RAG 技术

- [**从0到1快速搭建RAG应用**](https://mp.weixin.qq.com/s/89-bwZ4aPor4ySj5U3n5zw)
  - [配套代码](llm/rag/lession2.ipynb)
- [Evaluating Chunking Strategies for Retrieval 总结](llm/rag/Evaluating%20Chunking%20Strategies%20for%20Retrieval%20总结.md)
- [中文RAG系统Embedding模型选型技术文档](llm/rag/中文RAG系统Embedding模型选型技术文档.md)

#### 3.2 AI Agent 开发

- [**LangChain + 模型上下文协议（MCP）：AI 智能体 Demo**](llm/agent/README.md)
- [AI Agents for Beginners 课程之 AI Agent及使用场景简介](llm/AI%20Agents%20for%20Beginners%20课程之%20AI%20Agent及使用场景简介.md)
- [A Deep Dive Into MCP and the Future of AI Tooling](llm/mcp/A_Deep_Dive_Into_MCP_and_the_Future_of_AI_Tooling_zh_CN.md)
- [LangGraph 实战：用 Python 打造有状态智能体](llm/langgraph/langgraph_intro.md)
- [使用 n8n 构建多智能体系统的实践指南](llm/n8n_multi_agent_guide.md)
- [开源大模型应用编排平台：Dify、AnythingLLM、Ragflow 与 n8n 的功能与商用许可对比分析](llm/开源大模型应用编排平台：Dify、AnythingLLM、Ragflow%20与%20n8n%20的功能与商用许可对比分析.md)

---

## 第五部分：实践案例

### 1. AI 推理优化技术

完整的AI推理优化技术文档系列，涵盖从小型到大型集群的推理优化策略：

- [AI推理优化技术文档导航](inference/README.md)
- [背景与目标](inference/01-背景与目标.md)
- [集群规模分类与特征分析](inference/02-集群规模分类与特征分析.md)
- [核心推理优化技术深度解析](inference/03-核心推理优化技术深度解析.md)
- [不同集群规模的技术选型策略](inference/04-不同集群规模的技术选型策略.md)
- [性能评估指标体系](inference/05-性能评估指标体系.md)
- [推理服务架构设计](inference/06-推理服务架构设计.md)
- [实施建议与最佳实践](inference/07-实施建议与最佳实践.md)
- [参考资料与延伸阅读](inference/08-参考资料与延伸阅读.md)
- [安全性与合规性](inference/09-安全性与合规性.md)
- [多模态推理优化](inference/10-多模态推理优化.md)
- [边缘推理优化](inference/11-边缘推理优化.md)
- [场景问题解答](inference/12-场景问题解答.md)
- [实施检查清单](inference/13-实施检查清单.md)
- [总结与展望](inference/14-总结与展望.md)

### 2. 模型部署与推理

- [动手部署 ollama](llm/ollama/README.md)
- [在 Mac 上运行 DeepSeek-R1 模型](deepseek/mac-deepseek-r1.md)
- [DeepSeek r1 蒸馏模型和满血模型对比](deepseek/deepseek-r1-cmp.md)
- [Deepseek 3FS（ Fire-Flyer File System）设计笔记](deepseek/deepseek_3fs_design_notes.zh-CN.md)

### 3. 文档处理工具

- [深入探索：AI 驱动的 PDF 布局检测引擎源代码解析](llm/marker.zh-CN.md)
- [上海人工智能实验室开源工具 MinerU 助力复杂 PDF 高效解析提取](llm/minerU_intro.md)
- [Markitdown 入门](llm/markitdown/README.md)
- [DeepWiki 使用方法与技术原理深度分析](llm/DeepWiki%20使用方法与技术原理深度分析.md)

### 4. 特定领域应用

- [读者来信：请问7b阅读分析不同中医古籍的能力怎么样？可以进行专项训练大幅度提高这方面能力么？](llm/scenario/traditional-chinese-medicine.md)
- [中国大陆合同审核要点清单](llm/scenario/中国大陆合同审核要点清单.md)
- [让用户"说半句"话也能懂：ChatBox 的意图识别与语义理解机制解析](llm/让用户"说半句"话也能懂：ChatBox%20的意图识别与语义理解机制解析.md)

---

## 第六部分：工具与资源

### 1. AI 系统学习资源

[**AISystem**](AISystem/README.md) - 完整的AI系统学习资源，包含：

- [系统介绍](AISystem/01Introduction/README.md) - AI系统概述与发展历程
- [硬件基础](AISystem/02Hardware/README.md) - AI芯片与硬件架构
- [编译器技术](AISystem/03Compiler/README.md) - AI编译器原理与实践
- [推理优化](AISystem/04Inference/README.md) - 模型推理加速技术
- [框架设计](AISystem/05Framework/README.md) - AI框架架构与并行计算

### 2. Java AI 开发

- [使用 Spring AI 构建高效 LLM 代理](java_ai/spring_ai_cn.md) - 使用 Spring AI 构建高效 LLM 代理

### 4. AI 基础设施专业课程

#### 4.1 大模型原理与最新进展

- [AI Infra 课程 PPT 大纲](ai_infra_course/AI%20Infra%20课程%20PPT%20大纲.md) - 完整的AI基础设施课程设计
- [课程讲稿](ai_infra_course/讲稿.md) - 详细的演讲内容和技术要点
- **学习目标**：深入理解大模型工作原理和最新技术进展
- **核心内容**：
  - **Transformer架构**：编码器-解码器结构、注意力机制、文本生成过程
  - **训练规模数据**：GPT-3/4、PaLM等主流模型的参数量、成本和资源需求
  - **DeepSeek模型演进**：V1/V2/R1三代技术突破、MLA架构创新、MoE优化
  - **能力涌现现象**：规模效应、临界点突破、多模态发展趋势

#### 4.2 AI编程工具与应用

- **学习目标**：掌握AI编程工具的使用和应用场景
- **核心内容**：
  - **主流工具对比**：GitHub Copilot、Cursor、Trae AI的特色和优势
  - **应用场景**：代码生成、调试、重构、文档生成四大核心应用
  - **效率数据**：开发速度提升55%、学习时间减少60%、Bug修复时间减少31%
  - **企业案例**：Microsoft、Shopify等成功案例，ROI达1200%-3000%

#### 4.3 GPU架构与CUDA编程

- **学习目标**：理解GPU架构设计和CUDA编程模型
- **核心内容**：
  - **GPU vs CPU对比**：架构差异、性能对比、适用场景分析
  - **NVIDIA架构演进**：Tesla到Hopper的技术发展历程
  - **CUDA编程模型**：线程层次、内存模型、核函数设计
  - **性能优化**：内存访问优化、并行度调优、瓶颈分析

#### 4.4 云原生与AI Infra融合架构

- **学习目标**：构建现代化的AI基础设施架构
- **核心内容**：
  - **容器化技术**：Docker、Kubernetes在AI场景的应用
  - **资源管理**：GPU集群管理、资源调度、弹性扩缩容
  - **存储系统**：分布式存储、数据湖架构、缓存策略
  - **监控运维**：性能监控、日志管理、故障处理

#### 4.5 课程资源与实践

- **PPT演示文件**：包含37个HTML页面的完整课程演示
- **视频资源**：《流言终结者》GPU与CPU对比演示视频
- **实践指导**：基于真实项目的技术实践和案例分析
- **学习路径**：从基础理论到前沿实践的完整技术栈覆盖

### 5. 开源项目推荐

- [unstructured](https://github.com/Unstructured-IO/unstructured): Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.
- [MinerU](https://github.com/opendatalab/MinerU): A high-quality tool for convert PDF to Markdown and JSON.
- [markitdown](https://github.com/microsoft/markitdown): Python tool for converting files and office documents to Markdown.
- [unsloth](https://github.com/unslothai/unsloth): Finetune Llama 3.3, DeepSeek-R1 & Reasoning LLMs 2x faster with 70% less memory!
- [ktransformers](https://github.com/kvcache-ai/ktransformers): A Flexible Framework for Experiencing Cutting-edge LLM Inference Optimizations

---
