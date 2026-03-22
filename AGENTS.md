# 项目概览与工具使用说明

AI Fundamentals 是一个全面的人工智能基础设施（AI Infrastructure）学习资源集合，涵盖从底层硬件架构到上层高级应用的完整技术栈。内容体系包括 GPU 架构与编程、CUDA 开发、大语言模型（LLM）、AI 系统设计、性能调优以及企业级云原生部署等核心领域。该项目致力于为 AI 工程师、系统架构师以及技术研究人员提供系统化的学习路径与高价值的实践指导。所有内容来源于项目根目录的官方介绍文档。

---

## 1 核心技术模块

项目内容主要分为以下几个核心模块：

- **`01_hardware_architecture`（硬件架构）**：涵盖从单机计算芯片（GPU、TPU）到大规模集群互联的技术细节，包括 GPUDirect、PCIe 总线以及 NVLink 等互连技术。
- **`02_dpu_programming`（DPU 编程）**：包含 DOCA 编程指南等 DPU 相关的基础开发与实践内容。
- **`02_gpu_programming`（GPU 编程基础）**：提供完整的开发知识体系，包含环境构建、CUDA 核心编程范式以及性能分析与调优指南。
- **`03_ai_cluster_ops`（AI 集群运维与通信）**：包含 GPU 基础监控运维、InfiniBand 高性能网络以及 NCCL 分布式通信实战。
- **`04_cloud_native_ai_platform`（云原生 AI 基础设施）**：聚焦 Kubernetes 生态在 AI 场景下的应用，探讨 GPU 资源池化（如 HAMi）、弹性调度与分布式推理/存储系统设计。
- **`05_model_training_and_fine_tuning`（模型训练与微调）**：提供 AI 运维设计思路及垂域大模型 SFT 微调实践案例。
- **`06_llm_theory_and_fundamentals`（大语言模型理论与基础）**：深度剖析量化、混合专家模型（MoE）、Embedding 技术，以及深度研究（Deep Research）等前沿 Agent 框架设计。
- **`07_rag_and_tools`（RAG 与工具）**：包含知识图谱（KG）、GraphRAG 的实践指南、PDF 解析工具（如 MinerU、Marker）介绍，以及 RAG 系统的分块策略与 Embedding 选型分析。
- **`08_agentic_system`（智能体系统）**：探讨 Agent 基础设施、多智能体系统（Multi-Agent System）架构、记忆管理机制（MemoryOS、MemMachine）、MCP（Model Context Protocol）协议及上下文工程（Context Engineering）。
- **`09_inference_system`（推理系统与优化）**：专注于大模型推理服务优化，涵盖 KV Cache 核心技术（如 LMCache、KVBM）、vLLM 高级应用、内存计算分析及企业级推理解决方案设计。
- **`10_ai_related_course`（AI 相关课程）**：整理了 AI Infra、AI 编程实战（如 Trae 进阶教程）、多智能体培训等系统性课程的课件、讲稿与实战案例。
- **`98_llm_programming`（大模型编程）**：汇总了面向开发者的 LLM 编程实践指南，包括 LangGraph 框架应用、Java 生态（Spring AI）集成以及 Harness Engineering 探索。

---

## 2. Markdown 链接检查工具

[check_links.py](./tools/check_links.py) 是一个基于 Python 开发的自动化工具，专门用于验证项目内 Markdown 文档中本地及外部超链接的有效性，支持处理包含空格、URL 编码等复杂场景。详细的参数说明与使用示例，请参阅 [工具使用指南](./tools/README.md)。
