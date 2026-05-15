# RAG 基础

把 RAG 从「给 LLM 拼个向量库」做到「线上稳定跑」，真正决定效果的是三件事：**切得好不好**（chunking 策略）、**向量选得准不准**（embedding 模型）、**编排够不够灵活**（从 Naive RAG 到 Agentic RAG 的架构演进）。本目录围绕这三条主线组织文档，既包含理论与评测方法，也包含可上手的 notebook。

## 1. 核心文档

### 1.1 架构演进与对比

- **[RAG 策略对比](rag_comparison.md)** — 系统梳理多类 Agentic RAG 架构（单智能体路由器、多智能体系统、分层 RAG、Corrective RAG、Adaptive RAG、GraphRAG、Agent-G、GeAR、Agentic Document Workflow 等），分别给出核心思想、优势、挑战与典型应用场景，用于在项目启动阶段做架构选型。

### 1.2 分块策略评估

- **[Chunking 策略评估总结](evaluating_chunking_strategies_summary.md)** — 基于 Chroma 研究团队的 *Evaluating Chunking Strategies for Retrieval* 报告，提出 **token 级精确率 / 召回率 / IoU** 三指标取代传统 nDCG@K / MAP@K，用于量化不同 chunker（固定长度、递归、语义、HyDE、聚类）在真实检索中的 token 冗余与召回完整性。

### 1.3 中文 Embedding 模型选型

- **[中文 RAG Embedding 模型选型](chinese_rag_embedding_model_selection.md)** — 面向中文业务的完整选型文档：覆盖 BGE、GTE、M3E、Conan、Piccolo 等主流模型的 MTEB-zh 分数、维度与显存、推理吞吐、开源许可；结合检索准确率、部署成本、维护便利性给出分场景推荐。

## 2. 动手实验

- **[rag_lesson2.ipynb](notebooks/rag_lesson2.ipynb)** — 配套 notebook，覆盖文档加载、chunking、embedding、向量库写入与检索问答的端到端流程，可直接在本地 Jupyter 跑通。

## 3. 相关资源

- [GraphRAG 与知识图谱](../graph_rag/README.md) — 当 Naive RAG 在多跳推理上失效时的升级路径。
- [文档智能解析工具](../pdf_tools/README.md) — 从 PDF / Office 提取高质量 RAG 输入。
- [LLM 理论与基础 · Embedding](../../06_llm_theory_and_fundamentals/llm_basic_concepts/embedding/README.md) — Embedding 技术的理论背景。
