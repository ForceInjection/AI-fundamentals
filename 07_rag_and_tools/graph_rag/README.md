# GraphRAG：图驱动的检索增强生成

Naive RAG 依赖向量相似度，在「跨文档多跳推理」「数值/时间约束」「领域 Schema 对齐」这些场景很容易失效——向量能找到语义相关的片段，但拼不出一条逻辑链。**GraphRAG** 的思路是：先把文档变成结构化的知识图谱，再让 LLM 沿图走；**KAG** 则在此基础上更进一步，用逻辑符号引导求解。本目录收录这两条技术路径的深度文档。

## 1. 核心文档

### 1.1 GraphRAG 基础：面向 RAG 的知识图谱

- **[GraphRAG 学习指南](graph_rag_learning_guide.md)** — 基于 DeepLearning.AI × Neo4j 的 *Knowledge Graphs for RAG* 课程重组，围绕 SEC 10-K 年报问答系统，讲解 KG 概念、Cypher 查询、Text2Graph 抽取、向量检索与图检索的混合（Hybrid Retrieval）架构。适合首次接触 GraphRAG 的工程师建立完整心智模型。

### 1.2 KAG 框架：逻辑符号引导的混合推理

- **[KAG 框架介绍](kag_introduction.md)** — 解读 OpenSPG 团队提出的 **Knowledge Augmented Generation (KAG)** 框架：
  - **LLMFriSPG 知识表示**：借鉴 DIKW 金字塔，分层建模数据 / 信息 / 知识。
  - **互索引（Mutual Indexing）**：图结构与原始文本块双向索引，为图节点附带可追溯的文本上下文。
  - **逻辑形式（Logical Form）引导求解**：将自然语言问题分解为 `Retrieval` / `Math` / `Sort` / `Deduce` 等可执行算子，支持多轮反射。
  - 解决的痛点：传统 RAG 对数值、时间关系、专家规则不敏感；GraphRAG OpenIE 噪声大、难对齐领域 Schema。

## 2. 与其他目录的关系

- **图数据库底座**见 [`../knowledge_graph/`](../knowledge_graph/README.md)：Neo4j 的部署与 Cypher 查询语言。GraphRAG / KAG 的实战都依赖这层。
- **落地案例**见 [`../synergized_llms_kgs/`](../synergized_llms_kgs/README.md)：银行反电诈场景中 LLM + KG 的完整系统设计 + 可运行 demo。
- **RAG 基础能力**见 [`../rag_basics/`](../rag_basics/README.md)：chunking、embedding 等共通模块。

## 3. 相关资源

- [智能体系统（Agentic System）](../../08_agentic_system/README.md) — GraphRAG 中的多智能体协同与 Agent-G 等架构对应基础设施。
- [LLM 理论与基础](../../06_llm_theory_and_fundamentals/README.md) — CoT、幻觉等底层概念。
