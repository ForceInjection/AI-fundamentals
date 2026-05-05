# RAG 与工具生态

把「让大模型用上领域知识」这件事做好，有两条绕不开的路：**RAG**（用检索补全 LLM 知识的时效性与专精性）和**知识图谱**（把结构化关系喂给 LLM 做可控推理）。两条路各自又分出「基础能力」「架构升级」和「数据输入」三层关切——本目录就沿着这条脉络展开：从 chunking / embedding 基本功，到 GraphRAG / KAG 的图驱动推理，再到银行反电诈的 LLM + KG 协同案例；所有这些都建立在高质量的数据输入之上，所以也收录了 MinerU / Marker / MarkItDown 三款 PDF 解析工具的对比与实战。

## 1. RAG 基础能力

真正决定 RAG 线上效果的是「切、嵌、编排」三件事。详细索引见 [`rag_basics/README.md`](rag_basics/README.md)。

- [RAG 快速开发实战（从 0 到 1 搭建）](https://mp.weixin.qq.com/s/89-bwZ4aPor4ySj5U3n5zw) — RAG 技术全景导航。
- [RAG 策略对比](rag_basics/rag_comparison.md) — 10 类 Agentic RAG 架构的选型矩阵（Naive / Router / Multi-Agent / Corrective / Adaptive / Graph / Agent-G / GeAR / ADW）。
- [Chunking 策略评估](rag_basics/evaluating_chunking_strategies_summary.md) — token 级 Precision / Recall / IoU 替代 nDCG@K 的新评估方法。
- [中文 Embedding 模型选型](rag_basics/chinese_rag_embedding_model_selection.md) — BGE / GTE / M3E / Conan 等主流模型的 MTEB-zh 对比与场景推荐。

## 2. GraphRAG 与知识图谱

Naive RAG 在多跳推理、数值/时间约束、领域 Schema 对齐上会失效。图驱动的方案（GraphRAG / KAG）把文档变成结构化知识让 LLM 沿图走。详细索引见 [`graph_rag/README.md`](graph_rag/README.md) 与 [`knowledge_graph/README.md`](knowledge_graph/README.md)。

- [GraphRAG 学习指南](graph_rag/graph_rag_learning_guide.md) — 基于 DeepLearning.AI × Neo4j 的 _Knowledge Graphs for RAG_ 课程重组。
- [KAG 框架介绍](graph_rag/kag_introduction.md) — OpenSPG 的 LLMFriSPG 知识表示 + 逻辑形式（Logical Form）引导求解。
- [Neo4j Cypher 查询语言](knowledge_graph/neo4j_cypher_tutorial.md) — 节点 / 关系 / 模式匹配的权威教程。
- [Neo4j 实战指南](knowledge_graph/neo4j_handson_guide.md) — Docker 起容器、Browser 登录、真实反欺诈数据上手。

## 3. LLM + KG 协同落地案例

以银行反电信网络诈骗为标本，把设计方案与可运行 demo 打通。详细索引见 [`synergized_llms_kgs/README.md`](synergized_llms_kgs/README.md)。

- [银行反电诈智能系统设计方案](synergized_llms_kgs/anti_fraud_design.md) — 完整方案：场景挑战、图谱建模、LLM 可解释归因、合规边界、工程化路线。
- [反欺诈 Demo 源码](synergized_llms_kgs/demo/README.md) — 从合成数据、Neo4j 导入、LLM 研判到 API 服务的端到端示例。

## 4. 文档智能解析工具链

「Garbage in, garbage out」——RAG 的质量上限是 PDF 解析的质量。详细对比与选型见 [`pdf_tools/README.md`](pdf_tools/README.md)。

- [MinerU 高效解析 PDF](pdf_tools/miner_u_intro.md) — 上海 AI Lab 开源，复杂中文 PDF（公式 / 表格 / 多栏）首选。
- [Marker 源码解析（译）](pdf_tools/marker_zh_cn.md) — 英文文献深度解析引擎，比 nougat 快 10 倍。
- [MarkItDown 容器化](pdf_tools/markitdown/README.md) — Microsoft 通用文档转 Markdown，覆盖 Office / 图像 OCR / 音频转写。

## 5. 相关资源

- [LLM 理论与基础](../06_llm_theory_and_fundamentals/README.md) — Embedding、Token、Hallucination 等底层机制。
- [智能体系统（Agentic System）](../08_agentic_system/README.md) — Agentic RAG 背后的 Agent 基础设施与多智能体协作。
- [模型训练与微调](../05_model_training_and_fine_tuning/README.md) — 当 RAG 不够时，用垂域 SFT 进一步补齐领域能力。
- [推理系统与优化](../09_inference_system/README.md) — 大规模 RAG 在线服务的延迟与显存约束。
