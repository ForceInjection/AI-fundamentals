# RAG 与工具生态

本目录专注于检索增强生成 (RAG) 技术、知识图谱 (KG) 以及文档智能处理工具的深度探索与实践，旨在构建从非结构化数据处理到高阶推理应用的完整技术栈。

## 1. RAG 基础与进阶

探索 RAG 系统的核心组件、策略对比与模型选型，构建高效的检索增强生成系统。

- [**RAG 技术原理与实践**](rag/README.md) - RAG 技术全景导航，涵盖基础概念到进阶优化
- [**RAG 策略对比**](rag/RAG%20对比.md) - 不同 RAG 架构（Naive RAG, Advanced RAG 等）的优劣势分析
- [**Chunking 策略评估**](rag/Evaluating%20Chunking%20Strategies%20for%20Retrieval%20总结.md) - 检索分块策略的深度总结与最佳实践
- [**中文 Embedding 模型选型**](rag/中文RAG系统Embedding模型选型技术文档.md) - 面向中文场景的 Embedding 模型评测与推荐

## 2. GraphRAG 与知识图谱

结合知识图谱增强 RAG 的推理能力，深入 GraphRAG 前沿技术，解决复杂关系推理难题。

- [**GraphRAG 学习指南**](GraphRAG/GraphRAG_Learning_Guide.md) - GraphRAG 的核心概念、架构原理与入门路径
- [**KAG 框架介绍**](GraphRAG/KAG_Introduction.md) - Knowledge Augmented Generation (KAG) 框架深度解析
- [**Neo4j 实战指南**](KG/Neo4j_HandsOn_Guide.md) - 图数据库 Neo4j 的安装、配置与企业级实战
- [**Cypher 查询语言教程**](KG/Neo4j_Cypher_Tutorial.md) - Neo4j 查询语言 Cypher 的快速入门与进阶技巧

## 3. LLM + KG 协同应用

探索大语言模型与知识图谱的深度融合，构建高可信、可解释的智能应用。

- [**银行反电诈智能系统设计**](Synergized%20LLMs%20+%20KGs/anti_fraud_design.md) - 基于 LLM + KG 的金融风控系统设计方案，实战反欺诈场景
- [**反欺诈 Demo 源码**](Synergized%20LLMs%20+%20KGs/demo/README.md) - 完整的反欺诈系统演示代码，包含数据生成、图谱构建与智能体推理

## 4. 文档智能解析

高效处理非结构化文档（PDF, Office 等），为 RAG 系统提供高质量的数据输入，解决“Garbage In, Garbage Out”问题。

- [**MinerU 文档解析**](pdf/minerU_intro.md) - 上海人工智能实验室开源工具，助力复杂 PDF 高效解析
- [**Marker PDF 布局检测**](pdf/marker.zh-CN.md) - 基于深度学习的高精度 PDF 解析与布局分析引擎
- [**Markitdown 入门**](pdf/markitdown/README.md) - Microsoft 开源的文档转换工具，支持多种办公文档格式到 Markdown 的高质量转换
