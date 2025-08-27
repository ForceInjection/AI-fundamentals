# 大语言模型技术文档集

## 1. 概述

本目录包含大语言模型（LLM）技术的全面文档集合，涵盖核心架构、推理优化、应用开发、工具平台等多个技术领域，为 LLM 研究和应用提供深度技术指导。

## 2. 核心文档

### 2.1 模型架构与技术原理

- **[混合专家系统（MoE）图解指南](A%20Visual%20Guide%20to%20Mixture%20of%20Experts%20(MoE).zh-CN.md)** - MoE 架构原理与实现详解
- **[量化技术图解指南](A%20Visual%20Guide%20to%20Quantization.zh-CN.md)** - 模型量化技术深度解析
- **[大模型文件格式完整指南](大模型文件格式完整指南.md)** - 模型存储格式技术规范
- **[Token 机制解密](token/llm_token_intro.md)** - LLM 中 Token 处理机制详解

### 2.2 推理系统与优化

- **[Mooncake 架构详解](Mooncake%20架构详解：以%20KV%20缓存为中心的高效%20LLM%20推理系统设计.md)** - 以 KVCache 为中心的高效推理系统
- **[基于 LLM 的意图检测](Intent%20Detection%20using%20LLM.zh-CN.md)** - 意图识别系统设计与实现

## 3. AI Agent 相关

- [AI Agents for Beginners 课程之 AI Agent及使用场景简介](AI%20Agents%20for%20Beginners%20课程之%20AI%20Agent及使用场景简介.md)
- [使用 n8n 构建多智能体系统的实践指南](n8n_multi_agent_guide.md)
- [MCP 智能体演示项目](agent/README.md)

## 4. 模型微调与训练

- [一文入门垂域模型 SFT 微调](一文入门垂域模型SFT微调.md)
- [一文读懂思维链（Chain-of-Thought, CoT）](一文读懂思维链（Chain-of-Thought,%20CoT）.md)
- [微调示例/](fine-tuning/README.md)

## 5. RAG 检索增强生成

- [Evaluating Chunking Strategies for Retrieval 总结](rag/Evaluating%20Chunking%20Strategies%20for%20Retrieval%20总结.md)
- [中文 RAG 系统 Embedding 模型选型技术文档](rag/中文RAG系统Embedding模型选型技术文档.md)
- [Cursor DeepSearch 深度搜索技术](rag/cursor-deepsearch.md)
- [写作 Agentic Agent 智能体](rag/写作%20Agentic%20Agent.md)

## 6. 嵌入技术

- [LLM 嵌入技术详解：图文指南](embedding/LLM%20Embeddings%20Explained%20-%20A%20Visual%20and%20Intuitive%20Guide.zh-CN.md)
- [大模型 Embedding 层与独立 Embedding 模型：区别与联系](embedding/embedding.md)
- [文本嵌入（Text-Embedding） 技术快速入门](embedding/text_embeddings_guide.md)

## 7. 工具与平台

- [DeepWiki 使用方法与技术原理深度分析](DeepWiki%20使用方法与技术原理深度分析.md)
- [开源大模型应用编排平台：Dify、AnythingLLM、Ragflow 与 n8n 的功能与商用许可对比分析](开源大模型应用编排平台：Dify、AnythingLLM、Ragflow%20与%20n8n%20的功能与商用许可对比分析.md)
- [markitdown 入门](markitdown/README.md)
- [ollama 入门](ollama/README.md)

## 8. 模型技术与优化

- [大模型文件格式完整指南.md](大模型文件格式完整指南.md) - 大模型文件格式完整指南
- [大模型的幻觉及其应对措施.md](大模型的幻觉及其应对措施.md) - 大模型的幻觉及其应对措施
- [让用户"说半句"话也能懂：ChatBox 的意图识别与语义理解机制解析.md](让用户"说半句"话也能懂：ChatBox%20的意图识别与语义理解机制解析.md) - 让用户"说半句"话也能懂：ChatBox 的意图识别与语义理解机制解析
- [解密 LLM 中的 Tokens](token/llm_token_intro.md)

## 9. LangGraph 框架

- [LangGraph 实战：用 Python 打造有状态智能体](langgraph/langgraph_intro.md)

## 10. MCP 协议

- [深度解析 MCP 与 AI 工具化的未来](mcp/A_Deep_Dive_Into_MCP_and_the_Future_of_AI_Tooling_zh_CN.md)

## 11. 应用场景

- [读者来信：请问7b阅读分析不同中医古籍的能力怎么样？可以进行专项训练大幅度提高这方面能力么？](scenario/traditional-chinese-medicine.md)
- [中国大陆合同审核要点清单](scenario/中国大陆合同审核要点清单.md)

## 12. 文档处理与转换

- [Marker 文档转换工具](marker.zh-CN.md) - 高质量 PDF 到 Markdown 转换工具
- [MinerU 文档解析工具](minerU_intro.md) - 多模态文档智能解析系统

## 13. 研究报告

- [DeepWiki 深度研究报告](DeepWiki%20深度研究报告.pdf)

---
