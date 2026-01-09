# 大语言模型技术文档集

## 1. 概述

本目录包含大语言模型（LLM）技术的全面文档集合，涵盖核心架构、推理优化、应用开发、工具平台等多个技术领域，为 LLM 研究和应用提供深度技术指导。

## 2. 核心文档

### 2.1 模型架构与技术原理

- **[量化技术图解指南](A%20Visual%20Guide%20to%20Quantization.zh-CN.md)** - 模型量化技术深度解析
- **[大模型文件格式完整指南](大模型文件格式完整指南.md)** - 模型存储格式技术规范
- **[Token 机制解密](token/llm_token_intro.md)** - LLM 中 Token 处理机制详解

### 2.2 推理系统与优化

- **[Mooncake 架构详解](../../09_inference_system/Mooncake%20%E6%9E%B6%E6%9E%84%E8%AF%A6%E8%A7%A3%EF%BC%9A%E4%BB%A5%20KV%20%E7%BC%93%E5%AD%98%E4%B8%BA%E4%B8%AD%E5%BF%83%E7%9A%84%E9%AB%98%E6%95%88%20LLM%20%E6%8E%A8%E7%90%86%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1.md)** - 以 KVCache 为中心的高效推理系统
- **[基于 LLM 的意图检测](Intent%20Detection%20using%20LLM.zh-CN.md)** - 意图识别系统设计与实现

## 3. AI Agent 相关

- [AI Agents for Beginners 课程之 AI Agent及使用场景简介](../../10_ai_related_course/AI%20Agents%20for%20Beginners%20%E8%AF%BE%E7%A8%8B%E4%B9%8B%20AI%20Agent%E5%8F%8A%E4%BD%BF%E7%94%A8%E5%9C%BA%E6%99%AF%E7%AE%80%E4%BB%8B.md)
- [使用 n8n 构建多智能体系统的实践指南](../workflow/n8n_multi_agent_guide.md)
- [MCP 智能体演示项目](../../README.md)

## 4. 模型微调与训练

- [一文入门垂域模型 SFT 微调](../../05_model_training_and_fine_tuning/sft_example/一文入门垂域模型SFT微调.md)
- [一文读懂思维链（Chain-of-Thought, CoT）](一文读懂思维链（Chain-of-Thought,%20CoT）.md)
- [微调示例/](../../05_model_training_and_fine_tuning/sft_example/README.md)

## 5. RAG 检索增强生成

- [Evaluating Chunking Strategies for Retrieval 总结](../../07_rag_and_tools/rag/Evaluating%20Chunking%20Strategies%20for%20Retrieval%20%E6%80%BB%E7%BB%93.md)
- [中文 RAG 系统 Embedding 模型选型技术文档](../../07_rag_and_tools/rag/%E4%B8%AD%E6%96%87RAG%E7%B3%BB%E7%BB%9FEmbedding%E6%A8%A1%E5%9E%8B%E9%80%89%E5%9E%8B%E6%8A%80%E6%9C%AF%E6%96%87%E6%A1%A3.md)
- [Cursor DeepSearch 深度搜索技术](../deep_research/cursor-deepsearch.md)
- [写作 Agentic Agent 智能体](../../08_agentic_system/%E5%86%99%E4%BD%9C%20Agentic%20Agent.md)

## 6. 嵌入技术

- [LLM 嵌入技术详解：图文指南](embedding/LLM%20Embeddings%20Explained%20-%20A%20Visual%20and%20Intuitive%20Guide.zh-CN.md)
- [大模型 Embedding 层与独立 Embedding 模型：区别与联系](embedding/embedding.md)
- [文本嵌入（Text-Embedding） 技术快速入门](embedding/text_embeddings_guide.md)

## 7. 工具与平台

- [DeepWiki 使用方法与技术原理深度分析](../deep_research/DeepWiki%20%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95%E4%B8%8E%E6%8A%80%E6%9C%AF%E5%8E%9F%E7%90%86%E6%B7%B1%E5%BA%A6%E5%88%86%E6%9E%90.md)
- [开源大模型应用编排平台：Dify、AnythingLLM、Ragflow 与 n8n 的功能与商用许可对比分析](../workflow/%E5%BC%80%E6%BA%90%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8%E7%BC%96%E6%8E%92%E5%B9%B3%E5%8F%B0%EF%BC%9ADify%E3%80%81AnythingLLM%E3%80%81Ragflow%20%E4%B8%8E%20n8n%20%E7%9A%84%E5%8A%9F%E8%83%BD%E4%B8%8E%E5%95%86%E7%94%A8%E8%AE%B8%E5%8F%AF%E5%AF%B9%E6%AF%94%E5%88%86%E6%9E%90.md)
- [markitdown 入门](../../README.md)
- [ollama 入门](../../README.md)

## 8. 模型技术与优化

- [大模型文件格式完整指南.md](大模型文件格式完整指南.md) - 大模型文件格式完整指南
- [大模型的幻觉及其应对措施.md](大模型的幻觉及其应对措施.md) - 大模型的幻觉及其应对措施
- [解密 LLM 中的 Tokens](token/llm_token_intro.md)

## 9. LangGraph 框架

- [LangGraph 实战：用 Python 打造有状态智能体](../../98_llm_programming/langgraph/langgraph_intro.md)

## 10. MCP 协议

- [深度解析 MCP 与 AI 工具化的未来](../../08_agentic_system/mcp/A_Deep_Dive_Into_MCP_and_the_Future_of_AI_Tooling_zh_CN.md)

## 11. 应用场景

- [读者来信：请问7b阅读分析不同中医古籍的能力怎么样？可以进行专项训练大幅度提高这方面能力么？](../../99_misc/scenario/traditional-chinese-medicine.md)
- [中国大陆合同审核要点清单](../../99_misc/scenario/%E4%B8%AD%E5%9B%BD%E5%A4%A7%E9%99%86%E5%90%88%E5%90%8C%E5%AE%A1%E6%A0%B8%E8%A6%81%E7%82%B9%E6%B8%85%E5%8D%95.md)

## 12. 文档处理与转换

- [Marker 文档转换工具](../../07_rag_and_tools/pdf/marker.zh-CN.md) - 高质量 PDF 到 Markdown 转换工具
- [MinerU 文档解析工具](../../07_rag_and_tools/pdf/minerU_intro.md) - 多模态文档智能解析系统

## 13. 研究报告

- [DeepWiki 深度研究报告](../deep_research/DeepWiki%20%E6%B7%B1%E5%BA%A6%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A.pdf)

---
