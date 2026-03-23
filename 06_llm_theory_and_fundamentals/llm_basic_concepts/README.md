# 大语言模型技术理论与基础

本目录包含大语言模型（LLM）底层技术的全面理论文档，涵盖核心架构、模型优化、基础概念解析等多个技术领域，为深入理解和应用 LLM 提供技术指导。

## 1. 核心概念与理论

- **[思维链 (CoT)](cot/chain_of_thought_cot_intro.md)**: 一文读懂思维链（Chain-of-Thought）的工作原理及其在提升大模型推理能力中的作用。
- **[Token 机制](token/README.md)**: 解密 LLM 中的 Tokens，包括 Token 的处理机制、分词算法以及长度估算工具。
- **[模型幻觉 (Hallucination)](hallucination/llm_hallucination_and_mitigation.md)**: 大模型的幻觉现象解析及其应对与缓解措施。

## 2. 嵌入技术 (Embedding)

嵌入技术是将文本、图像等数据映射为高维向量空间的核心技术。本目录提供详细的指南：

- **[深入了解文本嵌入技术](embedding/text_embeddings_comprehensive_guide.md)**: 全面解析 Text Embeddings 的演变、距离度量及应用。
- **[LLM 嵌入技术详解：图文指南](embedding/LLM_embeddings_explained_visual_guide.zh-CN.md)**: 视觉化直观理解大模型 Embeddings。
- **[文本嵌入技术快速入门](embedding/text_embeddings_guide.md)**: 快速上手文本嵌入技术的实用指南。
- **[大模型 Embedding 层与独立 Embedding 模型：区别与联系](embedding/embedding.md)**: 解析模型内部 Embedding 层与外部独立模型的差异。
- _查看 [嵌入技术完整指南](embedding/README.md) 获取更多信息。_

## 3. 模型架构与优化

- **[混合专家模型 (MoE)](moe/mixture_of_experts_moe_visual_guide.zh-CN.md)**: 图解 Mixture of Experts 架构，解析如何通过稀疏激活提升模型规模与效率。
- **[模型量化 (Quantization)](quantization/01_visual_guide_to_quantization.md)**: 模型量化技术深度图解指南，探讨如何降低大模型推理资源需求。

## 4. 文件格式与应用层技术

- **[大模型文件格式](file_formats/llm_file_formats_complete_guide.md)**: 深入解析 GGUF, GGML, Safetensors 等主流大模型存储格式与技术规范。
- **[基于 LLM 的意图检测](intent_detection/intent_detection_using_llm.zh-CN.md)**: 意图识别系统设计与实现。
  - _参见：[ChatBox 意图识别与语义理解](intent_detection/chatbox_intent_recognition_and_semantic_understanding.md)_

## 5. 相关资源

- [推理系统与优化](../../09_inference_system/README.md)
- [模型训练与微调](../../05_model_training_and_fine_tuning/README.md)
- [智能体系统 (Agentic System)](../../08_agentic_system/README.md)
