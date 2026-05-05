# LLM 基础概念

一个大语言模型怎么把一段中文「读进去」、怎么在内部生成推理链、怎么压缩到能跑在消费级显卡、又怎么避免「听起来很对其实是编的」？本目录把这些看似独立但实际互相咬合的基础概念整理在一起，作为进一步研究训练、推理、RAG、Agent 等上层话题之前的统一参考系。

## 1. 核心概念与理论

- **[思维链 (CoT)](cot/chain_of_thought_cot_intro.md)** — 通过显式书写中间推理步骤提升复杂任务准确率的机制，及其与 few-shot / zero-shot / self-consistency 的关系。
- **[Token 机制](token/README.md)** — 切分算法（BPE / WordPiece）、长度估算工具与 Token-based 成本控制实战，配套 `token_estimation.py` 脚本与 Dockerfile。
- **[模型幻觉 (Hallucination)](hallucination/llm_hallucination_and_mitigation.md)** — 幻觉成因的分层解释（数据层 / 训练层 / 推理层）及检索、约束、校验三类缓解手段。

## 2. 嵌入（Embedding）

文本嵌入把离散符号压成稠密向量，是 RAG、聚类、分类、异常检测等几乎所有下游任务的共同底座。相关文档收在 [`embedding/`](embedding/README.md)：

- **[深入了解文本嵌入](embedding/text_embeddings_comprehensive_guide.md)** — 从 BoW、TF-IDF、Word2Vec 到 Transformer 句向量的完整演进，附 L2/曼哈顿/点积/余弦等距离度量与 PCA/t-SNE 可视化实战。
- **[LLM 嵌入技术图文指南](embedding/LLM_embeddings_explained_visual_guide.zh-CN.md)** — 以几何直觉讲清 Embedding 在向量空间里的位置与关系。
- **[文本嵌入快速入门](embedding/text_embeddings_guide.md)** — 面向新手的最短上手路径。
- **[LLM 内嵌 Embedding 层 vs. 独立 Embedding 模型](embedding/embedding.md)** — 剖析 LLM 内部 Embedding 层与 BGE / OpenAI text-embedding-3 等外部模型的架构差异与协作方式。

## 3. 模型架构与优化

- **[混合专家 (MoE)](moe/mixture_of_experts_moe_visual_guide.zh-CN.md)** — 稀疏激活、专家路由与负载均衡，如何让模型参数量增长而不线性增加推理成本。
- **[模型量化 (Quantization)](quantization/01_visual_guide_to_quantization.md)** — FP16 / INT8 / INT4 / GPTQ / AWQ 等量化路径的图解解析，以及精度—性能的折中决策。

## 4. 文件格式与应用层技术

- **[大模型文件格式](file_formats/llm_file_formats_complete_guide.md)** — GGUF / GGML / Safetensors 的存储结构、元数据布局与互转注意事项。
- **[基于 LLM 的意图检测](intent_detection/intent_detection_using_llm.zh-CN.md)** — 构建可靠意图识别系统的设计路径与常见工程陷阱。
  - 参见：[ChatBox 意图识别与语义理解](intent_detection/chatbox_intent_recognition_and_semantic_understanding.md)。

## 5. 相关资源

- [推理系统与优化](../../09_inference_system/README.md)
- [模型训练与微调](../../05_model_training_and_fine_tuning/README.md)
- [智能体系统（Agentic System）](../../08_agentic_system/README.md)
