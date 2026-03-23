# LLM 理论与基础

本目录集中了大语言模型（LLM）相关的底层理论基础、核心技术概念，同时也涵盖了前沿的深度研究（Deep Research）应用和多智能体工作流（Workflow）编排的实践指南。

## 1. 内容概览

本章节详细列出了各个子目录下的核心技术机制与架构原理。

### 1.1 LLM 基础概念

深入剖析大语言模型底层的核心技术机制与架构原理，详细内容请参考 [LLM 基础概念目录](llm_basic_concepts/README.md)。

- **[思维链 (CoT)](llm_basic_concepts/cot/chain_of_thought_cot_intro.md)**: 提升模型复杂推理能力的核心机制
- **[嵌入技术 (Embedding)](llm_basic_concepts/embedding/README.md)**: 文本向量化表示的演进、原理与图解指南
- **[混合专家模型 (MoE)](llm_basic_concepts/moe/mixture_of_experts_moe_visual_guide.zh-CN.md)**: 稀疏激活模型架构的视觉化解析
- **[模型量化 (Quantization)](llm_basic_concepts/quantization/01_visual_guide_to_quantization.md)**: 降低模型显存占用与推理成本的关键技术
- **[Token 机制](llm_basic_concepts/token/README.md)**: 模型的文本切分原理与计算估算工具
- **[模型幻觉 (Hallucination)](llm_basic_concepts/hallucination/llm_hallucination_and_mitigation.md)**: 幻觉现象的成因分析与应对策略
- **[文件格式 (File Formats)](llm_basic_concepts/file_formats/llm_file_formats_complete_guide.md)**: GGUF, GGML, Safetensors 等主流模型存储格式详解
- **[意图检测 (Intent Detection)](llm_basic_concepts/intent_detection/intent_detection_using_llm.zh-CN.md)**: 基于 LLM 的意图识别系统设计

### 1.2 深度研究

聚焦于利用 Agent 技术实现自动化科研与深度信息挖掘，详细内容请参考 [深度研究目录](deep_research/README.md)。

- **研究智能体分析**: 涵盖 [Cursor DeepSearch](deep_research/research_agents/cursor-deepsearch.md)、[通义 DeepResearch](deep_research/research_agents/qwen_deepresearch_analysis.md)、[Databricks Data Agent](deep_research/research_agents/databricks_data_agent.md) 等前沿产品与《Building Research Agents for Tech Insights》的技术解读
- **DeepWiki 项目**: [DeepWiki 技术原理与使用方法](deep_research/deepwiki/deepwiki_usage_and_technical_analysis.md)
- **智能体设计**: 针对[科研助手](deep_research/design/research_assistant.md)与[订单履约](deep_research/design/order_fulfillment_agent_system_design.md)场景的完整 Agent 架构设计与需求分析

### 1.3 工作流编排与应用平台

探讨如何将大模型能力转化为实际业务应用与自动化流程，详细内容请参考 [工作流编排与应用平台目录](workflow/README.md)。

- **[开源平台对比](workflow/open_source_llm_orchestration_platforms_comparison.md)**: Dify、AnythingLLM、Ragflow 与 n8n 的功能与商用许可全面分析
- **[多智能体实践](workflow/n8n_multi_agent_guide.md)**: 基于 n8n 构建多智能体系统的实战指南
- **[平台部署](workflow/coze_deployment_and_configuration_guide.md)**: Coze (扣子) 平台的部署、配置与插件集成手册

## 2. 学习路径建议

为了更好地掌握大语言模型的相关知识，我们为您提供以下学习路径建议。

1. **筑基**: 从 `llm_basic_concepts` 目录开始，重点理解 **Token**、**Embedding** 和 **CoT**，这是理解现代大模型应用的基础。
2. **进阶**: 深入学习 **MoE** 和 **Quantization**，理解大模型如何突破规模瓶颈并在有限算力下高效运行。
3. **应用**: 结合 `workflow` 目录，学习如何利用现有的开源编排平台（如 Dify、n8n）将大模型能力组装成智能体。
4. **前沿**: 探索 `deep_research` 目录，了解当前最复杂的 Agent 应用场景（自动化深度研究）的技术实现方案。

## 3. 相关资源

本目录侧重于理论基础与应用层编排，如需了解更底层的工程实践，请参阅以下相关资源。

- [模型训练与微调 (SFT/RLHF)](../05_model_training_and_fine_tuning/README.md)
- [RAG 与工具 (检索增强生成)](../07_rag_and_tools/README.md)
- [智能体系统底层架构 (Agentic System)](../08_agentic_system/README.md)
- [推理系统与优化 (KVCache/vLLM)](../09_inference_system/README.md)
