# LLM 理论与基础

用好大语言模型，前提是先理解它是怎么工作的：文本是如何被切成 token 的、语义是如何被压成向量的、推理能力从哪里涌现、幻觉又从哪里漏出来。本目录把这些底层概念集中到一处，同时向上延伸到两类应用：一类是把 LLM 组装成自动化深度研究智能体（Deep Research），另一类是用 Dify / n8n / Coze 等编排平台把模型能力装进业务流水线。

## 1. 内容概览

### 1.1 LLM 基础概念

围绕 Token、Embedding、注意力之外的「结构性」技术组成基础能力地图，详见 [`llm_basic_concepts/`](llm_basic_concepts/README.md)。

- **[思维链 (CoT)](llm_basic_concepts/cot/chain_of_thought_cot_intro.md)** — 通过显式推理步骤提升复杂问题求解能力，是多跳推理与工具调用的前置条件。
- **[嵌入 (Embedding)](llm_basic_concepts/embedding/README.md)** — 从 Bag-of-Words / TF-IDF 到 Transformer 句向量的演进，覆盖距离度量、降维可视化与 RAG/聚类/分类等下游用法。
- **[混合专家 (MoE)](llm_basic_concepts/moe/mixture_of_experts_moe_visual_guide.zh-CN.md)** — 稀疏激活架构如何在不线性放大推理成本的前提下扩展参数规模。
- **[模型量化 (Quantization)](llm_basic_concepts/quantization/01_visual_guide_to_quantization.md)** — FP16/INT8/INT4 等精度压缩路径，用于降低显存占用与推理延迟。
- **[Token 机制](llm_basic_concepts/token/README.md)** — BPE / WordPiece 的切分逻辑、长度估算工具与成本控制实践。
- **[幻觉 (Hallucination)](llm_basic_concepts/hallucination/llm_hallucination_and_mitigation.md)** — 幻觉的成因分层与检索/约束/校验三类缓解策略。
- **[模型文件格式](llm_basic_concepts/file_formats/llm_file_formats_complete_guide.md)** — GGUF / GGML / Safetensors 的存储结构与互转注意事项。
- **[意图检测](llm_basic_concepts/intent_detection/intent_detection_using_llm.zh-CN.md)** — 基于 LLM 的意图识别管线与常见工程陷阱。

### 1.2 深度研究（Deep Research）

当「回答」演变成「写一份带引用的研究报告」时，需要搜索 / 阅读 / 规划 / 写作多能力协同。本节收录业界产品解读与落地案例，详见 [`deep_research/`](deep_research/README.md)。

- **研究智能体拆解** — [Cursor DeepSearch](deep_research/research_agents/cursor-deepsearch.md)、[通义 DeepResearch](deep_research/research_agents/qwen_deepresearch_analysis.md)、[Databricks Data Agent](deep_research/research_agents/databricks_data_agent.md)，以及 [《Building Research Agents for Tech Insights》](deep_research/research_agents/building_research_agents_for_tech_insights.md) 的技术路径解读。
- **DeepWiki** — [技术原理与使用分析](deep_research/deepwiki/deepwiki_usage_and_technical_analysis.md)，把代码仓库自动转成可检索的结构化知识。
- **场景级设计** — [科研助手](deep_research/design/research_assistant.md)、[订单履约 Agent](deep_research/design/order_fulfillment_agent_system_design.md) 两套端到端的 Agent 架构与需求拆解样例。

### 1.3 工作流编排与应用平台

把 LLM 从 Playground 搬到业务系统，中间还隔着工作流、权限、插件、可观测等一整套工程工作。本节聚焦编排平台选型与落地，详见 [`workflow/`](workflow/README.md)。

- **[开源编排平台对比](workflow/open_source_llm_orchestration_platforms_comparison.md)** — Dify / AnythingLLM / Ragflow / n8n 的功能矩阵与商用许可对比，用于企业选型。
- **[n8n 多智能体实践](workflow/n8n_multi_agent_guide.md)** — 用 n8n 的工作流节点 + LLM 节点搭建多 Agent 协作系统。
- **[Coze 部署与配置](workflow/coze_deployment_and_configuration_guide.md)** — Coze（扣子）的私有化部署、插件接入与 Agent 发布流程。

## 2. 学习路径建议

1. **筑基** — 从 `llm_basic_concepts` 进入，先打通 **Token → Embedding → CoT** 三条主线，建立「模型怎么看文本、怎么推理」的底层直觉。
2. **进阶** — 转向 **MoE** 与 **Quantization**，理解现代大模型如何在规模与成本两头取舍。
3. **应用** — 借助 `workflow` 下的开源编排平台（Dify / n8n / Coze）把模型能力装进自动化流程。
4. **前沿** — 进入 `deep_research`，研究当下最复杂的 Agent 应用形态——多能力协同的自动化研究系统。

## 3. 相关资源

本目录聚焦理论与应用层编排；如需更底层的工程实践，参考：

- [模型训练与微调 (SFT/RLHF)](../05_model_training_and_fine_tuning/README.md)
- [RAG 与工具（检索增强生成）](../07_rag_and_tools/README.md)
- [智能体系统底层架构](../08_agentic_system/README.md)
- [推理系统与优化 (KV Cache / vLLM)](../09_inference_system/README.md)
