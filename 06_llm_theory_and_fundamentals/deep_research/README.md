# 深度研究（Deep Research）技术与应用

传统 LLM 聊天只能回答「问一答一」；而 Deep Research 要做的事是：给定一个开放式问题，自主做计划、并行搜索、跨源阅读、引用溯源，最后交付一份结构化的研究报告。这类系统同时考验模型的推理、工具使用、长上下文与工程编排能力。本目录收录业界已落地的产品解读、自动化知识生成实践，以及两套场景级的系统设计样例。

## 1. 研究智能体（Research Agents）产品拆解

这四份文档分别对应一条技术路径：从 IDE 内的代码检索、到通用研究模型、到数据分析 Agent、再到方法论层面的 tech-insights 研究流水线。

- **[《Building Research Agents for Tech Insights》解读](research_agents/building_research_agents_for_tech_insights.md)** — 梳理构建面向技术洞察的研究 Agent 所需的数据源、推理模板与评估反馈闭环。
- **[Cursor DeepSearch 技术解析](research_agents/cursor-deepsearch.md)** — Cursor 的代码 / 文档深搜路径，重点是索引策略与上下文窗口管理。
- **[Databricks Data Agent](research_agents/databricks_data_agent.md)** — 在数据仓库与笔记本语境下的 Agent 实践，关注 SQL 生成、工具编排与结果回收。
- **[通义 DeepResearch 分析](research_agents/qwen_deepresearch_analysis.md)** — Qwen DeepResearch 的模型与流程拆解，覆盖规划、检索、合成三阶段。

## 2. DeepWiki：自动化知识生成

DeepWiki 把一个 GitHub 仓库自动转成可问答的结构化 Wiki，是把 Deep Research 思路应用到「代码理解」场景的落地样本。

- **[DeepWiki 使用方法与技术原理](deepwiki/deepwiki_usage_and_technical_analysis.md)** — 工作流、技术栈与在企业知识库场景的适用性分析。
- **[DeepWiki 深度研究报告（PDF）](deepwiki/deepwiki_research_report.pdf)** — 由 DeepWiki 自身生成的完整研究报告样例。

## 3. 场景级系统设计（Design）

两套端到端的 Agent 设计样例，展示从需求拆解到架构落地的完整链路：

- **[科研助手 Agent 设计](design/research_assistant.md)** — 面向科研工作的助手系统，覆盖文献搜索、论文解读、写作辅助等能力编排。
- **[订单履约 Agent 需求分析](design/order_fulfillment_agent_requirement_analysis.md)** — 订单履约场景下的业务痛点、功能边界与评估指标拆解。
- **[订单履约 Agent 系统设计](design/order_fulfillment_agent_system_design.md)** — 与需求分析配套的完整系统架构方案。

## 4. 相关资源

- [智能体系统基础架构](../../08_agentic_system/README.md)
- [工作流编排与多智能体](../workflow/README.md)
