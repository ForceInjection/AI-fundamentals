# AI Agent 开发与实践

本目录包含 AI Agent 开发的完整技术体系，涵盖从基础理论到企业级实践的全方位内容，旨在为开发者提供构建生产级智能体系统的系统化指南。

## 1. 核心理论与框架

本章深入探讨构建智能体系统的理论基石，从多智能体协作机制到通用的设计模式，为复杂系统的构建提供指导原则。

### 1.1 多智能体系统

聚焦于多个智能体如何通过协作解决复杂问题，涵盖 BDI 架构、通信机制及企业级落地框架。

- [**多智能体 AI 系统基础：理论与框架**](./multi_agent/Part1-Multi-Agent-AI-Fundamentals.md) - BDI 架构、协作机制与 LangGraph 深度解析
- [**企业级多智能体 AI 系统构建实战**](./multi_agent/Part2-Enterprise-Multi-Agent-System-Implementation.md) - 基于 LangGraph 的架构设计与代码落地

### 1.2 智能体设计模式

总结了业界成熟的智能体设计范式，包括推理与行动协同、复杂内容创作及高阶对话管理。

- [**ReAct Agent 模式详解**](./agent_design/react-agent.md) - 推理 (Reasoning) 与行动 (Acting) 的协同机制
- [**写作 Agentic Agent 设计**](./agent_design/写作%20Agentic%20Agent.md) - 复杂内容创作领域的智能体架构设计
- [**多轮指代消解对话系统**](./agent_design/如何设计支持多轮指代消解的对话系统.md) - 高级对话状态管理与上下文理解
- [**12-Factor Agents**](./concepts/12-factor-agents-intro.md) - 构建可靠 LLM 应用的 12 要素原则

### 1.3 智能体认知模型

探讨智能体如何理解世界、预测未来并进行规划的内部认知机制。

- [**世界模型简介**](./concepts/world_model_introduction.md) - 智能体理解世界的内部引擎，涵盖 RSSM、JEPA 及生成式架构

## 2. 核心组件与工程

详细拆解智能体系统的关键工程组件，包括上下文管理、长期记忆、工具协议及底层基础设施。

### 2.1 上下文工程

探讨如何高效管理和优化 LLM 的上下文窗口，通过动态组装与压缩技术提升系统性能与响应质量。

- [**上下文工程原理**](./context/上下文工程原理.md) - 动态上下文组装的理论基础与实现机制
- [**Anthropic 上下文工程指南**](./context/anthropic_context_engineering_zh.md) - 来自 Anthropic 的最佳实践译文
- [**LangChain 上下文工程实践**](./context/langchain_with_context_engineering.md) - 结合 LangChain 框架的工程化落地

### 2.2 记忆系统

介绍智能体的记忆机制，从理论模型到 MemoryOS 与 Mem0 等实战架构，赋予智能体长期记忆与个性化能力。

- [**AI 智能体记忆系统综述**](./memory/docs/AI%20智能体记忆系统：理论与实践.md) - 记忆系统的理论模型与技术路线
- [**MemoryOS 架构设计**](./memory/docs/MemoryOS智能记忆系统架构设计与开发指南.md) - 模块化智能记忆管理系统设计
- [**Mem0 快速入门**](./memory/docs/mem0快速入门.md) - 个性化记忆库 Mem0 的实战指南

### 2.3 工具与互操作性 (Tools & MCP)

关注智能体与外部世界的交互能力，特别是通过 Model Context Protocol (MCP) 实现的标准化工具互操作。

- [**Model Context Protocol (MCP) 深度解析**](./mcp/A_Deep_Dive_Into_MCP_and_the_Future_of_AI_Tooling_zh_CN.md) - 定义 AI 工具互操作未来的通用标准
- [**Claude Skills 开发指南**](./agent-skills/claude_skills_guide.md) - 扩展智能体能力的工具定义与最佳实践

### 2.4 基础设施

解析支撑大规模智能体运行的技术栈，涵盖编排、监控、部署等关键环节，构建稳健的 Agent 运行环境。

- [**AI Agent 基础设施技术栈**](./agent_infra/ai-agent-infra-stack.md) - 工具层、数据层与编排层的三层架构
- [**AI Agent 基础设施的崛起**](./agent_infra/the-rise-of-ai-agent-infrastructure.md) - 基础设施生态的演进趋势

---

## 3. 实战项目与代码

提供可运行的代码示例与完整项目源码，帮助开发者从理论走向实践，快速构建自己的智能体应用。

### 3.1 完整系统

包含经过验证的端到端系统实现，展示了多智能体协作与 MCP 服务的完整代码结构。

- [**企业级多智能体系统源码**](./multi_agent/multi_agent_system/README.md) - 基于 Python 的完整 MAS 实现，包含通信总线与监控集成
- [**MCP Demo 服务**](./mcp/mcp_demo/README.md) - Model Context Protocol 服务端与客户端完整示例代码

### 3.2 专项工具

针对特定场景的实用工具库与示例，如文档处理与记忆集成，可作为构建复杂系统的积木。

- [**PDF 智能翻译器**](./agent-skills/pdf-translator/README.md) - 结合 OCR 与 LLM 的文档处理工具，支持多模态解析
- [**LangChain 记忆集成示例**](./memory/langchain/README.md) - 多种记忆模式 (ConversationBuffer, Summary 等) 的代码实现

---

## 4. 前沿研究与报告

追踪 AI Agent 领域的最新学术进展与行业动态，为技术选型与未来规划提供前瞻性参考。

### 4.1 行业报告

汇集主流技术社区与咨询机构的深度报告，分析 Agent 工程化的现状与发展趋势。

- [**LangChain Agent 工程现状报告**](./report/langchain-state-of-agent-engineering.md) - 2024 年度技术趋势与开发者生态分析

### 4.2 学术论文

精选 Agent 领域的核心论文，涵盖工作流综述与深度研究智能体等前沿课题。

- [**Deep Research Agents**](./paper/deepresearch-agent.md) - 深度研究智能体的定义、核心能力与评估
- [**Agent Workflow 综述**](./paper/agent-workflow-survey.md) - 涵盖 24 种主流工作流模式的系统性综述
- [**论文资源库**](./paper/README.md) - AI Agent 领域核心论文索引
