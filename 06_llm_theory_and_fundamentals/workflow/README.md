# 工作流编排与应用平台（Workflow）

写一个调 ChatGPT 的 demo 很简单；但要把 LLM 嵌入真实业务里，还得解决数据接入、多步流程、权限与审计、插件扩展、线上观测等一堆工程问题。工作流编排平台（Dify / n8n / Coze / AnythingLLM / Ragflow 等）就是填这道鸿沟的通用中间层。本目录聚焦选型对比与两套典型平台的落地实践。

## 1. 核心内容

### 1.1 平台对比与选型

- **[开源 LLM 应用编排平台对比](open_source_llm_orchestration_platforms_comparison.md)** — 横向对比 Dify、AnythingLLM、Ragflow、n8n 与 Coze Studio 在工作流建模、RAG 能力、Agent 支持、插件生态以及商用许可（License）上的差异，帮助企业按合规约束与功能需求做选型。

### 1.2 部署与配置指南

- **[Coze（扣子）部署与配置手册](coze_deployment_and_configuration_guide.md)** — Coze 开源版的私有化部署、工作流搭建、插件接入与 Agent 发布的端到端操作指引。

### 1.3 多智能体实践

- **[n8n 多智能体系统实战](n8n_multi_agent_guide.md)** — 以 n8n 作为编排引擎，用工作流节点串联多个 LLM Agent 形成协作链路，覆盖分工、消息路由与状态回收的完整范式。

## 2. 相关资源

- [智能体系统架构](../../08_agentic_system/README.md)
- [大语言模型基础理论](../llm_basic_concepts/README.md)
