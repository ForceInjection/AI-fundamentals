# AI Agent 开发与实践

## 1. 概述

本目录包含 AI Agent 开发的完整技术体系，涵盖从基础理论到企业级实践的全方位内容。文档体系分为理论基础、方法论指导、学术研究和实际应用四个层次，为 AI Agent 开发者提供系统性的学习和实践资源。

## 2. 核心文档

### 2.1 基础理论与框架

- **[多智能体 AI 系统基础：理论与框架](./Part1-Multi-Agent-AI-Fundamentals.md)** - 多智能体 AI 系统的核心理论、技术框架和 LangGraph/LangSmith 平台深度解析
- **[企业级多智能体 AI 系统构建实战](./Part2-Enterprise-Multi-Agent-System-Implementation.md)** - 基于 LangGraph 和 LangSmith 的企业级多智能体系统架构设计、代码实现和部署方案

### 2.2 方法论与最佳实践

- **[12-Factor Agents - 构建可靠 LLM 应用的原则](./12-factor-agents-intro.md)** - 借鉴 12 Factor Apps 方法论，为构建可靠的 LLM 应用提供的 12 个核心要素和最佳实践指南

### 2.3 专业技术领域

#### 2.3.1 上下文工程

- **[上下文工程原理](./context/上下文工程原理.md)** - 上下文工程的理论基础和实现原理
- **[上下文工程原理简介](./context/上下文工程原理简介.md)** - 上下文工程的入门介绍
- **[基于上下文工程的 LangChain 智能体应用](./context/langchain_with_context_engineering.md)** - LangChain 框架中的上下文工程实践

#### 2.3.2 记忆系统

- **[AI 智能体记忆系统：理论与实践](./memory/AI%20智能体记忆系统：理论与实践.md)** - 智能体记忆系统的设计原理和实现方法

### 2.4 实践项目

#### 2.4.1 多智能体系统


#### 2.4.2 多智能体训练

  - 理论基础
  - LangGraph 框架
  - LangSmith 监控
  - 企业级架构
  - 应用实践

### 2.5 应用案例

- **[Coze 部署和配置手册](../06_llm_theory_and_fundamentals/workflow/Coze%E9%83%A8%E7%BD%B2%E5%92%8C%E9%85%8D%E7%BD%AE%E6%89%8B%E5%86%8C.md)** - Coze 平台的部署配置指南
- **[支持多轮对话指代消解的 ChatBot 系统：架构设计与实现详解](./如何设计支持多轮指代消解的对话系统.md)** - 支持多轮对话的智能体系统设计方案

## 3. 学术研究

### 3.1 论文解读

- **[A Survey on Agent Workflow – Status and Future](./paper/agent-workflow-survey.md)** - Agent 工作流领域的系统性综述，涵盖 24 个主流框架的评估和未来发展方向
- **[深度研究智能体（Deep Research Agents）的定义与核心能力](./paper/deepresearch-agent.md)** - 深度研究智能体的正式定义、核心特点和技术实现
- **[论文资源汇总](./paper/README.md)** - Agent 相关论文的收集和分类整理

## 4. 实际应用场景

### 4.1 科研领域

- **[科研助手 Agent 需求与场景清单](../06_llm_theory_and_fundamentals/deep_research/design/%E7%A7%91%E7%A0%94%E5%8A%A9%E6%89%8B.md)** - 面向研究者的全生命周期智能助手系统设计，覆盖从研究发现到成果传播的完整闭环

### 4.2 企业应用

- **[订单履约 Agent 系统设计文档](../06_llm_theory_and_fundamentals/deep_research/design/%E8%AE%A2%E5%8D%95%E5%B1%A5%E7%BA%A6Agent%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3.md)** - 制造业订单履约业务的智能化 Agent 系统设计，实现从订单接收到库存分配的全流程自动化
- **[订单履约 Agent 需求分析](../06_llm_theory_and_fundamentals/deep_research/design/%E8%AE%A2%E5%8D%95%E5%B1%A5%E7%BA%A6Agent%E9%9C%80%E6%B1%82%E5%88%86%E6%9E%90.md)** - 订单履约 Agent 系统的详细需求分析和业务流程设计

### 4.3 数据科学


### 4.4 技术研究

- **[《Building Research Agents for Tech Insights》深度解读](../06_llm_theory_and_fundamentals/deep_research/%E3%80%8ABuilding%20Research%20Agents%20for%20Tech%20Insights%E3%80%8B%E6%B7%B1%E5%BA%A6%E8%A7%A3%E8%AF%BB.md)** - 技术洞察研究 Agent 的构建方法和实践案例

## 5. 技术栈

### 5.1 核心框架

- **LangChain** - 大语言模型应用开发框架
- **LangGraph** - 多智能体工作流编排框架
- **LangSmith** - LLM 应用监控和调试平台

### 5.2 部署与运维

- **Docker** - 容器化部署
- **Docker Compose** - 多容器应用编排
- **Python** - 主要编程语言

### 5.3 平台与工具

- **Coze** - 智能体开发平台
- **Databricks** - 数据科学和机器学习平台

## 6. 学习路径建议

### 6.1 初学者路径

1. 阅读 [多智能体 AI 系统基础：理论与框架](./Part1-Multi-Agent-AI-Fundamentals.md) 了解基础概念
2. 学习 [12-Factor Agents](./12-factor-agents-intro.md) 掌握最佳实践
3. 通过 [多智能体训练课程](../10_ai_related_course/multi_agent_system/multi_agent_training) 进行系统性学习
4. 参考 [应用场景](../99_misc/scenario) 了解实际应用

### 6.2 进阶开发者路径

1. 深入学习 [企业级多智能体 AI 系统构建实战](./Part2-Enterprise-Multi-Agent-System-Implementation.md)
2. 研究 [学术论文](./paper/) 了解前沿技术
3. 实践 [多智能体系统项目](../10_ai_related_course/multi_agent_system) 获得实战经验
4. 根据具体需求参考相应的 [应用场景文档](../99_misc/scenario)

## 7. 相关资源

### 7.1 官方文档

- **[LangChain 官方文档](https://python.langchain.com/en/latest/)** - 详细的 LangChain 框架文档
- **[LangGraph 官方文档](https://langgraph.readthedocs.io/en/latest/)** - 详细的 LangGraph 框架文档
- **[LangSmith 官方文档](https://docs.langchain.com/langsmith/)** - 详细的 LangSmith 监控文档
- **[Coze 官方文档](https://www.coze.com/docs/)** - 详细的 Coze 平台文档

### 7.2 学术资源

- **[AI Agents 论文集合](https://github.com/AI4Finance-Foundation/awesome-agent)** - AI Agents 相关论文的系统性收集

---
