# Java AI 开发指南

本文档聚焦于 Java 生态系统中的 AI 应用开发，重点介绍 Spring AI 及其相关生态技术栈。

---

## 1. Spring AI 介绍

Spring AI 是 Spring 官方推出的 AI 工程应用框架，旨在将 Spring 生态系统中的经典设计原则（如可移植性、模块化设计）应用于人工智能领域。

它为 Java 开发者提供了一套统一的抽象 API，使得开发者可以无缝对接并切换不同的 AI 提供商（如 OpenAI、Azure、Hugging Face 等），而无需重写核心业务逻辑。Spring AI 提供了包括大模型调用、Prompt 模板库、RAG（检索增强生成）工具链、向量数据库集成以及结构化输出解析等丰富的核心功能，极大地降低了企业级 Java 应用接入 AI 能力的工程门槛。

---

## 2. Spring AI Alibaba 介绍

Spring AI Alibaba 是基于 Spring AI 框架的阿里云生态开源实现，专为国内开发环境与国产大模型生态进行了深度定制。

它不仅完美兼容 Spring AI 的标准化 API，还深度集成了阿里云通义千问（DashScope）等国产优质大语言模型。结合 Spring Cloud Alibaba 的微服务治理体系，Spring AI Alibaba 能够为开发者提供从模型无缝接入、智能体（Agent）构建到云原生分布式部署的端到端企业级 AI 解决方案，非常适合国内业务场景的落地。

---

## 3. 实践案例

本节汇总了我们基于 Java 和 Spring AI 生态框架所沉淀的项目实战案例与最佳实践文档。

> [!NOTE]
> Spring AI 官方近期发布了关于构建有效 Agent（[Building Effective Agents](https://docs.spring.io/spring-ai/reference/api/effective-agents.html)）的最新架构指南，详细区分了工作流（Workflows）与智能体（Agents）。其中重点提到了以下五种核心的工作流设计模式：
>
> - **链式工作流（Chain Workflow）**：适用于有明确顺序步骤的任务，每一步基于前一步的输出。
> - **并行工作流（Parallelization Workflow）**：适用于可同时处理、相互独立的任务或视角聚合。
> - **路由工作流（Routing Workflow）**：用于将不同类别的输入智能分发给专门的处理逻辑。
> - **协调者-工作者模式（Orchestrator-Workers）**：由协调者动态分析任务并分配给多个工作者并行处理，适用于复杂、不可预知的子任务。
> - **评估者-优化者模式（Evaluator-Optimizer）**：通过生成、评估和优化的多轮迭代，适用于需要不断完善和有明确评价标准的任务。

较早的博客：[使用 Spring AI 构建高效 LLM 代理（第一部分）](spring_ai_cn.md) - 详细演示了如何使用 Spring AI 框架构建企业级的高效 LLM 代理（Agent）系统，涵盖了框架核心概念解析、多模型提供商集成策略以及架构设计最佳实践。
