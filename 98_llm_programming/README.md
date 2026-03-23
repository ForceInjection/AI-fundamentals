# 大语言模型应用开发与编排

本目录深入探讨在 AI 时代下新兴的应用层开发范式与工作流，重点关注如何利用各种编程语言、框架和编排工具构建复杂的 LLM 应用及智能体（Agent）系统。

---

## 1. AI 时代的软件工程：范式转移与重构

随着大语言模型（LLM）能力的爆发式增长，传统的软件工程正在经历一场深刻的**范式转移（Paradigm Shift）**。我们正从以“人类开发者写逻辑、机器死板执行”的 **Software 1.0/2.0** 时代，加速迈向以“自然语言驱动、Agent 自主决策与推理”为核心的 **Software 3.0** 时代。

在这种新范式下，“代码”不再仅仅是确定性的指令，而是成为了构建“上下文环境（Context）”与“边界约束（Constraints）”的工具。**Agent First（智能体优先）** 成为核心设计理念，要求开发者在系统架构、API 设计和文档编写时，首要考虑的是“机器（Agent）能否读懂并高效调用”，而非仅服务于人类用户。

这并非意味着软件工程标准的降低，而是**严谨性的迁移**。开发者面临的挑战从“如何写出无 Bug 的功能代码”，转变为“如何构建一套可控的驾驭系统（Harness System）”，以约束大模型的幻觉、管理不断增加的代码熵，并在不可预测的 AI 输出中建立工程上的确定性。

本节探讨了在 AI 辅助下新兴的编程范式与工作流，重点关注如何利用 AI 提升开发效率与代码质量：

- [Agent First：软件工程的下一个范式转移](Agent_First.md) - 梳理编程范式的演变历史，探讨 Agent First 的核心理念与实战指南。
- [驾驭工程](Harness_Engineering.md) - 深度解析如何构建驾驭系统，提升 AI 编程助手的可控性与效能。
- [OpenSpec 实战指南](https://github.com/ForceInjection/OpenSpec-practise/blob/main/README.md) - Spec 驱动开发 (Spec-Driven Development) 的工程实践，演示了“意图 -> Spec -> AI -> 代码 & 验证”的新一代开发工作流。

---

## 2. Java AI 开发

本节主要介绍在 Java 生态系统中开发大语言模型应用的技术栈。Spring AI 作为官方主推的 AI 工程框架，极大地降低了企业级 Java 应用接入 AI 能力的门槛。

- [Java AI 开发指南](java_ai/README.md) - Java 生态系统中的 AI 开发技术总览。
- [使用 Spring AI 构建高效 LLM 代理](java_ai/spring_ai_cn.md) - 基于 Spring AI 框架的企业级 AI 应用开发实践。

---

## 3. LangGraph 开发

LangGraph 是一个用于构建有状态、多智能体应用程序的库。它通过引入图计算模型，完美解决了传统 LLM 应用在循环逻辑和状态持久化方面的瓶颈，特别适合构建需要多轮推理和自我反思的复杂 Agent 工作流。

- [LangGraph 框架学习资源](langgraph/README.md) - LangGraph 框架的学习资源与实践案例总览。
- [LangGraph 简介](langgraph/langgraph_intro.md) - LangGraph 的核心概念与入门指南。
- [AI 客服系统实战](langgraph/aics.ipynb) - 基于 LangGraph 构建的 AI 客服系统 Notebook 实战。

---

## 4. AI 工作流与编排

除硬编码框架外，无代码或低代码（No-Code/Low-Code）工具也是 AI 应用落地的重要途径，它们能大幅提升编排效率。

- [Coze 部署和配置手册](../06_llm_theory_and_fundamentals/workflow/coze_deployment_and_configuration_guide.md) - Coze 平台的私有化部署与配置指南。
- [n8n 多智能体编排指南](../06_llm_theory_and_fundamentals/workflow/n8n_multi_agent_guide.md) - 基于 n8n 构建 Multi-Agent 系统。
- [开源大模型应用编排平台对比](../06_llm_theory_and_fundamentals/workflow/open_source_llm_orchestration_platforms_comparison.md) - 主流应用编排平台的深度横评。
