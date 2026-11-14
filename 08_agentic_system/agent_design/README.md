# 智能体设计（Agent Design）

智能体架构的源码级深拆与设计模式分析：从经典 ReAct 循环在真实产品中的工程落地，到确定性约束驯服非确定性 Agent 的开源系统解剖，再到 API 产品形态的服务端工具协议拆解。

## 深拆文章（deep-dives/）

| 文章 | 一句话 |
| --- | --- |
| [Cursor IDE ReAct Agent 技术架构深度分析](./deep-dives/01_react-agent.md) | 以 Cursor IDE 为标本解析 ReAct 循环的工程落地：分层架构、工具调用、上下文管理 |
| [确定性工程驯服不确定的 Agent：OpenCodeReview 三阶段架构深度解析](./deep-dives/06_open-code-review-deep-dive.md) | 阿里开源项目源码与论文分析：规则引导调度、有界工具审查循环、信息边界反思如何以确定性约束超越自由 Agent |
| [技术博客撰写 Agentic RAG Agent 系统设计](./deep-dives/03_writing-agentic-agent.md) | 复杂长文本创作领域的智能体工作流架构设计与实践优化 |
| [支持多轮对话指代消解的 ChatBot 系统：架构设计与实现详解](./deep-dives/04_coreference-resolution-dialogue-system.md) | 指代消解机制设计与实现建议；§四 按 v2 demo 展开：LLM 结构化输出 + 状态全量重建 |
| [TradingAgents-CN 多智能体设计与交互分析](./deep-dives/05_trading-agents-cn.md) | 交易领域的智能体设计与交互分析 |
| [All Agentic Architectures 深入详解](./deep-dives/02_all-agentic-architectures-deep-dive.md) | 17 种可运行的 LangChain + LangGraph 智能体架构：Reflection、ReAct、Planning、Blackboard、Tree of Thoughts 等 |
| [DeepSeek 把整个 App 开放成了 API：全家桶应用 API 深拆](./deep-dives/07_deepseek-app-api-deep-dive.md) | 服务端工具协议、signature 多轮记忆、被托管参数与工具计费口径，含独立实测与三张机制图 |

## 配套代码与课件

- [coref-dialogue-system/](./coref-dialogue-system/README.md) — 指代消解对话系统 v2（LLM 结构化输出 + 状态全量重建，规则路径对照）；2026-11-14 实测 pytest 27 条全过、DeepSeek API 实跑 5 轮消解全对
- [AI Agent 设计模式](./deep-dives/ai-agent-design-patterns.pptx) — 企业级智能体架构设计模式课件
