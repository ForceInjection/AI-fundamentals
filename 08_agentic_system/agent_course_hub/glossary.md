# Agent 术语速查

> 按「遇到不懂的词 → 查这里」设计的术语表。每个术语给一句话解释 + 深读入口：本仓库文章优先，没有深度文的指向官方文档或 Hub 清单中的外部课程。链接文字一律采用文章标题。
>
> 与其余两件套的分工：按阶段学 → [Agent 学习路线图：从小白到构建者](./learning-roadmap.md)；找课程与项目 → [Agent 学习课程 Hub](./agent-course-hub.md)。本文只做「词 → 解释 → 指针」，不重复路线图的阶段逻辑。

---

## 一、Agent 基础概念

| 术语                | 一句话解释                                                      | 深读入口                                                                                             |
| ------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Agent（智能体）     | 以 LLM 为大脑、能自主决策并调用工具完成任务的程序               | [hello-agents](https://github.com/datawhalechina/hello-agents) 第 1–3 章（外部课程）                 |
| Workflow vs Agent   | Anthropic 的核心区分：workflow 是预设路径，agent 是模型自主决策 | [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)（官方） |
| 多智能体系统（MAS） | 多个 Agent 通信协作，解决单个 Agent 搞不定的任务（如 BDI 架构） | [多智能体AI系统基础：理论与框架](../multi_agent/docs/part1_multi_agent_ai_fundamentals.md)           |
| Agent First         | 软件设计的范式转移：软件不只服务人类，更要先服务 AI Agent       | [Agent First：软件工程的下一个范式转移](../../98_llm_programming/Agent_First.md)                     |
| 12-Factor Agents    | 借鉴 12-Factor App 的 12 条构建可靠 LLM 应用的原则              | [12-Factor Agents - 构建可靠 LLM 应用的原则](../concepts/12-factor-agents-intro.md)                  |

## 二、LLM 前置基础

| 术语                             | 一句话解释                                              | 深读入口                                                                                            |
| -------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 提示词工程（Prompt Engineering） | 设计并迭代 system/user 消息，让模型输出更可控           | [dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)（外部课程） |
| Temperature（温度）              | 采样参数：越高输出越随机有创意，越低越确定保守          | [DeepSeek API 文档：参数设置](https://api-docs.deepseek.com/quick_start/parameter_settings)（官方） |
| 上下文窗口（Context Window）     | 模型单次能「看到」的最大 token 量，Agent 的工作记忆上限 | [上下文工程原理](../context/context-engineering-principles.md)                                      |
| 工具调用（Tool Calling）         | 让 LLM 输出结构化指令来调用外部函数或 API               | [12-Factor Agents - 构建可靠 LLM 应用的原则](../concepts/12-factor-agents-intro.md)（要素 1）       |
| 结构化输出（Structured Outputs） | 约束模型按 JSON Schema 输出，工具调用的本质             | [12-Factor Agents - 构建可靠 LLM 应用的原则](../concepts/12-factor-agents-intro.md)（要素 4）       |

## 三、Agent 设计模式

| 术语           | 一句话解释                                            | 深读入口                                                                                          |
| -------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| ReAct          | 「推理—行动」循环：边想边调用工具，Agent 最经典的范式 | [Cursor IDE ReAct Agent 技术架构深度分析](../agent_design/docs/react-agent.md)                    |
| Plan-and-Solve | 先让模型做完整计划，再按计划逐步执行                  | [All Agentic Architectures 深入详解](../agent_design/docs/all-agentic-architectures-deep-dive.md) |
| Reflection     | 生成结果后自我反思批评、迭代改进                      | [All Agentic Architectures 深入详解](../agent_design/docs/all-agentic-architectures-deep-dive.md) |
| Handoff        | 把对话与上下文交接给另一个更合适的 Agent              | [openai/openai-agents-python](https://github.com/openai/openai-agents-python)（官方 SDK）         |

## 四、协议与技能

| 术语                          | 一句话解释                                          | 深读入口                                                                                                 |
| ----------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| MCP（Model Context Protocol） | Agent 连接外部工具与数据的开放标准协议              | [深度解析 MCP 与 AI 工具化的未来](../mcp/docs/01_deep_dive_into_mcp_and_the_future_of_ai_tooling.md)     |
| A2A（Agent2Agent）            | Agent 之间通信与互操作的开放协议（Google 发起）     | [A2A 项目官网](https://a2aproject.org)（官方）                                                           |
| Agent Skills                  | 把能力、指令与资源打包成可复用技能，2026 年事实标准 | [给 Claude 写本“标准操作手册”：Agent Skills 实战与深度解析](../agent_skills/docs/claude_skills_guide.md) |

## 五、上下文与记忆

| 术语                              | 一句话解释                                             | 深读入口                                                                                                                  |
| --------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| 上下文工程（Context Engineering） | 动态组装、压缩、缓存上下文，让有限窗口装下最该装的信息 | [上下文工程原理](../context/context-engineering-principles.md)（入门先读[上下文工程原理简介](../context/context-engineering-intro.md)） |
| 上下文压缩                        | 长任务中主动丢弃冗余信息、保留核心逻辑                 | [Claude Code 上下文压缩机制深度解析](../context/claude-code-context-compression.md)                                       |
| 记忆系统（Memory）                | 让 Agent 跨会话记住用户与事实，分短期/长期与分层存储   | [AI 智能体记忆系统](../memory/README.md)                                                                                  |

## 六、工程与基础设施

| 术语                          | 一句话解释                                                | 深读入口                                                                                            |
| ----------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Harness（驾驭工程）           | 围绕 LLM 的运行时/工作环境，让 Agent 在真实环境自主干活   | [驾驭工程：为什么你的 AI 编程助手总在失控？](../../98_llm_programming/Harness_Engineering.md)       |
| Agent Sandbox                 | 给 Agent 提供安全隔离执行环境的沙箱，隔离从硬件级到策略级 | [Agent Sandbox 的演进与设计范式](../agent_infra/docs/agent-sandbox-design.md)                       |
| Agent Infra（智能体基础设施） | 工具、数据、编排三层底座，让 Agent 可靠运行、规模化托管   | [AI Agent 基础设施——三个决定性层次：工具、数据、编排](../agent_infra/docs/ai-agent-infra-stack.md)  |
| SDD（Spec 驱动开发）          | 先写 Spec 再让 AI 实现与验证的工程范式                    | [ForceInjection/OpenSpec-practise](https://github.com/ForceInjection/OpenSpec-practise)（开源项目） |

## 七、延伸相关

| 术语                    | 一句话解释                                                   | 深读入口                                                                                         |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| RAG（检索增强生成）     | 先从知识库检索相关片段再生成回答，给模型补知识               | [RAG 与工具生态](../../07_rag_and_tools/README.md)                                               |
| Data Agent              | 用自然语言查数据、做分析、出报表的 Agent，企业最易变现的方向 | [数据智能体：是重塑生产力的“自动驾驶”，还是换壳的平庸炒作？](../data_agent/data-agent-survey.md) |
| 世界模型（World Model） | Agent 对环境的内部预测模型，从任务助手走向自主智能的关键     | [世界模型简介：智能体理解世界的内部引擎](../concepts/world-model-introduction.md)                |
