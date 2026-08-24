# Agent 学习路线图：从小白到构建者

> 给零基础到初学者的 Agent 学习路径。七个阶段循序渐进，每阶段给出：学什么、参考课程（[Hub 清单](./agent-course-hub.md)）、参考项目、以及**本仓库的关联深度文章**（理论拔高时用）。
>
> 原则：**先跑通，再理解，最后造轮子**。每个阶段都有一个「验收标志」，完成即可进入下一阶段。

---

## 阶段 0：前置基础（1–2 周，可跳过部分）

**目标**：能调通 LLM API，会用 Python 写脚本。

- Python 基础（能写函数、类、async 即可）
- 一个主流 LLM 的 API 调用（OpenAI 兼容格式通用）
- Prompt 基础：system/user 消息、temperature 等参数的含义

**参考内容**：

- [dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)（77.7K）——挑前几章看
- [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm)（33.2K）——想补 LLM 原理再看

**验收标志**：写一个能多轮对话的 CLI 聊天脚本。

---

## 阶段 1：Agent 是什么（1 周）

**目标**：建立 Agent 心智模型——「LLM + 工具 + 循环」，理解 Agent 与普通 Chatbot 的本质区别。

**参考课程**：

- [microsoft/ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) 第 1–4 课（概念、框架概览、设计模式、工具调用）
- 或 [hello-agents](https://github.com/datawhalechina/hello-agents) 第 1–3 章（中文，定义/类型/发展史）
- 必读短文：Anthropic《Building Effective Agents》——**workflow（预设路径）vs agent（自主决策）** 的区分是全场最重要的概念

**本仓库关联**：

- [ReAct Agent 模式详解](../agent_design/docs/react-agent.md)——第一个要懂的 Agent 范式
- [12-Factor Agents](../concepts/12-factor-agents-intro.md)——Agent 工程质量标准的早期认知
- [微软 AI Agents for Beginners 课程笔记](../../10_ai_related_course/AI%20Agents%20for%20Beginners%20课程之%20AI%20Agent及使用场景简介.md)——中文笔记，可替代英文原文

**验收标志**：能向别人讲清楚「Agent 与 Chatbot 的区别是什么」。

---

## 阶段 2：手搓核心范式（2–3 周，最重要）

**目标**：用原生 API 亲手实现三个经典范式，理解 Agent 的"灵魂"在哪——**这一步决定了你是库的使用者还是系统的构建者**。

- ReAct（推理-行动循环）
- Plan-and-Solve（先规划后执行）
- Reflection（自我反思迭代）

**参考课程**：

- [hello-agents](https://github.com/datawhalechina/hello-agents) 第 4–7 章——专门带你把这三个范式用原生 API 手写一遍，并自研一个迷你框架 HelloAgents

**验收标志**：不依赖任何框架，手写出一个能查天气 + 算数 + 失败重试的 ReAct Agent。

---

## 阶段 3：框架实战（2–3 周）

**目标**：用主流框架快速构建真实应用，理解框架帮你解决了什么（状态、记忆、并行、错误恢复）。

**选型建议**（先精通一个，再触类旁通）：

- 快速出 demo：`crewAI`（角色扮演，最易上手）或 `smolagents`（代码极简）
- 企业级主攻：`LangGraph`（有状态图，生态最全，市场岗位需求最大）
- 对照理解：`openai-agents-python`（handoff 模式）、`google/adk-python`

**参考课程**：

- [huggingface/agents-course](https://github.com/huggingface/agents-course) —— 一门口课横跨 smolagents/LlamaIndex/LangGraph，有结业证书
- [NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents)（24.0K）——45+ 可运行 Notebook 跟练

**本仓库关联**：

- [企业级多智能体 AI 系统构建实战](../multi_agent/docs/part2_enterprise_multi_agent_system_implementation.md)——LangGraph 生产级落地
- [All Agentic Architectures 深入详解](../agent_design/docs/all-agentic-architectures-deep-dive.md)——17 种架构全景

**验收标志**：用 LangGraph 做出一个带状态、能中断恢复的多轮 Agent 应用。

---

## 阶段 4：协议、记忆与上下文工程（2–3 周）

**目标**：补齐 Agent 的"感官"与"记忆"——这是生产级 Agent 与 demo 的分水岭。

- **MCP**：Agent 连接外部工具的标准协议
- **Agent Skills**：可复用能力单元（2026 年事实标准）
- **A2A**：Agent 之间的通信协议
- **记忆系统**：短期/长期记忆、会话状态管理
- **上下文工程**：动态组装、压缩、缓存

**参考项目**：

- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)（89.8K）——跑通一个 MCP server
- [anthropics/skills](https://github.com/anthropics/skills)（171.2K）——读懂 SKILL.md 格式并写一个自己的技能
- [anthropics/courses](https://github.com/anthropics/courses)（22.7K）——Agent Skills 与上下文工程官方实操课

**本仓库关联**（本仓库在这一层的深度文章最全，重点利用）：

- [上下文工程原理](../context/context-engineering-principles.md)——动态组装、压缩与检索（[快速入门](../context/context-engineering-intro.md)先看）
- [记忆系统架构总览](../memory/README.md)——MemoryOS/Mem0/Claude Code 记忆机制解析
- [Model Context Protocol (MCP)](../mcp/docs/01_deep_dive_into_mcp_and_the_future_of_ai_tooling.md)——协议原理与实战
- [Claude Skills 开发指南](../agent_skills/docs/claude_skills_guide.md)——技能定义规范
- [awesome-skills](https://github.com/ForceInjection/awesome-skills)（[在线版](https://forceinjection.github.io/awesome-skills/)）——优秀 Agent Skill 实例合集；[mmx-cli](https://github.com/MiniMax-AI/cli)——遵循 agentskills.io 标准的官方技能实例

> 延伸阅读（按兴趣深挖）：[Claude Code 上下文压缩](../context/claude-code-context-compression.md) · [MemoryOS 架构](../memory/research/systems/memoryos-architecture-guide.md) · [Claude Code 记忆机制解析](../memory/research/case-studies/claude-code-memory-analysis.md)

**验收标志**：给你的 Agent 接上 3 个 MCP 工具 + 跨会话记忆，再写一个自己的 Agent Skill。

---

## 阶段 5：案例与多智能体项目（2–4 周）

**目标**：从"会写"到"会用"，通过案例库和完整项目建立产品感。

- 找灵感：[Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)（133.7K）
- 行业案例：[ashishpatel26/500-AI-Agents-Projects](https://github.com/ashishpatel26/500-AI-Agents-Projects)（36.9K）
- 多智能体理论：[geekan/MetaGPT](https://github.com/geekan/MetaGPT)（70.0K）、[camel-ai/camel](https://github.com/camel-ai/camel)（17.6K）

**本仓库关联**：

- [多智能体 AI 系统基础：理论与框架](../multi_agent/docs/part1_multi_agent_ai_fundamentals.md)——BDI 架构与协作机制
- [多智能体 AI 系统培训材料](../../10_ai_related_course/multi_agent_system/multi_agent_training/README.md)——五天结构化课程，配套理论+实战两篇文档
- [数据智能体综述](../data_agent/data-agent-survey.md)——Data Agent 是当下最容易变现的方向（配套 [PRD](../data_agent/enterprise-data-agent-prd.md) 与 [MVP 规划](../data_agent/data-agent-skill-mvp.md)）

> 延伸阅读（按兴趣深挖）：[写作 Agent 设计](../agent_design/docs/writing-agentic-agent.md) · [TradingAgents-CN](../agent_design/docs/trading-agents-cn.md) · [Kagent：K8s 运维智能体](../agent_infra/docs/deep-dive-kagent-k8s-ops-agent.md) · [Agent Workflow 综述](../papers/agent-workflow-survey.md)

**验收标志**：复刻或自创一个完整项目（智能客服、Data Agent、Deep Research 三选一）。

---

## 阶段 6：Harness 与工程化（持续）

**目标**：理解"驾驭工程（Harness Engineering）"——让 Agent 在真实环境中自主干活，并把它变成可维护的工程系统。

**参考项目**（从易到难）：

- [Aider-AI/aider](https://github.com/Aider-AI/aider)（48.4K）——终端结对编程，先当用户
- [cline/cline](https://github.com/cline/cline)（66.7K）——IDE 内自主编码 Agent
- [All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands)（84.9K）——读源码：浏览器+终端+编辑器全驾驭的通用 Harness 标本
- [microsoft/agent-framework](https://github.com/microsoft/agent-framework)（13.1K）——生产级编排框架（AutoGen 继任者）

**本仓库关联**（本仓库 Harness/Infra 文章最多最硬核，重点利用）：

- [驾驭工程](../../98_llm_programming/Harness_Engineering.md)——如何构建驾驭系统的深度解析（必读）
- [Agent First：软件工程的下一个范式转移](../../98_llm_programming/Agent_First.md)——范式认知
- [OpenHarness 深入浅出](../agent_infra/docs/openharness-deep-dive.md)——开源智能体基础设施全景
- [Agent Sandbox 的演进与设计范式](../agent_infra/docs/agent-sandbox-design.md)——沙箱隔离从硬件级到策略优先
- [确定性工程驯服不确定的 Agent：OpenCodeReview 三阶段架构](../agent_design/docs/open-code-review-deep-dive.md)——以确定性约束超越自由 Agent
- [DeepSeek-TUI 实战](../agent_infra/docs/deepseek-tui-in-practice.md)——榨干 DeepSeek V4 长上下文红利的命令行 Agent 实战
- [OpenSpec 实战指南](https://github.com/ForceInjection/OpenSpec-practise)——Spec 驱动开发：意图 → Spec → AI → 代码 & 验证

> 延伸阅读（按兴趣深挖）：[基础设施技术栈](../agent_infra/docs/ai-agent-infra-stack.md) · [Claude Code Sandbox 隔离机制](../agent_infra/docs/claude-code-sandbox.md) · [扩展托管智能体](../agent_infra/docs/scaling-managed-agents.md) · [基础设施的崛起](../agent_infra/docs/the-rise-of-ai-agent-infrastructure.md)

**验收标志**：日常开发由 Harness 类工具完成 30% 以上，且能说清它们的设计取舍。

---

## 两条可选路径

|      | 课程派（稳扎稳打）       | 实战派（以用带学）                         |
| ---- | ------------------------ | ------------------------------------------ |
| 主线 | 阶段 0 → 6 顺序推进      | 阶段 0 → 1 → 3 → 5 先出成果，再回头补 2、4 |
| 适合 | 学生、转行者、想打牢底子 | 在职工程师、急着落地                       |
| 时间 | 4–6 个月                 | 2–3 个月见效，原理后补                     |

## 附：本仓库的 Agent 内容地图

本路线图大量引用了仓库 `08_agentic_system/` 的深度文章——它们正好覆盖课程学不到的部分：**源码级机制解析与生产级设计取舍**。完整导航见 [AI Agent 开发与实践](../README.md)。
