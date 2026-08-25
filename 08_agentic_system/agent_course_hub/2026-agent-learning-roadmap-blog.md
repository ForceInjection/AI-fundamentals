# 2026 年 AI Agent 学习路线图：先跑通，再理解，最后造轮子

> 给零基础到初学者的 Agent 学习路径：七个阶段、每阶段一份课程与项目清单、一个「验收标志」，外加两条可选路径。总耗时 4–6 个月。文中所有 star 数于 2026-08-24 经 GitHub API 实时核实；遇到不懂的术语，文末附术语速查。

---

## 一、为什么是现在，为什么是你

大模型本身只会生成 token：它不会自己打开浏览器，不会调用 API，也不记得昨天的对话。让它端到端干完一件复杂任务的，是围绕模型搭起来的 Agent 系统：**Agent = LLM + 工具 + 循环 + 记忆**。动手这件事，得靠外面这套系统。

2026 年，Agent 明显在从概念走向工程化，有三个信号：

协议标准化。MCP（Model Context Protocol）已成为 Agent 连接外部工具的事实标准，主流模型、IDE 与框架全部接入；Agent Skills 成为 2026 年最流行的可复用能力封装方式，"给 Agent 写手册"这件事第一次有了统一格式。

Harness 爆发。让 Agent 自主干活的"驾驭系统"成为主战场：Cursor、Claude Code 已是程序员日常，DeepSeek 官方 deepseek-harness 上线 11 天拿下 19 万 star。

岗位变化。企业对 AI 工程师的要求正在从"会调 API"转向"能建系统"。状态管理、工具编排、记忆系统、错误恢复，这些工程能力比模型本身稀缺得多。

最省心的一点：**这条路不需要你会训练模型**。模型层的军备竞赛是少数实验室的事，Agent 层只要会写代码就能做。下面是我们在整理了大量课程与项目之后给出的路线图。

---

## 二、学习的三个陷阱

想学和学得会之间，隔着三个坑。这三个坑不怪学习者，怪这个领域迭代太快。整理课程清单的时候，我们在这几个坑里都栽过：

陷阱一：选择瘫痪。课程、框架、Harness 日更，star 数一个比一个高。收藏了几十个仓库、每个都看了一页、最后一个都没学完，这是这一行最常见的学习方式。star 数代表流行度，不代表适合你当前阶段。一个 200K star 的 Harness 对刚入门的你，价值还不如一本 33K star 的中文入门教程。

陷阱二：demo 层天花板。大多数教程教你到"调 API 输出一句话"为止。但从 demo 到生产，中间隔着四道关：状态管理、记忆系统、错误恢复、上下文工程。教程不会告诉你，这四道关才是 Agent 工程师真正的工作内容，也是面试真正会问的内容。

陷阱三：框架焦虑。今天 LangGraph，明天 crewAI，后天 smolagents。追新而不懂原理，学到的永远是库的用法。换一个框架，之前学的作废大半。

绕开这三个坑的办法，标题已经给了：**先跑通，再理解，最后造轮子**。下面每个阶段都配一个「验收标志」，达成即前进；没达成，就停下来把这一阶段补完。

---

## 三、七阶段路线图

![2026 年 AI Agent 学习路线图全景](2026-agent-learning-roadmap.png)

| 阶段 | 主题               | 周期   | 验收标志                                |
| ---- | ------------------ | ------ | --------------------------------------- |
| 0    | 前置基础           | 1–2 周 | 能写多轮对话 CLI 脚本                   |
| 1    | Agent 是什么       | 1 周   | 讲清 Agent 与 Chatbot 的区别            |
| 2    | 手搓核心范式       | 2–3 周 | 不靠框架手写 ReAct Agent                |
| 3    | 框架实战           | 2–3 周 | LangGraph 做出带状态可中断恢复的应用    |
| 4    | 协议、记忆与上下文 | 2–3 周 | 3 个 MCP 工具 + 跨会话记忆 + 自写 Skill |
| 5    | 案例与多智能体     | 2–4 周 | 独立完成一个完整项目                    |
| 6    | Harness 与工程化   | 持续   | 日常开发 30% 由 Harness 完成            |

这条路径的安排逻辑很简单：每一阶段的目标，都建立在上一步的产出之上。下一步永远是对上一步的自然延伸，不用另起炉灶。

### 阶段 0：前置基础（1–2 周，可跳过部分）

后面所有阶段都默认"你能跑通一个 LLM API"，所以起点从这里开始：Python 基础（函数、类、async 即可）、一个主流 LLM 的 API 调用（OpenAI 兼容格式通用）、prompt 基础（system/user 消息、temperature 等参数的含义）。

- 参考：[dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)（77.7K），挑前几章看即可；想补 LLM 原理再看 [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm)（33.2K，中文）。

**验收标志**：写一个能多轮对话的 CLI 聊天脚本。

### 阶段 1：Agent 是什么（1 周）

能调 API 之后，先别急着选框架。这一周只回答一个问题：Agent 和 Chatbot 到底差在哪。心智模型只有一句话：**Agent = LLM + 工具 + 循环**。Chatbot 是"一问一答"；Agent 是"自主规划步骤、调用工具、根据结果再决策"的循环。

- 参考：[microsoft/ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) 第 1–4 课（概念、框架概览、设计模式、工具调用）；中文入门可选 [hello-agents](https://github.com/datawhalechina/hello-agents) 第 1–3 章。
- 必读短文：Anthropic《[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)》。其中 **workflow（预设路径）vs agent（自主决策）** 的区分，是全场最重要的一个概念，值得反复读。
- 深度阅读：[12-Factor Agents - 构建可靠 LLM 应用的原则](https://github.com/ForceInjection/AI-fundamentals/blob/main/08_agentic_system/concepts/12-factor-agents-intro.md)，从云原生十二要素借来的工程质量标尺，越早建立越好。

**验收标志**：能向别人讲清楚「Agent 与 Chatbot 的区别是什么」。

### 阶段 2：手搓核心范式（2–3 周，最重要）

心智模型建立之后，下一步是亲手把"循环"实现一遍：用原生 API 手写三个经典范式：**ReAct**（推理-行动循环）、**Plan-and-Solve**（先规划后执行）、**Reflection**（自我反思迭代）。

这一步是整条路线图里最关键的一步：**它决定了你以后是"库的使用者"还是"系统的构建者"**。很多人想跳过它直接上框架，我的建议是别。手写过一次 ReAct，你才知道 LangGraph 的节点和边在替你做什么；没手写过，框架永远是个黑盒。面试能讲原理、看得懂框架源码、有能力自研系统，都从这一步开始。

- 参考：[hello-agents](https://github.com/datawhalechina/hello-agents) 第 4–7 章，专门带你把三个范式用原生 API 写一遍，并自研一个迷你框架 HelloAgents。
- 深度阅读：[Cursor IDE ReAct Agent 技术架构深度分析](https://github.com/ForceInjection/AI-fundamentals/blob/main/08_agentic_system/agent_design/docs/react-agent.md)，手搓完再看：同一范式在真实产品 Cursor 中的工程实现（分层架构、工具调用、上下文管理与性能优化）。

**验收标志**：不依赖任何框架，手写出一个能查天气 + 算数 + 失败重试的 ReAct Agent。

### 阶段 3：框架实战（2–3 周）

手搓过裸循环之后，框架的价值才真正显形：它替你解决状态、记忆、并行、错误恢复，而这些正是你上一阶段头疼过的问题。带着问题学框架，比一上来就学框架快得多，也扎实得多。

选型建议是"先精通一个，再触类旁通"：快速出 demo 用 [crewAI](https://github.com/crewAIInc/crewAI)（角色扮演，最易上手）或 [smolagents](https://github.com/huggingface/smolagents)（代码极简）；企业级主攻 [LangGraph](https://github.com/langchain-ai/langgraph)（有状态图，生态最全，市场岗位需求最大）；对照理解 [openai-agents-python](https://github.com/openai/openai-agents-python)（handoff 模式）与 [google/adk-python](https://github.com/google/adk-python)。

- 参考：[huggingface/agents-course](https://github.com/huggingface/agents-course)，一门口课横跨 smolagents/LlamaIndex/LangGraph，有结业证书；[NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents)（24.0K），45+ 个可运行 Notebook 跟练。
- 深度阅读：[All Agentic Architectures 深入详解](https://github.com/ForceInjection/AI-fundamentals/blob/main/08_agentic_system/agent_design/docs/all-agentic-architectures-deep-dive.md)，17 种可运行架构全景，从 Reflection 到 Tree of Thoughts，选型对照表。

**验收标志**：用 LangGraph 做出一个带状态、能中断恢复的多轮 Agent 应用。

### 阶段 4：协议、记忆与上下文工程（2–3 周）

能跑的单体 Agent 还缺两样东西："感官"与"记忆"。生产级和 demo 的差距就在这里。生产级需要：标准化的工具接口（MCP）、可复用的能力单元（Agent Skills）、跨会话的记忆系统，以及决定"哪些内容装进上下文窗口"的上下文工程。窗口是 Agent 的工作记忆，也是成本和性能的最大变量。

- 参考：[modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)（89.8K），先跑通一个 MCP server；[anthropics/skills](https://github.com/anthropics/skills)（171.2K），读懂 SKILL.md 格式并写一个自己的技能；[anthropics/courses](https://github.com/anthropics/courses)（22.7K），Agent Skills 与上下文工程的官方实操课。
- 深度阅读：[深度解析 MCP 与 AI 工具化的未来](https://github.com/ForceInjection/AI-fundamentals/blob/main/08_agentic_system/mcp/docs/01_deep_dive_into_mcp_and_the_future_of_ai_tooling.md)，协议原理与实战；[给 Claude 写本"标准操作手册"：Agent Skills 实战与深度解析](https://github.com/ForceInjection/AI-fundamentals/blob/main/08_agentic_system/agent_skills/docs/claude_skills_guide.md)，技能定义规范；[上下文工程原理](https://github.com/ForceInjection/AI-fundamentals/blob/main/08_agentic_system/context/context-engineering-principles.md)，动态组装、压缩与检索。
- 找灵感：[awesome-skills](https://github.com/ForceInjection/awesome-skills)，优秀 Agent Skill 实例合集。

**验收标志**：给你的 Agent 接上 3 个 MCP 工具 + 跨会话记忆，再写一个自己的 Agent Skill。

### 阶段 5：案例与多智能体项目（2–4 周）

单 Agent 的全栈能力齐了之后，下一步是把它用起来：用完整项目建立产品感。这个阶段练的是判断力：把需求翻译成 Agent 系统，哪些环节让 Agent 自主，哪些必须预设路径。

- 找灵感：[Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)（133.7K）；行业案例：[ashishpatel26/500-AI-Agents-Projects](https://github.com/ashishpatel26/500-AI-Agents-Projects)（36.9K）；多智能体理论：[geekan/MetaGPT](https://github.com/geekan/MetaGPT)（70.0K）与 [camel-ai/camel](https://github.com/camel-ai/camel)（17.6K）。
- 深度阅读：[数据智能体：是重塑生产力的"自动驾驶"，还是换壳的平庸炒作？](https://github.com/ForceInjection/AI-fundamentals/blob/main/08_agentic_system/data_agent/data-agent-survey.md)，Data Agent 是当下最容易变现的方向之一，能力分级与炒作风险都讲透了。

**验收标志**：复刻或自创一个完整项目（智能客服、Data Agent、Deep Research 三选一）。

### 阶段 6：Harness 与工程化（持续）

会造 Agent 之后，最后一课是 Harness Engineering（驾驭工程）：让 Agent 在真实环境中自主干活，并把它变成可维护的工程系统。本质上就是像管员工一样管 Agent：给工具、定边界、加监督、留容错。这一层正在成为软件工程的新范式。

按从易到难的顺序走：先用 [Aider-AI/aider](https://github.com/Aider-AI/aider)（48.4K）当用户，体验终端结对编程；顺手试试 [anomalyco/opencode](https://github.com/anomalyco/opencode)（201.2K），目前最流行的开源终端编码 Agent，和 Claude Code 同类；再上手 [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)（190.0K），DeepSeek 官方 Harness，「一切皆插件」架构，`npx @deepseek-ai/dsh web` 一条命令跑起 Web UI（开发者预览，更新快）；进阶看 [cline/cline](https://github.com/cline/cline)（66.7K）与 [All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands)（84.9K），浏览器 + 终端 + 编辑器全驾驭的通用 Harness 标本，值得读源码；编排层看 [microsoft/agent-framework](https://github.com/microsoft/agent-framework)（13.1K，生产级编排框架）。

- 深度阅读：[驾驭工程：为什么你的 AI 编程助手总在失控？](https://github.com/ForceInjection/AI-fundamentals/blob/main/98_llm_programming/Harness_Engineering.md)，如何构建驾驭系统的深度解析，本阶段必读；[Agent First：软件工程的下一个范式转移](https://github.com/ForceInjection/AI-fundamentals/blob/main/98_llm_programming/Agent_First.md)，范式认知。

**验收标志**：日常开发由 Harness 类工具完成 30% 以上，且能说清它们的设计取舍。

---

## 四、两条路径与路线图边界

七阶段不一定要按顺序走，按你的处境选：

|      | 课程派（稳扎稳打）       | 实战派（以用带学）                         |
| ---- | ------------------------ | ------------------------------------------ |
| 主线 | 阶段 0 → 6 顺序推进      | 阶段 0 → 1 → 3 → 5 先出成果，再回头补 2、4 |
| 适合 | 学生、转行者、想打牢底子 | 在职工程师、急着落地                       |
| 时间 | 4–6 个月                 | 2–3 个月见效，原理后补                     |

这条路线图没覆盖的事，也写清楚：

1. 不覆盖的领域：多模态 Agent、Agent 评测与 benchmark、模型微调与强化学习，这些是"下一个层次"的课题。本路线图聚焦一件事：用工程能力把现有模型变成系统。
2. star 数的时效性：文中所有 star 数于 2026-08-24 核实，而生态半年一换。star 数会过时，验收标志不会。**"学完"比"追新"重要**，不必逐条打卡。
3. 深度 vs 广度：阶段 3 的选型建议是"先精通一个"，不是"都学一遍"。框架会换，范式不换。把范式吃透，剩下的只是换语法。

---

## 五、结语

最后说句实在的：Agent 的门槛不在模型，在系统。模型的能力是模型厂商给的，把模型变成系统的能力是你自己的。七阶段走完，你手里会有一个能自己干活的系统。这种东西，2026 年的招聘市场正缺。

> 本文是 [AI-fundamentals 仓库](https://github.com/ForceInjection/AI-fundamentals) Agent 学习课程 Hub 的对外发布版。仓库内还有：完整版 [Agent 学习路线图：先跑通，再理解，最后造轮子](https://github.com/ForceInjection/AI-fundamentals/blob/main/08_agentic_system/agent_course_hub/learning-roadmap.md)（含更多深度文章与延伸阅读）、[Agent 学习课程 Hub](https://github.com/ForceInjection/AI-fundamentals/blob/main/08_agentic_system/agent_course_hub/agent-course-hub.md)（课程与项目清单全文），以及 [Agent 术语速查](https://github.com/ForceInjection/AI-fundamentals/blob/main/08_agentic_system/agent_course_hub/glossary.md)（27 个高频术语按类分组），学习途中遇到不懂的术语，随时可查。
