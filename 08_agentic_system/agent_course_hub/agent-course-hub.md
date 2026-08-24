# Agent 学习课程 Hub

> 面向想系统学习 AI Agent 的开发者，按「知名度 × star 数 × 开源」三个标准筛选的课程与项目清单。所有 star 数均于 2026-08-24 通过 GitHub API 实时核实。
>
> 配套阅读：[小白学习路线图](./learning-roadmap.md) —— 不知道怎么开始，先看路线图；已经上路，用本清单按类补齐弹药。

## 筛选标准

1. **知名度**：官方机构出品（微软/Anthropic/OpenAI/Google/Hugging Face/Datawhale）或社区公认标杆；
2. **star 数**：一般 ≥ 10K（官方出品且内容优质者可放宽）；
3. **开源**：许可证明确、可自由学习（部分课程为 CC 协议，商用注意条款）；
4. **活跃度**：2026 年仍有实质更新（停更项目单独标注）。

---

## 一、系统课程（4 门）

| 课程                                                                                      | ⭐        | License         | 一句话定位                                                                                                                                      |
| ----------------------------------------------------------------------------------------- | --------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| [datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents)             | **74.4K** | CC BY-NC-SA 4.0 | 中文 Agent 领域"黄埔军校"：不依赖现成框架，用原生 API 从零手搓 Agent 灵魂（16 章 5 阶段：基础认知 → 动手构建 → 高级进阶 → 综合案例 → 毕业设计） |
| [microsoft/ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) | **73.1K** | MIT             | 微软官方 15 课：概念 → 设计模式 → 生产部署，含视频 + 可运行代码，50+ 语言含中文，Agent 界的 CS50                                                |
| [huggingface/agents-course](https://github.com/huggingface/agents-course)                 | **31.3K** | Apache-2.0      | Hugging Face 官方：结业证书 + 基准项目，一门口课横跨 smolagents / LlamaIndex / LangGraph 三个框架                                               |
| [anthropics/courses](https://github.com/anthropics/courses)                               | **22.7K** | 未指定          | Anthropic 官方实操课：Agent Skills、上下文工程、可复现工作流等                                                                                  |

> **补充**：[datawhalechina/agent-skills-with-anthropic](https://github.com/datawhalechina/agent-skills-with-anthropic)（1.5K）——吴恩达 DeepLearning.AI《Agent Skills with Anthropic》课程的中文翻译与知识整理，star 不高但课程 IP 知名，适合不想啃英文的读者。
>
> **前置基础**（非 Agent 本体，小白按需补课）：[dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)（77.7K，MIT，提示工程全集）；[datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm)（33.2K，LLM 原理与手搓 Transformer）。

## 二、开源框架（8 个，学 Agent 工程绕不开的代码）

| 框架                                                                          | ⭐        | License    | 特点与适用场景                                                                                                                   |
| ----------------------------------------------------------------------------- | --------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------- |
| [geekan/MetaGPT](https://github.com/geekan/MetaGPT)                           | **70.0K** | MIT        | 多智能体协作与"软件公司 SOP"概念鼻祖，适合理解多 Agent 分工                                                                      |
| [microsoft/autogen](https://github.com/microsoft/autogen)                     | **60.6K** | CC-BY-4.0  | 多 Agent 对话框架；⚠️ 更新停于 2026-04，官方主线已并入 [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) |
| [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)                       | **57.5K** | MIT        | 角色扮演式多 Agent，最易上手，快速出 demo 首选                                                                                   |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)           | **40.3K** | MIT        | 有状态图工作流，企业级生产首选，生态最全                                                                                         |
| [huggingface/smolagents](https://github.com/huggingface/smolagents)           | **29.0K** | Apache-2.0 | 代码优先极简 Agent，几百行代码理解 Agent 本质                                                                                    |
| [openai/openai-agents-python](https://github.com/openai/openai-agents-python) | **28.9K** | MIT        | OpenAI 官方 Agents SDK，handoff 多 Agent 模式清晰                                                                                |
| [google/adk-python](https://github.com/google/adk-python)                     | **21.2K** | Apache-2.0 | Google Agent Development Kit，代码优先，A2A 协议原生                                                                             |
| [camel-ai/camel](https://github.com/camel-ai/camel)                           | **17.6K** | Apache-2.0 | 多智能体研究与角色扮演框架，学术向                                                                                               |

## 三、Harness 与编码 Agent（驾驭工程，6 个）

> 「Harness（驾驭）」指围绕 LLM 构建的运行时/工作环境——不是让你从零写 Agent，而是让 AI 在真实环境（终端、IDE、容器）中自主完成任务。推荐与仓库内 [Harness_Engineering.md](../../98_llm_programming/Harness_Engineering.md)、[Agent_First.md](../../98_llm_programming/Agent_First.md) 对照阅读。

| 项目                                                                | ⭐        | License    | 特点                                                                                                                                     |
| ------------------------------------------------------------------- | --------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| [All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands) | **84.9K** | MIT        | 通用 AI 软件工程 Agent：浏览器 + 终端 + 编辑器全驾驭，可自托管，研究 Agent 运行时的最佳开源标本                                          |
| [cline/cline](https://github.com/cline/cline)                       | **66.7K** | Apache-2.0 | IDE 内的自主编码 Agent（VS Code 插件 + SDK），人机协作 Plan/Act 模式                                                                     |
| [block/goose](https://github.com/block/goose)                       | **53.3K** | Apache-2.0 | Block 出品的本地优先编码 Agent，MCP 原生，可扩展任何工具                                                                                 |
| [Aider-AI/aider](https://github.com/Aider-AI/aider)                 | **48.4K** | Apache-2.0 | 终端里的 AI 结对编程，仓库地图 + 自动 commit，上手最快的命令行 Agent                                                                     |
| [Hmbown/DeepSeek-TUI](https://github.com/Hmbown/DeepSeek-TUI)       | **40.8K** | MIT        | 榨干 DeepSeek 长上下文的命令行编程 Agent，TUI + 审批模式 + MCP 扩展；本仓库有[实战解析](../agent_infra/docs/deepseek-tui-in-practice.md) |
| [opencode-ai/opencode](https://github.com/opencode-ai/opencode)     | **13.7K** | MIT        | 终端 AI 编码 Agent，开放式 Agent 协议；⚠️ 更新停于 2025-09                                                                               |

> 生产级 Agent 编排框架补充：[microsoft/agent-framework](https://github.com/microsoft/agent-framework)（13.1K，MIT）——微软新一代 Agent 框架，AutoGen 的官方继任者，适合框架选型时作为 LangGraph 的对照项。

## 四、案例库与速查（3 个）

| 项目                                                                                            | ⭐         | License    | 用途                                                                 |
| ----------------------------------------------------------------------------------------------- | ---------- | ---------- | -------------------------------------------------------------------- |
| [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)               | **133.7K** | Apache-2.0 | Agent 示例大合集，分类清晰（入门 → 进阶 → 多智能体），找灵感第一站   |
| [ashishpatel26/500-AI-Agents-Projects](https://github.com/ashishpatel26/500-AI-Agents-Projects) | **36.9K**  | MIT        | 500+ 行业落地案例（医疗/金融/教育/DevOps），部分案例为半成品，重思路 |
| [NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents)                           | **24.0K**  | 未指定     | 45+ 可直接运行的 Jupyter Notebook，90% 基于 LangGraph，跟练首选      |

## 五、生态协议与技能（3 个）

| 项目                                                                              | ⭐         | License | 用途                                                                                |
| --------------------------------------------------------------------------------- | ---------- | ------- | ----------------------------------------------------------------------------------- |
| [anthropics/skills](https://github.com/anthropics/skills)                         | **171.2K** | —       | Agent Skills 官方仓库：可复用技能的标准格式与实例，2026 年 Agent 能力扩展的事实标准 |
| [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)   | **89.8K**  | —       | MCP 官方参考服务器：Agent 连接外部工具的标准协议                                    |
| [anthropics/anthropic-cookbook](https://github.com/anthropics/anthropic-cookbook) | **52.1K**  | MIT     | Anthropic 官方实战手册：工具调用、Agent 模式、上下文工程代码示例                    |

## 六、本仓库关联项目与资料

> 本仓库自产或整理的学习资料，与外部清单互补——**star 不作为筛选标准**，但内容与本仓库深度文章一体同源。

| 项目/资料                                                                                                                                       | 类型     | 说明                                                                                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ForceInjection/awesome-skills](https://github.com/ForceInjection/awesome-skills)（[在线版](https://forceinjection.github.io/awesome-skills/)） | 开源项目 | 优秀认知技能（Agent Skill）合集：深度代码阅读、架构分析、文档评审等自动化工作流，与 [anthropics/skills](../agent_skills/docs/claude_skills_guide.md) 的学习配套 |
| [多智能体 AI 系统培训材料](../../10_ai_related_course/multi_agent_system/multi_agent_training/README.md)                                        | 仓库课程 | 五天结构化培训（理论基础 → LangGraph → LangSmith → 企业级架构 → 应用实践），配套本仓库两篇多智能体深度文档                                                      |
| [微软 AI Agents for Beginners 课程笔记](../../10_ai_related_course/AI%20Agents%20for%20Beginners%20课程之%20AI%20Agent及使用场景简介.md)        | 仓库笔记 | 微软 15 课的中文学习笔记，可与原课程配套使用                                                                                                                    |
| [ForceInjection/OpenSpec-practise](https://github.com/ForceInjection/OpenSpec-practise)                                                         | 开源项目 | Spec 驱动开发（SDD）工程实践：意图 → Spec → AI → 代码 & 验证，Harness 时代的工作流范式                                                                          |
| [MiniMax-AI/mmx-cli](https://github.com/MiniMax-AI/cli)                                                                                         | 开源项目 | MiniMax 官方 CLI 技能（遵循 agentskills.io 标准），Agent Skills 在多媒体生成场景的实例                                                                          |

---

## 学习顺序建议

1. **零基础**：微软 15 课（或 hello-agents 前 3 章）建立概念 → 详见 [学习路线图](./learning-roadmap.md)
2. **想懂原理**：hello-agents 手搓 ReAct/Plan-and-Solve/Reflection，再回头读 LangGraph 源码会豁然开朗
3. **想快速实战**：crewAI 或 smolagents 出第一个 demo → GenAI_Agents 跟练 → awesome-llm-apps 找灵感
4. **想做编码 Agent**：Aider 上手 → Cline/OpenHands 深挖 → 对照 Harness_Engineering 理解设计
5. **企业级**：LangGraph 主攻 → Microsoft Agent Framework 对照 → MCP/A2A/Agent Skills 补齐协议

> 数据时效：star 数会持续变化，本文数据快照于 2026-08-24；引用前可用 `curl -s https://api.github.com/repos/{owner}/{repo}` 复核。
