# AI Agent 基础设施的崛起

> **原文：** <https://www.madrona.com/the-rise-of-ai-agent-infrastructure/>
> 2024-6-5 by Jon Turow

![AI Agent 基础设施的崛起](https://www.madrona.com/wp-content/uploads/2024/05/AI-Agent-infrastructure-blog-post-ChatGPT.webp)

_（如需查看更新的 AI 基础设施市场图谱，请参阅 Jon Turow 在 [2025 年 3 月的文章](https://www.madrona.com/ai-agent-infrastructure-three-layers-tools-data-orchestration/)。）_

GenAI 应用的爆炸式增长显而易见，涵盖了生产力、开发、云基础设施管理、媒体消费，甚至医疗收入周期管理等领域。这种爆炸式增长得益于快速改进的模型和我们行业在过去 24 个月中构建的底层平台基础设施，这些基础设施简化了托管、微调、数据加载和内存管理，使应用构建变得更加容易。

因此，许多创始人和投资者的目光转向了技术栈的顶层，我们终于可以开始将最先进的技术为终端用户服务。但 GenAI 发展的惊人速度意味着很少有假设能够长期成立。应用现在正以一种新的方式构建，这将对底层基础设施提出新的要求。这些开发者正在一座半成品桥梁上疾驰。如果我们的行业未能在技术栈的底层为他们提供一套新的 AI Agent 基础设施组件支持，他们的应用将无法发挥其全部潜力。

## Agent 的崛起

一个关键变化是 AI Agent 的崛起：能够规划和执行多步骤任务的自主行为者。如今，AI Agent——而非对底层模型的直接提示——正在成为终端用户遇到的常见接口，甚至成为开发者构建应用的核心抽象。这进一步加速了新应用的构建速度，并在平台层创造了一系列新的机会。

从 2022 年的 [MRKL](https://arxiv.org/pdf/2205.00445) 项目开始，到 2023 年的 [ReAct](https://arxiv.org/abs/2210.03629)、[BabyAGI](https://github.com/yoheinakajima/babyagi) 和 [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)，开发者开始发现提示和响应链可以将大型任务分解为较小的任务（规划）并自主执行它们。像 [LangChain](https://www.langchain.com/)、[LlamaIndex](https://www.llamaindex.ai/)、[Semantic Kernel](https://github.com/microsoft/semantic-kernel)、[Griptape](https://www.griptape.ai/) 等框架显示了 Agent 可以通过代码与 API 交互，而 [Toolformer](https://arxiv.org/abs/2302.04761) 和 [Gorilla](https://arxiv.org/abs/2305.15334) 等研究论文表明底层模型可以学会有效使用 API。来自 [Microsoft](https://www.microsoft.com/en-us/research/project/autogen/)、[Stanford](https://arxiv.org/abs/2304.03442) 和 [Tencent](https://arxiv.org/abs/2402.05120) 的研究表明，AI Agent 协同工作比单独工作效果更好。

如今，Agent 这个词对不同的人[意味着不同的东西](https://blog.langchain.dev/openais-bet-on-a-cognitive-architecture/)。如果你与足够多的从业者交谈，会发现一个光谱，其中包含多个都可以称为 Agent 的概念。BabyAGI 创建者 Yohei Nakajima 有一个[很好的分类方法](https://twitter.com/yoheinakajima/status/1781470851621372238)：

1. **手工制作的 Agent：** 提示和 API 调用的链条，具有自主性但在狭窄约束内运行。
2. **专业化 Agent：** 在任务类型和工具的子集内动态决定要做什么。受约束，但比手工制作的 Agent 约束更少。
3. **通用 Agent：** Agent 的 AGI——仍在地平线上，而非今天的实际现实。

我们最先进的前沿模型（GPT-4o、Gemini 1.5 Pro、Claude 3 Opus 等）的推理限制是阻碍我们构建、部署和依赖更高级 Agent（专业化和通用）能力的关键约束。Agent 使用前沿模型进行规划、优先级排序和自我验证——即将大型任务分解为较小任务并确保输出正确。因此，适度的推理水平意味着 Agent 也受到约束。随着时间推移，具有更高级推理能力的新前沿模型（GPT-5、Gemini 2 等）将使更高级的 Agent 成为可能。

## 应用 Agent

如今，开发者表示表现最佳的 Agent 都是极度手工制作的。开发者正在想方设法应用这些技术的当前状态，通过找出在正确约束下今天可行的用例。尽管存在限制，Agent 仍在激增。终端用户有时会意识到它们的存在，比如在 Slack 上响应的编码 Agent。越来越多的 Agent 也被埋在其他 UX 抽象下，如搜索框、电子表格或画布。

考虑 [Matrices](https://matrices.app/)，一家成立于 2024 年的电子表格应用公司。Matrices 构建能够代表用户自动完成的电子表格，例如，根据行和列标题推断用户想要在（比如）单元格 A1:J100 中的信息，然后搜索网络并解析网页以找到每条数据。Matrices 的核心电子表格 UX 与 Excel（1985 年推出）甚至 Visicalc（1979 年推出）并无太大不同。但 Matrices 的开发者可以使用 1000+ 个 Agent 对每行、每列甚至每个单元格进行独立的多步推理。

或者考虑 [Gradial](https://gradial.com/)，一家成立于 2023 年的营销自动化公司。Gradial 让数字营销团队通过帮助创建资产变体、执行内容更新以及跨渠道创建/迁移页面来自动化其内容供应链。Gradial 提供聊天界面，但也可以通过响应 JIRA 或 Workfront 等跟踪系统中的工单来满足营销人员的现有工作流程。营销人员无需将高级任务分解为单个操作。相反，Gradial Agent 完成这项工作并在幕后代表营销人员完成任务。

当然，今天的 Agent 有很多限制。它们经常出错。它们需要管理。运行太多 Agent 会对带宽、成本、延迟和用户体验产生影响。开发者仍在学习如何有效使用它们。但读者会注意到这些限制与对基础模型本身的抱怨如出一辙。验证、投票和模型集成等技术为 AI Agent 强化了近期历史为 GenAI 整体所显示的：开发者指望快速的科学和工程改进，并以未来状态为目标进行构建。他们正在我上面提到的半成品桥梁上疾驰，假设它将快速完工。

## 用基础设施支持 Agent

所有这些都意味着我们的行业需要努力构建支持 AI Agent 和依赖它们的应用的基础设施。

如今，许多 Agent 几乎完全垂直集成，没有太多托管基础设施。这意味着：Agent 的自管理云主机、用于内存和状态的数据库、从外部源摄取上下文的连接器，以及称为 [Function Calling](https://platform.openai.com/docs/guides/function-calling)、[Tool Use](https://docs.anthropic.com/en/docs/tool-use) 或 [Tool Calling](https://python.langchain.com/docs/modules/model_io/chat/function_calling/) 的功能来使用外部 API。一些开发者使用 LangChain 等软件框架（特别是其评估产品 [Langsmith](https://www.langchain.com/langsmith)）将这些组合在一起。这个技术栈今天效果最好，因为开发者正在快速迭代，感觉他们需要端到端控制其产品。

![AI Agent 基础设施 - 当前状态](https://www.madrona.com/wp-content/uploads/2024/06/Madrona-2024-04-AIAgentsMarketMapCurrent_v9-0-HiRes.png)

但随着用例固化和设计模式改进，情况将在未来几个月发生变化。我们仍然牢固地处于手工制作和专业化 Agent 的时代。因此，近期最有用的基础设施原语将是那些满足开发者当前需求并让他们构建可控制的手工制作 Agent 网络的原语。该基础设施也可以具有前瞻性。随着时间推移，推理将逐渐改进，前沿模型将开始引导更多工作流程，开发者将希望专注于产品和数据——这些是区分他们的东西。他们希望底层平台能够"正常工作"，具备规模、性能和可靠性。

![AI Agent 基础设施 - 新兴生态系统](https://www.madrona.com/wp-content/uploads/2024/06/06-06024-AI_Agents_Infrastructure__Emerging_2024_V9_hiRes.png)

果然，当你这样看时，你可以看到一个丰富的生态系统已经开始形成，提供 AI Agent 基础设施。以下是一些关键主题：

### Agent 专用开发工具

像 [Flowplay](https://flowplay.ai/)、[Wordware](https://www.wordware.ai/) 和 [Rift](https://github.com/morph-labs/rift) 这样的工具原生支持常见设计模式（投票、集成、验证、"团队"），这将帮助更多开发者理解这些模式并将其用于构建 Agent。一个有用且有主见的开发工具可能是解锁基于这种强大 Agent 技术的下一波应用的最重要基础设施之一。

### Agent 即服务

针对特定任务的手工制作 Agent 开始充当基础设施，开发者可以选择购买而非构建。这些 Agent 提供有主见的功能，如：

- **UI 自动化：** [Tinyfish](https://www.tinyfish.io/)、[Reworkd](https://reworkd.ai/)、[Firecrawl](https://www.firecrawl.dev/)、[Superagent](https://superagent.sh/)、[Induced](https://induced.ai/) 和 [Browse.ai](http://browse.ai/)
- **工具选择：** [NPI](https://npi.ai/)、[Imprompt](https://imprompt.ai/)
- **提示创建和工程**

一些终端客户可能直接应用这些 Agent，但开发者也将通过 API 访问这些 Agent 并将它们组装成更广泛的应用。

### 浏览器基础设施

读取网络并对其采取行动是一个关键优先事项。开发者通过让 Agent 与 API、SaaS 应用和网络交互来丰富其 Agent。API 接口足够直接，但网站和 SaaS 应用访问、导航、解析和抓取都很复杂。这样做使得可以像使用 API 一样使用任何网页或 Web 应用来以结构化形式访问其信息和功能。这需要管理连接、代理和验证码。

提供浏览器基础设施的公司包括：

- [Browserbase](https://browserbase.com/)
- [Browserless](https://www.browserless.io/)
- [Apify](https://apify.com/)
- [Bright Data](https://brightdata.com/)
- [Platform.sh](https://platform.sh/blog/off-with-its-head-headless-chrome-as-a-service/)
- [Cloudflare Browser Rendering](https://developers.cloudflare.com/browser-rendering/)

### 个性化内存

当 Agent 在多个模型之间分配任务时，提供共享内存并确保每个模型都能访问相关历史数据和上下文变得重要。像 [Pinecone](https://www.pinecone.io/)、[Weaviate](https://weaviate.io/) 和 [Chroma](https://www.trychroma.com/) 这样的向量存储对此很有用。但存在一类具有互补、有主见功能的新公司，包括：

- [WhyHow](https://www.whyhow.ai/)
- [Cognee](https://www.cognee.ai/)
- [LangMem](https://python.langchain.com/v0.1/docs/modules/memory/)（LangChain 的一个功能）
- [MemGPT](https://memgpt.ai/)（流行的开源项目）

这些公司展示了如何为终端用户及其当前上下文个性化 Agent 内存。

### Agent 身份验证

这些 Agent 代表终端用户与外部系统交互时，需要管理身份验证和授权。如今，开发者使用 OAuth 令牌让 Agent 模拟终端用户（这很微妙），在某些情况下，甚至要求用户提供 API 密钥。但 UX 和安全影响是严重的，并非所有网络都支持 OAuth（这就是为什么 Plaid 在金融服务中存在的原因）。

Agent 身份验证解决方案的示例：

- [Anon.com](http://anon.com/)
- [Mindware](https://mindware.co/)
- [Statics.ai](http://statics.ai/)

这些是开发者在规模化时需要的示例：Agent 本身的托管身份验证和授权。

### "Agent 的 Vercel"

通过分布式系统无缝管理、编排和扩展 Agent 的托管。如今存在一组分散的原语：

**Agent 托管：**

- [E2b.dev](https://e2b.dev/)
- [Ollama](https://ollama.com/)
- [Langserve](https://www.langchain.com/langserve)

**持久化：**

- [Inngest](https://www.inngest.com/)
- [Hatchet.run](https://hatchet.run/)
- [Trigger.dev](https://trigger.dev/)
- [Temporal.io](http://temporal.io/)

**编排：**

- [DSPy](https://qdrant.tech/documentation/frameworks/dspy/)
- [AutoGen](https://www.microsoft.com/en-us/research/blog/autogen-enabling-next-generation-large-language-model-applications/)
- [CrewAI](https://www.crewai.com/)
- [Sema4.ai](http://sema4.ai/)
- [LangGraph](https://python.langchain.com/v0.1/docs/langgraph/)

一些平台（LangChain 和 [Griptape](https://www.griptape.ai/)）为这些功能的不同组合提供托管服务。一个提供可扩展、托管的整合服务，代表应用开发者提供持久化和编排，将意味着开发者不再需要在多个抽象级别（应用和 Agent）思考，而可以专注于他们希望解决的问题。

## 构建 AI Agent 基础设施的未来

AI Agent 基础设施的演进还处于非常早期阶段，今天我们看到的是运营服务和开源工具的混合，这些工具尚未商业化或纳入更广泛的服务中。谁将成为赢家还远不清楚——在这个领域，最终的赢家今天可能还很年轻，或者可能还不存在。所以让我们开始工作吧。

---
