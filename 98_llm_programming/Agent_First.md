# Agent First：软件工程的下一个范式转移

随着大语言模型（LLM）能力的爆发式增长，我们正处于软件开发历史上的一个转折点。从 Microsoft 联合创始人 Bill Gates [1] 到 OpenAI 的 Sam Altman [3]，行业领袖们一致认为：**AI Agent（智能体）将是继图形用户界面（GUI）之后计算领域的最大变革**。

软件开发的重心正在从"Mobile First"（移动优先）向 **"Agent First"（智能体优先）** 转移。这意味着，未来的软件不仅要服务于人类用户，更要首先服务于 AI Agent。本文将梳理编程范式的演变历史，探讨 Agent First 的核心理念，并结合权威专家观点，为大家提供构建未来软件的实战指南。

## 1. 编程范式的演变史：从主机到智能体

编程范式的每一次迭代，本质上都是**交互主体**与**计算环境**的重构。理解历史，才能更好地预判未来。

### 1.1 演变阶段概览

以下是过去六十年间软件开发范式的主要变迁：

| 阶段          | 核心范式           | 交互主体     | 开发重点                                       |
| :------------ | :----------------- | :----------- | :--------------------------------------------- |
| **1960s-70s** | **Hardware First** | 专家/操作员  | 针对特定硬件优化，资源昂贵，批处理为主。       |
| **1980s-90s** | **Desktop First**  | 个人用户     | 本地计算，GUI 图形界面，关注 OS API。          |
| **1995-2005** | **Web First**      | 互联网用户   | 浏览器即平台，无状态 HTTP 协议，C/S 架构。     |
| **2008-2015** | **Mobile First**   | 移动用户     | 触摸交互，碎片化场景，低功耗与弱网优化。       |
| **2015-2023** | **Cloud Native**   | 开发者/服务  | 微服务，容器化，API 经济，DevOps。             |
| **2024+**     | **Agent First**    | **AI Agent** | **自主决策，工具调用，环境沙箱，语义化 API。** |

### 1.2 范式转移的本质

每一次范式转移都伴随着 **"用户"定义的泛化**。在 `Mobile First` 时代，为了适应人类手指的触摸，我们重新设计了 UI；而在 `Agent First` 时代，为了适应 AI 的逻辑推理与工具调用，我们需要重新设计 API 和系统架构。

正如 **Satya Nadella** 在 Microsoft Build 2025 主题演讲中提出的"Agentic Web"愿景所示 [5]，AI Agent 正在成为人类与计算机交互的主要方式。这意味着软件的界面将不再局限于屏幕上的像素，而是延伸至代码层面的接口。

### 1.3 Software 3.0：从代码到自然语言

如果说范式转移描述的是软件"为谁服务"的变迁，那么 **Andrej Karpathy** 提出的 **Software 3.0 框架** [4] 则从另一个维度揭示了软件"如何被创造"的变革。这两条线索的交汇，正是 Agent First 诞生的理论基础。

Karpathy 将软件的演进划分为三个阶段：

- **Software 1.0**：人类用 C++、Python 等语言编写明确的规则和逻辑。
- **Software 2.0**：通过神经网络学习权重，由数据驱动行为（如图像识别、语音处理）。
- **Software 3.0**：以 LLM 为核心，用自然语言"编程"。正如 Karpathy 所说："LLM 是一种新型计算机，我们用英语为它编程。"

**Software 3.0 与 Agent First 的内在关联**：Software 3.0 定义了"如何与 LLM 交互"，而 Agent First 则回答了"如何为 LLM 驱动的 Agent 构建系统"。换言之，Software 3.0 是编程方式的变革，Agent First 是架构设计的变革——前者是因，后者是果。当自然语言成为新的编程接口，软件系统就必须从底层重新设计，以适应这种全新的"用户"——这正是 Agent First 的核心主张。

---

## 2. 什么是 Agent First？

**Agent First** 是一种软件设计理念 [6]，要求开发者在构建系统时，优先考虑 AI Agent 的使用场景和接入体验。

### 2.1 核心定义

在传统软件中，人是第一类公民，系统通过 UI（User Interface）与人交互。而在 Agent First 架构中，**Agent 是第一类公民** [7]，系统通过 API 和语义化描述与 Agent 交互。

- **传统模式**：人阅读文档 -> 理解逻辑 -> 点击按钮/输入命令。
- **Agent 模式**：Agent 读取规范 -> 规划任务 -> 调用工具 -> 分析结果 -> 自我修正。

### 2.2 专家观点：Agentic Workflow 与四种设计模式

人工智能专家 **Andrew Ng（吴恩达）** 曾多次强调 **Agentic Workflow（智能体工作流）** 的重要性 [2]。他认为，AI 的未来不在于拥有一个更强的单一模型（Zero-shot），而在于构建一个能像人类一样通过"规划、研究、起草、修改"循环工作的系统。

> "Thinking of AI as an agentic workflow allows the model to think, revise, and improve iteratively... This leads to significantly better outcomes."
> —— **Andrew Ng**

Ng 进一步总结了四种核心 Agentic 设计模式 [2]：

- **Reflection（反思）**：LLM 审视自身输出，寻找改进方法，实现自我迭代优化。
- **Tool Use（工具使用）**：LLM 调用外部工具（如搜索引擎、代码执行器、数据库）收集信息并执行操作。
- **Planning（规划）**：LLM 自主制定并执行多步骤计划以达成目标。
- **Multi-Agent Collaboration（多智能体协作）**：多个 Agent 分工合作，各司其职，通过讨论和辩论协同完成复杂任务。

在 Agent First 的世界里，我们的软件必须能够支持这种"迭代式"的工作流，提供原子化的工具让 Agent 能够分步执行任务，而不是黑盒式的一键操作。

---

## 3. Agent First 编程范式的核心原则

要构建对 Agent 友好的软件，开发者需要遵循以下核心原则。这些原则围绕一个统一的设计哲学展开：**降低 Agent 的认知负担，提升其自主行动能力**。

我们可以将这四个原则理解为一个递进的层次模型：

```text
┌─────────────────────────────────────────────────────────┐
│  3.4 安全、沙箱与可观测性 —— 信任层：确保 Agent 行为可控可追溯  │
├─────────────────────────────────────────────────────────┤
│  3.3 标准化协议 (MCP) —— 连接层：让 Agent 即插即用           │
├─────────────────────────────────────────────────────────┤
│  3.2 机器友好文档 —— 理解层：让 Agent 快速学习               │
├─────────────────────────────────────────────────────────┤
│  3.1 API 优先 —— 能力层：让 Agent 能够执行操作              │
└─────────────────────────────────────────────────────────┘
```

- **能力层**（API 优先）：解决"Agent 能做什么"的问题。
- **理解层**（机器友好文档）：解决"Agent 如何学会使用"的问题。
- **连接层**（标准化协议）：解决"Agent 如何便捷接入"的问题。
- **信任层**（安全、沙箱与可观测性）：解决"Agent 如何被安全地使用"的问题。

以下逐一展开：

### 3.1 API 优先与极致的自描述性（能力层）

Agent 没有直觉，它们依赖明确的定义。模糊的 API 参数是 Agent 执行任务的最大障碍。

- **语义化命名**：避免使用 `type=1`, `status=2` 这种魔法数字，应使用 `status="pending_review"` 等具备语义的值。
- **OpenAPI 规范**：必须提供详尽的 OpenAPI (Swagger) 文档。这是 Agent 理解软件能力的"说明书"。
- **原子化设计**：将复杂业务拆解为独立的、无副作用的原子操作（Atomic Actions），方便 Agent 根据需要灵活组合。

```yaml
# 良好的 OpenAPI 示例（片段）
paths:
  /orders/{orderId}/cancel:
    post:
      summary: "取消订单"
      description: "取消一个处于 'pending' 状态的订单。如果订单已发货，此操作将失败。"
      operationId: "cancelOrder"
      # ... 详细的参数与响应定义
```

### 3.2 面向机器的文档体系（理解层）

人类开发者喜欢阅读带截图和视频的教程，但 Agent 只需要纯文本和结构化数据。

- **`llms.txt` 标准**：在网站或项目根目录提供 `/llms.txt` 文件。这是一个新兴的标准，专门用于向 LLM 提供项目的核心上下文、API 索引和最佳实践。
- **Few-Shot Examples（少样本示例）**：在文档中提供大量的"输入-输出"对。Agent 通过学习示例（In-context Learning）掌握工具用法的效率远高于阅读抽象的文字说明。

### 3.3 拥抱标准化协议：MCP（连接层）

为了避免每个 Agent 都需要单独适配我们的系统，应积极采用行业标准协议。

- **Model Context Protocol (MCP)**：这是一个由 Anthropic 等公司推动的开放标准，旨在标准化 AI 模型与外部数据（如数据库、文件系统、API）的连接方式。支持 MCP 意味着我们的软件可以即插即用地被 Claude、Cursor 等先进 Agent 调用。

### 3.4 安全、沙箱与可观测性（信任层）

Bill Gates 在其博文中提到 [1]，Agent 的广泛应用带来了新的安全挑战。

- **Human-in-the-Loop (HITL)**：对于高风险操作（如资金转账、数据删除），必须设计"请求-批准"机制，强制要求人类确认。
- **资源配额（Budgeting）**：为 Agent 分配 Token 预算和 API 调用次数上限，防止其陷入死循环（Infinite Loops）消耗过多资源。
- **思考链追踪**：日志系统不仅要记录"结果"，还要记录 Agent 的 **Reasoning Trace（推理轨迹）**，以便调试 Agent 为什么会做出错误的决策。

---

## 4. Agent First 的局限性与挑战

任何技术范式都有其适用边界。在拥抱 Agent First 的同时，我们也必须正视其当前的局限性：

### 4.1 技术层面的挑战

当前 Agent 技术在可靠性、长程推理和上下文处理等关键维度仍面临瓶颈：

- **可靠性问题**：当前 LLM 仍存在"幻觉"（Hallucination）现象，Agent 可能基于错误的推理执行操作。在高风险场景（如金融交易、医疗诊断）中，完全依赖 Agent 的自主决策仍不现实。
- **长程任务的脆弱性**：Agent 在执行多步骤复杂任务时，错误会逐步累积。一个早期的错误判断可能导致整个任务链的失败。
- **上下文窗口限制**：尽管上下文长度不断增加，Agent 处理超大规模代码库或文档时仍面临信息丢失的风险。

### 4.2 生态与成本挑战

除技术本身外，围绕 Agent First 的生态体系和工程实践也面临现实阻力：

- **标准尚未统一**：MCP 等协议仍处于早期阶段，行业缺乏广泛认可的统一标准，可能导致生态碎片化。
- **开发成本增加**：为 Agent 设计语义化 API、编写 `llms.txt`、实现推理轨迹追踪等，都会增加开发和维护成本。对于资源有限的小团队，这可能是一个负担。
- **调试复杂度提升**：Agent 的行为具有非确定性，同样的输入可能产生不同的执行路径，这给测试和调试带来了新的挑战。

### 4.3 何时不应使用 Agent First

并非所有场景都适合 Agent First 架构：

- **确定性要求极高的系统**：如航空控制、核电站监控，应优先保证系统的可预测性。
- **简单的 CRUD 应用**：如果业务逻辑简单，传统 API 设计已足够，过度设计反而增加复杂性。
- **实时性要求极高的场景**：Agent 的多轮推理会引入延迟，不适合毫秒级响应的系统。

理解这些局限性，有助于我们更务实地应用 Agent First 理念——**它是一种强大的设计范式，但不是银弹**。

---

## 5. 总结与展望

**Sam Altman** 预测 [3]："有用的 Agent 将是 AI 的杀手级功能。"

Agent First 不仅仅是一个技术口号，它代表了软件价值链的重构。未来的软件竞争，将取决于**谁能更好地服务于 Agent**。

- **对于 UI 设计师**：传统的像素级 UI 可能会逐渐退居二线，取而代之的是"意图驱动的动态 UI"（Intent-based UI）。
- **对于开发者**：我们的工作不再只是写代码给机器跑，而是构建一个"数字生态系统"，制定清晰的规则，让成千上万的数字员工（Agents）能在其中高效、安全地协作。

现在，是时候开始为下一个项目编写 `llms.txt`，并重新审视我们的 API 设计了。

---

**参考文献**：

[1] B. Gates, "AI-powered agents are the future of computing," _GatesNotes_, Nov. 2023. [Online]. Available: https://www.gatesnotes.com/ai-agents

[2] A. Ng, "How Agents Can Improve LLM Performance," _DeepLearning.AI The Batch_, Mar. 2024. [Online]. Available: https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/

[3] S. Altman, "Helpful agents are poised to become AI's killer function," _MIT Technology Review_, May 2024. [Online]. Available: https://www.technologyreview.com/2024/05/01/1091979/sam-altman-says-helpful-agents-are-poised-to-become-ais-killer-function/

[4] A. Karpathy, "Software Is Changing (Again)," _Y Combinator_, Jun. 2025. [Online]. Available: https://www.ycombinator.com/library/MW-andrej-karpathy-software-is-changing-again

[5] S. Nadella, "Microsoft Build 2025 Opening Keynote," _Microsoft_, May 2025. [Online]. Available: https://news.microsoft.com/build-2025/

[6] Adesso, "AI Agent First: How AI is redefining our understanding of software development," _Adesso Blog_, 2024. [Online]. Available: https://www.adesso.de/en/news/blog/ai-agent-first-how-ai-is-redefining-our-understanding-of-software-development.jsp

[7] Tech Twitter, "Building for trillions of agents," _Tech Twitter_, 2024. [Online]. Available: https://www.techtwitter.com/articles/building-for-trillions-of-agents
