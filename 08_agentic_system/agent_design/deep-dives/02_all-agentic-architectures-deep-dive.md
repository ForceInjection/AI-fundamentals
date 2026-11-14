# All Agentic Architectures 深入详解

从 Reflection、ReAct 到 Blackboard、Ensemble、Tree of Thoughts、Graph World-Model、Metacognitive——以 **17 种可运行的 LangChain + LangGraph 架构**为线索，系统性梳理 Agent 领域从 2022 年论文原型到 2026 年生产系统的完整技术演进。每章均配有：**核心思想**的一句话定义、**Mermaid 架构图**、**State / Pydantic / Node / Graph 四要素实现解析**、**Head-to-Head 基线对比**、以及 **LLM-as-a-Judge 定量评分**。全文横跨单 Agent 模式（Reflection/Tool Use/ReAct/Planning）、多智能体协作（流水线/黑板/元控制器/集成）、高级记忆与推理（FAISS+Neo4j 双记忆/思维树/图世界模型）、安全与可靠性（PEV/心智模拟器/Dry-Run/元认知）、学习与自适应（RLHF 自改进/元胞自动机）五大主题，并在第六部分以**控制流分类、状态设计模式、Temperature 选型、错误处理等级、架构组合**五个维度进行横切分析，最终提供一棵可直接用于选型决策的 Mermaid 决策树。适合希望从「跑通 Demo」跨越到「设计生产系统」的工程师、架构师与研究者。

> 项目地址：<https://github.com/FareedKhan-dev/all-agentic-architectures>

---

## 前言与导读

AI Agent（智能体）正在从学术论文走向生产系统。2024-2026 年间，LangChain、LangGraph、AutoGen、CrewAI 等框架的成熟，使得构建复杂的 Agent 系统从「前沿研究」变为「工程实践」。然而，大多数资料要么停留在概念层面，要么只展示单一架构的片段代码，缺乏对完整架构谱系的系统性梳理。

本文基于一个包含 **17 种前沿 Agent 架构**的开源项目，逐一剖析每种架构的核心思想、完整实现、实际案例和适用场景。所有架构均使用 **LangChain + LangGraph** 实现，并配有 **LLM-as-a-Judge** 定量评估。

### 阅读建议

- **入门路径**：前言 → 技术基础与通用架构 → 第一部分（01-04）→ 第六部分选型指南
- **进阶路径**：第二~五部分按兴趣选读 → 第六部分横切分析
- **速查路径**：直接跳转到第六部分的分类法与选型决策树
- **决策者 / 产品经理路径**：前言 → 第七部分 7.1「17 种架构一览表」→ 第六部分 6.1「架构分类法与选型指南」+ 6.4「错误处理等级对比」→ 据此理解技术能力边界、评估实现代价、做出产品/资源决策（可略过代码细节）

### 全文架构导航

```mermaid
graph TB
    subgraph "Part 1: 基石模式"
        A1["01 Reflection<br/>自我批判"] --> A2["02 Tool Use<br/>工具使用"]
        A2 --> A3["03 ReAct<br/>推理+行动"]
        A3 --> A4["04 Planning<br/>规划"]
    end

    subgraph "Part 2: 多智能体协作"
        B1["05 Multi-Agent<br/>多智能体"] --> B2["07 Blackboard<br/>黑板系统"]
        B2 --> B3["11 Meta-Controller<br/>元控制器"]
        B3 --> B4["13 Ensemble<br/>集成决策"]
    end

    subgraph "Part 3: 高级记忆与推理"
        C1["08 Episodic+Semantic<br/>双记忆系统"] --> C2["09 Tree of Thoughts<br/>思维树"]
        C2 --> C3["12 Graph World-Model<br/>图世界模型"]
    end

    subgraph "Part 4: 安全与可靠性"
        D1["06 PEV<br/>计划-执行-验证"] --> D2["10 Mental Loop<br/>心智模拟器"]
        D2 --> D3["14 Dry-Run<br/>沙箱预演"]
        D3 --> D4["17 Metacognitive<br/>元认知"]
    end

    subgraph "Part 5: 学习与自适应"
        E1["15 RLHF<br/>自我改进"] --> E2["16 Cellular Automata<br/>元胞自动机"]
    end

    A4 --> B1
    A4 --> C1
    A4 --> D1
    B4 --> E1
```

---

## 技术基础与通用架构

在深入 17 种架构之前，先掌握它们共享的技术底座。理解这一章的内容，后续每种架构的学习成本会大幅降低。

### 技术栈全景

| 组件            | 角色       | 说明                                                  |
| --------------- | ---------- | ----------------------------------------------------- |
| **LangChain**   | LLM 交互层 | 提供与大模型交互的基础抽象：消息、提示模板、工具绑定  |
| **LangGraph**   | 工作流编排 | 基于有向图的状态机框架，支持循环、条件分支、并行执行  |
| **LangSmith**   | 可观测性   | 追踪每次 LLM 调用、工具执行和状态变化，用于调试和评估 |
| **ChatNebius**  | LLM 提供方 | OpenAI 兼容接口，提供 Llama 和 Mixtral 系列模型       |
| **Pydantic v2** | 数据建模   | 定义结构化输出 Schema，LLM 返回类型安全的 Python 对象 |
| **Rich**        | 终端输出   | 格式化打印（Markdown、Table、Panel、Syntax 高亮）     |
| **Tavily**      | 搜索工具   | 为 Agent 提供实时网络搜索能力                         |
| **Neo4j**       | 图数据库   | 存储知识图谱，支持 Cypher 查询的多跳推理              |
| **FAISS**       | 向量存储   | 高效的相似度搜索，用于情景记忆的检索                  |

**双模型策略**：本项目使用两个模型分工协作——

| 模型                                    | 定位     | Temperature | 典型用途                          |
| --------------------------------------- | -------- | ----------- | --------------------------------- |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 快速推理 | 0 - 0.2     | 简单任务、工具调用、基础规划      |
| `mistralai/Mixtral-8x22B-Instruct-v0.1` | 深度推理 | 0 - 0.5     | 复杂推理、多 Agent 编排、内容创作 |

> ⚠️ **重要澄清**：上表是本项目的*默认约定*，而非模型的*固有属性*。在生产实践中，**模型选择与 Temperature 是「节点 / 任务」的属性而非「模型」的属性**——同一个 Mixtral-8x22B 在 Planner 节点应用 temperature=0（求确定性），在 Creative Writing 节点可用 temperature=0.7（求多样性）；同一个 Llama-3.1-8B 在路由节点用 temperature=0，在文案润色节点可用 temperature=0.4。推荐的做法是：**按节点粒度声明 LLM 实例与参数**（如 `router_llm = ChatNebius(..., temperature=0)`、`creative_llm = ChatNebius(..., temperature=0.5)`），而不是全局共享 LLM 实例。

### 所有架构的「骨架代码」

17 个 Notebook 遵循统一的 **七阶段代码范式**：

```text
阶段 0: 环境配置
├── 加载 .env 环境变量
├── 设置 LangSmith 追踪
└── 导入 LangChain / LangGraph / Pydantic / Rich

阶段 1: 定义 Pydantic Schema
├── 主输出模型（如 DraftCode, Plan, MarketingEmail）
├── 辅助模型（如 Critique, VerificationResult）
└── 评估模型（如 CodeEvaluation, TaskEvaluation）

阶段 2: 初始化核心组件
├── 创建 ChatNebius LLM 实例
├── 初始化工具（TavilySearch 等）
└── 创建 Rich Console

阶段 3: 定义 TypedDict State
└── 如 AgentState(user_request, plan, final_answer, ...)

阶段 4: 实现 Node 函数
├── 处理节点（生成、执行、检索）
├── 决策节点（批评、验证、路由）
└── 路由函数（条件分支逻辑）

阶段 5: 构建 LangGraph 图
├── StateGraph(MyState)
├── add_node() / add_edge() / add_conditional_edges()
├── set_entry_point() → compile()
└── 调用 invoke() 或 stream()

阶段 6: LLM-as-a-Judge 评估
├── 定义评估 Schema（分项打分 + 理由）
└── 对比基线方案 vs 高级方案
```

以下是四个最核心的代码模式：

**模式 1：结构化输出**:

```python
from pydantic import BaseModel, Field

class DraftCode(BaseModel):
    code: str = Field(description="生成的 Python 代码")
    explanation: str = Field(description="代码说明")

# LLM 返回类型安全的 Pydantic 对象
structured_llm = llm.with_structured_output(DraftCode)
result = structured_llm.invoke("写一个排序函数")
# result.code, result.explanation 均为 str 类型
```

**模式 2：工具绑定**:

```python
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

search_tool = TavilySearch(max_results=2)
llm_with_tools = llm.bind_tools([search_tool])

# LangGraph 提供预构建的 ToolNode
tool_node = ToolNode([search_tool])
```

**模式 3：状态定义与消息归并**:

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import AnyMessage, add_messages

# 消息列表模式（自动合并，不覆盖）
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# 字段组合模式（精确控制每个字段）
class PlanningState(TypedDict):
    user_request: str
    plan: Optional[List[str]]
    intermediate_steps: List[str]
    final_answer: Optional[str]
```

**模式 4：图构建与条件路由**:

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(MyState)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.set_entry_point("planner")

# 条件路由：根据状态决定下一步
graph.add_conditional_edges("executor", lambda s: "executor" if s["plan"] else "end",
                            {"executor": "executor", "end": END})
compiled = graph.compile()
result = compiled.invoke({"user_request": "..."})
```

### LLM-as-a-Judge 评估范式

传统的自动评估指标（BLEU、ROUGE）无法衡量 Agent 输出的推理质量和任务完成度。本项目采用 **LLM-as-a-Judge** 模式——用一个 LLM 实例作为「裁判」，对 Agent 的输出进行结构化评分。

```python
class TaskEvaluation(BaseModel):
    task_completion_score: int = Field(description="任务完成度，1-10 分")
    reasoning_quality_score: int = Field(description="推理质量，1-10 分")
    justification: str = Field(description="评分理由")

judge_llm = llm.with_structured_output(TaskEvaluation)
evaluation = judge_llm.invoke(f"评估以下 Agent 输出：{agent_output}")
```

**LLM-as-a-Judge 的局限性（读者须知）**：

本文后续所有「A 架构 X 分 vs B 架构 Y 分」的对比结果，都建立在 LLM 裁判之上。在阅读时请始终记住：

- **裁判 ≠ 真理**：Judge LLM 本身也会犯错、有偏见——倾向于较长 / 较结构化的输出，对自己家族模型（同一基座或同一厂商训练的模型，例如用 GPT-4 裁判 GPT-4o 的输出）更宽容等，**分数是相对信号而非绝对度量**。
- **同模型自评有幸存者偏差**：用 GPT-4 评估 GPT-4 的输出往往高估；跨模型裁判（如用 Mixtral 裁判 Llama 的输出）更可信但依然非中立。
- **维度可见性局限**：LLM 能评价「可读性、完成度」，但难以评价「事实准确性、代码是否真能运行、API 是否真被正确调用」——这些需要**程序化测评**（执行代码、校验 API 返回）作为补充。
- **生产建议**：将 LLM-as-a-Judge 作为**快速反馈回路**（每次 PR 自动评分、回归检测），但关键决策仍应辅以**人工抽样审核 + 真实任务指标**（例如工具成功率、用户采纳率、延迟/成本）。

本文中各架构的评估维度：

| 架构           | 评估维度                                            |
| -------------- | --------------------------------------------------- |
| 01 Reflection  | correctness / efficiency / style                    |
| 02 Tool Use    | tool_selection / tool_input / synthesis_quality     |
| 03 ReAct       | task_completion / reasoning_quality                 |
| 04 Planning    | task_completion / process_efficiency                |
| 05 Multi-Agent | clarity_structure / analytical_depth / completeness |
| 06 PEV         | task_completion / error_handling                    |
| 07 Blackboard  | instruction_following / process_efficiency          |
| 13 Ensemble    | recommendation_quality / synthesis                  |
| 15 RLHF        | score (1-10) + feedback_points                      |

---

## 第一部分：基石模式

> 从单次生成到具备工具调用、推理循环和规划能力的单 Agent 进化之路。这四种模式是所有高级架构的基础。

### 1.1 Reflection（反思 / 自我批判）

#### 1.1.1 核心思想

**一句话定义**：Generate → Critique → Refine 三步自我改进循环。

类比人类写作中的「初稿 → 审校 → 终稿」流程。解决的核心问题是：单次 LLM 生成的质量不足，缺乏自我纠错能力。Reflection 通过让同一个 LLM 扮演「生成者」和「批评者」两个角色，实现了无需外部反馈的质量提升。

#### 1.1.2 架构图

```mermaid
graph LR
    A["Generator<br/>生成初始代码"] --> B["Critic<br/>结构化批评"]
    B --> C["Refiner<br/>定向改进"]
    C --> D["END"]
```

这是一个**线性 DAG**——没有循环，没有条件分支，三个节点依次执行。

#### 1.1.3 关键实现解析

**State 定义**：

```python
class ReflectionState(TypedDict):
    user_request: str
    draft: Optional[dict]          # Generator 输出
    critique: Optional[dict]       # Critic 输出
    refined_code: Optional[dict]   # Refiner 输出
```

**三个 Pydantic Schema 分别约束三个阶段的输出**：

```python
class DraftCode(BaseModel):
    code: str = Field(description="生成的 Python 代码")
    explanation: str = Field(description="代码工作原理说明")

class Critique(BaseModel):
    has_errors: bool = Field(description="是否有潜在 bug")
    is_efficient: bool = Field(description="算法效率是否合理")
    suggested_improvements: List[str] = Field(description="改进建议列表")
    critique_summary: str = Field(description="批评摘要")

class RefinedCode(BaseModel):
    refined_code: str = Field(description="改进后的代码")
    refinement_summary: str = Field(description="改进摘要")
```

**节点函数**遵循统一模式——接收 State，调用 LLM，返回 State 更新：

```python
def generator_node(state):
    generator_llm = llm.with_structured_output(DraftCode)
    prompt = f"你是 Python 专家，为以下需求编写代码：{state['user_request']}"
    draft = generator_llm.invoke(prompt)
    return {"draft": draft.model_dump()}

def critic_node(state):
    critic_llm = llm.with_structured_output(Critique)
    prompt = f"审查以下代码，指出问题和改进方向：\n{state['draft']['code']}"
    critique = critic_llm.invoke(prompt)
    return {"critique": critique.model_dump()}
```

**图构建**——三行代码完成编排：

```python
graph = StateGraph(ReflectionState)
graph.add_node("generator", generator_node)
graph.add_node("critic", critic_node)
graph.add_node("refiner", refiner_node)
graph.set_entry_point("generator")
graph.add_edge("generator", "critic")
graph.add_edge("critic", "refiner")
graph.add_edge("refiner", END)
```

#### 1.1.4 实际案例与效果

任务：「编写一个计算第 n 个 Fibonacci 数的函数」。

- **Generator** 产出递归实现（O(2^n) 时间复杂度）
- **Critic** 指出效率问题、缺少输入验证、没有缓存
- **Refiner** 改为迭代实现（O(n)），添加类型提示和边界检查

LLM-as-a-Judge 评分（`CodeEvaluation`，1-10 分制）对比：

| 维度        | 初始代码 | 改进后 |
| ----------- | -------- | ------ |
| correctness | 7        | 9      |
| efficiency  | 4        | 9      |
| style       | 6        | 8      |

#### 1.1.5 思考与延伸

- **适用场景**：代码生成、长文写作、结构化内容创作
- **局限性**：只有一轮批评-改进，无法迭代多次。对比 Notebook 15（RLHF），后者支持多轮循环直到质量达标
- **进化方向**：
  - **Iterative Reflection（迭代反思）**：在 `refiner → critic` 间加回边，配合「质量阈值 / 最大轮数」双停止条件，即进化为多轮自我优化——这正是 **Notebook 15 RLHF** 的前身形态，两者在第五部分形成完整闭环。
  - 在 Iterative Reflection 的基础上，如果再引入**跨任务的金标准记忆**（将历史上经过 Critic 高分认可的输出持久化，作为后续同类任务的 Few-Shot 示例），它便从「任务内自我修正」升级为「跨任务自我进化」，最终演化为完整的 Self-Improvement Loop——这正是 Notebook 15 RLHF 的核心机制。

---

### 1.2 Tool Use（工具使用）

#### 1.2.1 核心思想

**一句话定义**：让 LLM 自主决定「何时」及「如何」调用外部 API。

LLM 的知识有截止日期，无法访问实时数据。Tool Use 架构通过让 LLM 在回复中嵌入「工具调用请求」，将其从封闭系统变为开放系统。这是后续所有工具增强架构（ReAct、Planning、PEV 等）的基础。

#### 1.2.2 架构图

为避免图文不一致，此处分两张图呈现——**基础版**（Notebook 02 的实际行为）与**循环版**（形式上等价于 ReAct）：

```mermaid
graph LR
    subgraph "基础版（Notebook 02 实际实现）"
        A1["Agent<br/>分析问题"] -->|"有 tool_calls"| B1["Tool Node<br/>执行工具"]
        B1 --> A1
        A1 -->|"无 tool_calls"| C1["END"]
    end
    subgraph "循环版（加强版 ≈ ReAct）"
        A2["Agent<br/>推理+决策"] -->|"有 tool_calls"| B2["Tool Node"]
        B2 --> A2
        A2 -->|"无 tool_calls"| C2["END"]
    end
```

> **图示语义澄清**：Notebook 02 的 LangGraph 图**确实连接了** `tools → agent` 回边（代码与基础版一致），但由于系统提示和任务类型的约束，Agent 通常只调用一次工具就给出答案，**回边在实际执行中往往不被触发**。这意味着 Tool Use 与 ReAct 的真正边界**不在图结构，而在 Prompt + 任务复杂度**——当你用 ReAct 式的系统提示（鼓励多轮推理）替换 Notebook 02 的 Prompt 时，它的行为就等同于 ReAct。这也是下一节强调「ReAct 与 Tool Use 图结构相近、核心差异在语义」的根源。

#### 1.2.3 关键实现解析

**State 采用消息列表模式**——这是工具调用架构的标准选择：

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

`add_messages` 归并器确保新消息被**追加**而非覆盖，使得对话历史自然累积。

**工具绑定与路由**：

```python
from langchain_community.tools.tavily_search import TavilySearchResults

search_tool = TavilySearchResults(max_results=2, name="web_search")
llm_with_tools = llm.bind_tools([search_tool])

def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def router(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tool"
    return "__end__"
```

路由函数检查最后一条消息是否包含 `tool_calls`——这是 LangChain 中 LLM 表达「我需要调用工具」的标准方式。

**图构建**：

```python
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode([search_tool]))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", router)
graph.add_edge("tools", "agent")
```

#### 1.2.4 实际案例与效果

任务：「Apple 最新 WWDC 的主要公告有哪些？」

执行轨迹：

1. Agent 收到问题 → 生成 `tool_calls: [{"name": "web_search", "args": {"query": "Apple WWDC latest announcements"}}]`
2. ToolNode 执行搜索 → 返回搜索结果
3. Agent 综合搜索结果 → 输出最终答案

LLM-as-a-Judge 评分（`ToolUseEvaluation`，1-5 分制）：

| 维度                    | 得分 |
| ----------------------- | ---- |
| tool_selection_score    | 5    |
| tool_input_score        | 4    |
| synthesis_quality_score | 4    |

#### 1.2.5 思考与延伸

- **工具描述的质量直接影响路由准确率**——LLM 根据工具的 `description` 决定是否调用
- **单次调用的局限**：如果问题需要多步信息组合（如「A 公司 CEO 是谁，他之前在哪工作」），单次搜索无法满足
- **进化方向**：Tool Use → ReAct（多轮循环）

---

### 1.3 ReAct（推理 + 行动）

#### 1.3.1 核心思想

**一句话定义**：Think → Act → Observe 的动态多轮循环，交替推理和工具调用直到问题解决。

ReAct（Reasoning + Acting）源自 Yao et al. 2022 的论文。它与 Tool Use 的结构几乎相同，但语义完全不同——工具执行后**回到 Agent 继续推理**，而不是直接结束。这一看似微小的差异，使 Agent 获得了解决多跳问题的能力。

#### 1.3.2 架构图

```mermaid
graph LR
    A["ReAct Agent<br/>推理 + 决策"] -->|"需要更多信息"| B["Tool Node<br/>执行搜索"]
    B -->|"返回结果"| A
    A -->|"已有足够信息"| C["END<br/>综合输出"]
```

与 Tool Use 的图结构完全相同，**关键区别在于边的语义**：`Tool → Agent` 边意味着 Agent 会基于工具返回的新信息继续推理，而非简单复述。

#### 1.3.3 关键实现解析

Notebook 03 构建了两个 Agent 进行对比：

**Basic Agent**（基线）：强制单次回答

```python
def basic_agent_node(state: AgentState):
    # 系统提示强制只做一次搜索就回答
    system_msg = "你必须在一次工具调用后就给出最终答案。"
    response = llm_with_tools.invoke([system_msg] + state["messages"])
    return {"messages": [response]}
```

**ReAct Agent**：自由多轮推理

```python
def react_agent_node(state: AgentState):
    # 无限制，Agent 自主决定何时停止
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

两者的差异体现在**两处**：(1) 系统提示——Basic 强制 「在一次工具调用后就给出最终答案」；(2) 图结构——Basic 用 `add_edge("tools", END)` 形成线性 DAG，而 ReAct 用 `add_edge("tools", "agent")` 形成闭环。这两处共同作用，使 ReAct 具备了 Agent 自主控制循环次数的能力。

#### 1.3.4 Head-to-Head 对比实验

任务（多跳问答）：「Dune 这部科幻电影的制作公司的现任 CEO 是谁？该公司最新电影的预算是多少？」

这个问题需要至少 2-3 次搜索：查电影制作方 → 查 CEO → 查最新电影预算。

| 维度              | Basic Agent | ReAct Agent        |
| ----------------- | ----------- | ------------------ |
| 搜索次数          | 1           | 3                  |
| task_completion   | 3/10        | 9/10               |
| reasoning_quality | 2/10        | 8/10               |
| 结果              | 答案不完整  | 准确回答两个子问题 |

Basic Agent 只做了一次搜索，得到部分信息后就被迫输出答案。ReAct Agent 则进行了 3 轮搜索，逐步聚焦到最终答案。

#### 1.3.5 思考与延伸

- **ReAct 的探索性本质**：适合开放性、需要信息组合的问题
- **潜在风险**：可能陷入无限循环（需设置 `recursion_limit`）
- **效率 / 成本权衡**：每次循环都消耗 token + 延迟。ReAct 的「过度探索」对简单问题是明显浪费——2024 年社区实测表明：回答「Apple CEO 是谁」这类一跳事实型问题时，ReAct Agent 可能触发 2-3 次不必要的搜索，而单次 Tool Use 即可完成。**自然的解决方案**：在入口处接一个 **Meta-Controller（Notebook 11）先判断问题复杂度**，简单问答路由到 Tool Use、复杂多跳路由到 ReAct。这也是生产系统中常见的「路由 + 专家」组合范式。
- **与 Planning 的核心差异**：ReAct 是「边走边看」，Planning 是「先看地图再出发」

---

### 1.4 Planning（规划）

#### 1.4.1 核心思想

**一句话定义**：先制定完整计划（结构化步骤列表），再按步执行，最后综合所有结果。

Planning 架构将任务分为三个阶段：**Planner** 生成步骤列表 → **Executor** 逐步消费计划 → **Synthesizer** 综合中间结果。与 ReAct 的「探索式」不同，Planning 是「结构化」的——执行前就知道要做什么。

#### 1.4.2 架构图

```mermaid
graph LR
    A["Planner<br/>生成步骤列表"] --> B["Executor<br/>执行当前步骤"]
    B -->|"计划还有剩余步骤"| B
    B -->|"计划已执行完毕"| C["Synthesizer<br/>综合所有结果"]
    C --> D["END"]
```

#### 1.4.3 关键实现解析

**Plan Schema 用 Pydantic 约束 LLM 输出格式化的步骤列表**：

```python
class Plan(BaseModel):
    steps: List[str] = Field(description="按顺序排列的搜索查询列表")
```

**Planner Node 使用 few-shot 示例引导 LLM 生成高质量计划**：

```python
def planner_node(state):
    planner_llm = llm.with_structured_output(Plan)
    prompt = f"""为以下任务制定搜索计划。每步是一个具体的搜索查询。
    示例：
    任务: "法国首都的人口是多少？"
    计划: ["capital of France", "population of Paris"]

    任务: {state['user_request']}"""
    plan = planner_llm.invoke(prompt)
    return {"plan": plan.steps}
```

**Executor Node 逐步消费计划**——每次取第一步执行，将剩余步骤放回 State：

```python
def executor_node(state):
    current_step = state["plan"][0]
    remaining_plan = state["plan"][1:]
    result = search_tool.invoke({"query": current_step})
    return {
        "plan": remaining_plan,
        "intermediate_steps": state["intermediate_steps"] + [str(result)]
    }
```

**路由逻辑**——计划为空则综合，否则继续执行：

```python
def planning_router(state) -> str:
    if state["plan"]:
        return "execute"
    return "synthesize"
```

#### 1.4.4 Head-to-Head 对比实验

任务：「查询法国、德国、意大利首都的人口，求和后与美国人口对比」。

| 维度               | ReAct              | Planning            |
| ------------------ | ------------------ | ------------------- |
| 方法               | 逐步搜索，边找边想 | 一次性生成 4 步计划 |
| 搜索次数           | 5-7 次（含试错）   | 4 次（精确）        |
| task_completion    | 7/10               | 9/10                |
| process_efficiency | 5/10               | 9/10                |

Planning Agent 生成了清晰的 4 步计划（三国首都人口 + 美国人口），然后精确执行。ReAct Agent 虽然也完成了任务，但过程中有冗余搜索。

#### 1.4.5 思考与延伸

- **适用场景**：任务结构可预见、步骤间相对独立
- **核心问题**：如果某个步骤执行失败怎么办？计划没有容错机制
- **计划陈旧性问题（Plan Re-evaluation）**：即便每一步都执行成功，**计划的后半段可能已基于过时假设**——例如第 1 步发现标的公司已被收购，第 2 步的「查该公司财报」就不再适用。生产级方案应在 Executor 与 Synthesizer 之间插入一个 **Plan Reviewer 节点**：在每 N 步（或关键里程碑）之后，审视剩余计划是否仍然成立，不成立则触发重规划。这是 Planning 与 ReAct 的深度融合——「宏观规划 + 动态调整」。
- **进化方向**：Planning → PEV（Notebook 06），增加验证器和重规划能力
- **ReAct + Planning 混合**：先规划大框架，每步内部用 ReAct 灵活执行

---

## 第二部分：多智能体协作

> 基石模式解决了单 Agent 的能力闭环，但当任务复杂度超出单 Agent 的上下文窗口、专业深度或调度灵活性时，就需要多个专业化 Agent 分工协作——这正是多智能体系统的用武之地。这一部分按「编排耦合度」从低到高，展示四种不同的协作模式：固定流水线（Multi-Agent）→ 共享黑板（Blackboard）→ 显式路由（Meta-Controller）→ 并行扇出聚合（Ensemble）。

### 2.1 Multi-Agent Systems（多智能体系统）

#### 2.1.1 核心思想

**一句话定义**：专业化角色团队按固定流水线协作，各司其职。

类比一个分析师团队：新闻分析师负责舆情、技术分析师负责图表、财务分析师负责报表，最后报告撰写人综合所有分析。每个 Agent 拥有独立的系统提示（persona），决定了它的专业视角。

#### 2.1.2 架构图

```mermaid
graph LR
    A["News Analyst<br/>新闻分析师"] --> B["Technical Analyst<br/>技术分析师"]
    B --> C["Financial Analyst<br/>财务分析师"]
    C --> D["Report Writer<br/>报告撰写人"]
    D --> E["END"]
```

这是一个**固定顺序的流水线**——每个 Agent 把自己的分析结果写入 State，后续 Agent 可以读取前序结果。

#### 2.1.3 关键实现解析

**State 为每个 Agent 分配独立字段**：

```python
class MultiAgentState(TypedDict):
    user_request: str
    news_report: Optional[str]
    technical_report: Optional[str]
    financial_report: Optional[str]
    final_report: Optional[str]
```

**Agent 工厂函数**——用同一个模板创建不同 persona 的 Agent：

```python
def create_specialist_node(persona: str, output_key: str):
    def specialist_node(state: MultiAgentState):
        prompt = ChatPromptTemplate.from_messages([
            ("system", persona),
            ("human", "{input}")
        ])
        agent = prompt | llm_with_tools
        result = agent.invoke({"input": state["user_request"]})
        return {output_key: result.content}
    return specialist_node

news_analyst = create_specialist_node(
    persona="你是资深新闻分析师，擅长解读市场情绪和行业趋势...",
    output_key="news_report"
)
```

**Report Writer** 综合所有分析报告：

```python
def report_writer_node(state):
    prompt = f"""综合以下三份专家分析报告，撰写一份完整的市场分析报告：

    新闻分析：{state['news_report']}
    技术分析：{state['technical_report']}
    财务分析：{state['financial_report']}"""
    response = llm.invoke(prompt)
    return {"final_report": response.content}
```

#### 2.1.4 实际案例与效果

任务：生成 NVIDIA 市场分析报告。

对比单体 Agent（一个 generalist 独立完成所有分析）vs 多 Agent 团队：

| 维度              | 单体 Agent | Multi-Agent 团队 |
| ----------------- | ---------- | ---------------- |
| clarity_structure | 5/10       | 8/10             |
| analytical_depth  | 4/10       | 9/10             |
| completeness      | 5/10       | 9/10             |

多 Agent 团队在每个专业维度上都有显著提升，因为每个 Agent 可以用全部上下文窗口聚焦于自己的专业领域。

#### 2.1.5 思考与延伸

- **固定顺序的局限**：如果新闻情绪为负面，技术分析可能需要调整方向，但流水线无法回溯
- **固定顺序的反面价值**：确定性流水线意味着**更高的可预测性、可审计性与可复现性**，在金融/医疗/合规等强监管场景中，这往往比「灵活性」更重要——一份执行路径可追溯的分析报告，比一份「Controller 动态决定路径」的报告更易通过审计。
- **并行 vs 串行**：除了效率，**并行还能避免顺序偏见**——分析师 A 先写的观点会锚定分析师 B 的判断；让 A/B/C 并行产出后再由综合者合并，可显著降低集体思维（groupthink）。这也是 Ensemble（Notebook 13）相比顺序多 Agent 的根本优势之一。
- **进化方向**：Multi-Agent → Blackboard（动态调度）→ Meta-Controller（智能路由）

---

### 2.2 Blackboard Systems（黑板系统）

#### 2.2.1 核心思想

**一句话定义**：共享黑板（Blackboard）+ 智能控制器（Controller）动态调度专家 Agent。

Blackboard 系统源自 1980 年代的经典 AI 架构（如 Hearsay-II 语音识别系统）。核心思路是：有一块共享的「黑板」存储当前状态，多个 Agent 可以读取黑板上的信息并添加自己的贡献，一个 Controller 根据黑板内容决定下一步调用哪个 Agent。

与 Multi-Agent 的固定流水线不同，Blackboard 的执行顺序是**动态的**——Controller 可以根据中间结果跳过某些 Agent、重复调用某些 Agent，甚至改变执行计划。

#### 2.2.2 架构图

```mermaid
graph TD
    A["Controller<br/>读取黑板,决定下一步"] -->|"需要新闻分析"| B["News Analyst"]
    A -->|"需要技术分析"| C["Technical Analyst"]
    A -->|"需要财务分析"| D["Financial Analyst"]
    A -->|"准备输出"| E["Report Writer"]
    B --> A
    C --> A
    D --> A
    E --> F["END"]
```

#### 2.2.3 关键实现解析

**State 包含共享黑板和控制器日志**：

```python
class BlackboardState(TypedDict):
    user_request: str
    blackboard: List[str]        # 共享黑板，所有 Agent 追加内容
    available_agents: List[str]  # 当前可用的 Agent 列表
    next_agent: str              # Controller 决定的下一个 Agent 名称（或 "FINISH"）
```

**Controller 是整个系统最关键的组件**——它读取黑板内容，决定下一步：

```python
def controller_node(state):
    controller_llm = llm.with_structured_output(ControllerDecision)
    prompt = f"""你是一个分析任务的控制器。
    用户请求：{state['user_request']}
    黑板当前内容：{state['blackboard']}
    可用 Agent：{state['available_agents']}

    根据黑板上已有的信息，决定下一步应该调用哪个 Agent，
    或者信息已经足够，可以生成最终报告。"""
    decision = controller_llm.invoke(prompt)
    return {"next_agent": decision.next_agent}
```

**条件路由**基于 Controller 输出的 `next_agent` 字段动态选择下一个节点，`"FINISH"` 则路由到 END。为防止 Controller-Agent 多轮交互失控，调用时设置 `recursion_limit=10`。

#### 2.2.4 Head-to-Head 对比实验

任务：「分析 Nvidia 的最新重大新闻。根据新闻情绪决定下一步——如果为负面，跳过技术分析直接进行风险评估。」

| 维度                  | Sequential Agent       | Blackboard Agent                        |
| --------------------- | ---------------------- | --------------------------------------- |
| instruction_following | 4/10                   | 9/10                                    |
| process_efficiency    | 5/10                   | 8/10                                    |
| 行为                  | 执行所有分析，忽略条件 | Controller 检测到负面情绪后跳过技术分析 |

#### 2.2.5 思考与延伸

- **Controller 的 Prompt 质量决定整个系统的表现**——如果 Controller 无法正确理解条件分支逻辑，系统会退化为固定流水线
- **Controller 是单点故障 + 性能瓶颈**：Controller 错误的决策可能触发无效循环或错误结论，且**调试难度远大于固定流水线**——同样的输入每次执行路径可能不同，令回归测试变得困难。生产环境建议：(1) 对 Controller 决策打埋点日志；(2) 跟踪每条任务的执行谱（execution trace）；(3) 对 Controller 单独建立单元测试（固定黑板快照 → 验证决策输出）。
- **与 Meta-Controller 的区别**：Blackboard 是多轮调度（Controller 反复决策），Meta-Controller 是单次路由
- **潜在问题**：Controller 与 Agent 之间的多轮交互可能触发 `GraphRecursionError`，需要合理设置递归限制

---

### 2.3 Meta-Controller（元控制器 / 智能路由）

#### 2.3.1 核心思想

**一句话定义**：分析任务类型 → 一次性路由到最合适的专家 Agent。

Meta-Controller 是最简单的多 Agent 模式——没有协作、没有共享状态、没有循环。它只做一件事：理解用户意图，然后把请求转发给正确的专家。类比企业总机或 API Gateway。

#### 2.3.2 架构图

```mermaid
graph TD
    A["User Request"] --> B["Meta-Controller<br/>分析并路由"]
    B -->|"闲聊/简单问题"| C["Generalist Agent"]
    B -->|"需要搜索/研究"| D["Researcher Agent"]
    B -->|"需要写代码"| E["Coder Agent"]
    C --> F["END"]
    D --> F
    E --> F
```

#### 2.3.3 关键实现解析

**路由决策用 Pydantic 结构化输出**：

```python
class ControllerDecision(BaseModel):
    next_agent: str = Field(description="选择: Generalist, Researcher, Coder")
    reasoning: str = Field(description="路由理由")
```

**Controller Node**：

```python
def meta_controller_node(state):
    controller_llm = llm.with_structured_output(ControllerDecision)
    prompt = f"""分析以下用户请求，决定应该路由到哪个专家：
    - Generalist：闲聊、简单问答
    - Researcher：需要搜索实时信息的问题
    - Coder：需要编写代码的任务

    用户请求：{state['user_request']}"""
    decision = controller_llm.invoke(prompt)
    return {"next_agent_to_call": decision.next_agent}
```

**条件路由**——直接读取 State 字段：

```python
graph.add_conditional_edges(
    "meta_controller",
    lambda s: s["next_agent_to_call"],
    {"Generalist": "generalist", "Researcher": "researcher", "Coder": "coder"}
)
```

#### 2.3.4 实际案例

三次测试验证路由准确性：

| 输入                                                                        | 路由结果   | 正确性 |
| --------------------------------------------------------------------------- | ---------- | ------ |
| "Hello, how are you today?"                                                 | Generalist | 正确   |
| "What were NVIDIA's latest financial results?"                              | Researcher | 正确   |
| "Can you write me a python function to calculate the nth fibonacci number?" | Coder      | 正确   |

#### 2.3.5 思考与延伸

- **开销极低**：只需一次 LLM 调用做路由，后续由专家处理
- **与 Blackboard 的区别**：单次路由 vs 多轮调度
- **扩展性**：增加新专家只需在 Prompt 中添加描述 + 注册新节点
- **风险：路由错误时用户体验差**。正确的应对方式不是「加 fallback Agent」那么粗糙，而是在 `ControllerDecision` Schema 中新增 `confidence: float` 字段：
  - 若置信度 ≥ 阈值（0.7 作为经验值）→ 直接路由到选定专家
  - 若低于阈值→ 不就地路由，而是跳转到**「澄清 Agent」**向用户发起反问（「你是想查询财务数据还是市场新闻？」），或者交给一个通用 Agent 宽容处理。这是与 **Metacognitive（Notebook 17）** 思想的早期融合——Agent 不仅知道「路到哪里」，还知道「自己是否有把握」。

---

### 2.4 Ensemble（集成 / 扇出-扇入）

#### 2.4.1 核心思想

**一句话定义**：多个独立视角的 Agent 并行分析 → 综合决策者合并所有观点得出最终结论。

类比投资委员会：乐观派分析师、价值派分析师、量化分析师各自独立研究，最终由首席投资官（CIO）综合所有观点做出投资决策。关键创新在于**认知多样性**——不同 Agent 有不同的偏见和关注点，综合后比任何单一视角都更可靠。

#### 2.4.2 架构图

```mermaid
graph TD
    A["Start Analysis"] --> B["Bullish Analyst<br/>乐观视角"]
    A --> C["Value Analyst<br/>价值视角"]
    A --> D["Quant Analyst<br/>量化视角"]
    B --> E["CIO Synthesizer<br/>综合决策"]
    C --> E
    D --> E
    E --> F["END"]
```

这是一个**扇出-扇入**（Fan-out / Fan-in）模式——三个分析师并行执行，CIO 等待所有分析完成后综合。

#### 2.4.3 关键实现解析

**State 的关键设计——用 Dict 存储多个分析师的报告**：

```python
class EnsembleState(TypedDict):
    query: str
    analyses: Dict[str, str]  # {"bullish": "...", "value": "...", "quant": "..."}
    final_recommendation: Optional[FinalRecommendation]
```

**分析师工厂函数**——相同结构，不同人设：

```python
def create_analyst_node(persona: str, agent_name: str):
    def analyst_node(state):
        prompt = f"{persona}\n\n分析以下投资标的：{state['query']}"
        response = llm_with_tools.invoke(prompt)
        # 关键：合并而非覆盖
        current_analyses = state.get('analyses', {})
        current_analyses[agent_name] = response.content
        return {"analyses": current_analyses}
    return analyst_node
```

> 注意 State 合并模式：`current_analyses = state.get('analyses', {})` 然后更新——如果直接返回 `{"analyses": {agent_name: result}}`，会覆盖其他分析师的结果。

**CIO Synthesizer 输出结构化推荐**：

```python
class FinalRecommendation(BaseModel):
    final_recommendation: str = Field(description="Strong Buy/Buy/Hold/Sell/Strong Sell")
    confidence_score: float = Field(description="置信度 1.0-10.0")
    synthesis_summary: str
    identified_opportunities: List[str]
    identified_risks: List[str]
```

#### 2.4.4 实际案例

任务：NVIDIA 投资分析（2024 年中）。

三位分析师独立报告：

- **Bullish**：关注 AI 算力需求爆发、数据中心收入翻倍 → 推荐买入
- **Value**：关注 P/E 过高、泡沫风险 → 谨慎持有
- **Quant**：关注 RSI 指标、收入增长率 → 技术面偏多

CIO 综合：识别出共识（AI 需求强劲）和分歧（估值争议），给出"Buy, confidence 7.5"的最终推荐。

#### 2.4.5 思考与延伸

- **认知多样性的价值**：单个分析师可能有系统性偏见，Ensemble 通过多视角降低偏差
- **与 Tree of Thoughts 的区别**：ToT 探索不同推理路径，Ensemble 探索不同分析角度
- **进阶：Deliberative Ensemble（审议式集成）**：当 CIO 发现分析师间存在显著分歧或信息不足时，不立即综合，而是触发一轮**审议 / 信息澄清循环**——向特定分析师追问（「Bullish 分析师，Value 分析师提出的 P/E 估值风险你如何回应？」），或者让分析师对彼此的结论打分。这将 Ensemble 从单轮扇出-扇入推向多轮动态协作，贴近真实的投委会决策流程。
- **生产应用**：事实核查（多源验证）、风险评估、战略规划

---

## 第三部分：高级记忆与推理

> 多 Agent 协作解决了**并行分工**，但它们都共享一个根本局限：每次对话都是「失忆」的，且推理仍局限于一次前向生成。要让 Agent 在跨会话中保留上下文、并对复杂问题展开系统化的多路径搜索，需要为其配备**长期记忆**与**结构化推理能力**。这一部分介绍三种突破上下文窗口与单步推理瓶颈的高级架构：双轨记忆（Episodic + Semantic）、思维树（Tree of Thoughts）、图世界模型（Graph World-Model）。

### 3.1 Episodic + Semantic Memory（情景记忆 + 语义记忆）

#### 3.1.1 核心思想

**一句话定义**：FAISS 向量存储（情景记忆）+ Neo4j 图数据库（语义记忆）构成双记忆系统。

类比人类大脑：**情景记忆**（episodic）对应海马体——存储「上周二我和朋友讨论了 Apple 股票」这样的经历片段；**语义记忆**（semantic）对应新皮层——存储「Apple 是科技公司、库克是 CEO」这样的结构化知识。两者协同工作，使 Agent 能在多轮对话中实现真正的个性化。

#### 3.1.2 架构图

```mermaid
graph LR
    A["User Input"] --> B["Retrieve Memory<br/>检索 FAISS + Neo4j"]
    B --> C["Generate Response<br/>基于记忆生成回答"]
    C --> D["Update Memory<br/>提取并存储新记忆"]
    D --> E["END"]
```

#### 3.1.3 关键实现解析

**State 定义**：

```python
class AgentState(TypedDict):
    user_input: str
    retrieved_memories: Optional[str]
    generation: str
```

**双记忆检索**：

```python
def retrieve_memory(state):
    user_input = state['user_input']

    # 情景记忆：FAISS 向量相似度搜索
    episodic_results = episodic_vector_store.similarity_search(user_input, k=2)

    # 语义记忆：Neo4j 图查询（使用名为 'entity' 的全文索引）
    semantic_results = graph.query(
        "CALL db.index.fulltext.queryNodes('entity', $query) "
        "YIELD node RETURN node LIMIT 5",
        params={"query": user_input}
    )

    combined = f"情景记忆：{episodic_results}\n语义记忆：{semantic_results}"
    return {"retrieved_memories": combined}
```

**记忆更新——LLM 从对话中提取结构化知识**：

```python
class KnowledgeGraph(BaseModel):
    relationships: List[Relationship]

class Relationship(BaseModel):
    source: Node
    target: Node
    type: str = Field(description="关系类型，如 IS_A, INTERESTED_IN")

def update_memory(state):
    # 情景记忆：存储对话摘要
    summary = f"用户说：{state['user_input']}。助手回复：{state['generation']}"
    episodic_vector_store.add_documents([Document(page_content=summary)])

    # 语义记忆：LLM 提取实体-关系三元组，写入 Neo4j
    extractor = llm.with_structured_output(KnowledgeGraph)
    kg = extractor.invoke(f"从以下对话中提取实体和关系：{summary}")
    # 写入 Neo4j ...
```

#### 3.1.4 实际案例

三轮对话模拟：

| 轮次 | 用户输入                                            | Agent 行为                                                                      |
| ---- | --------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1    | 「我叫 Alex，是保守型投资者，主要关注成熟科技公司」 | 存储用户画像到 Neo4j；对话摘要存入 FAISS                                        |
| 2    | 「你怎么看 Apple (AAPL)？」                         | 从 Neo4j 检索到「Alex 是保守型投资者」 → 推荐基于稳定性分析                     |
| 3    | 「根据我的偏好，推荐其他股票」                      | 综合情景记忆（之前讨论了 AAPL）+ 语义记忆（保守型、成熟科技） → 推荐 MSFT、GOOG |

#### 3.1.5 思考与延伸

- **记忆衰减**：实际系统需要遗忘策略，避免过时信息干扰
- **时效性与相关性校准（关键工程难题）**：情景记忆不仅要**相似**，更要**关键**且**新鲜**。例如用户问「我的投资偏好」时，应优先召回「上周讨论的 Apple 股票看法」而非「一年前询问过 Apple 公司成立时间」。生产实践中的做法是：在向量相似度 `s_similarity` 的基础上，叠加两个附加因子——
  - **时效性衰减**：`s_recency = exp(-λ·Δt)`，越旧的记忆权重越低；
  - **重要性权重**：`s_importance`，用户明确点赞或重复提及的事实权重更高。

  最终综合评分为 `score = w₁·s_similarity + w₂·s_recency + w₃·s_importance`。这种「**相似 × 时效 × 重要性**」三因子打分是 **Generative Agents**（斯坦福 / Google 在 2023 年发布的 25-Agent 小镇模拟实验）、**MemGPT**（UC Berkeley 在 2024 年提出的 OS 式分层记忆管理 Agent）等前沿记忆系统的核心设计。

- **向量检索的召回率**：语义相似不等于逻辑相关——「保守型投资者」和「低风险策略」向量可能不够近
- **隐私问题**：长期记忆存储用户偏好，需要明确的数据管理策略

---

### 3.2 Tree of Thoughts（思维树）

#### 3.2.1 核心思想

**一句话定义**：并行探索多条推理路径 → 评估剪枝 → 沿最优路径搜索直到找到解。

Tree of Thoughts（Yao et al. 2023）将问题求解从**线性链**（Chain-of-Thought）扩展为**树搜索**。每一步都生成多个候选方案，评估后保留最有前景的分支，剪掉死胡同。这使得 Agent 可以「回溯」——如果某条路走不通，可以回到分叉点尝试另一条路。

#### 3.2.2 架构图

```mermaid
graph TD
    A["Initialize<br/>创建初始状态"] --> B["Expand<br/>生成所有合法后续状态"]
    B --> C["Prune<br/>剪除无效/重复路径"]
    C -->|"找到解"| D["END<br/>输出解路径"]
    C -->|"未找到解"| B
```

#### 3.2.3 关键实现解析

**PuzzleState 用 Pydantic 建模问题状态**（以经典「农夫过河」为例）：

```python
class PuzzleState(BaseModel):
    left_bank: set = Field(description="左岸的物品")
    right_bank: set = Field(description="右岸的物品")
    boat_location: str = Field(description="船的位置: left/right")
    move_description: str = Field(description="如何到达此状态")

    def is_valid(self) -> bool:
        """检查是否存在捕食关系（狼吃羊、羊吃白菜）：船不在的那一岸不能出现危险组合"""
        for bank, side in [(self.left_bank, "left"), (self.right_bank, "right")]:
            if self.boat_location != side:
                if {"wolf", "goat"}.issubset(bank) or {"goat", "cabbage"}.issubset(bank):
                    return False
        return True

    def is_goal(self) -> bool:
        return len(self.right_bank) == 3  # 所有物品到达右岸
```

**ToT State 维护多条活跃路径**：

```python
class ToTState(TypedDict):
    problem_description: str
    active_paths: List[List[PuzzleState]]  # 多条并行路径
    solution: Optional[List[PuzzleState]]
```

**Expand Node**——为每条活跃路径生成所有合法后续状态：

```python
def expand_paths(state):
    new_paths = []
    for path in state["active_paths"]:
        current = path[-1]
        for move in generate_possible_moves(current):
            next_state = apply_move(current, move)
            if next_state.is_valid():
                new_paths.append(path + [next_state])
    return {"active_paths": new_paths}
```

**Prune Node**——移除无效路径和循环：

```python
def prune_paths(state):
    valid_paths = []
    seen_states = set()
    for path in state["active_paths"]:
        state_hash = hash(path[-1])
        if state_hash not in seen_states:
            seen_states.add(state_hash)
            valid_paths.append(path)
    return {"active_paths": valid_paths}
```

执行时设置 `recursion_limit=15` 防止无限搜索。

#### 3.2.4 Head-to-Head 对比

问题：狼、羊、白菜过河。

| 方法             | 结果               | 过程                               |
| ---------------- | ------------------ | ---------------------------------- |
| Chain-of-Thought | 经常出错或陷入循环 | LLM 线性推理，一旦走错无法回溯     |
| Tree of Thoughts | 稳定找到正确解     | 系统化搜索所有合法路径，保证找到解 |

#### 3.2.5 思考与延伸

- **计算成本**：每层扩展可能生成指数级节点——需要有效的剪枝策略
- **LLM 在 ToT 中的角色**：可以用 LLM 评估哪些分支更有前景（启发式搜索），减少探索量
- **BFS vs DFS 的选择关乎成败**：Notebook 09 的实现是 **BFS（按层扩展 `active_paths`）**。BFS 保证找到最短解（分支因子有限、解深度较浅时适用，如渡河问题 ≤7 步），但内存随平面指数增长；DFS 内存友好，但在有环图中易陷入深深枝干，且不保证最优。对于「解很深但分支少」的问题（如多步数学证明），DFS+ 迭代深化更合适。对「解浅但分支多」的问题，BFS+ 激进剪枝更合适。实施时推荐用 `beam_width` + `max_depth` 双重限制。
- **适用场景**：逻辑谜题、约束满足问题、数学证明、策略规划
- **与 MCTS 的关系**：ToT 类似于蒙特卡洛树搜索的简化版，可以引入 UCB 等策略优化分支选择

---

### 3.3 Graph World-Model（图世界模型）

#### 3.3.1 核心思想

**一句话定义**：非结构化文本 → Neo4j 知识图谱 → Text-to-Cypher 实现多跳推理。

传统 RAG 通过向量检索找到「语义相似」的文档片段，但无法推理实体之间的关系链。Graph World-Model 先从文档中提取实体和关系（如「张三 → 就职于 → A 公司」），存入 Neo4j 图数据库，然后通过 LLM 生成 Cypher 查询来回答需要多跳推理的问题。

#### 3.3.2 架构图

```mermaid
graph TB
    subgraph "知识构建"
        A["非结构化文档"] --> B["LLM 实体关系提取"]
        B --> C["Neo4j 知识图谱"]
    end
    subgraph "知识查询"
        D["用户问题"] --> E["LLM 生成 Cypher"]
        E --> F["Neo4j 执行查询"]
        F --> G["LLM 综合回答"]
    end
    C -.-> F
```

#### 3.3.3 关键实现解析

**图 Schema 定义**——注意使用 Pydantic v1 以兼容 LangChain 的结构化输出：

```python
from langchain_core.pydantic_v1 import BaseModel as V1BaseModel

class Node(V1BaseModel):
    id: str = Field(description="实体标识，如 'AlphaCorp'")
    type: str = Field(description="实体类型，如 Person, Company, Product")

class Relationship(V1BaseModel):
    source: Node
    target: Node
    type: str = Field(description="关系类型，如 WORKS_FOR, ACQUIRED")

class KnowledgeGraph(V1BaseModel):
    relationships: List[Relationship]
```

**图构建流程**——LLM 从非结构化文本提取三元组：

```python
# 输入：非结构化文本
doc = "2023年5月，AlphaCorp 宣布收购 BetaSolutions，后者是云数据库领域的领先企业。"

# LLM 提取实体和关系
extractor = llm.with_structured_output(KnowledgeGraph)
kg = extractor.invoke(f"从以下文本提取实体和关系：\n{doc}")
# 结果：AlphaCorp --ACQUIRED--> BetaSolutions

# 写入 Neo4j
graph.add_graph_documents(kg_to_documents(kg))
```

**Text-to-Cypher 查询**——LLM 根据图 Schema 生成查询语句：

```python
cypher_prompt = ChatPromptTemplate.from_template(
    """你是 Neo4j Cypher 专家。根据以下图 Schema 生成查询：
    Schema：{schema}
    问题：{question}
    只返回 Cypher 查询，不要其他内容。"""
)

# 链式执行：生成 Cypher → 执行 → 综合回答
chain = cypher_prompt | llm | execute_cypher | synthesize_answer
```

#### 3.3.4 实际案例

三篇企业文档构建知识图谱后的查询测试：

| 查询                                                                    | 类型 | 结果                              |
| ----------------------------------------------------------------------- | ---- | --------------------------------- |
| 「谁在 AlphaCorp 工作？」                                               | 单跳 | 返回 AlphaCorp 的所有员工节点     |
| 「AlphaCorp 收购了哪家公司？」                                          | 单跳 | BetaSolutions                     |
| 「哪些公司与收购了 BetaSolutions 的那家公司所生产的产品存在竞争关系？」 | 多跳 | 通过 ACQUIRED→MAKES→COMPETES 三跳 |

#### 3.3.5 思考与延伸

- **图构建的准确性**是整个系统的瓶颈——实体/关系提取错误会导致下游推理错误
- **Cypher 生成的鲁棒性**：LLM 可能生成语法错误的 Cypher，需要 fallback 策略
- **混合检索与推理（Hybrid Retrieval）**：对于 Cypher 返回空结果或生成失败的查询，应 fallback 到**向量检索原始文档**，把语义相近的文本片段作为补充证据，然后由 LLM 综合**图谱结果 + 文档片段**生成最终答案。这种「GraphRAG 混合模式」可以在知识图谱不完备、或图谱维护滞后于新信息的场景下显著提升鲁棒性，是 2024-2026 年生产级 KG-Augmented RAG 的主流架构。
- **与 Episodic+Semantic 的选择**：Graph World-Model 适合静态知识库的结构化推理；Episodic+Semantic 适合动态对话的个性化记忆
- **规模化挑战**：图中节点和关系增多后，LLM 需要更长的 Schema 上下文

---

## 第四部分：安全与可靠性

> 从「能用」到「敢用」——面向生产环境的安全保障架构。这一部分的四种架构代表了从自动容错到人工审核再到自我认知的安全等级递进。

### 4.1 PEV（计划-执行-验证）

#### 4.1.1 核心思想

**一句话定义**：Plan → Execute → Verify 三阶段循环，每步执行后验证结果，失败则重规划。

PEV 是 Planning（Notebook 04）的增强版。Planning 架构假设「计划一定能执行成功」，但现实中工具可能失败、API 可能超时、数据可能不存在。PEV 在 Executor 之后增加了一个 **Verifier** 节点——它检查工具返回的结果是否有效，如果发现错误，会清空当前计划并触发 Planner 重新规划。

#### 4.1.2 架构图

```mermaid
graph TD
    A["Planner<br/>生成/重规划"] --> B["Executor<br/>执行当前步骤"]
    B --> C["Verifier<br/>验证结果"]
    C -->|"验证成功 + 还有步骤"| B
    C -->|"验证成功 + 计划完成"| D["Synthesizer<br/>综合结果"]
    C -->|"验证失败 + 重试次数未满"| A
    C -->|"验证失败 + 重试已满"| D
    D --> E["END"]
```

#### 4.1.3 关键实现解析

**State 增加了 retries 计数器和 last_tool_result**：

```python
class PEVState(TypedDict):
    user_request: str
    plan: Optional[List[str]]
    last_tool_result: Optional[str]
    intermediate_steps: List[str]
    final_answer: Optional[str]
    retries: int  # 重规划次数上限
```

**Verifier Node——用 LLM 判断工具结果是否有效**：

```python
class VerificationResult(BaseModel):
    is_successful: bool = Field(description="工具调用是否返回了有用信息")
    reasoning: str = Field(description="判断理由")

def verifier_node(state):
    verifier_llm = llm.with_structured_output(VerificationResult)
    prompt = f"""判断以下工具调用结果是否成功返回了有用信息：
    查询：{state['plan'][0] if state['plan'] else '未知'}
    结果：{state['last_tool_result']}

    注意：错误消息、空结果、超时都应判定为失败。"""
    result = verifier_llm.invoke(prompt)

    if result.is_successful:
        return {"intermediate_steps": state["intermediate_steps"] + [state["last_tool_result"]]}
    else:
        # 失败：清空计划，触发重规划
        return {"plan": None, "retries": state["retries"] + 1}
```

**"Flaky" 工具模拟**——Notebook 06 故意构造了一个在特定查询上失败的工具，用于测试 PEV 的容错能力：

```python
def flaky_web_search(query: str) -> str:
    if "employee count" in query.lower():
        return "ERROR: Service temporarily unavailable"
    return TavilySearch().invoke({"query": query})
```

#### 4.1.4 Head-to-Head 对比实验

任务：「查询 Apple 的 R&D 支出和员工数量」（员工查询会失败）。

| 维度            | Basic PE（无验证）           | PEV（有验证）                                       |
| --------------- | ---------------------------- | --------------------------------------------------- |
| task_completion | 4/10                         | 8/10                                                |
| error_handling  | 2/10                         | 9/10                                                |
| 行为            | 将错误信息当有效数据写入报告 | 检测到错误 → 重规划为"Apple total workforce" → 成功 |

#### 4.1.5 思考与延伸

- **Verifier 的成本**：每次工具调用都需要一次额外的 LLM 调用做验证——在低风险场景中可以跳过
- **重规划的智能性**：好的 Planner 应该能理解「为什么失败」并生成不同的查询策略
- **生产环境中的熔断器模式**：当连续失败达到阈值时，应该直接告知用户而不是无限重试

---

### 4.2 Mental Loop / Simulator（心智模拟器）

#### 4.2.1 核心思想

**一句话定义**：在内部模拟器中「预演」行动结果，评估风险后再决定是否真实执行。

类比象棋选手在脑中推演未来几步的结果。Mental Loop 让 Agent 在一个**沙箱化的模拟环境**中测试自己的策略，如果模拟结果不理想，可以修改策略再重新模拟，直到找到满意的方案才在真实环境中执行。

#### 4.2.2 架构图

```mermaid
graph LR
    A["Observe Market<br/>观察当前状态"] --> B["Analyst<br/>提出策略"]
    B --> C["Simulator<br/>运行5次模拟"]
    C --> D["Risk Manager<br/>分析模拟结果"]
    D --> E["Executor<br/>真实执行"]
    E --> F["END"]
```

#### 4.2.3 关键实现解析

**模拟环境——MarketSimulator 使用几何布朗运动模拟股价**：

```python
class MarketSimulator:
    def __init__(self):
        self.day = 0
        self.price = 100.0
        self.volatility = 0.1
        self.drift = 0.01
        self.portfolio = Portfolio(cash=10000.0, shares=0)

    def step(self, action: str, amount: float):
        """执行一步交易并更新市场"""
        # 执行交易
        if action == "buy":
            self.portfolio.shares += amount
            self.portfolio.cash -= amount * self.price
        # 更新价格（几何布朗运动）
        daily_return = np.random.normal(self.drift, self.volatility)
        self.price *= (1 + daily_return)
        self.day += 1
```

**Simulator Node——在模拟器的深拷贝上运行多次模拟**：

```python
import copy

def run_simulation_node(state):
    results = []
    for i in range(5):  # 5 次独立模拟
        sim = copy.deepcopy(state["real_market"])  # 深拷贝，不影响真实环境
        initial_value = sim.portfolio.value(sim.price)

        # 模拟 10 天
        for day in range(10):
            sim.step(action=translate_strategy(state["proposed_action"]), amount=...)

        final_value = sim.portfolio.value(sim.price)
        results.append({
            "return_pct": (final_value - initial_value) / initial_value * 100
        })
    return {"simulation_results": results}
```

**Risk Manager——分析模拟结果的方差，可能否决激进策略**：

```python
class FinalDecision(BaseModel):
    action: str = Field(description="buy/sell/hold")
    amount: float = Field(description="交易数量")
    reasoning: str

def refine_and_decide_node(state):
    refiner_llm = llm.with_structured_output(FinalDecision)
    results_summary = summarize_simulation_results(state["simulation_results"])
    prompt = f"""你是风险管理者。分析师建议：{state['proposed_action']}
    5次模拟结果：{results_summary}
    如果结果方差很大，应该降低仓位或转为持有。"""
    decision = refiner_llm.invoke(prompt)
    return {"final_decision": decision}
```

#### 4.2.4 实际案例

多日交易模拟：

- **Day 1**：好消息 → 分析师建议「积极买入」 → 5 次模拟中 4 次盈利 → Risk Manager 同意买入但降低仓位
- **Day 3**：坏消息 → 分析师建议「卖出」 → 5 次模拟中 3 次亏损 → Risk Manager 确认卖出

#### 4.2.5 思考与延伸

- **模拟器保真度**是关键——如果模拟环境与现实差距大，预演结果毫无意义
- **保真度与成本的根本权衡**：存在两种极端——低成本低保真（直接用 LLM 作为模拟器，依赖世界知识常识推理） vs 高成本高保真（调用专业领域模型、历史数据回放、区块链仿真）。**架构设计之初就必须明确选择**：用于低风险「前置筛查」时 LLM 模拟就够；用于终端决策（如实盘交易、自动驾驶规划）时必须误差有界的高保真模拟。**错误选择的影响**不是性能问题，而是给 Agent 植入了一个错误的世界模型，它后续的一切决策都将建立在这个错误基础之上。
- **「LLM 即模拟器」**：可以直接用 LLM 的世界知识作为模拟器，无需显式环境建模（但保真度不可期）
- **与 Dry-Run 的区别**：Mental Loop 是自动化的模拟+决策；Dry-Run 是展示给人看的预览+人工审批

---

### 4.3 Dry-Run Harness（沙箱预演）

#### 4.3.1 核心思想

**一句话定义**：Agent 提出行动 → 沙箱模拟执行 → 展示结果给人类 → 人类批准后真实执行。

Dry-Run 是 **Human-in-the-Loop（HITL）** 的标准实现。与 Mental Loop 的自动化模拟不同，Dry-Run 明确要求**人类审核和批准**，适用于不可逆操作（社交媒体发帖、数据删除、金融交易）。

#### 4.3.2 架构图

```mermaid
graph TD
    A["Propose Post<br/>LLM 起草内容"] --> B["Dry Run<br/>沙箱执行+人类审核"]
    B -->|"approve"| C["Live Execute<br/>真实发布"]
    B -->|"reject"| D["Abort<br/>中止"]
    C --> E["END"]
    D --> E
```

#### 4.3.3 关键实现解析

**Mock 社交媒体 API 支持 dry_run 模式**：

```python
class SocialMediaAPI:
    def publish_post(self, post: SocialMediaPost, dry_run: bool = True):
        if dry_run:
            return {
                "status": "DRY_RUN_SUCCESS",
                "log": f"[DRY RUN] 将发布：{post.content}",
                "proposed_post": post.content
            }
        else:
            return {
                "status": "LIVE_SUCCESS",
                "post_id": f"post_{uuid.uuid4().hex[:8]}"
            }
```

**Human Review Node——交互式审批**：

```python
def dry_run_review_node(state):
    # 沙箱执行
    result = social_media_tool.publish_post(state["proposed_post"], dry_run=True)

    # 展示给人类
    console.print(Panel(result["log"], title="Dry Run 预览"))

    # 等待人类输入
    decision = console.input("输入 'approve' 发布，'reject' 取消：")
    return {"review_decision": decision, "dry_run_log": result["log"]}
```

**条件路由**：

```python
graph.add_conditional_edges(
    "dry_run_review",
    lambda s: "execute" if s["review_decision"] == "approve" else "reject",
    {"execute": "execute_live", "reject": "post_rejected"}
)
```

#### 4.3.4 实际案例

- **安全帖子**：LLM 起草产品公告 → Dry Run 展示预览 → 用户输入 "approve" → 成功发布
- **风险帖子**：LLM 起草的内容措辞不当 → Dry Run 展示 → 用户输入 "reject" → 中止发布

#### 4.3.5 思考与延伸

- **HITL 的可扩展性问题**：每条操作都需要人工审批，无法处理高频场景
- **自动化审核**：可以用另一个 LLM 替代人类做初步审核，只将高风险操作上报人类
- **与 PEV 的组合**：PEV 处理工具层面的自动容错，Dry-Run 处理业务层面的人工审核

---

### 4.4 Reflexive Metacognitive（反思性元认知）

#### 4.4.1 核心思想

**一句话定义**：Agent 维护自我能力模型（Self-Model），根据置信度选择策略——直接回答、使用工具或上报人类。

这是本项目中最「高级」的安全架构。受认知科学中**元认知**（metacognition）概念启发——「知道自己知道什么，知道自己不知道什么」。Agent 在回答前先进行自我反思：这个问题在我的知识范围内吗？我有多大把握回答正确？如果把握不够，应该用工具辅助还是直接请求人类帮助？

#### 4.4.2 架构图

```mermaid
graph TD
    A["User Query"] --> B["Metacognitive Analysis<br/>自我能力评估"]
    B -->|"高置信度"| C["Reason Directly<br/>直接回答"]
    B -->|"中置信度 + 有工具"| D["Call Tool<br/>调用工具"]
    B -->|"低置信度 / 高风险"| E["Escalate<br/>上报人类"]
    D --> F["Synthesize Tool Response<br/>综合工具结果"]
    C --> G["END"]
    F --> G
    E --> G
```

#### 4.4.3 关键实现解析

**Self-Model——显式编码 Agent 的能力和局限**：

```python
class AgentSelfModel(BaseModel):
    name: str = "TriageBot-3000"
    role: str = "基础医疗分诊助手"
    knowledge_domain: List[str] = [
        "common_cold", "influenza", "allergies", "headaches", "basic_first_aid"
    ]
    available_tools: List[str] = ["drug_interaction_checker"]
    confidence_threshold: float = 0.6
```

**Metacognitive Analysis——LLM 自我反思**：

```python
class MetacognitiveAnalysis(BaseModel):
    confidence: float = Field(description="置信度 0.0-1.0")
    strategy: str = Field(description="reason_directly / use_tool / escalate")
    reasoning: str = Field(description="选择该策略的理由")
    tool_to_use: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None

def metacognitive_analysis_node(state):
    analysis_llm = llm.with_structured_output(MetacognitiveAnalysis)
    prompt = f"""你是 {state['self_model'].name}。
    你的知识范围：{state['self_model'].knowledge_domain}
    你的可用工具：{state['self_model'].available_tools}

    **安全第一**：
    - 潜在急症（胸痛、呼吸困难、严重出血）→ 必须 escalate
    - 超出知识范围 → 必须 escalate
    - 有任何不确定 → 倾向 escalate

    用户问题：{state['user_query']}

    分析你对这个问题的置信度和最佳应对策略。"""
    analysis = analysis_llm.invoke(prompt)
    return {"metacognitive_analysis": analysis}
```

**三策略路由**：

```python
graph.add_conditional_edges(
    "metacognitive_analysis",
    lambda s: s["metacognitive_analysis"].strategy,
    {
        "reason_directly": "reason_directly",
        "use_tool": "call_tool",
        "escalate": "escalate_to_human"
    }
)
```

#### 实际案例

| 用户问题                         | 置信度 | 策略            | 理由                 |
| -------------------------------- | ------ | --------------- | -------------------- |
| 「感冒的常见症状是什么？」       | 0.9    | reason_directly | 在知识范围内，低风险 |
| 「布洛芬和赖诺普利能一起吃吗？」 | 0.5    | use_tool        | 需要药物交互检查工具 |
| 「我胸口很痛，怎么办？」         | 0.1    | escalate        | 潜在急症，必须转人工 |

#### 4.4.4 思考与延伸

- **LLM 的过度自信问题：元认知架构的阿喀琉斯之踵**——一个无法准确自我评估的「元认知」毫无意义。仅靠 Prompt 工程很难从根本上解决这一问题。生产级缓解方案包括：
  - **判断前逻辑校准**（Reasoning Calibration）：要求模型在给出置信度前先**划分正反证据**（「支持我有把握的因素 / 质疑我有把握的因素」），让置信度基于演绎而非直觉；
  - **多模型自评投票**：用 ≥ 3 个异构模型独立评估同一 query 的置信度，取中位数或最保守值，降低单模型偏见；
  - **基于历史表现的统计式置信度**（Stats-based Confidence）：维护一张 `(query类型, 历史准确率)` 查表，Agent 的「自信」实际由过去在同类问题上的表现决定，而非当前 prompt 下的自白；
  - **保守先验**：在 Schema 中硬性约束 `confidence ≤ 0.9 除非命中安全白名单`，强制默认低估。
    在 Nebula-1 等医疗 Agent 产品中，上述策略组合使主动质疑上报率从 30% 提升到 80%，过度自信的误答率降低了一个数量级。这一指标的巨大跃迁印证了一个核心结论：**工程化校准**（多模型投票 + 历史统计 + Schema 约束）比单纯依赖 Prompt 工程更具决定性。
- **Self-Model 的动态更新**：随着工具库扩展或知识增长，Self-Model 应该能自动更新
- **与 Ensemble 的组合**：多个 Metacognitive Agent 投票，进一步提升置信度校准

---

## 第五部分：学习与自适应

> 从静态工具到持续进化——让 Agent 从经验中学习，或通过简单规则涌现复杂行为。

### 5.1 RLHF Self-Improvement（自我改进循环）

#### 5.1.1 核心思想

**一句话定义**：Generate → Critique → Revise 多轮循环 + 优质输出持久化为未来任务的 Few-Shot 示例。

Self-Improvement 实现了**两层改进**：

1. **任务内改进**（Self-Refine）：在当前任务中反复修改直到质量达标
2. **跨任务改进**（Few-Shot Memory）：将通过审核的高质量输出存入 GoldStandardMemory，下一次任务时作为 few-shot 示例

类比 RLHF：Generator 对应 Policy Model，Critique 的打分对应 Reward Model，修改过程对应 Policy Gradient 优化。

#### 5.1.2 架构图

```mermaid
graph TD
    A["Generator<br/>Junior Copywriter"] --> B["Critic<br/>Senior Editor"]
    B -->|"score >= 8 或已修改3次"| C["Save to GoldStandardMemory"]
    B -->|"score < 8 且未满3次"| D["Reviser<br/>基于反馈修改"]
    D --> B
    C --> E["END"]
```

#### 5.1.3 关键实现解析

**Critique Schema 包含打分和具体反馈**：

```python
class MarketingEmail(BaseModel):
    subject: str = Field(description="邮件标题")
    body: str = Field(description="邮件正文")

class Critique(BaseModel):
    score: int = Field(description="质量评分 1-10")
    feedback_points: List[str] = Field(description="具体改进建议")
    is_approved: bool = Field(description="score >= 8 则为 True")
```

**GoldStandardMemory——持久化高质量输出**：

```python
class GoldStandardMemory:
    def __init__(self):
        self.examples: List[MarketingEmail] = []

    def add_example(self, email: MarketingEmail):
        self.examples.append(email)

    def get_formatted_examples(self) -> str:
        if not self.examples:
            return "暂无历史优秀案例。"
        return "\n---\n".join(
            f"标题: {e.subject}\n正文: {e.body}" for e in self.examples
        )

gold_memory = GoldStandardMemory()
```

**条件路由——质量达标或达到最大修改次数则终止**：

```python
def should_continue(state) -> str:
    if state["critique"].is_approved:
        return "end"
    if state["revision_number"] >= 3:
        return "end"  # 最多修改3次
    return "revise"
```

**Phase 2——带记忆的 Generator 使用 few-shot 示例**：

```python
def generate_node_with_memory(state):
    examples = gold_memory.get_formatted_examples()
    prompt = f"""你是初级文案。以下是通过审核的优秀案例供参考：
    {examples}

    为以下产品撰写营销邮件：{state['user_request']}"""
    # ...
```

#### 5.1.4 实际案例

| 轮次                 | 任务                   | 初始分数 | 最终分数 | 修改次数 |
| -------------------- | ---------------------- | -------- | -------- | -------- |
| 第 1 轮              | InsightSphere 产品邮件 | 4/10     | 8/10     | 3        |
| 第 2 轮（有 memory） | Visionary CRM 邮件     | 6/10     | 9/10     | 2        |

第 2 轮因为有了第 1 轮的优秀案例作为 few-shot，初始质量更高，收敛更快。

#### 5.1.5 思考与延伸

- **与 Reflection（Notebook 01）的对比**：Reflection 只有一轮批评+改进；RLHF 支持多轮循环直到达标 + 跨任务学习
- **GoldStandardMemory 的检索策略**：当示例很多时，应该用向量检索找到最相关的示例，而非全部塞进 prompt
- **在线学习风险与「金标准审核者」防护**：如果错误的示例被加入 GoldStandardMemory（如 Critic 判断失误），会污染后续生成——这种逐步漂移的质量退化极难在早期被发现。**必需的防护措施**：引入一个**金标准审核者**（Gold Standard Auditor）角色——可以是一个更强的 LLM（如 GPT-4）、人类审核员，或一条硬性测试用例集——定期对已入库的金标准样本重新打分，低于阈值则淘汰；或在入库时就要求「双 Critic 独立评判 + 金标准审核者」一致通过方可入库。这等同于 RLHF 中的 reward model 验证，用于预防奖励黑客。

---

### 5.2 Cellular Automata（元胞自动机）

#### 5.2.1 核心思想

**一句话定义**：大量简单 Agent 按局部规则交互，涌现出全局智能行为。

这是 17 种架构中最特殊的一种——**没有 LLM 参与核心推理**。受经典元胞自动机（Conway's Game of Life、Wolfram 规则）启发，系统由一个网格上的大量「细胞 Agent」组成，每个细胞只看自己的邻居，按简单规则更新状态。这些局部规则的叠加产生了**涌现行为**——复杂的全局路径规划从简单的局部交互中自然出现。

应用场景：仓库物流路径规划。

#### 5.2.2 架构图

```mermaid
graph TD
    A["初始化仓库网格<br/>标记障碍物/货架/目标点"] --> B["设置目标点<br/>pathfinding_value = 0"]
    B --> C["Tick: 所有细胞并行更新<br/>value = min(邻居值) + 1"]
    C -->|"有变化"| C
    C -->|"收敛（无变化）"| D["梯度下降寻路<br/>从起点沿递减方向移动"]
    D --> E["END"]
```

#### 5.2.3 关键实现解析

**CellAgent——每个网格单元格**：

```python
class CellAgent:
    def __init__(self, cell_type: str, item: Optional[str] = None):
        self.type = cell_type  # EMPTY, OBSTACLE, SHELF, PACKING_STATION
        self.item = item
        self.pathfinding_value = float('inf')  # 初始为无穷大

    def update_value(self, neighbors: List['CellAgent']):
        """局部规则：value = min(邻居值) + 1"""
        if self.type == "OBSTACLE":
            return  # 障碍物永远不可达
        min_neighbor = min(n.pathfinding_value for n in neighbors)
        new_value = min_neighbor + 1
        if new_value < self.pathfinding_value:
            self.pathfinding_value = new_value
```

**WarehouseGrid——网格管理与波传播**：

```python
class WarehouseGrid:
    def __init__(self, layout: List[str]):
        self.grid = np.array([[CellAgent(char_to_type(c)) for c in row] for row in layout])

    def tick(self) -> bool:
        """同步更新所有细胞，返回是否有变化"""
        changed = False
        for r in range(self.grid.shape[0]):
            for c in range(self.grid.shape[1]):
                neighbors = self.get_neighbors(r, c)
                old_val = self.grid[r][c].pathfinding_value
                self.grid[r][c].update_value(neighbors)
                if self.grid[r][c].pathfinding_value != old_val:
                    changed = True
        return changed
```

**波传播算法**：

1. 将目标点（包装站）的 `pathfinding_value` 设为 0
2. 反复调用 `tick()` 直到收敛——值从目标点向外「波浪式」扩散
3. 从起点开始，沿 `pathfinding_value` 递减方向移动，即可得到最短路径

```python
def propagate_path_wave(grid, target):
    grid.grid[target].pathfinding_value = 0
    while grid.tick():
        pass  # 重复直到收敛

def trace_path(grid, start, target):
    path = [start]
    current = start
    while current != target:
        neighbors = grid.get_neighbors(*current)
        current = min(neighbors, key=lambda n: grid.grid[n].pathfinding_value)
        path.append(current)
    return path
```

**仓库布局示例**：

```text
#################
# A   B  P      #
#   #  #    C   #
#     D          #
#################
```

`#` = 障碍物，`P` = 包装站（目标），`A/B/C/D` = 货架上的商品。

#### 5.2.4 实际案例

订单：取货品 A 和 C 送到包装站 P。

1. 第一次波传播：从 P 向外扩散，计算所有可达格子的距离值
2. 从 A 出发沿梯度下降到 P → 得到 A→P 路径
3. 重新波传播（清空后重算）
4. 从 C 出发沿梯度下降到 P → 得到 C→P 路径

LLM 仅在最后用于生成人类可读的路径描述总结。

#### 5.2.5 思考与延伸

- **与 A\* 的对比**：A\* 是集中式最短路算法；Cellular Automata 是分布式的——每个细胞只知道邻居，无需全局视图
- **优势**：天然支持动态障碍物（只需局部重新传播）；可并行化（每个细胞独立更新）
- **更复杂的涌现行为**：同样的局部规则框架可以模拟交通流、群体行为、资源分配

---

## 第六部分：横切面分析

> 超越单个架构的比较分析，帮助读者建立系统性的选型和设计能力。

### 6.1 架构分类法与选型指南

#### 6.1.1 按控制流模式分类

| 模式           | 架构                              | 特点                           | 适用场景               |
| -------------- | --------------------------------- | ------------------------------ | ---------------------- |
| **线性 DAG**   | 01 Reflection                     | 确定性强，最易调试             | 单步生成优化           |
| **条件循环**   | 02 Tool Use, 03 ReAct, 06 PEV     | 动态适应，需要 recursion_limit | 需要外部信息的开放问题 |
| **固定流水线** | 04 Planning, 05 Multi-Agent       | 可预测，步骤明确               | 结构化多步任务         |
| **扇出/扇入**  | 13 Ensemble                       | 并行提效，综合质量             | 多视角分析             |
| **动态调度**   | 07 Blackboard, 11 Meta-Controller | 灵活路由，Controller 是瓶颈    | 条件分支、多域任务     |
| **树搜索**     | 09 ToT                            | 系统化探索，计算昂贵           | 逻辑谜题、约束规划     |
| **自循环进化** | 15 RLHF                           | 持续改进，需要评估标准         | 内容创作、质量迭代     |
| **去中心化**   | 16 Cellular Automata              | 涌现行为，不依赖 LLM           | 空间推理、物流         |

#### 6.1.2 按 Agent 数量与通信方式分类

| 类型                  | 架构                   | 通信模式                       |
| --------------------- | ---------------------- | ------------------------------ |
| **单 Agent**          | 01, 02, 03, 04, 09, 10 | State 自传递                   |
| **多 Agent 串行**     | 05, 06                 | State 依次传递                 |
| **多 Agent 并行**     | 13                     | Fan-out → State merge → Fan-in |
| **多 Agent 动态调度** | 07, 11                 | Controller 分发                |
| **群体 Agent**        | 16                     | 邻居交互                       |
| **自我对话**          | 01, 15                 | Generator/Critic 角色分离      |

#### 6.1.3 选型决策树

```mermaid
graph TD
    A["你的任务类型？"] --> B{"需要外部信息？"}
    B -->|"不需要"| C{"单步生成足够？"}
    C -->|"是"| D["01 Reflection"]
    C -->|"否，需要多步推理"| E{"问题结构已知？"}
    E -->|"是"| F["04 Planning"]
    E -->|"否，需要回溯"| G["09 Tree of Thoughts"]

    B -->|"需要"| H{"需要几次搜索？"}
    H -->|"一次"| I["02 Tool Use"]
    H -->|"多次，结构未知"| J["03 ReAct"]
    H -->|"多次，结构已知"| K["04 Planning"]

    A --> L{"需要多角度分析？"}
    L -->|"固定流程"| M["05 Multi-Agent"]
    L -->|"条件分支"| N["07 Blackboard"]
    L -->|"并行+综合"| O["13 Ensemble"]
    L -->|"智能路由"| P["11 Meta-Controller"]

    A --> Q{"需要长期记忆？"}
    Q -->|"对话记忆"| R["08 Episodic+Semantic"]
    Q -->|"知识推理"| S["12 Graph World-Model"]

    A --> T{"高风险决策？"}
    T -->|"自动容错"| U["06 PEV"]
    T -->|"模拟预演"| V["10 Mental Loop"]
    T -->|"人工审核"| W["14 Dry-Run"]
    T -->|"自我认知"| X["17 Metacognitive"]

    A --> Y{"持续改进？"}
    Y -->|"质量迭代"| Z["15 RLHF"]
    Y -->|"空间推理"| AA["16 Cellular Automata"]
```

### 6.2 LangGraph 状态设计模式对比

| 模式          | 代表架构       | State 结构                                  | 优点                       | 缺点                                                         |
| ------------- | -------------- | ------------------------------------------- | -------------------------- | ------------------------------------------------------------ |
| **消息列表**  | 02, 03         | `Annotated[list[AnyMessage], add_messages]` | 自然对话、工具调用无缝集成 | 随对话增长消耗 token                                         |
| **字段组合**  | 01, 04, 06, 15 | `TypedDict` with named fields               | 精确控制、便于条件路由     | 需要预定义所有字段、扩展性差（新增一个中间步骤就要改 State） |
| **共享黑板**  | 07             | `List[str]` 追加模式                        | 灵活，Agent 自由读写       | 缺乏类型安全                                                 |
| **Dict 聚合** | 13             | `Dict[str, str]` 分析师报告                 | 支持并行写入               | 需要小心合并逻辑                                             |
| **外部存储**  | 08, 12         | FAISS + Neo4j                               | 跨会话持久化、关系推理     | 基础设施复杂度高                                             |

**选择建议**：

- 工具调用场景 → 消息列表
- 多阶段处理 → 字段组合
- 并行 Agent → Dict 聚合
- 需要持久化 → 外部存储

### 6.3 LLM 模型选择策略

**Temperature 设置指南**：

| Temperature | 用途                           | 架构示例           |
| ----------- | ------------------------------ | ------------------ |
| **0**       | 确定性路由、安全决策、查询生成 | 02, 03, 11, 12, 17 |
| **0.2**     | 保守生成（代码）               | 01                 |
| **0.3**     | 轻微变化（多样化视角）         | 13                 |
| **0.4**     | 适度创造（策略、营销文案）     | 09, 10, 15         |
| **0.5**     | 创意内容（社交媒体）           | 14                 |

**规律**：越需要「准确性」的场景用越低的 temperature；越需要「创造性」的场景用越高的 temperature。安全相关的决策（路由、验证、元认知分析）一律用 temperature=0。

> 🔬 **无银弹原则**：上表仅是起点估值，**实际 temperature 必须通过 A/B 测试或评估来确定**。推荐的做法：在 LLM-as-a-Judge 的评估集上，对 `temperature ∈ {0, 0.1, 0.3, 0.5, 0.7}` 运行各 N 次，取平均得分最高者为该节点的生产值；改变 Prompt 后需重新标定。

**同一 Pipeline 混合模型**：在 Notebook 05-17 中，可以让路由/验证节点用 Llama-3.1-8B（快速+便宜），推理/生成节点用 Mixtral-8x22B（强大+准确），实现成本与质量的平衡。

### 6.4 从学术架构到生产部署

#### 6.4.1 错误处理等级对比

| 等级 | 机制          | 架构             | 适用场景                           |
| ---- | ------------- | ---------------- | ---------------------------------- |
| L0   | 无处理        | 01 Reflection    | 纯 LLM 生成，无外部依赖            |
| L1   | 条件路由      | 02, 03           | 根据 LLM 输出决定下一步            |
| L2   | 验证器 + 重试 | 06 PEV           | 工具调用可能失败                   |
| L3   | 模拟器预演    | 10 Mental Loop   | 高风险决策需要预测后果             |
| L4   | 人工审核      | 14 Dry-Run       | 不可逆操作需要人类批准             |
| L5   | 元认知自评    | 17 Metacognitive | 安全关键领域，Agent 知道自己的局限 |

#### 6.4.2 架构组合建议

单一架构很少能满足生产需求。以下是一些有价值的组合模式：

| 组合                             | 效果                            | 适用场景           |
| -------------------------------- | ------------------------------- | ------------------ |
| **PEV + Metacognitive**          | 自动容错 + 安全路由             | 高可靠性系统       |
| **Ensemble + RLHF**              | 多视角 + 质量迭代               | 自校准分析系统     |
| **Blackboard + Episodic Memory** | 动态协作 + 长期记忆             | 持续学习的协作系统 |
| **Meta-Controller + ReAct**      | 智能路由 + 每个专家内部多轮推理 | 多域助手           |
| **Planning + PEV + Dry-Run**     | 规划 + 自动验证 + 人工审核      | 企业级自动化流程   |

#### 6.4.3 可观测性

所有 Notebook 都集成了 LangSmith 追踪。在生产环境中，这意味着：

- 每次 LLM 调用的输入/输出/延迟都可以追踪
- 工具调用的成功/失败率可以监控
- Agent 的决策路径可以回溯和审计
- 评估分数可以作为系统健康指标

---

## 第七部分：结语与展望

### 7.1 17 种架构一览表

| #   | 架构              | 一句话定义              | 核心创新                 |
| --- | ----------------- | ----------------------- | ------------------------ |
| 01  | Reflection        | 生成→批评→改进          | 无需外部反馈的自我优化   |
| 02  | Tool Use          | LLM 自主调用外部 API    | 从封闭到开放系统         |
| 03  | ReAct             | 多轮推理+行动循环       | 动态多跳问题求解         |
| 04  | Planning          | 先规划后执行            | 结构化任务分解           |
| 05  | Multi-Agent       | 专家团队分工            | 专业深度超越通用广度     |
| 06  | PEV               | 执行后验证+重规划       | 工具失败自动容错         |
| 07  | Blackboard        | 共享黑板+动态调度       | 条件分支多 Agent 协调    |
| 08  | Episodic+Semantic | 向量+图谱双记忆         | 跨会话个性化             |
| 09  | Tree of Thoughts  | 树搜索+剪枝             | 系统化探索带回溯         |
| 10  | Mental Loop       | 模拟器预演              | 高风险决策前测试         |
| 11  | Meta-Controller   | 智能路由                | 最低开销多 Agent 调度    |
| 12  | Graph World-Model | 知识图谱+Text-to-Cypher | 多跳关系推理             |
| 13  | Ensemble          | 并行多视角+综合         | 认知多样性降低偏差       |
| 14  | Dry-Run           | 沙箱预览+人工审核       | 不可逆操作安全门         |
| 15  | RLHF              | 多轮修改+持久化记忆     | 跨任务质量提升           |
| 16  | Cellular Automata | 局部规则涌现全局行为    | 去中心化空间推理         |
| 17  | Metacognitive     | 自我能力评估+策略路由   | Agent 知道自己不知道什么 |

### 7.2 演进脉络

回顾这 17 种架构，可以看到一条清晰的能力升级链：

1. **输出质量**：单次生成（baseline）→ Reflection（单轮改进）→ RLHF（多轮+记忆）
2. **信息获取**：封闭系统 → Tool Use（单次）→ ReAct（多轮）→ Planning（结构化）
3. **协作规模**：单 Agent → Multi-Agent（流水线）→ Blackboard（动态）→ Ensemble（并行）
4. **安全保障**：无保护 → PEV（自动验证）→ Mental Loop（模拟）→ Dry-Run（人工）→ Metacognitive（自我认知）
5. **推理深度**：线性链 → Tree of Thoughts（树搜索）→ Graph World-Model（图推理）
6. **进化能力**：静态 → RLHF（经验学习）→ Cellular Automata（涌现行为）

### 7.3 2026 年 Agent 架构前沿趋势

- **多模态 Agent**：不仅处理文本，还能理解图像、视频、音频，并操作 GUI
- **Agent-to-Agent 协议**：MCP（Model Context Protocol）和 A2A（Agent-to-Agent）标准化了 Agent 之间的通信
- **自主 Agent OS**：Agent 作为操作系统级别的存在，管理工具、记忆、安全策略
- **可验证 Agent**：形式化方法确保 Agent 行为在安全边界内，而非仅靠 prompt 约束

---

## 附录

### A. 环境配置完整指南

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/all-agentic-architectures.git
cd all-agentic-architectures

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# .\venv\Scripts\activate  # Windows

# 3. 安装基础依赖
pip install -r requirements.txt

# 4. 安装特定 Notebook 的额外依赖
pip install faiss-cpu neo4j tiktoken langchain-openai  # Notebook 08, 12
pip install numpy  # Notebook 10, 16

# 5. 配置 .env 文件
cat > .env << 'EOF'
NEBIUS_API_KEY="your_nebius_api_key"
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="All-Agentic-Architectures"
TAVILY_API_KEY="your_tavily_api_key"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your_neo4j_password"
EOF

# 6. （可选）启动 Neo4j（Notebook 08, 12 需要）
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_neo4j_password \
  neo4j:latest

# 7. 启动 Jupyter
jupyter notebook
```

### B. 17 个架构 State 定义速查表

| #   | 架构            | State 类名        | 核心字段                                                          |
| --- | --------------- | ----------------- | ----------------------------------------------------------------- |
| 01  | Reflection      | `ReflectionState` | user_request, draft, critique, refined_code                       |
| 02  | Tool Use        | `AgentState`      | messages（add_messages 归并器）                                   |
| 03  | ReAct           | `AgentState`      | messages（add_messages 归并器）                                   |
| 04  | Planning        | `PlanningState`   | user_request, plan, intermediate_steps, final_answer              |
| 05  | Multi-Agent     | `MultiAgentState` | user_request, news/technical/financial_report, final_report       |
| 06  | PEV             | `PEVState`        | user_request, plan, last_tool_result, intermediate_steps, retries |
| 07  | Blackboard      | `BlackboardState` | user_request, blackboard, available_agents, next_agent            |
| 08  | Memory          | `AgentState`      | user_input, retrieved_memories, generation                        |
| 09  | ToT             | `ToTState`        | problem_description, active_paths, solution                       |
| 10  | Mental Loop     | `AgentState`      | real_market, proposed_action, simulation_results, final_decision  |
| 11  | Meta-Controller | `MetaAgentState`  | user_request, next_agent_to_call, generation                      |
| 12  | Graph           | N/A (chain)       | Neo4j graph + Cypher chain                                        |
| 13  | Ensemble        | `EnsembleState`   | query, analyses (Dict), final_recommendation                      |
| 14  | Dry-Run         | `AgentState`      | user_request, proposed_post, dry_run_log, review_decision         |
| 15  | RLHF            | `AgentState`      | user_request, draft_email, critique, revision_number              |
| 16  | Cellular        | N/A (simulation)  | WarehouseGrid + CellAgent (numpy)                                 |
| 17  | Metacognitive   | `AgentState`      | user_query, self_model, metacognitive_analysis, tool_output       |

### C. 中英术语对照表

| 英文                 | 中文             | 说明                                       |
| -------------------- | ---------------- | ------------------------------------------ |
| Agent                | 智能体 / Agent   | 能自主决策和行动的 AI 系统                 |
| Agentic Architecture | 智能体架构       | Agent 的结构设计模式                       |
| Reflection           | 反思 / 自我批判  | LLM 审查自己的输出                         |
| ReAct                | 推理+行动        | Reasoning + Acting 的缩写                  |
| Tool Use             | 工具使用         | LLM 调用外部 API                           |
| Planning             | 规划             | 先制定计划再执行                           |
| Multi-Agent          | 多智能体         | 多个 Agent 协作                            |
| Blackboard           | 黑板系统         | 共享状态的协作模式                         |
| Meta-Controller      | 元控制器         | 路由到专家的调度器                         |
| Ensemble             | 集成             | 多个独立分析的综合                         |
| Episodic Memory      | 情景记忆         | 事件/经历的向量存储                        |
| Semantic Memory      | 语义记忆         | 结构化知识的图存储                         |
| Tree of Thoughts     | 思维树           | 树搜索式推理                               |
| Graph World-Model    | 图世界模型       | 知识图谱 + 图查询                          |
| PEV                  | 计划-执行-验证   | Plan-Execute-Verify                        |
| Mental Loop          | 心智模拟器       | 模拟器内预演                               |
| Dry-Run              | 沙箱预演         | 模拟执行+人工审核                          |
| Metacognitive        | 元认知           | 自我能力评估                               |
| RLHF                 | 人类反馈强化学习 | Reinforcement Learning from Human Feedback |
| Cellular Automata    | 元胞自动机       | 局部规则涌现全局行为                       |
| LLM-as-a-Judge       | LLM 作为裁判     | 用 LLM 评估 Agent 输出                     |
| StateGraph           | 状态图           | LangGraph 的核心编排原语                   |
| Structured Output    | 结构化输出       | LLM 返回 Pydantic 对象                     |
| Conditional Edge     | 条件边           | 根据 State 决定下一步                      |
| Fan-out / Fan-in     | 扇出 / 扇入      | 并行执行后合并                             |
| Human-in-the-Loop    | 人在回路中       | 人工审核/批准机制                          |
| Few-Shot             | 少样本           | 用少量示例引导 LLM                         |
| Knowledge Graph      | 知识图谱         | 实体-关系-实体的图结构                     |
| Cypher               | Cypher 查询语言  | Neo4j 的查询语言                           |
| Recursion Limit      | 递归限制         | 防止图执行无限循环                         |
| Gold Standard Memory | 金标准记忆       | 存储高质量输出作为示例                     |
