# 使用 LangChain 实现智能对话机器人的记忆功能（LangChain 0.3+ / LangGraph 版）

在人工智能快速发展的今天，构建能够记住对话历史、理解上下文的智能对话机器人已成为一个重要需求。传统的大语言模型虽然具备强大的语言理解和生成能力，但在多轮对话中往往缺乏持续的记忆能力，无法有效维护对话的连贯性和个性化体验。

本文将深入探讨如何使用 LangChain 框架实现智能对话机器人的记忆功能，从 AI Agent 记忆系统的理论基础到 **LangChain 0.3+ 推荐的 LangGraph 记忆方案**，再到实际应用案例，为开发者提供完整的技术指南和可运行的代码示例。

> **重要更新（LangChain 0.3+ / 1.x）**
>
> 从 LangChain 0.3 起，`ConversationBufferMemory` / `ConversationSummaryMemory` / `ConversationBufferWindowMemory` / `ConversationSummaryBufferMemory` 以及 `ConversationChain` 均已被官方标记为 **deprecated**；从 LangChain **1.0** 起，这几个类已从主包 `langchain` 所在的命名空间整体迁出，转移至独立的 `langchain-classic` 包（导入路径由 `langchain.memory` / `langchain.chains` 变为 `langchain_classic.memory` / `langchain_classic.chains`）。官方推荐迁移到 **LangGraph 持久化**：以 `MessagesState` 定义对话状态，以 `MemorySaver` / `SqliteSaver` 作为 checkpointer，以 `thread_id` 实现会话隔离。
>
> 本文将旧 API 作为"迁移参考"保留一节，其余内容以 LangGraph 方案为主线重写，并在第 2.3 节给出逐项迁移对照。

---

## 1. AI Agent Memory 简介

### 1.1 什么是 AI Agent Memory

AI Agent Memory（智能体记忆）是指智能体在与环境交互过程中，存储、管理和检索历史信息的能力。与传统的大语言模型（LLM）不同，具备记忆功能的 AI Agent 能够：

- **持续学习**：从历史交互中积累经验和知识
- **上下文连贯**：维持长期对话的一致性和连贯性
- **个性化服务**：根据用户历史偏好提供定制化响应
- **任务延续**：在多轮交互中保持任务状态和进度

### 1.2 记忆系统的分层架构

参考 MemoryOS 等先进框架的设计理念，AI Agent 记忆系统通常采用分层架构：

- **短期记忆（STM）**：存储当前会话的即时信息，对应 LangGraph checkpointer 下的 `thread` 状态
- **中期记忆（MTM）**：保存近期的重要交互历史，通常以"摘要 + 窗口"形式体现
- **长期记忆（LTM）**：持久化存储关键知识和经验，在 LangGraph 中对应 `BaseStore`（本文不展开）

### 1.3 记忆系统的核心挑战

- **Token 限制**：LLM 输入长度的物理限制
- **信息检索**：从大量历史数据中快速定位相关信息
- **知识更新**：动态更新和维护记忆内容
- **成本控制**：平衡记忆容量与计算成本

---

## 2. LangChain Memory API 介绍

LangChain 的记忆能力在 0.3 版本前后经历了一次结构性重构。下面先看一眼"经典 API"（已弃用，仅作迁移参考），再重点介绍"现代方案"。

### 2.1 [已弃用] 经典 `ConversationXxxMemory`

下面四种是 0.2 之前最常见的写法，在 0.3 以后执行时会打印 `LangChainDeprecationWarning`；到 **1.0** 时这些类已完全移出主包，如想在 1.x 环境运行 Legacy 示例需先 `pip install langchain-classic`，并把下方代码段中的 `from langchain.memory` / `from langchain.chains` 换成 `from langchain_classic.memory` / `from langchain_classic.chains`。保留这些介绍只是为了帮助老代码读者理解含义；新项目请直接跳到 2.2。

#### 2.1.1 [已弃用] ConversationBufferMemory

`ConversationBufferMemory` 是最基础的记忆类型，它将所有的对话历史完整保存在内存中。这种方式简单直接，能够保持完整的对话上下文，但随着对话的进行，内存占用会不断增加。

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory(return_messages=True, memory_key="history")
conversation = ConversationChain(llm=llm, memory=memory)
response = conversation.predict(input="你好，我叫张三")
```

> 迁移目标：2.2 节的 `demo_langgraph_full_history`（`MessagesState + MemorySaver`）。

#### 2.1.2 [已弃用] ConversationSummaryMemory

`ConversationSummaryMemory` 通过 LLM 自动总结对话内容，将长对话压缩为简洁的摘要。

```python
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm, return_messages=True, memory_key="history")
```

> 迁移目标：2.2 节的 `demo_langgraph_summary`（`summarize_node` + `RemoveMessage`）。

#### 2.1.3 [已弃用] ConversationBufferWindowMemory

`ConversationBufferWindowMemory` 维护一个固定大小的滑动窗口，只保留最近的 k 轮对话。

```python
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=2, return_messages=True, memory_key="history")
```

> 迁移目标：2.2 节的 `demo_langgraph_windowed_history`（`trim_messages(strategy="last", max_tokens=k)`）。

#### 2.1.4 [已弃用] ConversationSummaryBufferMemory

`ConversationSummaryBufferMemory` 是一种混合策略，结合摘要记忆和缓冲记忆：当对话历史超过 token 限制时，较早的对话会被总结，最近的对话保持原始格式。

```python
from langchain.memory import ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=100, return_messages=True, memory_key="history"
)
```

> 迁移目标：2.2 节的 `demo_langgraph_summary_buffer`（`trim_messages` + `summarize_node`）。

### 2.2 现代方案：LangGraph 短期记忆

LangGraph 把对话"状态"从 `Memory` 对象里解放出来，用一张状态图 + 一个 checkpointer 承载所有会话历史。核心要素如下：

- **`MessagesState`**：内置 `messages: List[BaseMessage]` 字段和 `add_messages` reducer，每次节点返回新消息时会自动合并、去重、按 `id` 删除；
- **checkpointer**：`MemorySaver`（进程内，演示用）或 `SqliteSaver`（跨进程，生产可用）；
- **`thread_id`**：一个逻辑会话一把钥匙，不同 `thread_id` 之间状态完全隔离，不需要额外维护 `Dict[session_id, Memory]`；
- **`trim_messages`**：把"只喂给模型最近 N 条"或"按 token 预算裁剪"做成纯函数；
- **`RemoveMessage`**：把"摘要归档后删除旧消息"做成 reducer 级别的显式删除。

#### 2.2.1 最小可运行示例：完整缓冲

```python
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

def chat_node(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = (
    StateGraph(MessagesState)
    .add_node("chat", chat_node)
    .add_edge(START, "chat")
    .add_edge("chat", END)
    .compile(checkpointer=MemorySaver())
)

cfg = {"configurable": {"thread_id": "user_42"}}
graph.invoke({"messages": [HumanMessage("你好，我叫张三")]}, config=cfg)
graph.invoke({"messages": [HumanMessage("还记得我叫什么吗？")]}, config=cfg)
```

> 实现示例：[basic_memory_examples.py](code/basic_memory_examples.py) 中的 `demo_langgraph_full_history()`。

#### 2.2.2 窗口策略：`trim_messages`

```python
from langchain_core.messages import trim_messages

def chat_node(state: MessagesState) -> dict:
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,   # 按"消息条数"计数，简单直观
        max_tokens=6,        # 只保留最近 6 条
        start_on="human",
        include_system=True,
        allow_partial=False,
    )
    return {"messages": [llm.invoke(trimmed)]}
```

checkpointer 里仍然保留完整历史，只是每次调用 LLM 时"喂"一个窗口。相比老版的 `ConversationBufferWindowMemory`，可以随时改策略而不丢状态。

> 实现示例：`demo_langgraph_windowed_history()`。

#### 2.2.3 摘要策略：`summarize_node` + `RemoveMessage`

```python
from langchain_core.messages import RemoveMessage, SystemMessage

class SummaryState(MessagesState):
    summary: str

def chat_node(state: SummaryState) -> dict:
    system = [SystemMessage(f"此前对话摘要：{state['summary']}")] if state.get("summary") else []
    return {"messages": [llm.invoke(system + state["messages"])]}

def summarize_node(state: SummaryState) -> dict:
    if len(state["messages"]) <= 6:
        return {}
    prompt = "把下面对话压缩成一段中文摘要：\n" + "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))
    )
    summary = llm.invoke([HumanMessage(prompt)]).content
    to_remove = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": summary, "messages": to_remove}
```

关键点：`RemoveMessage(id=...)` 配合 `add_messages` reducer，在 checkpointer 里精准删除旧消息——这是相比老 `ConversationSummaryMemory` 最大的进步，因为它明确了"谁被删"。

> 实现示例：`demo_langgraph_summary()`。

#### 2.2.4 组合策略：摘要 + 窗口

生产环境最推荐的写法，用窗口控短期成本、用摘要保长期记忆。示意同上，只需在 `chat_node` 里先 `trim_messages`，再在 `summarize_node` 里把超过阈值的旧消息归档。

> 实现示例：`demo_langgraph_summary_buffer()`。

#### 2.2.5 会话隔离与跨进程持久化

```python
# MemorySaver：进程内，适合开发/测试
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# SqliteSaver：写入本地 SQLite，进程重启后状态仍在
from langgraph.checkpoint.sqlite import SqliteSaver
with SqliteSaver.from_conn_string("./sessions/checkpoints.sqlite") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    # 不同用户 / 会话用不同 thread_id
    graph.invoke(payload, config={"configurable": {"thread_id": "user_42:session_1"}})
```

`graph.get_state(cfg)` 读取最新状态，`graph.get_state_history(cfg)` 可以回溯到任意历史 checkpoint（"时间旅行"），方便调试与审计。

### 2.3 Legacy → Modern 迁移对照

| Legacy（已弃用）                  | Modern LangGraph 方案                                 | 等价 demo 方法                    |
| --------------------------------- | ----------------------------------------------------- | --------------------------------- |
| `ConversationBufferMemory`        | `StateGraph(MessagesState) + MemorySaver`             | `demo_langgraph_full_history`     |
| `ConversationBufferWindowMemory`  | 上面 + `trim_messages(strategy="last", max_tokens=k)` | `demo_langgraph_windowed_history` |
| `ConversationSummaryMemory`       | 上面 + `summarize_node` + `RemoveMessage(...)`        | `demo_langgraph_summary`          |
| `ConversationSummaryBufferMemory` | `trim_messages` + `summarize_node`（两者组合）        | `demo_langgraph_summary_buffer`   |

会话隔离：把 `session_id` 直接塞进 `config={"configurable": {"thread_id": session_id}}`，不再需要自建 `memories: Dict[session_id, Memory]`。

### 2.4 记忆类型选择指南

| 方案                                | 适用场景               | 优点                     | 缺点                     | 推荐度            |
| ----------------------------------- | ---------------------- | ------------------------ | ------------------------ | ----------------- |
| LangGraph + `MemorySaver`           | 开发、演示、单机短对话 | 零依赖、易调试           | 进程重启即失忆           | ★★★★（开发首选）  |
| LangGraph + `SqliteSaver`           | 单机生产、工具型应用   | 跨进程、轻量、可审计     | 受单机容量限制           | ★★★★★（生产首选） |
| LangGraph + `trim_messages`         | 多轮任务、成本敏感     | 上下文可控、不丢历史     | 窗口之外的信息需摘要兜底 | 搭配使用          |
| LangGraph + 摘要节点                | 长对话、客服、教练场景 | 保留关键事实、消息数可控 | 摘要生成本身需 LLM 调用  | 搭配使用          |
| ConversationBufferMemory 等经典 API | 已有老系统的维护       | 迁移成本低               | **已弃用**，0.3+ 会告警  | 仅维护期          |

---

## 3. 实战案例：智能客服机器人

为了展示 LangGraph 记忆能力的实际应用，我们用它重写了一个功能完整的智能客服机器人。这个案例覆盖从基础实现到高级功能的完整开发过程，包括多用户会话管理、智能记忆选择、性能监控等企业级功能。

### 3.1 系统架构设计

- **LLM 工厂**：统一管理 OpenAI / DeepSeek / MiniMax / 本地模型；
- **SessionManager**：只持有会话元数据（`SessionInfo`、`PerformanceMetrics`），对话正文全部交给 LangGraph checkpointer；
- **`build_graph(memory_type)`**：按 `buffer / window / summary_buffer` 编译三张图，共享同一个 checkpointer；
- **`CustomerServiceBot`**：业务门面，维持与老版相同的 API，方便已有 `main.py` / 前端无缝切换；
- **持久化**：`MemorySaver`（默认，演示）或 `SqliteSaver`（`--persistent`，生产）。

### 3.2 核心功能实现

#### 3.2.1 状态定义与图构建

```python
from langchain_core.messages import (
    AIMessage, HumanMessage, RemoveMessage, SystemMessage, trim_messages,
)
from langgraph.graph import END, START, MessagesState, StateGraph


class CustomerServiceState(MessagesState):
    summary: str


def build_graph(memory_type: str, llm, checkpointer, window_size=10, summarize_after=24):

    def _select_context(state: CustomerServiceState):
        messages = state["messages"]
        if memory_type in ("window", "summary_buffer"):
            messages = trim_messages(
                messages,
                strategy="last",
                token_counter=len,
                max_tokens=window_size,
                start_on="human",
                include_system=True,
                allow_partial=False,
            )
        system = [SystemMessage(f"此前对话摘要：{state['summary']}")] if state.get("summary") else []
        return system + list(messages)

    def chat_node(state):
        return {"messages": [llm.invoke(_select_context(state))]}

    def summarize_node(state):
        if len(state["messages"]) <= summarize_after:
            return {}
        prev = state.get("summary", "")
        prompt = (f"此前摘要：{prev}\n\n" if prev else "") + (
            "请把以下对话整合成一段简洁中文摘要：\n"
            + "\n".join(
                f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
                for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))
            )
        )
        summary = llm.invoke([HumanMessage(prompt)]).content
        to_remove = [RemoveMessage(id=m.id) for m in state["messages"][:-window_size]]
        return {"summary": summary, "messages": to_remove}

    builder = StateGraph(CustomerServiceState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    if memory_type == "summary_buffer":
        builder.add_node("summarize", summarize_node)
        builder.add_edge("chat", "summarize")
        builder.add_edge("summarize", END)
    else:
        builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)
```

> 完整实现：[smart_customer_service.py](code/smart_customer_service.py) 的 `SessionManager._build_graph`。

#### 3.2.2 会话管理：只管元数据，正文交给 checkpointer

```python
class SessionManager:
    def __init__(self, persistent: bool = False):
        self.sessions: Dict[str, SessionInfo] = {}          # 仅元数据
        self.performance_metrics: Dict[str, List[...]] = {} # 仅指标
        self.llm = get_llm()
        self.checkpointer = self._build_checkpointer(persistent)
        self._graphs = {                                     # 公用一个 checkpointer
            "buffer":         build_graph("buffer",         self.llm, self.checkpointer),
            "window":         build_graph("window",         self.llm, self.checkpointer),
            "summary_buffer": build_graph("summary_buffer", self.llm, self.checkpointer),
        }

    def chat(self, session_id: str, message: str, context=None):
        info = self.sessions[session_id]
        graph = self._graphs[info.memory_type]
        input_messages = []
        if context:
            input_messages.append(SystemMessage("\n".join(f"{k}: {v}" for k, v in context.items())))
        input_messages.append(HumanMessage(message))
        result = graph.invoke(
            {"messages": input_messages},
            config={"configurable": {"thread_id": session_id}},
        )
        last_ai = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        # ……记录 response_time / token_usage / memory_size / last_active / message_count
        return {"response": last_ai.content if last_ai else "(无响应)", ...}
```

和老版最大的两点不同：

1. 不再有 `self.memories: Dict[session_id, Memory]`——会话正文由 checkpointer 按 `thread_id` 自动隔离；
2. `save_session` 只负责把 `SessionInfo + PerformanceMetrics` 写到 `sessions/*.json`；`load_session` 只负责重建这些元数据。对话正文要么已经在 `MemorySaver` 的进程内字典里（重启即丢），要么已经在 `SqliteSaver` 的数据库里（重启后继续用同样的 `thread_id` 就能续上）。

#### 3.2.3 自动选择记忆策略

智能选择策略保留了下来，但只决定"用哪张图"，不再决定"怎么存"：

```python
def _auto_select_memory_type(self, user_id: str) -> str:
    user_sessions = [s for s in self.sessions.values() if s.user_id == user_id]
    if not user_sessions:
        return "buffer"
    avg_messages = sum(s.message_count for s in user_sessions) / len(user_sessions)
    if avg_messages < 10:
        return "buffer"
    elif avg_messages < 30:
        return "window"
    return "summary_buffer"
```

### 3.3 高级功能

#### 3.3.1 多用户会话隔离

不再需要显式的"用户会话表"：把 `session_id` 作为 `thread_id` 传给 LangGraph 即可，不同用户之间的状态天然隔离；同一用户的不同会话也可以共用一张图，互不干扰。

```python
cfg = {"configurable": {"thread_id": session_id}}
graph.invoke({"messages": [HumanMessage(user_input)]}, config=cfg)
```

会话元数据仍然集中维护，便于审计、列表页展示：

```python
@dataclass
class SessionInfo:
    session_id: str
    user_id: str
    created_at: datetime
    last_active: datetime
    message_count: int
    memory_type: str
    metadata: Dict[str, Any]
```

#### 3.3.2 性能监控与会话清理

```python
@dataclass
class PerformanceMetrics:
    response_time: float
    token_usage: int
    memory_size: int
    timestamp: datetime


def cleanup_inactive_sessions(self, hours: int = 24) -> None:
    cutoff = datetime.now() - timedelta(hours=hours)
    inactive = [sid for sid, info in self.sessions.items() if info.last_active < cutoff]
    for sid in inactive:
        self.save_session(sid)                    # 元数据落盘
        self.sessions.pop(sid, None)
        self.performance_metrics.pop(sid, None)
    # 如果使用 SqliteSaver，对话正文仍然保留在数据库里，未来可以重新挂载
```

`memory_size` 直接根据 `result["messages"]` 的内容长度粗算，用作 dashboard 预警。对于真实 token 计数可接入 `tiktoken` 或模型提供方 SDK。

---

## 4. 快速开始

### 4.1 环境配置

```bash
# 进入代码目录
cd /path/to/AI-fundamentals/08_agentic_system/memory/langchain/code/

# 安装依赖（LangChain 0.3+ / LangGraph 0.2+）
pip install -r requirements.txt

# 配置 LLM（二选一）
# 方式 1：复制配置文件
cp config.example.py config.py
# 编辑 config.py 设置你的 API Key

# 方式 2：设置环境变量
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # 可选
```

关键依赖（见 [requirements.txt](code/requirements.txt)）：

- `langchain>=0.3.0`、`langchain-core>=0.3.0`、`langchain-community>=0.3.0`
- `langchain-openai>=0.2.0`、`openai>=1.40.0`
- `langgraph>=0.2.50`、`langgraph-checkpoint-sqlite>=2.0.0`（启用 `--persistent` 时需要）

### 4.2 运行演示

```bash
# 检查配置
python main.py --config-check

# 基础记忆演示（Legacy + Modern 对照）
python main.py --demo basic

# 智能客服演示（LangGraph 短期记忆）
python main.py --demo customer

# LangGraph 记忆演示（MessagesState + checkpointer）
python main.py --demo langgraph

# 运行全部
python main.py --demo all

# 交互式聊天
python main.py --interactive

# 启用 SqliteSaver（跨进程短期记忆；可验证重启后仍能续上对话）
python main.py --demo customer  --persistent
python main.py --demo langgraph --persistent
python main.py --interactive    --persistent
```

### 4.3 项目文件说明

- **`config.py` / `config.example.py`**：LLM 配置管理，支持多提供商和通用参数；
- **`llm_factory.py`**：LLM 工厂，封装 OpenAI / DeepSeek / MiniMax / Ollama / 本地模型的创建；
- **`basic_memory_examples.py`**：四种 Legacy 记忆 + 四种等价的 LangGraph 实现；
- **`smart_customer_service.py`**：智能客服机器人完整实现（LangGraph 短期记忆）；
- **`langgraph_memory_example.py`**：LangGraph 现代记忆管理范本，含 `MessagesState`、摘要节点、`SqliteSaver`；
- **`main.py`**：主运行脚本，统一 CLI；
- **`requirements.txt`** / **`README.md`**：依赖清单与项目说明。

### 4.4 验证测试

```bash
# 1. 检查依赖安装
python -c "import langchain, langgraph, openai; print('dependencies ok')"

# 2. 验证配置
python main.py --config-check

# 3. 快速测试
python main.py --demo basic
```

---

## 5. 演示功能详解

### 5.1 基础记忆演示 (basic)

```bash
python main.py --demo basic
```

**功能说明**：这个演示分两部分。Part 1 运行四个 Legacy demo（每个都带 `[DEPRECATED]` 前缀，执行时会打印弃用警告，仅用于对照）；Part 2 运行四个等价的 LangGraph 实现，并打印"Legacy → Modern"迁移对照表。

**演示输出（节选）**：

```text
########################################################################
# Part 1 / Legacy API —— 已弃用，仅作迁移参考
########################################################################

============================================================
📝 [DEPRECATED] ConversationBufferMemory 演示
   迁移目标：demo_langgraph_full_history()
============================================================

👤 用户 1: 你好，我叫张三，是一名软件工程师
🤖 助手 1: 您好，张三！很高兴认识你……

########################################################################
# Part 2 / Modern LangGraph API —— 官方推荐
########################################################################

============================================================
✨ [MODERN] LangGraph 完整历史 (替代 ConversationBufferMemory)
============================================================

👤 用户 1: 你好，我叫张三，是一名软件工程师
🤖 助手 1: 您好，张三！作为软件工程师……

========================================================================
📊 Legacy ConversationXxxMemory → LangGraph 现代方案 迁移对照
========================================================================
Legacy (deprecated)               Modern LangGraph 模式                          适用场景
------------------------------------------------------------------------
ConversationBufferMemory           StateGraph(MessagesState) + MemorySaver        短对话、调试：完整上下文最简单
ConversationBufferWindowMemory     上面 + trim_messages(strategy='last', ...)     任务型多轮对话：只关注最近 N 条
ConversationSummaryMemory          上面 + summarize_node + RemoveMessage(...)     长对话、成本敏感：关键信息压缩
ConversationSummaryBufferMemory    trim_messages + summarize_node（两者组合）      生产环境默认选择：窗口控成本 + 摘要保记忆
```

### 5.2 智能客服演示 (customer)

```bash
python main.py --demo customer
python main.py --demo customer --persistent   # 使用 SqliteSaver
```

**功能说明**：模拟智能客服系统，展示多用户会话隔离（`thread_id = session_id`）、不同记忆策略（`buffer` / `window` / `summary_buffer`）下的响应时间与内存占用，并把会话元数据落盘到 `sessions/*.json`。

**演示输出（节选）**：

```text
🗂️  使用 MemorySaver（进程内 checkpointer）
✅ 会话管理器初始化成功，使用模型: ChatOpenAI

==================================================
🤖 智能客服机器人演示（LangGraph 短期记忆）
==================================================

📝 创建会话: 8a1b0c2d… (用户: user_001, 记忆类型: buffer)

👤 张三: 我想查询我的订单状态
🤖 客服: 好的，请提供您的订单号……
📊 响应时间: 1.23秒

👤 张三: 我的订单号是 ORD123456
🤖 客服: 收到，我来为您查询订单 ORD123456……
📊 响应时间: 0.97秒

==================================================
📋 会话摘要
==================================================

📊 对话摘要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👤 用户ID: user_001
🕐 创建时间: 2026-05-05 11:43:04
💬 消息数量: 4
🧠 记忆类型: buffer
⚡ 平均响应时间: 1.05秒
📝 平均Token使用: 186
```

### 5.3 LangGraph 记忆演示 (langgraph)

```bash
python main.py --demo langgraph
python main.py --demo langgraph --persistent
```

**功能说明**：展示 `MessagesState + add_messages` reducer、`summarize_node + RemoveMessage` 摘要机制，以及 `graph.get_state_history()` 的时间旅行能力。加上 `--persistent` 后会使用 `SqliteSaver`，演示会显式关闭现有连接、创建新 manager，验证"进程重启后对话仍能续上"。

**演示输出（节选）**：

```text
🗂️  checkpointer: MemorySaver (内存)
👤 用户: user_123
📱 会话: session_456

--- 第 1 轮对话 ---
👤 用户: 你好，我是新用户，想了解你们的服务
🤖 助手: 您好！很高兴为您介绍……
📊 checkpoint 消息数: 2

…（省略）…

==================================================
📋 对话历史（来自 checkpointer）
==================================================
📝 消息数量: 12
🧠 最终摘要: 用户是新用户，对 AI 感兴趣，询问了机器学习与深度学习……

🕰️  最近 5 个 checkpoint：
   checkpoint=1f5c… messages=12 summary=有
   checkpoint=1f5b… messages=10 summary=有
   checkpoint=1f5a… messages=8  summary=无
   checkpoint=1f59… messages=6  summary=无
   checkpoint=1f58… messages=4  summary=无

💡 提示：加上 --persistent 运行可验证 SqliteSaver 的跨进程持久化。
```

### 5.4 完整演示 (all)

```bash
python main.py --demo all
```

依次运行 `basic / customer / langgraph` 三类演示。

### 5.5 其他功能

- **配置检查**：`python main.py --config-check`
- **交互式聊天**：`python main.py --interactive`（加 `--persistent` 把短期记忆写入磁盘）

---

## 6. 总结与展望

### 6.1 技术要点回顾

- **经典 `ConversationXxxMemory` 已全部弃用**：在 0.3+ 的新项目里不要再新增这类代码；
- **LangGraph 短期记忆是官方迁移终点**：`MessagesState + checkpointer + thread_id` 是一个最小而完备的组合，几行代码就能把会话隔离、状态回溯、持久化一次性解决；
- **`trim_messages` 负责"喂给模型多少"**：纯函数、无副作用，可以随时调参；
- **`RemoveMessage` 负责"从 checkpointer 删掉什么"**：显式优于隐式，摘要节点配合它可以写出稳定的摘要策略；
- **`SqliteSaver` 把短期记忆延伸到生产**：进程重启、横向扩容前的最简可行方案，且与云上 `PostgresSaver` / `RedisSaver` 接口一致，后期平滑升级。

### 6.2 最佳实践建议

1. **会话隔离用 `thread_id`**：不要再写 `Dict[session_id, Memory]` 这样的结构，把 `session_id`（或 `f"{user_id}:{session_id}"`) 直接放到 `config` 里；
2. **窗口与摘要组合使用**：先 `trim_messages` 控本轮 prompt 长度，再用 `summarize_node + RemoveMessage` 控总体消息数；
3. **摘要阈值要留 buffer**：`summarize_after` 建议是 `window_size + 2` 以上，避免"刚触发摘要、下轮又触发"导致的抖动；
4. **生产环境用 `SqliteSaver` 起步**：本地、单机、跨进程都稳定，迁移到 `PostgresSaver` 只是改一行连接串；
5. **监控三件套**：`response_time`、`memory_size`（消息内容长度之和）、`len(state["messages"])`。前两者定位性能问题，后者定位摘要策略失效；
6. **Legacy 代码逐步替换**：对照第 2.3 节的迁移表，按文件灰度替换；不要追求一次性切换，新代码先写到 LangGraph 图里，老代码沿用 Legacy Memory，等所有调用路径统一后再删旧 import。

### 6.3 展望

`MessagesState` 之外，LangGraph 还提供 **`BaseStore`**（跨 `thread` 的长期记忆）与 **语义检索**（基于 embedding 的条件召回）等能力，可以进一步承载用户画像、长期偏好、知识库摘要等"跨会话信息"。把短期记忆（checkpointer）与长期记忆（Store）结合使用，就能构建出更接近人类记忆层次的 AI Agent——这是本文聚焦的"短期记忆"之外，值得下一篇专门展开的主题。
