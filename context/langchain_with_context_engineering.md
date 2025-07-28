# 基于上下文工程的 LangChain 智能体应用

上下文工程指在执行任务前为人工智能创建合理配置框架，该框架包含：

* **行为准则**：明确AI的职能定位（例如担任经济型旅行规划助手）
* **信息接入**：可访问数据库、文档或实时数据源的关键信息
* **会话记忆**：保留历史对话记录以避免重复或信息遗漏
* **工具集成**：支持调用计算器或搜索引擎等辅助功能
* **用户画像**：掌握个人偏好及地理位置等核心信息

![上下文工程](https://cdn-images-1.medium.com/max/1500/1*sCTOzjG6KP7slQuxLZUtNg.png)
*上下文工程（概念源自[LangChain](https://blog.langchain.com/context-engineering-for-agents/) 与[12Factor](https://github.com/humanlayer/12-factor-agents/tree/main) ）*

[当前AI工程师正逐步转型](https://diamantai.substack.com/p/why-ai-experts-are-moving-from-prompt) ，从提示词工程转向上下文工程，原因在于...

> 上下文工程专注于为人工智能提供合适的背景和工具，使其回答更加智能实用。

本文将探讨如何运用**LangChain**与**LangGraph**这两大构建人工智能智能体、RAG应用和LLM应用的利器，有效实施**上下文工程**以优化人工智能智能体。

本指南基于[langgchain ai](https://github.com/FareedKhan-dev/contextual-engineering-guide) 指南创建。

---

## 1. 什么是上下文工程？

大语言模型运作原理类似新型操作系统。大语言模型（LLM）如同CPU，其上下文窗口则类似RAM，充当短期记忆功能。但如同RAM的物理限制，上下文窗口存储不同信息的空间也有限。

> 正如操作系统决定RAM的存储内容，“上下文工程”的核心在于决策LLM应在上下文中保留哪些信息。

![不同上下文类型](https://cdn-images-1.medium.com/max/1000/1*kMEQSslFkhLiuJS8-WEMIg.png)

构建LLM应用时需管理多种上下文类型。上下文工程主要涵盖以下类型：

* 指令类：提示词、示例、记忆片段、工具描述
* 知识类：事实数据、存储信息、记忆库
* 工具类：工具调用的反馈与执行结果

今年因大语言模型在思维链与工具调用能力上的提升，智能体技术正获得更多关注。智能体群组通过协同LLM与工具处理长周期任务，并依据工具反馈动态决策后续操作。

![智能体工作流](https://cdn-images-1.medium.com/max/1500/1*Do44CZkpPYyIJefuNQ69GA.png)

但冗长任务和从工具收集过多反馈会消耗大量令牌。这将导致多重问题：上下文窗口可能溢出，成本与延迟增加，且智能体性能可能下降。

Drew Breunig 阐述了过多上下文损害性能的机制，包括：

* 上下文污染：[当错误或幻觉内容进入上下文时](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-poisoning)
* 上下文干扰：[过量上下文混淆模型判断时](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-distraction)
* 上下文混淆：[当额外无关细节影响答案时](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-confusion)
* 上下文冲突：[当部分上下文提供相互矛盾的信息时](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-clash)

![智能体的多轮对话](https://cdn-images-1.medium.com/max/1500/1*ZJeZJPKI5jC_1BMCoghZxA.png)

Anthropic[在其研究中](https://www.anthropic.com/engineering/built-multi-agent-research-system?ref=blog.langchain.com) 强调了其必要性：

> 智能体群组通常需要进行数百轮对话，因此谨慎管理上下文至关重要。

那么目前人们如何解决这一问题？人工智能智能体的上下文工程常用策略可归纳为四大主要类型：

* 编写：创建清晰有效的上下文
* 筛选：仅选取最相关信息
* 压缩：缩短上下文以节省空间
* 隔离：保持不同类型上下文独立

![上下文工程分类](https://cdn-images-1.medium.com/max/2600/1*CacnXVAI6wR4eSIWgnZ9sg.png)
*上下文工程分类（源自[LangChain文档](https://blog.langchain.com/context-engineering-for-agents/) ）*

[LangGraph](https://www.langchain.com/langgraph) 的设计全面支持所有这些策略。我们将通过`LangGrap`逐一解析这些组件，探究其如何优化人工智能智能体的运作效能。

## 2. LangGraph暂存区应用

如同人类通过笔记记录任务要点，智能体可利用[暂存区](https://www.anthropic.com/engineering/claude-think-tool) 实现相同功能——该机制将信息存储在上下文窗口之外，确保智能体随时调用关键数据。

![上下文工程(CE)的第一组件](https://cdn-images-1.medium.com/max/1000/1*aXpKxYt03iZPcrGkxsFvrQ.png)
*上下文工程的第一组件（摘自[LangChain文档](https://blog.langchain.com/context-engineering-for-agents/) ）*

典型案例参考[Anthropic多智能体研究系统](https://www.anthropic.com/engineering/built-multi-agent-research-system) ：

> *首席研究员制定计划后将其存入记忆，因为当上下文窗口超过200,000令牌容量时会被截断，保存计划可确保其完整性。*

暂存区集可通过不同方式实现：

* 通过[工具调用](https://www.anthropic.com/engineering/claude-think-tool) 实现[文件写入功能](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) 。
* 作为会话期间持续存在的运行时[状态对象](https://langchain-ai.github.io/langgraph/concepts/low_level/#state) 中的字段实现。

简而言之，暂存区帮助智能体在会话过程中保存重要笔记，从而高效完成任务。

在LangGraph框架中，系统同时支持[短期记忆](https://langchain-ai.github.io/langgraph/concepts/memory/#short-term-memory) （线程范围）和[长期记忆](https://langchain-ai.github.io/langgraph/concepts/memory/#long-term-memory) 机制。

* 短期记忆通过[检查点机制](https://langchain-ai.github.io/langgraph/concepts/persistence/) 在会话期间保存[智能体状态](https://langchain-ai.github.io/langgraph/concepts/low_level/#state) 。 其运作原理类似暂存区，允许在智能体运行时存储信息并在后续阶段调用。

状态对象是在图节点间传递的核心数据结构。可自定义其格式（通常采用Python字典形式）。 它充当共享暂存区，每个节点均可读取并更新特定字段。

> 我们将按需导入模块，以便以清晰的逻辑逐步学习实现过程。

为获得更清晰美观的输出效果，我们将使用Python的`pprint`模块进行美化输出，并采用`rich`库中的`Console`模块。首先导入并初始化这些模块：

```python
# 导入必要库
from typing import TypedDict  # 用于通过类型提示定义状态模式

from rich.console import Console  # 用于美化输出
from rich.pretty import pprint  # 用于美化Python对象显示

# 初始化控制台对象，实现notebook中的富文本格式化输出
console = Console()
```

接下来创建状态对象的`TypedDict`：

```python
# 使用TypedDict定义图状态模式
# 该类作为数据结构将在图的节点间传递
# 确保状态具有一致结构并提供类型提示
class State(TypedDict):
    """
    Defines the structure of the state for our joke generator workflow.

    Attributes:
        topic: The input topic for which a joke will be generated.
        joke: The output field where the generated joke will be stored.
    """

    topic: str
    joke: str
```

该状态对象将存储主题，以及要求智能体根据给定主题生成的笑话内容

## 3. 创建状态图

定义状态对象后，可通过[StateGraph](https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph) 向其写入上下文

StateGraph 是 LangGraph 构建有状态[智能体或工作流](https://langchain-ai.github.io/langgraph/concepts/workflows/) 的核心工具，可将其理解为有向图：

* 节点（nodes）代表工作流中的处理步骤。每个节点接收当前状态作为输入，更新后返回变更结果。
* 边（edges）连接节点并定义执行流向，支持线性、条件判断甚至循环路径。

接下来我们将执行以下操作：

1. 从[Anthropic模型](https://docs.anthropic.com/en/docs/about-claude/models/overview) 中选择并创建[聊天模型](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) 。
2. 将其应用于LangGraph工作流中。

```python
# 导入环境管理、显示功能和LangGraph所需的库
import getpass
import os

from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph

# --- 环境与模型配置 ---
# 设置Anthropic API密钥以验证请求
from dotenv import load_dotenv
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("Missing ANTHROPIC_API_KEY in environment")

# 初始化工作流中将使用的聊天模型
# 选用特定Claude模型并设定temperature=0以确保输出确定性
llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)
```

我们已初始化Sonnet模型。LangChain通过API支持多种开源和闭源模型，因此您可以使用其中任意模型。

现在需要创建使用该Sonnet模型生成响应的函数。

```python
# --- 定义工作流节点 ---
def generate_joke(state: State) -> dict[str, str]:
    """
    A node function that generates a joke based on the topic in the current state.

    This function reads the 'topic' from the state, uses the LLM to generate a joke,
    and returns a dictionary to update the 'joke' field in the state.

    Args:
        state: The current state of the graph, which must contain a 'topic'.

    Returns:
        A dictionary with the 'joke' key to update the state.
    """
    # 从状态中读取主题
    topic = state["topic"]
    print(f"Generating a joke about: {topic}")

    # 调用语言模型生成笑话
    msg = llm.invoke(f"Write a short joke about {topic}")

    # 将生成的笑话返回至状态
    return {"joke": msg.content}
```

此函数仅返回包含生成响应（笑话）的字典。

接下来我们将使用状态图轻松构建并编译该图。

```python
# --- 构建并编译图结构 ---
# 使用预定义的状态模式初始化新状态图
workflow = StateGraph(State)

# 将'生成笑话'函数添加为图节点
workflow.add_node("generate_joke", generate_joke)

# 定义工作流的执行路径：
# 图从START入口点开始流向'生成笑话'节点
workflow.add_edge(START, "generate_joke")
# '生成笑话'节点执行完毕后结束图流程
workflow.add_edge("generate_joke", END)

# 将工作流编译为可执行链
chain = workflow.compile()

# --- 可视化图谱 ---
# 展示编译后工作流图谱的可视化呈现
display(Image(chain.get_graph().draw_mermaid_png()))
```

![生成图谱](https://cdn-images-1.medium.com/max/1000/1*SxWwYN-oO_rG9xUFgeuB-A.png)

现在可以执行此工作流

```python
# --- 执行工作流 ---
# 使用包含主题的初始状态调用编译图谱
# `invoke`方法将运行从START节点到END节点的图谱
joke_generator_state = chain.invoke({"topic": "cats"})

# --- 显示最终状态 ---
# 打印执行后的图谱最终状态
# 这将同时显示写入状态的输入项'topic'和输出项'joke'
console.print("\n[bold blue]Joke Generator State:[/bold blue]")
pprint(joke_generator_state)

#### 输出结果 ####
{
  'topic': 'cats', 
  'joke': 'Why did the cat join a band?\n\nBecause it wanted to be the purr-cussionist!'
}
```

返回的字典本质上是智能体的笑话生成状态这个简单示例展示了如何将上下文写入状态

> 您可深入了解[检查点机制](https://langchain-ai.github.io/langgraph/concepts/persistence/) （用于保存和恢复图谱状态）及[人机协同](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) （暂停工作流以获取人工输入后继续执行）

## 4. LangGraph中的记忆写入

暂存区集支持智能体在单次会话中工作，但有时智能体需要跨多个会话保留记忆信息。

* [Reflexion](https://arxiv.org/abs/2303.11366) 引入了智能体在每轮交互后进行反思并复用自我生成提示的概念。
* [生成式智能体](https://ar5iv.labs.arxiv.org/html/2304.03442) 通过总结历史智能体反馈构建长期记忆系统。

![记忆写入](https://cdn-images-1.medium.com/max/1000/1*VaMVevdSVxDITLK1j0LfRQ.png)
*记忆写入（源自 [LangChain文档](https://blog.langchain.com/context-engineering-for-agents/) ）*

这些理念已应用于 [ChatGPT](https://help.openai.com/en/articles/8590148-memory-faq) 、[Cursor](https://forum.cursor.com/t/0-51-memories-feature/98509) 和 [Windsurf](https://docs.windsurf.com/windsurf/cascade/memories) 等产品，它们能自动从用户交互中生成长期记忆。

* 检查点机制会在每个步骤将图的[状态](https://langchain-ai.github.io/langgraph/concepts/persistence/) 保存到[线程](https://langchain-ai.github.io/langgraph/concepts/persistence/) 中。每个线程拥有唯一ID，通常代表一次交互——例如ChatGPT中的单次对话。
* 长期记忆功能支持您在不同线程间保持特定上下文。您可以保存[独立文件](https://langchain-ai.github.io/langgraph/concepts/memory/#profile) （例如用户档案）或记忆[集合](https://langchain-ai.github.io/langgraph/concepts/memory/#collection) 。
* 该功能采用键值存储的[BaseStore](https://langchain-ai.github.io/langgraph/reference/store/) 接口实现。既可如示例所示在内存中使用，也可用于[LangGraph平台部署](https://langchain-ai.github.io/langgraph/concepts/persistence/#langgraph-platform) 。

现在让我们创建一个`内存存储`，供本笔记本中多个会话共同使用。

```python
from langgraph.store.memory import InMemoryStore

# --- 初始化长期记忆存储 ---
# 创建InMemoryStore实例，它提供简单、非持久化的
# 键值存储系统，仅限当前会话使用
store = InMemoryStore()

# --- 定义组织用命名空间 ---
# 命名空间用于在存储中对相关数据进行逻辑分组
# 此处使用元组表示分层命名空间，
# 可关联用户ID和应用上下文
namespace = ("rlm", "joke_generator")

# --- 写入数据到记忆存储 ---
# 使用`put`方法将键值对保存到指定命名空间
# 此操作将上一步生成的笑话进行持久化存储，
# 使其可在不同会话或线程中检索
store.put(
    namespace,  # 目标写入命名空间
    "last_joke",  # 数据条目键名
    {"joke": joke_generator_state["joke"]},  # 待存储的值
)
```

我们将在后续章节讨论如何从命名空间中选择上下文当前可使用[search](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.search) 方法查看命名空间内的条目，确认写入成功

```python
# 搜索命名空间以查看所有存储项
stored_items = list(store.search(namespace))

# 以富文本格式显示存储项
console.print("\n[bold green]Stored Items in Memory:[/bold green]")
pprint(stored_items)

#### 输出结果 ####
[
  Item(namespace=['rlm', 'joke_generator'], key='last_joke', 
  value={'joke': 'Why did the cat join a band?\n\nBecause it wanted to be the purr-cussionist!'},
  created_at='2025-07-24T02:12:25.936238+00:00',
  updated_at='2025-07-24T02:12:25.936238+00:00', score=None)
]
```

现在，让我们将之前的所有操作整合到LangGraph工作流中

我们将使用两个参数编译工作流：

* `checkpointer` 在线程的每个步骤保存图状态
* `store` 实现跨线程的上下文保持

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

# 初始化存储组件
checkpointer = InMemorySaver()  # 用于线程级状态持久化
memory_store = InMemoryStore()  # 用于跨线程记忆存储


def generate_joke(state: State, store: BaseStore) -> dict[str, str]:
    """Generate a joke with memory awareness.
    
    This enhanced version checks for existing jokes in memory
    before generating new ones.
    
    Args:
        state: Current state containing the topic
        store: Memory store for persistent context
        
    Returns:
        Dictionary with the generated joke
    """
    # 检查长期记忆中是否存在现有笑话
    existing_jokes = list(store.search(namespace))
    if existing_jokes:
        existing_joke = existing_jokes[0].value
        print(f"Existing joke: {existing_joke}")
    else:
        print("Existing joke: No existing joke")

    # 根据主题生成新笑话
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    
    # 将新笑话存入长期记忆
    store.put(namespace, "last_joke", {"joke": msg.content})

    # 返回待添加到状态的笑话
    return {"joke": msg.content}


# 构建具备记忆能力的工作流
workflow = StateGraph(State)

# 添加支持记忆的笑话生成节点
workflow.add_node("generate_joke", generate_joke)

# 连接工作流组件
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", END)

# 同时启用检查点机制和记忆存储进行编译
chain = workflow.compile(checkpointer=checkpointer, store=memory_store)
```

很好！现在只需执行更新后的工作流，即可测试启用记忆功能的效果

```python
# 基于线程配置执行工作流
config = {"configurable": {"thread_id": "1"}}
joke_generator_state = chain.invoke({"topic": "cats"}, config)

# 以丰富格式展示工作流结果
console.print("\n[bold cyan]Workflow Result (Thread 1):[/bold cyan]")
pprint(joke_generator_state)

#### 输出结果 ####
Existing joke: No existing joke

Workflow Result (Thread 1):
{  'topic': 'cats', 
   'joke': 'Why did the cat join a band?\n\nBecause it wanted to be the purr-cussionist!'}
```

由于这是线程1，我们的AI智能体内存中没有存储现有笑话——这正是新线程的预期状态。

由于我们通过检查点机制编译了工作流，现在可以查看图的[最新状态](https://langchain-ai.github.io/langgraph/concepts/persistence/#get-state) 。

```python
# --- 检索并检查图状态 ---
# 使用`get_state`方法检索指定线程（本例中为线程"1"）的最新状态快照，
# 该操作之所以可行，是因为我们在编译图结构时启用了检查点机制
#
latest_state = chain.get_state(config)

# --- 显示状态快照 ---
# 将检索到的状态打印至控制台StateSnapshot不仅包含
# 数据（'topic', 'joke'），还包含执行元数据
console.print("\n[bold magenta]Latest Graph State (Thread 1):[/bold magenta]")
pprint(latest_state)
```

查看输出结果：

```text
### 最新状态输出 ###
Latest Graph State:

StateSnapshot(
    values={
        'topic': 'cats',
        'joke': 'Why did the cat join a band?\n\nBecause it wanted to be the purr-cussionist!'
    },
    next=(),
    config={
        'configurable': {
            'thread_id': '1',
            'checkpoint_ns': '',
            'checkpoint_id': '1f06833a-53a7-65a8-8001-548e412001c4'
        }
    },
    metadata={'source': 'loop', 'step': 1, 'parents': {}},
    created_at='2025-07-24T02:12:27.317802+00:00',
    parent_config={
        'configurable': {
            'thread_id': '1',
            'checkpoint_ns': '',
            'checkpoint_id': '1f06833a-4a50-6108-8000-245cde0c2411'
        }
    },
    tasks=(),
    interrupts=()
)
```

可见当前状态已记录我们与智能体的最近对话——本例中我们要求其讲述关于猫的笑话。让我们使用不同的ID重新运行工作流。

```python
# 使用不同线程ID执行工作流
config = {"configurable": {"thread_id": "2"}}
joke_generator_state = chain.invoke({"topic": "cats"}, config)

# 展示跨线程内存持久化的结果
console.print("\n[bold yellow]Workflow Result (Thread 2):[/bold yellow]")
pprint(joke_generator_state)

#### 输出结果 ####
Existing joke: {'joke': 'Why did the cat join a band?\n\nBecause it wanted to be the purr-cussionist!'}
Workflow Result (Thread 2):
{'topic': 'cats', 'joke': 'Why did the cat join a band?\n\nBecause it wanted to be the purr-cussionist!'}
```

可见首个线程的笑话已成功保存至内存。

> 您可通过[LangMem](https://langchain-ai.github.io/langmem/) 了解内存抽象机制，通过[Ambient智能体课程](https://github.com/langchain-ai/agents-from-scratch/blob/main/notebooks/memory.ipynb) 掌握LangGraph智能体中的内存管理概览。

## 5. 暂存区选择策略

从暂存区选择上下文的方式取决于其实现机制：

* 若属于[工具](https://www.anthropic.com/engineering/claude-think-tool) ，智能体可直接通过工具调用读取
* 若属于智能体运行时状态，开发者需自行决定每个步骤向智能体暴露的状态部分这使您能够对暴露的上下文实施细粒度控制

![上下文工程的第二组件](https://cdn-images-1.medium.com/max/1000/1*VZiHtQ_8AlNdV3HIMrbBZA.png)
*上下文工程第二组件（源自[LangChain文档](https://blog.langchain.com/context-engineering-for-agents/) ）*

在先前步骤中，我们学习了如何向LangGraph状态对象写入数据。现在我们将学习如何从状态中选择上下文，并将其传递至下游节点的大语言模型调用中。

这种选择性方法允许您精确控制大语言模型在执行过程中可见的上下文内容。

```python
def generate_joke(state: State) -> dict[str, str]:
    """Generate an initial joke about the topic.
    
    Args:
        state: Current state containing the topic
        
    Returns:
        Dictionary with the generated joke
    """
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}


def improve_joke(state: State) -> dict[str, str]:
    """Improve an existing joke by adding wordplay.
    
    This demonstrates selecting context from state - we read the existing
    joke from state and use it to generate an improved version.
    
    Args:
        state: Current state containing the original joke
        
    Returns:
        Dictionary with the improved joke
    """
    print(f"Initial joke: {state['joke']}")
    
    # 从状态中选取笑话呈现给大语言模型
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}
```

为增加复杂度，我们将在智能体中新增两个工作流：

1. 生成笑话（与之前流程相同）
2. 优化笑话（接收生成的笑话并进行改进）

此设置将帮助我们理解LangGraph中暂存区选择的运作机制。现在按照先前方式编译该工作流，并查看我们的图谱结构。

```python
# 构建包含两个顺序节点的工作流
workflow = StateGraph(State)

# 添加两个笑话生成节点
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)

# 按顺序连接节点
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", "improve_joke")
workflow.add_edge("improve_joke", END)

# 编译工作流
chain = workflow.compile()

# 显示工作流可视化视图
display(Image(chain.get_graph().draw_mermaid_png()))
```

![我们生成的图谱](https://cdn-images-1.medium.com/max/1000/1*XU_CMOwwboMYcK6lw3HjrA.png)

执行此工作流时，我们将获得如下输出。

```python
# 执行工作流以观察上下文选择机制的实际运作
joke_generator_state = chain.invoke({"topic": "cats"})

# 以富文本格式展示最终状态
console.print("\n[bold blue]Final Workflow State:[/bold blue]")
pprint(joke_generator_state)

#### 输出结果 ####
Initial joke: Why did the cat join a band?

Because it wanted to be the purr-cussionist!
Final Workflow State:
{
  'topic': 'cats',
  'joke': 'Why did the cat join a band?\n\nBecause it wanted to be the purr-cussionist!'}
```

工作流执行完成后，我们可继续将其应用于记忆选择阶段。

## 6. 记忆选择能力

若智能体具备记忆存储能力，则需为当前任务筛选相关记忆。该机制适用于：

* [情景记忆](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types) ——展示预期行为的少样本示例
* [程序性记忆](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types) ——指导行为操作的指令集
* [语义记忆](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types) ——提供任务相关背景的事实与关系网络

部分智能体使用狭窄的预定义文件存储记忆：

* Claude代码采用[`CLAUDE.md`](http://claude.md/) 文件。
* [Cursor](https://docs.cursor.com/context/rules) 与[Windsurf](https://windsurf.com/editor/directory) 通过「规则」文件存储指令或示例。

但当存储大量事实集合（语义记忆）时，记忆筛选会变得困难。

* [ChatGPT](https://help.openai.com/en/articles/8590148-memory-faq) 偶尔会检索无关记忆，[Simon Willison](https://simonwillison.net/2025/Jun/6/six-months-in-llms/) 曾指出：当ChatGPT错误获取其地理位置并注入图像生成过程时，会让人感觉上下文「不再属于自己」。
* 为优化筛选效果，通常采用嵌入向量或[知识图谱](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/#:~:text=changes%20since%20updates%20can%20trigger,and%20holistic%20memory%20for%20agentic) 建立索引机制。

在上一节中，我们向图中的`InMemoryStore`内存存储执行了写入操作。现在可以通过[get](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-storage) 方法从中选择上下文，将相关状态引入工作流。

```python
from langgraph.store.memory import InMemoryStore

# 初始化内存存储
store = InMemoryStore()

# 定义命名空间以组织记忆
namespace = ("rlm", "joke_generator")

# 将生成的笑话存入内存
store.put(
    namespace,                             # 用于组织的命名空间
    "last_joke",                          # 键标识符
    {"joke": joke_generator_state["joke"]} # 待存储的值
)

# 从内存中选择（检索）笑话
retrieved_joke = store.get(namespace, "last_joke").value

# 显示检索到的上下文
console.print("\n[bold green]Retrieved Context from Memory:[/bold green]")
pprint(retrieved_joke)

#### 输出结果 ####
Retrieved Context from Memory:
{'joke': 'Why did the cat join a band?\n\nBecause it wanted to be the purr-cussionist!'}
```

系统成功从内存中检索到正确的笑话。

现在需要编写规范的`generate_joke`函数，使其能够：

1. 获取当前状态（用于暂存区上下文）
2. 利用记忆（若执行笑话改进任务时获取过往笑话）

接下来我们将实现该函数。

```python
# 初始化存储组件
checkpointer = InMemorySaver()
memory_store = InMemoryStore()

def generate_joke(state: State, store: BaseStore) -> dict[str, str]:
    """Generate a joke with memory-aware context selection.
    
    This function demonstrates selecting context from memory before
    generating new content, ensuring consistency and avoiding duplication.
    
    Args:
        state: Current state containing the topic
        store: Memory store for persistent context
        
    Returns:
        Dictionary with the generated joke
    """
    # 若存在过往笑话则从内存中选取
    prior_joke = store.get(namespace, "last_joke")
    if prior_joke:
        prior_joke_text = prior_joke.value["joke"]
        print(f"Prior joke: {prior_joke_text}")
    else:
        print("Prior joke: None!")

    # 生成与先前不同的新笑话
    prompt = (
        f"Write a short joke about {state['topic']}, "
        f"but make it different from any prior joke you've written: {prior_joke_text if prior_joke else 'None'}"
    )
    msg = llm.invoke(prompt)

    # 将新笑话存入内存供后续上下文选择
    store.put(namespace, "last_joke", {"joke": msg.content})

    return {"joke": msg.content}
```

现在我们可以像之前那样直接执行这个具备记忆功能的工作流。

```python
# 构建具备记忆功能的工作流
workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)

# 连接工作流
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", END)

# 同时启用检查点机制和记忆存储进行编译
chain = workflow.compile(checkpointer=checkpointer, store=memory_store)

# 使用首个线程执行工作流
config = {"configurable": {"thread_id": "1"}}
joke_generator_state = chain.invoke({"topic": "cats"}, config)

#### 输出结果 ####
Prior joke: None!
```

未检测到先前的笑话，现在可以打印最新状态结构。

```python
# 获取图结构的最新状态
latest_state = chain.get_state(config)

console.print("\n[bold magenta]Latest Graph State:[/bold magenta]")
pprint(latest_state)
```

输出结果：

```text
#### 最新状态输出 ####
StateSnapshot(
    values={
        'topic': 'cats',
        'joke': "Here's a new one:\n\nWhy did the cat join a band?\n\nBecause it wanted to be the purr-cussionist!"
    },
    next=(),
    config={
        'configurable': {
            'thread_id': '1',
            'checkpoint_ns': '',
            'checkpoint_id': '1f068357-cc8d-68cb-8001-31f64daf7bb6'
        }
    },
    metadata={'source': 'loop', 'step': 1, 'parents': {}},
    created_at='2025-07-24T02:25:38.457825+00:00',
    parent_config={
        'configurable': {
            'thread_id': '1',
            'checkpoint_ns': '',
            'checkpoint_id': '1f068357-c459-6deb-8000-16ce383a5b6b'
        }
    },
    tasks=(),
    interrupts=()
)
```

我们从记忆中获取先前的笑话，并将其传递给大语言模型进行改进。

```python
# 使用第二个线程执行工作流以验证记忆持久性
config = {"configurable": {"thread_id": "2"}}
joke_generator_state = chain.invoke({"topic": "cats"}, config)


#### 输出结果 ####
Prior joke: Here is a new one:
Why did the cat join a band?
Because it wanted to be the purr-cussionist!
```

系统已成功**从记忆中获取正确的笑话**并按要求**对其进行了改进**。

## 7. LangGraph大工具调用优势

智能体群组会使用工具，但提供过多工具可能导致混淆，尤其在工具描述存在重叠时。这会增加模型选择正确工具的难度。

解决方案是采用RAG（检索增强生成）技术，通过工具描述的语义相似度筛选最相关工具——这种方法被Drew Breunig称为[工具负载方案](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html) 。

> 根据[最新研究](https://arxiv.org/abs/2505.03275) ，该方法将工具选择准确率最高提升至3倍。

在工具选择方面，[LangGraph Bigtool](https://github.com/langchain-ai/langgraph-bigtool) 库是最佳方案。它通过对工具描述进行语义相似性搜索，为任务筛选最相关的工具。该库利用LangGraph的长期记忆存储机制，使智能体群组能够搜索并检索适用于特定问题的工具。

让我们通过调用Python内置数学库的全部函数来理解`langgraph-bigtool`的工作机制。

```python
import math

# 收集`math`内置库的函数
all_tools = []
for function_name in dir(math):
    function = getattr(math, function_name)
    if not isinstance(
        function, types.BuiltinFunctionType
    ):
        continue
    # 这是`math`库的特殊设计
    if tool := convert_positional_only_function_to_tool(
        function
    ):
        all_tools.append(tool)
```

首先将Python数学模块的所有函数添加至列表。接下来需将这些工具描述转化为向量嵌入，以便智能体执行语义相似性搜索。

我们将采用嵌入模型实现此功能——本案例使用OpenAI文本嵌入模型。

```python
# 创建工具注册表（字典结构
# 用于将标识符映射到工具实例）
tool_registry = {
    str(uuid.uuid4()): tool
    for tool in all_tools
}

# 在LangGraph存储中建立工具名称与描述的索引
# 此处使用简易内存存储
embeddings = init_embeddings("openai:text-embedding-3-small")

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["description"],
    }
)
for tool_id, tool in tool_registry.items():
    store.put(
        ("tools",),
        tool_id,
        {
            "description": f"{tool.name}: {tool.description}",
        },
    )
```

每个函数被分配唯一ID，并按标准化格式进行结构化处理这种结构化格式确保函数可轻松转换为嵌入向量，以支持语义搜索

现在可视化智能体，观察所有数学函数嵌入后准备就绪的语义搜索形态！

```python
# 初始化智能体
builder = create_agent(llm, tool_registry)
agent = builder.compile(store=store)
agent
```

![我们的工具智能体](https://cdn-images-1.medium.com/max/1000/1*7uXCS9bgbNCwxB-6t6ZXOw.png)

现在可通过简单查询调用智能体，观察工具调用智能体如何选择并运用最相关的数学函数解答问题

```python
# 导入用于格式化与显示消息的实用函数
from utils import format_messages

# 定义智能体查询指令
# 该指令要求智能体使用数学工具求解反余弦值
query = "Use available tools to calculate arc cosine of 0.5."

# 使用查询调用智能体智能体将搜索其工具库，
# 根据查询语义选择'acos'工具并执行
result = agent.invoke({"messages": query})

# 格式化并展示智能体执行后的最终消息
format_messages(result['messages'])
```

```text
┌────────────── Human   ───────────────┐
│ Use available tools to calculate     │
│ arc cosine of 0.5.                   │
└──────────────────────────────────────┘

┌────────────── 📝 AI ─────────────────┐
│ I will search for a tool to calculate│
│ the arc cosine of 0.5.               │
│                                      │
│ 🔧 Tool Call: retrieve_tools         │
│ Args: {                              │
│   "query": "arc cosine arccos        │
│            inverse cosine trig"      │
│ }                                    │
└──────────────────────────────────────┘

┌────────────── 🔧 Tool Output ────────┐
│ Available tools: ['acos', 'acosh']   │
└──────────────────────────────────────┘

┌────────────── 📝 AI ─────────────────┐
│ Perfect! I found the `acos` function │
│ which calculates the arc cosine.     │
│ Now I will use it to calculate the   │
│ arc                                  │
│ cosine of 0.5.                       │
│                                      │
│ 🔧 Tool Call: acos                   │
│ Args: { "x": 0.5 }                   │
└──────────────────────────────────────┘

┌────────────── 🔧 Tool Output ────────┐
│ 1.0471975511965976                   │
└──────────────────────────────────────┘

┌────────────── 📝 AI ─────────────────┐
│ The arc cosine of 0.5 is ≈**1.047**  │
│ radians.                             │
│                                      │
│ ✔ Check: cos(π/3)=0.5, π/3≈1.047 rad │
│ (60°).                               │
└──────────────────────────────────────┘
```

可见我们的人工智能智能体能高效调用正确工具扩展阅读资源：

* [**Toolshed**](https://arxiv.org/abs/2410.14594) 提出Toolshed知识库与高级RAG-工具融合技术，优化AI智能体的工具选择能力
* [**图RAG-工具融合**](https://arxiv.org/abs/2502.07223) 结合向量检索与图遍历技术，捕捉工具间依赖关系
* [**LLM工具综述**](https://github.com/quchangle1/LLM-Tool-Survey) 大语言模型工具学习综合研究
* [**ToolRet**](https://arxiv.org/abs/2503.01763) 大语言模型工具检索能力评估与改进基准

## 8. 基于上下文工程的RAG系统

[RAG（检索增强生成）](https://github.com/langchain-ai/rag-from-scratch) 是一个广泛的研究领域，而代码智能体正是生产环境中智能体式RAG的最佳实践范例。

在实际应用中，RAG往往是上下文工程的核心挑战。正如[风帆冲浪公司的Varun](https://x.com/_mohansolo/status/1899630246862966837) 所指出的：
> 索引 ≠ 上下文检索。基于AST分块的嵌入搜索虽然有效，但随着代码库规模扩大会逐渐失效。我们需要混合检索方案：grep/文件搜索、知识图谱链接以及基于相关性的重排序。

LangGraph提供[教程与视频](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/) 指导如何将RAG集成至智能体系统。通常需要构建一个能综合运用上述RAG技术的检索工具。

为作演示，我们将从Lilian Weng的优秀博客中选取最近三篇文章作为RAG系统的文档来源。

我们将首先使用`WebBaseLoader`工具提取页面内容。

```python
# 导入WebBaseLoader，用于从URL获取文档
from langchain_community.document_loaders import WebBaseLoader

# 定义Lilian Weng博客文章的URL列表
urls = [
    "https://lilianweng.github.io/posts/2025-05-01-thinking/",
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

# 通过列表推导式从指定URL加载文档
# 此操作为每个URL创建WebBaseLoader实例并调用其load()方法
docs = [WebBaseLoader(url).load() for url in urls]
```

RAG存在多种数据分块方式，正确的分块处理对高效检索至关重要。

在将获取的文档索引到向量库之前，我们需要将其分割为更小的文本块。我们将采用简单直接的方法（如带重叠片段的递归分块），在保持分块适用于嵌入和检索的同时，确保跨文本块的上下文连贯性。

```python
# 导入用于文档分块的文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 将文档列表扁平化处理WebBaseLoader为每个URL返回文档列表，
# 因此我们得到的是嵌套列表结构该推导式将所有文档合并为单一列表
docs_list = [item for sublist in docs for item in sublist]

# 初始化文本分割器此操作将文档分割成更小的文本块
# 按指定大小分割文本块，分块间保留部分重叠以维持上下文连贯性。
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000, chunk_overlap=50
)

# 将文档分割成文本块
doc_splits = text_splitter.split_documents(docs_list)
```

完成文档分割后，我们可将其索引到向量存储中，用于后续语义搜索。

```python
# 导入创建内存向量存储所需的类
from langchain_core.vectorstores import InMemoryVectorStore

# 基于文档分割块创建内存向量存储
# 使用前文生成的'doc_splits'和已初始化的'embeddings'模型，为文本块创建向量表示
# 通过文本块生成向量化表征
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=embeddings
)

# 从向量存储创建检索器
# 该检索器提供基于查询搜索相关文档的接口
# 用于执行文档检索操作
retriever = vectorstore.as_retriever()
```

需创建可在智能体中使用的检索器工具

```python
# 导入创建检索器工具的函数
from langchain.tools.retriever import create_retriever_tool

# 基于向量存储检索器创建工具
# 此工具允许智能体根据查询
# 从博客文档中搜索并检索相关内容
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)

# 以下代码行展示直接调用该工具的示例
# 此行被注释掉，因其非智能体执行流程必需，但可用于测试验证。
# retriever_tool.invoke({"query": "types of reward hacking"})
```

现在，我们可以实现一个能够从工具中选择上下文的智能体。

```python
# 通过工具增强大语言模型能力
tools = [retriever_tool]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)
```

对于基于RAG的解决方案，需创建清晰的系统提示以指导智能体行为。该提示构成其核心指令集。

```python
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, ToolMessage
from typing_extensions import Literal

rag_prompt = """You are a helpful assistant tasked with retrieving information from a series of technical blog posts by Lilian Weng. 
Clarify the scope of research with the user before using your retrieval tool to gather context. Reflect on any context you fetch, and
proceed until you have sufficient context to answer the user's research request."""
```

接下来定义图结构的节点。我们需要两个主节点：

1. `llm_call`：智能体的核心决策中枢接收当前对话历史（用户查询+先前工具输出）据此决策下一步动作：调用工具或生成最终答案
2. `tool_node`：智能体的执行组件执行由`llm_call`发起的工具调用指令将工具执行结果返回给智能体

```python
# --- 定义智能体节点 ---

def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or generate a final answer."""
    # 将系统提示加入当前消息状态
    messages_with_prompt = [SystemMessage(content=rag_prompt)] + state["messages"]
    
    # 使用增强后的消息列表调用大语言模型
    response = llm_with_tools.invoke(messages_with_prompt)
    
    # 返回大语言模型的响应，该响应将被添加到状态中
    return {"messages": [response]}
    
def tool_node(state: dict):
    """Performs the tool call and returns the observation."""
    # 获取最后一条消息（应包含工具调用信息）
    last_message = state["messages"][-1]
    
    # 执行每个工具调用并收集结果
    result = []
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        
    # 将工具输出作为消息返回
    return {"messages": result}
```

需要一种机制来控制智能体流程，决定其应调用工具还是结束运行。

为此我们将创建名为 `should_continue` 的条件边函数。

* 该函数检查大语言模型最后返回的消息是否包含工具调用。
* 若存在工具调用，流程图将路由至 `tool_node` 节点
* 否则终止执行流程

```python
# --- 定义条件边 ---

def should_continue(state: MessagesState) -> Literal["Action", END]:
    """Decides the next step based on whether the LLM made a tool call."""
    last_message = state["messages"][-1]
    
    # 若大语言模型发起工具调用，则路由至工具节点
    if last_message.tool_calls:
        return "Action"
    # 否则结束工作流
    return END
```

现在即可构建工作流并编译流程图

```python
# 构建工作流
agent_builder = StateGraph(MessagesState)

# 添加节点
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# 添加连接节点的边
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # should_continue返回的名称 : 要访问的下一个节点名称
        "Action": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# 编译智能体
agent = agent_builder.compile()

# 显示智能体
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
```

![基于RAG的智能体](https://cdn-images-1.medium.com/max/1000/1*0QxVbzakDabkoMfgURIx2w.png)

该流程图呈现清晰循环：

1. 智能体启动后调用LLM
2. 根据LLM决策，执行操作（调用检索器工具）并循环回退，或完成处理返回答案

现在测试RAG智能体我们将提出关于**奖励黑客攻击**的具体问题，该问题只能通过检索已索引博客内容获取答案

```python
# 定义用户查询
query = "What are the types of reward hacking discussed in the blogs?"

# 通过查询调用智能体
result = agent.invoke({"messages": [("user", query)]})

# --- 显示最终消息 ---
# 格式化输出对话流程
format_messages(result['messages'])
```

```text
┌──────────────  Human  ───────────────┐
│ Clarify scope: I want types of       │
│ reward hacking from Lilian Weng’s    │
│ blog on RL.                          │
└──────────────────────────────────────┘

┌────────────── 📝 AI ─────────────────┐
│ Fetching context from her posts...   │
└──────────────────────────────────────┘

┌────────────── 🔧 Tool Output ────────┐
│ She lists 3 main types of reward     │
│ hacking in RL:                       │
└──────────────────────────────────────┘

┌────────────── 📝 AI ─────────────────┐
│ 1. **Spec gaming** – Exploit reward  │
│    loopholes, not real goal.         │
│                                      │
│ 2. **Reward tampering** – Change or  │
│    hack reward signals.              │
│                                      │
│ 3. **Wireheading** – Self-stimulate  │
│    reward instead of task.           │
└──────────────────────────────────────┘

┌────────────── 📝 AI ─────────────────┐
│ These can cause harmful, unintended  │
│ behaviors in RL agents.              │
└──────────────────────────────────────┘
```

可见智能体正确识别了需使用检索工具的需求随后成功从博客中检索到相关上下文，并利用该信息提供了详尽准确的答案

> 这完美展示了如何通过RAG实施上下文工程来构建功能强大、知识渊博的智能体群组。

## 9. 知识型智能体的压缩策略

智能体交互可跨越[数百轮次](https://www.anthropic.com/engineering/built-multi-agent-research-system) ，并涉及高token消耗的工具调用。摘要生成是管理此类问题的常用方法。

![上下文工程的第三组件](https://cdn-images-1.medium.com/max/1000/1*Xu76qgF1u2G3JipeIgHo5Q.png)
*上下文工程的第三组件（摘自[LangChain文档](https://blog.langchain.com/context-engineering-for-agents/) ）*

例如：

* 当上下文窗口使用率超过95%时，Claude代码采用"[自动压缩](https://docs.anthropic.com/en/docs/claude-code/costs) "机制，对完整的用户-智能体交互历史进行摘要生成。
* 通过[递归摘要](https://arxiv.org/pdf/2308.15022#:~:text=the%20retrieved%20utterances%20capture%20the,based%203) 或[分层摘要](https://alignment.anthropic.com/2025/summarization-for-monitoring/#:~:text=We%20addressed%20these%20issues%20by,of%20our%20computer%20use%20capability) 等策略，可压缩[智能体轨迹](https://langchain-ai.github.io/langgraph/concepts/memory/#manage-short-term-memory) 数据。

您也可以在特定节点添加摘要生成功能：

* 在令牌密集型工具调用后（例如搜索工具）[示例见此](https://github.com/langchain-ai/open_deep_research/blob/e5a5160a398a3699857d00d8569cb7fd0ac48a4f/src/open_deep_research/utils.py#L1407) 。
* 在智能体间边界进行知识传递时，[Cognition](https://cognition.ai/blog/dont-build-multi-agents#a-theory-of-building-long-running-agents) 在Devin中通过微调模型实现了该功能。

![LangGraph摘要生成方法示意图](https://cdn-images-1.medium.com/max/1500/1*y5AhaYoM_XDDrvlAnnFhcQ.png)
*LangGraph摘要生成方法（摘自[LangChain文档](https://blog.langchain.com/context-engineering-for-agents/) ）*

LangGraph是一个[底层编排框架](https://blog.langchain.com/how-to-think-about-agent-frameworks/) ，为您提供以下方面的完全控制权：

* 将智能体设计为一组[节点](https://www.youtube.com/watch?v=aHCDrAbH_go) 。
* 在每个节点内明确定义逻辑。
* 在节点间传递共享状态对象。

这使得通过不同方式压缩上下文变得简单。例如您可以：

* 使用消息列表作为智能体状态。
* 通过[内置实用工具](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/#manage-short-term-memory) 进行摘要生成。

我们将沿用先前编写的基于RAG的工具调用智能体，并增加对话历史的摘要功能。

首先需要扩展图的状态，添加用于存储最终摘要的字段。

```python
# 定义包含摘要字段的扩展状态
class State(MessagesState):
    """Extended state that includes a summary field for context compression."""
    summary: str
```

接下来定义专用的摘要生成提示词，同时保留之前的RAG提示词。

```python
# 定义摘要生成提示词
summarization_prompt = """Summarize the full chat history and all tool feedback to 
give an overview of what the user asked about and what the agent did."""
```

现在创建`摘要节点`。

* 该节点将在智能体工作结束时触发，生成整个交互过程的简明摘要。
* `llm_call` 与 `tool_node` 保持不变。

```python
def summary_node(state: MessagesState) -> dict:
    """
    Generate a summary of the conversation and tool interactions.

    Args:
        state: The current state of the graph, containing the message history.

    Returns:
        A dictionary with the key "summary" and the generated summary string
        as the value, which updates the state.
    """
    # 将摘要系统提示添加到消息历史记录开头
    messages = [SystemMessage(content=summarization_prompt)] + state["messages"]
    
    # 调用大语言模型生成摘要
    result = llm.invoke(messages)
    
    # 返回摘要内容存储至状态的'summary'字段
    return {"summary": result.content}
```

条件边 should_continue 现在需要决定调用工具还是跳转至新的 summary_node。

```python
def should_continue(state: MessagesState) -> Literal["Action", "summary_node"]:
    """Determine next step based on whether LLM made tool calls."""
    last_message = state["messages"][-1]
    
    # 若大语言模型发起工具调用，则执行调用
    if last_message.tool_calls:
        return "Action"
    # 否则进行摘要生成
    return "summary_node"
```

现在构建包含最终摘要步骤的新工作流图。

```python
# 构建 RAG 智能体工作流
agent_builder = StateGraph(State)

# 向工作流添加节点
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("Action", tool_node)
agent_builder.add_node("summary_node", summary_node)

# 定义工作流边
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "Action",
        "summary_node": "summary_node",
    },
)
agent_builder.add_edge("Action", "llm_call")
agent_builder.add_edge("summary_node", END)

# 编译智能体
agent = agent_builder.compile()

# 展示智能体工作流
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
```

![我们创建的智能体](https://cdn-images-1.medium.com/max/1000/1*UTtZj95DQ9_0hXb-h2UetQ.png)

现在执行需要获取大量上下文的查询测试。

```python
from rich.markdown import Markdown

query = "Why does RL improve LLM reasoning according to the blogs?"
result = agent.invoke({"messages": [("user", query)]})

# 向用户输出最终消息
format_message(result['messages'][-1])

# 输出生成的摘要
Markdown(result["summary"])


#### 输出结果 ####
The user asked about why reinforcement learning (RL) improves LLM re...
```

效果不错，但消耗了**11.5万令牌**！完整执行轨迹可[在此查看](https://smith.langchain.com/public/50d70503-1a8e-46c1-bbba-a1efb8626b05/r) 。这是工具调用令牌消耗量大的智能体群组普遍面临的挑战。

更高效的解决方案是在上下文进入智能体主暂存区*之前*进行压缩处理。让我们升级RAG智能体，实现工具调用输出的实时摘要生成。

首先，针对此任务设计新提示模板：

```python
tool_summarization_prompt = """You will be provided a doc from a RAG system.
Summarize the docs, ensuring to retain all relevant / essential information.
Your goal is simply to reduce the size of the doc (tokens) to a more manageable size."""
```

接下来修改**工具节点**，将摘要生成步骤整合其中

```python
def tool_node_with_summarization(state: dict):
    """Performs the tool call and then summarizes the output."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        
        # 文档摘要生成
        summary_msg = llm.invoke([
            SystemMessage(content=tool_summarization_prompt),
            ("user", str(observation))
        ])
        
        result.append(ToolMessage(content=summary_msg.content, tool_call_id=tool_call["id"]))
    return {"messages": result}
```

由于不再需要最终的摘要节点，现在可简化`应继续`判定逻辑

```python
def should_continue(state: MessagesState) -> Literal["Action", END]:
    """Decide if we should continue the loop or stop."""
    if state["messages"][-1].tool_calls:
        return "Action"
    return END
```

现在构建并编译这个高效版智能体程序

```python
# 构建工作流
agent_builder = StateGraph(MessagesState)

# 添加节点
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("Action", tool_node_with_summarization)

# 添加连接节点的边
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "Action",
        END: END,
    },
)
agent_builder.add_edge("Action", "llm_call")

# 编译智能体
agent = agent_builder.compile()

# 显示智能体
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
```

![优化后的智能体架构图示](https://cdn-images-1.medium.com/max/1000/1*FCRrXQxZveaQxyLHf6AROQ.png)

让我们执行相同查询并观察差异。

```python
query = "Why does RL improve LLM reasoning according to the blogs?"
result = agent.invoke({"messages": [("user", query)]})
format_messages(result['messages'])
```

```text
┌────────────── user ───────────────┐
│ Why does RL improve LLM reasoning?│
│ According to the blogs?            │
└───────────────────────────────────┘

┌────────────── 📝 AI ──────────────┐
│ Searching Lilian Weng’s blog for  │
│ how RL improves LLM reasoning...  │
│                                   │
│ 🔧 Tool Call: retrieve_blog_posts │
│ Args: {                           │
│ "query": "Reinforcement Learning  │
│ for LLM reasoning"                │
│ }                                │
└───────────────────────────────────┘

┌────────────── 🔧 Tool Output ─────┐
│ Lilian Weng explains RL helps LLM │
│ reasoning by training on rewards  │
│ for each reasoning step (Process- │
│ based Reward Models). This guides │
│ the model to think step-by-step,  │
│ improving coherence and logic.    │
└───────────────────────────────────┘

┌────────────── 📝 AI ──────────────┐
│ RL improves LLM reasoning by       │
│ rewarding stepwise thinking via    │
│ PRMs, encouraging coherent,        │
│ logical argumentation over final   │
│ answers. It helps the model self-  │
│ correct and explore better paths.  │
└───────────────────────────────────┘
```

> 本次智能体仅消耗**6万令牌**，查看此处的[追踪记录](https://smith.langchain.com/public/994cdf93-e837-4708-9628-c83b397dd4b5/r) 。

此简单改动使令牌消耗减少近半，显著提升智能体效率与成本效益。

扩展阅读资源：

* [**启发式压缩与消息裁剪**](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/#trim-messages) ：通过裁剪消息防止上下文溢出，实现令牌限额管理。
* [**摘要生成节点作为模型前钩**](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-manage-message-history/) ：在ReAct智能体中通过汇总对话历史控制令牌消耗。
* [**LangMem摘要生成**](https://langchain-ai.github.io/langmem/guides/summarization/) ：采用消息摘要与动态摘要策略实现长上下文管理。

## 10. 通过子智能体架构隔离上下文

隔离上下文的常见方法是通过子智能体群组分担上下文处理。OpenAI的[Swarm](https://github.com/openai/swarm) 库正是为实现这种「[关注点分离](https://openai.github.io/openai-agents-python/ref/agent/) 」而设计，每个智能体通过专属工具、指令和上下文窗口管理特定子任务。

![上下文工程第四组件](https://cdn-images-1.medium.com/max/1000/1*-b9BLPkLHkYsy2iLQIdxUg.png)
*上下文工程第四组件（源自[LangChain文档](https://blog.langchain.com/context-engineering-for-agents/) ）*

Anthropic的[多智能体研究系统](https://www.anthropic.com/engineering/built-multi-agent-research-system) 表明，采用独立上下文的多智能体群组相较单智能体性能提升90.2%，因为每个子智能体能聚焦于更窄的子任务。

> *子智能体通过各自的上下文窗口并行运作，同时探索问题的不同维度。*

然而，多智能体系统面临以下挑战：

* 令牌使用量大幅增加（有时达到单智能体聊天的15倍以上）。
* 需要精心设计[提示词工程](https://www.anthropic.com/engineering/built-multi-agent-research-system) 来规划子智能体任务。
* 协调子智能体的过程可能非常复杂。

![多智能体并行处理](https://cdn-images-1.medium.com/max/1000/1*N_BT9M5OyYB7UJfDkpcL-g.png)
*多智能体并行处理（源自[LangChain文档](https://blog.langchain.com/context-engineering-for-agents/) ）*

LangGraph支持多智能体架构部署。常用方案是采用[监督器](https://github.com/langchain-ai/langgraph-supervisor-py) 架构，该方案同样应用于Anthropic多智能体研究系统。监督器将任务委派给子智能体，每个子智能体在独立的上下文窗口中运行。

下面我们构建一个管理两个智能体的简易监督器：

* `math_expert` 处理数学计算任务。
* `research_expert` 搜索并提供研究信息。

监督器将根据查询内容决定调用哪位专家，并在LangGraph工作流中协调其响应。

```python
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

# --- 定义各智能体工具集 ---
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def web_search(query: str) -> str:
    """Mock web search function that returns FAANG company headcounts."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )
```

现在我们可以创建专业智能体及其管理监督器。

```python
# --- 创建具有隔离上下文的专业智能体 ---
math_agent = create_react_agent(
    model=llm,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)

research_agent = create_react_agent(
    model=llm,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)

# --- 创建协调智能体的监督器工作流 ---
workflow = create_supervisor(
    [research_agent, math_agent],
    model=llm,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "Delegate tasks to the appropriate agent to answer the user's query. "
        "For current events or facts, use research_agent. "
        "For math problems, use math_agent."
    )
)

# 编译多智能体应用
app = workflow.compile()
```

执行工作流，观察监督器如何分配任务。

```python
# --- 执行多智能体工作流 ---
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what's the combined headcount of the FAANG companies in 2024?"
        }
    ]
})

# 格式化并显示结果
format_messages(result['messages'])
```

```text
┌────────────── user ───────────────┐
│ Learn more about LangGraph Swarm  │
│ and multi-agent systems.          │
└───────────────────────────────────┘

┌────────────── 📝 AI ──────────────┐
│ Fetching details on LangGraph     │
│ Swarm and related resources...    │
└───────────────────────────────────┘

┌────────────── 🔧 Tool Output ─────┐
│ **LangGraph Swarm**               │
│ Repo:                             │
│ https://github.com/langchain-ai/  │
│ langgraph-swarm-py                │
│                                   │
│ • Python library for multi-agent  │
│   AI with dynamic collaboration.  │
│ • Agents hand off control based   │
│   on specialization, keeping      │
│   conversation context.           │
│ • Supports custom handoffs,       │
│   streaming, memory, and human-   │
│   in-the-loop.                    │
│ • Install:                        │
│   `pip install langgraph-swarm`   │
└───────────────────────────────────┘

┌────────────── 🔧 Tool Output ─────┐
│ **Videos on multi-agent systems** │
│ 1. https://youtu.be/4nZl32FwU-o   │
│ 2. https://youtu.be/JeyDrn1dSUQ   │
│ 3. https://youtu.be/B_0TNuYi56w   │
└───────────────────────────────────┘

┌────────────── 📝 AI ──────────────┐
│ LangGraph Swarm makes it easy to  │
│ build context-aware multi-agent    │
│ systems. Check videos for deeper   │
│ insights on multi-agent behavior.  │
└───────────────────────────────────┘
```

此处监督器正确隔离了每项任务的上下文——将研究查询发送至研究员，数学问题发送至数学家，展现了高效的上下文隔离机制。

扩展阅读资源：

* [**LangGraph群智系统**](https://github.com/langchain-ai/langgraph-swarm-py) ：支持动态任务交接、记忆存储与人机协同的Python多智能体系统开发库。
* [**多智能体系统专题视频**](https://www.youtube.com/watch?v=4nZl32FwU-o) 获取构建协同式人工智能智能体的深度解析（[视频2](https://www.youtube.com/watch?v=JeyDrn1dSUQ) , [视频3](https://www.youtube.com/watch?v=B_0TNuYi56w) ）。

## 11. 沙盒环境隔离技术

HuggingFace的[深度研究](https://huggingface.co/blog/open-deep-research#:~:text=From%20building%20,it%20can%20still%20use%20it) 展示了一种隔离上下文的创新方法。大多数智能体使用[工具调用API](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) ，这些接口返回JSON参数来运行搜索API等工具并获取结果。

HuggingFace采用[代码智能体](https://huggingface.co/papers/2402.01030) ，通过编写代码来调用工具。该代码在安全的[沙盒环境](https://e2b.dev/) 中运行，执行结果将返回给大语言模型（LLM）。

这种方法使海量数据（如图像或音频）可突破LLM的token限制。HuggingFace阐释道：

> *[代码智能体能够]更优地处理状态……需要暂存此图像/音频/其他数据供后续使用？只需将其保存为状态中的变量，后续调用即可。*

在LangGraph中使用沙箱环境十分简便。[LangChain沙箱](https://github.com/langchain-ai/langchain-sandbox) 通过Pyodide（编译为WebAssembly的Python）安全执行非受信Python代码。您可将其作为工具集成至任意LangGraph智能体。

**注意：** 需预先安装Deno。 安装指南：<https://docs.deno.com/runtime/getting_started/installation/>

```python
from langchain_sandbox import PyodideSandboxTool
from langgraph.prebuilt import create_react_agent

# 创建支持网络访问的沙箱工具（用于软件包安装）
tool = PyodideSandboxTool(allow_net=True)

# 创建包含沙箱工具的ReAct智能体
agent = create_react_agent(llm, tools=[tool])

# 使用沙箱执行数学查询
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what's 5 + 7?"}]},
)

# 格式化并显示结果
format_messages(result['messages'])
```

```text
┌────────────── user ───────────────┐
│ what's 5 + 7?                    │
└──────────────────────────────────┘

┌────────────── 📝 AI ──────────────┐
│ I can solve this by executing     │
│ Python code in the sandbox.       │
│                                  │
│ 🔧 Tool Call: pyodide_sandbox     │
│ Args: {                          │
│   "code": "print(5 + 7)"          │
│ }                                │
└──────────────────────────────────┘

┌────────────── 🔧 Tool Output ─────┐
│ 12                               │
└──────────────────────────────────┘

┌────────────── 📝 AI ──────────────┐
│ The answer is 12.                 │
└──────────────────────────────────┘
```

## 12. LangGraph中的状态隔离

智能体的**运行时状态对象**是隔离上下文的绝佳方式，其作用类似于沙箱机制。可通过模式设计此状态（如采用Pydantic模型），其中不同字段专用于存储上下文信息。

例如：某字段（如`messages`）每轮次均向大语言模型(LLM)展示，而其他字段存储的信息则保持隔离直至需要时调用。

LangGraph 围绕[**状态**](https://langchain-ai.github.io/langgraph/concepts/low_level/#state) 对象构建，允许您创建自定义状态模式并在智能体工作流中访问其字段。

例如，您可将工具调用结果存储于特定字段，使其在必要时才对大语言模型可见。您已在上述笔记中看到诸多相关示例。

## 13. 全面总结

让我们总结当前完成的工作：

* 我们使用 LangGraph 的 `StateGraph` 创建短期记忆**「暂存区」**，并采用 `InMemoryStore` 实现长期记忆存储，使智能体具备信息存取能力。
* 我们演示了如何从智能体状态和长期记忆中精准提取相关信息。这包括运用检索增强生成（`RAG`）获取特定知识，以及通过 `langgraph-bigtool` 从多工具中精准选择。
* 为管理长对话及占用大量令牌的工具输出，我们实现了摘要生成机制。
* 我们演示了如何动态压缩`RAG`结果，以提升智能体效率并减少令牌使用量。
* 通过构建多智能体系统（由监督员将任务分派给专业子智能体）以及使用沙盒环境运行代码，我们实现了上下文隔离机制以避免混淆。

这些技术均属于**“上下文工程”**策略范畴，该策略通过精细管理工作内存（`context`）来优化人工智能智能体，使其更高效、更精准，并能处理复杂且长期运行的任务。
