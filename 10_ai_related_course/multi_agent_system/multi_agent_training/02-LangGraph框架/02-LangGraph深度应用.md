# 第二天：LangGraph框架深度应用

## 学习目标

1. LangGraph基础与核心概念
   - 理解LangGraph的设计理念和核心概念
   - 掌握图结构在多智能体系统中的应用
   - 学会创建基础的LangGraph工作流

2. 高级工作流构建与模式
   - 掌握复杂工作流的设计模式
   - 学习并行处理和循环控制
   - 理解工作流的性能优化技巧

---

## 参考项目

**💡 实际代码参考**：完整的LangGraph工作流引擎实现可参考项目中的以下文件：

- `langgraph_workflow.py` - 企业级工作流管理引擎
- `multi_agent_system/src/workflows/` - 工作流模板和实现
- `multi_agent_system/src/workflows/langgraph_workflow.py` - 状态管理机制
- `src/examples/customer_service_system.py` - 客服系统工作流案例
- `src/monitoring/langsmith_integration.py` - 工作流性能监控

**代码引用**: 完整的企业级工作流引擎实现请参考 `multi_agent_system/src/workflows/langgraph_workflow.py`

企业级LangGraph工作流引擎的核心特性：

**核心组件**：

- `WorkflowStatus`: 工作流状态管理（待处理、运行中、已完成、失败、暂停）
- `WorkflowMetrics`: 性能指标监控（执行时间、内存使用、成功率等）
- `EnterpriseWorkflowEngine`: 主引擎类，支持并发工作流管理

**企业级特性**：

- 并发工作流限制和资源管理
- 实时性能指标收集和监控
- 状态持久化和恢复机制
- 异常处理和自动清理

---

## 1. LangGraph基础与核心概念（2学时）

### 1.1 LangGraph框架概述

LangGraph是LangChain生态系统中专门用于构建多智能体工作流的框架，它将复杂的智能体交互建模为有向图结构，每个节点代表一个智能体或处理步骤，边代表数据流和控制流。

> **💡 实际代码参考**：完整的LangGraph工作流引擎实现可参考项目中的 `langgraph_workflow.py` 文件，该文件提供了企业级的工作流管理、状态验证和性能监控功能。

#### 1.1.1 核心设计理念

| 设计原则 | 具体体现 | 技术优势 | 应用价值 |
|---------|---------|---------|---------|
| **图结构建模** | 节点-边模型表示工作流 | 直观可视化、易于理解 | 复杂流程管理、调试便利 |
| **状态管理** | 集中式状态存储与更新 | 一致性保证、并发安全 | 多智能体协作、数据共享 |
| **流程控制** | 条件分支、循环、并行 | 灵活的执行逻辑 | 动态决策、自适应流程 |
| **可扩展性** | 模块化节点、插件机制 | 组件复用、系统扩展 | 快速开发、维护便利 |
| **错误恢复** | 智能错误检测与恢复 | 自动重试、状态回滚 | 系统稳定性、容错能力 |

#### 1.1.2 企业级特性对比

| 对比维度 | LangGraph | 传统工作流引擎 | 企业级增强 |
|---------|-----------|---------------|------------|
| **AI集成** | 原生支持LLM和AI工具 | 需要额外集成 | 智能决策、自适应优化 |
| **状态管理** | 智能状态推理 | 静态状态定义 | 状态验证、一致性保证 |
| **错误处理** | AI驱动的错误恢复 | 预定义错误处理 | 智能重试、自动恢复 |
| **性能监控** | 实时性能追踪 | 基础日志记录 | 全链路监控、性能优化 |
| **学习能力** | 支持在线学习优化 | 静态流程定义 | 持续改进、模式识别 |

### 1.2 核心概念深度解析

#### 1.2.1 图（Graph）结构

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

# 定义状态结构
class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]
    current_agent: str
    task_status: str
    shared_context: dict

# 创建状态图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("research_agent", research_node)
workflow.add_node("analysis_agent", analysis_node)
workflow.add_node("decision_agent", decision_node)

# 定义边和条件
workflow.add_edge("research_agent", "analysis_agent")
workflow.add_conditional_edges(
    "analysis_agent",
    decide_next_step,
    {
        "continue": "decision_agent",
        "need_more_data": "research_agent",
        "complete": END
    }
)

# 设置入口点
workflow.set_entry_point("research_agent")

# 编译图
app = workflow.compile()
```

#### 1.2.2 节点（Node）类型与实现

| 节点类型 | 功能描述 | 实现方式 | 使用场景 |
|---------|---------|---------|---------|
| **智能体节点** | 封装单个智能体的处理逻辑 | LLM + 工具调用 | 专业任务处理 |
| **工具节点** | 执行特定工具或API调用 | 函数封装 | 外部系统集成 |
| **决策节点** | 基于条件进行路径选择 | 条件逻辑 | 流程控制 |
| **聚合节点** | 合并多个输入的结果 | 数据合并算法 | 结果整合 |

```python
# 智能体节点实现示例
async def research_agent_node(state: AgentState) -> AgentState:
    """研究智能体节点"""
    
    # 获取当前状态
    messages = state["messages"]
    context = state["shared_context"]
    
    # 构建提示词
    prompt = f"""
    作为研究智能体，请基于以下信息进行深度研究：
    历史消息：{messages}
    上下文：{context}
    
    请提供详细的研究报告。
    """
    
    # 调用LLM
    llm = ChatOpenAI(model="gpt-4")
    response = await llm.ainvoke(prompt)
    
    # 更新状态
    return {
        "messages": [f"Research Agent: {response.content}"],
        "current_agent": "research_agent",
        "task_status": "research_completed",
        "shared_context": {
            **context,
            "research_data": response.content
        }
    }

# 条件决策节点
def decide_next_step(state: AgentState) -> str:
    """决策下一步执行路径"""
    
    task_status = state["task_status"]
    context = state["shared_context"]
    
    # 检查研究数据质量
    research_data = context.get("research_data", "")
    
    if len(research_data) < 100:
        return "need_more_data"
    elif "analysis_required" in research_data.lower():
        return "continue"
    else:
        return "complete"
```

#### 1.2.3 状态管理机制

LangGraph的状态管理是其核心特性之一，支持多种状态更新模式：

```python
from typing import Annotated
import operator

# 状态定义示例
class MultiAgentState(TypedDict):
    # 累加模式：新消息追加到列表
    messages: Annotated[List[str], operator.add]
    
    # 替换模式：新值直接替换旧值
    current_step: str
    
    # 合并模式：字典合并
    agent_outputs: Annotated[dict, lambda x, y: {**x, **y}]
    
    # 自定义更新函数
    confidence_scores: Annotated[List[float], lambda x, y: x + y if x else y]

# 状态更新示例
def update_confidence(existing: List[float], new: List[float]) -> List[float]:
    """自定义置信度更新逻辑"""
    if not existing:
        return new
    
    # 计算加权平均
    combined = []
    for i in range(min(len(existing), len(new))):
        weighted_score = (existing[i] * 0.7 + new[i] * 0.3)
        combined.append(weighted_score)
    
    return combined

# 状态访问和修改
class StateManager:
    """状态管理器"""
    
    def __init__(self, initial_state: MultiAgentState):
        self.state = initial_state
        self.history = [initial_state.copy()]
    
    def update_state(self, updates: dict) -> MultiAgentState:
        """更新状态"""
        new_state = self.state.copy()
        
        for key, value in updates.items():
            if key in new_state:
                # 根据注解类型进行更新
                annotation = MultiAgentState.__annotations__.get(key)
                if hasattr(annotation, '__metadata__'):
                    update_func = annotation.__metadata__[0]
                    new_state[key] = update_func(new_state[key], value)
                else:
                    new_state[key] = value
        
        self.state = new_state
        self.history.append(new_state.copy())
        return new_state
    
    def rollback(self, steps: int = 1) -> MultiAgentState:
        """状态回滚"""
        if len(self.history) > steps:
            self.state = self.history[-(steps + 1)].copy()
            self.history = self.history[:-(steps)]
        return self.state
```

### 1.4 基础工作流构建

#### 1.4.1 简单线性工作流

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI

# 定义简单的客服工作流
class CustomerServiceState(TypedDict):
    customer_query: str
    intent: str
    response: str
    satisfaction_score: float

def intent_recognition_node(state: CustomerServiceState) -> CustomerServiceState:
    """意图识别节点"""
    query = state["customer_query"]
    
    # 使用LLM进行意图识别
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = f"分析以下客户查询的意图：{query}\n返回意图类别：技术支持/账单查询/产品咨询/投诉建议"
    
    response = llm.invoke(prompt)
    intent = response.content.strip()
    
    return {
        **state,
        "intent": intent
    }

def response_generation_node(state: CustomerServiceState) -> CustomerServiceState:
    """响应生成节点"""
    query = state["customer_query"]
    intent = state["intent"]
    
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"""
    客户查询：{query}
    识别意图：{intent}
    
    请生成专业、友好的客服响应。
    """
    
    response = llm.invoke(prompt)
    
    return {
        **state,
        "response": response.content
    }

def satisfaction_evaluation_node(state: CustomerServiceState) -> CustomerServiceState:
    """满意度评估节点"""
    query = state["customer_query"]
    response = state["response"]
    
    # 简化的满意度评估
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = f"""
    评估以下客服对话的满意度（0-10分）：
    客户问题：{query}
    客服回复：{response}
    
    只返回数字分数。
    """
    
    score_response = llm.invoke(prompt)
    try:
        score = float(score_response.content.strip())
    except:
        score = 7.0  # 默认分数
    
    return {
        **state,
        "satisfaction_score": score
    }

# 构建工作流
def create_customer_service_workflow():
    """创建客服工作流"""
    
    workflow = StateGraph(CustomerServiceState)
    
    # 添加节点
    workflow.add_node("intent_recognition", intent_recognition_node)
    workflow.add_node("response_generation", response_generation_node)
    workflow.add_node("satisfaction_evaluation", satisfaction_evaluation_node)
    
    # 定义执行顺序
    workflow.add_edge("intent_recognition", "response_generation")
    workflow.add_edge("response_generation", "satisfaction_evaluation")
    workflow.add_edge("satisfaction_evaluation", END)
    
    # 设置入口点
    workflow.set_entry_point("intent_recognition")
    
    return workflow.compile()

# 使用示例
async def run_customer_service():
    """运行客服工作流"""
    
    app = create_customer_service_workflow()
    
    # 初始状态
    initial_state = {
        "customer_query": "我的账单有问题，为什么这个月费用这么高？",
        "intent": "",
        "response": "",
        "satisfaction_score": 0.0
    }
    
    # 执行工作流
    result = await app.ainvoke(initial_state)
    
    print(f"客户查询：{result['customer_query']}")
    print(f"识别意图：{result['intent']}")
    print(f"客服回复：{result['response']}")
    print(f"满意度评分：{result['satisfaction_score']}")
    
    return result
```

#### 1.4.2 条件分支工作流

```python
# 复杂的条件分支工作流
class ComplexTaskState(TypedDict):
    task_description: str
    complexity_level: str
    assigned_agents: List[str]
    results: dict
    final_output: str

def task_analysis_node(state: ComplexTaskState) -> ComplexTaskState:
    """任务分析节点"""
    description = state["task_description"]
    
    # 分析任务复杂度
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"""
    分析以下任务的复杂度：{description}
    
    返回复杂度级别：简单/中等/复杂/极复杂
    """
    
    response = llm.invoke(prompt)
    complexity = response.content.strip()
    
    return {
        **state,
        "complexity_level": complexity
    }

def simple_task_node(state: ComplexTaskState) -> ComplexTaskState:
    """简单任务处理节点"""
    description = state["task_description"]
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(f"直接处理这个简单任务：{description}")
    
    return {
        **state,
        "assigned_agents": ["general_agent"],
        "results": {"general_agent": response.content},
        "final_output": response.content
    }

def complex_task_node(state: ComplexTaskState) -> ComplexTaskState:
    """复杂任务处理节点"""
    description = state["task_description"]
    
    # 分配给多个专业智能体
    agents = ["research_agent", "analysis_agent", "synthesis_agent"]
    results = {}
    
    for agent in agents:
        llm = ChatOpenAI(model="gpt-4")
        prompt = f"作为{agent}，处理任务的相关部分：{description}"
        response = llm.invoke(prompt)
        results[agent] = response.content
    
    return {
        **state,
        "assigned_agents": agents,
        "results": results
    }

def result_synthesis_node(state: ComplexTaskState) -> ComplexTaskState:
    """结果综合节点"""
    results = state["results"]
    
    # 综合所有智能体的结果
    combined_results = "\n".join([f"{agent}: {result}" for agent, result in results.items()])
    
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"""
    综合以下智能体的处理结果，生成最终输出：
    {combined_results}
    """
    
    final_output = llm.invoke(prompt)
    
    return {
        **state,
        "final_output": final_output.content
    }

def route_by_complexity(state: ComplexTaskState) -> str:
    """根据复杂度路由"""
    complexity = state["complexity_level"]
    
    if complexity in ["简单"]:
        return "simple_task"
    else:
        return "complex_task"

def check_need_synthesis(state: ComplexTaskState) -> str:
    """检查是否需要结果综合"""
    agents = state["assigned_agents"]
    
    if len(agents) > 1:
        return "synthesis"
    else:
        return "end"

# 构建条件分支工作流
def create_adaptive_workflow():
    """创建自适应工作流"""
    
    workflow = StateGraph(ComplexTaskState)
    
    # 添加节点
    workflow.add_node("task_analysis", task_analysis_node)
    workflow.add_node("simple_task", simple_task_node)
    workflow.add_node("complex_task", complex_task_node)
    workflow.add_node("result_synthesis", result_synthesis_node)
    
    # 设置条件分支
    workflow.add_conditional_edges(
        "task_analysis",
        route_by_complexity,
        {
            "simple_task": "simple_task",
            "complex_task": "complex_task"
        }
    )
    
    workflow.add_conditional_edges(
        "simple_task",
        check_need_synthesis,
        {
            "synthesis": "result_synthesis",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "complex_task",
        check_need_synthesis,
        {
            "synthesis": "result_synthesis",
            "end": END
        }
    )
    
    workflow.add_edge("result_synthesis", END)
    workflow.set_entry_point("task_analysis")
    
    return workflow.compile()
```

### 1.5 实践环节

#### 1.5.1 练习1：构建简单的多智能体对话系统

```python
# 多智能体对话系统
class ConversationState(TypedDict):
    user_input: str
    conversation_history: Annotated[List[str], operator.add]
    current_speaker: str
    topic: str
    sentiment: str

def moderator_node(state: ConversationState) -> ConversationState:
    """主持人节点"""
    user_input = state["user_input"]
    history = state["conversation_history"]
    
    # 分析话题和情感
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    topic_prompt = f"分析对话主题：{user_input}\n历史：{history[-3:]}\n返回主题关键词"
    topic_response = llm.invoke(topic_prompt)
    
    sentiment_prompt = f"分析情感倾向：{user_input}\n返回：积极/中性/消极"
    sentiment_response = llm.invoke(sentiment_prompt)
    
    return {
        **state,
        "topic": topic_response.content.strip(),
        "sentiment": sentiment_response.content.strip(),
        "current_speaker": "moderator"
    }

def expert_agent_node(state: ConversationState) -> ConversationState:
    """专家智能体节点"""
    user_input = state["user_input"]
    topic = state["topic"]
    
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"""
    作为{topic}领域的专家，回应用户：{user_input}
    提供专业、详细的回答。
    """
    
    response = llm.invoke(prompt)
    
    return {
        **state,
        "conversation_history": [f"Expert: {response.content}"],
        "current_speaker": "expert"
    }

def empathy_agent_node(state: ConversationState) -> ConversationState:
    """共情智能体节点"""
    user_input = state["user_input"]
    sentiment = state["sentiment"]
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = f"""
    用户情感状态：{sentiment}
    用户输入：{user_input}
    
    提供共情和情感支持的回应。
    """
    
    response = llm.invoke(prompt)
    
    return {
        **state,
        "conversation_history": [f"Empathy Agent: {response.content}"],
        "current_speaker": "empathy"
    }

def route_conversation(state: ConversationState) -> str:
    """对话路由"""
    sentiment = state["sentiment"]
    topic = state["topic"]
    
    # 根据情感和话题决定路由
    if sentiment == "消极":
        return "empathy_agent"
    elif any(keyword in topic.lower() for keyword in ["技术", "专业", "科学"]):
        return "expert_agent"
    else:
        return "expert_agent"  # 默认路由

# 构建对话工作流
def create_conversation_workflow():
    """创建对话工作流"""
    
    workflow = StateGraph(ConversationState)
    
    workflow.add_node("moderator", moderator_node)
    workflow.add_node("expert_agent", expert_agent_node)
    workflow.add_node("empathy_agent", empathy_agent_node)
    
    workflow.add_edge("moderator", "route_conversation")
    workflow.add_conditional_edges(
        "moderator",
        route_conversation,
        {
            "expert_agent": "expert_agent",
            "empathy_agent": "empathy_agent"
        }
    )
    
    workflow.add_edge("expert_agent", END)
    workflow.add_edge("empathy_agent", END)
    workflow.set_entry_point("moderator")
    
    return workflow.compile()

# 使用示例
async def run_conversation():
    """运行对话系统"""
    
    app = create_conversation_workflow()
    
    test_inputs = [
        "我最近工作压力很大，感觉很焦虑",
        "能解释一下机器学习的基本原理吗？",
        "今天天气真好，心情不错"
    ]
    
    for user_input in test_inputs:
        initial_state = {
            "user_input": user_input,
            "conversation_history": [],
            "current_speaker": "",
            "topic": "",
            "sentiment": ""
        }
        
        result = await app.ainvoke(initial_state)
        print(f"用户：{user_input}")
        print(f"系统回应：{result['conversation_history'][-1]}")
        print(f"话题：{result['topic']}, 情感：{result['sentiment']}")
        print("-" * 50)
```

---

## 2. 高级工作流构建与模式（3学时）

### 2.1 并行处理模式

在多智能体系统中，并行处理是提高效率的关键技术。LangGraph提供了多种并行处理模式。

#### 2.1.1 并行执行模式对比

| 并行模式 | 适用场景 | 优势 | 注意事项 |
|---------|---------|------|----------|
| **独立并行** | 无依赖的独立任务 | 最大化并行度 | 资源竞争 |
| **管道并行** | 流水线式处理 | 持续吞吐 | 同步复杂 |
| **分治并行** | 可分解的大任务 | 负载均衡 | 结果合并 |
| **条件并行** | 基于条件的选择性并行 | 资源优化 | 逻辑复杂 |

#### 2.1.2 独立并行处理实现

```python
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel
import asyncio
from typing import Dict, List

class ParallelAnalysisState(TypedDict):
    input_data: str
    market_analysis: str
    technical_analysis: str
    sentiment_analysis: str
    risk_analysis: str
    final_report: str

async def market_analysis_node(state: ParallelAnalysisState) -> ParallelAnalysisState:
    """市场分析节点"""
    data = state["input_data"]
    
    # 模拟市场分析处理
    await asyncio.sleep(2)  # 模拟处理时间
    
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"进行市场分析：{data}"
    response = await llm.ainvoke(prompt)
    
    return {
        **state,
        "market_analysis": response.content
    }

async def technical_analysis_node(state: ParallelAnalysisState) -> ParallelAnalysisState:
    """技术分析节点"""
    data = state["input_data"]
    
    await asyncio.sleep(1.5)  # 模拟处理时间
    
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"进行技术分析：{data}"
    response = await llm.ainvoke(prompt)
    
    return {
        **state,
        "technical_analysis": response.content
    }

async def sentiment_analysis_node(state: ParallelAnalysisState) -> ParallelAnalysisState:
    """情感分析节点"""
    data = state["input_data"]
    
    await asyncio.sleep(1)  # 模拟处理时间
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = f"进行情感分析：{data}"
    response = await llm.ainvoke(prompt)
    
    return {
        **state,
        "sentiment_analysis": response.content
    }

async def risk_analysis_node(state: ParallelAnalysisState) -> ParallelAnalysisState:
    """风险分析节点"""
    data = state["input_data"]
    
    await asyncio.sleep(2.5)  # 模拟处理时间
    
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"进行风险分析：{data}"
    response = await llm.ainvoke(prompt)
    
    return {
        **state,
        "risk_analysis": response.content
    }

async def synthesis_node(state: ParallelAnalysisState) -> ParallelAnalysisState:
    """综合分析节点"""
    
    # 等待所有分析完成
    analyses = {
        "市场分析": state["market_analysis"],
        "技术分析": state["technical_analysis"],
        "情感分析": state["sentiment_analysis"],
        "风险分析": state["risk_analysis"]
    }
    
    # 综合所有分析结果
    combined_analysis = "\n".join([f"{key}: {value}" for key, value in analyses.items()])
    
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"""
    基于以下分析结果，生成综合投资建议：
    {combined_analysis}
    
    请提供明确的投资建议和风险提示。
    """
    
    response = await llm.ainvoke(prompt)
    
    return {
        **state,
        "final_report": response.content
    }

def create_parallel_analysis_workflow():
    """创建并行分析工作流"""
    
    workflow = StateGraph(ParallelAnalysisState)
    
    # 添加并行分析节点
    workflow.add_node("market_analysis", market_analysis_node)
    workflow.add_node("technical_analysis", technical_analysis_node)
    workflow.add_node("sentiment_analysis", sentiment_analysis_node)
    workflow.add_node("risk_analysis", risk_analysis_node)
    workflow.add_node("synthesis", synthesis_node)
    
    # 设置并行执行
    # 所有分析节点并行执行
    workflow.set_entry_point("market_analysis")
    workflow.set_entry_point("technical_analysis")
    workflow.set_entry_point("sentiment_analysis")
    workflow.set_entry_point("risk_analysis")
    
    # 所有分析完成后进行综合
    workflow.add_edge("market_analysis", "synthesis")
    workflow.add_edge("technical_analysis", "synthesis")
    workflow.add_edge("sentiment_analysis", "synthesis")
    workflow.add_edge("risk_analysis", "synthesis")
    
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()

# 使用示例
async def run_parallel_analysis():
    """运行并行分析"""
    
    app = create_parallel_analysis_workflow()
    
    initial_state = {
        "input_data": "AAPL股票，当前价格150美元，近期财报显示营收增长15%",
        "market_analysis": "",
        "technical_analysis": "",
        "sentiment_analysis": "",
        "risk_analysis": "",
        "final_report": ""
    }
    
    start_time = asyncio.get_event_loop().time()
    result = await app.ainvoke(initial_state)
    end_time = asyncio.get_event_loop().time()
    
    print(f"并行处理完成，耗时：{end_time - start_time:.2f}秒")
    print(f"最终报告：{result['final_report']}")
    
    return result
```

#### 2.1.3 管道并行处理

```python
class PipelineState(TypedDict):
    raw_data: List[str]
    processed_items: Annotated[List[dict], operator.add]
    current_batch: List[str]
    batch_index: int
    total_batches: int

async def data_ingestion_node(state: PipelineState) -> PipelineState:
    """数据摄取节点"""
    raw_data = state["raw_data"]
    batch_size = 5
    
    # 分批处理
    batches = [raw_data[i:i+batch_size] for i in range(0, len(raw_data), batch_size)]
    
    return {
        **state,
        "total_batches": len(batches),
        "batch_index": 0,
        "current_batch": batches[0] if batches else []
    }

async def processing_node(state: PipelineState) -> PipelineState:
    """处理节点"""
    current_batch = state["current_batch"]
    batch_index = state["batch_index"]
    
    # 处理当前批次
    processed_batch = []
    for item in current_batch:
        # 模拟处理
        processed_item = {
            "original": item,
            "processed": f"processed_{item}",
            "batch": batch_index,
            "timestamp": datetime.now().isoformat()
        }
        processed_batch.append(processed_item)
    
    return {
        **state,
        "processed_items": processed_batch
    }

async def batch_controller_node(state: PipelineState) -> PipelineState:
    """批次控制节点"""
    batch_index = state["batch_index"]
    total_batches = state["total_batches"]
    raw_data = state["raw_data"]
    batch_size = 5
    
    # 准备下一批次
    next_batch_index = batch_index + 1
    
    if next_batch_index < total_batches:
        start_idx = next_batch_index * batch_size
        end_idx = min(start_idx + batch_size, len(raw_data))
        next_batch = raw_data[start_idx:end_idx]
        
        return {
            **state,
            "batch_index": next_batch_index,
            "current_batch": next_batch
        }
    else:
        return {
            **state,
            "current_batch": []
        }

def should_continue_pipeline(state: PipelineState) -> str:
    """判断是否继续管道处理"""
    current_batch = state["current_batch"]
    
    if current_batch:
        return "continue"
    else:
        return "end"

def create_pipeline_workflow():
    """创建管道工作流"""
    
    workflow = StateGraph(PipelineState)
    
    workflow.add_node("data_ingestion", data_ingestion_node)
    workflow.add_node("processing", processing_node)
    workflow.add_node("batch_controller", batch_controller_node)
    
    # 设置管道流程
    workflow.add_edge("data_ingestion", "processing")
    workflow.add_edge("processing", "batch_controller")
    
    workflow.add_conditional_edges(
        "batch_controller",
        should_continue_pipeline,
        {
            "continue": "processing",
            "end": END
        }
    )
    
    workflow.set_entry_point("data_ingestion")
    
    return workflow.compile()
```

### 2.3 循环控制与迭代优化

循环控制是处理需要多次迭代的任务的关键机制。

#### 2.3.1 循环控制模式

| 循环类型 | 控制条件 | 适用场景 | 实现要点 |
|---------|---------|---------|----------|
| **计数循环** | 固定次数 | 批处理任务 | 计数器管理 |
| **条件循环** | 动态条件 | 优化算法 | 收敛判断 |
| **反馈循环** | 质量评估 | 迭代改进 | 反馈机制 |
| **自适应循环** | 智能决策 | 复杂问题 | 学习调整 |

#### 2.3.2 迭代优化实现

```python
class IterativeOptimizationState(TypedDict):
    problem_description: str
    current_solution: str
    iteration_count: int
    max_iterations: int
    quality_score: float
    improvement_threshold: float
    optimization_history: Annotated[List[dict], operator.add]

async def solution_generator_node(state: IterativeOptimizationState) -> IterativeOptimizationState:
    """解决方案生成节点"""
    
    problem = state["problem_description"]
    current_solution = state["current_solution"]
    iteration = state["iteration_count"]
    history = state["optimization_history"]
    
    llm = ChatOpenAI(model="gpt-4")
    
    if iteration == 0:
        # 初始解决方案
        prompt = f"为以下问题生成初始解决方案：{problem}"
    else:
        # 基于历史改进
        recent_feedback = history[-1] if history else {}
        prompt = f"""
        问题：{problem}
        当前解决方案：{current_solution}
        上次反馈：{recent_feedback.get('feedback', '')}
        
        请生成改进的解决方案。
        """
    
    response = await llm.ainvoke(prompt)
    new_solution = response.content
    
    return {
        **state,
        "current_solution": new_solution,
        "iteration_count": iteration + 1
    }

async def quality_evaluator_node(state: IterativeOptimizationState) -> IterativeOptimizationState:
    """质量评估节点"""
    
    problem = state["problem_description"]
    solution = state["current_solution"]
    
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"""
    评估以下解决方案的质量（0-10分）：
    问题：{problem}
    解决方案：{solution}
    
    评估维度：
    1. 可行性 (0-10)
    2. 创新性 (0-10)
    3. 效率 (0-10)
    4. 完整性 (0-10)
    
    返回格式：
    总分：X.X
    可行性：X.X
    创新性：X.X
    效率：X.X
    完整性：X.X
    反馈：具体改进建议
    """
    
    response = await llm.ainvoke(prompt)
    evaluation = response.content
    
    # 解析评分
    try:
        lines = evaluation.split('\n')
        total_score = float(lines[0].split('：')[1])
    except:
        total_score = 5.0  # 默认分数
    
    # 记录优化历史
    history_entry = {
        "iteration": state["iteration_count"],
        "solution": solution,
        "score": total_score,
        "evaluation": evaluation,
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        **state,
        "quality_score": total_score,
        "optimization_history": [history_entry]
    }

def should_continue_optimization(state: IterativeOptimizationState) -> str:
    """判断是否继续优化"""
    
    iteration = state["iteration_count"]
    max_iterations = state["max_iterations"]
    quality_score = state["quality_score"]
    threshold = state["improvement_threshold"]
    history = state["optimization_history"]
    
    # 检查最大迭代次数
    if iteration >= max_iterations:
        return "max_iterations_reached"
    
    # 检查质量阈值
    if quality_score >= 9.0:
        return "quality_threshold_reached"
    
    # 检查改进幅度
    if len(history) >= 2:
        recent_scores = [entry["score"] for entry in history[-2:]]
        improvement = recent_scores[-1] - recent_scores[-2]
        
        if improvement < threshold:
            return "improvement_threshold_not_met"
    
    return "continue_optimization"

async def final_optimization_node(state: IterativeOptimizationState) -> IterativeOptimizationState:
    """最终优化节点"""
    
    solution = state["current_solution"]
    history = state["optimization_history"]
    
    # 生成优化报告
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"""
    基于优化历史，生成最终优化报告：
    
    最终解决方案：{solution}
    优化历史：{history}
    
    请总结：
    1. 优化过程
    2. 关键改进点
    3. 最终方案优势
    4. 实施建议
    """
    
    response = await llm.ainvoke(prompt)
    
    return {
        **state,
        "final_report": response.content
    }

def create_iterative_optimization_workflow():
    """创建迭代优化工作流"""
    
    workflow = StateGraph(IterativeOptimizationState)
    
    workflow.add_node("solution_generator", solution_generator_node)
    workflow.add_node("quality_evaluator", quality_evaluator_node)
    workflow.add_node("final_optimization", final_optimization_node)
    
    # 设置循环流程
    workflow.add_edge("solution_generator", "quality_evaluator")
    
    workflow.add_conditional_edges(
        "quality_evaluator",
        should_continue_optimization,
        {
            "continue_optimization": "solution_generator",
            "max_iterations_reached": "final_optimization",
            "quality_threshold_reached": "final_optimization",
            "improvement_threshold_not_met": "final_optimization"
        }
    )
    
    workflow.add_edge("final_optimization", END)
    workflow.set_entry_point("solution_generator")
    
    return workflow.compile()

# 使用示例
async def run_iterative_optimization():
    """运行迭代优化"""
    
    app = create_iterative_optimization_workflow()
    
    initial_state = {
        "problem_description": "设计一个高效的客户服务系统，能够处理多种类型的客户查询",
        "current_solution": "",
        "iteration_count": 0,
        "max_iterations": 5,
        "quality_score": 0.0,
        "improvement_threshold": 0.5,
        "optimization_history": []
    }
    
    result = await app.ainvoke(initial_state)
    
    print(f"优化完成，共进行{result['iteration_count']}次迭代")
    print(f"最终质量评分：{result['quality_score']}")
    print(f"最终解决方案：{result['current_solution']}")
    
    return result
```

### 2.4 错误处理与恢复机制

在复杂的多智能体工作流中，错误处理和恢复机制至关重要。企业级应用需要具备完善的容错能力和自动恢复机制。

> **💡 实际代码参考**：完整的错误处理和状态管理实现可参考项目中的 `langgraph_workflow.py` 文件，该文件提供了企业级的异常处理、状态验证和自动恢复功能。

#### 2.4.1 企业级错误处理策略

| 错误类型 | 处理策略 | 实现方式 | 适用场景 | 企业级增强 |
|---------|---------|---------|---------|------------|
| **网络错误** | 重试机制 | 指数退避 | API调用失败 | 智能重试、熔断器 |
| **数据错误** | 数据清洗 | 异常值处理 | 输入数据异常 | 数据验证、自动修复 |
| **逻辑错误** | 回滚重试 | 状态恢复 | 业务逻辑错误 | 智能回滚、版本控制 |
| **资源错误** | 降级处理 | 备用方案 | 资源不足 | 动态扩容、负载均衡 |
| **超时错误** | 异步处理 | 任务队列 | 长时间任务 | 分布式处理、进度追踪 |

#### 2.4.2 企业级错误处理实现

```python
# 基于项目 multi_agent_system/src/workflows/langgraph_workflow.py 的增强实现
from typing import Optional, Dict, Any, List
import traceback
import logging
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    """恢复策略"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"

@dataclass
class ErrorContext:
    """错误上下文"""
    error_id: str
    timestamp: datetime
    node_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    recoverable: bool
    retry_count: int
    max_retries: int
    recovery_strategy: RecoveryStrategy
    stack_trace: str
    system_state: Dict[str, Any]

class EnterpriseWorkflowState(TypedDict):
    """企业级工作流状态"""
    input_data: str
    processing_steps: List[str]
    error_log: Annotated[List[ErrorContext], operator.add]
    retry_count: int
    max_retries: int
    fallback_used: bool
    final_result: str
    performance_metrics: Dict[str, Any]
    circuit_breaker_status: Dict[str, bool]
    health_check_results: Dict[str, Any]

class CircuitBreaker:
    """熔断器实现"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """执行函数调用，带熔断保护"""
        if self.state == "OPEN":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e

class EnterpriseErrorHandler:
    """企业级错误处理器"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_patterns: Dict[str, int] = {}
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.logger = logging.getLogger("EnterpriseErrorHandler")
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """处理错误并生成错误上下文"""
        
        error_id = str(uuid.uuid4())
        node_name = context.get("node_name", "unknown")
        
        # 分析错误类型和严重程度
        error_type, severity = self._analyze_error(error)
        
        # 确定恢复策略
        recovery_strategy = self._determine_recovery_strategy(
            error_type, severity, context
        )
        
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            node_name=node_name,
            error_type=error_type,
            error_message=str(error),
            severity=severity,
            recoverable=recovery_strategy != RecoveryStrategy.ABORT,
            retry_count=context.get("retry_count", 0),
            max_retries=context.get("max_retries", 3),
            recovery_strategy=recovery_strategy,
            stack_trace=traceback.format_exc(),
            system_state=self._capture_system_state()
        )
        
        # 记录错误模式
        self._record_error_pattern(error_type)
        
        # 发送告警（如果是高严重程度错误）
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_alert(error_context)
        
        return error_context
    
    def _analyze_error(self, error: Exception) -> tuple[str, ErrorSeverity]:
        """分析错误类型和严重程度"""
        error_type = type(error).__name__
        
        # 根据错误类型确定严重程度
        severity_mapping = {
            "ConnectionError": ErrorSeverity.MEDIUM,
            "TimeoutError": ErrorSeverity.MEDIUM,
            "ValidationError": ErrorSeverity.LOW,
            "AuthenticationError": ErrorSeverity.HIGH,
            "SystemError": ErrorSeverity.CRITICAL,
            "MemoryError": ErrorSeverity.CRITICAL,
        }
        
        severity = severity_mapping.get(error_type, ErrorSeverity.MEDIUM)
        return error_type, severity
    
    def _determine_recovery_strategy(self, error_type: str, severity: ErrorSeverity, 
                                   context: Dict[str, Any]) -> RecoveryStrategy:
        """确定恢复策略"""
        
        retry_count = context.get("retry_count", 0)
        max_retries = context.get("max_retries", 3)
        
        # 基于错误类型和严重程度的策略矩阵
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ABORT
        
        if error_type in ["ConnectionError", "TimeoutError"] and retry_count < max_retries:
            return RecoveryStrategy.RETRY
        
        if error_type in ["ValidationError", "DataError"]:
            return RecoveryStrategy.FALLBACK
        
        if retry_count >= max_retries:
            return RecoveryStrategy.FALLBACK
        
        return RecoveryStrategy.RETRY
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """捕获系统状态"""
        return {
            "timestamp": datetime.now().isoformat(),
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent(),
            "disk_usage": psutil.disk_usage('/').percent,
            "active_connections": len(psutil.net_connections()),
        }
    
    def _record_error_pattern(self, error_type: str):
        """记录错误模式"""
        self.error_patterns[error_type] = self.error_patterns.get(error_type, 0) + 1
    
    def _send_alert(self, error_context: ErrorContext):
        """发送告警"""
        alert_message = {
            "error_id": error_context.error_id,
            "severity": error_context.severity.value,
            "node_name": error_context.node_name,
            "error_message": error_context.error_message,
            "timestamp": error_context.timestamp.isoformat()
        }
        
        # 这里可以集成实际的告警系统
        self.logger.critical(f"High severity error detected: {json.dumps(alert_message)}")

async def enterprise_risky_processing_node(state: EnterpriseWorkflowState) -> EnterpriseWorkflowState:
    """企业级风险处理节点"""
    
    input_data = state["input_data"]
    retry_count = state["retry_count"]
    error_handler = EnterpriseErrorHandler()
    
    try:
        # 健康检查
        health_status = await perform_health_check()
        
        # 使用熔断器保护
        circuit_breaker = error_handler.circuit_breakers.get("processing", CircuitBreaker())
        
        def risky_operation():
            if "error" in input_data.lower() and retry_count < 2:
                raise ConnectionError("Simulated network error")
            
            # 模拟处理
            return f"Processed: {input_data}"
        
        result = circuit_breaker.call(risky_operation)
        
        return {
            **state,
            "processing_steps": [f"Successfully processed: {result}"],
            "final_result": result,
            "health_check_results": health_status,
            "circuit_breaker_status": {"processing": circuit_breaker.state}
        }
        
    except Exception as e:
        # 使用企业级错误处理
        error_context = error_handler.handle_error(e, {
            "node_name": "enterprise_risky_processing",
            "retry_count": retry_count,
            "max_retries": state["max_retries"]
        })
        
        return {
            **state,
            "error_log": [error_context],
            "retry_count": retry_count + 1,
            "circuit_breaker_status": {"processing": "OPEN"}
        }

async def perform_health_check() -> Dict[str, Any]:
    """执行健康检查"""
    return {
        "database_connection": "healthy",
        "external_api": "healthy",
        "memory_usage": "normal",
        "response_time": "acceptable"
    }

def enterprise_should_retry_or_fallback(state: EnterpriseWorkflowState) -> str:
    """企业级重试或降级决策"""
    
    error_log = state["error_log"]
    circuit_breaker_status = state.get("circuit_breaker_status", {})
    
    if not error_log:
        return "success"
    
    latest_error = error_log[-1]
    
    # 检查熔断器状态
    if circuit_breaker_status.get("processing") == "OPEN":
        return "fallback"
    
    # 基于恢复策略决策
    if latest_error.recovery_strategy == RecoveryStrategy.RETRY:
        return "retry"
    elif latest_error.recovery_strategy == RecoveryStrategy.FALLBACK:
        return "fallback"
    elif latest_error.recovery_strategy == RecoveryStrategy.ABORT:
        return "abort"
    else:
        return "success"
```

> **🔧 企业级特性**：
>
> - **智能熔断器**：防止级联故障
> - **错误模式识别**：自动学习错误规律
> - **多级告警机制**：及时响应严重错误
> - **系统状态监控**：全面的健康检查
> - **自适应恢复策略**：基于错误类型智能选择恢复方案

#### 2.4.2 错误处理实现

```python
from typing import Optional
import traceback
import logging

class RobustWorkflowState(TypedDict):
    input_data: str
    processing_steps: List[str]
    error_log: Annotated[List[dict], operator.add]
    retry_count: int
    max_retries: int
    fallback_used: bool
    final_result: str

class WorkflowError(Exception):
    """工作流错误基类"""
    def __init__(self, message: str, error_type: str, recoverable: bool = True):
        super().__init__(message)
        self.error_type = error_type
        self.recoverable = recoverable

async def risky_processing_node(state: RobustWorkflowState) -> RobustWorkflowState:
    """可能出错的处理节点"""
    
    input_data = state["input_data"]
    retry_count = state["retry_count"]
    
    try:
        # 模拟可能失败的处理
        if "error" in input_data.lower() and retry_count < 2:
            raise WorkflowError(
                "Processing failed due to invalid input",
                "DATA_ERROR",
                recoverable=True
            )
        
        # 正常处理
        llm = ChatOpenAI(model="gpt-4")
        response = await llm.ainvoke(f"处理数据：{input_data}")
        
        return {
            **state,
            "processing_steps": [f"Step completed: {response.content[:100]}..."],
            "final_result": response.content
        }
        
    except WorkflowError as e:
        # 记录错误
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": e.error_type,
            "error_message": str(e),
            "retry_count": retry_count,
            "recoverable": e.recoverable,
            "stack_trace": traceback.format_exc()
        }
        
        return {
            **state,
            "error_log": [error_entry],
            "retry_count": retry_count + 1
        }
    
    except Exception as e:
        # 未预期的错误
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": "UNEXPECTED_ERROR",
            "error_message": str(e),
            "retry_count": retry_count,
            "recoverable": False,
            "stack_trace": traceback.format_exc()
        }
        
        return {
            **state,
            "error_log": [error_entry],
            "retry_count": retry_count + 1
        }

async def fallback_processing_node(state: RobustWorkflowState) -> RobustWorkflowState:
    """备用处理节点"""
    
    input_data = state["input_data"]
    
    # 使用简化的处理逻辑
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = f"使用简化方式处理：{input_data}"
    response = await llm.ainvoke(prompt)
    
    return {
        **state,
        "processing_steps": ["Fallback processing completed"],
        "fallback_used": True,
        "final_result": f"[Fallback] {response.content}"
    }

async def error_analysis_node(state: RobustWorkflowState) -> RobustWorkflowState:
    """错误分析节点"""
    
    error_log = state["error_log"]
    
    if error_log:
        # 分析错误模式
        error_types = [entry["error_type"] for entry in error_log]
        most_common_error = max(set(error_types), key=error_types.count)
        
        analysis = {
            "total_errors": len(error_log),
            "most_common_error": most_common_error,
            "error_pattern": error_types,
            "recommendations": generate_error_recommendations(error_log)
        }
        
        return {
            **state,
            "error_analysis": analysis
        }
    
    return state

def generate_error_recommendations(error_log: List[dict]) -> List[str]:
    """生成错误处理建议"""
    recommendations = []
    
    error_types = [entry["error_type"] for entry in error_log]
    
    if "DATA_ERROR" in error_types:
        recommendations.append("建议增强输入数据验证")
    
    if "NETWORK_ERROR" in error_types:
        recommendations.append("建议实施更强的重试机制")
    
    if len(error_log) > 3:
        recommendations.append("建议检查系统稳定性")
    
    return recommendations

def should_retry_or_fallback(state: RobustWorkflowState) -> str:
    """决定重试还是使用备用方案"""
    
    retry_count = state["retry_count"]
    max_retries = state["max_retries"]
    error_log = state["error_log"]
    
    # 检查是否有错误
    if not error_log:
        return "success"
    
    # 检查最新错误是否可恢复
    latest_error = error_log[-1]
    
    if not latest_error["recoverable"]:
        return "fallback"
    
    if retry_count < max_retries:
        return "retry"
    else:
        return "fallback"

def create_robust_workflow():
    """创建健壮的工作流"""
    
    workflow = StateGraph(RobustWorkflowState)
    
    workflow.add_node("risky_processing", risky_processing_node)
    workflow.add_node("fallback_processing", fallback_processing_node)
    workflow.add_node("error_analysis", error_analysis_node)
    
    # 设置错误处理流程
    workflow.add_conditional_edges(
        "risky_processing",
        should_retry_or_fallback,
        {
            "success": "error_analysis",
            "retry": "risky_processing",
            "fallback": "fallback_processing"
        }
    )
    
    workflow.add_edge("fallback_processing", "error_analysis")
    workflow.add_edge("error_analysis", END)
    workflow.set_entry_point("risky_processing")
    
    return workflow.compile()

# 使用示例
async def test_error_handling():
    """测试错误处理"""
    
    app = create_robust_workflow()
    
    test_cases = [
        "正常数据处理",
        "包含error的数据",  # 会触发错误
        "另一个正常数据"
    ]
    
    for test_data in test_cases:
        initial_state = {
            "input_data": test_data,
            "processing_steps": [],
            "error_log": [],
            "retry_count": 0,
            "max_retries": 3,
            "fallback_used": False,
            "final_result": ""
        }
        
        result = await app.ainvoke(initial_state)
        
        print(f"输入：{test_data}")
        print(f"结果：{result['final_result']}")
        print(f"错误次数：{len(result['error_log'])}")
        print(f"使用备用方案：{result['fallback_used']}")
        print("-" * 50)
```

### 2.5 性能优化技巧

#### 2.5.1 优化策略总结

| 优化维度 | 具体技巧 | 实现方法 | 性能提升 |
|---------|---------|---------|----------|
| **并发优化** | 异步处理、并行执行 | asyncio、多线程 | 2-5倍 |
| **缓存优化** | 结果缓存、状态缓存 | Redis、内存缓存 | 3-10倍 |
| **资源优化** | 连接池、对象复用 | 连接管理、对象池 | 20-50% |
| **算法优化** | 智能路由、负载均衡 | 启发式算法 | 30-100% |

#### 2.5.2 性能监控实现

```python
import time
import psutil
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PerformanceMetrics:
    """性能指标"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    node_execution_times: Dict[str, float]
    throughput: float

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.node_timings: Dict[str, List[float]] = {}
    
    async def monitor_workflow_execution(self, workflow_func, *args, **kwargs):
        """监控工作流执行"""
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        start_cpu = psutil.cpu_percent()
        
        # 执行工作流
        result = await workflow_func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        end_cpu = psutil.cpu_percent()
        
        # 计算指标
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = end_cpu - start_cpu
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            node_execution_times={},
            throughput=1.0 / execution_time if execution_time > 0 else 0
        )
        
        self.metrics_history.append(metrics)
        
        return result, metrics
    
    def get_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        recent_metrics = self.metrics_history[-10:]  # 最近10次执行
        
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        
        return {
            "average_execution_time": avg_execution_time,
            "average_memory_usage": avg_memory_usage,
            "average_throughput": avg_throughput,
            "total_executions": len(self.metrics_history),
            "performance_trend": self.analyze_performance_trend()
        }
    
    def analyze_performance_trend(self) -> str:
        """分析性能趋势"""
        
        if len(self.metrics_history) < 5:
            return "数据不足，无法分析趋势"
        
        recent_times = [m.execution_time for m in self.metrics_history[-5:]]
        earlier_times = [m.execution_time for m in self.metrics_history[-10:-5]]
        
        recent_avg = sum(recent_times) / len(recent_times)
        earlier_avg = sum(earlier_times) / len(earlier_times)
        
        if recent_avg < earlier_avg * 0.9:
            return "性能提升"
        elif recent_avg > earlier_avg * 1.1:
            return "性能下降"
        else:
            return "性能稳定"

# 使用示例
async def performance_test():
    """性能测试示例"""
    
    monitor = PerformanceMonitor()
    
    # 测试不同的工作流
    workflows = [
        create_customer_service_workflow(),
        create_parallel_analysis_workflow(),
        create_iterative_optimization_workflow()
    ]
    
    for i, workflow in enumerate(workflows):
        print(f"测试工作流 {i+1}")
        
        # 执行多次测试
        for j in range(5):
            test_state = {
                "input_data": f"测试数据 {j+1}",
                # ... 其他初始状态
            }
            
            result, metrics = await monitor.monitor_workflow_execution(
                workflow.ainvoke, test_state
            )
            
            print(f"  执行 {j+1}: {metrics.execution_time:.2f}s, "
                  f"内存: {metrics.memory_usage:.2f}MB")
        
        # 生成报告
        report = monitor.get_performance_report()
        print(f"  平均执行时间: {report['average_execution_time']:.2f}s")
        print(f"  性能趋势: {report['performance_trend']}")
        print("-" * 50)
```

### 2.6 实践环节

#### 2.6.1 综合练习：构建企业级多智能体客服系统

```python
# 企业级客服系统完整实现
class EnterpriseCustomerServiceState(TypedDict):
    customer_id: str
    query: str
    priority: str
    category: str
    assigned_agent: str
    conversation_history: Annotated[List[dict], operator.add]
    resolution_status: str
    satisfaction_score: float
    escalation_level: int
    processing_time: float

# 实现完整的企业级客服工作流
def create_enterprise_customer_service():
    """创建企业级客服系统"""
    
    workflow = StateGraph(EnterpriseCustomerServiceState)
    
    # 添加所有必要的节点
    workflow.add_node("intake_classification", intake_classification_node)
    workflow.add_node("priority_assessment", priority_assessment_node)
    workflow.add_node("agent_assignment", agent_assignment_node)
    workflow.add_node("primary_response", primary_response_node)
    workflow.add_node("quality_check", quality_check_node)
    workflow.add_node("escalation_handler", escalation_handler_node)
    workflow.add_node("satisfaction_survey", satisfaction_survey_node)
    workflow.add_node("case_closure", case_closure_node)
    
    # 设置复杂的条件流程
    workflow.add_edge("intake_classification", "priority_assessment")
    workflow.add_edge("priority_assessment", "agent_assignment")
    workflow.add_edge("agent_assignment", "primary_response")
    
    workflow.add_conditional_edges(
        "primary_response",
        check_response_quality,
        {
            "approved": "satisfaction_survey",
            "needs_improvement": "quality_check",
            "escalate": "escalation_handler"
        }
    )
    
    workflow.add_edge("quality_check", "primary_response")
    workflow.add_edge("escalation_handler", "satisfaction_survey")
    workflow.add_edge("satisfaction_survey", "case_closure")
    workflow.add_edge("case_closure", END)
    
    workflow.set_entry_point("intake_classification")
    
    return workflow.compile()

# 运行完整测试
async def run_enterprise_test():
    """运行企业级测试"""
    
    app = create_enterprise_customer_service()
    monitor = PerformanceMonitor()
    
    test_cases = [
        {
            "customer_id": "CUST001",
            "query": "我的账单有问题，费用比预期高很多",
            "priority": "",
            "category": "",
            "assigned_agent": "",
            "conversation_history": [],
            "resolution_status": "open",
            "satisfaction_score": 0.0,
            "escalation_level": 0,
            "processing_time": 0.0
        }
    ]
    
    for test_case in test_cases:
        result, metrics = await monitor.monitor_workflow_execution(
            app.ainvoke, test_case
        )
        
        print(f"客户ID: {result['customer_id']}")
        print(f"查询: {result['query']}")
        print(f"分配智能体: {result['assigned_agent']}")
        print(f"解决状态: {result['resolution_status']}")
        print(f"满意度: {result['satisfaction_score']}")
        print(f"处理时间: {metrics.execution_time:.2f}s")
        print("-" * 50)
    
    # 生成性能报告
    report = monitor.get_performance_report()
    print("性能报告:")
    print(f"平均执行时间: {report['average_execution_time']:.2f}s")
    print(f"平均吞吐量: {report['average_throughput']:.2f} requests/s")
```

这个第二天的培训材料涵盖了LangGraph框架的高级应用，包括并行处理、循环控制、错误处理和性能优化等关键技术，为学员提供了构建企业级多智能体系统的完整知识体系。
