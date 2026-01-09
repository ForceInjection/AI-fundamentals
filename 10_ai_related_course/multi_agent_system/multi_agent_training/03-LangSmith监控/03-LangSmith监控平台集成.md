# 第三天：LangSmith监控平台集成

## 学习目标

- 掌握LangSmith平台的核心功能和架构
- 学会集成LangSmith到多智能体系统
- 掌握系统监控、日志分析和性能优化技术
- 学会使用LangSmith进行调试和故障排除
- 了解企业级监控系统的设计和实现
- 掌握智能告警和自动化运维技术

## 参考项目

本课程深入讲解LangSmith监控平台的集成与应用，帮助学员掌握多智能体系统的监控、调试和性能优化技术。

**💡 实际代码参考**：完整的LangSmith监控集成实现可参考项目中的以下文件：

- `langsmith_integration.py` - 企业级全链路追踪系统
- `monitoring/system_monitor.py` - 系统性能监控
- `monitoring/alert_manager.py` - 智能告警管理
- `multi_agent_system/src/monitoring/` - 监控组件集合
- `customer_service_system.py` - 客服系统监控案例

**模拟环境说明**：本培训使用模拟的LangSmith环境进行教学，所有功能和API接口与真实LangSmith保持一致。在实际项目中，只需要替换为真实的LangSmith API密钥即可无缝切换到生产环境。

**代码引用**: 完整的企业级监控系统实现请参考 `multi_agent_system/src/monitoring/langsmith_integration.py`

企业级监控系统的核心特性：

**核心组件**：

- `AlertLevel`: 告警级别管理（信息、警告、错误、严重）
- `MetricType`: 指标类型分类（性能、业务、系统、安全）
- `MonitoringConfig`: 监控配置管理（采样率、批处理、保留策略）
- `EnterpriseMonitoringSystem`: 主监控系统，支持全链路追踪

**企业级特性**：

- 智能体执行全链路追踪
- 实时告警规则检查
- 性能指标自动收集
- 可配置的数据保留策略

---

## 1. LangSmith平台概述

### 1.1 什么是LangSmith

LangSmith是LangChain生态系统中的企业级监控和调试平台，专为LLM应用和多智能体系统设计。它提供了完整的可观测性解决方案，帮助开发者更好地理解和优化他们的AI应用。

### 1.2 核心价值

- **全链路追踪**：完整记录智能体交互过程，包括输入输出和中间步骤
- **性能监控**：实时监控系统性能指标，如响应时间、成功率等
- **调试支持**：提供强大的调试和故障排除工具，快速定位问题
- **数据分析**：深入分析智能体行为和系统效率，支持数据驱动的优化
- **成本控制**：监控Token使用量和API调用成本
- **企业级特性**：支持多租户、权限管理、数据安全等企业需求

### 1.3 企业级架构组件

```python
# 基于项目 multi_agent_system/src/monitoring/langsmith_integration.py 的架构设计
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid

@dataclass
class TracingContext:
    """追踪上下文"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    tags: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class RunMetrics:
    """运行指标"""
    run_id: str
    duration_ms: float
    token_usage: int
    cost_usd: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_count: int

class EnterpriseTracing:
    """企业级全链路追踪系统"""
    
    def __init__(self, project_name: str, environment: str = "production"):
        self.project_name = project_name
        self.environment = environment
        self.active_traces: Dict[str, TracingContext] = {}
        self.metrics_buffer: List[RunMetrics] = []
        self.alert_thresholds = {
            "error_rate": 0.05,  # 5% 错误率阈值
            "response_time": 5000,  # 5秒响应时间阈值
            "memory_usage": 80,  # 80% 内存使用率阈值
        }
    
    def start_trace(self, operation_name: str, 
                   parent_span_id: Optional[str] = None,
                   tags: Dict[str, Any] = None) -> str:
        """开始追踪"""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        context = TracingContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            tags=tags or {},
            metadata={}
        )
        
        self.active_traces[trace_id] = context
        return trace_id
    
    def end_trace(self, trace_id: str, 
                 outputs: Dict[str, Any] = None,
                 error: Optional[str] = None) -> RunMetrics:
        """结束追踪并生成指标"""
        if trace_id not in self.active_traces:
            raise ValueError(f"Trace {trace_id} not found")
        
        context = self.active_traces[trace_id]
        end_time = datetime.now()
        duration_ms = (end_time - context.start_time).total_seconds() * 1000
        
        # 生成运行指标
        metrics = RunMetrics(
            run_id=trace_id,
            duration_ms=duration_ms,
            token_usage=self._calculate_token_usage(outputs),
            cost_usd=self._calculate_cost(outputs),
            memory_usage_mb=self._get_memory_usage(),
            cpu_usage_percent=self._get_cpu_usage(),
            success_rate=1.0 if error is None else 0.0,
            error_count=1 if error else 0
        )
        
        self.metrics_buffer.append(metrics)
        
        # 检查告警阈值
        self._check_alerts(metrics)
        
        # 清理追踪上下文
        del self.active_traces[trace_id]
        
        return metrics
```

### 1.4 模拟环境与生产环境对比

| 特性 | 模拟环境 | 生产环境 | 说明 |
|------|---------|---------|------|
| **API接口** | 完全兼容 | 真实API | 接口设计完全一致 |
| **数据存储** | 内存存储 | 云端存储 | 模拟环境数据重启后清空 |
| **性能指标** | 模拟数据 | 真实数据 | 算法和逻辑完全相同 |
| **告警系统** | 本地日志 | 多渠道告警 | 支持邮件、短信、钉钉等 |
| **数据可视化** | 简化版本 | 完整仪表板 | 核心图表和指标一致 |
| **权限管理** | 简化版本 | 企业级RBAC | 支持多租户和细粒度权限 |

> **🚀 快速切换到生产环境**：
>
> ```python
> # 开发/测试环境
> tracer = EnterpriseTracing(
>     project_name="my-project",
>     environment="development",
>     api_key="mock-key"  # 使用模拟密钥
> )
> 
> # 生产环境
> tracer = EnterpriseTracing(
>     project_name="my-project", 
>     environment="production",
>     api_key=os.getenv("LANGSMITH_API_KEY")  # 使用真实密钥
> )
> ```

---

## 2. 核心功能详解

### 2.1 链路追踪(Tracing)

#### 2.1.1 基本概念

```python
from langsmith import Client
from langchain.callbacks import LangChainTracer

# 初始化LangSmith客户端
client = Client(
    api_url="https://api.smith.langchain.com",
    api_key="your-api-key"
)

# 配置追踪器
tracer = LangChainTracer(
    project_name="multi-agent-system",
    client=client
)
```

#### 2.1.2 智能体追踪实现

```python
from langchain.schema import BaseMessage
from typing import List, Dict, Any

class TracedAgent:
    def __init__(self, name: str, tracer: LangChainTracer):
        self.name = name
        self.tracer = tracer
    
    def process_message(self, message: str, context: Dict[str, Any] = None):
        """带追踪的消息处理"""
        with self.tracer.trace(
            name=f"{self.name}_process",
            inputs={"message": message, "context": context}
        ) as trace:
            try:
                # 模拟智能体处理逻辑
                result = self._internal_process(message, context)
                trace.outputs = {"result": result}
                return result
            except Exception as e:
                trace.error = str(e)
                raise
    
    def _internal_process(self, message: str, context: Dict[str, Any]):
        # 实际处理逻辑
        return f"Processed: {message}"
```

### 2.2 日志管理

#### 2.2.1 结构化日志

```python
import logging
from langsmith import Client
from datetime import datetime

class LangSmithLogger:
    def __init__(self, project_name: str):
        self.client = Client()
        self.project_name = project_name
        
    def log_agent_action(self, agent_name: str, action: str, 
                        inputs: Dict, outputs: Dict, metadata: Dict = None):
        """记录智能体行为"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_name": agent_name,
            "action": action,
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata or {}
        }
        
        self.client.create_run(
            name=f"{agent_name}_{action}",
            project_name=self.project_name,
            inputs=inputs,
            outputs=outputs,
            extra=metadata
        )
```

### 2.3 性能指标收集

#### 2.3.1 关键指标定义

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class PerformanceMetrics:
    agent_name: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    token_count: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None

class MetricsCollector:
    def __init__(self, langsmith_client: Client):
        self.client = langsmith_client
        self.metrics = []
    
    def start_operation(self, agent_name: str, operation: str) -> str:
        """开始操作计时"""
        run_id = f"{agent_name}_{operation}_{int(time.time())}"
        metric = PerformanceMetrics(
            agent_name=agent_name,
            operation=operation,
            start_time=time.time()
        )
        self.metrics.append(metric)
        return run_id
    
    def end_operation(self, run_id: str, success: bool = True, 
                     error_message: str = None):
        """结束操作计时"""
        # 查找对应的指标记录并更新
        for metric in self.metrics:
            if f"{metric.agent_name}_{metric.operation}" in run_id:
                metric.end_time = time.time()
                metric.duration = metric.end_time - metric.start_time
                metric.success = success
                metric.error_message = error_message
                
                # 发送到LangSmith
                self._send_metrics(metric)
                break
    
    def _send_metrics(self, metric: PerformanceMetrics):
        """发送指标到LangSmith"""
        self.client.create_run(
            name=f"metrics_{metric.agent_name}_{metric.operation}",
            inputs={"operation": metric.operation},
            outputs={
                "duration": metric.duration,
                "success": metric.success,
                "memory_usage": metric.memory_usage,
                "token_count": metric.token_count
            },
            extra={
                "agent_name": metric.agent_name,
                "error_message": metric.error_message
            }
        )
```

---

## 3. 集成实现

### 3.1 LangGraph + LangSmith集成

#### 3.1.1 基础集成配置

```python
from langgraph import StateGraph
from langsmith import Client
from langchain.callbacks import LangChainTracer
import os

# 环境配置
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "multi-agent-system"

class MonitoredMultiAgentSystem:
    def __init__(self):
        self.client = Client()
        self.tracer = LangChainTracer(project_name="multi-agent-system")
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """构建带监控的智能体图"""
        graph = StateGraph(dict)
        
        # 添加节点（智能体）
        graph.add_node("coordinator", self._coordinator_with_monitoring)
        graph.add_node("analyzer", self._analyzer_with_monitoring)
        graph.add_node("executor", self._executor_with_monitoring)
        
        # 定义边
        graph.add_edge("coordinator", "analyzer")
        graph.add_edge("analyzer", "executor")
        graph.add_edge("executor", "coordinator")
        
        # 设置入口点
        graph.set_entry_point("coordinator")
        
        return graph.compile()
    
    def _coordinator_with_monitoring(self, state: dict):
        """带监控的协调器智能体"""
        with self.tracer.trace(
            name="coordinator_process",
            inputs=state
        ) as trace:
            try:
                # 协调器逻辑
                result = self._coordinate_tasks(state)
                trace.outputs = result
                return result
            except Exception as e:
                trace.error = str(e)
                raise
    
    def _analyzer_with_monitoring(self, state: dict):
        """带监控的分析器智能体"""
        with self.tracer.trace(
            name="analyzer_process",
            inputs=state
        ) as trace:
            try:
                # 分析器逻辑
                result = self._analyze_data(state)
                trace.outputs = result
                return result
            except Exception as e:
                trace.error = str(e)
                raise
    
    def _executor_with_monitoring(self, state: dict):
        """带监控的执行器智能体"""
        with self.tracer.trace(
            name="executor_process",
            inputs=state
        ) as trace:
            try:
                # 执行器逻辑
                result = self._execute_tasks(state)
                trace.outputs = result
                return result
            except Exception as e:
                trace.error = str(e)
                raise
```

### 3.2 自定义监控装饰器

#### 3.2.1 智能体监控装饰器

```python
from functools import wraps
from typing import Callable, Any
import inspect

def monitor_agent(agent_name: str, operation: str = None):
    """智能体监控装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 获取操作名称
            op_name = operation or func.__name__
            
            # 开始追踪
            with LangChainTracer().trace(
                name=f"{agent_name}_{op_name}",
                inputs={"args": args, "kwargs": kwargs}
            ) as trace:
                try:
                    # 执行原函数
                    result = func(*args, **kwargs)
                    trace.outputs = {"result": result}
                    return result
                except Exception as e:
                    trace.error = str(e)
                    raise
        return wrapper
    return decorator

# 使用示例
class SmartAgent:
    @monitor_agent("smart_agent", "process_request")
    def process_request(self, request: str) -> str:
        """处理请求"""
        # 实际处理逻辑
        return f"Processed: {request}"
    
    @monitor_agent("smart_agent", "analyze_data")
    def analyze_data(self, data: dict) -> dict:
        """分析数据"""
        # 分析逻辑
        return {"analysis": "completed", "insights": ["insight1", "insight2"]}
```

---

## 4. 监控与分析

### 4.1 实时监控面板

#### 4.1.1 关键指标监控

```python
from langsmith import Client
from typing import List, Dict
import asyncio

class RealTimeMonitor:
    def __init__(self, project_name: str):
        self.client = Client()
        self.project_name = project_name
        self.is_monitoring = False
    
    async def start_monitoring(self, interval: int = 30):
        """开始实时监控"""
        self.is_monitoring = True
        while self.is_monitoring:
            try:
                metrics = await self._collect_metrics()
                await self._analyze_metrics(metrics)
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"监控错误: {e}")
    
    async def _collect_metrics(self) -> Dict:
        """收集系统指标"""
        runs = self.client.list_runs(
            project_name=self.project_name,
            limit=100
        )
        
        metrics = {
            "total_runs": len(runs),
            "success_rate": 0,
            "avg_duration": 0,
            "error_count": 0,
            "agent_performance": {}
        }
        
        if runs:
            successful_runs = [r for r in runs if not r.error]
            metrics["success_rate"] = len(successful_runs) / len(runs)
            
            durations = [r.total_time for r in runs if r.total_time]
            if durations:
                metrics["avg_duration"] = sum(durations) / len(durations)
            
            metrics["error_count"] = len(runs) - len(successful_runs)
        
        return metrics
    
    async def _analyze_metrics(self, metrics: Dict):
        """分析指标并生成告警"""
        # 性能告警
        if metrics["avg_duration"] > 5.0:  # 5秒阈值
            await self._send_alert("性能告警", f"平均响应时间过长: {metrics['avg_duration']:.2f}s")
        
        # 错误率告警
        if metrics["success_rate"] < 0.95:  # 95%成功率阈值
            await self._send_alert("错误率告警", f"成功率过低: {metrics['success_rate']:.2%}")
    
    async def _send_alert(self, alert_type: str, message: str):
        """发送告警"""
        print(f"[{alert_type}] {message}")
        # 这里可以集成邮件、Slack等告警渠道
```

### 4.2 数据分析与洞察

#### 4.2.1 智能体行为分析

```python
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

class AgentAnalytics:
    def __init__(self, langsmith_client: Client):
        self.client = langsmith_client
    
    def analyze_agent_performance(self, agent_name: str, 
                                 time_range: int = 24) -> Dict:
        """分析智能体性能"""
        # 获取指定时间范围内的运行数据
        runs = self.client.list_runs(
            project_name="multi-agent-system",
            filter=f"name LIKE '%{agent_name}%'",
            limit=1000
        )
        
        if not runs:
            return {"error": "没有找到相关数据"}
        
        # 转换为DataFrame进行分析
        data = []
        for run in runs:
            data.append({
                "timestamp": run.start_time,
                "duration": run.total_time or 0,
                "success": not bool(run.error),
                "tokens": run.prompt_tokens + run.completion_tokens if hasattr(run, 'prompt_tokens') else 0,
                "operation": run.name
            })
        
        df = pd.DataFrame(data)
        
        # 计算关键指标
        analysis = {
            "total_operations": len(df),
            "success_rate": df["success"].mean(),
            "avg_duration": df["duration"].mean(),
            "max_duration": df["duration"].max(),
            "min_duration": df["duration"].min(),
            "total_tokens": df["tokens"].sum(),
            "operations_per_hour": self._calculate_ops_per_hour(df),
            "performance_trend": self._calculate_trend(df)
        }
        
        return analysis
    
    def _calculate_ops_per_hour(self, df: pd.DataFrame) -> float:
        """计算每小时操作数"""
        if df.empty:
            return 0
        
        time_span = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
        return len(df) / max(time_span, 1)
    
    def _calculate_trend(self, df: pd.DataFrame) -> str:
        """计算性能趋势"""
        if len(df) < 10:
            return "数据不足"
        
        # 按时间排序并计算移动平均
        df_sorted = df.sort_values("timestamp")
        df_sorted["duration_ma"] = df_sorted["duration"].rolling(window=10).mean()
        
        # 比较最近和早期的平均值
        recent_avg = df_sorted["duration_ma"].tail(10).mean()
        early_avg = df_sorted["duration_ma"].head(10).mean()
        
        if recent_avg > early_avg * 1.1:
            return "性能下降"
        elif recent_avg < early_avg * 0.9:
            return "性能提升"
        else:
            return "性能稳定"
```

---

## 5. 性能优化

### 5.1 基于监控数据的优化

#### 5.1.1 自动优化建议

```python
class PerformanceOptimizer:
    def __init__(self, analytics: AgentAnalytics):
        self.analytics = analytics
    
    def generate_optimization_suggestions(self, agent_name: str) -> List[Dict]:
        """生成优化建议"""
        analysis = self.analytics.analyze_agent_performance(agent_name)
        suggestions = []
        
        # 响应时间优化
        if analysis["avg_duration"] > 3.0:
            suggestions.append({
                "type": "performance",
                "priority": "high",
                "issue": "响应时间过长",
                "suggestion": "考虑优化模型调用、增加缓存或并行处理",
                "expected_improvement": "30-50%响应时间减少"
            })
        
        # 成功率优化
        if analysis["success_rate"] < 0.95:
            suggestions.append({
                "type": "reliability",
                "priority": "critical",
                "issue": "成功率过低",
                "suggestion": "增加错误处理、重试机制和输入验证",
                "expected_improvement": "成功率提升至98%+"
            })
        
        # 资源使用优化
        if analysis["total_tokens"] > 100000:
            suggestions.append({
                "type": "cost",
                "priority": "medium",
                "issue": "Token使用量过高",
                "suggestion": "优化提示词、使用更小的模型或实现智能缓存",
                "expected_improvement": "20-40%成本降低"
            })
        
        return suggestions
    
    def implement_caching_optimization(self, agent_class):
        """实现缓存优化"""
        from functools import lru_cache
        import hashlib
        
        class CachedAgent(agent_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._cache = {}
            
            def _get_cache_key(self, inputs: Dict) -> str:
                """生成缓存键"""
                content = str(sorted(inputs.items()))
                return hashlib.md5(content.encode()).hexdigest()
            
            def process_with_cache(self, inputs: Dict, ttl: int = 3600):
                """带缓存的处理"""
                cache_key = self._get_cache_key(inputs)
                
                # 检查缓存
                if cache_key in self._cache:
                    cached_result, timestamp = self._cache[cache_key]
                    if time.time() - timestamp < ttl:
                        return cached_result
                
                # 执行处理
                result = self.process(inputs)
                
                # 存储到缓存
                self._cache[cache_key] = (result, time.time())
                
                return result
        
        return CachedAgent
```

### 5.2 智能负载均衡

#### 5.2.1 基于性能的负载分配

```python
import heapq
from typing import List, Dict, Any

class IntelligentLoadBalancer:
    def __init__(self, agents: List[Any]):
        self.agents = agents
        self.agent_metrics = {id(agent): {"load": 0, "avg_response_time": 0} 
                             for agent in agents}
    
    def select_optimal_agent(self, task_complexity: float = 1.0) -> Any:
        """选择最优智能体"""
        # 计算每个智能体的负载分数
        scores = []
        for agent in self.agents:
            agent_id = id(agent)
            metrics = self.agent_metrics[agent_id]
            
            # 综合考虑负载和响应时间
            score = (metrics["load"] * 0.6 + 
                    metrics["avg_response_time"] * 0.4) * task_complexity
            
            heapq.heappush(scores, (score, agent))
        
        # 返回负载最低的智能体
        _, optimal_agent = heapq.heappop(scores)
        return optimal_agent
    
    def update_agent_metrics(self, agent: Any, response_time: float):
        """更新智能体指标"""
        agent_id = id(agent)
        metrics = self.agent_metrics[agent_id]
        
        # 更新平均响应时间（指数移动平均）
        alpha = 0.3
        metrics["avg_response_time"] = (
            alpha * response_time + 
            (1 - alpha) * metrics["avg_response_time"]
        )
        
        # 更新负载（简单递减）
        metrics["load"] = max(0, metrics["load"] - 0.1)
    
    def assign_task(self, task: Dict[str, Any]) -> Any:
        """分配任务"""
        # 评估任务复杂度
        complexity = self._evaluate_task_complexity(task)
        
        # 选择最优智能体
        agent = self.select_optimal_agent(complexity)
        
        # 增加负载
        agent_id = id(agent)
        self.agent_metrics[agent_id]["load"] += complexity
        
        return agent
    
    def _evaluate_task_complexity(self, task: Dict[str, Any]) -> float:
        """评估任务复杂度"""
        # 简单的复杂度评估逻辑
        base_complexity = 1.0
        
        # 根据任务类型调整
        if task.get("type") == "analysis":
            base_complexity *= 1.5
        elif task.get("type") == "generation":
            base_complexity *= 2.0
        
        # 根据数据量调整
        data_size = len(str(task.get("data", "")))
        if data_size > 1000:
            base_complexity *= 1.2
        
        return base_complexity
```

---

## 6. 实践练习

### 练习1：基础监控集成

**目标**：为现有的多智能体系统集成LangSmith监控

**任务**：

1. 配置LangSmith环境
2. 为智能体添加基础追踪
3. 实现日志收集
4. 创建简单的监控面板

**代码框架**：

```python
# 学员需要完成的代码
class MonitoredAgent:
    def __init__(self, name: str):
        self.name = name
        # TODO: 初始化LangSmith客户端和追踪器
    
    def process_task(self, task: str):
        # TODO: 添加监控和追踪逻辑
        pass
    
    def get_performance_metrics(self):
        # TODO: 获取性能指标
        pass
```

### 练习2：性能分析与优化

**目标**：分析智能体性能并实现优化

**任务**：

1. 收集性能数据
2. 分析瓶颈
3. 实现缓存机制
4. 验证优化效果

### 练习3：告警系统实现

**目标**：构建智能告警系统

**任务**：

1. 定义告警规则
2. 实现告警触发逻辑
3. 集成通知渠道
4. 测试告警功能

## 总结

本课程涵盖了LangSmith监控平台的核心功能和集成方法，通过实践练习帮助学员掌握：

1. **监控集成**：完整的LangSmith集成流程
2. **性能分析**：深入的性能分析和优化技术
3. **实时监控**：构建实时监控和告警系统
4. **最佳实践**：企业级监控的最佳实践

下一课程将学习企业级系统架构设计与实现。
