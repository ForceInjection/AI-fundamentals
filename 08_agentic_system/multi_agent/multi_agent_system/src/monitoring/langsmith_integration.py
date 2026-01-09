import asyncio
import json
import uuid
import time
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from contextlib import asynccontextmanager
import traceback
from collections import defaultdict, deque

# LangSmith相关导入（模拟）
try:
    from langsmith import Client, traceable
    from langsmith.run_trees import RunTree
except ImportError:
    # 如果LangSmith未安装，提供模拟实现
    class Client:
        def __init__(self, api_key=None, api_url=None):
            self.api_key = api_key
            self.api_url = api_url
            
        def create_run(self, **kwargs):
            return MockRun(kwargs)
            
        def update_run(self, run_id, **kwargs):
            pass
            
        def create_feedback(self, **kwargs):
            pass
    
    class MockRun:
        def __init__(self, data):
            self.id = str(uuid.uuid4())
            self.data = data
    
    def traceable(func):
        return func
    
    class RunTree:
        def __init__(self, **kwargs):
            self.id = str(uuid.uuid4())
            self.data = kwargs

class TraceLevel(Enum):
    """追踪级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class RunStatus(Enum):
    """运行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TraceContext:
    """追踪上下文"""
    trace_id: str
    parent_run_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class RunMetrics:
    """运行指标"""
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def calculate_duration(self):
        """计算执行时长"""
        if self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

@dataclass
class TraceEvent:
    """追踪事件"""
    event_id: str
    trace_id: str
    run_id: str
    event_type: str
    timestamp: datetime
    level: TraceLevel
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    stack_trace: Optional[str] = None

class EnterpriseTracing:
    """企业级全链路追踪系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # LangSmith客户端 - 始终使用模拟客户端以避免认证问题
        # 创建模拟客户端类
        class MockLangSmithClient:
            def __init__(self, api_key=None, api_url=None):
                self.api_key = api_key
                self.api_url = api_url
                
            def create_run(self, **kwargs):
                # 创建模拟运行对象
                class MockRun:
                    def __init__(self, data):
                        self.id = str(uuid.uuid4())
                        self.data = data
                return MockRun(kwargs)
                
            def update_run(self, run_id, **kwargs):
                pass
                
            def create_feedback(self, **kwargs):
                pass
        
        self.langsmith_client = MockLangSmithClient(
            api_key=self.config.get("langsmith_api_key"),
            api_url=self.config.get("langsmith_api_url")
        )
        self.use_real_langsmith = False  # 标记使用模拟客户端
        
        # 追踪状态管理
        self.active_traces: Dict[str, TraceContext] = {}
        self.active_runs: Dict[str, Dict[str, Any]] = {}
        
        # 事件存储
        self.trace_events: deque = deque(maxlen=self.config.get("max_events", 10000))
        self.event_handlers: List[Callable[[TraceEvent], None]] = []
        
        # 性能监控
        self.performance_metrics = {
            "total_traces": 0,
            "active_traces_count": 0,
            "avg_trace_duration": 0.0,
            "error_rate": 0.0,
            "throughput_per_minute": 0.0
        }
        
        # 采样配置
        self.sampling_rate = self.config.get("sampling_rate", 1.0)  # 1.0 = 100%采样
        self.error_sampling_rate = self.config.get("error_sampling_rate", 1.0)
        
        # 批处理配置
        self.batch_size = self.config.get("batch_size", 100)
        self.batch_timeout = self.config.get("batch_timeout", 5.0)  # 秒
        self.pending_events: List[TraceEvent] = []
        
        # 启动后台任务
        self._background_tasks: List[asyncio.Task] = []
        self._running = False
    
    async def start(self):
        """启动追踪系统"""
        if self._running:
            return
        
        self._running = True
        self.logger.info("Starting enterprise tracing system")
        
        # 启动批处理任务
        batch_task = asyncio.create_task(self._batch_processor())
        self._background_tasks.append(batch_task)
        
        # 启动性能监控任务
        metrics_task = asyncio.create_task(self._metrics_collector())
        self._background_tasks.append(metrics_task)
        
        self.logger.info("Enterprise tracing system started")
    
    async def stop(self):
        """停止追踪系统"""
        if not self._running:
            return
        
        self._running = False
        self.logger.info("Stopping enterprise tracing system")
        
        # 取消后台任务
        for task in self._background_tasks:
            task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # 处理剩余的事件
        if self.pending_events:
            await self._flush_events()
        
        self.logger.info("Enterprise tracing system stopped")
    
    def create_trace(self, name: str, session_id: Optional[str] = None, 
                    user_id: Optional[str] = None, tags: List[str] = None,
                    metadata: Dict[str, Any] = None) -> str:
        """创建新的追踪"""
        trace_id = str(uuid.uuid4())
        
        trace_context = TraceContext(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.active_traces[trace_id] = trace_context
        self.performance_metrics["total_traces"] += 1
        
        # 记录追踪开始事件
        self._add_event(
            trace_id=trace_id,
            run_id=trace_id,
            event_type="trace_start",
            level=TraceLevel.INFO,
            message=f"Trace started: {name}",
            data={"name": name, "tags": tags, "metadata": metadata}
        )
        
        self.logger.debug(f"Created trace {trace_id} for {name}")
        return trace_id
    
    @asynccontextmanager
    async def trace_run(self, trace_id: str, name: str, run_type: str = "llm",
                       inputs: Dict[str, Any] = None, tags: List[str] = None,
                       metadata: Dict[str, Any] = None):
        """追踪运行上下文管理器"""
        run_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # 创建运行记录
        run_data = {
            "id": run_id,
            "trace_id": trace_id,
            "name": name,
            "run_type": run_type,
            "inputs": inputs or {},
            "tags": tags or [],
            "metadata": metadata or {},
            "status": RunStatus.RUNNING,
            "metrics": RunMetrics(start_time=start_time)
        }
        
        self.active_runs[run_id] = run_data
        
        # 记录运行开始事件
        self._add_event(
            trace_id=trace_id,
            run_id=run_id,
            event_type="run_start",
            level=TraceLevel.INFO,
            message=f"Run started: {name}",
            data={"run_type": run_type, "inputs": inputs}
        )
        
        langsmith_run = None
        try:
            # 创建LangSmith运行
            langsmith_run = self.langsmith_client.create_run(
                name=name,
                run_type=run_type,
                inputs=inputs or {},
                tags=tags or [],
                extra=metadata or {}
            )
            
            run_data["langsmith_run_id"] = langsmith_run.id
            
            yield RunContext(self, trace_id, run_id, run_data)
            
            # 运行成功完成
            run_data["status"] = RunStatus.COMPLETED
            end_time = datetime.now()
            run_data["metrics"].end_time = end_time
            run_data["metrics"].calculate_duration()
            
            # 更新LangSmith运行
            if langsmith_run:
                self.langsmith_client.update_run(
                    langsmith_run.id,
                    outputs=run_data.get("outputs", {}),
                    end_time=end_time
                )
            
            self._add_event(
                trace_id=trace_id,
                run_id=run_id,
                event_type="run_complete",
                level=TraceLevel.INFO,
                message=f"Run completed: {name}",
                data={
                    "duration_ms": run_data["metrics"].duration_ms,
                    "outputs": run_data.get("outputs", {})
                }
            )
            
        except Exception as e:
            # 运行失败
            run_data["status"] = RunStatus.FAILED
            run_data["error"] = str(e)
            run_data["stack_trace"] = traceback.format_exc()
            
            end_time = datetime.now()
            run_data["metrics"].end_time = end_time
            run_data["metrics"].calculate_duration()
            
            # 更新LangSmith运行
            if langsmith_run:
                self.langsmith_client.update_run(
                    langsmith_run.id,
                    error=str(e),
                    end_time=end_time
                )
            
            self._add_event(
                trace_id=trace_id,
                run_id=run_id,
                event_type="run_error",
                level=TraceLevel.ERROR,
                message=f"Run failed: {name}",
                data={"duration_ms": run_data["metrics"].duration_ms},
                error=str(e),
                stack_trace=run_data["stack_trace"]
            )
            
            raise
        
        finally:
            # 清理运行记录
            if run_id in self.active_runs:
                del self.active_runs[run_id]
    
    def _add_event(self, trace_id: str, run_id: str, event_type: str,
                  level: TraceLevel, message: str, data: Dict[str, Any] = None,
                  error: str = None, stack_trace: str = None):
        """添加追踪事件"""
        event = TraceEvent(
            event_id=str(uuid.uuid4()),
            trace_id=trace_id,
            run_id=run_id,
            event_type=event_type,
            timestamp=datetime.now(),
            level=level,
            message=message,
            data=data or {},
            error=error,
            stack_trace=stack_trace
        )
        
        # 应用采样
        if self._should_sample(event):
            self.trace_events.append(event)
            self.pending_events.append(event)
            
            # 调用事件处理器
            for handler in self.event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler failed: {str(e)}")
    
    def _should_sample(self, event: TraceEvent) -> bool:
        """判断是否应该采样此事件"""
        if event.level in [TraceLevel.ERROR, TraceLevel.CRITICAL]:
            return True  # 总是采样错误事件
        
        import random
        return random.random() < self.sampling_rate
    
    async def _batch_processor(self):
        """批处理事件处理器"""
        while self._running:
            try:
                # 等待批处理超时或达到批处理大小
                await asyncio.sleep(self.batch_timeout)
                
                if self.pending_events:
                    await self._flush_events()
                    
            except Exception as e:
                self.logger.error(f"Batch processor error: {str(e)}")
    
    async def _flush_events(self):
        """刷新待处理事件"""
        if not self.pending_events:
            return
        
        events_to_process = self.pending_events[:self.batch_size]
        self.pending_events = self.pending_events[self.batch_size:]
        
        # 这里可以添加将事件发送到外部系统的逻辑
        # 例如：发送到日志系统、指标系统等
        
        self.logger.debug(f"Processed {len(events_to_process)} events")
    
    async def _metrics_collector(self):
        """性能指标收集器"""
        while self._running:
            try:
                await asyncio.sleep(60)  # 每分钟收集一次指标
                
                # 更新性能指标
                self.performance_metrics["active_traces_count"] = len(self.active_traces)
                
                # 计算错误率
                recent_events = [e for e in self.trace_events 
                               if (datetime.now() - e.timestamp).total_seconds() < 3600]  # 最近1小时
                
                if recent_events:
                    error_events = [e for e in recent_events if e.level == TraceLevel.ERROR]
                    self.performance_metrics["error_rate"] = len(error_events) / len(recent_events)
                
                # 计算吞吐量
                recent_traces = [e for e in recent_events if e.event_type == "trace_start"]
                self.performance_metrics["throughput_per_minute"] = len(recent_traces)
                
            except Exception as e:
                self.logger.error(f"Metrics collector error: {str(e)}")
    
    def add_event_handler(self, handler: Callable[[TraceEvent], None]):
        """添加事件处理器"""
        self.event_handlers.append(handler)
    
    def remove_event_handler(self, handler: Callable[[TraceEvent], None]):
        """移除事件处理器"""
        if handler in self.event_handlers:
            self.event_handlers.remove(handler)
    
    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """获取追踪摘要"""
        if trace_id not in self.active_traces:
            return None
        
        trace_context = self.active_traces[trace_id]
        trace_events = [e for e in self.trace_events if e.trace_id == trace_id]
        
        # 计算统计信息
        start_time = min(e.timestamp for e in trace_events) if trace_events else None
        end_time = max(e.timestamp for e in trace_events) if trace_events else None
        duration = (end_time - start_time).total_seconds() if start_time and end_time else 0
        
        error_count = len([e for e in trace_events if e.level == TraceLevel.ERROR])
        
        return {
            "trace_id": trace_id,
            "context": asdict(trace_context),
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "duration_seconds": duration,
            "event_count": len(trace_events),
            "error_count": error_count,
            "status": "error" if error_count > 0 else "success"
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return dict(self.performance_metrics)
    
    def search_traces(self, filters: Dict[str, Any] = None, 
                     limit: int = 100) -> List[Dict[str, Any]]:
        """搜索追踪"""
        filters = filters or {}
        
        # 过滤事件
        filtered_events = list(self.trace_events)
        
        if "trace_id" in filters:
            filtered_events = [e for e in filtered_events if e.trace_id == filters["trace_id"]]
        
        if "level" in filters:
            filtered_events = [e for e in filtered_events if e.level == TraceLevel(filters["level"])]
        
        if "event_type" in filters:
            filtered_events = [e for e in filtered_events if e.event_type == filters["event_type"]]
        
        if "start_time" in filters:
            start_time = datetime.fromisoformat(filters["start_time"])
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if "end_time" in filters:
            end_time = datetime.fromisoformat(filters["end_time"])
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        # 按追踪ID分组
        traces_by_id = defaultdict(list)
        for event in filtered_events:
            traces_by_id[event.trace_id].append(event)
        
        # 生成追踪摘要
        results = []
        for trace_id, events in list(traces_by_id.items())[:limit]:
            summary = self.get_trace_summary(trace_id)
            if summary:
                summary["events"] = [asdict(e) for e in events]
                results.append(summary)
        
        return results

class RunContext:
    """运行上下文"""
    
    def __init__(self, tracer: EnterpriseTracing, trace_id: str, 
                 run_id: str, run_data: Dict[str, Any]):
        self.tracer = tracer
        self.trace_id = trace_id
        self.run_id = run_id
        self.run_data = run_data
    
    def log(self, message: str, level: TraceLevel = TraceLevel.INFO, 
           data: Dict[str, Any] = None):
        """记录日志"""
        self.tracer._add_event(
            trace_id=self.trace_id,
            run_id=self.run_id,
            event_type="log",
            level=level,
            message=message,
            data=data
        )
    
    def set_outputs(self, outputs: Dict[str, Any]):
        """设置输出"""
        self.run_data["outputs"] = outputs
    
    def update_metrics(self, **metrics):
        """更新指标"""
        for key, value in metrics.items():
            setattr(self.run_data["metrics"], key, value)
    
    def add_feedback(self, score: float, comment: str = None, 
                    feedback_type: str = "user"):
        """添加反馈"""
        try:
            self.tracer.langsmith_client.create_feedback(
                run_id=self.run_data.get("langsmith_run_id"),
                score=score,
                comment=comment,
                feedback_source_type=feedback_type
            )
            
            self.log(
                f"Feedback added: {score}",
                level=TraceLevel.INFO,
                data={"score": score, "comment": comment, "type": feedback_type}
            )
            
        except Exception as e:
            self.log(
                f"Failed to add feedback: {str(e)}",
                level=TraceLevel.ERROR,
                data={"error": str(e)}
            )

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, tracer: EnterpriseTracing):
        self.tracer = tracer
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 性能阈值
        self.thresholds = {
            "max_duration_ms": 30000,  # 30秒
            "max_error_rate": 0.05,    # 5%
            "max_memory_mb": 1000,     # 1GB
            "max_cpu_percent": 80      # 80%
        }
        
        # 告警处理器
        self.alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def check_performance_thresholds(self, metrics: Dict[str, Any]):
        """检查性能阈值"""
        alerts = []
        
        # 检查执行时长
        if metrics.get("avg_trace_duration", 0) > self.thresholds["max_duration_ms"]:
            alerts.append({
                "type": "high_duration",
                "message": f"Average trace duration exceeds threshold",
                "value": metrics["avg_trace_duration"],
                "threshold": self.thresholds["max_duration_ms"]
            })
        
        # 检查错误率
        if metrics.get("error_rate", 0) > self.thresholds["max_error_rate"]:
            alerts.append({
                "type": "high_error_rate",
                "message": f"Error rate exceeds threshold",
                "value": metrics["error_rate"],
                "threshold": self.thresholds["max_error_rate"]
            })
        
        # 触发告警
        for alert in alerts:
            self._trigger_alert(alert["type"], alert)
    
    def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """触发告警"""
        self.logger.warning(f"Performance alert: {alert_type} - {alert_data['message']}")
        
        for handler in self.alert_handlers:
            try:
                handler(alert_type, alert_data)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {str(e)}")
    
    def add_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)

# 装饰器函数
def trace_agent_execution(tracer: EnterpriseTracing, agent_name: str):
    """智能体执行追踪装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            trace_id = tracer.create_trace(
                name=f"Agent Execution: {agent_name}",
                tags=["agent", agent_name],
                metadata={"function": func.__name__}
            )
            
            async with tracer.trace_run(
                trace_id=trace_id,
                name=func.__name__,
                run_type="agent",
                inputs={"args": str(args), "kwargs": kwargs}
            ) as run_context:
                try:
                    result = await func(*args, **kwargs)
                    run_context.set_outputs({"result": result})
                    return result
                except Exception as e:
                    run_context.log(f"Agent execution failed: {str(e)}", TraceLevel.ERROR)
                    raise
        
        return wrapper
    return decorator

def trace_workflow_step(tracer: EnterpriseTracing, step_name: str):
    """工作流步骤追踪装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 从参数中提取trace_id（如果存在）
            trace_id = kwargs.get("trace_id") or tracer.create_trace(
                name=f"Workflow Step: {step_name}",
                tags=["workflow", step_name]
            )
            
            async with tracer.trace_run(
                trace_id=trace_id,
                name=step_name,
                run_type="workflow_step",
                inputs=kwargs
            ) as run_context:
                result = await func(*args, **kwargs)
                run_context.set_outputs({"result": result})
                return result
        
        return wrapper
    return decorator