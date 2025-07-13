from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import uuid
from datetime import datetime
import json

class AgentStatus(Enum):
    """智能体状态枚举"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    PAUSED = "paused"

class BeliefType(Enum):
    """信念类型"""
    FACT = "fact"
    ASSUMPTION = "assumption"
    PREDICTION = "prediction"
    OBSERVATION = "observation"

@dataclass
class Belief:
    """信念数据结构"""
    key: str
    value: Any
    belief_type: BeliefType
    confidence: float  # 0.0 - 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    
    def is_valid(self) -> bool:
        """检查信念是否仍然有效"""
        # 可以根据时间戳和置信度判断
        return self.confidence > 0.1

@dataclass
class Desire:
    """愿望/目标数据结构"""
    goal_id: str
    description: str
    priority: int  # 1-10, 10为最高优先级
    deadline: Optional[datetime] = None
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_achievable(self, current_beliefs: Dict[str, Belief]) -> bool:
        """判断目标是否可达成"""
        for prereq in self.prerequisites:
            if prereq not in current_beliefs or not current_beliefs[prereq].is_valid():
                return False
        return True

@dataclass
class Intention:
    """意图/计划数据结构"""
    intention_id: str
    goal_id: str
    plan_steps: List[Dict[str, Any]]
    current_step: int = 0
    status: str = "pending"
    estimated_duration: Optional[float] = None
    actual_start_time: Optional[datetime] = None
    
    def get_current_step(self) -> Optional[Dict[str, Any]]:
        """获取当前执行步骤"""
        if self.current_step < len(self.plan_steps):
            return self.plan_steps[self.current_step]
        return None
    
    def advance_step(self):
        """推进到下一步"""
        self.current_step += 1

@dataclass
class AgentResult:
    """智能体执行结果"""
    agent_id: str
    status: AgentStatus
    data: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    
class BaseAgent(ABC):
    """基础智能体类 - 实现BDI架构"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.name = config.get("name", agent_id)  # 从配置中获取名称，默认使用agent_id
        self.status = AgentStatus.IDLE
        self.logger = logging.getLogger(f"agent.{agent_id}")
        
        # BDI架构核心组件
        self.beliefs: Dict[str, Belief] = {}  # 信念库
        self.desires: Dict[str, Desire] = {}  # 愿望库
        self.intentions: Dict[str, Intention] = {}  # 意图库
        
        # 执行历史和性能指标
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {
            "total_executions": 0,
            "successful_executions": 0,
            "average_execution_time": 0.0,
            "success_rate": 0.0
        }
        
        # 能力和工具
        self.capabilities: List[str] = []
        self.tools: Dict[str, Callable] = {}
        
        # 初始化智能体
        self._initialize_agent()
    
    def _initialize_agent(self):
        """初始化智能体"""
        # 加载初始信念
        self._load_initial_beliefs()
        
        # 设置基本能力（扩展而不是覆盖）
        config_capabilities = self.config.get("capabilities", [])
        self.capabilities.extend(config_capabilities)
        
        # 加载工具
        self._load_tools()
        
        self.logger.info(f"Agent {self.agent_id} initialized with {len(self.capabilities)} capabilities")
    
    def _load_initial_beliefs(self):
        """加载初始信念"""
        initial_beliefs = self.config.get("initial_beliefs", {})
        for key, value in initial_beliefs.items():
            self.add_belief(key, value, BeliefType.FACT, 1.0, "initialization")
    
    def _load_tools(self):
        """加载工具"""
        # 子类可以重写此方法来加载特定工具
        pass
    
    # BDI架构核心方法
    def add_belief(self, key: str, value: Any, belief_type: BeliefType, 
                   confidence: float, source: str = "unknown"):
        """添加信念"""
        belief = Belief(
            key=key,
            value=value,
            belief_type=belief_type,
            confidence=confidence,
            source=source
        )
        self.beliefs[key] = belief
        self.logger.debug(f"Added belief: {key} = {value} (confidence: {confidence})")
    
    def update_belief(self, key: str, value: Any, confidence: float):
        """更新信念"""
        if key in self.beliefs:
            self.beliefs[key].value = value
            self.beliefs[key].confidence = confidence
            self.beliefs[key].timestamp = datetime.now()
            self.logger.debug(f"Updated belief: {key} = {value}")
    
    def get_belief(self, key: str) -> Optional[Belief]:
        """获取信念"""
        return self.beliefs.get(key)
    
    def add_desire(self, goal_id: str, description: str, priority: int, 
                   deadline: Optional[datetime] = None, 
                   prerequisites: List[str] = None,
                   success_criteria: Dict[str, Any] = None):
        """添加愿望/目标"""
        desire = Desire(
            goal_id=goal_id,
            description=description,
            priority=priority,
            deadline=deadline,
            prerequisites=prerequisites or [],
            success_criteria=success_criteria or {}
        )
        self.desires[goal_id] = desire
        self.logger.debug(f"Added desire: {goal_id} - {description}")
    
    def create_intention(self, goal_id: str, plan_steps: List[Dict[str, Any]]) -> str:
        """创建意图/计划"""
        intention_id = str(uuid.uuid4())
        intention = Intention(
            intention_id=intention_id,
            goal_id=goal_id,
            plan_steps=plan_steps
        )
        self.intentions[intention_id] = intention
        self.logger.debug(f"Created intention {intention_id} for goal {goal_id}")
        return intention_id
    
    def perceive(self, environment: Dict[str, Any]):
        """感知环境并更新信念"""
        self.logger.debug("Perceiving environment...")
        
        # 从环境中提取信息并更新信念
        for key, value in environment.items():
            if self._should_believe(key, value):
                self.add_belief(
                    key, value, BeliefType.OBSERVATION, 
                    self._calculate_belief_confidence(key, value),
                    "environment_perception"
                )
    
    def deliberate(self):
        """基于信念生成或更新愿望"""
        self.logger.debug("Deliberating on goals...")
        
        # 分析当前信念，生成新的目标或调整现有目标
        current_context = self._analyze_current_context()
        new_goals = self._generate_goals(current_context)
        
        for goal in new_goals:
            if goal["goal_id"] not in self.desires:
                self.add_desire(**goal)
    
    def plan(self):
        """将愿望转化为具体的执行意图"""
        self.logger.debug("Planning actions...")
        
        # 选择最高优先级且可达成的目标
        achievable_goals = [
            desire for desire in self.desires.values()
            if desire.is_achievable(self.beliefs)
        ]
        
        if not achievable_goals:
            return
        
        # 按优先级排序
        achievable_goals.sort(key=lambda x: x.priority, reverse=True)
        
        # 为最高优先级目标创建计划
        top_goal = achievable_goals[0]
        plan_steps = self._create_plan(top_goal)
        
        if plan_steps:
            self.create_intention(top_goal.goal_id, plan_steps)
    
    async def execute_intentions(self) -> List[AgentResult]:
        """执行当前意图"""
        results = []
        
        for intention in list(self.intentions.values()):
            if intention.status == "pending":
                result = await self._execute_intention(intention)
                results.append(result)
        
        return results
    
    async def _execute_intention(self, intention: Intention) -> AgentResult:
        """执行单个意图"""
        start_time = datetime.now()
        intention.actual_start_time = start_time
        intention.status = "executing"
        
        try:
            while intention.current_step < len(intention.plan_steps):
                current_step = intention.get_current_step()
                if current_step:
                    await self._execute_step(current_step)
                    intention.advance_step()
            
            intention.status = "completed"
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data={"intention_id": intention.intention_id, "goal_id": intention.goal_id},
                execution_time=execution_time
            )
            
            self._update_performance_metrics(result)
            return result
            
        except Exception as e:
            intention.status = "failed"
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.ERROR,
                data={},
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self.logger.error(f"Intention execution failed: {str(e)}")
            return result
    
    async def _execute_step(self, step: Dict[str, Any]):
        """执行计划步骤"""
        action = step.get("action")
        parameters = step.get("parameters", {})
        
        if action in self.tools:
            await self.tools[action](**parameters)
        else:
            self.logger.warning(f"Unknown action: {action}")
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """执行智能体主要任务 - 子类必须实现"""
        pass
    
    # 辅助方法
    def _should_believe(self, key: str, value: Any) -> bool:
        """判断是否应该相信某个信息"""
        # 简单实现，子类可以重写
        return True
    
    def _calculate_belief_confidence(self, key: str, value: Any) -> float:
        """计算信念的置信度"""
        # 简单实现，子类可以重写
        return 0.8
    
    def _analyze_current_context(self) -> Dict[str, Any]:
        """分析当前上下文"""
        return {
            "beliefs_count": len(self.beliefs),
            "active_desires": len(self.desires),
            "pending_intentions": len([i for i in self.intentions.values() if i.status == "pending"])
        }
    
    def _generate_goals(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于上下文生成目标"""
        # 子类应该重写此方法
        return []
    
    def _create_plan(self, goal: Desire) -> List[Dict[str, Any]]:
        """为目标创建执行计划"""
        # 子类应该重写此方法
        return []
    
    def _update_performance_metrics(self, result: AgentResult):
        """更新性能指标"""
        self.performance_metrics["total_executions"] += 1
        
        if result.status == AgentStatus.COMPLETED:
            self.performance_metrics["successful_executions"] += 1
        
        # 更新平均执行时间
        total_time = (self.performance_metrics["average_execution_time"] * 
                     (self.performance_metrics["total_executions"] - 1) + 
                     result.execution_time)
        self.performance_metrics["average_execution_time"] = total_time / self.performance_metrics["total_executions"]
        
        # 更新成功率
        self.performance_metrics["success_rate"] = (
            self.performance_metrics["successful_executions"] / 
            self.performance_metrics["total_executions"]
        )
    
    async def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """运行智能体的完整BDI循环"""
        start_time = datetime.now()
        self.status = AgentStatus.RUNNING
        
        try:
            # 1. 感知环境
            environment = input_data.get("environment", {})
            self.perceive(environment)
            
            # 2. 思考和规划
            self.deliberate()
            self.plan()
            
            # 3. 执行主要任务
            result = await self.execute(input_data)
            
            # 4. 执行意图
            intention_results = await self.execute_intentions()
            
            # 5. 更新状态
            self.status = result.status
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # 6. 记录执行历史
            self.execution_history.append({
                "timestamp": start_time,
                "input_data": input_data,
                "result": result,
                "intention_results": intention_results
            })
            
            self._update_performance_metrics(result)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_result = AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.ERROR,
                data={},
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self.status = AgentStatus.ERROR
            self.logger.error(f"Agent execution failed: {str(e)}")
            
            return error_result
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "beliefs_count": len(self.beliefs),
            "desires_count": len(self.desires),
            "intentions_count": len(self.intentions),
            "capabilities": self.capabilities,
            "performance_metrics": self.performance_metrics
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return dict(self.performance_metrics)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取详细的性能指标"""
        return {
            **self.performance_metrics,
            "current_status": self.status.value,
            "beliefs_summary": {
                "total": len(self.beliefs),
                "by_type": {
                    belief_type.value: len([b for b in self.beliefs.values() if b.belief_type == belief_type])
                    for belief_type in BeliefType
                }
            },
            "goals_summary": {
                "total": len(self.desires),
                "by_priority": {
                    f"priority_{i}": len([d for d in self.desires.values() if d.priority == i])
                    for i in range(1, 11)
                }
            }
        }
    
    async def handle_message(self, message: Dict[str, Any]):
        """处理接收到的消息"""
        try:
            message_type = message.get("type", "unknown")
            sender = message.get("sender", "unknown")
            content = message.get("content", {})
            
            self.logger.info(f"Received message of type '{message_type}' from {sender}")
            
            # 根据消息类型处理
            if message_type == "task_request":
                # 处理任务请求
                result = await self.execute(content)
                return result
            elif message_type == "belief_update":
                # 更新信念
                belief_key = content.get("key")
                belief_value = content.get("value")
                if belief_key and belief_value is not None:
                    self.add_belief(belief_key, belief_value, BeliefType.OBSERVATION, 0.8, sender)
            elif message_type == "status_request":
                # 返回状态信息
                return self.get_status()
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            return {"error": str(e)}