#!/usr/bin/env python3
"""
企业级多智能体AI系统 - 系统测试

这个测试文件包含了系统的核心功能测试，确保各个组件能够正常工作。
"""

import asyncio
import pytest
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入测试目标
from src.agents.base_agent import BaseAgent, Belief, Desire, Intention
from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.communication.message_bus import MessageBus, Message, MessageType, MessagePriority
from src.workflows.langgraph_workflow import (
    EnhancedAgentState, StateValidator, EnterpriseWorkflowEngine
)
from src.monitoring.langsmith_integration import (
    EnterpriseTracing, TraceLevel, TraceContext
)
from src.examples.customer_service_system import (
    CustomerServiceWorkflow, CustomerServiceAgent, CustomerProfile
)
from main import MultiAgentSystem

class TestBaseAgent:
    """基础智能体测试"""
    
    @pytest.fixture
    def base_agent(self):
        """创建基础智能体实例"""
        return BaseAgent("test_agent_1", "Test Agent")
    
    def test_agent_initialization(self, base_agent):
        """测试智能体初始化"""
        assert base_agent.agent_id == "test_agent_1"
        assert base_agent.name == "Test Agent"
        assert base_agent.status == "idle"
        assert len(base_agent.beliefs) == 0
        assert len(base_agent.desires) == 0
        assert len(base_agent.intentions) == 0
    
    def test_belief_management(self, base_agent):
        """测试信念管理"""
        # 添加信念
        belief = Belief(
            content={"test": "data"},
            confidence=0.8,
            source="test",
            timestamp=datetime.now()
        )
        base_agent.beliefs.append(belief)
        
        assert len(base_agent.beliefs) == 1
        assert base_agent.beliefs[0].confidence == 0.8
    
    def test_desire_management(self, base_agent):
        """测试愿望管理"""
        # 添加愿望
        desire = Desire(
            goal="test_goal",
            priority=0.9,
            context={"test": "context"},
            timestamp=datetime.now()
        )
        base_agent.desires.append(desire)
        
        assert len(base_agent.desires) == 1
        assert base_agent.desires[0].goal == "test_goal"
    
    def test_intention_management(self, base_agent):
        """测试意图管理"""
        # 添加意图
        intention = Intention(
            action="test_action",
            parameters={"param": "value"},
            priority=0.7,
            timestamp=datetime.now()
        )
        base_agent.intentions.append(intention)
        
        assert len(base_agent.intentions) == 1
        assert base_agent.intentions[0].action == "test_action"
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, base_agent):
        """测试智能体生命周期"""
        # 启动智能体
        await base_agent.start()
        assert base_agent.status == "active"
        
        # 停止智能体
        await base_agent.stop()
        assert base_agent.status == "stopped"

class TestResearchAgent:
    """研究智能体测试"""
    
    @pytest.fixture
    def research_agent(self):
        """创建研究智能体实例"""
        return ResearchAgent(
            "research_agent_1", "Test Research Agent",
            specializations=["web_search", "analysis"]
        )
    
    def test_research_agent_initialization(self, research_agent):
        """测试研究智能体初始化"""
        assert research_agent.agent_id == "research_agent_1"
        assert "web_search" in research_agent.specializations
        assert "search_web" in research_agent.tools
        assert "analyze_results" in research_agent.tools
    
    @pytest.mark.asyncio
    async def test_research_task_execution(self, research_agent):
        """测试研究任务执行"""
        task = {
            "query": "artificial intelligence trends",
            "depth": "basic",
            "sources": ["web"]
        }
        
        result = await research_agent.execute_research_task(task)
        
        assert "summary" in result
        assert "confidence" in result
        assert "sources" in result
        assert result["confidence"] > 0

class TestAnalysisAgent:
    """分析智能体测试"""
    
    @pytest.fixture
    def analysis_agent(self):
        """创建分析智能体实例"""
        return AnalysisAgent(
            "analysis_agent_1", "Test Analysis Agent",
            specializations=["statistical_analysis"]
        )
    
    def test_analysis_agent_initialization(self, analysis_agent):
        """测试分析智能体初始化"""
        assert analysis_agent.agent_id == "analysis_agent_1"
        assert "statistical_analysis" in analysis_agent.specializations
        assert "analyze_data" in analysis_agent.tools
    
    @pytest.mark.asyncio
    async def test_data_analysis(self, analysis_agent):
        """测试数据分析"""
        # 模拟数据
        data = {
            "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "labels": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        }
        
        result = await analysis_agent.analyze_data(data)
        
        assert "insights" in result
        assert "statistics" in result
        assert "confidence" in result
        assert result["confidence"] > 0

class TestMessageBus:
    """消息总线测试"""
    
    @pytest.fixture
    async def message_bus(self):
        """创建消息总线实例"""
        bus = MessageBus()
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_message_publishing(self, message_bus):
        """测试消息发布"""
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.REQUEST,
            sender="test_sender",
            recipient="test_recipient",
            content={"test": "data"},
            priority=MessagePriority.NORMAL
        )
        
        await message_bus.publish("test_topic", message)
        
        # 验证消息统计
        stats = message_bus.get_statistics()
        assert stats["messages_published"] > 0
    
    @pytest.mark.asyncio
    async def test_message_subscription(self, message_bus):
        """测试消息订阅"""
        received_messages = []
        
        async def message_handler(message: Message):
            received_messages.append(message)
        
        # 订阅主题
        await message_bus.subscribe("test_topic", message_handler)
        
        # 发布消息
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.REQUEST,
            sender="test_sender",
            recipient="test_recipient",
            content={"test": "data"},
            priority=MessagePriority.NORMAL
        )
        
        await message_bus.publish("test_topic", message)
        
        # 等待消息处理
        await asyncio.sleep(0.1)
        
        assert len(received_messages) == 1
        assert received_messages[0].content["test"] == "data"

class TestEnterpriseTracing:
    """企业追踪测试"""
    
    @pytest.fixture
    async def tracer(self):
        """创建追踪器实例"""
        tracer = EnterpriseTracing()
        await tracer.start()
        yield tracer
        await tracer.stop()
    
    def test_trace_creation(self, tracer):
        """测试追踪创建"""
        trace_id = tracer.create_trace(
            name="test_trace",
            tags=["test"],
            metadata={"test": "data"}
        )
        
        assert trace_id in tracer.active_traces
        assert tracer.active_traces[trace_id].tags == ["test"]
    
    @pytest.mark.asyncio
    async def test_trace_run(self, tracer):
        """测试追踪运行"""
        trace_id = tracer.create_trace("test_trace")
        
        async with tracer.trace_run(
            trace_id=trace_id,
            name="test_run",
            run_type="test",
            inputs={"test": "input"}
        ) as run_context:
            run_context.log("Test log message")
            run_context.set_outputs({"test": "output"})
        
        # 验证追踪事件
        assert len(tracer.trace_events) > 0
    
    def test_performance_metrics(self, tracer):
        """测试性能指标"""
        metrics = tracer.get_performance_metrics()
        
        assert "total_traces" in metrics
        assert "active_traces_count" in metrics
        assert "error_rate" in metrics

class TestCustomerServiceSystem:
    """客服系统测试"""
    
    @pytest.fixture
    async def customer_service_system(self):
        """创建客服系统实例"""
        tracer = EnterpriseTracing()
        await tracer.start()
        
        message_bus = MessageBus()
        await message_bus.start()
        
        system = CustomerServiceWorkflow(tracer, message_bus)
        
        # 添加测试智能体
        agent = CustomerServiceAgent(
            "test_cs_agent", "Test Customer Service Agent"
        )
        system.add_agent(agent)
        
        yield system
        
        await message_bus.stop()
        await tracer.stop()
    
    @pytest.mark.asyncio
    async def test_customer_request_processing(self, customer_service_system):
        """测试客户请求处理"""
        customer_message = "I need help with my account"
        customer_id = "test_customer_1"
        
        result = await customer_service_system.process_customer_request(
            customer_message, customer_id
        )
        
        assert result.workflow_status in ["completed", "active"]
        assert len(result.messages) > 0
        assert "context" in result.__dict__
    
    def test_customer_service_metrics(self, customer_service_system):
        """测试客服系统指标"""
        metrics = customer_service_system.get_system_metrics()
        
        assert "total_agents" in metrics
        assert "total_tickets_handled" in metrics
        assert "agent_utilization" in metrics

class TestMultiAgentSystem:
    """多智能体系统集成测试"""
    
    @pytest.fixture
    async def system(self):
        """创建系统实例"""
        config = {
            "system": {"name": "Test System", "version": "1.0.0", "environment": "test"},
            "tracing": {"enabled": True, "sampling_rate": 1.0},
            "message_bus": {"max_queue_size": 1000, "worker_count": 2},
            "agents": {"max_concurrent_tasks": 5},
            "workflows": {"max_parallel_executions": 3},
            "customer_service": {"enabled": True, "max_agents": 5}
        }
        
        system = MultiAgentSystem(config)
        await system.start()
        yield system
        await system.shutdown()
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, system):
        """测试系统初始化"""
        assert system.running is True
        assert system.tracer is not None
        assert system.message_bus is not None
        assert system.workflow_engine is not None
        assert len(system.agents) > 0
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, system):
        """测试智能体注册"""
        initial_count = len(system.agents)
        
        # 创建新智能体
        new_agent = BaseAgent("test_new_agent", "New Test Agent")
        await system.register_agent(new_agent)
        
        assert len(system.agents) == initial_count + 1
        assert "test_new_agent" in system.agents
        
        # 注销智能体
        await system.unregister_agent("test_new_agent")
        assert len(system.agents) == initial_count
        assert "test_new_agent" not in system.agents
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, system):
        """测试系统指标收集"""
        metrics = await system._collect_system_metrics()
        
        assert "timestamp" in metrics
        assert "uptime_seconds" in metrics
        assert "agents" in metrics
        assert "message_bus" in metrics
        assert "tracing" in metrics

class TestWorkflowEngine:
    """工作流引擎测试"""
    
    @pytest.fixture
    async def workflow_engine(self):
        """创建工作流引擎实例"""
        tracer = EnterpriseTracing()
        await tracer.start()
        
        message_bus = MessageBus()
        await message_bus.start()
        
        engine = EnterpriseWorkflowEngine()
        # 设置工作流引擎的依赖
        if hasattr(engine, 'set_dependencies'):
            engine.set_dependencies(tracer, message_bus)
        
        yield engine
        
        # EnterpriseWorkflowEngine没有stop方法
        await message_bus.stop()
        await tracer.stop()
    
    def test_state_validation(self, workflow_engine):
        """测试状态验证"""
        validator = StateValidator()
        
        # 创建测试状态
        state = EnhancedAgentState(
            messages=[{"role": "user", "content": "test"}],
            current_agent="test_agent",
            workflow_status="active"
        )
        
        # 验证状态
        is_valid, errors = validator.validate_state(state)
        assert is_valid is True
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, workflow_engine):
        """测试工作流执行"""
        # 创建简单的研究分析工作流
        workflow = workflow_engine.create_research_analysis_workflow()
        
        # 初始状态
        initial_state = EnhancedAgentState(
            messages=[{"role": "user", "content": "Analyze AI trends"}],
            current_agent="research_agent",
            workflow_status="active",
            context={"research_query": "AI trends"}
        )
        
        # 执行工作流（模拟）
        result = await workflow.ainvoke(initial_state)
        
        assert result is not None
        assert hasattr(result, 'workflow_status')

# 性能测试
class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self):
        """测试并发消息处理"""
        message_bus = MessageBus()
        await message_bus.start()
        
        received_count = 0
        
        async def message_handler(message: Message):
            nonlocal received_count
            received_count += 1
            await asyncio.sleep(0.01)  # 模拟处理时间
        
        await message_bus.subscribe("perf_test", message_handler)
        
        # 发送多个消息
        tasks = []
        for i in range(100):
            message = Message(
                id=str(uuid.uuid4()),
                type=MessageType.REQUEST,
                sender="perf_test",
                recipient="handler",
                content={"index": i},
                priority=MessagePriority.NORMAL
            )
            task = message_bus.publish("perf_test", message)
            tasks.append(task)
        
        # 等待所有消息发送完成
        await asyncio.gather(*tasks)
        
        # 等待消息处理完成
        await asyncio.sleep(2)
        
        await message_bus.stop()
        
        # 验证所有消息都被处理
        assert received_count == 100
    
    @pytest.mark.asyncio
    async def test_system_startup_time(self):
        """测试系统启动时间"""
        start_time = datetime.now()
        
        config = {
            "system": {"name": "Perf Test", "version": "1.0.0", "environment": "test"},
            "tracing": {"enabled": False},  # 禁用追踪以提高启动速度
            "message_bus": {"worker_count": 1},
            "customer_service": {"enabled": False}
        }
        
        system = MultiAgentSystem(config)
        await system.start()
        
        startup_time = (datetime.now() - start_time).total_seconds()
        
        await system.shutdown()
        
        # 启动时间应该在合理范围内（小于10秒）
        assert startup_time < 10.0

# 运行测试的主函数
def run_tests():
    """运行所有测试"""
    import subprocess
    import sys
    
    # 运行pytest
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v",  # 详细输出
        "--tb=short",  # 简短的错误回溯
        "--asyncio-mode=auto"  # 自动异步模式
    ])
    
    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)