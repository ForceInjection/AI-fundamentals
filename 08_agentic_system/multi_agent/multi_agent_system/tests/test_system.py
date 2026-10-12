#!/usr/bin/env python3
"""
企业级多智能体AI系统 - 系统测试

对齐 src/ 真实 API 的核心功能测试。2026-10 修订：初版测试是按一套
未落地的 API 设计稿写的（构造签名、容器语义、生命周期方法均与实现
不符），本版全部改为对齐 src/ 的真实接口。
"""

import asyncio
import pytest
import uuid
from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入测试目标
from src.agents.base_agent import (
    BaseAgent, Belief, Desire, Intention, AgentStatus, BeliefType, AgentResult,
)
from src.agents.research_agent import ResearchAgent, SearchResult
from src.agents.analysis_agent import AnalysisAgent
from src.communication.message_bus import MessageBus, Message, MessageType, MessagePriority
from src.workflows.langgraph_workflow import (
    EnhancedAgentState, StateValidator, EnterpriseWorkflowEngine,
)
from src.monitoring.langsmith_integration import EnterpriseTracing
from src.examples.customer_service_system import (
    CustomerServiceWorkflow, CustomerServiceAgent,
)
from main import MultiAgentSystem


class ConcreteTestAgent(BaseAgent):
    """测试用具体智能体——BaseAgent.execute 是抽象方法，必须落地才能实例化"""

    async def execute(self, input_data):
        return AgentResult(
            agent_id=self.agent_id,
            status=AgentStatus.COMPLETED,
            data={"echo": input_data},
            execution_time=0.0,
        )


def make_system_config(**overrides):
    """构造 MultiAgentSystem 需要的完整配置（main.py 按键直取，缺键会 KeyError）"""
    config = {
        "system": {"name": "Test System", "version": "1.0.0", "environment": "test"},
        "tracing": {"enabled": True, "sampling_rate": 1.0},
        "message_bus": {"max_queue_size": 1000, "worker_count": 2},
        "agents": {"max_concurrent_tasks": 5},
        "workflows": {},
        "customer_service": {"enabled": False},
    }
    config.update(overrides)
    return config


class TestBaseAgent:
    """基础智能体测试（BDI 架构：信念/愿望/意图为 key 索引的字典）"""

    @pytest.fixture
    def base_agent(self):
        return ConcreteTestAgent("test_agent_1", {"name": "Test Agent"})

    def test_agent_initialization(self, base_agent):
        assert base_agent.agent_id == "test_agent_1"
        assert base_agent.name == "Test Agent"
        assert base_agent.status == AgentStatus.IDLE
        assert base_agent.beliefs == {}
        assert base_agent.desires == {}
        assert base_agent.intentions == {}

    def test_belief_management(self, base_agent):
        base_agent.add_belief("test_key", {"test": "data"}, BeliefType.FACT, 0.8, "test")

        assert "test_key" in base_agent.beliefs
        assert base_agent.beliefs["test_key"].confidence == 0.8

        base_agent.update_belief("test_key", {"updated": True}, 0.9)
        assert base_agent.beliefs["test_key"].value == {"updated": True}
        assert base_agent.get_belief("missing") is None

    def test_desire_management(self, base_agent):
        base_agent.add_desire("test_goal", "完成测试", priority=5)

        assert "test_goal" in base_agent.desires
        assert base_agent.desires["test_goal"].description == "完成测试"
        assert base_agent.desires["test_goal"].is_achievable({}) is True

    def test_intention_management(self, base_agent):
        intention_id = base_agent.create_intention(
            "test_goal", [{"action": "step_one"}, {"action": "step_two"}]
        )

        intention = base_agent.intentions[intention_id]
        assert intention.goal_id == "test_goal"
        assert intention.get_current_step()["action"] == "step_one"
        intention.advance_step()
        assert intention.get_current_step()["action"] == "step_two"

    @pytest.mark.asyncio
    async def test_agent_run_completes_bdi_loop(self, base_agent):
        """run() 走完 感知→思考→规划→执行 整个 BDI 循环"""
        result = await base_agent.run({"environment": {"signal": "go"}})

        assert result.status == AgentStatus.COMPLETED
        assert result.agent_id == "test_agent_1"
        assert base_agent.status == AgentStatus.COMPLETED


class TestResearchAgent:
    """研究智能体测试"""

    @pytest.fixture
    def research_agent(self):
        return ResearchAgent(
            "research_agent_1",
            {"name": "Test Research Agent", "expertise_domains": ["web_search", "analysis"]},
        )

    def test_research_agent_initialization(self, research_agent):
        assert research_agent.agent_id == "research_agent_1"
        assert research_agent.expertise_domains == ["web_search", "analysis"]
        assert "web_search" in research_agent.capabilities
        for tool in ("web_search", "analyze_content", "validate_source", "extract_insights"):
            assert tool in research_agent.tools

    @pytest.mark.asyncio
    async def test_research_task_execution(self, research_agent, monkeypatch):
        """execute_research_task：mock 掉网络搜索，验证研究管道与返回结构"""
        async def fake_web_search(query, max_results=10):
            return [
                SearchResult(
                    title="AI trends",
                    content="AI is evolving fast",
                    url="https://example.org/ai-trends",
                    relevance_score=0.9,
                    source_type="web",
                    timestamp=datetime.now(),
                )
            ]

        monkeypatch.setattr(research_agent, "_web_search", fake_web_search)

        result = await research_agent.execute_research_task(
            {"query": "artificial intelligence trends", "depth": "basic", "sources": ["web"]}
        )

        assert "summary" in result
        assert "confidence" in result
        assert "sources" in result
        assert result["confidence"] > 0


class TestAnalysisAgent:
    """分析智能体测试"""

    @pytest.fixture
    def analysis_agent(self):
        return AnalysisAgent(
            "analysis_agent_1",
            {"name": "Test Analysis Agent", "analysis_types": ["statistical_analysis"]},
        )

    def test_analysis_agent_initialization(self, analysis_agent):
        assert analysis_agent.agent_id == "analysis_agent_1"
        assert "statistical_analysis" in analysis_agent.capabilities
        assert "descriptive_stats" in analysis_agent.tools

    @pytest.mark.asyncio
    async def test_data_analysis(self, analysis_agent):
        """execute 输入 data + analysis_type，产出自适应的洞察与统计"""
        data = {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        result = await analysis_agent.execute(
            {"data": data, "analysis_type": "descriptive"}
        )

        assert result.status == AgentStatus.COMPLETED
        assert result.data


class TestMessageBus:
    """消息总线测试

    真实语义：subscribe(agent_id, message_types, callback) 按消息类型订阅；
    publish(topic, content, sender_id) 对外广播（内部构造 Message）。
    """

    @pytest.fixture
    async def message_bus(self):
        bus = MessageBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest.mark.asyncio
    async def test_message_publishing(self, message_bus):
        await message_bus.publish(
            "test_topic", {"test": "data"}, sender_id="test_sender"
        )

        stats = message_bus.get_statistics()
        assert stats["message_stats"]["total_sent"] > 0

    @pytest.mark.asyncio
    async def test_message_subscription(self, message_bus):
        received_messages = []

        async def message_handler(message: Message):
            received_messages.append(message)

        # 订阅者必须先注册，路由才会为其建队列
        await message_bus.register_agent("test_subscriber")
        await message_bus.subscribe(
            "test_subscriber", [MessageType.REQUEST, MessageType.BROADCAST], message_handler
        )

        await message_bus.publish(
            "test_topic", {"test": "data"}, sender_id="test_sender"
        )

        await asyncio.sleep(0.1)

        assert len(received_messages) == 1
        assert received_messages[0].content == {"test": "data"}

    @pytest.mark.asyncio
    async def test_handler_can_call_bus_methods(self, message_bus):
        """回归测试：handler 内再调用总线方法不得自锁死
        （分发在 _subscription_lock 外触发回调——asyncio.Lock 不可重入）"""
        received = []
        forwarded = []

        async def forwarding_handler(message: Message):
            received.append(message)
            if message.content.get("forward"):
                await message_bus.publish(
                    "forwarded_topic", {"forwarded": True}, sender_id="handler"
                )
                forwarded.append(True)

        await message_bus.register_agent("forward_sub")
        await message_bus.subscribe(
            "forward_sub", [MessageType.BROADCAST], forwarding_handler
        )

        await message_bus.publish("origin", {"forward": True}, sender_id="origin")
        await asyncio.sleep(0.2)

        assert received  # handler 被调用且未死锁
        assert forwarded  # handler 内的 publish 也正常完成

    @pytest.mark.asyncio
    async def test_type_filtering(self, message_bus):
        """类型过滤：publish 的 BROADCAST 不命中 REQUEST 订阅；点对点直发进入队列由拉取消费"""
        received = []

        async def handler(message: Message):
            received.append(message)

        await message_bus.register_agent("filter_sub")
        await message_bus.subscribe("filter_sub", [MessageType.REQUEST], handler)

        # publish 的主题消息固定为 BROADCAST 类型——不匹配 REQUEST 订阅
        await message_bus.publish("t1", {"kind": "broadcast"}, sender_id="s")
        await asyncio.sleep(0.1)
        assert received == []

        # send_message 点对点直发：无显式 callback 时走拉取模式
        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id="s",
            receiver_id="filter_sub",
            message_type=MessageType.REQUEST,
            priority=MessagePriority.NORMAL,
            content={"kind": "request"},
            timestamp=datetime.now(),
        )
        await message_bus.send_message(message)

        pulled = await message_bus.receive_message("filter_sub", timeout=1.0)
        assert pulled is not None
        assert pulled.content == {"kind": "request"}
        assert received == []  # callback 未挂载时不推送，避免重复消费


class TestEnterpriseTracing:
    """企业追踪测试"""

    @pytest.fixture
    async def tracer(self):
        tracer = EnterpriseTracing()
        await tracer.start()
        yield tracer
        await tracer.stop()

    @pytest.mark.asyncio
    async def test_trace_creation(self, tracer):
        trace_id = tracer.create_trace(
            name="test_trace", tags=["test"], metadata={"test": "data"}
        )

        assert trace_id in tracer.active_traces
        assert tracer.active_traces[trace_id].tags == ["test"]

    @pytest.mark.asyncio
    async def test_trace_run(self, tracer):
        trace_id = tracer.create_trace("test_trace")

        async with tracer.trace_run(
            trace_id=trace_id, name="test_run", run_type="test",
            inputs={"test": "input"},
        ) as run_context:
            run_context.log("Test log message")
            run_context.set_outputs({"test": "output"})

        assert len(tracer.trace_events) > 0

    @pytest.mark.asyncio
    async def test_performance_metrics(self, tracer):
        metrics = tracer.get_performance_metrics()

        assert "total_traces" in metrics
        assert "active_traces_count" in metrics
        assert "error_rate" in metrics


class TestCustomerServiceSystem:
    """客服系统测试"""

    @pytest.fixture
    async def customer_service_system(self):
        tracer = EnterpriseTracing()
        await tracer.start()

        message_bus = MessageBus()
        await message_bus.start()

        system = CustomerServiceWorkflow(tracer, message_bus)

        agent = CustomerServiceAgent("test_cs_agent", "Test Customer Service Agent")
        system.add_agent(agent)

        yield system

        await message_bus.stop()
        await tracer.stop()

    @pytest.mark.asyncio
    async def test_customer_request_processing(self, customer_service_system):
        result = await customer_service_system.process_customer_request(
            "I need help with my account", "test_customer_1"
        )

        # ainvoke 返回最终状态 dict
        assert result["workflow_status"] in ["completed", "active"]
        assert len(result["messages"]) > 0

    def test_customer_service_metrics(self, customer_service_system):
        metrics = customer_service_system.get_system_metrics()

        assert "total_agents" in metrics
        assert "total_tickets_handled" in metrics
        assert "average_customer_satisfaction" in metrics


class TestMultiAgentSystem:
    """多智能体系统集成测试"""

    @pytest.fixture
    async def system(self):
        system = MultiAgentSystem(make_system_config())
        await system.start()
        yield system
        await system.shutdown()

    @pytest.mark.asyncio
    async def test_system_initialization(self, system):
        assert system.running is True
        assert system.tracer is not None
        assert system.message_bus is not None
        assert system.workflow_engine is not None
        assert len(system.agents) > 0

    @pytest.mark.asyncio
    async def test_agent_registration(self, system):
        initial_count = len(system.agents)

        new_agent = ConcreteTestAgent("test_new_agent", {"name": "New Test Agent"})
        await system.register_agent(new_agent)

        assert len(system.agents) == initial_count + 1
        assert "test_new_agent" in system.agents

        await system.unregister_agent("test_new_agent")
        assert len(system.agents) == initial_count
        assert "test_new_agent" not in system.agents

    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, system):
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
        tracer = EnterpriseTracing()
        await tracer.start()

        message_bus = MessageBus()
        await message_bus.start()

        engine = EnterpriseWorkflowEngine()
        if hasattr(engine, 'set_dependencies'):
            engine.set_dependencies(tracer, message_bus)

        yield engine

        await message_bus.stop()
        await tracer.stop()

    def test_state_validation(self, workflow_engine):
        """validate_state 返回错误列表：为空即合法"""
        validator = StateValidator()

        state = EnhancedAgentState(
            messages=[{"role": "user", "content": "test"}],
            current_agent="test_agent",
            workflow_status="active",
        )

        errors = validator.validate_state(state)
        assert errors == []

    def test_workflow_template_registered(self, workflow_engine):
        """引擎初始化时注册了内置工作流模板"""
        assert "research_analysis" in workflow_engine.workflow_templates
        assert "data_processing" in workflow_engine.workflow_templates
        assert "content_generation" in workflow_engine.workflow_templates


class TestPerformance:
    """性能测试"""

    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self):
        message_bus = MessageBus()
        await message_bus.start()

        received_count = 0

        async def message_handler(message: Message):
            nonlocal received_count
            received_count += 1
            await asyncio.sleep(0.01)

        await message_bus.subscribe("perf_sub", [MessageType.BROADCAST], message_handler)

        tasks = []
        for i in range(100):
            tasks.append(
                message_bus.publish("perf_test", {"index": i}, sender_id="perf_test")
            )

        await asyncio.gather(*tasks)
        await asyncio.sleep(2)

        await message_bus.stop()

        assert received_count == 100

    @pytest.mark.asyncio
    async def test_system_startup_time(self):
        start_time = datetime.now()

        config = make_system_config(
            system={"name": "Perf Test", "version": "1.0.0", "environment": "test"},
            tracing={"enabled": False},
            customer_service={"enabled": False},
        )
        config["tracing"] = {"enabled": False}

        system = MultiAgentSystem(config)
        await system.start()

        startup_time = (datetime.now() - start_time).total_seconds()

        await system.shutdown()

        assert startup_time < 10.0


def run_tests():
    """运行所有测试"""
    import subprocess
    import sys

    result = subprocess.run([
        sys.executable, "-m", "pytest",
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])

    return result.returncode


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
