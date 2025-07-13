#!/usr/bin/env python3
"""
企业级多智能体AI系统 - 主应用程序

这是系统的主入口点，整合了所有核心组件：
- 基础智能体架构（BDI模式）
- 专业化智能体（研究、分析等）
- 通信中间件（消息总线）
- 工作流引擎（LangGraph集成）
- 监控系统（LangSmith集成）
- 示例应用（智能客服系统）
"""

import asyncio
import logging
import signal
import sys
import json
from typing import Dict, Any, Optional
from datetime import datetime
import argparse
from pathlib import Path

# 导入核心组件
from src.agents.base_agent import BaseAgent
from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.communication.message_bus import MessageBus, Message, MessageType, MessagePriority
from src.workflows.langgraph_workflow import (
    EnhancedAgentState, EnterpriseWorkflowEngine
)
from src.monitoring.langsmith_integration import (
    EnterpriseTracing, TraceLevel, PerformanceMonitor
)
from src.examples.customer_service_system import (
    CustomerServiceWorkflow, CustomerServiceAgent
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('multi_agent_system.log')
    ]
)

logger = logging.getLogger(__name__)

class MultiAgentSystem:
    """多智能体系统主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 核心组件
        self.tracer: Optional[EnterpriseTracing] = None
        self.message_bus: Optional[MessageBus] = None
        self.workflow_engine: Optional[EnterpriseWorkflowEngine] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # 智能体注册表
        self.agents: Dict[str, BaseAgent] = {}
        
        # 示例应用
        self.customer_service: Optional[CustomerServiceWorkflow] = None
        
        # 系统状态
        self.running = False
        self.startup_time: Optional[datetime] = None
        
        # 信号处理
        self._setup_signal_handlers()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            "system": {
                "name": "Enterprise Multi-Agent AI System",
                "version": "1.0.0",
                "environment": "development"
            },
            "tracing": {
                "enabled": True,
                "sampling_rate": 1.0,
                "batch_size": 100,
                "batch_timeout": 5.0
            },
            "message_bus": {
                "max_queue_size": 10000,
                "worker_count": 4,
                "retry_attempts": 3
            },
            "agents": {
                "max_concurrent_tasks": 10,
                "default_timeout": 300,
                "performance_tracking": True
            },
            "workflows": {
                "max_parallel_executions": 5,
                "state_persistence": True,
                "checkpoint_interval": 60
            },
            "customer_service": {
                "enabled": True,
                "max_agents": 10,
                "auto_scaling": True
            }
        }
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self):
        """初始化系统"""
        self.logger.info("Initializing Multi-Agent System...")
        
        try:
            # 初始化追踪系统
            if self.config["tracing"]["enabled"]:
                self.tracer = EnterpriseTracing(self.config["tracing"])
                await self.tracer.start()
                self.logger.info("Enterprise tracing system initialized")
            
            # 初始化消息总线
            self.message_bus = MessageBus(self.config["message_bus"])
            await self.message_bus.start()
            self.logger.info("Message bus initialized")
            
            # 初始化工作流引擎
            self.workflow_engine = EnterpriseWorkflowEngine(
                self.config["workflows"]
            )
            # 设置工作流引擎的依赖
            if hasattr(self.workflow_engine, 'set_dependencies'):
                self.workflow_engine.set_dependencies(self.tracer, self.message_bus)
            self.logger.info("Workflow engine initialized")
            
            # 初始化性能监控
            if self.tracer:
                self.performance_monitor = PerformanceMonitor(self.tracer)
                self.logger.info("Performance monitor initialized")
            
            # 初始化核心智能体
            await self._initialize_core_agents()
            
            # 初始化示例应用
            if self.config["customer_service"]["enabled"]:
                await self._initialize_customer_service()
            
            self.startup_time = datetime.now()
            self.logger.info("Multi-Agent System initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {str(e)}")
            await self.shutdown()
            raise
    
    async def _initialize_core_agents(self):
        """初始化核心智能体"""
        self.logger.info("Initializing core agents...")
        
        try:
            # 创建研究智能体
            research_agent = ResearchAgent(
                agent_id="research_agent_1",
                config={
                    "name": "Primary Research Agent",
                    "specializations": ["web_search", "data_analysis", "trend_analysis"]
                }
            )
            
            # 创建分析智能体
            analysis_agent = AnalysisAgent(
                agent_id="analysis_agent_1",
                config={
                    "name": "Primary Analysis Agent",
                    "specializations": ["statistical_analysis", "data_visualization", "insights"]
                }
            )
            
            # 注册智能体
            await self.register_agent(research_agent)
            await self.register_agent(analysis_agent)
            
            self.logger.info(f"Initialized {len(self.agents)} core agents")
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error in _initialize_core_agents: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def _initialize_customer_service(self):
        """初始化客服系统"""
        self.logger.info("Initializing customer service system...")
        
        self.customer_service = CustomerServiceWorkflow(
            self.tracer, self.message_bus
        )
        
        # 创建客服智能体
        tech_agent = CustomerServiceAgent(
            "tech_support_1", "Technical Support Agent",
            specializations=["Technical", "Product"]
        )
        
        billing_agent = CustomerServiceAgent(
            "billing_support_1", "Billing Support Agent",
            specializations=["Billing", "Account"]
        )
        
        general_agent = CustomerServiceAgent(
            "general_support_1", "General Support Agent",
            specializations=["General", "Account"]
        )
        
        # 添加到客服系统
        self.customer_service.add_agent(tech_agent)
        self.customer_service.add_agent(billing_agent)
        self.customer_service.add_agent(general_agent)
        
        self.logger.info("Customer service system initialized")
    
    async def register_agent(self, agent: BaseAgent):
        """注册智能体"""
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")
        
        self.agents[agent.agent_id] = agent
        
        # 订阅消息总线
        if self.message_bus:
            from src.communication.message_bus import MessageType
            await self.message_bus.subscribe(
                agent.agent_id,
                [MessageType.REQUEST, MessageType.NOTIFICATION, MessageType.STATUS_UPDATE],
                agent.handle_message
            )
        
        self.logger.info(f"Registered agent: {agent.agent_id} ({agent.name})")
    
    async def unregister_agent(self, agent_id: str):
        """注销智能体"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        
        # 取消订阅
        if self.message_bus:
            await self.message_bus.unsubscribe(
                f"agent.{agent_id}",
                agent.handle_message
            )
        
        del self.agents[agent_id]
        self.logger.info(f"Unregistered agent: {agent_id}")
    
    async def start(self):
        """启动系统"""
        if self.running:
            self.logger.warning("System is already running")
            return
        
        await self.initialize()
        self.running = True
        
        self.logger.info("Multi-Agent System started successfully")
        
        # 启动系统监控任务
        asyncio.create_task(self._system_monitor())
        
        # 显示系统信息
        await self._display_system_info()
    
    async def shutdown(self):
        """关闭系统"""
        if not self.running:
            return
        
        self.logger.info("Shutting down Multi-Agent System...")
        self.running = False
        
        try:
            # 停止工作流引擎
            if self.workflow_engine:
                # EnterpriseWorkflowEngine没有stop方法
                self.logger.info("Workflow engine stopped")
            
            # 停止消息总线
            if self.message_bus:
                await self.message_bus.stop()
                self.logger.info("Message bus stopped")
            
            # 停止追踪系统
            if self.tracer:
                await self.tracer.stop()
                self.logger.info("Tracing system stopped")
            
            self.logger.info("Multi-Agent System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    async def _system_monitor(self):
        """系统监控任务"""
        while self.running:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次
                
                # 收集系统指标
                metrics = await self._collect_system_metrics()
                
                # 检查性能阈值
                if self.performance_monitor:
                    self.performance_monitor.check_performance_thresholds(metrics)
                
                # 记录系统状态
                self.logger.debug(f"System metrics: {json.dumps(metrics, indent=2)}")
                
            except Exception as e:
                self.logger.error(f"System monitor error: {str(e)}")
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,
            "agents": {
                "total_count": len(self.agents),
                "active_count": len([a for a in self.agents.values() if a.status == "active"]),
                "performance": {
                    agent_id: agent.get_performance_metrics()
                    for agent_id, agent in self.agents.items()
                }
            }
        }
        
        # 消息总线指标
        if self.message_bus:
            metrics["message_bus"] = self.message_bus.get_statistics()
        
        # 追踪系统指标
        if self.tracer:
            metrics["tracing"] = self.tracer.get_performance_metrics()
        
        # 工作流引擎指标
        if self.workflow_engine:
            metrics["workflows"] = self.workflow_engine.get_metrics()
        
        # 客服系统指标
        if self.customer_service:
            metrics["customer_service"] = self.customer_service.get_system_metrics()
        
        return metrics
    
    async def _display_system_info(self):
        """显示系统信息"""
        info = f"""
╔══════════════════════════════════════════════════════════════╗
║                 Multi-Agent AI System                        ║
╠══════════════════════════════════════════════════════════════╣
║ Version: {self.config['system']['version']:<47} ║
║ Environment: {self.config['system']['environment']:<43} ║
║ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<47} ║
╠══════════════════════════════════════════════════════════════╣
║ Components:                                                  ║
║   • Agents: {len(self.agents):<47} ║
║   • Tracing: {'Enabled' if self.tracer else 'Disabled':<46} ║
║   • Message Bus: {'Running' if self.message_bus else 'Stopped':<42} ║
║   • Workflows: {'Active' if self.workflow_engine else 'Inactive':<44} ║
║   • Customer Service: {'Enabled' if self.customer_service else 'Disabled':<37} ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        print(info)
        
        # 显示可用的演示命令
        print("\nAvailable Demo Commands:")
        print("  • Research Analysis: python main.py --demo research")
        print("  • Customer Service: python main.py --demo customer_service")
        print("  • System Metrics: python main.py --metrics")
        print("  • Interactive Mode: python main.py --interactive")
    
    async def run_research_demo(self):
        """运行研究分析演示"""
        self.logger.info("Running research analysis demo...")
        
        if "research_agent_1" not in self.agents:
            self.logger.error("Research agent not available")
            return
        
        research_agent = self.agents["research_agent_1"]
        
        # 示例研究任务
        research_tasks = [
            "Analyze the latest trends in artificial intelligence",
            "Research the impact of remote work on productivity",
            "Investigate sustainable energy solutions"
        ]
        
        for task in research_tasks:
            print(f"\n🔍 Research Task: {task}")
            
            try:
                result = await research_agent.execute_research_task({
                    "query": task,
                    "depth": "comprehensive",
                    "sources": ["web", "academic", "news"]
                })
                
                print(f"✅ Research completed:")
                print(f"   Summary: {result.get('summary', 'N/A')[:200]}...")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
                print(f"   Sources: {len(result.get('sources', []))}")
                
            except Exception as e:
                print(f"❌ Research failed: {str(e)}")
    
    async def run_customer_service_demo(self):
        """运行客服系统演示"""
        self.logger.info("Running customer service demo...")
        
        if not self.customer_service:
            self.logger.error("Customer service system not available")
            return
        
        # 示例客户请求
        customer_requests = [
            ("customer_001", "I can't log into my account, it says my password is wrong"),
            ("customer_002", "My last bill was charged twice, can you help me?"),
            ("customer_003", "The app keeps crashing when I try to upload files"),
            ("customer_004", "I love your service! Just wanted to say thanks"),
            ("customer_005", "This is urgent! My entire system is down!")
        ]
        
        for customer_id, request in customer_requests:
            print(f"\n💬 Customer {customer_id}: {request}")
            
            try:
                result = await self.customer_service.process_customer_request(
                    request, customer_id
                )
                
                if result and isinstance(result, dict):
                    messages = result.get('messages', [])
                    if messages and len(messages) > 0:
                        response = messages[-1].get('content', 'No response content')
                        print(f"🤖 Agent Response: {response[:200]}...")
                    else:
                        print("🤖 Agent Response: Processing completed")
                    
                    workflow_status = result.get('workflow_status', 'unknown')
                    print(f"📊 Status: {workflow_status}")
                    
                    context = result.get('context', {})
                    if context.get('escalated'):
                        print("⚠️  Request escalated to human agent")
                else:
                    print("🤖 Agent Response: Processing completed (no detailed result)")
                
            except Exception as e:
                import traceback
                print(f"❌ Processing failed: {str(e)}")
                print(f"📋 Full traceback: {traceback.format_exc()}")
                self.logger.error(f"Customer service processing failed: {str(e)}", exc_info=True)
    
    async def show_metrics(self):
        """显示系统指标"""
        metrics = await self._collect_system_metrics()
        print("\n📊 System Metrics:")
        print(json.dumps(metrics, indent=2, default=str))
    
    async def interactive_mode(self):
        """交互模式"""
        print("\n🎮 Interactive Mode - Type 'help' for commands, 'quit' to exit")
        
        while self.running:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'help':
                    print("""
Available commands:
  agents     - List all agents
  metrics    - Show system metrics
  research   - Run research demo
  customer   - Run customer service demo
  status     - Show system status
  quit       - Exit interactive mode
                    """)
                elif command == 'agents':
                    print(f"\nRegistered Agents ({len(self.agents)}):")
                    for agent_id, agent in self.agents.items():
                        print(f"  • {agent_id}: {agent.name} ({agent.status})")
                
                elif command == 'metrics':
                    await self.show_metrics()
                
                elif command == 'research':
                    await self.run_research_demo()
                
                elif command == 'customer':
                    await self.run_customer_service_demo()
                
                elif command == 'status':
                    uptime = (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0
                    print(f"\nSystem Status:")
                    print(f"  Running: {self.running}")
                    print(f"  Uptime: {uptime:.0f} seconds")
                    print(f"  Agents: {len(self.agents)}")
                    print(f"  Memory Usage: {self._get_memory_usage():.1f} MB")
                
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Enterprise Multi-Agent AI System")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--demo", choices=["research", "customer_service"], help="Run demo")
    parser.add_argument("--metrics", action="store_true", help="Show system metrics")
    parser.add_argument("--interactive", action="store_true", help="Start in interactive mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 加载配置
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            logger.error(f"Configuration file not found: {args.config}")
            return 1
    
    # 创建系统实例
    system = MultiAgentSystem(config)
    
    try:
        # 启动系统
        await system.start()
        
        # 根据参数执行相应操作
        if args.demo == "research":
            await system.run_research_demo()
        elif args.demo == "customer_service":
            await system.run_customer_service_demo()
        elif args.metrics:
            await system.show_metrics()
        elif args.interactive:
            await system.interactive_mode()
        else:
            # 默认保持运行状态
            print("\nSystem is running. Press Ctrl+C to stop.")
            try:
                while system.running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                pass
        
        return 0
        
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        return 1
    
    finally:
        await system.shutdown()

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)