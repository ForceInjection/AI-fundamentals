import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import asynccontextmanager

# 导入基础组件
from ..agents.base_agent import BaseAgent, Belief, Desire, Intention, AgentResult, AgentStatus
from ..communication.message_bus import MessageBus, Message, MessageType, MessagePriority
from ..workflows.langgraph_workflow import (
    EnhancedAgentState, StateValidator, WorkflowNode, 
    ConditionalRouter, StateGraph, CompiledGraph
)
from ..monitoring.langsmith_integration import (
    EnterpriseTracing, TraceLevel, trace_agent_execution, trace_workflow_step
)

class CustomerServicePriority(Enum):
    """客服优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TicketStatus(Enum):
    """工单状态"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_CUSTOMER = "pending_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"

class CustomerSentiment(Enum):
    """客户情感"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

@dataclass
class CustomerProfile:
    """客户档案"""
    customer_id: str
    name: str
    email: str
    phone: Optional[str] = None
    tier: str = "standard"  # standard, premium, vip
    language: str = "en"
    timezone: str = "UTC"
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    satisfaction_score: float = 0.0
    total_interactions: int = 0

@dataclass
class SupportTicket:
    """支持工单"""
    ticket_id: str
    customer_id: str
    subject: str
    description: str
    category: str
    priority: CustomerServicePriority
    status: TicketStatus
    created_at: datetime
    updated_at: datetime
    assigned_agent: Optional[str] = None
    resolution: Optional[str] = None
    satisfaction_rating: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

class CustomerServiceAgent(BaseAgent):
    """客服智能体"""
    
    def __init__(self, agent_id: str, name: str, specializations: List[str] = None,
                 max_concurrent_tickets: int = 5):
        config = {
            "name": name,
            "specializations": specializations or [],
            "max_concurrent_tickets": max_concurrent_tickets
        }
        super().__init__(agent_id, config)
        
        self.specializations = self.config.get("specializations", [])
        self.max_concurrent_tickets = self.config.get("max_concurrent_tickets", 5)
        self.active_tickets: Dict[str, SupportTicket] = {}
        
        # 客服特定能力
        self.capabilities.extend([
            "sentiment_analysis",
            "intent_recognition",
            "knowledge_base_search",
            "escalation_management",
            "multilingual_support"
        ])
        
        # 客服工具
        self.tools.update({
            "ticket_management": self._manage_ticket,
            "knowledge_search": self._search_knowledge_base,
            "sentiment_analyzer": self._analyze_sentiment,
            "intent_classifier": self._classify_intent,
            "response_generator": self._generate_response,
            "escalation_handler": self._handle_escalation
        })
        
        # 性能指标
        self.performance_metrics.update({
            "tickets_handled": 0,
            "avg_response_time": 0.0,
            "avg_resolution_time": 0.0,
            "customer_satisfaction": 0.0,
            "escalation_rate": 0.0,
            "first_contact_resolution": 0.0
        })

    async def execute(self, input_data: Dict[str, Any]) -> "AgentResult":
        """执行客服智能体的主要任务"""
        customer_message = input_data.get("customer_message")
        customer_profile = input_data.get("customer_profile")
        context = input_data.get("context", {})

        if not customer_message or not customer_profile:
            raise ValueError("Missing customer_message or customer_profile in input_data")

        result = await self.handle_customer_inquiry(
            customer_message, customer_profile, context
        )

        return AgentResult(
            agent_id=self.agent_id,
            status=AgentStatus.COMPLETED,
            data=result,
            execution_time=0,  # This will be updated by the run method
            confidence_score=result.get("confidence", 0.85)
        )
    
    async def handle_customer_inquiry(self, customer_message: str, 
                                    customer_profile: CustomerProfile,
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理客户咨询"""
        context = context or {}
        
        # 创建或更新工单
        ticket = await self._create_or_update_ticket(
            customer_message, customer_profile, context
        )
        
        # 分析客户情感和意图
        sentiment = await self._analyze_sentiment(customer_message)
        intent = await self._classify_intent(customer_message)
        
        # 更新信念
        self.beliefs.append(Belief(
            content={
                "customer_sentiment": sentiment,
                "customer_intent": intent,
                "ticket_info": ticket,
                "customer_profile": customer_profile
            },
            confidence=0.9,
            source="customer_inquiry",
            timestamp=datetime.now()
        ))
        
        # 生成响应策略
        response_strategy = await self._plan_response_strategy(
            ticket, sentiment, intent, customer_profile
        )
        
        # 执行响应
        response = await self._execute_response_strategy(response_strategy)
        
        # 更新工单
        ticket.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "agent_response",
            "content": response["message"],
            "sentiment": sentiment.value,
            "intent": intent
        })
        
        # 更新性能指标
        await self._update_performance_metrics(ticket, response)
        
        return {
            "ticket_id": ticket.ticket_id,
            "response": response,
            "sentiment": sentiment.value,
            "intent": intent,
            "next_actions": response_strategy.get("next_actions", [])
        }
    
    async def _create_or_update_ticket(self, message: str, 
                                     customer_profile: CustomerProfile,
                                     context: Dict[str, Any]) -> SupportTicket:
        """创建或更新工单"""
        # 检查是否有现有的开放工单
        existing_ticket = None
        for ticket in self.active_tickets.values():
            if (ticket.customer_id == customer_profile.customer_id and 
                ticket.status in [TicketStatus.OPEN, TicketStatus.IN_PROGRESS]):
                existing_ticket = ticket
                break
        
        if existing_ticket:
            # 更新现有工单
            existing_ticket.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "customer_message",
                "content": message
            })
            existing_ticket.updated_at = datetime.now()
            return existing_ticket
        else:
            # 创建新工单
            ticket_id = str(uuid.uuid4())
            
            # 分类工单
            category = await self._classify_ticket_category(message)
            priority = await self._determine_priority(message, customer_profile)
            
            ticket = SupportTicket(
                ticket_id=ticket_id,
                customer_id=customer_profile.customer_id,
                subject=message[:100] + "..." if len(message) > 100 else message,
                description=message,
                category=category,
                priority=priority,
                status=TicketStatus.OPEN,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                assigned_agent=self.agent_id,
                conversation_history=[{
                    "timestamp": datetime.now().isoformat(),
                    "type": "customer_message",
                    "content": message
                }]
            )
            
            self.active_tickets[ticket_id] = ticket
            return ticket
    
    async def _analyze_sentiment(self, text: str) -> CustomerSentiment:
        """分析客户情感"""
        # 模拟情感分析
        negative_keywords = ["angry", "frustrated", "terrible", "awful", "hate", "worst"]
        positive_keywords = ["great", "excellent", "love", "amazing", "perfect", "wonderful"]
        
        text_lower = text.lower()
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        
        if negative_count > positive_count:
            return CustomerSentiment.NEGATIVE if negative_count == 1 else CustomerSentiment.VERY_NEGATIVE
        elif positive_count > negative_count:
            return CustomerSentiment.POSITIVE if positive_count == 1 else CustomerSentiment.VERY_POSITIVE
        else:
            return CustomerSentiment.NEUTRAL
    
    async def _classify_intent(self, text: str) -> str:
        """分类客户意图"""
        # 模拟意图分类
        intent_keywords = {
            "billing_inquiry": ["bill", "charge", "payment", "invoice", "cost"],
            "technical_support": ["error", "bug", "not working", "broken", "issue"],
            "account_management": ["account", "profile", "settings", "password", "login"],
            "product_information": ["how to", "feature", "function", "capability"],
            "complaint": ["complain", "dissatisfied", "problem", "issue", "wrong"],
            "compliment": ["thank", "great", "excellent", "satisfied", "good job"]
        }
        
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        else:
            return "general_inquiry"
    
    async def _classify_ticket_category(self, message: str) -> str:
        """分类工单类别"""
        categories = {
            "Technical": ["error", "bug", "not working", "broken", "crash"],
            "Billing": ["bill", "charge", "payment", "invoice", "refund"],
            "Account": ["account", "profile", "password", "login", "access"],
            "Product": ["feature", "how to", "tutorial", "guide", "function"],
            "General": ["question", "help", "support", "information"]
        }
        
        message_lower = message.lower()
        for category, keywords in categories.items():
            if any(keyword in message_lower for keyword in keywords):
                return category
        
        return "General"
    
    async def _determine_priority(self, message: str, 
                                customer_profile: CustomerProfile) -> CustomerServicePriority:
        """确定工单优先级"""
        # VIP客户自动高优先级
        if customer_profile.tier == "vip":
            return CustomerServicePriority.HIGH
        
        # 基于关键词确定优先级
        urgent_keywords = ["urgent", "emergency", "critical", "down", "not working"]
        high_keywords = ["important", "asap", "quickly", "soon"]
        
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in urgent_keywords):
            return CustomerServicePriority.URGENT
        elif any(keyword in message_lower for keyword in high_keywords):
            return CustomerServicePriority.HIGH
        elif customer_profile.tier == "premium":
            return CustomerServicePriority.MEDIUM
        else:
            return CustomerServicePriority.LOW
    
    async def _plan_response_strategy(self, ticket: SupportTicket, 
                                    sentiment: CustomerSentiment,
                                    intent: str, 
                                    customer_profile: CustomerProfile) -> Dict[str, Any]:
        """规划响应策略"""
        strategy = {
            "response_type": "standard",
            "tone": "professional",
            "actions": [],
            "escalation_needed": False,
            "knowledge_base_search": True,
            "follow_up_required": False
        }
        
        # 根据情感调整语调
        if sentiment in [CustomerSentiment.NEGATIVE, CustomerSentiment.VERY_NEGATIVE]:
            strategy["tone"] = "empathetic"
            strategy["actions"].append("acknowledge_frustration")
            
            if sentiment == CustomerSentiment.VERY_NEGATIVE:
                strategy["escalation_needed"] = True
        
        # 根据意图确定行动
        if intent == "technical_support":
            strategy["actions"].extend(["gather_technical_details", "provide_troubleshooting"])
        elif intent == "billing_inquiry":
            strategy["actions"].extend(["verify_account", "explain_charges"])
        elif intent == "complaint":
            strategy["actions"].extend(["acknowledge_issue", "investigate_problem"])
            strategy["follow_up_required"] = True
        
        # 根据优先级调整
        if ticket.priority == CustomerServicePriority.URGENT:
            strategy["response_type"] = "immediate"
            strategy["escalation_needed"] = True
        
        return strategy
    
    async def _execute_response_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """执行响应策略"""
        response_parts = []
        
        # 根据语调生成开场白
        if strategy["tone"] == "empathetic":
            response_parts.append("I understand your frustration, and I'm here to help resolve this issue for you.")
        else:
            response_parts.append("Thank you for contacting us. I'll be happy to assist you today.")
        
        # 执行策略行动
        for action in strategy["actions"]:
            if action == "acknowledge_frustration":
                response_parts.append("I sincerely apologize for any inconvenience this has caused.")
            elif action == "gather_technical_details":
                response_parts.append("To better assist you, could you please provide more details about the issue you're experiencing?")
            elif action == "provide_troubleshooting":
                response_parts.append("Let me walk you through some troubleshooting steps that should resolve this issue.")
            elif action == "verify_account":
                response_parts.append("For security purposes, I'll need to verify your account information.")
            elif action == "explain_charges":
                response_parts.append("I'll review your billing details and explain each charge for you.")
            elif action == "acknowledge_issue":
                response_parts.append("I take your concerns seriously and will investigate this matter thoroughly.")
        
        # 知识库搜索（模拟）
        if strategy["knowledge_base_search"]:
            kb_result = await self._search_knowledge_base("relevant_solution")
            if kb_result:
                response_parts.append(f"Based on our knowledge base: {kb_result}")
        
        # 生成完整响应
        full_response = " ".join(response_parts)
        
        return {
            "message": full_response,
            "tone": strategy["tone"],
            "escalation_needed": strategy["escalation_needed"],
            "follow_up_required": strategy["follow_up_required"],
            "confidence": 0.85
        }
    
    async def _search_knowledge_base(self, query: str) -> Optional[str]:
        """搜索知识库"""
        # 模拟知识库搜索
        knowledge_base = {
            "billing": "You can view your billing details in the Account section of your dashboard.",
            "technical": "Please try clearing your browser cache and cookies, then restart the application.",
            "account": "You can reset your password using the 'Forgot Password' link on the login page.",
            "general": "For additional assistance, please refer to our comprehensive help documentation."
        }
        
        # 简单的关键词匹配
        for category, solution in knowledge_base.items():
            if category in query.lower():
                return solution
        
        return knowledge_base["general"]
    
    async def _handle_escalation(self, ticket: SupportTicket, reason: str) -> Dict[str, Any]:
        """处理升级"""
        # 更新工单状态
        ticket.status = TicketStatus.IN_PROGRESS
        ticket.tags.append("escalated")
        ticket.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "system_note",
            "content": f"Ticket escalated: {reason}"
        })
        
        # 更新性能指标
        self.performance_metrics["escalation_rate"] += 1
        
        return {
            "escalated": True,
            "reason": reason,
            "new_status": ticket.status.value,
            "escalation_time": datetime.now().isoformat()
        }
    
    async def _update_performance_metrics(self, ticket: SupportTicket, 
                                        response: Dict[str, Any]):
        """更新性能指标"""
        self.performance_metrics["tickets_handled"] += 1
        
        # 计算响应时间（模拟）
        response_time = 30.0  # 秒
        current_avg = self.performance_metrics["avg_response_time"]
        total_tickets = self.performance_metrics["tickets_handled"]
        
        self.performance_metrics["avg_response_time"] = (
            (current_avg * (total_tickets - 1) + response_time) / total_tickets
        )

    async def _manage_ticket(self, ticket_id: str, action: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """管理工单"""
        if ticket_id not in self.active_tickets:
            return {"error": "Ticket not found"}

        ticket = self.active_tickets[ticket_id]
        data = data or {}

        if action == "update_status":
            new_status = data.get("status")
            if new_status and hasattr(TicketStatus, new_status.upper()):
                ticket.status = TicketStatus[new_status.upper()]
                ticket.updated_at = datetime.now()
                return {"status": "success", "new_status": ticket.status.value}
            else:
                return {"error": "Invalid status"}
        elif action == "add_note":
            note = data.get("note")
            if note:
                ticket.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "internal_note",
                    "content": note,
                    "agent_id": self.agent_id
                })
                return {"status": "success"}
            else:
                return {"error": "Note content is missing"}
        else:
            return {"error": f"Unknown ticket action: {action}"}

    async def _generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """生成客服回复"""
        full_prompt = f"{context.get('history', '')}\n\nCustomer: {prompt}\nAgent:"
        
        # 此处应调用一个真正的语言模型
        # 为简化，我们返回一个基于模板的响应
        if "password" in prompt.lower():
            return "I can help with that. To reset your password, please go to the login page and click 'Forgot Password'."
        elif "billing" in prompt.lower():
            return "For billing inquiries, I can connect you to our billing department. Would you like me to do that?"
        else:
            return "Thank you for contacting us. How can I help you today?"

class CustomerServiceWorkflow:
    """客服工作流"""
    
    def __init__(self, tracer: EnterpriseTracing, message_bus: MessageBus):
        self.tracer = tracer
        self.message_bus = message_bus
        self.agents: Dict[str, CustomerServiceAgent] = {}
        self.customer_profiles: Dict[str, CustomerProfile] = {}
        
        # 工作流状态
        self.state_validator = StateValidator()
        self.workflow_graph = None
        
        self._setup_workflow()
    
    def _setup_workflow(self):
        """设置工作流"""
        # 创建工作流图
        self.workflow_graph = StateGraph(EnhancedAgentState)
        
        # 添加节点
        self.workflow_graph.add_node("intake", self._intake_node)
        self.workflow_graph.add_node("triage", self._triage_node)
        self.workflow_graph.add_node("assignment", self._assignment_node)
        self.workflow_graph.add_node("processing", self._processing_node)
        self.workflow_graph.add_node("escalation", self._escalation_node)
        self.workflow_graph.add_node("resolution", self._resolution_node)
        
        # 设置入口点
        self.workflow_graph.set_entry_point("intake")
        
        # 添加边
        self.workflow_graph.add_edge("intake", "triage")
        self.workflow_graph.add_conditional_edges(
            "triage",
            self._triage_router,
            {
                "standard": "assignment",
                "escalation": "escalation",
                "auto_resolve": "resolution"
            }
        )
        self.workflow_graph.add_edge("assignment", "processing")
        self.workflow_graph.add_conditional_edges(
            "processing",
            self._processing_router,
            {
                "resolved": "resolution",
                "escalate": "escalation",
                "continue": "processing"
            }
        )
        self.workflow_graph.add_edge("escalation", "assignment")
        self.workflow_graph.add_edge("resolution", "__end__")
        
        # 编译工作流
        self.compiled_workflow = self.workflow_graph.compile()
    
    async def process_customer_request(self, customer_message: str, 
                                     customer_id: str,
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理客户请求"""
        trace_id = self.tracer.create_trace(
            name="Customer Service Request",
            tags=["customer_service", "workflow"],
            metadata={"customer_id": customer_id}
        )
        
        # 获取或创建客户档案
        customer_profile = self._get_or_create_customer_profile(customer_id)
        
        # 初始化工作流状态
        initial_state = EnhancedAgentState(
            messages=[{"role": "user", "content": customer_message}],
            current_agent="system",
            workflow_status="active",
            context={
                "customer_profile": customer_profile,
                "trace_id": trace_id,
                "request_context": context or {}
            }
        )
        
        async with self.tracer.trace_run(
            trace_id=trace_id,
            name="Customer Service Workflow",
            run_type="workflow",
            inputs={"message": customer_message, "customer_id": customer_id}
        ) as run_context:
            try:
                # 执行工作流
                result = await self.compiled_workflow.ainvoke(initial_state)
                
                run_context.set_outputs({"result": result})
                return result
                
            except Exception as e:
                run_context.log(f"Workflow execution failed: {str(e)}", TraceLevel.ERROR)
                raise
    
    def _get_or_create_customer_profile(self, customer_id: str) -> CustomerProfile:
        """获取或创建客户档案"""
        if customer_id not in self.customer_profiles:
            self.customer_profiles[customer_id] = CustomerProfile(
                customer_id=customer_id,
                name=f"Customer_{customer_id[:8]}",
                email=f"{customer_id}@example.com",
                tier="standard"
            )
        
        return self.customer_profiles[customer_id]
    
    async def _intake_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """接收节点"""
        # 记录客户请求
        customer_message = state.messages[-1]["content"]
        customer_profile = state.context["customer_profile"]
        
        # 更新客户档案
        customer_profile.total_interactions += 1
        customer_profile.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "incoming_request",
            "content": customer_message
        })
        
        state.context["intake_completed"] = True
        state.context["intake_timestamp"] = datetime.now().isoformat()
        
        return state
    
    async def _triage_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """分流节点"""
        customer_message = state.messages[-1]["content"]
        customer_profile = state.context["customer_profile"]
        
        # 创建临时智能体进行分析
        temp_agent = CustomerServiceAgent("triage_agent", "Triage Agent")
        
        # 分析情感和意图
        sentiment = await temp_agent._analyze_sentiment(customer_message)
        intent = await temp_agent._classify_intent(customer_message)
        category = await temp_agent._classify_ticket_category(customer_message)
        priority = await temp_agent._determine_priority(customer_message, customer_profile)
        
        # 更新状态
        state.context.update({
            "sentiment": sentiment.value,
            "intent": intent,
            "category": category,
            "priority": priority.value,
            "triage_completed": True
        })
        
        return state
    
    def _triage_router(self, state: EnhancedAgentState) -> str:
        """分流路由器"""
        priority = state.context.get("priority")
        sentiment = state.context.get("sentiment")
        
        # 紧急情况直接升级
        if priority == "urgent" or sentiment == "very_negative":
            return "escalation"
        
        # 简单问题自动解决
        if state.context.get("intent") in ["compliment", "general_inquiry"]:
            return "auto_resolve"
        
        return "standard"
    
    async def _assignment_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """分配节点"""
        category = state.context.get("category")
        priority = state.context.get("priority")
        
        # 选择最合适的智能体
        assigned_agent = self._select_best_agent(category, priority)
        
        if assigned_agent:
            state.current_agent = assigned_agent.agent_id
            state.context["assigned_agent"] = assigned_agent.agent_id
        else:
            # 如果没有可用智能体，创建新的
            agent_id = f"agent_{len(self.agents) + 1}"
            new_agent = CustomerServiceAgent(
                agent_id, f"Service Agent {len(self.agents) + 1}",
                specializations=[category]
            )
            self.agents[agent_id] = new_agent
            state.current_agent = agent_id
            state.context["assigned_agent"] = agent_id
        
        return state
    
    def _select_best_agent(self, category: str, priority: str) -> Optional[CustomerServiceAgent]:
        """选择最佳智能体"""
        available_agents = [
            agent for agent in self.agents.values()
            if len(agent.active_tickets) < agent.max_concurrent_tickets
        ]
        
        if not available_agents:
            return None
        
        # 优先选择专业匹配的智能体
        specialized_agents = [
            agent for agent in available_agents
            if category.lower() in [spec.lower() for spec in agent.specializations]
        ]
        
        if specialized_agents:
            # 选择工作负载最轻的专业智能体
            return min(specialized_agents, key=lambda a: len(a.active_tickets))
        else:
            # 选择工作负载最轻的通用智能体
            return min(available_agents, key=lambda a: len(a.active_tickets))
    
    async def _processing_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """处理节点"""
        agent_id = state.context.get("assigned_agent")
        agent = self.agents.get(agent_id)
        
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        customer_message = state.messages[-1]["content"]
        customer_profile = state.context["customer_profile"]
        
        # 处理客户咨询
        response = await agent.handle_customer_inquiry(
            customer_message, customer_profile, state.context
        )
        
        # 更新状态
        state.messages.append({
            "role": "assistant",
            "content": response["response"]["message"]
        })
        
        state.context.update({
            "agent_response": response,
            "processing_completed": True,
            "escalation_needed": response["response"].get("escalation_needed", False)
        })
        
        return state
    
    def _processing_router(self, state: EnhancedAgentState) -> str:
        """处理路由器"""
        if state.context.get("escalation_needed"):
            return "escalate"
        
        # 检查是否需要进一步处理
        response = state.context.get("agent_response", {})
        if response.get("response", {}).get("follow_up_required"):
            return "continue"
        
        return "resolved"
    
    async def _escalation_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """升级节点"""
        # 标记为升级状态
        state.context["escalated"] = True
        state.context["escalation_timestamp"] = datetime.now().isoformat()
        
        # 这里可以添加升级到人工客服的逻辑
        state.messages.append({
            "role": "system",
            "content": "Your request has been escalated to a senior representative who will assist you shortly."
        })
        
        return state
    
    async def _resolution_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """解决节点"""
        state.workflow_status = "completed"
        state.context["resolution_timestamp"] = datetime.now().isoformat()
        
        # 计算处理时长
        if "intake_timestamp" in state.context:
            intake_time = datetime.fromisoformat(state.context["intake_timestamp"])
            resolution_time = datetime.now()
            processing_duration = (resolution_time - intake_time).total_seconds()
            state.context["processing_duration_seconds"] = processing_duration
        
        return state
    
    def add_agent(self, agent: CustomerServiceAgent):
        """添加智能体"""
        self.agents[agent.agent_id] = agent
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        total_tickets = sum(agent.performance_metrics["tickets_handled"] for agent in self.agents.values())
        avg_satisfaction = sum(agent.performance_metrics["customer_satisfaction"] for agent in self.agents.values()) / len(self.agents) if self.agents else 0
        
        return {
            "total_agents": len(self.agents),
            "total_tickets_handled": total_tickets,
            "average_customer_satisfaction": avg_satisfaction,
            "active_tickets": sum(len(agent.active_tickets) for agent in self.agents.values()),
            "agent_utilization": {
                agent_id: len(agent.active_tickets) / agent.max_concurrent_tickets
                for agent_id, agent in self.agents.items()
            }
        }

# 使用示例
async def main():
    """主函数示例"""
    # 初始化组件
    tracer = EnterpriseTracing()
    await tracer.start()
    
    message_bus = MessageBus()
    await message_bus.start()
    
    # 创建客服系统
    customer_service = CustomerServiceWorkflow(tracer, message_bus)
    
    # 添加智能体
    tech_agent = CustomerServiceAgent(
        "tech_agent_1", "Technical Support Agent",
        specializations=["Technical", "Product"]
    )
    billing_agent = CustomerServiceAgent(
        "billing_agent_1", "Billing Support Agent",
        specializations=["Billing", "Account"]
    )
    
    customer_service.add_agent(tech_agent)
    customer_service.add_agent(billing_agent)
    
    # 处理客户请求
    customer_requests = [
        "I'm having trouble logging into my account",
        "My bill seems incorrect this month",
        "The application keeps crashing when I try to save",
        "I want to upgrade my subscription"
    ]
    
    for i, request in enumerate(customer_requests):
        customer_id = f"customer_{i+1}"
        print(f"\nProcessing request from {customer_id}: {request}")
        
        try:
            result = await customer_service.process_customer_request(
                request, customer_id
            )
            
            print(f"Response: {result.messages[-1]['content']}")
            print(f"Status: {result.workflow_status}")
            
        except Exception as e:
            print(f"Error processing request: {str(e)}")
    
    # 显示系统指标
    metrics = customer_service.get_system_metrics()
    print(f"\nSystem Metrics: {json.dumps(metrics, indent=2)}")
    
    # 清理
    await message_bus.stop()
    await tracer.stop()

if __name__ == "__main__":
    asyncio.run(main())