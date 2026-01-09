#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能客服机器人 - 完整示例

功能特性：
1. 多用户会话管理
2. 智能记忆选择
3. 性能监控
4. 会话持久化
5. 异常处理
"""

import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory
)
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import BaseMemory

try:
    from .llm_factory import get_llm
    from .config import config
except ImportError:
    from llm_factory import get_llm
    from config import config

@dataclass
class SessionInfo:
    """会话信息"""
    session_id: str
    user_id: str
    created_at: datetime
    last_active: datetime
    message_count: int
    memory_type: str
    metadata: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """性能指标"""
    response_time: float
    token_usage: int
    memory_size: int
    timestamp: datetime

class SessionManager:
    """会话管理器"""
    
    def __init__(self, storage_dir: str = "./sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.sessions: Dict[str, SessionInfo] = {}
        self.memories: Dict[str, BaseMemory] = {}
        self.conversations: Dict[str, ConversationChain] = {}
        self.performance_metrics: Dict[str, List[PerformanceMetrics]] = {}
        
        # 初始化LLM
        try:
            self.llm = get_llm()
            print(f"✅ 会话管理器初始化成功，使用模型: {type(self.llm).__name__}")
        except Exception as e:
            print(f"❌ LLM初始化失败: {e}")
            raise
    
    def create_session(
        self, 
        user_id: str, 
        memory_type: str = "auto",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        创建新会话
        
        Args:
            user_id: 用户ID
            memory_type: 记忆类型 ("buffer", "summary", "window", "summary_buffer", "auto")
            metadata: 会话元数据
            
        Returns:
            会话ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        # 自动选择记忆类型
        if memory_type == "auto":
            memory_type = self._auto_select_memory_type(user_id)
        
        # 创建会话信息
        session_info = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_active=now,
            message_count=0,
            memory_type=memory_type,
            metadata=metadata or {}
        )
        
        # 创建记忆实例
        memory = self._create_memory(memory_type)
        
        # 创建对话链
        conversation = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=False
        )
        
        # 存储会话
        self.sessions[session_id] = session_info
        self.memories[session_id] = memory
        self.conversations[session_id] = conversation
        self.performance_metrics[session_id] = []
        
        print(f"📝 创建会话: {session_id[:8]}... (用户: {user_id}, 记忆类型: {memory_type})")
        return session_id
    
    def _auto_select_memory_type(self, user_id: str) -> str:
        """
        自动选择记忆类型
        
        Args:
            user_id: 用户ID
            
        Returns:
            记忆类型
        """
        # 获取用户历史会话统计
        user_sessions = [s for s in self.sessions.values() if s.user_id == user_id]
        
        if not user_sessions:
            # 新用户，使用缓冲记忆
            return "buffer"
        
        # 计算平均消息数
        avg_messages = sum(s.message_count for s in user_sessions) / len(user_sessions)
        
        if avg_messages < 10:
            return "buffer"  # 短对话
        elif avg_messages < 30:
            return "window"  # 中等长度对话
        else:
            return "summary_buffer"  # 长对话
    
    def _create_memory(self, memory_type: str) -> BaseMemory:
        """
        创建记忆实例
        
        Args:
            memory_type: 记忆类型
            
        Returns:
            记忆实例
        """
        if memory_type == "buffer":
            return ConversationBufferMemory(
                return_messages=True,
                memory_key="history"
            )
        elif memory_type == "summary":
            return ConversationSummaryMemory(
                llm=self.llm,
                return_messages=True,
                memory_key="history"
            )
        elif memory_type == "window":
            return ConversationBufferWindowMemory(
                k=config.max_history_length // 2,
                return_messages=True,
                memory_key="history"
            )
        elif memory_type == "summary_buffer":
            return ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=config.max_token_limit,
                return_messages=True,
                memory_key="history"
            )
        else:
            raise ValueError(f"不支持的记忆类型: {memory_type}")
    
    def chat(
        self, 
        session_id: str, 
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理聊天消息
        
        Args:
            session_id: 会话ID
            message: 用户消息
            context: 额外上下文信息
            
        Returns:
            响应结果
        """
        if session_id not in self.sessions:
            raise ValueError(f"会话不存在: {session_id}")
        
        start_time = time.time()
        
        try:
            # 获取会话组件
            session_info = self.sessions[session_id]
            conversation = self.conversations[session_id]
            memory = self.memories[session_id]
            
            # 构建输入
            input_text = message
            if context:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                input_text = f"上下文信息：\n{context_str}\n\n用户消息：{message}"
            
            # 生成响应
            response = conversation.predict(input=input_text)
            
            # 更新会话信息
            session_info.last_active = datetime.now()
            session_info.message_count += 1
            
            # 记录性能指标
            response_time = time.time() - start_time
            self._record_performance(
                session_id, 
                response_time, 
                len(message) + len(response),
                self._get_memory_size(memory)
            )
            
            return {
                "response": response,
                "session_id": session_id,
                "message_count": session_info.message_count,
                "response_time": response_time,
                "memory_type": session_info.memory_type
            }
            
        except Exception as e:
            print(f"❌ 聊天处理失败: {e}")
            return {
                "error": str(e),
                "session_id": session_id
            }
    
    def _record_performance(
        self, 
        session_id: str, 
        response_time: float, 
        token_usage: int, 
        memory_size: int
    ):
        """
        记录性能指标
        """
        metrics = PerformanceMetrics(
            response_time=response_time,
            token_usage=token_usage,
            memory_size=memory_size,
            timestamp=datetime.now()
        )
        
        self.performance_metrics[session_id].append(metrics)
        
        # 保留最近100条记录
        if len(self.performance_metrics[session_id]) > 100:
            self.performance_metrics[session_id] = self.performance_metrics[session_id][-100:]
    
    def _get_memory_size(self, memory: BaseMemory) -> int:
        """
        获取记忆大小（估算）
        """
        try:
            if hasattr(memory, 'buffer'):
                return len(str(memory.buffer))
            elif hasattr(memory, 'chat_memory'):
                return len(str(memory.chat_memory.messages))
            else:
                return 0
        except:
            return 0
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息
        """
        if session_id not in self.sessions:
            return None
        
        session_info = self.sessions[session_id]
        metrics = self.performance_metrics.get(session_id, [])
        
        # 计算平均性能指标
        avg_response_time = sum(m.response_time for m in metrics) / len(metrics) if metrics else 0
        avg_token_usage = sum(m.token_usage for m in metrics) / len(metrics) if metrics else 0
        
        return {
            "session_info": asdict(session_info),
            "performance": {
                "avg_response_time": avg_response_time,
                "avg_token_usage": avg_token_usage,
                "total_interactions": len(metrics)
            }
        }
    
    def list_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        列出用户的所有会话
        """
        user_sessions = [
            {
                "session_id": session_id,
                "created_at": session_info.created_at.isoformat(),
                "last_active": session_info.last_active.isoformat(),
                "message_count": session_info.message_count,
                "memory_type": session_info.memory_type
            }
            for session_id, session_info in self.sessions.items()
            if session_info.user_id == user_id
        ]
        
        return sorted(user_sessions, key=lambda x: x["last_active"], reverse=True)
    
    def save_session(self, session_id: str) -> bool:
        """
        保存会话到磁盘
        """
        if session_id not in self.sessions:
            return False
        
        try:
            session_file = self.storage_dir / f"{session_id}.json"
            
            # 准备保存数据
            save_data = {
                "session_info": asdict(self.sessions[session_id]),
                "memory_type": self.sessions[session_id].memory_type,
                "performance_metrics": [
                    asdict(m) for m in self.performance_metrics.get(session_id, [])
                ]
            }
            
            # 序列化datetime对象
            def datetime_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=datetime_serializer)
            
            return True
            
        except Exception as e:
            print(f"❌ 保存会话失败: {e}")
            return False
    
    def load_session(self, session_id: str) -> bool:
        """
        从磁盘加载会话
        """
        try:
            session_file = self.storage_dir / f"{session_id}.json"
            
            if not session_file.exists():
                return False
            
            with open(session_file, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            # 恢复会话信息
            session_data = save_data["session_info"]
            session_data["created_at"] = datetime.fromisoformat(session_data["created_at"])
            session_data["last_active"] = datetime.fromisoformat(session_data["last_active"])
            
            session_info = SessionInfo(**session_data)
            
            # 重新创建记忆和对话链
            memory = self._create_memory(session_info.memory_type)
            conversation = ConversationChain(
                llm=self.llm,
                memory=memory,
                verbose=False
            )
            
            # 恢复性能指标
            metrics_data = save_data.get("performance_metrics", [])
            metrics = [
                PerformanceMetrics(
                    response_time=m["response_time"],
                    token_usage=m["token_usage"],
                    memory_size=m["memory_size"],
                    timestamp=datetime.fromisoformat(m["timestamp"])
                )
                for m in metrics_data
            ]
            
            # 存储到内存
            self.sessions[session_id] = session_info
            self.memories[session_id] = memory
            self.conversations[session_id] = conversation
            self.performance_metrics[session_id] = metrics
            
            print(f"✅ 成功加载会话: {session_id[:8]}...")
            return True
            
        except Exception as e:
            print(f"❌ 加载会话失败: {e}")
            return False
    
    def cleanup_inactive_sessions(self, hours: int = 24):
        """
        清理不活跃的会话
        
        Args:
            hours: 不活跃时间阈值（小时）
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        inactive_sessions = [
            session_id for session_id, session_info in self.sessions.items()
            if session_info.last_active < cutoff_time
        ]
        
        for session_id in inactive_sessions:
            # 保存会话
            self.save_session(session_id)
            
            # 从内存中移除
            del self.sessions[session_id]
            del self.memories[session_id]
            del self.conversations[session_id]
            del self.performance_metrics[session_id]
        
        print(f"🧹 清理了 {len(inactive_sessions)} 个不活跃会话")

class CustomerServiceBot:
    """智能客服机器人"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.system_prompt = """
你是一个专业的客服助手，具有以下特点：
1. 友好、耐心、专业
2. 能够记住对话历史
3. 提供准确的帮助和建议
4. 在无法解决问题时，会引导用户联系人工客服

请根据用户的问题提供有帮助的回答。
        """.strip()
    
    def start_conversation(self, user_id: str, user_name: str = None) -> str:
        """
        开始新对话
        """
        metadata = {}
        if user_name:
            metadata["user_name"] = user_name
        
        session_id = self.session_manager.create_session(
            user_id=user_id,
            memory_type="auto",
            metadata=metadata
        )
        
        # 发送欢迎消息
        welcome_msg = f"您好{user_name or ''}！我是智能客服助手，很高兴为您服务。请问有什么可以帮助您的吗？"
        
        return session_id, welcome_msg
    
    def chat(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        处理用户消息
        """
        # 添加系统提示
        context = {"系统角色": self.system_prompt}
        
        return self.session_manager.chat(session_id, message, context)
    
    def get_conversation_summary(self, session_id: str) -> str:
        """
        获取对话摘要
        """
        session_info = self.session_manager.get_session_info(session_id)
        if not session_info:
            return "会话不存在"
        
        info = session_info["session_info"]
        perf = session_info["performance"]
        
        return f"""
📊 对话摘要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👤 用户ID: {info['user_id']}
🕐 创建时间: {info['created_at']}
💬 消息数量: {info['message_count']}
🧠 记忆类型: {info['memory_type']}
⚡ 平均响应时间: {perf['avg_response_time']:.2f}秒
📝 平均Token使用: {perf['avg_token_usage']:.0f}
        """.strip()

def demo_customer_service():
    """演示智能客服功能"""
    print("🤖 智能客服机器人演示")
    print("=" * 40)
    
    # 创建客服机器人
    bot = CustomerServiceBot()
    
    # 模拟多个用户的对话
    users = [
        {"user_id": "user_001", "name": "张三"},
        {"user_id": "user_002", "name": "李四"}
    ]
    
    sessions = {}
    
    # 为每个用户创建会话
    for user in users:
        session_id, welcome = bot.start_conversation(user["user_id"], user["name"])
        sessions[user["user_id"]] = session_id
        print(f"\n👤 {user['name']} 开始对话")
        print(f"🤖 {welcome}")
    
    # 模拟对话
    conversations = {
        "user_001": [
            "我想查询我的订单状态",
            "我的订单号是 ORD123456",
            "什么时候能发货？",
            "好的，谢谢你的帮助"
        ],
        "user_002": [
            "我收到的商品有质量问题",
            "是一件衣服，颜色和描述不符",
            "我想申请退货",
            "需要什么手续吗？"
        ]
    }
    
    # 交替进行对话
    max_rounds = max(len(convs) for convs in conversations.values())
    
    for round_num in range(max_rounds):
        for user_id, convs in conversations.items():
            if round_num < len(convs):
                user_name = next(u["name"] for u in users if u["user_id"] == user_id)
                session_id = sessions[user_id]
                message = convs[round_num]
                
                print(f"\n👤 {user_name}: {message}")
                
                result = bot.chat(session_id, message)
                if "response" in result:
                    print(f"🤖 客服: {result['response']}")
                    print(f"📊 响应时间: {result['response_time']:.2f}秒")
                else:
                    print(f"❌ 错误: {result.get('error', '未知错误')}")
    
    # 显示会话摘要
    print("\n" + "=" * 50)
    print("📋 会话摘要")
    print("=" * 50)
    
    for user in users:
        session_id = sessions[user["user_id"]]
        summary = bot.get_conversation_summary(session_id)
        print(f"\n{summary}")
    
    # 保存会话
    print("\n💾 保存会话...")
    for session_id in sessions.values():
        if bot.session_manager.save_session(session_id):
            print(f"✅ 会话 {session_id[:8]}... 保存成功")
        else:
            print(f"❌ 会话 {session_id[:8]}... 保存失败")

def main():
    """主函数"""
    print("智能客服机器人 - 完整示例")
    print("=" * 40)
    
    # 检查配置
    if not config.validate_config():
        print("❌ 配置验证失败！请检查配置文件或环境变量")
        return
    
    # 运行演示
    demo_customer_service()

if __name__ == "__main__":
    main()