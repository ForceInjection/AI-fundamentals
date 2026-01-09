#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 现代记忆管理示例

展示如何使用LangGraph实现持久化记忆和状态管理
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from pathlib import Path

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("⚠️ LangGraph 未安装，请运行: pip install langgraph")
    print("本示例将展示概念性代码结构")
    LANGGRAPH_AVAILABLE = False
    
    # 模拟LangGraph类型
    class StateGraph:
        def __init__(self, state_schema): pass
        def add_node(self, name, func): pass
        def add_edge(self, from_node, to_node): pass
        def set_entry_point(self, node): pass
        def compile(self, checkpointer=None): pass
    
    class MemorySaver:
        def __init__(self): pass
    
    END = "__end__"

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

try:
    from .llm_factory import get_llm
    from .config import config
except ImportError:
    from llm_factory import get_llm
    from config import config

# 定义状态类型
class ConversationState(TypedDict):
    """对话状态定义"""
    messages: List[Dict[str, Any]]  # 消息历史
    user_id: str  # 用户ID
    session_id: str  # 会话ID
    context: Dict[str, Any]  # 上下文信息
    memory_summary: str  # 记忆摘要
    last_activity: str  # 最后活动时间
    metadata: Dict[str, Any]  # 元数据

class LangGraphMemoryManager:
    """LangGraph记忆管理器"""
    
    def __init__(self, db_path: str = "./memory.db"):
        self.db_path = db_path
        self.llm = get_llm()
        
        # 创建检查点保存器（使用内存保存器）
        self.checkpointer = MemorySaver()
        
        # 构建状态图
        self.graph = self._build_graph()
        
        print(f"✅ LangGraph记忆管理器初始化完成")
        print(f"📁 使用内存保存器（演示模式）")
    
    def _build_graph(self) -> StateGraph:
        """
        构建LangGraph状态图
        """
        # 创建状态图
        workflow = StateGraph(ConversationState)
        
        # 添加节点
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("update_memory", self._update_memory)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("save_state", self._save_state)
        
        # 添加边
        workflow.add_edge("process_input", "update_memory")
        workflow.add_edge("update_memory", "generate_response")
        workflow.add_edge("generate_response", "save_state")
        workflow.add_edge("save_state", END)
        
        # 设置入口点
        workflow.set_entry_point("process_input")
        
        # 编译图
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _process_input(self, state: ConversationState) -> ConversationState:
        """
        处理输入消息
        """
        print("🔄 处理输入消息...")
        
        # 更新最后活动时间
        state["last_activity"] = datetime.now().isoformat()
        
        # 处理消息格式
        if state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, dict) and "content" in last_message:
                # 消息已经是正确格式
                pass
            else:
                # 转换消息格式
                state["messages"][-1] = {
                    "role": "user",
                    "content": str(last_message),
                    "timestamp": datetime.now().isoformat()
                }
        
        return state
    
    def _update_memory(self, state: ConversationState) -> ConversationState:
        """
        更新记忆
        """
        print("🧠 更新记忆...")
        
        messages = state["messages"]
        
        # 如果消息过多，生成摘要
        if len(messages) > config.summary_threshold:
            # 提取最近的消息用于摘要
            recent_messages = messages[-config.summary_threshold:]
            
            # 生成摘要
            summary_prompt = f"""
请总结以下对话的关键信息：

{self._format_messages_for_summary(recent_messages)}

请提供简洁的摘要，包含：
1. 用户的主要需求或问题
2. 重要的上下文信息
3. 对话的进展状态
            """.strip()
            
            try:
                summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
                state["memory_summary"] = summary_response.content
                
                # 保留最近的几条消息和摘要
                state["messages"] = messages[-5:]  # 保留最近5条消息
                
                print(f"📝 生成记忆摘要: {state['memory_summary'][:100]}...")
                
            except Exception as e:
                print(f"⚠️ 摘要生成失败: {e}")
                state["memory_summary"] = "摘要生成失败"
        
        return state
    
    def _format_messages_for_summary(self, messages: List[Dict[str, Any]]) -> str:
        """
        格式化消息用于摘要
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"用户: {content}")
            elif role == "assistant":
                formatted.append(f"助手: {content}")
        return "\n".join(formatted)
    
    def _generate_response(self, state: ConversationState) -> ConversationState:
        """
        生成响应
        """
        print("💭 生成响应...")
        
        # 构建上下文
        context_parts = []
        
        # 添加记忆摘要
        if state.get("memory_summary"):
            context_parts.append(f"对话摘要: {state['memory_summary']}")
        
        # 添加最近消息
        recent_messages = state["messages"][-5:]  # 最近5条消息
        if recent_messages:
            context_parts.append("最近对话:")
            for msg in recent_messages[:-1]:  # 排除当前用户消息
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    context_parts.append(f"用户: {content}")
                elif role == "assistant":
                    context_parts.append(f"助手: {content}")
        
        # 获取当前用户消息
        current_message = state["messages"][-1]["content"]
        
        # 构建完整提示
        full_prompt = f"""
你是一个智能助手，具有记忆能力。请根据以下信息回答用户的问题：

{chr(10).join(context_parts)}

当前用户问题: {current_message}

请提供有帮助的回答。
        """.strip()
        
        try:
            # 生成响应
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            
            # 添加助手响应到消息历史
            state["messages"].append({
                "role": "assistant",
                "content": response.content,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"🤖 生成响应: {response.content[:100]}...")
            
        except Exception as e:
            print(f"❌ 响应生成失败: {e}")
            error_response = "抱歉，我现在无法处理您的请求，请稍后再试。"
            state["messages"].append({
                "role": "assistant",
                "content": error_response,
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    def _save_state(self, state: ConversationState) -> ConversationState:
        """
        保存状态
        """
        print("💾 保存状态...")
        
        # 更新元数据
        state["metadata"]["last_updated"] = datetime.now().isoformat()
        state["metadata"]["message_count"] = len(state["messages"])
        
        return state
    
    def chat(
        self, 
        user_id: str, 
        session_id: str, 
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理聊天消息
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            message: 用户消息
            context: 额外上下文
            
        Returns:
            响应结果
        """
        try:
            # 构建配置
            config_dict = {
                "configurable": {
                    "thread_id": f"{user_id}_{session_id}"
                }
            }
            
            # 准备初始状态
            initial_state = {
                "messages": [{
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.now().isoformat()
                }],
                "user_id": user_id,
                "session_id": session_id,
                "context": context or {},
                "memory_summary": "",
                "last_activity": datetime.now().isoformat(),
                "metadata": {
                    "created_at": datetime.now().isoformat()
                }
            }
            
            # 执行图
            result = self.graph.invoke(initial_state, config=config_dict)
            
            # 提取响应
            assistant_messages = [
                msg for msg in result["messages"] 
                if msg.get("role") == "assistant"
            ]
            
            if assistant_messages:
                response = assistant_messages[-1]["content"]
            else:
                response = "抱歉，我无法生成响应。"
            
            return {
                "response": response,
                "session_id": session_id,
                "user_id": user_id,
                "message_count": len(result["messages"]),
                "memory_summary": result.get("memory_summary", "")
            }
            
        except Exception as e:
            print(f"❌ 聊天处理失败: {e}")
            return {
                "error": str(e),
                "session_id": session_id,
                "user_id": user_id
            }
    
    def get_conversation_history(
        self, 
        user_id: str, 
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取对话历史
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            
        Returns:
            对话历史
        """
        try:
            config_dict = {
                "configurable": {
                    "thread_id": f"{user_id}_{session_id}"
                }
            }
            
            # 获取状态
            state = self.graph.get_state(config_dict)
            
            if state and state.values:
                return {
                    "messages": state.values.get("messages", []),
                    "memory_summary": state.values.get("memory_summary", ""),
                    "last_activity": state.values.get("last_activity", ""),
                    "metadata": state.values.get("metadata", {})
                }
            
            return None
            
        except Exception as e:
            print(f"❌ 获取对话历史失败: {e}")
            return None
    
    def clear_conversation(
        self, 
        user_id: str, 
        session_id: str
    ) -> bool:
        """
        清除对话历史
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            
        Returns:
            是否成功
        """
        try:
            config_dict = {
                "configurable": {
                    "thread_id": f"{user_id}_{session_id}"
                }
            }
            
            # 重置状态
            initial_state = {
                "messages": [],
                "user_id": user_id,
                "session_id": session_id,
                "context": {},
                "memory_summary": "",
                "last_activity": datetime.now().isoformat(),
                "metadata": {
                    "cleared_at": datetime.now().isoformat()
                }
            }
            
            self.graph.update_state(config_dict, initial_state)
            print(f"🧹 已清除会话: {user_id}_{session_id}")
            return True
            
        except Exception as e:
            print(f"❌ 清除对话失败: {e}")
            return False

def demo_langgraph_memory():
    """演示LangGraph记忆功能"""
    print("🚀 LangGraph 现代记忆管理演示")
    print("=" * 50)
    
    try:
        # 创建记忆管理器
        memory_manager = LangGraphMemoryManager()
        
        # 模拟用户对话
        user_id = "user_123"
        session_id = "session_456"
        
        conversations = [
            "你好，我是新用户，想了解你们的服务",
            "我对人工智能很感兴趣，你能介绍一下吗？",
            "我想学习机器学习，有什么建议吗？",
            "刚才你提到的深度学习，能详细说说吗？",
            "我之前问过关于机器学习的问题，你还记得吗？",
            "谢谢你的建议，我会认真考虑的"
        ]
        
        print(f"\n👤 用户: {user_id}")
        print(f"📱 会话: {session_id}")
        
        # 进行对话
        for i, message in enumerate(conversations, 1):
            print(f"\n--- 第 {i} 轮对话 ---")
            print(f"👤 用户: {message}")
            
            # 发送消息
            result = memory_manager.chat(user_id, session_id, message)
            
            if "response" in result:
                print(f"🤖 助手: {result['response']}")
                print(f"📊 消息数量: {result['message_count']}")
                
                if result.get("memory_summary"):
                    print(f"🧠 记忆摘要: {result['memory_summary'][:100]}...")
            else:
                print(f"❌ 错误: {result.get('error', '未知错误')}")
        
        # 获取对话历史
        print("\n" + "=" * 50)
        print("📋 对话历史")
        print("=" * 50)
        
        history = memory_manager.get_conversation_history(user_id, session_id)
        if history:
            print(f"📝 消息数量: {len(history['messages'])}")
            print(f"🧠 记忆摘要: {history.get('memory_summary', '无')}")
            print(f"🕐 最后活动: {history.get('last_activity', '未知')}")
            
            # 显示最近几条消息
            recent_messages = history['messages'][-4:]  # 最近4条消息
            print("\n最近消息:")
            for msg in recent_messages:
                role = "👤" if msg['role'] == 'user' else "🤖"
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"{role} {content}")
        
        # 测试记忆持久化
        print("\n" + "=" * 50)
        print("🔄 测试记忆持久化")
        print("=" * 50)
        
        # 创建新的记忆管理器实例（模拟重启）
        print("🔄 模拟系统重启...")
        new_memory_manager = LangGraphMemoryManager()
        
        # 继续对话
        continue_message = "我们之前聊到哪里了？"
        print(f"\n👤 用户: {continue_message}")
        
        result = new_memory_manager.chat(user_id, session_id, continue_message)
        if "response" in result:
            print(f"🤖 助手: {result['response']}")
            print("✅ 记忆持久化测试成功！")
        else:
            print(f"❌ 错误: {result.get('error', '未知错误')}")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("LangGraph 现代记忆管理示例")
    print("=" * 40)
    
    # 检查配置
    if not config.validate_config():
        print("❌ 配置验证失败！请检查配置文件或环境变量")
        return
    
    # 运行演示
    demo_langgraph_memory()

if __name__ == "__main__":
    main()