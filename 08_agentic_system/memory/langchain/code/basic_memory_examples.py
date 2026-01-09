#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础记忆示例 - 展示LangChain不同类型的记忆功能
"""

import os
import sys
from typing import Dict, Any
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory
)
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage

try:
    from .llm_factory import get_llm
    from .config import config
except ImportError:
    from llm_factory import get_llm
    from config import config

class MemoryExamples:
    """记忆功能示例类"""
    
    def __init__(self):
        """初始化LLM"""
        try:
            self.llm = get_llm()
            print(f"✅ 成功初始化LLM: {type(self.llm).__name__}")
        except Exception as e:
            print(f"❌ LLM初始化失败: {e}")
            sys.exit(1)
    
    def demo_conversation_buffer_memory(self):
        """
        演示ConversationBufferMemory
        保存完整的对话历史
        """
        print("\n" + "="*50)
        print("📝 ConversationBufferMemory 演示")
        print("功能：保存完整的对话历史")
        print("="*50)
        
        # 创建记忆实例
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="history"
        )
        
        # 创建对话链
        conversation = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=True
        )
        
        # 模拟对话
        conversations = [
            "你好，我叫张三，是一名软件工程师",
            "我正在学习LangChain的记忆功能",
            "你还记得我的名字和职业吗？"
        ]
        
        for i, user_input in enumerate(conversations, 1):
            print(f"\n👤 用户 {i}: {user_input}")
            response = conversation.predict(input=user_input)
            print(f"🤖 助手 {i}: {response}")
        
        # 显示记忆内容
        print("\n📋 记忆内容:")
        print(memory.buffer)
        
        return memory
    
    def demo_conversation_summary_memory(self):
        """
        演示ConversationSummaryMemory
        自动总结对话历史
        """
        print("\n" + "="*50)
        print("📝 ConversationSummaryMemory 演示")
        print("功能：自动总结对话历史，节省token")
        print("="*50)
        
        # 创建记忆实例
        memory = ConversationSummaryMemory(
            llm=self.llm,
            return_messages=True,
            memory_key="history"
        )
        
        # 创建对话链
        conversation = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=True
        )
        
        # 模拟较长的对话
        conversations = [
            "你好，我是李四，今年25岁，住在北京",
            "我是一名数据科学家，专门研究机器学习",
            "我最近在做一个关于自然语言处理的项目",
            "这个项目使用了BERT和GPT模型",
            "你能总结一下我们刚才聊的内容吗？"
        ]
        
        for i, user_input in enumerate(conversations, 1):
            print(f"\n👤 用户 {i}: {user_input}")
            response = conversation.predict(input=user_input)
            print(f"🤖 助手 {i}: {response}")
        
        # 显示记忆摘要
        print("\n📋 记忆摘要:")
        print(memory.buffer)
        
        return memory
    
    def demo_conversation_buffer_window_memory(self):
        """
        演示ConversationBufferWindowMemory
        只保留最近N轮对话
        """
        print("\n" + "="*50)
        print("📝 ConversationBufferWindowMemory 演示")
        print("功能：只保留最近N轮对话，控制记忆窗口大小")
        print("="*50)
        
        # 创建记忆实例（只保留最近2轮对话）
        memory = ConversationBufferWindowMemory(
            k=2,  # 保留最近2轮对话
            return_messages=True,
            memory_key="history"
        )
        
        # 创建对话链
        conversation = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=True
        )
        
        # 模拟多轮对话
        conversations = [
            "我叫王五，是一名教师",  # 第1轮
            "我教数学，已经工作5年了",  # 第2轮
            "我喜欢看书和旅游",  # 第3轮（此时第1轮应该被遗忘）
            "我最近去了日本旅游",  # 第4轮（此时第2轮应该被遗忘）
            "你还记得我的名字吗？"  # 第5轮（测试是否记得第1轮的信息）
        ]
        
        for i, user_input in enumerate(conversations, 1):
            print(f"\n👤 用户 {i}: {user_input}")
            response = conversation.predict(input=user_input)
            print(f"🤖 助手 {i}: {response}")
            
            # 显示当前记忆窗口
            print(f"📋 当前记忆窗口 (k={memory.k}):")
            print(memory.buffer)
        
        return memory
    
    def demo_conversation_summary_buffer_memory(self):
        """
        演示ConversationSummaryBufferMemory
        结合摘要和缓冲区的混合记忆
        """
        print("\n" + "="*50)
        print("📝 ConversationSummaryBufferMemory 演示")
        print("功能：结合摘要和缓冲区，智能管理记忆")
        print("="*50)
        
        # 创建记忆实例
        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=100,  # 设置较小的限制以避免token计算问题
            return_messages=True,
            memory_key="history"
        )
        
        # 创建对话链
        conversation = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=True
        )
        
        # 模拟长对话
        conversations = [
            "你好，我是赵六，是一名产品经理",
            "我在一家互联网公司工作，负责移动应用产品",
            "我们公司主要做电商平台，用户量超过1000万",
            "我最近在负责一个新的推荐系统项目",
            "这个推荐系统使用了深度学习和协同过滤算法",
            "我们希望通过这个系统提高用户的购买转化率",
            "你能帮我分析一下推荐系统的优化方向吗？"
        ]
        
        for i, user_input in enumerate(conversations, 1):
            print(f"\n👤 用户 {i}: {user_input}")
            response = conversation.predict(input=user_input)
            print(f"🤖 助手 {i}: {response}")
            
            # 显示当前记忆状态
            print(f"📋 当前记忆状态 (token限制: {memory.max_token_limit}):")
            if hasattr(memory, 'moving_summary_buffer') and memory.moving_summary_buffer:
                print(f"摘要: {memory.moving_summary_buffer}")
            print(f"缓冲区: {memory.chat_memory.messages[-2:] if memory.chat_memory.messages else '空'}")
        
        return memory
    
    def compare_memory_types(self):
        """
        比较不同记忆类型的特点
        """
        print("\n" + "="*60)
        print("📊 记忆类型对比")
        print("="*60)
        
        comparison_data = [
            {
                "类型": "ConversationBufferMemory",
                "特点": "保存完整对话历史",
                "优点": "信息完整，上下文丰富",
                "缺点": "token消耗大，成本高",
                "适用场景": "短对话，信息密度高的场景"
            },
            {
                "类型": "ConversationSummaryMemory",
                "特点": "自动总结对话历史",
                "优点": "节省token，成本低",
                "缺点": "可能丢失细节信息",
                "适用场景": "长对话，成本敏感的场景"
            },
            {
                "类型": "ConversationBufferWindowMemory",
                "特点": "只保留最近N轮对话",
                "优点": "固定内存使用，可预测成本",
                "缺点": "会遗忘早期重要信息",
                "适用场景": "注重最近上下文的场景"
            },
            {
                "类型": "ConversationSummaryBufferMemory",
                "特点": "结合摘要和缓冲区",
                "优点": "平衡信息保留和成本",
                "缺点": "实现复杂，调优困难",
                "适用场景": "需要平衡性能和成本的场景"
            }
        ]
        
        for item in comparison_data:
            print(f"\n🔹 {item['类型']}")
            print(f"   特点: {item['特点']}")
            print(f"   优点: {item['优点']}")
            print(f"   缺点: {item['缺点']}")
            print(f"   适用场景: {item['适用场景']}")
    
    def run_all_demos(self):
        """
        运行所有演示
        """
        print("🚀 开始LangChain记忆功能演示")
        print(f"使用模型: {type(self.llm).__name__}")
        
        try:
            # 运行各种记忆演示
            self.demo_conversation_buffer_memory()
            self.demo_conversation_summary_memory()
            self.demo_conversation_buffer_window_memory()
            
            # 跳过 ConversationSummaryBufferMemory，因为它与某些模型不兼容
            print("\n⚠️ 跳过 ConversationSummaryBufferMemory 演示（与当前模型不兼容）")
            
            # 显示对比
            self.compare_memory_types()
            
            print("\n✅ 所有演示完成！")
            
        except Exception as e:
            print(f"❌ 演示过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("LangChain 记忆功能基础演示")
    print("=" * 40)
    
    # 检查配置
    if not config.validate_config():
        print("❌ 配置验证失败！请检查配置文件或环境变量")
        print("\n请参考 config.example.py 文件进行配置")
        return
    
    # 创建演示实例
    demo = MemoryExamples()
    
    # 运行演示
    demo.run_all_demos()

if __name__ == "__main__":
    main()