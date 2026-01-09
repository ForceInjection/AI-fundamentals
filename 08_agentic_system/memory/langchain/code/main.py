#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主运行脚本 - LangChain 记忆功能演示

使用方法：
    python main.py --demo basic          # 基础记忆演示
    python main.py --demo customer       # 智能客服演示
    python main.py --demo langgraph      # LangGraph记忆演示
    python main.py --demo all            # 运行所有演示
    python main.py --interactive         # 交互式聊天
    python main.py --config-check        # 检查配置
"""

import os
import sys
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import config
    from llm_factory import get_llm
    from basic_memory_examples import MemoryExamples
    from smart_customer_service import CustomerServiceBot
    from langgraph_memory_example import LangGraphMemoryManager, LANGGRAPH_AVAILABLE
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保所有依赖已安装: pip install -r requirements.txt")
    sys.exit(1)

def check_config():
    """检查配置"""
    print("🔍 检查配置...")
    print("=" * 40)
    
    # 检查配置文件
    config_file = Path("config.py")
    if config_file.exists():
        print("✅ 配置文件存在: config.py")
    else:
        print("⚠️ 配置文件不存在，使用环境变量")
        print("💡 建议复制 config.example.py 为 config.py 并修改配置")
    
    # 检查各种配置
    configs = {
        "OpenAI": config.validate_config("openai"),
        "DeepSeek": config.validate_config("deepseek"),
        "本地模型": config.validate_config("local"),
        "任意配置": config.validate_config()
    }
    
    print("\n📋 配置状态:")
    for name, status in configs.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {name}: {'可用' if status else '不可用'}")
    
    # 显示当前配置值
    print("\n⚙️ 当前配置:")
    print(f"  OpenAI API Key: {'已设置' if config.openai_api_key else '未设置'}")
    print(f"  OpenAI Base URL: {config.openai_base_url}")
    print(f"  OpenAI Model: {config.openai_model}")
    print(f"  DeepSeek API Key: {'已设置' if config.deepseek_api_key else '未设置'}")
    print(f"  DeepSeek Base URL: {config.deepseek_base_url}")
    print(f"  DeepSeek Model: {config.deepseek_model}")
    print(f"  Local Base URL: {config.local_base_url}")
    print(f"  Local Model: {config.local_model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max Tokens: {config.max_tokens}")
    print(f"  Max History Length: {config.max_history_length}")
    
    # 测试LLM连接
    if configs["任意配置"]:
        print("\n🔗 测试LLM连接...")
        try:
            llm = get_llm()
            print(f"✅ LLM连接成功: {type(llm).__name__}")
            
            # 简单测试
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content="你好")])
            print(f"🤖 测试响应: {response.content[:50]}...")
            
        except Exception as e:
            print(f"❌ LLM连接失败: {e}")
    else:
        print("\n❌ 无可用配置，无法测试LLM连接")
        print("\n💡 配置建议:")
        print("  1. 设置 OPENAI_API_KEY 环境变量")
        print("  2. 或设置 DEEPSEEK_API_KEY 环境变量")
        print("  3. 或配置本地模型 LOCAL_BASE_URL")
        print("  4. 或启动 Ollama 服务")

def run_basic_demo():
    """运行基础记忆演示"""
    print("🚀 运行基础记忆演示")
    print("=" * 40)
    
    try:
        demo = MemoryExamples()
        demo.run_all_demos()
    except Exception as e:
        print(f"❌ 基础演示失败: {e}")
        import traceback
        traceback.print_exc()

def run_customer_service_demo():
    """运行智能客服演示"""
    print("🚀 运行智能客服演示")
    print("=" * 40)
    
    try:
        from smart_customer_service import demo_customer_service
        demo_customer_service()
    except Exception as e:
        print(f"❌ 客服演示失败: {e}")
        import traceback
        traceback.print_exc()

def run_langgraph_demo():
    """运行LangGraph记忆演示"""
    print("🚀 运行LangGraph记忆演示")
    print("=" * 40)
    
    if not LANGGRAPH_AVAILABLE:
        print("⚠️ LangGraph 未正确安装，无法运行演示")
        print("请运行: pip install langgraph")
        return
    
    try:
        from langgraph_memory_example import demo_langgraph_memory
        demo_langgraph_memory()
    except Exception as e:
        print(f"❌ LangGraph演示失败: {e}")
        import traceback
        traceback.print_exc()

def run_interactive_chat():
    """运行交互式聊天"""
    print("💬 交互式聊天模式")
    print("=" * 40)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清除对话历史")
    print("输入 'summary' 查看对话摘要")
    print("=" * 40)
    
    try:
        # 创建客服机器人
        bot = CustomerServiceBot()
        
        # 开始对话
        user_id = "interactive_user"
        session_id, welcome = bot.start_conversation(user_id, "交互用户")
        
        print(f"🤖 {welcome}")
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 您: ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                elif user_input.lower() in ['clear', '清除']:
                    # 重新开始对话
                    session_id, welcome = bot.start_conversation(user_id, "交互用户")
                    print(f"🧹 对话已清除")
                    print(f"🤖 {welcome}")
                    continue
                elif user_input.lower() in ['summary', '摘要']:
                    summary = bot.get_conversation_summary(session_id)
                    print(f"\n{summary}")
                    continue
                
                # 处理正常消息
                result = bot.chat(session_id, user_input)
                
                if "response" in result:
                    print(f"🤖 助手: {result['response']}")
                    print(f"⏱️ 响应时间: {result['response_time']:.2f}秒")
                else:
                    print(f"❌ 错误: {result.get('error', '未知错误')}")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 处理消息时出错: {e}")
        
        # 保存会话
        if bot.session_manager.save_session(session_id):
            print("💾 会话已保存")
        
    except Exception as e:
        print(f"❌ 交互式聊天失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="LangChain 记忆功能演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --demo basic          # 基础记忆演示
  python main.py --demo customer       # 智能客服演示
  python main.py --demo langgraph      # LangGraph记忆演示
  python main.py --demo all            # 运行所有演示
  python main.py --interactive         # 交互式聊天
  python main.py --config-check        # 检查配置
        """
    )
    
    parser.add_argument(
        "--demo",
        choices=["basic", "customer", "langgraph", "all"],
        help="运行指定的演示"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="启动交互式聊天模式"
    )
    
    parser.add_argument(
        "--config-check",
        action="store_true",
        help="检查配置"
    )
    
    args = parser.parse_args()
    
    # 显示标题
    print("🤖 LangChain 记忆功能演示")
    print("=" * 50)
    
    # 处理命令
    if args.config_check:
        check_config()
    elif args.interactive:
        # 先检查配置
        if not config.validate_config():
            print("❌ 配置验证失败！请先配置LLM")
            check_config()
            return
        run_interactive_chat()
    elif args.demo:
        # 先检查配置
        if not config.validate_config():
            print("❌ 配置验证失败！请先配置LLM")
            check_config()
            return
        
        if args.demo == "basic":
            run_basic_demo()
        elif args.demo == "customer":
            run_customer_service_demo()
        elif args.demo == "langgraph":
            run_langgraph_demo()
        elif args.demo == "all":
            print("🚀 运行所有演示")
            print("=" * 50)
            
            print("\n1️⃣ 基础记忆演示")
            run_basic_demo()
            
            print("\n2️⃣ 智能客服演示")
            run_customer_service_demo()
            
            print("\n3️⃣ LangGraph记忆演示")
            run_langgraph_demo()
            
            print("\n✅ 所有演示完成！")
    else:
        # 默认显示帮助
        parser.print_help()
        print("\n💡 建议先运行: python main.py --config-check")

if __name__ == "__main__":
    main()