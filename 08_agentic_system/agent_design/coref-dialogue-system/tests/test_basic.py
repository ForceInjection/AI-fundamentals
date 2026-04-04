#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import pytest
from example_usage import IntegratedDialogueSystem

@pytest.mark.asyncio
async def test_basic_functionality():
    """测试基本功能"""
    print("开始测试基本功能...")
    
    try:
        # 创建系统实例
        system = IntegratedDialogueSystem()
        print("✅ 系统初始化成功")
        
        # 测试基本对话处理
        result = await system.process_user_input('你好，我是张三。')
        print(f"✅ 基本对话处理测试成功: {result[:50] if result else '无响应'}")
        
        # 测试指代消解
        result2 = await system.process_user_input('他今天来公司了吗？')
        print(f"✅ 指代消解测试成功: {result2[:50] if result2 else '无响应'}")
        
        # 获取系统统计
        stats = system.get_system_stats()
        print(f"✅ 系统统计获取成功: 处理了 {stats['conversation_turns']} 轮对话")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

def main():
    """主函数，供run_tests.py调用"""
    success = asyncio.run(test_basic_functionality())
    if success:
        print("\n🎉 所有基本功能测试通过！")
    else:
        print("\n💥 测试失败，请检查错误信息")
    return success

if __name__ == "__main__":
    main()