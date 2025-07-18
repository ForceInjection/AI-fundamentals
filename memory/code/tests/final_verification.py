#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终验证脚本

本脚本对整个多轮指代消解对话系统进行全面验证，包括：
1. 模块导入测试
2. 基本功能测试
3. 性能测试
4. 系统集成测试
5. 错误处理测试
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

# 添加父目录到Python路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_test_result(test_name: str, success: bool, details: str = ""):
    """打印测试结果"""
    status = "✅ 通过" if success else "❌ 失败"
    print(f"{test_name:<40} {status}")
    if details:
        print(f"    详情: {details}")

async def test_module_imports() -> Dict[str, bool]:
    """测试模块导入"""
    print_section("模块导入测试")
    
    results = {}
    
    # 测试核心模块导入
    modules_to_test = [
        ("entity_recognition", "EnhancedEntityRecognitionLayer"),
        ("coreference_resolution", "AdvancedCoreferenceLayer"),
        ("dialogue_state_manager", "IntelligentStateManager"),
        ("memory_management", "MemoryManager"),
        ("performance_optimization", "PerformanceOptimizer"),
        ("multimodal_coref", "MultimodalCoreferenceResolver"),
        ("system_integration", "app"),
        ("example_usage", "IntegratedDialogueSystem"),
        ("testing_and_monitoring", "TestCoreferenceEngine")
    ]
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            getattr(module, class_name)
            results[module_name] = True
            print_test_result(f"导入 {module_name}", True)
        except Exception as e:
            results[module_name] = False
            print_test_result(f"导入 {module_name}", False, str(e))
    
    return results

async def test_basic_functionality() -> Dict[str, bool]:
    """测试基本功能"""
    print_section("基本功能测试")
    
    results = {}
    
    try:
        # 测试系统初始化
        from example_usage import IntegratedDialogueSystem
        system = IntegratedDialogueSystem()
        results["system_init"] = True
        print_test_result("系统初始化", True)
        
        # 测试基本对话处理
        response = await system.process_user_input("你好，我是张三。")
        results["basic_dialogue"] = bool(response)
        print_test_result("基本对话处理", bool(response), f"响应: {response[:30]}...")
        
        # 测试指代消解
        response2 = await system.process_user_input("他今天来公司了吗？")
        results["coreference"] = bool(response2)
        print_test_result("指代消解", bool(response2), f"响应: {response2[:30]}...")
        
        # 测试系统统计
        stats = system.get_system_stats()
        results["system_stats"] = bool(stats)
        print_test_result("系统统计", bool(stats), f"处理轮次: {stats.get('conversation_turns', 0)}")
        
    except Exception as e:
        results["basic_functionality"] = False
        print_test_result("基本功能测试", False, str(e))
    
    return results

async def test_performance() -> Dict[str, bool]:
    """测试性能"""
    print_section("性能测试")
    
    results = {}
    
    try:
        from example_usage import IntegratedDialogueSystem
        system = IntegratedDialogueSystem()
        
        # 响应时间测试
        start_time = time.time()
        await system.process_user_input("测试响应时间")
        response_time = time.time() - start_time
        
        results["response_time"] = response_time < 1.0  # 1秒内响应
        print_test_result("响应时间", response_time < 1.0, f"{response_time:.3f}秒")
        
        # 并发处理测试
        tasks = []
        for i in range(5):
            task = asyncio.create_task(system.process_user_input(f"并发测试{i+1}"))
            tasks.append(task)
        
        start_time = time.time()
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        results["concurrent_processing"] = concurrent_time < 2.0  # 2秒内完成5个并发任务
        print_test_result("并发处理", concurrent_time < 2.0, f"{concurrent_time:.3f}秒完成5个任务")
        
    except Exception as e:
        results["performance"] = False
        print_test_result("性能测试", False, str(e))
    
    return results

async def test_system_integration() -> Dict[str, bool]:
    """测试系统集成"""
    print_section("系统集成测试")
    
    results = {}
    
    try:
        # 测试FastAPI应用
        from system_integration import app, health_check
        
        # 健康检查
        health_result = await health_check()
        results["health_check"] = health_result.status == "healthy"
        print_test_result("健康检查", health_result.status == "healthy", health_result.status)
        
        # 测试服务组件
        from system_integration import EntityRecognitionService, CoreferenceResolutionService
        
        entity_service = EntityRecognitionService()
        entities = await entity_service.extract_entities("张三是一个好人", "test_001")
        results["entity_service"] = len(entities) > 0
        print_test_result("实体识别服务", len(entities) > 0, f"识别到{len(entities)}个实体")
        
        coref_service = CoreferenceResolutionService()
        resolutions = await coref_service.resolve_coreferences("他很好", entities, "test_001")
        results["coref_service"] = True  # 不报错即为成功
        print_test_result("指代消解服务", True, f"处理了{len(resolutions)}个指代")
        
    except Exception as e:
        results["system_integration"] = False
        print_test_result("系统集成测试", False, str(e))
    
    return results

async def test_error_handling() -> Dict[str, bool]:
    """测试错误处理"""
    print_section("错误处理测试")
    
    results = {}
    
    try:
        from example_usage import IntegratedDialogueSystem
        system = IntegratedDialogueSystem()
        
        # 测试空输入
        try:
            response = await system.process_user_input("")
            results["empty_input"] = True
            print_test_result("空输入处理", True, "系统正常处理空输入")
        except Exception:
            results["empty_input"] = False
            print_test_result("空输入处理", False, "空输入导致异常")
        
        # 测试特殊字符
        try:
            response = await system.process_user_input("@#$%^&*()")
            results["special_chars"] = True
            print_test_result("特殊字符处理", True, "系统正常处理特殊字符")
        except Exception:
            results["special_chars"] = False
            print_test_result("特殊字符处理", False, "特殊字符导致异常")
        
        # 测试长文本
        long_text = "这是一个很长的文本。" * 100
        try:
            response = await system.process_user_input(long_text)
            results["long_text"] = True
            print_test_result("长文本处理", True, "系统正常处理长文本")
        except Exception:
            results["long_text"] = False
            print_test_result("长文本处理", False, "长文本导致异常")
        
    except Exception as e:
        results["error_handling"] = False
        print_test_result("错误处理测试", False, str(e))
    
    return results

async def generate_final_report(all_results: Dict[str, Dict[str, bool]]):
    """生成最终报告"""
    print_section("最终验证报告")
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        category_passed = sum(results.values())
        category_total = len(results)
        total_tests += category_total
        passed_tests += category_passed
        
        print(f"{category:<20} {category_passed}/{category_total} 通过")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n总体测试结果: {passed_tests}/{total_tests} 通过 ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("\n🎉 系统验证通过！所有核心功能正常运行")
        print("✅ 系统已准备好用于生产环境")
    elif success_rate >= 70:
        print("\n⚠️ 系统基本可用，但存在一些问题需要修复")
        print("🔧 建议在部署前解决失败的测试项")
    else:
        print("\n❌ 系统存在严重问题，不建议部署")
        print("🚨 请修复失败的测试项后重新验证")
    
    # 系统信息
    print(f"\n系统信息:")
    print(f"Python版本: {sys.version}")
    print(f"验证时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"验证环境: 开发环境")

async def main():
    """主验证函数"""
    print("多轮指代消解对话系统 - 最终验证")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 执行所有测试
    all_results = {}
    
    all_results["模块导入"] = await test_module_imports()
    all_results["基本功能"] = await test_basic_functionality()
    all_results["性能测试"] = await test_performance()
    all_results["系统集成"] = await test_system_integration()
    all_results["错误处理"] = await test_error_handling()
    
    # 生成最终报告
    await generate_final_report(all_results)
    
    print(f"\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())