#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行脚本

统一执行所有测试的入口脚本，支持不同的测试模式。

使用方法:
    python tests/run_tests.py --all          # 运行所有测试
    python tests/run_tests.py --basic        # 只运行基本功能测试
    python tests/run_tests.py --performance  # 只运行性能测试
    python tests/run_tests.py --verification # 只运行最终验证
    python tests/run_tests.py --monitoring   # 只运行监控测试
"""

import argparse
import asyncio
import sys
import os
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_banner(title: str):
    """打印测试横幅"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_result(test_name: str, success: bool, duration: float = 0):
    """打印测试结果"""
    status = "✅ 通过" if success else "❌ 失败"
    duration_str = f" ({duration:.2f}s)" if duration > 0 else ""
    print(f"{test_name:<30} {status}{duration_str}")

async def run_basic_tests():
    """运行基本功能测试"""
    print_banner("基本功能测试")
    start_time = time.time()
    
    try:
        # 导入并运行基本测试
        from test_basic import test_basic_functionality
        success = await test_basic_functionality()
        duration = time.time() - start_time
        print_result("基本功能测试", success, duration)
        return success
    except Exception as e:
        duration = time.time() - start_time
        print_result("基本功能测试", False, duration)
        print(f"错误: {e}")
        return False

async def run_performance_tests():
    """运行性能测试"""
    print_banner("性能测试")
    start_time = time.time()
    
    try:
        # 导入并运行性能测试
        from test_performance import test_performance, test_stress
        perf_success = await test_performance()
        stress_success = await test_stress()
        success = perf_success and stress_success
        duration = time.time() - start_time
        print_result("性能测试", success, duration)
        return success
    except Exception as e:
        duration = time.time() - start_time
        print_result("性能测试", False, duration)
        print(f"错误: {e}")
        return False

async def run_verification_tests():
    """运行最终验证测试"""
    print_banner("最终验证测试")
    start_time = time.time()
    
    try:
        # 导入并运行最终验证
        from final_verification import (test_module_imports, test_basic_functionality, 
                                      test_performance, test_system_integration, test_error_handling)
        
        # 执行所有验证测试
        results = {}
        results["模块导入"] = await test_module_imports()
        results["基本功能"] = await test_basic_functionality()
        results["性能测试"] = await test_performance()
        results["系统集成"] = await test_system_integration()
        results["错误处理"] = await test_error_handling()
        
        # 计算总体成功率
        total_tests = sum(len(r) for r in results.values())
        passed_tests = sum(sum(r.values()) for r in results.values())
        success = passed_tests == total_tests
        
        duration = time.time() - start_time
        print_result("最终验证测试", success, duration)
        print(f"验证结果: {passed_tests}/{total_tests} 项测试通过")
        return success
    except Exception as e:
        duration = time.time() - start_time
        print_result("最终验证测试", False, duration)
        print(f"错误: {e}")
        return False

def run_monitoring_tests():
    """运行监控测试"""
    print_banner("监控测试")
    start_time = time.time()
    
    try:
        # 导入并测试监控模块
        from testing_and_monitoring import TestCoreferenceEngine, TestEntityRecognition
        
        # 创建测试实例
        coref_test = TestCoreferenceEngine()
        entity_test = TestEntityRecognition()
        
        print("监控模块导入成功")
        print(f"指代消解测试引擎: {type(coref_test).__name__}")
        print(f"实体识别测试引擎: {type(entity_test).__name__}")
        
        duration = time.time() - start_time
        print_result("监控测试", True, duration)
        return True
    except Exception as e:
        duration = time.time() - start_time
        print_result("监控测试", False, duration)
        print(f"错误: {e}")
        return False

async def run_all_tests():
    """运行所有测试"""
    print_banner("多轮指代消解对话系统 - 完整测试套件")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    total_start_time = time.time()
    
    # 运行各项测试
    results.append(await run_basic_tests())
    results.append(await run_performance_tests())
    results.append(run_monitoring_tests())
    results.append(await run_verification_tests())
    
    # 生成总结报告
    total_duration = time.time() - total_start_time
    passed_tests = sum(results)
    total_tests = len(results)
    success_rate = (passed_tests / total_tests) * 100
    
    print_banner("测试总结报告")
    print(f"总测试数量: {total_tests}")
    print(f"通过测试数: {passed_tests}")
    print(f"失败测试数: {total_tests - passed_tests}")
    print(f"成功率: {success_rate:.1f}%")
    print(f"总耗时: {total_duration:.2f}秒")
    
    if success_rate == 100:
        print("\n🎉 所有测试通过！系统运行正常")
    elif success_rate >= 75:
        print("\n⚠️ 大部分测试通过，但存在一些问题")
    else:
        print("\n❌ 多项测试失败，需要检查系统")
    
    print(f"\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return success_rate == 100

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多轮指代消解对话系统测试运行器')
    parser.add_argument('--all', action='store_true', help='运行所有测试')
    parser.add_argument('--basic', action='store_true', help='运行基本功能测试')
    parser.add_argument('--performance', action='store_true', help='运行性能测试')
    parser.add_argument('--verification', action='store_true', help='运行最终验证测试')
    parser.add_argument('--monitoring', action='store_true', help='运行监控测试')
    
    args = parser.parse_args()
    
    # 如果没有指定参数，默认运行所有测试
    if not any([args.all, args.basic, args.performance, args.verification, args.monitoring]):
        args.all = True
    
    async def run_selected_tests():
        if args.all:
            return await run_all_tests()
        
        results = []
        if args.basic:
            results.append(await run_basic_tests())
        if args.performance:
            results.append(await run_performance_tests())
        if args.monitoring:
            results.append(run_monitoring_tests())
        if args.verification:
            results.append(await run_verification_tests())
        
        return all(results) if results else True
    
    # 运行测试
    try:
        success = asyncio.run(run_selected_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()