#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import time
import statistics
import pytest
from example_usage import IntegratedDialogueSystem

@pytest.mark.asyncio
async def test_performance():
    """性能测试"""
    print("开始性能测试...")
    
    # 创建系统实例
    system = IntegratedDialogueSystem()
    
    # 测试用例
    test_cases = [
        "你好，我是张三。",
        "他今天来公司了吗？",
        "他的工作表现怎么样？",
        "李四和他是同事吗？",
        "她也在同一个部门工作吗？",
        "这个项目的负责人是谁？",
        "他们什么时候开始的？",
        "公司的业绩如何？",
        "它在市场上的表现怎样？",
        "我想了解更多信息。"
    ]
    
    processing_times = []
    
    print(f"\n测试 {len(test_cases)} 个对话轮次...")
    
    for i, test_input in enumerate(test_cases, 1):
        start_time = time.time()
        
        try:
            result = await system.process_user_input(test_input)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            print(f"轮次 {i}: {processing_time:.3f}秒 - {test_input[:20]}...")
            
        except Exception as e:
            print(f"轮次 {i} 失败: {str(e)}")
    
    # 性能统计
    if processing_times:
        avg_time = statistics.mean(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        median_time = statistics.median(processing_times)
        
        print(f"\n=== 性能统计 ===")
        print(f"平均处理时间: {avg_time:.3f}秒")
        print(f"最快处理时间: {min_time:.3f}秒")
        print(f"最慢处理时间: {max_time:.3f}秒")
        print(f"中位数处理时间: {median_time:.3f}秒")
        print(f"总处理轮次: {len(processing_times)}")
        
        # 系统统计
        stats = system.get_system_stats()
        print(f"\n=== 系统统计 ===")
        print(f"对话轮次: {stats['conversation_turns']}")
        print(f"实体总数: {stats['memory']['total_entities']}")
        print(f"缓存命中: {stats['performance'].get('cache_hits', 0)}")
        print(f"缓存未命中: {stats['performance'].get('cache_misses', 0)}")
        print(f"内存使用: {stats['memory']['memory_usage_mb']:.2f} MB")
        
        # 性能评估
        if avg_time < 0.1:
            print("\n🚀 性能优秀！平均响应时间小于100ms")
        elif avg_time < 0.5:
            print("\n✅ 性能良好！平均响应时间小于500ms")
        elif avg_time < 1.0:
            print("\n⚠️ 性能一般，平均响应时间小于1秒")
        else:
            print("\n❌ 性能需要优化，平均响应时间超过1秒")
        
        return True
    else:
        print("\n❌ 没有成功的测试用例")
        return False

@pytest.mark.asyncio
async def test_stress():
    """压力测试"""
    print("\n开始压力测试...")
    
    system = IntegratedDialogueSystem()
    
    # 并发测试
    concurrent_tasks = []
    test_input = "他今天来公司了吗？"
    
    start_time = time.time()
    
    # 创建10个并发任务
    for i in range(10):
        task = asyncio.create_task(system.process_user_input(f"{test_input} (任务{i+1})"))
        concurrent_tasks.append(task)
    
    # 等待所有任务完成
    results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    successful_tasks = sum(1 for r in results if not isinstance(r, Exception))
    failed_tasks = len(results) - successful_tasks
    
    print(f"\n=== 压力测试结果 ===")
    print(f"并发任务数: {len(concurrent_tasks)}")
    print(f"成功任务数: {successful_tasks}")
    print(f"失败任务数: {failed_tasks}")
    print(f"总耗时: {total_time:.3f}秒")
    print(f"平均每任务耗时: {total_time/len(concurrent_tasks):.3f}秒")
    
    if failed_tasks == 0:
        print("🎉 压力测试通过！所有并发任务都成功完成")
        return True
    else:
        print(f"⚠️ 压力测试部分失败，{failed_tasks}个任务失败")
        return False

async def main():
    """主测试函数"""
    print("=== 多轮指代消解对话系统性能测试 ===")
    
    # 基本性能测试
    perf_success = await test_performance()
    
    # 压力测试
    stress_success = await test_stress()
    
    # 总结
    print("\n=== 测试总结 ===")
    if perf_success and stress_success:
        print("🎉 所有测试通过！系统运行正常")
    elif perf_success:
        print("✅ 基本功能正常，但压力测试有问题")
    elif stress_success:
        print("✅ 并发处理正常，但基本性能有问题")
    else:
        print("❌ 测试失败，系统需要检查")

if __name__ == "__main__":
    asyncio.run(main())