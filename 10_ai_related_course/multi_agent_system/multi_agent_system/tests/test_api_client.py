#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 客户端测试脚本

本脚本用于验证 README.md 文档中描述的 API 使用示例：
1. 创建研究任务
2. 查询任务状态
3. 数据分析
4. 智能客服对话
5. 系统监控
"""

import asyncio
import json
import sys
from typing import Dict, Any
import time

try:
    import httpx
except ImportError:
    print("❌ 请安装 httpx: pip install httpx")
    sys.exit(1)


class APITester:
    """API 测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
    
    async def run_all_tests(self):
        """运行所有 API 测试"""
        print("🌐 开始 API 接口测试")
        print("=" * 50)
        
        try:
            # 等待服务器完全启动
            await self.wait_for_server()
            
            # 测试 1: 健康检查
            await self.test_health_check()
            
            # 测试 2: 创建研究任务
            task_id = await self.test_create_research_task()
            
            # 测试 3: 查询任务状态
            if task_id:
                await self.test_query_task_status(task_id)
            
            # 测试 4: 数据分析
            await self.test_data_analysis()
            
            # 测试 5: 智能客服对话
            await self.test_customer_service()
            
            # 测试 6: 系统监控
            await self.test_system_metrics()
            
        except Exception as e:
            print(f"❌ 测试过程中发生错误：{e}")
        finally:
            await self.client.aclose()
        
        # 输出测试结果
        self.print_test_summary()
    
    async def wait_for_server(self, max_attempts: int = 10):
        """等待服务器启动"""
        print("⏳ 等待服务器启动...")
        
        for attempt in range(max_attempts):
            try:
                response = await self.client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    print("✅ 服务器已就绪")
                    return
            except Exception:
                pass
            
            if attempt < max_attempts - 1:
                await asyncio.sleep(1)
        
        raise Exception("服务器启动超时")
    
    async def test_health_check(self):
        """测试健康检查"""
        print("\n🏥 测试 1: 健康检查")
        print("-" * 30)
        
        try:
            response = await self.client.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 健康检查成功")
                print(f"📊 状态: {data.get('status')}")
                print(f"🕐 时间: {data.get('timestamp')}")
                print(f"📦 版本: {data.get('version')}")
                self.test_results.append(("健康检查", True, "服务器运行正常"))
            else:
                print(f"❌ 健康检查失败，状态码: {response.status_code}")
                self.test_results.append(("健康检查", False, f"状态码: {response.status_code}"))
                
        except Exception as e:
            print(f"❌ 健康检查异常：{e}")
            self.test_results.append(("健康检查", False, str(e)))
    
    async def test_create_research_task(self) -> str:
        """测试创建研究任务（对应 README.md 示例）"""
        print("\n🔬 测试 2: 创建研究任务")
        print("-" * 30)
        
        try:
            # 对应 README.md 中的 curl 示例
            task_data = {
                "query": "分析人工智能在医疗领域的应用前景",
                "priority": "high",
                "agent_type": "research"
            }
            
            response = await self.client.post(
                f"{self.base_url}/tasks",
                json=task_data
            )
            
            if response.status_code == 201:
                data = response.json()
                task_id = data.get('task_id')
                print(f"✅ 研究任务创建成功")
                print(f"📝 任务ID: {task_id}")
                print(f"🎯 查询: {data.get('query')}")
                print(f"⚡ 优先级: {data.get('priority')}")
                print(f"🤖 智能体类型: {data.get('agent_type')}")
                print(f"⏰ 预计完成时间: {data.get('estimated_completion')}")
                self.test_results.append(("创建研究任务", True, f"任务ID: {task_id}"))
                return task_id
            else:
                print(f"❌ 任务创建失败，状态码: {response.status_code}")
                print(f"📄 响应内容: {response.text}")
                self.test_results.append(("创建研究任务", False, f"状态码: {response.status_code}"))
                return None
                
        except Exception as e:
            print(f"❌ 任务创建异常：{e}")
            self.test_results.append(("创建研究任务", False, str(e)))
            return None
    
    async def test_query_task_status(self, task_id: str):
        """测试查询任务状态"""
        print("\n📊 测试 3: 查询任务状态")
        print("-" * 30)
        
        try:
            response = await self.client.get(f"{self.base_url}/tasks/{task_id}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 任务状态查询成功")
                print(f"📝 任务ID: {data.get('task_id')}")
                print(f"📈 状态: {data.get('status')}")
                print(f"🎯 进度: {data.get('progress')}%")
                
                result = data.get('result', {})
                if result:
                    print(f"📋 摘要: {result.get('summary')}")
                    print(f"🔍 发现: {result.get('findings')}")
                    print(f"🎯 置信度: {result.get('confidence')}")
                
                self.test_results.append(("查询任务状态", True, f"状态: {data.get('status')}"))
            else:
                print(f"❌ 任务状态查询失败，状态码: {response.status_code}")
                self.test_results.append(("查询任务状态", False, f"状态码: {response.status_code}"))
                
        except Exception as e:
            print(f"❌ 任务状态查询异常：{e}")
            self.test_results.append(("查询任务状态", False, str(e)))
    
    async def test_data_analysis(self):
        """测试数据分析（对应 README.md 示例）"""
        print("\n📈 测试 4: 数据分析")
        print("-" * 30)
        
        try:
            # 对应 README.md 中的 curl 示例
            analysis_data = {
                "data_source": "sales_data_2024.csv",
                "analysis_type": "trend_analysis",
                "parameters": {
                    "time_range": "2024-01-01 to 2024-12-31",
                    "metrics": ["revenue", "growth_rate", "customer_acquisition"]
                }
            }
            
            response = await self.client.post(
                f"{self.base_url}/analysis",
                json=analysis_data
            )
            
            if response.status_code == 201:
                data = response.json()
                print(f"✅ 数据分析任务创建成功")
                print(f"📊 分析ID: {data.get('analysis_id')}")
                print(f"📁 数据源: {data.get('data_source')}")
                print(f"🔍 分析类型: {data.get('analysis_type')}")
                print(f"⚙️ 参数: {json.dumps(data.get('parameters', {}), ensure_ascii=False, indent=2)}")
                print(f"⏰ 预计完成时间: {data.get('estimated_completion')}")
                self.test_results.append(("数据分析", True, f"分析ID: {data.get('analysis_id')}"))
            else:
                print(f"❌ 数据分析失败，状态码: {response.status_code}")
                self.test_results.append(("数据分析", False, f"状态码: {response.status_code}"))
                
        except Exception as e:
            print(f"❌ 数据分析异常：{e}")
            self.test_results.append(("数据分析", False, str(e)))
    
    async def test_customer_service(self):
        """测试智能客服对话（对应 README.md 示例）"""
        print("\n💬 测试 5: 智能客服对话")
        print("-" * 30)
        
        try:
            # 对应 README.md 中的 curl 示例
            chat_data = {
                "message": "我想了解你们的产品价格和服务内容",
                "session_id": "session_123",
                "customer_id": "customer_456"
            }
            
            response = await self.client.post(
                f"{self.base_url}/chat",
                json=chat_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 智能客服对话成功")
                print(f"🆔 会话ID: {data.get('session_id')}")
                print(f"👤 客户消息: {data.get('customer_message')}")
                print(f"🤖 机器人回复: {data.get('bot_response')}")
                print(f"🎯 置信度: {data.get('confidence')}")
                print(f"🔍 意图识别: {data.get('intent')}")
                print(f"💡 建议操作: {data.get('suggested_actions')}")
                self.test_results.append(("智能客服对话", True, "对话成功"))
            else:
                print(f"❌ 智能客服对话失败，状态码: {response.status_code}")
                self.test_results.append(("智能客服对话", False, f"状态码: {response.status_code}"))
                
        except Exception as e:
            print(f"❌ 智能客服对话异常：{e}")
            self.test_results.append(("智能客服对话", False, str(e)))
    
    async def test_system_metrics(self):
        """测试系统监控（对应 README.md 示例）"""
        print("\n📊 测试 6: 系统监控")
        print("-" * 30)
        
        try:
            response = await self.client.get(f"{self.base_url}/metrics")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 系统指标获取成功")
                print(f"🕐 时间戳: {data.get('timestamp')}")
                print(f"💚 系统状态: {data.get('system_status')}")
                
                agents = data.get('agents', {})
                print(f"🤖 智能体总数: {agents.get('total_count')}")
                print(f"⚡ 活跃智能体: {agents.get('active_count')}")
                
                resources = data.get('resources', {})
                print(f"💾 内存使用: {resources.get('memory_usage_mb')}MB")
                print(f"🖥️ CPU 使用率: {resources.get('cpu_usage_percent')}%")
                print(f"💿 磁盘使用率: {resources.get('disk_usage_percent')}%")
                
                performance = data.get('performance', {})
                print(f"📈 每分钟请求数: {performance.get('requests_per_minute')}")
                print(f"⏱️ 平均响应时间: {performance.get('average_response_time_ms')}ms")
                print(f"❌ 错误率: {performance.get('error_rate_percent')}%")
                
                self.test_results.append(("系统监控", True, "指标获取成功"))
            else:
                print(f"❌ 系统指标获取失败，状态码: {response.status_code}")
                self.test_results.append(("系统监控", False, f"状态码: {response.status_code}"))
                
        except Exception as e:
            print(f"❌ 系统指标获取异常：{e}")
            self.test_results.append(("系统监控", False, str(e)))
    
    def print_test_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 50)
        print("📋 API 测试结果总结")
        print("=" * 50)
        
        passed = 0
        failed = 0
        
        for test_name, success, message in self.test_results:
            status = "✅ 通过" if success else "❌ 失败"
            print(f"{status} {test_name}: {message}")
            if success:
                passed += 1
            else:
                failed += 1
        
        print(f"\n📊 总计: {passed + failed} 个测试")
        print(f"✅ 通过: {passed} 个")
        print(f"❌ 失败: {failed} 个")
        
        if failed == 0:
            print("\n🎉 所有 API 测试通过！接口运行正常。")
            print("\n📖 您可以访问以下地址查看 API 文档：")
            print(f"   • Swagger UI: http://localhost:8000/docs")
            print(f"   • ReDoc: http://localhost:8000/redoc")
        else:
            print(f"\n⚠️ 有 {failed} 个测试失败，请检查相关接口。")


async def main():
    """主函数"""
    print("🌐 多智能体系统 API 接口验证")
    print(f"🎯 测试目标: http://localhost:8000")
    
    tester = APITester()
    await tester.run_all_tests()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n💥 程序异常退出：{e}")
        sys.exit(1)