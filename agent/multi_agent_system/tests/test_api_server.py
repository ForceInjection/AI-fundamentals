#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 服务器测试脚本

本脚本用于验证 README.md 文档中描述的 API 接口：
1. POST /tasks - 创建研究任务
2. POST /analysis - 数据分析
3. POST /chat - 智能客服对话
4. GET /metrics - 系统监控
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    from main import MultiAgentSystem
except ImportError as e:
    print(f"❌ 导入错误：{e}")
    print("请确保已安装所有依赖包")
    sys.exit(1)


# 请求模型
class TaskRequest(BaseModel):
    """研究任务请求"""
    query: str
    priority: str = "medium"
    agent_type: str = "research"


class AnalysisRequest(BaseModel):
    """数据分析请求"""
    data_source: str
    analysis_type: str = "statistical"
    parameters: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    """智能客服对话请求"""
    message: str
    session_id: Optional[str] = None
    customer_id: Optional[str] = None


# 创建 FastAPI 应用
app = FastAPI(
    title="多智能体系统 API",
    description="企业级多智能体AI系统的RESTful API接口",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 全局系统实例
system: Optional[MultiAgentSystem] = None


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    global system
    try:
        print("🚀 启动多智能体系统...")
        system = MultiAgentSystem()
        await system.start()
        print("✅ 多智能体系统启动成功")
    except Exception as e:
        print(f"❌ 系统启动失败：{e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    global system
    if system:
        try:
            print("🔄 关闭多智能体系统...")
            await system.shutdown()
            print("✅ 多智能体系统已关闭")
        except Exception as e:
            print(f"⚠️ 系统关闭时出现警告：{e}")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "欢迎使用多智能体系统 API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running" if system else "initializing"
    }


@app.post("/tasks")
async def create_task(request: TaskRequest):
    """创建研究任务（对应 README.md API 示例）"""
    try:
        if not system:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        # 模拟任务创建
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 这里可以调用实际的智能体处理逻辑
        # 为了演示，我们返回模拟响应
        response = {
            "task_id": task_id,
            "status": "created",
            "query": request.query,
            "priority": request.priority,
            "agent_type": request.agent_type,
            "created_at": datetime.now().isoformat(),
            "estimated_completion": "2-5 minutes"
        }
        
        print(f"📝 创建任务：{task_id} - {request.query}")
        return JSONResponse(content=response, status_code=201)
        
    except Exception as e:
        print(f"❌ 任务创建失败：{e}")
        raise HTTPException(status_code=500, detail=f"任务创建失败：{str(e)}")


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    try:
        if not system:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        # 模拟任务状态查询
        response = {
            "task_id": task_id,
            "status": "completed",
            "progress": 100,
            "result": {
                "summary": "任务已完成",
                "findings": ["发现1", "发现2", "发现3"],
                "confidence": 0.95
            },
            "completed_at": datetime.now().isoformat()
        }
        
        print(f"📊 查询任务状态：{task_id}")
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"❌ 任务状态查询失败：{e}")
        raise HTTPException(status_code=500, detail=f"任务状态查询失败：{str(e)}")


@app.post("/analysis")
async def create_analysis(request: AnalysisRequest):
    """数据分析（对应 README.md API 示例）"""
    try:
        if not system:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        # 模拟数据分析
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        response = {
            "analysis_id": analysis_id,
            "status": "processing",
            "data_source": request.data_source,
            "analysis_type": request.analysis_type,
            "parameters": request.parameters,
            "created_at": datetime.now().isoformat(),
            "estimated_completion": "3-10 minutes"
        }
        
        print(f"📈 创建分析任务：{analysis_id} - {request.data_source}")
        return JSONResponse(content=response, status_code=201)
        
    except Exception as e:
        print(f"❌ 分析任务创建失败：{e}")
        raise HTTPException(status_code=500, detail=f"分析任务创建失败：{str(e)}")


@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """智能客服对话（对应 README.md API 示例）"""
    try:
        if not system:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        # 模拟智能客服响应
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 简单的响应逻辑
        if "帮助" in request.message or "help" in request.message.lower():
            bot_response = "我是智能客服助手，可以帮助您解决各种问题。请告诉我您需要什么帮助？"
        elif "价格" in request.message or "费用" in request.message:
            bot_response = "关于价格信息，我需要了解您的具体需求。请提供更多详细信息。"
        else:
            bot_response = f"我理解您说的是：{request.message}。让我为您查找相关信息..."
        
        response = {
            "session_id": session_id,
            "customer_message": request.message,
            "bot_response": bot_response,
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.88,
            "intent": "general_inquiry",
            "suggested_actions": ["继续对话", "转人工客服"]
        }
        
        print(f"💬 客服对话：{session_id} - {request.message[:50]}...")
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"❌ 客服对话失败：{e}")
        raise HTTPException(status_code=500, detail=f"客服对话失败：{str(e)}")


@app.get("/metrics")
async def get_system_metrics():
    """获取系统指标（对应 README.md API 示例）"""
    try:
        if not system:
            raise HTTPException(status_code=503, detail="系统尚未初始化")
        
        # 获取实际系统指标
        if hasattr(system, '_collect_system_metrics'):
            metrics = await system._collect_system_metrics()
        else:
            # 模拟系统指标
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system_status": "healthy",
                "agents": {
                    "total_count": 3,
                    "active_count": 2,
                    "performance": {
                        "research_agent": {"status": "active", "load": 0.3},
                        "analysis_agent": {"status": "active", "load": 0.5},
                        "customer_service_agent": {"status": "idle", "load": 0.1}
                    }
                },
                "resources": {
                    "memory_usage_mb": 256,
                    "cpu_usage_percent": 15.5,
                    "disk_usage_percent": 45.2
                },
                "performance": {
                    "requests_per_minute": 12,
                    "average_response_time_ms": 850,
                    "error_rate_percent": 0.5
                }
            }
        
        print("📊 获取系统指标")
        return JSONResponse(content=metrics)
        
    except Exception as e:
        print(f"❌ 系统指标获取失败：{e}")
        raise HTTPException(status_code=500, detail=f"系统指标获取失败：{str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if system else "initializing",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    print("🌐 启动 FastAPI 服务器...")
    print("📖 API 文档地址：http://localhost:8000/docs")
    print("🔍 ReDoc 文档地址：http://localhost:8000/redoc")
    print("⏹️ 按 Ctrl+C 停止服务器")
    
    try:
        uvicorn.run(
            "test_api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n⏹️ 服务器已停止")
    except Exception as e:
        print(f"\n💥 服务器启动失败：{e}")
        sys.exit(1)