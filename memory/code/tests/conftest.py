#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest配置文件

为pytest测试框架提供配置和fixture。
"""

import pytest
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环用于异步测试"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def dialogue_system():
    """创建对话系统实例用于测试"""
    try:
        from example_usage import IntegratedDialogueSystem
        return IntegratedDialogueSystem()
    except ImportError:
        pytest.skip("无法导入IntegratedDialogueSystem")

@pytest.fixture(scope="function")
def sample_texts():
    """提供测试用的样本文本"""
    return [
        "你好，我是张三。",
        "他今天来公司了吗？",
        "李四也在这里工作。",
        "她的工作表现很好。",
        "公司的业务发展不错。",
        "这个项目很重要。"
    ]

@pytest.fixture(scope="function")
def sample_entities():
    """提供测试用的样本实体"""
    return [
        {"text": "张三", "type": "PERSON", "confidence": 0.9},
        {"text": "李四", "type": "PERSON", "confidence": 0.9},
        {"text": "公司", "type": "ORG", "confidence": 0.85},
        {"text": "项目", "type": "MISC", "confidence": 0.8}
    ]

# pytest配置
pytest_plugins = []

# 测试收集配置
collect_ignore = [
    "__pycache__",
    "*.pyc"
]

# 标记配置
markers = [
    "basic: 基本功能测试",
    "performance: 性能测试", 
    "integration: 集成测试",
    "slow: 慢速测试",
    "async: 异步测试"
]