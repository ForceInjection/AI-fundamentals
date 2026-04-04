#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮指代消解对话系统测试包

本包包含了系统的所有测试模块：
- test_basic.py: 基本功能测试
- test_performance.py: 性能测试
- testing_and_monitoring.py: 测试框架和监控
- final_verification.py: 最终验证脚本

使用方法:
    # 运行所有测试
    python -m pytest tests/
    
    # 运行特定测试
    python tests/test_basic.py
    python tests/test_performance.py
    
    # 运行最终验证
    python tests/final_verification.py
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

# 导入测试模块
try:
    from .test_basic import *
    from .test_performance import *
    from .testing_and_monitoring import *
    from .final_verification import *
except ImportError:
    # 如果导入失败，可能是因为依赖问题，忽略错误
    pass

__all__ = [
    'test_basic',
    'test_performance', 
    'testing_and_monitoring',
    'final_verification'
]