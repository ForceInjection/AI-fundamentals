#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置示例文件

使用方法：
1. 复制此文件为 config.py
2. 修改下面的配置项
3. 或者设置对应的环境变量
"""

import os

# ================================
# OpenAI 配置
# ================================

# OpenAI API Key（必填）
# 获取地址：https://platform.openai.com/api-keys
OPENAI_API_KEY = "sk-your-openai-api-key-here"

# OpenAI API Base URL（可选，默认为官方地址）
# 如果使用代理或第三方服务，可以修改此地址
OPENAI_BASE_URL = "https://api.openai.com/v1"

# OpenAI 模型名称（可选）
OPENAI_MODEL = "gpt-3.5-turbo"

# ================================
# 本地模型配置（可选）
# ================================

# 本地模型 API Key（如果本地模型需要认证）
LOCAL_API_KEY = "your-local-api-key"

# 本地模型 API 地址
# 例如：Ollama 默认地址为 http://localhost:11434/v1
# 例如：vLLM 默认地址为 http://localhost:8000/v1
LOCAL_BASE_URL = "http://localhost:8000/v1"

# 本地模型名称
LOCAL_MODEL = "qwen2.5:7b"

# ================================
# 模型参数配置
# ================================

# 温度参数（0-1，控制输出的随机性）
LLM_TEMPERATURE = 0.7

# 最大输出token数
LLM_MAX_TOKENS = 2048

# 请求超时时间（秒）
LLM_TIMEOUT = 60

# ================================
# 记忆系统配置
# ================================

# 最大历史消息数量
MAX_HISTORY_LENGTH = 20

# 触发摘要的消息数阈值
SUMMARY_THRESHOLD = 10

# ConversationSummaryBufferMemory 的最大token限制
MAX_TOKEN_LIMIT = 1000

# ================================
# 环境变量设置示例
# ================================
"""
你也可以通过设置环境变量来配置，而不是修改此文件：

# Linux/macOS:
export OPENAI_API_KEY="sk-your-openai-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-3.5-turbo"
export LOCAL_BASE_URL="http://localhost:8000/v1"
export LOCAL_MODEL="qwen2.5:7b"
export LLM_TEMPERATURE="0.7"
export LLM_MAX_TOKENS="2048"
export MAX_HISTORY_LENGTH="20"
export SUMMARY_THRESHOLD="10"
export MAX_TOKEN_LIMIT="1000"

# Windows:
set OPENAI_API_KEY=sk-your-openai-api-key-here
set OPENAI_BASE_URL=https://api.openai.com/v1
set OPENAI_MODEL=gpt-3.5-turbo
set LOCAL_BASE_URL=http://localhost:8000/v1
set LOCAL_MODEL=qwen2.5:7b
set LLM_TEMPERATURE=0.7
set LLM_MAX_TOKENS=2048
set MAX_HISTORY_LENGTH=20
set SUMMARY_THRESHOLD=10
set MAX_TOKEN_LIMIT=1000
"""

# ================================
# 常用模型配置参考
# ================================
"""
# OpenAI 模型
- gpt-4o: 最新的GPT-4模型
- gpt-4o-mini: 轻量版GPT-4模型
- gpt-3.5-turbo: 经典的GPT-3.5模型

# 本地模型（Ollama）
- qwen2.5:7b: 通义千问2.5 7B参数版本
- llama3.1:8b: Meta Llama 3.1 8B参数版本
- mistral:7b: Mistral 7B参数版本
- codellama:7b: 专门用于代码的Llama模型

# 本地模型（vLLM）
- Qwen/Qwen2.5-7B-Instruct
- meta-llama/Llama-3.1-8B-Instruct
- mistralai/Mistral-7B-Instruct-v0.3
"""