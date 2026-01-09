#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 大模型API配置

使用方法：
1. 复制 config.example.py 为 config.py
2. 修改相应的配置项
3. 或者设置环境变量
"""

import os
from typing import Optional

class LLMConfig:
    """大模型配置类"""
    
    def __init__(self):
        # OpenAI 配置
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # DeepSeek 配置
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        # 其他模型配置（如本地部署的模型）
        self.local_api_key = os.getenv("LOCAL_API_KEY", "")
        self.local_base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
        self.local_model = os.getenv("LOCAL_MODEL", "qwen2.5")
        
        # 通用配置
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        self.timeout = int(os.getenv("LLM_TIMEOUT", "60"))
        
        # 记忆配置
        self.max_history_length = int(os.getenv("MAX_HISTORY_LENGTH", "20"))
        self.summary_threshold = int(os.getenv("SUMMARY_THRESHOLD", "10"))
        self.max_token_limit = int(os.getenv("MAX_TOKEN_LIMIT", "1000"))
    
    def get_openai_config(self) -> dict:
        """获取OpenAI配置"""
        return {
            "api_key": self.openai_api_key,
            "base_url": self.openai_base_url,
            "model": self.openai_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout
        }
    
    def get_deepseek_config(self) -> dict:
        """获取DeepSeek配置"""
        return {
            "api_key": self.deepseek_api_key,
            "base_url": self.deepseek_base_url,
            "model": self.deepseek_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout
        }
    
    def get_local_config(self) -> dict:
        """获取本地模型配置"""
        return {
            "api_key": self.local_api_key,
            "base_url": self.local_base_url,
            "model": self.local_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout
        }
    
    def validate_config(self, provider: str = None) -> bool:
        """验证配置是否完整"""
        if provider is None:
            # 检查是否有任何可用的配置
            return (bool(self.openai_api_key) or 
                   bool(self.deepseek_api_key) or 
                   bool(self.local_base_url))
        elif provider == "openai":
            return bool(self.openai_api_key)
        elif provider == "deepseek":
            return bool(self.deepseek_api_key)
        elif provider == "local":
            return bool(self.local_base_url)
        return False
    
    def get_memory_config(self) -> dict:
        """获取记忆配置"""
        return {
            "max_history_length": self.max_history_length,
            "summary_threshold": self.summary_threshold,
            "max_token_limit": self.max_token_limit
        }

# 全局配置实例
config = LLMConfig()

# 配置验证函数
def check_config(provider: str = "openai") -> None:
    """检查配置是否有效"""
    if not config.validate_config(provider):
        if provider == "openai":
            raise ValueError(
                "OpenAI API Key 未设置！请设置环境变量 OPENAI_API_KEY 或修改 config.py"
            )
        elif provider == "deepseek":
            raise ValueError(
                "DeepSeek API Key 未设置！请设置环境变量 DEEPSEEK_API_KEY 或修改 config.py"
            )
        elif provider == "local":
            raise ValueError(
                "本地模型URL未设置！请设置环境变量 LOCAL_BASE_URL 或修改 config.py"
            )
    print(f"✅ {provider.upper()} 配置验证通过")

if __name__ == "__main__":
    # 测试配置
    print("=== 配置信息 ===")
    print(f"OpenAI API Key: {'已设置' if config.openai_api_key else '未设置'}")
    print(f"OpenAI Base URL: {config.openai_base_url}")
    print(f"OpenAI Model: {config.openai_model}")
    print(f"DeepSeek API Key: {'已设置' if config.deepseek_api_key else '未设置'}")
    print(f"DeepSeek Base URL: {config.deepseek_base_url}")
    print(f"DeepSeek Model: {config.deepseek_model}")
    print(f"Local Base URL: {config.local_base_url}")
    print(f"Local Model: {config.local_model}")
    print(f"Temperature: {config.temperature}")
    print(f"Max Tokens: {config.max_tokens}")
    
    # 验证配置
    try:
        check_config("openai")
    except ValueError as e:
        print(f"❌ {e}")
    
    try:
        check_config("deepseek")
    except ValueError as e:
        print(f"❌ {e}")
    
    try:
        check_config("local")
    except ValueError as e:
        print(f"❌ {e}")