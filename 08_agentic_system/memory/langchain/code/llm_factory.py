#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM工厂类 - 根据配置创建不同的LLM实例
"""

import os
from typing import Optional, Union
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.language_models.base import BaseLanguageModel

try:
    from .config import config, check_config
except ImportError:
    from config import config, check_config

class LLMFactory:
    """LLM工厂类，用于创建不同类型的语言模型实例"""
    
    @staticmethod
    def create_openai_llm(
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> ChatOpenAI:
        """
        创建OpenAI兼容的LLM实例
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时时间
            
        Returns:
            ChatOpenAI实例
        """
        # 使用传入参数或配置文件中的默认值
        api_key = api_key or config.openai_api_key
        base_url = base_url or config.openai_base_url
        model = model or config.openai_model
        temperature = temperature if temperature is not None else config.temperature
        max_tokens = max_tokens or config.max_tokens
        timeout = timeout or config.timeout
        
        # 验证必要参数
        if not api_key:
            raise ValueError("API Key 不能为空！请设置 OPENAI_API_KEY 环境变量或在配置文件中指定")
        
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
    
    @staticmethod
    def create_local_llm(
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> ChatOpenAI:
        """
        创建本地部署的LLM实例（使用OpenAI兼容接口）
        
        Args:
            base_url: 本地API地址
            model: 模型名称
            api_key: API密钥（如果需要）
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时时间
            
        Returns:
            ChatOpenAI实例
        """
        # 使用传入参数或配置文件中的默认值
        base_url = base_url or config.local_base_url
        model = model or config.local_model
        api_key = api_key or config.local_api_key or "dummy-key"  # 本地模型可能不需要真实的API key
        temperature = temperature if temperature is not None else config.temperature
        max_tokens = max_tokens or config.max_tokens
        timeout = timeout or config.timeout
        
        # 验证必要参数
        if not base_url:
            raise ValueError("本地模型URL不能为空！请设置 LOCAL_BASE_URL 环境变量或在配置文件中指定")
        
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
    
    @staticmethod
    def create_ollama_llm(
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Ollama:
        """
        创建Ollama LLM实例
        
        Args:
            model: 模型名称
            base_url: Ollama服务地址
            temperature: 温度参数
            
        Returns:
            Ollama实例
        """
        model = model or config.local_model
        base_url = base_url or "http://localhost:11434"  # Ollama默认地址
        temperature = temperature if temperature is not None else config.temperature
        
        return Ollama(
            model=model,
            base_url=base_url,
            temperature=temperature
        )
    
    @staticmethod
    def create_deepseek_llm(
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> ChatOpenAI:
        """
        创建DeepSeek LLM实例
        
        Args:
            api_key: DeepSeek API密钥
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时时间
            
        Returns:
            ChatOpenAI实例
        """
        # 使用传入参数或配置文件中的默认值
        api_key = api_key or config.deepseek_api_key
        model = model or config.deepseek_model or "deepseek-chat"
        temperature = temperature if temperature is not None else config.temperature
        max_tokens = max_tokens or config.max_tokens
        timeout = timeout or config.timeout
        
        # 验证必要参数
        if not api_key:
            raise ValueError("DeepSeek API Key 不能为空！请设置 DEEPSEEK_API_KEY 环境变量或在配置文件中指定")
        
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
    
    @staticmethod
    def create_llm(
        provider: str = "openai",
        **kwargs
    ) -> BaseLanguageModel:
        """
        根据提供商创建LLM实例
        
        Args:
            provider: 提供商类型 ("openai", "deepseek", "local", "ollama")
            **kwargs: 其他参数
            
        Returns:
            LLM实例
        """
        provider = provider.lower()
        
        if provider == "openai":
            return LLMFactory.create_openai_llm(**kwargs)
        elif provider == "deepseek":
            return LLMFactory.create_deepseek_llm(**kwargs)
        elif provider == "local":
            return LLMFactory.create_local_llm(**kwargs)
        elif provider == "ollama":
            return LLMFactory.create_ollama_llm(**kwargs)
        else:
            raise ValueError(f"不支持的提供商: {provider}。支持的提供商: openai, deepseek, local, ollama")
    
    @staticmethod
    def auto_create_llm() -> BaseLanguageModel:
        """
        自动选择可用的LLM实例
        
        优先级：OpenAI > DeepSeek > 本地模型 > Ollama
        
        Returns:
            LLM实例
        """
        # 尝试OpenAI
        try:
            if config.validate_config("openai"):
                print("🤖 使用 OpenAI 模型")
                return LLMFactory.create_openai_llm()
        except Exception as e:
            print(f"⚠️ OpenAI 模型创建失败: {e}")
        
        # 尝试DeepSeek
        try:
            if config.validate_config("deepseek"):
                print("🤖 使用 DeepSeek 模型")
                return LLMFactory.create_deepseek_llm()
        except Exception as e:
            print(f"⚠️ DeepSeek 模型创建失败: {e}")
        
        # 尝试本地模型
        try:
            if config.validate_config("local"):
                print("🤖 使用本地模型")
                return LLMFactory.create_local_llm()
        except Exception as e:
            print(f"⚠️ 本地模型创建失败: {e}")
        
        # 尝试Ollama
        try:
            print("🤖 尝试使用 Ollama 模型")
            return LLMFactory.create_ollama_llm()
        except Exception as e:
            print(f"⚠️ Ollama 模型创建失败: {e}")
        
        raise RuntimeError(
            "无法创建任何LLM实例！请检查配置：\n"
            "1. 设置 OPENAI_API_KEY 环境变量\n"
            "2. 或设置 DEEPSEEK_API_KEY 环境变量\n"
            "3. 或配置本地模型 LOCAL_BASE_URL\n"
            "4. 或启动 Ollama 服务"
        )

def get_openai_llm(**kwargs):
    """
    获取OpenAI LLM实例
    
    Args:
        **kwargs: 额外参数
    
    Returns:
        ChatOpenAI实例
    """
    return LLMFactory.create_openai_llm(**kwargs)

def get_deepseek_llm(**kwargs):
    """
    获取DeepSeek LLM实例
    
    Args:
        **kwargs: 额外参数
    
    Returns:
        ChatOpenAI实例
    """
    return LLMFactory.create_deepseek_llm(**kwargs)

def get_llm(provider: Optional[str] = None, **kwargs) -> BaseLanguageModel:
    """
    便捷函数：获取LLM实例
    
    Args:
        provider: 提供商类型 ("openai", "deepseek", "local", "ollama")，如果为None则自动选择
        **kwargs: 其他参数
        
    Returns:
        LLM实例
    """
    if provider is None:
        return LLMFactory.auto_create_llm()
    else:
        return LLMFactory.create_llm(provider, **kwargs)

if __name__ == "__main__":
    # 测试LLM工厂
    print("=== LLM工厂测试 ===")
    
    try:
        # 自动创建LLM
        llm = LLMFactory.auto_create_llm()
        print(f"✅ 成功创建LLM: {type(llm).__name__}")
        
        # 测试简单对话
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content="你好，请简单介绍一下自己")])
        print(f"🤖 模型响应: {response.content[:100]}...")
        
    except Exception as e:
        print(f"❌ LLM创建失败: {e}")
        print("\n请检查配置文件或环境变量设置")
    
    # 显示配置信息
    print("\n=== 当前配置 ===")
    print(f"OpenAI API Key: {'已设置' if config.openai_api_key else '未设置'}")
    print(f"OpenAI Base URL: {config.openai_base_url}")
    print(f"Local Base URL: {config.local_base_url}")
    print(f"Temperature: {config.temperature}")