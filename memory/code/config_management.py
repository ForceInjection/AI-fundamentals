# -*- coding: utf-8 -*-
"""
配置管理与环境隔离

本模块实现了基于Pydantic的配置管理系统，支持环境隔离和配置验证。
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from enum import Enum

class Environment(str, Enum):
    """环境类型枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class DatabaseConfig(BaseSettings):
    """数据库配置"""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    username: str = Field(env="DB_USERNAME")
    password: str = Field(env="DB_PASSWORD")
    database: str = Field(env="DB_NAME")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('端口号必须在1-65535之间')
        return v
    
    @property
    def url(self) -> str:
        """构建数据库连接URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False

class RedisConfig(BaseSettings):
    """Redis配置"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    socket_connect_timeout: int = Field(default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('端口号必须在1-65535之间')
        return v
    
    @validator('db')
    def validate_db(cls, v):
        if not 0 <= v <= 15:
            raise ValueError('Redis数据库索引必须在0-15之间')
        return v
    
    @property
    def url(self) -> str:
        """构建Redis连接URL"""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"
    
    class Config:
        env_prefix = "REDIS_"
        case_sensitive = False

class ModelConfig(BaseSettings):
    """模型配置"""
    # 模型路径
    ner_model_path: str = Field(env="NER_MODEL_PATH")
    coref_model_path: str = Field(env="COREF_MODEL_PATH")
    embedding_model_path: str = Field(env="EMBEDDING_MODEL_PATH")
    
    # 模型推理配置
    batch_size: int = Field(default=32, env="MODEL_BATCH_SIZE")
    max_sequence_length: int = Field(default=512, env="MODEL_MAX_SEQ_LEN")
    device: str = Field(default="cuda", env="MODEL_DEVICE")
    num_workers: int = Field(default=4, env="MODEL_NUM_WORKERS")
    
    # 缓存配置
    enable_model_cache: bool = Field(default=True, env="MODEL_ENABLE_CACHE")
    cache_size: int = Field(default=1000, env="MODEL_CACHE_SIZE")
    cache_ttl: int = Field(default=3600, env="MODEL_CACHE_TTL")
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        if not 1 <= v <= 256:
            raise ValueError('批处理大小必须在1-256之间')
        return v
    
    @validator('max_sequence_length')
    def validate_max_sequence_length(cls, v):
        if not 64 <= v <= 2048:
            raise ValueError('最大序列长度必须在64-2048之间')
        return v
    
    @validator('device')
    def validate_device(cls, v):
        if v not in ['cpu', 'cuda', 'mps']:
            raise ValueError('设备类型必须是cpu、cuda或mps之一')
        return v
    
    class Config:
        env_prefix = "MODEL_"
        case_sensitive = False

class APIConfig(BaseSettings):
    """API配置"""
    # OpenAI配置
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=1000, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    
    # 其他API配置
    api_timeout: int = Field(default=30, env="API_TIMEOUT")
    api_retry_times: int = Field(default=3, env="API_RETRY_TIMES")
    api_retry_delay: float = Field(default=1.0, env="API_RETRY_DELAY")
    
    @validator('openai_temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('温度参数必须在0.0-2.0之间')
        return v
    
    @validator('api_timeout')
    def validate_timeout(cls, v):
        if not 1 <= v <= 300:
            raise ValueError('API超时时间必须在1-300秒之间')
        return v
    
    class Config:
        env_prefix = "API_"
        case_sensitive = False

class SystemConfig(BaseSettings):
    """系统配置"""
    # 环境配置
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    
    # 服务配置
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # 安全配置
    secret_key: str = Field(env="SECRET_KEY")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # 缓存配置
    entity_cache_size: int = Field(default=10000, env="ENTITY_CACHE_SIZE")
    entity_cache_ttl: int = Field(default=3600, env="ENTITY_CACHE_TTL")
    dialogue_cache_size: int = Field(default=1000, env="DIALOGUE_CACHE_SIZE")
    dialogue_cache_ttl: int = Field(default=7200, env="DIALOGUE_CACHE_TTL")
    
    # 性能配置
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    
    # 子配置
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    model: ModelConfig = ModelConfig()
    api: APIConfig = APIConfig()
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('端口号必须在1-65535之间')
        return v
    
    @validator('max_workers')
    def validate_max_workers(cls, v):
        if not 1 <= v <= 32:
            raise ValueError('最大工作进程数必须在1-32之间')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class DevelopmentConfig(SystemConfig):
    """开发环境配置"""
    debug: bool = True
    log_level: LogLevel = LogLevel.DEBUG
    environment: Environment = Environment.DEVELOPMENT
    
    # 开发环境特定配置
    enable_auto_reload: bool = True
    enable_debug_toolbar: bool = True
    
    class Config:
        env_file = ".env.development"

class TestingConfig(SystemConfig):
    """测试环境配置"""
    debug: bool = True
    log_level: LogLevel = LogLevel.DEBUG
    environment: Environment = Environment.TESTING
    
    # 测试环境特定配置
    database: DatabaseConfig = DatabaseConfig(
        host="localhost",
        database="test_db",
        username="test_user",
        password="test_password"
    )
    
    redis: RedisConfig = RedisConfig(
        host="localhost",
        db=1  # 使用不同的数据库
    )
    
    class Config:
        env_file = ".env.testing"

class StagingConfig(SystemConfig):
    """预发布环境配置"""
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    environment: Environment = Environment.STAGING
    
    # 预发布环境特定配置
    max_workers: int = 8
    enable_metrics: bool = True
    enable_tracing: bool = True
    
    class Config:
        env_file = ".env.staging"

class ProductionConfig(SystemConfig):
    """生产环境配置"""
    debug: bool = False
    log_level: LogLevel = LogLevel.WARNING
    environment: Environment = Environment.PRODUCTION
    
    # 生产环境特定配置
    max_workers: int = 16
    request_timeout: int = 60
    enable_metrics: bool = True
    enable_tracing: bool = True
    
    # 生产环境安全配置
    allowed_hosts: List[str] = Field(env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(env="CORS_ORIGINS")
    
    class Config:
        env_file = ".env.production"

# 配置工厂
def get_config() -> SystemConfig:
    """获取配置实例"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "testing": TestingConfig,
        "staging": StagingConfig,
        "production": ProductionConfig
    }
    
    config_class = config_map.get(env, DevelopmentConfig)
    return config_class()

def validate_config(config: SystemConfig) -> bool:
    """验证配置"""
    try:
        # 验证必需的配置项
        required_fields = [
            'secret_key',
            'model.ner_model_path',
            'model.coref_model_path'
        ]
        
        for field in required_fields:
            value = config
            for part in field.split('.'):
                value = getattr(value, part)
            
            if not value:
                raise ValueError(f"必需的配置项 {field} 未设置")
        
        # 验证文件路径
        model_paths = [
            config.model.ner_model_path,
            config.model.coref_model_path,
            config.model.embedding_model_path
        ]
        
        for path in model_paths:
            if path and not os.path.exists(path):
                print(f"警告: 模型文件 {path} 不存在")
        
        return True
        
    except Exception as e:
        print(f"配置验证失败: {str(e)}")
        return False

def create_env_template(env_type: str = "development") -> str:
    """创建环境配置模板"""
    templates = {
        "development": """
# 开发环境配置
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# 服务配置
HOST=0.0.0.0
PORT=8000
MAX_WORKERS=4

# 安全配置
SECRET_KEY=your-secret-key-here

# 数据库配置
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=your-db-username
DB_PASSWORD=your-db-password
DB_NAME=your-db-name

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# 模型配置
NER_MODEL_PATH=./models/ner_model
COREF_MODEL_PATH=./models/coref_model
EMBEDDING_MODEL_PATH=./models/embedding_model
MODEL_DEVICE=cpu
MODEL_BATCH_SIZE=32

# API配置
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-3.5-turbo
""",
        "production": """
# 生产环境配置
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# 服务配置
HOST=0.0.0.0
PORT=8000
MAX_WORKERS=16
REQUEST_TIMEOUT=60

# 安全配置
SECRET_KEY=your-production-secret-key
ALLOWED_HOSTS=your-domain.com,api.your-domain.com
CORS_ORIGINS=https://your-frontend.com

# 数据库配置
DB_HOST=your-db-host
DB_PORT=5432
DB_USERNAME=your-db-username
DB_PASSWORD=your-secure-db-password
DB_NAME=your-production-db
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40

# Redis配置
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password
REDIS_DB=0
REDIS_MAX_CONNECTIONS=100

# 模型配置
NER_MODEL_PATH=/app/models/ner_model
COREF_MODEL_PATH=/app/models/coref_model
EMBEDDING_MODEL_PATH=/app/models/embedding_model
MODEL_DEVICE=cuda
MODEL_BATCH_SIZE=64
MODEL_ENABLE_CACHE=true

# API配置
OPENAI_API_KEY=your-production-openai-api-key
OPENAI_MODEL=gpt-4
API_TIMEOUT=60
API_RETRY_TIMES=3

# 监控配置
ENABLE_METRICS=true
METRICS_PORT=9090
ENABLE_TRACING=true
"""
    }
    
    return templates.get(env_type, templates["development"])

# 使用示例
if __name__ == "__main__":
    # 获取配置
    config = get_config()
    
    # 验证配置
    if validate_config(config):
        print("配置验证通过")
        print(f"当前环境: {config.environment}")
        print(f"数据库URL: {config.database.url}")
        print(f"Redis URL: {config.redis.url}")
    else:
        print("配置验证失败")
    
    # 生成配置模板
    print("\n开发环境配置模板:")
    print(create_env_template("development"))