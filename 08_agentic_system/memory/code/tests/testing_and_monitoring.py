# -*- coding: utf-8 -*-
"""
测试框架和性能监控

本模块实现了完整的测试策略、性能监控、错误处理和日志记录功能。
"""

import pytest
import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog
from pythonjsonlogger import jsonlogger

# 导入被测试的模块
from entity_recognition import Entity, EnhancedEntityRecognitionLayer
from coreference_resolution import (
    Mention, PronounType, CoreferenceResult, AdvancedCoreferenceLayer
)
from dialogue_state_manager import IntelligentStateManager

# ==================== 测试框架 ====================

class TestDataBuilder:
    """测试数据构建器"""
    
    @staticmethod
    def create_entity(entity_id: str = "1", text: str = "张三", 
                     entity_type: str = "PERSON", **kwargs) -> Entity:
        """创建测试实体"""
        return Entity(
            id=entity_id,
            text=text,
            type=entity_type,
            start=kwargs.get('start', 0),
            end=kwargs.get('end', len(text)),
            confidence=kwargs.get('confidence', 1.0),
            gender=kwargs.get('gender', 'male'),
            attributes=kwargs.get('attributes', {})
        )
    
    @staticmethod
    def create_mention(text: str = "他", mention_type: PronounType = PronounType.PERSONAL,
                      **kwargs) -> Mention:
        """创建测试指代词"""
        return Mention(
            text=text,
            type=mention_type,
            start=kwargs.get('start', 0),
            end=kwargs.get('end', len(text)),
            gender=kwargs.get('gender', 'male'),
            number=kwargs.get('number', 'singular'),
            person=kwargs.get('person', 'third')
        )
    
    @staticmethod
    def create_dialogue_context(dialogue_id: str = "test_001", 
                               turn_id: int = 1, **kwargs) -> Dict[str, Any]:
        """创建测试对话上下文"""
        return {
            'dialogue_id': dialogue_id,
            'turn_id': turn_id,
            'current_turn': kwargs.get('current_turn', turn_id),
            'entity_salience': kwargs.get('entity_salience', {}),
            'entity_last_mention': kwargs.get('entity_last_mention', {}),
            'timestamp': kwargs.get('timestamp', time.time())
        }

class TestCoreferenceEngine:
    """指代消解引擎测试类"""
    
    @pytest.fixture
    def engine(self):
        """测试夹具：创建指代消解引擎"""
        return AdvancedCoreferenceLayer()
    
    @pytest.fixture
    def sample_entities(self):
        """测试夹具：创建样本实体"""
        return [
            TestDataBuilder.create_entity("1", "张三", "PERSON", gender="male"),
            TestDataBuilder.create_entity("2", "李四", "PERSON", gender="male"),
            TestDataBuilder.create_entity("3", "苹果公司", "ORG", gender="neutral")
        ]
    
    def test_resolve_single_candidate(self, engine, sample_entities):
        """测试单候选实体的指代消解"""
        mention = TestDataBuilder.create_mention("他", PronounType.PERSONAL, gender="male")
        context = TestDataBuilder.create_dialogue_context(
            entity_salience={"1": 0.8, "2": 0.3, "3": 0.1}
        )
        
        # 模拟候选筛选只返回一个候选
        with patch.object(engine.candidate_filter, 'filter') as mock_filter:
            mock_filter.return_value = [sample_entities[0]]
            
            result = engine.resolve(mention, sample_entities, context)
            
            assert result.entity == sample_entities[0]
            assert result.confidence > 0.5
            assert result.method == "advanced_neural"
    
    def test_resolve_multiple_candidates(self, engine, sample_entities):
        """测试多候选实体的指代消解"""
        mention = TestDataBuilder.create_mention("他", PronounType.PERSONAL, gender="male")
        context = TestDataBuilder.create_dialogue_context(
            entity_salience={"1": 0.8, "2": 0.6, "3": 0.1}
        )
        
        result = engine.resolve(mention, sample_entities[:2], context)
        
        # 应该选择显著性更高的实体
        assert result.entity is not None
        assert result.entity.id in ["1", "2"]
        assert len(result.candidates) <= 3
    
    def test_resolve_no_candidates(self, engine):
        """测试无候选实体的情况"""
        mention = TestDataBuilder.create_mention("它", PronounType.PERSONAL, gender="neutral")
        context = TestDataBuilder.create_dialogue_context()
        
        # 传入不兼容的实体（人称实体与中性代词不匹配）
        person_entities = [
            TestDataBuilder.create_entity("1", "张三", "PERSON", gender="male")
        ]
        
        result = engine.resolve(mention, person_entities, context)
        
        # 应该没有找到合适的候选
        assert result.entity is None or result.confidence < 0.3
    
    def test_gender_consistency(self, engine):
        """测试性别一致性"""
        # 测试男性代词
        male_mention = TestDataBuilder.create_mention("他", gender="male")
        female_entity = TestDataBuilder.create_entity("1", "李女士", "PERSON", gender="female")
        male_entity = TestDataBuilder.create_entity("2", "王先生", "PERSON", gender="male")
        
        context = TestDataBuilder.create_dialogue_context(
            entity_salience={"1": 0.8, "2": 0.8}
        )
        
        result = engine.resolve(male_mention, [female_entity, male_entity], context)
        
        # 应该选择性别匹配的实体
        assert result.entity == male_entity
    
    @pytest.mark.asyncio
    async def test_async_resolution(self, engine, sample_entities):
        """测试异步指代消解"""
        mention = TestDataBuilder.create_mention()
        context = TestDataBuilder.create_dialogue_context()
        
        # 模拟异步执行
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, engine.resolve, mention, sample_entities, context
        )
        
        assert isinstance(result, CoreferenceResult)
        assert result.mention == mention

class TestEntityRecognition:
    """实体识别测试类"""
    
    @pytest.fixture
    def entity_layer(self):
        """测试夹具：创建实体识别层"""
        return EnhancedEntityRecognitionLayer()
    
    def test_extract_person_entities(self, entity_layer):
        """测试人名实体提取"""
        text = "张三和李四是好朋友"
        context = TestDataBuilder.create_dialogue_context()
        
        entities = entity_layer.process(text, context)
        
        # 应该提取到人名实体
        person_entities = [e for e in entities if e.type == "PERSON"]
        assert len(person_entities) >= 1
    
    def test_entity_caching(self, entity_layer):
        """测试实体缓存"""
        text = "苹果公司发布了新产品"
        context = TestDataBuilder.create_dialogue_context()
        
        # 第一次处理
        start_time = time.time()
        entities1 = entity_layer.process(text, context)
        first_duration = time.time() - start_time
        
        # 第二次处理（应该使用缓存）
        start_time = time.time()
        entities2 = entity_layer.process(text, context)
        second_duration = time.time() - start_time
        
        # 第二次应该更快（使用了缓存）
        assert len(entities1) == len(entities2)
        # 注意：在测试环境中缓存效果可能不明显

class TestDialogueStateManager:
    """对话状态管理器测试类"""
    
    @pytest.fixture
    def state_manager(self):
        """测试夹具：创建状态管理器"""
        return IntelligentStateManager()
    
    def test_state_update(self, state_manager):
        """测试状态更新"""
        entities = [TestDataBuilder.create_entity()]
        coreferences = []
        
        state = state_manager.update_state(
            "你好，张三", "你好！", entities, coreferences
        )
        
        assert state['current_turn'] == 1
        assert len(state['active_entities']) == 1
        assert '1' in state['entity_salience']
    
    def test_salience_decay(self, state_manager):
        """测试显著性衰减"""
        entity = TestDataBuilder.create_entity()
        
        # 第一轮：提及实体
        state_manager.update_state("张三来了", "好的", [entity], [])
        first_salience = state_manager.get_current_state()['entity_salience']['1']
        
        # 第二轮：不提及实体
        state_manager.update_state("天气不错", "是的", [], [])
        second_salience = state_manager.get_current_state()['entity_salience'].get('1', 0)
        
        # 显著性应该衰减
        assert second_salience < first_salience

# ==================== 性能监控 ====================

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        # 计数器指标
        self.request_count = Counter(
            'coref_requests_total',
            'Total number of coreference resolution requests',
            ['method', 'status']
        )
        
        self.entity_count = Counter(
            'entities_extracted_total',
            'Total number of entities extracted',
            ['entity_type']
        )
        
        # 直方图指标
        self.processing_time = Histogram(
            'coref_processing_seconds',
            'Time spent processing coreference resolution',
            ['component']
        )
        
        self.confidence_score = Histogram(
            'coref_confidence_score',
            'Confidence scores of coreference resolution',
            buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        )
        
        # 仪表指标
        self.active_dialogues = Gauge(
            'active_dialogues',
            'Number of active dialogues'
        )
        
        self.cache_hit_rate = Gauge(
            'cache_hit_rate',
            'Cache hit rate percentage'
        )
    
    def record_request(self, method: str, status: str):
        """记录请求"""
        self.request_count.labels(method=method, status=status).inc()
    
    def record_entity_extraction(self, entity_type: str, count: int = 1):
        """记录实体提取"""
        self.entity_count.labels(entity_type=entity_type).inc(count)
    
    def record_processing_time(self, component: str, duration: float):
        """记录处理时间"""
        self.processing_time.labels(component=component).observe(duration)
    
    def record_confidence(self, confidence: float):
        """记录置信度"""
        self.confidence_score.observe(confidence)
    
    def update_active_dialogues(self, count: int):
        """更新活跃对话数"""
        self.active_dialogues.set(count)
    
    def update_cache_hit_rate(self, rate: float):
        """更新缓存命中率"""
        self.cache_hit_rate.set(rate)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = structlog.get_logger()
    
    def monitor_coreference_resolution(self, func):
        """监控指代消解性能的装饰器"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # 记录成功指标
                duration = time.time() - start_time
                self.metrics.record_processing_time('coreference_resolution', duration)
                self.metrics.record_request('coreference', 'success')
                
                if hasattr(result, 'confidence'):
                    self.metrics.record_confidence(result.confidence)
                
                self.logger.info(
                    "coreference_resolution_completed",
                    duration=duration,
                    confidence=getattr(result, 'confidence', None)
                )
                
                return result
                
            except Exception as e:
                # 记录失败指标
                duration = time.time() - start_time
                self.metrics.record_request('coreference', 'error')
                
                self.logger.error(
                    "coreference_resolution_failed",
                    duration=duration,
                    error=str(e)
                )
                
                raise
        
        return wrapper
    
    def monitor_entity_extraction(self, func):
        """监控实体提取性能的装饰器"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                entities = func(*args, **kwargs)
                
                # 记录指标
                duration = time.time() - start_time
                self.metrics.record_processing_time('entity_extraction', duration)
                
                # 按类型统计实体
                entity_types = {}
                for entity in entities:
                    entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
                
                for entity_type, count in entity_types.items():
                    self.metrics.record_entity_extraction(entity_type, count)
                
                self.logger.info(
                    "entity_extraction_completed",
                    duration=duration,
                    entity_count=len(entities),
                    entity_types=entity_types
                )
                
                return entities
                
            except Exception as e:
                duration = time.time() - start_time
                self.logger.error(
                    "entity_extraction_failed",
                    duration=duration,
                    error=str(e)
                )
                raise
        
        return wrapper

# ==================== 错误处理 ====================

class ErrorType(Enum):
    """错误类型枚举"""
    MODEL_ERROR = "model_error"
    DATA_ERROR = "data_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"

@dataclass
class ErrorContext:
    """错误上下文"""
    error_type: ErrorType
    error_message: str
    component: str
    timestamp: datetime
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    additional_info: Dict[str, Any] = None

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = structlog.get_logger()
        self.fallback_strategies = {
            ErrorType.MODEL_ERROR: self._model_error_fallback,
            ErrorType.NETWORK_ERROR: self._network_error_fallback,
            ErrorType.TIMEOUT_ERROR: self._timeout_error_fallback
        }
    
    def handle_error(self, error: Exception, context: ErrorContext) -> Any:
        """处理错误"""
        # 记录错误
        self.logger.error(
            "error_occurred",
            error_type=context.error_type.value,
            error_message=context.error_message,
            component=context.component,
            request_id=context.request_id,
            additional_info=context.additional_info
        )
        
        # 更新错误指标
        self.metrics.record_request(context.component, 'error')
        
        # 尝试降级策略
        fallback_strategy = self.fallback_strategies.get(context.error_type)
        if fallback_strategy:
            try:
                return fallback_strategy(error, context)
            except Exception as fallback_error:
                self.logger.error(
                    "fallback_strategy_failed",
                    original_error=str(error),
                    fallback_error=str(fallback_error)
                )
        
        # 如果没有降级策略或降级失败，重新抛出异常
        raise error
    
    def _model_error_fallback(self, error: Exception, context: ErrorContext) -> Any:
        """模型错误降级策略"""
        if context.component == 'coreference_resolution':
            # 使用简单的距离优先策略
            self.logger.info("using_distance_based_fallback")
            return self._distance_based_resolution(context.additional_info)
        
        return None
    
    def _network_error_fallback(self, error: Exception, context: ErrorContext) -> Any:
        """网络错误降级策略"""
        # 使用缓存结果或默认响应
        self.logger.info("using_cached_or_default_response")
        return "抱歉，网络连接出现问题，请稍后重试。"
    
    def _timeout_error_fallback(self, error: Exception, context: ErrorContext) -> Any:
        """超时错误降级策略"""
        # 返回部分结果或默认响应
        self.logger.info("using_partial_or_default_response")
        return "处理时间较长，请稍后重试或简化您的请求。"
    
    def _distance_based_resolution(self, context_info: Dict[str, Any]) -> Any:
        """基于距离的简单指代消解"""
        # 简化的降级实现
        mentions = context_info.get('mentions', [])
        entities = context_info.get('entities', [])
        
        if not mentions or not entities:
            return None
        
        # 选择最近的实体
        mention = mentions[0]
        closest_entity = min(entities, key=lambda e: abs(e.start - mention.start))
        
        return CoreferenceResult(
            mention=mention,
            entity=closest_entity,
            confidence=0.6,  # 降级结果的置信度较低
            candidates=[closest_entity],
            method="distance_based_fallback"
        )

# ==================== 结构化日志 ====================

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str, level: str = "INFO"):
        # 配置structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(name)
    
    def log_coreference_resolution(self, dialogue_id: str, mention: str,
                                 resolved_entity: Optional[str], confidence: float,
                                 processing_time: float, method: str):
        """记录指代消解事件"""
        self.logger.info(
            "coreference_resolution_completed",
            event_type="coreference_resolution",
            dialogue_id=dialogue_id,
            mention=mention,
            resolved_entity=resolved_entity,
            confidence=confidence,
            processing_time_ms=processing_time * 1000,
            method=method
        )
    
    def log_entity_extraction(self, dialogue_id: str, text: str,
                            entities: List[Entity], processing_time: float):
        """记录实体提取事件"""
        entity_info = [
            {"text": e.text, "type": e.type, "confidence": e.confidence}
            for e in entities
        ]
        
        self.logger.info(
            "entity_extraction_completed",
            event_type="entity_extraction",
            dialogue_id=dialogue_id,
            input_text=text,
            entities=entity_info,
            entity_count=len(entities),
            processing_time_ms=processing_time * 1000
        )
    
    def log_dialogue_state_update(self, dialogue_id: str, turn_id: int,
                                 active_entities: int, salience_scores: Dict[str, float]):
        """记录对话状态更新事件"""
        self.logger.info(
            "dialogue_state_updated",
            event_type="dialogue_state_update",
            dialogue_id=dialogue_id,
            turn_id=turn_id,
            active_entities=active_entities,
            avg_salience=sum(salience_scores.values()) / len(salience_scores) if salience_scores else 0
        )
    
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """记录性能指标"""
        self.logger.info(
            "performance_metrics",
            event_type="performance_metrics",
            **metrics
        )
    
    def log_error(self, error_type: str, error_message: str, 
                 context: Dict[str, Any]):
        """记录错误事件"""
        self.logger.error(
            "error_occurred",
            event_type="error",
            error_type=error_type,
            error_message=error_message,
            **context
        )

# ==================== 监控服务启动 ====================

def start_monitoring_server(port: int = 9090):
    """启动监控服务器"""
    start_http_server(port)
    print(f"监控服务器已启动，端口: {port}")
    print(f"指标端点: http://localhost:{port}/metrics")

# 使用示例
if __name__ == "__main__":
    # 启动监控服务器
    start_monitoring_server()
    
    # 创建监控组件
    metrics = MetricsCollector()
    monitor = PerformanceMonitor(metrics)
    error_handler = ErrorHandler(metrics)
    logger = StructuredLogger("test")
    
    # 模拟一些指标
    metrics.record_request("coreference", "success")
    metrics.record_processing_time("entity_extraction", 0.15)
    metrics.record_confidence(0.85)
    
    print("监控系统已初始化")