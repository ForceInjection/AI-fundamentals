# -*- coding: utf-8 -*-
"""
实体识别与管理模块

本模块实现了增强的实体识别层，包括NER模型、实体注册器、缓存管理和实体链接等功能。
"""

import spacy
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
import time

@dataclass
class Entity:
    """实体数据类"""
    id: str
    text: str
    type: str
    start: int
    end: int
    confidence: float = 1.0
    gender: Optional[str] = None
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

class EntityCache:
    """实体缓存管理器"""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Entity]:
        """获取缓存的实体"""
        if key in self.cache:
            # 检查是否过期
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                # 清理过期缓存
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def put(self, key: str, entity: Entity):
        """存储实体到缓存"""
        # 如果缓存已满，清理最旧的条目
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = entity
        self.timestamps[key] = time.time()

class EntityRegistry:
    """实体注册器"""
    
    def __init__(self):
        self.entity_counter = 0
        self.registered_entities = {}
    
    def register(self, entity: Entity, dialogue_context: Dict) -> Entity:
        """注册新实体"""
        # 生成唯一ID
        if not entity.id:
            self.entity_counter += 1
            entity.id = f"entity_{self.entity_counter}"
        
        # 添加上下文信息
        entity.attributes.update({
            'dialogue_id': dialogue_context.get('dialogue_id'),
            'turn_id': dialogue_context.get('turn_id'),
            'timestamp': time.time()
        })
        
        self.registered_entities[entity.id] = entity
        return entity

class EntityLinker:
    """实体链接与消歧器"""
    
    def __init__(self):
        self.knowledge_base = {}  # 简化的知识库
    
    def link(self, entities: List[Entity], dialogue_context: Dict) -> List[Entity]:
        """链接实体到知识库"""
        linked_entities = []
        
        for entity in entities:
            # 简单的实体链接逻辑
            linked_entity = self._link_single_entity(entity, dialogue_context)
            linked_entities.append(linked_entity)
        
        return linked_entities
    
    def _link_single_entity(self, entity: Entity, context: Dict) -> Entity:
        """链接单个实体"""
        # 这里可以实现复杂的实体链接逻辑
        # 例如：查询知识库、计算相似度等
        
        # 简化实现：直接返回原实体
        return entity

class EnhancedEntityRecognitionLayer:
    """增强的实体识别层"""
    
    def __init__(self, model_name: str = "zh_core_web_sm"):
        self.ner_model = self._load_ner_model(model_name)
        self.entity_registry = EntityRegistry()
        self.entity_cache = EntityCache(max_size=10000, ttl=3600)
        self.entity_linker = EntityLinker()
    
    def _load_ner_model(self, model_name: str):
        """加载NER模型"""
        try:
            return spacy.load(model_name)
        except OSError:
            print(f"模型 {model_name} 未找到，尝试下载...")
            spacy.cli.download(model_name)
            return spacy.load(model_name)
    
    def process(self, text: str, dialogue_context: Dict) -> List[Entity]:
        """处理文本并提取实体"""
        # 1. NER识别
        raw_entities = self._extract_entities(text)
        
        # 2. 实体注册与缓存
        registered_entities = []
        for entity in raw_entities:
            cached_entity = self.entity_cache.get(entity.text)
            if cached_entity:
                registered_entities.append(cached_entity)
            else:
                reg_entity = self.entity_registry.register(
                    entity, dialogue_context
                )
                self.entity_cache.put(entity.text, reg_entity)
                registered_entities.append(reg_entity)
        
        # 3. 实体链接与消歧
        linked_entities = self.entity_linker.link(
            registered_entities, dialogue_context
        )
        
        return linked_entities
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """使用spaCy提取实体"""
        doc = self.ner_model(text)
        entities = []
        
        for ent in doc.ents:
            entity = Entity(
                id="",  # 将在注册时分配
                text=ent.text,
                type=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0  # spaCy不直接提供置信度
            )
            entities.append(entity)
        
        return entities

# 使用示例
if __name__ == "__main__":
    # 初始化实体识别层
    entity_layer = EnhancedEntityRecognitionLayer()
    
    # 测试文本
    test_text = "穆勒是拜仁慕尼黑的球员，他来自德国。"
    dialogue_context = {
        'dialogue_id': 'test_001',
        'turn_id': 1
    }
    
    # 提取实体
    entities = entity_layer.process(test_text, dialogue_context)
    
    # 打印结果
    for entity in entities:
        print(f"实体: {entity.text}, 类型: {entity.type}, ID: {entity.id}")