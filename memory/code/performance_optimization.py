#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化模块

本模块提供多轮指代消解对话系统的性能优化功能，包括：
- 候选实体预筛选
- 缓存机制
- 批处理优化
- 内存管理

作者: AI Assistant
版本: 1.0.0
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict, OrderedDict
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor


@dataclass
class Entity:
    """实体数据结构"""
    id: str
    text: str
    type: str
    position: Tuple[int, int]
    confidence: float
    features: Optional[Dict[str, Any]] = None


@dataclass
class CacheConfig:
    """缓存配置"""
    max_size: int = 10000
    ttl: int = 3600  # 秒
    enable_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379


class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key in self.cache:
            # 更新访问时间
            self.access_times[key] = time.time()
            # 移动到末尾（最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """设置缓存值"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # 移除最久未使用的项
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)


class EntityPrefilter:
    """候选实体预筛选器"""
    
    def __init__(self, max_distance: int = 5, type_compatibility: Dict[str, List[str]] = None):
        self.max_distance = max_distance
        self.type_compatibility = type_compatibility or {
            'PERSON': ['PERSON', 'PRONOUN'],
            'ORG': ['ORG', 'PRONOUN'],
            'LOC': ['LOC', 'PRONOUN'],
            'MISC': ['MISC', 'PRONOUN']
        }
    
    def filter_candidates(self, mention: Entity, candidates: List[Entity]) -> List[Entity]:
        """筛选候选实体"""
        filtered = []
        
        for candidate in candidates:
            # 距离筛选
            if self._calculate_distance(mention, candidate) > self.max_distance:
                continue
            
            # 类型兼容性筛选
            if not self._is_type_compatible(mention.type, candidate.type):
                continue
            
            # 性别一致性筛选（如果有性别信息）
            if not self._is_gender_compatible(mention, candidate):
                continue
            
            filtered.append(candidate)
        
        return filtered
    
    def _calculate_distance(self, mention: Entity, candidate: Entity) -> int:
        """计算实体间距离（句子数）"""
        # 简化实现：基于位置计算距离
        return abs(mention.position[0] - candidate.position[0])
    
    def _is_type_compatible(self, mention_type: str, candidate_type: str) -> bool:
        """检查类型兼容性"""
        return candidate_type in self.type_compatibility.get(mention_type, [])
    
    def _is_gender_compatible(self, mention: Entity, candidate: Entity) -> bool:
        """检查性别兼容性"""
        if not mention.features or not candidate.features:
            return True
        
        mention_gender = mention.features.get('gender')
        candidate_gender = candidate.features.get('gender')
        
        if mention_gender and candidate_gender:
            return mention_gender == candidate_gender
        
        return True


class BatchProcessor:
    """批处理优化器"""
    
    def __init__(self, batch_size: int = 32, max_wait_time: float = 0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_batch(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """批量处理请求"""
        if not requests:
            return []
        
        # 批量特征提取
        features_batch = await self._extract_features_batch(requests)
        
        # 批量模型推理
        predictions_batch = await self._model_inference_batch(features_batch)
        
        # 批量后处理
        results_batch = await self._postprocess_batch(predictions_batch)
        
        return results_batch
    
    async def _extract_features_batch(self, requests: List[Dict[str, Any]]) -> List[np.ndarray]:
        """批量特征提取"""
        # 模拟特征提取
        features = []
        for request in requests:
            # 这里应该是实际的特征提取逻辑
            feature_vector = np.random.rand(768)  # 模拟BERT特征
            features.append(feature_vector)
        
        return features
    
    async def _model_inference_batch(self, features_batch: List[np.ndarray]) -> List[float]:
        """批量模型推理"""
        # 模拟批量推理
        batch_array = np.stack(features_batch)
        
        # 这里应该是实际的模型推理
        # 模拟推理结果
        predictions = np.random.rand(len(features_batch))
        
        return predictions.tolist()
    
    async def _postprocess_batch(self, predictions_batch: List[float]) -> List[Dict[str, Any]]:
        """批量后处理"""
        results = []
        for pred in predictions_batch:
            result = {
                'confidence': pred,
                'prediction': pred > 0.5,
                'timestamp': time.time()
            }
            results.append(result)
        
        return results


class PerformanceOptimizer:
    """性能优化主类"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        
        # 初始化缓存
        self.entity_cache = LRUCache(max_size=1000)
        self.similarity_cache = LRUCache(max_size=10000)
        self.model_cache = LRUCache(max_size=5000)
        
        # 初始化组件
        self.prefilter = EntityPrefilter()
        self.batch_processor = BatchProcessor()
        
        # 性能统计
        self.stats = defaultdict(int)
        self.timing_stats = defaultdict(list)
    
    def get_cached_entities(self, text: str) -> Optional[List[Entity]]:
        """获取缓存的实体"""
        cache_key = self._generate_cache_key(text)
        cached_result = self.entity_cache.get(cache_key)
        
        if cached_result:
            self.stats['cache_hits'] += 1
            return cached_result
        
        self.stats['cache_misses'] += 1
        return None
    
    def cache_entities(self, text: str, entities: List[Entity]) -> None:
        """缓存实体结果"""
        cache_key = self._generate_cache_key(text)
        self.entity_cache.put(cache_key, entities)
    
    def get_cached_similarity(self, entity1_id: str, entity2_id: str) -> Optional[float]:
        """获取缓存的相似度"""
        cache_key = f"{entity1_id}:{entity2_id}"
        return self.similarity_cache.get(cache_key)
    
    def cache_similarity(self, entity1_id: str, entity2_id: str, similarity: float) -> None:
        """缓存相似度结果"""
        cache_key = f"{entity1_id}:{entity2_id}"
        self.similarity_cache.put(cache_key, similarity)
    
    def optimize_candidate_selection(self, mention: Entity, candidates: List[Entity]) -> List[Entity]:
        """优化候选实体选择"""
        start_time = time.time()
        
        # 预筛选
        filtered_candidates = self.prefilter.filter_candidates(mention, candidates)
        
        # 记录性能统计
        processing_time = time.time() - start_time
        self.timing_stats['candidate_filtering'].append(processing_time)
        
        return filtered_candidates
    
    async def batch_process_mentions(self, mentions: List[Dict[str, Any]]) -> List[Any]:
        """批量处理指代消解"""
        start_time = time.time()
        
        results = await self.batch_processor.process_batch(mentions)
        
        # 记录性能统计
        processing_time = time.time() - start_time
        self.timing_stats['batch_processing'].append(processing_time)
        
        return results
    
    def _generate_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = dict(self.stats)
        
        # 计算平均处理时间
        for operation, times in self.timing_stats.items():
            if times:
                stats[f'{operation}_avg_time'] = sum(times) / len(times)
                stats[f'{operation}_max_time'] = max(times)
                stats[f'{operation}_min_time'] = min(times)
        
        # 缓存统计
        stats['entity_cache_size'] = self.entity_cache.size()
        stats['similarity_cache_size'] = self.similarity_cache.size()
        stats['model_cache_size'] = self.model_cache.size()
        
        return stats
    
    def clear_caches(self) -> None:
        """清空所有缓存"""
        self.entity_cache.clear()
        self.similarity_cache.clear()
        self.model_cache.clear()
    
    def reset_stats(self) -> None:
        """重置性能统计"""
        self.stats.clear()
        self.timing_stats.clear()


# 使用示例
if __name__ == "__main__":
    # 创建性能优化器
    optimizer = PerformanceOptimizer()
    
    # 模拟实体
    entities = [
        Entity("1", "张三", "PERSON", (0, 2), 0.9),
        Entity("2", "他", "PRONOUN", (1, 1), 0.8),
        Entity("3", "公司", "ORG", (2, 2), 0.85)
    ]
    
    # 测试候选实体筛选
    mention = entities[1]  # "他"
    candidates = [entities[0], entities[2]]  # "张三", "公司"
    
    filtered = optimizer.optimize_candidate_selection(mention, candidates)
    print(f"筛选后的候选实体数量: {len(filtered)}")
    
    # 测试缓存
    text = "张三说他要去公司"
    optimizer.cache_entities(text, entities)
    cached_entities = optimizer.get_cached_entities(text)
    print(f"缓存命中: {cached_entities is not None}")
    
    # 获取性能统计
    stats = optimizer.get_performance_stats()
    print(f"性能统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")