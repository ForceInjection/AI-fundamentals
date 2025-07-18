#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存管理模块

本模块提供多轮指代消解对话系统的内存管理功能，包括：
- 实体记忆管理
- 对话历史压缩
- 显著性计算
- 内存优化策略

作者: AI Assistant
版本: 1.0.0
"""

import time
import math
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import heapq


class EntityType(Enum):
    """实体类型枚举"""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    MISCELLANEOUS = "MISC"
    PRONOUN = "PRONOUN"


class SaliencyType(Enum):
    """显著性类型枚举"""
    RECENCY = "recency"  # 时间显著性
    FREQUENCY = "frequency"  # 频率显著性
    IMPORTANCE = "importance"  # 重要性显著性
    SEMANTIC = "semantic"  # 语义显著性


@dataclass
class EntityMemory:
    """实体记忆数据结构"""
    entity_id: str
    entity_text: str
    entity_type: EntityType
    first_mention_time: float
    last_mention_time: float
    mention_count: int = 0
    mention_positions: List[Tuple[int, int]] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    coreference_chain: List[str] = field(default_factory=list)
    salience_score: float = 0.0
    context_embeddings: List[List[float]] = field(default_factory=list)
    
    def update_mention(self, position: Tuple[int, int], context_embedding: Optional[List[float]] = None):
        """更新实体提及信息"""
        self.last_mention_time = time.time()
        self.mention_count += 1
        self.mention_positions.append(position)
        
        if context_embedding:
            self.context_embeddings.append(context_embedding)
            # 保持最近的10个上下文嵌入
            if len(self.context_embeddings) > 10:
                self.context_embeddings.pop(0)
    
    def get_age(self) -> float:
        """获取实体年龄（秒）"""
        return time.time() - self.last_mention_time
    
    def get_lifespan(self) -> float:
        """获取实体生命周期（秒）"""
        return self.last_mention_time - self.first_mention_time


@dataclass
class DialogueTurn:
    """对话轮次数据结构"""
    turn_id: int
    timestamp: float
    user_input: str
    system_response: str
    entities: List[str]  # 实体ID列表
    coreferences: List[Tuple[str, str]]  # 指代关系对
    importance_score: float = 0.0
    compressed: bool = False
    summary: Optional[str] = None


class SalienceCalculator:
    """显著性计算器"""
    
    def __init__(self, 
                 recency_weight: float = 0.3,
                 frequency_weight: float = 0.3,
                 importance_weight: float = 0.2,
                 semantic_weight: float = 0.2,
                 decay_factor: float = 0.1):
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.importance_weight = importance_weight
        self.semantic_weight = semantic_weight
        self.decay_factor = decay_factor
    
    def calculate_salience(self, entity: EntityMemory, current_time: float = None) -> float:
        """计算实体显著性分数"""
        if current_time is None:
            current_time = time.time()
        
        # 时间显著性（基于最近提及时间的指数衰减）
        time_diff = current_time - entity.last_mention_time
        recency_score = math.exp(-self.decay_factor * time_diff)
        
        # 频率显著性（基于提及次数的对数函数）
        frequency_score = math.log(1 + entity.mention_count) / math.log(10)
        
        # 重要性显著性（基于实体类型和属性）
        importance_score = self._calculate_importance(entity)
        
        # 语义显著性（基于上下文嵌入的一致性）
        semantic_score = self._calculate_semantic_consistency(entity)
        
        # 加权计算总显著性
        total_salience = (
            self.recency_weight * recency_score +
            self.frequency_weight * frequency_score +
            self.importance_weight * importance_score +
            self.semantic_weight * semantic_score
        )
        
        return min(1.0, total_salience)  # 限制在[0, 1]范围内
    
    def _calculate_importance(self, entity: EntityMemory) -> float:
        """计算实体重要性"""
        # 基于实体类型的基础重要性
        type_importance = {
            EntityType.PERSON: 0.8,
            EntityType.ORGANIZATION: 0.7,
            EntityType.LOCATION: 0.6,
            EntityType.MISCELLANEOUS: 0.5,
            EntityType.PRONOUN: 0.3
        }
        
        base_score = type_importance.get(entity.entity_type, 0.5)
        
        # 基于属性的重要性调整
        if entity.attributes.get('is_main_character', False):
            base_score += 0.2
        
        if entity.attributes.get('has_proper_name', False):
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _calculate_semantic_consistency(self, entity: EntityMemory) -> float:
        """计算语义一致性"""
        if len(entity.context_embeddings) < 2:
            return 0.5  # 默认中等一致性
        
        # 计算上下文嵌入之间的平均余弦相似度
        similarities = []
        embeddings = entity.context_embeddings
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        if similarities:
            return sum(similarities) / len(similarities)
        
        return 0.5
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class MemoryCompressor:
    """内存压缩器"""
    
    def __init__(self, compression_threshold: int = 100, summary_length: int = 50):
        self.compression_threshold = compression_threshold
        self.summary_length = summary_length
    
    def compress_dialogue_history(self, dialogue_turns: List[DialogueTurn]) -> List[DialogueTurn]:
        """压缩对话历史"""
        if len(dialogue_turns) <= self.compression_threshold:
            return dialogue_turns
        
        # 保留最近的对话轮次
        recent_turns = dialogue_turns[-20:]
        
        # 压缩较早的对话轮次
        old_turns = dialogue_turns[:-20]
        compressed_turns = self._compress_turns(old_turns)
        
        return compressed_turns + recent_turns
    
    def _compress_turns(self, turns: List[DialogueTurn]) -> List[DialogueTurn]:
        """压缩对话轮次"""
        if not turns:
            return []
        
        # 按重要性分组
        important_turns = [t for t in turns if t.importance_score > 0.7]
        normal_turns = [t for t in turns if 0.3 <= t.importance_score <= 0.7]
        unimportant_turns = [t for t in turns if t.importance_score < 0.3]
        
        compressed = []
        
        # 保留所有重要轮次
        compressed.extend(important_turns)
        
        # 采样保留部分普通轮次
        if normal_turns:
            sample_size = max(1, len(normal_turns) // 3)
            sampled_normal = normal_turns[::len(normal_turns)//sample_size][:sample_size]
            compressed.extend(sampled_normal)
        
        # 创建不重要轮次的摘要
        if unimportant_turns:
            summary_turn = self._create_summary_turn(unimportant_turns)
            compressed.append(summary_turn)
        
        return sorted(compressed, key=lambda x: x.turn_id)
    
    def _create_summary_turn(self, turns: List[DialogueTurn]) -> DialogueTurn:
        """创建摘要轮次"""
        if not turns:
            return None
        
        # 提取关键信息
        all_entities = set()
        all_coreferences = []
        
        for turn in turns:
            all_entities.update(turn.entities)
            all_coreferences.extend(turn.coreferences)
        
        # 创建摘要
        summary = f"压缩了{len(turns)}个对话轮次，涉及{len(all_entities)}个实体"
        
        return DialogueTurn(
            turn_id=turns[0].turn_id,
            timestamp=turns[0].timestamp,
            user_input="[压缩内容]",
            system_response=summary,
            entities=list(all_entities),
            coreferences=all_coreferences,
            importance_score=0.1,
            compressed=True,
            summary=summary
        )


class MemoryManager:
    """内存管理器主类"""
    
    def __init__(self, 
                 max_entities: int = 1000,
                 max_dialogue_turns: int = 500,
                 cleanup_interval: int = 100):
        self.max_entities = max_entities
        self.max_dialogue_turns = max_dialogue_turns
        self.cleanup_interval = cleanup_interval
        
        # 存储结构
        self.entity_memories: Dict[str, EntityMemory] = {}
        self.dialogue_history: deque = deque(maxlen=max_dialogue_turns)
        
        # 组件
        self.salience_calculator = SalienceCalculator()
        self.memory_compressor = MemoryCompressor()
        
        # 统计信息
        self.operation_count = 0
        self.last_cleanup_time = time.time()
    
    def add_entity(self, entity_id: str, entity_text: str, entity_type: EntityType,
                   position: Tuple[int, int], context_embedding: Optional[List[float]] = None) -> None:
        """添加或更新实体"""
        current_time = time.time()
        
        if entity_id in self.entity_memories:
            # 更新现有实体
            entity = self.entity_memories[entity_id]
            entity.update_mention(position, context_embedding)
        else:
            # 创建新实体
            entity = EntityMemory(
                entity_id=entity_id,
                entity_text=entity_text,
                entity_type=entity_type,
                first_mention_time=current_time,
                last_mention_time=current_time,
                mention_count=1,
                mention_positions=[position]
            )
            
            if context_embedding:
                entity.context_embeddings.append(context_embedding)
            
            self.entity_memories[entity_id] = entity
        
        # 更新显著性分数
        entity.salience_score = self.salience_calculator.calculate_salience(entity, current_time)
        
        # 检查是否需要清理
        self.operation_count += 1
        if self.operation_count % self.cleanup_interval == 0:
            self._cleanup_memory()
    
    def add_dialogue_turn(self, turn: DialogueTurn) -> None:
        """添加对话轮次"""
        self.dialogue_history.append(turn)
        
        # 如果历史过长，进行压缩
        if len(self.dialogue_history) >= self.max_dialogue_turns:
            compressed_history = self.memory_compressor.compress_dialogue_history(
                list(self.dialogue_history)
            )
            self.dialogue_history.clear()
            self.dialogue_history.extend(compressed_history)
    
    def get_entity_memory(self, entity_id: str) -> Optional[EntityMemory]:
        """获取实体记忆"""
        return self.entity_memories.get(entity_id)
    
    def get_salient_entities(self, top_k: int = 10) -> List[EntityMemory]:
        """获取最显著的实体"""
        # 更新所有实体的显著性分数
        current_time = time.time()
        for entity in self.entity_memories.values():
            entity.salience_score = self.salience_calculator.calculate_salience(entity, current_time)
        
        # 按显著性排序
        sorted_entities = sorted(
            self.entity_memories.values(),
            key=lambda x: x.salience_score,
            reverse=True
        )
        
        return sorted_entities[:top_k]
    
    def get_recent_dialogue(self, num_turns: int = 10) -> List[DialogueTurn]:
        """获取最近的对话历史"""
        return list(self.dialogue_history)[-num_turns:]
    
    def _cleanup_memory(self) -> None:
        """清理内存"""
        current_time = time.time()
        
        # 如果实体数量超过限制，移除最不显著的实体
        if len(self.entity_memories) > self.max_entities:
            # 计算所有实体的显著性
            entity_salience = []
            for entity_id, entity in self.entity_memories.items():
                salience = self.salience_calculator.calculate_salience(entity, current_time)
                entity_salience.append((salience, entity_id))
            
            # 按显著性排序，移除最不显著的实体
            entity_salience.sort()
            entities_to_remove = entity_salience[:len(self.entity_memories) - self.max_entities]
            
            for _, entity_id in entities_to_remove:
                del self.entity_memories[entity_id]
        
        # 移除过期的实体（超过24小时未提及）
        expired_entities = []
        for entity_id, entity in self.entity_memories.items():
            if entity.get_age() > 24 * 3600:  # 24小时
                expired_entities.append(entity_id)
        
        for entity_id in expired_entities:
            del self.entity_memories[entity_id]
        
        self.last_cleanup_time = current_time
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        current_time = time.time()
        
        # 计算实体统计
        entity_types = defaultdict(int)
        total_mentions = 0
        avg_salience = 0
        
        for entity in self.entity_memories.values():
            entity_types[entity.entity_type.value] += 1
            total_mentions += entity.mention_count
            avg_salience += entity.salience_score
        
        if self.entity_memories:
            avg_salience /= len(self.entity_memories)
        
        return {
            'total_entities': len(self.entity_memories),
            'entity_types': dict(entity_types),
            'total_mentions': total_mentions,
            'avg_salience': avg_salience,
            'dialogue_turns': len(self.dialogue_history),
            'compressed_turns': sum(1 for turn in self.dialogue_history if turn.compressed),
            'memory_usage_mb': self._estimate_memory_usage(),
            'last_cleanup_time': self.last_cleanup_time
        }
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量（MB）"""
        # 简化的内存估算
        entity_size = len(self.entity_memories) * 1024  # 每个实体约1KB
        dialogue_size = len(self.dialogue_history) * 512  # 每个对话轮次约512B
        
        total_bytes = entity_size + dialogue_size
        return total_bytes / (1024 * 1024)  # 转换为MB
    
    def clear_memory(self) -> None:
        """清空所有内存"""
        self.entity_memories.clear()
        self.dialogue_history.clear()
        self.operation_count = 0
        self.last_cleanup_time = time.time()


# 使用示例
if __name__ == "__main__":
    # 创建内存管理器
    memory_manager = MemoryManager()
    
    # 添加实体
    memory_manager.add_entity(
        entity_id="person_1",
        entity_text="张三",
        entity_type=EntityType.PERSON,
        position=(0, 2)
    )
    
    memory_manager.add_entity(
        entity_id="pronoun_1",
        entity_text="他",
        entity_type=EntityType.PRONOUN,
        position=(1, 1)
    )
    
    # 添加对话轮次
    turn = DialogueTurn(
        turn_id=1,
        timestamp=time.time(),
        user_input="张三今天来了吗？",
        system_response="是的，他刚刚到达。",
        entities=["person_1", "pronoun_1"],
        coreferences=[("pronoun_1", "person_1")],
        importance_score=0.8
    )
    
    memory_manager.add_dialogue_turn(turn)
    
    # 获取显著实体
    salient_entities = memory_manager.get_salient_entities(top_k=5)
    print(f"显著实体数量: {len(salient_entities)}")
    
    # 获取内存统计
    stats = memory_manager.get_memory_stats()
    print(f"内存统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")