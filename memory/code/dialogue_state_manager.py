# -*- coding: utf-8 -*-
"""
对话状态管理器

本模块实现了智能的对话状态管理，包括状态跟踪、显著性更新、上下文压缩和分层记忆管理等功能。
"""

import time
import math
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from entity_recognition import Entity
from coreference_resolution import CoreferenceResult, Mention
import json

@dataclass
class DialogueTurn:
    """对话轮次数据类"""
    turn_id: int
    user_input: str
    system_response: str
    entities: List[Entity]
    coreferences: List[CoreferenceResult]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EntityState:
    """实体状态数据类"""
    entity: Entity
    salience: float = 0.5
    last_mention_turn: int = 0
    mention_count: int = 0
    mention_history: List[int] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def update_mention(self, turn_id: int):
        """更新提及信息"""
        self.last_mention_turn = turn_id
        self.mention_count += 1
        self.mention_history.append(turn_id)
        
        # 保持历史记录在合理范围内
        if len(self.mention_history) > 10:
            self.mention_history = self.mention_history[-10:]

class DialogueStateTracker:
    """对话状态跟踪器"""
    
    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.turns: List[DialogueTurn] = []
        self.current_turn_id = 0
        self.entity_states: Dict[str, EntityState] = {}
        self.global_context: Dict[str, Any] = {}
    
    def add_turn(self, user_input: str, system_response: str, 
                entities: List[Entity], coreferences: List[CoreferenceResult]):
        """添加新的对话轮次"""
        self.current_turn_id += 1
        
        turn = DialogueTurn(
            turn_id=self.current_turn_id,
            user_input=user_input,
            system_response=system_response,
            entities=entities,
            coreferences=coreferences
        )
        
        self.turns.append(turn)
        
        # 维护最大轮次限制
        if len(self.turns) > self.max_turns:
            removed_turn = self.turns.pop(0)
            self._cleanup_old_turn(removed_turn)
        
        # 更新实体状态
        self._update_entity_states(entities)
    
    def _update_entity_states(self, entities: List[Entity]):
        """更新实体状态"""
        for entity in entities:
            if entity.id not in self.entity_states:
                self.entity_states[entity.id] = EntityState(entity=entity)
            
            self.entity_states[entity.id].update_mention(self.current_turn_id)
    
    def _cleanup_old_turn(self, turn: DialogueTurn):
        """清理旧轮次的数据"""
        # 移除不再被提及的实体
        entities_to_remove = []
        for entity_id, state in self.entity_states.items():
            if state.last_mention_turn <= turn.turn_id:
                entities_to_remove.append(entity_id)
        
        for entity_id in entities_to_remove:
            del self.entity_states[entity_id]
    
    def get_recent_turns(self, n: int = 5) -> List[DialogueTurn]:
        """获取最近的n轮对话"""
        return self.turns[-n:] if len(self.turns) >= n else self.turns
    
    def get_all_entities(self) -> List[Entity]:
        """获取所有活跃实体"""
        return [state.entity for state in self.entity_states.values()]
    
    def get_entity_state(self, entity_id: str) -> Optional[EntityState]:
        """获取实体状态"""
        return self.entity_states.get(entity_id)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """获取上下文摘要"""
        return {
            'current_turn': self.current_turn_id,
            'total_turns': len(self.turns),
            'active_entities': len(self.entity_states),
            'recent_entities': [e.text for e in self.get_all_entities()[-5:]]
        }

class SalienceUpdater:
    """显著性更新器"""
    
    def __init__(self, decay_factor: float = 0.1, recency_weight: float = 0.3):
        self.decay_factor = decay_factor
        self.recency_weight = recency_weight
    
    def update_salience(self, entity_states: Dict[str, EntityState], 
                       current_turn: int):
        """更新所有实体的显著性"""
        for entity_id, state in entity_states.items():
            new_salience = self._calculate_salience(state, current_turn)
            state.salience = new_salience
    
    def _calculate_salience(self, state: EntityState, current_turn: int) -> float:
        """计算单个实体的显著性"""
        # 基础显著性（基于提及频率）
        frequency_score = min(state.mention_count / 10.0, 1.0)
        
        # 时间衰减（距离上次提及的时间）
        time_decay = math.exp(-self.decay_factor * (current_turn - state.last_mention_turn))
        
        # 最近性权重
        recency_score = 1.0 if state.last_mention_turn == current_turn else time_decay
        
        # 综合计算
        salience = (
            frequency_score * (1 - self.recency_weight) + 
            recency_score * self.recency_weight
        )
        
        return max(0.0, min(1.0, salience))

class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(self, max_context_length: int = 2000):
        self.max_context_length = max_context_length
    
    def compress(self, turns: List[DialogueTurn]) -> Dict[str, Any]:
        """压缩对话上下文"""
        if not turns:
            return {}
        
        # 计算当前上下文长度
        total_length = sum(len(turn.user_input) + len(turn.system_response) 
                          for turn in turns)
        
        if total_length <= self.max_context_length:
            return self._create_full_context(turns)
        
        # 需要压缩
        return self._create_compressed_context(turns)
    
    def _create_full_context(self, turns: List[DialogueTurn]) -> Dict[str, Any]:
        """创建完整上下文"""
        return {
            'type': 'full',
            'turns': turns,
            'summary': f"包含{len(turns)}轮完整对话",
            'compression_ratio': 1.0
        }
    
    def _create_compressed_context(self, turns: List[DialogueTurn]) -> Dict[str, Any]:
        """创建压缩上下文"""
        # 保留最近的几轮完整对话
        recent_turns = turns[-3:] if len(turns) >= 3 else turns
        older_turns = turns[:-3] if len(turns) > 3 else []
        
        # 压缩较旧的对话
        compressed_summary = self._summarize_turns(older_turns)
        
        original_length = sum(len(turn.user_input) + len(turn.system_response) 
                             for turn in turns)
        compressed_length = (len(compressed_summary) + 
                           sum(len(turn.user_input) + len(turn.system_response) 
                               for turn in recent_turns))
        
        return {
            'type': 'compressed',
            'recent_turns': recent_turns,
            'summary': compressed_summary,
            'compression_ratio': compressed_length / original_length if original_length > 0 else 1.0
        }
    
    def _summarize_turns(self, turns: List[DialogueTurn]) -> str:
        """总结对话轮次"""
        if not turns:
            return ""
        
        # 提取关键实体
        all_entities = set()
        for turn in turns:
            for entity in turn.entities:
                all_entities.add(entity.text)
        
        # 创建简要总结
        entity_list = ", ".join(list(all_entities)[:5])
        summary = f"前{len(turns)}轮对话涉及实体: {entity_list}"
        
        if len(all_entities) > 5:
            summary += f"等{len(all_entities)}个实体"
        
        return summary

class HierarchicalMemoryManager:
    """分层记忆管理器"""
    
    def __init__(self):
        # 短期记忆：最近几轮对话
        self.short_term_memory = deque(maxlen=5)
        
        # 中期记忆：重要实体和关系
        self.medium_term_memory = {}
        
        # 长期记忆：持久化的关键信息
        self.long_term_memory = {}
    
    def organize_memory(self, compressed_context: Dict[str, Any], 
                      new_entities: List[Entity]):
        """组织分层记忆"""
        # 更新短期记忆
        if 'recent_turns' in compressed_context:
            for turn in compressed_context['recent_turns']:
                self.short_term_memory.append(turn)
        
        # 更新中期记忆
        self._update_medium_term_memory(new_entities)
        
        # 更新长期记忆
        self._update_long_term_memory(compressed_context)
    
    def _update_medium_term_memory(self, entities: List[Entity]):
        """更新中期记忆"""
        for entity in entities:
            if entity.id not in self.medium_term_memory:
                self.medium_term_memory[entity.id] = {
                    'entity': entity,
                    'importance': 0.5,
                    'relations': [],
                    'last_updated': time.time()
                }
            else:
                # 更新重要性
                self.medium_term_memory[entity.id]['importance'] = min(
                    1.0, self.medium_term_memory[entity.id]['importance'] + 0.1
                )
                self.medium_term_memory[entity.id]['last_updated'] = time.time()
    
    def _update_long_term_memory(self, context: Dict[str, Any]):
        """更新长期记忆"""
        # 存储压缩后的上下文摘要
        if 'summary' in context:
            timestamp = time.time()
            self.long_term_memory[f"summary_{timestamp}"] = {
                'content': context['summary'],
                'timestamp': timestamp,
                'type': 'context_summary'
            }
        
        # 清理过旧的长期记忆
        self._cleanup_long_term_memory()
    
    def _cleanup_long_term_memory(self, max_entries: int = 100):
        """清理长期记忆"""
        if len(self.long_term_memory) > max_entries:
            # 按时间戳排序，保留最新的条目
            sorted_items = sorted(
                self.long_term_memory.items(),
                key=lambda x: x[1].get('timestamp', 0),
                reverse=True
            )
            
            self.long_term_memory = dict(sorted_items[:max_entries])
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        return {
            'short_term_turns': len(self.short_term_memory),
            'medium_term_entities': len(self.medium_term_memory),
            'long_term_entries': len(self.long_term_memory)
        }

class IntelligentStateManager:
    """智能状态管理器"""
    
    def __init__(self):
        self.dialogue_tracker = DialogueStateTracker()
        self.salience_updater = SalienceUpdater()
        self.context_compressor = ContextCompressor()
        self.memory_manager = HierarchicalMemoryManager()
    
    def update_state(self, user_input: str, system_response: str,
                    entities: List[Entity], 
                    coreferences: List[CoreferenceResult]) -> Dict[str, Any]:
        """更新对话状态"""
        # 1. 更新对话跟踪器
        self.dialogue_tracker.add_turn(
            user_input, system_response, entities, coreferences
        )
        
        # 2. 更新实体显著性
        self.salience_updater.update_salience(
            self.dialogue_tracker.entity_states,
            self.dialogue_tracker.current_turn_id
        )
        
        # 3. 压缩上下文
        recent_turns = self.dialogue_tracker.get_recent_turns()
        compressed_context = self.context_compressor.compress(recent_turns)
        
        # 4. 组织分层记忆
        self.memory_manager.organize_memory(compressed_context, entities)
        
        return self.get_current_state()
    
    def get_current_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'dialogue_summary': self.dialogue_tracker.get_context_summary(),
            'entity_salience': {
                entity_id: state.salience 
                for entity_id, state in self.dialogue_tracker.entity_states.items()
            },
            'entity_last_mention': {
                entity_id: state.last_mention_turn 
                for entity_id, state in self.dialogue_tracker.entity_states.items()
            },
            'current_turn': self.dialogue_tracker.current_turn_id,
            'memory_summary': self.memory_manager.get_memory_summary(),
            'active_entities': self.dialogue_tracker.get_all_entities()
        }
    
    def get_context_for_resolution(self) -> Dict[str, Any]:
        """获取用于指代消解的上下文"""
        current_state = self.get_current_state()
        
        # 添加额外的上下文信息
        recent_turns = self.dialogue_tracker.get_recent_turns(3)
        current_state['recent_dialogue'] = [
            {
                'turn_id': turn.turn_id,
                'user_input': turn.user_input,
                'entities': [e.text for e in turn.entities]
            }
            for turn in recent_turns
        ]
        
        return current_state
    
    def reset_dialogue(self):
        """重置对话状态"""
        self.dialogue_tracker = DialogueStateTracker()
        # 保留长期记忆，重置其他组件
        long_term_backup = self.memory_manager.long_term_memory.copy()
        self.memory_manager = HierarchicalMemoryManager()
        self.memory_manager.long_term_memory = long_term_backup

# 使用示例
if __name__ == "__main__":
    # 创建状态管理器
    state_manager = IntelligentStateManager()
    
    # 模拟对话
    entities1 = [
        Entity(id="1", text="穆勒", type="PERSON", start=0, end=2, gender="male")
    ]
    
    entities2 = [
        Entity(id="1", text="他", type="PRONOUN", start=0, end=1, gender="male")
    ]
    
    coreferences = []
    
    # 第一轮对话
    state1 = state_manager.update_state(
        "穆勒这个赛季表现怎么样？",
        "穆勒本赛季表现出色...",
        entities1,
        coreferences
    )
    
    # 第二轮对话
    state2 = state_manager.update_state(
        "他进了几个球？",
        "他本赛季进了15个球。",
        entities2,
        coreferences
    )
    
    print("对话状态:")
    print(json.dumps(state2, indent=2, ensure_ascii=False, default=str))