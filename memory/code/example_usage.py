#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮指代消解对话系统使用示例

本文件展示如何使用各个模块构建完整的多轮指代消解对话系统。

作者: AI Assistant
版本: 1.0.0
"""

import asyncio
import time
from typing import List, Dict, Any

# 导入各个模块
from performance_optimization import PerformanceOptimizer, Entity
from memory_management import MemoryManager, EntityType, DialogueTurn
from multimodal_coref import (
    MultimodalCoreferenceResolver, 
    TextualMention, 
    EntityType as MultimodalEntityType
)


class IntegratedDialogueSystem:
    """集成的多轮指代消解对话系统"""
    
    def __init__(self):
        # 初始化各个组件
        self.performance_optimizer = PerformanceOptimizer()
        self.memory_manager = MemoryManager()
        self.multimodal_resolver = MultimodalCoreferenceResolver()
        
        # 对话状态
        self.current_turn_id = 0
        self.conversation_history = []
    
    async def process_user_input(self, user_input: str, image_data=None) -> str:
        """处理用户输入"""
        self.current_turn_id += 1
        start_time = time.time()
        
        print(f"\n=== 处理第 {self.current_turn_id} 轮对话 ===")
        print(f"用户输入: {user_input}")
        
        # 1. 实体识别（使用缓存优化）
        entities = await self._extract_entities_with_cache(user_input)
        print(f"识别到的实体: {[e.text for e in entities]}")
        
        # 2. 处理多模态输入
        visual_entity_ids = []
        if image_data is not None:
            visual_entity_ids = self.multimodal_resolver.process_visual_input(image_data)
            print(f"视觉实体: {visual_entity_ids}")
        
        # 3. 指代消解
        resolved_entities = await self._resolve_coreferences(entities)
        print(f"指代消解结果: {resolved_entities}")
        
        # 4. 更新记忆管理
        self._update_memory(entities, resolved_entities)
        
        # 5. 生成响应
        response = self._generate_response(user_input, resolved_entities)
        
        # 6. 记录对话轮次
        processing_time = time.time() - start_time
        self._record_dialogue_turn(user_input, response, entities, processing_time)
        
        print(f"系统响应: {response}")
        print(f"处理时间: {processing_time:.3f}秒")
        
        return response
    
    async def _extract_entities_with_cache(self, text: str) -> List[Entity]:
        """带缓存的实体识别"""
        # 检查缓存
        cached_entities = self.performance_optimizer.get_cached_entities(text)
        if cached_entities:
            print("使用缓存的实体识别结果")
            return cached_entities
        
        # 模拟实体识别
        entities = []
        
        # 简单的实体识别逻辑（实际应使用NLP模型）
        entity_patterns = {
            '张三': ('PERSON', 0.9),
            '李四': ('PERSON', 0.9),
            '他': ('PRONOUN', 0.8),
            '她': ('PRONOUN', 0.8),
            '公司': ('ORG', 0.85),
            '北京': ('LOC', 0.9)
        }
        
        for pattern, (entity_type, confidence) in entity_patterns.items():
            if pattern in text:
                start_pos = text.find(pattern)
                entity = Entity(
                    id=f"entity_{len(entities)}",
                    text=pattern,
                    type=entity_type,
                    position=(start_pos, start_pos + len(pattern)),
                    confidence=confidence
                )
                entities.append(entity)
        
        # 缓存结果
        self.performance_optimizer.cache_entities(text, entities)
        
        return entities
    
    async def _resolve_coreferences(self, entities: List[Entity]) -> Dict[str, str]:
        """指代消解"""
        resolved = {}
        
        # 获取显著实体作为候选
        salient_entities = self.memory_manager.get_salient_entities(top_k=10)
        
        for entity in entities:
            if entity.type == 'PRONOUN':
                # 对代词进行指代消解
                candidates = [e for e in salient_entities 
                            if e.entity_type.value in ['PERSON', 'ORG', 'LOC']]
                
                if candidates:
                    # 简化的指代消解：选择最显著的候选实体
                    best_candidate = max(candidates, key=lambda x: x.salience_score)
                    resolved[entity.id] = best_candidate.entity_id
                    print(f"指代消解: {entity.text} -> {best_candidate.entity_text}")
        
        return resolved
    
    def _update_memory(self, entities: List[Entity], resolved_entities: Dict[str, str]):
        """更新记忆管理"""
        for entity in entities:
            # 转换实体类型
            memory_entity_type = self._convert_entity_type(entity.type)
            
            # 添加到记忆管理器
            self.memory_manager.add_entity(
                entity_id=entity.id,
                entity_text=entity.text,
                entity_type=memory_entity_type,
                position=entity.position
            )
    
    def _convert_entity_type(self, entity_type_str: str) -> EntityType:
        """转换实体类型"""
        type_mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'LOC': EntityType.LOCATION,
            'PRONOUN': EntityType.PERSON,  # 简化处理
            'MISC': EntityType.MISCELLANEOUS
        }
        return type_mapping.get(entity_type_str, EntityType.MISCELLANEOUS)
    
    def _generate_response(self, user_input: str, resolved_entities: Dict[str, str]) -> str:
        """生成响应"""
        # 简化的响应生成逻辑
        if '你好' in user_input:
            return "你好！我是多轮指代消解对话系统，很高兴为您服务。"
        elif '张三' in user_input:
            return "我了解您提到了张三，有什么关于他的问题吗？"
        elif '他' in user_input and resolved_entities:
            # 使用指代消解结果
            return f"我明白您指的是之前提到的实体，让我为您处理相关信息。"
        else:
            return "我理解了您的输入，正在为您处理相关信息。"
    
    def _record_dialogue_turn(self, user_input: str, response: str, 
                            entities: List[Entity], processing_time: float):
        """记录对话轮次"""
        # 计算重要性分数
        importance_score = self._calculate_turn_importance(user_input, entities)
        
        turn = DialogueTurn(
            turn_id=self.current_turn_id,
            timestamp=time.time(),
            user_input=user_input,
            system_response=response,
            entities=[e.id for e in entities],
            coreferences=[],  # 简化处理
            importance_score=importance_score
        )
        
        self.memory_manager.add_dialogue_turn(turn)
        self.conversation_history.append({
            'turn_id': self.current_turn_id,
            'user_input': user_input,
            'response': response,
            'processing_time': processing_time,
            'entities_count': len(entities)
        })
    
    def _calculate_turn_importance(self, user_input: str, entities: List[Entity]) -> float:
        """计算对话轮次重要性"""
        importance = 0.5  # 基础重要性
        
        # 基于实体数量调整
        importance += min(0.3, len(entities) * 0.1)
        
        # 基于关键词调整
        important_keywords = ['重要', '关键', '问题', '帮助']
        for keyword in important_keywords:
            if keyword in user_input:
                importance += 0.1
        
        return min(1.0, importance)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        performance_stats = self.performance_optimizer.get_performance_stats()
        memory_stats = self.memory_manager.get_memory_stats()
        multimodal_stats = {
            'multimodal_entities_count': len(self.multimodal_resolver.multimodal_entities)
        }
        
        return {
            'conversation_turns': len(self.conversation_history),
            'current_turn_id': self.current_turn_id,
            'performance': performance_stats,
            'memory': memory_stats,
            'multimodal': multimodal_stats
        }
    
    def reset_system(self):
        """重置系统状态"""
        self.current_turn_id = 0
        self.conversation_history.clear()
        self.performance_optimizer.clear_caches()
        self.performance_optimizer.reset_stats()
        self.memory_manager.clear_memory()
        self.multimodal_resolver.multimodal_entities.clear()
        print("系统状态已重置")


async def main():
    """主函数 - 演示系统使用"""
    print("=== 多轮指代消解对话系统演示 ===")
    
    # 创建系统实例
    dialogue_system = IntegratedDialogueSystem()
    
    # 模拟对话序列
    conversation_examples = [
        "你好，我想了解一下张三的情况。",
        "他今天来公司了吗？",
        "他的工作表现怎么样？",
        "李四和他是同事吗？",
        "她也在同一个部门工作吗？"
    ]
    
    # 处理对话
    for user_input in conversation_examples:
        await dialogue_system.process_user_input(user_input)
        await asyncio.sleep(0.5)  # 模拟思考时间
    
    # 显示系统统计信息
    print("\n=== 系统统计信息 ===")
    stats = dialogue_system.get_system_stats()
    
    print(f"对话轮次总数: {stats['conversation_turns']}")
    print(f"当前轮次ID: {stats['current_turn_id']}")
    print(f"实体总数: {stats['memory']['total_entities']}")
    print(f"缓存命中次数: {stats['performance'].get('cache_hits', 0)}")
    print(f"缓存未命中次数: {stats['performance'].get('cache_misses', 0)}")
    print(f"内存使用量: {stats['memory']['memory_usage_mb']:.2f} MB")
    
    # 显示显著实体
    print("\n=== 显著实体 ===")
    salient_entities = dialogue_system.memory_manager.get_salient_entities(top_k=5)
    for i, entity in enumerate(salient_entities, 1):
        print(f"{i}. {entity.entity_text} (类型: {entity.entity_type.value}, "
              f"显著性: {entity.salience_score:.3f}, 提及次数: {entity.mention_count})")
    
    # 演示多模态功能
    print("\n=== 多模态功能演示 ===")
    import numpy as np
    mock_image = np.random.rand(224, 224, 3)  # 模拟图像
    
    await dialogue_system.process_user_input(
        "这张图片中的人是谁？", 
        image_data=mock_image
    )
    
    # 最终统计
    final_stats = dialogue_system.get_system_stats()
    print(f"\n最终处理的对话轮次: {final_stats['conversation_turns']}")
    print(f"多模态实体数量: {final_stats['multimodal']['multimodal_entities_count']}")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())