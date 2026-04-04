#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态指代消解模块

本模块提供多模态指代消解功能，支持：
- 文本-图像跨模态指代消解
- 视觉实体识别与跟踪
- 多模态特征融合
- 跨模态注意力机制

作者: AI Assistant
版本: 1.0.0
"""

import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import base64
from abc import ABC, abstractmethod


class ModalityType(Enum):
    """模态类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class EntityType(Enum):
    """实体类型枚举"""
    PERSON = "PERSON"
    OBJECT = "OBJECT"
    LOCATION = "LOCATION"
    CONCEPT = "CONCEPT"


@dataclass
class BoundingBox:
    """边界框数据结构"""
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    
    def area(self) -> float:
        """计算面积"""
        return self.width * self.height
    
    def iou(self, other: 'BoundingBox') -> float:
        """计算IoU（交并比）"""
        # 计算交集
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area() + other.area() - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class VisualEntity:
    """视觉实体数据结构"""
    entity_id: str
    entity_type: EntityType
    bounding_box: BoundingBox
    visual_features: np.ndarray
    confidence: float
    frame_id: Optional[int] = None
    timestamp: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class TextualMention:
    """文本指代数据结构"""
    mention_id: str
    text: str
    start_pos: int
    end_pos: int
    entity_type: EntityType
    confidence: float
    linguistic_features: Dict[str, Any] = field(default_factory=dict)
    context: Optional[str] = None


@dataclass
class MultimodalEntity:
    """多模态实体数据结构"""
    entity_id: str
    entity_type: EntityType
    textual_mentions: List[TextualMention] = field(default_factory=list)
    visual_entities: List[VisualEntity] = field(default_factory=list)
    cross_modal_confidence: float = 0.0
    last_update_time: float = field(default_factory=time.time)
    
    def add_textual_mention(self, mention: TextualMention) -> None:
        """添加文本指代"""
        self.textual_mentions.append(mention)
        self.last_update_time = time.time()
    
    def add_visual_entity(self, visual_entity: VisualEntity) -> None:
        """添加视觉实体"""
        self.visual_entities.append(visual_entity)
        self.last_update_time = time.time()


class VisualFeatureExtractor(ABC):
    """视觉特征提取器抽象基类"""
    
    @abstractmethod
    def extract_features(self, image_data: Union[str, np.ndarray]) -> np.ndarray:
        """提取视觉特征"""
        pass
    
    @abstractmethod
    def detect_objects(self, image_data: Union[str, np.ndarray]) -> List[VisualEntity]:
        """检测视觉对象"""
        pass


class MockVisualFeatureExtractor(VisualFeatureExtractor):
    """模拟视觉特征提取器（用于演示）"""
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
    
    def extract_features(self, image_data: Union[str, np.ndarray]) -> np.ndarray:
        """提取视觉特征（模拟实现）"""
        # 模拟特征提取，实际应使用CNN模型
        return np.random.rand(self.feature_dim).astype(np.float32)
    
    def detect_objects(self, image_data: Union[str, np.ndarray]) -> List[VisualEntity]:
        """检测视觉对象（模拟实现）"""
        # 模拟对象检测，实际应使用YOLO、R-CNN等模型
        objects = []
        
        # 模拟检测到的对象
        mock_objects = [
            ("person_1", EntityType.PERSON, (0.1, 0.2, 0.3, 0.4), 0.9),
            ("object_1", EntityType.OBJECT, (0.5, 0.3, 0.2, 0.3), 0.8),
            ("location_1", EntityType.LOCATION, (0.0, 0.0, 1.0, 0.5), 0.7)
        ]
        
        for obj_id, obj_type, bbox_coords, confidence in mock_objects:
            bbox = BoundingBox(*bbox_coords, confidence)
            visual_features = self.extract_features(image_data)
            
            visual_entity = VisualEntity(
                entity_id=obj_id,
                entity_type=obj_type,
                bounding_box=bbox,
                visual_features=visual_features,
                confidence=confidence
            )
            
            objects.append(visual_entity)
        
        return objects


class CrossModalAttention:
    """跨模态注意力机制"""
    
    def __init__(self, text_dim: int = 768, visual_dim: int = 512, hidden_dim: int = 256):
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        
        # 模拟注意力权重矩阵
        self.text_projection = np.random.randn(text_dim, hidden_dim).astype(np.float32)
        self.visual_projection = np.random.randn(visual_dim, hidden_dim).astype(np.float32)
        self.attention_weights = np.random.randn(hidden_dim, 1).astype(np.float32)
    
    def compute_attention(self, text_features: np.ndarray, visual_features: np.ndarray) -> Tuple[float, np.ndarray]:
        """计算跨模态注意力"""
        # 投影到共同空间
        text_proj = np.dot(text_features, self.text_projection)
        visual_proj = np.dot(visual_features, self.visual_projection)
        
        # 计算注意力分数
        combined_features = text_proj + visual_proj
        attention_score = np.dot(combined_features, self.attention_weights).item()
        
        # 应用softmax归一化
        attention_weight = 1.0 / (1.0 + np.exp(-attention_score))  # sigmoid
        
        # 融合特征
        fused_features = attention_weight * text_proj + (1 - attention_weight) * visual_proj
        
        return attention_weight, fused_features


class MultimodalFeatureFusion:
    """多模态特征融合器"""
    
    def __init__(self, fusion_strategy: str = "attention"):
        self.fusion_strategy = fusion_strategy
        self.cross_modal_attention = CrossModalAttention()
    
    def fuse_features(self, textual_features: np.ndarray, visual_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """融合多模态特征"""
        if self.fusion_strategy == "concatenation":
            return self._concatenate_features(textual_features, visual_features)
        elif self.fusion_strategy == "attention":
            return self._attention_fusion(textual_features, visual_features)
        elif self.fusion_strategy == "weighted_sum":
            return self._weighted_sum_fusion(textual_features, visual_features)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def _concatenate_features(self, text_feat: np.ndarray, visual_feat: np.ndarray) -> Tuple[np.ndarray, float]:
        """特征拼接融合"""
        fused = np.concatenate([text_feat, visual_feat])
        confidence = 0.8  # 固定置信度
        return fused, confidence
    
    def _attention_fusion(self, text_feat: np.ndarray, visual_feat: np.ndarray) -> Tuple[np.ndarray, float]:
        """注意力机制融合"""
        attention_weight, fused_features = self.cross_modal_attention.compute_attention(text_feat, visual_feat)
        return fused_features, attention_weight
    
    def _weighted_sum_fusion(self, text_feat: np.ndarray, visual_feat: np.ndarray, 
                           text_weight: float = 0.6) -> Tuple[np.ndarray, float]:
        """加权求和融合"""
        # 确保特征维度一致
        min_dim = min(len(text_feat), len(visual_feat))
        text_feat_norm = text_feat[:min_dim]
        visual_feat_norm = visual_feat[:min_dim]
        
        fused = text_weight * text_feat_norm + (1 - text_weight) * visual_feat_norm
        confidence = 0.7
        return fused, confidence


class MultimodalCoreferenceResolver:
    """多模态指代消解器"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.visual_extractor = MockVisualFeatureExtractor()
        self.feature_fusion = MultimodalFeatureFusion(fusion_strategy="attention")
        
        # 存储多模态实体
        self.multimodal_entities: Dict[str, MultimodalEntity] = {}
        self.entity_counter = 0
    
    def process_text_mention(self, mention: TextualMention, 
                           text_features: Optional[np.ndarray] = None) -> str:
        """处理文本指代"""
        if text_features is None:
            # 模拟文本特征提取
            text_features = np.random.rand(768).astype(np.float32)
        
        # 查找匹配的多模态实体
        best_match_id = self._find_best_textual_match(mention, text_features)
        
        if best_match_id:
            # 添加到现有实体
            self.multimodal_entities[best_match_id].add_textual_mention(mention)
            return best_match_id
        else:
            # 创建新的多模态实体
            entity_id = f"multimodal_entity_{self.entity_counter}"
            self.entity_counter += 1
            
            new_entity = MultimodalEntity(
                entity_id=entity_id,
                entity_type=mention.entity_type
            )
            new_entity.add_textual_mention(mention)
            
            self.multimodal_entities[entity_id] = new_entity
            return entity_id
    
    def process_visual_input(self, image_data: Union[str, np.ndarray], 
                           frame_id: Optional[int] = None) -> List[str]:
        """处理视觉输入"""
        # 检测视觉实体
        visual_entities = self.visual_extractor.detect_objects(image_data)
        
        processed_entity_ids = []
        
        for visual_entity in visual_entities:
            if frame_id is not None:
                visual_entity.frame_id = frame_id
            
            # 查找匹配的多模态实体
            best_match_id = self._find_best_visual_match(visual_entity)
            
            if best_match_id:
                # 添加到现有实体
                self.multimodal_entities[best_match_id].add_visual_entity(visual_entity)
                processed_entity_ids.append(best_match_id)
            else:
                # 创建新的多模态实体
                entity_id = f"multimodal_entity_{self.entity_counter}"
                self.entity_counter += 1
                
                new_entity = MultimodalEntity(
                    entity_id=entity_id,
                    entity_type=visual_entity.entity_type
                )
                new_entity.add_visual_entity(visual_entity)
                
                self.multimodal_entities[entity_id] = new_entity
                processed_entity_ids.append(entity_id)
        
        return processed_entity_ids
    
    def resolve_cross_modal_coreference(self, text_mention_id: str, 
                                       visual_entity_id: str) -> Tuple[bool, float]:
        """解决跨模态指代关系"""
        # 获取文本和视觉特征
        text_entity = None
        visual_entity = None
        
        for entity in self.multimodal_entities.values():
            for mention in entity.textual_mentions:
                if mention.mention_id == text_mention_id:
                    text_entity = entity
                    break
            
            for visual in entity.visual_entities:
                if visual.entity_id == visual_entity_id:
                    visual_entity = entity
                    break
        
        if not text_entity or not visual_entity:
            return False, 0.0
        
        # 提取特征进行比较
        text_features = self._get_entity_text_features(text_entity)
        visual_features = self._get_entity_visual_features(visual_entity)
        
        # 计算跨模态相似度
        similarity = self._compute_cross_modal_similarity(text_features, visual_features)
        
        # 判断是否为指代关系
        is_coreference = similarity > self.similarity_threshold
        
        if is_coreference:
            # 合并实体
            self._merge_entities(text_entity.entity_id, visual_entity.entity_id)
        
        return is_coreference, similarity
    
    def _find_best_textual_match(self, mention: TextualMention, 
                               text_features: np.ndarray) -> Optional[str]:
        """查找最佳文本匹配"""
        best_score = 0.0
        best_entity_id = None
        
        for entity_id, entity in self.multimodal_entities.items():
            if entity.entity_type != mention.entity_type:
                continue
            
            if not entity.textual_mentions:
                continue
            
            # 计算与现有文本指代的相似度
            entity_text_features = self._get_entity_text_features(entity)
            similarity = self._cosine_similarity(text_features, entity_text_features)
            
            if similarity > best_score and similarity > self.similarity_threshold:
                best_score = similarity
                best_entity_id = entity_id
        
        return best_entity_id
    
    def _find_best_visual_match(self, visual_entity: VisualEntity) -> Optional[str]:
        """查找最佳视觉匹配"""
        best_score = 0.0
        best_entity_id = None
        
        for entity_id, entity in self.multimodal_entities.items():
            if entity.entity_type != visual_entity.entity_type:
                continue
            
            if not entity.visual_entities:
                continue
            
            # 计算与现有视觉实体的相似度
            entity_visual_features = self._get_entity_visual_features(entity)
            similarity = self._cosine_similarity(visual_entity.visual_features, entity_visual_features)
            
            # 考虑空间位置相似度（对于视频序列）
            spatial_similarity = self._compute_spatial_similarity(visual_entity, entity)
            
            combined_similarity = 0.7 * similarity + 0.3 * spatial_similarity
            
            if combined_similarity > best_score and combined_similarity > self.similarity_threshold:
                best_score = combined_similarity
                best_entity_id = entity_id
        
        return best_entity_id
    
    def _compute_cross_modal_similarity(self, text_features: np.ndarray, 
                                      visual_features: np.ndarray) -> float:
        """计算跨模态相似度"""
        # 使用特征融合器计算相似度
        fused_features, confidence = self.feature_fusion.fuse_features(text_features, visual_features)
        
        # 基于融合特征和置信度计算最终相似度
        feature_norm = np.linalg.norm(fused_features)
        normalized_similarity = confidence * (1.0 / (1.0 + np.exp(-feature_norm)))
        
        return float(normalized_similarity)
    
    def _get_entity_text_features(self, entity: MultimodalEntity) -> np.ndarray:
        """获取实体的文本特征"""
        # 简化实现：返回随机特征
        # 实际应该基于实体的所有文本指代计算平均特征
        return np.random.rand(768).astype(np.float32)
    
    def _get_entity_visual_features(self, entity: MultimodalEntity) -> np.ndarray:
        """获取实体的视觉特征"""
        if not entity.visual_entities:
            return np.zeros(512, dtype=np.float32)
        
        # 计算所有视觉实体特征的平均值
        features = [ve.visual_features for ve in entity.visual_entities]
        return np.mean(features, axis=0)
    
    def _compute_spatial_similarity(self, visual_entity: VisualEntity, 
                                  entity: MultimodalEntity) -> float:
        """计算空间位置相似度"""
        if not entity.visual_entities:
            return 0.0
        
        max_iou = 0.0
        for existing_visual in entity.visual_entities:
            iou = visual_entity.bounding_box.iou(existing_visual.bounding_box)
            max_iou = max(max_iou, iou)
        
        return max_iou
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _merge_entities(self, entity_id1: str, entity_id2: str) -> None:
        """合并两个实体"""
        if entity_id1 not in self.multimodal_entities or entity_id2 not in self.multimodal_entities:
            return
        
        entity1 = self.multimodal_entities[entity_id1]
        entity2 = self.multimodal_entities[entity_id2]
        
        # 将entity2的所有指代和视觉实体合并到entity1
        entity1.textual_mentions.extend(entity2.textual_mentions)
        entity1.visual_entities.extend(entity2.visual_entities)
        entity1.last_update_time = time.time()
        
        # 删除entity2
        del self.multimodal_entities[entity_id2]
    
    def get_entity_summary(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """获取实体摘要信息"""
        if entity_id not in self.multimodal_entities:
            return None
        
        entity = self.multimodal_entities[entity_id]
        
        return {
            'entity_id': entity.entity_id,
            'entity_type': entity.entity_type.value,
            'textual_mentions_count': len(entity.textual_mentions),
            'visual_entities_count': len(entity.visual_entities),
            'cross_modal_confidence': entity.cross_modal_confidence,
            'last_update_time': entity.last_update_time,
            'textual_mentions': [
                {
                    'mention_id': m.mention_id,
                    'text': m.text,
                    'confidence': m.confidence
                } for m in entity.textual_mentions
            ],
            'visual_entities': [
                {
                    'entity_id': v.entity_id,
                    'confidence': v.confidence,
                    'bounding_box': {
                        'x': v.bounding_box.x,
                        'y': v.bounding_box.y,
                        'width': v.bounding_box.width,
                        'height': v.bounding_box.height
                    },
                    'frame_id': v.frame_id
                } for v in entity.visual_entities
            ]
        }
    
    def get_all_entities_summary(self) -> List[Dict[str, Any]]:
        """获取所有实体的摘要信息"""
        summaries = []
        for entity_id in self.multimodal_entities:
            summary = self.get_entity_summary(entity_id)
            if summary:
                summaries.append(summary)
        return summaries


# 使用示例
if __name__ == "__main__":
    # 创建多模态指代消解器
    resolver = MultimodalCoreferenceResolver()
    
    # 处理文本指代
    text_mention = TextualMention(
        mention_id="mention_1",
        text="这个人",
        start_pos=0,
        end_pos=3,
        entity_type=EntityType.PERSON,
        confidence=0.9,
        context="这个人正在走路"
    )
    
    entity_id = resolver.process_text_mention(text_mention)
    print(f"处理文本指代，实体ID: {entity_id}")
    
    # 处理视觉输入（模拟图像数据）
    mock_image = np.random.rand(224, 224, 3)  # 模拟图像
    visual_entity_ids = resolver.process_visual_input(mock_image, frame_id=1)
    print(f"处理视觉输入，检测到实体: {visual_entity_ids}")
    
    # 获取所有实体摘要
    all_entities = resolver.get_all_entities_summary()
    print(f"\n所有实体摘要:")
    print(json.dumps(all_entities, indent=2, ensure_ascii=False))