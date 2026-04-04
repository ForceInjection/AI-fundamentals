# -*- coding: utf-8 -*-
"""
指代消解核心引擎

本模块实现了高级的指代消解层，包括候选实体筛选、特征提取、概率计算和消歧决策等功能。
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from entity_recognition import Entity
import math

class PronounType(Enum):
    """代词类型枚举"""
    PERSONAL = "personal"  # 人称代词：他、她、它
    DEMONSTRATIVE = "demonstrative"  # 指示代词：这、那
    POSSESSIVE = "possessive"  # 物主代词：他的、她的
    REFLEXIVE = "reflexive"  # 反身代词：自己

@dataclass
class Mention:
    """指代词数据类"""
    text: str
    type: PronounType
    start: int
    end: int
    gender: Optional[str] = None
    number: Optional[str] = None  # singular/plural
    person: Optional[str] = None  # first/second/third

@dataclass
class CoreferenceResult:
    """指代消解结果"""
    mention: Mention
    entity: Optional[Entity]
    confidence: float
    candidates: List[Entity]
    method: str = "neural_model"
    features: Dict[str, float] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = {}

class CandidateFilter:
    """候选实体筛选器"""
    
    def __init__(self):
        self.gender_mapping = {
            '他': 'male',
            '她': 'female', 
            '它': 'neutral',
            'he': 'male',
            'she': 'female',
            'it': 'neutral'
        }
    
    def filter(self, mention: Mention, entities: List[Entity], 
              dialogue_state: Dict) -> List[Entity]:
        """筛选候选实体"""
        candidates = []
        
        for entity in entities:
            if self._is_valid_candidate(mention, entity, dialogue_state):
                candidates.append(entity)
        
        # 按显著性和距离排序
        candidates.sort(key=lambda e: self._calculate_priority(e, dialogue_state), 
                       reverse=True)
        
        return candidates[:10]  # 返回前10个候选
    
    def _is_valid_candidate(self, mention: Mention, entity: Entity, 
                           dialogue_state: Dict) -> bool:
        """检查实体是否为有效候选"""
        # 性别一致性检查
        mention_gender = self.gender_mapping.get(mention.text.lower())
        if mention_gender and entity.gender and mention_gender != entity.gender:
            return False
        
        # 类型兼容性检查
        if mention.type == PronounType.PERSONAL:
            if mention_gender in ['male', 'female'] and entity.type not in ['PERSON']:
                return False
            if mention_gender == 'neutral' and entity.type in ['PERSON']:
                return False
        
        return True
    
    def _calculate_priority(self, entity: Entity, dialogue_state: Dict) -> float:
        """计算实体优先级"""
        priority = 0.0
        
        # 显著性分数
        salience = dialogue_state.get('entity_salience', {}).get(entity.id, 0.5)
        priority += salience * 0.6
        
        # 距离分数（越近越好）
        last_mention_turn = dialogue_state.get('entity_last_mention', {}).get(entity.id, 0)
        current_turn = dialogue_state.get('current_turn', 1)
        distance_score = 1.0 / (1.0 + current_turn - last_mention_turn)
        priority += distance_score * 0.4
        
        return priority

class MultiModalFeatureExtractor:
    """多模态特征提取器"""
    
    def __init__(self):
        self.feature_names = [
            'distance_score', 'salience_score', 'gender_match',
            'type_compatibility', 'syntactic_role', 'semantic_similarity'
        ]
    
    def extract(self, mention: Mention, candidates: List[Entity], 
               dialogue_state: Dict) -> Dict[str, np.ndarray]:
        """提取多模态特征"""
        features = {}
        
        for i, candidate in enumerate(candidates):
            candidate_features = self._extract_single_candidate_features(
                mention, candidate, dialogue_state
            )
            
            for feature_name in self.feature_names:
                if feature_name not in features:
                    features[feature_name] = []
                features[feature_name].append(
                    candidate_features.get(feature_name, 0.0)
                )
        
        # 转换为numpy数组
        for feature_name in features:
            features[feature_name] = np.array(features[feature_name])
        
        return features
    
    def _extract_single_candidate_features(self, mention: Mention, 
                                          candidate: Entity, 
                                          dialogue_state: Dict) -> Dict[str, float]:
        """提取单个候选实体的特征"""
        features = {}
        
        # 距离特征
        features['distance_score'] = self._calculate_distance_score(
            mention, candidate, dialogue_state
        )
        
        # 显著性特征
        features['salience_score'] = dialogue_state.get(
            'entity_salience', {}
        ).get(candidate.id, 0.5)
        
        # 性别匹配特征
        features['gender_match'] = self._calculate_gender_match(
            mention, candidate
        )
        
        # 类型兼容性特征
        features['type_compatibility'] = self._calculate_type_compatibility(
            mention, candidate
        )
        
        # 句法角色特征
        features['syntactic_role'] = self._calculate_syntactic_role(
            mention, candidate, dialogue_state
        )
        
        # 语义相似度特征
        features['semantic_similarity'] = self._calculate_semantic_similarity(
            mention, candidate, dialogue_state
        )
        
        return features
    
    def _calculate_distance_score(self, mention: Mention, candidate: Entity, 
                                 dialogue_state: Dict) -> float:
        """计算距离分数"""
        last_mention_turn = dialogue_state.get('entity_last_mention', {}).get(
            candidate.id, 0
        )
        current_turn = dialogue_state.get('current_turn', 1)
        distance = current_turn - last_mention_turn
        
        # 使用指数衰减
        return math.exp(-0.5 * distance)
    
    def _calculate_gender_match(self, mention: Mention, candidate: Entity) -> float:
        """计算性别匹配分数"""
        gender_mapping = {
            '他': 'male', '她': 'female', '它': 'neutral',
            'he': 'male', 'she': 'female', 'it': 'neutral'
        }
        
        mention_gender = gender_mapping.get(mention.text.lower())
        if not mention_gender or not candidate.gender:
            return 0.5  # 中性分数
        
        return 1.0 if mention_gender == candidate.gender else 0.0
    
    def _calculate_type_compatibility(self, mention: Mention, 
                                    candidate: Entity) -> float:
        """计算类型兼容性分数"""
        if mention.type == PronounType.PERSONAL:
            if mention.text.lower() in ['他', '她', 'he', 'she']:
                return 1.0 if candidate.type == 'PERSON' else 0.0
            elif mention.text.lower() in ['它', 'it']:
                return 1.0 if candidate.type != 'PERSON' else 0.0
        
        return 0.5  # 默认中性分数
    
    def _calculate_syntactic_role(self, mention: Mention, candidate: Entity, 
                                 dialogue_state: Dict) -> float:
        """计算句法角色分数"""
        # 简化实现：基于实体在句子中的位置
        return 0.5  # 占位符实现
    
    def _calculate_semantic_similarity(self, mention: Mention, candidate: Entity, 
                                      dialogue_state: Dict) -> float:
        """计算语义相似度分数"""
        # 简化实现：可以使用词向量计算相似度
        return 0.5  # 占位符实现

class ProbabilityCalculator:
    """概率计算器"""
    
    def __init__(self):
        # 简化的权重参数
        self.feature_weights = {
            'distance_score': 0.3,
            'salience_score': 0.25,
            'gender_match': 0.2,
            'type_compatibility': 0.15,
            'syntactic_role': 0.05,
            'semantic_similarity': 0.05
        }
    
    def compute(self, features: Dict[str, np.ndarray], 
               dialogue_state: Dict) -> np.ndarray:
        """计算候选实体的概率分布"""
        if not features or len(list(features.values())[0]) == 0:
            return np.array([])
        
        # 计算加权特征分数
        scores = np.zeros(len(list(features.values())[0]))
        
        for feature_name, feature_values in features.items():
            weight = self.feature_weights.get(feature_name, 0.0)
            scores += weight * feature_values
        
        # 应用softmax获得概率分布
        probabilities = self._softmax(scores)
        
        return probabilities
    
    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        if len(scores) == 0:
            return scores
        
        exp_scores = np.exp(scores - np.max(scores))  # 数值稳定性
        return exp_scores / np.sum(exp_scores)

class DisambiguationEngine:
    """消歧决策引擎"""
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
    
    def decide(self, candidates: List[Entity], 
              probabilities: np.ndarray) -> Optional[Entity]:
        """做出消歧决策"""
        if len(candidates) == 0 or len(probabilities) == 0:
            return None
        
        # 找到最高概率的候选
        best_idx = np.argmax(probabilities)
        best_probability = probabilities[best_idx]
        
        # 检查置信度阈值
        if best_probability < self.confidence_threshold:
            return None
        
        return candidates[best_idx]

class ConfidenceEstimator:
    """置信度评估器"""
    
    def estimate(self, best_candidate: Optional[Entity], 
                probabilities: np.ndarray) -> float:
        """评估置信度"""
        if best_candidate is None or len(probabilities) == 0:
            return 0.0
        
        max_prob = np.max(probabilities)
        
        # 如果只有一个候选，置信度等于其概率
        if len(probabilities) == 1:
            return float(max_prob)
        
        # 计算最高概率与第二高概率的差距
        sorted_probs = np.sort(probabilities)[::-1]
        if len(sorted_probs) >= 2:
            confidence = float(sorted_probs[0] - sorted_probs[1] + sorted_probs[0])
            return min(confidence, 1.0)
        
        return float(max_prob)

class AdvancedCoreferenceLayer:
    """高级指代消解层"""
    
    def __init__(self):
        self.candidate_filter = CandidateFilter()
        self.feature_extractor = MultiModalFeatureExtractor()
        self.probability_calculator = ProbabilityCalculator()
        self.disambiguation_engine = DisambiguationEngine()
        self.confidence_estimator = ConfidenceEstimator()
    
    def resolve(self, mention: Mention, entities: List[Entity], 
               dialogue_state: Dict) -> CoreferenceResult:
        """执行指代消解"""
        # 1. 候选实体筛选
        candidates = self.candidate_filter.filter(
            mention, entities, dialogue_state
        )
        
        if not candidates:
            return CoreferenceResult(
                mention=mention,
                entity=None,
                confidence=0.0,
                candidates=[],
                method="no_candidates"
            )
        
        # 2. 多模态特征提取
        features = self.feature_extractor.extract(
            mention, candidates, dialogue_state
        )
        
        # 3. 概率计算
        probabilities = self.probability_calculator.compute(
            features, dialogue_state
        )
        
        # 4. 消歧决策
        best_candidate = self.disambiguation_engine.decide(
            candidates, probabilities
        )
        
        # 5. 置信度评估
        confidence = self.confidence_estimator.estimate(
            best_candidate, probabilities
        )
        
        return CoreferenceResult(
            mention=mention,
            entity=best_candidate,
            confidence=confidence,
            candidates=candidates[:3],  # 返回前3个候选
            method="advanced_neural",
            features=dict(zip(self.feature_extractor.feature_names, 
                            [float(f[0]) if len(f) > 0 else 0.0 
                             for f in features.values()]))
        )

# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    mention = Mention(
        text="他",
        type=PronounType.PERSONAL,
        start=10,
        end=11,
        gender="male"
    )
    
    entities = [
        Entity(id="1", text="穆勒", type="PERSON", start=0, end=2, gender="male"),
        Entity(id="2", text="拜仁", type="ORG", start=3, end=5, gender="neutral")
    ]
    
    dialogue_state = {
        'current_turn': 2,
        'entity_salience': {'1': 0.8, '2': 0.3},
        'entity_last_mention': {'1': 1, '2': 1}
    }
    
    # 执行指代消解
    coref_layer = AdvancedCoreferenceLayer()
    result = coref_layer.resolve(mention, entities, dialogue_state)
    
    print(f"指代词: {result.mention.text}")
    print(f"解析结果: {result.entity.text if result.entity else 'None'}")
    print(f"置信度: {result.confidence:.3f}")
    print(f"候选实体: {[c.text for c in result.candidates]}")