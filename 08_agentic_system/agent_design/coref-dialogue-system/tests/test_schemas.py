"""schema 契约测试：LLM 输出的每条约束都要有对应的拒绝用例。"""

import pytest
from pydantic import ValidationError

from schemas import DialogueState, EntityRecord, EntityType, Resolution


def base_entity(**over) -> dict:
    d = dict(
        name="张三",
        entity_type=EntityType.PERSON,
        aliases=[],
        attributes={"role": "后端工程师"},
        first_seen_turn=1,
        last_mention_turn=1,
        mention_count=1,
    )
    d.update(over)
    return d


def test_accept_valid_state():
    state = DialogueState.model_validate(
        {
            "entities": [base_entity()],
            "resolutions": [
                {
                    "mention_text": "他",
                    "turn_id": 2,
                    "resolved_to": "张三",
                    "confidence": 0.9,
                    "reasoning": "上轮主语继承",
                }
            ],
            "current_turn": 2,
        }
    )
    assert state.entities[0].name == "张三"
    assert state.resolutions[0].resolved_to == "张三"


def test_reject_pronoun_as_entity_name():
    """v1 经典 bug 的回归测试：代词不能成为实体（否则它会成为自己的先行语）。"""
    with pytest.raises(ValidationError):
        EntityRecord.model_validate(base_entity(name="他"))


def test_reject_last_turn_before_first():
    with pytest.raises(ValidationError):
        EntityRecord.model_validate(base_entity(first_seen_turn=3, last_mention_turn=1))


def test_reject_confidence_out_of_range():
    with pytest.raises(ValidationError):
        Resolution(mention_text="他", turn_id=1, resolved_to="张三", confidence=1.5)


def test_reject_unknown_entity_type():
    with pytest.raises(ValidationError):
        EntityRecord.model_validate(base_entity(entity_type="ALIEN"))
