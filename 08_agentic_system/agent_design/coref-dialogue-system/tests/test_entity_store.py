"""实体账本测试：ID 分配、跨轮合并、显著性单位、信息保鲜。"""

from schemas import DialogueState, EntityRecord, EntityType, Resolution
from entity_store import EntityStore


def make_state(turn: int, resolutions: list[dict], entities: list[dict] | None = None) -> DialogueState:
    return DialogueState.model_validate(
        {
            "entities": entities or [],
            "resolutions": resolutions,
            "current_turn": turn,
        }
    )


def entity_dict(name: str, turn: int, **over) -> dict:
    d = dict(
        name=name,
        entity_type=EntityType.PERSON.value,
        aliases=[],
        attributes={},
        first_seen_turn=turn,
        last_mention_turn=turn,
        mention_count=1,
    )
    d.update(over)
    return d


def test_new_entity_gets_stable_id(store):
    """v1 碰撞 bug 的回归：ID 由 store 顺序分配，跨轮不重置。"""
    outcomes = store.apply_state(
        "d1",
        make_state(1, [{"mention_text": "张三", "turn_id": 1, "resolved_to": "NEW",
                        "confidence": 1.0, "reasoning": ""}],
                   [entity_dict("张三", 1)]),
    )
    assert outcomes[0].status == "new"
    id1 = outcomes[0].entity_id

    outcomes = store.apply_state(
        "d1",
        make_state(2, [{"mention_text": "他", "turn_id": 2, "resolved_to": "张三",
                        "confidence": 0.95, "reasoning": "上轮主语继承"}]),
    )
    assert outcomes[0].status == "resolved"
    assert outcomes[0].entity_id == id1, "第二轮回指必须命中同一 ID"


def test_second_entity_not_swallowed(store):
    """v1 实测 bug：李四被并进张三。两轮各引入一人，账本必须有两个实体。"""
    store.apply_state("d1", make_state(1, [{"mention_text": "张三", "turn_id": 1, "resolved_to": "NEW",
                                            "confidence": 1.0, "reasoning": ""}],
                                       [entity_dict("张三", 1)]))
    store.apply_state("d1", make_state(2, [{"mention_text": "李四", "turn_id": 2, "resolved_to": "NEW",
                                            "confidence": 1.0, "reasoning": ""}],
                                       [entity_dict("李四", 2)]))
    snapshot = store.snapshot("d1")
    names = {e.name for e in snapshot}
    assert names == {"张三", "李四"}, f"李四不得被吞并: {names}"


def test_alias_recorded_on_pronoun_resolution(store):
    store.apply_state("d1", make_state(1, [{"mention_text": "张三", "turn_id": 1, "resolved_to": "NEW",
                                            "confidence": 1.0, "reasoning": ""}],
                                       [entity_dict("张三", 1)]))
    store.apply_state("d1", make_state(2, [{"mention_text": "他", "turn_id": 2, "resolved_to": "张三",
                                            "confidence": 0.9, "reasoning": ""}]))
    ent = store.session("d1").by_name["张三"]
    assert "他" in ent.record.aliases
    assert ent.mention_turns == [1, 2]


def test_salience_decays_by_turns_not_seconds(store):
    """v1 教训：同名衰减机制一处按秒一处按轮。这里锁定按轮的数值契约。"""
    import math

    store.apply_state("d1", make_state(1, [{"mention_text": "张三", "turn_id": 1, "resolved_to": "NEW",
                                            "confidence": 1.0, "reasoning": ""}],
                                       [entity_dict("张三", 1)]))
    ent = store.session("d1").by_name["张三"]
    s_now = ent.salience(current_turn=1)
    s_later = ent.salience(current_turn=3)
    assert math.isclose(s_later / s_now, math.exp(-0.3 * 2), rel_tol=1e-9)


def test_attribute_refresh_keeps_history(store):
    """信息保鲜：新属性覆盖旧值，旧值进 history。"""
    store.apply_state("d1", make_state(1, [], [entity_dict("张三", 1, attributes={"role": "后端"})]))
    store.apply_state("d1", make_state(2, [], [entity_dict("张三", 1, last_mention_turn=1,
                                                           attributes={"role": "后端负责人"})]))
    ent = store.session("d1").by_name["张三"]
    assert ent.record.attributes == {"role": "后端负责人"}
    assert ent.attribute_history == [(2, {"role": "后端"})]


def test_ambiguous_mention_not_written(store):
    outcomes = store.apply_state(
        "d1",
        make_state(1, [{"mention_text": "他", "turn_id": 1, "resolved_to": "AMBIGUOUS",
                        "confidence": 0.2, "reasoning": "无候选"}]),
    )
    assert outcomes[0].status == "ambiguous"
    assert store.snapshot("d1") == []


def test_sessions_isolated(store):
    store.apply_state("a", make_state(1, [{"mention_text": "张三", "turn_id": 1, "resolved_to": "NEW",
                                           "confidence": 1.0, "reasoning": ""}],
                                      [entity_dict("张三", 1)]))
    assert store.snapshot("b") == []
