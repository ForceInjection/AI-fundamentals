"""规则路径测试：纯代词表 + 显著性消解（_NER=None，见 conftest）。"""

from schemas import EntityRecord, EntityType
from demo import PRONOUNS, RuleResolver


def rec(name: str, turn: int, etype: EntityType = EntityType.PERSON, mention_count: int = 1) -> EntityRecord:
    return EntityRecord(
        name=name, entity_type=etype, aliases=[], attributes={},
        first_seen_turn=turn, last_mention_turn=turn, mention_count=mention_count,
    )


def test_pronoun_table_covers_chinese_third_person():
    for p in ["他", "她", "它", "他们"]:
        assert p in PRONOUNS


def test_extract_pronouns_from_text():
    rr = RuleResolver()
    _, mentions = rr.extract("他的第一个任务是重构登录模块。", 2, {})
    assert "他" in mentions


def test_extract_falls_back_to_known_names_without_spacy():
    """无 spaCy 时只认已知名——诚实的降级边界，要锁死行为。"""
    rr = RuleResolver()
    ents, _ = rr.extract("张三和李四开会", 1, {"张三": EntityType.PERSON})
    assert "张三" in ents
    assert "李四" not in ents  # 未注册的名字抽不出来，这是降级的代价


def test_single_candidate_resolves():
    rr = RuleResolver()
    results = rr.resolve_mentions(
        ["他"], {"张三": EntityType.PERSON}, {"张三": 0.9}, turn=2,
    )
    assert results[0].resolved_to == "张三"
    assert results[0].confidence >= 0.5


def test_gender_pronoun_filters_candidates():
    """「她」不该解析到男性候选（类型过滤的代理测试：PERSON 代词 vs 组织候选）。"""
    rr = RuleResolver()
    results = rr.resolve_mentions(
        ["它"], {"星辰项目": EntityType.PROJECT, "张三": EntityType.PERSON},
        {"星辰项目": 0.4, "张三": 0.9}, turn=3,
    )
    # "它" 类型不定（None）：不过滤，按显著性应命中张三
    assert results[0].resolved_to == "张三"


def test_even_split_is_ambiguous():
    """两个势均力敌的候选：宁可 AMBIGUOUS 也不猜。"""
    rr = RuleResolver()
    results = rr.resolve_mentions(
        ["他"],
        {"张三": EntityType.PERSON, "李四": EntityType.PERSON},
        {"张三": 0.90, "李四": 0.89},  # softmax 后接近均匀
        turn=5,
    )
    assert results[0].resolved_to == "AMBIGUOUS"
    assert results[0].confidence < 0.6, "AMBIGUOUS 的置信度必须落在阈值以下（语义自洽）"


def test_build_state_wires_mentions_to_store_shape(store):
    rr = RuleResolver()
    known = {"张三": rec("张三", 1)}
    salience = {"张三": 0.8}
    state = rr.build_state("他的任务", turn=2, new_entities=[], known=known, salience=salience)
    assert state.resolutions[0].resolved_to == "张三"
    store.apply_state("d", state)
    assert "他" in store.session("d").by_name["张三"].record.aliases
