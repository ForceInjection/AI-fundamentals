"""API 层测试：多轮状态回灌闭环（v1 断链 bug 的回归测试）。"""

from fastapi.testclient import TestClient

import service
from schemas import DialogueState, EntityRecord, EntityType, Resolution


def make_client() -> TestClient:
    return TestClient(service.app)


def seed_entity(dialogue_id: str = "t1"):
    """绕过 NER 直接注账：模拟第一轮已完成（历史与账本同步更新，与 /chat 行为一致）。"""
    store = service._store
    store.apply_state(
        dialogue_id,
        DialogueState(
            entities=[EntityRecord(
                name="张三", entity_type=EntityType.PERSON, aliases=[],
                attributes={"role": "后端工程师"}, first_seen_turn=1,
                last_mention_turn=1, mention_count=1,
            )],
            resolutions=[],
            current_turn=1,
        ),
    )
    service._histories[dialogue_id] = [
        {"role": "user", "content": "张三加入了星辰项目，他是后端工程师。"},
        {"role": "assistant", "content": "（已解析）"},
    ]


def test_rule_path_resolves_pronoun_across_turns(monkeypatch):
    """核心闭环：第一轮注册张三，第二轮「他」必须解析到张三。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = make_client()
    seed_entity("t1")

    # 第二轮由规则路径消解（turn 递增到 2）
    resp = client.post("/chat", json={"dialogue_id": "t1", "message": "他的第一个任务是什么？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["used_path"] == "rules"
    assert body["turn_id"] == 2
    assert body["resolutions"], "规则路径应给出消解结论"
    assert body["resolutions"][0]["mention_text"] == "他"
    assert body["resolutions"][0]["resolved_to"] == "张三"
    # 服务端状态被真实更新（v1 是伪造的 current_turn=1）
    state = client.get("/dialogues/t1/state").json()
    assert state["turn"] == 2
    names = {e["name"] for e in state["entities"]}
    assert "张三" in names


def test_llm_path_wired(monkeypatch):
    """有 Key 时走 LLM 路径：resolver.resolve 被调用且其输出直接进响应。"""
    import resolver as resolver_mod

    calls = []

    def fake_resolve(self, history, entity_summary=""):
        calls.append(history)
        return DialogueState(
            entities=[EntityRecord(
                name="张三", entity_type=EntityType.PERSON, aliases=["他"],
                attributes={}, first_seen_turn=1, last_mention_turn=1, mention_count=1,
            )],
            resolutions=[Resolution(mention_text="他", turn_id=2, resolved_to="张三",
                                    confidence=0.9, reasoning="stub")],
            current_turn=2,
        )

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(resolver_mod.LLMResolver, "resolve", fake_resolve)
    client = make_client()
    seed_entity("t2")

    resp = client.post("/chat", json={"dialogue_id": "t2", "message": "他还在吗？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["used_path"] == "llm"
    assert body["resolutions"][0]["resolved_to"] == "张三"
    # 历史里包含本轮用户消息（全量重建的输入）
    assert any(m["content"] == "他还在吗？" for m in calls[0])


def test_unknown_dialogue_state_404():
    client = make_client()
    assert client.get("/dialogues/nope/state").status_code == 404


def test_delete_dialogue():
    client = make_client()
    client.post("/chat", json={"dialogue_id": "t3", "message": "你好"})
    assert client.delete("/dialogues/t3").status_code == 200
    assert client.get("/dialogues/t3/state").status_code == 404
