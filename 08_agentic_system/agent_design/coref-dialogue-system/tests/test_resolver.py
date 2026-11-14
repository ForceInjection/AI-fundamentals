"""LLM 消解器测试：全部走 stub client，不产生真实 API 调用。"""

import json

import pytest

from resolver import LLMResolver
from schemas import ResolverError

GOOD_JSON = json.dumps(
    {
        "entities": [
            {
                "name": "张三",
                "entity_type": "PERSON",
                "aliases": ["他"],
                "attributes": {"role": "后端工程师"},
                "first_seen_turn": 1,
                "last_mention_turn": 2,
                "mention_count": 2,
            }
        ],
        "resolutions": [
            {
                "mention_text": "他",
                "turn_id": 2,
                "resolved_to": "张三",
                "confidence": 0.93,
                "reasoning": "上轮主语继承",
            }
        ],
        "current_turn": 2,
    },
    ensure_ascii=False,
)

HISTORY = [
    {"role": "user", "content": "张三加入了星辰项目，他是后端工程师。"},
    {"role": "assistant", "content": "（已解析）"},
    {"role": "user", "content": "他的第一个任务是什么？"},
]


class StubClient:
    """按预设序列返回响应，记录调用次数与收到的 messages。"""

    def __init__(self, payloads: list[str | Exception]) -> None:
        self._payloads = payloads
        self.calls = 0
        self.last_messages = None

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        self.last_messages = kwargs.get("messages")
        payload = self._payloads[min(self.calls - 1, len(self._payloads) - 1)]
        if isinstance(payload, Exception):
            raise payload
        obj = type("R", (), {})()
        obj.choices = [type("C", (), {})()]
        obj.choices[0].message = type("M", (), {})()
        obj.choices[0].message.content = payload
        return obj


def make_resolver(payloads) -> LLMResolver:
    r = LLMResolver(api_key="test-key")
    r._client = StubClient(payloads)
    return r


def test_golden_pronoun_resolves_to_zhangsan():
    """黄金断言：「他」必须解析到「张三」——v1 全项目没有一条这样的断言。"""
    r = make_resolver([GOOD_JSON])
    state = r.resolve(HISTORY)
    assert state.resolutions[0].mention_text == "他"
    assert state.resolutions[0].resolved_to == "张三"
    assert state.entities[0].name == "张三"


def test_prompt_contains_schema_and_history():
    r = make_resolver([GOOD_JSON])
    r.resolve(HISTORY)
    system_msg = r._client.last_messages[0]["content"]
    user_msg = r._client.last_messages[-1]["content"]
    assert "JSON Schema" in system_msg
    assert "张三加入了星辰项目" in user_msg


def test_retry_after_malformed_json():
    bad = '{"entities": [截断的'
    r = make_resolver([bad, GOOD_JSON])
    state = r.resolve(HISTORY)
    assert r._client.calls == 2
    assert state.resolutions[0].resolved_to == "张三"
    # 错误信息被回喂给下一轮
    retry_hint = r._client.last_messages[-1]["content"]
    assert "重新输出" in retry_hint


def test_retry_exhausted_raises():
    r = make_resolver(["not json", "still not json", "nope"])
    with pytest.raises(ResolverError):
        r.resolve(HISTORY)
    assert r._client.calls == 3  # 1 + max_retries(2)
