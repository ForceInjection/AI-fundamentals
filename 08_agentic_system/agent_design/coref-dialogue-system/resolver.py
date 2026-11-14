"""LLM 指代消解器：OpenAI 兼容接口 + JSON 约束 + 校验重试。

设计要点（对应主流实践）：
1. **全量重建**：每轮把完整对话历史交给 LLM，让它输出整个 DialogueState，
   而不是在旧状态上打增量补丁——增量管线是 v1 实体 ID 碰撞、状态断链的根源。
2. **结构化输出**：prompt 内嵌 JSON Schema，响应强制 JSON；pydantic 校验
   失败时把错误信息喂回重试（最多 2 次）。schema 校验在任何 OpenAI 兼容
   端点都可用，不依赖特定厂商的 json_schema 受限解码。
3. **可降级**：无 Key / 网络 / 重试耗尽 → 抛 ResolverError，调用方切换到
   demo.py 的规则路径。LLM 通道与规则通道是并列的两条腿，不是绑定。

配置经环境变量（12-factor）：
    DEEPSEEK_API_KEY / OPENAI_API_KEY   二者任一即可
    COREF_BASE_URL                       默认 https://api.deepseek.com
    COREF_MODEL                          默认 deepseek-chat
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

from schemas import DialogueState, ResolverError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是对话状态跟踪（DST）与指代消解引擎。给定多轮对话历史，输出当前完整对话状态。

规则：
1. 输出单一 JSON 对象，符合下面的 JSON Schema，不要输出任何其他文字。
2. entities 是你当前确认的全部实体（全量重建，不是增量）；代词（他/她/它/这个/那个等）
   不能作为实体名。mention_count 包含代词回指次数。
3. resolutions 覆盖最新一轮用户消息里的每一个指代性 mention（代词与省略主语）：
   - resolved_to 填实体规范名（回指成功）
   - 引入新实体时填 "NEW"（此时 entities 里必须有对应条目）
   - 无法确定时填 "AMBIGUOUS"，confidence 给低值。宁可 AMBIGUOUS 也不猜。
4. confidence 在 0-1 之间；reasoning 用一句话给出消解依据（如：上轮主语继承/性别一致/近因）。
5. current_turn = 对话历史的总轮数。

JSON Schema：
{schema}"""

SCHEMA_HINT = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "entity_type": {
                        "type": "string",
                        "enum": ["PERSON", "ORG", "LOCATION", "PRODUCT", "PROJECT", "TIME", "OTHER"],
                    },
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
                    "first_seen_turn": {"type": "integer", "minimum": 1},
                    "last_mention_turn": {"type": "integer", "minimum": 1},
                    "mention_count": {"type": "integer", "minimum": 1},
                },
                "required": [
                    "name", "entity_type", "aliases", "attributes",
                    "first_seen_turn", "last_mention_turn", "mention_count",
                ],
            },
        },
        "resolutions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mention_text": {"type": "string"},
                    "turn_id": {"type": "integer", "minimum": 1},
                    "resolved_to": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"},
                },
                "required": ["mention_text", "turn_id", "resolved_to", "confidence", "reasoning"],
            },
        },
        "current_turn": {"type": "integer", "minimum": 1},
    },
    "required": ["entities", "resolutions", "current_turn"],
}


class LLMResolver:
    """把「对话历史 → DialogueState」委托给 LLM，负责 prompt 组装与结果校验。"""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or os.environ.get("COREF_BASE_URL", "https://api.deepseek.com"),
        )
        self._model = model or os.environ.get("COREF_MODEL", "deepseek-chat")
        self._max_retries = max_retries

    def resolve(self, history: list[dict[str, str]], entity_summary: str = "") -> DialogueState:
        """history: [{"role": "user"|"assistant", "content": ...}]，按时间序，含最新用户消息。"""
        user_content = self._render_history(history)
        if entity_summary:
            user_content += "\n\n" + entity_summary
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(schema=json.dumps(SCHEMA_HINT, ensure_ascii=False))},
            {"role": "user", "content": user_content},
        ]
        last_err: Exception | None = None
        raw = ""
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=2000,
                )
                raw = resp.choices[0].message.content or ""
                return self._parse(raw)
            except json.JSONDecodeError as e:
                last_err = e
                logger.warning("LLM 输出不是合法 JSON（第 %d 次）：%s", attempt + 1, e)
            except ValueError as e:  # pydantic ValidationError 的基类链
                last_err = e
                logger.warning("LLM 输出未通过 schema 校验（第 %d 次）：%s", attempt + 1, e)
            # 校验失败：把错误回喂，要求修正——这是结构化输出的标准重试姿势
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"上面的输出有问题：{last_err}。请严格按 JSON Schema 重新输出完整 JSON，不要任何解释文字。",
            })
        raise ResolverError(f"LLM 输出在 {self._max_retries + 1} 次尝试后仍不合法: {last_err}")

    # ---------- 内部 ----------

    @staticmethod
    def _render_history(history: list[dict[str, str]]) -> str:
        lines = []
        for i, msg in enumerate(history, start=1):
            role = "用户" if msg["role"] == "user" else "客服"
            lines.append(f"[第{i}轮·{role}] {msg['content']}")
        return "对话历史：\n" + "\n".join(lines)

    def _parse(self, raw: str) -> DialogueState:
        data: Any = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"期望 JSON 对象，得到 {type(data).__name__}")
        return DialogueState.model_validate(data)
