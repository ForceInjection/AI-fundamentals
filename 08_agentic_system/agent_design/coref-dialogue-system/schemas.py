"""结构化输出契约（pydantic v2）。

v2 的核心约定：LLM 每一轮输出的是**完整的对话状态**（DialogueState），
而不是对旧状态的增量补丁。这是当前对话状态跟踪（DST）的主流做法：
从对话历史全量重建状态，天然处理用户的修正、撤销与话题切换，
也把「实体 ID 碰撞」这类增量管线的经典 bug 从机制上排除——
实体以 name 为逻辑键，ID 由 EntityStore 在服务端统一分配。

所有跨边界的数据都走这里的模型：LLM 输出经 `DialogueState` 校验，
HTTP 请求/响应经 `ChatRequest` / `ChatResponse` 校验。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EntityType(str, Enum):
    """实体类型。LLM 输出必须落在这个枚举内，否则校验失败触发重试。"""

    PERSON = "PERSON"
    ORG = "ORG"
    LOCATION = "LOCATION"
    PRODUCT = "PRODUCT"
    PROJECT = "PROJECT"
    TIME = "TIME"
    OTHER = "OTHER"


class EntityRecord(BaseModel):
    """单个实体的最新视图。

    LLM 全量重建时输出它当前「相信」的完整属性；EntityStore 负责把
    连续两轮的视图合并（attributes 取新值覆盖旧值，历史进 audit trail）。
    """

    name: str = Field(min_length=1, description="实体规范名，作为跨轮合并的逻辑键")
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list, description="对话中出现过的其他称呼")
    attributes: dict[str, str] = Field(
        default_factory=dict, description="该实体的已知属性，如 role=后端工程师"
    )
    first_seen_turn: int = Field(ge=1, description="首次出现的轮次（从 1 计）")
    last_mention_turn: int = Field(ge=1, description="最近一次被提及的轮次")
    mention_count: int = Field(ge=1, description="累计提及次数（含代词回指）")

    @field_validator("name")
    @classmethod
    def name_not_pronoun(cls, v: str) -> str:
        """代词不能成为实体本身——v1 的经典 bug：把「他」存成实体后它成了自己的先行语。"""
        pronouns = {"他", "她", "它", "他们", "她们", "它们", "这个", "那个", "上次那个"}
        if v.strip() in pronouns:
            raise ValueError(f"代词不能作为实体名: {v}")
        return v.strip()

    @field_validator("last_mention_turn")
    @classmethod
    def last_not_before_first(cls, v: int, info: Any) -> int:
        first = info.data.get("first_seen_turn")
        if first is not None and v < first:
            raise ValueError(f"last_mention_turn({v}) 早于 first_seen_turn({first})")
        return v


class Resolution(BaseModel):
    """本轮某个 mention 的消解结论。

    resolved_to 的取值约定：
    - 实体规范名：成功回指到已有实体
    - "NEW"：引入新实体（该 mention 本身就是实体名）
    - "AMBIGUOUS"：无法确定先行语，宁可挂起也不猜——错误消解比不消解伤害大
    """

    mention_text: str = Field(min_length=1, description="原文中的指代表面形式，如「他」「那个项目」")
    turn_id: int = Field(ge=1)
    resolved_to: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=200, description="一两句消解依据，供人工核查")


class DialogueState(BaseModel):
    """LLM 每轮输出的完整对话状态。"""

    entities: list[EntityRecord] = Field(default_factory=list)
    resolutions: list[Resolution] = Field(default_factory=list, description="本轮各 mention 的消解结论")
    current_turn: int = Field(ge=1)


class ChatRequest(BaseModel):
    """POST /chat 的请求体。"""

    dialogue_id: str = Field(min_length=1, description="会话标识，客户端生成并保持稳定")
    message: str = Field(min_length=1, description="本轮用户消息原文")


class ChatResponse(BaseModel):
    """POST /chat 的响应体。"""

    dialogue_id: str
    turn_id: int
    reply: str = Field(default="", description="占位回复文本；本 demo 只做理解，不做生成")
    resolutions: list[Resolution]
    entities: list[EntityRecord]
    used_path: str = Field(description="llm | rules —— 实际走的消解路径，教学对照用")


class ResolverError(Exception):
    """LLM 通道不可用（无 Key / 网络失败 / 重试后仍不合法）。调用方据此降级到规则路径。"""
