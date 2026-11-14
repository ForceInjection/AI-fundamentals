"""实体记忆：服务端的唯一事实源（single source of truth）。

职责边界（与 resolver.py 的分工）：
- resolver（LLM）：理解——输出「它当前相信的完整状态」（DialogueState）
- entity_store（本文件）：记账——ID 分配、跨轮合并、显著性、保鲜审计

为什么这么分：LLM 的全量重建输出可能自相矛盾（mention_count 漂移、
first_seen_turn 回退），直接采信会让状态随轮次漂移。store 以「服务端
记账」校正 LLM 视图：合并按逻辑键（name+type），计数与轮次以本地
观测为准，LLM 的属性视图（attributes）作为可覆盖的最新值合并进来。

显著性公式（v1 的教训：一处按秒一处按轮，同名机制两种物理含义）：
    salience = exp(-0.3 * turn_gap) * log1p(mention_count)
统一按「轮」衰减；0.3 意味着隔 2 轮显著性降到约 55%，隔 5 轮约 22%。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from schemas import DialogueState, EntityRecord, EntityType


@dataclass
class StoredEntity:
    """store 内部的实体账本条目。id 是服务端分配的稳定主键。"""

    id: str
    record: EntityRecord
    mention_turns: list[int] = field(default_factory=list)  # 每次被提及（含代词）的轮次
    attribute_history: list[tuple[int, dict[str, str]]] = field(default_factory=list)

    def salience(self, current_turn: int, decay: float = 0.3) -> float:
        if not self.mention_turns:
            return 0.0
        gap = current_turn - max(self.mention_turns)
        return math.exp(-decay * gap) * math.log1p(len(self.mention_turns))


class EntityStore:
    """按 dialogue_id 隔离的内存实体账本。

    生产替换点：把 self._sessions 换成 Redis/DB 并给 SessionState 加
    TTL 即可，接口不变。
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}

    # ---------- 会话生命周期 ----------

    def session(self, dialogue_id: str) -> _Session:
        return self._sessions.setdefault(dialogue_id, _Session(dialogue_id))

    def drop(self, dialogue_id: str) -> bool:
        return self._sessions.pop(dialogue_id, None) is not None

    # ---------- 状态合入 ----------

    def apply_state(self, dialogue_id: str, state: DialogueState) -> list[ResolutionOutcome]:
        """把 LLM 全量重建的视图合并进账本，返回本轮 mention 的落地结果。

        分配/合并规则：
        1. mention 是实体名或别名 → 命中已有实体（或别名扩展）
        2. mention 解析到某实体规范名 → 该实体 mention_turns 追加本轮
        3. resolved_to == "NEW" → 新建实体，分配稳定 ID
        4. resolved_to == "AMBIGUOUS" → 只记账 mention，不落实体
        """
        sess = self.session(dialogue_id)
        turn = state.current_turn
        sess.turn = turn
        outcomes: list[ResolutionOutcome] = []

        for res in state.resolutions:
            outcome = ResolutionOutcome(mention=res.mention_text, resolution=res)
            target = res.resolved_to
            if target == "AMBIGUOUS":
                outcome.status = "ambiguous"
                outcomes.append(outcome)
                continue
            if target == "NEW":
                # 新实体的规范名取 mention 原文；LLM 视图里应有一个同名 EntityRecord
                rec = self._llm_record_for(state, res.mention_text)
                ent = sess.by_name.get(res.mention_text)
                if ent is None:
                    ent = StoredEntity(
                        id=f"ent_{sess.next_id:04d}",
                        record=rec
                        or EntityRecord(
                            name=res.mention_text,
                            entity_type=EntityType.OTHER,
                            first_seen_turn=turn,
                            last_mention_turn=turn,
                            mention_count=1,
                        ),
                    )
                    sess.next_id += 1
                    sess.entities.append(ent)
                    sess.by_name[ent.record.name] = ent
                if turn not in ent.mention_turns:
                    ent.mention_turns.append(turn)
                outcome.entity_id = ent.id
                outcome.status = "new"
                outcomes.append(outcome)
                continue
            ent = sess.by_name.get(target)
            if ent is None:
                # LLM 引用了 store 里还没有的实体（首轮即回指的罕见情况）：补记
                rec = self._llm_record_for(state, target)
                if rec is None:
                    outcome.status = "unresolved"
                    outcomes.append(outcome)
                    continue
                ent = StoredEntity(
                    id=f"ent_{sess.next_id:04d}",
                    record=rec,
                )
                sess.next_id += 1
                sess.entities.append(ent)
                sess.by_name[ent.record.name] = ent
            # 命中：提及轮次追加 + 别名归并
            if turn not in ent.mention_turns:
                ent.mention_turns.append(turn)
            ent.record.last_mention_turn = turn
            if res.mention_text != ent.record.name and res.mention_text not in ent.record.aliases:
                ent.record.aliases.append(res.mention_text)
            outcome.entity_id = ent.id
            outcome.status = "resolved"
            outcomes.append(outcome)

        # 实体属性视图合并（保鲜）：LLM 输出的最新 attributes 覆盖旧值，历史留痕
        for rec in state.entities:
            ent = sess.by_name.get(rec.name)
            if ent is None:
                # LLM 视图里有而账本没有（例如本轮文本显式提到的新实体）→ 补记
                ent = StoredEntity(
                    id=f"ent_{sess.next_id:04d}",
                    record=rec.model_copy(),
                )
                sess.next_id += 1
                sess.entities.append(ent)
                sess.by_name[ent.record.name] = ent
                if turn not in ent.mention_turns:
                    ent.mention_turns.append(turn)
            else:
                if rec.attributes != ent.record.attributes:
                    ent.attribute_history.append((turn, dict(ent.record.attributes)))
                    ent.record.attributes = rec.attributes
                if rec.aliases:
                    for a in rec.aliases:
                        if a not in ent.record.aliases and a != ent.record.name:
                            ent.record.aliases.append(a)
                ent.record.mention_count = max(ent.record.mention_count, rec.mention_count)
                ent.record.entity_type = rec.entity_type

        return outcomes

    # ---------- 查询 ----------

    def top_entities(self, dialogue_id: str, k: int = 5) -> list[tuple[StoredEntity, float]]:
        sess = self.session(dialogue_id)
        scored = [(e, e.salience(sess.turn)) for e in sess.entities]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def context_summary(self, dialogue_id: str, k: int = 5) -> str:
        """给下一轮 LLM prompt 的实体速览。空会话返回空串。"""
        sess = self.session(dialogue_id)
        if not sess.entities:
            return ""
        lines = []
        for ent, score in self.top_entities(dialogue_id, k):
            attrs = "; ".join(f"{a}={v}" for a, v in ent.record.attributes.items()) or "-"
            aliases = "/".join(ent.record.aliases) or "-"
            lines.append(
                f"- {ent.record.name}({ent.id}, {ent.record.entity_type.value}, "
                f"别名: {aliases}, 提及 {len(ent.mention_turns)} 次, 显著性 {score:.2f}, 属性: {attrs})"
            )
        return "当前已知实体：\n" + "\n".join(lines)

    def snapshot(self, dialogue_id: str) -> list[EntityRecord]:
        return [e.record.model_copy() for e in self.session(dialogue_id).entities]

    # ---------- 内部 ----------

    @staticmethod
    def _llm_record_for(state: DialogueState, name: str) -> EntityRecord | None:
        for rec in state.entities:
            if rec.name == name or name in rec.aliases:
                return rec.model_copy()
        return None


@dataclass
class _Session:
    dialogue_id: str
    turn: int = 0
    next_id: int = 1
    entities: list[StoredEntity] = field(default_factory=list)
    by_name: dict[str, StoredEntity] = field(default_factory=dict)


@dataclass
class ResolutionOutcome:
    """一条 mention 在 store 侧的落地结果，供 API 响应与调试。"""

    mention: str
    resolution: object
    entity_id: str | None = None
    status: str = "pending"  # resolved / new / ambiguous / unresolved
