"""离线规则引擎 + 演示入口。

RuleResolver 是「无 LLM 时的规则路径」：保留 v1 可讲的骨架
（候选筛选 → 特征加权 → softmax → 决策 + 中文代词表），修正 v1 的
三处硬伤——ID 由 EntityStore 统一分配（不再轮内自增）、置信度直接取
softmax 最大概率（不再 2p1-p2）、阈值 0.5（不再永不触发的 0.3）。

运行：
    python demo.py            # 有 DEEPSEEK_API_KEY 走 LLM，否则自动切规则路径
    python demo.py --rules    # 强制规则路径（教学对照）
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

from entity_store import EntityStore
from resolver import LLMResolver, ResolverError
from schemas import (
    DialogueState,
    EntityRecord,
    EntityType,
    Resolution,
    ResolverError,
)

# 中文第三人称代词表（v1 coreference_resolution.py 的遗产，精简后保留）
PRONOUNS: dict[str, str | None] = {
    "他": "PERSON",
    "她": "PERSON",
    "它": None,  # 指物，类型不定
    "他们": "PERSON",
    "她们": "PERSON",
    "它们": None,
}

_NER: object | None = None
_NER_TRIED = False


def _try_load_spacy():
    """spaCy zh 模型可选加载：失败返回 None，规则路径照常工作（仅 NER 退化）。"""
    global _NER, _NER_TRIED
    if _NER_TRIED:
        return _NER
    _NER_TRIED = True
    try:
        import spacy

        _NER = spacy.load("zh_core_web_sm")
        print("[demo] spaCy zh_core_web_sm 已加载：规则路径具备真实 NER")
    except Exception as e:  # 模型未安装 / spacy 未装
        print(f"[demo] spaCy zh 模型不可用（{type(e).__name__}），NER 降级：只认显式实体名与代词")
        _NER = None
    return _NER


class RuleResolver:
    """规则路径：NER（可选 spaCy）+ 代词匹配 + 显著性加权消解。"""

    def __init__(self) -> None:
        self._nlp = _try_load_spacy()

    # ---------- 提取 ----------

    def extract(self, text: str, turn: int, known_names: dict[str, EntityType]) -> tuple[list[str], list[str]]:
        """返回 (本轮新出现的实体名, 本轮的指代 mention 列表)。"""
        entities: list[str] = []
        if self._nlp is not None:
            doc = self._nlp(text)
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "PRODUCT", "LOC", "WORK_OF_ART"} and ent.text not in PRONOUNS:
                    entities.append(ent.text)
        else:
            # 无 spaCy：把「已知名之外的连续中文专名」交给不了——诚实降级，只标记已知名
            for name in known_names:
                if name in text:
                    entities.append(name)

        mentions: list[str] = []
        for p in PRONOUNS:
            if re.search(re.escape(p), text):
                mentions.append(p)
        return entities, mentions

    # ---------- 消解 ----------

    def resolve_mentions(
        self,
        mentions: list[str],
        candidates: dict[str, EntityType],
        salience: dict[str, float],
        turn: int,
    ) -> list[Resolution]:
        """对每个代词 mention 打分消解。

        候选过滤：类型一致（代词类型已知时）；打分特征：recency + frequency，
        归一化取 argmax，p1 < 0.6 判 AMBIGUOUS。阈值取 0.6 而不是 0.5：
        规则路径没有语义信号，两个候选 50.2% vs 49.8% 时押注是赌博，宁可挂起；
        （v1 的 0.3 在 2-3 候选下数学上永不触发，是装饰品。）
        """
        results: list[Resolution] = []
        for m in mentions:
            want_type = PRONOUNS.get(m)
            cand = {
                name: t
                for name, t in candidates.items()
                if want_type is None or t == want_type or t == EntityType.OTHER
            }
            if not cand:
                results.append(Resolution(mention_text=m, turn_id=turn, resolved_to="AMBIGUOUS",
                                          confidence=0.0, reasoning="无类型一致的候选实体"))
                continue
            weights = {name: max(salience.get(name, 0.0), 1e-4) for name in cand}
            total = sum(weights.values())
            probs = {name: w / total for name, w in weights.items()}
            best = max(probs, key=probs.get)
            p1 = probs[best]
            if p1 < 0.6:
                results.append(Resolution(
                    mention_text=m, turn_id=turn, resolved_to="AMBIGUOUS", confidence=round(p1, 3),
                    reasoning=f"候选 {list(cand)} 概率分散（top={p1:.2f}），不猜",
                ))
                continue
            results.append(Resolution(
                mention_text=m, turn_id=turn, resolved_to=best, confidence=round(p1, 3),
                reasoning=f"规则路径：显著性最高候选（softmax p={p1:.2f}）",
            ))
        return results

    # ---------- 组装状态（与 LLM 路径共用 EntityStore 记账） ----------

    def build_state(
        self,
        text: str,
        turn: int,
        new_entities: list[str],
        known: dict[str, EntityRecord],
        salience: dict[str, float],
    ) -> DialogueState:
        _, mentions = self.extract(text, turn, {r.name: r.entity_type for r in known.values()})
        # 本轮新引入的实体也参与消解候选（同轮近因是规则路径最强的信号之一）
        candidates = {r.name: r.entity_type for r in known.values()}
        cand_salience = dict(salience)
        for name in new_entities:
            candidates[name] = self._guess_type(name)
            cand_salience[name] = math.log1p(1)  # 同轮出现：recency 项取满
        resolutions = self.resolve_mentions(mentions, candidates, cand_salience, turn)

        entities: list[EntityRecord] = []
        for name in new_entities:
            entities.append(EntityRecord(
                name=name,
                entity_type=self._guess_type(name),
                aliases=[],
                attributes={},
                first_seen_turn=turn,
                last_mention_turn=turn,
                mention_count=1,
            ))
        for res in resolutions:
            source = known.get(res.resolved_to)
            if source is not None:
                rec = source.model_copy()
                rec.last_mention_turn = turn
                rec.mention_count += 1
                if res.mention_text != rec.name and res.mention_text not in rec.aliases:
                    rec.aliases.append(res.mention_text)
                entities.append(rec)
        return DialogueState(entities=entities, resolutions=resolutions, current_turn=turn)

    @staticmethod
    def _guess_type(name: str) -> EntityType:
        if _NER is not None:
            doc = _NER(name)
            if doc.ents:
                label = doc.ents[0].label_
                return {"PERSON": EntityType.PERSON, "ORG": EntityType.ORG, "LOC": EntityType.LOCATION}.get(
                    label, EntityType.OTHER
                )
        return EntityType.OTHER


# ---------- 演示 ----------

DEMO_TURNS = [
    "张三加入了星辰项目，他是后端工程师。",
    "李四也在星辰项目，她负责前端。",
    "他的第一个任务是重构登录模块。",
    "那个项目月底要上线，它的负责人是谁？",
    "上次会议说的事和她有关吗？",
]


def run_llm_demo(history: list[dict[str, str]], store: EntityStore, user_turn: str) -> DialogueState:
    resolver = LLMResolver(api_key=os.environ["DEEPSEEK_API_KEY"])
    dialogue_id = "demo"
    summary = store.context_summary(dialogue_id)
    history.append({"role": "user", "content": user_turn})
    state = resolver.resolve(history, entity_summary=summary)
    store.apply_state(dialogue_id, state)
    history.append({"role": "assistant", "content": "（已解析）"})
    return state


def run_rules_demo(store: EntityStore, user_turn: str) -> DialogueState:
    rr = RuleResolver()
    dialogue_id = "demo"
    sess = store.session(dialogue_id)
    turn = sess.turn + 1
    known = {e.record.name: e.record for e in sess.entities}
    salience = {e.record.name: e.salience(turn) for e in sess.entities}
    new_ents, _ = rr.extract(user_turn, turn, {n: r.entity_type for n, r in known.items()})
    new_ents = [n for n in new_ents if n not in known]
    state = rr.build_state(user_turn, turn, new_ents, known, salience)
    store.apply_state(dialogue_id, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="多轮指代消解演示")
    parser.add_argument("--rules", action="store_true", help="强制规则路径（教学对照）")
    args = parser.parse_args()

    has_key = bool(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    use_llm = has_key and not args.rules
    store = EntityStore()
    history: list[dict[str, str]] = []

    print(f"=== 指代消解演示（路径：{'LLM' if use_llm else '规则'}）===\n")
    for i, user_turn in enumerate(DEMO_TURNS, start=1):
        print(f"[第{i}轮] {user_turn}")
        if use_llm:
            try:
                state = run_llm_demo(history, store, user_turn)
            except ResolverError as e:
                print(f"  LLM 通道失败（{e}），本Demo切换规则路径\n")
                use_llm = False
                state = run_rules_demo(store, user_turn)
        else:
            state = run_rules_demo(store, user_turn)

        for res in state.resolutions:
            print(f"  {res.mention_text!r} → {res.resolved_to}  (conf={res.confidence}, {res.reasoning})")
        ents = ", ".join(f"{e.name}({e.entity_type.value})" for e in state.entities) or "无"
        print(f"  实体：{ents}\n")

    print("=== 最终实体账本 ===")
    for ent, score in store.top_entities("demo", k=10):
        print(f"  {ent.id}  {ent.record.name}  提及{len(ent.mention_turns)}次  显著性{score:.2f}  "
              f"属性{ent.record.attributes or '-'}  别名{ent.record.aliases or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
