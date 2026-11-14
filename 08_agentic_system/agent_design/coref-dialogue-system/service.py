"""FastAPI 服务：真实状态回灌的多轮闭环。

v1 的断链教训：每请求伪造 dialogue_state（current_turn 恒 1、显著性恒 0.5），
API 层的「多轮」从未真正发生。v2 的闭环是硬性的：

    用户消息 → history（含全部历史轮）→ LLMResolver → EntityStore.apply_state
    → 下一轮请求自动带上更新后的 history 与 entity_summary

会话状态存内存（dict），生产替换点见 entity_store.py 模块注释。
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException

from entity_store import EntityStore
from resolver import LLMResolver
from schemas import ChatRequest, ChatResponse, Resolution, ResolverError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="coref-dialogue-system", version="2.0.0")

_store = EntityStore()
_histories: dict[str, list[dict[str, str]]] = {}  # dialogue_id -> 完整对话历史


def _get_resolver() -> LLMResolver | None:
    """按环境变量构造 resolver；无 Key 返回 None（由调用方降级）。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return LLMResolver(api_key=api_key)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if req.dialogue_id not in _histories:
        _histories[req.dialogue_id] = []
    history = _histories[req.dialogue_id]
    history.append({"role": "user", "content": req.message})
    turn = sum(1 for m in history if m["role"] == "user")

    resolver = _get_resolver()
    if resolver is not None:
        summary = _store.context_summary(req.dialogue_id)
        try:
            state = resolver.resolve(history, entity_summary=summary)
        except ResolverError as e:
            logger.error("LLM 通道失败，本请求降级为规则路径: %s", e)
            return _rule_fallback(req, history, turn)
    else:
        return _rule_fallback(req, history, turn)

    outcomes = _store.apply_state(req.dialogue_id, state)
    history.append({"role": "assistant", "content": "（已解析本轮指代）"})
    for o in outcomes:
        logger.info("turn %d mention=%r → %s (%s)", turn, o.mention, o.entity_id, o.status)

    return ChatResponse(
        dialogue_id=req.dialogue_id,
        turn_id=turn,
        reply="已解析本轮指代。",
        resolutions=state.resolutions,
        entities=_store.snapshot(req.dialogue_id),
        used_path="llm",
    )


def _rule_fallback(req: ChatRequest, history: list[dict[str, str]], turn: int) -> ChatResponse:
    """无 Key / LLM 失败时的降级路径：demo.py 的规则引擎，显著性加权消解。"""
    from demo import RuleResolver  # 延迟 import，避免无 Key 环境强依赖 spaCy

    rr = RuleResolver()
    sess = _store.session(req.dialogue_id)
    known = {e.record.name: e.record for e in sess.entities}
    salience = {e.record.name: e.salience(turn) for e in sess.entities}
    new_ents, _ = rr.extract(req.message, turn, {n: r.entity_type for n, r in known.items()})
    new_ents = [n for n in new_ents if n not in known]
    state = rr.build_state(req.message, turn, new_ents, known, salience)
    _store.apply_state(req.dialogue_id, state)
    history.append({"role": "assistant", "content": "（已解析本轮指代[规则路径]）"})
    return ChatResponse(
        dialogue_id=req.dialogue_id,
        turn_id=turn,
        reply="已解析本轮指代（规则路径）。",
        resolutions=state.resolutions,
        entities=_store.snapshot(req.dialogue_id),
        used_path="rules",
    )


@app.get("/dialogues/{dialogue_id}/state")
def get_state(dialogue_id: str) -> dict:
    if dialogue_id not in _histories:
        raise HTTPException(status_code=404, detail="dialogue not found")
    return {
        "dialogue_id": dialogue_id,
        "turn": _store.session(dialogue_id).turn,
        "entities": [e.record.model_dump() for e, _ in _store.top_entities(dialogue_id, k=20)],
    }


@app.delete("/dialogues/{dialogue_id}")
def delete_dialogue(dialogue_id: str) -> dict:
    ok = _store.drop(dialogue_id)
    _histories.pop(dialogue_id, None)
    if not ok:
        raise HTTPException(status_code=404, detail="dialogue not found")
    return {"deleted": dialogue_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8600)
