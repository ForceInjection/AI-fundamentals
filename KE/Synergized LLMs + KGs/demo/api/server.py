"""FastAPI 服务入口。

提供三类能力：
- 健康检查：`GET /health`
- 图谱子图检索：`GET /graph/subgraph`
- 风险判决：`POST /agent/verdict`（规则/特征版）与 `POST /agent/verdict_llm`（LLM 增强解释版）

运行方式（在本目录上级执行）：
- `python -m demo.api.server`（仅用于导入检查）
- `uvicorn demo.api.server:app --reload --port 8000`
"""

import os
from typing import Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from neo4j import GraphDatabase
from demo.agent.verdict import RiskVerdict
from demo.agent.llm_client import llm_verdict
from demo.configs.config import load_env

load_env()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

app = FastAPI()
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


class Alert(BaseModel):
    account_id: str
    window_minutes: int = 60
    rule: Optional[str] = None


@app.get("/health")
def health():
    with driver.session() as session:
        session.run("RETURN 1")
    return {"status": "ok"}


@app.get("/graph/subgraph")
def subgraph(account_id: str, hops: int = 2, window_minutes: int = 60):
    """返回以 `account_id` 为中心的子图（用于前端展示/研判取证）。

实现方式：
- 使用 APOC 的 `apoc.path.subgraphNodes` 获取限定跳数的节点集合；
- 再补充节点间边集合；
- 对 `TRANSFER` 边可按时间窗过滤（Demo 中为可扩展参数）。
    """
    with driver.session() as session:
        result = session.run(
            """
            WITH datetime() - duration({minutes: $window_minutes}) AS since
            MATCH (a:Account {account_id: $aid})
            CALL apoc.path.subgraphNodes(a, {maxLevel: $hops}) YIELD node
            WITH since, collect(node) AS nodes
            MATCH (n)-[r]->(m)
            WHERE n IN nodes AND m IN nodes AND (type(r) <> 'TRANSFER' OR r.timestamp >= since)
            RETURN
              [n IN nodes | {id: id(n), labels: labels(n), props: properties(n)}] AS nodes,
              collect({type: type(r), props: properties(r), start: id(startNode(r)), end: id(endNode(r))}) AS edges
            """,
            {"aid": account_id, "hops": hops, "window_minutes": int(window_minutes)},
        )
        rec = result.single()
        return {"nodes": rec["nodes"], "edges": rec["edges"]}


@app.post("/agent/verdict")
def agent_verdict(alert: Alert):
    """规则/特征版判决：输出结构化证据与建议处置。"""
    rv = RiskVerdict(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    out = rv.analyze(alert.account_id, alert.window_minutes)
    rv.close()
    return out


@app.post("/agent/verdict_llm")
def agent_verdict_llm(alert: Alert):
    """LLM 增强判决：在规则证据基础上生成可审计研判结论。

注意：
- LLM 输出不直接驱动冻结等强处置，仅作为解释与复核辅助；
- 强处置应由确定性证据链与审批链路驱动（Demo 仅返回建议，不执行实际处置）。
    """
    rv = RiskVerdict(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    base = rv.analyze(alert.account_id, alert.window_minutes)
    rv.close()
    enriched = llm_verdict(base["evidence"], base["suggested_action"])
    return {
        "account_id": alert.account_id,
        "risk_type": enriched.get("risk_type", base["risk_type"]),
        "score": base["score"],
        "evidence": base["evidence"],
        "suggested_action": enriched.get("suggested_action", base["suggested_action"]),
        "llm": enriched,
    }
