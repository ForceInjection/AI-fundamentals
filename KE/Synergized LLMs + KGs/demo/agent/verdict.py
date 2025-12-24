"""规则/特征版风险判决器。

该模块实现“KG 侧确定性判决”的最小可运行版本：
- 从 Neo4j 拉取时间窗内的入/出账边；
- 计算入度、来源离散度、快进快出等资金拓扑特征；
- 计算设备社区规模作为团伙环境证据的简化代理；
- 形成结构化证据并输出处置建议分级（Notify/Delay/Block）。

设计目的：
- 在线拦截优先依赖可控的确定性链路；
- LLM 仅在 `/agent/verdict_llm` 中用于解释与补充审计结论。
"""

from datetime import datetime
from typing import Dict, Any, List, Tuple
from neo4j import GraphDatabase


class RiskVerdict:
    """基于图谱特征的风险判决器。"""

    def __init__(self, uri: str, user: str, password: str):
        """创建 Neo4j Driver。

        参数：
        - uri：Bolt 地址，例如 `bolt://localhost:7687`
        - user/password：数据库账号
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """释放 Driver 资源。"""
        self.driver.close()

    def _query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行 Cypher 并返回记录列表（dict）。"""
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [r.data() for r in result]

    def _to_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if hasattr(value, "to_native"):
            native = value.to_native()
            if isinstance(native, datetime):
                return native
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return datetime.fromisoformat(str(value))

    def _fetch_txns(self, account_id: str, window_minutes: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """拉取时间窗内入账与出账转账边。"""
        inbound = self._query(
            """
            MATCH (src:Account)-[t:TRANSFER]->(dst:Account {account_id: $aid})
            WHERE t.timestamp >= datetime() - duration({minutes: $window_minutes})
            RETURN src.account_id AS src, dst.account_id AS dst, t.amount AS amount, t.timestamp AS ts, t.memo AS memo
            ORDER BY ts ASC
            """,
            {"aid": account_id, "window_minutes": int(window_minutes)},
        )
        outbound = self._query(
            """
            MATCH (src:Account {account_id: $aid})-[t:TRANSFER]->(dst:Account)
            WHERE t.timestamp >= datetime() - duration({minutes: $window_minutes})
            RETURN src.account_id AS src, dst.account_id AS dst, t.amount AS amount, t.timestamp AS ts, t.memo AS memo
            ORDER BY ts ASC
            """,
            {"aid": account_id, "window_minutes": int(window_minutes)},
        )
        return inbound, outbound

    def _source_diversity(self, inbound: List[Dict[str, Any]]) -> int:
        """入账来源账户去重计数，用于衡量“来源离散度”。"""
        return len({t["src"] for t in inbound})

    def _fast_out(self, inbound: List[Dict[str, Any]], outbound: List[Dict[str, Any]], threshold_seconds: int = 60) -> bool:
        """判断“快进快出”：首次出账是否在首次入账后阈值内发生。"""
        if not inbound or not outbound:
            return False
        first_in_ts = self._to_datetime(inbound[0]["ts"])
        first_out_ts = self._to_datetime(outbound[0]["ts"])
        return (first_out_ts - first_in_ts).total_seconds() <= int(threshold_seconds)

    def _in_degree(self, account_id: str, window_minutes: int) -> int:
        """时间窗内入账笔数（入度）。"""
        rows = self._query(
            """
            MATCH (:Account)-[t:TRANSFER]->(dst:Account {account_id: $aid})
            WHERE t.timestamp >= datetime() - duration({minutes: $window_minutes})
            RETURN count(t) AS cnt
            """,
            {"aid": account_id, "window_minutes": int(window_minutes)},
        )
        return int(rows[0]["cnt"]) if rows else 0

    def _device_community(self, account_id: str) -> int:
        """设备社区规模：与目标账户持有人共用设备的客户数量（简化团伙信号）。"""
        rows = self._query(
            """
            MATCH (c:Customer)-[:OWNS]->(:Account {account_id: $aid})
            MATCH (c)-[:USED_DEVICE]->(d:Device)
            MATCH (other:Customer)-[:USED_DEVICE]->(d)
            WITH collect(DISTINCT other.cust_id) AS others
            RETURN size(others) AS community_size
            """,
            {"aid": account_id},
        )
        return int(rows[0]["community_size"]) if rows else 0

    def _memo_risk(self, inbound: List[Dict[str, Any]]) -> bool:
        """备注关键词命中（Demo 版：投资/保证金/解冻）。"""
        keywords = {"投资", "保证金", "解冻"}
        return any(str(t.get("memo", "")) in keywords for t in inbound)

    def analyze(self, account_id: str, window_minutes: int = 60) -> Dict[str, Any]:
        """输出结构化研判结果。

        返回字段：
        - `risk_type`：风险类型（Demo 版以“资金归集/可疑资金/低风险”分类）
        - `score`：规则打分
        - `evidence`：可审计证据字典
        - `suggested_action`：处置建议分级（Notify/Delay/Block）
        """
        inbound, outbound = self._fetch_txns(account_id, window_minutes)
        diversity = self._source_diversity(inbound)
        indeg = self._in_degree(account_id, window_minutes)
        fast = self._fast_out(inbound, outbound)
        community = self._device_community(account_id)
        memo_flag = self._memo_risk(inbound)

        evidence = {
            "in_degree": indeg,
            "source_diversity": diversity,
            "fast_out": fast,
            "device_community_size": community,
            "memo_risk": memo_flag,
        }

        score = 0
        if indeg >= 10:
            score += 3
        if diversity >= 5:
            score += 2
        if fast:
            score += 3
        if community >= 5:
            score += 2
        if memo_flag:
            score += 1

        if score >= 7:
            action = "Block"
            risk_type = "资金归集"
        elif score >= 4:
            action = "Delay"
            risk_type = "可疑资金"
        else:
            action = "Notify"
            risk_type = "低风险"

        return {
            "account_id": account_id,
            "risk_type": risk_type,
            "score": score,
            "evidence": evidence,
            "suggested_action": action,
        }
