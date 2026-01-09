# 反电诈 Demo（Synergized LLMs + KGs）— Neo4j + Moonshot LLM

## 环境准备

- macOS / Python 3.10+
- Docker 与 Docker Compose

## 安装依赖

说明：macOS 的系统 Python 可能启用 PEP 668，建议使用虚拟环境安装依赖。

```bash
# 在仓库根目录创建虚拟环境
python3 -m venv .venv
# 安装 Demo 依赖（路径包含空格，需加引号）
.venv/bin/python -m pip install -r 'KE/Synergized LLMs + KGs/demo/requirements.txt'
```

## 启动 Neo4j 容器

说明：不同环境可能使用 `docker compose` 或 `docker-compose`。如遇到命令不存在，请切换另一种写法。

```bash
# 进入 Demo 目录
cd 'KE/Synergized LLMs + KGs/demo'
docker-compose up -d neo4j
```

- Web 控制台: `http://localhost:7474`（默认 `neo4j/password`）
- Bolt 端口: `bolt://localhost:7687`

## 生成样本数据

说明：生成合成数据，用于复现“归集/长链/正常样本”的最小可运行链路。

```bash
# 从仓库根目录执行（路径包含空格，需加引号）
.venv/bin/python 'KE/Synergized LLMs + KGs/demo/etl/generate_data.py'
```

- 生成至 `demo/data/`，容器已映射到 Neo4j `import` 目录

## 导入图谱

说明：通过 Bolt Driver 执行 `etl/cypher_import.cypher`，在 Neo4j 中写入顶点与边。

```bash
# 从仓库根目录执行（路径包含空格，需加引号）
.venv/bin/python 'KE/Synergized LLMs + KGs/demo/etl/load_to_neo4j.py'
```

- 包含约束与设备日志、交易转账关系

## 启动 API

说明：从本目录的上级路径启动，以 `demo` 作为顶层包导入。

```bash
# 从仓库根目录进入 Demo 包所在目录
cd 'KE/Synergized LLMs + KGs'
.venv/bin/python -m uvicorn demo.api.server:app --reload --port 8000
```

- 健康检查: `http://localhost:8000/health`
- 子图检索: `GET /graph/subgraph?account_id=<Axxx>&hops=2&window_minutes=60`
- 判决接口: `POST /agent/verdict`（JSON: `{"account_id": "A0010", "window_minutes": 60}`）
- LLM 判决接口: `POST /agent/verdict_llm`（需设置 `MOONSHOT_API_KEY`）

## 配置 Moonshot LLM

说明：推荐使用本地配置文件（避免在 shell 历史中留下密钥），并由程序启动时自动加载。

```bash
export MOONSHOT_API_KEY=YOUR_KEY
export MOONSHOT_API_URL=https://api.moonshot.cn/v1/chat/completions
export MOONSHOT_MODEL=kimi-k2-turbo-preview
```

或使用配置文件（已在启动时自动加载）：

```bash
# 复制示例为本地文件
cp demo/configs/.env.example demo/configs/.env.local
# 编辑 demo/configs/.env.local，填写密钥与模型
# 或使用 JSON:
cp demo/configs/env.example.json demo/configs/env.local.json
```

- `.gitignore` 已忽略 `demo/configs/.env*` 与 `env.local.json`，不会提交密钥。
- 兼容从任意子目录启动：加载逻辑会沿当前目录向上回溯并查找 `.env` / `.env.local`。

说明：以下命令用于验证“规则证据 + LLM 解释”端到端链路。

```bash
curl -s -X POST http://127.0.0.1:8000/agent/verdict_llm \
  -H 'Content-Type: application/json' \
  -d '{"account_id":"A0010","window_minutes":120}' | python -m json.tool
```

## 场景验证

- 数据包含“分散转入 → 集中转出”“长链清洗”“正常样本”，判决依据入度、来源离散度、快进快出、设备社区大小、备注关键词。
- 处置建议分级：`Notify` / `Delay` / `Block`。

## 全链路自检（2025-12-23 实测输出）

```bash
# 1) 启动 Neo4j
cd 'KE/Synergized LLMs + KGs/demo'
docker-compose up -d neo4j
```

```text
# 输出示例
[+] up 1/1
 ✔ Container demo-neo4j Recreated
```

```bash
# 2) 生成合成数据
cd '/Users/wangtianqing/Project/AI-fundermentals'
.venv/bin/python 'KE/Synergized LLMs + KGs/demo/etl/generate_data.py'
```

```text
# 输出示例
Data generated in /Users/wangtianqing/Project/AI-fundermentals/KE/Synergized LLMs + KGs/demo/data
```

```bash
# 3) 导入 Neo4j
cd '/Users/wangtianqing/Project/AI-fundermentals'
.venv/bin/python 'KE/Synergized LLMs + KGs/demo/etl/load_to_neo4j.py'
```

```text
# 输出示例
Import completed
```

```bash
# 4) 验证图谱已落地（节点数）
cd 'KE/Synergized LLMs + KGs/demo'
docker exec demo-neo4j cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n) AS nodes"
```

```text
# 输出示例
nodes
315
```

```bash
# 5) 启动 API（另开一个终端窗口执行）
cd 'KE/Synergized LLMs + KGs'
/Users/wangtianqing/Project/AI-fundermentals/.venv/bin/python -m uvicorn demo.api.server:app --port 8000
```

```bash
# 6) 健康检查
curl -s http://127.0.0.1:8000/health
```

```text
# 输出示例
{"status":"ok"}
```

```bash
# 7) 选取“归集候选账户”（按两天内入账笔数排序）
cd 'KE/Synergized LLMs + KGs/demo'
docker exec demo-neo4j cypher-shell -u neo4j -p password \
  "MATCH (src:Account)-[t:TRANSFER]->(dst:Account) \
   WHERE t.timestamp >= datetime() - duration({days: 2}) \
   RETURN dst.account_id AS account, count(t) AS inbound, count(DISTINCT src.account_id) AS srcs \
   ORDER BY inbound DESC LIMIT 5"
```

```text
# 输出示例
account, inbound, srcs
"A0250", 36, 9
"A0041", 34, 8
"A0020", 32, 23
"A0260", 27, 19
"A0220", 19, 17
```

```bash
# 8) 基础判分（规则版）
curl -s -X POST http://127.0.0.1:8000/agent/verdict \
  -H 'Content-Type: application/json' \
  -d '{"account_id":"A0250","window_minutes":2880}' \
  | python -c "import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))"
```

```text
# 输出示例
{
  "account_id": "A0250",
  "risk_type": "资金归集",
  "score": 8,
  "evidence": {
    "in_degree": 36,
    "source_diversity": 9,
    "fast_out": false,
    "device_community_size": 30,
    "memo_risk": true
  },
  "suggested_action": "Block"
}
```

```bash
# 9) 子图取证（2 跳，2 天时间窗）
curl -s 'http://127.0.0.1:8000/graph/subgraph?account_id=A0250&hops=2&window_minutes=2880' \
  | python -c "import json,sys; obj=json.load(sys.stdin); print(json.dumps({'nodes':len(obj.get('nodes',[])),'edges':len(obj.get('edges',[]))}, ensure_ascii=False))"
```

```text
# 输出示例（仅展示规模，便于 Demo）
{"nodes": 82, "edges": 670}
```

```bash
# 10) LLM 判决（在规则证据基础上生成可审计解释）
curl -s -X POST http://127.0.0.1:8000/agent/verdict_llm \
  -H 'Content-Type: application/json' \
  -d '{"account_id":"A0250","window_minutes":2880}' \
  | python -c "import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))"
```

```text
# 输出示例（字段已结构化，便于审计）
{
  "account_id": "A0250",
  "risk_type": "涉诈可疑账户",
  "score": 8,
  "evidence": {
    "in_degree": 36,
    "source_diversity": 9,
    "fast_out": false,
    "device_community_size": 30,
    "memo_risk": true
  },
  "suggested_action": "维持Block，同步报送反诈平台；冻结关联设备指纹下所有账户；发起回溯调查，追溯上游9个资金来源账户；向客户发送风险告知并限期提交合法资金来源证明，逾期不解冻。",
  "llm": {
    "risk_type": "涉诈可疑账户",
    "evidence_points": "1. 36个不同对手方短期内集中转入，呈现‘分散转入’特征；2. 资金来源仅9个不同渠道，集中度异常高，疑似同一团伙控制；3. 交易备注出现涉诈关键词，命中‘memo_risk’规则；4. 设备关联社区达30个账户，远超正常用户范围，存在设备农场嫌疑；5. 未触发‘fast_out’，暂未见立即转出，但符合‘养卡期’行为模式。",
    "verdict": "综合判定该账户处于电信网络诈骗链条的‘资金归集’阶段，风险等级高，具备继续转移赃款或洗钱的潜在威胁。",
    "suggested_action": "维持Block，同步报送反诈平台；冻结关联设备指纹下所有账户；发起回溯调查，追溯上游9个资金来源账户；向客户发送风险告知并限期提交合法资金来源证明，逾期不解冻。"
  }
}
```

## 参考

[1] Moonshot AI. "Moonshot API Reference." 访问日期: 2025-12-23. [Online]. Available: https://platform.moonshot.cn/docs/api-reference
