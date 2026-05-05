# LLM + 知识图谱协同：反电诈案例

单独看 LLM，它能写、能算、但对事实不严谨；单独看知识图谱，它结构严谨、但缺乏自然语言理解。两者协同（**Synergized LLMs + KGs**）的价值，要放到「业务场景」里才讲得清——本目录以**银行反电信网络诈骗**为标本案例，把「设计方案 → 可运行 demo → 汇报材料」一次性打通：图谱承担结构化证据与资金流追踪，LLM 承担自然语言解释与策略建议。

## 1. 目录内容

### 1.1 设计方案（Design）

- **[银行反电诈智能系统设计方案](anti_fraud_design.md)** — 完整的系统设计文档：
  - **背景与挑战**：电诈「短时间 / 高频次 / 多层级」资金流转特征为何让传统规则引擎失效。
  - **核心目标**：分钟级实时拦截（以「黄金 5 分钟」为工程 SLA 示例）、团伙风险模式识别、可解释归因。
  - **架构设计**：图谱建模（账户 / 设备 / 交易）、LLM 证据要点生成、规则引擎 + 图算法 + LLM 三层联动。
  - **法规合规**：引用《反电信网络诈骗法》、银发〔2016〕86 号、银发〔2016〕261 号等文件，明确强处置由确定性证据链驱动、LLM 仅作解释辅助的边界。
  - **工程化路线**：从 demo 到生产的演进路径。

### 1.2 可运行 Demo（`demo/`）

- **[反欺诈 Demo 源码](demo/README.md)** — 完整 258 行说明的端到端示例系统：
  - `etl/generate_data.py` + `load_to_neo4j.py` — 合成客户 / 账户 / 设备 / 交易数据并导入 Neo4j。
  - `agent/` — LLM 客户端与 `verdict` 决策模块。
  - `api/server.py` — 查询/研判 API 服务。
  - `docker-compose.yml` — 一键拉起 Neo4j 依赖。

### 1.3 配套材料

- **`assets/anti_fraud_system_design.pptx`** — 系统设计汇报 PPT，可用于内部评审或布道演讲。

## 2. 阅读顺序

1. **先读** [`anti_fraud_design.md`](anti_fraud_design.md) 建立对场景与架构的整体认知。
2. **再跑** [`demo/`](demo/README.md)：起 Neo4j → 生成并导入数据 → 启动 API → 调用研判接口。
3. **可选**：用 `assets/anti_fraud_system_design.pptx` 对外分享方案。

## 3. 与其他目录的关系

- **理论支撑**见 [`../graph_rag/`](../graph_rag/README.md)：GraphRAG / KAG 为何是这个场景的正解。
- **图数据库底座**见 [`../knowledge_graph/`](../knowledge_graph/README.md)：Neo4j + Cypher 的基础，demo 中的 `docker-compose.yml` 与该目录的教程共用。

## 4. 相关资源

- [智能体系统（Agentic System）](../../08_agentic_system/README.md) — 多智能体协作与上下文工程。
- [AI 基础设施模块总览](../../README.md)
