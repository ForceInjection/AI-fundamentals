# 知识图谱：Neo4j 与 Cypher

要用 GraphRAG，先得有图数据库；要用 Neo4j，先得会 Cypher。本目录把「查询语言」和「部署实战」拆成两份互补文档，配合 [`../synergized_llms_kgs/demo`](../synergized_llms_kgs/demo) 中的真实数据集（银行反欺诈），让读者一条线从语法学到图上线。

## 1. 阅读顺序

先学查询语言，再动手部署——两份文档共用同一份 anti-fraud 数据集，能无缝衔接。

### 1.1 查询语言：Cypher

- **[Neo4j Cypher 查询语言权威指南](neo4j_cypher_tutorial.md)** — 基于 Neo4j 官方文档重组的从入门到进阶教程，围绕银行反欺诈场景（`Customer` / `Account` / `Device` / `Transaction`）讲解节点、关系、属性、标签四大构件，覆盖 `MATCH / WHERE / RETURN` 基础查询、聚合与排序、最短路径（`shortestPath`）、变长关系（`-[*1..3]->`）、`MERGE` 幂等写入等实战技巧。

### 1.2 实战部署：Docker + Neo4j Browser

- **[Neo4j 快速上手实战指南](neo4j_handson_guide.md)** — 配合 `neo4j_cypher_tutorial.md` 的动手篇：
  - 用 `docker-compose up -d` 拉起 Neo4j 容器（复用项目内置配置）。
  - 通过 `http://localhost:7474` 登录 Neo4j Browser。
  - 导入银行反欺诈样本数据（客户、账户、设备、转账链）。
  - 在 WebUI 中交互式执行 Cypher，观察图谱可视化结果。

## 2. 与其他目录的关系

- **上游场景**见 [`../graph_rag/`](../graph_rag/README.md)：GraphRAG / KAG 为什么需要图数据库。
- **配套数据与容器**见 [`../synergized_llms_kgs/demo/`](../synergized_llms_kgs/demo/README.md)：现成的 `docker-compose.yml` 与反欺诈样本数据。

## 3. 相关资源

- [RAG 基础](../rag_basics/README.md) — 图检索 vs 向量检索的场景分工。
- [智能体系统（Agentic System）](../../08_agentic_system/README.md) — 当 Agent 需要在图上做规划时的协同模式。
