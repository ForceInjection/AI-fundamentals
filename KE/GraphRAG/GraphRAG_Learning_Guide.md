# Knowledge Graphs for RAG 深度学习教案

本教案基于 DeepLearning.AI 与 Neo4j 合作的 "Knowledge Graphs for RAG" 课程设计，旨在为学习者提供从理论到实践的完整路径。课程围绕构建一个基于 SEC 财报（10-K 表格）的问答系统展开，通过实战掌握 GraphRAG 的核心技术。

---

## 课程信息

- **课程名称**: Knowledge Graphs for RAG (GraphRAG)
- **适用人群**: AI 工程师、数据科学家、希望提升 RAG 系统性能的开发者
- **先决条件**:
  - 熟悉 Python 编程
  - 了解基本的 LLM 和 LangChain 概念
  - 拥有 OpenAI API Key
  - 已安装 Docker（用于运行 Neo4j）
- **学习目标**:
  1. 理解知识图谱（KG）如何解决传统 RAG 的上下文缺失问题。
  2. 掌握 Neo4j 图数据库的基本操作与 Cypher 查询语言。
  3. 学会使用 LangChain 将非结构化文本转化为结构化图谱（Text2Graph）。
  4. 构建一个结合向量检索与图检索的混合检索（Hybrid Retrieval）QA 系统。

---

## 课程大纲

| **模块**     | **主题**                   | **核心内容**                              | **时长** |
| :----------- | :------------------------- | :---------------------------------------- | :------- |
| **Module 1** | **基础理论与环境搭建**     | KG 概念、GraphRAG 优势、Neo4j Docker 部署 | 1 小时   |
| **Module 2** | **Cypher 查询语言入门**    | 节点、关系、属性、MATCH/RETURN 语句       | 1.5 小时 |
| **Module 3** | **非结构化文本的图谱构建** | Text2Graph 流程、Schema 定义、LLM 提取    | 2 小时   |
| **Module 4** | **混合检索与问答系统**     | 向量索引、图谱检索、LangChain 集成        | 2 小时   |
| **Module 5** | **实战项目：SEC 财报分析** | 处理真实 10-K 数据、多文档关联、复杂推理  | 2.5 小时 |

---

## 详细教学内容

### Module 1: 基础理论与环境搭建

#### 1.1 理论：为什么需要 GraphRAG？

- **传统 RAG 的局限性**:
  - **切片（Chunking）导致语义断裂**: 简单的文本切分可能将实体与其属性或关系分开。
  - **缺乏全局视角**: 无法回答跨文档或跨段落的综合性问题（例如："A 公司的供应商 B 的高管 C 之前在哪工作？"）。
- **知识图谱的优势**:
  - **显式关系**: 直接存储实体间的连接。
  - **结构化上下文**: 提供更精准的上下文窗口，减少幻觉。

#### 1.2 环境搭建 (Hands-on)

**任务**: 启动 Neo4j 数据库并配置 Python 环境。

1. **启动 Neo4j (Docker)**:

   ```bash
   docker run \
       --name neo4j-graphrag \
       -p 7474:7474 -p 7687:7687 \
       -e NEO4J_AUTH=neo4j/password \
       -e NEO4J_PLUGINS='["apoc"]' \
       neo4j:5.18.0
   ```

   - 访问 `http://localhost:7474` 验证启动成功。

2. **安装 Python 依赖**:

   ```bash
   pip install langchain langchain-community langchain-openai neo4j tiktoken
   ```

---

### Module 2: Cypher 查询语言入门

#### 2.1 核心概念

- **Node (节点)**: 实体，如 `(p:Person {name: "Elon Musk"})`
- **Relationship (关系)**: 连接，如 `[:FOUNDED {year: 2002}]`
- **Label (标签)**: 类别，如 `:Person`, `:Company`

#### 2.2 基础查询练习

**场景**: 电影数据库。

- **创建数据**:

  ```cypher
  CREATE (m:Movie {title: "The Matrix", released: 1999})
  CREATE (p:Person {name: "Keanu Reeves", born: 1964})
  CREATE (p)-[:ACTED_IN {roles: ["Neo"]}]->(m)
  ```

- **查询数据**:

  ```cypher
  // 查找 Keanu Reeves 演过的电影
  MATCH (p:Person {name: "Keanu Reeves"})-[:ACTED_IN]->(m:Movie)
  RETURN m.title, m.released
  ```

#### 课后练习

- 编写 Cypher 语句，创建一个包含 "User" 和 "Product" 的简单购买关系图。

---

### Module 3: 非结构化文本的图谱构建 (Text2Graph)

这是本课程的核心。我们将使用 LLM 自动从文本中提取实体和关系。

#### 3.1 流程图解

```mermaid
graph LR
    A[非结构化文档] --> B[文本切片 (Chunks)]
    B --> C{LLM (Graph Transformer)}
    C --> D[提取节点 & 关系]
    D --> E[(Neo4j 数据库)]
```

#### 3.2 实战代码：从文本到图谱

**步骤**:

1. **定义 Schema**: 告诉 LLM 我们关心哪些类型的节点和关系。
2. **执行提取**: 使用 LangChain 的 `LLMGraphTransformer`。

```python
import os
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

# 配置
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

graph = Neo4jGraph()

# 示例文本
text = """
Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
Tim Cook is the CEO of Apple.
"""
documents = [Document(page_content=text)]

# 初始化转换器 (GPT-4 效果最佳)
llm = ChatOpenAI(temperature=0, model="gpt-4")
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Company", "Person", "City"],
    allowed_relationships=["HEADQUARTERED_IN", "CEO_OF", "LOCATED_IN"]
)

# 转换并存储
graph_documents = llm_transformer.convert_to_graph_documents(documents)
graph.add_graph_documents(graph_documents)
print(f"提取了 {len(graph_documents[0].nodes)} 个节点和 {len(graph_documents[0].relationships)} 条关系。")
```

---

### Module 4: 混合检索与问答系统

#### 4.1 混合检索架构

单纯的图检索可能漏掉非结构化的细节，单纯的向量检索缺乏结构。混合检索结合两者。

- **向量检索 (Vector Search)**: 寻找语义相似的文档块。
- **图检索 (Graph Traversal)**: 寻找实体及其邻居。

#### 4.2 实现 QA Chain

```python
from langchain.chains import GraphCypherQAChain

# 自动将自然语言转为 Cypher
chain = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    verbose=True,
    allow_dangerous_requests=True
)

# 提问
response = chain.invoke({"query": "Who is the CEO of the company located in Cupertino?"})
print(response['result'])
```

---

### Module 5: 实战项目 - SEC 财报分析

**背景**: 投资者需要分析大量的 SEC 10-K 表格（年度报告），了解公司的风险、高管和投资关系。

#### 5.1 数据准备

- 下载示例 SEC 10-K JSON 数据（如 NetApp Inc. 的财报）。
- 提取其中的 `Item 1` (Business) 和 `Item 1A` (Risk Factors) 部分。

#### 5.2 构建策略

1. **Chunking**: 将长文本切分为 500-1000 tokens 的块。
2. **向量化**: 为每个 Chunk 创建向量索引，存入 Neo4j。
3. **图谱化**: 提取关键实体（如 `Manager`, `Company`, `Product`）。
4. **连接**: 将 Chunk 节点链接到提取出的实体节点（`MENTIONS` 关系）。

#### 5.3 复杂查询示例

**问题**: "NetApp 提到了哪些关于供应链的风险？"

- **检索路径**:
  1. 向量检索: 搜索 "supply chain risk" 相关的 Chunk 节点。
  2. 图扩展: 查找这些 Chunk 关联的 `Risk` 实体。
  3. 生成: LLM 综合这些信息生成回答。

---

## 常见问题 (FAQ)

1. **Q: 必须使用 GPT-4 吗？**

   - A: 推荐使用 GPT-4 进行图谱提取（Text2Graph），因为它对指令的遵循能力更强，能生成更规范的 Schema。问答阶段可以使用 GPT-3.5 或其他模型。

2. **Q: 如何处理实体消歧（Entity Resolution）？**

   - A: 比如 "Apple" 和 "Apple Inc."。可以在提取后使用简单的规则合并，或者在提取前给 LLM 提供明确的命名规范。

3. **Q: 数据量很大怎么办？**
   - A: 对于大规模数据，建议并行处理 Text2Graph 步骤，并使用 Neo4j 的批量导入工具。

---

## 拓展思考：GraphRAG vs. Synergized LLMs + KGs

在实际的企业级应用（如银行反欺诈）中，我们常听到 "Synergized LLMs + KGs"（大模型与知识图谱协同）这一概念。虽然它与 GraphRAG 都涉及 KG 和 LLM 的结合，但在**设计理念**和**KG 的角色**上有显著区别。

| **维度**       | **GraphRAG (本课程模式)**                                                            | **Synergized LLMs + KGs (如银行反欺诈)**                                                                    |
| :------------- | :----------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| **核心目标**   | **文档问答与检索增强**。解决 LLM 在长文本、跨文档信息上的遗忘和幻觉问题。            | **复杂推理与决策支持**。解决高风险场景下的精确性、可解释性和实时性问题。                                    |
| **KG 的来源**  | 主要来自**非结构化文本** (Text2Graph)。KG 是为了更好地索引文本而临时或半自动构建的。 | 主要来自**结构化业务数据** (Core Banking System)。KG 是核心资产，包含高质量、经校验的账户、交易、设备信息。 |
| **KG 的质量**  | **概率性**。依赖 LLM 提取，可能存在噪声、错误或不完整的关系。                        | **确定性 (High Quality)**。作为 "Ground Truth" (事实基准)，数据的准确性至关重要。                           |
| **LLM 的角色** | **构建者 & 接口**。LLM 既用于构建图谱，也用于生成最终的自然语言回答。                | **推理者 & 解释者**。LLM 利用高质量 KG 进行逻辑推理 (Reasoning) 和结果解释，而非构建基础事实。              |
| **典型场景**   | 研报分析、法律文档审查、百科问答。                                                   | 资金链路追踪、团伙挖掘、反洗钱 (AML)。                                                                      |

**总结**:

- 本课程教授的 **GraphRAG** 更多关注如何"从无到有"利用 LLM 将文本转化为图谱，以提升检索效果。
- **Synergized LLMs + KGs** 则更多关注如何"强强联合"，利用已有的高质量图谱为 LLM 提供可靠的推理边界和事实依据。

---

## 参考文献与资源

1. **DeepLearning.AI Course**: [Knowledge Graphs for RAG](https://learn.deeplearning.ai/courses/knowledge-graphs-rag/)
2. **Neo4j GenAI Stack**: [GitHub Repository](https://github.com/neo4j/neo4j-genai-python)
3. **LangChain Graph Documentation**: [Docs](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/)
