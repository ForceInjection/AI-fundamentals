# Neo4j Cypher 查询语言权威指南

本指南基于 Neo4j 官方文档编写，结合本项目中的**银行反欺诈场景**，提供从入门到进阶的 Cypher 语言教程。Cypher 是 Neo4j 的声明式图查询语言，设计理念类似 SQL，但专门针对图数据的模式匹配进行了优化。

---

## 1. 核心概念 (Core Concepts)

在开始编写查询之前，回顾一下图数据库的四个基本构建块：

1. **节点 (Nodes)**: 图中的实体。例如：`(:Customer)`, `(:Account)`。
2. **关系 (Relationships)**: 连接两个节点，必须有方向和类型。例如：`-[:OWNS]->`, `-[:TRANSFER]->`。
3. **属性 (Properties)**: 存储在节点或关系上的键值对数据。例如：`{name: "张三", amount: 1000}`。
4. **标签 (Labels)**: 用于对节点进行分组和分类。例如：`:VIP`, `:Suspect`。

---

## 2. 基础查询 (Reading Data)

### 2.1 模式匹配 (MATCH)

`MATCH` 是 Cypher 中最核心的子句，用于描述你想要在图中找到的结构（模式）。

```cypher
// 1. 查询所有 Customer 节点
MATCH (c:Customer)
RETURN c.name, c.cust_id

// 2. 查询特定属性的节点
MATCH (a:Account {account_id: "A001"})
RETURN a

// 3. 查询特定关系（客户拥有的账户）
// 模式：(客户)-[拥有]->(账户)
MATCH (c:Customer)-[:OWNS]->(a:Account)
RETURN c.name, a.account_id
```

### 2.2 过滤条件 (WHERE)

虽然可以在 `MATCH` 中直接指定属性，但 `WHERE` 子句提供了更强大的过滤能力（布尔逻辑、范围查询、字符串匹配等）。

```cypher
MATCH (t:Transfer)
WHERE t.amount > 10000
  AND t.currency = "CNY"
  AND (t.status = "Failed" OR t.channel = "Mobile")
RETURN t
```

**常用操作符：**

- **比较**: `=`, `<>`, `<`, `>`, `<=`, `>=`
- **逻辑**: `AND`, `OR`, `NOT`, `XOR`
- **字符串**: `STARTS WITH`, `ENDS WITH`, `CONTAINS`
- **空值检查**: `IS NULL`, `IS NOT NULL`
- **存在性检查**: `EXISTS { MATCH ... }` (高级用法)

```cypher
// 查找名字中包含 "王" 且有手机号的客户
MATCH (c:Customer)
WHERE c.name CONTAINS "王" AND c.phone IS NOT NULL
RETURN c
```

### 2.3 可选匹配 (OPTIONAL MATCH)

类似于 SQL 中的 `LEFT JOIN`。如果模式匹配失败，`OPTIONAL MATCH` 会返回 `null` 而不是丢弃当前行。

```cypher
// 查询所有客户，以及他们可能拥有的 "高风险" 标签（如果有的话）
MATCH (c:Customer)
OPTIONAL MATCH (c)-[:HAS_TAG]->(t:RiskTag)
RETURN c.name, t.tag_name
// 结果：如果客户没有风险标签，t.tag_name 将为 null
```

### 2.4 排序与分页 (ORDER BY, SKIP, LIMIT)

```cypher
MATCH (a:Account)
RETURN a.account_id, a.balance
ORDER BY a.balance DESC  // 按余额降序
SKIP 10                  // 跳过前 10 条
LIMIT 5                  // 取接下来的 5 条
```

### 2.5 别名与去重 (AS, DISTINCT)

```cypher
MATCH (c:Customer)-[:OWNS]->(a:Account)
RETURN DISTINCT c.name AS CustomerName, count(a) AS AccountCount
```

---

## 3. 数据写入与更新 (Writing Data)

### 3.1 创建数据 (CREATE)

`CREATE` 用于无条件创建新的节点和关系。

```cypher
// 创建单节点
CREATE (:Customer {name: "李四", cust_id: "C002"})

// 创建带关系的路径
CREATE (c:Customer {name: "王五"})-[:OWNS]->(a:Account {id: "A100", balance: 0})
```

### 3.2 智能合并 (MERGE)

`MERGE` 是 "Match or Create" 的操作。它首先尝试在图中匹配模式，如果存在则重用，不存在则创建。**这是保证数据幂等性的关键。**

```cypher
// 确保账户存在
MERGE (a:Account {account_id: "A999"})
ON CREATE SET a.created_at = datetime(), a.balance = 0
ON MATCH SET a.last_accessed = datetime()
```

**实战技巧**：在导入 CSV 数据时，务必使用 `MERGE` 而非 `CREATE`，以避免重复数据。

### 3.3 更新属性与标签 (SET)

```cypher
MATCH (c:Customer {name: "张三"})
// 更新属性
SET c.age = 35, c.risk_level = "Medium"
// 添加标签
SET c:VIP:HighNetWorth
```

### 3.4 删除数据 (DELETE / REMOVE)

- `DELETE`: 删除节点或关系。
- `REMOVE`: 删除属性或标签。
- `DETACH DELETE`: 删除节点及其所有关联关系（强制删除）。

```cypher
// 删除关系
MATCH (c:Customer)-[r:OWNS]->(a:Account)
WHERE c.name = "已注销用户"
DELETE r

// 强制删除节点及其关系
MATCH (n:RiskTag {name: "过期标签"})
DETACH DELETE n

// 移除标签和属性
MATCH (c:Customer {cust_id: "C001"})
REMOVE c:VIP, c.temp_token
```

---

## 4. 进阶模式与算法 (Advanced Patterns)

### 4.1 可变长路径 (Variable Length Paths)

在反欺诈中，经常需要查找多跳关系（例如：资金经过多次转账最终流向哪里）。

```cypher
// 查找从账户 A 到账户 B 的转账路径，长度为 1 到 5 跳
MATCH p = (start:Account {id: "A001"})-[:TRANSFER*1..5]->(end:Account {id: "A002"})
RETURN p
```

- `*`: 任意长度
- `*2`: 固定 2 跳
- `*2..5`: 2 到 5 跳

### 4.2 最短路径 (Shortest Path)

查找两个节点之间的最短连接。

```cypher
MATCH (start:Customer {name: "张三"}), (end:Customer {name: "李四"})
// 查找两人之间最短的交互路径（无论什么关系）
MATCH p = shortestPath((start)-[*]-(end))
RETURN p
```

### 4.3 列表与聚合 (List & Aggregation)

Cypher 提供了强大的列表处理能力。

- `collect()`: 将多行结果聚合为一个列表。
- `UNWIND`: 将列表展开为多行（`collect` 的逆操作）。

```cypher
// 聚合：列出每个客户拥有的所有账户ID列表
MATCH (c:Customer)-[:OWNS]->(a:Account)
RETURN c.name, collect(a.account_id) AS account_ids, count(a) AS total_accounts

// 展开：处理输入的列表数据
WITH ["A001", "A002", "A003"] AS target_ids
UNWIND target_ids AS aid
MATCH (a:Account {account_id: aid})
RETURN a
```

### 4.4 WITH 子句 (Piping)

`WITH` 用于连接查询的多个部分，它可以对中间结果进行过滤、聚合或排序，然后传递给下一个查询部分。

```cypher
// 查找转账次数超过 10 次的账户，并标记为高频账户
MATCH (a:Account)-[:TRANSFER]->()
WITH a, count(*) AS txn_count
WHERE txn_count > 10
SET a:HighFrequency
RETURN a.account_id, txn_count
```

---

## 5. 常用函数 (Functions)

### 5.1 字符串函数

- `toUpper(s)`, `toLower(s)`
- `substring(s, start, length)`
- `replace(s, search, replace)`
- `split(s, delimiter)` -> List

### 5.2 标量与转换函数

- `coalesce(val1, val2, ...)`: 返回第一个非空值。
- `head(list)`, `last(list)`, `size(list)`
- `toInteger()`, `toFloat()`, `toString()`, `toBoolean()`
- `type(r)`: 返回关系类型。
- `labels(n)`: 返回节点标签列表。

### 5.3 时间日期函数

- `date()`: 当前日期。
- `datetime()`: 当前日期时间。
- `date("2023-01-01")`: 字符串转日期。
- `duration.between(date1, date2)`: 计算时间差。

```cypher
// 查询过去 30 天内的转账
MATCH (t:Transfer)
WHERE t.timestamp > datetime() - duration({days: 30})
RETURN t
```

---

## 6. 性能优化 (Performance)

### 6.1 索引与约束 (Index & Constraint)

在生产环境中，**必须**为常用的查询字段创建索引。

```cypher
// 创建唯一性约束（同时也是索引）
CREATE CONSTRAINT cust_id_unique IF NOT EXISTS FOR (c:Customer) REQUIRE c.cust_id IS UNIQUE;

// 创建普通索引
CREATE INDEX account_balance_idx IF NOT EXISTS FOR (a:Account) ON (a.balance);
```

### 6.2 PROFILE 与 EXPLAIN

在查询前加上 `PROFILE` 关键字，可以看到查询的执行计划和实际 DB 命中次数，用于调优。

```cypher
PROFILE MATCH (c:Customer {name: "张三"}) RETURN c
```

---

## 7. 数据导入 (Loading Data)

`LOAD CSV` 是 Neo4j 处理 CSV 文件的标准方式。

```cypher
// 示例：从 CSV 导入交易数据
LOAD CSV WITH HEADERS FROM 'file:///transactions.csv' AS row
// 1. 查找或创建关联节点
MERGE (src:Account {id: row.from_acc})
MERGE (dst:Account {id: row.to_acc})
// 2. 创建关系
CREATE (src)-[:TRANSFER {
    amount: toInteger(row.amount),
    date: datetime(row.date)
}]->(dst)
```

---

## 8. 实战案例：反欺诈查询

结合项目中的数据结构，以下是一些高级查询示例：

### 案例 1: 资金回流检测 (Cycle Detection)

检测资金是否转出一圈后又回到了原账户（典型的洗钱模式）。

```cypher
MATCH path = (a:Account)-[:TRANSFER*2..5]->(a)
WHERE a.account_id = "A001"
RETURN path
```

### 案例 2: 关联团伙挖掘 (Community Detection)

查找使用同一设备登录的不同客户。

```cypher
MATCH (c1:Customer)-[:USED_DEVICE]->(d:Device)<-[:USED_DEVICE]-(c2:Customer)
WHERE c1.cust_id < c2.cust_id  // 避免重复对 (A-B, B-A)
RETURN c1.name, c2.name, d.device_fp AS SharedDevice
```

### 案例 3: 多跳资金追踪

查找从"黑名单客户"转出的资金，在 3 跳之内进入了哪些账户。

```cypher
MATCH (suspect:Customer {risk_level: "Blacklist"})-[:OWNS]->(sa:Account)
MATCH p = (sa)-[:TRANSFER*1..3]->(target:Account)
RETURN target.account_id, reduce(s = 0, r in relationships(p) | s + r.amount) AS total_amount
ORDER BY total_amount DESC
```

---

## 参考资料

- [Neo4j Cypher Manual (Official)](https://neo4j.com/docs/cypher-manual/current/)
- [Neo4j Cypher Cheat Sheet](https://neo4j.com/docs/cypher-cheat-sheet/current/)
