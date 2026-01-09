# Neo4j Cypher 快速上手实战指南

本指南旨在配合 `Neo4j_Cypher_Tutorial.md`，带您一步步搭建本地 Neo4j 图数据库环境，并通过**银行反欺诈**场景的真实数据进行实战演练。

---

## 1. 准备工作

在开始之前，请确保您的电脑上已经安装并启动了 **Docker**。

我们将直接复用项目中现有的 Docker 配置，无需复杂的本地安装。

### 1.1 启动数据库服务

打开终端（Terminal），执行以下命令启动 Neo4j 容器：

```bash
# 1. 进入 demo 目录
cd "../Synergized LLMs + KGs/demo"

# 2. 启动 Neo4j 服务（后台运行）
docker-compose up -d
```

> **提示**：首次启动可能需要下载 Neo4j 镜像，请耐心等待。

### 1.2 登录 Neo4j 控制台 (WebUI)

Neo4j 提供了一个可视化的交互界面 **Neo4j Browser**，我们将在这里编写和执行 Cypher 查询。

1. **访问地址**：打开浏览器，访问 [http://localhost:7474](http://localhost:7474)
2. **登录凭据**：
   - **Connect URL**: 保持默认 (`neo4j://localhost:7687`)
   - **Username**: `neo4j`
   - **Password**: `password` (我们在 docker-compose.yml 中预设的密码)

登录成功后，您将看到一个带有输入框的控制台界面。

---

## 2. 数据装载 (Data Loading)

为了模拟真实的银行反欺诈场景，我们需要导入预置的 CSV 数据（包括客户、账户、设备、交易记录等）。

**操作步骤**：
请将下方完整的 Cypher 脚本复制并粘贴到 Neo4j Browser 顶部的输入框中，然后点击右侧的 ▶️ 按钮（或按 `Ctrl+Enter` / `Cmd+Enter`）执行。

```cypher
// ==========================================
// 银行反欺诈场景数据导入脚本
// ==========================================

// --- 步骤 1: 建立性能约束与索引 ---
// 确保客户ID和账户ID的唯一性，并为高频查询字段建立索引
CREATE CONSTRAINT customer_id_unique IF NOT EXISTS FOR (c:Customer) REQUIRE c.cust_id IS UNIQUE;
CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE;
CREATE INDEX device_fp_idx IF NOT EXISTS FOR (d:Device) ON (d.device_fp);
CREATE INDEX ip_addr_idx IF NOT EXISTS FOR (ip:IPAddress) ON (ip.ip_addr);

// --- 步骤 2: 导入客户档案 (Customer) ---
LOAD CSV WITH HEADERS FROM 'file:///customers.csv' AS row
MERGE (c:Customer {cust_id: row.cust_id})
SET c.name = row.name;

// --- 步骤 3: 导入账户及资产关系 (OWNS) ---
LOAD CSV WITH HEADERS FROM 'file:///accounts.csv' AS row
MERGE (a:Account {account_id: row.account_id})
SET a.open_date = date(row.open_date),
    a.status = row.status,
    a.balance = toInteger(row.balance)
MERGE (c:Customer {cust_id: row.cust_id})
MERGE (c)-[:OWNS]->(a);

// --- 步骤 4: 导入环境指纹 (Device & IP) ---
// 建立 Customer -> Device 和 Customer -> IP 的关联
LOAD CSV WITH HEADERS FROM 'file:///device_logs.csv' AS row
MERGE (c:Customer {cust_id: row.cust_id})
MERGE (d:Device {device_fp: row.fingerprint})
SET d.is_emulator = toBoolean(row.is_emulator)
MERGE (c)-[r:USED_DEVICE]->(d)
SET r.last_seen = datetime(row.login_time)
MERGE (ip:IPAddress {ip_addr: row.ip_address})
MERGE (c)-[l:LOGIN_AT]->(ip)
ON CREATE SET l.cnt = 1, l.last_seen = datetime(row.login_time)
ON MATCH SET l.cnt = l.cnt + 1, l.last_seen = datetime(row.login_time);

// --- 步骤 5: 导入资金交易链路 (TRANSFER) ---
LOAD CSV WITH HEADERS FROM 'file:///transactions.csv' AS row
MERGE (src:Account {account_id: row.src_account_id})
MERGE (dst:Account {account_id: row.dst_account_id})
MERGE (src)-[t:TRANSFER {txn_id: row.txn_id}]->(dst)
SET t.amount = toInteger(row.amount),
    t.timestamp = datetime(row.timestamp),
    t.currency = row.currency,
    t.channel = row.channel,
    t.memo = row.memo;
```

> **成功验证**：执行完毕后，系统应提示创建了数百个节点和关系。您可以运行 `MATCH (n) RETURN count(n)` 来查看节点总数。

---

## 3. 基础查询实战 (Basic Queries)

数据就绪后，让我们热热身，熟悉一下基础的查询操作。

### 练习 1: 全局数据概览

_目标：直观感受图数据的样子。_

```cypher
// 随机展示 50 个节点及其关系
MATCH (n) RETURN n LIMIT 50
```

_体验：在可视化界面中，试着拖拽节点，双击节点展开更多关系。_

### 练习 2: 客户资产视图

_目标：查找特定客户（如 "用户008"）及其名下所有账户。_

```cypher
MATCH (c:Customer {name: "用户008"})-[:OWNS]->(a:Account)
RETURN c, a
```

### 练习 3: 大额交易筛选

_目标：找出金额超过 5000 的所有转账记录，并按金额降序排列。_

```cypher
MATCH (src:Account)-[t:TRANSFER]->(dst:Account)
WHERE t.amount > 5000
RETURN src.account_id AS 转出方, dst.account_id AS 接收方, t.amount AS 金额, t.timestamp AS 时间
ORDER BY t.amount DESC
```

---

## 4. 进阶反欺诈分析 (Advanced Analysis)

这部分我们将利用图算法的特性，挖掘隐藏在数据中的风险模式。

### 场景 4: 资金链路追踪 (Path Finding)

_业务含义：追踪资金流向，发现最终受益人或洗钱路径。_

```cypher
// 随机选取一个账户作为起点，追踪 1 到 3 跳的资金流向
MATCH (start:Account) 
WITH start LIMIT 1
MATCH path = (start)-[:TRANSFER*1..3]->(end:Account)
RETURN path
```

### 场景 5: 团伙关联挖掘 (Community Detection)

_业务含义：发现使用同一设备登录的不同客户，这通常是欺诈团伙或代办黑产的特征。_

```cypher
MATCH (c1:Customer)-[:USED_DEVICE]->(d:Device)<-[:USED_DEVICE]-(c2:Customer)
WHERE c1.cust_id < c2.cust_id // 避免 A-B 和 B-A 重复显示
RETURN c1.name AS 客户A, c2.name AS 客户B, d.device_fp AS 共享设备指纹, d.is_emulator AS 是否模拟器
```

### 场景 6: 资金循环回流检测 (Cycle Detection)

_业务含义：检测资金是否转出一圈后又回到了原点，这是典型的刷量或洗钱模式。_

```cypher
// 查找长度在 2 到 5 跳之间的闭环路径
MATCH path = (a:Account)-[:TRANSFER*2..5]->(a)
RETURN path
LIMIT 5
```

---

## 5. 环境管理与清理

### 5.1 重置数据库

如果您想清空所有数据重新开始：

```cypher
// 警告：这将删除所有节点和关系！
MATCH (n) DETACH DELETE n
```

### 5.2 停止服务

练习结束后，建议停止 Docker 容器以释放资源：

```bash
docker-compose down
```

---

## 常见问题 (FAQ)

**Q1: 浏览器访问 http://localhost:7474 无法连接？**

- **检查**: 确保 Docker 容器正在运行 (`docker ps`)。
- **检查**: 确保没有其他服务占用 7474 或 7687 端口。

**Q2: 执行 LOAD CSV 时报错 "Couldn't load the external resource"？**

- **原因**: Neo4j 无法找到 CSV 文件。
- **解决**: 我们的 `docker-compose.yml` 已经配置了自动映射。请确保您是在 `Synergized LLMs + KGs/demo` 目录下启动的 Docker，且该目录下的 `data` 文件夹中包含 CSV 文件。

---

**祝您学习愉快！更多语法细节请参考同目录下的 `Neo4j_Cypher_Tutorial.md`。**
