// 用途: 以最小 Schema 导入 Demo 数据，支撑图谱检索与风险判决。
// 数据来源: demo/etl/generate_data.py 合成数据（仅用于流程验证）。

// 1) 约束与索引（性能关键）
CREATE CONSTRAINT customer_id_unique IF NOT EXISTS FOR (c:Customer) REQUIRE c.cust_id IS UNIQUE;
CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE;
CREATE INDEX device_fp_idx IF NOT EXISTS FOR (d:Device) ON (d.device_fp);
CREATE INDEX ip_addr_idx IF NOT EXISTS FOR (ip:IPAddress) ON (ip.ip_addr);

// 2) 客户（Customer）
LOAD CSV WITH HEADERS FROM 'file:///customers.csv' AS row
MERGE (c:Customer {cust_id: row.cust_id})
SET c.name = row.name;

// 3) 账户（Account）及归属关系（OWNS）
LOAD CSV WITH HEADERS FROM 'file:///accounts.csv' AS row
MERGE (a:Account {account_id: row.account_id})
SET a.open_date = date(row.open_date),
    a.status = row.status,
    a.balance = toInteger(row.balance)
MERGE (c:Customer {cust_id: row.cust_id})
MERGE (c)-[:OWNS]->(a);

// 4) 设备与 IP 登录（USED_DEVICE / LOGIN_AT）
// 说明: Demo 用设备指纹与 IP 聚合近似团伙环境证据，不引入更敏感的 GPS 等字段。
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

// 5) 交易边（TRANSFER）
// 说明: Demo 以转账边承载交易要素（金额/时间/备注），便于在图中追踪资金链路。
LOAD CSV WITH HEADERS FROM 'file:///transactions.csv' AS row
MERGE (src:Account {account_id: row.src_account_id})
MERGE (dst:Account {account_id: row.dst_account_id})
MERGE (src)-[t:TRANSFER {txn_id: row.txn_id}]->(dst)
SET t.amount = toInteger(row.amount),
    t.timestamp = datetime(row.timestamp),
    t.currency = row.currency,
    t.channel = row.channel,
    t.memo = row.memo;
