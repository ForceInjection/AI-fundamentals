"""将本地 CSV 导入 Neo4j（Demo 版）。

导入方式：
- 通过 Bolt Driver 执行 `cypher_import.cypher`；
- 脚本内按 `;` 分割语句逐条执行，避免“Schema 修改 + 写入”同事务导致的限制。

前置条件：
- Neo4j 容器已启动，且 `demo/data` 已映射至 Neo4j `import` 目录。
"""

import os
from neo4j import GraphDatabase

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ETL_DIR = os.path.join(BASE_DIR, "etl")


def run_import(uri="bolt://localhost:7687", user="neo4j", password="password"):
    """执行导入脚本。"""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    cypher_path = os.path.join(ETL_DIR, "cypher_import.cypher")
    with open(cypher_path, "r", encoding="utf-8") as f:
        cypher = f.read()
    with driver.session() as session:
        for stmt in [s.strip() for s in cypher.split(";") if s.strip()]:
            session.run(stmt)
    driver.close()
    print("Import completed")


if __name__ == "__main__":
    run_import()
