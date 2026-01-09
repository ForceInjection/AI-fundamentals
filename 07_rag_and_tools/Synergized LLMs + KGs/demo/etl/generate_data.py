"""Demo 数据生成器。

说明：
- 该脚本用于生成可复现的“最小可运行”样本数据集，以支撑图谱导入与判决演示；
- 数据为合成数据，仅用于工程流程验证，不代表真实业务分布；
- 生成的数据将写入 `demo/data/`，并通过 docker volume 映射到 Neo4j `import` 目录。
"""

import csv
import os
import random
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def ensure_dirs():
    """创建数据输出目录。"""
    os.makedirs(DATA_DIR, exist_ok=True)


def gen_customers_accounts(n_customers=30):
    """生成客户与账户样本。"""
    customers = []
    accounts = []
    for i in range(1, n_customers + 1):
        cust_id = f"C{i:03d}"
        customers.append({"cust_id": cust_id, "name": f"用户{i:03d}"})
        # 每个客户 1-2 个账户（简化建模：一个客户拥有多账户）
        for j in range(random.choice([1, 2])):
            acc_id = f"A{i:03d}{j}"
            accounts.append(
                {
                    "account_id": acc_id,
                    "cust_id": cust_id,
                    "open_date": (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 300))).date().isoformat(),
                    "status": "active",
                    "balance": random.randint(1000, 50000),
                }
            )
    return customers, accounts


def gen_device_logs(customers, n_logs_per_cust=5):
    """生成客户登录设备与 IP 记录。

特征注入：
- `is_emulator` 以较小概率为 true，用于模拟 GOIP/模拟器集群；
- IP 使用私网段示意，Demo 中不做真实地理解析。
    """
    device_logs = []
    now = datetime.now()
    for c in customers:
        for _ in range(n_logs_per_cust):
            ts = now - timedelta(days=random.randint(0, 7), hours=random.randint(0, 23), minutes=random.randint(0, 59))
            # 模拟器设备指纹与普通设备指纹
            is_emulator = random.random() < 0.2
            fingerprint = (
                "Android 10 / Pixel 4" if is_emulator else random.choice(["iOS 16 / iPhone 12", "Android 13 / Mi 12"])
            )
            ip = random.choice(
                ["10.0.0.%d" % random.randint(1, 200), "172.16.0.%d" % random.randint(1, 200), "100.64.0.%d" % random.randint(1, 200)]
            )
            device_logs.append(
                {
                    "cust_id": c["cust_id"],
                    "fingerprint": fingerprint,
                    "login_time": ts.isoformat(),
                    "ip_address": ip,
                    "is_emulator": "true" if is_emulator else "false",
                }
            )
    return device_logs


def gen_transactions(accounts, n_txn=150):
    """生成交易流水。

包含四类模式：
- normal：随机转账，模拟正常交易噪声；
- gathering：分散转入到“二道卡/归集卡”；
- scattering：由“二道卡”向外分散转出（分赃/发薪等）；
- long_chain：构造 5 跳长链，模拟逐层清洗。
    """
    transactions = []
    # 归集卡集合与最终主账户（仅用于合成模式）
    mule_accounts = random.sample(accounts, k=max(3, len(accounts) // 10))
    master_account = random.choice(accounts)["account_id"]
    start = datetime.now() - timedelta(days=2)
    for i in range(1, n_txn + 1):
        mode = random.choice(["normal", "gathering", "scattering", "long_chain"])
        ts = start + timedelta(minutes=i * random.choice([1, 2, 3]))
        amount = random.choice([500, 1000, 2000, 5000, 10000, 20000])
        if mode == "gathering":
            src = random.choice(accounts)["account_id"]
            dst = random.choice(mule_accounts)["account_id"]
        elif mode == "scattering":
            src = random.choice(mule_accounts)["account_id"]
            dst = random.choice(accounts)["account_id"]
        elif mode == "long_chain":
            chain = random.sample(accounts, k=5)
            for a, b in zip(chain[:-1], chain[1:]):
                transactions.append(
                    {
                        "txn_id": f"T{i:04d}_{a['account_id']}_{b['account_id']}",
                        "src_account_id": a["account_id"],
                        "dst_account_id": b["account_id"],
                        "amount": amount,
                        "timestamp": (ts + timedelta(seconds=random.randint(0, 120))).isoformat(),
                        "currency": "CNY",
                        "channel": random.choice(["APP", "WEB", "ATM"]),
                        "memo": random.choice(["转账", "投资", "保证金", "解冻", "工资", "生活费"]),
                    }
                )
            continue
        else:
            src = random.choice(accounts)["account_id"]
            dst = random.choice(accounts)["account_id"]
            if src == dst:
                continue
        transactions.append(
            {
                "txn_id": f"T{i:04d}",
                "src_account_id": src,
                "dst_account_id": dst,
                "amount": amount,
                "timestamp": ts.isoformat(),
                "currency": "CNY",
                "channel": random.choice(["APP", "WEB", "ATM"]),
                "memo": random.choice(["转账", "投资", "保证金", "解冻", "工资", "生活费"]),
            }
        )
    # 归集到 master
    for i in range(30):
        ts = start + timedelta(minutes=5 * i + 1)
        src = random.choice(mule_accounts)["account_id"]
        transactions.append(
            {
                "txn_id": f"TM{i:03d}",
                "src_account_id": src,
                "dst_account_id": master_account,
                "amount": random.choice([5000, 10000, 20000]),
                "timestamp": ts.isoformat(),
                "currency": "CNY",
                "channel": "APP",
                "memo": random.choice(["投资", "保证金", "解冻"]),
            }
        )
    return transactions


def write_csv(path, rows, headers):
    """将 dict 列表写为 CSV。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    """生成数据并落盘。"""
    ensure_dirs()
    customers, accounts = gen_customers_accounts()
    device_logs = gen_device_logs(customers)
    transactions = gen_transactions(accounts)

    write_csv(
        os.path.join(DATA_DIR, "customers.csv"),
        customers,
        ["cust_id", "name"],
    )
    write_csv(
        os.path.join(DATA_DIR, "accounts.csv"),
        accounts,
        ["account_id", "cust_id", "open_date", "status", "balance"],
    )
    write_csv(
        os.path.join(DATA_DIR, "device_logs.csv"),
        device_logs,
        ["cust_id", "fingerprint", "login_time", "ip_address", "is_emulator"],
    )
    write_csv(
        os.path.join(DATA_DIR, "transactions.csv"),
        transactions,
        ["txn_id", "src_account_id", "dst_account_id", "amount", "timestamp", "currency", "channel", "memo"],
    )
    print(f"Data generated in {DATA_DIR}")


if __name__ == "__main__":
    main()
