# AI 工厂的物理极限

> 互连、内存、存储、供电四条线在 2026 年同时撞墙——这篇文章把它们放在同一张账本上算。

## 这篇讲什么

算力之外，AI 工厂受四组物理约束：铜缆传不远、单柜功率压不住、HBM 装不下、冷存储供不上。四条线各自都有成熟的技术叙事（CPO、CXL、分层存储、新介质），但真正决定集群形态的是它们的耦合：CPO 省下的电被液冷吃掉一部分，CXL 扩容要用延迟换，HDD 缺货把冷数据挤到 QLC 再推高推理成本。

- **[AI 工厂的物理极限：CPO 光互连、CXL 解耦内存与冷存储](01_cpo_cxl_cold_storage.md)**：完整长文，含产品规格表、延迟账本、规模阈值估算、总评与口径折扣、来源索引。

## 速查表

| 极限 | 2026 年的位置 | 主要对策 | 代价 |
| --- | --- | --- | --- |
| 铜缆距离 | 200G/lane 下无源铜约 1.5 m | CPO / OCI-MSA / OCP SROI | 激光器运维、液冷、故障域上移 |
| 功率密度 | NVL72 级 >100 kW；Kyber 约 600 kW | 800 V DC、全液冷 | 机房改造成本、PUE 口径重算 |
| 内存容量 | HBM4 单栈 36 GB，8 栈约 288 GB | CXL 扩容/池化/共享 | 延迟高 2–5 倍、软件栈不成熟 |
| 存储供给 | HDD 2026 售罄、特定容量段 SSD 约 20× HDD $/TB | 磁带、陶瓷介质、长协锁产能 | RTO 拉长、归档层成本上升 |

## 五个结论

1. CPO 的经济性看 **scale-out 光口总量**，万卡以上才明显；scale-up 跨机柜是 OCI-MSA 的战场，和 CPO 交换机不是同一件事。
2. 「3.5× 能效」是端口级口径，不是整机 PUE；液冷与激光器维护要单独入账。
3. CXL 的 TCO 甜点在**机架内（sub-rack）**，它做容量层，不做延迟敏感层，也不会取代数据中心网络。
4. 池化 ≠ 共享：2026 年能落地的主要是 MLD 静态切分，多主机硬件一致性共享仍处早期。
5. 2026–2027 年最短板是**电力 + 存储供给**，不是 GPU；规划顺序应先算这两笔账。

## 阅读路径

- 只关心互连 → 读第二节（CPO）与 [NVLink 技术入门](../nvlink/nvlink_intro.md)
- 只关心内存 → 读第三节，再进 [CXL 互联协议全景解读](../pcie/08_cxl_interconnect_overview.md)
- 只关心存储 → 读第四节，配合 [HBM 显存技术演进](../performance/01_hbm_evolution.md) 理解分层起点
- 做选型/采购 → 读第二节 2.7（规模阈值）与第五节（总评与口径折扣）

## 关联文章

- [CXL 互联协议全景解读](../pcie/08_cxl_interconnect_overview.md) — 三代协议、Flit 布局、池化机制与延迟实测
- [CXL 命令行工具链全景](../pcie/09_cxl_cli_tools.md) — `cxl` / `daxctl` / `ndctl` 实操与内核成熟度边界
- [HBM 显存技术演进](../performance/01_hbm_evolution.md) — HBM2 → HBM3e 的代际与实测
- [UCIe 4.0 芯粒互连](../ucie/README.md) — 封装内互连的能耗-距离分层
- [GB300 NVL72 机架级架构](../superchips/nvidia_gb300.md) — 机架即系统的系统级形态
- [AI 基础设施延迟金字塔](../performance/ai_latency_pyramid.md) — 各级延迟基准

## 口径提醒

文中数字分三类：**厂商口径**（NVIDIA/Broadcom/Samsung/SK hynix/Cerabyte 的公开材料）、**第三方口径**（TrendForce、heise、CXL 联盟）、**本文估算**（端口数与省电的线性外推）。引用时请保留标签，as-of 日期为 2026-11-26。
