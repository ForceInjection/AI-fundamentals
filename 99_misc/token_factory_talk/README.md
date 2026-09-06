# Token Factory 演讲项目

对外演讲《Token Factory: AI 推理的成本革命》（南京大学校友会）的完整素材库。

## 内容结构

| 文件                                                                                           | 说明                                                                                                                   |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| [outline.md](outline.md)                                                                       | **可排练版演讲提纲**：四部分结构 + 时间分配 + 逐小节要点/数据/转场句/观众收获 + Q&A 预判 + 数据核查清单                |
| [talk-illustrated.md](talk-illustrated.md)                                                     | **对外图文讲解**：PPT 24 页逐页配图（img/cover+01–23）讲解——"白菜化/Token 单位化 → 产品制造商 → 两件事 → 国产算力差距"全文口径 |
| [cluster-capacity-measurement.md](cluster-capacity-measurement.md)                             | **对外发布文档**：《Token 工厂的产能怎么算？》——Goodput 框架 + AIPerf 落地（概念公开、内部数据脱敏）                  |
| [references/01-wechat-token-factory-model.md](references/01-wechat-token-factory-model.md)     | 两年砍柴《Token工厂财务运营测算工具》微信原文（8 张 Sheet 模型全拆解，含下载方法）                                     |
| [references/02-ai-factory-ecosystem.md](references/02-ai-factory-ecosystem.md)                 | AI Factory 定义/黄仁勋叙事/分类框架/全球格局                                                                           |
| [references/03-business-model-and-finance.md](references/03-business-model-and-finance.md)     | 商业模式变革、财务测算结果、**电力占比口径矛盾辨析**、仓库成本锚点                                                     |
| [references/04-cost-technology.md](references/04-cost-technology.md)                           | **物理三维框架（算/存/传 + 能量统一）**、模型架构稀疏化（MoE/MoD/MLA/DSA）、KV Cache、硬件代际、价格历史               |
| [references/05-jevons-demand-future.md](references/05-jevons-demand-future.md)                 | 杰文斯悖论、需求爆发数据、政策落地、未来方向                                                                           |
| [references/06-listed-companies.md](references/06-listed-companies.md)                         | **上市公司财报证据**：硅基流动招股书（毛利率 −24%）、CoreWeave 季报（66% 毛利仍净亏）、一级市场同行数据                |
| [references/07-tokenfactory-whitepaper-2026.md](references/07-tokenfactory-whitepaper-2026.md) | **TokenFactory 白皮书 2026**（九章云极 × InfoQ）结构化全文：八章核心内容、成本结构、产业链三层、九章云极实践、趋势风险 |
| [references/08-hf-open-models-summer-2026.md](references/08-hf-open-models-summer-2026.md) | HF《State of Open Models: Summer 2026 Observations》报告摘要：模型能力与供给丰富度信号（万亿参数开源竞赛、开源追平闭源、296 万个模型）与 Agent 流量 |
| [references/09-model-price-performance.md](references/09-model-price-performance.md) | **模型性价比跃迁**：GLM-5.3-Flash 与 Qwen3.8-Flash 同日双发（2026-08-26）——性能/价格/架构数据、演讲用法、口径注 |
| [references/10-cluster-capacity-assessment.md](references/10-cluster-capacity-assessment.md) | **集群产能评估框架**（内部仓库引入）：SLO Goodput、单卡产能/集群日上限/日产值四数、品质承诺 P99 体系——"利用率是第一杠杆"的运营量化版 |
| [references/11-parameter-vs-throughput.md](references/11-parameter-vs-throughput.md) | **模型选型对产能的影响**：V4 Pro/Flash 实测——单卡产能 ~2×、同卡数端到端 3.8×、通信税（拓扑 NVLink/IB 决定税率） |
| [references/12-silicon-data-token-index.md](references/12-silicon-data-token-index.md) | **Silicon Data LLM Token 支出指数**：全市场平均支付价跌破 $1/百万（历史新低、较峰值腰斩）——市场级价格锚点 |

## 注意

所有市场数据均为 2026 年 8 月搜索所得，引用前请复核。
