# 推理成本分析

业界普遍估计，**推理成本占大模型服务总成本的 80% 以上**，而这笔支出又分成两条截然不同的曲线：一条是按 token 计费的 **API 调用成本**（OpenAI / Anthropic / DeepSeek / Qwen 等），另一条是按座位 / 用量计费的 **Coding Plan 订阅成本**（Cursor / GitHub Copilot / Kimi / 智谱龙虾等）。两者定价逻辑、计量口径、限流条款都不同——"每月花多少才不冤"需要分开建模。本目录提供一套"数据驱动 + 脚本自动化"的成本测算框架，避免被首月促销价、5 小时限流窗口、"禁止 API 调用"条款等营销陷阱误导。此外，单卡/集群层面的生产经济学，决定了同样的牌价下厂商能留多少毛利。

## 1. API 定价定量分析

面向平台方 / 大规模业务方，基于 **OpenRouter 聚合定价接口** 构建动态、可复现的成本测算链路，替代手工查阅各厂商官网的低效方式。

- **[大模型 API 定价策略定量分析框架](llm_api_pricing_analysis.md)**：主流模型分级矩阵（OpenAI / Anthropic / Google / DeepSeek / Qwen / Llama）、定价采集机制、缓存命中率与汇率敏感性分析（表内价格为 2026-04-15 OpenRouter 快照，历史口径）
- **[`fetch_pricing.py`](fetch_pricing.py)**：零依赖（仅 Python 3 标准库）的动态定价测算脚本，命令行参数仅有 `--exchange-rate`（汇率）与 `--no-cache`（跳过缓存）；场景 A/B/C 的 token 数与缓存命中率为脚本内置参数
- **[`openrouter_models_cache.json`](openrouter_models_cache.json)**：2026-04-15 手工归档的 OpenRouter 全量模型报价快照，用于离线复现与历史对比（脚本运行时的缓存写在系统临时目录，仓库内这份为手工拷贝）

> 典型用法：`python fetch_pricing.py --exchange-rate 7.25`（场景 A/B/C 为脚本内置：A 20K 入 / 1K 出 / 20% 命中，B 30K 入 / 1K 出 / 80% 命中，C 5K 入 / 5K 出 / 0% 命中）

## 2. Coding Plan 订阅深度对比

面向开发者 / 中小团队，深度扒清 2026 年 4 月国内外 11 款主流 AI 编程订阅套餐的真实成本，识别"首月七块九"、"5 小时限流"、"禁止 API 调用"等隐藏条款。

- **[Coding Plan 深度对比与避坑指南](coding_plan/coding_plan_report.md)**：2026 年国内外 11 款主流 AI 编程工具（Cursor / GitHub Copilot / Windsurf / Kimi / 腾讯云 Lite / 智谱龙虾等）的价格、限流、协议红线对比，附角色-场景选型速查表与三条避坑准则
- **[Coding Plan 数据看板](coding_plan/objective_pricing_comparison.md)**：归一化后的厂商定价源数据，以结构化表格 + 分组柱状图呈现费用阶梯与用量限制，作为纯数据附录
- **[`scripts/fetch_coding_plan_pricing.py`](coding_plan/scripts/fetch_coding_plan_pricing.py)**：定价页面自动化采集脚本（HTML 抓取 + 结构化抽取，支持快照保存）
- **[`scripts/generate_objective_comparison.py`](coding_plan/scripts/generate_objective_comparison.py)**：从归一化 JSON 生成对比报告与图表的渲染工具

配套数据目录：

- `coding_plan/data/pricing_raw/<日期>/`：按采集日归档的原始 HTML / 截图 / 文本抽取结果（国内外厂商各自独立子目录）
- `coding_plan/data/pricing_normalized.json`：清洗并归一化后的结构化定价数据（驱动报告与图表）
- `coding_plan/data/manual_overrides.json`：无法从页面稳定抽取时的人工覆盖值，保证报告可复现
- [`coding_plan/data/pricing_table.md`](coding_plan/data/pricing_table.md)：各厂商套餐定价结构化总表（月费 / 额度 / 限制条款 / 官方来源链接与采集日期，2026-04-17/18 采集口径）

## 3. 生产经济学案例

- **[DeepSeek 推理集群的单卡经济学](deepseek_inference_cluster_unit_economics.md)**：用官方披露的 155 万亿 tokens/天、96% 缓存命中、38 美元/卡时反推 2,370 张有效卡、单卡年收入 33 万美元与 93–96% 单卡运营利润率；附官方价格页、成本假设与价格/折旧敏感性。

## 4. 如何选择分析口径

面对一个成本问题，先分清用户画像再决定看哪份文档：

| 用户画像              | 核心诉求                                  | 主参考文档                          |
| --------------------- | ----------------------------------------- | ----------------------------------- |
| 平台方 / 大规模业务   | 百万 token 单价、月账单预算、模型切换 ROI | §1 API 定价定量分析                 |
| 平台方 / 基础设施投资 | 单卡收入、折旧、负载与牌价敏感性          | §3 生产经济学案例                   |
| 个人开发者 / 中小团队 | 订阅月费、限流条款、是否支持外接 Agent    | §2 Coding Plan 深度对比             |
| 政企 / 合规敏感       | 数据出境、私有化交付、审计可观测          | §2 Coding Plan 报告<br>§1.2 选型速查表 |
