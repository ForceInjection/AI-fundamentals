# 推理成本分析

在大型 AI 服务商的总运营成本中，**推理成本通常占到 80–90%**，而这笔支出又分成两条截然不同的曲线：一条是按 token 计费的 **API 调用成本**（OpenAI / Anthropic / DeepSeek / Qwen 等），另一条是按座位 / 用量计费的 **Coding Plan 订阅成本**（Cursor / GitHub Copilot / Kimi / 智谱龙虾等）。两者定价逻辑、计量口径、限流条款都不同——"每月花多少才不冤"需要分开建模。本目录提供一套"数据驱动 + 脚本自动化"的成本测算框架，避免被首月促销价、5 小时限流窗口、"禁止 API 调用"条款等营销陷阱误导。

## 1. API 定价定量分析

面向平台方 / 大规模业务方，基于 **OpenRouter 聚合定价接口** 构建动态、可复现的成本测算链路，替代手工查阅各厂商官网的低效方式。

- **[大模型 API 定价策略定量分析框架](llm_api_pricing_analysis.md)**：主流模型分级矩阵（OpenAI / Anthropic / Google / DeepSeek / Qwen / Llama）、定价采集机制、缓存命中率与汇率敏感性分析
- **[`fetch_pricing.py`](fetch_pricing.py)**：零依赖（仅 Python 3 标准库）的动态定价测算脚本，支持自定义缓存命中率、输入/输出 token 数、人民币汇率等参数
- **[`openrouter_models_cache.json`](openrouter_models_cache.json)**：OpenRouter 全量模型报价的本地快照，用于离线复现与历史对比

> 典型用法：`python fetch_pricing.py --hit-rate 0.6 --input-tokens 30.0 --output-tokens 1.0 --exchange-rate 6.9`

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

## 3. 如何选择分析口径

面对一个成本问题，先分清用户画像再决定看哪份文档：

| 用户画像 | 核心诉求 | 主参考文档 |
|---|---|---|
| 平台方 / 大规模业务 | 百万 token 单价、月账单预算、模型切换 ROI | §1 API 定价定量分析 |
| 个人开发者 / 中小团队 | 订阅月费、限流条款、是否支持外接 Agent | §2 Coding Plan 深度对比 |
| 政企 / 合规敏感 | 数据出境、私有化交付、审计可观测 | §2 Coding Plan 报告 §1.2 选型速查表 |
