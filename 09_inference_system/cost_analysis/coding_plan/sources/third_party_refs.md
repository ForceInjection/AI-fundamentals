# 第三方资料对账清单（Coding Plan）

## 一、说明

本文件用于记录网络第三方资料（媒体文章、测评、数据库等）对各厂商 Coding Plan 定价与用量口径的描述，并与官方来源进行对账。对账原则为“以官方来源为最终裁决”，第三方仅用于交叉验证与识别过期信息。

- 访问日期：2026-04-17
- 官方来源清单：见 [official_links.md](file:///Users/wangtianqing/Project/study/AI-fundermentals/09_inference_system/cost_analysis/sources/official_links.md)
- 定价结构化数据：见 `data/pricing_normalized.json` 与 `data/pricing_table.md`
- 第三方对账工具站点（参考）：https://codingplan.org/

## 二、海外主流工具对账

### 2.1 GitHub Copilot

本节用于交叉验证 GitHub Copilot 的个人与团队套餐价格口径，重点关注 Pro 与 Pro+ 的月费与高级请求额度是否存在版本差异。

- 官方定价页：https://github.com/features/copilot/plans/
- 第三方资料：
  - https://costbench.com/software/ai-coding-assistants/github-copilot/
    - 摘录：宣称 Copilot Pro 为 $10 / month，Pro+ 为 $39 / month。
    - 对账结论：match（与官方定价页展示口径一致）。
  - https://awesomeagents.ai/reviews/review-github-copilot/
    - 摘录：宣称 Pro 为 $10 / month，Pro+ 为 $39 / month，Business 为 $19 / user / month。
    - 对账结论：match（价格与官方定价页展示口径一致；额度与功能需以官方为准）。

### 2.2 Cursor

本节用于交叉验证 Cursor 的个人套餐（Hobby、Pro、Ultra）与团队套餐（Teams）的价格口径。

- 官方定价页：https://www.cursor.com/pricing
- 第三方资料：
  - https://costbench.com/software/ai-coding-assistants/cursor/
    - 摘录：宣称 Pro 为 $20 / month，Ultra 为 $200 / month，Teams 为 $40 / user / month。
    - 对账结论：match（与官方定价页展示口径一致）。
  - https://www.nocode.mba/articles/cursor-pricing
    - 摘录：宣称 Hobby 为免费，Pro 为 $20 / month，Ultra 为 $200 / month。
    - 对账结论：match（与官方定价页展示口径一致）。

### 2.3 Windsurf（Codeium）

本节用于交叉验证 Windsurf 的个人与团队价格口径，并关注 Credits 计量方式是否与第三方描述一致。

- 官方定价页：https://windsurf.com/pricing
- 官方用量规则文档：https://docs.windsurf.com/plugins/accounts/usage
- 第三方资料：
  - https://costbench.com/software/ai-coding-assistants/windsurf/
    - 摘录：宣称 Pro 为 $15 / month，Teams 为 $30 / user / month。
    - 对账结论：match（价格与官方文档与定价页口径一致；Credits 额度需以官方文档为准）。
  - https://pecollective.com/tools/windsurf-pricing/
    - 摘录：宣称 Pro 为 $15 / month，Team 为 $30 / user / month。
    - 对账结论：match（价格口径一致；功能与模型阵容需以官方为准）。

### 2.4 Amazon Q Developer

本节用于交叉验证 Amazon Q Developer 的 Pro 订阅价格口径，并识别“按量计费”与“固定订阅”的表述差异。

- 官方定价页：https://aws.amazon.com/q/developer/pricing/
- 第三方资料：
  - https://aws.amazon.com/q/developer/build/
    - 摘录：页面中出现 $19 / month / user 的订阅描述（与定价页口径一致）。
    - 对账结论：match（与官方定价页一致）。

### 2.5 Tabnine

本节用于交叉验证 Tabnine 的个人与企业定价口径。由于部分区域访问官方定价页可能出现 403，本节同时记录官方可访问的备用入口。

- 官方定价页：https://www.tabnine.com/pricing/
- 官方备用入口：https://old-www.tabnine.com/pricing
- 第三方资料：
  - https://costbench.com/software/ai-coding-assistants/tabnine/
    - 摘录：宣称 Dev 为 $12 / user / month，Enterprise 为 $39 / user / month。
    - 对账结论：pending（当前官方定价页更偏向 “Get a quote” 报价口径，未在可公开页面中稳定展示固定价格；以官方报价为准）。

### 2.6 Replit

本节用于交叉验证 Replit 的订阅层级与价格口径，并识别“月付与年付折扣”导致的价格差异。

- 官方定价页：https://replit.com/pricing
- 第三方资料：
  - https://replit.com/pricing
    - 摘录：页面展示 Core 为 $25 / month（年付折算约 $20 / month），Pro 为 $100 / month（年付折算约 $95 / month）。
    - 对账结论：match（以官方定价页为准；折扣口径需明确计费周期）。

## 三、国内主流工具对账

### 3.1 阿里云百炼 Coding Plan

本节用于交叉验证阿里云百炼 Coding Plan 的常规定价与新客首月活动价口径。由于活动价具有时间敏感性，本节仅记录“是否存在活动价”与“口径一致性”，不承诺活动长期有效。

- 官方文档：https://help.aliyun.com/zh/model-studio/coding-plan
- 第三方资料：
  - 待补充：优先选择近 30 天内更新的官方社区或媒体文章，并逐条摘录价格字段做对账。
  - 对账结论：pending（当前以官方文档为准；第三方资料待补齐）。

### 3.2 腾讯云 Coding Plan

本节用于交叉验证腾讯云 Coding Plan 的常规定价与新客首月活动价口径。由于活动页内容可能随时间变化，本节对账优先依赖官方活动页快照与访问日期。

- 官方活动页：https://cloud.tencent.com/act/pro/codingplan
- 第三方资料：
  - https://www.cnblogs.com/maplerlate/articles/19774736
    - 摘录：作为国内 Coding Plan 汇总文章，包含对腾讯云 Coding Plan 的价格与限额描述（可能存在更新滞后）。
    - 对账结论：partial match（作为交叉验证参考；活动价与日常价以官方活动页为准）。
  - 对账结论：partial match（第三方描述可作为参考，但活动价高度时间敏感，以官方活动页为准）。

### 3.3 智谱 GLM Coding Plan

本节用于交叉验证智谱 GLM Coding Plan 的套餐与额度口径。若定价主要在控制台内展示，则以官方控制台入口与可公开访问的产品页为基准。

- 官方入口：https://open.bigmodel.cn/
- 第三方工具站点（汇总参考）：https://codingplan.org/（仅用于交叉验证，不作为最终裁决）
- 第三方资料：
  - https://github.com/imjuya/juya-ai-daily/issues/28
    - 摘录：提及 GLM Coding Plan 不同套餐与时间窗口系数等细则（价格字段可能随活动变化）。
    - 对账结论：partial match（作为“规则与口径”参考；价格仍需回到官方页面或控制台核对）。

#### 3.3.1 智谱龙虾套餐（团队协作版）

本节用于交叉验证智谱“龙虾套餐”的公开标价与包含 tokens 额度口径。该页面为前端渲染页面，抓取侧以“截图 + OCR”方式保留证据链，价格以官方页面可视文本为准。

- 官方页面：https://www.bigmodel.cn/claw-plan-team
- 第三方资料：
  - 待补充：优先选择可公开访问且明确标注“￥39 / ￥99”等价格字段的汇总文章，并在此记录对账结论。
  - 对账结论：pending（当前以官方页面为准；第三方资料待补齐）。

### 3.4 火山方舟 Coding Plan

本节用于交叉验证火山方舟 Coding Plan 的定价与“仅限 AI 编程工具、禁止 API 调用”的限制口径。

- 官方文档：https://www.volcengine.com/docs/82379/1925114?lang=zh
- 官方活动页：https://www.volcengine.com/activity/codingplan
- 第三方资料：
  - https://juejin.cn/post/7613191044306829339
    - 摘录：对火山方舟 Coding Plan 的价格与额度机制进行横评整理。
    - 对账结论：match（价格与额度口径可在官方文档与活动页中找到对应字段；以官方为最终裁决）。

### 3.5 MiniMax Token Plan

本节用于交叉验证 MiniMax Token Plan 的“按月/按年订阅 + 5 小时滚动窗口”计量口径，以及不同档位的价格与配额。

- 官方定价文档：https://platform.minimaxi.com/docs/guides/pricing-token-plan
- 第三方资料：
  - https://zhuanlan.zhihu.com/p/2020125047965131295
    - 摘录：提及 Token Plan 入门价与“全模态订阅”的产品定位。
    - 对账结论：partial match（价格需以官方定价文档为准；第三方偏产品解读）。
  - https://codingplan.org/plans/minimax
    - 摘录：对各档位价格与请求额度进行汇总。
    - 对账结论：match（关键价格与 5 小时请求额度可在官方定价文档中核对）。

### 3.6 Kimi Code Plan

本节用于交叉验证 Kimi Code Plan 的价格档位与“月度活动价/常规月费并列展示”的口径。

- 官方入口：https://www.kimi.com/code
- 官方文档：https://www.kimi.com/coding/docs/
- 第三方资料：
  - https://codingplan.org/plans/kimi
    - 摘录：对 Kimi Code Plan 档位与权益进行汇总（可能存在口径差异）。
    - 对账结论：partial match（价格以 Kimi Code 页面为准；权益与用量规则以官方文档为准）。
