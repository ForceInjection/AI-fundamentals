# 素材：杰文斯悖论、需求增长与未来方向

> 用途：支撑演讲第四部分「革命之后发生什么」。

## 一、杰文斯悖论：价格跌、需求涨

### 1.1 核心结论

- 中金、中泰、摩根士丹利、高盛一致认为：**杰文斯悖论在 AI 推理领域并未失效**——单位 Token 成本快速下降非但没有压缩总开支，反而通过降低使用门槛激活更多应用场景，推动 Token 消耗量指数级爆发，带来更高的总需求与总收入。
- 风险信号不是 Token 价格继续下跌，而是"价格跌但用量停滞"——目前尚未出现。

### 1.2 价格端（下降）

- ~~每个 AI token 的成本三年内下降约 1000 倍~~（Edgen 报道口径）**【演讲勿用此倍数】**：为任务综合口径，端点不可复现；统一用可验证口径——纯单价 GPT-4 $60 → DeepSeek V4 $0.28 为 214 倍（见 04 §4.2 弃用注）。
- GPT-3.5 推理成本较初期降 280 倍；2022–2026 单价降幅达 99.9%。〔待复核：未找到原始出处，演讲前核实或删除〕
- 2020–2026 大模型推理价格下降约 600 倍。〔待复核：同上〕

### 1.3 需求端（爆发）

- OpenAI GPT-5.6 Luna 降价 80% → 两周 Token 调用量约 **×14**（周调用量 1.95 万亿），收入 **+34%**（TD Cowen / OpenRouter 数据）；Terra 降价 20% → 用量 **×5**、收入 **+45%**。（早期报道"一周 ×7、收入 +42%"为合并口径，引用以 TD Cowen 原始数据为准）
- AT&T 每日处理 Token 从 18 个月前的 10 亿增至 270 亿。
- Uber 在 4 月就烧光了整个 2026 年的 AI 预算。
- **Google 月度 Token 处理量：2024-05 的 9.7 万亿 → 2025-10 的 1300 万亿（约 134 倍）**；Google I/O 2026 最新口径已达约 3200 万亿/月，演讲前建议用最新值。
- **中国日均 Token 调用量：2024 初 1000 亿 → 2026 年 3 月 140 万亿（两年约 1400 倍）**——**国家数据局局长刘烈宏官方披露**（2026-03-23 中国发展高层论坛、03-24 国新办发布会；工信部 2026-04 引用："比 2024 年同期增长 1000 多倍"）。中间锚点：2025-06 底 30 万亿（官方）、2025 底 100 万亿、2026-03 140 万亿——曲线连贯可交叉验证。2026 上半年同比增长约 10–20 倍，主要来自 AI 编程和智能体。
- **Token 官方中文名「词元」**（2026-03 国家数据局定名，澎湃等报道）——token 被纳入官方统计口径，是"token 成为经济计量单位"的直接证据，可用于 outline 2.2 的论据。
- IDC：全球年度 Token 消耗量 2025 年 0.0005 Peta Token → 2030 年 15 万 Peta Token（CAGR 3418%）。〔口径注：此三数互斥——0.0005→15 万 Peta 的五年 CAGR 约 4860%，3418% 对应约 2.68 万 Peta；且 0.0005 PT/年 ≈ 13.7 亿 token/天，与其他来源差 4 个数量级。引用时二选一：用"五年增长约 3 亿倍"或改用 IDC 自洽的"2030 年 22 亿 AI Agent"预测〕

### 1.4 需求爆发的主要驱动力：Agent（消耗主体从人变机器）

- 智能体每完成一项任务触发 **10–20 次**模型调用；Gartner：智能体场景 Token 耗用量是普通对话的 **5–30 倍**。
- 多步推理、工具调用、Agent 协同、子 Agent 递归调用产生大量"隐形 token"。
- 案例：一家大型医疗保险公司不到一年月度 AI token 消耗量从 300 万飙升至超 1.5 亿。
- 2026–2028 可能是关键拐点，Agent 大规模部署有望带来一到两个数量级的"J 型爆发"。

## 二、商业模式变化（用量计费成为主流）

- 所有主要 AI 供应商从统一收费转向按 Token 计量：Anthropic 取消企业统一收费方案，OpenAI 将 Codex 改为按 token 计费。
- 前沿模型转向"能力溢价"定价（Anthropic Claude Fable 5 定价较前代翻倍）。
- Token 成为 AI 时代的"计量单位、结算单位、统计单位"，催生 Token 运营层（套餐化、路由聚合）与结算基础设施（Stripe 以 10 亿美元收购 Metronome）。
- 案例参考：周鸿祎透露内部员工用智能体做一份 PPT 累计消耗近 1 亿 token，按高端模型计费折算成本接近 1 万元。

## 三、投资含义（成本革命利好谁）

- 算力基础设施是核心受益环节：Token 单价下跌但总消耗量爆炸式增长，利好 GPU（英伟达）、光模块、网络设备、数据中心、电力等"卖铲人"。
- 高盛：AI 基础设施年度支出可能从 2026 年的 7650 亿美元增至 2031 年的 1.6 万亿美元。
- 大摩：开放权重模型降低 AI 成本不会削弱算力需求，反而通过杰文斯悖论推高 Token、算力、电力和基础设施总需求。
- 典型案例 **DeepSeek**：MoE + Flash MLA + KV Cache 压缩 + PD 分离等系统级优化，服务成本下降 90%、输出价格仅为海外前沿模型 2%–14%、仍实现 50–80% 毛利率——"成本陡降 + 需求放量"的盈利逻辑闭环。（数据来自券商研报与 SemiAnalysis 等公开分析，方向一致但具体数值无官方口径，引用时标注"公开分析"）

## 四、政策与产业落地（中国）

- 算力网被纳入国家重点规划"六张网"体系。
- 中国电信宁夏分公司发布首个以"Token 工厂"命名的百亿级集采项目（预估规模 **164.51 亿元**不含税）。
- 弘信电子在无锡建设大规模 Token 工厂；阿里巴巴成立 Token Hub（ATH）事业群。
- 2026 年初算力需求激增，H200/H100 租金环比上涨 15–30%；云厂商与模型厂商相继涨价 20–34%。

## 五、未来技术方向（GTC 2026）

- **Agent 经济**："每家 SaaS 公司都将变成 Agent-as-a-Service 公司"；每个工程师将拥有年度 Token 预算（额度约为年薪的一半），成为硅谷新招聘筹码。
- 架构演进：Feynman 带来新 GPU、LPU（LP40）和 CPU Rosa，首次同时支持铜线与光学封装水平扩展。
- 太空算力：NVIDIA Space One（太空无对流无传导，只能辐射散热）——演讲中可作"想象力"点缀，不宜展开。
- 技术路线：铜缆、光芯片、CPO 三者都要扩产，不是二选一。

## 六、来源链接

### 中国日均 token 调用量（官方口径，重点数据）

- http://big5.www.gov.cn/gate/big5/www.gov.cn/lianbo/202603/content_7063595.htm（中国政府网：我国日均词元调用量突破 140 万亿）
- http://news.china.com.cn/2026-03/24/content_118398028.shtml（中国网：两年增长超千倍）
- http://finance.people.com.cn/n1/2026/0325/c1004-40688455.html（人民网：日均 Token 调用量爆发式增长折射中国 AI 产业新图景）
- https://finance.cnr.cn/jjgd/20260325/t20260325_527562091.shtml（央广网：突破 140 万亿）
- https://m.bjnews.com.cn/detail/1777369716129170.html（新京报：工信部引用"截至 3 月底突破 140 万亿，比 2024 年同期增长 1000 多倍"）
- https://m.gmw.cn/2026-03/25/content_1304390930.htm（光明网：从日均"词元"调用量透视）
- http://www.chinanews.com.cn/cj/2025/08-14/10464675.shtml（中新网：截至 2025 年 6 月底日均 30 万亿——中间锚点）
- https://www.thepaper.cn/newsDetail_forward_32829374（澎湃：Token 中文名定了——词元）

### 其他

- https://finance.sina.com.cn/stock/stockzmt/2026-08-11/doc-inimwyqt7609059.shtml（中金：杰文斯悖论在 AI 推理未失效）
- https://www.edgen.tech/zh/news/post/jevons-paradox-drives-ai-compute-demand-as-token-costs-fall-1000x（Edgen：成本下降 1000 倍）
- https://www.investing.com/analysis/collapsing-token-prices-jevons-paradox-not-a-demand-crash-200686144（Investing.com：不是需求崩溃）
- https://finance.sina.com.cn/roll/2026-08-04/doc-inimeknz5549153.shtml（大摩：杰文斯悖论推高算力需求）
- https://m.21jingji.com/timeline/bc8e1cf8a816faa6b31381e6e9ad627f.html（21 经济：开放权重模型与杰文斯悖论）
- http://vip.stock.finance.sina.com.cn/q/go.php/vReport_Show/kind/lastest/rptid/832605479700/index.phtml（2026 AI 应用系列：TOKEN 经济学）
- https://news.qq.com/rain/a/20260427A01GXA00（腾讯新闻：智能定价革命）
- https://d.drcnet.com.cn/?version=integrated&docid=8276129&leafid=3025&chnid=1020（国研网：Token 经济价值结算基础设施）
- https://www.janushenderson.com/zh-hk/investor/article/chart-to-watch-insatiable-demand-for-compute-power-rationalises-massive-capex/（计算需求为资本开支提供依据）
- https://tech.ifeng.com/c/8vTA5uP72HS（智东西：魔形智能日卖数万亿 Token）
- https://finance.eastmoney.com/a/202608043831158026.html（中经：每一次 AI 问答都藏着一门暴涨千倍的生意）
- https://www.htx.com/zh-cn/feed/community/21503256/（OpenAI 降价后使用量与收入齐升，TD Cowen/OpenRouter 数据）
- https://caifuhao.eastmoney.com/news/20260816073817987968060（Token 工厂专家交流：政策/集采/租金）
