# Auditing Prompt Caching（ICML 2025）精读报告

> 来源：papers/auditing-prompt-caching-icml25.pdf（本目录下的 papers/，PDF 共 20 页，PMLR 正式版）。页码约定：正文引用统一写「p.X」，X 为 PDF 页脚页码（1–20）；与 PMLR 印刷页码的映射为 **印刷页 = 20476 + X**（如 p.5 = PMLR 20481），论文范围为 PMLR 267:20477–20496（PMLR v267 官方页，本仓外核实 2026-10-13）。核对方式：全文 pdftotext 抽取逐条对照 + 关键页（pp.3–8）转图目读复核 + 附录表格（pp.15–17、表 3–9）逐行抽取核对；表格与引用均回原文核对。只有推断/点评标「精读员注」；未核实写「未核实」。

## 一、速览

- 完整标题：Auditing Prompt Caching in Language Model APIs（p.1，无副标题）
- Venue 与页码：ICML 2025——第 42 届 International Conference on Machine Learning，Vancouver，Canada；PMLR 267:20477–20496；PDF 本体 20 页。
- 作者与机构：Chenchen Gu、Xiang Lisa Li、Rohith Kuditipudi、Percy Liang、Tatsunori Hashimoto，Stanford University（p.1）。
- 一句话贡献：把"缓存命中快、未命中慢"的时序差做成带**有效 p 值**的统计审计（KS 检验、单侧、假阳性率有保证），对 17 家真实 API 实测：8 家检出 prompt 缓存、7 家检出跨用户/跨组织的**全局共享**（含 OpenAI 的 embedding 接口）；副产品是推断出 OpenAI text-embedding-3-small 为 decoder-only 架构（此前未公开）（p.1、p.6）。
- 开源情况：代码与数据 `https://github.com/chenchenygu/auditing-prompt-caching`（p.1 脚注 1）。本仓外核实 2026-10-13（GitHub API）：创建 2025-02-13、最后 push 2026-05-15、Apache-2.0、10 stars、仓库描述为「Code for "Auditing Prompt Caching in Language Model APIs" (ICML 2025)」。
- Venue 链接：[PMLR gu25b](https://proceedings.mlr.press/v267/gu25b.html)、[OpenReview gUj2fxQcLZ](https://openreview.net/forum?id=gUj2fxQcLZ)（均本仓外核实 2026-10-13）。

## 二、问题与动机

（以下均为论文自述；关键表述给英文短引）

- **机制与信号**：prompt caching 复用 attention KV cache；"cache hits will tend to have a faster time to first token (TTFT)"（p.2）。在 decoder-only Transformer 里前缀复用精确保持模型行为；论文利用两条性质："1. Cache hits occur on prefix matches between prompts. 2. Cache hits tend to have a faster TTFT than cache misses (after accounting for prompt length)."（p.2）。
- **攻击模型**（图 1，p.1）：受害者 prompt 进缓存 → 攻击者发无关 prompt 得慢响应（未命中）→ 发共享前缀 prompt 得快响应（命中）。跨用户含义："If the cache is shared across users, then a prompt being cached implies that another user recently sent that prompt."（p.1）。§1 给攻击的可达性定性："an attacker could, in principle, detect cache hits from timing differences to infer that another user sent a prompt that shares a prefix with a given prompt."（p.1）。
- **动机：透明度**。Anthropic、DeepSeek、Fireworks、OpenAI 宣布了 prompt caching，但"even if a provider announces prompt caching, they may not state the level of cache sharing, i.e., per-user, per-organization, or global."（p.2）。
- **零假设定得很宽**：H0 不是"没有 KV 缓存"，而是"when u_victim sends a prompt x to the API, the API does not store any information about x that affects the TTFT T(x̃) for any future prompt x̃ sent by u_attacker"（p.3）。因此审计不依赖实现细节："we do not assume any particular implementation of prompt caching in our attacks. Indeed, we do not know technical details about the caching mechanisms used by the APIs we audited."（pp.8–9）。
- **三级共享与风险**（pp.3，图 2）：per-user（无泄露）、per-organization（组织内风险，例：CEO 的敏感数据；组织所有者可控）、global（最高风险，可学任何用户与组织）。
- **测量假设**（§2.2，p.3）：max tokens 设为 1 时"总响应时间 = TTFT"（"supported by most, if not all, real-world LLM APIs"）；客户端/服务端计时皆可，服务端计时在"more than half of the APIs we test"可测（HTTP 头未公开字段，脚注 3）。
- **与既有攻击工作的定位**（§7，pp.8–9）：最相关的是 Song et al. (2024) 与 Zheng et al. (2024b)，"primarily in simulated, controlled environments"；本篇的差别是审计实用、带统计保证、能精确定级共享层级、且能得出架构信息。

## 三、核心机制

### 3.1 统计假设检验框架（§3.1，pp.3–4）

- **H0: D_hit = D_miss；H1: D_hit ≠ D_miss**——无缓存时两个程序只能产生未命中，分布相同；有缓存时命中更快。
- **检验**：两样本 Kolmogorov-Smirnov（SciPy 实现），非参数；统计量为经验累积分布函数的最大差；因预期方向为"命中更快"，做**单侧**检验。
- **保证**："outputs valid p-values with respect to the null hypothesis of no caching, enabling guarantees on the false positive rate."（p.1）。

### 3.2 两个采样程序（§3.2，p.4）

- **命中程序**：采随机 prompt x → 受害者连发 NUM_VICTIM_REQUESTS 次缓存 x → 采 x̃ 与 x 共享**恰好 PREFIX_FRACTION × PROMPT_LENGTH** 个 token 前缀 → 攻击者发 x̃ 测 TTFT。PREFIX_FRACTION=1 测精确匹配；<1（审计用 0.95）测同前缀不同后缀（例 "a b c d" vs "a b c x"）。
- **未命中程序**：发随机 prompt 测 TTFT。prompt 分布是均匀随机的大小写英文字母、空格分隔——BPE 分词器都在空白处分词，"all prompts in P will be exactly PROMPT_LENGTH tokens long"；两随机 prompt 共享 ≥15 token 前缀的概率小于 10⁻²⁵，"this procedure accurately measures a distribution of cache miss times"。
- **顺序随机化**：两组样本随机顺序采集（防时间漂移）。
- **命中程序不保证命中**（图 6 注，p.14）：服务器路由等原因会让"命中分布"混入未命中，压低 recall。

### 3.3 四级递进、参数与成本（§4.1，pp.4–6）

- 审计四级：① 同 prompt、per-user（固定 NUM_VICTIM_REQUESTS=25）；② 同前缀不同后缀、per-user；③ per-org；④ global（不支持组织的 API 跳过级 3；级 4 退化为"不同用户"）。**只在上一级检出后继续**。
- 参数：PROMPT_LENGTH = 5000、NUM_SAMPLES = 250、α = 10⁻⁸；级 2–4 用 NUM_VICTIM_REQUESTS ∈ {1, 5, 25} 递增、首次显著即停；Bonferroni 校正——三种请求数 ÷3，两种计时法再 ÷2。
- 成本（p.6）：NUM_VICTIM_REQUESTS=25 时单测约 **3400 万 prompt token**；按 0.05–0.25 USD/百万 prompt token（chat API）计，**1.69–8.44 USD/测**。
- 模型选择（p.4）：开源权重 provider 测 Llama 3/3.1 8B Instruct；专有模型测其最新家族里最便宜的 chat 模型；另测专有 embedding 模型（不做开源权重 embedding 模型——论文未找到提供开源权重 decoder-only embedding 模型的服务）。

### 3.4 审计对象与主结果（§4.2，p.6；表 1/2，p.5）

- 被测 17 家：Anthropic、Amazon Bedrock、Microsoft Azure OpenAI、Cohere、Deep Infra、DeepSeek、Fireworks AI、Google、Groq、Hyperbolic、Lepton AI、Mistral、OctoAI、OpenAI、Perplexity、Replicate、Together AI（p.4）。
- 检出 8 家 / 全局共享 7 家；Anthropic（Claude 3 Haiku）与 OpenAI GPT-4o mini 恰检出 per-org、未检出 global——与两家文档一致（"confirming the efficacy of our audit procedure"），"we do not consider it a security vulnerability"（p.6）。
- OpenAI text-embedding-3-small 的全局共享属漏洞，"has been patched after our responsible disclosure prior to the release of this paper"（p.6）——全篇唯一被点名已修复项。
- DeepSeek：有 prompt caching 功能、响应返回 cache hit token 数（用于确认命中），但时间分布无显著差异（连双侧检验亦然）；"DeepSeek states that the cache is isolated per-user, and we empirically verified that this is the case based on the number of cache hit tokens returned in the API responses."（p.7）。
- 检出速度：最高层级多数 API 只需 NUM_VICTIM_REQUESTS = 1；仅 OpenAI 与 Azure 的 embedding 需要 25（推测多服务器独立缓存、随机路由）（p.6）。"In all APIs where we detected caching, all available timing methods resulted in significant p-values."（p.6）

## 四、关键数字

> 全部为论文对真实 API 的实测/表格值（审计窗口 2024-09 至 10 月初）；无仿真、无建模推算。凡"平均精度"均为分类器口径（命中程序为正类）。

| 指标                   | 数值                                                                                                                                    | 对照基线/条件                                                                                                 | 出处                          |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| 审计规模               | 17 家 provider；22 个模型 API（表 1 9 行 + 表 2 13 行）                                                                                 | 表 2 覆盖 10 家、Azure 两表重复（8 + 10 − 1 = 17）                                                            | p.4、p.6、表 1/2 p.5          |
| 检出缓存               | 8 家检出；其余 9 家（Amazon、Cohere、DeepSeek、Google、Groq、Hyperbolic、Mistral、OctoAI、Together）全部被测模型未检出                   | provider 口径                                                                                                 | p.1、p.6                      |
| 全局共享               | 7 家：Azure、Deep Infra、Fireworks、Lepton、Perplexity、Replicate、OpenAI                                                                | 另 2 家只到 per-org（Anthropic Claude 3 Haiku、OpenAI GPT-4o mini）                                            | p.1、p.6、表 1 p.5            |
| 平均精度（逐行）       | Azure 0.80；Deep Infra 0.84；Fireworks 0.77/0.79；Lepton 0.71/0.70；OpenAI embedding 0.78/0.79；Perplexity 0.90；Replicate —/1.00；Anthropic 0.84；OpenAI GPT-4o mini 0.79/0.86 | 客户端/服务端分开；最高检出层级；"大多数在 0.8 上下"（p.6）                                                  | 表 1 p.5                      |
| 检出所需 victim 请求数 | 最高层级多数 1 次即显著；OpenAI 与 Azure 的 embedding 需 25 次                                                                           | 级 2–4 按 {1,5,25} 递增、首次显著即停                                                                          | p.6、图 3 p.6                 |
| p 值示例               | Lepton 2.2×10⁻¹³⁸（级 1，双计时）；Replicate 服务端 8.6×10⁻¹⁵⁰（级 2）；Deep Infra 9.5×10⁻¹¹⁶（级 1 客户端）                              | α = 10⁻⁸；"many orders of magnitude smaller"                                                                   | 表 3–4 p.15                   |
| 审计配置               | PROMPT_LENGTH=5000；NUM_SAMPLES=250；PREFIX_FRACTION∈{1, 0.95}；NUM_VICTIM_REQUESTS∈{1,5,25}；α=10⁻⁸；Bonferroni ÷3、÷2                   | 四级递进；级 1 固定 25 次                                                                                     | pp.4–6                        |
| 单测成本               | 约 3400 万 prompt token；0.05–0.25 USD/百万 token → 1.69–8.44 USD/测                                                                     | NUM_VICTIM_REQUESTS=25 口径；2024 年价格                                                                      | p.6                           |
| 随机 prompt 前缀碰撞   | 两随机 prompt 共享 ≥15 token 前缀概率 < 10⁻²⁵                                                                                            | 均匀随机字母、空格分隔                                                                                        | p.4                           |
| 命中/未命中响应时延    | 服务端：命中 0.032–0.039 s；未命中 0.076–0.127 s；另有 0.216/0.280 s 慢样本                                                              | OpenAI text-embedding-3-small、同 prompt 连发 25 次、PROMPT_LENGTH=5000（附录 D 表 7 逐行读数）                | 表 7 p.17（定性描述 p.8）     |
| embedding 浮点差异     | 命中 vs 重算差 10⁻⁴–10⁻⁵/坐标                                                                                                            | 低精度 KV 缓存假设；个别慢响应返回第三种数值（疑不同 GPU）                                                     | p.8                           |
| 消融                   | prompt 长度 ≳1000 时平均精度高且稳、趋 0 掉到随机；前缀匹配长度变短则降；模型 1B–405B 均检出、与规模无明显关系                            | Fireworks/Perplexity/Replicate 的 Llama 8B、v=1                                                               | §4.3 pp.7–8、图 5 p.8         |
| 披露与修复             | 2024-10 披露；60 天窗口（"the actual time elapsed ended up being longer"）；"at least five providers made changes"；OpenAI embedding 已修补 | providers 名单未点名                                                                                          | p.1、p.6、p.9                 |
| 模拟攻击对照           | Song et al.：每 token 平均 200+ 次测量、需显式清缓存、本地无网络时延                                                                      | 论文据此判断其"currently unlikely to be real-world privacy threats"                                           | p.9                           |
| 计时与 TTFT 口径       | max tokens=1 → 总响应时间 ≈ TTFT；服务端计时可测于 "more than half" 被测 API                                                             | —                                                                                                             | p.3、脚注 3                   |

表 1 逐行（p.5，转图目读复核）：9 行中 "same prompt / per-user" 与 "same prefix / per-user" 全部 ✓；per-org 列为 ✓ 的仅 Fireworks、OpenAI embedding、Anthropic、OpenAI GPT-4o mini（其余为"—"未测）；global 列 ✓ 的为前 7 行（Azure、Deep Infra、Fireworks、Lepton、OpenAI embedding、Perplexity、Replicate），✗ 的为 Anthropic 与 OpenAI GPT-4o mini。

## 五、局限与边界

**论文自认的局限：**

1. **提取攻击未成功**（§4.4，p.7）："we were unable to execute practical prompt extraction attacks"；"Just one incorrect token causes complete failure due to the exact prefix match required for a cache hit"；并存在自污染——攻击者发送候选 prompt 后，"future measurements may produce a cache hit not because another user sent the prompt, but because the attacker sent it"。论文不声称提取不可能；已知 prompt 模板的场景可能更易。
2. **审计不解释实现**（pp.8–9）：不假设、不知道各 API 的缓存实现细节。
3. **不研究非精确复用**（p.9）：语义缓存、"前缀不精确匹配也复用 KV"（CacheBlend 等）不在研究范围；但"they are also likely susceptible to similar cache timing attacks, and our audit can easily be adapted"。
4. **per-user 之下架构泄露仍在**：架构泄露只需 per-user 缓存即成立，是否接受该风险"up to the provider"（p.8）。
5. **修复状态不可知全貌**："at least five providers made changes"，未逐家点名内容与时间（p.9）。

**精读员注（疑点/空白）：**

1. **时点性最强的一处**：审计窗口 2024-09 至 10 月初、客户端在加州（p.6），距今约两年；论文发布（2025-06 创建 PDF 元数据）前的修复只在 OpenAI embedding 一处点名。"至少 5 家"具体改了谁、改成什么，**未核实**；当下各 API 的共享层级也**未核实**（引用时必须带 2024 年秋口径）。
2. **缓解措施**：论文 §6 四条为 per-user 隔离、披露共享层级、关缓存、时延拉平（p.8）；未提出「cache key 随机化」类密钥侧缓解。
3. **论文内部一处主谓不一致**：§3.1 写未命中程序由 u_attacker 发随机 prompt（p.3），§3.2 却写由 u_victim 发（p.4）。对 H0 语义无影响（两程序都只产生未命中），疑为笔误。
4. **计数口径易误读**：正文 "8 out of 17 API providers" 是 provider 口径；表 1 有 9 行（OpenAI 两行）、表 2 有 13 行覆盖 10 家（Azure 两表重复）。8 + 9（未检出家数）= 17，与表 2 的 10 家相差 1 正因 Azure 重复出现。
5. **摘要与正文无冲突**：摘要 "seven API providers, including OpenAI"、"8 out of 17" 与 §4.2 一致；未见本系列常见的一手/二手数字冲突。
6. **无 "up to" 类夸大式宣称**：主要结果均为确定计数（8/17、7 家、p 值、平均精度），引用风险低；唯一含糊处是修复规模（"at least five"）。
7. **图 3 未逐面板核数**：直方图为分布示意，报告仅取表 7（p.17）的逐行数值作为时延例证；图 3 各面板的横轴范围（如 Anthropic 0.6–1.4 s）为读图印象，未逐点核。

## 六、写作素材

- **系列定位（三种取证位置之一：服务时延）**：本篇不接触任何 KV 数据，只用商业 API 的响应时间线；与 I Know What You Asked（NDSS '25）的共享行为取证、Shadow in the Cache（NDSS '26）的 KV 数据取证并列。攻击面最"轻"：无需共置、无需数据访问权、一根秒表即可发起。
- **与另外两篇的对照点**（写作时按各篇精读报告落定细节）：
  - **vs I Know What You Asked（NDSS '25）**：那篇在自建/多租户 vLLM 环境里利用 APC 共享恢复 prompt；本篇是在**真实商业 API** 上先证明"共享确实存在、且多数没有文档说明"，是攻击面的现实性背书——两篇合起来是"机制可行 × 现实存在"。
  - **vs Shadow in the Cache（NDSS '26）**：那篇要求拿到 KV 张量数据（前提较强），从数据本身重建输入；本篇反而证明**看都不用看数据**，时序差就够探测。
- **与本仓已有文章可勾连处（候选）**：`01_concepts/prefix_caching/prefix_caching.md`（前缀缓存机制）、`01_concepts/prefix_caching/claude_prompt_caching.md`（Anthropic API 的缓存功能与文档口径）、`01_concepts/prefix_caching/radix_attention.md`（SGLang RadixAttention 的共享实现）、`01_concepts/prefix_caching/rope_and_prefix_caching.md`（复用为何会失败——与"命中即泄露"互为表里：正是复用判定逻辑决定了 oracle 的灵敏度）、`reference_design/09-安全性与合规性.md`（系统侧安全设计）。
- **适合文章开头的"反差/场景"素材**：
  - **表 7 三档时延**：同 prompt 连发 25 次 → 0.03–0.04 s（命中）/ 0.08–0.13 s（未命中）/ 0.2 s+（第三种，疑跨 GPU），且命中档 embedding 与正常档差 10⁻⁵ 量级（p.17、p.8）——"响应时间的两档分布 + 数值的第三种值"是全文最具体的一个现场。
  - **"最好的审计结果是审计自己的失败"**：论文对 Anthropic 与 GPT-4o mini 的 per-org 检出与官方文档吻合，被当作**审计有效性的自我验证**（p.6）——统计工具先对已知答案校准，再报未知答案，这个方法学点值得写。
  - **"漏了一层就很难解释"**：审计逐级递进（per-user → per-org → global），只有在上一级显著后才升级测试；Level 4 的全球共享实锤背后是一个可复用的递进式审计设计（§4.1，p.5）。
  - **"最锋利的发现是一行架构信息"**：全局共享等发现大多被修补或属文档内行为，真正不可撤销的是"text-embedding-3-small 是 decoder-only"——一次缓存行为观察泄了模型架构（§5，p.8）。
