# 0.03 秒命中、0.1 秒未命中：ICML'25 审计发现 7 家 LLM API 跨用户共享 prompt 缓存

> **元信息**：论文 _Auditing Prompt Caching in Language Model APIs_，Chenchen Gu、Xiang Lisa Li、Rohith Kuditipudi、Percy Liang、Tatsunori Hashimoto（Stanford University），ICML 2025，Vancouver。PMLR 267:20477–20496；本 PDF 页脚编号 1–20，正文引用页码 p.X 即 PDF 页脚页（PMLR 印刷页 = 20476 + X）。代码与数据 `github.com/chenchenygu/auditing-prompt-caching`（Apache-2.0，论文 p.1 脚注 1）。精读方式：全文逐条抽取核对 + 关键表图页版式复核（报告见 [references/auditing-prompt-caching-report.md](references/auditing-prompt-caching-report.md)），数字均标论文页码。阅读日期 2026-10-13。

同一个 prompt 连续发 25 次到 OpenAI 的 text-embedding-3-small 接口，抓到的响应时间分成几档：大多数落在约 0.08–0.13 秒，少数只要 0.03–0.04 秒，个别超过 0.2 秒；再细看返回的 embedding，快档的数值与"正常"档在小数点后第 4–5 位不一致（p.17 表 7、p.8）。快档是缓存命中——这个 prompt 的 KV cache 被复用、直接"取回"；正常档是从头计算；超慢的第三档，论文推测是请求落到了不同 GPU 型号上（p.8）。一条时间线，就是一个能读出"缓存里有什么"的 oracle。

Stanford 这篇 ICML'25 论文把这个 oracle 做成了对 17 家商业 API 的统计审计，结论是：prompt 缓存被检出 8 家，其中 7 家是跨用户、跨组织的**全局共享**——包括 OpenAI（p.1、p.6）。攻击者不需要读取任何 KV 数据、不需要与受害者共置，仅凭响应时间就能探测出"另一个用户最近发过与我共享前缀的 prompt"。这是一篇审计与测量论文，不是攻击系统论文：它给的是带有效 p 值的检测方法、覆盖 17 家的实测账本和一条负责任披露时间线；逐 token 提取全文的攻击它明说没能做到（§4.4，p.7）。

## 一、问题：缓存命中为什么构成一条泄露通道

### 1.1 机制：KV cache 复用把 TTFT 变成数据依赖量

prompt caching 在 serving 侧就是 attention KV cache 的跨请求复用：prompt 的 KV 被存进缓存，后续请求只要与缓存内容有匹配前缀，就能取回这段 KV，"cache hits will tend to have a faster time to first token (TTFT)"（p.2）。论文特意点明，在 decoder-only Transformer 上（每个 token 只注意之前的 token），前缀复用**精确保持模型行为**，即使后缀不同（p.2）。论文利用的性质就两条：一是命中发生在前缀匹配处；二是命中更快（在考虑 prompt 长度之后）（p.2）。

### 1.2 攻击三角：先缓存、再对照、后探测

论文的 Figure 1（p.1）给出最小攻击模型：受害者先发一个 prompt，它进了缓存（1）；攻击者发一个不相干 prompt，未命中、响应慢（2）；攻击者再发一个与受害者共享前缀的 prompt，命中、响应快（3）。由快慢之差即可推断命中与否，而"an attacker could identify prompts that yield fast API response times; such prompts are likely cached. If the cache is shared across users, then a prompt being cached implies that another user recently sent that prompt."（p.1）。同类"命中/未命中时序差"在安全史上从 Meltdown、Spectre 一路用到 AES 密钥提取与 web 隐私（§7，p.8–9）。

### 1.3 三级共享，三级风险

论文把共享粒度定义为三级（§2.3，图 2，p.3）：per-user（命中只发生在自己发过的 prompt 上，"no potential privacy leakage"）、per-organization（组织内跨用户，风险存在但组织所有者可控——论文举的例子是"CEO 知道敏感数据"）、global（跨所有用户与组织，"the highest risk of privacy leakage"）。这三级的递进也直接变成了审计的递进梯度（见 2.3）。

### 1.4 为什么值得审：透明度是一种安全属性

动机落在披露上：Anthropic、DeepSeek、Fireworks、OpenAI 都宣布了 prompt caching 功能，但论文指出"even if a provider announces prompt caching, they may not state the level of cache sharing, i.e., per-user, per-organization, or global"（p.2）。审计的零假设也刻意定得很宽——不是"没有 KV 前缀缓存"，而是"当受害者发送 prompt x 时，API 不存储任何影响攻击者未来 TTFT 的信息"（p.3）。这使审计与具体实现解耦："we do not assume any particular implementation of prompt caching in our attacks. Indeed, we do not know technical details about the caching mechanisms used by the APIs we audited"（pp.8–9）。也因此它不挑实现：语义缓存等变体"are also likely susceptible to similar cache timing attacks"（p.9）。

### 1.5 测量假设

两个关键假设（§2.2，p.3）：把 max tokens 参数设为 1，让整段响应时间近似等于 TTFT——"supported by most, if not all, real-world LLM APIs"；计时客户端侧、服务端侧皆可，服务端计时可从 HTTP 响应头的未公开字段读出，论文称在"more than half of the APIs we test"上可行（脚注 3，p.3）。

## 二、机制：一个带 p 值的审计怎么工作

### 2.1 假设检验框架（§3.1，pp.3–4）

审计被写成一次统计假设检验：设 D_hit 与 D_miss 分别为"命中程序"与"未命中程序"采样到的 TTFT 分布。零假设 H0：D_hit = D_miss（无缓存时两个程序都只会产生未命中）；备择假设 H1：D_hit ≠ D_miss（缓存时命中更快）。检验用两样本 Kolmogorov-Smirnov（SciPy 实现）——非参数、比较两条经验累积分布函数的最大差；因为预期方向是"命中更快"，所以做单侧检验。论文强调其产出是"valid p-values with respect to the null hypothesis of no caching, enabling guarantees on the false positive rate"（p.1）——这就是"审计"与"演示攻击"的区别：它有假阳性率保证。

### 2.2 两个采样程序（§3.2，p.4）

- **命中程序**：先采一个随机 prompt x（PROMPT_LENGTH 个大小写英文字母、空格分隔——因为主流 BPE 分词器都在空白处分词，所以 x 恰好是 PROMPT_LENGTH 个 token）；受害者连续发 NUM_VICTIM_REQUESTS 次把 x 缓存；再采一个与 x 共享**恰好** PREFIX_FRACTION × PROMPT_LENGTH 个 token 前缀的新 prompt x̃；攻击者发 x̃ 并测 TTFT。PREFIX_FRACTION = 1 时测精确匹配，= 0.95 时测"同前缀、不同后缀"（如 "a b c d" 对 "a b c x"）。
- **未命中程序**：发随机 prompt 并测 TTFT。随机字母 prompt 之间共享 15 个 token 以上前缀的概率小于 10⁻²⁵（p.4），所以这是干净的未命中分布。
- 两组样本**随机顺序**采集，防时间漂移；缓存命中程序不保证命中（服务器路由等原因），所以"命中分布"里混有少量未命中（图 6 注，p.14）。

### 2.3 四级递进与配置（§4.1，p.5；§4.2，p.6）

审计分四级递进，每级只在上一级检出缓存后才继续：(1) 同 prompt、per-user——粗筛；(2) 同前缀不同后缀、per-user；(3) per-org；(4) global（不支持组织的 API 跳过第 3 级，第 4 级退化为"不同用户"）。配置：PROMPT_LENGTH = 5000、NUM_SAMPLES = 250、显著性水平 α = 10⁻⁸；第 2–4 级按 NUM_VICTIM_REQUESTS ∈ {1, 5, 25} 递增试探、首次显著即停；多重检验做 Bonferroni 校正——三种请求数除以 3，客户端/服务端两种计时再除以 2。成本：NUM_VICTIM_REQUESTS = 25 时单次测试约 3400 万 prompt token，按 0.05–0.25 美元/百万 token 计，1.69–8.44 美元一次（p.6）。

## 三、评估：17 家的实测账本

审计窗口：2024 年 9 月至 10 月初，客户端位于加州（p.6）。这一时点对全篇数字都适用（见「诚实声明」）。

### 3.1 总账：8 家检出、7 家全局共享（§4.2，p.6；表 1，p.5）

17 家 provider 中检出缓存的 8 家；其余 9 家——Amazon Bedrock、Cohere、DeepSeek、Google、Groq、Hyperbolic、Mistral、OctoAI、Together——在所有被测模型上均未检出。同一模型换个 host 结论会翻转：Claude 3 Haiku 在 Anthropic 上检出 per-org、在 Amazon Bedrock 上未检出；GPT-4o mini 在 OpenAI 上检出 per-org、在 Azure 上未检出（表 1/2，p.5）。按最高检出层级归类：

| provider（被害模型）                                  | 检出的最高层级 | 平均精度（客户端/服务端） |
| ----------------------------------------------------- | -------------- | ------------------------- |
| Azure、Deep Infra、Fireworks、Lepton、Perplexity、Replicate、OpenAI（text-embedding-3-small） | **全局共享**   | 0.70–1.00，大多 ~0.8      |
| Anthropic（Claude 3 Haiku）、OpenAI（GPT-4o mini）    | per-org（未检出 global） | 0.84；0.79/0.86 |

表 1 的逐行读数：Azure 0.80、Deep Infra 0.84、Fireworks 0.77/0.79、Lepton 0.71/0.70、OpenAI embedding 0.78/0.79、Perplexity 0.90、Replicate 服务端 1.00、Anthropic 0.84、OpenAI GPT-4o mini 0.79/0.86（p.5）。平均精度（average precision，命中程序为正类）体现的是"分类器"口径："cache hits can be detected with near perfect precision up to moderate recall scores"（p.6，图 4）；响应时间直方图上命中/未命中两组分布明显分离（图 3，p.6）。注意表 1 的 9 行对应 8 家 provider（OpenAI 占两行）——"8 out of 17"按供应商计。

### 3.2 检出有多容易、显著性有多硬（p.6；附录 B 表 3–6，pp.15–16）

"在多数检出缓存的 API 上，最高共享层级只需 NUM_VICTIM_REQUESTS = 1 就够"——只有 OpenAI 与 Azure 的 text-embedding-3-small 需要 25 次，论文据此推测这两家"have multiple servers with separate caches and that requests are randomly routed to a server"（p.6）。p 值常常比 10⁻⁸ 低几十个数量级（例：Lepton 2.2×10⁻¹³⁸、Replicate 服务端 8.6×10⁻¹⁵⁰，表 3–4，p.15）；"In all APIs where we detected caching, all available timing methods resulted in significant p-values"（p.6）。

### 3.3 对照文档：有的命中有据、有的属于漏洞（p.6）

Anthropic Claude 3 Haiku 与 OpenAI GPT-4o mini 恰好检出了 per-org、未检出 global——这与两家自己的文档一致，"confirming the efficacy of our audit procedure"，论文明确"we do not consider it a security vulnerability"。而 OpenAI text-embedding-3-small 的全局共享"was a potential vulnerability, but has been patched after our responsible disclosure prior to the release of this paper"（p.6）——这是全篇唯一被点名已修复的发现。DeepSeek 是个反例样本：它自己有 prompt caching 功能、响应里还返回 cache hit token 数（论文用它确认自己确实制造了命中），但时间分布查无差异，连双侧检验都不显著；其"缓存按用户隔离"的自述被论文用 cache hit token 计数经验证实（p.7）。

### 3.4 副产品一：从缓存行为推断模型架构（§5，pp.7–8）

"同前缀、不同后缀可命中"只在 decoder-only Transformer 里可能——encoder-only / encoder-decoder 里每个 token 注意整段 prompt，前缀复用不成立。论文在 text-embedding-3-small 上检出该行为，并确认改了后缀 embedding 就变（排除"返回近似 prompt 的缓存输出"），由此推断"text-embedding-3-small is a decoder-only Transformer. This is new information, as OpenAI has not released any information about the architecture of their embedding models"（p.8）。对专有 embedding 模型，架构信息本身就是知识产权的泄露——这条泄露只需要 per-user 缓存即成立。

### 3.5 副产品二：缓存里的浮点痕迹（§5，p.8；附录 D，p.17）

命中返回的 embedding 与从头计算的相比，每个坐标差 **10⁻⁴ 到 10⁻⁵** 量级。论文的假设：被复用的 KV cache 以更低浮点精度存储，命中时"取回"与未命中时"重算"产生细微差异；个别特别慢的响应返回第三种数值，可能与落到不同 GPU 型号有关（p.8，表 7–9 给出原始数据）。这条是把"缓存存在性"证据细化到数值层面的一笔注脚。

### 3.6 消融（§4.3，pp.7–8；图 5，p.8）

在 Fireworks、Perplexity、Replicate 的 Llama 3/3.1 8B API 上（v = 1）：prompt 长度 ≳1000 token 时平均精度高且稳定，趋近 0 时掉到随机水平；前缀匹配长度变短，平均精度随之下降（PROMPT_LENGTH = 1000 下扫描）；模型规模从 1B 到 405B 都能检出，与规模无明显关系（图 5(d)，p.8）。

## 四、边界：审计能证明什么、不能证明什么

### 4.1 提取攻击没能实用化（§4.4，p.7）

逐 token 广度优先搜索在原理上可行——但需要极准的单 token 命中检测："Just one incorrect token causes complete failure due to the exact prefix match required for a cache hit"；论文自述在预实验中未能做到——"we were unable to reliably detect the presence of one additional cached token"（p.7）。还有一层自污染：攻击者一旦发送候选 prompt，它就进了缓存，"future measurements may produce a cache hit not because another user sent the prompt, but because the attacker sent it"。论文不声称提取攻击不可能（"We do not claim that prompt extraction attacks are necessarily impossible"），已知 prompt 模板的场景也许更容易；但当下它证明了"可探测性"，没有证明"可提取性"。

### 4.2 与模拟环境攻击的距离（§7，pp.8–9）

最相关的工作 Song et al. (2024) 与 Zheng et al. (2024b) 在模拟、受控环境里研究同类泄露。论文对前者逐条对比：攻击跑在本地、没有网络抖动；需要知道 prompt 分布；需要显式清空缓存以重复测量；每提取一个 token 平均要 200+ 次测量——"Due to these limitations, we believe that these simulated attacks are currently unlikely to be real-world privacy threats"（p.9）。本篇的差别在于：审计在真实 API 上跑、且带统计保证。

### 4.3 缓解手段（§6，p.8）

论文给出的缓解清单就四条，都作用于**共享边界与响应可观测性**，不动缓存算法本身：

- **per-user 隔离**：完全防泄露的唯一档位（"To completely prevent any privacy leakage from prompt caching, only per-user caching should be allowed"），论文判断性能代价小——不同用户本就很少发长前缀相同的 prompt，"per-user caching should retain many of the performance benefits from global cache sharing"。
- **披露共享层级**：让用户能做决策，例：企业知道是 per-org 后可按员工组拆组织。
- **关缓存**：对"只需 per-user 就会发生"的泄露（即架构泄露）最强；但论文把权衡留给 provider（"up to the provider to determine their level of risk tolerance"）。
- **时延拉平**："intentionally delay the response time for cache hits so that they look like cache misses"——用户失去缓存收益，provider 仍省 GPU 时间。

### 4.4 负责任披露与修复（p.1、p.6、p.9）

2024 年 10 月逐家披露、给 60 天修复窗口（"the actual time elapsed ended up being longer"）、"at least five providers made changes to mitigate vulnerabilities, e.g., disabling global cache sharing across organizations and updating documentation"；论文发布前，OpenAI embedding 的全局共享已修补（p.6）。论文另声明利益冲突事项：Percy Liang 是 Together AI 联合创始人，本工作以 Stanford 身份完成、方法和结果在公开前未与 Together 共享，所有 provider 用同一套程序审计（含 Together）（p.9）。

## 论文结构索引

| 论文位置              | 内容                                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| §1 / p.1–2            | 动机与总账（8/17、7 家全局）；Meltdown/Spectre 谱系；架构泄露预告；负责任披露时间线                      |
| §2.1–2.2 / p.2–3      | prompt 缓存机制与 TTFT；两条被利用性质；API 假设（max tokens=1、双计时渠道）                           |
| §2.3 / p.3            | 用户/组织定义与三级共享及对应隐私风险（图 2）                                                          |
| §3.1 / p.3–4          | 审计形式化：H0/H1、D_hit = D_miss、KS 检验                                                             |
| §3.2 / p.4            | 实现细节：随机字母 prompt 分布、命中/未命中程序、单侧检验、四配置参数                                   |
| §4.1 / p.4–6          | 审计配置：17 家与模型选择；四级递进；α=10⁻⁸ 与 Bonferroni；单测成本                                   |
| §4.2 / p.6            | 主结果：8 家检出、7 家全局；与文档对表；OpenAI embedding 已修补；DeepSeek 时序不可检出（图 3、表 1/2） |
| §4.3 / p.7–8          | 消融：prompt 长度、前缀匹配长度、模型规模（图 5）                                                      |
| §4.4 / p.7            | 提取攻击失败分析：单 token 检测精度、自污染、模板场景例外                                              |
| §5 / p.7–8            | 架构泄露（decoder-only 证据）与缓存浮点精度差异                                                        |
| §6 / p.8              | 缓解四件套：per-user、披露、关缓存、时延拉平                                                           |
| §7–8 / p.8–9          | 相关工作（与模拟环境攻击的距离）与结论                                                                 |
| Impact Statement / p.9 | 负责任披露与更广影响；利益冲突声明                                                                    |
| 附录 A–D / p.13–19    | 全量 PR 曲线、p 值表（表 3–6）、消融的 p 值图（图 7，p.20）、embedding 与响应时间样例（表 7–9）         |

## 与系列其他篇目

三篇凑齐了 KV cache 泄露面的三种取证位置：[I Know What You Asked](i-know-what-you-asked.md)（NDSS '25）从多租户服务的**调度行为**（LPM 下的请求返回顺序）逐 token 恢复他人 prompt；[Shadow in the Cache](shadow-in-the-cache.md)（NDSS '26）走得更深，从 KV **数据本身**重建用户输入、并给出 KV-Cloak 防御；本篇两样都不碰——只用商业 API 的**响应时间线**审计出"跨用户共享正在发生"这一事实。它是系列里攻击面最轻的一篇：不要求与受害者在同一服务上共置，也不要求任何数据访问权，一根秒表即可发起。横切总纲见 [KV Cache 安全总纲](kv-security-overview.md)。

## 诚实声明

- **这是一份 2024 年秋的快照**：审计窗口为 2024 年 9 月至 10 月初（p.6），距今约两年；论文只说"at least five providers"修复、未逐家点名改了什么，各家 API 当下的缓存与共享策略**未核实**——引用本期数字必须带"2024 年秋"的口径。
- **未复现**：本仓未做任何独立审计或复现，全部数字来自论文原文；提供商名称与模型名也是 2024 年的口径（如 OctoAI 等此后是否仍以原形态提供服务，未核实）。
- **"8 家"是 provider 口径，不是接口口径**：表 1 有 9 行（OpenAI 占两行）；表 2 共 13 行、覆盖 10 家（Azure 因 embedding 与 GPT-4o mini 结论不同，两表都出现）。"7 家全局共享"同样按 provider 计。
- **平均精度不是攻击成功率**：average precision 是分类器口径（命中程序为正类）；且命中程序不保证命中（服务器路由会带来假阴性），recall 本身有上限（图 6 注，p.14）。
- **标题与正文开篇的 0.03/0.1 秒**来自表 7（p.17）：OpenAI text-embedding-3-small、服务端计时、同 prompt 连发 25 次的单次审计样例，不是全 API 一般值；其他 API 的时延量级不同（图 3，p.6）。
- **论文内部一处小出入**：§3.1 写未命中程序由攻击者发随机 prompt（p.3），§3.2 却写由受害者发（p.4）；对 H0 的检验语义没有影响，原样记录（详见精读报告疑点清单）。
- **论文自认未做的**：未研究语义缓存等非精确复用机制（p.9）；未尝试实用化提取攻击（p.7）；论文只知道"至少 5 家做了改动"，修复的完整清单与时间线不可得。

## 参考资料

- Chenchen Gu, Xiang Lisa Li, Rohith Kuditipudi, Percy Liang, Tatsunori Hashimoto. _Auditing Prompt Caching in Language Model APIs_. ICML 2025, PMLR 267:20477–20496。本地 PDF：[references/papers/auditing-prompt-caching-icml25.pdf](references/papers/auditing-prompt-caching-icml25.pdf)；[PMLR 论文页](https://proceedings.mlr.press/v267/gu25b.html)；[OpenReview](https://openreview.net/forum?id=gUj2fxQcLZ)
- 代码与数据：[github.com/chenchenygu/auditing-prompt-caching](https://github.com/chenchenygu/auditing-prompt-caching)（Apache-2.0，论文 p.1 脚注 1；本仓外核实 2026-10-13，仓库于 2025-02 创建）
- 精读报告（含逐条页码核对与疑点清单）：[references/auditing-prompt-caching-report.md](references/auditing-prompt-caching-report.md)
