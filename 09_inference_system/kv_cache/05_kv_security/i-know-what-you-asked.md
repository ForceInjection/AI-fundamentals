# 缓存命中的 1 个 bit 出卖了你：NDSS'25 PromptPeek 平均 99% 成功率复原他人 prompt

> **元信息**：论文 _I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving_，Guanlong Wu、Weili Wang、Jianyu Niu、Yinqian Zhang（南方科技大学 SUSTech，Yinqian Zhang 为通讯作者）+ Zheng Zhang、Yao Zhang、Ye Wu（ByteDance Inc.），NDSS Symposium 2025（2025-02-24~28，San Diego），DOI 10.14722/ndss.2025.241772，PDF 共 15 页。精读方式：官方论文 PDF 全文逐条抽取核对 + 表格与图 300–400 dpi 渲染逐标签复核（报告见 [references/i-know-what-you-asked-report.md](references/i-know-what-you-asked-report.md)）；数字均标论文印刷页码，读图值另行标注。阅读日期 2026-10-13。

60 个请求，四个占位符全部读出。受害者用的模板是 "Calculate BMI with an explanation, then create two plans ... using the following inputs: [your gender], [age], [weight], [height]"，攻击者把占位符填成 "male"、"35"、"90kg"、"5 feet 9 inches"——论文对这一案例的结论是 "the attacker only need 60 requests to reverse all the placeholders and uncover the user's secrets"（§VI-C，p.11；引言的表述是借此认定攻击"低成本"，"with just 60 requests in total"，p.2）。攻击者没碰模型权重、没有 KV 张量、没有任何服务器权限，做的只是反复向同一个多租户 LLM 服务提交自己的猜测，然后观察这些请求被**返回的顺序**。

这套攻击叫 PromptPeek。NDSS'25 的这篇论文给自己的定位是 "the first investigation on security risks associated with multi-tenant LLM serving"，判断 "the state-of-the-art mechanisms of KV cache sharing may lead to new side channel attack vectors"（p.1）。它盯上的不是 KV 里存了什么，而是共享行为本身：攻击者需要的全部信息量，就是「我的请求命中了别人已缓存的前缀吗」这一个 bit——而这个 bit 可以从 SGLang 默认调度策略造成的服务顺序里读出来。在 A100 80G + Llama2-13B 上，无背景知识时全 prompt 复原的平均成功率 95%，知道模板后复原用户输入 99%、已知输入反推模板 98%（p.2）。

## 一、问题：省显存的共享，就是跨租户的信道

### 1.1 共享机制：前缀一致，缓存一致

KV cache 是多租户服务的显存大头：论文给的数量级是每 token 约 1 MB，且很容易比模型权重还大（"It consumes roughly 1MB per token [13], and can easily grow larger than the model weights [12]"，p.2）。省它的主流办法是复用——"the KV cache produced by a token in the request can be reused by another request, as long as all preceding tokens match"（§II-B，p.3）：只要从第一个 token 起完全一致，后面的 KV 就能被直接继承。Figure 1 的例子是两个用户都在开头写 "Imagine you are an IT expert and tell me how to install"，第二个用户只需计算 "Linux" 那一段（p.1）。

论文对着两个主流框架梳理了共享实现（§II-B，p.3；§III-A，p.4）：

- **vLLM**：请求生成带元数据的 KV Cache Block，块上记录前缀 token 的哈希、最近访问时间与引用计数；"These caches are retained in memory for as long as possible"，显存到限时逐出最旧的，新请求靠哈希比对找可复用块（p.3）。
- **SGLang**：基数树（radix tree）存 KV；调度用 LPM（Longest Prefix Match）——"those having longer shared KV will be served earlier"，匹配越长越先被服务；逐出用 LRU，且到达容量上限时一次性清空——"all KV cache entries except those in the current batch are evicted simultaneously"（p.4）。

实验以 SGLang 为准，理由写在 §VI-A：vLLM "first introduced the concept of KV cache sharing" 但机制 "is still in development"，而 SGLang "offers a more comprehensive discussion on KV cache sharing, scheduling, and eviction policy"（p.4、p.9）。

LPM 那条"匹配越长越先服务"是整篇攻击的支点：匹配长度哪怕只多一个 token 都会被优先（"even by one token"，§IV-A，p.5），于是多租户共用的等待队列变成了一个攻击者可读的端口。与以往"近似复原 prompt"的工作（Sha 等、PRSA）相比，论文强调这条路能拿到 exact reconstruction："P ROMPTPEEK accurately reconstructs prompts"，且场景 2、3 里不依赖受害者的模型输出（pp.1、13）。

### 1.2 威胁模型：攻击者是"一个普通用户"

论文的威胁模型不含任何特权假设（§III-B，p.4）：攻击者的能力 "are the same as those of an ordinary user, with no direct control over the inference server"；对系统内部只知默认机制（调度、逐出等，这是侧信道研究的标准假设）；模型层面的背景知识 "extends only to the tokenizer, which is always accessible to the adversary."；交互面只有框架的公开客户端 API——"adversaries can only interact with the server through the commonly defined client APIs"（例如 max_tokens 这类参数，p.4）。目标一句话："The adversary's goal is to reconstruct the prompts sent by other users using the side channel information."（p.4）

## 二、机制：把调度顺序用成一个 1-bit 判决器

### 2.1 候选生成：本地模型猜词，加一个 dummy token

攻击逐 token 进行（§IV-A，p.5）：每轮先让本地 LLM "generate the top k most likely tokens as candidates, based on the already reconstructed fragment"——已还原到 "Imagine you are"，本地模型给出 "Imagine you are an" / "a" / "the" 等候选（p.5，Fig.5）。本地模型只需与目标服务共用同一个 tokenizer（论文为简化直接假设用同款 LLM），避免候选与目标被分词方式岔开（p.5）。

与候选一起生成的还有 dummy 请求：本地模型挑出"最不可能的 token"（如 "%"）拼成 "Imagine you are %"，它的职责是把侧信道图案衬出来；一旦 dummy token 真的命中（概率极小），本轮立即换靶（p.5）。

### 2.2 token 选择：命中与未命中是两种可区分的返回序列

每轮按 dummy → 候选 → dummy 三批发送（Fig.5/6）。设计要点（§IV-A，pp.5–6）：

- **pre-dummy** 数量要超过服务端最大批大小，先把运行批填满，逼候选进等待队列、接受 LPM 重排；**post-dummy** 同样要超过最大批，作用是 "amplify the order effect"，把命中候选从其余候选里隔出来（p.6）。
- **判读规则**：命中时返回序是 "pre-candidate dummy requests, the matched request, post-candidate dummy requests, and then the remaining candidate requests"；未命中时是 "pre-candidate dummy requests, post-candidate dummy requests, followed by all candidate requests"（p.6）。
- 图案为什么成立：dummy 请求自己也先跑过一轮、把 "%" 写进了缓存，于是 dummy 的匹配长度比未命中候选多一位；命中候选与 dummy 等长，而 "Requests with equal match lengths retain their original order in the queue."——于是它恰好嵌在两批 dummy 之间（p.6）。
- **成本控制**：所有请求的输出长度都被客户端 API 压到 1 个 token，"This approach minimizes the strain on GPU memory and maintains the effectiveness of our attack."（p.6）。流式返回 + 单 token 输出，使"返回顺序"本身成为可观测量（§III-A 的输出策略，p.4）。

### 2.3 逐 prompt 重建：换靶、清场、挂钩

token 提取循环之外是 prompt 级策略（§IV-B，pp.6–7）：

- **换靶判据**：靶被逐出后要换下一条。攻击者看不到逐出事件，但可以重发一个必然未命中的旧候选、两侧夹 dummy；若它的次序错位一格，说明它已被冲掉——它比目标后进内存，"the eviction of a candidate prompt indicates that the target prompt has also been evicted based on LRU"（p.6）。
- **清场**：为了让每次重建从干净状态开始，主动把显存打爆：发"互不相同"的 dummy 请求（每个以唯一 token 开头接长序列，保证零共享，输出长度拉满），把 KV 池挤到触发 SGLang 的批量清空；再等一段时间 t，让 N 个用户按频率 f 攒出 "N × f × t" 条新 prompt，勾住其中一条开攻（§IV-B.2，p.7）。

### 2.4 三个场景：先验知识从无到有

- **场景 1（全 prompt 复原）**：无背景知识，对应"陌生用户各自提问"的常态，作为整体有效性基线（§V-A，p.7）。
- **场景 2（输入复原）**：攻击者知道用户所用模板（"users of the same application often employing identical prompt templates"），只需反推填入其中的个人信息；覆盖 cloze-style 与 prefix-style（role-based、instruction-based）两类模板，用提示工程引导本地模型生成候选（§V-B，pp.7–8）。
- **场景 3（模板复原）**：输入已知、模板未知——反向窃取提示工程服务本身的资产；攻击者还能利用模型输出辅助反推（§V-C，p.8）。

## 三、评估：成功率、成本与三个决定因素

### 3.1 固定装置

SGLang 默认配置、共享相关特性全开；temperature 0、无 beam search；max running batch size 16；每请求最大输出 128 token；用户频率 40 请求/3 小时（0.004 req/s，按 OpenAI ChatGPT-4 的使用标准；250 个用户折合 1 req/s）；硬件为单卡 A100 80G，目标模型 Llama2-13B（§VI-A，p.9）。数据集四份（Table I，p.8）：ultrachat 77,444 条（真实对话，用于场景 1 的有效性）、PromptBase 180 条（cloze）、awesome-chatgpt-prompts 153 条（role）、alpaca-gpt4 1,000 条（instruction）。

### 3.2 效果：三场景 × 三数据集

Table II（p.11）的整体结果。SR 为"完全 + 部分复原"的合并成功率，RR 为平均复原长度比（extracted length / total length，p.11），请求数只计候选请求（口径见 3.3）：

| 场景               | SR（cloze / role / instruction） | RR（同序）           | 请求数/条目标 | 请求数/token |
| ------------------ | -------------------------------- | -------------------- | ------------- | ------------ |
| 全 prompt 复原（场景 1） | 87% / 100% / 100%                | 64% / 87% / 93%      | 4843 / 1502 / 2183 | 212 / 126 / 172 |
| 输入复原（场景 2）       | 96% / 100% / 100%                | 98% / 99% / 99%      | 3115 / 1234 / 948  | 132 / 68 / 50 |
| 模板复原（场景 3）       | 94% / 100% / 100%                | 77% / 99% / 99%      | 4641 / 1687 / 1298 | 59 / 21 / 18 |

跨数据集平均后，论文的总结是："In the input reversing scenario, our attack reaches 99% of average success rate and an average reversal ratio of 99% across all cases. In the template reversing scenario, the average success rate is 98%, with an average reversal ratio of 91%. Even in the whole prompt reversing scenario, our attack can still reach a success rate of 95%, with a reversal ratio of 81%."（§VI-D，p.12）

两个值得标注的细节：cloze 数据集明显更难——PromptBase 的 prompt 含不规则结构、多余符号、emoji 与拼写错误（p.11）；而部分复原本身就有杀伤——理财顾问模板的案例里只复原出 [Monthly Income]="$4,000"、[Regular Expenses]="Housing $1,200, Utilities $300, Food $500 ..."，论文说 "it still poses a severe secret leakage"（p.11）。

### 3.3 成本：九成 token 十次以内就中

图 13 的分布结论是 "over 90% of tokens are extracted with fewer than 10 candidate requests"，但平均请求数仍"从几十到几百"——有些 token 天生难猜（"some tokens are inherently irretrievable"，p.11）。论文给了几个极端与典型（pp.10–11）：

- 同一句 "How did the division of the Frankish Kingdom..."，token "Frank" 无背景知识时要 4,093 个请求，有相关知识时 24 个（p.10）。
- "CONTEXT / TASK" 式少见格式的 prompt，个别 token 平均要 2,145 个请求（p.11）。
- 模板场景里，"istanbulu cok seviyom..." 的首 token "ist" 猜了 1,200+ 请求仍未命中；但把输入交给攻击者后，反推整条模板只需 129 个请求（p.11）。
- 有输出辅助时，"Spongebob's Magic Conch Shell" 中的 "Spon" 只要 34 个请求（p.11）。

成本口径要注意两点：论文只统计**候选请求**，dummy 请求被排除在外——"Since the size of dummy requests is fixed that determined by the adversary, we solely count the number of candidate requests."（p.10）；并且成本实验把候选批大小设为 1（每个候选单独一条请求），是最保守的配置（p.10）。

### 3.4 决定成败的三个因素

§VI-B 用三组控制实验拆解攻击空间（p.10；以下括号内为读图值，正文未给数值）：

- **并发用户数**：不超过 200 用户时提取长度中位数稳定在约 155 tokens，之后随用户数回落，1000 用户时降至约 55——"when the number of users exceeds 200, the average extracted prompt length significantly decreases"（Fig.10）。
- **显存容量**：40 GB→80 GB，中位数约 93 → 约 150；"capacities below 70GB start to affect P ROMPTPEEK's effectiveness"（Fig.11）。
- **攻击参数**：候选批大小存在最优点——"the effectiveness first goes up then goes down ... and peaks at a batch size of 40"；dummy 批越大效果越差（dummy 批 40 的峰值约 160，dummy 批 160 的峰值约 125；Fig.12）。并发实验的默认攻击参数是 dummy 批 20 + 候选批 50（p.9）。

### 3.5 换 GQA：成功率升到 100%

换用 Llama3-8B-GQA（默认 8 个 KV head）在 role 数据集上复测：三个场景各 153 条全部完全复原，SR 与 RR 均为 100%（Table III，p.12）。论文的解释是 GQA 压缩了 KV 尺寸，"allowing the attacker to send more requests before the target prompt is flushed"（p.12）。精读员注：GQA 行的每 token 请求数反而更高（306 / 148 / 35，对 Llama2-13B 的 126 / 68 / 21），论文没有解释这一成本上升。

## 四、边界与讨论

论文自己划的边界（§VIII，p.13）：

- **公有云未验证**：实验全部在本地完成；云上还有 colocation 问题——请求可能到不了存有目标 KV 的那块 GPU，共享就无从发生，"The colocation problem is complex and represents a non-trivial research question"，留作 future work；旁证是已有框架（阿里的 RTP-LLM）记录过云上自动定位 KV 缓存以促成共享（p.13）。
- **调度策略限定 LPM**：FCFS/random 不支持 KV 共享、甚至性能退化到无缓存，不在研究范围；论文强调自己给出的攻击条件可外推到未来的缓存感知调度（p.13）。
- **模型形态单一**：主实验是多用户共享同一模型；附录 A 指出跨微调变体（S-LoRA/Punica 式基座共享）等形态也可能有风险，未展开（pp.14–15）。

论文给防御方的输出是三条"成立条件"与对应教训（§VII，pp.12–13）：**条件 1** 缓存足够持久（"The KV cache persists as long as there is sufficient GPU memory."——max_tokens 这类客户端参数还能被用来刻意拉长目标缓存的寿命）；**条件 2** 共享对客户端可观测、至少泄漏 1 bit（"The sharing of KV cache is observable by the client through a side channel that leaks at least 1-bit information."）；**条件 3** 客户端拥有充分的请求参数控制与不受限的发送能力（vLLM 的 "pos" 参数即为一例）。可选的缓解包括：给 LPM 引入随机性、要求至少多匹配 M 个 token（M=2、3、4）才优先，以及用罕见 token 混淆 prompt（pp.12–13）。

伦理侧，论文称已向框架方（SGLang 为主要目标）与厂商（OpenAI、ByteDance、Anthropic）负责任披露，收到 ByteDance 的确认回复，并与 SGLang 讨论反制更新（§I，p.2）。

## 论文结构索引

| 论文位置 | 内容 |
| --- | --- |
| §I / pp.1–2 | 动机、贡献清单、伦理披露；"first investigation" 定位 |
| §II / pp.2–3 | 背景：推理两阶段、KV cache、共享机制（vLLM blocks / SGLang radix tree + LPM + LRU） |
| §III / pp.3–5 | 系统模型（五类策略 P_S/P_B/P_KV/P_O/P_E）、威胁模型、攻击总览 |
| §IV / pp.5–7 | PromptPeek：候选生成、token 选择（dummy 夹心）、重建与清场优化 |
| §V / pp.7–8 | 三个攻击场景（全 prompt / 输入 / 模板） |
| §VI / pp.8–12 | 评估：装置与数据集（Table I）、三因素（Fig.10–12）、成本（Table II、Fig.13）、GQA（Table III）、小结 |
| §VII / pp.12–13 | 三个攻击条件与三条教训 |
| §VIII / p.13 | 讨论：公有云 colocation、定向攻击、其他调度策略与缓解方向 |
| §IX–X / pp.13–14 | 相关工作与结论 |
| 附录 A–B / pp.14–15 | 其他多租户模型形态（S-LoRA/Punica）；其他共享资源（GPTCache、Redis 连接池事故） |

## 与系列其他篇目

同一份被多租户共享的 KV cache，三篇论文从三个位置取证：[Shadow in the Cache](shadow-in-the-cache.md)（NDSS '26）把威胁模型推到数据侧——攻击者直接拿到 KV 张量、逐 token 复原用户输入，并给出 KV-Cloak 防御；[Auditing Prompt Caching](auditing-prompt-caching.md)（ICML '25）站在纯黑盒一侧，只用商业 API 的响应时间线审计"跨用户共享是否正在发生"；本篇 PromptPeek 两边都不碰——不读 KV 数据、也不测绝对延迟，只用 LPM 调度造成的**返回顺序**这一个结构化行为信号，加上反复提交的自造请求，逐 token 复原他人 prompt。三篇合起来回答同一问题的三种问法：缓存的"行为"泄露什么、缓存的"数据"泄露什么、缓存的"服务"暴露什么；而 PromptPeek 归纳的三条攻击条件（缓存持久性、共享可观测性、客户端控制力）恰可作为判断任意部署落在哪一档的清单。横切总纲见 [KV Cache 安全总纲](kv-security-overview.md)。

## 诚实声明

- **实验性质**：全部结果均为论文在单台 A100 80G + SGLang（附加配置）上的自测，非公有云生产环境；论文自认云上 colocation 可行性未验证（p.13）。
- **成功率口径**："平均成功率 99% / 98% / 95%" 是"完全 + 部分复原"合并口径（fully or partially，p.2），且为三数据集算术平均；完全复原比例与逐数据集数字见 Table II / Table III（pp.11–12），场景 1 的 cloze 子集 SR 只有 87%。跨数据集重算为 95.9% / 98.9% / 98.2%，论文报 95% / 99% / 98%，似为下取整。
- **论文内部一处数字不一致**：Table II 的 cloze-模板行 Succ./Part./Fail = 102/78/10，合计 190，与 Table I 的 PromptBase 180 条对不上；SR 94% 按 190 为底自洽。本文引用该行时保留原数字（p.11）。
- **成本口径**：攻击请求数只统计候选请求、不含 dummy 请求；成本实验用候选批 = 1 的最保守配置（p.10）。真实攻击总流量高于表中数字。
- **读图值**：Fig.10–12 的中位数与峰值（约 155 → 约 55；约 93 → 约 150；约 160 对约 125）为精读员目测，正文未给数值。
- **未核实项**：GQA 行每 token 请求数高于 Llama2-13B 的原因论文未解释；SGLang/vLLM 版本号论文未给出；论文未提供公开代码仓库。
- **场景先验强度**：场景 2 / 3 假设攻击者掌握模板（或输入与输出），属较强先验；无先验的场景 1 效果对数据集质量高度敏感（cloze 明显更难，p.11）。

## 参考资料

- Guanlong Wu, Zheng Zhang, Yao Zhang, Weili Wang, Jianyu Niu, Ye Wu, Yinqian Zhang. _I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving_. NDSS Symposium 2025, pp. 1–15. DOI: [10.14722/ndss.2025.241772](https://dx.doi.org/10.14722/ndss.2025.241772)。[官方论文 PDF](https://www.ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf)（本文页码所指）；[NDSS 论文页](https://www.ndss-symposium.org/ndss-paper/i-know-what-you-asked-prompt-leakage-via-kv-cache-sharing-in-multi-tenant-llm-serving/)
- 本仓存本：[references/papers/i-know-what-you-asked-ndss25.pdf](references/papers/i-know-what-you-asked-ndss25.pdf)（官方论文 PDF，15 页）
- 代码：论文未提供公开仓库
- 精读报告（逐条页码核对与疑点清单）：[references/i-know-what-you-asked-report.md](references/i-know-what-you-asked-report.md)
