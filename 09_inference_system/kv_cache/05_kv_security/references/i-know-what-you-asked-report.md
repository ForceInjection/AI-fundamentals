# PromptPeek（NDSS '25）精读报告 ——《I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving》

> 来源：论文正文核对自官方 NDSS PDF（15 页，letter 版式；本仓存本为 `references/papers/i-know-what-you-asked-ndss25.pdf`，原始下载于 https://www.ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf）。页码约定：「p.X」一律指官方论文印刷页 1–15（p.1 为标题页无页码，pp.2–15 页脚有页码，PDF 页号与印刷页号一致，已逐页核对；正文含结论 pp.1–14，参考文献与附录 A–B 同占 pp.14–15）。核对方式：全文 pdftotext 抽取逐条对照 + 表格与图 300–400 dpi 渲染逐标签核对；推断/点评标「精读员注」，未核实写「未核实」。

## 一、速览

- 完整标题：I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving（p.1，无副标题）
- Venue：NDSS Symposium 2025，2025-02-24~28，San Diego, CA, USA（p.1 页脚）；ISBN 979-8-9894372-8-3；DOI 10.14722/ndss.2025.241772（p.1）
- 作者与机构：Guanlong Wu、Weili Wang、Jianyu Niu、Yinqian Zhang（南方科技大学 SUSTech；Yinqian Zhang 通讯）＋ Zheng Zhang、Yao Zhang、Ye Wu（ByteDance Inc.）（p.1）
- 攻击名：PromptPeek——论文以小型大写（small caps）排印，pdftotext 抽取为 "P ROMPT P EEK"，渲染核对为「全尺寸首字母 + 小型大写后续」的 PROMPTPEEK 样式（p.1、正文各处标题）
- 一句话贡献：首篇系统论证多租户 LLM serving 的 KV cache 共享构成跨租户侧信道；提出 PromptPeek，用「请求是否命中他人已缓存前缀」这 1 bit 信号（从 LPM 调度造成的返回顺序中读出）逐 token 复原他人 prompt；A100 80G + Llama2-13B 上三场景平均成功率 95%–99%（p.2），GQA 模型 100%（p.12）
- 开源/工件：论文未提供代码仓库（全文无 artifact 表述，仅引用 vLLM/SGLang 等第三方链接）→ 无

## 二、问题与动机

（以下均为论文自述；关键表述给英文短引）

- **KV cache 是显存大头，共享是主流对策。** "It consumes roughly 1MB per token [13], and can easily grow larger than the model weights [12]"（p.2）。共享规则：KV "can be reused by another request, as long as all preceding tokens match"（§II-B，p.3）——从首 token 起前缀一致是复用前提。
- **两个框架的共享实现**（p.3、p.4）：vLLM 用带元数据的 KV Cache Block（前缀 token 哈希、最近访问时间、引用计数），"These caches are retained in memory for as long as possible"，显存到限时逐出最旧块；SGLang 用基数树（radix tree），并配三件套——调度 LPM、连续批、LRU 逐出（到容量上限时 "all KV cache entries except those in the current batch are evicted simultaneously"，p.4）。实验选 SGLang 的理由："vLLM first introduced the concept of KV cache sharing" 但 "still in development"，SGLang "offers a more comprehensive discussion on KV cache sharing, scheduling, and eviction policy"（p.4、p.9）。
- **核心洞察（论文的立论）。** "the KV cache sharing may inadvertently create side channel information, which can be leveraged by the adversary to carefully craft requests sent to the LLM server to determine if its requests match the other users', thereby recovering other users' prompts."（p.1）"The intuition behind P ROMPTPEEK is that the state of one tenant's request can be influenced by others, inadvertently creating cross-tenant side channels."（p.5）
- **侧信道的载体是 LPM 造成的服务顺序。** LPM "prioritizes the longest matching token sequences... those with a longer matched token sequence—even by one token—will be prioritized and served first"（p.5）；论文把它转述为"命中共享即较快返回"，攻击者观测的是**一批自造请求的返回次序**（结构化的顺序信号，非对绝对延迟的测量；流式输出把首 token 立即回传是前提，p.4）。
- **威胁模型（§III-B，p.4）。** 攻击者 "capabilities are the same as those of an ordinary user, with no direct control over the inference server"；知道服务端默认机制（调度/逐出等，侧信道研究的标准假设）；LLM 层面背景知识 "extends only to the tokenizer, which is always accessible to the adversary"；只经公开客户端 API 交互（如 max_tokens）。目标："to reconstruct the prompts sent by other users using the side channel information."（p.4）
- **系统模型（§III-A，p.3–4）。** 三实体（users / scheduler / GPU worker）+ 五类策略：调度 P_S（LPM，"the only scheduling policy designed to enhance KV cache sharing efficiency"）、批处理 P_B（连续批）、KV 策略 P_KV（基数树）、输出 P_O（流式，对齐 OpenAI）、逐出 P_E（LRU）。容量关系 T_max = M_KV / m_ti（m_ti 为单 token KV 尺寸）。
- **与既往 prompt 窃取工作的差别**：Sha 等需要输出访问且只做近似复原；"P ROMPTPEEK accurately reconstructs prompts"（p.1），且场景 2/3 不依赖输出——"our work allows an attacker to reverse-engineer a prompt without access to the output, and achieve exact replication of the original prompt"（p.13）。
- **伦理披露（§I，p.2）**："We responsibly disclosed our findings to both the framework developers (SGLang, our primary target) and service providers, including OpenAI, ByteDance, and Anthropic. So far, we have received a response from ByteDance acknowledging the new side-channel vector identified in our work. Also, we are actively engaging with SGLang to discuss updates to counter the attack."

## 三、核心机制

### 3.1 攻击原语：token extraction（§IV-A，pp.5–6）

- **候选生成**：本地 LLM "generate the top k most likely tokens as candidates, based on the already reconstructed fragment"（例：已还原 "Imagine you are" → 候选 "an" / "a" / "the"）。本地模型只需与目标服务同 tokenizer；论文为简化假设用同款 LLM（p.5）。
- **dummy token**：本地模型挑 "one least likely token"（如 "%"）拼成 dummy 请求 "Imagine you are %"；其作用是让顺序图案可分辨。dummy 极小概率命中时立即换靶（p.5）。
- **token 选择**：每轮发 三批 = pre-dummy → 候选 → post-dummy。pre-dummy 数量须**超过服务端 max batch size**（先把运行批填满、逼候选排队受 LPM 重排）；post-dummy 同样超过最大批，"amplify the order effect"（p.6）。所有请求输出长度经客户端 API 设为 1 token，以压低显存占用（p.6）。
- **判读图案**（p.6）：命中 → "pre-candidate dummy requests, the matched request, post-candidate dummy requests, and then the remaining candidate requests"；未命中 → "pre-candidate dummy requests, post-candidate dummy requests, followed by all candidate requests"。机理：dummy 自身跑过一轮后其 "%" 也进了缓存，dummy 的匹配长度比未命中候选长一位；命中候选与 dummy 等长，靠 "Requests with equal match lengths retain their original order in the queue." 恰好夹在两批 dummy 之间（p.6、Fig.6）。

### 3.2 prompt 级流程：换靶、清场、挂钩（§IV-B，pp.6–7）

- **换靶判据**：重发一个必然未命中的旧候选、两侧夹 dummy；次序错位一格说明它已被逐出；它晚于目标入存，故 "the eviction of a candidate prompt indicates that the target prompt has also been evicted based on LRU"（p.6；论文声明不考虑目标因活跃使用而更晚被逐出的 corner case，p.7）。
- **清场优化**：发"互不相同"的 dummy 请求（唯一首 token + 长序列，保证零共享；输出长度拉满）挤爆显存，触发 SGLang 的批量清空（"To trigger the eviction, we send non-identical dummy requests to saturate the GPU memory"，p.7）；等待 t 后，N 个用户按频率 f 已攒出 N × f × t 条 prompt，任选一条开攻（p.7）。
- **循环**：token 提取循环 → 一条 prompt 重建完成 → 换下一条（§III-C，p.5）。

### 3.3 三个场景（§V，pp.7–8）

- 场景 1 全 prompt 复原：无先验（基线）；场景 2 输入复原：知道模板（cloze / prefix 两类，提示工程引导本地模型生成候选）；场景 3 模板复原：知道输入与输出、反推模板（提示工程服务的核心资产）。

### 3.4 论文归纳的三个攻击条件与三条教训（§VII，pp.12–13）

- 条件 1 "The KV cache persists as long as there is sufficient GPU memory."（另有客户端参数 trick，如 max_tokens 拉长目标缓存寿命）；条件 2 "The sharing of KV cache is observable by the client through a side channel that leaks at least 1-bit information."；条件 3 "Clients are granted with extensive control over request parameters and unrestricted dispatching capabilities."（例：vLLM 的 "pos" 参数，p.13）。
- 缓解方向（未实现）：LPM 加随机性 / 要求至少多匹配 M 个 token（M=2、3、4）才优先（p.13）；rare tokens 混淆 prompt（p.12）。

## 四、关键数字

> 全部为论文真机实测（A100 80G 上模拟多用户负载）；除标注「读图值」「重算」外均直接取自正文/表格。攻击成本只计候选请求（p.10 口径）。「成功率 SR」为"完全 + 部分复原"合并口径（p.2）。

引用了本报告即等于引用附录：4.1 总表（本文与文章全部数字）、4.2 Table II 原表、4.3 Table III 原表。

### 4.1 数字总表

| 指标                        | 数值                                                                                                           | 对照基线 / 条件                                               | 出处               |
| --------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------ |
| 平均成功率·场景 2 输入复原  | 99%（SR，fully or partially）                                                                                  | Llama2-13B / A100 80G，三数据集均值                           | p.2、p.12          |
| 平均成功率·场景 3 模板复原  | 98%                                                                                                            | 同上                                                          | p.2、p.12          |
| 平均成功率·场景 1 全 prompt | 95%                                                                                                            | 同上                                                          | p.2、p.12          |
| 平均 RR（复原长度比）       | 99% / 91% / 81%（输入 / 模板 / 全 prompt）                                                                     | 同上                                                          | p.12               |
| 平均成功率重算              | 95.9% / 98.9% / 98.2%（按 Table II 三数据集 SR 算术平均）                                                      | 精读员重算；论文报 95 / 99 / 98，似下取整                     | p.11–12（重算）    |
| 最低成本案例                | 60 个请求复原 4 个占位符                                                                                       | 场景 2（cloze），占位符 = male / 35 / 90kg / 5 feet 9 inches  | §I p.2、§VI-C p.11 |
| 部分复原的杀伤案例          | [Monthly Income]=$4,000、[Regular Expenses]=Housing $1,200 / Utilities $300 / Food $500 / …                    | 理财顾问模板的部分复原，"still poses a severe secret leakage" | p.11               |
| 单 token 请求数分布         | >90% 的 token <10 个候选请求；均值跨"几十到几百"                                                               | Fig.13，全场景                                                | p.11               |
| 极端 token 案例 1           | 4,093（无背景） vs 24（有背景）                                                                                | "Frank"（ultrachat 句）                                       | p.10               |
| 极端 token 案例 2           | 平均 2,145 请求 / token                                                                                        | "CONTEXT / TASK" 少见格式（cloze）                            | p.11               |
| 极端 token 案例 3           | >1,200 未命中 vs 129                                                                                           | "ist"（role） vs 有输入时整条模板                             | p.11               |
| 极端 token 案例 4           | 34                                                                                                             | "Spon"，有输出辅助（role）                                    | p.11               |
| 并发影响（Fig.10）          | 中位数 ≈155 稳定（≤200 用户）→ 350 用户 ≈136、500 ≈107、750 ≈93、1000 ≈55                                      | 读图值；正文：>200 用户 "significantly decreases"             | p.10               |
| 显存影响（Fig.11）          | 中位数 ≈93（40 GB）→ ≈115（50）→ ≈130（60）→ ≈145（70）→ ≈150（80 GB）                                         | 读图值；正文："capacities below 70GB start to affect"         | p.10               |
| 攻击参数影响（Fig.12）      | 峰值在候选批 40；dummy 批 40 曲线峰值 ≈160 vs dummy 批 160 峰值 ≈125（batch 200 端 ≈104 / ≈83）                | 读图值；正文：峰值在 40、更大的 dummy 批更差                  | p.9–10             |
| KV 体量                     | ≈1 MB/token，可超过模型权重                                                                                    | 背景引用（[13][12]）                                          | p.2                |
| 实验装置                    | A100 80G；Llama2-13B；temperature 0；无 beam search；max running batch 16；max output 128；SGLang 共享特性全开 | —                                                             | p.9                |
| 用户频率                    | 40 请求/3 小时 = 0.004 req/s；250 用户 ≈ 1 req/s                                                               | 按 OpenAI ChatGPT-4 使用标准                                  | p.9                |
| 有效性实验默认攻击参数      | dummy 批 20 + 候选批 50；可用显存满 80 GB                                                                      | 并发实验                                                      | p.9                |
| 成本实验参数                | 候选批 = 1；单用户；逐条打全数据集                                                                             | 最保守成本配置                                                | p.10               |
| 数据集规模                  | ultrachat 77,444；PromptBase 180（18 类 × 10）；awesome-chatgpt-prompts 153；alpaca-gpt4 1,000                 | Table I                                                       | p.8                |
| GQA 结果                    | Llama3-8B-GQA（8 个 KV head）：153/153 全部完全复原，SR = RR = 100%                                            | role 数据集，Table III                                        | p.12               |
| GQA 成本对照                | 请求/token 306 / 148 / 35 vs Llama2-13B role 行 126 / 68 / 21                                                  | 论文未解释成本上升（精读员注）                                | p.11–12            |
| 缓解建议                    | LPM 至少多匹配 M=2 / 3 / 4 token 才优先；rare tokens 混淆                                                      | 未实现、未实测，"leave ... to future work"                    | pp.12–13           |
| 论文内部不一致              | Table II cloze-模板行 102 + 78 + 10 = 190 ≠ PromptBase 180                                                     | 400 dpi 渲染逐格复核确认 102/78/10                            | p.11               |

### 4.2 Table II 原表（p.11；SR=(Succ+Part)/total，Req. 只计候选请求）

| 数据集      | 场景      | Succ./Part./Fail | SR   | RR  | Req./prompt | Req./token |
| ----------- | --------- | ---------------- | ---- | --- | ----------- | ---------- |
| cloze       | 全 prompt | 56/102/22        | 87%  | 64% | 4843        | 212        |
| role        | 全 prompt | 120/33/0         | 100% | 87% | 1502        | 126        |
| instruction | 全 prompt | 899/101/0        | 100% | 93% | 2183        | 172        |
| cloze       | 输入      | 170/4/6          | 96%  | 98% | 3115        | 132        |
| role        | 输入      | 151/2/0          | 100% | 99% | 1234        | 68         |
| instruction | 输入      | 997/3/0          | 100% | 99% | 948         | 50         |
| cloze       | 模板      | 102/78/10        | 94%  | 77% | 4641        | 59         |
| role        | 模板      | 150/3/0          | 100% | 99% | 1687        | 21         |
| instruction | 模板      | 995/5/0          | 100% | 99% | 1298        | 18         |

### 4.3 Table III 原表（p.12，Llama3-8B-GQA，role 数据集）

| 场景      | Succ./Part./Fail | SR   | RR   | Req./prompt | Req./token |
| --------- | ---------------- | ---- | ---- | ----------- | ---------- |
| 全 prompt | 153/0/0          | 100% | 100% | 5505        | 306        |
| 输入      | 153/0/0          | 100% | 100% | 2670        | 148        |
| 模板      | 153/0/0          | 100% | 100% | 2675        | 35         |

## 五、局限与边界

**论文自认的局限：**

1. **公有云可行性未验证。** 实验全在本地；云上受 colocation 制约——"an input request might not reach the correct GPU in the cloud to access a previously stored KV cache, preventing KV cache sharing from happening"；"The colocation problem is complex and represents a non-trivial research question [45]"，明确留作 future work；旁证仅 RTP-LLM 文档记录云上 colocation 机制（p.13）。
2. **调度策略只覆盖 LPM。** "Other policies, such as FCFS or random, do not support KV cache sharing and may even degrade performance to no-cache [61], thus not considered in this study."（p.13）；论文自认攻击条件（§VII）可外推到未来缓存感知调度。
3. **模型形态单一。** 主实验是"多用户共享同一模型"；附录 A 列出另两种形态（不同用户不同微调模型、完全不同模型），只指出 S-LoRA/Punica 式基座共享也可能有风险，未展开（pp.14–15）。
4. **定向攻击需要背景知识。** 普通攻击者难以区分个体；知道目标（如同事）用特定模板时可行（p.13）。
5. **防御只给方向、未实现。** M-token LPM 与 rare tokens 都 "leave this exploration to future work"（p.13、p.12）。
6. **条件 3 的隐式前提**："unrestricted dispatching capabilities"——真实服务的限流/配额如何影响攻击，论文未讨论（精读员注归入此处）。

**精读员注（疑点与空白）：**

1. **Table II cloze-模板行计数对不上。** 102+78+10=190 > 数据集 180（Table I）；该行 SR 94% 按 190 为底自洽，说明不是抄错一格那么简单，可能是评测时多打入了样本或计数口径不同。渲染 400 dpi 复核确认三个数字为 102/78/10 → 论文内部不一致，引用保留原样。
2. **平均数的舍入口径未说明。** 按 Table II 逐数据集重算：SR 均值 95.9 / 98.9 / 98.2，RR 均值 81.3 / 98.7 / 91.7，论文报 95 / 99 / 98 与 81 / 99 / 91，疑为下取整——写文章引用时建议同时给表值。
3. **成本口径被低估的风险。** 攻击请求数只统计候选请求；每轮还有 pre/post 两批 dummy（都要超过 max batch size），dummy 流量被排除（p.10）。且有效性实验的候选批是 50、成本实验是 1，两类数字不可直接换算——"Req./prompt" 是最保守（批=1）口径，而攻击者若加大批，请求效率与显存压力同时变化（Fig.12）。
4. **GQA 的成本上升没有解释。** Table III 的成功率全部 100%，但每 token 请求数（306/148/35）全面高于 Table II 的 role 行（126/68/21）。论文只解释成功率为何更高（KV 压缩 → 攻击空间更大），未提成本为何更高。另：Table III 是否与 VI-C 同用候选批 = 1，论文未明说（写作时按"同协议"处理）→ 未核实。
5. **观测量是"返回顺序"，不是延迟测量。** 论文的判读基于一批请求的相对次序（结构化图案），不依赖对单请求延迟的绝对值测量；流式输出（P_O）与 LPM 是前提。本篇更准确的称呼是**顺序/排队行为侧信道**（系列行文统一为「调度行为 / 返回顺序」）。
6. **复现信息缺口。** SGLang/vLLM 版本号、§VI-A 的 "additional configurations" 具体内容未给出 → 复现性细节**未核实**。
7. **攻击者如何获知服务端 max batch size 未讨论。** 论文要求 dummy 批超过该值（p.6），但没说明攻击者从哪知道它（默认值可从代码/文档推断？）→ 精读员注。
8. **评测环境为单攻击者、单模型、单卡模拟负载。** 用户行为按 0.004 req/s 的 ChatGPT-4 标准生成，非生产分布；没有多攻击者并发、没有与真实网关/限流的交互。
9. **camera-ready 修订未核**：上述一切结论以落盘 PDF（2025-02-18 最后修改）为准。

## 六、写作素材

- **系列定位（三篇的三个取证位置）**：PromptPeek = 行为侧信道（不看 KV 数据、不测绝对延迟，用 LPM 调度下的返回顺序 + 自造请求逐 token 复原）；Shadow in the Cache = 数据侧（拿到 KV 张量即近明文）；Auditing Prompt Caching = 服务侧黑盒审计（真实商业 API 的响应时间线）。三问：缓存的"行为"泄露什么、"数据"泄露什么、"服务"暴露什么。
- **总纲可用的三条攻击条件**：持久性（KV 留得越久、攻击窗口越大，客户端参数还能主动拉长）／可观测性（至少 1 bit 的共享信号对客户端可见）／客户端控制力（参数与发送能力不受约束）。这三条同时是给部署方的 checklist。
- **反差素材（适合开头或小节引子）**：
  - 60 个请求读出 gender/age/weight/height 四个隐私字段（p.2、p.11）；
  - "平均成功率 99%" 对应的信号量是 1 bit/次（命中与否）；
  - 90%+ 的 token 十次以内猜中，但最难的一个 token 曾烧掉 4,093 个请求（p.10–11）；
  - 无背景知识 4,093 vs 有背景 24（"Frank"，p.10）；1,200+ 猜不出 "ist" vs 129 反推整条模板（p.11）——先验知识把成本压缩两个数量级。
- **机制解说可用的两个"图案"描述**：dummy 夹心（pre-dummy 灌满运行批 → 命中候选被 LPM 提到 post-dummy 之前）；等长保序（"Requests with equal match lengths retain their original order"，p.6）。
- **与姊妹篇的对读点**：本篇原始机制是**返回次序**的结构化信号（不测延迟值；LPM 重排 + 等长保序）——与 Auditing 的"响应时间 oracle"是两种不同口径，系列行文统一为"调度行为 / 返回顺序"。
- **与二手调研报告（references/reuse-security-report.md）的核对结论**：团队（SUSTech + ByteDance，Guanlong Wu 等）、NDSS 2025 正式录用、"首篇系统论证多租户 KV 共享侧信道、可恢复他人 prompt"的定位，均与 PDF 核实一致；"被 PrefixWall 等防御工作引为攻击基线"属外部事实，本次未核（未核实）。
- **与本仓已有内容的勾连候选**：`01_concepts/prefix_caching/prefix_caching.md`、`radix_attention.md`、`claude_prompt_caching.md`（共享机制与商业 prompt caching 的原理侧）；`01_concepts/scheduling/`（调度策略侧）；`02_systems/`（vLLM/SGLang 系统篇，若有）；`04_memory_tiering/` 系列讲"KV 放哪"的账，本篇讲"KV 共享"的险，可互文。
- **伦理与时代注脚**：论文附录 B 提到 2023-03-24 ChatGPT 下线事故（开源库 Redis 连接池复用导致用户看到他人会话标题）作为"共享资源风险"的旁证（pp.14–15），写"共享即风险"的主题段可用。
