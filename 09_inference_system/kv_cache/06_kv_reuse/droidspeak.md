# 借来的 KV 不能全借：NSDI'26 DroidSpeak 只重算约一成层，预填充最高快 3.1×

> **元信息**：论文 _DroidSpeak: KV Cache Sharing Across Fine-tuned Model Variants_，Yuhan Liu、Yuyang Huang、Jiayi Yao、Shaoting Feng、Zhuohan Gu、Kuntai Du、Hanchen Li、Yihua Cheng、Junchen Jiang（University of Chicago）+ Shan Lu、Madan Musuvathi、Esha Choukse（Microsoft），NSDI '26 正式论文集 pp. 319–338，2026-05-04~06，Renton, WA（本地 PDF 21 页，含 USENIX 封面 1 页；arXiv:2411.02820 沿用旧题 "DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving"，引用口径以本 PDF 为准）。论文未提供公开代码仓库（2026-10-13 本仓外核实）。精读方式：全文逐条抽取核对 + 关键图表 110–440 dpi 渲染复核（报告见 [references/droidspeak-report.md](references/droidspeak-report.md)），数字均标论文印刷页码，读图值另行标注。阅读日期 2026-10-13。

先摆出论文自己给的"理想账"：A100 上跑 Llama-3.1-8B-Instruct，40K token 的输入如果能把别人算好的 KV cache 直接拿来用，"reduce prefill latency from 4s to 0.08 seconds"——两个绝对值相差 50×（论文未直接给倍数，此为精读员换算，p.322）。机会到处都有：论文统计的一条 LangGraph 多智能体应用真实 trace 里，对话中平均 49% 的 token 属于前缀复用（p.321），多模型系统里同一段上下文被不同微调变体轮流读是常态。

但这笔账不能直接兑现。把发送方模型（sender）的 KV cache 原样交给接收方模型（receiver）继续 decode，质量塌方——HotpotQA 上多数模型对 "lose more than 50% of the accuracy points"（p.322，Fig.3）。NSDI '26 的 DroidSpeak 的答案是"挑着重算"：八个模型对、三个数据集上，只重算一小撮敏感层——跨全部模型对平均而言，被视为 critical 的层只占 11%（其余层的 KV 复用于接收方）——预填充延迟相比从头 full prefill 降 1.7–3.1×（平均 2.1×），在线服务吞吐最高 4×，质量压在配置的 5%（profiling 默认）到 1%（在线实验）容差内（pp.319、325–326、328）。

## 一、问题：同一个基座的微调变体，为什么不能直接互借 KV

### 1.1 场景：compound AI 系统里的重复 prefill

论文把目标负载分为三类（§2.2，p.321）：多智能体工作流（编码 agent 的产出要整段喂给测试 agent，对话历史作为前缀在模型间流转）、多模型/多 LoRA 服务（论文举例：ChatGPT API 基于同一基座的微调版本"about every two months"更新一次，新旧版本要处理同一段历史）、个性化助手（给每个用户微调一个模型，同一篇新闻被多个助手读）。这些系统都在做同一件事：为同一段前缀反复 prefill。

重复 prefill 的代价用 goodput 计量最直观。论文的小实验——Llama-3-8B 单卡 A100、5K 与 20K 合成输入、3 秒延迟 SLO——"increasing the input size by only 4× can reduce goodput by as much as 32×"（p.322，Fig.2b）：TTFT 超线性增长，decode 必须等 prefill，长输入把 prefill 变成端到端瓶颈。而现有分布式推理系统（vLLM Production Stack、Dynamo、AIBrix）的跨节点 KV 共享只服务同款模型——"how to enable one model to reuse the prefix KV caches of a different model remains an open question"（摘要，p.319）。

DroidSpeak 画的前提一句话："the two LLMs have different weights but the same architecture"（p.320）——即同一基座的微调变体之间，论文自称是 "the first distributed LLM inference system that enables KV cache reuse across distributed nodes running inference of different LLMs"（摘要，p.319）。

### 1.2 直接复用为什么不行：三条实证洞察

权重不同，同一段输入在两边各层算出的激活就是错位的；KV cache 又是逐层喂给 attention 的中间状态，错位会一路传导。论文为此建了 8 对模型的 benchmark（入选标准：同基座、数据集与 receiver 微调任务相关、receiver 质量高于 sender，§3.1，p.322），得到三条洞察（§3.2，pp.322–323）：

- **Insight 1：整个 KV 直接复用，精度损失巨大。** "Reusing the whole KV cache between models leads to a huge loss in accuracy."（p.322）Fig.3 里，receiver 用 sender 的 KV 后质量仍好于 sender 自己，但远逊于 receiver 原本的水平；HotpotQA 多数模型对掉一半以上精度。
- **Insight 2：只有一小撮层对这个偏差敏感。** "Only a small subset of layers are sensitive to KV cache reuse in terms of accuracy."（p.322）Fig.4 逐层测试（只复用一层、其余重算），把 F1 掉超 10% 的层标红、称为 critical layers；跨全部模型对平均，"we identify 11% of layers to be critical"（p.323）。
- **Insight 3：关键层的位置跨输入稳定。** "The variation in KV cache patterns across inputs is only notable for critical layers."（p.323）非关键层在不同输入上的质量方差可以忽略——所以"关键层名单"可以离线 profile 一次、在线复用。

## 二、机制：连续关键层组 + 传输与重算流水

### 2.1 为什么重算的是"连续层组"

要理解重算的入口，先要认识 E cache。每层 transformer 的输入是一个 embedding 张量 E（attention 的 Q/K/V 都由它算出），复用的对象是 K、V，重算的起点却是 E：要重算第 l 层的 KV，就要有该层在整个上下文上的输入 E——接收方必须从这份 E 出发，用自己的权重重新算出 K、V。接收方跳过了 prefill，手里没有这份 E——所以 DroidSpeak 让发送方把"复用切到重算"那一层的 E cache 也存下来传过去，这个层叫 transition layer（Fig.6，p.324）。E cache 不便宜："reaching up to twice the size of the KV cache for the Mistral-7B or Llama-3-8B model families, and up to four times larger for Llama-3.1-70B"（GQA 把 KV 压小了，p.324）。

这带来两个挑战（§4.1，p.324）：

- **效率账**：关键层分散时会有多个 transition point，每个点都要存、传一份 E cache；
- **精度账**：每加载一次发送方的 E cache，就注入一次"发送方初始化 vs 接收方应有的 E"的偏差，且随 forward 向后传播。Fig.7a 的例子——只重算分散的关键层 16–18、20、25–27 要在三处加载 E——偏差叠加后输出误差相当可观；改成重算连续段 16–27（把夹在中间的层一并重算）就只剩一次加载。

所以论文的方案是：为每个模型对离线挑一个**连续层组**，"we refer to the critical layer group as the re-computation configuration"（p.324）。

### 2.2 离线 profiling：从 Pareto 前沿上挑一个点

配置怎么挑（§4.2，pp.324–325）：在训练集上枚举连续层组、逐组测质量，得到 F1 分数量级下"重算层数 vs 质量"的 Pareto 前沿（Fig.7b）；然后在用户给定的容差内选重算层数最少的一档。论文举例："a configuration with 11 re-computed layers is a good example on the Pareto frontier, since the F1 score drop is within 5% of the original accuracy of the receiver model"（p.325）——5% 是用户可以自行调整的超参。训练集直接取微调数据集（推理请求的分布理应贴着微调域），论文实验里统一用 HotpotQA 的 50 条上下文 profile（p.326）。

profiling 不便宜但只做一次：复杂度 O(l²)，Llama-3-8B 32 层逐层扫一轮在 A100 上要三小时；按 2 层粒度分组可以把时间降到约 1/3，论文评估即用 2 层分组（p.325）。profile 结果的跨数据集泛化性在 §5.4 单独验证（见下）。

### 2.3 在线：两件事——选配置、把加载藏进重算

**配置自适应**（附录 C，p.338）：DroidSpeak 看 vLLM 引擎里等待队列的长短判断负载——队列有积压就切"重算层数最少"的激进档，空载则按延迟 SLO 估一版最保守（重算最多、精度最高）的档。

**传输与重算流水**（§4.3，pp.325–326）：接收方收到 transition layer 的 E cache 就能开算，不必等其余层的 KV 传完。Fig.9 的玩具时间线（两个 10 层模型、每层传输或重算各 1 个时间单位）：先收全再算的基线 TTFT 47、只收复用层再算 30、流水化收到 E 即开算 17——"approximately a 2× improvement over the baseline in (b)"（p.325）。E cache 里单层的那一份被单独保留，"only the E cache for a single layer – the transition layer – needs to be kept"（p.325），成本示例：Llama-3.1-8B 全层 KV 按 10K token 算占 1.2 GB 显存，单层 E 只要 0.08 GB，约 6%（p.325）。

### 2.4 实现

约 3K 行 Python，PyTorch v2.0、CUDA 12.0、LMCache 0.1.4（§4.4，p.326）。三个接口：`store(Cache, context, LLM)`、`fetch(context, LLM, layer_id) -> Cache`、`partial_prefill(recompute_config, context) -> text`；fetch_kv（取复用层 KV）与 fetch_e（取 transition 层 E）都用 `torch.distributed` 跨节点取，且 "All transmission will be placed on a CUDA Stream different from PyTorch's default computation stream"（p.326）——这正是流水的落点。

## 三、评估：1.7–3.1× 预填充、2–4× 吞吐的口径

### 3.1 装置

8 对模型，覆盖 Mistral-7B/24B、Llama-3.1-8B、Llama-3-8B、Phi-3.5-mini、Llama-3-70B/3.1-70B 的微调族（其中 70B 用 4-bit AWQ 量化塞进单卡 A100；receiver 未必直接微调自 sender，含两对同基座变体，Table 1 p.337、p.326）；6 个数据集、三类任务（QA：HotpotQA / 2wikimQA / multifieldQA_en，摘要：multi_news，代码：lcc / repobench-p；每对模型挑其中的 3 个，Table 2 p.337 给出上下文长度中位数 7.5K–14.3K tokens）；硬件是两台 A100 虚拟机（每台 8×80GB），"connected with 200 Gbps NVIDIA Mellanox HDR InfiniBand links"（p.326）。基线四类（p.327）：full prefill（vLLM 从头算，质量天花板）、full KV reuse（全复用）、CacheBlend（作者自行扩展为跨模型版本）、更小的模型。

### 3.2 质量-延迟：3.1× 与 16% 的对照口径要说清

主结果在 Fig.10（p.327）："Across 8 pairs of models on three datasets, DroidSpeak achieves 1.7–3.1× reduction in prefill delay over the full prefill method, without compromising generation quality."（§5.2，p.328）——这里的 prefill latency 含 GPU 计算与跨节点取 KV/E 的加载延迟（§5.1 口径，p.327）；"without compromising" 对应 profile 的 5% 容差档。三笔对照互为坐标：

- **vs full KV reuse**：全复用最快（根本不 prefill）但质量崩（Insight 1）；DroidSpeak 用略高的延迟换回质量；
- **vs CacheBlend**："DroidSpeak has 5–33% (average 16%) higher quality than CacheBlend at a similar prefill latency"（p.328）——CacheBlend 按首层差异挑 token 重算，论文认为它 "fails to spot the critical layers since it selects the tokens to re-compute based on the first layer"（p.328）；
- **vs full prefill**：1.7–3.1× 即此项，也是摘要 "about 3.1× faster prefill (time to first token)" 的来源（p.319）。

### 3.3 在线服务：吞吐最高 4×，且不止 TTFT

在线实验部署在 Kubernetes 上、跑 vLLM Production Stack 定制镜像，每模型 8 个副本（每个节点 4 个 sender + 4 个 receiver），Poisson 到达（§5.3，p.328）。Fig.11 里，full prefill 的 TTFT 曲线在低 QPS 处就出现排队拐点（"the queuing delay affects (knee in the curve) its TTFT at a much lower QPS than what DroidSpeak can support"，p.328）；论文的总结口径是 "DroidSpeak can support 2-4× higher throughput"（p.328），摘要与 §5 开场写作 "up to 4× improvement in throughput"（pp.319、326）。次要效应也在：TTFT 压下去后排队与调度变好，"brings down the TBT and E2E latency too"（p.328，Fig.11 中 TBT 一列 full prefill 早早冲到秒级，读图值约 1.3–2.6 s，DroidSpeak 则贴地）。

### 3.4 泛化性与组件账

- **profile 跨数据集泛化**（§5.4，p.329）：在 A 数据集 profile、用在 B 数据集上，Pareto 前沿几乎重合，"the maximum difference in the score is 4 points, with the average being 2 points"。
- **真实 agent 工作流**（p.329）：MetaGPT 双 agent（coder 用 evolcode、tester 用 tool-8b）、HumanEval 题目，"DroidSpeak significantly improves the TTFT by 2.7× and brings down the E2E delay"（Fig.12）。
- **MoE 模型**（p.329）：Mixtral-8x7B → -Instruct 同样有效——专家选择发生在 attention 之后的线性层，"DroidSpeak's KV cache sharing remains effective for MoE architectures"。
- **数学任务**（附录 B.1，p.337）：MAmmoTH2 上 GSM8K，Pareto 形态与其他任务一致。
- **对比更小模型**（附录 B.2，p.338）：Llama-3.1-8B 替 70B 换来约 4× 预填充缩减但 F1 掉一半，TinyLlama-1.1B 对 Ultrachat-8B 是 4.7× 预填充缩减——DroidSpeak 宣称在质量-延迟权衡上更优。
- **网络带宽**（§5.5，p.329，Fig.15，读图值）：4 Gbps 低速档下 DroidSpeak 的 TTFT 约 1.7×10² s，而两条未流水化的传输基线直接冲出图外（≥3×10² s）；64 Gbps 时各方案收敛到 1.2–1.4×10² s，而"纯重算"基线不随带宽变化、固定约 2.2×10² s。论文自己承认："as networking speed increases, the advantage of loading KV caches in parallel with recomputation decreases"（p.329）。
- **profiling 开销**（§5.6，p.329）：3.6 小时（逐层）→ 1 小时（2 层组）→ 0.375 小时（3 层组），质量不降；附录 B.3 在另一模型对上复现。

## 四、边界：论文自认与精读员注

论文自己划的线（§6，p.330）有五条：**跨基座不行**——"DroidSpeak as-is only works for KV cache sharing across models originated from the same foundational model"；**只做了两两模型**——"Our current evaluation focuses on KV cache reuse between two models"，多模型场景要拆成模型对集合，跨模型共享层的最优化留给未来；**自适应只看负载**——现版本不感知网络带宽变化；**有数据漂移风险**——profiling 离线做一次，若真实流量分布漂移，配置可能失准，论文建议周期性重 profile；**不是所有模型对都灵**——"it may not work for all model pairs, even those sharing the same foundation model"。

精读员注补充几个论文没展开的点：

- **质量口径是数据集标准指标的容差**：F1 / Rouge-L / 代码编辑相似度，且"可接受"由 5%（profile 默认）与 1%（在线实验配置）两个容差超参定义；没有 perplexity、人工评测或更长对话的误差累积实验。"negligible quality loss"（摘要）应对照"容差内"理解。
- **3.1× 是预填充（TTFT）口径，对照 full prefill**；摘要与 Fig.1 题注的措辞更宽（"reduce the latency"、"brings down the computation latency"），端到端与吞吐另有口径（2–4×）。引用时不要把三个数字混成一句。
- **评估环境是两台 200 Gbps InfiniBand 节点**；§5.5 明说收益对带宽敏感，低带宽下换来的主要是"不被传输拖死"。此外 Fig.10 图例里系统名写作 "DroidSync"、Fig.20 写作 "CacheSync"——同一系统的遗留旧名，正文口径统一为 DroidSpeak。
- **§5.2 有一句主语混乱**："In contrast to full KV reuse, DroidSpeak has a longer prefill latency because it does not perform prefill at all."（p.328）——按上下文应为"full KV reuse 因不做 prefill 才是最快（但质量差）"，引用该段时按语义理解。
- **CacheBlend 基线是作者改造的跨模型版**（"we extend CacheBlend's algorithm"，p.327），不是原版实现；对比数字应带上这层限定。
- **无公开代码仓库**（USENIX 页与 arXiv 页均无链接，2026-10-13 核实）；实现基于 vLLM 与 LMCache，复现要从这两个底座起步。
- **缓存命中路径未细化**：store 发生在首次生成之后（p.326），论文未讨论冷启动、缓存淘汰与 E cache 在发送方的驻留成本。

## 论文结构索引

| 论文位置         | 内容                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| §1 / p.319–320   | 摘要与动机：compound AI 系统的三类场景、贡献清单、"first distributed LLM inference system" 定位               |
| §2 / p.321–322   | 背景：E/Q/K/V 与 prefill/decode；上下文共享三场景；goodput 与 prefill 干扰（Fig.1/2）                          |
| §3.1 / p.322     | benchmark 构建：8 对模型的入选三标准（同基座、任务相关、receiver 更准）                                        |
| §3.2 / p.322–323 | 三条洞察：整体复用质量塌方（Fig.3）、关键层约 11%（Fig.4）、关键层跨输入稳定（Fig.5）                          |
| §4.1 / p.324     | 两个挑战：分散重算低效、多 transition point 误差传导（Fig.6/7a）；E cache 与 transition layer                  |
| §4.2 / p.324–325 | 离线 profiling：Pareto 前沿、5% 容差、O(l²) 与 2 层分组（Fig.7b）                                              |
| §4.3 / p.325–326 | 在线设计：按 SLO 选配置（附录 C）；三策略对比与传输/重算流水（Fig.8/9）；E cache 单层驻留账                    |
| §4.4 / p.326     | 实现：约 3K 行 Python + vLLM/LMCache 三接口、独立 CUDA Stream                                                  |
| §5.1 / p.326–327 | 装置：8 模型对、6 数据集、两台 A100+IB；四类基线定义                                                           |
| §5.2 / p.327–328 | 预填充延迟 1.7–3.1×；对 CacheBlend 质量 +5–33%（Fig.10）                                                       |
| §5.3 / p.328     | 在线服务：TTFT/TBT/E2E 拐点与 2–4× 吞吐（Fig.11）                                                             |
| §5.4 / p.329     | profile 跨数据集泛化（4/2 分差）；MetaGPT 编码工作流 2.7×；MoE 验证（Fig.12–14）                               |
| §5.5–5.6 / p.329 | 带宽敏感性（Fig.15）；profiling 开销 3.6→1→0.375 小时（Fig.16）                                                |
| §6 / p.330       | 五条局限与未来工作；§7–8 相关工作与结论                                                                        |
| 附录 A–C / p.337–338 | 数据集与模型对清单（Table 1/2）；数学任务与更小模型对比（Fig.18/19）；动态配置自适应（Fig.20）             |

## 与系列其他篇目

同一份 KV cache，两条扩展路线：[DroidSpeak](droidspeak.md)（NSDI '26，即本篇）把"谁的 KV"从同一个模型扩到同基座的微调变体——复用的对象换成了另一组权重产出的中间状态，代价是要先学会哪些层可以信、哪些层必须重算；[KVLink](kvlink.md)（NeurIPS 2025）把"复用什么"从前缀匹配扩到文档级预计算与任意拼接——复用的对象仍是同一个模型，但把复用粒度从 token 前缀放松到整篇文档，靠位置重编码与 link token 缝合。横切总纲见 [KV Cache 复用总纲](kv-reuse-overview.md)。

## 诚实声明

- **版本判断**：本 PDF 首页为 NSDI '26 论文集封面（Renton, WA，2026-05-04~06，ISBN 978-1-939133-54-0），正文页脚 319–338，是会议正式版；arXiv:2411.02820（最新 v4，2025-07-14）仍用旧题 "…Cross-LLM Communication and Multi-LLM Serving"。两版数字未逐页比对，仅摘要口径一致（4× / 3.1×）。
- **全部性能数字为论文真机实测**（A100 与双节点 InfiniBand 环境），无仿真；唯一标"读图值"的是 Fig.15 的 TTFT 区间（精读员目测，论文正文未给数值）。
- **"1.7–3.1×" 与 "最高 4×" 的口径**：前者是预填充延迟缩减、对照 full prefill、覆盖 8 对模型 × 3 数据集；后者是在线吞吐（§5.3 的原文范围是 2–4×）；两者不能合并叙述。
- **质量承诺是"容差内"**：profiling 默认 5% F1 容差、在线实验用 1% 容差配置；未评估长对话下误差累积、也未给出 perplexity / 人工评测。
- **覆盖范围**：仅同架构同基座的模型对、两两复用；跨基座、多模型联合复用、冷启动与缓存/内存治理均未验证。
- 论文未提供公开代码仓库；§5.5 明说优势随网络带宽升高而缩小。

## 参考资料

- Yuhan Liu, Yuyang Huang, Jiayi Yao, Shaoting Feng, Zhuohan Gu, Kuntai Du, Hanchen Li, Yihua Cheng, Junchen Jiang, Shan Lu, Madan Musuvathi, Esha Choukse. _DroidSpeak: KV Cache Sharing Across Fine-tuned Model Variants_. NSDI '26, pp. 319–338. 本地 PDF：[references/papers/droidspeak-nsdi26.pdf](references/papers/droidspeak-nsdi26.pdf)；[USENIX 论文页](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan)（官方 PDF：`https://www.usenix.org/system/files/nsdi26-liu-yuhan.pdf`）；[arXiv:2411.02820](https://arxiv.org/abs/2411.02820)（旧题名版本）
- 代码仓库：论文未提供公开仓库（2026-10-13 经 USENIX 论文页与 arXiv 页核实无链接）；实现底座为 [vLLM](https://github.com/vllm-project/vllm) 与 [LMCache](https://github.com/LMCache/LMCache)
- 精读报告（含逐条页码核对、关键数字表与疑点清单）：[references/droidspeak-report.md](references/droidspeak-report.md)
