# C 线：KV Cache 压缩/驱逐论文调研报告（2025 末–2026）

调研日期：2026-10-12。验证渠道：arXiv API（元数据/摘要/HTML 正文）、NeurIPS & ICLR 官方虚拟站（grep 全量论文列表）、ACL Anthology 2026 各卷页面（grep 全文）、AAAI OJS 官方页、NVIDIA 官方博客与 GitHub Releases、多路 web 检索交叉。OpenReview 论坛与 DBLP 页面被反爬拦截，凡仅能追溯到 OpenReview 的信息均已标注。**评级标准**：A = 顶会录用且数字扎实；B = 有录用或著名团队；否则淘汰。

## 评级汇总

| #   | 论文                  | Venue（来源已核）                         | 团队                    | 评级              |
| --- | --------------------- | ----------------------------------------- | ----------------------- | ----------------- |
| 1   | G-KV                  | 未录用（ARR 在投）                        | 微软                    | B                 |
| 2   | EvolKV                | 未录用                                    | UCAS + ETH              | B（下限）         |
| 3   | Universal Attention   | 未录用（arXiv 2026-10-06）                | IBM + UIUC              | B（声明需带限定） |
| 4   | Self-Indexing KVCache | **AAAI 2026**（OJS 核实）                 | 湖南大学                | **A**             |
| 5   | EchoKV                | **NeurIPS 2026 poster**（官方虚拟站核实） | 哈工大 SCIR             | **A**             |
| 6   | SALS                  | **NeurIPS 2025 poster**（官方虚拟站核实） | 北交大 + ByteDance Seed | **A**             |
| 7   | KARA                  | 未录用                                    | CMU                     | B                 |
| 8   | SAGE-KV               | **ICLR 2025 SLLM workshop**（非主会）     | SambaNova               | B                 |
| 9   | kvpress + NVFP4       | 开源工具/工程博客                         | NVIDIA                  | 生态补充          |

**关键纠错**：① G-KV「Findings of ACL 2026」说法系与同团队 Zipage 混淆，实为在投；② SAGE-KV 只是 ICLR 2025 workshop；③ Universal Attention 的「10× 且提升性能」确有摘要原文，但需端到端重训练且属作者口径。

---

## 1. G-KV: Decoding-Time KV Cache Eviction with Global Attention（未录用/在投）

**录用验证**：arXiv:2512.00504（2025-11-29）。OpenReview 论坛 4Uxtzc5Jns 为「Anonymous ACL submission」（ARR 在投，修订至 2026-01）。**已逐一 grep ACL Anthology 2026 主会与 Findings 两个全量卷页面（各约 6.7MB、2100+ 篇）：均无此文**。检索摘要中流传的「Findings of ACL 2026, pp. 7716–7737」系与同团队后续论文 Zipage（2026.findings-acl.381）混淆，不可采信。结论：**未录用**（来源：ACL Anthology 官方卷页面 + OpenReview 状态）。

**作者验证**：Mengqi Liao、Lu Wang、Chaoyun Zhang、Zekai Shen、Xiaowei Mao、Si Qin、Qingwei Lin、Saravan Rajmohan、Dongmei Zhang（微软）、Huaiyu Wan（北邮），代码在 `github.com/microsoft/G-KV`——**微软团队**，加分。

**核心贡献与数字**（来源：arXiv HTML 正文）：解码期驱逐，用「局部+历史」注意力全局评分（含记忆衰减因子 α）替代单窗口局部分；配套 RL + 蒸馏后训练让模型适配压缩 KV。512 token 预算下较基线方法提升 5%–20%（AMC 23 近 +20%）；吞吐 2.73–5.09×（预算 512/1024/2048），大批量下至多 12.18×/19.7× vs Full-KV；16k 上下文 KV 内存省约 90%（512 预算 96.1%）。模型为 DeepSeek-R1-Distill 7B/8B，评测集中在数学推理。

**评级：B**（著名团队 + 工程数字扎实，但顶会未录用）。

---

## 2. EvolKV: Evolutionary KV Cache Compression for LLM Inference（未录用）

**录用验证**：arXiv:2509.08315（2025-09-10），引用约 35。OpenReview 有 revisions 记录（id=RkCLb1Y9N2，疑 ARR 在投）；**已 grep ICLR 2026 官方虚拟站全量论文页：无此文**；ACL Anthology 2026 各卷亦无。结论：**未录用**。

**作者验证**：Bohan Yu（中国科学院大学）、Yekun Chai（ETH Zurich），两人小团队；ETH 知名但非 KV 方向核心组。

**核心贡献与数字**（来源：arXiv 摘要 + HTML 正文）：首个把「逐层 KV 预算分配」形式化为黑盒多目标优化的框架，按层分组做进化搜索，以下游任务表现为适应度。11 个任务上跨预算（128–2048）全面超基线；GSM8K 在 128 预算下较最强基线 +7 个百分点（保留全模型 95.7% 性能，而最强基线在 512 预算才保住 84.5%）；代码补全任务以仅 **1.5%** 原始预算反超全量 KV——这条「超全量」数字仅限该单一任务，引用时应带限定语。缺点：进化搜索评估成本高，且未录用。

**评级：B**（下限；若按严格「著名团队」口径可降为淘汰，建议保留观察）。

---

## 3. A Self-Pruning Transformer: Extreme KV-Cache Compression with Universal Attention（未录用，惊人声明核实）

**录用验证**：arXiv:2610.09051，**2026-10-06 提交（调研时仅 6 天）**，comments 仅「27 pages, 3 figures, 14 tables」，无任何 venue 线索。**未录用**（arXiv-only）。

**作者验证**：Davis Wertheimer、Naigang Wang、Raghu K. Ganti、Mudhakar Srivatsa 等（IBM Research，USA）+ Haochen Shen、Ahan Gupta、Derrick Liu、Minjia Zhang（UIUC SSAIL 实验室）——**IBM + UIUC 名团队**（来源：arXiv HTML 首页机构标注）。

**核心贡献与数字**（来源：arXiv 摘要逐字；含「10× 压缩还提升性能」声明的原文核实）：摘要原文为 "achieves state-of-the-art **10× compression** … while **improving** downstream performance compared to both state-of-the-art baselines **and unpruned oracles**"，另称 "unprecedented **25×** compression at length 16k"——声明确有原文依据，但属**作者口径**，未经独立验证。

**关键限定**：方法需把注意力层的 RoPE 替换为可学习的复合衰减机制并**端到端重训练**，非 training-free 即插即用；「压缩后胜过未压缩 oracle」更接近衰减机制带来的正则效应，复现门槛高、时间敏感。

**评级：B**（著名团队；声明引用时必须标注「需重训练 + 作者口径」）。

---

## 4. Self-Indexing KVCache: Predicting Sparse Attention from Compressed Keys（AAAI 2026）

**录用验证**：**AAAI-26 正式录用**，Proceedings of the Fortieth AAAI Conference on Artificial Intelligence, Vol. 40 No. 33, pp. 27675–27683，DOI 10.1609/aaai.v40i33.39988（来源：AAAI OJS 官方论文页直接核实）。arXiv:2603.14224（2026-03-15，对应候选「2026-03」）。

**作者验证**：Xu Yang、Jiapeng Zhang、Dongyang Zhao、Guo Chen、Zhuo Tang，全部**湖南大学**（含湖大深圳研究院）；团队非国际第一梯队，但论文为 AAAI 正会 ML Track 正文。另有同班底后续版 arXiv:2609.13205（2026-08）。

**核心贡献与数字**（来源：arXiv HTML 正文 + AAAI 页面）：统一「压缩」与「稀疏检索」——压缩后的 key 表示（sign-based 1-bit 向量量化）本身即检索索引，免外挂索引与学习式预测器；定制 LUT-GEMV + 稀疏 FlashAttention CUDA kernel。KV 内存至多省 **5×**（正文口径：896 bits/（token·头）、省 78%），稀疏 attention 计算 **6.7×** 加速，端到端最高 **2×** vs FlashAttention v2（「时延」为贡献列表措辞、证据实为解码吞吐）；格式支持 token 级随机访问。

**评级：A**（CCF-A 顶会录用 + 硬件友好设计 + 数字完整）。

---

## 5. EchoKV: Efficient KV Cache Compression via Similarity-Based Reconstruction（NeurIPS 2026）

**录用验证**：**NeurIPS 2026 主会 poster 正式确认**——官方虚拟站列出该文（neurips.cc/virtual/2026/poster/152942，本次调研直接抓取官方页面 grep 证实）；NeurIPS 2026 于 2026-12-06–12 悉尼举行。arXiv:2603.22910（v1 2026-03-24，v2 2026-05-13，对应候选「2026-05」）。

**作者验证**：Shiyu Ji、Yixuan Wang（共同一作）、Yijun Liu、Qingfu Zhu、Wanxiang Che（车万翔），**哈工大 SCIR**——NLP 名团队，加分。

**核心贡献与数字**（来源：arXiv 摘要 + HTML 正文）：低秩系压缩（Palu/CommonKV 类）需改模型投影、不可逆；EchoKV 不改原模型，用轻量网络从部分 KV 子集利用**层间/层内相似性**重建被丢弃分量，支持运行时按需切换（摘要口径为全量→压缩；贡献列表称双向，回切方向无机制与实测）。LongBench + RULER、7B–24B 多 backbone 多压缩率一致超现有方法；两阶段轻量训练单卡 A100：短序列约 2 分钟、常序列口径约 27 分钟；短上下文场景保持全量缓存吞吐。

**评级：A**（NeurIPS 2026 主会 + 名团队 + 「按需切换」是差异化卖点）。

---

## 6. SALS: Sparse Attention in Latent Space for KV Cache Compression（NeurIPS 2025）

**录用验证**：**NeurIPS 2025 主会 poster 确认**（多源交叉：neurips.cc 虚拟站 poster 页、作者 OpenReview profile 标注「NeurIPS 2025 poster」、NeurIPS 2025 12/3 日程 poster 环节）。arXiv:2510.24273（2025-10-28）。

**作者验证**：Junlin Mu、Jihang Zhang、Minghui Yu、Tao Wang、Yidong Li（**北京交通大学**）+ Hantao Huang；论文标注工作完成于 **ByteDance Seed**——产业名团队，加分。

**核心贡献与数字**（来源：arXiv HTML 正文）：洞察 RoPE 会抬高 key 方差（rank 升高），故在 **pre-RoPE** 低秩投影到共享单头潜空间（所有头共享一个投影），并在潜空间内直接选关键 token、仅对被选 token 重建再施加 RoPE，规避重建瓶颈。LLaMA2-7B/Mistral-7B + RULER（LLaMA3.1-8B；序列长度口径：摘要写 128k、表题标 4k，互相矛盾）验证：**6.4×** KV 压缩（全文未给推导，作者口径）+ **5.7×** attention 算子加速 vs FlashAttention2（4K、bs=16）；端到端吞吐 vs GPT-fast **1.4×@4k / 4.5×@32k**；25%/12.5% 两档压缩，12.5% 档内存访问降至基线 5.8% 且精度超所有稀疏基线，25% 档 RULER 均值 80.81 ≈ 全量基线。

**评级：A**（NeurIPS 主会 + 数字完整 + 训练后即用）。

---

## 7. KARA: Efficient Reasoning LLM Serving via Sliding-Window KV Cache Compression（未录用）

**录用验证**：arXiv:2607.01237（v1 2026-05-01，v2 2026-07-03 改题并对应候选「2026-07」），comments 仅「9 pages, 6 figures」；多轮检索**未发现任何 venue 记录**（无 OpenReview 论坛、无会议列表收录）。**未录用**（arXiv-only）。

**作者验证**：Shen Han、Yuyang Wu（共同一作）、Junpu Yu、Olexandr Isayev（通讯），全部 **CMU**（Isayev 组，来源：arXiv HTML 机构标注）。

**核心贡献与数字**（来源：arXiv HTML 正文）：两个被忽视的问题——阈值触发的解码期压缩在内存受限并发下会出现「并发-吞吐反转」（vLLM+SnapKV 在 batch 128 时吞吐反降 11.15%），且反复压缩旧区块造成不可逆信息丢失。Kara 只在**滑动窗口内**做解码期压缩，用双向注意力打分、保留「孤立 KV 对 + 任意位置弹性块」组合；适配 PagedAttention 后封装为基于 vLLM 的 **KvLLM**。保留 **20%** KV 时近乎 100% 全量精度，KvLLM 内存受限并发服务下较 vanilla vLLM 平均吞吐 **+12.75%**。提升幅度温和、篇幅短（9 页），偏 workshop 体型。

**评级：B**（CMU 团队 + 问题定位准，但未录用且增益偏工程）。

---

## 8. LLMs Know What to Drop: Self-Attention Guided KV Cache Eviction（SAGE-KV）（ICLR 2025 Workshop）

**录用验证**：arXiv:2503.08879（2025-03-11）。**ICLR 2025 Workshop on Sparsity in LLMs（SLLM）poster**，iclr.cc 官方 workshop 页面列出（来源：iclr.cc 虚拟站检索快照）。**注意：是 workshop 论文，非 ICLR 主会**——候选单所写「ICLR」需降格。

**作者验证**：Guangtao Wang、Shubhangi Upasani、Chen Wu、Darshan Gandhi、Jonathan Li、Changran Hu、Bo Li、Urmish Thakker，**SambaNova Systems**（AI 芯片公司，Thakker 为 Principal Engineer，组页 urmish.github.io）——业界知名团队。

**核心贡献与数字**（来源：arXiv 摘要）：发现 prefill 后注意力分数已足以在 **head 级**判断哪些 token 可丢；prefill 后一次性做 token+head 双层 top-k 选择，之后以压缩缓存推理。LongBench、Llama-3.1-8B-Instruct-128k 等：内存效率较静态 StreamingLLM **4×** 且精度反升，较 H2O 类 **2×**，长上下文精度接近全量。方法简单（免训练），但属一次性行前压缩、无解码期动态，数字在 2026 年已不突出。

**评级：B**（workshop 录用 + 知名公司）。

---

## 9. NVIDIA/kvpress（含 DecodingPress）与 NVFP4 KV 格式（工程生态补充）

**验证**：`github.com/NVIDIA/kvpress`（2024-11 创建，1228 stars，最新 v0.5.5 发布于 2026-09-17，来源：GitHub API）。README 确认 **DecodingPress** 为实验性解码期压缩 wrapper：按 `compression_interval` 周期性压缩、可保留近期 hidden states 缓冲、仅接受 ScorerPress 子类（KNormPress/CriticalKVPress 等）；衍生有 CompressionRatioDecodingPress（按固定比例压缩）、PrefillDecodingPress、CAMPress（驱逐时合并进保留 token）等。**NVFP4 KV**：NVIDIA 官方博客《Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache》（**2025-12-08**，来源：博客页面日期）——vs FP8 KV 内存足迹至多省 **50%**（上下文预算翻倍），代码生成/知识类任务精度损失 **<1%**，on-device 可驻留约 2× 上下文从而提升缓存命中率；TensorRT-LLM 生态落地，LMSYS 2026-09-16 亦发博文跟进。
**定位**：非论文，工程生态基线与部署参照。不参与 A/B 评级；写作中可作为「工业界把 KV 压缩做成默认能力」的证据。
