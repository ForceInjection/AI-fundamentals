# 压 KV 要赶在旋转之前：NeurIPS 2025 SALS 压到 1/8，访存只剩基线 5.8%

> **元信息**：论文 _SALS: Sparse Attention in Latent Space for KV cache Compression_，Junlin Mu、Hantao Huang、Jihang Zhang、Minghui Yu、Tao Wang、Yidong Li（北京交通大学 + ByteDance Seed，脚注注明 "Work done at ByteDance Seed"），NeurIPS 2025（第 39 届，poster，编号 115064）。本地 PDF 为 arXiv:2510.24273v1（2025-10-28 提交，正文落款 2025-10-29），共 15 页。源码：论文称 "The source code will be publicly available in the future."（p.1），截至 2026-10-13 未检索到公开仓库。精读方式：全文逐条抽取核对 + 关键页 110–200 dpi 渲染复核（报告见 [references/sals-report.md](references/sals-report.md)），数字均标论文页码，读图值与精读员推算另行标注。阅读日期 2026-10-13。

KV cache 在隐层维度上低秩——低秩压缩一路引用的这个经验事实，SALS 发现它有个前提：只在 RoPE 之前成立。key 一旦乘上旋转矩阵，方差被抬高、秩随之上升，一个固定投影矩阵再也贴不住随 token 位置转动的子空间（pp.1、4）；而反过来坚持在 RoPE 之前压，低秩缓存必须先重建回全秩才能施加旋转——"the time spent reconstructing low-rank keys/values from their rank-r form and applying the RoPE rotation grows linearly with sequence length and soon dominates the total attention runtime at 32 K tokens"（p.5）。前后两条路，一条丢精度，一条丢速度。

NeurIPS 2025 的 SALS 把第三件事也搬进了压缩域：token 选择本身。所有头共享一个 pre-RoPE 低秩投影，query 与 key 在这个单头潜空间里做一次廉价内积打分、选出关键 token，只把被选中的少数重建出来、就地施加 RoPE 送进 FlashAttention。两档设置下的结果：25% 档 LongBench 平均分 32.26 对基线 32.65、访存降到基线约 11%；12.5% 档访存只剩 5.8%，平均分 31.97 反而高过全部稀疏基线（pp.8–9）。摘要汇总的效率口径是 4K 序列上 attention 算子加速 5.7×、端到端吞吐对 GPT-Fast 1.4×@4k 与 4.5×@32k（p.1；口径拆解见 §三）。

## 一、问题：RoPE 之后压不狠，RoPE 之前重建贵

### 1.1 低秩压缩的两难：逐头省重建、联合保精度

低秩 KV 压缩的出发点是一个反复被验证的观察：key cache 在隐层维度上低秩，"a relatively small rank to retain approximately 90% of the energy"（p.4，引 Palu 与 Loki 的结论）。但落到实现上有一个两难，论文借 Palu 概括："Compressing each KV head separately reduces the overhead but at the cost of losing accuracy. However, compressing all heads together by low-rank projection improves accuracy by preserving the global information but results in significantly increased reconstruction cost."（p.2）——逐头压，重建便宜但精度差；联合压（所有头共享一个投影），精度好但重建要动整个大矩阵。Palu 用分组头分解折中，SALS 选的是第三条：把重建对象从全部 token 砍到少数被选中的 token（p.2）。

按论文自己的分类表（Table 1，p.3），既有方案里 Palu 是纯低秩、StreamingLLM 是固定模式、Quest 与 Double Sparse 是动态稀疏，SALS 是唯一 "Dynamic + Low Rank" 的组合——低秩负责存储，动态稀疏负责访问。

### 1.2 证据一：RoPE 抬高方差与秩，post-RoPE 压缩走不通

论文先把两种压缩顺序写进同一个式子（Eq.3，p.4）：attention 分数有两种低秩近似——对 key 先旋转再压（post-RoPE，缓存 `K̃ = K R_j U`，query 同样投影后免去重建），或先压再旋转（pre-RoPE，缓存 `K̃`，用时重建 `K̃U^T`）。唯一的免重建路线是前者，但论文的结论是走不通："applying RoPE will rotate the key vectors with larger variance, making them difficult to approximate using a low-rank matrix."（p.4）

证据分直观与定量两块。直观版是一个二维 key 散点（Fig.1(b)，p.4）："applying RoPE to a set of key vectors pushes the data points outward and rotates their principal axes"——旋转后点云外扩、主成分轴被转走（读图值：旋转后的红点在纵轴散布明显大于旋转前的蓝点）；更要命的是"the principal components rotate based on the token position, suggesting that a single fixed projection matrix may no longer approximate all of these rotated subspaces"（p.4）——不同位置转出的角度不同，一个固定投影罩不住所有子空间。定量版在附录（Figure 4，pp.14–15）：对 key 协方差做特征分解，post-RoPE 的特征值谱整体上移；按 Loki 的 `Rank_l(v)` 定义算"留住 90% 方差所需的主成分数"，两模型上 post-RoPE 的秩曲线在几乎所有层都高于 pre-RoPE（读图值：LLaMA2-7B 从约 500–1300 升到约 800–2200，满维 4096；Mistral-7B 从约 100–260 升到约 150–460，满维 1024）。前人做 post-RoPE 压缩精度掉得多也被顺手引用——Eigen Attention 被定性为 "suffers a relatively large accuracy loss"（p.3）。

### 1.3 证据二：pre-RoPE 压缩的重建税

那就老老实实在 RoPE 之前压？重建开销随序列线性增长。Fig.1(a)（p.4）把 attention 耗时按模块拆开（TopK / Recover / Pre-K / Softmax / γ-PV），低秩重建版的 Recover 块随序列变长，到 32K 时已是总时间的主体，64K/128K 更甚（读图值；论文正文的表述见开头引句，p.5）。post-RoPE 压不动、pre-RoPE 压得起但用不起——重建这一关绕不过去，除非需要重建的 token 先少下来。

## 二、机制：把"选择"也搬进压缩域

### 2.1 离线标定出一个所有头共享的 pre-RoPE 潜空间

投影矩阵来自离线标定（§4.2，pp.5–6；标定配方见 §5.1，p.8）：从 C4 语料随机抽 512 条长度 4096 的序列，收集其 pre-RoPE key，把每个 token 的「头数 × 头维」拼成 `nd` 维向量，算经验协方差 `C = K^T K`，再做特征分解 `C = UΣU^T`，取前 `r` 个特征向量作投影 `U_r ∈ R^{nd×r}`——所有头共享同一个 `U_r`（论文称 multi-head joint projection），把多头键压进 "a compact single-head latent space"（§4.1，p.5）。键侧压缩率即 `d/r`，实验取两档：`r = 25%` 与 `12.5%`（p.8）。

为什么敢让所有头共享：Lemma 1（p.6）证明逐头分块投影集 `B_r` 是联合投影集 `U_r` 的子集，因此联合投影的最优能量捕获不低于逐头。精读员注：这段证明是集合包含论证（`B_r ⊆ U_r`），"不亏"是必然而非洞见；真正的精度依据是 Palu 观察到的"联合压保留全局信息"，而联合投影重建更贵的代价，交给下一节的稀疏选择去抵消。

### 2.2 潜空间内做选择：不重建、不加存储

打分只用投影后的前 `r⋆` 维（§4.3，p.6）：query 投影一次 `q̃ = U_r^T q`，只取 `q̃_{:r⋆}`（`r⋆ = 0.5r`，p.8）；每个缓存 key 本来就存着 `k̃_j = U_r^T k_j`，直接取前 `r⋆` 维做内积 `s_j = q̃_{:r⋆}^T k̃_{j,:r⋆}`——"critical tokens are identified directly from the existing key cache, without extra storage and at a fraction of the original compute"（p.6）。

选择分数的正当性由 §3.2 的 overlap score 背书（p.5）：用潜空间分数的 top-Nc 圈出的位置，能覆盖多少全量注意力的质量——三个模型上"the average overlap score mostly remains above 90% for layers 2—29 but drops below 50% in layers 0, 1"，结论是"even when RoPE is completely omitted, latent space tokens can still preserve almost all of the full attention scores across the majority of layers"（p.5）。据此，层 0、1、31 干脆不稀疏、全量计算（p.8）。

### 2.3 只重建被选 token：一个融合 kernel 完成重建 + RoPE

Algorithm 1（p.7）把 decode 一步的流程写清楚：新 token 投影入潜空间并入缓存（步骤 2–3）→ 前 `r⋆` 维内积出近似分数（4）→ TopK 选出索引集 C（5）→ 只对 C 做 `K_C = K̃_C U_r^T` 重建（6）→ 对 query 与被选 key 施加 RoPE（7）→ 在选中子集上跑标准 FlashAttention（8）→ `y = p V̂_C`（9）。

关键工程动作是把 token 选择、重建、RoPE 融合进一个 Triton kernel，它"reduces memory traffic by 7.69× to 14.28×, depending on the chosen sparsity level and low-rank compression ratio, compared to the standard FlashAttention implementation"（p.7）。流量模型（§4.5，p.7）：全量 attention 每步搬 `2sd` 个元素；SALS 分两步——选择读 `s·r⋆`，选中后二次访问低秩 key 与 value 各 `kr`，合计 `sr⋆ + 2kr`，记忆带宽口径的加速比 `2sd/(sr⋆ + 2kr) = 2/(d_{r⋆} + 2 d_r k_s)`（其中 `d_{r⋆} = r⋆/d`、`d_r = r/d`、`k_s = k/s`）。精读员注：后文 Table 2/3 的 Memory Access 列与这个公式吻合（见报告 §四），说明该列量的是"访存比"而非"缓存大小比"——后者是另一列 Comp. ratio。

### 2.4 键走低秩、值走量化，近期窗口留精度

值不参与低秩，理由很直接："Since value tensors are almost full rank and play a pivotal role in attention, we forgo low-rank projection for them and instead perform channel-wise group quantisation that mirrors the key-cache setting (4 bit at 25% and 2 bit at 12.5%)."（p.8）——所以 "SALS-25%/12.5%" 是键秩比例与值位宽同步的两档。另有 KIVI 式混合精度："tokens in the most recent window are compressed by only 50%"，且高精度窗口与"总会被选中的最近 w 个 token"对齐（p.8）。值侧这套通道分组量化的粒度坐标（Per-Tensor / Per-Token-Head / NVFP4 三级）见 [《KV Cache 量化深度解析》](../01_concepts/compression/kv_cache_quantization.md)。

## 三、评估：两档设置下的精度、访存与加速

### 3.1 装置与口径

模型为 LLaMA2-7B-Chat（MHA）与 Mistral-7B-v0.2（GQA），RULER 上加测 LLaMA3.1-8B-Instruct；机器是 Xeon Platinum 8336C + 一张 Ampere 架构 GPU + 128 GB 内存——GPU 型号未给（p.7）。对照三路：压缩方向 Palu（低秩）与 KIVI（量化），稀疏方向 Double Sparse、HShare、Loki，效率方向 FlashAttention-v2 与 GPT-Fast（pp.7–10）。稀疏基线"只在解码期引入稀疏"且 KV 不压缩（p.8）。稀疏预算：GSM8K/CoQA 固定 1/4 稀疏、保留最近 w=128 token；LongBench 统一 1/8——LLaMA2-7B 保留 Nc=512 token（Hshare 配置：16 sink + 432 critical + 64 recent）、Mistral 翻倍到 1024（p.8）。

### 3.2 精度：压缩档压过 Palu，稀疏档反超全部稀疏基线

**GSM8K/CoQA**（Table 2，p.7）：baseline 0.2335（严格口径）/0.5997，SALS-25% 0.2312/0.5975，SALS-12.5% 0.2176/0.6070；对照 Palu-50%(3bit) 崩到 0.0614/0.5560、KIVI-2bit 0.2047/0.6010。25% 档是压缩基线里最高的（"SALS–25% achieves the best accuracy compared to KIVI and Palu, with neligible loss compared to the baseline"，p.8）；CoQA 上 12.5% 档的 0.6070 甚至略高于全量——单次评测，不宜过度解读。

**LongBench**（Table 3，p.8，均值列）：LLaMA2-7B-chat baseline 32.65 → SALS-25% 32.26、SALS-12.5% 31.97（同表 KIVI-4bit 32.70、KIVI-2bit 31.93、Palu-30% 30.91、Palu-50% 26.92）；Mistral-7B baseline 43.12 → 42.79/40.99（Palu-50% 掉到 35.71）。

**对稀疏基线**（Table 4，p.9）：baseline 32.65；SALS-25% 32.26 > Loki 31.95 > HShare 31.83 > Double Sparse 31.64，SALS-12.5% 的 31.97 同为最高。访存侧："about 11% of the baseline's memory traffic"（25% 档，p.9）与"the Memory–Access Ratio drops to just 5.8%, yet SALS still outperforms all competing sparse methods in accuracy"（12.5% 档，p.9）。

**RULER**（Table 5，p.9）：baseline 均值 81.60 → SALS-25% 80.81（"nearly identical"，p.9），SALS-12.5% 75.86；12.5% 档的退化集中在检索依赖最强的 NIAH-Single-1 与 Multi-Key-2（p.10）。该表的序列长度口径自相矛盾（"128k context" 对 "4k sequence length"），见诚实声明。

### 3.3 效率：长序列越跑越快，短序列倒贴

**算子延迟**（Table 6，p.10，1000 次重复）：bs=16、4k 序列时 FlashAttention-v2 要 3.230 ms，SALS-12.5% 0.565 ms、SALS-25% 0.757 ms——摘要的 5.7× 与这一格吻合（精读员按表推算 3.230/0.565 ≈ 5.7）；bs=8、4k 时 1.630 → 0.439/0.530 ms（3.7×/3.1×）；同为 4k、bs=8，SALS-12.5% 也快过稀疏基线（HShare 0.576、Double Sparse 0.724、Loki 1.102 ms）。短序列是例外："SALS will introduce some overhead for short sequences (e.g. 1k)"（p.10）——bs=8、1k 时 FA2 只要 0.230 ms，SALS 两档 0.409/0.357 ms。

**端到端吞吐**（Table 7，p.10，对 GPT-Fast，50 次重复，token/s）：4k 118 → 163.5（1.4×）；8k 70.6 → 150.1（正文口径 2.13×，p.10，"This is due to the overhead of reconstructing the selected token."）；16k 38.3 → 122.7；32k 19.8 → 89.47（正文写 4.57×，按表算 4.5×）；64k（bs=4）9.2 → 42.96。

**压缩率口径**：摘要另称 "Under different settings, SALS achieves 6.4-fold KV cache compression"（p.1），但全文未给该数字的推导——Table 2 的 Comp. ratio 是 0.28/0.15（约 3.6×/6.7×），与标称的 4×/8× 也对不严。引用时视为作者口径，详见诚实声明。

## 四、边界与讨论

论文自认的边界有四条：短序列有开销（p.10）；12.5% 档并非无损，退化集中在 "where the retrieval dependency is strongest" 的 NIAH-Single-1 与 Multi-Key-2（p.10）；附录末句提出层自适应秩选择 "could further enhance compression efficiency"（p.15），即当前"一秩到底"留了余量；方法属校准型，换模型需重算投影与标定（p.8）。

精读员注补充几条论文没展开的：

- **硬件只有一句 "one GPU (ampere architecture)"**（p.7）：型号、显存、带宽全缺，5.7× 在什么器件上成立无法复现判定；效率对照中 FA2 与 GPT-Fast 的精度与 kernel 实现也未说明。
- **附录 Figure 4(c) 的读数与 12.5% 档的精度损失方向一致**：LLaMA2-7B 多数层留住 90% 方差要 500–1300 个主成分（满维 4096，读图值），12.5% 档只给 512 维——连 90% 方差都盖不满；RULER 均值掉到 75.86 与其呼应。25% 档（1024 维）则大体盖住这条线。
- **"访存 5.8%" 不全是选择策略的功劳**：稀疏基线只在解码期稀疏、KV 仍全精度存储（p.8），SALS 的访存优势含低秩与量化的存储收益；Table 4 的 0.11 对 0.14 不能全记在选择头上。
- **跳过的三层没有量化**：层 0、1、31 不稀疏、全量读写（p.8），这部分对总访存的拉扯未给数字。
- **无 serving 实测**：没有 vLLM/SGLang 类集成、没有分页/批处理适配讨论；端到端只有 GPT-Fast 一条对照，且 4k–64k 各只有一个 batch size。

## 论文结构索引

| 论文位置         | 内容                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------- |
| §1 / pp.1–2      | 动机：重建瓶颈与 RoPE 抬秩；两大洞察、三项贡献、关键数字                                  |
| §2 / pp.2–3      | 相关工作：潜空间注意力（Palu/Eigen Attention）、稀疏注意力（Quest/DS/NSA）；Table 1 分类   |
| §3.1 / pp.3–5    | 两种压缩顺序的形式化（Eq.3）；方差放大论证（Fig.1(b)）；重建开销（Fig.1(a)）              |
| §3.2 / p.5       | 潜空间 token 表示：overlap score 定义与 Figure 2                                          |
| §4.1 / p.5       | 三阶段流水总览（Figure 3）                                                                |
| §4.2 / pp.5–6    | 离线标定：协方差 + 特征分解；共享投影；Lemma 1（联合 ≥ 逐头）                             |
| §4.3 / p.6       | 潜空间打分选 token（前 r⋆ 维内积）                                                        |
| §4.4 / pp.6–7    | 稀疏注意力定义（Eq.4–5）与 Algorithm 1（只重建被选 token）                                |
| §4.5 / p.7       | roofline 流量模型（`2/(d_{r⋆}+2d_r k_s)`）与融合 kernel（7.69–14.28×）                    |
| §5.1–5.2 / pp.7–8 | 装置、两档设置与标定配方；Table 2/3 精度与访存                                            |
| §5.3 / p.9       | 对稀疏基线 Table 4（25% 档访存约 11%、12.5% 档 5.8%）                                     |
| §5.4 / pp.9–10   | RULER Table 5（80.81 vs 81.60；12.5% 档 75.86）                                           |
| §5.5 / p.10      | 效率：Table 6 算子延迟、Table 7 端到端吞吐                                                |
| §6 / p.11        | 结论                                                                                     |
| A / pp.14–15     | 附录取证：特征值谱与 `Rank_l(90)` 层曲线（Figure 4）；层自适应秩建议                       |

## 与系列其他篇目

同一件事——在压缩表示上直接做决策——系列里给了三种落法：[SALS](sals.md)（NeurIPS '25，本篇）在 pre-RoPE 低秩潜空间里选 token、只重建被选中的少数；[Self-Indexing](self-indexing.md)（AAAI '26）把压缩后的 key 直接当作稀疏检索索引，压缩与检索合一；[EchoKV](echokv.md)（NeurIPS '26）用层间/头间相似性重建被丢弃的分量，并支持运行时按需切换。放回本仓既有的四维冗余框架（[《KV Cache 压缩技术详解》](../01_concepts/compression/kv_cache_compression.md)），SALS 同时踩 Token 维（稀疏选择）与特征维（低秩投影），并把两个维度的接线点焊死在 RoPE 之前——这既区别于传统 post-RoPE 低秩方案（Eigen Attention 一路），也区别于"KV 本体不压缩、只做稀疏读取"的路线（StreamingLLM/Quest 一路，论文对后者的批评是 "leaving the KV cache uncompressed and thus a potential bottleneck"，p.3）。横切总纲见 [KV Cache 压缩总纲](kv-compression-overview.md)。

## 诚实声明

- **venue 与版本**：本地 PDF 为 arXiv:2510.24273v1（2025-10-28，正文落款 2025-10-29），PDF 自身未标 venue；NeurIPS 2025 poster（编号 115064）依据 NeurIPS 官方虚拟站与调研多源交叉核实。camera-ready（OpenReview zJSZupQ889）本次不可取回，全部数字以本地 v1 为准。
- **6.4× 无推导**：摘要、贡献与结论三处写 "6.4-fold KV cache compression"（pp.1–2、11），正文没有任何算式或表格与之精确对应——Table 2 的 Comp. ratio 0.28/0.15 对应约 3.6×/6.7×，与标称的 25%→4×、12.5%→8× 也对不严。
- **5.7× 的出处**：按 Table 6（bs=16、4k）3.230 ms ÷ 0.565 ms ≈ 5.7 与摘要吻合（精读员推算，论文未指明）；同为 4k 的 bs=8 档只有 3.1×–3.7×。
- **§5.5 的 "7.46×" 对不上表**：原文 "SALS accelerates the stand-alone attention operator by 7.46× speed-up for batch size (bs) =8 and 4k sequence length (4k)"（p.10），但 Table 6 该配置（bs=8、4k）FA2 1.630 ms 对 SALS 0.439/0.530 ms 是 3.7×/3.1×。
- **RULER 口径自相矛盾**：§5.4 写 "We evaluate all methods on a 128k-token context, applying an 8× sparsity ratio, i.e., selecting 16k active tokens"，同段末句又写 "with a 4k sequence length"，Table 5 题注亦标 "(4k sequence length)"（p.9）。均值对比与退化位置不受影响，但"128k 实测"不能照抄。
- **硬件不完整**：评估机只写 Xeon Platinum 8336C、一张 Ampere 架构 GPU、128 GB 内存（p.7）；GPU 型号、显存、带宽缺失。
- **口径与公平性**：稀疏基线只在解码期稀疏、KV 不压缩（p.8），SALS 的访存优势含存储压缩的贡献；层 0、1、31 不稀疏、全量读写（p.8），其影响未量化。Table 3 的 0.06 与正文 5.8%（p.9）是同一数字的两种粒度。
- **未实测**：真实 serving 集成、分页与批处理、更激进的压缩档位（论文只给 25%/12.5% 两档）、标定语料与在测分布的漂移敏感性——均无数据；准确率实验未说明重复次数（效率实验注明 1000 次/50 次重复）。
- **读图值**：Fig.1(a)、Fig.1(b)、Fig.2、Fig.4(c)(d) 与 Figure 3 的读数为目测估计，正文引用均已标"约"。

## 参考资料

- Junlin Mu, Hantao Huang, Jihang Zhang, Minghui Yu, Tao Wang, Yidong Li. _SALS: Sparse Attention in Latent Space for KV cache Compression_. NeurIPS 2025（第 39 届，poster，编号 115064）。本地 PDF：[references/papers/sals-arxiv-2510.24273.pdf](references/papers/sals-arxiv-2510.24273.pdf)（arXiv:2510.24273v1，2025-10-28，15 页）；[arXiv 摘要页](https://arxiv.org/abs/2510.24273)；[NeurIPS 2025 poster 页](https://neurips.cc/virtual/2025/poster/115064)
- 源码：论文称 "The source code will be publicly available in the future."（p.1）；截至 2026-10-13 本仓外未检索到公开仓库（GitHub 检索无结果）。
- 精读报告（逐条页码核对与疑点清单）：[references/sals-report.md](references/sals-report.md)
