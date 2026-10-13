# SALS（NeurIPS 2025）精读报告

> 来源：papers/sals-arxiv-2510.24273.pdf（本目录下 papers/，PDF 共 15 页，单栏 arXiv 预印本，印刷页码 = PDF 页号，页脚 1–15 已逐页核对）。版本：arXiv:2510.24273v1，2025-10-28 提交（正文落款 "Date: October 29, 2025"）。核对方式：`pdftotext -layout` 全文抽取逐条对照 + 关键页 110–200 dpi 渲染目读（pp.1、4–10、14–15）；数字标页码「p.X」，读图值与精读员推算另行标注；未核实写「未核实」。venue：arXiv 版 PDF 自身未标 venue；NeurIPS 2025 poster（编号 115064）经 NeurIPS 官方虚拟站核实（neurips.cc/virtual/2025/poster/115064，本仓外核实 2026-10-13），OpenReview 论坛 zJSZupQ889（直接抓取被 403 拦截，未取回 camera-ready 比对）。

## 一、速览

- 完整标题：SALS: Sparse Attention in Latent Space for KV cache Compression（p.1，无副标题；NeurIPS 虚拟站页面把 "cache" 印作 "Cache"）。
- 作者与机构：Junlin Mu^1,2,\*、Hantao Huang^2,†、Jihang Zhang^2、Minghui Yu^2,†、Tao Wang^1,†、Yidong Li^1,†；1 = Beijing Jiaotong University（北京交通大学），2 = ByteDance Seed；脚注 "\* Work done at ByteDance Seed, † Corresponding authors"（p.1）。首页有 ByteDance Seed 标识。
- 一句话贡献：发现 KV 低秩性只在 pre-RoPE key 上成立（RoPE 抬高方差与秩），据此在共享单头 pre-RoPE 潜空间里完成低秩存储 + top-k token 选择 + 只重建被选 token（重建后施加 RoPE），从而避开"全量低秩重建"瓶颈；LLaMA2-7B-chat / Mistral-7B 上 25% 与 12.5% 两档压缩，LongBench 12.5% 档访存 5.8%、均值反超全部稀疏基线（pp.1、8–9）。
- 开源/落地：论文称 "The source code will be publicly available in the future."（p.1）；截至 2026-10-13 本仓外 GitHub 检索未发现公开仓库。无生产集成描述。
- 评估装置：Xeon Platinum 8336C CPU + 一张 Ampere 架构 GPU + 128 GB RAM（p.7，GPU 型号未给）；速度实验走 PyTorch 后端 + 自研 Triton 融合 kernel（p.10）。
- 二档设置一句话：键侧低秩 `d_r = r/d = 25% / 12.5%`，值侧同步 4-bit / 2-bit；打分用前 `r⋆ = 0.5r` 维（p.8）。

## 二、问题与动机

（均为论文自述，关键处给英文短引）

- **低秩压缩的两难（Palu 继承）**："Compressing each KV head separately reduces the overhead but at the cost of losing accuracy. However, compressing all heads together by low-rank projection improves accuracy by preserving the global information but results in significantly increased reconstruction cost."（p.2）逐头压便宜但丢精度，联合压保精度但重建贵；SALS 的答案："sparsely selecting a small subset of tokens from the KV cache significantly reduces reconstruction error while preserving model accuracy"（p.2）。§2.1 对 Palu 的表态："We agree with Palu's observation but tackle the computation from the sparse perspective."（p.3）
- **post-RoPE 压缩的先例与代价**：Eigen Attention 在 RoPE 之后压，"suffers a relatively large accuracy loss"（p.3）。
- **两种压缩顺序（Eq.3，p.4）**：pre-RoPE 形式 `Q_i K_j^T ≈ (Q R_i R_j^T U U^T) K^T`（缓存 `K̃ = KU`，用时重建 `K̃U^T`）；post-RoPE 形式 `≈ (Q R_i U U^T R_j^T) K^T`（缓存 `K̃ = K R_j U`，query 也投影 `Q̃ = Q R_i U`，免重建）。论文措辞："Ideally, we can adopt the post-RoPE transformation ... to avoid the reconstruction complexity. However ... applying RoPE will rotate the key vectors with larger variance, making them difficult to approximate using a low-rank matrix."（p.4）
- **RoPE 抬秩的两块证据**：（1）Fig.1(b) 二维示意——"applying RoPE to a set of key vectors pushes the data points outward and rotates their principal axes"、"the principal components rotate based on the token position, suggesting that a single fixed projection matrix may no longer approximate all of these rotated subspaces"（p.4）；（2）附录 Figure 4（Pp.14–15）——特征值谱 post-RoPE 整体上移；`Rank_l(90)`（留住 90% 方差的主成分数）post-RoPE 在几乎所有层高于 pre-RoPE。
- **重建税（Fig.1(a)）**："the time spent reconstructing low-rank keys/values from their rank-r form and applying the RoPE rotation grows linearly with sequence length and soon dominates the total attention runtime at 32 K tokens"（p.5）。
- **第二位洞察（选择侧）**：潜空间 token 表示跨层稳定——overlap score "mostly remains above 90% for layers 2—29 but drops below 50% in layers 0, 1"（p.5）；"even when RoPE is completely omitted, latent space tokens can still preserve almost all of the full attention scores across the majority of layers"（p.5）。灵感来源写为 Double Sparse（p.1）。
- **对既有稀疏方法的批评**："most of these post-training approaches mainly reduce bandwidth and computation cost, leaving the KV cache uncompressed and thus a potential bottleneck"（p.3）；与 NSA 的区别："NSA requires training from scratch... In contrast, SALS can be calibrated post-training, performing top-k token selection based on low-rank latent approximations rather than a trained sparse module."（p.3）

## 三、核心机制

### 3.1 离线标定与共享单头潜空间（§4.2 pp.5–6；配方 §5.1 p.8）

- 标定：C4 随机抽 "512 sequences of length 4096"，收集 pre-RoPE key；把每个 token 的头数与头维拼成 `nd` 维（"reshaped by merging the head number with head dimension"）；经验协方差 `C = K^T K` → 特征分解 `C = UΣU^T` → 取前 `r` 个特征向量 `U_r ∈ R^{nd×r}`，所有头共享（multi-head joint projection），产出 "a compact single-head latent space"（§4.1，p.5）。键压缩率 `d/r`。
- Lemma 1（p.6）：逐头分块投影集 `B_r = {diag(U_1,…,U_n) | U_i^T U_i = I_{r'}}`（`r' = r/n`）是联合投影集 `U_r = {U ∈ R^{nd×r} | U^T U = I_r}` 的子集，故 `max_{U∈U_r} E(U) ≥ max_{U∈B_r} E(U)`。证明即集合包含（`B_r ⊆ U_r`）。
- 精读员注：Lemma 1 的"联合不低于逐头"是平凡结论（超集上的最大化）；其唯一数学内容是排除"联合反而更差"的可能。论文没有证明"跨头冗余"之类的经验命题——联合投影的精度依据来自 Palu 的观察，重建代价由 §4.3 的稀疏选择摊薄。

### 3.2 潜空间内选 token（§4.3 p.6）

- query 投影一次 `q̃ = U_r^T q`，取前 `r⋆` 维；每个缓存 key 天然存有 `k̃_j = U_r^T k_j`，取前 `r⋆` 维；近似分数 `s_j = q̃_{:r⋆}^T k̃_{j,:r⋆}`。原文："critical tokens are identified directly from the existing key cache, without extra storage and at a fraction of the original compute."
- overlap score（§3.2 p.5）：`OS = Σ_{i∈C} p_i / Σ_{i=1}^s p_i`——潜空间 top-Nc 索引 C 覆盖全量注意力质量的比例；Figure 2 覆盖 LLaMA2-7B-Chat / LLaMA3-8B / Mistral-7B-v0.2。
- 据此跳过层 0、1、31 的稀疏（"Based on the observations in Figure 2"，p.8）。

### 3.3 只重建被选 token 与融合 kernel（§4.4 pp.6–7；§4.5 p.7）

- Algorithm 1（p.7）：新 token 投影入潜空间并入缓存 → 前 r⋆ 维内积打分 → TopK → 只重建 `K_C = K̃_C U_r^T` → 对 query 与 `K_C` 施加 RoPE → 选中子集上标准 FA → `y = p V̂_C`（`V̂` 为量化值缓存）。
- 融合 kernel：token 选择 + 重建 + RoPE 一个 Triton kernel，"reduces memory traffic by 7.69× to 14.28×, depending on the chosen sparsity level and low-rank compression ratio, compared to the standard FlashAttention implementation"（p.7）。
- 流量模型（§4.5）：全量 `2sd`；SALS `sr⋆ + 2kr`；加速比 `2sd/(sr⋆ + 2kr) = 2/(d_{r⋆} + 2 d_r k_s)`，其中 `d_{r⋆} = r⋆/d`、`d_r = r/d`、`k_s = k/s`（p.7，公式版式经 200 dpi 渲染核对）。

### 3.4 值的处理与混合精度（§5.1 p.8）

- 值不退低秩："Since value tensors are almost full rank and play a pivotal role in attention, we forgo low-rank projection for them and instead perform channel-wise group quantisation that mirrors the key-cache setting (4 bit at 25% and 2 bit at 12.5%)."
- KIVI 式混合精度："tokens in the most recent window are compressed by only 50%, whereas all preceding tokens are compressed according to the target ratio"；高精度窗口与稀疏窗口对齐（最近 w 个 token 总被选中）。

## 四、关键数字

> 除注明外均为真机实测（Ampere GPU，型号未给）；accuracy 实验未见重复次数说明，效率实验注明 Table 6 为 1000 次、Table 7 为 50 次重复（p.10）。「精读员推算」标出论文未给的换算。

| 指标 | 数值 | 对照基线/条件 | 出处 |
| ---- | ---- | ------------- | ---- |
| 摘要汇总效率口径 | 6.4× KV 压缩 + 4K 序列 attention 算子 5.7× | vs FlashAttention2；**6.4× 全文无推导** | p.1 摘要、p.2 贡献、p.11 结论 |
| 端到端吞吐 | 1.4×@4k、4.5×@32k | vs GPT-Fast | p.1；Table 7 p.10 |
| LongBench 均值（LLaMA2-7B-chat） | baseline 32.65 / SALS-25% 32.26 / SALS-12.5% 31.97 | KIVI-4bit 32.70、KIVI-2bit 31.93、Palu-30% 30.91、Palu-50% 26.92 | Table 3 p.8 |
| LongBench 均值（Mistral-7B-v0.2） | baseline 43.12 / 42.79 / 40.99 | KIVI-4bit 43.11、KIVI-2bit 41.80、Palu-30% 34.74、Palu-50% 35.71 | Table 3 p.8 |
| LongBench 访存比 | 25% 档 "about 11%"；12.5% 档 "just 5.8%"（表值 0.11 / 0.06） | vs baseline，稀疏 1/8 | Table 3 p.8；正文 pp.8–9 |
| LongBench 稀疏基线对照 | baseline 32.65；SALS-25% 32.26；SALS-12.5% 31.97；Loki 31.95（0.19）；HShare 31.83（0.14）；Double Sparse 31.64（0.16） | 同稀疏预算（x=16, y=432, z=64，1/8） | Table 4 p.9 |
| GSM8K 严格/柔性（LLaMA2-7B-chat） | baseline 0.2335/0.2335；SALS-25% 0.2312/0.2343；SALS-12.5% 0.2176/0.2229 | KIVI-4 0.2297/0.2305、KIVI-2 0.2047/0.2047、Palu-30%(3bit) 0.1713/0.1759、Palu-50%(3bit) 0.0614/0.0879 | Table 2 p.7 |
| CoQA（同上） | baseline 0.5997；SALS-25% 0.5975；SALS-12.5% 0.6070 | KIVI-2 0.6010、KIVI-4 0.5992、Palu-30% 0.5938、Palu-50% 0.5560 | Table 2 p.7 |
| 压缩比 / 访存（Table 2 列） | SALS-25% Comp. 0.28、Mem. 0.13；SALS-12.5% Comp. 0.15、Mem. 0.07 | KIVI-4 0.31/0.31、KIVI-2 0.19/0.19、Palu-30% 0.14/0.14、Palu-50% 0.09/0.09 | Table 2 p.7；与 §4.5 公式互证见 §五注 7 |
| RULER 均值（LLaMA3.1-8B-Instruct） | baseline 81.60；SALS-25% 80.81；SALS-12.5% 75.86 | 表题标 "(4k sequence length)"；正文又写 128k context/16k active tokens（矛盾见 §五注 4） | Table 5 p.9；正文 pp.9–10 |
| 算子延迟 bs=16, 4k | FA2 3.230 ms；SALS-25% 0.757±0.026；SALS-12.5% 0.565±0.013（**5.7× 与 3.230/0.565≈5.7 吻合，精读员推算**） | 单位 ms，1000 次重复 | Table 6 p.10；摘要 p.1 |
| 算子延迟 bs=8, 4k | FA2 1.630；Loki 1.102；DS 0.724；HShare 0.576；SALS-25% 0.530±0.008；SALS-12.5% 0.439±0.016（3.1×/3.7×，精读员算） | 同上 | Table 6 p.10 |
| 算子延迟 bs=8, 1k（短序列例外） | FA2 0.230；SALS-25% 0.409±0.093；SALS-12.5% 0.357±0.051 | "SALS will introduce some overhead for short sequences (e.g. 1k)" | Table 6 p.10 |
| 端到端吞吐（token/s） | 4k: 118→154.1/163.5；8k: 70.6→128.6/150.1（正文 2.13×）；16k: 38.3→102.5/122.7；32k: 19.8→67.97/89.47（正文 4.57×，按表算 4.5×）；64k(bs=4): 9.2→32.92/42.96 | vs GPT-Fast；50 次重复 | Table 7 p.10 |
| §5.5 文字口径 | "accelerates ... by 7.46× speed-up for batch size (bs) =8 and 4k sequence length" | 与 Table 6 同配置 3.1×–3.7× 不符（见 §五注 3） | p.10 |
| 流量模型公式 | `2sd/(sr⋆ + 2kr) = 2/(d_{r⋆} + 2d_r k_s)`；全量 2sd vs SALS sr⋆ + 2kr | 记忆带宽口径 | §4.5 p.7 |
| 融合 kernel 流量降幅 | 7.69×–14.28× | vs 标准 FlashAttention 实现 | §4.5 p.7 |
| overlap score | 层 2–29 多数 >90%；层 0、1 <50% | LLaMA2-7B-Chat / LLaMA3-8B / Mistral-7B-v0.2 | §3.2、Figure 2 p.5 |
| Fig.1(a) 重建开销 | Recover 块随序列线性变厚，32K 起主导总时间（读图值） | 模块：TopK/Recover/Pre-K/Softmax/γ-PV | 正文 p.5、图 p.4 |
| Fig.4(c)(d) 秩曲线 | 90% 方差秩 post-RoPE 高于 pre-RoPE：LLaMA2-7B 约 500–1300→约 800–2200（满维 4096）；Mistral-7B 约 100–260→约 150–460（满维 1024）（读图值） | `Rank_l(v)` 定义引 Loki | 附录 pp.14–15 |
| 设置 | 键 `d_r`=25%/12.5%；打分 `r⋆=0.5r`；值 4-bit/2-bit；近窗 50% 压缩；标定 512×4096（C4） | 层 0、1、31 不稀疏 | §5.1–5.2 pp.7–8 |
| 稀疏预算 | GSM8K/CoQA：w=128、1/4；LongBench：Nc=512（LLaMA2）/1024（Mistral）、1/8（x=16, y=432, z=64） | — | §5.2 p.8 |
| 装置 | Xeon 8336C + 1×Ampere GPU（型号未给）+ 128 GB RAM；速度实验 PyTorch + Triton kernel | — | pp.7、10 |

## 五、局限与边界

**论文自认的限制：**

1. **短序列有开销**："SALS will introduce some overhead for short sequences (e.g. 1k)"（p.10）。
2. **12.5% 档非无损**：退化集中在 NIAH-Single-1 与 Multi-Key-2，"where the retrieval dependency is strongest"（p.10）。
3. **秩选择留有余量**：附录末句——"the required rank varies substantially across layers, indicating that a layer–adaptive rank selection scheme could further enhance compression efficiency"（p.15）。
4. **校准型方法**：投影来自离线标定（C4 512×4096，p.8），换模型/分布需重标定；论文未讨论标定数据敏感性。

**精读员注（疑点与空白）：**

1. **6.4× 无来源**：摘要（p.1）、贡献（p.2）、结论（p.11）三处写 6.4×，正文无任何推导。Table 2 的 Comp. ratio 0.28/0.15 对应约 3.6×/6.7×；与标称 25%→4×、12.5%→8× 亦对不严。1/6.4 = 0.15625 亦无法由键/值两档组合推出（键 1/8 + 值 2-bit/16-bit = 1/8 等常见组合都不是）。**引用须带"作者口径"。**
2. **5.7× 的出处论文未写明**：可回溯到 Table 6（bs=16、4k）3.230/0.565 = 5.72 与摘要吻合（精读员推算）；bs=8、4k 同表只有 3.08×–3.71×。与之对照，调研报告（compression-report.md §6）的"6.4× + 5.7×（4K 序列）"表述与 PDF 摘要一致；引用这两个数字时带"5.7× 对应 4K、bs=16；6.4× 全文未给推导"的限定。
3. **§5.5 的 "7.46×" 与 Table 6 冲突**：正文称 bs=8、4k 时加速 7.46×（p.10），表内同配置 FA2 1.630 ms 对 SALS 0.439/0.530 ms，为 3.71×/3.08×。无法用任何表的列组合复现 7.46；疑为笔误或未披露配置。
4. **RULER 口径自相矛盾**：§5.4 写 "We evaluate all methods on a 128k-token context, applying an 8× sparsity ratio, i.e., selecting 16k active tokens during inference"，同段末句写 "with a 4k sequence length"，Table 5 题注标 "(4k sequence length)"（p.9）。摘要称 "RULER-128k"。表格均值（81.60/80.81/75.86）无论按哪种口径都可引用，但"128k 上下文实测"不能照抄；调研报告本条注明同一矛盾（摘要写 128k、表题标 4k）。
5. **硬件不完整**："one GPU (ampere architecture)"（p.7）——型号、显存、带宽缺失；FA2/GPT-Fast 对照的实现精度未说明。
6. **公平性口径**：稀疏基线"只在解码期引入稀疏"且 KV 不压缩（p.8）——Table 4 中 SALS 的访存优势混入了低秩+量化的存储收益，不是纯选择策略的对比。另：层 0、1、31 不稀疏、全量读写（p.8），影响未量化。
7. **Memory Access 列的语义可被 §4.5 公式复现（精读员推算）**：按 `d_r⋆/2 + d_r k_s`（d_r⋆=0.5d_r），25%×稀疏1/4 → 0.125、12.5%×1/4 → 0.0625，与 Table 2 的 0.13/0.07 吻合；LongBench（1/8）预测 0.094/0.047，Table 3 的 0.11/0.06 略高（近窗高精度可解释部分差异）。即该列 = 解码访存比，Comp. ratio 列 = 缓存尺寸比，两列不可混引；"访存 5.8%"（正文 p.9）与表值 0.06 是同一数字的两种粒度。
8. **版本口径**：本稿一切数字以 arXiv:2510.24273v1（2025-10-28）为准；camera-ready（OpenReview zJSZupQ889）本次因 403 未能比对（既有先例：KVLink v1→v4 有口径变化），引用正式版数字前建议复核。
9. **笔误与命名不一致**：Table 1 中 SALS 行 "Midian"（p.3）、Lemma 1 陈述处 `U ∈ R^{h×r}` 疑为 `nd×r`（与下文集定义不符）、"enery then"（p.6）、附录标题 "Dimensianlity"（p.14）；模型名主文写 Mistral-7B-v0.2、附录与 Figure 4 写 Mistral-7B-Instruct-v0.2。
10. **与附录读数的呼应（精读员推断）**：Figure 4(c) 显示 LLaMA2-7B 多数层留住 90% 方差需 500–1300 个主成分（满维 4096 的 12%–32%），而 12.5% 档只给 512 维——多数层连 90% 方差都盖不满，与 RULER 12.5% 档掉到 75.86 方向一致；25% 档（1024 维）大体盖住。跨模型（Mistral 满维 1024）与跨层差异论文未展开。
11. **未实测**：真实 serving 集成、分页/批处理适配、与 PagedAttention 的交互、标定漂移、>12.5% 的更激进档位——均无数据；accuracy 实验无重复次数/error bar 说明。

## 六、写作素材

- **系列定位（在压缩表示上做决策的第三种落法）**：SALS 与同系列的 Self-Indexing（压缩 key 即检索索引）与 EchoKV（相似性重建 + 运行时按需切换）构成"压缩三篇"；SALS 的独特处是把两个传统流派焊在一根时间线上——低秩压缩必须发生在 RoPE 之前，token 选择则直接复用压缩后的表示。
- **四维冗余框架落点（供与《KV Cache 压缩技术详解》互引）**：SALS 同时踩 **Token 维**（动态稀疏选择，§2.2 系谱：StreamingLLM/Quest/Double Sparse）与 **特征维**（低秩投影，§2.3 系谱：Palu/Eigen Attention/Loki）；其与 Palu 的关系是继承+改进（联合投影保精度 + 稀疏选择摊薄重建）。口径衔接点两处：① 该篇的 KQ-SVD/ReCalKV 属 post-RoPE 或对 K-Q 耦合做联合优化，SALS 的"必须 pre-RoPE"论证可直接补充"为什么 post-RoPE 低秩有谱域上限"（附录 Figure 4）；② "压缩率"表述要统一——SALS 的 0.28/0.15（Comp. ratio）与"6.4×（无推导）/4×/8×（标称）"并存，与既有文章谈 Palu 的 30%/50% 口径可直接对比。
- **适合放文章开头的反差素材**：① "低秩性只在旋转前存在"——Fig.1(b) 的散点与附录秩曲线把一句经验事实变成可检验的命题；② "32K 之墙"——低秩重建从"省显存"翻转为"费时间"（Fig.1(a)、p.5）；③ "1k 倒贴、32k 起飞"——Table 6/7 的短序列 penalty（0.230→0.357 ms）与 64k 的 4.7× 对比，说明收益全在长上下文；④ "访存 5.8% 反超全部稀疏基线"——12.5% 档是全文最干净的单点。
- **与系列其他篇目的对照点**：三篇都在"压缩后的表示"上做二次决策，区别在决策目标——SALS 选 token（决定算哪些）、Self-Indexing 找 token（压缩即索引）、EchoKV 补 token（重建被丢分量）；SALS 的代价是校准 + RoPE 前压的架构约束，EchoKV 的卖点是训练无关可回退，可留对照问题："到底是把决策放在压缩域（SALS/Self-Indexing）划算，还是保留一个可回退的重建通道（EchoKV）更稳？"
- **口径清单（写作与互引时直接抄）**：5.7× = Table 6 bs=16/4k（精读员推算）；1.4×/4.5× = Table 7 4k/32k（4.5 按表算，正文写 4.57）；32.26/32.65、31.97 与 5.8% = Table 3/4 与 §5.3；80.81/81.60 = Table 5（序列长度口径有矛盾）；7.46× 与 6.4× 均不照抄或带疑点标注。
