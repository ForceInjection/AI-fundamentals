# 当压缩表示被继续使用：三篇论文给出的三种二次决策

> **元信息**：本文是「KV Cache 压缩」系列总纲。系列含三篇单篇精读：[Self-Indexing KVCache](self-indexing.md)（AAAI '26，压缩码兼任稀疏检索索引）· [SALS](sals.md)（NeurIPS 2025，pre-RoPE 潜空间选 token）· [EchoKV](echokv.md)（NeurIPS 2026，丢弃维度 + 轻量重建），每篇都附逐条页码核对的精读报告（`references/`），论文原文 PDF 在 `references/papers/`。本文只做横向组织与口径提示，机制细节与数字以单篇为准。整理日期 2026-10-13。

同样「只保留三成缓存」，三个方法在 LongBench 上的平均分摆在一起：Palu 4.91、CommonKV 31.08、EchoKV 45.27——全量是 49.88（EchoKV 论文 p.6，Table 1）。这一格的差距来自压缩哲学的分歧：低秩系把整份缓存压进一个固定子空间，预算收紧时误差没有退路；EchoKV 不压，直接丢掉一部分，再用轻量网络把丢掉的预测回来。

本系列三篇的共同点在于：压缩产物继续参与后续计算。Self-Indexing 把量化码直接当检索索引，回答「该读谁」；SALS 把低秩潜空间当选 token 的场地、只重建被选中的少数，回答「该算谁」；EchoKV 把丢弃的维度当作待预测对象、用层间与层内相似性补回，回答「丢了怎么办」。三篇的压缩目标不同、决策位置不同，但都在同一份「被压过的表示」上继续做文章——而既有压缩方案（量化、低秩、驱逐）大多止步于「把存储账做小」。

## 一、三种落法：同一份压缩表示，三种用法

三篇可以放进同一根轴：**压缩表示被继续使用时，承担什么角色**。

- **Self-Indexing：压缩码即索引。** 出发点是「top-k 稀疏预测本质上是基于余弦相似度的检索」——余弦只关心方向，而 key 的符号正是方向最紧凑的编码。于是 sign-based 1-bit 量化产出的符号码同时是存储格式与检索结构：query 与 key 同切 4 维子向量，每组预生成 16 项查找表，检索直接在压缩域用查表加和完成，免去外挂索引与学习式预测器。
- **SALS：潜空间即选择场。** 低秩压缩必须发生在 RoPE 之前（旋转抬高 key 方差与秩，post-RoPE 低秩贴不住）；而 pre-RoPE 低秩缓存全量重建又太贵。SALS 的答案是把选择也搬进压缩域——所有头共享一个离线标定的单头潜空间，query 与 key 用它做廉价内积选出关键 token，只对被选中的少数重建并就地施加 RoPE，「选谁」与「重建谁」融合进一个 Triton kernel。
- **EchoKV：丢弃的维度即待预测对象。** 它放弃「压缩-解压」范式：每层 KV 展平后按组丢弃维度，组首层保留全量作全局信号、本层保留前 m 维作局部信号，一个轻量线性网络把丢弃的维度预测回来。原模型参数一个不改，压缩因此退化成运行时可选项。

三种姿势各占一个位置：**直接查**（检索）、**直接选**（门控）、**反向补**（重建）。前两者把决策留在压缩域、省掉重建；第三篇把重建做成轻量预测、换回全量域再算。取舍不同，前提相同：KV 表示里的冗余足够结构化，少存一点照样能算。

放回本仓 [《KV Cache 压缩技术详解》](../01_concepts/compression/kv_cache_compression.md) 的四维冗余框架，三篇的坐标也清楚：Self-Indexing 把 Token 维与精度维折叠进同一份数据（量化码即索引）；SALS 把 Token 维与特征维的接线点焊在 pre-RoPE；EchoKV 横跨特征维（丢弃展平子集预测重建）与结构维（层间相似性）。

## 二、三篇各自的机制与账本

### 2.1 Self-Indexing KVCache（AAAI '26）：压缩码兼任索引

**机制**：key 沿最后一维按 4 维一组取符号（16 种组合即 16 槽 codebook，单遍扫描建好）；检索侧 query 同切 4 维、每组一张 16 项分数查找表，由定制 LUT-GEMV kernel 执行；选定 token 后由融合反量化的稀疏 FlashAttention kernel 消费。量化参数按 token（而非按通道）存储——这是被选 token 能随机访问的前提。

**账本**：单 token 单头 896 bits（128 符号位 + 512 位 2-bit K/V + 256 位量化参数）对 FP16 的 4096 bits，省 78%；LongBench 160-token 预算下 58.2 对全量 58.7，且高过 K/V 保持 16-bit 的 Quest（56.5）与 DoubleSparse（54.8）；RULER 32K、7.5% 计算稀疏 89.2 对 90.8；64K 解码吞吐约 2×（读图 42.1 vs 20.3 t/s）。

**前提**：收益依赖两个自研 CUDA kernel，算法收益与 kernel 收益没有分开记账；codebook 16、子向量 4 维、sink 64 等超参固定、无敏感性分析；「7.5%」是参与注意力计算的比例，不是存储比例（全部 token 以压缩格式留存）。

### 2.2 SALS（NeurIPS 2025）：在旋转之前压、在潜空间里选

**机制**：两块证据把方案焊在 RoPE 之前——post-RoPE 的特征值谱上移、留 90% 方差所需秩上升（附录 Figure 4）；pre-RoPE 全量重建的开销随序列线性增长、32K 起主导总时间。投影矩阵来自 C4 语料 512 条序列的离线标定，所有头共享；打分只用投影后的前 r⋆ = 0.5r 维，top-k 选出后只重建被选者。

**账本**：键秩 25%/12.5% 两档（值同步 4-bit/2-bit）；LongBench 均值 32.26 / 31.97 对基线 32.65；12.5% 档访存只剩基线 5.8%，均值反超全部稀疏基线（Loki 31.95、HShare 31.83、Double Sparse 31.64）；RULER 80.81 对 81.60（25% 档）；4K、bs=16 的 attention 算子 5.7×（对 FlashAttention2）；端到端对 GPT-Fast 1.4×@4k / 4.5×@32k。

**前提**：校准型方法，换模型或数据分布需重标定；短序列倒贴（1k、bs=8 时 0.357 ms 对 FA2 的 0.230 ms）；层 0、1、31 不稀疏、全量读写；GPU 型号未给。

### 2.3 EchoKV（NeurIPS 2026）：丢掉维度，再预测回来

**机制**：三个 backbone 同为 8 个 KV 头 × 128 维 = 每 token 每层 1024 维。层按大小 S 分组，组首层保留全量作 global input，本层保留展平维度的前 m 维作 local input，每层一个线性层把丢弃的 1024−m 维预测回来；训练分两阶段（KV-MSE 表示对齐 + O-MSE 输出对齐）。键的输入取 pre-RoPE 版本。

**账本**：0.5 档（保留一半 ≈ 2×）LongBench 48.53 对全量 49.88（97.3%）、RULER 83.52 对 87.63；0.3 档 45.27 对 CommonKV 31.08、Palu 4.91；重建网络全部层合计 28.8–50.2M 参数，训练数据 1.6K 条、单卡 A100 短序列约 2 分钟。

**前提**：需要训练（虽然便宜，「a few minutes」对应短序列口径）；局部维度的「前缀」选择是朴素启发式；回切全量的方向论文无机制与实测（摘要口径是「全量→压缩」，贡献列表称双向）；「nearly lossless」只对 0.5 档成立。

## 三、对号入座

| 你的条件 | 该看哪篇 | 先决条件与口径提示 |
| -------- | -------- | ------------------ |
| 部署能上自研 kernel，想在 2-bit 存储上直接做稀疏读取、不再养索引 | [Self-Indexing KVCache](self-indexing.md) | 收益含两个自研 kernel 的贡献（未分开记账）；超参固定无敏感性分析；「5×」按正文 896 bits 口径理解 |
| 接受离线标定，想把键压到 1/8 级、值同步低比特，且主要跑长序列 | [SALS](sals.md) | 换模型/分布需重标定；短序列有开销；RULER 序列长度口径（128k 对 4k）只影响「多长上下文」的表述、不影响均值对比 |
| 不想动模型参数、要一个能按需启用与停用的压缩开关，可接受一次轻量训练 | [EchoKV](echokv.md) | 压缩率是保留比例口径（0.5 ≈ 2×、0.3 ≈ 3.3×）；「nearly lossless」仅 0.5 档；回切无实测 |
| 想给现有量化方案（FP8/NVFP4）之上再叠一层「免索引稀疏」或「可重建丢维」 | 三篇对读 + [量化解析](../01_concepts/compression/kv_cache_quantization.md) | Self-Indexing 本身即「2-bit 量化 + 1-bit 索引」的折叠；EchoKV 明言与量化正交、可叠加；叠加后的总账三篇均未给出 |

## 四、共同的边界

**三份账本都建立在定制实现上。** Self-Indexing 的收益来自自研 LUT-GEMV 与稀疏 FlashAttention 两个 kernel（组件对照全部以自研实现为准）；SALS 的端到端走自研 Triton 融合 kernel；EchoKV 的门槛最低（一个线性层加两阶段训练）。三篇都没有 vLLM/SGLang 类 serving 集成，没有 batching、分页适配的讨论——数字都出自单卡自测环境。

**压缩率口径三套，横向比较前必须换算。** Self-Indexing 用比特模型（896 对 4096 bits）、SALS 用秩比例与访存比（25%/12.5%，5.8% 是访存不是存储）、EchoKV 用保留比例（0.5/0.3，与业界 2×/3.3× 倍率写法相反）。三个「压缩率」不处在同一坐标轴上，本文不做跨篇拉直。

**精度承诺各有边界。** Self-Indexing 的评测止步 32K（作者自述两模型超 32K 后退化明显、比较失去意义）；SALS 的 12.5% 档退化集中在检索依赖最强的 NIAH-Single-1 与 Multi-Key-2；EchoKV 的 0.3 档定位是「maintains basic model usability」而非无损。

**与 04、05、06 系列的分工。** [内存层级系列](../04_memory_tiering/memory-tiering-overview.md)算「KV 放哪层」的账（显存装不下之后，DRAM、互联、调度怎么做交换）；[安全系列](../05_kv_security/kv-security-overview.md)算「KV 被谁看见」；[复用系列](../06_kv_reuse/kv-reuse-overview.md)算「KV 借给谁」；本系列算第四笔——**KV 变少之后，这份变小的表示还能不能用**。前三个系列关心 KV 的位置、权属与可见性，本系列关心压缩产物本身的剩余价值：它是索引、是选择场，还是重建的输入。

## 五、数字怎么读

- **Self-Indexing**：内存账按 896 bits/（token·头）算——原文总量写「768L」漏计了 128L 符号位，78% 恰按 896/4096 计算；贡献列表的「5×」是 4096/896 ≈ 4.57 的取整；「2× 端到端」字面写的是时延、证据实为解码吞吐（读图 64K 处 2.07×）；「7.5%」是计算口径。
- **SALS**：「6.4×」在摘要、贡献、结论出现三次而全文无推导，引用须带「作者口径」；「5.7×」可回溯到 Table 6（4K、bs=16）的 3.230/0.565 一格（精读员推算），同表 bs=8 档只有 3.1×–3.7×；§5.5 的「7.46×」与表对不上，不要引用；RULER 均值（80.81/81.60）可用，「128k 实测」不可照抄。
- **EchoKV**：「数分钟」对应短序列口径（约 2 分钟），常序列（1.6K 样本）口径两阶段合计约 27 分钟；Mistral-Small-24B 0.3 档的 Palu 行 16 个数值与 Mistral-7B 同档行完全相同，疑为论文错表，引用避开该行；Fig.3a 的绝对值高于 Table 1、只可用其相对趋势。
- **三篇对读**：LongBench 与 RULER 同名，但模型、预算、压缩配置各不相同（一个压比特、一个压秩、一个丢维度），除注明外数字不可直接互比。

## 系列篇目

- [压缩后的 Key 就是索引：AAAI'26 Self-Indexing KVCache 让 KV 内存少占 78%](self-indexing.md)——**压缩码兼任索引**（AAAI '26）
- [压 KV 要赶在旋转之前：NeurIPS 2025 SALS 压到 1/8，访存只剩基线 5.8%](sals.md)——**潜空间选 token**（NeurIPS 2025）
- [丢掉的 KV 还能重建：NeurIPS 2026 EchoKV 缓存减半保住 97% 精度、可按需切换](echokv.md)——**丢弃 + 重建**（NeurIPS 2026）
- 精读报告（逐条页码核对版）：[Self-Indexing](references/self-indexing-report.md) · [SALS](references/sals-report.md) · [EchoKV](references/echokv-report.md)；论文原文 PDF 在 `references/papers/`

## 诚实声明

- 三篇的数字均为论文自测，本仓未复现；本文的横向比较跨论文、跨任务、跨环境，除注明外**不可直接互比**。
- 本文每个数字都取自对应单篇精读；各单篇的「诚实声明」列出了自己的口径限定与未核实项（Self-Indexing 的读图值与 768L 笔误；SALS 的 6.4×/7.46× 疑点与硬件不完整；EchoKV 的错表嫌疑、回切未实测与切换开销空白）。
- 「三种二次决策：索引 / 选择 / 重建」的框架是本文的组织视角，不是任一论文的主张。

## 参考资料

- Xu Yang, Jiapeng Zhang, Dongyang Zhao, Guo Chen, Zhuo Tang. _Self-Indexing KVCache: Predicting Sparse Attention from Compressed Keys_. AAAI '26, Vol. 40 No. 33, pp. 27675–27683. [arXiv:2603.14224](https://arxiv.org/abs/2603.14224)
- Junlin Mu, Hantao Huang, Jihang Zhang, Minghui Yu, Tao Wang, Yidong Li. _SALS: Sparse Attention in Latent Space for KV cache Compression_. NeurIPS 2025（poster 115064）. [arXiv:2510.24273](https://arxiv.org/abs/2510.24273)
- Shiyu Ji, Yixuan Wang, Yijun Liu, Qingfu Zhu, Wanxiang Che. _EchoKV: Efficient KV Cache Compression via Similarity-Based Reconstruction_. NeurIPS 2026（poster）. [arXiv:2603.22910](https://arxiv.org/abs/2603.22910)
- 本仓相关：[KV Cache 压缩技术详解](../01_concepts/compression/kv_cache_compression.md) · [KV Cache 量化深度解析](../01_concepts/compression/kv_cache_quantization.md) · [KV Cache 内存层级总纲（04）](../04_memory_tiering/memory-tiering-overview.md) · [KV Cache 复用总纲（06）](../06_kv_reuse/kv-reuse-overview.md)
