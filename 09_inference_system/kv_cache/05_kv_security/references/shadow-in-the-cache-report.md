# Shadow in the Cache（NDSS '26）精读报告

> 来源：papers/shadow-in-the-cache-arxiv-2508.09442.pdf（本目录 papers/ 下，PDF 共 18 页）。核对方式：pdftotext 全文抽取逐条对照 + 关键表格与图经 110–130 dpi 渲染目读（Table I/II/VII–XII、Fig. 4/5/8/9 已逐格核对）。页码约定：本 PDF 为 arXiv:2508.09442v4（2026-03-11），无独立印刷页码，页脚页码与 PDF 页号一一对应（抽查 p.1/7/13/17/18 页脚确认），故「p.X」即 PDF 页号。本 PDF 自带 NDSS 2026 会议页脚（DOI 10.14722/ndss.2026.240258、ISBN 979-8-9919276-8-0），版式即会议版。只有推断/点评标「精读员注」；未核实写「未核实」。阅读日期 2026-10-13。

## 一、速览

- 完整标题：Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference（p.1，无副标题）
- Venue：Network and Distributed System Security (NDSS) Symposium 2026，2026-02-23~27，San Diego, CA, USA（p.1 页脚）；DOI 10.14722/ndss.2026.240258，ISBN 979-8-9919276-8-0。本 PDF 为 arXiv:2508.09442v4（cs.CR，2026-03-11），页面版式已含 NDSS 会议页脚。
- 作者与机构：Zhifan Luo、Shuo Shao、Yuke Hu、Chenxu Zhao、Zhihao Liu、Zhan Qin（Zhejiang University，State Key Laboratory of Blockchain and Data Security）；Su Zhang、Lijing Zhou（Huawei Technology）；Zhan Qin 兼 Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security。通讯作者：Yuke Hu、Zhan Qin（p.1 署名与脚注）。
- 一句话贡献：论文自称首篇对 KV-cache 隐私风险的系统性分析（"the first comprehensive analysis of these vulnerabilities"，p.1 摘要）——拿到明文 KV-cache 的对手可重建用户输入；设计三类攻击（Inversion / Collision / Injection）与防御 KV-Cloak（可逆矩阵混淆 + 算子融合），使攻击重建质量回落到随机噪声水平、模型精度零损失、融合实现延迟开销 15.41 ms/GB（prefill 的 0.45%）。
- 开源情况：论文正文未给出任何代码链接（全文无 github/URL 字样，已 grep 确认）；官方实现仓库 [github.com/SiO-2/kvcloak](https://github.com/SiO-2/kvcloak) 自述 "Official implementation of 'Shadow in the Cache…' (NDSS 2026)"，Apache-2.0 License，创建于 2026-02-27、最后 push 2026-02-28（GitHub API 核实 2026-10-13，22 stars）。落地形态：以 vLLM 为目标集成对象，设计上"块内自包含、无跨块依赖"（§VI-G p.18）；无真实服务部署描述，无与 serving 栈的集成实测。

## 二、问题与动机

（以下均为论文自述；关键表述给英文短引）

- **背景：KV-cache 是明文外置的。** 端到端加密只覆盖用户与服务之间的通信，而"the KV-cache itself is almost always processed, transmitted between compute nodes, and persisted in plaintext"（p.1）；这是性能妥协——"This design choice is a concession to performance, as the unacceptable latency overhead from cryptographically securing the often gigabyte-scale KV-cache would violate the stringent demands of real-time inference"（p.1）。
- **动机：Confidential MaaS 的架构性暴露。** TEE 保护推理时，高吞吐架构必须把 KV-cache 外置出 TEE 保护边界："This architectural design—not a vulnerability—directly exposes the KV-cache to the Cloud Service Provider (CSP), who inherently possesses it without requiring any additional exploit"（p.2）。
- **威胁模型（§III-A，p.3）**：对手=推理服务提供方（CSP）。目标：逐字（verbatim）重建用户输入 prompt。能力：(1) 取得服务端存储/传输的 KV-cache——论文视其为"intentional performance-security trade-off, not a traditional vulnerability"（p.3）；(2) 拿到模型权重（灰盒）——闭源服务 CSP 天然持有，开源模型可因许可证披露或指纹识别获知（[8]、[27]）；不观察 GPU 寄存器里的瞬态激活。
- **KV-cache 为何比隐藏态更危险**："Unlike deep hidden states that represent fused semantic information, entries in the KV-cache maintain a direct, element-wise correspondence to the tokens in the user's input sequence"（p.3）——拿到缓存"does not merely obtain abstract embeddings, but potentially holds the raw materials to reverse-engineer the original user prompt"（p.3）。
- **与侧信道路线的分野。** 既有 KV-cache 威胁研究是时序侧信道（PromptPeek [39]，即 NDSS '25 的 I Know What You Asked）；论文指出这类攻击的实用性被现代内存管理削弱：PagedAttention [17] 默认 16-token 块，命中信号要求整块猜对，搜索空间 |V|¹⁶（|V| ≈ 10⁵），暴力侧信道不可行（p.3）。本文因此选择"直接访问"威胁模型：拿到数据后做"precise, per-token"重建（p.3）。
- **防御现状的五个坑（§V-A，p.8）**：全量加密/HE 对数十至数百 GB 的 KV-cache 开销不可接受（"Given that the KV-cache can be tens or hundreds of gigabytes, applying full encryption or HE is unsuitable"，p.8）；DP 加噪与稀疏注意力结构冲突，有效隐私预算下精度崩塌；KV-Shield [41]（唯一既有轻量混淆方案）有两处硬伤——固定置换保留统计分布（可被本文 Collision 攻击击穿）、固定密钥无动态随机性（无法防御 CPA），且打乱相对位置与 RoPE 不兼容（p.8）。
- **设计目标（p.8）**：Robust Security（同时遮蔽代数与统计性质）、Lossless Model Fidelity（注意力数学等价、零精度退化）、Negligible Overhead（开销尽量挪到离线）。

## 三、核心机制

### 3.1 三类攻击（§III-B，p.4–5；流程见 Fig. 2 p.5）

- **Inversion Attack（§III-B1，p.4）**：代数基线。式(4)：`x_i = k_i (R^d_Θ,i)^{-1} (W_k^T)^{-1}`（K 路；V 路对称），拿到明文缓存与 W_k、W_v 即可反算注意力层输入。两条硬约束：(1) "The attack strictly requires the projection matrices W_k and W_v to be square and full-rank invertible"——只在传统 MHA（如 LLaMA-7B）成立；GQA/MLA（LLaMA-3、Qwen、DeepSeek）把输入投影到低维子空间，矩阵非方阵、唯一逆不存在（ill-posed）；(2) "the attack's effectiveness is largely confined to the first decoder layer's KV-cache"——第一层逆推直接得到输入 embedding、可映射回词表；深层只能得到语义融合的隐藏态，需要训练辅助反演模型 [37]，成本高、保真降。定位：proof-of-concept，受架构与层位限制。例外：GQA 的 LLaMA-3.2-1B 第一层仍可被逆推（≈100%），论文归因于其"high projection rank relative to its hidden state dimension"——最小二乘在非方阵下仍有效（p.6）。
- **Collision Attack（§III-B2，p.4–5）**：通用前向匹配。把重建改写为搜索——对每个未知位置 i，找最小化距离的 token：式(5) `t*_i = argmin_t D(K_leaked^(i), K_local^(i)(t|x_<i))`（D 取 Frobenius 范数）。前提是"the KV-cache exhibits distance-preserving properties for identical tokens under the same context"（p.4）：错误 token 的距离近似高斯分布，正确 token 是显著离群点。逐 token 迭代（候选生成→本地前向→碰撞匹配，Fig. 2b）。三项工程优化（p.4）：(1) Batched Outlier Detection（批量异常值检测 + 提前退出）——`d_t < μ_batch − 3σ_batch` 即判定碰撞、终止搜索；(2) Probability-Guided Prioritization——按模型预测概率排序候选，正确 token 通常在前几批；(3) Search Space Pruning——只保留 top-k% 候选，限制最坏情况。CPA（选择明文攻击）增强：对手可投喂已知输入、观察产生的缓存，刻画 d_target、d_other 分布后自适应阈值，式(6) `P(success|r) = P(d_other > τ)^{r-1} × P(d_target < τ)`（p.5）。
- **Injection Attack（§III-B3，p.5）**：语义路子。架构约束——"Architecturally, a stand-alone KV-cache is computationally dormant"（p.5）：自注意力需要当前 Query 驱动，单独一份缓存不能自己"续跑"。攻击方式（contextual hijacking）：在截获的缓存后追加指令（最优为 "Repeat the previous content."，附录 C2），指令生成必要 Q 向量（stimulus generation），模型被对齐训练绑定、执行指令、把缓存里的隐私"回声"出来（forced exfiltration）："the model executes the instruction, effectively 'echoing' or summarizing the latent private information stored in the cache"（p.5）。优势：对 H2O [46] 类有损逐出鲁棒（语义要点仍在）、只需单次生成。论文由此得出结论："a comprehensive defense must render the KV-cache semantically unintelligible, not just algebraically obfuscated"（p.5）。

### 3.2 KV-Cloak 防御（§V-B~V-D，p.8–10；图见 Fig. 6 p.9、Fig. 7 p.9）

- **朴素可逆线性混淆先被自己攻破（§V-B，p.8–9）**：`K' = S K M`（S ∈ R^{b×b}、M ∈ R^{d×d} 秘密可逆矩阵）。CPA 下不安全：对手构造两个输入控制差值 ΔK = K1 − K2，观测 ΔK' = S(ΔK)M；令 ΔK 逼近标准基矩阵即可解出 S 的列与 M 的行，复杂度仅 O(b²d + bd²)。
- **实际方案：一次性置换 + 加性掩码（§V-C，p.9）**：式(9) `K' = S P̂ (K + A) M`。三个部件：
  - P̂ ∈ {0,1}^{b×b}：每块一次性随机置换矩阵（one-time pad）。依据是"the physical memory ordering of k, v vectors is computationally superfluous during inference, as positional semantics are intrinsic to the vectors via RoPE"（p.9）——块内重排保持注意力输出不变（Fig. 6：`o' = a P̂^T P̂ V = aV = o`），却给对手加上 b! 的组合复杂度（b ∈ {16, 32, 64}，对应 vLLM PagedAttention 标准块大小）；P̂ 用后即弃、"does not require storage"（p.9）。
  - A：结构化加性掩码，兼作 rank 保险与"位置信标"。低秩缓存（如重复 token 的 rank-1 K）会让置换熵坍缩；A 的元素从 [3θ_K, 4θ_K] 均匀采样、padding 置 1.5θ_K（θ_K 为校准文本上 K-cache 的最大绝对值，Fig. 7 记录 LLaMA-3.2-1B 上 K 的 μ=0.030/σ²=2.452、V 的 μ=−0.000/σ²=0.313），使 (K + A) 满秩；解混淆时按高幅值离群点动态定位置换后的掩码并减掉——"This mechanism effectively offloads the storage of P̂ into the data itself"（p.9）。
  - S、M：秘密可逆矩阵，遮蔽统计量（均值、方差），使"profiling and data mining attacks"失效（p.9）。密钥恢复复杂度 O((b²d + bd²)·b!)（p.9）。
- **算子融合（§V-D，p.10）**：另引入两个秘密可逆矩阵 M1、M2 ∈ R^{d×d} 改写 q/k/v 与输出权重（式(10)：`q^m = q(M1^{-1})^T`、`k^m = k M1`、`v^m = v M2`、`Wo^m = Wo (M2^{-1})^T`），注意力分数与输出数学不变（式(11)(12)）——"guaranteeing lossless accuracy"（p.10）。融合的关键约束：M1 必须与 RoPE 矩阵交换——"The fusion is algebraically feasible if and only if the secret matrix M1 and the RoPE matrix commute"（p.10）；因此 M1 结构化设计为 2×2 旋转-缩放子矩阵构成的块对角矩阵（附录 A 给出交换性证明），M2 取随机可逆旋转-缩放矩阵。四个权重离线预计算（式(15)：`Wq^m = M1^{-1} Wq` 等），在线只剩式(16) `K' = S P̂ (K^m + A)`，每块约 b³ + 2b²d 次浮点乘。LLaMA-3.1-8B（b=16、d=128、D=4096）下为 KV 重算代价的 0.83%；naive 实现（b³ + 2b²d + 2bd²）为 7.1%（附录 F p.17）。
- **与 PagedAttention 的兼容设计（§VI-G，p.18）**：全部密码学操作"self-contained within a single physical memory block"、无跨块依赖，内存管理器可以照常 schedule/copy/swap/share 物理块而无需理解其内容——这是"可直接接进 vLLM"的论证（无集成实测）。

## 四、关键数字

> 攻击与防御数字均为论文真机实测（单机、论文自建实验）；**唯一属建模性质的**是附录 F 用浮点运算量推导的开销占比（b³+2b²d 等，p.17）与 §V-C 的复杂度论证（O(·) 记号）。攻击目标平台论文未给出 GPU 型号（**未核实**）。

| 指标                                   | 数值                                                                                                                                   | 对照基线/条件                                                                                  | 出处                             |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------- |
| 攻击评估配置                           | 7 个模型：LLaMA-7B、LLaMA-2-7B、LLaMA-3.2-1B、LLaMA-3.2-3B-Instruct、LLaMA-3.1-8B、Qwen2.5-Math-7B、DeepSeek-R1-Distill-LLaMA-8B（LLaMA-3.1-8B-Distilled）；数据 LMSYS-Chat-1M 随机 1,000 条；指标 BERTScore（all-mpnet-base-v2）与 ROUGE-L | 攻击 LLaMA-3.1-8B-Distilled 时只用 base 权重；跨架构用 LLaMA-7B 攻 LLaMA-2-7B | p.5 §IV-A、p.5–6  |
| 侧信道可行性（动机数）                 | PagedAttention 默认块 16 token；词表 \|V\| ≈ 10⁵；猜中整块搜索空间 \|V\|¹⁶                                                             | 时序侧信道（PromptPeek [39]）的可行性论证                                                     | p.3 §II-B2                       |
| Inversion：LLaMA-7B 第一层             | BERTScore 1.000 / ROUGE-L 1.000（≈100% 逐字）；深层塌为噪声（Mid 0.065/0.036、Last 0.092/0.062，<10%）                                  | MHA 方阵满秩 + 第一层两个条件同时成立                                                          | p.6–7 §IV-B2、Table I            |
| Inversion：GQA 例外与失效              | LLaMA-3.2-1B 第一层 1.000/0.994（最小二乘）；LLaMA-3.1-8B 0.071/0.000、LLaMA-3.2-3B-Instruct 0.055/0.000、Qwen2.5-Math-7B 0.229/0.186 | GQA/MLA 非方阵投影下唯一逆不存在，1B 因投影秩相对隐藏维度高成为例外                            | p.6–7 §IV-B2、Table I            |
| Collision：全层重建（原文对照）        | LLaMA-3.2-1B 三层 BERTScore 0.877（First）/0.791（Mid）/0.894（Last）、ROUGE-L 0.709/0.617/0.680；LLaMA-3.1-8B 0.873/0.652/0.764；Qwen2.5-Math-7B 0.918/0.552/0.783 | 3σ 阈值、batch=256；未受保护 KV-cache                                                          | p.6–7 §IV-B3、Table I            |
| Collision：微调失配（灰盒）            | LLaMA-3.1-8B-Distilled（攻方仅持 base 权重）：First 0.642/0.633、Mid 0.492/0.227、Last 0.635/0.413                                     | "fine-tuning preserves the fundamental weight correlations"（p.8）                             | p.7 Table I、p.8 §IV-C           |
| Collision：跨架构失配（黑盒）          | LLaMA-7B 攻 LLaMA-2-7B 全部 ≈0.06–0.09（"< 0.1"，随机猜测水平）                                                                        | 攻击依赖权重相关性，架构不相交即失效                                                           | p.8 §IV-C、Table I               |
| Collision：统计阈值                    | 启发式 3σ_other：假阳率 0.13%、per-token 成功率 91.84%（r=64）；统计推导阈值→100%；微调模型 91.79%→97.82%                              | LLaMA-3.1-8B / Distilled，Last-layer 缓存                                                      | p.6 §IV-B3                       |
| Collision：效率（top-k 裁剪）          | 只搜 top 1/8 → 保持 96.1% 保真；单层平均重建 5.06s → 2.17s（<43%）。Fig. 4 全表：1/16→90.9%/1.62s、1/4→98.4%/2.76s、1/2→99.5%/3.54s、1→100%/5.06s | LLaMA-3.2-1B，batch=256、3σ                                                                    | p.7 §IV-B3、Fig. 4               |
| Collision+（先验知识增强）             | 假定 rank r=8 → 开源模型 ≈100%；Distilled：第一层 138.6% of baseline（实际平均 rank≈4）、最后一层 139.0%（64≤r≤128）、中间层无改善（实际 rank>128） | 相对该模型 Collision 基线；Fig. 5 读图第一层点 136–139%、最后一层点最高 135.8%（位置 24）、中间层 44–57% | p.6 §IV-B4、Fig. 5（p.8）        |
| Injection：语义重建                    | 平均 BERTScore 0.58 / ROUGE-L 0.42；峰值在 LLaMA-7B（0.765/0.687）；最优指令 "Repeat the previous content."；另三条指令见 Table VI         | 与 Collision 的精确匹配相比语义相似度为主                                                      | p.6–7 §IV-B5、p.16 Table VI      |
| Injection：数据与鲁棒性                | 需全层完整缓存（碎片化即失效）；对 H2O 等有损逐出鲁棒；单次生成；Collision 则单层即可                                                   | 两类攻击的数据需求对比                                                                         | p.5 §III-B3、p.7 §IV-C           |
| 攻击数据集泛化                         | LLaMA-3.2-1B：Alpaca Collision First 0.942/0.897；GSM8K 0.890/0.784；LMSYS-Chat-1M 0.877/0.709（BERTScore/ROUGE-L）                    | 三个不同领域数据集                                                                             | p.17 Table VII                   |
| 朴素防御被 CPA 攻破                    | `K' = S K M` 差分攻击（控制 ΔK）恢复 S、M，复杂度 O(b²d + bd²)                                                                         | 说明"固定线性结构"不安全                                                                       | p.9 §V-B                         |
| KV-Cloak 安全性（Table II）            | 全部攻击输出 BERTScore ≤ 0.093、ROUGE-L ≈0（例：LLaMA-7B Inversion First 1.000→0.091）                                                  | 对照 (10⁸, 10⁻⁵)-DP：Collision First 0.849、Collision+ 最高 1.000（LLaMA-3.2-1B Last）         | p.11 Table II                    |
| 防御后攻击成功率                       | 分布坍缩：d_target 3429.40±1215.19 vs d_other 3429.51±1214.97（不可分）→ 0% success                                                     | DP：10⁷ 下 37.09±1.05 vs 48.41±2.79（可分，per-token 94.84%）；10⁸ 下 11.71±0.38 vs 33.20±3.78（近 100%） | p.11–12 §VI-B、Fig. 8（p.12）    |
| KV-Cloak 精度（Table III）             | 6 模型 MMLU/SQuAD 与明文完全一致（如 LLaMA-3.1-8B 0.668/0.708 → 0.668/0.709）                                                          | (10⁸, 10⁻⁵)-DP 崩塌：LLaMA-7B 0.016/0.000、8B 0.283/0.026、Distilled 0.108/0.001               | p.12 Table III                   |
| DP 精度-隐私曲线                       | LLaMA-3.2-1B：ε=1/10/10⁷ → MMLU ≈0.05；ε=10⁸ → 0.262；ε=10⁹ → 0.299（明文 0.335）；10⁷-DP 下 Collision 仍 >55% 语义相似度               | 选 ε=10⁸ 作为"可用效用"的 DP 基线（更严预算精度归零）                                          | p.11、p.17 §VI-A/D、Table VIII/IX |
| 延迟开销（Fig. 9）                     | Prefill 3438.5 ms/GB；AES 3020.9 ms/GB；KV-Cloak 52.73 ms/GB（未融合）；KV-Cloak 融合 15.41 ms/GB = prefill 的 0.45%；DP 4.88 ms/GB     | 串行 PyTorch 实现、无自定义 CUDA kernel（"upper bound"口径）                                   | p.13 §VI-D、Fig. 9               |
| 块大小扫描（Table XII）                | 不融合：52.73/45.45/28.60 ms/GB（b=16/32/64 → +1.53%/+1.32%/+0.83%）；融合：15.41/10.17/12.33（+0.45%/+0.30%/+0.36%）                   | 相对 prefill 3438.5 ms/GB                                                                      | p.18 Table XII                   |
| 块大小对精度                           | LLaMA-3.2-1B：MMLU 0.335（16/32/64 全同明文）；SQuAD 0.463/0.462/0.460（明文 0.457）                                                   | "virtually lossless"（p.18）                                                                   | p.18 Table XI                    |
| 存储开销                               | 密钥总大小 2·n_kv·l·(b² + 3/2·d + 1)；LLaMA-3.1-8B（b=16）898 KB；Ling-1T（b=64）20.9 MB                                                | MB 级密钥可驻 TEE                                                                              | p.13 §VI-D                       |
| 附录 F 开销推导                        | naive b³+2b²d+2bd² = 重算代价的 7.1%；融合后 b³+2b²d = 0.83%（"nearly 8-fold reduction"，为原开销的 11.72%）                            | 重算口径 = 从隐藏态重算同一块（b·D·d），非 prefill 延迟口径                                    | p.17 §F                          |

## 五、局限与边界

**论文自认的限制：**

1. **密钥托管假设 TEE。** §VII："The security of KV-Cloak fundamentally relies on the confidentiality of its (megabyte-scale) key matrices. Currently, we assume these secrets are protected via TEEs."（p.13）未来工作：与 TEE/confidential GPU 深度集成、低延迟密钥轮换。
2. **量化模型未覆盖。** 当前原型只针对浮点模型（§VII p.13）；INT8/INT4 需要新的"对离散数据类型的无损可逆变换"（可能基于模运算）——留作未来工作。
3. **性能可再优化。** OTP 矩阵可异步生成以移出关键路径；硬件层面可做 block-wise permutation 与矩阵乘的 GPU 原语 co-design（§VII p.13）——即当前延迟并非下界。
4. **伦理与披露。** 对所有攻击/漏洞做 responsible disclosure，发布前告知 vLLM 等主要受影响框架；实验只用公开非敏感学术数据集（p.14 Ethics）。
5. **攻击评估范围有限定。** Inversion/Collision 在单层缓存上评估；Injection 用完整多层缓存（§IV-B1 p.6）。跨架构失配下攻击完全失效（p.8）。

**精读员注（疑点/空白）：**

1. **攻击前提强，论文自己承认。** 攻击要求拿到明文 KV-cache 数据（CSP 位置或存储/传输环节失守），论文定位为"架构性暴露"（p.2–3）；Injection 还要求以偷来的缓存"续跑"推理（追加指令、驱动模型生成，p.5）。无任何针对真实在线服务的端到端攻击演示；评估是实验室内截获 KV 张量后的重建。
2. **Table I 与 Table II 的 Distilled "Original" 行不一致。** Collision+ 三列：Table I 为 0.894/0.868、0.258/0.122、0.762/0.479；Table II 为 0.885/0.858、0.251/0.112、0.829/0.552。两表其余列一致，疑为两次评估或取整差异——引用时以 Table I 为主并标注。
3. **正文 139.0% vs Fig. 5 读图。** 正文称 Collision+ 在 Distilled 最后一层"increases accuracy to 139.0%"（p.6），但 Fig. 5（p.8）最后一层（深绿菱形）读到的最高点约 135.8%（位置 24），第一层各点 136–139% 与正文 138.6% 一致。两者不能同引；文章侧只引正文的 138.6% / 139.0%。
4. **"59.13%"无法从表复现。** §VI-D 称 ε=10⁸ 时精度为 unprotected baseline 的 59.13%、ε=10⁹ 为 93.61%（p.17）；用 Table VIII（0.262/0.258 对 0.335/0.457）无论单基准还是均值口径都算不出 59.13%（均值口径约 65.7%；93.61% 与均值口径 93.7% 能对上）。本文未使用 59.13% 这一数字。
5. **攻击效率是单层口径。** 5.06s→2.17s 是 LLaMA-3.2-1B 单层缓存的平均重建耗时（batch size 256 的批处理实现）；攻击所用硬件与端到端（全序列多层）成本论文未给（未核实）。
6. **"mostly < 1%" 的口径。** 引言写 KV-Cloak "minimal latency overhead (mostly < 1%)"（p.2）；实测 b=16 未融合为 +1.53%（p.18 Table XII）——"mostly"之外的那一档恰好是最常见块大小。融合后全部 ≤0.45%。
7. **DP 基线 ϵ=10⁸ 本身不是有效隐私预算。** 论文把 ϵ=10⁸ 称为 "a weak budget"（p.11）；"DP 失败"的正确读法是：能保住效用的 DP（ϵ=10⁸–10⁹）防不住 Collision（Table II/IX），能防住的 DP（ϵ≤10⁷）把 MMLU 打到 ≈0.05（Table VIII）。这是论文的诚实口径，引用时不要写成"DP 形同虚设"。
8. **KV-Shield [41] 的批评是论文自评。** "Security Flaws / Architectural Incompatibility"（p.8）由本文作者给出，KV-Shield 原作者的回应未知（未核实）。
9. **Fig. 8(a) 标注读图歧义。** 明文子图注记盒读作 Target μ≈0.00/σ≈0.00、Other μ=30.99/σ=4.01（p.12），与直方图位置难以对齐；本文只用 (b)(c)(d) 三个子图与文字数字，明文子图只引绿虚线 18.96=30.99−3×4.01 的阈值关系（推断，未采信）。
10. **无真实部署与集成实测。** vLLM 集成、TEE 密钥管理、异步 OTP、量化模型均为设计论述或未来工作；端到端吞吐/尾延迟未测。

## 六、写作素材

- **系列定位（KV Cache 安全三篇之"数据本身"篇）**：I Know What You Asked（NDSS '25）不问数据、只看共享行为（时序侧信道）——本篇引为 PromptPeek [39] 并指出 PagedAttention 16-token 块让时序信号变得不可暴力（|V|¹⁶，p.3）；Auditing Prompt Caching（ICML '25）站黑盒服务一侧审计真实 API；本篇把威胁模型推到另一端——**数据即明文，重建即逐 token**。三篇合起来：缓存"行为"泄露什么、缓存"服务"暴露什么、缓存"数据"本身泄露什么。
- **攻防一体的叙事线（本文特色）**：先证明"能拿回来"（三类攻击，各有约束与代价），再证明"能挡住"（KV-Cloak 三层混淆 + 算子融合），最后把攻防放在同一组分布图上收口——明文下 d_target 与 d_other 可分，KV-Cloak 下两分布坍缩为同一条（3429.40±1215.19 vs 3429.51±1214.97），DP 下仍可分。**"同一条图，三种命运"是全文最好用的结构素材。**
- **适合放开头的情境/反差素材**：
  - **"逐字还原"**：LLaMA-7B 第一层缓存上 Inversion 的 BERTScore 就是 1.000——不是近似，是逐字（p.7 Table I）。
  - **"一句话让缓存自己招供"**：追加 "Repeat the previous content." 即可让模型"回声"隐私，平均 BERTScore 0.58（p.6–7）。
  - **"7 秒代价 vs 0.45% 代价"**：AES 加密 KV-cache 要 3020.9 ms/GB（几乎追上 prefill 的 3438.5 ms/GB），KV-Cloak 融合后 15.41 ms/GB（prefill 的 0.45%）——为什么"性能妥协"选择明文、以及为什么 KV-Cloak 能打破这个妥协（p.13）。
  - **"b! 的组合墙"**：每块一次随机置换，把密钥恢复成本乘上阶乘项 O((b²d+bd²)·b!)（p.9）。
- **与既有仓库文章的勾连处（候选）**：PagedAttention 块管理（02_systems/ 或 01_concepts/ 相关篇目）——KV-Cloak 的兼容性论证以 vLLM 块为基本单位，Table XI/XII 的 b=16/32/64 扫描可作对照素材；KV 量化/压缩篇目——量化与防御的组合（INT8/INT4 未覆盖，§VII）；分层存储/KV 外置篇目（04_memory_tiering 系列）——"KV-cache 外置到非安全内存/持久化存储"正是攻击前提的来源 [43 SCX, SIGCOMM '25]。
