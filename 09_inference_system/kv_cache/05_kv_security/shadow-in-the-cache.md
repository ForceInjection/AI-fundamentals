# 你的 prompt 就存在 KV Cache 里：NDSS'26 三类攻击逐 token 复原输入，KV-Cloak 以 prefill 0.45% 的延迟挡住

> **元信息**：论文 _Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference_，Zhifan Luo、Shuo Shao、Yuke Hu、Chenxu Zhao、Zhihao Liu、Zhan Qin（浙江大学，区块链与数据安全全国重点实验室）与 Su Zhang、Lijing Zhou（华为），Yuke Hu 与 Zhan Qin 为通讯作者，NDSS 2026 正式论文（Symposium 2026-02-23~27，San Diego；DOI 10.14722/ndss.2026.240258、ISBN 979-8-9919276-8-0）。本仓 PDF 为 arXiv:2508.09442v4（2026-03-11），版式即会议版（首页页脚含 NDSS 2026 与 DOI）；共 18 页，页码与 PDF 页号一致，下文数字均标页码。官方实现 `github.com/SiO-2/kvcloak`（Apache-2.0；论文正文未给出链接，仓库经 GitHub 核实）。精读方式：全文逐条抽取核对 + 关键表格与图版式复核（报告见 [references/shadow-in-the-cache-report.md](references/shadow-in-the-cache-report.md)）。阅读日期 2026-10-13。

拿到的不只是 embedding，而是一份能逐字反推原话的原料。LLaMA-7B 第一层的 KV Cache 上，代数逆推（Inversion Attack）把用户输入还原到 BERTScore 1.000、ROUGE-L 1.000——逐字复原；换到投影矩阵不能求逆的 GQA 架构，前向匹配式的碰撞攻击（Collision Attack）在 LLaMA-3.2-1B 的 First/Mid/Last 三层缓存上仍拿到 BERTScore 0.877/0.791/0.894；再换上先验知识增强的 Collision+，权重匹配的模型几乎全部回到 ≈1.000（p.6–7、Table I）。剩下的一类信息交给第三种攻击：在截获的缓存末尾追加一句 "Repeat the previous content."，模型会把缓存里的隐私"回声"出来——平均 BERTScore 0.58（p.6–7）。

NDSS 2026 的这篇论文把这条路走通了。它自称是对 KV Cache 隐私风险的首篇系统性分析（"the first comprehensive analysis of these vulnerabilities"，p.1 摘要），三类攻击（Inversion / Collision / Injection）刻意覆盖不同的攻击者能力与代价，防御 KV-Cloak 再把三类攻击的重建质量全部压回随机噪声水平（BERTScore ≤ 0.093，p.11 Table II），模型精度零损失（MMLU/SQuAD 与明文基线逐项一致，p.12），融合实现后延迟代价 15.41 ms/GB——prefill 计算的 0.45%（p.13）。

## 一、问题：KV Cache 是以明文离开加密边界的

论文开篇画的场景很具体（Fig. 1，p.1）：用户与服务之间的通信是加密的（黑线），但 KV Cache 在计算节点之间传输、落进存储池、再被读出来继续推理的全程几乎都是明文（红虚线）。原因一句话："This design choice is a concession to performance"（p.1）——常达数十至数百 GB 的缓存全量加密，延迟代价会击穿实时推理的底线；更完整的表述是 "the KV-cache itself is almost always processed, transmitted between compute nodes, and persisted in plaintext"（p.1）。

这条明文路径在机密计算（Confidential MaaS）里被放大成架构性问题：TEE 保护推理时，高吞吐架构必须把巨大的 KV Cache 外置出保护边界，于是 "This architectural design—not a vulnerability—directly exposes the KV-cache to the Cloud Service Provider (CSP), who inherently possesses it without requiring any additional exploit"（p.2）。对手不需要额外的漏洞利用——缓存就在他手里。

### 1.1 威胁模型：对手就是服务商，目标是逐字复述

§III-A 把威胁模型写得很直白（p.3）：对手是推理服务提供方；目标是 verbatim reconstruction——逐字重建用户输入，因为那意味着可以直接提取凭证、PII 与专有逻辑。能力是两条：一是拿到服务端存储/传输的 KV Cache，论文视其为 "an intentional performance-security trade-off, not a traditional vulnerability"（p.3）；二是拿到模型权重（灰盒）——闭源服务的 CSP 天然持有，开源模型则可能因许可证披露或指纹识别 [8, 27] 被识别。唯一不在能力范围内的是 GPU 寄存器里的瞬态激活——也就是说，攻击面是"静态/在途的缓存数据"，不是运行时窥探。

### 1.2 为什么绕开侧信道也能成立

既有 KV Cache 威胁研究走的是时序侧信道（论文引为 PromptPeek [39]，即 NDSS '25 的 I Know What You Asked）；本篇指出这条路被现代内存管理削弱：PagedAttention [17] 默认 16 token 一块，要触发命中信号必须整块猜对，词表 |V| ≈ 10⁵ 时搜索空间爆到 |V|¹⁶，"rendering brute-force side-channels computationally infeasible"（p.3）。本文因此切换威胁模型：既然缓存是明文处理的，"a compromised system allows adversaries to bypass probabilistic guessing and perform precise, per-token collision attacks"（p.3）——不猜了，直接读数据。

而 KV Cache 之所以比隐藏态更危险，在于它的结构："Unlike deep hidden states that represent fused semantic information, entries in the KV-cache maintain a direct, element-wise correspondence to the tokens in the user's input sequence"（p.3）。拿到它，"does not merely obtain abstract embeddings, but potentially holds the raw materials to reverse-engineer the original user prompt"（p.3）。

## 二、机制：三类攻击 + 一层可逆混淆

### 2.1 Inversion：用权重矩阵原路算回去

最直接的攻击是代数逆推（§III-B1，p.4）：既然 `k_i = x_i·Wk^T·R^d_Θ,i`，拿到明文缓存 `k_i` 与权重 `Wk`，就能算出 `x_i = k_i·(R^d_Θ,i)^{-1}·(Wk^T)^{-1}`（式(4)，V 路对称）。它受两条硬约束："The attack strictly requires the projection matrices Wk and Wv to be square and full-rank invertible"（p.4）——只在传统 MHA（如 LLaMA-7B）成立；LLaMA-3、Qwen、DeepSeek 用的 GQA/MLA 把输入投影到低维子空间，矩阵非方阵、唯一逆不存在（ill-posed）。其次，"the attack's effectiveness is largely confined to the first decoder layer's KV-cache"（p.4）——第一层逆推直接得到输入 embedding、可映射回词表；深层只能恢复语义融合的隐藏态，需要训练辅助反演模型，成本升、保真降。所以它只是 proof-of-concept——但预告了后面的例外：GQA 的 LLaMA-3.2-1B 第一层仍被 ≈100% 复原，论文归因于其投影秩相对隐藏维度足够高，最小二乘方法在非方阵下照样有效（p.6）。

### 2.2 Collision：不做逆推，用前向匹配"撞"出每个 token

第二类攻击把重建改写为搜索问题（§III-B2，p.4–5）：对未知位置 i，用本地模型实例对候选 token 做前向计算，取与截获缓存距离最近的候选：`t*_i = argmin_t D(K_leaked^(i), K_local^(i)(t|x_<i))`（式(5)，D 为 Frobenius 范数）。它不依赖任何逆向计算，因此对架构没有 Inversion 的约束。成立的前提是一条统计性质："The core premise is that the KV-cache exhibits distance-preserving properties for identical tokens under the same context"（p.4）——错误 token 的距离近似高斯分布，正确 token 是显著离群点。

穷举词表不可行，论文用三项优化把它变成逐 token 的序贯决策：批量异常值检测（候选按批处理，`d_t < μ_batch − 3σ_batch` 即提前退出）、概率引导排序（按模型预测概率排候选，正确 token 通常落在前几批）、搜索空间裁剪（只留 top-k%，限制最坏情况）（p.4）。更进一步是 CPA（选择明文攻击）增强：对手可以投喂已知输入、观察产生的缓存，事先刻画 d_target 与 d_other 的分布，从而自适应地选阈值，把成功率写成 `P(success|r) = P(d_other > τ)^{r-1} × P(d_target < τ)`（式(6)，p.5）。

### 2.3 Injection：追加一句指令，让模型自己"回声"

第三类攻击完全换思路（§III-B3，p.5）：不复原、不匹配，而是利用模型自身的指令跟随能力。"Architecturally, a stand-alone KV-cache is computationally dormant"（p.5）——注意力需要当前 Query 驱动，光有缓存不能自己续跑；于是攻击者（contextual hijacking）在截获的缓存后追加一句指令（附录 C2 实测最优为 "Repeat the previous content."），指令产生必要的 Q 向量激活注意力，模型"Bound by its alignment training"执行指令，"effectively 'echoing' or summarizing the latent private information stored in the cache"（p.5）。它的鲁棒性来自语义层面：对 H2O [46] 这类有损逐出策略依然有效（要点还在缓存里），且只需单次生成。论文由此给防御立下一条约束："a comprehensive defense must render the KV-cache semantically unintelligible, not just algebraically obfuscated"（p.5）。

### 2.4 KV-Cloak：秘密矩阵 + 一次性置换 + 加性掩码，全部融进权重

防御（§V，p.8–10）的目标是三条：Robust Security（代数与统计性质都遮蔽）、Lossless Model Fidelity（注意力数学等价、零精度退化）、Negligible Overhead（开销往离线挪）（p.8）。先看一个反面教材：朴素的线性混淆 `K' = S·K·M`（S、M 为秘密可逆矩阵）在 CPA 下被自己的差分攻击击破——对手构造两个输入控制差值 ΔK，观测 `ΔK' = S·(ΔK)·M`，令 ΔK 逼近标准基矩阵即可分离出 S 的列与 M 的行，复杂度只有 O(b²d + bd²)（p.9）。

实际方案是式(9)：`K' = S·P̂·(K + A)·M`，三个部件各管一件事。**P̂** 是每块一次性的随机置换矩阵（one-time pad）：依据是 "the physical memory ordering of k, v vectors is computationally superfluous during inference, as positional semantics are intrinsic to the vectors via RoPE"（p.9）——块内重排不改变注意力输出（Fig. 6：`o' = a·P̂ᵀ·P̂·V = aV = o`），却给对手加上 b! 的组合复杂度，且 P̂ 用后即弃、"does not require storage"（p.9）。**A** 是加性掩码，兼作秩保险与"位置信标"：重复 token 会让缓存低秩、置换熵坍缩，A 的元素从 [3θ_K, 4θ_K] 采样（θ_K 为校准文本上 K 缓存的最大绝对值，典型 < 100），保证 (K + A) 满秩；解混淆时按高幅值离群点动态找出置换后的掩码并减掉——"This mechanism effectively offloads the storage of P̂ into the data itself"（p.9）。**S、M** 遮蔽均值、方差等统计量，使画像与数据挖掘类分析失效（p.9）。密钥恢复复杂度被抬到 O((b²d + bd²)·b!)（p.9），b 取 vLLM 的标准块大小 {16, 32, 64}。

剩下的开销用算子融合消掉（§V-D，p.10）：再引入两个秘密可逆矩阵 M1、M2 改写 q/k/v 与输出权重（`q^m = q(M1^{-1})ᵀ`、`k^m = k·M1`、`v^m = v·M2`、`Wo^m = Wo(M2^{-1})ᵀ`，式(10)），注意力分数与输出数学不变（式(11)(12)）。融合的前提是最微妙的一处："The fusion is algebraically feasible if and only if the secret matrix M1 and the RoPE matrix commute"（p.10）——于是 M1 被结构化设计成 2×2 旋转-缩放子矩阵拼成的块对角矩阵（附录 A 给出交换性证明），刚好与 RoPE 的旋转结构同构。四个权重离线预计算（式(15)），在线只剩 `K' = S·P̂·(K^m + A)`（式(16)），每块约 b³ + 2b²d 次浮点运算。到最后，与 vLLM 的兼容性只依赖一条设计原则：所有密码学操作 "self-contained within a single physical memory block"（p.18）——块与块之间零依赖，内存管理器可以照常调度、拷贝、换出、共享物理块而无需理解内容（§VI-G）。

## 三、评估：攻防两侧的数字

实验覆盖 7 个模型：LLaMA-7B、LLaMA-2-7B（MHA）、LLaMA-3.2-1B/3B-Instruct、LLaMA-3.1-8B、Qwen2.5-Math-7B（GQA）、DeepSeek-R1-Distill-LLaMA-8B（即 LLaMA-3.1-8B-Distilled，用于微调场景）；数据取 LMSYS-Chat-1M 随机 1,000 条；指标为 BERTScore（all-mpnet-base-v2）与 ROUGE-L（p.5 §IV-A）。

**攻击侧（§IV-B，p.6–8）**。Inversion 的边界清晰：LLaMA-7B 第一层 BERTScore 1.000/ROUGE-L 1.000（≈100% 逐字），深层塌到 <10% 的噪声（Mid 0.065/Last 0.092）；GQA 模型的 8B 与 3B 版本第一层就失效（0.071/0.055），Qwen2.5-Math-7B 0.229，而 LLaMA-3.2-1B 因高投影秩成为例外（1.000/0.994）。Collision 是全场主力：LLaMA-3.2-1B 三层 0.877/0.791/0.894（BERTScore），微调失配（攻方只拿 base 权重）下仍有 0.642（First），跨架构失配（用 LLaMA-7B 攻 LLaMA-2-7B）则全部跌回 ≈0.06–0.09、随机猜测水平（"< 0.1"，p.8）——攻击吃的是权重参数的相关性。统计上，3σ 阈值下假阳率 0.13%、per-token 成功率 91.84%（r=64），换成统计推导的阈值后到 100%；微调模型从 91.79% 升到 97.82%（p.6）。效率上，只搜 top 1/8 的词表就能保住 96.1% 的保真度，单层平均重建从 5.06 s 降到 2.17 s（<43%）；1/16 是 90.9%/1.62 s，1/4 是 98.4%/2.76 s（p.7，Fig. 4）。先验知识增强的 Collision+ 在权重匹配的模型上（假定 rank r=8）几乎全部 ≈100%；对微调模型，第一层提升到基线的 138.6%（实际平均 rank ≈ 4）、最后一层 139.0%（对齐 64≤r≤128），中间层无改善——说明该模型中间层的实际 rank > 128（p.6）。数据集泛化（Alpaca/GSM8K）上 Collision 第一层 0.942/0.890（p.17 Table VII）。Injection 平均 BERTScore 0.58 / ROUGE-L 0.42，峰值出现在 LLaMA-7B（0.765/0.687）——论文的解释是 MHA 缓存比 GQA 压缩版保留更高的上下文保真、语义上更"可读"（p.6–7）。数据需求上两类攻击互补：Collision 拿到任意单层就够，Injection 必须全层完整缓存（p.7）。

**防御侧（§VI，p.10–13）**。安全评估把攻击逐个打在受保护缓存上（Table II，p.11）：KV-Cloak 下所有攻击输出的 BERTScore 最高只有 0.093、ROUGE-L 几乎为 0——例如 LLaMA-7B 的 Inversion 从 1.000 掉到 0.091，Distilled 的 Collision+ 从 0.885 掉到 0.041。对照组 DP 则是"留了门"：同样配置下 Collision 第一层还有 0.849，Collision+ 在 LLaMA-3.2-1B 最后一层甚至回到 1.000。更直观的证据是距离分布图（Fig. 8，p.12）：KV-Cloak 下 d_target 与 d_other 两条分布完全重叠（3429.40 ± 1215.19 对 3429.51 ± 1214.97），统计可分性消失，攻击成功率 0%；而 (10⁷, 10⁻⁵)-DP 下两条分布（37.09 ± 1.05 对 48.41 ± 2.79）仍可分，配合先验阈值 per-token 成功率 94.84%。精度上（Table III，p.12），6 个模型的 MMLU/SQuAD 在 KV-Cloak 下与明文逐项一致（如 LLaMA-3.1-8B 0.668/0.708 → 0.668/0.709），而 DP 全面崩塌（LLaMA-7B 掉到 0.016/0.000，8B 掉到 0.283/0.026）。DP 的困境有完整曲线：LLaMA-3.2-1B 在 ε=1/10/10⁷ 时 MMLU 只有 ≈0.05，ε=10⁸ 才回到 0.262、ε=10⁹ 回到 0.299（明文 0.335）——能保住效用的预算防不住 Collision（10⁷ 下 Collision 仍有 >55% 语义相似度），防得住的预算精度归零（p.11、p.17）。

性能开销是本篇最接近"免费"的部分（Fig. 9，p.13）：同一块 LLaMA-3.1-8B 上，prefill 延迟 3438.5 ms/GB，AES 加密要 3020.9 ms/GB、几乎追上 prefill 本身；KV-Cloak 未融合 52.73 ms/GB，融合后 15.41 ms/GB——prefill 的 0.45%。块大小扫描（Table XII，p.18）显示融合版在 b=16/32/64 下分别是 +0.45%/+0.30%/+0.36%，全部 <0.45%；精度不受块大小影响（MMLU 0.335 恒定，SQuAD 0.460–0.463，p.18 Table XI）。附录 F 给出口径更宽的推导：朴素实现（b³+2b²d+2bd²）是"从隐藏态重算同一块"代价的 7.1%，融合后 0.83%，约 8 倍降幅（p.17）。密钥存储是 898 KB（LLaMA-3.1-8B、b=16；公式 2·n_kv·l·(b² + 3/2·d + 1)），1T 参数的 Ling-1T 上也只有 20.9 MB（p.13）——MB 级密钥正好塞进 TEE 的有限内存。

## 四、边界：这套攻防在什么条件下成立

论文自认的第一条边界关乎防御的根：**KV-Cloak 的安全性吊在密钥上**。"The security of KV-Cloak fundamentally relies on the confidentiality of its (megabyte-scale) key matrices. Currently, we assume these secrets are protected via TEEs"（p.13）——密钥失守，整套混淆归零；因此密钥轮换、与 TEE/confidential GPU 的深度集成被列为未来工作。第二条是 **量化模型未覆盖**：当前原型只面向浮点模型，INT8/INT4 需要为离散数据类型另造无损可逆变换（可能基于模运算，p.13）——主流 KV 量化的粒度与误差量级见本仓 [KV Cache 量化深度解析](../01_concepts/compression/kv_cache_quantization.md)。第三，性能数字还不是下界：OTP 矩阵可异步生成移出关键路径、块置换与矩阵乘可做 GPU 原语 co-design（p.13）。

更需要读者自己掂量的，是**攻击面前提的强度**——这也是论文最坦诚的地方。三类攻击都要求对手拿到明文的 KV Cache 数据（CSP 位置，或存储/传输环节失守），Injection 还要求以偷来的缓存"续跑"推理（追加指令、驱动模型生成，p.5）；论文不把它写成传统漏洞，而是写成性能与安全的结构性交换："intentional performance-security trade-off, not a traditional vulnerability"（p.3）。因此它的现实性取决于一个组织问题：你是否接受"云服务商能看到你的缓存明文"。与之相比，I Know What You Asked 那一类侧信道攻击者只是同集群的普通租户——前提更弱，但信号也弱得多（|V|¹⁶ 的块级猜中率把暴力式侧信道压到不可行，p.3）。同一条威胁轴上的两端，值得放在一起读。

精读员注补充几点。其一，**评估是实验室内截获 KV 张量后的重建**：没有对任何真实在线服务的端到端攻击演示，攻击基础设施的 GPU 型号论文未给（未核实）。其二，**攻击效率是单层口径**：5.06 s → 2.17 s 是 LLaMA-3.2-1B 单层缓存的平均重建耗时（batch 256 的批处理实现），攻击所用硬件与端到端成本论文未给。其三，**两处表格/图文小出入**已在精读报告第五节列出（Table I 与 Table II 的 Distilled Collision+ 三列数字不完全一致；正文 139.0% 与 Fig. 5 最后一层读到的约 135.8% 不能同引），本文数字均以正文与 Table I 为主。其四，引言的 "mostly < 1%"（p.2）指的是延迟开销：实测未融合版在 b=16 时是 +1.53%（p.18 Table XII）——恰好是最常见块大小那一档，融合后全部 ≤0.45%，这也是全文推荐引用的口径。其五，"DP 挡不住"的准确读法见上节：不是"DP 无效"，而是**有效隐私预算与可用精度在当前方案下不可兼得**；论文自己也把 ε=10⁸ 称为 "a weak budget"（p.11）。最后，KV-Shield 的两处批评（固定置换可被统计击穿、与 RoPE 不兼容）来自本文作者的分析（p.8），KV-Shield 原作者的回应本文未涉及（未核实）。

## 论文结构索引

| 论文位置           | 内容                                                                                                                    |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| §I / p.1–2         | 动机：KV Cache 明文外置的性能妥协、TEE/MaaS 的架构性暴露；RQ1/RQ2；三类攻击与 KV-Cloak 概览；"mostly < 1%" 开销声明      |
| §II / p.2–3        | 背景：Transformer 前向与 RoPE、自回归；隐私攻击分类；侧信道在 PagedAttention 下的不可行性（\|V\|¹⁶）                     |
| §III-A / p.3       | 威胁模型：对手=CSP；目标=逐字重建；能力=KV Cache 访问 + 模型权重（灰盒）；"架构性暴露，非漏洞"                            |
| §III-B1 / p.4      | Inversion Attack：式(4)；方阵满秩 + 第一层两条约束；GQA/MLA 下 ill-posed；深层需反演模型                                  |
| §III-B2 / p.4–5    | Collision Attack：式(5) 前向匹配；距离保持前提；三项优化（批异常值/概率排序/裁剪）；CPA 增强式(6)                         |
| §III-B3 / p.5      | Injection Attack：缓存"computationally dormant"；contextual hijacking；"Repeat the previous content."；对 H2O 鲁棒       |
| §IV-A–B / p.5–7    | 攻击评估：7 模型、LMSYS-Chat-1M 1,000 条、BERTScore/ROUGE-L；Table I、Fig. 3（距离分布）、Fig. 4（裁剪效率）、Fig. 5（Collision+ 阈值） |
| §IV-C / p.7–8      | 鲁棒性：微调失配（有效）vs 跨架构失配（<0.1 失效）；单层 vs 全层的数据需求                                                |
| §V-A–B / p.8–9     | 防御动机：crypto/HE、DP、KV-Shield 的缺陷；三条设计目标；朴素 `K'=SKM` 及其 CPA 差分破解                                   |
| §V-C / p.9         | KV-Cloak 核心：式(9) `S·P̂·(K+A)·M`；OTP 块内置换（b! 屏障）；掩码 A 的秩保险与零存储恢复；O((b²d+bd²)·b!)                |
| §V-D / p.10        | 算子融合：M1/M2 变换式(10)–(12)；RoPE 交换性约束（块对角 2×2 旋转-缩放）；离线式(15)、在线式(16)；在线成本 b³+2b²d        |
| §VI-A–B / p.10–12  | 防御评估配置与安全：Table II（KV-Cloak vs DP）；Fig. 8 分布坍缩与 0% 成功率                                               |
| §VI-C–D / p.12–13  | 精度（Table III：零损失）与开销（Fig. 9：15.41 ms/GB）；存储开销（898 KB / 20.9 MB）                                      |
| §VII–VIII / p.13   | 讨论（TEE 密钥假设、密钥轮换、异步 OTP、硬件 co-design、量化模型）与结论                                                   |
| Ethics / p.14      | 责任披露：发布前告知 vLLM 等框架；实验仅用公开学术数据集                                                                  |
| 附录 A–H / p.15–18 | RoPE 交换性证明；校准文本；攻击消融（阈值/批大小/指令/数据集，Table IV–VII）；DP 参数选择（Table VIII–IX）；其余模型防御（Table X）；开销推导（附录 F）；PagedAttention 兼容（Table XI–XII） |

## 与系列其他篇目

KV Cache 安全三篇各从一个位置切入。[I Know What You Asked](i-know-what-you-asked.md)（NDSS '25）不看数据本身，从多租户共享 KV Cache 的调度行为（LPM 造成的请求返回顺序，不测绝对延迟）恢复他人 prompt——本篇把它引为 PromptPeek [39]，并论证 PagedAttention 的 16 token 块让这类侧信道的块级猜中率落到 |V|¹⁶、不再可暴力（p.3）；[Auditing Prompt Caching](auditing-prompt-caching.md)（ICML '25）站在黑盒一侧，审计真实商业 API 的 prompt caching 行为；本篇则把威胁模型推到另一端——攻击者直接拿到 KV Cache 数据（服务商位置或存储/传输失守），数据即明文，逐 token 复原，不需要任何侧信道。三篇合起来恰好回答同一个问题的三种问法：缓存的"行为"泄露什么、缓存的"服务"暴露什么、缓存的"数据"本身泄露什么。横切总纲见 [KV Cache 安全总纲](kv-security-overview.md)。

## 诚实声明

- **本文数字全部转录自论文，未独立复测**；攻击与防御均为论文自建实验（单机、公开数据集），非第三方复现。
- **攻击前提强于侧信道**：三类攻击都要求拿到明文 KV Cache 数据，Injection 还需以偷来的缓存驱动模型推理（p.3–5）；论文将此定位为"架构性交换"而非漏洞。无真实在线服务的端到端攻击演示。
- **性能口径**：15.41 ms/GB（0.45%）出自 Fig. 9 的串行 PyTorch 微基准、无自定义 CUDA kernel，模拟 prefill-decode 拆分；论文自称保守上界口径。附录 F 的 7.1% → 0.83% 以"从隐藏态重算同一块"为分母，与 0.45% 的 prefill 分母不是同一笔账，不可混引。
- **两处论文内部出入**（详见[精读报告](references/shadow-in-the-cache-report.md)第五节）：Table I 与 Table II 的 Distilled Collision+ 三列数字不一致；正文 139.0% 与 Fig. 5 读图约 135.8% 冲突，本文只引正文。另有一处 "59.13%" 无法从表格复现，未采用。
- **未核实项**：官方代码仓库为 GitHub 侧核实（论文正文未给链接）；攻击实验的 GPU 型号、vLLM 真实集成、TEE 密钥管理与量化模型（INT8/INT4）支持均无实测数据。KV-Shield 的缺陷评估是本文作者的分析。
- 论文 venue：本仓 PDF 为 arXiv:2508.09442v4，首页页脚已含 NDSS 2026 会议信息（DOI 10.14722/ndss.2026.240258），即会议版式；本文按此表述。

## 参考资料

- Zhifan Luo, Shuo Shao, Su Zhang, Lijing Zhou, Yuke Hu, Chenxu Zhao, Zhihao Liu, Zhan Qin. _Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference_. NDSS 2026. 本地 PDF：[references/papers/shadow-in-the-cache-arxiv-2508.09442.pdf](references/papers/shadow-in-the-cache-arxiv-2508.09442.pdf)；[arXiv:2508.09442](https://arxiv.org/abs/2508.09442)；[DOI](https://dx.doi.org/10.14722/ndss.2026.240258)
- 官方实现：[github.com/SiO-2/kvcloak](https://github.com/SiO-2/kvcloak)（Apache-2.0）
- 精读报告（含逐条页码核对与疑点清单）：[references/shadow-in-the-cache-report.md](references/shadow-in-the-cache-report.md)
