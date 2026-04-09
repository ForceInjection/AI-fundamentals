# 大模型 KV Cache 压缩技术详解：原理、架构与趋势

## 引言

大语言模型正以前所未有的速度推高上下文窗口的上限，然而上下文每增加一倍，KV Cache 的显存占用也随之线性翻番。以 70B 参数的模型配合 128K 上下文为例，仅 KV Cache 的内存开销就约 50–60GB，已然超越了模型权重本身。对云端推理服务而言，这意味着每张 GPU 能同时处理的请求数量被严重挤压，显存、带宽与延迟构成了三重约束。

正因如此，KV Cache 的压缩，已不再是可选项，而是决定大模型服务成本与可扩展性的核心命题。

本文系统梳理当前 KV Cache 压缩技术的全貌，分析各类方法的原理、效果与局限，并在此基础上谨慎展望 2026 年的技术趋势。本文的核心观点可概括为以下四个方面：

- **四维冗余**：KV Cache 存在 Token、Feature、Structure、System 四个维度的冗余，构成了所有压缩技术的理论基础。
- **不可能三角**：生产环境中，压缩率、精度与吞吐量难以同时兼顾，需基于具体任务动态权衡。
- **主流落地方案**：当前产业界倾向于“Token 剪枝 + 混合量化 + 存储卸载”的三位一体组合。
- **未来趋势**：极低比特（3-bit 及以下）量化、内核感知压缩（Kernel-aware Compression）、事前评估指标的标准化，以及从“物理存储”向“实时预测”的范式演进。

---

## 一、规模瓶颈与四维冗余模型

理解 KV Cache 压缩的第一步是解构其成因。本节从基础计算公式出发，分析其呈爆炸性增长的核心驱动力；在揭示生产环境中“压缩率、精度与吞吐”不可兼顾的铁律（不可能三角）之后，我们基于注意力机制的内部特性将冗余解构为四个维度（Token、特征、结构、系统）。这四个维度不仅勾勒出了 KV Cache 的内在结构特征，也是后续分类所有压缩算法和演进路线的导航坐标。

### 1.1 KV Cache 何以成为瓶颈

Transformer 的每层自注意力机制需要将每个 token 投影为 Key 向量和 Value 向量。为了解决自回归生成中的重复计算问题，模型通常采用 KV Cache 技术，将已计算过的历史 token 的 K 和 V 缓存下来，供后续每一步解码直接复用。其显存占用可通过以下公式估算：

> $\text{Memory}_{KV} \approx 2 \times b_{kv} \times L \times B \times S \times N_{kv} \times d_{head}$
>
> _(注：$b_{kv}$ 为单个数值的字节数（如 FP16 为 2）；系数 2 代表同时缓存 Key 和 Value 矩阵、$L$ 为层数、$B$ 为批处理规模、$S$ 为上下文序列长度、$N_{kv}$ 为 KV 头数、$d_{head}$ 为每个头的特征维度。若模型采用 GQA/MQA 架构，$N_{kv}$ 将小于 Query 头数 $N_{attn}$，整体缓存规模随之大幅缩减。)\_
>
> 据此可估算系统最大并发请求数（理论上限）：$B_{max} \approx \frac{M_{GPU} - M_{model}}{M_{KV\_per\_sample}}$，其中 $M_{GPU}$ 为 GPU 总显存，$M_{model}$ 为模型权重占用，$M_{KV\_per\_sample}$ 为单个请求（给定上下文长度下）的 KV Cache 占用。压缩 $M_{KV\_per\_sample}$ 将直接等比例提升 $B_{max}$，从而推高系统吞吐上限。

该公式揭示了 KV Cache 显存占用的严峻挑战：动态增长的缓存数据由多个维度共同驱动——更深的网络、更宽的隐层、更长的上下文序列、更大的并发请求数（Batch Size）。推高其中任何一个维度，KV Cache 的显存需求都会呈线性膨胀，成为制约系统吞吐量的核心瓶颈。此外，PagedAttention 论文指出，在未优化的传统连续内存分配方式下，有效利用率仅 20%–38%，内存碎片化与预分配浪费显著，进一步加剧了显存压力。

> **Prefill 与 Decode 的非对称瓶颈**  
> 在分析 KV Cache 压缩的实际收益之前，需要区分推理的两个阶段：Prefill（预填充）阶段一次性处理全部输入 token，其计算量为 $O(S^2)$，属于 **compute-bound**；而 Decode（自回归生成）阶段每步仅处理一个新 token，计算量为 $O(S)$，但需反复从显存读取完整的 KV Cache，因此属于 **memory-bound**。这意味着，KV Cache 的压缩对 Decode 阶段的吞吐与延迟改善最为直接——压缩后的 KV Cache 减少了每步生成的访存量，直接缓解了内存带宽瓶颈。理解这一差异，是评估各类压缩方法实际工程收益的前提。

### 1.2 三角约束

KV Cache 压缩在生产环境中普遍面临内存缩减（压缩率）、精度保持与计算效率（推理延迟与吞吐量）的三元约束（常被称为“不可能三角”），这一约束在高压缩率与长上下文场景下尤为尖锐[1]。高压缩率往往难以同时兼顾精度与效率：

1. **隐藏的精度缺陷**：很多研究通过计算整体准确率掩盖了特定场景下的性能退化。进一步针对单个样本维度的评测表明，压缩 KV Cache 会引发“负样本（Negative Samples）”问题——即原本在全量 KV 下能够正确回答的问题，在压缩后由于关键上下文的丢失或失真，导致模型出现严重的幻觉或错误。
2. **框架适配与吞吐损耗**：现有的压缩算法实现通常未能针对生产级系统（如 PagedAttention 或 FlashAttention）进行底层深度优化。这导致虽然内存占用的绝对值下降了，但由于引入了复杂的压缩/解压算子，计算延迟反而增加。此外，被压缩的 KV Cache 可能导致模型生成冗余或重复内容，从而意外增加了端到端的总延迟。

因此，不存在普适的最优方案，唯有基于具体任务的准确度要求和硬件条件，在三者间寻求动态权衡。

### 1.3 四维冗余模型

KV Cache 的规模计算公式在揭示爆炸性增长的同时，也指明了潜在的优化切入点：序列长度（Token）、隐层维度（Feature）、网络深度（Structure）以及物理显存的布局方式（System）。我们通过剖析这四个基础维度的结构特性与运行机制，归纳出 KV Cache 的四维冗余模型：

- **Token 维冗余**：并非所有 token 对未来生成同等重要，许多中间 token 可以安全删除（如 SnapKV [2]、H2O [3] 等研究的观察）。
- **特征维冗余**：隐层维度中存在大量低秩结构和数值精度冗余（如 KIVI [4]、KQ-SVD [5] 等工作的验证）。
- **结构维冗余**：不同层之间的 KV 具有高度相似性，可以跨层共享；此外还存在头间冗余，即不同注意力头之间可能共享信息——GQA 与 MQA 等架构设计正是对头间冗余在训练阶段的主动利用，而 CommonKV [6]、xKV [7] 等方法则在推理阶段进一步对层间与头间结构进行联合压缩。
- **系统维冗余**：内存分配方式和存储位置选择存在极大的优化空间（如 PagedAttention [8]、ShadowKV [9] 等系统级实践）。

后续各技术路径之所以分化，根本原因就在于它们分别锚定了上述不同的冗余维度作为压缩主战场。

---

## 二、KV Cache 压缩技术全景

基于第一章所构建的四维冗余模型，近年来学术界与工业界涌现出了大量针对特定维度优化的 KV Cache 压缩技术。本章将对这些技术路径进行全景式的梳理：从简单粗暴的 Token 剪枝，到榨取数值精度的量化；从深度网络特征降维的低秩分解，到系统架构层面的内存分页与 CPU/存储卸载。我们通过剖析各流派的代表性工作，展示它们在“压缩率、精度、吞吐”这个不可能三角中作出了何种权衡。

### 2.1 Token 维压缩：Pruning 与 Eviction

Token 维压缩是最直接的方法——识别并删除重要性低的 token 的 KV 对，从根本上削减 KV 序列的长度。

早期方法主要依赖注意力分数作为重要性判据：SnapKV 通过观察窗口内的注意力模式选择保留 token，H2O 将累积注意力权重作为 token 重要性的代理指标，StreamingLLM [10] 则发现保留首尾 token 可以有效保持长文本性能。NVIDIA 的 KVzap [11] 进一步将这些思想工程化落地，在 2×–4× 压缩率下基本实现了无损。

2025 年，该方向迎来了方法论层面的重要升级。OBCache [12] 将 KV Cache 驱逐问题形式化为一个层级结构化剪枝问题，借鉴最优脑损伤理论，通过度量 token 剪枝在注意力输出中引起的扰动来精确量化 token 显著性。与传统方法仅依赖累积注意力权重不同，OBCache 的评分同时考虑了 Key、Value 和注意力输出的联合信息，提供了一种更具原则性的剪枝策略。同时期，Lookahead Q-Cache [13] 通过生成低成本伪前瞻查询来预判 token 重要性，在 LongBench 上获得了 1–4 个百分点的准确率提升。

此外，业界还涌现出了更多细粒度的控制方案：例如，DynamicKV 与 DBudgetKV 引入了动态预算分配机制；Lethe 探索了层自适应的压缩策略（在推理密集型任务中实现了 2.56× 的吞吐量提升）；SparK 则实现了查询感知的非结构化稀疏与可恢复的通道级剪枝。

Token 维压缩的优势在于延迟低、工程可落地，其根本局限在于信息的不可逆丢失——一旦剪枝，被删除 token 的信息便永久消失。

> **一句话本质**：Token 维压缩的本质是通过重要性筛选，用最少的信息丢弃换取最大的显存释放。

### 2.2 精度维压缩：量化与低精度表示

如果说 Pruning 做的是“减法”，那么量化做的就是“压缩”——通过降低 KV 的数值精度（从 FP16 到 INT4/INT2 甚至更低），在不改变序列长度的前提下大幅削减显存占用。

量化技术的发展经历了从“一刀切”到“差异化对待”的演进。早期的均匀量化对所有 token 和所有层施加相同的量化策略，简单但精度损失明显。QAQ [14] 发现了 Key 与 Value 在量化敏感度上的显著差异，据此对二者施以不同策略。KVC-Q 引入了 recency-aware 和 head-aware 的自适应机制，在约 70% 的显存缩减下保持了 94% 以上的精度。AQUA-KV 通过动态调整量化参数进一步提升鲁棒性，Oaken 探索了在线-离线混合量化范式，UQ-KV Cache 则提供了一个支持 1-bit 量化的混合精度统一框架。此外，KV-AdaQuant 提出了一种不对称的位分配策略（对 Keys 多分配、对 Values 少分配），进一步优化了量化资源的利用效率。

此外，向量量化（Vector Quantization）在精度维度上提供了离散化压缩的另一条路径。例如，VecInfer [15] 通过异常值抑制策略实现了低比特 KV Cache 的向量量化表示，有效降低了高维特征的数值存储开销。

2026 年 3 月，Google Research 的 TurboQuant [16] 在 ICLR 上正式发表，将量化推至一个引人注目的新高度。TurboQuant 的核心技术组合包括 PolarQuant 和 QJL：PolarQuant 通过随机旋转使高维向量的几何结构变得更规整，便于后续高保真量化；QJL 利用 JL 引理在低维空间保持向量距离，配合 PolarQuant 的旋转归一化以实现 3-bit 高效编码。最终成果是：KV Cache 被压缩至 3-bit 精度，内存需求压缩约 6×，推理速度提升最高达 8 倍，且在 H100 上的 LongBench 等评测中实现了“零精度损失”（与 FP16 相比无统计显著退化）。在 Llama-3.1-8B-Instruct 的 LongBench 测试中，TurboQuant 展现出了相较于各主流压缩方法的稳健性能优势。

需要审慎指出的是，极低比特量化的“零损失”声明存在场景依赖性。TurboQuant 在中低端推理芯片上的效果与 4-bit 相比优势并不明显，3-bit 是否已逼近量化极限仍是一个待解的开放问题。KIVI [4] 在 2-bit 非对称量化上的探索，以及 RaBitQ [17] 在高维向量量化理论方面的贡献，均为极低比特 KV Cache 压缩铺平了道路。

> **一句话本质**：精度维压缩的本质是用数值表示的粒度换取存储空间，其边界由模型对噪声的容忍度决定。

### 2.3 特征维压缩：低秩分解

特征维压缩的核心在于利用 KV 在隐层维度上的冗余——通过奇异值分解（SVD）等降维技术，提取具有主导作用的特征子空间，用更紧凑的低秩表示替代原始的高维稠密向量。这种方法在不删减序列长度的前提下，从特征表示的维度“榨取”了空间。

2025 年下半年，低秩压缩方向迎来数个突破性进展。KQ-SVD [5] 提出直接对完整的注意力矩阵做最优低秩分解。其核心理论洞察在于：由于注意力计算本质上是 Q 与 K 的内积，如果仅孤立地对 K 做低秩分解而忽略了与 Q 的耦合效应，将不可避免地导致次优的重建误差。KQ-SVD 通过闭式解实现了全局最优分解，为低秩压缩提供了严谨的可证明保真度保证。

另一方面，考虑到 Key 和 Value 在注意力计算中的角色差异，ReCalKV [18] 提出了一种极具工程参考价值的非对称压缩策略。对于 Key，ReCalKV 引入了头相似性感知重排序（Head-wise Similarity-aware Reordering），将结构相似的注意力头聚为若干组后再执行分组 SVD，以此降低低秩近似的损失；对于 Value，则引入了离线校准（Offline Calibration）机制，利用少量无标注数据预先优化投影矩阵。在高达 8× 的压缩比下，ReCalKV 依然能够维持极低的性能退化。同时，xKV [7] 进一步打破了层间壁垒，探索了跨层 SVD 共享，将多层 KV Cache 映射合并到一个统一的共享低秩子空间中。

此外，基于残差或 Delta 编码的方案（如 DeltaKV）同样聚焦于特征维度，通过仅存储局部变化量来进一步降低显存开销。

需要清醒认识的是，低秩方法虽然在理论上限上表现优异，但其计算开销（尤其是自回归生成过程中高频且高复杂度的动态 SVD 分解矩阵计算）仍是工程落地的最大现实瓶颈。如何将离线投影与在线轻量级更新结合，是该方向未来走向生产级的关键挑战。

> **一句话本质**：特征维压缩的本质是利用高维向量的低秩结构，在子空间中紧凑地保留主导信息。

### 2.4 结构维压缩：跨层共享

结构维压缩的逻辑非常简洁：既然 Transformer 的中深层网络所提取的 KV 特征具有高度相似性，为什么不跨层合并存储？

MiniCache [19] 率先探索了这一方向。其基于相邻层 KV Cache 状态高度相似的观察，提出了一种沿着网络深度（Depth Dimension）进行跨层合并的压缩策略，通过保留关键层的完整 KV 并让相邻层共享近似表示，大幅减少了总存储量。CommonKV [6] 则更进一步，提出了一种仅需少量校准数据的跨层参数共享方案，通过权重共享来融合跨层的 KV Cache，在 LongBench 和 Ruler 等多个基准测试中，即使在 50% 的高压缩率下依然能保持超过 95% 的性能。xKV [7] 同样立足于跨层相似性，但它将其与低秩子空间框架结合，实现了多层 KV 的联合 SVD 压缩。

该方向的核心理论可概括为：**KV ≈ 低秩骨干 + 局部变化残差**——即通过层间共享一个共同的低秩或共享主干，各层仅需额外存储少量的差异化残差信息，从而巧妙地在结构维度释放了显存空间。

> **一句话本质**：结构维压缩的本质是将层间共享信息提取为公共骨干，各层仅保留差异化的“残差”。

### 2.5 系统维压缩：Offloading、Paging 与 Prefix Caching

系统维压缩的特殊之处在于它并不直接改变 KV 数据的数学内容（如精度或数量），而是从计算机体系结构的视角出发，优化 KV Cache 的“居住方式”（内存布局）和“访问模式”（I/O 调度）。

PagedAttention [8] 是这一范式的奠基性里程碑。vLLM 团队观察到，由于 LLM 文本生成的不可预测性，传统的连续内存预分配会导致高达 60%–80% 的内部和外部碎片浪费。通过将 KV Cache 拆分为固定大小的物理 Block（例如包含 16 个 token 的 KV），并引入类似操作系统虚拟内存的 Block Table 映射机制，PagedAttention 用逻辑连续替代了物理连续，从根本上消除了内存碎片化问题，使显存利用率飙升至接近 100%。此后，TGI、TensorRT-LLM、FlashInfer 等主流推理引擎都围绕“分页/分块 KV Cache”展开了底层 Kernel 和架构适配。

GPU-CPU 协同卸载（Offloading）则将冷热分层的存储理念引入了 KV Cache 管理。2024 年，字节跳动提出的 ShadowKV [9] 采用了巧妙的非对称“影子”机制：将经过低秩压缩后的 Key Cache 保留在高带宽的 GPU HBM 上作为“热数据影子”（Key 在 GPU 侧以低秩近似形式留存）用于快速注意力筛选，而将庞大的、全精度的完整 Value Cache 卸载（Offload）至低成本的 CPU 内存作为“大本营”。在注意力计算时，模型仅按需通过 PCIe 将选出的稀疏 Value 数据传回 GPU 重建。在 A100 GPU 上（针对特定模型与长上下文设置），ShadowKV 实现了高达 6× 的批处理规模扩展和 3.04× 的吞吐量提升，甚至在 RULER 等长文本基准下，其性能表现超越了在同等调度约束下、假设显存无限时的全量缓存批处理基线。

Prefix Caching（前缀缓存）是另一个在系统层被广泛采纳的 I/O 优化策略。由于多轮对话（Multi-turn Chat）、System Prompt 或检索增强生成（RAG）等场景中存在大量重复的前缀输入，系统可以将这部分已计算的 KV Cache 进行哈希签名并跨请求共享。当新请求到来时，若命中缓存，即可直接跳过重复的 Prefill（预填充）计算阶段，显著降低了 Time To First Token（TTFT）。例如 SGLang 的 RadixAttention 就将这一机制做到了极致。此外，如 LMCache 这样的层级缓存管理系统，也为系统维度的缓存调度提供了高性能的解决方案。

### 2.6 混合方法

单一维度的压缩终究有其天花板（例如，剪枝会导致不可逆的信息丢失，而极低比特量化则存在精度崩塌的风险），因此跨越多个维度的联合优化（Hybrid Compression）正在成为实际生产环境中的必然演进方向。

LeanKV [20] 是混合路线的典型代表，它敏锐地捕捉到了模型中普遍存在的“三层差异化”特性并以此构建了一个统一的无需重新训练（Training-free）的压缩框架：首先是 **KV 的不对称性**，即 Key 和 Value 对注意力输出的贡献度截然不同；其次是 **Token 的不均等性**，少数核心 Token 承载了大部分上下文语义；最后是 **Attention Head 的动态稀疏性**，不同注意力头在处理特定模式时表现出多样的稀疏激活行为。实验表明，LeanKV 可实现 2.7×–5.7× 的 KV Cache 压缩，吞吐量提升 1.9×–5.4×，且在复杂推理和长文本生成任务中保持了近乎无损的精度。另一方面，Compactor [21] 则基于近似杠杆分数（Approximate Leverage Scores），通过精确衡量不同 KV 对对最终注意力矩阵近似程度的相对重要性，在同样无需训练的条件下实现了 LongBench 评测集上高达 68% 的显存缩减。

混合方法的理论收益毋庸置疑，但工程复杂度也随之呈指数级增加。将多种压缩算子（如 Token 剪枝决策、混合精度量化解码、非对称反量化机制）融合在一起，并深度集成到现有的高效推理框架（如 vLLM/TensorRT-LLM 的 PagedAttention 内核）中，不仅需要极高的 Kernel 编写技巧，还要在运行期动态调度计算与访存以防止“压缩带来的解码延迟掩盖了显存节省的红利”。这也是混合框架目前仍主要停留在“早期工程化”阶段的核心原因。

### 2.7 技术效果对比一览

为了更直观地评估上述各类压缩路径的实用价值与系统代价，下表横向对比了目前主流方法在压缩率、精度保持、额外开销及工程成熟度等维度的核心指标。可以看出，追求极致压缩率（如低比特量化与低秩分解）往往会引入不可忽视的计算/通信开销，而成熟落地的方案（如 Token 剪枝与内存分页）则更多是在规避运行时负担与容忍一定的信息丢失之间寻求平衡。

| 方法类别             | 代表性工作       | 压缩率           | 精度保持           | 额外开销                  | 工程成熟度 | 主要局限       |
| -------------------- | ---------------- | ---------------- | ------------------ | ------------------------- | ---------- | -------------- |
| Token Pruning        | SnapKV / H2O     | 2×–4×            | 高                 | 极小（轻量 Prefill 计算） | 成熟       | 信息不可逆丢失 |
| Output-aware Pruning | OBCache          | 2×–5×            | 高（输出感知）     | 计算开销                  | 研究       | 计算开销       |
| Quantization         | KVC-Q / QAQ      | 3×–8×            | 94%+               | 解码计算开销              | 较成熟     | 极低比特退化   |
| Ultra-low-bit Quant. | TurboQuant       | ≈6×（相对 FP16） | 零损失（特定场景） | 预处理开销+解码计算开销   | 实验室     | 场景局限性     |
| Low-rank             | KQ-SVD / ReCalKV | 2×–8×            | 高                 | Prefill 端 SVD 分解开销   | 研究       | SVD 计算开销   |
| Cross-layer          | CommonKV / xKV   | 2×–3×            | 高                 | 无                        | 研究       | 架构侵入性     |
| Offloading           | ShadowKV         | 3×–6×            | 高                 | PCIe 通信开销             | 较成熟     | CPU-GPU 延迟   |
| Hybrid               | LeanKV           | 2.7×–5.7×        | 高                 | 综合开销                  | 早期       | 工程复杂度高   |

---

## 三、四层压缩模型

基于上述技术全景的分析，可以将散落在各篇论文中的优化思路提炼为一个统一的、自上而下的四层压缩模型。该模型不仅是本文的核心理论贡献，也为系统架构师在实际业务中进行技术选型提供了清晰的坐标系。

### 3.1 层级架构

为了系统化地理解和实施压缩策略，我们将压缩技术抽象为四个自上而下的结构层次。该架构能够指导从最顶层的序列缩减到底层的硬件存储调度的全方位优化。

四层模型按照数据流向和系统抽象级别，自上而下划分为以下层级：

```text
Level 1: Token-Level Compression（Token 层压缩）
    ↓
Level 2: Feature-Level Compression（特征层压缩）
    ↓
Level 3: Structure-Level Compression（结构层压缩）
    ↓
Level 4: System-Level Compression（系统层压缩）
```

### 3.2 各层映射

下表详细展示了这四个层级对应的具体压缩对象、代表性核心方法以及相应的性能折中情况，为实际工程选型提供直接参考。

| 层级             | 压缩对象            | 核心方法                                | 代表性工作                                           | 压缩率范围 | 精度代价       |
| ---------------- | ------------------- | --------------------------------------- | ---------------------------------------------------- | ---------- | -------------- |
| **Token 层**     | KV 序列长度         | Pruning / Eviction                      | SnapKV [2], H2O [3], StreamingLLM [10], OBCache [12] | 2×–5×      | 低–中          |
| **Feature 层**   | 隐层维度 / 数值精度 | Quantization / Low-rank / Vector Quant. | KIVI [4], TurboQuant [16], KQ-SVD [5], ReCalKV [18]  | 3×–10×     | 低             |
| **Structure 层** | 层间冗余            | Cross-layer Sharing                     | MiniCache [19], CommonKV [6], xKV [7]                | 2×–3×      | 低             |
| **System 层**    | 存储位置 / 访问模式 | Offloading / Paging / Prefix            | PagedAttention [8], ShadowKV [9]                     | 3×–6×      | 无（延迟代价） |

### 3.3 核心洞察

四层之间并非彼此孤立，而是存在着强烈的互补关系。其中，Token 层（数量缩减）与 Feature 层（精度/维度缩减）的联合优化可以产生显著的乘数效应。总压缩率 $C_{total} \approx C_{token} \times C_{feature} \times C_{structure} \times C_{system}$，但需注意各维度压缩的**相互作用**（例如，Token 剪枝后序列长度客观缩短，这会导致后续量化所能带来的“绝对显存节省量”相应缩减）。因此，在混合框架设计中必须综合评估二者的交互边际收益。与此同时，Structure 层和 System 层则完全是在不改变单个 KV 块数学内容的前提下，进一步通过共享或外存置换来释放高昂的 GPU HBM 空间。

**所有方法本质上都是在消除模型运转不同维度的冗余。由于四个维度各有侧重、优劣互补，未来的最优方案必然是场景驱动的多维混合。** 例如：

- **高并发短文本场景（如 API 问答）**：System 层的 Paging（内存分页）和 Prefix Caching（前缀共享）通常是最具性价比的手段；
- **超长上下文场景（如 RAG 或文档分析）**：由于上下文规模占据绝对主导，采用 Token 层 Pruning 配合 Feature 层低比特量化往往是必选项；
- **平台型多模型路由服务**：由于面临频繁的模型切换，Structure 层的跨层共享或统一低秩投影提供了极具弹性的、模型无关的压缩潜力。

---

## 四、系统视角：工程落地的现实约束

技术论文中的压缩率和精度曲线再漂亮，最终都要接受底层推理框架（Inference Framework）和 GPU Kernel 执行效率的现实考验。本章将从工程落地的视角，探讨压缩算法如何与系统解耦与融合。

### 4.1 与推理框架的耦合

主流的生产级推理系统已经深度集成了针对 KV Cache 的底层优化，任何新的压缩算法都必须与这些既有架构进行适配：

- **vLLM**：以 PagedAttention 为核心消除内存碎片，支持 FP8 等低精度 KV Cache 格式，并在 0.11.0 版本正式引入了分层卸载（Offloading）功能。
- **TensorRT-LLM**：追求极致的 Kernel 融合与量化深度集成，通过 C++ 层面的定制化算子提供端到端的优化链路。
- **SGLang**：利用 RadixAttention 实现了基于基数树的精细化前缀缓存复用（Prefix Caching）。
- **ShadowKV [9]**：采用“Key 低秩存储 + Value 卸载”的异构模式，在 A100 GPU 上实现了超越内存物理上限的理论批处理性能。
- **LMCache**：专注于高性能的层级缓存管理，支持多种存储后端的跨请求缓存复用。
- **H2O / InfiniGen**：结合了 Token 剪枝启发式算法（Eviction Heuristics）与张量卸载技术。

### 4.2 三种集成模式

将压缩算法集成到上述框架中，通常面临三种工程模式的选择与权衡：

- **内联压缩（Inline Compression）**：量化、剪枝与注意力 Kernel 深度融合。压缩、计算与解压缩在同一次 Kernel 调用中在 GPU SRAM 内完成。这种模式延迟极低，但需要极其复杂的 CUDA/Triton 算子重写，工程门槛最高。
- **核外压缩（Out-of-core Compression）**：将温冷状态的 KV Cache 卸载至 CPU 内存或 SSD 并进行高压缩比存储，仅在注意力计算时按需加载至 GPU。该模式非常适合超长上下文场景，但会引入显著的 PCIe 数据搬运延迟。
- **混合压缩（Hybrid Compression）**：这是一种分级存储策略——热 KV 在 GPU 保持高精度、温 KV 在 GPU 内降精度存储、冷 KV 卸载至 CPU。这是实际生产中最具弹性和性价比的折中方案。

### 4.3 系统关键约束

KV Cache 的逻辑内存布局（Layout）通常是 `[L, H, T, D]`（层数、头数、序列长度、特征维度）的四维张量结构，但在 PagedAttention 等机制下，其物理布局因分页而碎片化，常被重构为 `[num_blocks, block_size, num_heads, head_dim]` 的块状结构。在评估压缩方法的落地可行性时，必须考虑以下关键的系统级约束：

1. **Kernel 融合难度**：压缩粒度（如 Per-token 还是 Per-channel 量化）是否能与底层注意力计算的矩阵分块大小对齐？如果解压缩过程不能在极快的 SRAM 内顺滑完成，而是需要频繁访问全局显存（Global Memory），压缩带来的速度红利将被访存延迟彻底抹平。
2. **分页对齐问题**：在 PagedAttention 机制下，Token 是按固定大小的 Block 分页存储的。如果剪枝算法（Pruning）随机删除了某些 Token，将导致物理 Block 内部出现空洞，进而破坏了连续的内存读取效率。目前业界的折中方案是在 Block 粒度上进行淘汰（Block-level Eviction），但这会牺牲细粒度的 Token 选择精度。
3. **编解码延迟**：以 TurboQuant-vLLM 插件为例，其核心设计思路是直接从页表读取压缩后的 KV，其核心设计是在 SRAM 中完成解压缩与注意力计算的一体化处理。反观部分基于复杂编码（如哈夫曼编码）的 KVTC 压缩方法，由于其 Encode/Decode 延迟与 Attention Kernel 存在本质的融合困难，短期内难以成为主流。

---

## 五、2026 技术趋势预测

随着长文本大模型（如 1M+ Context 的 Gemini 1.5 Pro）的普及与边缘端 AI 算力的演进，KV Cache 的管理范式正面临深刻重构。基于前文的技术梳理与业界最新的研究动态，我们预判了 2026 年 KV Cache 压缩领域的七大核心技术演进趋势。

### 5.1 趋势一：量化进入“3-bit 时代”

TurboQuant [16] 已证明 3-bit 无损量化的可行性，4-bit → 3-bit 的迁移将在 2026 年加速成为端侧和云端推理的标准配置（值得注意的是，2025 年末多家厂商如 NVIDIA、AMD 已在硬件层增加对 INT4/FP4 的原生支持，为极低比特软件模拟提供了参考）。然而，我们需要审慎看待“零损失”声明的泛化能力——极低比特量化的精度保持存在着对模型架构（如 MQA 与 MHA 的差异）、具体下游任务（如代码生成 vs 常规问答）以及硬件平台特性的三重依赖。从 3-bit 迈向 2-bit 甚至 1-bit（如 UQ-KV Cache 所探索的极限），是否还能在长文本寻回任务中维持可接受的精度，仍然是一个悬而未决的开放问题。

### 5.2 趋势二：混合压缩成为实际部署首选

正如第 2.6 节所述，Pruning + Quantization + Offloading（剪枝 + 量化 + 卸载）的三位一体方案正在成为工程实践的标准范式。LeanKV [20]、Compactor [21] 等混合框架已经展示了从 2.7× 到接近 10× 的综合压缩比潜力。但也必须承认，混合方案增加了极大的系统复杂度，未来的推理引擎（如 vLLM）需要提供更加模块化、可插拔的 Cache Manager，以允许开发者在特定应用场景下仔细权衡工程投入与性能收益。

### 5.3 趋势三：低秩方法从理论研究走向工程验证

KQ-SVD [5] 的可证明保真度保证和 ReCalKV [18] 的差异化策略，为低秩压缩奠定了坚实的理论基础；xKV [7] 的跨层 SVD 共享则展示了低秩方法在结构维度的扩展潜力。自回归生成时的 SVD 矩阵乘法计算开销依然是瓶颈，但“离线特征预处理配合在线轻量级更新”的范式有望在 2026 年取得工程突破，从而将其推向生产线。

### 5.4 趋势四：KV Cache 演变为分层存储系统

随着单次推理动辄 GB 级别的缓存需求，从 GPU HBM 到 CPU DRAM 再到 NVMe SSD 的分层存储架构正在成型。Hot KV 以高精度驻留 GPU、Warm KV 在 GPU 内降精度存储或卸载至 CPU、Cold KV 经压缩编码后存入 SSD——这样的三级存储体系已经具备初步的工程可行性。冷热分层引入的 PCIe/CXL 数据传输延迟仍需通过“预取策略（Prefetching）”与“计算与通信重叠流水线（Overlap）”来智能缓解，但分层存储的方向已不可逆转。

### 5.5 趋势五：系统与算法深度融合

ShadowKV [9] 的“Key 低秩存储 + Value 卸载”异构模式，以及 TurboQuant-vLLM 插件的端到端集成设计，清晰地表明了一个趋势：最优的 KV Cache 压缩不再是纯粹的学术算法推导问题，而必须在算法设计阶段就充分考虑底层 Kernel 融合、SRAM 内存布局对齐和底层硬件带宽瓶颈。**Kernel-aware Compression（内核感知压缩）**将成为 2026 年一个重要且主流的设计范式。

### 5.6 趋势六：Benchmark 与评测体系标准化

当前主流评测基准包括 LongBench（综合长上下文评测）、RULER（极长上下文评测）和 Needle In A Haystack（针尖寻回测试，NIAH）。2026 年 2 月，KV-CoRE [22] 提出了首个系统性评估 KV Cache 数据依赖低秩可压缩性的基准框架，通过 Normalized Effective Rank（归一化有效秩）等指标在事前评估（a priori evaluation）不同模型和数据集对低秩压缩的适应潜力。评测体系的逐步标准化与统一化，将为各类压缩方法的横向盲测比较提供更可靠的“试金石”。

### 5.7 趋势七：从“压缩”到“预测”的范式探索

一个更长远、更根本的问题是：KV Cache 是否一定要被物理存储？是否可以通过学习模型的隐状态流形（Manifold）来实时预测或生成 KV？这是一个将 KV Cache 问题从工程上的“数据压缩”升维到理论上的“表示学习（Representation Learning）”的范式转换。需要明确的是，此方向目前仍为概念性前瞻，尚无成熟实证支撑（仅有诸如 Draft & Verify 等推测解码框架中对于 KV 复用的零星探索）。但这一方向彻底颠覆了自回归计算的基础，无疑是下一个十年最值得期待的方向。

---

## 六、关键挑战与未解决问题

尽管 KV Cache 压缩技术在过去两年取得了长足进步，但在追求极致性能与理论上限的道路上，仍有诸多核心难题悬而未决。本章将探讨当前研究的盲区与未来亟需突破的系统级挑战。

### 6.1 精度与压缩率的上限

是否存在信息论意义上的理论最优压缩率（如借鉴率失真理论，探索给定任务失真容忍度下的最小比特率）？不同任务、不同模型对压缩的容忍度差异巨大，如何建立统一的压缩边界理论？例如，虽然 TurboQuant [16] 展示了 3-bit 甚至探索了更低比特的潜力，但 3-bit 是否已逼近实用极限？这些问题尚无定论。

### 6.2 硬件适配性

同一压缩方法在不同 GPU 架构上的 Kernel 效率可能相差数倍。随着 AI 芯片呈现多样化（如 NVIDIA Blackwell、AMD MI300X 以及各类专用 NPU），是否应该在算法设计阶段就考虑底层指令集与硬件约束？答案越来越倾向于肯定，Hardware-Algorithm Co-design（软硬协同设计）将是不可回避的挑战。

### 6.3 KV 表示的可重构性

从“物理存储 KV”到“学习 KV”再到“预测 KV”，这条路径上的每一步都蕴含着对现有 Transformer 架构的根本性挑战。如果有朝一日 KV 状态可以被一个轻量的辅助网络（如推测解码中的 Draft Model）实时生成，那么传统意义上的物理 KV Cache 本身将不复存在。

### 6.4 与注意力机制的深度耦合

是否需要重新设计“Compression-aware Attention（压缩感知注意力）”？传统的注意力计算假设 KV 以全精度、全序列的形式存在。当 KV 已经被剪枝、被量化甚至被低秩投影时，标准的 Softmax 注意力机制本身是否还有更好的、适配压缩状态的数学表达方式？

### 6.5 标准化缺失

无统一的评测基准、无统一的压缩 API、不同框架间方法难以横向比较——这些问题正随着 KV-CoRE [22] 等工作的出现而逐步改善，但距离真正的标准化仍有相当距离。随着 2025 年 MLCommons 开始讨论将 LLM 推理纳入 MLPerf，KV Cache 压缩有望成为新的评测维度，但如何建立一套标准的评测套件，仍是产业界亟待解决的问题。

---

## 七、总结

KV Cache 压缩，本质上是一个多维冗余消除的系统工程——序列维（Token）、特征维（Feature）、结构维（Structure）和系统维（System），这四个维度各有其独特的冗余模式与对应的压缩策略。

当前产业界的主流落地方案已初步收敛为“Token 剪枝 + 混合量化 + 存储卸载”的三位一体组合。而学术界的前沿探索则聚焦于 3-bit 甚至更低精度的无损量化边界、低秩分解的理论深化，以及内核感知压缩（Kernel-aware Compression）的工程化落地。没有一个单一的算法能够统治所有推理场景，基于任务特征和硬件约束的多维混合优化才是唯一的最优解。

| 典型场景                       | 核心瓶颈                    | 推荐压缩组合                     | 预期收益             |
| :----------------------------- | :-------------------------- | :------------------------------- | :------------------- |
| 高并发短文本问答（API 服务）   | 并发数 × 短序列的 KV 碎片化 | PagedAttention + Prefix Caching  | 吞吐提升 2×–4×       |
| 长文档分析 / RAG（1M context） | 单个请求的序列长度          | Token Pruning + 3-bit 量化       | 显存缩减 4×–8×       |
| 多轮对话（带长历史）           | 历史轮次累积的冷 KV         | Token Eviction + CPU Offloading  | 显存缩减 3×–6×       |
| 边缘 / 端侧部署                | 总显存严格受限              | 2-bit 量化 + 跨层共享            | 使大模型在端侧可运行 |
| 多模型路由服务                 | 模型切换时 Cache 重建开销   | 统一低秩投影（Structure 层压缩） | 切换延迟显著降低     |

展望 2026 年，几个值得重点关注的关键信号包括：TurboQuant 等极低比特量化方案能否在多样化下游任务中大规模落地、跨层 SVD 方法能否跨越矩阵重构计算开销的工程门槛、以及 KV-CoRE 等评测基准能否真正推动行业的横向评测标准化。而从更宏大的周期来看，“大模型推理是否可以不存储物理 KV”这一根本性问题，或许才是颠覆自回归计算范式、引领下一个十年的星辰大海。

**一句话概括：KV Cache 正在从“大模型推理的被动中间状态”演化为“可压缩、可分层、可调度的主动数据管理系统”。**

---

## 参考文献

[1] Wei Gao et al., "Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving," arXiv preprint arXiv:2503.24000, 2025.

[2] Yuhong Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation," arXiv preprint arXiv:2404.14469, 2024.

[3] Zhenyu Zhang et al., "H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models," arXiv preprint arXiv:2306.14048, 2023.

[4] Zirui Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," arXiv preprint arXiv:2402.02750, 2024.

[5] Beheshteh T. Rakhshan et al., "KQ-SVD: Compressing the KV Cache with Provable Guarantees on Attention Fidelity," arXiv preprint arXiv:2512.05916, 2025.

[6] Haojie Wang et al., "CommonKV: Compressing KV Cache with Cross-layer Parameter Sharing," arXiv preprint arXiv:2508.16134, 2025.

[7] Xinyu Liu et al., "xKV: Cross-Layer SVD for KV-Cache Compression," arXiv preprint arXiv:2503.18893, 2025.

[8] Woosuk Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," arXiv preprint arXiv:2309.06180, 2023.

[9] Yichao Zhong et al., "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference," arXiv preprint arXiv:2410.21465, 2024.

[10] Guangxuan Xiao et al., "Efficient Streaming Language Models with Attention Sinks," arXiv preprint arXiv:2309.17453, 2023.

[11] Simon Jegou et al., "KVzap: Fast, Adaptive, and Faithful KV Cache Pruning," arXiv preprint arXiv:2601.07891, 2026.

[12] Yunhe Wang et al., "OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference," arXiv preprint arXiv:2510.07651, 2025.

[13] Yixuan Wang et al., "Lookahead Q-Cache: Achieving More Consistent KV Cache Eviction via Pseudo Query," arXiv preprint arXiv:2505.20334, 2025.

[14] Shichen Dong et al., "QAQ: Quality Adaptive Quantization for LLM KV Cache," arXiv preprint arXiv:2403.04643, 2024.

[15] Dingyu Yao et al., "VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization," arXiv preprint arXiv:2510.06175, 2025.

[16] Amir Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate," arXiv preprint arXiv:2504.19874, 2025.

[17] Jianyang Gao et al., "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search," arXiv preprint arXiv:2405.12497, 2024.

[18] Xianglong Yan et al., "ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration," arXiv preprint arXiv:2505.24357, 2025.

[19] Akide Liu et al., "MiniCache: KV Cache Compression in Depth Dimension for Large Language Models," arXiv preprint arXiv:2405.14366, 2024.

[20] Yijiang Li et al., "LeanKV: A Unified Framework for KV Cache Compression," arXiv preprint arXiv:2407.12345, 2024.

[21] Yunqi Li et al., "Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores," arXiv preprint arXiv:2507.08143, 2025.

[22] Jian Chen et al., "KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs," arXiv preprint arXiv:2602.05929, 2026.
