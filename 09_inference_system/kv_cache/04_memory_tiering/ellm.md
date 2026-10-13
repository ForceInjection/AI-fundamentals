# 30.8% 的显存按 200K 上下文预留，实际只用 1%：eLLM 让 Activation 与 KV Cache 互相借内存

> **元信息**：论文 _eLLM: Elastic Memory Management Framework for Efficient LLM Serving_，上海交大 + 上海期智研究院 + 中科大 + NUS 等（Jiale Xu、Yi Xiong、Rui Zhang 共同一作，通讯作者 Cong Guo、Junping Zhao、Zongwei Zhu），DAC '26 论文集 8 页，arXiv:2506.15155v2。精读方式：全文逐页通读（8 页）逐条核对，数字标论文页码。阅读日期 2026-10-12。

在 vLLM 里显存快满时会发生什么：新请求进不来，先排队，等跑着的请求把 KV Cache 释放出来（p.2）。队列等的是 KV Cache 的空间。同一时刻，GPU 上另一块空间可能正闲着：按最大上下文长度预留、准备给激活（activation）用的那块。两个内存池互不认识，一个爆满、一个闲置，中间没有通道。

论文给的极端例子：MTP-30B 的最大上下文从 2K 调到 200K，同一批 A100 上为激活预留的显存占比从 0.3% 涨到 30.8%（p.2）。而真实流量长这样：ShareGPT 上 99% 的 prompt 不超过 2K token；超过 90% 的 prefill batch 用不到模型最大上下文的 30%；到 decoding 阶段，每步只处理少量 token，激活空间利用率约 1%（p.2）。上下文越拉越长（论文点名 CoT 拉长了解码周期），这 1% 的闲置就越是「主导」整个推理过程。

预留按最坏情况算，使用按典型情况来，中间的空间就沉淀成了碎片。上海交大团队的 eLLM 把 activation 和 KV cache 从两套内存抽象里解放出来，放进同一个物理内存池，谁紧张就向对方借：inflate 借入、deflate 归还，实在不够再把 KV 溢出到 CPU DRAM。论文的系列收益：解码吞吐最高 2.32×（p.6）。

## 一、一道不能互借的墙

两套抽象各自出现时都是对的。activation 继承 PyTorch 等框架的静态张量抽象，直接分配物理内存块、用 memory pool 复用（p.2）；KV cache 走 PagedAttention 的页表虚拟化，初始化时预留好池子、按请求动态发块（p.2）。问题出在两者叠在一起之后：抽象层级不同，谁也不认识另一个的账本。原文写得很直接：

> the two memory spaces cannot share resources, even when the KV cache space is exhausted while the activation memory sits idle (or vice versa), the underutilized space cannot be repurposed.（p.2）

论文把这种现象命名为 **space-wise internal fragmentation**（空间维度的内部碎片）。摘要里给出了系统级代价：「leading to suboptimal utilization and 20% throughput degradation」（p.1）；Figure 2 用两组内存快照做对照，vLLM 的图里 activation 和 KV 两块被标成 "Isolated!"，换成 eLLM 后同样负载多装下约 20% 的吞吐（图注记为 1.2× speedup，p.3）。

这里有个容易混的数字：论文里「20%」出现两次、含义不同。摘要的 20% 是碎片造成的吞吐损失（问题侧），Figure 2 的约 20% 是弹性互借收回的吞吐（收益侧）。引用时别串了。

论文还逐一回应了「交给调度层解决」的已有方案：Chunked Prefill 把请求切块压低激活峰值，代价是 read amplification：每个新 chunk 要重读此前全部 KV，内存开销随序列长度超线性增长；PD 分离（prefill/decode 拆到不同 GPU）降低解码侧激活预留，但论文援引 semi-PD 的总结列出代价，包括模型权重复制、利用率倾斜、跨实例 KV 传输（p.2）。eLLM 的姿态是绕开这些既有取舍：直接在内存管理层消除空间内部碎片，「avoiding their limitations and remaining orthogonal to them」（p.2）。

统一两条内存抽象也有两条看着更省事的路，论文都否了（p.3）：把 KV 交给框架做细粒度管理，元数据开销大、还会破坏 KV 的共享结构（prefix 复用依赖块级结构）；做完全不区分类型的统一虚拟化，则抹掉了两类张量访问模式的差别，没法各自优化。

## 二、eTensor：逻辑分开，物理归一

eLLM 的抽象叫 **eTensor**：KV cache 和 activation 在逻辑层是两个独立实体，各自保留定制化管理；物理层共享一个统一内存池，池内块可以互相转手（Figure 3，p.3）。

![eLLM：eTensor 统一内存池与 inflation / deflation](../../assets/mtier-ellm-elastic-pool.svg)

### 2.1 两类张量，两套虚拟化

**KV eTensor** 面对的是大而规整、增长可预测、访问频度低、全程驻留的张量。它的虚拟地址空间按每个请求的**最大上下文长度**一次性预留（保证逻辑连续，顺序访问和索引计算都简单），物理块则按实际写入 on-demand 分配，虚拟预留不花物理钱。**activation eTensor** 是另一头：小块、生命周期短、访问频度高，用变长虚拟地址段、细粒度管理。两类张量的虚拟地址段（tensor slots）都严格按物理块粒度对齐，论文的说法是在访问效率和碎片控制之间取平衡（p.3）。

### 2.2 池管理：不急着 unmap，KV 用 Best-Fit

针对虚拟化本身的开销，池管理做了两个选择（p.3）。其一，eTensor 生命周期结束后**不立即解除映射**，而是标记为 mapped 且 available 的 tensor slot、记录映射尺寸，留给后来的请求复用。其二，KV eTensor pool 用 **Best-Fit** 分配：在可用槽集合里选能满足目标尺寸 s 的最小槽位（`arg min{size(r_j) | size(r_j) ≥ s}`），选不出就现场 on-demand 映射；activation eTensor pool 则保留 PyTorch 原生的 **BFC**（Best-Fit with Coalescing）策略。

## 三、弹性内存：膨胀、收缩与 CPU 后备

### 3.1 Inflation：四步借内存

当 KV 池不够用时，eLLM 向 activation 池「借」物理块，整个过程像给气球充气（论文原话 "much like inflating a balloon"，p.4），分四步走（p.4）：

1. **Inflation trigger**：KV 分配请求先查 KV pool，空间不够就向 activation pool 发借用请求；
2. **Memory reclamation**：activation pool 触发一轮轻量 GC，把不活跃 eTensor 对象占的物理块解除映射；
3. **Ownership transfer**：回收的块在逻辑上从 activation pool 转手给 KV pool——只改所有权，不搬数据；
4. **On-demand remapping**：GPU 虚拟内存管理器（VMM）把这些块映射进目标 KV eTensor 的虚拟地址空间。

deflation 是逆过程，把块还给 activation，且**惰性触发**（lazy），不是每步都做，免得付无谓开销。Figure 4 用四个步骤演示了一遍：历史 KV 占满 → prefill 请求到来，inflation 借闲置块立刻开跑 → 负载高时 inflate 出更大 batch → decoding 收尾后 deflate 归还（p.4）。

### 3.2 GPU-CPU 弹性：CPU DRAM 当后备池

第二级弹性把 CPU DRAM 拉进来当溢出 buffer（p.4）。prefill 阶段，系统主动把一部分请求的 KV cache 卸到 CPU，降低请求执行的内存准入门槛，排队时间缩短、TTFT 改善；顺带能聚合出更大的解码 batch。论文给了三条可行性依据：新生成、当下用不到的 KV 可以先卸后取；Transformer 的多层结构天然支持逐层流水（layer-wise pipelining）把传输和计算叠起来；KV 迁移是 O(N) 的线性通信量，而 prefill 的 self-attention 是 O(N²)，量级差摆在那，传输藏得进计算。论文称在 A100 + LLaMA-3-8B 的真实负载下，「offloading overhead can be completely hidden」（p.4）。

代价论文自己认：CPU buffer 改善排队，但引入对 prefill 的偏好、伤 TPOT，所以必须配一套折衷策略（见第四节）。

### 3.3 把映射开销藏进计算

虚拟化的账要还，eLLM 在实现层藏了两笔（p.4–5 §3.1）。**投机预映射**：自回归每序列每迭代只产 1 个 token，就按分支投机（branch speculation）的思路假定它还会接着产出，提前异步发起内存分配，用不到 50MB 的额外内存换来映射开销与解码计算的完全重叠。**异步 unmap**：GPU VMM 支持把一个物理块映射到多个虚拟地址，于是新 tensor slot 可以立刻接手物理块，旧槽位的解除映射异步地慢慢做。净效果见第五节：CPU 调度开销低于 1%、VMM 操作开销 1–5%。

## 四、SLO 感知调度：让弹性跟着延迟指标走

弹性机制负责「能借能还」，调度负责决定「何时借、借多少」。

### 4.1 分阶段调度

activation 的需求在推理两阶段差异巨大，eLLM 就按阶段分配（p.4）：prefill 阶段 activation 和 KV 需求都大，把 KV 卸到 CPU buffer，给 GPU 腾地方；decoding 阶段 activation 需求极小，把 KV fetch 回 GPU，把腾出来的利用率吃满。两个阶段都靠 inflation/deflation 把内存利用率顶到上限，从而容纳更多并发请求。

### 4.2 logical buffer：不逐出数据的「缩容」

CPU buffer 的大小直接决定 TTFT 和 TPOT 之间的取舍。eLLM 的解法是 **logical buffer**：物理容量固定，可用尺寸动态调整，关键是缩容时**不逐出已存数据**。缩放规则写得很朴素（p.4）：检测到 TPOT 违反 SLO，就把 logical buffer 调小，限制 prefill 请求的进入、把 TPOT 拉回来；检测到 TTFT 违反，就调大 buffer 改善 TTFT。违反的判定是「一个调度窗口（5 个 iteration）内超标 3 次」，调整因子 α 是超参，默认取 2。

## 五、评估：收益有多大

测试台是一台 8×NVIDIA A100 (80GB) NVLink 服务器，配 Xeon 8369B 和 1TB RAM；软件栈 PyTorch 2.4.0 + CUDA 12.4 + vLLM v0.5.5，eLLM 本体是 vLLM v0.5.5 上的约 4000 行 C++/Python 修改（KV Cache tensor 被整个替换、最小分配粒度从 page 改成 physical memory chunk、PyTorch 分配器改用 GPU VMM）（p.5）。SLO 按 LoongServe 口径定义为无竞争延迟的 25×，goodput 定义为达到 90% SLO attainment 的最大请求速率（p.5）。三个模型各有代表性：Llama3-8B-262K（GQA、支持 Chunked Prefill）、Jamba（Mamba-MoE 混合架构、暂时不支持 Chunked Prefill）、OPT-13B（MHA、配 PD 分离基线 DistServe）。

- **在线单卡（Figure 5，Llama3-8B-262K，1×A100）**：TTFT 最高比 vLLM 快 **295×**、比 vLLM-CP 快 140×（p.5）；这个倍数的来源是 vLLM 在请求速率升高后 TTFT 撑不住、排队时间爆炸，eLLM 的曲线则被弹性内存压平。goodput 最高 **2.5×**（vs vLLM）与 2.26×（vs vLLM-CP）（p.5）。
- **在线多卡（Figure 6，OPT-13B，2 GPU）**：对照 DistServe（P=1, D=1，其余系统 TP=2），eLLM 吞吐高 1.5×、goodput 高 **2.5×**。论文对 DistServe 的两点批评：prefill 与 decode 各占专用 GPU，对应阶段空闲时算力闲置；权重在两种实例上各存一份，挤掉了解码 batch（p.6）。
- **离线（Figure 7）**：全部请求在实验开始时到达。128k-4k 配置（Jamba-Mini，2 GPU）下总吞吐最高 1.82×、**解码吞吐最高 2.32×**；同一配置读图可见 Max Batch Size 归一化约 3×（正文未给数字）。收益在长序列和高「activation-to-KV 比」架构（如 Jamba）上最明显：vLLM 的静态分配在那种形状下内存瓶颈最严重。Figure 7 右图（Llama3-8B-262K，1 GPU）的增幅就温和得多。
- **消融（Figure 8）**：只用 intra-GPU 弹性（vLLM+intra）和只用 GPU-CPU 弹性（vLLM+inter）各自都能压低 TTFT，组合起来最高降 295×，且 TPOT 稳定性不掉（p.6）。
- **开销分解（§4.6）**：CPU 调度开销低于总执行时间的 1%，VMM 操作开销占 1–5%（在线场景）。论文归功于三项优化：eTensor 池化复用映射、解码投机预映射、异步 unmap（p.6）。

## 六、边界与代价

eLLM 的收益不是免费的，论文把这些账目也列了出来。

**TPOT 要还账。** §4.2 原文：「because eLLM and vLLM-CP have more available KV Cache space for decoding, their TPOT is higher than vLLM.」（p.5）eLLM 用 TTFT/TPOT 上的让步，换来更高的 SLO 达成率与 goodput。§2.2.2 里也承认 CPU buffer 会引入 prefill-preference（p.4）。

**PCIe 带宽是天花板。** 消融显示两种弹性组合后，吞吐有时反而低于只用 intra-GPU 弹性，原因是 decoding 阶段的 PCIe 传输无法完全重叠（p.6）。反过来只开 intra-GPU 弹性也不行：高负载下 prefill 阶段的瓶颈挡不住，goodput 会退化（p.6）。eLLM 的对策是让双机制按负载动态编排，平衡「GPU 内互借」的收益与「CPU 后备」的成本。

**前提条件。** GPU-CPU 弹性依赖 CPU DRAM 容量（测试台给了 1TB）与 PCIe 带宽；「传输开销完全隐藏」的结论也是 A100 + LLaMA-3-8B 这一组负载下的实测（p.4）。换更慢的互连或更大的模型，账要重算。

评估覆盖面的边界（精读员注）：只有一台 8×A100 测试台；GPU-CPU 弹性只跟 vLLM 自身比，没对照 LMCache、Mooncake 这类专用 KV offload 系统；精度与能耗没有报告（纯内存管理理论上不影响精度，但论文没有给数据）。

## 论文结构索引

| 论文位置        | 内容                                                                                                                                    |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| §1 / p.1–2      | 动机：双层抽象割裂与 "space-wise internal fragmentation" 命名；MTP-30B 0.3%→30.8%、ShareGPT 99%≤2K、decode 激活利用率 ∼1%；CP 与 PD 分离的代价（Fig.1、Fig.2） |
| §2 / p.2–3      | 三组件架构总览（eTensor → 弹性内存 → 轻量调度）；弹性互借约 20% 吞吐改善（Fig.2）                                                        |
| §2.1 / p.3      | eTensor：KV/activation 两类虚拟化设计；tensor slots 对齐物理块；Best-Fit 与 BFC 池管理（Fig.3）                                          |
| §2.2.1 / p.3–4  | inflation 四步（触发→GC→所有权转移→重映射）与 deflation；四状态示例（Fig.4）                                                             |
| §2.2.2 / p.4    | GPU-CPU 弹性：prefill 阶段 offload；三条依据（新 KV 可后取、层流水、O(N) vs O(N²)）；prefill-preference 代价                            |
| §2.3 / p.4      | 轻量调度：分阶段分配；logical buffer 与 SLO 缩放（α=2、5 iteration 内 3 次违反触发）                                                     |
| §3.1 / p.4–5    | (un)map 开销重叠：decoding 投机预映射（<50MB）与异步 unmapping                                                                          |
| §3.2 / p.5      | 实现：vLLM v0.5.5、~4000 行；page→physical chunk 粒度；GPU VMM 替换 cudaMalloc                                                          |
| §4.1–4.2 / p.5  | 实验配置；在线单卡：TTFT 295×/140×、goodput 2.5×/2.26×（Fig.5）                                                                         |
| §4.3 / p.5–6    | 在线多卡 vs DistServe：1.5× 吞吐、2.5× goodput（Fig.6）                                                                                 |
| §4.4–4.5 / p.6  | 离线 1.82×/2.32×（128k-4k，batch 读图≈3×）；消融 intra/inter 与 PCIe 反例（Fig.7、Fig.8）                                              |
| §4.6 / p.6      | 系统执行时间分解：CPU 调度 <1%、VMM 操作 1–5%（三项优化）                                                                                |
| §5–6 / p.6      | 相关工作（GMLake、vAttention 的定位）与结论；参考文献 p.7–8                                                                              |

## 与系列其他篇目

eLLM 是「KV Cache 内存层级」系列的**抽象虚拟化**路线：不动搬运路径、不加新硬件层，只在同一块 HBM 内部把 activation 与 KV cache 两套账本并成一本，靠 inflation/deflation 动态调额度。同系列的 [DirectKV](directkv.md)（OSDI '26）走**零拷贝去搬运**：在 GH200/GB200 的 NVLink-C2C 上把 staging buffer 干掉，正好是 eLLM 仍要付 PCIe 账单的那部分成本；[NEO](neo.md)（MLSys 2025）走**计算下移**：把 attention 连同 KV 一起沉到 CPU 侧，让 KV 不必回传，与 eLLM「搬回 GPU、用 O(N) 传输藏 O(N²) 计算」的选择正面对照；[Strata](strata.md)（OSDI '26）走**分层调度**：在 GPU/CPU/SSD 的层与层之间做缓存感知调度，与 eLLM 的池内弹性正交、可叠加。四条路线的横向对照与数字口径注解见总纲《[当 KV 装不进显存：四篇论文的四种内存账，与一个「统一命名空间 + 成本旋钮」的设想](memory-tiering-overview.md)》。

## 诚实声明

- 本文数字全部出自论文原文并按页码标注；写作者没有做任何复现或第三方交叉验证。论文未开源（全文无仓库链接），无法核对实现细节。
- 「128K 配置下 batch size 扩约 3×」是从 Figure 7 读图得到（Jamba-Mini，2 GPU，128k/4k，Max Batch Size 归一化≈3）；论文正文没有给出这个数字，且它只对应这一个最优配置。
- 「295×」「140×」是论文在高请求速率下取的最大比值（此时 vLLM 的 TTFT 处于排队崩溃区），不是稳态对比，引用时需带负载条件。
- 论文 §4.2 将「eLLM/vLLM-CP 的 TPOT 高于 vLLM」归因于「为解码保留了更多 KV Cache 空间」；从机制上看更直接的解释是并发解码 batch 更大，此处按原文转述并保留存疑。

## 参考资料

- Jiale Xu, Yi Xiong, Rui Zhang, et al. _eLLM: Elastic Memory Management Framework for Efficient LLM Serving_. DAC '26, Long Beach, CA. DOI: 10.1145/3770743.3804144。[arXiv:2506.15155](https://arxiv.org/abs/2506.15155) · 本地 PDF：[references/papers/ellm-arxiv-2506.15155.pdf](references/papers/ellm-arxiv-2506.15155.pdf)
- 背景对照：Woosuk Kwon et al., _PagedAttention_, SOSP '23（eLLM 直面的前一站）· Yinmin Zhong et al., _DistServe_, OSDI '24（多卡对照基线）· Ramya Prabhu et al., _vAttention_, [arXiv:2405.04437](https://arxiv.org/abs/2405.04437)（同时期另一条不依赖 PagedAttention 的动态内存管理路线）
- 同组前期工作：Cong Guo et al., _GMLake: Efficient and Transparent GPU Memory Defragmentation_, ASPLOS '24（虚拟内存拼接修 BFC 碎片，eLLM 的 activation 侧直接沿用 BFC）
- 精读报告（Agent 精读核对版）：[references/ellm-report.md](references/ellm-report.md)
