# NEO（MLSys 2025）精读报告

> 来源：papers/neo-arxiv-2411.01142.pdf（本目录下的 papers/；arXiv 版，PDF 共 13 页）。核对方式：全文抽取逐条对照，数字标页码「p.X」。只有推断/点评标「精读员注」；未核实写「未核实」。

## 一、速览

- 完整标题（含副标题）：NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference（arXiv:2411.01142v1，2024-11-02，cs.DC。PDF 页脚自标 "Preprint, under review"，p.1——本 PDF 内没有 venue 行，MLSys 2025 标注来自调研阶段的 MLSys 官方 poster 页核实）
- Venue：MLSys 2025（系列给定背景 + 外部核实：poster 页 mlsys.org/virtual/2025/poster/3346、OpenReview CFQlQ34z2g；本 PDF 为投稿版，PDF 内未核实 venue）
- 作者与机构：Xuanlin Jiang\*（北京大学；\* 标注 "Work done during an internship at Harvard University"）、Yang Zhou（UC Berkeley / UC Davis）、Shiyi Cao（UC Berkeley）、Ion Stoica（UC Berkeley）、Minlan Yu（Harvard University）（p.1）
- 一句话贡献：把一部分 decoding attention 的**计算连同 KV cache** 从 GPU 下放到主机 CPU，用非对称 GPU-CPU 流水线 + 负载感知调度，在时延不变的前提下把在线推理吞吐提高最高 7.5×（T4）/ 26%（A10G）/ 14%（H100）（p.1）。论文自称是首个在在线推理下"CPU offload 后仍比 GPU-only 更快、且硬件成本与精度相同"的系统（p.2）。
- 开源/集成情况（仓库、落地系统）：论文结论承诺 "We will open source NEO codebase"（p.11）；实现基于 SwiftLLM，论文称 "could be adapted to other frameworks like vLLM ... and SGLang"（p.2）。实际仓库 github.com/NEO-MLSys25/NEO（外部核实，非 PDF 内容）；**论文未描述生产落地**，"integrating NEO into vLLM to measure our 4-GPU and 8-GPU performance gains" 被列为 future work（p.10）。

## 二、问题与动机

- 显存危机：批大小决定吞吐，而批大小被显存框死。"the memory-bounded LLM inference workloads have created the GPU memory crisis where people demand expensive high-end GPUs with large memory sizes"（p.1）。论文引测量：吞吐随批大小近似线性增长直到数百，但 "these GPUs fail to accommodate hundreds or even dozens of batch size, leaving significant GPU compute resources unutilized"（§2.1，p.2）。
- 既有路线不够用：量化掉精度；offload 系（FlexGen/PowerInfer/TwinPilots/HeteGen/FastDecode）大多 "trades inference latency for throughput by using huge GPU batch sizes and layer-by-layer swapping, thus not suitable for online inference"（p.1）。最接近的 FastDecode 被点名："it uses 8 32-core AMD Epyc CPUs in remote servers to handle the offloaded compute of only one A10G GPU, where the CPUs cost much more than the GPU"（p.1）。本文立场是只用 "local host CPUs as the offloading target that comes with the GPU 'as free'"（p.1）。
- 为什么可以下移 attention：只有 decoding attention 依赖 KV cache，"offloading this part of computation to the CPU would help avoid repetitively transferring the KV cache between the GPU and CPU"，而且这一操作计算占比极小、不需要加载模型权重（§2.2，p.3）。
- 为什么 CPU 顶得住：decoding attention 在 GPU 和 CPU 上都是 memory-bandwidth-bound，"The memory bandwidth gap between GPUs and CPUs is much smaller than their compute gap"——文中给了具体对照：A10G 600 GB/s 带宽 + 125 TFLOPS，x86 服务器约 200 GB/s + 1.2 TFLOPS（算力差 100×，带宽差 3×）；并提到 AWS Graviton4 每 socket 537.6 GB/s（§2.2，p.3，数据引自 He & Zhai 2024 / WikiChip）。
- 两个挑战（§2.3，p.3）：Challenge #1 单次迭代内如何重叠 GPU 与 CPU 且不过载任一方（"not overloading any hardware"，p.3）；Challenge #2 跨迭代如何对动态 input/output 长度做自适应调度——点名 FlexGen 与 FastDecode "assumes fixed input/output lengths ... leverages static optimal offloading policies that were determined by one-time offline profiling"（p.1；§2.3 p.3）。

## 三、核心机制

> 精读员注（术语更正）：本报告全文检索后确认，**NEO 论文中不存在 "Streaming Forward Loading"（SFL）与 "Partial Overlap" 这两个术语**——它们是 FastDecode（He & Zhai 2024）的机制名。公开转述里常见把这两个术语安在 NEO 头上，属讹误，引用时勿沿用。NEO 自己的机制名是 **asymmetric pipelining**（含 partial offloading、layer-wise swapping、asymmetric batch division）与 **load-aware scheduling**；其中与 SFL 职能最接近的是 layer-wise swapping（见 3.1 机制 c）。

### 3.1 非对称流水线（asymmetric pipelining，§3.1 pp.4–5）

**做什么**：每个迭代并行跑两个**不对等**的 sub-batch——batch-0 基本在 GPU 上（含全部 prefill 请求与 GPU decoding 请求，外加少量 CPU decoding 请求），batch-1 基本在 CPU 上（只装 decoding attention）；两者互相重叠，"to balance the load of GPU and CPU"（§3.1，p.4；Figure 5，p.5）。

**为什么不是两个 obvious 做法**（两个 strawman，均 p.4，Figure 3/4）：

- Strawman #1 simple offloading：把全部 decoding attention 连同 KV 卸到 CPU、其余留 GPU，但 "during these linear operations, the CPU always remains idle ... fails to leverage the compute and memory resources of CPUs"（p.4；Figure 3 caption 给出传输语义：TrQKV = 传 Q,K,V 到 CPU，TrO = 传 attention 输出回 GPU）。
- Strawman #2 symmetric pipelining（FastDecode 等在用）：把一个 decoding batch 均分成两个相同 sub-batch，GPU 线性段与 CPU attention 段对跑。论文列三宗罪（p.4–5）：① GPU 内存显著闲置（GPU 只留权重和 activation，其余原本给 KV cache 的内存空着）；② 重叠不平衡——prefill 与 KV swap-out 期间 CPU 空转，且 CPU attention 段通常长于 GPU 线性段，"causing the CPU to become the bottleneck"；③ 实践中无法均分——"it is nearly impossible to ensure that a single batch can be divided into two identical sub-batches"，会制造 pipeline "bubbles"（p.5）。

**四个关键设计（怎么做）**（均 p.5）：

- a. **Partial offloading**：KV cache 拆成 GPU HBM 里的 "GPU-cache" 与 CPU 主存里的 "CPU-cache"；已 prefill 的请求整段驻留其一，分别叫 "GPU-request"/"CPU-request"；"Requests are prioritized for storage in the GPU-cache to maximize GPU memory utilization."。这是与 FastDecode 全量卸载的分野，防止 CPU 被算力或带宽压垮（§1 p.2 亦称 "partial offloading to limit the offloaded computation and memory bandwidth consumption"）。
- b. **Prefill 并入 batch-0**：把 prefill 计算放进 GPU decoding sub-batch，使 "the prefilling stage computation (in GPU) also happens in parallel with the CPU attention computation"，思路与 Orca 的 selective batching 同源；作用是拉长 GPU 计算时长，给 CPU 侧留出更长的可重叠窗口，从而容纳更多 CPU-requests。
- c. **Layer-wise swapping**：新 prefill 请求的 KV 按层产出，于是 "we can initiate PCIe transmission immediately after each layer's KV value is computed, rather than deferring this process until the end of the entire iteration."——把 KV 传输与后续层计算重叠（这是与 SFL 职能对应的机制，但论文没有用 SFL 之名）。
- d. **Asymmetric batch division**：不做均分，而是 "consolidate all prefilling requests and GPU decoding requests with a few CPU decoding requests to batch-0, while dispatching the majority of CPU decoding requests to batch-1"。两个 batch 互补：batch-0 线性段长、CPU attention 少；batch-1 attention 段长、线性段极短。迭代呈现 "alternative 'long-stages' and 'short-stages'"，而非均匀分段。

**附带收益**（p.5）：首尾非重叠段被压到最小；**GPU attention kernel 每迭代只调用一次**（原文："the GPU attention kernel is only invoked once per iteration, as only one sub-batch will need the GPU attention"，即只有 batch-0 需要——batch-1 的 attention 全在 CPU），对比 symmetric pipelining 的两次，从而减少 Python GIL 下昂贵的 kernel 启动开销。

**伪代码**：论文未给流水线伪代码。精读员注——按 §3.1 文字与 Figure 5 可整理为（术语用原文）：

```text
每个迭代：
  batch-0 = [全部 prefilling 请求] + [GPU decoding 请求] + [少量 CPU decoding 请求]
  batch-1 = [其余 CPU decoding 请求]
  # GPU 侧：batch-0 pre-layer → 逐层 (linear: pr/po + FFN) → 需要时 GPU attention (ga0)
  # CPU 侧：batch-1 的 decoding attention (ca1) 与 batch-0 的 linear 段并行
  #          batch-0 中那少量 CPU 请求的 attention (ca0) 与 batch-1 的 linear 段并行
  # KV 传输：batch-0 每层 KV 算完即刻经 PCIe 传出（layer-wise swapping）
  时间 ≈ L × ( max{Tpo0+Tpr0, Tca1} + max{Tpo1+Tpr1+Tga0, Tca0} )   # 见 3.2 估算式
```

### 3.2 负载感知调度（load-aware scheduling，§3.2 pp.5–6）

**做什么**：每迭代在线决定"新请求进 GPU 还是 CPU、两个 batch 怎么组"，目标是 "to keep both busy while not overloading anyone"（p.5）。

**比 vLLM 难在哪（三个难点的原文）**（p.5）：① 要同时做 batch selection 与 batch splitting；② "blindly putting as many requests as possible into the CPU decoding runqueue does not work well"（CPU 带宽/算力会被压垮）；③ "the two-batch asymmetric pipelining does not always work better than GPU-only inference"——双 batch 并跑时每个 sub-batch 最多只能拿一半 batch size，若卸得不够多，"the sub-batch size might be even smaller than the GPU-only inference, yielding lower throughput"。

**四条设计原则**（原文名，p.6）：**Greedy**（每迭代各构造一份 GPU-only 方案与一份两 batch 方案，取估算吞吐更高者）、**Balancing**（把 pipeline bubble 压到最小，即估算的 CPU busy time 与 GPU busy time 尽量接近）、**Hiding CPU**（不允许"CPU 忙而 GPU 闲"）、**Maximizing GPU**（从 prefill waitqueue、GPU runqueue、CPU runqueue 里尽量多取请求）。

**代价模型**（p.6）：每次迭代时间 T、batch size x，目标最大化 T/x。定义 pre-layer 时间 Tprl、post-layer 时间 Tpol，主体是 transformer 层时间 Ttr（"usually takes more than 95% of time in each iteration"）。估算式逐字：

```text
T ≈ Ttr = L × ( max{Tpo0 + Tpr0, Tca1} + max{Tpo1 + Tpr1 + Tga0, Tca0} )
```

其中 Tpox/Tprx/Tgax/Tcax 是 batch-x 的 post-projection、pre-projection、GPU attention、CPU attention 时间，Tlx = Tpox + Tprx。Tl、Tga、Tca 的估值靠 "offline profiling for typical input/output lengths and uses linear interpolation"（p.6）。Balancing/Hiding CPU 落地为两条不等式：**Tl0 ≥ Tca1** 与 **Tl1 + Tga0 ≥ Tca0**，且尽量缩小白左边与右边的差距（p.6）。

**每迭代六步调度流程（论文原文即伪代码，逐条转述，p.6）**：

1. **Initialization**：清空两个 batch schedule——batch-0 主要跑 GPU（含 prefill 与 decoding 请求），batch-1 主要跑 CPU（只含 decoding attention 请求）。
2. **Schedule GPU decoding requests**：GPU runqueue 里的请求尽量进 batch-0；随后按需 swap out 腾显存、若有富余则 swap in（Maximizing GPU）。
3. **Schedule prefilling requests**：从 prefilling waitqueue 取请求进 batch-0，显存够就把新 KV 留在 GPU，否则 swap out；重复直到 "the GPU cannot hold the activations in the batch"。
4. **Schedule CPU decoding requests**：扫 CPU runqueue，把请求放进 batch-0 或 batch-1，维持 **Tca0 ≤ Tl1 + Tga0** 与 **Tca1 ≤ Tl0**；放进任一侧会破坏不等式就跳过、留给下一迭代（Balancing and hiding CPU）。
5. **Reduce prefilling requests**：在不破坏不等式的前提下，把 batch-0 中"需要把新生成 KV swap out"的 prefill 请求移出，避免 CPU 空转（Balancing）。
6. **Make decisions**：两 batch 方案成型后，由 batch-0（去掉第 4 步加入的 CPU decoding 请求）构造 GPU-only 方案，比较两者估算的 Ttr / x，取大者（Greed）。原文原则名是 "Greedy"，第 6 步写作 "Greed"（p.6，原文如此）。

### 3.3 实现配套（§4 pp.6–7）

- **基座**：SwiftLLM（"a simplified copy of vLLM"，约 2K 行，单卡默认调度下达到 vLLM 等价性能，p.6）；NEO 自身 "4K lines of Python and 1.5K lines of C++"（p.6）。
- **CPU kernel（PACPU，p.6）**：自定义 C++ torch extension "Paged-Attention-for-CPU"，Python 侧直接调多线程 C++；内层用 **ISPC**（Intel SPMD Program Compiler）生成可编译到 AVX2 / AVX512 / ARM Neon 的向量化代码，因此可跨 CPU 移植；PACPU 沿用 vLLM 式 paged KV cache 抗碎片。优化两点：单核内用 SIMD load/store 且刻意保证访存连续性；跨核用类似 Flash Decoding 的策略——按 request 维切任务、每任务在 block 粒度访问独占连续内存、按 block 数均分给线程、最后聚合部分输出（pp.6–7）。
- **减 kernel 启动开销（p.7）**：Python GIL 让"数据面 CPU kernel"无法与控制面的 CUDA 启动并行，启动落在关键路径上；对策是把 SwiftLLM 大部分 Triton-JIT kernel 换成手写 CUDA C++。
- **多卡（p.7）**：重写 SwiftLLM 架构支持 model sharding 与 tensor parallelism；用 Ray actors 持有模型分片、PyTorch 预建通信库（NCCL）做跨 GPU 通信；每个 Ray actor 同时持有自己的 CPU KV cache 分片（按 KV heads 划分，避免跨 CPU 通信）。

## 四、关键数字

| 指标 | 数值 | 对照基线/条件 | 出处 |
| ---- | ---- | ------------- | ---- |
| 吞吐提升（摘要总口径，T4） | 最高 **7.5×**（实测） | "compared to GPU-only approach"，时延相同；§5.4 展开为 synthetic workload、T4 + LLaMa-2-7B、基线 = GPU-only NEO（即 SwiftLLM） | 摘要 p.1；§5.4 p.8（"750%"）、Figure 9c p.9 |
| 吞吐提升（A10G） | 最高 **26%**（实测） | 同上，A10G + LLaMa-3.1-8B，synthetic，基线 = SwiftLLM | 摘要 p.1；§5.4 p.8 |
| 吞吐提升（H100） | 最高 **14%**（实测） | 同上，2×H100 + LLaMa-3.1-70B，synthetic，基线 = SwiftLLM | 摘要 p.1；§5.4 p.8 |
| 更强 CPU 下的提升（A10G） | 最高 **79.3%**（实测） | g5.16xlarge（CPU 带宽约为 g5.4xlarge 的 4×），基线 = SwiftLLM | 摘要 p.1；§5.5 p.9 |
| CPU 容量敏感性 | **12.2% / 13.3% / 29.7% / 79.3%**（实测） | 四档 AWS 实例 g5.2/4/8/16xlarge，同为 1×A10G、baseline 相同（精读员注：数值与实例的对应按论文行文顺序推定） | §5.5 p.9，Figure 10a p.9 |
| 在线服务 vs vLLM（H100） | 吞吐高 **14.3%**（实测） | 2×H100 + LLaMa-3.1-70B + AC trace，Poisson 到达，取 2 s 时延点 | §5.2 p.7，Figure 6a p.8 |
| 在线服务 vs vLLM（A10G） | 吞吐高 **6.40%**（实测） | A10G + LLaMa-3.1-8B + AC，2 s 时延点 | §5.2 p.7 |
| 在线服务 vs vLLM（T4） | 吞吐高 **563%**（实测） | T4 + LLaMa-2-7B + OSC，1 s 时延点；同段文字亦称 "nearly 6×" | §5.2 p.7 |
| 全量卸载对照（FastDecode+） | 输出变长时跌至基线 **< 60%**（实测） | 2×H100 + LLaMa-3.1-70B，输入固定 2000、扫输出长度，基线 = GPU-only NEO | §5.3 p.8，Figure 8b p.8 |
| 基线系统本身的差距 | 2 卡下 SwiftLLM 比 vLLM 低 **8.8%**（实测） | 2×H100 + "LLaMa-3.1-80B"（原文写法），单卡 A10G+8B 时两者相当 | §5.5 pp.9–10，Figure 10b p.9 |
| 动机数字（GPU/CPU 带宽与算力） | A10G：600 GB/s + 125 TFLOPS；x86 服务器：约 200 GB/s + 1.2 TFLOPS；Graviton4 每 socket 537.6 GB/s | 引自 He & Zhai 2024 与 WikiChip，非本文实测 | §2.2 p.3 |
| 迭代时间构成 | Ttr（transformer 层）占每迭代 > **95%** | 调度代价模型的假设前提 | §3.2 p.6 |
| 代码规模 | NEO 4K 行 Python + 1.5K 行 C++；SwiftLLM 约 2K 行 | — | §4 p.6 |
| 硬件配置 | g5.nxlarge：A10G / EPYC 7R32（2n 核）/ 16n GB 主机内存；g4.4xlarge：T4 / Xeon P-8259CL（8 核）/ 64 GB；HGX：8×H100 / Xeon 8462Y+（64 核）/ 2 TB | Table 1；g4 实体名在两个位置写作 "g4dn.4xlarge"（p.7 正文）与 "g4.4xlarge"（表内） | §5.1 p.7 |
| 覆盖范围 | 模型 7B/8B/70B；GPU T4/A10G/H100（2×H100）；数据集 AC（代码）+ OSC（摘要）+ synthetic | AC 用于 H100/A10G，OSC 用于 T4 | §5.1 p.7 |

口径提醒（精读员注）：第一节的 7.5×/26%/14% 与 §5.2 的 14.3%/6.40%/563% **不是同一组数**——前者基线是 GPU-only NEO（SwiftLLM）、取自 synthetic 扫描的峰值相对吞吐（Figure 9）；后者基线是 vLLM、取固定时延点（2 s / 1 s）上的吞吐。两套数在文中都写作 "up to"/单点，引用时不能混用。

## 五、局限与边界

论文自认的限制：

- **多卡未测全**：4 卡 / 8 卡收益未测，"We leave as future work integrating NEO into vLLM to measure our 4-GPU and 8-GPU performance gains"（p.10）；且自认多卡场景工程优化不如 vLLM，高负载下时延略高于 vLLM（两条理由原文：vLLM 优化更重；"NEO actively seeks opportunities to offload requests, even though offloading may not help"，带来调度与 swap 的额外开销，§5.2 p.8）。
- **调度器依赖离线 profiling + 线性插值**，估值不准会带来次优决策：输出很长时 NEO "would be close to the baseline ... sometimes slightly worse due to suboptimal scheduling decisions"（§5.4 pp.8–9）。
- **收益依赖大 CPU 内存**："NEO relies on large CPU memory to ensure throughput gains"（§6 p.10 与 chunked-prefill 对比段）。
- **假设权重全驻 GPU**："NEO implicitly assumes putting all model weights on the GPU and only offloading attention computation to the CPU is the most efficient way"；把 dense 算子也卸到 CPU 只在极端负载（输出极短）下可能有益，"the actual gain needs to be validated by further exploration"（§6 p.10）。
- **远端 CPU 未做**：扩展支持 remote CPU workers 留作 future work，理由是在线云 CPU 内存带宽仍贵、跨机传输延迟可能是新瓶颈（§6 p.10）。
- **与 chunked-prefill 整合未做**："we believe NEO could integrate with chunked-prefill, providing a larger design space. We leave it for future work."（§6 p.10）

前提条件与评估边界（精读员注整理）：

- **PCIe 传输是隐含成本**：每次迭代都有 TrQKV/TrO（Q,K,V 下传、attention 输出回传），论文没有给出 PCIe 代数、带宽或单次传输量数字（未核实），也没有测 PCIe 成为瓶颈的临界点；两条不等式只约束了 CPU attention 与 GPU 线性段时间，没有把传输时间写进代价模型（§3.2 p.6；Figure 3/4 caption p.4）。
- **CPU 带宽决定上限**：§5.5 的结论原文是 "the memory bandwidth, rather than computing power (i.e., number of cores), is the factor that determines the performance of attention operation on CPUs"（p.9）——换更多核不如换更大带宽；g5.2xlarge 与 g5.4xlarge 峰值带宽几乎相同，所以 12.2% 与 13.3% 两档接近。
- **跨 GPU 的数字不可互比**：T4 用 OSC（短请求），H100/A10G 用 AC（长请求）（§5.1 p.7）；且 T4 的大倍数部分来自 vLLM 在极小显存下 batch 被卡死（§5.2 p.7）。
- **2 卡实验只用一个 NUMA 节点**（"1/4th of the total memory size and bandwidth"，§5.1 p.7）——2×H100 的数字是在 CPU 资源被人为削到 1/4 的条件下取得的。
- **模型规模**：只覆盖 7B/8B/70B，未覆盖 MoE、超长上下文、MLA 类注意力；论文也没有给出 CPU 侧 KV 容量上限的定量边界。
- **版本**：以上数字均出自 arXiv v1（2024-11-02）；MLSys 2025 camera-ready 是否修订未核实。

精读员注（疑点）：

1. **术语**：全文无 "Streaming Forward Loading" / "Partial Overlap"（见三、开头）。对外转述 NEO 的机制时应用 asymmetric pipelining / partial offloading / layer-wise swapping / load-aware scheduling。
2. **§5.5 的模型名**："2xH100 + LLaMa-3.1-80B"（p.10）——LLaMA-3.1 无 80B 规模，疑为 70B 笔误；同章其余位置均写作 70B。
3. **12.2/13.3/29.7/79.3 与四个实例的对应**：论文未逐一点名配对，按行文顺序推定 g5.2/4/8/16xlarge；与摘要 "with more powerful CPUs ... on A10G" 一致。
4. **原文明文小错**：§6 写 "to balance GPU and GPU loads"（p.10，应为 GPU 与 CPU）；Table 1 实例名写作 g4.4xlarge 而正文写 g4dn.4xlarge（p.7）。
5. **Table 1 的 Memory 列是主机内存**（按 g5.nxlarge = 16n GB、即 g5.4xlarge 的 64 GB 可核对），不是显存——引用时勿误读为 GPU 显存。

## 六、写作素材

- 系列定位（四种"内存账"之一：NEO=计算下移；DirectKV=零拷贝去搬运；eLLM=抽象虚拟化；Strata=分层调度）：NEO 是四篇里唯一**把计算本身搬下 GPU**的一篇——CPU 不只是 KV 的仓库，而是承担一部分 decoding attention；因此它的对照物是"数据不动、算力动"。叙事锚点：论文对 FastDecode 的成本批评（8 颗 32 核 EPYC 伺候 1 张 A10G，CPU 比 GPU 贵，p.1）与本文只用"随卡白送"的主机 CPU、且**部分**卸载的取舍，正好构成"下移多少算力才划算"这条轴。
- 与另外三篇的对照点：① 搬运路径：NEO 仍走 PCIe 传 QKV/输出（传的是小张量而非整份 KV），DirectKV 则靠在 GH200/GB200 的 NVLink-C2C 上让 GPU kernel 直接解引用 CPU 内存、把搬运彻底消掉——一个是"传得少"，一个是"不用传"；② 抽象层级：NEO 在运行时决定请求归谁（请求级卸载），eLLM 用 Virtual Tensor Abstraction 把内存管理虚拟化（分配级），Strata 用分层缓存 + 缓存感知调度组织跨层数据流（缓存级）；③ 场景：NEO 瞄准在线低时延（"while maintaining the same latency"，p.1）与"显存卡死 batch 的低端卡"（T4/A10G，p.10 的 usage 段），Strata/DirectKV 更偏高端平台与长上下文吞吐；④ 共同前提都有"两侧总线带宽差远小于算力差"，可作总纲的横切观察。
- 与本仓已有文章可勾连处（01_concepts/offloading/、01_concepts/pd_transfer/、03_fast26/ 等，列候选即可）：
  - `01_concepts/offloading/01_kv_offloading.md`（vLLM 原生 offload / LMCacheConnector 只卸数据）——NEO 是最直接的反面参照："卸数据"与"卸计算"的分野；
  - `01_concepts/offloading/02_layerwise_pipeline.md`——NEO 的 layer-wise swapping 是同一思路在单机内的系统化，可对照 PD 分离里的逐层流水线；
  - `01_concepts/offloading/03_kv_cache_prefetching.md`、`01_concepts/offloading/sparse_attention_driven_offloading_problems.md`——预取/信号质量问题里 NEO 只做静态估算，可作"在线系统不依赖重要性信号"的对照样本；
  - `01_concepts/scheduling/01_vllm_chunked_prefill.md`——NEO §6（p.10）专门讨论了与 chunked-prefill 的关系（对方更耗 GPU 带宽、在显存受限卡上失效）；
  - `01_concepts/basic/paged_attention.md`——PACPU 沿用 paged KV cache（p.6）；
  - `03_fast26/`（FAST'26 五场景，SSD/端侧层级）——存储侧层级，与本篇的内存侧层级互补。
- 适合放文章开头的"反差/场景"素材：
  - "低端卡上 GPU 算力在挨饿"：T4 + LLaMa-2-7B 上 vLLM 的 batch 被显存卡死，"NEO could offload the KV cache to the CPU, achieving a much larger batch size"，吞吐近 6×（p.7）；
  - "CPU 是买 GPU 白送的"：原文把这个当作整个方案的立论基础（"comes with the GPU 'as free'"，p.1），与 FastDecode 的远程 EPYC 阵列形成成本反差（p.1）；
  - "带宽差 3 倍、算力差 100 倍"：600 GB/s/125 TFLOPS vs 200 GB/s/1.2 TFLOPS 的对照（p.3），解释为什么 attention 这种 memory-bound 算子下移不亏；
  - "GPU 越造越快、显存原地踏步"："H100 triples the compute of A100 but with the same 80GB memory size"（p.10），论文据此判断它瞄准的场景"will likely hold for a long time"；
  - "反向的诚实"：NEO 的高负载时延比 vLLM 略高，原因是它"actively seeks opportunities to offload requests, even though offloading may not help"（p.8）——可作为"调度激进 vs 时延保守"的讨论引子。
