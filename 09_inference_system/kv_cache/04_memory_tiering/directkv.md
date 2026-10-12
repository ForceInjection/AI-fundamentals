# 不搬 KV 反而更快：OSDI'26 DirectKV 让显存少占 43%

> **元信息**：论文 _No Buffer, No Bottleneck: Efficient Zero-Copy KV Cache Offloading for Long-Context LLMs_，Shutian Luo、Haiying Shen（University of Virginia），OSDI '26 正式论文集 pp. 39–54，2026-07-13~15，Seattle。源码 `github.com/shutianluo/DirectKV`（MIT，论文脚注 p.40）。精读方式：全文逐条抽取核对 + 关键图版式复核（报告见 [references/directkv-report.md](references/directkv-report.md)），数字均标论文印刷页码。阅读日期 2026-10-12。

把 KV cache 留在 CPU 内存里、让 GPU 的 attention kernel 直接去读它，这条路的实测代价先摆出来：同一个矩阵乘（`C = A × B`，10240 方阵，各 400 MB），用 200 MB 显存缓冲把数据换进来（Swap），延迟 65 ms（PCIe）/54 ms（NVLink-C2C）；换成零拷贝、让 GPU 直接读 CPU 内存，PCIe 上 1122 ms、NVLink-C2C 上 106 ms——对照全显存 baseline 的 56/52 ms，PCIe 上慢 20 倍，NVLink-C2C 上也还慢 2 倍（p.42，Fig.2a 读数）。省下的显存是实打实的：零拷贝的显存占用比全显存方案少 50%、比 swap 方案少 33%（p.42），但省下的时间一笔勾销。

OSDI '26 的 DirectKV 从这个反直觉的起点出发，最后给出的账本是：GH200 上 CPU–GPU 传输量最高降 50%、GPU 显存占用降 43%、端到端性能最高 1.2×（p.39 摘要）。它要达到的状态，论文标题自己概括成两句话——"no buffer"（取消 HBM staging buffer）与 "no bottleneck"（压掉 CPU–GPU 搬运瓶颈）。

## 一、问题：既有 offload 都假设"KV 必须先到 HBM"

长上下文模型的 KV cache 按序列长度线性增长，论文举的极端例子是 128K tokens 的缓存可以要几百 GB，远超 HBM 容量（p.39）。把 KV 放到 CPU 内存是业界共识的做法，但论文指出所有既有方案共享同一个隐含假设：

> "all of these designs still stage KV cache into GPU memory along the critical path and depend on extra GPU-resident buffers during attention computation, because their attention kernels assume that all operands (including KV) reside in HBM"（p.39）

这条假设带来三笔代价。为盖住传输延迟，系统必须预留大块 GPU 缓冲，"To hide transfer latency, a large GPU buffer is required, wasting valuable GPU memory"（p.41）；swap 类设计按层换入换出，又"doubling interconnect traffic and further constraining scalability"（p.41）；`cudaMemcpyAsync` 的地址随 decode step 变化，还让 CUDA Graph 无法原样重放——"These dynamic transfer addresses can force graph updates or re-capture, reducing the benefits of CUDA Graphs in practical serving systems"（p.41）。把部分或全部 attention 计算下放到 CPU 是另一条路线，论文一句话定性："sacrifices the full compute capability of the GPU"（p.39）。

改变游戏规则的是硬件。GH200/GB200 这类 CPU–GPU 超芯用 NVLink-C2C 提供最高 900 GB/s 双向带宽，"7× higher than PCIe Gen5"（p.39）。900 GB/s 相对 HBM 的 4 TB/s 仍有约 4 倍差距（§2.4.3 里单向口径是 450 GB/s 对 4 TB/s，p.43），但相对 PCIe 的 64 GB/s（§2.2.1 口径，p.41；§1 另写 40–60 GB/s，p.39）已经不是同一种互联。DirectKV 的赌注是：零拷贝访问的粒度恰好适配这条通道——每个 SM 取约 100 KB 的 KV tile，100 多个 SM 并发取，"creates MB-scale aggregate data movement"，于是决定性能的是持续带宽而非单次访问延迟（p.41）。

而朴素零拷贝的失败原因也被定量了：它"repeatedly fetches B from CPU memory throughout the O(n³) computation, fully exposing the bandwidth gap"（p.42）。连带伤害是 L2 局部性——Baseline/Swap 的 L2 命中率约 77%、HBM→L2 流量 21.1 GB；零拷贝的命中率掉到 32.3%，论文称此为"a 58% drop compared to Baseline/Swap"（p.42，Fig.3）。论文的结论是这两件事叠加——带宽限制加上被打坏的 cache 局部性——才把朴素零拷贝压得这么低（p.42）。

## 二、机制：三招把带宽压力从互联挪到 HBM

论文的核心判断只有一句："By tiling through SMEM, we shift bandwidth pressure from the CPU–GPU interconnect to HBM, fully exploiting NVLink-C2C while maintaining GPU compute efficiency"（p.40）。围绕这一句，落成三件事。

### 2.1 CPU-memory-aware tiling：换掉矩阵乘里"驻留"的那个操作数

常规 tiling 里，C 驻留在寄存器/SMEM 反复累加，A、B tile 从 HBM 反复取（Algorithm 1 的 `NativeTiling`，p.45）。DirectKV 把角色对调：B tile（也就是 CPU 内存里的那部分）只取一次、跨所有 i 复用，A 与 C 改从 HBM 取（`CpuAwareTiling`，p.45）。B 的流量从 O(n³) 降到 O(n²)，代价是 C 每步都要从 HBM 读回累加、流量从 2·O(n³) 涨到 3·O(n³)（p.45）。

这笔交换在数字上算得过来，因为 "HBM bandwidth is roughly an order of magnitude higher than CPU–GPU interconnect bandwidth"（p.42）。GEMM 案例里，零拷贝下 B 的 CPU→GPU 流量被重复访问放大到 33.5 GB（"over 3× the original tile size"），改成 CPU-aware 后降到 0.4 GB；HBM 流量涨到 61.8 GB + 33.9 GB，论文的表述是 "increasing HBM traffic by about 2×"（p.42，Fig.4）。延迟从 106 ms 降到 54 ms（49%），L2 命中率从 32.3% 回到 75.1%（p.42–43，Fig.5）。

prefill 与 decode 用的是两套相反的策略（§5.1，pp.45–46）。prefill 阶段 KV 在 CPU 内存里、重载代价高，于是按 HBM 里的 Q 迭代，每个 (K,V) tile 被多个 Q tile 复用，对应 `CpuAwareTiling`；decode 阶段 Q 只有一个 token，遍历一遍 KV 就够，于是对给定 Q 迭代 (K,V)，还能把输出 O 留在寄存器里跨迭代复用、不必反复回 HBM 取，对应 `NativeTiling`。

### 2.2 Warp 级流水：把搬运藏进计算

三个 warp group 分工：producer 从 HBM 取 tile、consumer 计算、storer 把生成的 K/V 写回 CPU 内存（Algorithm 2 Step 3，p.46）。consumer 算当前 tile 时 producer 预取下一 tile（Algorithm 2 Step 6/7 与 16/18；Algorithm 3 Step 5/6，pp.46–47），论文把它类比为 swap 方案里 host–device 传输与计算的重叠（p.46），Hopper 上的 TMA 硬件拷贝引擎进一步帮衬（p.46）。decode kernel 只需 producer 与 consumer 两组，因为每次只写回一个新的 KV 对，写回由 consumer 兼任（§5.4，p.47）。

有一处调度选择值得单记：不同 attention head 与 batch 被分派到不同 SM，以避开输出 O 的跨 SM 写–读依赖——"Since the product of the number of heads (e.g., 8) and batch size (e.g., 32) typically exceeds the number of SMs (e.g., 132 on GH200), this scheduling ensures that no two SMs simultaneously operate on the same head–batch pair with different (K,V) tiles"（p.47）。

微基准给出的收益是吞吐而非流量：有/无 pipe fetching 的两版传输量几乎相同，但 warp 级流水把 HBM 吞吐从 0.3 TB/s 提到 1.3 TB/s（4.3×），延迟 54 ms → 48 ms（11%）（p.43，Fig.6）。

### 2.3 Kernel fusion：K/V 生成后不许回 CPU 内存兜一圈

分两个 kernel 跑时，projection 生成的 K/V 必须写回 CPU 内存更新 KV cache，随后被 attention kernel 重新读回来，"creating redundant CPU–GPU transfers"（p.43）。DirectKV 把 QKV projection 与 attention 合成一次 kernel launch，K/V tile 生成后留在 SMEM 直接被 attention 消费（p.46），X、Wk、Q 也常驻 SMEM（p.43）。论文说这一步"breaks the strict read–write dependency in traditional Zero-Copy"（p.43）。微基准里两版传输量相近（约 125.2 GB），分离 kernel 因冗余写回与重取使 HBM 吞吐低 15.8%，融合版跑到 1.9 TB/s，延迟 85 ms → 57 ms（论文写 "a 49% speedup"）（p.43，Fig.7）。

### 2.4 装进系统：离线生成内核、在线选内核、KV 放 pinned 内存

支撑上述三招的是一个四件套运行时（§3–§4、§6、§7.1，pp.44–47）：Kernel Generator 用 C++ 模板按 `<T, Dim, N>` 离线预编译候选 kernel 池（覆盖 FP16/BF16 与 prefill/decode 两套，Listing 1 p.45）；Kernel Adaptor 在线按张量精度、head dimension、执行阶段匹配最近的实例；Attention Fusion Engine 负责融合（p.44）；KV Cache Manager 用 `cudaHostAlloc` 在 host 端分配 pinned 内存，让 kernel 通过 device-visible pointer 直接寻址，"without staging them in GPU memory"（§6 p.47）。SMEM 的预算被写成一条约束：默认留 80%（α）给 SMEM，从 256 KB 的 L1/SMEM 池里划，约束式 `α · P ≥ 3 · m · size(T) · (Dim · N)` 决定 tile 尺寸 N 的上限（p.44）。

还有个容易被略过的设计声明：DirectKV 用的是标准零拷贝 pinned memory，不是 GH200 的统一页表（UPT）；不需要 coherence 的原因很朴素——"KV blocks are not concurrently updated by the CPU"（p.50）。这把方案的适用面从支持 UPT 的平台扩到了任何能用 pinned 内存的地方。

## 三、评估：GH200 上的实测与三笔账的口径

评估环境是一台 GH200 Grace–Hopper（Hopper GPU、96 GB HBM3、LPDDR5X CPU 内存），模型 Llama-3.1-8B 与 OPT-13B/30B，数据集 ShareGPT 与 Alpaca，请求率最高 30 req/s、上下文 1k–32k（§7.2，p.48）；baseline 是 SGLang（HBM 常驻）、Pie（大 GPU buffer + NVLink-C2C swap）、FlexGen（GPU/CPU/disk 三层 + 4-bit 压缩）、Neo（部分 attention 与 KV 下放 CPU）（p.48）。实现约 5,300 行 CUDA/C++，基于 FlashAttention-3 扩展，CUDA 12.4 + CUTLASS 3.0+（p.47–48）。

**端到端延迟**（§7.3.1，p.48，Fig.10）：Llama-3.1-8B 在 30 req/s 下，DirectKV 把延迟从其他 offload 系统的 1.55–2.95 s 压到 0.75 s；OPT-13B 在同样负载下 DirectKV 0.75 s、其他 offload 系统 1.55–3.95 s，而 SGLang 在这一负载下 OOM；OPT-30B 上 SGLang 只支持低请求率，DirectKV 能撑到 30 req/s。SGLang 只要放得下就最快，论文的定位是"DirectKV offers a better performance–capacity tradeoff"（p.48）。

**上下文长度**（§7.3.2，p.49，Fig.11a）：DirectKV 在"每个上下文长度下支持该长度的系统里"延迟最低，"with a 1.2× average speedup"；优势随长度放大——16k tokens 时对 Neo/Pie 约 1.3×、对 FlexGen 约 1.7×；32k 时 Neo、Pie、SGLang 全部 OOM。

**显存**（§7.3.3，p.49，Fig.11b）：SGLang 平均吃 92 GB 并逼近 96 GB 上限；Neo 86 GB、Pie 88 GB、FlexGen 74 GB（offload 系统仍因 KV 搬运用到的运行时缓冲而占用可观显存）；DirectKV 只用 47 GB，论文给的结论是"节省 35 GB 平均、对应 43% 的显存降幅"。精读员注：35 GB 与 43% 都对得上 Neo/Pie/FlexGen 三者的均值 82.7 GB，不含 SGLang 的 92 GB——引用这两个数时基线要说清。

**组件级**（§7.4，pp.49–50）：CPU-aware 零拷贝相对朴素零拷贝，传输量最高降 50%、推理延迟最高降 70%（同一对照，Fig.12）；融合 kernel 相对分离 kernel，HBM 吞吐最高 3.5×、per-token 延迟低 2.5–3.0×（Fig.13）；NVLink-C2C 相对 PCIe 把 attention 延迟降最高 4.2×（512 序列长度，Fig.14）。

## 四、边界：这套账在什么条件下成立

论文自己划的第一条线是硬件（§8.2，p.50）：DirectKV 适合有高带宽互联的平台，"on PCIe-only platforms, DirectKV may still reduce GPU memory usage, but throughput benefits are often limited by interconnect bandwidth"——在 PCIe 上它更接近一个容量扩展机制（§7.4.3，p.50）。第二条线是范围（§8.3，p.51）：这是节点内的异构内存优化，context-parallel 扩展与多级 KV 层级的集成都需要额外运行时支持，而且"zero-copy benefits diminish once accesses fall back to pageable or disk-backed tiers"，都被留作 future work。第三条线是定位（§8.1，p.50）：它不改调度、批与 KV 管理策略，要接进 vLLM/SGLang 的做法是把 KV allocator 扩展出 CPU-pinned 存储，其余不动。

精读员注补充几点论文没覆盖的：评估只有 GH200 一台机器（外加 H100 PCIe 做互联对照），最长上下文 32k 而 motivating 场景是 128K（p.39、p.48）；`α`=80%、两级流水、tile 尺寸都是固定缺省，没有敏感性分析（p.44）；端到端结果全部是 per-token latency，没有 tokens/s 吞吐或尾延迟；CUDA Graph 适合零拷贝这条论证（p.41）没有配套实验。另外，"50% 传输量降幅"的对照口径在文内不一致：§7.4.1 明确写 "compared with naive zero-copy"（p.49），摘要与 §1 却写成 "compared to existing solutions" / "Compared to prior offloading approaches"（p.39–40）——同一段里的 43% 显存与 1.2× 端到端是对系统 baseline，50% 是对朴素零拷贝，两笔账不能合并引用。

## 论文结构索引

| 论文位置                       | 内容                                                                                            |
| ------------------------------ | ----------------------------------------------------------------------------------------------- |
| §1 / p.39–40                   | 动机、贡献、源码脚注；"no buffer, no bottleneck"                                               |
| §2.1–2.2 / p.40–41             | attention 与 KV offload 基础；swap 的三笔代价（显存、双倍流量、CUDA Graph）                     |
| §2.3 / p.41–42                 | GEMM 案例：Swap vs Zero-Copy 的 20×/2× 惩罚、L2 命中率塌陷（Fig.2/3）                           |
| §2.4.1–2.4.3 / p.42–43         | 三个 takeaway：CPU-memory-aware tiling、warp 级流水、kernel fusion（Fig.4–7）                  |
| §3–§4 / p.44–45                | 架构四件套；SMEM 分区复用与约束式 (1)；离线生成 + 在线选择（Fig.8、Listing 1）                  |
| §5.1–5.4 / p.45–47             | 两套迭代策略；三/两个 warp group；prefill 与 decode 的融合 kernel（Algorithm 1–3）             |
| §6–§7.1 / p.47                 | KV Cache Manager：pinned 内存 + device-visible pointer；5,300 行实现                            |
| §7.2–7.3 / p.48–49             | 评估配置、端到端延迟、显存（Fig.10/11）                                                         |
| §7.4 / p.49–50                 | 组件级：CPU-aware（50%/70%）、融合（3.5×）、NVLink-C2C vs PCIe（4.2×）（Fig.12–14）            |
| §8 / p.50–51                   | 讨论：与 serving 系统集成、硬件适用性、可扩展性                                                 |
| §9–10 / p.51–54                | 相关工作与结论                                                                                 |

## 与系列其他篇目

同一块 CPU 内存，四种花法：[NEO](neo.md) 把计算连同 KV 一起下移（算力换显存）、[eLLM](ellm.md) 引入弹性抽象让分层内存用起来像一层、[Strata](strata.md) 保留搬运但把 I/O 路径与调度重做，DirectKV 干脆取消搬运、只留直接寻址。横切总纲见 [KV Cache 内存层级总纲](memory-tiering-overview.md)。

## 诚实声明

- **精度未核实**：全文没有 accuracy / perplexity 实验，唯一相关表述是 §1 的 "all while maintaining full attention accuracy"（p.40）。融合 kernel 改了 streaming softmax 的累加路径与 RoPE 的施加位置，论文未给与基线的数值等价或误差对比。
- **50% / 70% 的对照是"朴素零拷贝"**，不是 Pie/Neo/FlexGen（§7.4.1，p.49）；摘要与 §1 的措辞更宽，引用时以 §7.4.1 为准。
- **§2.3 正文与 Fig.3 的读数有一处冲突**：正文写 "increasing CPU-to-GPU traffic to 22.7 GB"，Fig.3 零拷贝组是橙柱 22.7（GPU→L2）、蓝柱堆叠顶标 56.3、绿柱 32.3（命中率），对照 Fig.4 的 22.7/33.5 判断 22.7 GB 是 GPU→L2、CPU→L2 约 33.5 GB，正文那句当属笔误。本文未使用该冲突句的数字。
- **Swap 开销的正文与图不一致**：正文称 Swap 在 PCIe 上开销"不到 10%"、NVLink-C2C 上"约 5%"（p.42），Fig.2(a) 的绝对值是 56→65 ms（+16%，PCIe）与 52→54 ms（+3.8%，NVLink-C2C）——NVLink-C2C 一侧对得上，PCIe 一侧对不上。本文统一引用图中绝对值。
- **未实测**：GB200、128K 上下文、与 vLLM/SGLang 的真实集成、CUDA Graph replay、host 内存压力，论文均无实验数据；GH200 的 LPDDR5X 容量论文未给出。
- 本文所有性能数字均为论文在 GH200/H100 上的真机实测；唯一属建模性质的是 §5.1 用 O(n³)/O(n²) 对流量的复杂度论证与 §4.1 的 SMEM 约束式。

## 参考资料

- Shutian Luo, Haiying Shen. _No Buffer, No Bottleneck: Efficient Zero-Copy KV Cache Offloading for Long-Context LLMs_. OSDI '26, pp. 39–54. 本地 PDF：[references/papers/directkv-osdi26.pdf](references/papers/directkv-osdi26.pdf)；[USENIX 论文页](https://www.usenix.org/conference/osdi26/presentation/luo)
- 源码：[github.com/shutianluo/DirectKV](https://github.com/shutianluo/DirectKV)（MIT License）
- 精读报告（含逐条页码核对与疑点清单）：[references/directkv-report.md](references/directkv-report.md)
