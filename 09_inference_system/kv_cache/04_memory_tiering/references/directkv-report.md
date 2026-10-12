# DirectKV（OSDI '26）精读报告

> 来源：papers/directkv-osdi26.pdf（本目录下的 papers/，PDF 共 17 页）。核对方式：全文抽取逐条对照，数字标页码「p.X」——p.X 为论文印刷页码（本 PDF 第 1 页为 USENIX 封面页，印刷页 = PDF 页号 + 37，正文 pp.39–54；已用页脚逐页核对）。只有推断/点评标「精读员注」；未核实写「未核实」。

## 一、速览

- 完整标题（含副标题）：No Buffer, No Bottleneck: Efficient Zero-Copy KV Cache Offloading for Long-Context LLMs（p.39，无副标题）
- Venue 与页码：OSDI '26——第 20 届 USENIX Symposium on Operating Systems Design and Implementation，2026-07-13~15，Seattle, WA（p.39 封面）；论文集连续页码 pp.39–54（ISBN 978-1-939133-55-7）。论文正文印刷页码 39–54。
- 作者与机构：Shutian Luo、Haiying Shen，University of Virginia, Department of Computer Science（p.39）
- 一句话贡献：在 GH200/GB200 这类 NVLink-C2C 紧耦合超芯上，让 GPU kernel 直接读 CPU 内存里的 KV cache（零拷贝、取消 HBM staging buffer），并以 CPU-memory-aware tiling + warp 级流水 + projection/attention 融合三招把带宽瓶颈从 CPU–GPU 互联挪到 HBM；GH200 上 CPU–GPU 传输量最高降 50%、GPU 显存占用降 43%、端到端性能最高 1.2×（p.39 摘要）。
- 开源/集成情况（仓库、落地系统、许可证若有）：论文脚注给出源码 `https://github.com/shutianluo/DirectKV`（p.40，脚注 1）。仓库存在（本仓外核实 2026-10-12 经 GitHub API：创建 2026-05-08、最后 push 2026-06-04、15 stars；目录含 `csrc/`、`kernel/`、`baseline/`、`evaluation/`、`tests/`），LICENSE 为 MIT License（"Copyright (c) 2026 Shutian Luo"；GitHub 的 license 识别返回 NOASSERTION，即未按标准 SPDX 归类）。落地形态：论文只作设计论述，定位为 vLLM/SGLang 的补充而非替代——"A serving system can integrate DirectKV by extending its KV allocator to support CPU-pinned storage for offloaded blocks, while leaving the scheduler, model execution graph, and frontend API unchanged"（p.50）；无生产部署描述，无与真实 serving stack 的集成实测。

## 二、问题与动机

（以下均为论文自述；关键表述给英文短引）

- **既有 offload 的共同前提：KV 必须先进 HBM。** "all of these designs still stage KV cache into GPU memory along the critical path and depend on extra GPU-resident buffers during attention computation, because their attention kernels assume that all operands (including KV) reside in HBM"（p.39）。第三条路线（把部分或全部 attention 计算下放 CPU，如 Neo、FastDecode 一类）"sacrifices the full compute capability of the GPU"（p.39）。
- **代价一：staging buffer 吃显存。** "To hide transfer latency, a large GPU buffer is required, wasting valuable GPU memory"（p.41）。
- **代价二：layer-by-layer swap in/out 使互联流量翻倍。** "swap-based designs perform layer-by-layer swap-in and swap-out of KV blocks, doubling interconnect traffic and further constraining scalability"（p.41）。
- **代价三：CUDA Graph 无法原样重放。** swap 引入显式 `cudaMemcpyAsync`，地址随 decode step 变化，"These dynamic transfer addresses can force graph updates or re-capture, reducing the benefits of CUDA Graphs in practical serving systems"（p.41）。零拷贝的 KV block 是在 attention kernel 内部被加载（而非显式 memcpy），动态块选择只改变 kernel 输入或内部地址解析、不改变图拓扑，因此可原样重放（p.41）。
- **机会：NVLink-C2C 把 CPU 内存变成 KV 容量的延伸。** "Recent CPU–GPU superchips such as NVIDIA GH200 and GB200 [14, 15] provide up to 900 GB/s bidirectional bandwidth via NVLink-C2C—7× higher than PCIe Gen5 [16]"（p.39）。但带宽差仍在：PCIe 40–60 GB/s、HBM 3–4 TB/s（p.39）；§2.2.1 用 PCIe 64 GB/s、HBM 4 TB/s 的写法（p.41）。
- **零拷贝的粒度恰好合适。** "each SM fetches KV tiles of around 100 KB, aligned with its SMEM capacity, and more than 100 SMs can fetch such tiles concurrently. This creates MB-scale aggregate data movement from CPU memory to the GPU. In this regime, sustained CPU–GPU bandwidth becomes the dominant bottleneck"（p.41）。所以有效性的关键是 CPU–GPU 带宽，而非单次访问延迟（p.41）。
- **但朴素零拷贝反而更慢（论文的抓手）。** GEMM 案例：`C = A × B`，A/B/C 为 10240×10240 方阵（各 400 MB），三种配置——Baseline（A、B 全在 GPU 显存）、Swap（200 MB GPU buffer 暂存从 CPU 内存传过来的 B）、Zero-Copy（B 全在 CPU 内存、GPU 直接访问）（p.41）。结果：Swap 在 PCIe 上加 <10%、NVLink-C2C 上约 5% 开销（异步传输掩盖）——**精读员注：这句与 Fig.2(a) 的绝对值对不上（PCIe 56→65 ms 即 +16%），见 §五注 12**；Zero-Copy "incurs over a 20× slowdown on PCIe (1122 ms vs. 56 ms) and a 2× slowdown on NVLink-C2C (106 ms vs. 52 ms)"（p.42，图 Fig.2(a) 经 bbox 逐标签核对：Baseline 56/52、Swap 65/54 ms）。原因："Zero-Copy repeatedly fetches B from CPU memory throughout the O(n³) computation, fully exposing the bandwidth gap between CPU–GPU interconnects and HBM"（p.42）。零拷贝唯一的好处是内存：比 Baseline 省 50%、比 Swap 省 33%（p.42）。
- **隐性代价：L2 命中率塌陷。** Baseline 与 Swap 表现相近，L2 命中率约 77%、HBM→L2 流量 21.1 GB（远超矩阵本身大小，来自 O(n³) 的重复访问）；零拷贝则 32.3%（正文口径）——"a 58% drop compared to Baseline/Swap"，因为 CPU–GPU 传输（900 GB/s，单向 450 GB/s）远慢于 HBM→L2（4 TB/s）（p.42，Fig.3）。论文的判断：降低的 cache 局部性叠加带宽限制，"compounds the performance gap of naïve zero-copy offloading"（p.42）。
- **设计结论（论文自己的 takeaway）。** "By tiling through SMEM, we shift bandwidth pressure from the CPU–GPU interconnect to HBM, fully exploiting NVLink-C2C while maintaining GPU compute efficiency"（p.40）；§2.4.1 的 takeaway："Iterating over GPU-resident matrices rather than CPU-resident ones shifts bandwidth demand to HBM"（p.43）。

## 三、核心机制

### 3.1 CPU-memory-aware tiling：把 GEMM 的驻留操作数换掉（§5.1 pp.45–46；概念与量化先见于 §2.4.1 pp.42–43）

- **做什么**：重排矩阵乘的访问模式，把带宽压力从 CPU–GPU 互联挪到 HBM。
- **怎么做**（Algorithm 1，p.45）：`NativeTiling` 让 C 驻留（寄存器/SMEM），A、B tile 反复从 HBM 取；`CpuAwareTiling` 反过来——"Load B_{k,j}（CPU Memory → SMEM, zero copy）"只取一次，随后对所有 i 复用 B tile，A 与 C 从 HBM 取。B 的流量从 O(n³) 降到 O(n²)；代价是 C 每一步都要从 HBM 读回累加，流量从 2·O(n³) 涨到 3·O(n³)（p.45）。
- **关键设计选择**：接受 HBM 流量的增加，因为 "HBM bandwidth is roughly an order of magnitude higher than CPU–GPU interconnect bandwidth"（p.42）。
- **量化验证**（GEMM 案例）：零拷贝下 B 的 CPU→GPU 流量被重复访问放大到 33.5 GB（"over 3× the original tile size"）；CPU-aware 后降到 0.4 GB；HBM 流量涨到 61.8 GB + 33.9 GB（论文表述："increasing HBM traffic by about 2×（from 33.5 GB over the CPU–GPU interconnect to 61.8 GB + 33.9 GB in HBM）"）（p.42，Fig.4）。延迟 106 ms → 54 ms（49% 改进），L2 命中率 32.3% → 75.1%（p.42–43，Fig.5）。
- **两种迭代策略的取舍**（§5.1，pp.45–46）：
  - (i) 对每个 (K,V) 对迭代 Q —— 对应 CpuAwareTiling，**prefill 用**：KV 在 CPU 内存里、重载代价高，宁可按 HBM 里的 Q 迭代；
  - (ii) 对给定 Q 迭代 (K,V) —— 对应 NativeTiling，**decode 用**：Q 只有一个 token，遍历一遍 KV 即可，且 O 可以留在寄存器里跨迭代复用，不必反复回 HBM 取。
- **配套的 SMEM 预算**（§4.1，p.44）：SMEM 逻辑分 projection buffer 与 attention buffer 两区，并让二者重叠使用——K 复用 Wk 的 buffer、V 复用 Wv 的 buffer；decode 阶段 O 直接从寄存器写 HBM，prefill 阶段 O 复用 Wv 的 buffer。默认 2 级流水（使 buffering 需求翻倍），从统一 L1/SMEM 池（GH200 上 256 KB）里保守划出比例 α（默认 80%）给 SMEM，约束式为 α·P ≥ 3·m·size(T)·(Dim·N)（P 为 L1/SMEM 池大小，因子 3 对应 projection 与 attention 中需要缓冲的三个张量，m 为流水级数默认 2，N 为 tile 尺寸）；给定 <T, Dim, N> 求出最大可行 N，作为离线编译的候选集（p.44）。

### 3.2 Warp 级流水：把搬运藏进计算里（§5.2 pp.46；§5.3 pp.46–47；§5.4 p.47）

- **做什么**：在同一个 kernel 内让不同 warp group 分别负责搬运与计算，重叠 CPU 内存访问的停顿。
- **怎么做**：prefill kernel 初始化三个 warp group——producer（从 HBM 取 tile）、consumer（计算）、storer（把生成的 K/V 写回 CPU 内存；需要时写回中间结果）（Algorithm 2 Step 3，p.46）；decode kernel 只需 producer 与 consumer 两组，写回由 consumer 兼任（只写一个新的 KV 对）（§5.4，p.47）。consumer 算当前 tile 时 producer 预取下一 tile（Algorithm 2 Step 6/7、16/18；Algorithm 3 Step 5/6，pp.46–47）。论文把它类比为 swap 方案里"host–device 传输与计算重叠"的异步拷贝（p.46），并指出 Hopper 的 TMA 硬件拷贝引擎进一步强化这一机制（p.46）。
- **关键设计选择**：把不同 attention head 与 batch 分派到不同 SM，避免 O_i 的跨 SM 写–读依赖——"Since the product of the number of heads (e.g., 8) and batch size (e.g., 32) typically exceeds the number of SMs (e.g., 132 on GH200), this scheduling ensures that no two SMs simultaneously operate on the same head–batch pair with different (K,V) tiles"（p.47）。
- **保留项**：streaming softmax 的 m、ℓ 状态仍要反复从 HBM 取回累加（p.46），论文判断这笔 HBM 流量相对 CPU–GPU 传输"remains modest ... and thus does not become the dominant bottleneck"（p.46）；RoPE 按 tile 独立施加（Step 8），因为只依赖位置相关的预计算 sin/cos，无 tile 间依赖（p.47）。
- **量化验证**（§2.4.2 微基准，Fig.6，p.43）：有/无 pipe fetching 的两版传输量几乎相同，但 warp 级流水把 HBM 吞吐从 0.3 TB/s 提到 1.3 TB/s（4.3×），延迟 54 ms → 48 ms（11%）。

### 3.3 Kernel fusion：projection 与 attention 合成一个 kernel（§2.4.3 p.43；§5.3 pp.46–47）

- **做什么**：消除"K/V 写回 CPU 内存更新 cache、下一个 kernel 再读回来"的往返。
- **怎么做**：K/V tile 生成后留在 SMEM 里被后续 attention 立即消费（p.46）；X、Wk、Q 常驻 SMEM（p.43）。论文把这条描述为打破传统零拷贝里的"严格读写依赖"（"This design breaks the strict read–write dependency in traditional Zero-Copy"），减少停顿并提高有效吞吐利用率（p.43）。
- **量化验证**（§2.4.3 微基准，Fig.7，p.43）：两版传输量相近（约 125.2 GB，其中 GPU→L2 37.3 GB），分离 kernel 因冗余写回 K + 下一 kernel 重取，HBM 吞吐低 15.8%；融合版达 1.9 TB/s，延迟 85 ms → 57 ms（论文写 "a 49% speedup"）。
- **端到端对照**（§7.4.2，Fig.13，p.49）：相对分离 kernel，融合设计 HBM 吞吐最高 3.5×、per-token 延迟低 2.5–3.0×（Llama-8B、OPT-13B、OPT-30B 三模型）。

### 3.4 运行时系统：Kernel Generator / Kernel Adaptor + KV Cache Manager（§3 pp.44；§4 pp.44–45；§6 p.47；§7.1 p.47）

- **Kernel Generator（离线）**：用 C++ 模板按 <T, Dim, N> 预编译融合 kernel 候选池，覆盖不同精度（FP16/BF16）与执行阶段（prefill/decode），Listing 1 给出 prefill kernel 模板签名（`template <typename T, int Dim, int N> __global__ void prefill_kernel(...)`）与实例化方式（p.44–45）。
- **Kernel Adaptor（在线）**：按张量精度、head dimension、执行阶段（multi-token prefill / single-token decode）匹配候选池里最近的实例，过程"lightweight, as it only requires substituting the template parameters with the pre-compiled kernels"（p.44、§4.3 p.45）。
- **KV Cache Manager**：用 `cudaHostAlloc` 在 host 分配 pinned 内存，"Since these buffers are page-locked and GPU-accessible, DirectKV kernels can access CPU-resident KV blocks through device-visible pointers without staging them in GPU memory"；KV 一旦写入就留在 pinned buffer 里跨 decoding iteration 复用，不重算不重传（§6 p.47、§7.1 p.47）。
- **明确不依赖 UPT**：§8.2 说明它用的是标准零拷贝 pinned memory，而非 GH200 的统一页表（UPT）；不需要 coherence 的原因是 "KV blocks are not concurrently updated by the CPU"（p.50）。
- **实现规模**：prototype 约 5,300 行 CUDA/C++（在 FlashAttention-3 之上扩展），外挂 Python binding 以接入 PyTorch 2.2；CUDA 12.4、CUTLASS 3.0+（经 CuTe 支持 Hopper）（p.47、p.48）。

## 四、关键数字

> 全部为真机实测（GH200 与 H100-PCIe），**未见仿真或建模推算**；唯一属"建模"性质的是 §5.1 用 O(n³)/O(n²) 对数据流量的复杂度论证（p.45），以及 §4.1 的 SMEM 约束式（1）（p.44）。

| 指标                                       | 数值                                                                                                                            | 对照基线/条件                                                                                                                                | 出处                                 |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| CPU–GPU 传输量                             | 最高降 50%                                                                                                                      | **vs 朴素零拷贝**（§7.4.1 原文 "compared with naive zero-copy"）；摘要写 "compared to existing solutions"、§1 写 "Compared to prior offloading approaches [34, 46, 53]"，口径不一致，见§五 | p.39 摘要、p.40 §1、p.49 §7.4.1      |
| GPU 显存占用                               | 降 43%；DirectKV 47 GB，"saving 35GB on average"                                                                                 | Neo 86 GB / Pie 88 GB / FlexGen 74 GB（三者均值 82.7 GB → 省 35.7 GB、降 43.2%，算术自洽）；SGLang 92 GB；GH200 上限 96 GB                  | p.49 §7.3.3、Fig.11(b)               |
| 端到端性能                                 | 最高 1.2×；跨上下文长度 "1.2× average speedup"                                                                                  | vs SGLang/Neo/Pie/FlexGen；16k tokens 时对 Neo/Pie 约 1.3×、对 FlexGen 约 1.7×；32k 时 Neo/Pie/SGLang OOM                                | p.39 摘要、p.49 §7.3.2               |
| 推理延迟（组件口径）                       | 最高降 70%                                                                                                                      | CPU-aware 零拷贝 vs 朴素零拷贝（Llama-8B/OPT-13B/OPT-30B）                                                                                   | p.49 §7.4.1、Fig.12(b)               |
| HBM 吞吐 / 延迟（融合 kernel）             | 吞吐最高 3.5×；延迟低 2.5–3.0×                                                                                                  | 融合 kernel vs 分离 kernel（同三模型）                                                                                                       | p.49 §7.4.2、Fig.13                  |
| NVLink-C2C vs PCIe                         | attention 延迟降最高 4.2×                                                                                                       | 512 序列长度                                                                                                                                | p.50 §7.4.3、Fig.14                  |
| 端到端单点延迟                             | 30 req/s：Llama-3.1-8B 从 1.55–2.95 s 降至 0.75 s；OPT-13B DirectKV 0.75 s vs 其他 offload 系 1.55–3.95 s；OPT-30B SGLang 仅支持低请求率 | GH200，ShareGPT/Alpaca，请求率 ≤30 req/s，上下文 1k–32k                                                                                      | p.48 §7.3.1、Fig.10                  |
| 微基准 GEMM 延迟（n=10240，各 400 MB）     | PCIe：Baseline 56 / Swap 65 / Zero-Copy 1122 ms；NVLink-C2C：Baseline 52 / Swap 54 / Zero-Copy 106 ms                            | Swap 配 200 MB GPU staging buffer                                                                                                            | p.41 §2.3、p.42（Fig.2(a) 读数）     |
| 微基准零拷贝惩罚                           | PCIe 上 >20×（1122 vs 56 ms）、NVLink-C2C 上 2×（106 vs 52 ms）                                                                  | vs Baseline                                                                                                                                  | p.40 §1、p.42 §2.3                   |
| 微基准 Swap 开销                           | 正文口径：PCIe "less than 10%"、NVLink-C2C "about 5%"；Fig.2(a) 绝对值实为 +16%（56→65 ms）与 +3.8%（52→54 ms）                  | vs Baseline                                                                                                                                  | p.42 §2.3、Fig.2(a)（冲突见§五注 12）|
| 微基准零拷贝内存收益                       | 比 Baseline 省 50%、比 Swap 省 33%                                                                                              | 同上（GEMM 微基准，非端到端）                                                                                                                | p.42 §2.3                            |
| L2 命中率                                  | Baseline 77.0% / Swap 76.8% / Zero-Copy 32.3%（正文口径）；HBM→L2 流量：Baseline 21.1 GB、Swap 21.1 GB                          | GH200；Fig.3 零拷贝组橙柱 22.7（GPU→L2）、蓝柱堆叠顶标 56.3（≈22.7+33.5）、绿柱 32.3 | p.42 §2.3、Fig.3（读图见§五注 4） |
| CPU-aware tiling 的 GEMM 收益              | 延迟 106 → 54 ms（49%）；L2 命中率 32.3% → 75.1%；B 的 CPU→GPU 流量 33.5 → 0.4 GB；HBM 流量 61.8 GB + 33.9 GB                    | vs 朴素零拷贝                                                                                                                                | p.42–43 §2.4.1、Fig.4/Fig.5          |
| Warp 级流水收益                            | HBM 吞吐 0.3 → 1.3 TB/s（4.3×）；延迟 54 → 48 ms（11%）                                                                          | 两版传输量几乎相同                                                                                                                           | p.43 §2.4.2、Fig.6                   |
| Kernel 融合收益（微基准）                  | HBM 吞吐 1.9 TB/s（分离版低 15.8%）；延迟 85 → 57 ms（论文称 "49% speedup"）                                                     | 两版传输量均约 125.2 GB                                                                                                                      | p.43 §2.4.3、Fig.7                   |
| 实现规模                                   | 约 5,300 行 CUDA/C++（基于 FlashAttention-3）+ Python binding                                                                    | CUDA 12.4 / CUTLASS 3.0+ / 接入 PyTorch 2.2                                                                                                  | p.47 §7.1、p.48 §7.2                 |
| 硬件与带宽参数                             | GH200：96 GB HBM3 + LPDDR5X（**容量论文未给**）；L1/SMEM 池 256 KB；α=80%；SM 数举例 132；NVLink-C2C 900 GB/s 双向（单向 450 GB/s）；HBM 4 TB/s；PCIe 64 GB/s（§1 另写 40–60 GB/s）；TMA 存在于 Hopper | 对照平台 H100 PCIe Gen5                                                                                                                      | p.39、p.41、p.42、p.43、p.44、p.46、p.47、p.48 |
| 评估配置                                   | 模型 Llama-3.1-8B、OPT-13B、OPT-30B；数据集 ShareGPT、Alpaca；Poisson 到达，请求率 ≤30 req/s；上下文 1k–32k；baseline = SGLang、Pie、Neo、FlexGen（前两者为系统对照，FlexGen 用 4-bit 压缩） | 主实验区间被论文自述为 "a stable operating region"；另有压力场景但未给数字                                                                    | p.48 §7.2                            |

## 五、局限与边界

**论文自认的限制：**

1. **硬件前提是宽互联。** §8.2："DirectKV is best suited to CPU–GPU platforms with high-bandwidth interconnects, such as NVLink-C2C on GH200/GB200. Its coarse-grained access pattern—∼100 KB KV blocks accessed concurrently by over 100 SMs—creates MB-scale aggregate transfers, making performance primarily limited by sustained CPU–GPU bandwidth rather than individual remote-load latency. Thus, on PCIe-only platforms, DirectKV may still reduce GPU memory usage, but throughput benefits are often limited by interconnect bandwidth."（p.50）§7.4.3 进一步把 PCIe 场景定性为"容量扩展机制"（capacity-extension mechanism）而非性能手段（p.50）。
2. **只做节点内。** §8.3：node-local heterogeneous-memory optimization；"Its main limitations are context-parallel scale-out and host-memory pressure: coordinating remotely owned KV blocks or integrating with a multi-tier KV hierarchy would require additional runtime support, and zero-copy benefits diminish once accesses fall back to pageable or disk-backed tiers. We leave these extensions to future work."（p.51）
3. **不改上层语义。** §8.1：定位为补充而非替代，"DirectKV does not change scheduling, batching, or KV-management policies; instead, it provides an efficient execution path for CPU-resident KV blocks"（p.50）。
4. **依赖 pinned 内存，且要求 CPU 不同时改写 KV。** 用 `cudaHostAlloc` 的 page-locked buffer（§6 p.47），不需要 UPT 的 coherence 是因为 KV block 不被 CPU 并发更新（§8.2 p.50）。

**精读员注（疑点与空白）：**

1. **精度验证缺席。** 全文没有任何 accuracy / perplexity / 误差实验；唯一相关表述是 §1 的 "all while maintaining full attention accuracy"（p.40）。融合 kernel 改了 softmax 累加路径与 RoPE 施加位置，论文未给数值等价或误差对比 → **未核实**。
2. **口径不一致（50%）。** §7.4.1 明确是 "compared with naive zero-copy"（p.49），摘要与 §1 却写成 "compared to existing solutions" / "Compared to prior offloading approaches [34, 46, 53]"（p.39、p.40）。同一句里的 43% 与 1.2× 是**对系统 baseline** 的口径，50% 却是**对朴素零拷贝**的口径——引用时不可混用。
3. **50% 与 70% 是一组，不是两组独立收益。** "up to 50% transfer volume" 与 "up to 70% lower inference latency"（Fig.12）都出自 §7.4.1 的 CPU-aware vs 朴素零拷贝对照，不是 vs Pie/Neo/FlexGen。
4. **§2.3 正文与 Fig.3 读法冲突。** 正文写 "increasing CPU-to-GPU traffic to 22.7 GB while reducing the L2 hit rate to 32.3%—a 58% drop compared to Baseline/Swap"（p.42），但 Fig.3 的零拷贝组是：橙色柱（GPU Memory→GPU L2）22.7、蓝色堆叠柱顶标 56.3（左侧数据轴）、绿色柱（Hit Rate）32.3（右侧轴）。用 Fig.4 交叉核对：零拷贝下 GPU→L2 22.7 GB、CPU→L2 33.5 GB，22.7 + 33.5 ≈ 56.2。因此合理读法是 GPU→L2 22.7 GB、CPU→L2 约 33.5 GB、命中率 32.3%，正文的 "CPU-to-GPU traffic to 22.7 GB" 当属笔误（本报告按图 + Fig.4 交叉理解；Fig.5 里零拷贝命中率亦标 32.3%，与正文一致）。
5. **CUDA Graph 是纯论证、无实验。** §2.2.2 用一段话论证零拷贝更适合 CUDA Graph replay（p.41），评估里没有任何 CUDA Graph 相关数据。
6. **评估硬件单一。** 只有 GH200 一台机器（+ H100 PCIe 做互联对照，p.48）；GB200 只出现在标题性表述里（p.39、p.50），无实测。最长上下文 32k（Fig.11），而 motivation 举的是 128K（p.39）→ 128K 场景**未核实**。
7. **host 侧参数缺失。** GH200 的 LPDDR5X 容量、host 内存带宽均未给出（p.48 只写 "LPDDR5X CPU memory"）；"CPU memory as cheap storage" 这一前提在 LPDDR5X 与普通 DDR5 上的差别未讨论 → 容量**未核实**。
8. **无敏感性分析。** α=80%、流水级数 2、tile 尺寸 N 均为固定缺省（p.44），未给敏感性数据。
9. **吞吐指标缺位。** 所有端到端结果以 per-token latency（秒）与 request rate（req/s）呈现，没有 tokens/s 吞吐或尾延迟（P99）；主实验区间自述为 "stable operating region"（请求率 ≤30 req/s），压力场景只声明做了、未给数字（p.48）。
10. **集成路径未实测。** §8.1 的"扩 KV allocator 支持 CPU-pinned 存储"是设计论述（p.50）；端到端对照是 DirectKV 自家 kernel 栈 vs 原系统整体，不存在 "vLLM/SGLang + DirectKV" 的实测数据。
11. **一个隐含前提值得记下：** 零拷贝要求 CPU 不并发写 KV block（p.50）。多进程/多线程 serving 里若有 CPU 侧 KV 复用或共享内存场景，这条前提需要重新审。
12. **Swap 开销的正文与图不一致。** §2.3 正文称 Swap 配 200 MB buffer 在 PCIe 上开销"不到 10%"、NVLink-C2C 上"约 5%"（p.42），但 Fig.2(a) 的绝对值（经 bbox 逐标签核对）是 Baseline 56/52 ms、Swap 65/54 ms：PCIe 一侧为 +16%，与"不到 10%"不符；NVLink-C2C 一侧 +3.8%，与"约 5%"相符。引用时用图中绝对值更稳。

## 六、写作素材

- **系列定位（四种"内存账"之一：零拷贝去搬运）**：DirectKV 解的是"KV 放在 CPU 内存里，但不必搬进显存"——取消 staging buffer（no buffer）、把 CPU–GPU 互联的搬运量压下去（no bottleneck），靠的是 kernel 直接寻址 host pinned memory + 访问模式重排。它与 NEO（计算下移）、eLLM（抽象虚拟化）、Strata（分层调度）共同构成系列：**同一块 CPU 内存，四种花法。**
- **与另外三篇的对照点**：
  - **vs NEO（计算下移）**：NEO 把 decode attention 的计算连同 KV 一起搬到 CPU，用"CPU 算"换显存；DirectKV 保留全部 GPU 算力，只把 KV 的存放地挪到 CPU、让 GPU 直接寻址。两者前提互反：NEO 要求 CPU 算力够用（并接受 GPU 计算单元闲置），DirectKV 要求 CPU–GPU 互联够宽。论战线索可以直接引用——DirectKV §1 批评"把 attention 下放 CPU"的路线 "sacrifices the full compute capability of the GPU"（p.39），而 NEO 正是被它列为 baseline 的 [34]（p.48）。
  - **vs eLLM（抽象虚拟化）**：eLLM 的路线是在内存管理层引入弹性抽象（其具体机制以该篇精读报告为准，本报告不做断言）；DirectKV 相反，不新增抽象层、不改调度与 KV 管理接口——"leaving the scheduler, model execution graph, and frontend API unchanged"（p.50）。对照句可以这么写：**一边靠抽象让 CPU 内存"看起来像"显存，一边靠 kernel 直接寻址让它"不必被搬运"。**
  - **vs Strata（分层调度，同为 OSDI '26）**：两篇都在解"KV 躺在 CPU 内存里怎么被高效用到"，但切法相反。Strata 保留 copy 语义、修的是 I/O 路径与调度本身（GPU-assisted I/O kernel、GPU/host 布局解耦、把 CPU-GPU 带宽当一等资源），其测量给出反面证据：SGLang 常规配置下 74% 的 prefill 时间阻塞在 KV 传输、page size 32 时 8192 token 的 KV 传输只跑到 PCIe 5.0 理论带宽的 ≈22%（Strata pp.1–4）；DirectKV 干脆取消搬运，并刻意把访问做成粗粒度以免撞上小传输低效——"∼100 KB KV blocks accessed concurrently by over 100 SMs—creates MB-scale aggregate transfers"（p.50）。可提的对照问题：**取消搬运 vs 优化搬运，各自的边界条件是什么**（前者要 NVLink-C2C 级带宽，后者要能把传输做大、把停顿排掉）。
- **与本仓已有文章可勾连处（候选）**：
  - `01_concepts/offloading/01_kv_offloading.md`（offload 全景）、`02_layerwise_pipeline.md`（layer-by-layer swap 正是 DirectKV §2.2.1 批评的对象，p.41）、`sparse_attention_driven_offloading_problems.md`（offload 粒度与被访问模式绑定的问题）
  - `02_systems/lmcache/local_cpu_backend.md`、`lmcache_storage_overview.md`（CPU 内存 tier 的工程实况；论文 §8.1 引 LMCache [24] 作为 hierarchical KV caching 的既有机制）、`cachegen.md`（KV 流式传输的另一条路，论文引为 [39]）
  - `01_concepts/basic/attention_kv_cache_formats.md`（head dim / KV 布局）、`01_concepts/capacity_planning/hbm_kv_pool_concurrency.md`、`kv_cache_roi.md`（"KV 该放哪一层"的成本账）
  - `03_fast26/` 系列（SolidAttention 把这条账链延伸到 SSD；可作为"再下一层"的对照）
  - `04_memory_tiering/` 本篇 + NEO/eLLM/Strata 三篇 + 总纲；`02_systems/mooncake`、`nixl`（跨节点/传输层对照）
- **适合放文章开头的"反差/场景"素材**：
  - **"不搬反而更慢"**：把 KV 留在 CPU 内存、让 GPU 直接读（零拷贝），在 PCIe 上慢 20 倍（1122 ms vs 56 ms），在 NVLink-C2C 上也还慢 2 倍（106 ms vs 52 ms）——先给这个反直觉数字，再解释为什么"零拷贝"必须配"内核协同设计"（p.42）。
  - **"显存 92 GB → 47 GB"**：96 GB 的 GH200 上，SGLang 把权重和 KV 都放 HBM、平均吃 92 GB 并在长上下文 OOM；DirectKV 把 KV 放 host pinned 内存，显存 47 GB（p.49）。
  - **"一个 1.2×，两种口径"**：论文明面上给的是 1.2×（对系统），真正扎眼的数字是组件级的 50%/70%（对朴素零拷贝）——写文章时把两笔账分开讲，本身就是有用的阅读示范。
  - **"不搬 KV，搬瓶颈"**：论文标题的自述 "no buffer, no bottleneck" 是同一件事的两面——去掉 staging buffer（省显存）与去掉带宽瓶颈（提性能）在零拷贝方案里是同一个动作的两个后果。
