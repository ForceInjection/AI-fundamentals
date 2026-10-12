# eLLM（DAC 2026）精读报告

> 来源：papers/ellm-arxiv-2506.15155.pdf（本目录下的 papers/；arXiv 版 v2，PDF 共 8 页）。核对方式：全文抽取逐条对照，数字标页码「p.X」。只有推断/点评标「精读员注」；未核实写「未核实」。

## 一、速览

- 完整标题（含副标题）：eLLM: Elastic Memory Management Framework for Efficient LLM Serving（无副标题，p.1）
- Venue：DAC '26，第 63 届 ACM/IEEE Design Automation Conference，2026-07-26~29，Long Beach, CA, USA；ACM 论文集 8 页；DOI 10.1145/3770743.3804144；arXiv:2506.15155v2 [cs.DC]（页眉标注 v2 日期 2026-05-06，p.1）
- 作者与机构：Jiale Xu\*、Yi Xiong\*、Rui Zhang\*（三人共同一作，equal contribution）；Cong Guo†、Zihan Liu、Yangjie Zhou、Weiming Hu、Hao Wu、Boyu Li、Junping Zhao†、Minyi Guo（过敏意）、Zongwei Zhu†、Xuehai Zhou、Jingwen Leng（冷静文）。† 为通讯作者：Cong Guo、Junping Zhao、Zongwei Zhu。机构：上海交通大学、上海期智研究院、中国科学技术大学、新加坡国立大学、以及若干标注 Unaffiliated 的产业界作者（p.1）
- 一句话贡献：提出 eLLM：用统一虚拟张量抽象 eTensor 把 activation 与 KV cache 放进同一个物理内存池，以「内存气球式」的 inflation/deflation 让两者互相借内存，外加 CPU DRAM 后备与 SLO 感知调度；解码吞吐最高 2.32×（p.1、p.6）
- 开源/集成情况（仓库、落地系统）：**论文未提及任何代码开源、artifact 或落地系统**（全文含致谢在内无仓库链接；「未核实」是否存在后续开源）。集成形态是修改 vLLM v0.5.5 的私有实现：替换 KV Cache tensor object、改 PyTorch caching allocator（cudaMalloc → GPU VMM API），约 4000 行 C++/Python（p.5 §3.2）。相关工作 GMLake（同组，ASPLOS '24）与 vAttention（同期同问题域）见 §5（p.6）

## 二、问题与动机

论文把痛点命名为 **space-wise internal fragmentation**（空间维度内部碎片），根因是 LLM 服务的**两层内存抽象割裂**：

- **运行时内存（activation）与 KV cache 分属两套抽象**。activation 继承 PyTorch 等框架的静态张量抽象，直接分配物理内存块、用 memory pool 复用；KV cache 走 PagedAttention 页表虚拟化。两套机制都不能感知对方（Figure 1 三联图：PyTorch 碎片 → vLLM 隔离 → eLLM 统一，p.3）。
- **割裂的直接后果**，原文：「the two memory spaces cannot share resources, even when the KV cache space is exhausted while the activation memory sits idle (or vice versa), the underutilized space cannot be repurposed. We refer to this phenomenon as space-wise internal fragmentation.」（p.2）
- **量化（p.2 §1）**：
  - 「increasing the maximum context of MTP-30B from 2K to 200K on the same A100 GPUs increases the activation reservation from only 0.3% to 30.8% of GPU memory.」——预留守着 30.8% 显存，而「99% of prompts fall within 2K tokens in ShareGPT」；
  - 「more than 90% of prefill batches use less than 30% of the model's maximum context length」；
  - decode 阶段激活利用率更低：「even lower activation space utilization (∼1%)」；CoT 拉长解码周期后，「this extremely low utilization increasingly dominates the inference process」。
- **摘要给出的系统级代价**：「leading to suboptimal utilization and 20% throughput degradation」（p.1）。论文用 Figure 2 的内存快照对比（vLLM 的 "Isolated!" 与 eLLM 的 "20% Speedup!"）示意互借可回收约 20% 性能（p.2、Figure 2 图注 p.3 记为 1.2× speedup）。
- **存量优化为何不够（p.2）**：调度层方案被逐一指出局限。Chunked Prefill 有 read amplification（每 chunk 需重读之前全部 KV，内存开销随序列长度超线性）；PD 分离方面，论文援引 semi-PD 的总结——原文写「they also exhibit four limitations [8]」，但正文实列三条：(1) Model Replication Overhead、(2) Memory Utilization Skew、(3) Inter-instance Communication Cost。论文明确自己的定位：「we directly address the space-wise internal fragmentation at the memory management level, avoiding their limitations and remaining orthogonal to them.」
- **两条 naive 统一路线被否（p.3 §2.1）**：把 KV 交给框架级细粒度管理会引入大量元数据开销、破坏 KV 的 sharing 结构；做类型无关的统一虚拟化会忽略 activation 与 KV 访问模式的本质差异。

精读员注：20% 这个数字在论文里出现两次、含义不同——摘要的 20% 是「碎片造成的吞吐损失」（问题量化），§1 与 Figure 2 的 ~20%/1.2× 是「弹性互借带来的吞吐收益」（机制演示）。引用时须写明是哪一侧。

## 三、核心机制

三组件自底向上：Virtual Tensor Abstraction → Elastic Memory Mechanism → Lightweight Scheduling Strategy（p.2 §2）。

### 3.1 Virtual Tensor Abstraction：eTensor 逻辑分离、物理归一

- **做什么**：在逻辑层把 KV cache 与 activation 视为同一抽象层级上的两个独立逻辑实体（各自保留定制化管理），物理层共享一个 unified memory pool，池内可互相转换与借用（Figure 3，p.3）。
- **怎么做**：两种 eTensor 类型分别适配访问模式（p.3 §2.1.1）：
  - **KV eTensor**：大而规整、增长可预测、访问频度低、全程驻留。每个请求**预分配等于最大上下文长度的虚拟地址空间**（保证逻辑连续、便于顺序访问和索引计算），物理块按写入 on-demand 分配。
  - **activation eTensor**：小块、生命周期短、访问频度高，用变长虚拟地址段、细粒度管理。
  - 两者的虚拟地址段（tensor slots）**严格按物理块粒度对齐**，在访问效率与碎片控制间取平衡。
- **关键设计选择**——内存池策略（p.3 §2.1.2）：eTensor 生命周期结束后**不立即 unmap**，标记为 mapped 且 available 的 tensor slots 并记录 mapping size，供复用（省映射开销）。KV eTensor pool 用 **Best-Fit**：给定可用槽集合 R 与目标大小 s，选 arg min{size(r_j) | size(r_j) ≥ s}，无满足者触发 on-demand mapping；activation eTensor pool 保留框架原生 **BFC**（Best-Fit with Coalescing）。
- 页码：p.3 §2.1–§2.1.2、Figure 3（p.3）。

### 3.2 Elastic Memory Mechanism：inflation/deflation + GPU-CPU 弹性

**intra-GPU：inflation / deflation（p.3 §2.2.1–p.4）**——像给气球充放气一样借还内存（原文「much like inflating a balloon」，p.4）：

- Inflation 四步：❶ Inflation trigger：KV 分配请求先查 KV pool 是否够，不够则向 activation pool 发借用请求；❷ Memory reclamation：activation pool 触发**轻量 GC**，unmap 不活跃 eTensor 对象占的物理块；❸ Ownership transfer：回收块**逻辑上**从 activation pool 迁移到 KV pool（所有权转移）；❹ On-demand remapping：GPU VMM 把块动态映射进目标 KV eTensor 虚拟地址空间。Deflation 是逆过程，**惰性触发**（lazy）以免无谓开销。
- Figure 4 走了一个四状态例子（p.4）：step-0 历史 KV 占满 → step-1 prefill inflation 借 idle activation chunks 立即接新请求 → step-2 inflation 支撑更大 batch → step-3 decoding 完成后 deflation 归还。
- 关键设计选择：膨胀/收缩作用在**映射关系层**（mapping relation propagation），不动数据本身，只改虚拟-物理绑定。

**GPU-CPU 弹性：offloading / fetching（p.4 §2.2.2）**：

- 做什么：CPU DRAM 作为 GPU 内存的 elastic buffer。prefill 阶段主动把部分请求的 KV 卸到 CPU，降低内存准入门槛（admission barrier）、缓解排队，改善 TTFT；同时能聚合更大 decoding batch 提吞吐。
- 技术依据三条观察（p.4）：新生成且暂不需要的 KV 可先卸后取（等调度到 decoding 再 fetch）；Transformer 多层结构天然支持 layer-wise pipelining 重叠传输与计算；KV 迁移是 O(N) 线性通信，而 prefill 的 self-attention 是 O(N²)，传输可以被藏住——论文称在 A100 + LLaMA-3-8B 的真实负载下「offloading overhead can be completely hidden」。
- 关键设计选择：论文自认 CPU buffer 改善排队的同时引入 **prefill-preference，伤 TPOT**，所以必须配 §2.3.2 的折衷策略（p.4）。

### 3.3 Lightweight Scheduling Strategy：SLO 感知的弹性编排

- **请求调度（p.4 §2.3.1）**：利用 activation 需求在推理两阶段差异巨大的特点做 phase-specific allocation——prefill 阶段（activation/KV 需求都大）把 KV 卸到 CPU buffer 换 GPU 压力；decoding 阶段（activation 需求极小）把 KV fetch 回 GPU。两阶段都靠 inflation/deflation 把利用率顶满。
- **SLO-aware buffer scaling（p.4 §2.3.2）**：定义 **logical buffer**——物理 buffer 固定容量内可动态调整的可用尺寸抽象，**调整时不逐出已存数据**。缩放规则：检测到 TPOT violation → 缩小 logical buffer，限制 prefill 请求，保 TPOT；检测到 TTFT violation → 扩大 buffer，改善 TTFT。violation 判定：TTFT 或 TPOT 超过 SLO 阈值**在一个调度窗口（5 iterations）内出现 3 次**；调整因子 α 为超参，默认 α=2。

### 3.4 实现层：把 (un)map 开销藏进计算（p.4–5 §3.1）

- **Decoding speculative pre-mapping**：自回归每序列每迭代只产 1 token，按 branch speculation 思路假定会产下一 token，提前异步发起分配；用不到 50MB 的额外内存把映射开销完全盖住。
- **Asynchronous unmapping**：利用 GPU VMM「一个物理块可映射到多个虚拟地址」的能力，新 tensor slot 先接收物理块，旧槽位的 unmap 异步稍后做。
- 实测净开销见 §四（CPU 调度 <1%、VMM 操作 1–5%）。

## 四、关键数字

全部为论文报告值。测试台（p.5 §4.1）：8×NVIDIA A100 (80GB) NVLink + Intel Xeon Platinum 8369B + 1TB RAM；PyTorch 2.4.0、CUDA Driver 12.4、vLLM v0.5.5。SLO 定义为无竞争下基准延迟的 25×（LoongServe 口径），goodput 定义为达到 90% SLO attainment 的最大请求速率（p.5）。

| 指标 | 数值 | 对照基线/条件 | 出处 |
| ---- | ---- | ------------- | ---- |
| 解码吞吐（离线） | **最高 2.32×** | vLLM 为基线，128k-4k 配置，Jamba-Mini 2 GPU（实测） | p.6 §4.4、Figure 7（p.6） |
| 总吞吐（离线） | 最高 1.82× | 同上条件（实测） | p.6 §4.4、Figure 7 |
| Max batch size（离线） | ≈3× | 128k/4k，Jamba-Mini 2 GPU，归一化 vLLM；**读图值，正文未给数字** | Figure 7（p.6），精读员注 |
| Max batch size（单卡） | ≈2× | Llama3-8B-262K 1 GPU，8k/2k–32k/4k 各配置均为 ~2（读图） | Figure 7 右图（p.6） |
| TTFT（在线单卡） | 最高快 **295×**（vs vLLM）、**140×**（vs vLLM-CP） | 1×A100，Llama3-8B-262K，2K-2K 与 ShareGPT 两个负载取最大（实测） | p.5 §4.2 |
| Goodput（在线单卡） | 最高 **2.5×**（vs vLLM）、**2.26×**（vs vLLM-CP） | 90% SLO attainment 口径（实测） | p.5 §4.2、Figure 5c/5g |
| 吞吐（在线多卡） | 1.5× | vs DistServe（OPT-13B，2 GPU；DistServe P=1,D=1，其余 TP=2）（实测） | p.6 §4.3 |
| Goodput（在线多卡） | 2.5× | 同上 | p.6 §4.3 |
| 消融：组合 TTFT 降幅 | 最高 295× | vLLM 基线，2K-2K 负载，intra+inter 组合（实测） | p.6 §4.5 |
| 消融负向结果 | 组合吞吐**有时低于**仅 intra-GPU | PCIe 传输开销在 decoding 阶段无法完全重叠（实测） | p.6 §4.5 |
| CPU 调度开销 | <1% 总执行时间 | 在线服务（实测） | p.6 §4.6 |
| VMM 操作开销 | 1–5% 总执行时间 | 在线服务（实测） | p.6 §4.6 |
| 投机预映射额外内存 | <50MB | decoding 阶段（论文口径） | p.4 §3.1 |
| 碎片导致的吞吐损失 | 20% | 论文对问题侧的量化表述（原文口径，未见推导细节） | p.1 Abstract |
| 弹性互借收益 | ~20% / 1.2× speedup | Figure 2 内存快照对比（论文示意数据） | p.2 §1、Figure 2 图注（p.3） |
| Activation 预留占比 | 0.3% → 30.8% GPU 显存 | MTP-30B，max context 2K→200K，A100（论文测算） | p.2 §1 |
| Activation 利用率 | prefill：>90% 的 batch 用不到 max context 的 30%；decode：∼1% | ShareGPT 统计：99% prompts ≤2K tokens（论文口径） | p.2 §1 |
| 实现规模 | ~4000 行 C++/Python | 基于 vLLM v0.5.5 修改（论文自述） | p.5 §3.2 |
| 调参默认值 | α=2；violation 窗口 5 iterations 内 3 次 | SLO-aware buffer scaling（论文设定） | p.4 §2.3.2 |
| 测试床 | 8×A100 80GB NVLink + Xeon 8369B + 1TB RAM | PyTorch 2.4.0 / CUDA 12.4 / vLLM v0.5.5 | p.5 §4.1 |
| SLO 定义 | 25× 无竞争 TTFT/TPOT；goodput 取 90% attainment | LoongServe 口径引用 | p.5 §4.1 |

口径说明：表中「实测」= A100 测试台上的测量结果（单测试台、论文未开源，无法第三方复核）；「论文测算/口径」= 论文给出的统计或推算，无独立推导过程；「读图」= 数值从 Figure 7 柱状/折线读取，正文无对应数字。摘要的「up to 2.32× higher throughput」与离线 128k-4k 的解码吞吐 2.32× 是同一口径（p.1 与 p.6 互证）。

## 五、局限与边界

论文自认的代价与限制：

- **TPOT 侧让步**：CPU buffer 改善排队但引入 prefill-preference、损害 TPOT（p.4 §2.2.2）。§4.2 原文：「because eLLM and vLLM-CP have more available KV Cache space for decoding, their TPOT is higher than vLLM.」（p.5）论文以 TTFT/TPOT 的取舍换 SLO attainment 与 goodput。
- **PCIe 是不可绕过的账单**：消融显示 intra-GPU 与 GPU-CPU 两种弹性组合后，吞吐有时反而低于仅 intra-GPU，原因是 decoding 阶段的 PCIe 传输无法完全重叠（p.6 §4.5）；单靠 intra-GPU 弹性又挡不住高负载下的 prefill-phase 瓶颈、goodput 退化（p.6 §4.5 尾）。eLLM 的解法是「动态双机制编排」（dual-mechanism orchestration）来平衡两者的收益/成本（p.6）。
- **前提条件**：GPU-CPU 弹性的收益建立在 CPU DRAM 容量（测试台给了 1TB）与 PCIe 带宽之上；论文引用的「offloading overhead 完全隐藏」给出前提是 A100 + LLaMA-3-8B 这一组真实负载（p.4）。

精读员注（推断与疑点）：

1. **295× 是高负载排队场景的极值口径**。其来源是 vLLM 在请求速率升高后 TTFT 崩溃（Figure 5a/5e 的对数轴曲线），比值不等于低负载下的稳态差距。引用必须带负载条件，否则严重误导。
2. **≈3× batch 是我从 Figure 7 读图所得**，正文只写「maintains larger batch sizes than vLLM」；且该最优值是 Jamba-Mini（高 activation-to-KV 比的 Mamba-MoE 混合架构、2 GPU、128k/4k）配置，不是通用保证。系列总纲里「128K batch 扩 3×」的表述建议补「Jamba 配置、读图值」限定。
3. **§4.2 那句 TPOT 归因存疑**：原文将「TPOT 更高」归因于「有更多可用 KV Cache 空间」，机制上更自然的解释是 admit 了更大的并发 batch（每 token 延迟升高、吞吐与 SLO 达成率上升）。引用该句时注意其字面归因。
4. **对比基线不含 KV offload 系统**：GPU-CPU 弹性只与 vLLM 自身比，未对照 LMCache/Mooncake 等同代 CPU offload 方案，无法判断「eLLM 式 offload」相对专用系统的优劣。
5. **未报告的维度**：精度（纯内存管理理论上无损，但论文未给数据）、能耗、CPU 端开销、多租户/安全；「lightweight GC」的触发阈值与回收粒度、inflation 在极端负载下的最坏情形均未展开（8 页篇幅所限）。
6. **不可复现风险**：不开源 + 强依赖 GPU VMM 与修改过的 PyTorch allocator，第三方复现成本高；所有数字来自单一 8×A100 测试台。
7. MTP-30B 只出现在动机章节的测算里，不在评估模型列表（Llama3-8B-262K、Jamba、OPT-13B）中。
8. 原文称 PD 分离「four limitations」但正文只列三条，随后直接转折到「Unlike these approaches」——引用「四条限制」说法时注意原文自身的计数与列表不一致（p.2）。

## 六、写作素材

- **系列定位（四种「内存账」之一：eLLM=抽象虚拟化；DirectKV=零拷贝去搬运；NEO=计算下移；Strata=分层调度）**：eLLM 管的是「同一块 HBM 里两个内存池之间的账」。它不新增硬件层级、不搬实现路径，先把 activation 与 KV cache 从两层割裂的抽象里解放出来放进同一弹性池——碎片问题在内存管理层直接消掉（「we directly address the space-wise internal fragmentation at the memory management level」，p.2）。这是四篇里唯一的「池内再分配」路线：DirectKV/NEO/Strata 都在动 HBM 与下层/旁路的关系，eLLM 只在池内挪额度。
- **与另外三篇的对照点**：
  - **vs DirectKV（零拷贝去搬运）**：eLLM 的 GPU-CPU 弹性仍走 PCIe 搬运，消融里它正是「无法完全重叠」的短板（p.6 §4.5）；DirectKV 在 GH200 NVLink-C2C 上想消灭的恰是这笔搬运成本。两篇合读能看到「eLLM 用 O(N) 传输藏 O(N²) 计算」的脆弱性——当互连不是 NVLink-C2C 时账更紧。
  - **vs NEO（计算下移）**：NEO 把 attention 计算搬到 CPU，使 KV 不必回传；eLLM 选择把 KV 直接搬回 GPU，靠 prefill 计算量级差藏传输。一个消去搬运，一个让搬运可被藏住，是同一成本项的两条对抗路线。
  - **vs Strata（分层调度）**：Strata 建 GPU→CPU→SSD 的层级与缓存感知调度并进入 SGLang；eLLM 在层内做弹性重分配、改动点更靠下（VMM 映射层）。两者正交、可叠加：eLLM 的池内弹性解决「同一层内的账」，Strata 解决「层与层之间的账」。
- **与本仓已有文章可勾连处（候选）**：
  - [paged_attention.md](../../01_concepts/basic/paged_attention.md)：eLLM 的前提与批判对象——PagedAttention 解了 KV 碎片，同时造出「activation 静态抽象 vs KV 页表虚拟化」两层割裂；eLLM 的 Figure 1(b)→(c) 就是接着这篇文章往下讲。
  - [hbm_kv_pool_concurrency.md](../../01_concepts/capacity_planning/hbm_kv_pool_concurrency.md)：「L1 KV Pool 是并发上限」的现象，eLLM 给出了一条互补解法——激活池闲置时把它借给 KV；两篇合读可写「KV pool 上限的两种解题方向」。
  - [01_kv_offloading.md](../../01_concepts/offloading/01_kv_offloading.md) 与 [02_layerwise_pipeline.md](../../01_concepts/offloading/02_layerwise_pipeline.md)：eLLM 的 GPU-CPU 弹性本质是一种 offload（逻辑 buffer 缩放 vs LMCache 的存储分层），且其「技术依据之二」就是 layer-wise pipelining。
  - [01_vllm_chunked_prefill.md](../../01_concepts/scheduling/01_vllm_chunked_prefill.md)：eLLM 把 vLLM-CP 当基线之一，并直接引用其 read amplification 缺点（p.2）。
  - [01_disaggregated_prefill_kv_transfer.md](../../01_concepts/pd_transfer/01_disaggregated_prefill_kv_transfer.md)：DistServe 对照（eLLM 在动机与多卡实验两处逐条回应论文列出的 PD 分离限制，p.2/p.6）。
  - [glm5_kv_cache_capacity_planning.md](../../01_concepts/capacity_planning/glm5_kv_cache_capacity_planning.md)：容量规划视角——「给 activation 的预留如何蚕食可用 KV 容量」正是 eLLM 要抹掉的账。
  - [kv_cache_roi.md](../../01_concepts/capacity_planning/kv_cache_roi.md)：把「20% 的吞没在碎片里」翻译成 ROI 语言的材料。
- **适合放文章开头的「反差/场景」素材**：
  - 反差数字组：为 200K 上下文预留的 activation 空间占 30.8% GPU 显存（p.2），但 99% 的 ShareGPT prompt 在 2K 以内、decode 阶段激活利用率约 1%——「预留按最坏情况，使用按典型情况」。
  - 场景：vLLM 显存快满时新请求只能排队等 KV 释放（p.2 开头），同一时刻 activation 池里躺着大块按 max context 预留、实际闲置的空间——两池之间「不能互相借」（Figure 2a 标着 "Isolated!"，快照对比标着 "20% Speedup!"）。
  - 术语钩子：操作系统里的 memory ballooning（虚拟机气球）被搬到 LLM serving 的显存管理——inflation/deflation 借还内存（p.4）。
