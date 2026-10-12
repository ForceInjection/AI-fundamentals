# Strata（OSDI '26）精读报告

> 来源：papers/strata-osdi26.pdf（本目录下的 papers/，PDF 共 17 页）。核对方式：全文抽取逐条对照，数字标页码「p.X」——p.X 为论文印刷页码（本 PDF 第 1 页为 USENIX 封面页，印刷页 = PDF 页号 − 1，正文 pp.1–16）。只有推断/点评标「精读员注」；未核实写「未核实」。

## 一、速览

- 完整标题（含副标题）：Strata: Hierarchical Context Caching for Long Context Language Model Serving（p.1）
- Venue 与页码：OSDI '26——第 20 届 USENIX Symposium on Operating Systems Design and Implementation，2026-07-13~15，Seattle，WA（p.1 封面）；本 PDF 正文印刷页码 pp.1–16。论文集连续页码：未核实（封面未标注，仅有 ISBN 978-1-939133-55-7）。
- 作者与机构：Zhiqiang Xie（Stanford University & NVIDIA）、Ziyi Xu（上海交通大学）、Mark Zhao（University of Colorado Boulder）、Yuwei An（CMU）、Vikram Sharma Mailthody（NVIDIA）、Scott Mahlke（NVIDIA & University of Michigan）、Michael Garland（NVIDIA）、Christos Kozyrakis（NVIDIA & Stanford）（p.1）
- 一句话贡献：给长上下文 LLM 服务的分层上下文缓存补两块短板——GPU-assisted I/O 把 KV 搬运改成「GPU 内核搬小块 + GPU/host 布局解耦」，cache-aware 调度处理 delay hit、加载停顿与 batch 失衡；集成进 SGLang 并在多家公司生产部署（p.1–2）。
- 开源/集成情况：论文自述 "Implemented as part of SGLang and deployed in production"、"has been deployed in production environments at several leading AI companies"（p.1–2）。集成形态：Cache Controller（数据面）+ Scheduler（控制面）作为 SGLang 组件，HiRadixTree 是对 SGLang RadixTree 的扩展（p.5）；调度器保留 SGLang 的 Python 实现骨架、用 C++ 重写策略热路径以降开销（p.6）。论文未给出独立 Strata 仓库或 artifact 地址（全文无指向 Strata 本体的代码链接）→ 公开形态即 SGLang 主干。精读员注：本仓 `09_inference_system/sglang/hicache_deep_dive.md` 所述 SGLang HiCache 的 HiRadixTree、page-first、prefetch/backup 构件与本文设计同名同构，可作工程实况对照；但「生产部署涉及哪些公司、何种配置」论文未披露：未核实。

## 二、问题与动机

论文自述的总判断：naïve 分层缓存设计会变成 I/O-bound。"naïve designs often become I/O-bound: fragmented KV layouts lead to small transfers that underutilize bandwidth, cache loading stalls prefill, and schedulers that ignore cache-loading latency and delay hits ... suffer severe throughput degradation"（p.1, Abstract）。展开为四条：

1. **fragmented KV layouts → 小传输打不满带宽**。PagedAttention [26] 以细粒度分页省显存，代价是 KV 散落多个非连续页，"This yields many small data transfers, sometimes only a few kilobytes, that fail to saturate PCIe bandwidth"（p.2）。量化测量：8192 token 的 KV 以 page size 32 从 CPU 传到 GPU，只达到 PCIe 5.0 理论带宽的约 22%；在 GH200（NVLink 替代 PCIe，峰值带宽 6×）上低至 5%（p.3–4, Fig.3）。根因用 Little's Law 表述：可持续吞吐 X = C·S/L，实用上最大杠杆是提高单次传输尺寸 S——而吃满 PCIe 5.0 的 75–80% 需要 1–2MB 级传输（p.3）。
2. **大页的代价**。"Using larger pages could mitigate fragmentation and improve I/O efficiency, but it worsens cache granularity and can reduce cache hit rate"（p.2）。实测：页大小 1→512（Mistral-24B、ShareGPT、H200），KV 命中率显著下滑，平均 TTFT 与 P90 最高涨 2× / 2.9×（p.3, Fig.2）。
3. **cache loading stalls prefill**。SGLang 把 KV offload 到 CPU 内存、按常规配置：74% 的 prefill 时间阻塞在 KV 传输（Fig.1 红曲线），吞吐最高降 4×（p.1）。即使换上本文的优化 I/O，"up to 24% of prefill execution time remains stalled on cache loading"（p.2）。
4. **scheduler 对 I/O 与 delay hit 视而不见**。"current schedulers are blind to cache-loading costs and delay hits"（p.2）。既有系统的隐含假设——层 N 的计算足够盖住层 N+1 的 KV 加载——在「长缓存上下文 + 少量新 token」时失效；重算则随上下文变贵（p.4）。delay hit 借自网络社区 [SIGCOMM '20, ref 8]：同一数据对象的多个请求在首次 miss 尚未解决时到达，"subsequent requests are effectively treated as cache misses"（p.2 与 p.4 两处出现）。量化：Mooncake agent tool trace 中 38% 的请求与至少另一个请求在 1 秒窗口内共享 ≥6k token 前缀（不含 system prompt）（p.4）；异步调度器在批执行完成前就备下一批，把 miss 解决窗口拉长到整批执行时间，放大 delay hit（p.4）。论文结论句："Schedulers must explicitly treat CPU-GPU bandwidth as a first-class resource, balancing computation and data transfer when batching requests"（p.4）。

## 三、核心机制

### 3.1 GPU-assisted I/O（GPU 与 host 布局解耦、大块传输）（§4.2、§4.2.1）

- **做什么**：用 GPU 侧的 I/O kernel 承担 CPU↔GPU 的 KV 页搬运，替代对小传输低效的 cudaMemcpyAsync 重复调用（p.5）。
- **怎么做**：launch 一个 CUDA kernel、起数千线程；每个线程把一小块数据从源（GPU global memory 或注册的 CPU pinned memory）读进寄存器，再流式写到目的端（p.5）。三项收益恰好对应 Little's Law 三个变量：并发 C（数千并发 I/O vs CPU 侧典型数十）、小传输 S 友好（高效粒度只需 128 bytes [39]，单页 KV 是 KB 级，无需为效率放大页）、布局灵活（I/O kernel 里的轻计算近乎免费，可顺带做布局转换）（p.5）。
- **关键设计（干扰控制）**：GPU-assisted I/O 的已知问题是与计算 kernel 抢 SM/寄存器/污染 cache，硬件调度器管不好这种争抢（p.5）。Strata 的做法：launch 少量大 CUDA block，诱使硬件调度器把 I/O kernel 压进极少数 SM（"as few as 1"），配合绕过 cache 的低层指令（p.5）。微基准（Fig.5，H200，与 prefill 批 2 请求×4k、decode 批 16 请求×4k 并发）：2 个 1024 线程 block 达成 48 GB/s，prefill 性能损失 <5%、decode 10%（p.5）。默认配额：CPU→GPU 加载（关键路径）2 block；GPU→CPU 备份（非关键路径）1 block（p.5）。
- **布局解耦**（§4.2.1, Fig.6）：GPU 侧保留 layer-first（一层所有 token 连续，attention kernel 顺序读 K/V 最优）；host 与磁盘侧用 page-first（一页跨所有层连续）。layer-first 对传输是 pessimal 的——一个逻辑页被散成 L 个 KB 级碎片；page-first 则逼 attention kernel 多一层地址间接（p.6）。关键设计：GPU-assisted I/O 的线程在计算目的地址时多做一次算术即可完成布局转换，"This operation has negligible overhead"（p.6）。磁盘端复用同一机制（§5.3.5）。
- **兼容性声明**：ROCm 后端下同一类 kernel 实现亦兼容 AMD GPU（p.5）——无实验数据：未核实。

### 3.2 Cache-aware Scheduler（delay hit 规避、均衡 batch、bubble filling）（§4.3）

目标：最大化缓存收益——避免 delay hit 与加载停顿（p.6）。三阶段：

1. **Deferral on delay hit（§4.3.1）**：HiRadixTree 中引入 transient node，用 token ID 做遍历键，但不指向内存索引、只带两种标记：in-queue（有请求引用了该新上下文）与 in-flight（对应缓存正在计算）。迭代请求队列时按需插入 in-queue 节点；匹配到 transient node 的请求推迟到下一轮调度，但插到等待队首，以吃「即将变热」的缓存、把 TTFT 影响压到最小；执行时节点转 in-flight，完成后转标准节点（指向已就绪的缓存索引）（p.6–7）。防误伤阈值：匹配 token 数超过阈值才推迟，默认 100——既放过短前缀的偶然撞车，又抓住长上下文 delay hit（p.7）。
2. **Balanced batch formation（§4.3.2, Algorithm 1）**：默认 FIFO 批形成只看 token 预算/显存，不看不平衡的加载需求；Strata 逐请求检查「加入后聚合 load/compute 比是否越限」（`loading_bound`，line 10）：越限则移入 deprioritized 列表，否则入批、并优先补入与其 **bundle hit** 的请求——"bundle hit, the counterpart of a delay hit"（同上下文请求进同一批，既平衡加载又省显存与 on-device 带宽）；批未满时用 deprioritized 请求回填。反饥饿：deprioritized 保持原序、每轮批形成都从队首请求起算（p.7）。阈值硬件/模型相关，可单独 profile；默认比值 100，对应 Fig.1 停顿开始出现的位置（p.7）。
3. **Bubble filling（§4.3.3）**：均衡后仍 loading-bound 的批，推迟原 prefill、先发一个 decoding batch 与加载并发——decode 主要吃 HBM 带宽、加载吃 PCIe，两者资源面不撞；P-D 分离系统里也可插 prefill batch 填气泡（p.7）。此策略补足 SGLang 默认的 prefill-first 策略（p.7）。

实现注记：SGLang 调度核心是 Python，这些策略用 C++ 重写以压低调度开销（p.6）。

### 3.3 分级缓存组织结构（HiRadixTree、Cache Controller、写策略与存储预取）（§4.1、§4.2.1、§4.4）

- **两组件架构**（Fig.4）：请求入队 → Scheduler 在执行当前批时持续估算资源与排队请求需求、选出下一批 → 交 GPU executor 并触发 KV 加载；prefill 执行中，GPU executor 用 CUDA events 与 Cache Controller 同步，确保某层 KV 在用到前就绪；prefill 完成后经 continuous batching 并入 decode 批；备份/驱逐由 Cache Controller 异步做（p.5）。
- **P-D co-location**：prefill 与 decode 批在同一 GPU 上时间复用，沿用 SGLang 的 prefill 优先实践（降 TTFT、凑大 decode 批拉吞吐）（p.5）。
- **HiRadixTree**：SGLang RadixTree 的扩展，兼任 page table 与逐页 metadata（Fig.4 给出 [Token_ids]: {GPU_indices, CPU_indices, hit_count, …}）（p.5）。
- **写策略三档**（§4.4）：单一 write-once 不适用——不同负载在写带宽/容量/持久性/复用间取舍不同（p.8）。**write-back**（濒临驱逐才备份，省带宽容量，资源受限场景）、**write-through**（每次生成即备份，会话全量持久化场景）、**selective-write-through**（默认：节点计数器超阈值才备份；阈值 1 等价 write-through；带宽充裕时默认阈值 2）（p.8）。所有层默认 **LRU** 驱逐（p.8）。
- **存储层数据管理**（§4.2.1）：磁盘层命中即触发 host memory 预取，与请求排队时延重叠；一旦被调度选中便终止未完成预取、用现有缓存（**best-effort** 默认，原因：存储层延迟比 host-GPU 传输高一个量级且不可预测）；可选 **wait-complete** / **timeout** 策略（p.6）。

## 四、关键数字

| 指标 | 数值 | 对照基线/条件 | 出处 |
| ---- | ---- | ------------- | ---- |
| 端到端吞吐（vs vLLM-LMCache） | 最高 5× | Llama-3.1-70B（4 GPU TP）· LooGLE · 同 TTFT 比较 · H200 | p.9 §5.2.1（摘要同口径 p.1） |
| 端到端吞吐（vs TRT-LLM-HiCache） | 最高 3.75× | 同上 | p.9 §5.2.1（摘要同口径 p.1） |
| LooGLE 吞吐（Llama-8B） | 最高 3.2× / 2.6× / 1.9× | 对照 SGLang-HiCache / vLLM-LMCache / TRT-LLM-HiCache；同 TTFT · H200 | p.9 §5.2.1 |
| LooGLE 吞吐（Qwen2.5-14B） | 最高 3.9× / 2.1× / 1.9× | 同上三基线 | p.9 §5.2.1 |
| ReviewMT 吞吐（Llama-8B） | 1.7× / 2.3× / 2.3× | 多 Agent 对话数据集（输出较长，prefill 占比下降） | p.9 §5.2.1 |
| 暖缓存 NarrativeQA（vs vLLM-LMCache） | 2.3×（8B）/ 2.6×（14B）/ 2.5×（70B） | 预热 CPU 缓存全部上下文后重启负载；TRT-LLM 不支持预热 | p.9 §5.2.2 |
| 消融（SGLang-HiCache 之上） | 调度最高 1.8×、I/O 最高 2.3×（峰值吞吐） | LooGLE · Qwen-14B · H200；另比 Strata-IO / Strata-IO-LPM | p.10 §5.3.1 |
| 页大小不敏感性 | SGLang-HiCache 最优设置（页 512）仅达 Strata-IO 的 93%，因其命中率低 2.4% | LooGLE · Qwen-14B · H200 | p.11 §5.3.2（Fig.10） |
| cache distance 分解（峰值吞吐增量） | delay hit 规避 +42%（最小距离场景）；I/O +76% / +95%（shuffle / 最大距离）；balance batch +11% / +12%；stall hiding +8% / +3% | LooGLE 派生三负载（shuffle/min/max cache distance） | p.11 §5.3.3（Fig.11） |
| delay hit 放大条件 | 1k token 输入的 cache resolve time 约数百 ms–数秒；resolve time 与吞吐越高，delay hit 造成的 miss 越多 | **模拟执行** Mooncake Tool-Agent trace，无限缓存排除驱逐 | p.11–12 §5.3.4（Fig.12，模拟） |
| 磁盘布局（DeepSeek-V3 · 8×H20 · page 32 · 12 req/s） | 平均 TTFT 5.03→2.42 s（2.1×）；吞吐 27.43→36.41 token/s（1.3×） | page-first 布局 vs 原布局（后者页本身已够大） | p.12 §5.3.5（Fig.13） |
| 持续传输带宽 | Strata-IO：40.30→150.50 GB/s（PCIe→GH200）；对照 SGLang-HiCache：10.80（PCIe）/ 19.43（GH） | Llama-3.1-8B · LooGLE；同日比 Strata-Oracle（模拟无限带宽，Strata-GH 近 Oracle） | p.12 §5.4（Fig.15） |
| 带宽利用率（背景测量） | 8192 token KV @ page 32 ≈ 22% PCIe 5.0 理论带宽；GH200 低至 5% | Llama-3.1-8B · CPU→GPU | p.3–4（Fig.3） |
| I/O 停顿（背景测量） | naïve：74% prefill 时间阻塞于 KV 传输、吞吐最高降 4×；优化 I/O 后仍至多 24% 停顿 | SGLang + CPU offload · LooGLE · Qwen2.5-14B | p.1–2（Fig.1） |
| 大页代价 | 页 1→512：平均 TTFT 最高 2×、P90 最高 2.9× | Mistral-24B · ShareGPT · H200 | p.3（Fig.2） |
| I/O kernel 微基准 | 2×1024 线程 block：48 GB/s；prefill 干扰 <5%、decode 10% | H200 · Llama-3.1-8B（prefill 2×4k、decode 16×4k 并发） | p.5（Fig.5） |
| 分层缓存命中率 | 各分层系统约 95%（CPU 内存扩展后） | LooGLE 三模型 | p.9 §5.2.1 |
| CUDA 12.8 cudaMemcpyBatchAsync 对照 | 38 GB/s vs 本文 kernel 48 GB/s（同一微基准） | 论文建议：加载走 GPU kernel、备份走 batch-copy API | p.13 §6 |
| 容量换算 | 40 GB HBM ≈ 0.3M token（Llama-8B） | 展示 KV 足迹量级 | p.1 |

实测 vs 推算：#10 为模拟执行（论文原文 "a simulated execution of the Mooncake trace"）；#12 中 Strata-Oracle 为模拟无限带宽；#19 为换算；其余为论文实测。「最高/至多」均按原文 "up to" 转写。生产部署规模无数字（定性，p.2）。

## 五、局限与边界

论文自认（§6 Discussion，p.13）：

- **I/O kernel 干扰是固有代价**：GPU-assisted I/O 以少量 SM 争抢换小块传输效率。CUDA 12.8 的 cudaMemcpyBatchAsync 不占 SM 但吞吐更低（38 vs 48 GB/s，同一微基准）；论文建议配置成「CPU→GPU 加载用 GPU kernel、GPU→CPU 备份用 batch-copy API」；长期方向是更通用的片上 I/O 加速器（p.13）。
- **公平性缺口**：调度以聚合 I/O/计算效率为先，可能区别对待请求、触发个别 SLO 违规；SLO/公平感知的 batch formation 留作未来工作（p.13）。
- **只覆盖 host DRAM→HBM 的隐藏**：从更慢存储层预取当前只与排队时延重叠，为其调度留 headroom 是公开方向（p.13）。
- **模型覆盖**：只针对标准注意力的 dense KV；sparse/linear 等 hybrid 模型的缓存管理与调度是另一组问题（p.13；§7 把 Marconi 列为互补方向）。
- **作用域**：单计算实例内的内存管理与调度，不依赖专用网络硬件；与 Mooncake/MemServe 类分布式池互补（p.13）。

前提条件：评测依赖大容量 host pinned memory（1 TB；GH200 上 400 GB）（p.9）；Strata/SGLang 用 page size 1（SGLang 默认），基线 SGLang-HiCache 用 32（p.8）；写策略需按部署形态选档（§4.4）。基准负载为 Poisson 到达、在途请求上限 128、ShareGPT 限 GPU 内存约 500K token（p.8–9）。

精读员注（疑点）：

- 测试平台全部为 NVIDIA（H200 / H20-storage / GH200）；ROCm/AMD 兼容仅口头声明。
- 「最高」增益跨度大：vs vLLM-LMCache 在 LooGLE 上 2.1×–5×、vs TRT-LLM-HiCache 1.9×–3.75×，ReviewMT 只有 1.7×/2.3×——增益强依赖「长缓存前缀 + 少量新 token」的负载形态（p.9）。
- 主力基线 SGLang-HiCache 是作者自实现（据 CachedAttention/Pensieve/FlashGen 路线、cudaMemcpyAsync 逐层重叠，p.8），不是 SGLang 发行版 HiCache 的直测；引用 5×/3.75× 必须带硬件与负载条件（H200、70B、LooGLE、同 TTFT）。
- 短上下文只有一个维度：ShareGPT 的平均 TTFT，且论文自认 SGLang 引擎在 Llama-8B/70B 上有轻微 kernel 劣势（p.9–10）——「不伤短上下文」是可比而非占优。
- delay hit 的严重性研究为模拟（p.11），生产部署为定性叙述（p.2），两者均无实测数字。
- Fig.1 的 74%/4× 与 24% 均出自 Qwen2.5-14B + LooGLE + H200 单一配置（§5.3 给明分析默认设置），外推到其他负载需谨慎。

## 六、写作素材

- **系列定位（四种「内存账」之一）**：Strata = 分层调度。它把层级缓存的账拆成两本：传输账（GPU-assisted I/O——用并发 C、小粒度 S、布局解耦把 KB 级碎片搬出 MB 级带宽）与调度账（把 CPU-GPU 带宽当一等资源，delay hit / bundle hit / loading_bound 全部进批形成决策）。四篇里唯一以「调度器」为主角的一篇。
- **与另外三篇的对照点**：
  - **DirectKV（OSDI '26，同会）**：同治「搬运贵」，但 DirectKV 走零拷贝去 staging、warp 级流水，宿主是 NVLink-C2C 单机；Strata 承认 layer-wise DMA 在长上下文下必漏，把搬运器整个换成 GPU kernel。对照点：「同一段 8192 token 的 KV，两篇各砍掉了哪个环节」——DirectKV 省拷贝次数，Strata 提并发与传输粒度。
  - **NEO（MLSys 2025）**：计算下移（attention 连同 KV 放 CPU 算）；Strata 反向——数据拉回 GPU、让 GPU 顺便干搬运。对「PCIe/NVLink 带宽 vs CPU 算力谁是瓶颈」的判断相反，是系列内最标准的正反论证。
  - **eLLM（DAC 2026）**：虚拟化/弹性抽象（ballooning、Virtual Tensor）；Strata 是显式三级缓存 + 策略化写回 + LRU。对照「容量决策该交给 OS 式抽象，还是应用级策略」。
  - 统一分析框架候选：Little's Law X = C·S/L（Strata 用它论证抬 S；DirectKV 减拷贝次数；NEO 换 C 的来源，值得在总纲里统一成一张表）。
- **与本仓已有文章可勾连处（候选）**：
  - `09_inference_system/sglang/hicache_deep_dive.md`：HiRadixTree / page-first / 零拷贝 / prefetch-backup 与本文同线；Strata「已集成进 SGLang」的工程形态可用它作实况对照（注意区分论文的自实现基线 SGLang-HiCache 与发行版 HiCache，见 §五精读员注）。
  - `09_inference_system/kv_cache/02_systems/lmcache/`（lmcache_overview.md、local_cpu_backend.md、local_disk_backend.md、gds_backend.md 等）：论文头号基线 vLLM-LMCache 即 LMCache 社区版（v0.2.1，chunk 256，p.8）——「被超越方」的机制细节在本仓。
  - `09_inference_system/kv_cache/03_fast26/`：Bidaw（多轮对话复用）、CacheSlide（agent 复用）与 delay hit 问题同题不同解；solidattention.md 的「块粒度 vs 传输效率」两难是其端侧 batch=1 版本，可与本文「页大小两难」并读；fast26-kv-storage-five-scenarios.md 可作本系列总纲的写法范式。
  - `09_inference_system/kv_cache/01_concepts/offloading/`（01_kv_offloading.md、02_layerwise_pipeline.md、03_kv_cache_prefetching.md）：layer-wise overlap 正是 Strata 要超越的假设（「层 N+1 加载被层 N 计算盖住」），可作「旧方案」锚点。
  - `09_inference_system/kv_cache/02_systems/mooncake/mooncake_architecture.md`：delay hit 的 38% 数据来自 Mooncake agent trace；Mooncake 也是远端 KV 池代表。
  - `09_inference_system/kv_cache/README.md`：门户的「内存层级专题」入口收录本系列四篇（含本篇），可作对外入口的坐标。
- **反差/场景素材（文章开头候选）**：
  - 「40 GB HBM 只装得下约 0.3M token（Llama-8B）——几份文档、几百轮对话就见底」（p.1）。
  - 「8192 token 的 KV 从 CPU 搬回 GPU，只吃到 22% 的 PCIe 5.0 带宽；换 GH200（互联峰值 6×），比例反而掉到 5%」（p.3–4）——硬件越新，软件浪费越刺眼。
  - 「页大小从 1 调到 512，平均 TTFT 涨 2×、P90 涨 2.9×——为搬得快而放大页，命中率先崩」（p.3）。
  - 「74% 的 prefill 时间在等 KV 传输」（p.1）——长上下文 prefill 有一大半是 I/O 等待。
  - 「agent trace 里 38% 的请求与另一请求在 1 秒内撞同一 ≥6k 前缀」（p.4）——delay hit 从网络缓存文献走进 LLM 服务。
  - 「同一份 DeepSeek-V3，只改磁盘上的 KV 布局（layer-first→page-first），TTFT 从 5.03 s 到 2.42 s」（p.12）。
  - 「GH200 上新旧软件的差距：DMA 拷贝 19.43 GB/s vs GPU kernel 150.50 GB/s」（p.12, Fig.15）。
