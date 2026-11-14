# 01 · Janus：生产规模的多 LLM 服务

> SOSP 2026 论文精读笔记。来源：`../papers/01_janus-multi-llm-serving.pdf`（作者版 PDF，2026-11-14 下载）。文中数字均为论文口径（作者测得或自述），引用时注意区分。

## 一句话

双时间尺度调度数百个异构 LLM：慢路 30 秒 bin-packing 共置互补模型，快路 0.5 秒弹性扩缩，D2D fork 秒级激活副本吃突发。

## 问题与背景

京东 MaaS 集群同一张 trace 里有 62 个应用、9 个模型（Qwen3-0.6B 到 DeepSeek-V3，规模跨三个数量级），日请求 45.6 M（§1，p.1）。作者从 2026-03-10 的生产数据提炼出三个叠加矛盾（§2.2，p.3）：

| 观察         | 论文证据                                                                                                                                     |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 突发不可预测 | 少数应用请求率数秒内涨 5–10×；传统 autoscaling 靠容器启动与 collective group 建立，时延十秒到分钟级，仅 H2D 权重加载就 0.4–2.7 s（measured） |
| 流量幂律     | top 12% 应用占 80% 请求、top 30% 占 80% token；尾部应用请求少但模型多样、必须保留                                                            |
| 资源互补     | 应用在 HBM 容量（HBM-C）、SM 算力、HBM 带宽（HBM-BW）三维画像互补，memory-heavy 与 compute-heavy 可低干扰共存                                |

现有系统各管一段（§1，p.1，作者自述）：MusServe 有多维感知但资源静态分配；ServerlessLLM 只加速冷启动、无集群级弹性；BlitzScale 每应用独立扩缩、缺跨应用协调；Prism 用页重映射做多模型共置，但周期性全局重优化会暂停请求，运行期也不能重分配设备。主流引擎 one-instance-one-model，互补性无从利用（§2.2，p.4）。问题因此从单模型吞吐变成数百模型在共享设备上放置、扩缩、路由的联合决策。

## 核心设计

Janus 分集中式 Service 层与分布式 Engine 层，两条时间尺度：慢路 30 秒、快路 0.5 秒（§3，p.4）。Engine 把设备分成两个池：Steady Pool 单实例共置多模型，承接慢路 packing；Elastic Pool 单实例单模型，承接快路扩缩与突发（§3，p.4）。

**Performance Oracle（§4.1，p.4–5）**：两个 Gaussian process。Capacity GP 把负载特征映射为三维稳态需求向量 (HBM-C, SM, HBM-BW) ∈ [0,1]³，供慢路 packing；Elasticity GP 把负载特征加实例数映射为 SLO 达标率，供快路二分搜索最小实例数。选 GP 是因为观测稀疏：Sherman-Morrison 增量更新后单次预测 O(n)，自带校准不确定度，冷启动（8–50 个训练样本）比 kNN 准约 2×，不确定度高时倾向超额供给。输入特征刻意低维：token 率、请求率、输入输出长度的均值方差，NMJ 分析支持这组统计量足以刻画资源消耗。

**Model Scheduler（§4.2，p.5–6）**：慢路做在线三维向量 bin-packing，bin 容量 (1,1,1)，新模型到达即贪心配对互补者，bin 失衡超阈值才 repacking 而非每次模型离场都重算，后台每 30 秒做一次 first-fit decreasing 整理；插入与 repacking 均摊 O(1)。快路每 0.5 秒按 Elasticity GP 二分找每模型最小实例数；Elastic Pool 预算是 Steady Pool 占用后的剩余设备，配额按比例分、Hamilton 最大余数法取整。防抖窗口存最近两个方案，新方案须与两个前案都不同才生效，避免需求贴阈值时来回抖动。缩模型让出的设备立即给扩模型复用，重叠执行把切换时延减半（p.6）。

**Request Scheduler：LST-IMH（§4.3，p.6）**：按模型维护集中队列，跟踪 Estimated Prefill Done Time，任一实例 EPDT 超 20 ms 就对全 backlog 跑一次 LST-IMH。算法把单机 Moore-Hodgson 推广到并行机 deadline 调度：实例按 EPDT 降序，逐台机器解有效截止期 d_i − T_i 的移位单机问题，被拒作业退回 backlog。有 2-近似保证（证明在补充材料），O(mn log n)、实测亚毫秒。被拒请求可放宽截止期以 reentry 模式重新入队，用略低达标率换更低失败率。

**xTensor 虚拟化 HBM（§5.1，p.6–7）**：动机是一个实测数字：驱动级 map/unmap 每 2 MB 页要 37/235 μs，逐页映射大模型权重需数秒。xTensor 插在框架 allocator 之下，把 HBM 组织成统一物理页池，上层张量只持虚拟地址、绑定延迟到首次使用。权重与 KV 分治：权重是大块连续段，实例启动时在预注册区域 best-fit 切段，分配释放退化为段级预留，不再触驱动；KV 细粒度且短命，驱动映射挪出请求热路径，已映射页缓存把分配压到 50 ns（裸路径 33.7 μs，675×，measured）。权重从页池高端取页、KV 从低端取，物理隔离长短命对象。算力维度用多 stream 复用，不依赖 MPS/MIG 这类厂商特性。

**三态生命周期与双激活路径（§5.2，p.7–8）**：模型生命周期与实例解耦。active（权重在 HBM）/warm（换出到 host pinned 内存，collective group 与框架上下文保留）/cold（只留最小框架状态）。激活分两路：H2D wakeup 恢复 Steady Pool 的 warm 模型，传输与执行重叠；D2D fork 服务 Elastic Pool 突发，目标实例按 global xTensor 登记的偏移直接从源设备拉权重、不经 host 中转。物理基础是实测 D2D 163.6 GB/s 对 H2D 59.0 GB/s，2.8×（§7.1，p.8）。只有 P/D 分离需要改框架（KV 地址注册），其余 per-model 优化原样保留（§5.3，p.8）。实现：Service 层 ~8 KLOC、Engine 层 ~21 KLOC C++，D2D 传输复用 Mooncake（§6，p.8）。

机制衔接：Oracle 从 Engine telemetry 持续回流更新，慢路定共置格局、快路在格局内扩缩、请求调度器把流量落到已供应的实例上；两个池子的划分正是两条时间尺度在物理层的对应物。

## 效果与代价

环境（§7.1，p.8）：CloudMatrix 384 supernode，48 台服务器 × 8 个 Ascend 910C 共 768 设备，每设备 64 GB HBM、376 TFLOPS，UnifiedBus 1.0 互联。主实验 W0 为一天生产流量的 603 K 请求回放，24 小时压成 30 分钟、约 64× 平均到达率，open-loop 回放（p.9）。

端到端（§7.2，p.9–10，measured）：Janus 全天 SLO 达标 0.97–1.0（日峰 ≥0.98）；BlitzScale 夜间约 96%、白天跌到 0.60–0.65；Prism 白天 0.80–0.92、低点 0.51；ServerlessLLM 白天 0.22–0.30，近四分之三请求违反 SLO。白天 goodput：Janus 400–500 req/s（峰值约 620），BlitzScale 约 150，ServerlessLLM 50–150。设备开销：Janus 13,440 device-hours/day，比静态占满 768 设备的 ServerlessLLM（18,432）省 27%。

消融（measured）：去掉跨应用协调，均值 SLO 99.4%→96.9%、最差应用 90.0%→82.0%，说明 fork 与 packing 能兜底、但头部竞争仍需全局分配（§7.2，p.10）；单 Steady Pool 只有 37.1%、单 Elastic Pool 59.0%，双池 97.8%（§7.4，p.11）；去掉 xTensor，SLO 跌到 81%、P99 TTFT 到秒级（p.12）。速率扫到 6.19× 时，Janus 在 ≥90% 达标下承载 BlitzScale 1.3–1.5×、Prism 与 ServerlessLLM 3× 以上的请求率（§7.3，p.10）。

组件数字（p.11–12，measured）：671B DeepSeek-V3 突发下 D2D fork 0.74 s 激活副本、SLO 保持约 98%，H2D 要 2.71 s、跌到约 90%；fork 对源实例有扰动，671B 源 P99 TPOT 最多涨 5.7%（42.6 ms 基线）。生命周期时延：deactivation 307–1278 ms，D2D fork 388–783 ms（Qwen3 系列）、742 ms（671B）。调度消融：2× 负载下 LST-IMH TTFT 达标 91.2%（reentry）对 Round-Robin 64.8%，3× 时 47.0% 对 9.2–13.4%。bin-packing 与 OR-Tools 精确解比，99.99% 的时间 bin 数相同、从不差超过 1 个 bin，决策快 4.3×（0.30 对 1.28 ms）。绝对时延：全 trace P50/P90/P99 TTFT 57.5/161/395 ms，TPOT 7.2/12.3/18.5 ms；5× 负载下平均时延涨到 1928 ms，排队占比从 3.4% 升到 43.3%（835 ms），瓶颈转到调度管线。

局限散落各节，拼成清单：Poisson/均匀到达的 trace（W6/W7）上优势收窄，且该实验设备池更小、请求率更低（§7.3，p.10）；动态并行度重配置不做、创建时定死（§5.2，p.7）；P/D 分离要改框架；fork 扰动源实例；全文无独立 limitations 节。

## 总评

论文回答的问题很具体：幂律流量下，共置省设备与弹性保 SLO 能否兼得。答案是要，且消融把贡献拆得干净：去掉全局协调最差应用掉 8 个百分点，单池只剩 37–59%。给调度算法配 2-近似证明、给预测器带校准不确定度，在 serving 论文里少见，快慢两路的阈值设计因此有依据而非纯调参。整套系统跑在昇腾 CloudMatrix 上而非 NVIDIA，也是少见的生产级非 CUDA 栈完整数据点。

三处需要打折。其一，主实验把 24 小时压成 30 分钟、64× 平均到达率的 open-loop 回放，绝对时延与排队占比都是压缩口径下的产物，外推到真实节奏要小心。其二，基线配置偏弱：ServerlessLLM 被切成静态均分设备池，Prism 被限制不能运行期重分配设备，差距里机制与配置各占多少只做了部分剥离（Prism-with-global 一组）。其三，D2D fork 的 2.8× 带宽优势绑定 UnifiedBus 这类超节点互联，论文自述商用平台存在类似拓扑但没给数字（§2，p.3），换到 PCIe 或普通 NVLink 场景，快路时序要重算。

对做多模型平台的读者，可借鉴三个抽象：把资源画像压成 HBM-C/SM/HBM-BW 三维向量再做 packing；用 GP 加二分搜索替代启发式扩缩；权重段级预留与 KV 页级缓存分治，绕开驱动映射开销。要警惕集中式 Service 层的前提：全集群 telemetry、亚毫秒调度循环、30 秒级 repacking，规模再上一档或多集群时是否仍成立，论文没讨论。

## 疑点与待核

- 评估全在昇腾/CloudMatrix 栈（xLLM 引擎 + Mooncake），NVIDIA 栈上 xTensor 的页粒度与 map/unmap 开销不可直接迁移，开源仓库的硬件覆盖范围待查。
- Capacity GP 对 SM/BW 的 P90 误差有 24.18%/23.24%，而 bin-packing 以 (1,1,1) 硬容量判可行，预测偏乐观导致超卖时靠什么恢复，文中没展开。
- 共置干扰的刻画停留在多 stream 交错与 1.07× 碎片开销，高利用率下共置模型的尾时延干扰无单独实验。
- 603 K 是从 45.6 M 按比例抽样的子集，论文称采样保留 burst 形状，但无量化指标。
