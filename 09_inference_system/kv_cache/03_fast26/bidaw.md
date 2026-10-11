# 多轮对话的 KV 放上 SSD，延迟为什么反涨 3.8 倍

> **元信息**：论文 _Bidaw: Enhancing Key-Value Caching for Interactive LLM Serving via Bidirectional Computation–Storage Awareness_，清华大学（胡世鹏、张广艳等）+ 中国地质大学(北京) + 中国电信，FAST '26 正式论文集 pp. 101–116（16 页全本）。**勘误（2026-10-11）**：本篇曾因 PDF 解析器缺陷被误判为「6 页 short paper」，实际为全本论文，已按全文重写。精读方式：Agent 精读 + 主会话全文通读交叉核对（报告见 [references/bidaw-report.md](references/bidaw-report.md)，报告中「short paper 未覆盖」相关标注已失效）。阅读日期 2026-10-11。

多轮对话的推理服务有一个绕不开的矛盾：每轮回复都要复用历史轮次的 KV Cache，显存放不下，只能放进「主机内存 + SSD」的两级存储。直觉上这只是加一层盘的事，论文量化的现实是：相比「所有 KV 都驻留内存」的理想设置，响应延迟最高升 3.8 倍、吞吐最低降到一半（摘要、§2.2）。

慢了一倍多，问题出在哪？Bidaw（清华，与中国电信合作）的回答分两步：先用量化定位根因——计算引擎和两级存储互相不感知；再让两边互相看见。

## 一、量化根因：两个盲区

论文自建了交互式对话 workload（产业伙伴提供的百万级对话轮：平均 query 36 token、平均回复 45 token、平均 22 轮/用户），在 A800 + OPT-13B + 200 GB 主机内存 + 1.5 GB/s SSD 的配置上测量，两个盲区各有硬数字。

**计算引擎的盲区：调度不看加载时延。** 请求的 KV 大小差异极大（从不到 100 MB 到超过 800 MB，图 8），叠加两级存储的带宽差，KV 加载时间的变异系数超过 90%——即使同一 5 秒窗口内到达的请求（图 7）。而调度器对此无感：长 KV 请求占住 GPU，后续短 KV 请求排队空转，论文称之为 severe request blocking。

**存储层的盲区：逐出不看对话语义。** 逐出只看自身的排队历史，而多轮对话 KV 的访问时间局部性极差：加权重用距离（两次访问之间其他 KV 的总量）80% 超过 200 GB 性能层容量（图 6(a)），并发缓存量可达性能层的 3.91×（图 5）。结果性能层平均只装得下 40.1% 的 KV，命中率约 20%——与 FIFO 相当（图 6(b)）。该省的没省，该留的被挤了。

## 二、双向感知：各自补对方的盲

既然是互盲，解法就是双向看见。

**计算侧感知存储**：调度器掌握每个请求 KV 的位置与大小，加载快的（性能层命中）进 ready 队列直接送 GPU，加载慢的（在 SSD）进 preparing 队列先预取——GPU 不被慢 I/O 卡住。这里有个设计判断值得注意：为什么是预取而不是计算与 I/O 重叠？论文算了笔账，GPU 一轮 decode 只有首轮数十毫秒可以遮盖 I/O，而容量层的 I/O 要数百毫秒，重叠无效，只能提前搬。

**存储侧感知计算**：用 LLM 刚生成的回复长度预测该用户下次访问的重用距离——回复越长，用户读、听、理解再组织下一问的时间越长，重用距离越大。论文强调这是只有计算引擎才有的语义信息，I/O trace 里拿不到。

## 三、四个机制

双队列调度（§3.2）：ready 队列 FCFS 保公平；preparing 队列按 disk-HRRN 排序，响应比 = 1 + 等待时间 / KV 大小（式 1）——小 KV 先发读，等待时间累积又防止大 KV 饿死；从 preparing 提升到 ready 时按原始到达时间（而非提升时刻）插入，避免尾延迟。代价是每请求维护 KV 位置与大小的元数据。

回复预测重用（§3.1 流程 3）：eviction manager 后台捕获每用户的最新回复，以长度预测下次访问的加权重用距离。

预测式逐出（§1）：ghost cache 模拟「已知未来的最优逐出」并结合历史 I/O trace，估计各重用距离下的命中概率，逐出估计命中概率最低的 KV。代价是每用户维护回复记录与 ghost 元数据。

选择性缓存历史张量（§1、§3.1）：推理产生的中间张量大小差异大且可互相转换，Bidaw 度量「每单位存储换回多少计算」，只缓存最划算的历史张量。论文称对 MHA 模型通用，并提到「GQA 模型缓存 MHA 形式更合适」——细节论文未覆盖。

逐出策略的细节（§3.3）比直觉更讲究。存储侧先在计算引擎里维护「上一轮回复长度」的在线统计——因为它与下一次访问的加权重用距离下界正相关（论文在 12 组不同时段、不同到达率的 trace 上验证，Spearman 秩相关系数 0.94–0.98）。逐出决策分两步：先用 ghost cache 模拟「已知未来的 Belady 最优逐出」得到各重用距离区间的命中率上界，再把重用距离分成小桶（小于性能层容量，命中率 1.0）、若干 promise 桶与 extreme 桶（超出后最优策略命中率也为零），每个桶携带命中率概率；最后按式 2 计算每个用户下次访问的总命中潜力 `Overall_potential = prob_small×1.0 + Σ prob_promising(i)×hit_promising(i) + prob_extreme×0.0`，逐出潜力最低者。这套设计的聪明处在于把「不确定的未来」转化为「带概率的区间估计」，并用回复长度做下界约束剪掉不可行的桶。

完整的请求工作流见论文图 9：到达 → 按 KV 位置与大小分队列 → 计算 → 回复交 eviction manager → 性能层低于阈值时按预测逐出；采用包容式缓存（容量层留副本）。

无损性是 Bidaw 的一个安静卖点：它只做 I/O 感知调度与逐出，不做任何 KV 重算或压缩，「computing user 1's request or user 2's request first does not change their LLM response」（§5.1）——精度与直接跑推理完全一致，这与量化类方案（掉精度换显存）是本质不同的取舍。

## 四、评估：五个模型、公开负载、逐项消融

完整评估（§5）比摘要口径丰富得多。

**总体性能**（Figure 15）：五模型扫描（OPT-6.7B/Qwen-7B/OPT-13B/Qwen-14B/OPT-30B，A800 + 200 GB 内存 + RAID-5 SSD 共 1.5 GB/s）。对比基线 vLLM、CachedAttention（ATC'24）、FlashGen（ASPLOS'25）与「全内存最优」模拟上界。OPT-13B 上 Bidaw 延迟最高降 3.58×；总口径延迟较 vLLM 降 83.9%。有意思的细节：FlashGen 在 OPT-30B 上延迟高于 CachedAttention——重计算成本随模型增大，这种交叉正是「选型要看模型」的实例。

**吞吐与到达率**：总口径较基线提升 1.43×–1.83×（§5.1）。SSD 带宽敏感性（§5.1 末）：模拟 5 GB/s SSD，FlashGen 从 15.18 提到 20.23 users/min，仍远低于 Bidaw 的 27.81——加大 SSD 带宽不消除瓶颈，因为瓶颈在 KV 加载效率本身。

**内存敏感性**（Figure 16）：host 内存从 2.5× 扫到 1.5× GPU 内存（200→120 GB），内存缩减时各方案延迟都上升（更多 KV 需从容量层加载），Bidaw 保持显著优势，支撑 1.75×–2.19× 的更高到达率。

**公开负载**（§5.3，ShareGPT + OPT-13B，Figure 17）：Bidaw 吞吐较 CachedAttention 高 1.40×，与 FlashGen 相当（平均延迟 8.70s vs 9.09s）；但对 vLLM 延迟降最高 69.8%/65.8%/56.9%。论文诚实承认：ShareGPT trace 不含时间戳（用 Poisson 模拟），previous-answer 逐出在此负载上不再降低 miss rate——存储侧机制确实依赖「按用户归因 + 真实时间戳」的部署前提。

**性能层 miss rate**（§5.4，Figure 18）：对 queue-enhanced 逐出降 57.6%、对通用逐出（LFU/LRU/FIFO）降 69.9%（OPT-13B，25 users/min），且论文指出 queue-enhanced 需要排队中请求少于 50 个才可在线决策，不可用于在线服务。

**排队与开销**（§5.5–5.6，Figure 19–20）：I/O-aware 调度器把平均排队时间从 5.76s 降到 2.45s（−57%）；调度开销每次 0.62ms，逐出操作 0.35ms（后台，性能层低于 5% 时触发），storage-efficient tensor 到 KV 的变换为数十 ms——对推理延迟影响可忽略。

**逐项消融**（§5.7，Figure 21，OPT-30B）：+I/O-aware 调度延迟降 1.58×；+previous-answer 逐出吞吐再升 1.25×；+storage-efficient tensor 缓存再升 1.10×——三个机制各自独立贡献，不是一招通吃。

**尾延迟**（§5.8，Figure 22，OPT-30B）：P90 较 CachedAttention 降 52.96%、较 FlashGen 降 66.63%；P95 降 49.30%/62.64%；P99 降 47.03%/56.81%——对 SLA 敏感的服务这组数字比均值更有意义。

## 五、边界

四个需要留意的边界。其一，「回复越长 → 思考越久」是启发式，边读边打字、秒回追问、多会话并行、粘贴长文本的用户都会打破它——论文用 ShareGPT 场景如实承认了这一点（逐出机制在无时间戳负载上失效）；非交互/批处理场景该信号消失。其二，逐出预测靠历史 trace 拟合，workload 热点切换会失准（论文的 ghost cache 分桶是缓解，不是消除）。其三，主评估模型是 OPT 系列（MHA）加 Qwen-7B/14B（GQA），覆盖面比想象的好；但「GQA 模型缓存 MHA 形式更合适」的机制细节论文确实未展开（§4）。其四，机制强依赖「按用户归因请求 + 真实时间戳」的部署（论文以 ShareGPT/Mooncake workload 缺信息为反例）。

## 六、生态关联与同会分工

Bidaw 的直接基线是 CachedAttention 与 FlashGen——论文的根因分析正是指出二者「计算-存储互盲」，并沿用了 FlashGen 的包容式缓存与跳过式准入。与 Mooncake 的关系是定位互补：Mooncake 式 KV 池化分离架构面向 RDMA 集群，Bidaw 明确自己的场景是无 RDMA 网卡、低成本本地部署（架构对比为精读员注，正文未展开）。

放进 FAST'26 的「KV Cache × 存储」全景里，三篇 KV 论文各占一个象限、机制互不重叠：CacheSlide 消除复用的计算（位置漂移），SolidAttention 压缩单请求的 I/O（端侧稀疏化），Bidaw 优化多用户服务端的分层调度与逐出——场景与机制均不同，见[系列总纲](fast26-kv-storage-five-scenarios.md)。

## 论文结构索引

| 论文位置        | 内容                                                                                     |
| --------------- | ---------------------------------------------------------------------------------------- |
| §1–2 / 图 1–8   | 两级存储的量化根因：延迟 3.8×、变异系数 90%、命中率 20%、重用距离 80% 超 200 GB          |
| §3.1 / 图 9     | 双向感知设计：双队列、回复预测重用、ghost cache 逐出、选择性缓存、工作流                 |
| §3.2 / 式 1     | disk-HRRN 调度（响应比 = 1 + 等待/KV 大小）与提升插入规则                                |
| §3.3 / 图 12–13 | 回复长度-重用距离下界（Spearman 0.94–0.98）、Belady 上界、三桶概率模型                   |
| §4 / 图 14      | 实现：混粒度 GPU 分配、storage-efficient tensor（tensor 6 成本效率 51.0）、变换开销      |
| §5 / Fig.15–22  | 完整评估：五模型总体、内存敏感性、ShareGPT 公开负载、miss rate、排队、开销、消融、尾延迟 |
| §6–7            | 相关工作与结论（与 Mooncake 定位互补）                                                   |

## 参考资料

- Shipeng Hu et al., _Bidaw: Enhancing Key-Value Caching for Interactive LLM Serving via Bidirectional Computation–Storage Awareness_, FAST '26, pp. 101–116. [USENIX 页面](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)
- 精读报告（Agent 交叉核对）：[references/bidaw-report.md](references/bidaw-report.md)
- 系列关联：[五场景总纲](fast26-kv-storage-five-scenarios.md) · [SolidAttention 单篇](solidattention.md) · [AITURBO 单篇](aiturbo.md)
