# 10 · Borges：CXL 内存/SSD 混合上的低延迟分布式共享日志

> SOSP 2026 论文精读笔记。来源：`../papers/10_borges-cxl-shared-log.pdf`（作者版 PDF，UT Austin Witchel 主页，2026-11-14 下载）。文中数字均为论文口径。页码按论文集页码（211-228）。UT Austin + Anthropic + UIUC，其中 Zhipeng Jia（Boki 一作）工作完成于 Google。

## 一句话

把共享日志的元数据与排序状态放进 CXL 内存/SSD 混合设备：单写者元数据加双索引发布，追加 p99 降到 184μs。

## 问题与背景

共享日志统一复制、协调与恢复，是锁、元数据服务、状态机复制的共同底座（§1, p.211）。但现有设计围绕网络：支撑每秒百万级追加的系统 append 延迟仍在 1-2ms 量级（§1, p.211）。拆开看，Scalog 的服务器侧通信占端到端 append 延迟的 51.0%，Boki 占 40.1%，为摊薄通信引入的批处理又各占 21.7%/21.4%（§2.1, p.213, Figure 2）。回放还有一道两难：流绑死一个 shard 回放本地，但热流会压垮该 shard，vScalog-CXL 在高 skew 下吞吐掉近 30%；流跨 shard 分散了负载，回放却要远程取记录，Boki-CXL 单次远程 fetch p50 为 673μs、p99 超 2.5ms（§2.1, p.213）。

CXL 提供了新基材。Samsung CMM-H 是 CXL Type-3 DRAM/SSD 混合设备：48GB DRAM 缓存（8 路组相联、LRU）背靠 NAND，呈现为 CPU-less 的持久 NUMA 节点（§2.2, p.213）。实测 1GB 工作集访问延迟 583ns，32GB 内低于 590ns，64-512GB 也只有 621-638ns，远低于 RDMA 的微秒级；带宽 32 核下从 9.3GB/s（16GB 工作集）缓降到 4.9-5.4GB/s（64-256GB）。持久性靠平台 eADR 加两段刷写：掉电先刷 CPU cache 行进设备，再全局刷 DRAM 进 NAND。论文面向约 8-16 台后端的 CXL pod（无 CXL switch），协调服务核心本来就小：Chubby、ZooKeeper 通常 3-5 副本，etcd 建议 ≤7 个（§2.2, p.213）。

直接把网络日志移植上去不管用，论文归纳出两个挑战（§2.3, p.214）。挑战一是无跨主机缓存一致性：现有 CXL pod 原型（Octopus 与 Niagara 2.0 均未实现 CXL 3.2 规定的那部分），POSIX 锁与 futex 等隐含硬件一致性的原语失效；唯一的跨主机原子原语 mCAS 只有研究原型，1 写者 2.3μs、16 写者中位数 38.5μs（p99 241.9μs），比一致性缓存内的 CAS 贵一个数量级。挑战二是共享可见性带来发布问题：每个 shard 的存储对所有主机直接可读，复制与提交在时间上解耦，读者可能读到已落存储但未排序的数据；global cut 与索引必须成对一致，否则回放会漏记录、混入未提交记录或错序。Scalog、Boki 换 CXL 传输解决不了这两个问题；LazyLog、SpecLog 靠投机排序降「感知延迟」，不适合锁与 exactly-once 这类要等实际提交的负载（§2.4, p.214, Table 1）。

## 核心设计

三个想法支撑全部设计（§3 开头, p.214）：单写者元数据、双索引发布、字节偏移索引。

**架构**（§3.1, Figure 3, p.215）。沿用 replication-first 分解：shard server 在数据面，单 sequencer 在控制面；pod 内服务器只经复制 CXL 存储通信，客户端在 pod 外走网络。日志按流组织，每流归一个 shard 但可跨 shard：刻意不绑死单 shard，用放置灵活换 skew 友好，回放开销另由机制压掉。流是 2KB 固定段的链表；全局索引只记每流每 shard 一个字节偏移。

**元数据所有权**（§3.1, p.215）。CXL 驻留元数据分两区。local 区由一个 shard 单写、sequencer 读：local cut 上报复制进度，stream delta 上报每流新增字节数。global 区只有 sequencer 写、全体 shard 读：global cut 划出已提交前缀，global index 记录每流每 shard 的已提交字节边界。每个共享词单写者，数据与元数据按 cache line 对齐填充，互不共享，协调只靠普通 load/store，mCAS 离开快路径（Takeaway 1）。

**追加路径**（§3.2, Figure 4, p.215-216）分三段。复制：请求落到 shard server 后分配本地序号，两个复制线程并行写主、备设备，两份都越过该记录才算 durable。排序：indexer 把 (流 id, 新增字节数) 的紧凑 delta 写进与 sequencer 共享的 ring buffer；sequencer 收各 shard 的 local cut（64 位原子字打包），取最小值即完全复制的进度，之后的 delta 先忽略。delta 无 per-record 信息，与复制几乎重叠，排序层不拖慢追加。发布：front index 供读者读，back index 由 sequencer 私有更新。安全前缀确定后，sequencer 先把 delta 应用进 back index，再构造 global cut：头含发布轮次与重配置 epoch、指向该 back index，payload 记各 shard 全局进度；在每台设备上原子安装 checksum 作有效性标记，只复制 global cut（恢复的唯一权威）。先切读者可见指针，再推进 cut ring 的生产者位置。核心不变量由此而来：每个可见 global cut 都对应一个物化了完全相同已提交前缀的完整 index（Takeaway 2）。发布用 RCU 式 publish-and-wait（Table 2, p.217）：读者在非临时槽位上宣告活跃，sequencer 切换指针后等旧读者退场即可。

**线性一致点**（§3.2, p.217）。写只在 cut/index 对发布时可见；读只回放当前 front index 与 global cut 命名的前缀，重构出的必是线性一致的已提交前缀。

**回放路径**（§3.3, Figure 5, p.217）。回放只依赖两样东西：每 shard 的已提交字节边界与 global cut 序列；其余从日志数据重建。global cut 是 (shard id, local seqnum) 对，按字典序合并各 shard-local 流即得全局序（同 Scalog/Boki）。跨分片回放因此是共享内存直接读。字节偏移索引是中间态：per-record 索引要 sequencer 发布更多共享元数据，绑死单 shard 会重现热点，每流每 shard 一个字节边界两头兼顾。trim（§3.4, p.217）也走发布通道：零字节偏移的 delta 表示删整条流，惰性回收。

**其余机制**。单 sequencer 是刻意选择（§3.5, p.218）：每轮每 shard 只消费固定大小元数据摘要，突发请求不增加每轮 CPU 工作；更大部署按 pod 放多个 sequencer（Boki logBook 式），只线性化 pod 内记录，放弃部署级全序。重配置（§3.6, p.218）复用同一发布边界：配置记录带单调递增 epoch、双设备持久化，变更与 global cut 同批发布才生效。容错（§3.7, p.218-219）假设 fail-stop，靠 lease 检测：sequencer 无状态，从最后完全复制的 cut 重启即可，半成品 cut 被下一个覆盖；shard 替换从 cut 恢复本地序号、delta 游标与字节边界，未发布状态直接丢弃；CXL 设备 f+1 副本，备坏降级运行、主坏暂停提交，恢复时倒放跳过 torn tail、验 checksum 选最新完整 cut。exactly-once 工作流（§3.8, p.219）是直接产物：工作流状态全在日志里，操作 ID 跨重试稳定，写靠回放去重，读追加读记录后回放到该点。

## 效果与代价

评估环境（§4.1, p.219）：pod 用单台 Intel Xeon 6 服务器加一块物理 1TB Samsung CMM-H（PCIe 5.0 x8）模拟，8 个 VM（各 5 vCPU、10GB DRAM），X550 10GbE TCP/IP。按现有 CXL pod 原型的假设，共享数据用 clflushopt 加 sfence 访问；CMM-H 无跨主机原子，mCAS 以 flush 加 CPU CAS 建模，p50/p99 为 2.7/3.2μs，只比 Cxlalloc 的 FPGA 原型（2.3/2.8μs）高 0.4μs，4 写者时 7.58/34.25μs 对 6.80/32.34μs。物理设备只有一块，拆两个逻辑设备，双副本争同一带宽（保守）。基线是同框架重实现的 Scalog、Boki，各有 '-NET'（原传输）与 '-CXL'（CXL 消息队列换传输）两档；Borges-RDMA 用 ConnectX-6（25Gb/s、1.86μs）同机制对照。实验重复 5 次，标准差低于均值 5%。

主结果：追加（Table 3/4, §4.2-4.3, p.220-221）饱和吞吐 161.91K req/s，p50/p99 157.24/184.21μs，对 Scalog-CXL 吞吐 2.16×、p99 2.87×（528.57μs），对 Boki-CXL 吞吐 2.66×、p99 2.78×（511.39μs）；负载加到 40K/50K/60K ops/s，p99 优势扩大到 2.9×-4.0× 与 3.3×-4.3×。机制不动只换 RDMA，吞吐掉 4.22×，收益大头来自 CXL 本身；只换传输、不重设计协调则追不上：Scalog-CXL 仅 mCAS 锁获取就占 p99 的 24.1%，Borges 把复制、排序、提交延迟分别降 3.1×、8.9×、4.9×。客户端 TCP/IP I/O 占 175μs append 中的 105μs（60%），理想化零客户端 I/O 后服务器侧 70μs（§4.2, p.220）。锁（Table 5, §4.4, p.221）饱和吞吐 37.12K req/s，对 vScalog-CXL 8.50K（4.37×）、Boki-CXL 0.38K（97.68×）；跨分片取记录降到 p99 10.1μs；双索引发布开销 p50 3.4μs、p99 9.8μs（4 读者），12 读者时 8.9/22.6μs。skew（Table 6, §4.5, p.222）θ 到 0.99 时 Borges 吞吐只从 147.3 掉到 144.2 Kreq/s（2.1%），vScalog-CXL 掉到 69.5%，p50/p99 涨 48.6%/30.0%。KV 与工作流（§4.6-4.7, p.222-223）YCSB A-D 吞吐 156.4K-198.6K req/s（对 Scalog-CXL 1.6×-2.1×），Retwis p50 806.7μs 对 Scalog-CXL 4.6ms（5.7×）、Boki-CXL 18.1ms（22.5×），吞吐 27,722 对 1,426 req/s（19.4×）。配置敏感性（Table 7, §4.8, p.223-224）都比较温和：副本 1 到 3 只降吞吐 12.3%、p50/p99 涨 7.1%/7.5%。16 分片时单 sequencer 吞吐 612.1K req/s、五核只占 40% CPU（§4.9, p.223），但该实验用合成 shard 流喂 sequencer，端到端延迟由模拟 shard 延迟加客户端 I/O 合成。

局限（论文自认, §4.1, p.219）：单物理设备拆双副本；mCAS 是建模而非硬件原语；面向 pod 规模（约 8-16 台）；不依赖 NAND 层，无 NAND 的 CXL 共享内存同样适用。

## 总评

论文回答的问题很清楚：在无跨主机缓存一致性、原子原语贵一个数量级的共享持久内存上，共享日志能否同时低延迟追加、跨分片低延迟回放并保持线性一致。答案是能，路径三件套：单写者元数据把协调压回普通 load/store，双索引发布用成对不变量分离前缀构造与可见性切换，字节偏移索引让回放只依赖两种轻量元数据。对存储与分布式系统读者，最有复用价值的是 Table 2 的读者宣告协议加 cut/index 成对不变量：非一致共享内存上发布一致前缀的正确性论证。§4.2 的双向对照有说服力：同一设计换 RDMA 吞吐掉 4.22×，Scalog/Boki 换 CXL 传输追不上，两头夹住「收益来自按非一致共享内存重设计协调」。

折扣清单要一次说清。评估在单台服务器加单块 CMM-H 上模拟 pod，双逻辑副本争同一带宽，真实双设备部署的数字会变，方向未知；exactly-once 只测了 Retwis 一个工作流，「读也追加记录再回放」在重读负载下的放大未单独评估。另外客户端走内核 TCP/IP 占延迟 60%，换用户态或 RDMA 客户端路径后各系统的相对位置会变，论文只在服务器侧数字间接展示。

CXL 当日志介质的成熟度分三层看。设备层已有量产样品（CMM-H），但 NAND 刷写的长期行为与双设备部署的可靠性数据仍缺；协议层目前无跨主机一致性，这正是 Borges 的出发点，论文自认向前兼容未来的一致性硬件，未量化届时还剩多少优势；系统层，8-16 台 pod 的经济性只有「协调服务核心很小」的论据，无成本测算。推理系统里需要低延迟协调的场景（共享元数据、KV cache 索引）可借鉴这套发布协议；当生产组件用，等真实多设备部署的数据。

## 疑点与待核

- CMM-H 的持久性是两段式：eADR 先刷 CPU cache 行，设备再全局刷 DRAM 进 NAND（§2.2, p.213）。若第二段刷写中再次掉电，checksum 与数据副本是否仍满足 §3.7 的恢复假设，论文未讨论，需查 Samsung 设备规格。
- §4.9 的 16 分片扩展用合成流，客户端请求并非真实流量，「单 sequencer 撑住 pod」外推到更大 pod（如 64 分片）无数据支撑。
- mCAS 建模只对照 Cxlalloc 的 FPGA 原型；若未来 CXL 3.2 硬件提供更快原生原子，单写者设计还剩多少优势，论文只有一句 forward-compatible，未量化。
- lease 超时只要求「高于测得的更高分位调度抖动」（§3.7, p.218），具体数值与误检率没有给出。
- 论文说吞吐标准差低于均值 5%（§4.1, p.220），正文未附误差范围，无法逐点复核。
