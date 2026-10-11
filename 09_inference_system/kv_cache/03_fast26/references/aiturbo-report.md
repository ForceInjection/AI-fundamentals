# AITURBO 精读报告(FAST'26)

**论文**:Fast Cloud Storage for AI Jobs via Grouped I/O API with Transparent Read/Write Optimizations(上海交大 IPADS + 华为云,Yingyi Hao 等,FAST'26,pp.255–270)
**版本说明**:实际 PDF 为 17 页全文(正文至 p.267),并非 7 页删节版,评估数字完整可用。系统名 AITURBO。

## 一、问题与动机

AI 负载正在吃穿分离式云存储的带宽:训练侧是周期性 checkpoint 写(数百 GB 量级),推理侧是 autoscaling 时的 checkpoint 读与 prefix 共享的 KVCache 读(§2.2)。量化证据:华为云观察到 AI 任务已占本地数据中心云存储带宽的 10% 以上(§1,p.255,**厂商口径**)。补带宽有两条堵路:(1) 花钱扩 backend 带宽——带宽从 1.6 提到 80 GBps,每 GB 成本涨 16 倍(Figure 1(c),p.259);(2) 计算侧自行优化——Megatron 用了约四分之一个代码库做 checkpoint I/O 优化仍不最优,因为它看不到云存储的分离式架构(§1,p.255)。根因是硬件口径差:compute fabric 200 Gbps/XPU,而 storage fabric 仅 100 Gbps/节点(Figure 1,p.256),且 fabric 优先保障 XPU 间通信。

## 二、核心洞察

两条(§1,pp.255–256):其一,AI 集群的 host DRAM 与 compute fabric 是闲置的,可作为快速 staging buffer 中转 XPU↔存储的数据—— durability 换性能对 checkpoint 这类可重写数据可接受(丢失靠副本或上一版 checkpoint 兜底)。其二,存储层可以自动推导 dedup 与负载均衡读写计划,只要应用通过 grouped read/write API(受 NCCL group communicator 启发)声明「哪些客户端将一起参与这次 I/O」:单条 read/write 只暴露单文件语义,而 group 语义让云存储看到完整 I/O 模式,且存储提供商比应用开发者更懂底层架构,于是能透明做出优于应用层优化的计划。这两个洞察为何可行:AI 负载天然成组(训练全体 XPU 同步写 checkpoint、推理扩缩容同时读同一模型、agent 同前缀任务并发读同段 KVCache,Figure 6),组内重复内容多(Table 1),优化空间大。

## 三、系统设计

**API 形态**(Figure 7,p.259):`group_getfile / group_putfile`,比原 getfile/putfile 多一个 `group=` 参数,返回两个 future——`future_0`(数据已入 CPU DRAM buffer)、`future_1`(数据已持久化到 backend);checkpoint 场景 await future_1 再进下一迭代。写路径三步(§4.1,Figure 9):① XPU 侧用优化过的 BLAKE3 内核算校验和做去重(1 GB 文件 CPU 35.6 ms → XPU 7.8 ms,p.261,粒度 4 MB 块到整文件);②③ 把去重后的负载写入 host DRAM 并写回存储,写计划形式化为 bilinear programming(7 条约束,目标最小化完成时间 t),利用 t 的上下界二分 + 迭代解 LP 可行性求 plan,38B 模型 64 XPU 场景未优化 Python 单线程 4 秒出 plan(p.262),之后按 job 生命周期缓存复用。执行时 DRAM→存储流水线化(Figure 11)。读路径(§4.2):重复块只从存储取一份到某节点,再用 compute fabric 按 BlitzScale 式串行转发链广播给其余节点,两阶段流水;命中他节点 DRAM 缓存时直接走 compute fabric 取数。配套机制(§4.3):tensor-native 文件类型免序列化、cgroup 管理缓冲池、自建点对点 QP 代替 NCCL(建链 15 ms 且可与存储初始化重叠,避开 NCCL communicator 秒~分钟级冷启动)、RoCE QS 最低优先级做性能隔离。

## 四、评估

测试床:Ascend 910B(64 GB HBM)与 A800(80 GB)两集群,至多 64 XPU,每节点 192 核 / 1.5 TB DRAM、1 张 100 Gbps S-NIC,30 Gbps backend 带宽摊到 XPU(§5)。基线:Megatron+SFSTurbo(华为通用云存储,类 3FS)、Megatron+Gemini(闭源,作者在 Megatron commit cac60ce 上复刻其技术)、ServerlessLLM(ckpt 读)、Mooncake(KVCache)。
- **checkpoint 写**(Figure 13,p.264):1.5B/13B/38B × 有无 ZeRo 六种配置,AITURBO 比 Megatron+SFSTurbo 快至多 5×,Gemini 关掉重复写场景下再快至多 5.9×;生产环境口径为比 SFSTurbo 快 3.9–58.8×(§1,p.256,厂商声称)。factor analysis:+Dedup 降 4.3–47.2%,+Write plan 再降至多 76%(p.265)。
- **checkpoint 读**(Figure 17,p.265):8 XPU、1 GBps 下所有系统读 135 GB 的 Qwen-72B 都要 173 s(存储瓶颈);但一旦单副本进入 DRAM,AITURBO 在 64 XPU 上 2.25 s 完成部署且近乎与 XPU 数无关,ServerlessLLM 需 1,384 s。
- **KVCache 读**(Figure 18,p.266):Qwen-14B + Qwen-Bailian 真实 trace 重放 30 min,Mean TTFT 0.8 ms → 0.65 ms(降 23%);推理整体口径比 Mooncake 快至多 1.28×(§1)。
- **工程量**(Table 2,p.266):Megatron 自研 checkpoint 优化 2,228 行,换 AITURBO 只改 286 行;Mooncake 仅改 44 行;group 协调开销 64 XPU 最大 45 ms(Figure 20)。

## 五、局限与适用边界

论文承认(§1 Discussion,§4.3):只优化 XPU 间大块传输,小块 I/O 不受益;性能隔离仅用硬件 QoS 最低优先级,同 XPU 混部等复杂场景留待未来;BLAKE3 碰撞概率按生日界论证为可忽略;前几迭代优化器状态全零导致假去重,靠 pattern 稳定后缓存 plan 缓解。**精读员注**:① 收益强依赖「compute fabric 有富余 + host DRAM 闲置」的前提,论文自己承认推荐类业务的 DRAM 是满的(p.256);通信密集负载(如大规模 MoE all-to-all)与存储借用竞争时,最低优先级 QoS 下的实际带宽未刻画。② future_0/future_1 语义本质是拿持久性换性能,DRAM 副本全失即丢数据,对通用存储不成立。③ Mooncake 实验里并未真正使用 group API(「同步运行中的推理实例太困难」,p.267),只替换了 miss 后的存储读路径——推理侧 23% 的改善并非来自组去重本身。④ 计划生成按 job 生命周期缓存,弹性训练/频繁扩缩容导致的组成员变化是否触发重计划、代价多大,论文未覆盖。

## 六、关联

与 **Mooncake**(FAST'25)是分层互补而非竞争:Mooncake 用 compute fabric 建分布式内存 KVCache,AITURBO 位于其下的存储层,接管 cache miss 后的存储读,故能直接叠加(44 行改动)。与 **Gemini**(SOSP'23)同为 DRAM staging + 多副本容错,但 Gemini 不做去重与写计划优化;与 **ByteCheckpoint**(NSDI'25)对比,后者需改框架,AITURBO 主张优化下沉存储层。**JuiceFS/GooseFS 论文中未出现**(精读员注):二者是对象存储上的 POSIX/缓存型文件系统,本质差异在于 AITURBO 的 group 语义把跨客户端 I/O 模式暴露给存储层、并把计算网络与闲置 DRAM 当作传输介质——这两点是通用云 FS 缓存层目前不具备的,也是本文最可迁移的思想。「已在华为云生产部署」(§1、§7)意味着: separating 设计经受真实多租户与故障率(1 XPU/小时的失败率假设来自生产集群)检验,3.9–58.8× 是生产数字;但同时也意味着结论绑定华为云硬件口径(Ascend/A800、RoCE、SFSTurbo),迁移到其他厂商 fabric 需自行复核。
