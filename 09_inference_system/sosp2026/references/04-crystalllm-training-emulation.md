# 04 · CrystalLLM：用少量 GPU 忠实仿真大规模 LLM 训练

> SOSP 2026 论文精读笔记。来源：`../papers/04_crystalllm-training-emulation.pdf`（作者版 PDF，2026-11-14 下载）。文中数字均为论文口径。

## 一句话

录下大规模训练 job 的执行图，关心的 rank 跑真实程序、其余 rank 按图重放，用不到 1% 的 GPU 复现千卡训练的时长与显存行为。

## 问题与背景

在 LLM 训练的 DevOps 循环里，复现一个性能问题、验证一处改动，往往意味着动用生产集群（§1, p.1）。论文给的现实约束是：前沿模型按 3-6 个月的节奏发布，绝大多数 GPU 被生产 workload 占用，留给开发与调试的只有几百张甚至几十张卡（§2.1, p.2）。

已有两条路都不理想（p.2-3）。仿真路线中，SimAI 要为 ML framework 维护一套 mock 版本，CUDA operation 与网络行为靠手工建模，stack 演进后模型漂移，profiling 加重新实现的循环反复出现（§2.2）；降配路线更隐蔽：缩 batch、改 VPP 依赖会迫使全局配置偏离部署形态，而 OOM 与 offloading 的取舍随显存容量变化，降配环境量出的结论到生产不成立（§2.3）。

作者的出发点：调试时工程师本来就只盯少数 rank（比如 8 个），其余 rank 只是分布式执行的参与者（§2.4, p.3）。那么可以只让关心的 rank 真跑、其余虚拟化。难点有两个（p.3）：一是表示规模，采图按理要全规模跑一遍，而 rank 复用到少量 GPU 后测到的 timing 已失真；二是小规模下保住大规模行为，上千个 virtual rank 要真实参与通信，挤在几台机器上会耗尽内存、争抢资源。

## 核心设计

**两阶段架构**（§3.2, p.4）：preparation 阶段采出全 job 的执行图，emulation 阶段让目标 rank 跑原程序、其余重放。前提是两个假设（§3.1, p.3-4）：同程序同配置下执行序列确定，多次运行的图可合并重放；DP 维度对称，非对称维度（如 pipeline stage）要单独采集。原型只支持 PyTorch + NCCL，fused compute-communication kernel 尚不支持（§3.1, p.4）。

**CrystalTrace 执行图**（§4.1, p.5）：节点是计算段或通信事件，带唯一 id 与 GPU 侧 CUDA event 测得的时长；边分两类，directional（rank 内通信完成后计算才继续）与 synchronization（配对的 send-receive）。只记 GPU 侧时间戳是刻意取舍：CPU 侧的 data loading stall、kernel launch 排队都会表现为 GPU 事件之间的空隙，重放时原样保留。作为对照，PyTorch Profiler 在 128-GPU Qwen3 job 上带来 20.04% 的迭代时间增幅（§4.1, p.5）。

**上下文切换采集**（§4.2, p.5-6）：协调器在 N 张可用卡上调度 N 个 rank，各跑到通信点即阻塞，checkpoint GPU 上下文后换出；等某 collective 的全部参与者 tensor 就绪，就在 CPU 上算出结果解锁各方。纯走这条路径，128-GPU job 要 128×80 GB 约 10 TB host memory，cuda-checkpoint 恢复 80 GB 上下文要两分钟以上。替代方案是用户自定义 tensor：绕开上下文切换，直接注入通信的输入输出。Megatron-LM 里注入三类值：dataloader status（rank-0 广播 successful）、training samples（embedding 层 TP 组内广播合法 token id）、MoE dispatch all-to-all 的切分值（注入合法 in-vocab 值避免 OOM）。效果：235B MoE 在 2048-GPU target 上用 32 GPU 采图，纯上下文切换约 9 小时，注入后降为 64 轮、每轮约 1 分钟；轮数还能除以 DP 组大小，单 replica（32 ranks）只要一轮约 1 分钟。只有图组织变化（并行设置、collective 元数据、kernel fusion、recomputation）才需要重采。

**slicing 与跨片校准**（§4.3, p.7）：采集切成 slice 顺序执行，每片内一部分 rank 真跑、其余用 bare graph 重放，round-robin（0-7、8-15…）保证每个 rank 都以真身执行至少一次。片内 timing 准、跨片不对齐，靠图里的依赖关系做全局校准（例：pipeline degree 4 切两片，slice 1 的 receive 依赖 slice 0 的 send，整体后移）。论文称无校准时仿真误差很容易超过 10%（§7.3, p.12）。

**混合仿真**（§5.1, p.7-8）：sandbox rank 跑原程序；virtual rank 遍历图，计算节点等满记录时长，通信节点做真实通信。通信 tensor 由 runtime prefetch pipeline 供数，GPU buffer 满则 staging 到 CPU buffer pool；512-GPU job 配一台 assistant 机时，每 GPU 可提供 100 GB 以上 replay buffer（§5.2, p.8）。

**virtual rank 初始化**（§5.2, p.7-8）：上千个 virtual rank 各建 communicator 与 buffer（每组 500 MB）会耗尽显存，初始化还会争抢 CUDA context 与驱动锁、串行化到小时级。NCCL group reduction 只实例化与 sandbox rank 有交集的 group，其余在 PyTorch c10d 层根本不建：2000 virtual rank 时 active group 从 8,617 降到 82。world size 不变，barrier 走 TCPStore 计数，缺席参与者由 assistant rank 里的 leader 代发请求；NCCL 初始化的 control channel 也由 leader 代管，代为交换 intra-node topology。

**运行时通信剪枝**（§5.3, p.8-9）：只让与 sandbox 相邻的 virtual rank 参与集体通信，靠算法性质保数值正确。ring all-reduce 的 reduction 阶段每个 rank 只对自己负责的 chunk 的正确性负责，最左 vRank 准备扣除 sandbox 贡献的补偿值，sandbox rank 沿环累加后得到正确的全值；broadcast 阶段由持有全值的 vRank 供值。tree 算法里 sandbox head 按 root、leaf、intermediate 三种边界角色处理（p.9）。剪枝后执行的传输数量与体积与原 collective 相同，避免引入非确定性（p.10）。

## 效果与代价

评测 setup（§7, p.10-11）：2048-GPU testbed（8 GPU/台），Megatron-LM 上的 Qwen3 MoE 预训练，模型 M.1 235B-A22B、M.2 90B-A20B、M.3 1.01T-A43B，策略 S.A-S.D（TP 1-2、PP 4-16、VPP 0-2、EP 8-16、GA 8-32），在 512/1024/2048 三档目标规模与全规模 baseline 对比，每组 85 次迭代取中间 70 的中位数。

精度：迭代时间平均误差 0.58%、最大 1.98%（Figure 7, p.11）；峰值显存误差全部低于 0.01%，且复现了 baseline 的 OOM 位置（Figure 8, p.11）。kernel 时长落在 baseline 自然抖动带内，启动时刻最大偏差 3.64%（Figure 12, p.12）。初始化是重放系统的常见坑：baseline NCCL 在 512 rank 就要 544.5 GB CPU 内存加 103.7 GB GPU 内存、756 s，≥1024 rank 直接 OOM；CrystalLLM 初始化 8192-rank 仿真只要 44.3 s、每卡 5.4 GB（§7.3, p.12）。

规模与开销（§7.2, p.11）：仿真 8192-GPU 集群只需 32 台物理机（16 sandbox + 16 assistant），不到目标 GPU 的 0.4%，节省超过 99%。assistant 与 sandbox 按 1:1 扩展后，端到端仿真时间稳定在 80 分钟内，仿真以与真实训练相同的每迭代 wall-clock 推进；assistant 不随规模扩展时，1024-GPU 案例慢 1.6 倍（125 分钟）。对比实验：Phantora 在其支持的配置内典型误差 12%，序列长度降到 64 时误差升到 64%（§7.4, p.12-13）；SimAI 在相同配置下平均误差 77.2%（§7.4）。

论文自认的边界（§3.1, p.4；Appendix C.1, p.18）：目标代码跑不起来或硬件尚不存在时应选 simulation；runtime 数据改变控制流的负载（RL post-training）固定图重放失效；收敛性、数值漂移、跨全 rank 的瞬态 straggler 分布必须全规模执行。128-rank 大消息下单 Node 承载全部 virtual rank 会出现明显延迟偏差（32 GB 消息 267.42 vs 177.62 ms），需横向扩展 assistant 缓解。

## 总评

这套工作回答的问题很具体：对迭代时长与峰值显存这两个训练工程里最高频的问题，全规模行为能用多少资源复现到什么精度。0.58% 与 <0.01% 说明，只要执行结构（what、in what order、how long）保真，通信真跑、计算按时长重放，就足以逼近真集群。它把「哪些行为必须在全规模下才成立」拆成了可判定的条件（§3.1），这份边界清单比误差数字更有复用价值。

没回答的部分同样清楚。仿真速度与真实训练 1:1（§7.2），论文不提供时间压缩，收益全部落在资源侧与可复现性上；评测集中在 MoE 预训练的 85 次迭代短窗口，长时间训练的误差累积没有数据；对未见过的负载没有外推能力，并行设置一变就要重采图（好在单 replica 采集约 1 分钟，§4.2）。

对做训练平台与集群调度的读者，价值在于它把生产集群行为变成一份可离线重放的执行图资产。§8 的三个案例（一次搜索 90 个配置策略、把一个内存 bug 的保留-已分配缺口从 19% 定位到 8%、筛出被热节流到 900 MHz 的故障 node）都在不动生产集群的前提下完成。对仿真保真度类工作的通用矛盾（保真度与维护成本），这份方案绕开了建模：真硬件跑真代码，把建模问题换成录制与重放的工程问题，代价是对运行时数据相关的负载直接失效，适用面比分析模型窄。

## 疑点与待核

1. §7.3 校准例子的数字方向存疑：原文写 emulated iteration time "before inter-slice calibration quickly drops from 5.7s to 5.13s"，其中 5.13 s 与 Table 4 的 baseline 5.62 s 相差 8.7%（对应「误差可超 10%」的叙述），5.7 s 反而更接近 baseline；数字与叙述的对应关系需核对原文。
2. §5.2 称单 GPU 提供 "over 100 GB of buffer space"，但全篇未写评估用 GPU 型号，显存代际无法推断，此数字待核。
3. Table 3 的足迹缺推导：512-GPU target 只占 15 台 sandbox node、2048-GPU 用 25 台，与「每 slice 装 max(TP, EP) 级紧耦合组」（§3.2, p.4）之间的换算论文未展开。
4. 多迭代下经 virtual rank 中转的 PP activation 取值陈旧：重放喂的是录制或注入值，论文证明了 collective 的数值正确性（§5.3），但 PP 点对点路径上 sandbox rank 拿到的是录制时刻的 activation，若数值触发数据相关控制流，图会偏离，论文未讨论。
5. SimAI 77.2% 的平均误差远高于 SimAI 原论文报告的精度；论文归因于 workload 文件缺 PP 通信 bubble 与 MoE 开销，但评测用的是含 VPP/GA 的自有策略组合，引用该数字前需注意配置差异。
6. 每组只评 85 次迭代（§7.1），且仿真 wall-clock 与真实 1:1，速度上无压缩收益；长时间运行的漂移累积与「缩短实验时间」类需求未被覆盖。
