# CXL 互联协议全景解读：从 PCIe 时代到可组合数据中心

> **论文**：《An Introduction to the Compute Express Link™ (CXL™) Interconnect》
> **作者**：Debendra Das Sharma（Intel，CXL 规范首席架构师）、Robert Blankenship（Intel）、Daniel S. Berger（Microsoft Azure）
> **来源**：arXiv:2306.11227（2023 年 6 月），35 页
> **性质**：面向系统研究者的 CXL 1.0/1.1/2.0/3.0 完整技术综述（tutorial），第一作者是 CXL 规范本身的撰写者，本文是其 IEEE Micro 系列论文与 CXL 白皮书的浓缩版

这篇论文的价值不在理论新颖性——它本身是标准文档的"人话版"——而在于把散落在三版规范中的协议设计动机、取舍逻辑与实测数据压缩进一篇文章。对系统工程师而言，理解 CXL 不需要背 Flit 布局表，而需要回答四个递进的问题：**为什么需要它（Context）、它解决什么问题（Issue）、每代协议如何逐步逼近目标（Solution）、付出了什么代价（Trade-off）**。本文按这一叙事展开，并在最后给出 CXL 对 AI 基础设施（GPU 服务器、内存墙、机架级系统）的启示。

---

## 一、背景：互连的旧秩序与 CXL 的登场

处理器与外部设备之间的通信，长期以来由两条互不交叉的路线承担：**PCIe 串行总线**连接 CPU 与设备（GPU、NIC、FPGA、存储），**DDR 并行总线**连接 CPU 与内存。二者各有所长，但都带着与生俱来的结构性缺陷——而正是这些缺陷，在 AI/ML 工作负载兴起后变成了不可回避的瓶颈。

PCIe 的成功建立在两点上：向后兼容与开放生态。它承担"控制面"职责时无可挑剔——设备发现、配置、DMA、中断，全部通过非一致的 load-store 语义完成。但**非一致意味着设备无法缓存系统内存**：每次访问都穿越 Root Complex，无法利用时间/空间局部性，无法执行原子操作序列。当加速器（GPU、智能 NIC、PIM 设备）需要与 CPU 共享同一份数据结构时，只能整块搬运、软件同步——这正是 AI 推理中常见的"数据先拷入设备、算完再拷回"模式背后的协议根源。

DDR 的问题则是**引脚效率**。论文给出了一个决定性的对比：x16 Gen5 PCIe 端口（32 GT/s）用 64 根信号引脚提供 64 GB/s/方向（双向 128 GB/s）；DDR5-6400 用约 200 根引脚只提供 50 GB/s——**PCIe 每引脚带宽高出 DDR 约 8 倍**（128/64 = 2 GB/s/引脚 vs 50/200 = 0.25 GB/s/引脚）。（论文原文此处写作"256 GB/s"，疑为笔误：32 GT/s × 16 lanes 的物理上限即 64 GB/s/方向，且论文自己的 8× 结论按 128 GB/s 双向计算才成立。）PCIe 引脚原则上完全可以替代 DDR 承担内存互连，但它不提供一致性与内存语义，设备侧内存无法映射进可缓存的系统地址空间，因此替代无从谈起。内存墙的根源不只是延迟，还有这条协议层面的"各管一段"。

第三条路线是历史上反复尝试过的**对称一致性互联**：Intel QPI/UPI、AMD Infinity Fabric、IBM Bluelink、NVIDIA NVLink，以及试图把一致性推向更大范围的 CCIX 与 OpenCAPI。对称互联要求每个节点实现相同的缓存一致性算法并跟踪全部状态，复杂度高、各架构实现差异大、演进缓慢；而且并非所有场景都需要一致性——内存扩容（Challenge 2）与资源池化（Challenge 3）需要的是**内存语义**而非完整一致性。对称路线因此始终没有成为跨厂商的统一标准。

CXL 的诞生是上述三条线索的汇合。Intel 早在 2005 年就有"在 PCIe 上加简化一致性机制"的想法，并先后在 PCIe 3.0 中加入原子语义、尝试过共享内存控制器（SMC）方案——但 PCIe 3.0 仅 8.0 GT/s 的带宽让池化失去意义。2019 年，PCIe 5.0 的 32.0 GT/s 与成熟的线缆生态让带宽不再成为障碍，Intel 以自有协议 Intel Accelerator Link（IAL）为基础，**于 2019 年 3 月将 IAL 1.0 更名发布为 CXL 1.0**，与 Alibaba、Cisco、Dell、Google、Huawei、Meta、Microsoft、HPE 共同成立 CXL 联盟。此后竞争标准（CCIX、GenZ、OpenCAPI）相继合并入局，联盟成员膨胀至约 250 家公司，版本演进为：CXL 1.0（2019.3）→ CXL 1.1（2019.9，加入合规测试机制）→ CXL 2.0（2020.11）→ CXL 3.0（2022.8），全程向后兼容。

CXL 的设计哲学可概括为三个词，它们贯穿三代协议：**非对称**（主机承担一致性编排，设备只需实现简化 MESI，降低设备端实现门槛）、**向后兼容**（复用 PCIe 物理层、槽位、软件栈，设备可以从 PCIe 平滑迁移）、**开放**（多协议分层，设备只需实现与自己用途相关的协议子集）。这套哲学是理解后续所有技术选择的钥匙。

---

## 二、问题：四个驱动 CXL 的行业挑战

论文将 CXL 要解决的行业问题收敛为四个挑战，全部有具体量化支撑，而非泛泛之谈。

**挑战一：系统内存与设备内存之间缺少一致性访问。** PCIe 设备不能缓存系统内存，加速器访问共享数据必须整块搬运并在软件层用屏障机制防止 CPU 与设备并发冲突。这直接阻碍了 AI/ML、智能 NIC、近内存计算（PIM）等"CPU 与设备同时访问同一数据结构"的新范式。PIM 尤其受困：没有一个标准化的方式让 PIM 设备一致性访问可能驻留在 CPU 缓存层次中的数据，编程模型因此极度笨重，阻碍了 PIM 的普及。

**挑战二：内存可扩展性跟不上算力增长。** 计算呈指数增长，DDR 却因引脚效率低而拖了后腿：加 DDR 通道带来平台成本上升与信号完整性难题；DRAM 每比特成本近年持平，而 DDR 命令集（Refresh、Bank Pre-charge 等）又与 DRAM 介质强绑定，阻碍 ReRAM、3D XPoint 等新介质类型被采纳。PCIe 引脚本是更优的内存互连载体（每引脚 8 倍带宽、支持 retimer 长距离布线、单 DIMM 功耗可超 15W），唯独缺一致性与内存语义。

**挑战三：资源滞留（stranding）导致的算力与内存双重浪费。** 一台服务器的计算、内存、I/O 设备被物理绑定为"一台机器"，当工作负载峰值需求波动时，每台服务器都必须按峰值超额配置内存与加速器：算力跑满的服务器内存闲置，内存吃紧的服务器却无法从机架上其他服务器借用。阿里、AWS、Google、Meta、Microsoft 的数据均报告了低资源利用率，stranding 同时带来功耗、成本与可持续性三重损失。根本原因是**资源的耦合粒度过细**——资源属于服务器，而不属于资源池。

**挑战四：分布式系统的细粒度数据共享。** Web 级应用（搜索、社交内容合成、广告选择）的 partition/aggregate 模式中，查询更新通常小于 2kB；分布式数据库依赖 kB 级页面，共识协议更新更小。这些更新对延迟极度敏感——4kB 数据在 50 GB/s 链路上传输不足 2μs，但数据中心网络通信延迟普遍超过 10μs，**传输时间只占等待时间的五分之一**。一致共享内存可以把通信延迟压缩到亚微秒级。

四者轻重不同：挑战一与二指向单机内的协议能力（一致性与内存语义），挑战三与四指向系统级重构（池化与跨主机共享）。三代 CXL 的演进顺序恰好对应了这个由内而外的次序——先解决单机内的语义，再解决机器间的组合方式。

---

## 三、方案：三代协议的演进之路

### 3.1 CXL 1.1：单机内的一致性与内存语义

CXL 1.1 是整座大厦的地基：它在 PCIe 物理层之上复用了三套协议，定义了三种设备类型，并确立了低延迟的实现路径。后续两代的所有能力都建立在这套分层之上，因此值得逐层拆解。

**三协议复用同一物理层。** CXL.io、CXL.cache、CXL.mem 被动态复用在同一对 PCIe 链路上（论文图 2 的多协议复用模型）。CXL.io 基于 PCIe，承担设备发现、状态上报、虚拟地址翻译与 DMA，是**所有设备必选**的协议；CXL.cache 让设备可以缓存系统内存；CXL.mem 让设备内存映射为可缓存的主机内存（Host-managed Device Memory，HDM），主机获得"本机内存 + 设备内存"的统一视图。三者的存在与否，恰好划分出三种设备类型：**Type 1**（仅 io+cache，如智能 NIC）、**Type 2**（io+cache+mem，如带本地内存的 GP-GPU/FPGA）、**Type 3**（仅 io+mem，如内存扩展设备）。设备按用途实现协议子集——这是 CXL 控制设备复杂度的核心手段。

**68 字节 Flit 与低延迟路径。** 三种协议的传输单元是 Flit（流控单元）：2 字节协议 ID + 64 字节负载 + 2 字节 CRC。64 字节负载与 cache line 等长，使 CXL.cache/mem 无需分组/重组开销；协议 ID 带冗余编码（Hamming 距离 4 且重复两次），CRC 可检测 64 字节负载内最多 4 个随机位翻转。与 PCIe 的 TLP/DLLP 变长分组相比，Flit 化让链路层与事务层路径大幅缩短，再加上**在物理层做多路复用**（而非更高层）、128b/130b 编码可协商旁路等优化，CXL.cache/mem 的延迟被压到与"本地 socket 的 DDR 访问"同一量级——这正是"与对称互联相当、但只承担非对称复杂度"的关键兑现。

**CXL.cache：主机编排的简化一致性。** 设备通过 CXL.cache 缓存主机内存，一致性协议采用经典 MESI，但**主机负责全部对等缓存跟踪**，设备从不直接与对等缓存交互——这是非对称设计的精髓。协议由每方向三条通道构成（Request/Response/Data），设备侧 D2H Request 提供 4 类 15 个命令（Read、Read0、Read0-Write、Write），主机侧 H2D Request 用于对设备缓存发起 Snoop。地址翻译通过 PCIe ATS 扩展完成：设备实现 DTLB 缓存页表项，CXL 扩展 ATS 以区分"允许走 CXL.cache"与"仅限 CXL.io"的访问。协议细节中最见功力的是冲突处理：GO（Global Observation）消息必须排在同地址后续 Snoop 之前（Req-to-Snoop 情形），保证设备先感知自己已获 E 状态所有权、才能正确响应后续 Snoop；Evict-to-Snoop 情形下，脏数据驱逐在途时到达的 Snoop 必须拿到 M 态数据，随后到达的 GO_WritePull 虽仍需返回数据但必须标记为"可能过期"，主机据此丢弃。这些规则保证了多代理并发下的正确性，而代价全部由主机承担。

**CXL.mem：主机可缓存访问的设备内存。** 协议只有两对通道（M2S 的 Request/RwD 与 S2M 的 NDR/DRS），通道间不强制排序以换取简单与低延迟。CXL 3.0 术语将 HDM 分为两类：**HDM-H**（host-only coherent，主机独占一致性，典型于 Type 3 内存扩展器，可选 2-bit Meta Value 供主机使用）、**HDM-D**（device-managed coherent，典型于 Type 2 加速器，设备通过 DCOH agent 跟踪主机缓存状态，并通过 **Bias Flip 流程**——基于设备侧 Bias Table 记录 Host-S/Host-A/Device 三态——主动翻转地址的"偏好"以回收所有权）。HDM-D 的一致性仲裁最终落在设备侧：设备是它拥有地址的一致性的最终裁决者。

**协议依赖图：无环保证的死锁自由。** 论文用协议依赖图（L1=CXL.cache、L2=host-specific、L3=CXL.mem 三层的通道依赖关系）论证协议内部与跨协议层的**无死锁性**：只要依赖图无环，就不会出现通道间循环等待。这是一致性协议设计中罕见的"可证伪"论证方式，也为 CXL 3.0 的 Back-Invalidate 通道加入时如何维持无环提供了检验工具。

### 3.2 CXL 2.0：资源池化与运行时重构

CXL 1.1 把内存语义带给了单台机器，但资源仍然绑定在机器上——stranding 依旧。CXL 2.0 的目标是把资源从"属于某台服务器"变成"属于一个可重分配的池"：**与其为两台服务器按峰值分别配置内存，不如按均值配置一个共享池**，运行时把空闲资源动态重分配给吃紧的主机。这个转变依赖一组新能力：热插拔、单级交换、内存与设备池化、内存 QoS 与全局持久刷新（GPF），并由 Fabric Manager 统管池分配。

**交换与虚拟层次。** CXL 2.0 引入单级交换：主机看到的 CXL 拓扑被抽象为虚拟层次（VH）——每台主机拥有独立的 Virtual CXL Switch（VCS），只含分配给它的桥与设备。Flit 依据虚拟桥路由，每个主机-设备对之间只有一条路径（有向树拓扑）。设备池化建立在此之上：标准设备是 Single-Logical-Device（SLD，同一时刻只归属一台主机）；**Multi-Logical-Device（MLD）**则把一个 CXL.mem 设备的资源切成最多 16 个逻辑设备（LD），按 256MB 粒度经 Set-LD 命令创建，各 LD 可同时归属不同主机。LD-ID 只存在于交换与设备之间的链路上，对主机不可见——交换根据主机端口打标签。CXL 2.0 的拓扑由此被严格限制为"单交换、单路径"，这个限制要到 CXL 3.0 才被打破。

**"池"的主机视图：多个独立设备，而非一个聚合设备。** 这里澄清一个常见误解：交换机并不把众多内存设备"聚合"成一个大设备呈现给主机——每台主机枚举到的是分配给它的每个设备（SLD）或每个 LD（MLD），各自拥有独立的 PCIe 配置空间、CDAT 属性表与 HDM decoder，设备粒度在主机侧完全可见。"一个统一内存池"的统一性来自软件层：主机侧 HDM decoder 可跨多个设备做地址交错（Linux 下 `cxl create-region -m -w 2 -g <粒度> mem0 mem1`——`-m` 指示位置参数为 memdev 名，`-w` 与 memdev 数量一致，把一段 HPA 条带化到多个 memdev）；OS 内存层级（memory tier）把它们归入同一慢内存层统一调度；应用层再以 mmap + 偏移约定实现划分与共享。另一层语义边界同样值得记住：标准 CXL 2.0 下同一 SLD 或同一 LD 同一时刻只归属一台主机——多主机并发访问同一设备地址的硬件一致性共享内存是 CXL 3.0 借 Back-Invalidate/GFD 才进入规范的；2.0 时代的跨主机共享只能以"设备/LD 归属切换 + 软件协议"实现。

**Fabric Manager 与池管理。** 设备如何动态绑定到主机，由 CXL Fabric Manager（FM）裁决：FM 可以是主机上的软件、BMC 固件、交换内嵌固件或专用设备，通过 Component Command Interface（CCI）经带内（MMIO）或带外（MCTP：SMBus/I2C、PCIe、USB、串口）下发命令。绑定（bind）需四个参数：交换 ID、虚拟桥 ID、物理端口 ID、逻辑设备 ID；解绑（unbind）支持三种语义——等待主机停用、热移除并等待主机、**强制热移除**（后者允许 FM 与不配合的主机协作，例如裸金属云中归属第三方客户的主机）。设备归属变更对主机表现为标准热插拔事件，无需重启。QoS 层面，CXL.mem 响应消息携带 DevLoad 字段报告设备负载，主机据此调节注入速率；多源场景用源节流（source throttling）控制各源对 MLD 资源的占用上限。

**主机软件的新角色。** 池化把"静态资源配置"变成了"运行时事件"。传统内存拓扑经 ACPI 静态表（SRAT/HMAT）在开机时固化；CXL 2.0 引入 **CDAT（Coherent Device Attribute Table）**——内存设备运行时热插时，主机读取其 CDAT 寄存器，获得内部 NUMA 域、内存区间、带宽与延迟属性，由 OS 分配空闲 HPA 并编程 HDM decoder。设备驱动的还是 PCIe 驱动框架，但内存资源的管理从固件静态配置转向 OS 运行时管理。

### 3.3 CXL 3.0：织网与规模跃迁

CXL 3.0 的目标是把 CXL 2.0 的池化从"单交换、16 台主机"推向"多级交换、4096 个端点（主机、内存、加速器、NIC 均可）"。论文给出的核心矛盾是：CXL 2.0 的单路径树拓扑源于 PCIe 的 ordering 规则——只要一致性/有序性依赖树形路由，规模就被锁死。CXL 3.0 的每一步都是对这把锁的拆除。

**64 GT/s 与 256 字节 Flit。** 带宽翻倍依赖 PCIe 6.0 的 PAM-4 电信号（4 电平，每 UI 编码 2 bit），但 PAM-4 的眼图高度/宽度显著劣化，首比特错误率高达 10⁻⁶，必须引入前向纠错（FEC）。CXL 3.0 定义了两种 256B Flit：常规型（8B Reed-Solomon CRC + 3 路交错单符号纠错 FEC）与**延迟优化型（128B LO Flit）**——由两个 128B 子 Flit 组成，FEC 跨整个 256B 共享，但每个 128B 半 Flit 有独立 6B CRC。关键工程洞察是 **CRC 先于 FEC 执行**：CRC 约 10 级逻辑门，FEC 约 50 级，先查 CRC 在无错路径上省约 2ns（x16 链路），且 128B 的累积延迟还略优于 32 GT/s 下的 68B Flit。出错时再累积全 Flit 做 FEC、重查 CRC。Flit 模式（68B/256B/LO）在链路协商时确定，与运行速率解耦——即使 64 GT/s 链路降速到 32 GT/s，LO Flit 模式仍然生效。

**UIO：无序 I/O 拆掉树的锁。** PCIe ordering 规则是树拓扑的制度性根源。CXL 3.0（随后被 PCIe 6.0 采纳）定义 **Unordered I/O（UIO）**：在 VC1–VC7 上传输完全无序的事务，**排序责任上移到源节点**——每个 UIO Write 都带 completion（实质上是非 posted 的），生产者必须先等数据写完成、再写 Flag；UIO Read 正常返回带数据的 completion。跨流控类与流控类内部的排序要求全部取消，于是任意源-目的对之间可以走多条路径、仍保住生产者-消费者语义。VC0 保留传统有序传输用于向后兼容。一句话概括：**树拓扑是为了全局有序，UIO 把有序性从"全网属性"降级为"源点行为"，全网就可以是任意拓扑。**

**Back-Invalidate：绕过主机的一致性回写。** CXL.mem 新增 S2M 方向的 Back-Invalidate 通道（BISnp/BIRsp），设备可主动把主机缓存中的副本作废（对应新的 HDM-DB 内存类型，Type 2/3 均支持）。由此解锁三个用法：其一，**设备直连设备**——设备经 UIO 直接访问交换上其他设备的 HDM 内存，命中 I/S 态直接服务，需要一致性时才用 BI 反向回写主机（论文例：NIC→内存直连 8 跳，经主机 16 跳往返）；其二，**Type 2 设备可用 snoop filter 取代全量目录**——容量未命中时用 BI 驱逐，从而能把大块本地内存映射进 HDM-DB 区域；其三，**多主机硬件一致性共享内存**——内存设备（MLD 或 GFD）维护目录（每 cache line 两 bit 一致性状态 + 共享者列表：I/S/E），Host 1 取共享副本置 S 并记录，Host 4 要独占时向 Host 1/Host 3 广播 Back-Invalidate、收到全部确认后才转交所有权。这是第一次把"多台独立主机的缓存一致性"放进一个内存设备：GFD（Global Fabric-Attached Memory）可同时服务 4096 个独立节点（MLD 上限 32），通过不参与各节点配置空间枚举实现规模扩张。

**PBR：面向 4096 端点的路由。** 端口基路由（Port Based Routing）放弃 CXL 2.0 的分层地址路由（交换需维护每台主机的 VH 与地址映射），改为 12-bit PID 纯端口路由：边缘端口（edge port）做**无状态**的 HBR↔PBR 转换（Address→PID 经 FAST 表、LD-ID→PID 经 16 深查找表、CacheID/Bus Number→PID），内部交换（spline）只按 DPID 查转发表、完全不做地址解码。路由表由 FM 集中配置，支持多条目的端口实现负载均衡与故障重配置——再次印证"中心控制、简化转发"的 CXL 哲学。ISL（交换间链路）需要承载 12 条对称通道（上行/下行各 12，含独立流控），与主机/设备链路的单向上行或下行形成对比。CXL.cache 也借 CacheID 字段（4-bit）从每 root port 一个缓存设备扩展到 16 个，但主机 snoop filter 的规模仍约束其总数。

### 3.4 实测数据：延迟与带宽的量化验证

论文的第三部分价值在于给出可审计的数字。延迟方面（论文图 19 的微架构拆解）：CXL 端口内部往返延迟为 **21ns（共同时钟）/ 25ns（独立时钟）**，一次 CXL.cache+mem 访问要穿过端口两次（CPU 侧 + 设备侧），加上 15ns 带 retimer 的飞行时间，得到 **57ns 的端到端延迟增量**——与 CPU-CPU 一致性链路相当，且落在规范 pin-to-pin 目标（内存访问 80ns、snoop 响应 50ns）之内。CXL.io 的内存读延迟约 275ns（LLC miss + IOTLB hit，与 PCIe 相同）。按拓扑估算：CPU→Type 3 SLD 内存约 **170ns**；经一个 CXL 交换约 **250ns**；同 SMC 的两 CPU 消息往返约 **220ns**；跨两个交换约 **270ns**。CPU 侧 load-to-use 均 <100ns。

带宽方面，x16 @ 32 GT/s 的原始带宽 64 GB/s/方向、@ 64 GT/s 为 128 GB/s/方向，乘以链路效率（68B Flit 模式 0.924、256B/128B LO 模式 0.938——LO 模式"15 槽数据 + 1 槽开销"的结构让各 Flit 类型效率趋同）再扣除协议头占用：CXL.cache 读约 **56.6 GB/s**（68B @ 32 GT/s）、112 GB/s（256B @ 64 GT/s）、104 GB/s（LO @ 64 GT/s）；写约 40 GB/s（68B @ 32 GT/s，每次写要 Req+Evict+MemWr 三个头）、73.8 GB/s（256B/LO @ 64 GT/s）。CXL.mem 纯读约 53.5 GB/s（68B @ 32 GT/s）。论文特别强调 UIO/BI 的收益：即使 100% 访问都触发 Back-Invalidate 的病理情形，其链路效率仍优于传统"全部绕经主机"的多 cache line 传输。

硬件生态方面，论文调研时点（2023 年中）：Intel Sapphire Rapids 与 Agilex7 FPGA 支持全部三协议，AMD Genoa/Bergamo 支持 CXL，ARM V2/N2/E2 宣布 CXL 2.0；Samsung 公开了 CXL 1.1 内存扩展器的基准数据，SK Hynix/Micron/Microchip/Astera 等宣布 Type 3 设备，Micron 原型了 CXL 1.1 近内存计算设备。论文坦承实测数据以 Intel 为主（公开数据主要来自 Intel 平台），并以"代表性性能指标"定位。

---

## 四、实践：CPU 访问 CXL 内存的设备形态

协议标准定义了 CXL 设备与主机之间的"线路语言"，但 CPU 到底以什么形态使用这块内存，取决于操作系统把这批地址空间交给谁管理。Linux 的 CXL 子系统给出了一条值得玩味的统一路径：**无论易失还是持久，每个 CXL 内存 region 创建后首先以 DAX 设备（`/dev/daxN.Y`）的形态存在，再由使用者决定"留在用户态做 mmap"还是"转交给内核页分配器"**。本章基于内核 6.19 源码（`drivers/cxl/`、`drivers/dax/`），拆解这条路径上的三种最终形态——System RAM、devdax、持久内存栈——以及各自对应的 CPU 访问方式。前文讨论的是协议层语义，本章是操作系统层语义。

### 4.1 统一入口：region 创建与 DAX 设备

CPU 访问 CXL 内存的第一步，是让内存进入系统地址空间。管理员用 cxl-cli 在 root decoder 上创建 region——`cxl create-region -t ram` 或 `-t pmem`：前者对应易失内存（论文的 HDM-H 语义），后者对应持久内存。内核的 region 类型也只有这两种（`drivers/cxl/core/region.c:618-633` 的 `mode_show`，区分 `CXL_PARTMODE_RAM` 与 `CXL_PARTMODE_PMEM`）。region 提交时，内核从 CFMWS 预留的窗口分配 Host Physical Address（`alloc_hpa()`，`drivers/cxl/core/region.c:635`）——这正是论文 4.3 节"主机软件编程 HDM decoder"的内核实现。

随后发生关键一步：region 进入 commit 状态时，**`cxl_dax_region` 驱动自动为每个 region 创建 dev_dax 设备**（`drivers/dax/cxl.c:9-33`）：以 2MB（`PMD_SIZE`）对齐、`memmap_on_memory` 默认开启，暴露为 `/dev/daxX.Y`。这意味着 devdax 不是与 ram/pmem 并列的第三种 region 类型，而是所有 region 的"出厂形态"——内核先交出一把直接映射的钥匙，而非自行决定内存归属。

### 4.2 形态一：System RAM——转交给内核页分配器

若想让 CXL 内存"像普通内存一样"被系统使用，执行 `daxctl reconfigure-device --mode=system-ram dax0.0`（内核动作是把该 region 的 dev_dax 绑定到 `dax_kmem` 驱动；默认将内存块 online 为 movable，`--no-movable` 上 ZONE_NORMAL，`--no-online` + `daxctl online-memory` 可手动控制 online 时机；sysfs 等价路径是 `echo dax0.0 > /sys/bus/dax/drivers/device_dax/unbind` 后 `echo dax0.0 > /sys/bus/dax/drivers/kmem/bind`），CXL 内存即进入内存热插拔框架，成为页分配器管理的普通内存。转换由 `dev_dax_kmem_probe()` 完成（`drivers/dax/kmem.c:68`），其流程忠实体现了前文协议章的全部概念：

- **NUMA 归属**：取 `dev_dax->target_node`（源自 SRAT PXM/CEDT 映射），无效节点直接拒绝（`kmem.c:85-90`）——保证慢内存不会被混进快节点，这是论文 CDAT 属性的落地；
- **内存层级（memory tier）**：`mt_calc_adistance()`（`kmem.c:92`）根据 CDAT/HMAT 距离计算该内存所属的层级，供内核做层级感知的内存分配；
- **资源标记**：`res->flags = IORESOURCE_SYSTEM_RAM`（`kmem.c:167`）后调用 `add_memory_driver_managed()`（`kmem.c:177`）——"driver managed"意味着这批内存**明确排除在 kexec 内核之外**（memory-hotplug.rst 解释：防止 CXL 设备在系统重启时被 reset 导致新内核被覆盖）。

在线化后的行为由三处配置决定（优先级递减）：`CONFIG_MHP_DEFAULT_ONLINE_TYPE` 编译项 → `memhp_default_state` 内核参数 → `/sys/devices/system/memory/auto_online_blocks`，决定内存块进入 `ZONE_NORMAL`（几乎任何分配可用）还是 `ZONE_MOVABLE`（仅可迁移分配，保住未来整区热移除的能力）。`struct folio` 描述符的来源由 `memmap_on_memory` 控制：从本块内存中划出，或从执行热插拔 CPU 的本地 `ZONE_NORMAL` 分配——对高延迟的 CXL 内存池，若 folio 落在 CXL 内存上且被争用，会带来显著性能损失。

**CPU 访问方式**：完全透明。普通 load/store 指令，经 CPU 缓存层次，协议层一致性（HDM-H）由主机保证——与本地 DDR 的唯一差异是 NUMA 距离与带宽/延迟特征，应用程序甚至无需感知它的存在。这是三种形态中唯一"不需要任何程序改造"的形态，也是 Tiered Memory、内存池化的默认选择。

### 4.3 形态二：devdax——留在用户态，mmap 直访

若创建 region 后不做 online，`/dev/daxN.Y` 就保持设备形态：用户程序 `mmap()` 该设备（offset 需按 2MB 对齐），把 CXL 物理页直接映射进进程页表。内核路径是 `dax_mmap()`（`drivers/dax/device.c:288-305`），page fault 时建立到 CXL 内存的映射（`ZONE_DEVICE` 直映射，vmemmap 可由 `vmemmap_shift` 配置为 1GB 粒度以节省描述符开销）。

**与 System RAM 形态的本质差异不在 CPU 访问路径**——mmap 之后同样是 load/store、同样可缓存、同样一致性——而在于**分配与回收由谁管理**：System RAM 形态下内核页分配器掌握所有权（malloc/free 背后是它），devdax 形态下用户态代码自己决定这块内存怎么分、给谁用。它绕过了页缓存、页分配器与文件语义，适合"自定义内存池""固定大小内存预留"这类需要精确控制的使用场景——把 CXL 内存切给特定应用而非全系统。

### 4.4 形态三：持久内存栈——pmem 的块设备与文件语义

`-t pmem` 的 region 走另一条独立路径：`cxl_pmem_region` 驱动（`drivers/cxl/pmem.c:375`）经 nvdimm 总线把 region 暴露为持久内存，最终由 pmem 驱动呈现为块设备（`pmemN`）。在块设备之上有三种用途：

| 用途                  | 访问方式                                          | 适用场景                                                                                                                                |
| --------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 普通块设备            | 块 I/O（经页缓存）                                | 传统文件系统，不利用 DAX 能力                                                                                                           |
| fsdax（文件系统 DAX） | 页错误时文件页**直接映射** CXL 物理页，绕过页缓存 | 持久内存的典型用法：ext2/ext4/xfs/virtiofs/erofs 五个文件系统支持 `-o dax`（Documentation/filesystems/dax.rst），映射后 load/store 直访 |
| 持久 devdax           | mmap 直访                                         | 持久内存的裸设备管理（同 4.3，但数据掉电不丢）                                                                                          |

fsdax 的语义与 devdax 的差异在于接口：devdax 是"一块内存"（整个 region 一个文件），fsdax 是"一个文件系统"（文件粒度、权限、共享、快照语义），文件页缺失时直接把 CXL 物理页映射进页表，省去页缓存这层拷贝——这正是 DAX（Direct Access）的名字由来。

### 4.5 决策框架：三种形态怎么选

| 维度     | System RAM（`daxctl reconfigure-device`） | devdax（不 reconfigure）                | pmem + fsdax                        |
| -------- | ----------------------------------------- | --------------------------------------- | ----------------------------------- |
| 所有权   | 内核页分配器                              | 用户态代码                              | 文件系统                            |
| 程序改造 | 无                                        | mmap 编程                               | 挂载/DAX 感知                       |
| CPU 访问 | load/store（可缓存、一致）                | load/store（mmap 后同左）               | load/store 或块 I/O                 |
| 持久性   | 易失                                      | 易失（ram region）或持久（pmem region） | 持久                                |
| 热移除   | 可（ZONE_MOVABLE 时）                     | 直接（未使用中）                        | 先卸载                              |
| 典型场景 | Tiered Memory 冷层、内存池化、普通扩容    | 用户态内存池、精确资源切分              | 持久内存文件系统、数据库 DAX 表空间 |

选择逻辑遵循一个简单原则：**谁应该管理这块内存？** 想让内核全权管理 → System RAM；想自己精确控制 → devdax；需要持久化与文件语义 → pmem。注意三种形态的底层都走同一条 CXL.mem 一致性路径，形态差异是纯软件层的所有权划分——这是 CXL 设计"设备简单、主机软件灵活"哲学的最终体现。

### 4.6 与论文的衔接：池化愿景与内核现状

论文第 7 节的池化、共享与可组合愿景，在内核中的落地进度是分层的：单机形态（本章三种）已完全成熟；**跨主机形态仍是已知缺口**——Linux CXL 子系统的成熟度地图（Documentation/driver-api/cxl/maturity-map.rst）中，"Fabrics / G-FAM"与多主机硬件一致性共享内存的评分均为 [0]，而设备池化相关能力中，LD 热插拔（经 PCIe hotplug）为 [1] 初始可用，动态容量设备（DCD）为 [0]（`maturity-map.rst:84,147-157`）。这正好呼应论文自己的判断：协议就绪与部署成熟之间，隔着系统软件的长周期演进。配套工具链同步演进：kmem 转换目前由 `daxctl reconfigure-device` 承担，2026-01 的内核 patch 系列（`cxl_sysram_region`/`cxl_dax_kmem_region` 两级显式绑定 + `online_type` sysfs 属性）正在把在线化策略显式化，预计随新内核落地。

---

## 五、权衡：设计取舍与部署边界

CXL 的每一层设计选择都伴随明确代价。论文对"代价"的交代堪称诚实，这些权衡正是架构评审最值得吸收的部分。

**非对称一致性：设备简单，主机买单。** 主机承担全部对等缓存跟踪、snoop 编排与 DTLB 失效管理，设备只实现简化 MESI。换来的是设备端实现门槛大幅降低、生态繁荣；代价是主机的 snoop filter 规模成为 CXL.cache 设备数量的硬约束，且跨主机的一致性要靠内存设备目录（CXL 3.0）继续把成本外移——一致性的负担总是落在"离内存最近、最集中的那一个点"上。

**CXL vs DDR：带宽与空闲延迟的置换。** CXL 每引脚带宽是 DDR 的 8 倍，支持更长布线（散热/供电更灵活）与异构介质；但**空闲延迟约为 DDR 的两倍**。关键洞察在负载延迟：CXL 的带宽优势使其在系统繁忙时可能反超 DDR——延迟的可感知性取决于排队，而带宽决定排队。这解释了 Tiered Memory（CXL 内存做冷层）为何成为主流部署形态：不是"CXL 替代 DDR"，而是"CXL 扩容量、DDR/封装内存保低延迟"。论文据此推断：最终 CXL 会成为 CPU/加速器唯一的**外部**内存挂载点——注意"外部"二字，片内与封装内内存（含 UCIe 芯片粒）继续承担延迟敏感部分。

**机架级而非数据中心级。** CXL 延迟比以太网/InfiniBand 低一个数量级，但需要专用线缆、retimer、严格长度约束，成本与灵活性都劣于现有网络。论文引用的财务模型显示 **sub-rack（机架内）是 TCO 甜点**；标准化的推进可以让 CXL 扩展到 cluster/pod 级，但**不可能取代以太网成为数据中心级网络**。CXL 与以太网的关系是"互连分工"而非替代：细粒度、一致性、load-store 语义走 CXL；粗粒度、任意规模、容错走网络。

**可组合 ≠ 单一巨系统。** CXL 3.0 的 4096 端点共享内存描绘了一幅诱人的图景，但论文明确泼了一盆冷水：工作负载仍会偏好局部性、尽量少跨 CXL 链路——因为要最小化一致性流量与**故障爆炸半径**。可组合系统的价值是多路复用与按需装配，而不是把机架当一台机器。这给系统软件提出全新课题：远程主机引发内存压力时的 QoS 保障、内存织网上的拥塞与故障遏制，目前 CXL 的 QoS 只覆盖 CXL.mem 而不覆盖织网拥塞，且不支持事务级动态多路径——均被列为规范演进方向。

**工程层面的微观权衡**同样值得记取：CRC 先行 vs FEC 的 2ns 延迟博弈；LO Flit 的"半 Flit 独立 CRC"换取的无错路径低延迟 vs 出错路径的全量累积；68B Flit 模式的写带宽折损（每 cache line 写要 3 个协议头）vs 256B Flit 的带宽恢复。这些数字说明：互联协议的性能天花板不是单点工程，而是负载模式、Flit 布局、调度算法三层交互的结果。

---

## 六、启示：CXL 对计算基础设施的影响与未来方向

论文最后给出四个影响判断，其中两条与 AI 基础设施直接相关，值得单独强调。

**其一，内存带宽与容量扩展是当下最确定的收益。** GPU 服务器的内存墙（HBM 容量有限、DDR 带宽见顶）在 CPU 侧同样成立：Type 3 内存扩展器以远低于 DDR 通道的引脚/成本代价扩展容量，比远端 RDMA 内存池的延迟低一个数量级、又比本地 DRAM 廉价；Tiered Memory 软件栈（论文引用的 TPP、Pond 等）已经成熟到可部署。

**其二，CXL 是"机架即系统"叙事的基础设施前提。** 论文第 3.3 节描绘的多主机共享一致内存、设备直连、load-store 消息传递，正是 2023 年以来"机架级单域"趋势（如 GPU 机架的 NVL72 类比）在通用互连层面的展开：CXL 3.0 让"多个 CPU 主机 + 共享内存 + 直连加速器"可以在一个交换织网内以亚微秒一致性协作。对多机推理/训练中的细粒度同步、内存池化，CXL 提供了网络之外的第二个工具。

**其三，池化与可组合是云厂商的长期红利。** Azure（Berger 本人）与各大云厂商的 stranding 数据是 CXL 池化最有力的论据：从按峰值配置改为按均值配置，节省的是真金白银的内存成本。Berger 团队 2023 年的设计权衡研究（Pond，论文引文 [18]）表明池化的收益在超配率高的场景最显著。

**未来方向的清单**同样有参考价值：内存控制器独立演进（自适应刷新、可靠性改进）；CPU 架构适配（更高内存延迟下的预取与缓冲策略）；系统软件（远程内存压力下的 QoS、织网拥塞控制、面向可组合性的调度）；工程层面（延迟进一步压低、错误遏制与爆炸半径管理、动态多路径）；以及通过 UCIe retimer 引入共封装光学，把 CXL 的物理距离延伸到 pod 级。

**从 2026 年的时点回看这篇 2023 年的综述**：CXL 2.0 的 Type 3 设备已进入量产服务器，Tiered Memory 成为默认部署形态；CXL 3.0 的规格后续又迭代出 3.1/3.2，PCIe 7.0 的 128 GT/s 已在路上。论文的技术分析没有过时——它的价值恰恰在于把三代协议的动机、机制与代价固化成了可检索的知识，读者不需要在规范文档中自行拼图。对想判断"CXL 在我的系统里应该扮演什么角色"的工程师，这篇论文是比任何厂商白皮书都权威的起点。

---

## 附：核心概念速查

| 术语                         | 含义                                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------- |
| CXL.io / CXL.cache / CXL.mem | CXL 三大协议：非一致控制面 / 设备缓存主机内存 / 主机可缓存访问设备内存                      |
| HDM / HDM-H / HDM-D / HDM-DB | 主机可管理设备内存；主机独占一致性 / 设备管理一致性 / 设备管理一致性 + Back-Invalidate      |
| Flit（68B / 256B / 128B LO） | CXL 传输单元；LO 为延迟优化型（子 Flit + CRC 先行）                                         |
| MESI / GO                    | 缓存一致性协议状态；全局观察点（一致性提交点）                                              |
| Bias Flip                    | Type 2 设备通过 CXL.cache 请求翻转主机缓存状态的所有权                                      |
| MLD / LD                     | 多逻辑设备：一个 CXL.mem 设备切成可独立归属的逻辑设备（CXL 2.0 上限 16 个、3.0 上限 32 个） |
| FM / CCI / MCTP              | Fabric Manager（池分配决策者）/ 组件命令接口 / 带外管理传输协议                             |
| CDAT                         | 一致设备属性表：运行时热插内存的拓扑、带宽、延迟描述                                        |
| UIO                          | 无序 I/O：VC1-7 上无排序要求、源端负责排序语义                                              |
| BI（Back-Invalidate）        | CXL.mem 新增的设备→主机缓存作废通道，支撑 P2P、snoop filter、多主机共享                     |
| PBR / PID / FAST             | 端口基路由 / 12-bit 端点标识 / 织网地址段查找表                                             |
| GFD / GFAM                   | 全局织网附加内存设备：直接参与 PBR、最多 4096 节点共享                                      |
| devdax / kmem                | DAX 设备形态（/dev/daxN.Y，用户态 mmap 直访）/ kmem 转换（转为 System RAM 交给页分配器）    |
| HPA / HDM decoder            | 主机物理地址；region 在 CFMWS 预留窗口中的地址分配与解码编程                                |

## 参考

- 论文原文：[arXiv:2306.11227](https://arxiv.org/abs/2306.11227)
- 本文解读的 68B/256B Flit 布局、延迟微架构与带宽表格均出自论文第 3/6 节，文中数据与论文表 1-5 及图 19 对应
- 第四章源码引用基于本地内核 6.19-rc1 源码（`drivers/cxl/`、`drivers/dax/`），公开文档见 [Linux CXL 子系统文档](https://docs.kernel.org/driver-api/cxl/)（overview / memory-hotplug / dax-driver / maturity-map）
