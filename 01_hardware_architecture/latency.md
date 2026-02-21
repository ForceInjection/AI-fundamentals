# AI 基础设施延迟金字塔

本文旨在为 AI 工程师提供一份直观的系统延迟参考指南。通过建立 **"延迟金字塔" (Latency Pyramid)** 模型，将从芯片内部到跨数据中心的各级延迟数据可视化，帮助开发者建立对系统性能数量级的第一性认知。

> ⚠️ **基准说明**：数据基于 NVIDIA H100/A100 + NVLink/NVSwitch + IB NDR 网络环境下的典型热路径估算。

---

## 1. 延迟金字塔总览

为了更直观地理解**数据移动的代价**，我们将时间延迟映射为**"人类距离尺度" (Human Distance Scale)**。

> **核心隐喻**：
> 在 AI 基础设施中，我们将基准定义为 **1 GPU 周期 (~0.5 ns)**。
> 假设 **1 GPU 周期 (0.5 ns)** 相当于 **迈出 1 步 (0.5 米)**。
>
> 👉 **速算口诀**: `1 ns ≈ 1 米`

```text
# -----------------------------------------------------------------------------
# Level 1 — 芯片内部 (Chip Internal)
# -----------------------------------------------------------------------------
事件 (Event)                        延迟 (Latency)        人类距离尺度 (Distance Scale)
-----------------------------------------------------------------------------
Register Access                 ~0.5 ns               0.5 m   (伸手拿纸)
L1 Cache                        15-30 ns              15-30 m (穿过大办公室)
Shared Memory (Bank Conflicts)  20-30 ns              20-30 m (去隔壁会议室)
GPU L2 Cache                    100-120 ns            100-120 m (下楼去便利店)

# -----------------------------------------------------------------------------
# Level 2 — 显存与互联 (Memory & Interconnect)
# -----------------------------------------------------------------------------
NVLink hop (Direct)             80-120 ns             80-120 m (去隔壁楼栋)
HBM3 Access                     250-350 ns            250-350 m (绕小区半圈)
NVSwitch traversal              150-250 ns            150-250 m (去小区门口)
PCIe Gen5 Transaction           400-600 ns            400-600 m (步行去地铁站)

# -----------------------------------------------------------------------------
# Level 3 — 通信与网络 (Communication & Network)
# -----------------------------------------------------------------------------
RDMA 1KB (One-way)              1.5-2 us              1.5-2 km (共享单车去地铁站)
Host-Device RTT (PCIe)          1-3 us                1-3 km   (跑步去邻村)
Kernel Launch (CPU->GPU)        5-15 us               5-15 km  (10公里长跑)
NCCL AllReduce (NVLink)         20-200 us             20-200 km (开车跨城上班)
TCP RTT (Intra/Cross Rack)      20-150 us             20-150 km (北京到天津)
NCCL AllReduce (Cross Node)     200 us - 5 ms         200 km+   (高铁跨省)

# -----------------------------------------------------------------------------
# Level 4 — 存储与 I/O (Storage & I/O)
# -----------------------------------------------------------------------------
NVMe Random Read (4KB)          ~100 us               100 km   (北京到保定)
NVMe Seq Read (1MB)             100-300 us            100-300 km (北京到石家庄)
HDD Seek                        ~10 ms                10,000 km (坐飞机去美国)
Object Storage First Byte       30-100 ms             30,000-100,000 km (绕地球 1-2.5 圈)
Region-to-Region RTT            80-150 ms             80,000-150,000 km (地月距离的 1/4)
```

![AI Latency Pyramid Guide Overview](./The%20AI%20Latency%20Pyramid%20Guide%20Overview.png)

---

## 2. 核心层级详解

### 2.1 Level 1: 计算核心层

这是 AI 计算的微观世界，性能单位为 **纳秒 (ns)**。

- **Register (~0.5 ns)**: 数据的最终目的地，计算发生的场所。
- **L1 / Shared Memory (15-30 ns)**:
  - **L1 Cache (15-30 ns)**: 约 30-60 个 GPU 周期，远高于 CPU L1 的 4-5 个周期。
  - **Shared Memory (20-30 ns)**: 开发者可控的暂存区，若发生 Bank Conflict，延迟会显著增加。
- **L2 Cache (100-120 ns)**: 所有 SM 共享的片上缓存 (50MB)。相比 A100 (40MB)，H100 的 L2 更大，延迟约 200+ 周期。

> **关键认知**: 即使是访问 L2 Cache (100ns+)，相比寄存器 (0.5ns) 也慢了 **200倍**。

### 2.2 Level 2: 显存与互联层

这是 "Memory Wall" (内存墙) 的发生地，性能单位为 **百纳秒 (100s ns)**。

- **HBM3 (250-350 ns)**: 高带宽内存。虽然带宽极大 (3TB/s)，但延迟并未显著降低 (400-600+ 周期)。
- **NVLink**:
  - **Direct Hop (80-120 ns)**: 点对点直连延迟。
  - **NVSwitch Path (150-250 ns)**: 跨 NVSwitch 的通信延迟。
- **PCIe Gen5 (400 ns - 3 us)**:
  - **Transaction (400-600 ns)**: 纯硬件总线传输延迟。
  - **Host-Device RTT (1-3 us)**: 包含驱动和 CPU 交互的端到端往返延迟。

> **关键认知**: 通过 NVLink 访问邻居 GPU 的 L2 Cache (Total ~200-300 ns) 甚至比访问本地 HBM (250-350 ns) **更快**。这意味着利用"邻居缓存" (Cooperative Caching) 在物理上是可行的。

### 2.3 Level 3: 分布式通信层

这是分布式训练 Scaling 的关键，性能单位为 **微秒 (us)**。

- **RDMA (1.5-2 us)**: NDR InfiniBand 网络的典型单向延迟。
- **Kernel Launch (5-15 us)**: CPU 下发指令到 GPU 开始执行的开销。使用 CUDA Graphs 可将其降低到 3-5 us。
- **NCCL AllReduce (NVLink)**:
  - **Small Tensor (20-50 us)**: 延迟敏感型的小包通信。
  - **Typical Payload (50-200 us)**: 正常训练负载下的通信耗时。
- **TCP RTT (20-150 us)**: 同机架 (20-50 us) vs 跨机架 (50-150 us)。

> **关键认知**: 跨机通信 (Network) 相比机内通信 (NVLink) 有 **10-100 倍** 的延迟差距，这是设计分布式算法时必须考虑的物理约束。

### 2.4 Level 4: 存储与 I/O 层

这是数据持久化层，性能单位为 **毫秒 (ms)**。

- **NVMe SSD**:
  - **Random Read (80-120 us)**: 4KB 随机读取。
  - **Seq Read 1MB (100-300 us)**: 大块顺序读取效率极高，远快于 1ms。
- **Object Storage (30-100 ms)**: 首字节延迟 (TTFB) 波动较大，受网络和云服务负载影响。

> **关键认知**: I/O 延迟是计算延迟的 **10,000 倍** 以上。任何同步 I/O 操作都会导致 GPU 算力的巨大浪费。

---

## 3. 常见操作延迟速查表

| 操作 (Operation)  | 典型延迟   | 数量级 | 备注                          |
| :---------------- | :--------- | :----- | :---------------------------- |
| **Math**          | < 1 ns     | Nano   | FMA, Tensor Core op           |
| **SRAM Access**   | 15-30 ns   | Nano   | L1 (~30 cycles), Shared       |
| **DRAM Access**   | 250-350 ns | Micro  | HBM3 (500 cycles+), DDR5      |
| **GPU-GPU Raw**   | 80-250 ns  | Micro  | NVLink (Direct vs Switch)     |
| **Host-Device**   | 1-3 us     | Micro  | PCIe RTT                      |
| **Net-Device**    | 2-4 us     | Micro  | GPUDirect RDMA                |
| **Kernel Launch** | 5-15 us    | Micro  | CUDA API Overhead             |
| **Small Kernel**  | ~5 us      | Micro  | Element-wise op               |
| **Attention**     | 50-300 us  | Micro  | FlashAttn (seq len dependent) |
| **AllReduce**     | 50us - 5ms | Milli  | Single Node vs Multi Node     |
| **Disk I/O**      | 0.1-10 ms  | Milli  | NVMe vs HDD                   |

---

## 4. 延迟估算公式

在没有实测数据时，可用以下简易模型进行估算：

1. **数据传输 (Data Transfer)**:
   `Time = Latency + (Size / Bandwidth)`
   - _小数据看 Latency，大数据看 Bandwidth。_

2. **集合通信 (Ring AllReduce)**:
   `Time ≈ 2 * (N-1) / N * (Latency + Size/Bandwidth)`
   - _随着节点数 N 增加，带宽利用率趋近恒定，但 Latency 线性增长。_

3. **推理首字 (TTFT)**:
   `TTFT ≈ Launch_Overhead + HBM_Read + KV_Fetch + Network_RTT + Queue_Delay`
   - _除了计算和显存，KV Cache 获取和调度排队往往是长尾延迟的来源。_

---

## 5. 总结

**"延迟金字塔"** 揭示了 AI 系统设计的核心挑战：

1. **层级跨越代价巨大**: 每跨越一层，延迟增加约 **1-2 个数量级**。
2. **局部性 (Locality) 是王道**: 将计算留在 Level 1/2，避免频繁跌落到 Level 3/4。
3. **隐藏延迟 (Hiding)**: 高层级的延迟 (Level 3/4) 必须通过异步流水线被低层级的计算 (Level 1) 掩盖。

记住这张金字塔图，当你在写代码或设计架构时，时刻问自己：**"我现在的数据在第几层？"**
