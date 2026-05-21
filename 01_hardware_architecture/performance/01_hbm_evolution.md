# HBM 显存技术演进：从 HBM2 到 HBM3e

> HBM (High Bandwidth Memory) 是 AI 算力的"燃料管道"——GPU 算力每年翻倍，但如果显存带宽跟不上，Tensor Core 只能空转。本文覆盖 HBM 各代的技术参数、关键创新及 A100 (HBM2e) 与 RTX 5090 (GDDR7) 的实测对比。

---

## 1. 为什么 HBM 对 AI 至关重要

GPU 的计算吞吐和显存带宽必须匹配。以 A100 为例：

```text
A100 Tensor Core (BF16): 312 TFLOPS
A100 HBM2e 带宽:        2039 GB/s
每 FLOP 可用带宽:        2039 GB/s ÷ 312 TFLOPS ≈ 6.5 bytes/FLOP
```

对于矩阵乘法等计算密集型算子，这个比例是够的。但 Attention 机制每次计算都要读取整个 KV Cache——如果序列长度 32K、hidden size 8192，KV Cache 约 2 GB，在 A100 上每次 Attention 计算仅读取 KV Cache 就需要 ~1 ms，远超计算时间。**显存带宽是 LLM 推理的真正瓶颈。**

---

## 2. HBM vs GDDR：两种显存路线

|          | HBM (High Bandwidth Memory)       | GDDR (Graphics Double Data Rate) |
| -------- | --------------------------------- | -------------------------------- |
| 设计理念 | **位宽优先**：1024-bit 起，堆栈式 | **频率优先**：32-bit 接口，高频  |
| 位宽     | 1024-6144 bits (多堆栈)           | 32-512 bits (多芯片)             |
| 频率     | 1.2-3.2 GHz                       | 14-28 GHz                        |
| 带宽     | 410 GB/s → 4.8 TB/s (HBM3e)       | 1792 GB/s (RTX 5090 GDDR7)       |
| 功耗效率 | 高 (~3.5 pJ/bit)                  | 较低 (~7 pJ/bit)                 |
| 封装     | 3D 堆叠 + Si Interposer           | PCB 上平面排布                   |
| 成本     | 极高                              | 低                               |
| 典型产品 | A100, H100, B200                  | RTX 4090, RTX 5090               |

> 消费级 GPU 使用 GDDR 是因为成本敏感。数据中心 GPU 使用 HBM 是因为带宽需求压倒成本考量。

---

## 3. HBM 代际演进

| 代际      | 带宽/堆栈    | 最大堆栈数 | 最大位宽     | 最大带宽      | 代表 GPU                      |
| --------- | ------------ | ---------- | ------------ | ------------- | ----------------------------- |
| HBM2      | 307 GB/s     | 4          | 4096-bit     | 900 GB/s      | V100 (32 GB)                  |
| **HBM2e** | **410 GB/s** | **5**      | **5120-bit** | **2039 GB/s** | **A100 (80 GB)**              |
| HBM3      | 665 GB/s     | 6          | 6144-bit     | 3.35 TB/s     | H100 (80 GB)                  |
| HBM3e     | 1.15 TB/s    | 8          | 8192-bit     | 4.8 TB/s      | H200 (141 GB) / B200 (192 GB) |

**公式**：`总带宽 = 堆栈数 × 每堆栈带宽 × 2 (DDR)`

以 A100 为例：

```text
5 stacks × 1024 bits × 1.593 GHz × 2 (DDR) ÷ 8 = 2039 GB/s
```

### 3.1 HBM2e：A100 的精妙之处

A100 的 NVLink 3.0 设计逻辑与 HBM2e 一脉相承：**减少每条链路的信号线数量，翻倍增加链路数**。

|            | V100 HBM2 | A100 HBM2e    | 变化  |
| ---------- | --------- | ------------- | ----- |
| 堆栈数     | 4         | **5**         | +25%  |
| 每堆栈位宽 | 1024-bit  | 1024-bit      | 不变  |
| 总位宽     | 4096-bit  | **5120-bit**  | +25%  |
| 频率       | 1.75 GHz  | **1.59 GHz**  | -9%   |
| 总带宽     | 900 GB/s  | **2039 GB/s** | +2.3× |

频率降低但位宽大增——这就是 HBM 的哲学。更宽的位宽意味着可以同时处理更多并发内存请求，减少排队延迟。

### 3.2 A100 vs RTX 5090：HBM2e vs GDDR7 实测

| 指标              | A100 HBM2e           | RTX 5090 GDDR7   | 说明                                |
| ----------------- | -------------------- | ---------------- | ----------------------------------- |
| 位宽              | **5120-bit**         | 512-bit          | A100 宽 10 倍                       |
| 频率              | 1593 MHz             | **14001 MHz**    | RTX 5090 高 8.8 倍                  |
| 理论带宽          | **2039 GB/s**        | 1792 GB/s        | A100 高 14%                         |
| 实测 D2D (4MB)    | **~1188 GB/s** (58%) | ~1341 GB/s (75%) | RTX 5090 L2 (96MB) 更大，4MB 全命中 |
| 实测 D2D (大矩阵) | 预计 ~1500+          | ~762 GB/s        | A100 宽位宽优势在大矩阵体现         |

> 数据来源：[`03_hbm_bandwidth_test.md`](../../02_gpu_programming/04_profiling/03_hbm_bandwidth_test.md) + transpose 官方 sample 实测。

---

## 4. HBM 的 3D 封装技术

HBM 的性能来自于独特的物理结构：

```text
          HBM 堆栈 (侧视图)
          ┌──────────┐
          │ DRAM Die │ ← 第 8 层
          │ DRAM Die │ ← 第 7 层
          │ DRAM Die │ ← 第 6 层
          │ DRAM Die │ ← 第 5 层
          │ DRAM Die │ ← 第 4 层
          │ DRAM Die │ ← 第 3 层
          │ DRAM Die │ ← 第 2 层
          │ DRAM Die │ ← 第 1 层 (顶层)
          │ Logic Die│ ← 控制器 + PHY
    ┌─────┴──────────┴─────┐
    │   Silicon Interposer │ ← 硅中介层
    │   (连接 GPU 与 HBM)   │
    └──────────────────────┘
    ┌──────────────────────┐
    │    GPU Die (GA100)   │
    └──────────────────────┘
```

- **3D 堆叠**：DRAM die 垂直堆叠，通过 **TSV (Through-Silicon Via)** 连接
- **Silicon Interposer**：微凸点（micro-bump）连接的硅中介层，GPU 和 HBM 堆栈共享
- **物理距离**：GPU 到 HBM 的信号路径只有 ~1 mm，而 GDDR 走 PCB 需要 ~20-30 mm——这就是为什么 HBM 能用更低频率实现更高带宽

---

## 5. L2 Cache：HBM 的最后一级加速器

从 A100 开始，NVIDIA 大幅增加了 L2 Cache：

| GPU      | L2 Cache  | HBM 带宽  | L2/HBM 比 |
| -------- | --------- | --------- | --------- |
| V100     | 6 MB      | 900 GB/s  | 1:150     |
| A100     | **40 MB** | 2039 GB/s | 1:51      |
| H100     | 50 MB     | 3350 GB/s | 1:67      |
| B200     | 96 MB     | 4800 GB/s | 1:50      |
| RTX 5090 | **96 MB** | 1792 GB/s | 1:19      |

**L2 越大，越能掩盖 HBM 延迟**。40 MB L2 在 A100 上足以容纳 4M 元素的 fp32 矩阵——这意味着中等大小的矩阵操作可以完全在 cache 内完成，带宽利用率可达 58% 以上。

---

## 6. 编程启示

- **尽量把数据留在显存**：A100 HBM2e 内部带宽 (2039 GB/s) vs PCIe Gen4 (~28 GB/s) = **73倍**
- **利用 L2 cache**：中等矩阵 (≤ 数 MB) 操作受益于 L2 命中，实际带宽可达到峰值 50% 以上
- **大矩阵操作**：A100 的宽位宽 (5120-bit) 比 RTX 5090 (512-bit) 更适合大数据流
- **GDDR vs HBM**：RTX 5090 的 GDDR7 在 L2 命中时很快（~1341 GB/s），但一旦需要 DRAM 访问，窄位宽立刻成为瓶颈

## 参考

- [HBM 显存带宽测试](../../02_gpu_programming/04_profiling/03_hbm_bandwidth_test.md) — A100 vs RTX 5090 实测对比
- [NVIDIA A100 架构详解](../nvidia/understand_gpu_architecture/07_a100_architecture.md) — HBM2e 在 A100 上的完整规格
- [PCIe & NVLink 带宽速查表](05_pcie_nvlink_speed_reference.md) — 带宽全景对比
- [AI 基础设施延迟金字塔](ai_latency_pyramid.md) — 各级延迟基准
- [JEDEC HBM Specs](https://www.jedec.org/) — HBM 标准制定者
