# CUDA 性能调优方法论——从 Roofline Model 到 ncu 诊断

你已经读完了 01 到 16——知道如何分配 Shared Memory（[#13](13_shared_memory_bank_conflict.md)），怎样用 Warp Shuffle 替代 `__syncthreads()`（[#14](14_warp_level_programming.md)），何时上 `cp.async` + pipeline（[#16](16_async_copy_pipeline.md)），以及如何用 NCCL 做多 GPU 通信（[#15](15_multi_gpu_programming.md)）。

但现在你面对一个新的 kernel——它慢了，你不知道为什么。你的第一反应可能是："加 Shared Memory？""改 block size？""上 Warp Shuffle？"——但这些都是**手段**，不是**诊断**。Roofline Model 给你的是"天花板在哪里"——在动任何一行代码之前，先知道最多能快多少、瓶颈在算力还是带宽。

本文提供一个分级诊断框架。第一层：Roofline Model（确定天花板）。第二层：ncu 关键指标（定位瓶颈）。第三层：逐级优化 checklist（从 occupancy 到 arithmetic intensity，6 级递进）。最后用一个朴素 kernel 从头到尾走一遍完整调优。

> **前置要求**：你已理解此系列的前置概念——Shared Memory、Warp、Stream、Global Memory 延迟。建议阅读 [#10](10_reduction.md)（Reduction 逐级优化展示）、[#13](13_shared_memory_bank_conflict.md)（Bank Conflict 诊断 SOP）。
>
> **验证代码**：本文为方法论文章，不依赖单一配套 benchmark。建议在任意 GPU 上对现有 kernel 按照 §6.1 的分级诊断 SOP 实操一遍。ncu 命令参考：[ncu 诊断示例](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。

---

## 1. Roofline Model——"你的天花板在哪里"

### 1.1 两层天花板

GPU kernel 的性能最终由两个硬上限约束：

```text
实际 TFLOPS = min(峰值算力, 峰值带宽 × 算术强度)

            ┌────────────────────────────┐
            │   峰值算力 (Compute Roof)   │ ← SM 全部跑满时的 TFLOPS
            │   A100 FP32: 19.5 TFLOPS   │
            └──────────┬─────────────────┘
                       │
            ┌──────────┴─────────────────┐
            │   峰值带宽 (Memory Roof)    │ ← HBM 带宽能喂饱多少 FLOPS
            │   A100 HBM: 2.0 TB/s       │
            └────────────────────────────┘
```

**算术强度**（Arithmetic Intensity）= FLOP / Byte——每从 HBM 读 1 字节数据，能做多少次浮点运算。它是连接算力和带宽两个维度的桥梁。

CPU 类比：Roofline Model 最早由 UC Berkeley 的 Williams 等人提出，是 HPC 社区的标准分析工具。GPU 版本的区别在于：GPU 的 Compute Roof 远比 CPU 高，但 Memory Roof 相对窄——大多数 kernel 被带宽限制，而非算力。

### 1.2 你的 kernel 站在哪里

```text
算数强度 (FLOP/Byte)

  0.1         1          10        100       1000      10000
   │          │          │          │          │          │
   ├──────────┼──────────┼──────────┼──────────┼──────────┤
   │          │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────┐
│  Memory-Bound          │       Compute-Bound                │
│  (瓶颈在带宽)           │       (瓶颈在算力)                  │
│  Element-wise ops      │       GEMM, Conv                   │
│  Reduce, Scan          │       Tensor Core ops              │
│  Attention (naive)     │                                    │
└─────────────────────────────────────────────────────────────┘

具体数值（A100 FP32）:
  峰值算力: 19.5 TFLOPS
  峰值 HBM 带宽: 2.0 TB/s
  拐点 = 19.5 TFLOPS / 2.0 TB/s ≈ 9.75 FLOP/Byte

  Kernel 算术强度 < 10 FLOP/Byte → Memory-Bound（优先优化带宽）
  Kernel 算术强度 > 10 FLOP/Byte → Compute-Bound（优先优化算力利用率）
```

### 1.3 快速估算算术强度

不需要精确计算——估算到数量级就够：

```text
典型 kernel 的算术强度 (FP32):

  Element-wise (A+B, ReLU):    1 FLOP / 12 bytes = 0.08   ← 极低，带宽瓶颈
  Reduction:                   1 FLOP / 4 bytes  = 0.25   ← 带宽瓶颈
  Stencil (3D 7-point):        7 FLOP / 4 bytes  = 1.75   ← 带宽瓶颈
  GEMM (N=128):                2N³ / (3N² × 4)   = 21.3   ← 算力瓶颈 (N=128)
  GEMM (N=4096):               2N³ / (3N² × 4)   = 682    ← 严重算力瓶颈
```

> N=128 时 GEMM 卡的算术强度 21.3 已超过 A100 拐点，但实际利用率受限于 tile 太小导致的 edge effect。这就是为什么 GEMM 是算力瓶颈，但小矩阵的实测 TFLOPS 仍然远低于峰值（见 [#11](11_tensor_core_gemm.md)）。

---

## 2. ncu 关键指标——"数据告诉你瓶颈在哪里"

Roofline 告诉你方向，ncu（Nsight Compute）告诉你具体问题。

### 2.1 三个必看指标

```bash
ncu --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    ./my_kernel
```

| 指标                                                 | 含义                              | 怎么解读                                  |
| ---------------------------------------------------- | --------------------------------- | ----------------------------------------- |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed`   | SM 算力利用率                     | > 80% → Compute-Bound                     |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽利用率                    | > 80% → Memory-Bound                      |
| `achieved_occupancy`                                 | 实际每 SM 驻留 warp 数 / 理论最大 | < 50% → Latency-Bound（优先提 occupancy） |

三者全低（< 60%）→ **Latency-Bound**——GPU 在等数据、等 barrier、等同步。

### 2.2 诊断决策树

```text
ncu 三个指标读出来：

  sm_throughput > 80%?
    ├─ 是 → Compute-Bound
    │   ├─ FP32 → 考虑 FP16/BF16
    │   ├─ 非 GEMM → 检查是否能重构为 GEMM
    │   └─ GEMM → 检查 tile 大小、Tensor Core 利用率
    │
    └─ 否 → dram_throughput > 80%?
        ├─ 是 → Memory-Bound
        │   ├─ Global Mem → coalesced access? ([#12](12_gpu_memory_management.md) 第 1 节)
        │   ├─ Shared Mem → bank conflict? ([#13](13_shared_memory_bank_conflict.md))
        │   └─ 数据复用 → 引入 Shared Memory tile ([#13](13_shared_memory_bank_conflict.md))
        │
        └─ 否 → 两者都低 → Latency-Bound
            ├─ occupancy < 50%? → 调整 block size / SMEM 用量
            ├─ 很多 __syncthreads? → 用 Warp Shuffle 替代 ([#14](14_warp_level_programming.md))
            ├─ Load-Compute 串行? → cp.async + pipeline ([#16](16_async_copy_pipeline.md))
            └─ 小 kernel launch 太多? → CUDA Graphs ([#09](09_cuda_graphs.md))
```

### 2.3 只看一个数字：算术强度的 ncu 版本

ncu 在 Memory Workload Analysis 中直接给出了每个 byte 的 FLOP：

```bash
ncu --section MemoryWorkloadAnalysis_Tables ./my_kernel
```

找 `dram__bytes_read` + `dram__bytes_write` 和 `flop_count_sp` 或 `flop_count_hp`：

```text
FLOP / (bytes_read + bytes_write) = 实测算术强度

对比 A100 拐点 9.75 FLOP/Byte:
  < 10 → 带宽瓶颈
  > 10 → 算力瓶颈
```

---

## 3. 逐级优化 Checklist——6 级递进

从"最粗"到"最细"——每一级的收益通常比下一级大一个数量级。**不要跳级**。

### Level 0: 确认 GPU 在做事

- 检查 `nvidia-smi` 中 GPU 利用率（`nvidia-smi dmon -s u`）
- CPU 侧：确认 kernel launch 时间 vs 数据准备时间（`perf top` 或 `nsys` timeline）
- 如果 GPU 大部分时间在空转：检查 data pipeline、stream 是否真正并发（[#07](07_cuda_streams_concurrency.md)）

### Level 1: Occupancy——"够不够 warp 切换"

Occupancy 决定了 GPU 在等待 HBM 时有足够 warp 可以切换。这是**一切优化的前提**——如果 occupancy < 50%，后面所有优化都可能被延迟隐藏不足所掩盖。

```bash
ncu --section Occupancy ./my_kernel
```

**常见问题和修复**：

| 问题                          | 症状                       | 修复                                      |
| ----------------------------- | -------------------------- | ----------------------------------------- |
| 每 block 用太多 Shared Memory | occupancy 被 SMEM 限制     | 减少 SMEM 用量或改用更小的 tile           |
| 每 thread 用太多寄存器        | occupancy 被 register 限制 | 加 `__launch_bounds__` hint、减少局部变量 |
| block 太大                    | 每 SM 放不下 2 个 block    | 减小 block size                           |

```cuda
// 示例：用 __launch_bounds__ 控制寄存器用量
__global__ void __launch_bounds__(256, 2)  // 256 threads, 至少 2 blocks/SM
my_kernel(...) { ... }
```

### Level 2: Global Memory Coalescing——"一次读 128 字节"

Warp 内 32 线程的全局内存访问如果能被合并为一次 128B transaction，延迟降低 32×。这是**单次改动收益最大**的优化。

```cuda
// 非 coalesced（Stride 访问——每线程一次 transaction）
float x = g_data[tid * N];  // stride = N → 32 次 4B transaction → 32× 延迟

// Coalesced（连续访问——32 线程共用一次 128B transaction）
float x = g_data[tid];      // 32 线程读 [0..31] → 1 次 128B transaction
```

详见 [#12 GPU 内存管理](12_gpu_memory_management.md) 第 1 节。

### Level 3: Shared Memory——"把复用的数据搬进来"

如果多个线程读同一个 Global Memory 地址，或同一线程多次读同一个位置——Shared Memory 使后续访问延迟从 400ns 降到 5ns。

```cuda
// 每次迭代从 Global Memory 读: 400ns × 100 iterations = 40μs
// 搬进 Shared Memory: 400ns (1 次 load) + 5ns × 99 = ~900ns
```

详见 [#13 Shared Memory 与 Bank Conflict](13_shared_memory_bank_conflict.md)、[#16 异步拷贝与 Pipeline](16_async_copy_pipeline.md)。

### Level 4: Bank Conflict——"Shared Memory 用对了吗"

用了 Shared Memory ≠ 没有 Bank Conflict。检查 stride 访问模式：

```text
Bank = (addr / 4) % 32
stride 与 32 互质 → 无冲突
stride 为 2 的幂次 → gcd(stride, 32) > 1 → d-way conflict
```

详见 [#13](13_shared_memory_bank_conflict.md)。

### Level 5: Latency Hiding——"计算掩盖等待"

- Level 5a: **Warp Shuffle** 替代 Warp 内 Shared Memory（[#14](14_warp_level_programming.md)）
- Level 5b: **`cp.async` + pipeline** 重叠 Load 和 Compute（[#16](16_async_copy_pipeline.md)）(SM80+)
- Level 5c: **CUDA Graphs** 消除多次 kernel launch 的 CPU→GPU overhead（[#09](09_cuda_graphs.md)）

### Level 6: Arithmetic Intensity——"少算、精算"

- FP32 → FP16/BF16（[#11](11_tensor_core_gemm.md)）：Tensor Core 的 2×-4× 加速
- GEMM 融合：conv → im2col → GEMM，避免中间多一轮读写
- Kernel 融合：element-wise ops 合并为一个 kernel

---

_注：虽然本节以 ncu 为主，但在实际工作流中，一个互补的做法是先用 `nsys` (Nsight Systems) 做宏观时间线分析——它可以快速定位是哪些 kernel 在消耗大部分时间、是否存在 CPU-GPU 之间的 Idle Gap。当需要深入单个 kernel 的微观瓶颈时，再用 ncu。本文聚焦微观优化，因此以 ncu 为主线。_

---

## 4. 案例：一个朴素 GEMM 的完整调优

以一个简单但完整的矩阵乘法 kernel 为例，展示 Level 0-6 的全流程。

### 4.1 Level 0 kernel：朴素实现

```cuda
__global__ void gemm_naive(float *C, const float *A, const float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];  // 每个元素从 HBM 读
        C[row * N + col] = sum;
    }
}
```

```bash
ncu --section SpeedOfLight py ./gemm_naive N=1024
```

假设 ncu 输出：

```text
sm_throughput:    ~5%     ← 严重低
dram_throughput:  ~15%    ← 也低
occupancy:        ~50%    ← 尚可
```

**诊断**：算术强度 = 2 FLOP / (4+4+4 bytes) = 0.17 FLOP/Byte（极低，因为完全无数据复用）→ Memory-Bound，但带宽也只有 15%——原因是 Global Memory 访问没有 coalesced？

**问题**：`B[k * N + col]`——col 是 `blockIdx.x * TILE + threadIdx.x`，所以同一 warp 内的线程的 col 是**相邻的**。`A[row * N + k]`——同一个 warp 内 row 相同，k 也相同，所以 32 个线程读**同一个地址**！这反而是 broadcast——但不是 coalesced（32 次访问同一地址，L1 命中但没有合并）。

### 4.2 Level 2-3 fix：Shared Memory + Coalescing

```cuda
#define TILE 32
__global__ void gemm_tiled(float *C, const float *A, const float *B, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N; t += TILE) {
        // Cooperative load（coalesced 到 Shared Memory）
        As[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}
```

ncu 输出预期变化：

```text
sm_throughput:   ~15%    ← +10%
dram_throughput: ~60%    ← +45%  (coalesced loads)
```

数据复用 → 算术强度提升 → sm_throughput 上升。但 dram_throughput 离 80% 还有距离——因为 `__syncthreads()` 的 barrier 在消耗时间。

### 4.3 Level 5 fix：Double Buffer + `__syncthreads` 优化

用两个 tile buffer（[#16](16_async_copy_pipeline.md)），Load[N+1] 与 Compute[N] 重叠：

- 预加载第一个 tile
- 循环：Load next tile → 同时 Compute current tile
- 用 double buffer 交替

ncu 输出预期变化：

```text
dram_throughput: ~80%    ← +20% (Load 和 Compute 重叠)
sm_throughput:   ~25%    ← +10%
```

### 4.4 Level 6 fix：FP16 + Tensor Core

将数据转为 FP16，用 WMMA API 调用 Tensor Core（[#11](11_tensor_core_gemm.md)）：

```cuda
// ... WMMA fragment 声明 + mma_sync ...
```

ncu 输出预期变化：

```text
sm_throughput: ~55%    ← 质的飞跃
```

### 4.5 全流程加速总结

```text
Kernel Version          | 相对加速 | 瓶颈类型        | 关键优化
------------------------|---------|----------------|------------------
Level 0 (naive)         | 1.0×    | Memory-Bound    | (baseline)
Level 2-3 (tiled SMEM)  | ~8×     | Memory-Bound    | coalesced + data reuse
Level 5 (double buffer) | ~10×    | Memory→Compute  | Load/Compute overlap
Level 6 (FP16+TC)       | ~80×    | Compute-Bound   | Tensor Core FP16

从 Level 0 到 Level 6: ~80× 加速。其中:
  Level 0→2-3 (数据复用):   ~8×   ← 最大单级收益
  Level 2-3→6 (Tensor Core): ~10×  ← 第二级
  Level 3→5 (latency hiding): ~1.25× ← 精细化收益
```

> 注意：以上加速比为基于理论估算的相对值，实际绝对值取决于矩阵大小和 GPU 型号。`ncu` 实测数据因环境而异，建议在自己的 GPU 上使用配套 benchmark 复现。

---

## 5. 优化边界——"什么时候该停手"

### 5.1 Roofline 决定的硬上限

如果你的 kernel 实测 TFLOPS 已经达到峰值 80%+——再优化也不会超 Roofline。这时的优化 switch：

- FP32 算力瓶颈 → 换 FP16/BF16 提天花板（Tensor Core）
- FP16 算力瓶颈 → 你的 kernel 已经在峰值附近，剩下的空间很小
- HBM 带宽瓶颈 → 数据压缩（FP16、INT8）、算子融合

### 5.2 边际收益递减

```text
优化投入 vs 收益曲线:

  收益
   ↑
   │  ★ Level 2 (coalescing): 巨大收益，改几行代码
   │   ★ Level 3 (shared mem): 显著收益，重构 kernel
   │     ★ Level 4 (bank conflict): 中等收益，改索引/tile 大小
   │       ★ Level 5 (pipeline): 高中等收益，重写 loop
   │          ★ Level 6 (FP16/TC): 巨大收益但需要改精度
   │             ★★ 微调: 收益 < 5%，但耗时以天计
   └──────────────────────────────────────→ 投入

  80/20 法则: 20% 的努力拿到 80% 的收益 (Level 0-3)
  剩下的 20% 需要 80% 的努力 (Level 4-6)
```

### 5.3 "够好了"的判断标准

```text
你的 kernel 是否已经"够快"了？

  ✓ sm_throughput > 70% OR dram_throughput > 70%
  ✓ occupancy > 50%
  ✓ ncu 中没有明显的 red flag（如 32-way bank conflict）
  ✓ 与 cuBLAS/cuDNN 对标在 2× 以内
  → 可以了。投入时间到其他 kernel 或系统层面优化。
```

---

## 6. 总结——优化决策速查

### 6.1 分级诊断 SOP

```text
Step 1: 估算算术强度
  → < 10 FLOP/Byte → Memory-Bound → 优先 Level 2-4
  → > 10 FLOP/Byte → Compute-Bound → 优先 Level 5-6

Step 2: ncu 三指标
  → sm_throughput > 70%? → Compute-Bound 确认
  → dram_throughput > 70%? → Memory-Bound 确认
  → 两者 < 60%? → Latency-Bound → 先查 occupancy (Level 1)

Step 3: 逐级排查
  Level 0: GPU 在做吗?         → nvidia-smi
  Level 1: Occupancy 够吗?     → ncu --section Occupancy
  Level 2: Coalesced 了吗?     → ncu Memory Workload
  Level 3: Shared Memory 复用? → 数据复用分析
  Level 4: Bank Conflict?      → ncu Shared Memory
  Level 5: 延迟隐藏?           → __syncthreads 开销分析
  Level 6: 算术强度?           → FP16/Tensor Core/算子融合

Step 4: 对标
  → cuBLAS/cuDNN 是最优实现的下界——不要试图超过它们
  → 如果差距 > 2×，继续优化；< 1.5×，考虑停手
```

### 6.2 跨文章衔接

| 优化 Level                | 技术                                      | 详见                                                          |
| ------------------------- | ----------------------------------------- | ------------------------------------------------------------- |
| Level 1 (Occupancy)       | `__launch_bounds__`、SMEM/Register budget | [#13 Shared Memory](13_shared_memory_bank_conflict.md) §7.2   |
| Level 2 (Coalescing)      | Global Memory 访问模式                    | [#12 GPU 内存管理](12_gpu_memory_management.md) §1            |
| Level 3 (SMEM)            | Shared Memory tile、`__syncthreads`       | [#13 Shared Memory](13_shared_memory_bank_conflict.md)        |
| Level 4 (Bank Conflict)   | Padding / Swizzle / Reorder               | [#13 Shared Memory](13_shared_memory_bank_conflict.md) §4-5   |
| Level 5a (Warp Shuffle)   | `__shfl_down_sync` 等                     | [#14 Warp-level Programming](14_warp_level_programming.md) §2 |
| Level 5b (Pipeline)       | `cp.async`、`cuda::memcpy_async`          | [#16 Async Copy & Pipeline](16_async_copy_pipeline.md)        |
| Level 5c (CUDA Graphs)    | Graph Capture + Re-launch                 | [#09 CUDA Graphs](09_cuda_graphs.md)                          |
| Level 6 (Arith Intensity) | FP16/BF16、Tensor Core、Kernel 融合       | [#11 Tensor Core GEMM](11_tensor_core_gemm.md)                |
| Multi-GPU                 | NCCL AllReduce、P2P                       | [#15 Multi-GPU](15_multi_gpu_programming.md)                  |
