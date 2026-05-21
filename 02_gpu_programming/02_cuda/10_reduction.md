# Reduction：从朴素实现到 Warp Shuffle

> 基于 A100-SXM4-80GB (CC 8.0) + CUDA 13.1 实测。Reduction 是 GPU 并行编程中最经典的教学案例——它简单到能看懂，复杂到能展示几乎所有 GPU 优化技巧。本文沿 NVIDIA 官方 cuda-samples 的 8 个 kernel 变体，展示从 0.048 ms 到 0.022 ms 的逐级优化过程。

---

## 1. 什么是 Reduction

Reduction 将数组中所有元素通过二元结合操作合并为单个值：

```text
输入: [a₀, a₁, a₂, ..., aₙ₋₁]
输出: a₀ ⊕ a₁ ⊕ a₂ ⊕ ... ⊕ aₙ₋₁    (⊕ 可以是 +, max, min, ...)
```

**在实际 AI 系统中的应用**：

| 场景           | Reduction 类型            |
| -------------- | ------------------------- |
| Softmax 分母   | `sum(exp(x_i))`           |
| Layer Norm     | `sum(x_i)` + `sum(x_i²)`  |
| Attention Mask | `max(score)` 用于数值稳定 |
| Loss 计算      | `sum(cross_entropy)`      |

**为什么 Reduction 是经典教学案例**：

Reduction 只有几行核心逻辑，但每改一行就能触及一个硬件特性：global memory coalescing → shared memory → bank conflict → warp divergence → warp shuffle → template unroll → Cooperative Groups。搞懂这 7 个优化，你就能回答"我的 kernel 为什么慢"中的 80% 的场景。

---

## 2. 朴素实现：Interleaved Addressing（Kernel 0）

这是最直观的 tree-based reduction：每次迭代 stride 翻倍，stride=1 时线程 0 和 1 相加，stride=2 时线程 0 和 2 相加，依此类推。

```c
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if ((tid % (2 * s)) == 0) {        // ← 取模运算 + 不连续活跃
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

**两个问题**：

1. **Warp Divergence**：当 `s=1` 时只有偶数线程活跃（50%），`s=2` 时只有 4 的倍数活跃（25%），`s=4` 时仅 12.5%... 在 warp 级别（32 线程），divergence 让 SIMT 执行效率不断降低。

2. **Shared Memory Bank Conflict**：interleaved addressing 每次访问的地址间隔为 `s`，当 `s` 是 2 的幂次时，大量线程落在同一个 bank 上 → bank conflict。

```text
s=1: thread 0 → bank[0], thread 1 → bank[0+1]  ← 连续，无 conflict
s=2: thread 0 → bank[0], thread 2 → bank[2]     ← 部分 conflict
s=4: thread 0 → bank[0], thread 4 → bank[4]     ← 更严重
s=32: thread 0 → bank[0], thread 32 → bank[0]   ← 完全同 bank！
```

---

## 3. 逐级优化路径

### Kernel 1：修复 Warp Divergence

将条件从 `tid % (2*s) == 0` 改为只让前一半连续线程工作：

```c
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    // 原: if ((tid % (2*s)) == 0)
    if ((tid & (2*s - 1)) == 0)          // ← 位运算替代取模
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

将取模替换为位运算（`&`），减少指令开销。但 divergence 问题依然存在——活跃线程还是分散的。

### Kernel 2：Sequential Addressing

**核心 rewrite**：从 "stride 从小到大" 改为 "stride 从大到小"，让活跃线程始终在前半部分连续排列：

```c
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {                       // ← 前半部分连续活跃
        sdata[tid] += sdata[tid + s];    // ← 操作相邻元素
    }
    __syncthreads();
}
```

```text
s=128: thread 0-127 活跃 (连续) ← 128 个连续线程，4 个完整 warp
s=64:  thread 0-63  活跃 (连续) ← 2 个完整 warp
s=32:  thread 0-31  活跃         ← 1 个完整 warp，完全无 divergence！
s=16:  thread 0-15  活跃         ← 半个 warp（divergence 不可避免，但 < warpSize 之后走不同路径）
```

**收益**：warp 内部的活跃线程始终连续 → divergence 只在最后 5 次迭代（s < 32）出现，之前的迭代完全无 divergence。

### Kernel 3：展开最后一个 Warp

当 s < 32 时，活跃线程不足一个 warp。与其让 `__syncthreads()` 和分支继续运行，不如在编译期就把最后 5 层展开：

```c
// 手动展开最后 5 次迭代 (s=16,8,4,2,1)
if (tid < 32) {
    sdata[tid] += sdata[tid + 32];  // 附加的 2nd warp 合并
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
```

消除了 5 次 `__syncthreads()` 调用和对应的 warp stall。

### Kernel 4：引入 Warp Shuffle

**彻底消除 shared memory 在 warp 内的使用**——用 `__shfl_down_sync` 直接在线程寄存器间交换数据：

```c
cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

// 先在 shared memory 中完成 pre-warp 阶段 (s > 32)
for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] = mySum = mySum + sdata[tid + s];
    cg::sync(cta);
}

// 最后 32 个元素全部在寄存器中完成，零 shared memory 访问
if (cta.thread_rank() < 32) {
    if (blockSize >= 64) mySum += sdata[tid + 32];  // 合并 2nd warp
    for (int offset = 16; offset > 0; offset /= 2)
        mySum += tile32.shfl_down(mySum, offset);   // ← 寄存器级 warp reduce
}
```

**为什么 shuffle 更快**：shared memory ~100 cycles/access，shuffle 是单指令寄存器交换（~1 cycle）。512 线程 block 中最后 6 层迭代从 ~600 cycles 直接降到 ~6 cycles。

### Kernel 5：Template 编译期展开

将 block size 作为模板参数，让编译器在编译期为每个 block size 生成特化代码——循环完全消失：

```c
template <class T, unsigned int blockSize>
__global__ void reduce5(T *g_idata, T *g_odata, unsigned int n) {
    // blockSize 是编译期常量 → 所有 s > 32 的迭代被编译器展开为固定数量的加法和 sync
    ...
}
```

代价：需要针对每个 `blockSize = 512, 256, 128, ..., 1` 分别实例化 kernel。

### Kernel 6：Cooperative Groups + 多元素/线程

综合所有优化并引入 grid-stride loop——每个线程处理多个元素：

```c
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n) {
    T mySum = 0;
    unsigned int gridSize = blockSize * 2 * gridDim.x;  // 全局 stride

    // 每个线程累加 gridSize 跨度的多个元素
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    while (i < n) {
        mySum += g_idata[i];
        if ((i + blockSize) < n) mySum += g_idata[i + blockSize];
        i += gridSize;
    }
    // ... 然后是 Kernel 5 的完全展开 + warp shuffle ...
}
```

**关键思想**：让每个线程处理多个元素，减少需要的 block 数量 → 减少 block 之间的 `__syncthreads` 开销。

### Kernel 7：单 Kernel 多 Block（Cooperative Groups）

Kernel 0-6 都需要两阶段：Kernel 将数组 reduce 为 per-block 结果 → CPU 或第二个 kernel 做最终 reduce。Kernel 7 用 Cooperative Groups 的 `grid_group` 和 `cg::reduce` 在单个 kernel 内完成全部 reduction——利用了 CC 8.0 (A100) 的 `__reduce_add_sync` 硬件加速指令：

```c
// A100 (SM 8.0+) 硬件加速 warp reduce
int warpReduceSum(int mySum) {
    return __reduce_add_sync(0xffffffff, mySum);  // 单指令完成！
}
```

---

## 4. A100 实测性能

```bash
cd cuda-samples/Samples/2_Concepts_and_Techniques/reduction
nvcc -arch=sm_80 -I../../../Common -o reduction reduction.cpp reduction_kernel.cu -lcudart
./reduction --shmoo
```

`--shmoo` 输出表（int 类型，1M - 16M elements 关键切片，时间单位 ms）：

| Elements  | K0 (interleaved) | K1 (div fix) | K2 (seq) | K3 (unroll) | K4 (shuffle) | K5 (tmpl) | K6 (CG+grid) | K7 (single-pass) |
| --------- | ---------------- | ------------ | -------- | ----------- | ------------ | --------- | ------------ | ---------------- |
| 131,072   | 0.029            | 0.028        | 0.026    | 0.020       | 0.016        | 0.020     | **0.017**    | 0.017            |
| 262,144   | 0.032            | 0.028        | 0.028    | 0.020       | 0.019        | 0.016     | **0.018**    | 0.018            |
| 524,288   | 0.037            | 0.032        | 0.030    | 0.027       | 0.023        | **0.023** | **0.019**    | **0.019**        |
| 1,048,576 | 0.048            | 0.038        | 0.035    | 0.030       | 0.024        | **0.023** | **0.022**    | **0.022**        |
| 2,097,152 | 0.070            | 0.051        | 0.045    | 0.034       | 0.026        | **0.026** | **0.028**    | 0.029            |
| 4,194,304 | 0.112            | 0.077        | 0.063    | 0.096       | 0.031        | **0.031** | **0.041**    | **0.041**        |
| 8,388,608 | 0.199            | 0.130        | 0.109    | 0.065       | 0.046        | **0.045** | 0.141        | 0.139            |

> K0-K2 在 16M elements 返回 -1（kernel 执行超时被 kill），因为这些 kernel 的 block 数量无法支撑大规模数据。

**1M elements 加速比**：

```text
K0 (interleaved):  0.048 ms  ██████████████████████████████████  1.00x (baseline)
K1 (div fix):      0.038 ms  ██████████████████████████▌           1.26x
K2 (sequential):   0.035 ms  ████████████████████████▌             1.37x
K3 (unroll warp):  0.030 ms  █████████████████████▌                1.60x
K4 (warp shuffle): 0.024 ms  ████████████████▊                     2.00x
K5 (template):     0.023 ms  ███████████████▋                      2.09x
K6 (CG+grid):      0.022 ms  ██████████████▊                       2.18x  ← 最优
K7 (single-pass):  0.022 ms  ██████████████▊                       2.18x
```

**关键发现**：

- K0→K2 (addressing 优化)：+37%，改的只是索引计算方式
- K2→K4 (warp shuffle)：+46%，这是单次改动中收益最大的
- K4→K6 (template + grid loop)：+9%，接近零开销的工程完善
- **总计 K0→K6：2.18×**——没有任何算法改变，纯工程优化

---

## 5. Warp Shuffle 原理

Warp shuffle 是 CUDA 中最快的线程间通信机制——数据通过寄存器直接交换，不经过 shared memory：

```c
// 标准 warp reduce 模板
int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

执行过程（以 8 线程为例）：

```text
offset=4:  t0←t0+t4  t1←t1+t5  t2←t2+t6  t3←t3+t7
offset=2:  t0←t0+t2  t1←t1+t3
offset=1:  t0←t0+t1
最终 t0 = sum(all 8 threads)
```

| 通信方式      | 延迟          | 带宽              | 适用              |
| ------------- | ------------- | ----------------- | ----------------- |
| Global Memory | ~500 cycles   | ~1.5 TB/s (HBM2e) | 跨 block          |
| Shared Memory | ~20-30 cycles | ~12 TB/s          | block 内          |
| Warp Shuffle  | **~1 cycle**  | 寄存器级          | warp 内 (32 线程) |

> A100 (SM 8.0+) 更进一步：`__reduce_add_sync(mask, val)` 直接用硬件做 warp reduce，整个 32-thread reduce 在一条指令内完成。

---

## 6. 编程启示

```text
层次        优化手段                       适用场景
─────────────────────────────────────────────────
global mem  coalesced access              所有 kernel
shared mem  sequential addressing          block 内数据复用
warp        ── 之上的都不需要 sync ──
warp        __shfl_down_sync              warp 内 reduce/scan
instruction template unroll                固定大小 block
hardware    __reduce_add_sync (A100+)      求和 reduce
```

- **先消除 global memory 瓶颈**（coalesced access），再优化 shared memory（sequential addressing）
- **warp shuffle 是 free lunch**：一行代码替换 5+ 层循环 + sync
- **template block size** 让编译器替你 unroll——编译期常量是 CUDA 性能的关键
- **不要过早优化**：先写对的（K0），再写快的（K6）。K0 的代码 20 行就能读懂；K6 的优化版 80 行但读者理解 K0 后才能看懂
- **Reduction 是基础原语**：softmax、layer norm、attention score 归一化、所有 `all_reduce` 都依赖它

---

## 参考

- [NVIDIA cuda-samples: reduction](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/reduction)
- [CUDA C Programming Guide: Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
- [GPU 编程导论](01_gpu_programming_introduction.md)
- [CUDA 核心详解](02_cuda_cores.md)
