# Reduction：从朴素实现到 Warp Shuffle

> 基于 A100-SXM4-80GB (CC 8.0) + CUDA 13.1 实测。Reduction 是 GPU 并行编程的经典教学案例：几行核心逻辑，能依次触发 coalescing、shared memory、divergence、shuffle 等几乎所有优化点。本文沿 NVIDIA 官方 cuda-samples 的 7 个 kernel 变体，展示从 0.048 ms 到 0.022 ms 的逐级优化过程。
>
> **验证代码**：本文所有实测数据来自配套 Benchmark：[10_reduction_bench.cu](code/10_reduction_bench.cu)（包含 K0-K6 共 7 个 kernel 变体）。

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

Reduction 只有几行核心逻辑，但每改一行就能触及一个硬件特性：global memory coalescing → shared memory → warp divergence → warp shuffle → template unroll → Cooperative Groups。搞懂这 7 个优化，就掌握了"我的 kernel 为什么慢"的绝大多数场景。

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

2. **无效工作**：随 `s` 增大，绝大多数线程被 mask 掉（`s=16` 时仅 6% 的线程活跃），SIMT 单元仍要为这些 warp 发射指令和同步；`%` 取模本身也是额外指令。（注：这里的寻址是加法 `tid + s`，Bank 分布只是轮转，**不会**产生 Bank Conflict——"stride 是 2 的幂就冲突"是乘法寻址（`tid * s`）才有的问题，常见误解，见 [#13](13_shared_memory_bank_conflict.md) 第 6.3 节。）

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
s=16:  thread 0-15  活跃         ← 半个 warp（s<32 后活跃线程不足一个 warp，divergence 无法避免）
```

**收益**：warp 内部的活跃线程始终连续 → divergence 只在最后 5 次迭代（s < 32）出现，之前的迭代完全无 divergence。

### Kernel 3：展开最后一个 Warp

当 s < 32 时，活跃线程不足一个 warp。与其让 `__syncthreads()` 和分支继续运行，不如在编译期就把最后 5 层展开：

```c
// 手动展开最后 5 次迭代 (s=16,8,4,2,1)
// 注意 volatile：最后这几层没有 __syncthreads 保护，编译器可能把共享内存读写
// 重排/缓存进寄存器，官方 reduce3 正是用 volatile 防住这一点
if (tid < 32) {
    volatile float* smem = sdata;   // T 为 int 时用 volatile int*
    smem[tid] += smem[tid + 32];    // 先把 2nd warp 的部分和并进来
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
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

**为什么 shuffle 更快**：shared memory 一次访问 ~20-30 cycles 且每层迭代要付一次 `__syncthreads()`，shuffle 是单指令寄存器交换（~1 cycle、无 sync）。最后 6 层迭代的开销差一个数量级以上。

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

### 关于"单 Kernel 全局归约"

Kernel 0-6 都需要两阶段：kernel 把数组归约为 per-block 结果，再由第二个 kernel（或 CPU）做最终归约。要在单个 kernel 内跨 block 归约，需要 Cooperative Groups 的 grid sync（`cg::this_grid().sync()`，要求 cooperative launch），或改用原子操作的 single-pass 方案——两者都不在官方 reduction sample 与配套 bench 的 K0-K6 之列，本文不给未实测的数字。

顺带说明：A100 的硬件归约指令 `__reduce_add_sync` 只支持整型（见 [#14](14_warp_level_programming.md) 第 7 节）。上面各 kernel 是 float 数据，warp 内归约用 shuffle 循环实现。

---

## 4. A100 实测性能

```bash
cd cuda-samples/Samples/2_Concepts_and_Techniques/reduction
nvcc -arch=sm_80 -I../../../Common -o reduction reduction.cpp reduction_kernel.cu -lcudart
./reduction --shmoo
```

`--shmoo` 输出表（int 类型，128K - 8M elements 关键切片，时间单位 ms）：

| Elements  | K0 (interleaved) | K1 (div fix) | K2 (seq) | K3 (unroll) | K4 (shuffle) | K5 (tmpl) | K6 (CG+grid) |
| --------- | ---------------- | ------------ | -------- | ----------- | ------------ | --------- | ------------ |
| 131,072   | 0.029            | 0.028        | 0.026    | 0.020       | 0.016        | 0.020     | **0.017**    |
| 262,144   | 0.032            | 0.028        | 0.028    | 0.020       | 0.019        | 0.016     | **0.018**    |
| 524,288   | 0.037            | 0.032        | 0.030    | 0.027       | 0.023        | **0.023** | **0.019**    |
| 1,048,576 | 0.048            | 0.038        | 0.035    | 0.030       | 0.024        | **0.023** | **0.022**    |
| 2,097,152 | 0.070            | 0.051        | 0.045    | 0.034       | 0.026        | **0.026** | **0.028**    |
| 4,194,304 | 0.112            | 0.077        | 0.063    | 0.096       | 0.031        | **0.031** | **0.041**    |
| 8,388,608 | 0.199            | 0.130        | 0.109    | 0.065       | 0.046        | **0.045** | 0.141        |

> K0-K2 在 16M elements 时返回 -1——这是 sample 自身的时间上限约定（这些 kernel 的 block 数量无法支撑大规模数据），不是硬件 watchdog 触发。

**1M elements 加速比**：

```text
K0 (interleaved):  0.048 ms  ██████████████████████████████████  1.00x (baseline)
K1 (div fix):      0.038 ms  ██████████████████████████▌           1.26x
K2 (sequential):   0.035 ms  ████████████████████████▌             1.37x
K3 (unroll warp):  0.030 ms  █████████████████████▌                1.60x
K4 (warp shuffle): 0.024 ms  ████████████████▊                     2.00x
K5 (template):     0.023 ms  ███████████████▋                      2.09x
K6 (CG+grid):      0.022 ms  ██████████████▊                       2.18x  ← 最优
```

**关键发现**：

- K0→K2 (addressing 优化)：+37%，改的只是索引计算方式
- K2→K4 (warp shuffle)：+46%，这是单次改动中收益最大的
- K4→K6 (template + grid loop)：+9%，接近零开销的工程完善
- 总计 K0→K6：2.18×，没有任何算法层面的改变，全部是工程优化

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
warp        shuffle（寄存器内完成，无需 block 级 sync）   warp 内 reduce/scan
instruction template unroll                固定大小 block
hardware    __reduce_add_sync (A100+, 整型)  求和 reduce
```

- 先消除 global memory 瓶颈（coalesced access），再优化 shared memory（sequential addressing）
- warp shuffle 用一行代码替换 5+ 层循环 + sync
- template block size 让编译器替你 unroll——编译期常量是 CUDA 性能的关键
- 不要过早优化：先写对的（K0），再写快的（K6）。K0 的代码 20 行就能读懂；K6 的优化版 80 行，但要读懂 K0 才能看懂
- Reduction 是基础原语：softmax、layer norm、attention score 归一化、所有 `all_reduce` 都依赖它

---

## 参考

- [NVIDIA cuda-samples: reduction](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/reduction)
- [CUDA C Programming Guide: Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
- [GPU 编程导论](01_gpu_programming_introduction.md)
- [CUDA 核心详解](02_cuda_cores.md)
