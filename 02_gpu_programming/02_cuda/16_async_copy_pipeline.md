# 异步拷贝与现代 Pipeline——从 cp.async 到 memcpy_async

你已经知道 Shared Memory 比 Global Memory 快 80 倍（[#13](13_shared_memory_bank_conflict.md)），也知道 Tensor Core 做 GEMM 能达到 ~90 TFLOPS（[#11](11_tensor_core_gemm.md)）。但两者之间有一个一直被忽略的环节：**数据是怎么从 HBM 搬进 Shared Memory 的**。

传统的做法是：一个 block 的所有线程一起 load，然后 `__syncthreads()`，然后开始计算。等这一批算完了，再 load 下一批，再 barrier，再算。在这个模型下，SM 的计算单元在 load 期间**完全空闲**——数据搬运和计算是串行的。

SM80 (A100) 引入了一个关键能力：**`cp.async`**——让 DMA 引擎在后台把数据从 Global Memory 搬进 Shared Memory，同时 SM 的计算单元继续执行其他指令。配合 Pipeline 原语管理"第几批数据到了、第几批可以用了"，你可以实现**搬运和计算的完全重叠**——double buffer 甚至 triple buffer。

这就是 FlashAttention 和 bf16TensorCoreGemm 能达到高利用率的核心技巧之一。本文将拆解这组技术：`cp.async` → Pipeline 原语 → `memcpy_async` → Double Buffering → 实战。

> **前置要求**：你已经理解 Shared Memory 的使用模式（见 [#13](13_shared_memory_bank_conflict.md)），知道 Block/Warp 执行模型（见 [#01](01_gpu_programming_introduction.md)）。
>
> **验证代码**：本文所有实测数据来自配套 Benchmark：[16_async_copy_bench.cu](code/16_async_copy_bench.cu)。对比 1-stage 同步 vs 2-stage double buffer 的 tile 处理性能。注意：benchmark 使用同步 `ld` 实现 double buffer（`cp.async` 需 PTX inline asm），因此加速比有限（~1.02x）——这恰好证明了"pipeline 的真正收益来自异步 DMA，不是简单的双 buffer 切换"。

---

## 1. 传统拷贝的瓶颈——"为什么 `__syncthreads()` 在吃掉你的 SM 时间"

### 1.1 同步 Load 的时间线

一个典型的 tile-based GEMM kernel 的循环体：

```cuda
for (int k = 0; k < K; k += TILE) {
    // Step 1: Load tile from Global Memory → Shared Memory (同步)
    As[threadIdx.y][threadIdx.x] = A[row * K + (k + threadIdx.x)];
    Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
    __syncthreads();  // ← 所有线程等待最慢的 load

    // Step 2: Compute from Shared Memory
    for (int kk = 0; kk < TILE; kk++)
        sum += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
    __syncthreads();  // ← 等计算完成，然后下一轮 load 覆盖 As/Bs
}
```

时间线画出来是这样的：

```text
时间 →
  Load   |████████████|            |████████████|            |████████████|
  Sync   |            █|           |            █|           |            █|
  Compute|             |████████████|             |████████████|             |

  SM 计算单元:  ░░░░空闲░░░░|████计算████|░░░░空闲░░░░|████计算████|
```

SM 的计算单元在 Load 阶段**完全空闲**。Global Memory 延迟约 400ns——在这 400ns 里，一个 A100 SM 本可以做 ~7800 次 FP32 FMA 操作。这些周期白白流走了。

### 1.2 本质：生产者-消费者问题

Shared Memory 的使用模式是一个经典的生产者-消费者问题：

- **生产者**：Load 操作——把数据从 HBM 搬到 Shared Memory
- **消费者**：Compute 操作——从 Shared Memory 读取、计算

传统同步方案里，生产者在工作时消费者干等，消费者在工作时生产者干等。解决方案也很经典：**让生产者和消费者同时工作**，用一个 pipeline buffer 解耦二者的速度。

CPU 类比：软件流水线（Software Pipelining）——编译器的 loop optimization 中，将循环体拆分为"预取下一轮数据 + 计算当前数据"两个阶段，使得 `load[i+1]` 与 `compute[i]` 重叠。GPU 的 Pipeline 原语将此模式硬件化。

---

## 2. `cp.async`——SM80+ 的异步拷贝指令

### 2.1 硬件机制

`cp.async`（SM80+ / A100+）是 PTX 级别的异步拷贝指令。它与传统 `ld.shared` 的核心区别：

| 特性         | `ld.shared` (传统)                | `cp.async` (SM80+)                                     |
| ------------ | --------------------------------- | ------------------------------------------------------ |
| 执行方式     | 线程自己 load，等数据到达         | DMA 引擎搬运，线程不等                                 |
| 数据路径     | 线程寄存器 → Shared Mem           | Global Mem → Shared Mem（不经过寄存器！）              |
| 完成通知     | 隐式（线程到达时数据已就绪）      | 显式（通过 `cp.async.wait_group` 或 pipeline barrier） |
| 对线程的影响 | 阻塞（stall）当前指令直到数据到达 | 非阻塞——线程继续执行后续指令                           |

**关键洞察**：`cp.async` 走了 GPU 内部的 DMA 路径——数据从 HBM 直接进入 Shared Memory，**不占用线程的寄存器**。这意味着发起拷贝后，线程可以立即执行不依赖 Shared Memory 的其他指令（比如寄存器上的计算、或者 `cp.async` 另外一批数据）。

CPU 类比：`DMA`（Direct Memory Access）——CPU 告诉 DMA 控制器"把这块内存拷到那边去"，然后 CPU 继续执行其他指令。`cp.async` 就是 GPU 内部的 DMA，只是源和目标分别是 Global Memory 和 Shared Memory。

### 2.2 PTX 语法与三种变体

```cuda
// 在 CUDA C++ 中通过 inline PTX 调用
asm volatile(
    "cp.async.ca.shared.global [%0], [%1], %2;\n"
    :: "r"(smem_addr), "l"(gmem_addr), "n"(copy_size)
);
```

三种缓存策略变体（影响数据在 L1/L2 cache 中的留存方式）：

| 变体          | 全称         | L2 行为       | 适用场景                    |
| ------------- | ------------ | ------------- | --------------------------- |
| `cp.async.ca` | Cache-All    | 数据保留在 L2 | 数据会被其他线程/block 复用 |
| `cp.async.cg` | Cache-Global | 数据 evict L2 | 数据仅用一次，避免污染 L2   |
| `cp.async`    | 默认         | 同 `ca`       | 默认保留在缓存层级          |

大多数 tile-based kernel 用 **`cp.async.ca`**——因为 tile 数据大概率在 L2 里被后续的 block 重复使用（L2 是跨 SM 共享的）。

### 2.3 与 Pipeline 的配合——commit group

`cp.async` 指令入队后，硬件会为这批拷贝分配一个 **commit group**。后续可以用 `cp.async.wait_group N` 等待"最近 N 个 commit group 还没完成的，等它们全部完成"：

```cuda
// 发起第 0 批拷贝
asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem));
asm volatile("cp.async.commit_group;\n");  // ← 标记这一批的边界

// 发起第 1 批拷贝
asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem+16), "l"(gmem+16));
asm volatile("cp.async.commit_group;\n");

// 等待最近的 2 个 commit group 全部完成
asm volatile("cp.async.wait_group 2;\n");
```

**关键概念**：`commit_group` 是 batch 边界。你可以在 commit_group 之间插入计算指令——这些计算在数据到达之前就已开始执行。

---

## 3. Pipeline 原语——"怎么知道第二批数据到了"

PTX 级别的 `cp.async.commit_group` / `cp.async.wait_group` 是功能完整的，但手工管理 commit group 的数量（"等 2 个还是 3 个"）容易出错。CUDA C++ 从 11.1 开始提供了 `cuda::pipeline`——一个更高层的抽象。

### 3.1 Pipeline 对象的生命周期

```cuda
#include <cuda/pipeline>

// 创建一个 pipeline，最多同时进行 4 个阶段的异步拷贝
__shared__ cuda::pipeline_shared_state<cuda::thread_scope::block, 4> pipe_state;
auto pipe = cuda::make_pipeline(cta, &pipe_state);
```

Pipeline 对象管理一个固定容量的"飞行中"请求队列。核心操作：

| 操作                            | 角色     | 语义                                                                   |
| ------------------------------- | -------- | ---------------------------------------------------------------------- |
| `pipe.producer_acquire()`       | Producer | 获取下一个可用的 buffer slot（如果需要保证 buffer 空闲，内部可能等待） |
| `cuda::memcpy_async(..., pipe)` | Producer | 发起异步拷贝，关联到一个 pipeline stage                                |
| `pipe.producer_commit()`        | Producer | 提交当前 stage 的拷贝（标记"这批数据已经发出"）                        |
| `pipe.consumer_wait()`          | Consumer | 等待下一个 stage 的数据到达 Shared Memory                              |
| `pipe.consumer_release()`       | Consumer | 释放已消费的 stage（让 producer_acquire 可以复用 buffer）              |

### 3.2 Producer-Consumer 时序

用一个 2-stage pipeline 为例（stage 0 和 stage 1 交替使用）：

```text
Stage 0:  [  Load A₀   ][  Compute A₀  ]
Stage 1:      [  Load A₁  ][  Compute A₁  ]

Thread 0 视角:
  producer_acquire → memcpy_async(A₀) → producer_commit
  consumer_wait → compute(A₀) → consumer_release
  producer_acquire → memcpy_async(A₁) → producer_commit
  consumer_wait → compute(A₁) → consumer_release
  ...
```

**关键点**：`producer_acquire` 在 buffer 被 consumer_release 释放之前会 block（等待旧数据被消费完）。`consumer_wait` 在 producer_commit 完成之前会 block（等待新数据到达 Shared Memory）。

### 3.3 Pipeline 状态机

```text
              producer_acquire
    [empty] ──────────────────→ [acquired]
       ↑                            │
       │ consumer_release            │ memcpy_async + producer_commit
       │                            ↓
    [consumed] ←──────────────── [committed]
                    consumer_wait
```

`pipeline_shared_state` 的模板参数 `<scope, stages>` 中，`stages` 决定了最多有多少个 buffer slot 在飞行中。stages=2 就是 double buffering。

---

## 4. `cuda::memcpy_async`——CUDA C++ 层面的统一 API

### 4.1 与 `cp.async` 的关系

`cuda::memcpy_async` 是 C++ 层面的封装。编译器在 SM80+ 上自动将其映射为 `cp.async` PTX 指令；在老架构上则回退到同步的 `ld.shared`。

```cuda
// 基本用法：从 Global Memory 异步拷贝到 Shared Memory
cuda::memcpy_async(group, smem_dst, gmem_src, sizeof(float) * n, pipe);
```

参数：

- `group`：`cooperative_groups::thread_block`——本 block 内的所有线程
- `smem_dst`：目标 Shared Memory 地址
- `gmem_src`：源 Global Memory 地址
- `size`：拷贝字节数
- `pipe`：关联的 pipeline 对象

### 4.2 与非 pipeline 的 barrier 配合

如果你只需要简单的异步拷贝（不需要完整的 pipeline），可以用 `cuda::barrier` 替代：

```cuda
#include <cuda/barrier>

__shared__ cuda::barrier<cuda::thread_scope::block> bar;
if (threadIdx.x == 0) init(&bar, blockDim.x);

// 发起异步拷贝
cuda::memcpy_async(cta, smem, gmem, size, bar);

// 等待拷贝完成
bar.arrive_and_wait();
// 现在可以用 smem 了
```

这种单 barrier 模式比 `pipeline` 简单，但没有 double buffering——一轮拷贝完成后才能开始下一轮。

### 4.3 Pipeline 模式示例

完整的 2-stage pipeline with double buffering：

```cuda
#include <cuda/pipeline>

__global__ void gemm_pipelined(float *C, const float *A, const float *B, int N) {
    extern __shared__ float smem[];
    float *As0 = smem;                          // buffer 0
    float *Bs0 = smem + TILE * TILE;
    float *As1 = smem + 2 * TILE * TILE;        // buffer 1
    float *Bs1 = smem + 3 * TILE * TILE;

    auto cta = cg::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::block, 2> ps;
    auto pipe = cuda::make_pipeline(cta, &ps);

    // 预加载第 0 批（stage 0）
    pipe.producer_acquire();
    cuda::memcpy_async(cta, As0, &A[row * N + 0], sizeof(float) * TILE, pipe);
    cuda::memcpy_async(cta, Bs0, &B[0 * N + col], sizeof(float) * TILE, pipe);
    pipe.producer_commit();

    for (int k = TILE; k < K; k += TILE) {
        // 发起第 (k/TILE) 批的异步拷贝
        pipe.producer_acquire();
        int stage = (k / TILE) % 2;
        float *As = (stage == 0) ? As0 : As1;
        float *Bs = (stage == 0) ? Bs0 : Bs1;
        cuda::memcpy_async(cta, As, &A[row * N + k], sizeof(float) * TILE, pipe);
        cuda::memcpy_async(cta, Bs, &B[k * N + col], sizeof(float) * TILE, pipe);
        pipe.producer_commit();

        // 消费上一批数据（与当前拷贝并发！）
        pipe.consumer_wait();
        float *As_prev = (stage == 1) ? As0 : As1;
        float *Bs_prev = (stage == 1) ? Bs0 : Bs1;
        for (int kk = 0; kk < TILE; kk++)
            sum += As_prev[threadIdx.y * TILE + kk] * Bs_prev[kk * TILE + threadIdx.x];
        pipe.consumer_release();
    }
    // 消费最后一批
    pipe.consumer_wait();
    // ... 最后一次 compute ...
    pipe.consumer_release();

    C[row * N + col] = sum;
}
```

时间线对比：

```text
传统（同步）:
  Load  |████████|            |████████|            |████████|
  Comp  |        |████████████|        |████████████|        |

Pipeline (2-stage):
  Load  |████████|████████|████████|
  Comp  |        |████████████|████████████|
          ↑ 重叠！Load[k+1] 与 Compute[k] 同时进行
```

---

## 5. Double Buffering——"永不停机的流水线"

### 5.1 为什么需要两个 Buffer

如果只有一个 Shared Memory buffer：

- Producer 写数据时，Consumer 不能读（数据不完整）
- Consumer 读数据时，Producer 不能写（会破坏正在使用中的数据）

Double Buffer 将 Shared Memory 分成两个等大的 tile 区域（buffer A 和 buffer B），producer 和 consumer 交替使用：

```text
         Buffer A         Buffer B
 iter 0: [Producer→]     [空闲]
 iter 1: [Consumer←]     [Producer→]     ← 重叠！
 iter 2: [Producer→]     [Consumer←]     ← 重叠！
 iter 3: [Consumer←]     [Producer→]
 ...
```

### 5.2 Shared Memory 布局

```text
Shared Memory 总用量（double buffer):
  |── As0 (TILE×TILE×4B) ──|── Bs0 (TILE×TILE×4B) ──|
  |── As1 (TILE×TILE×4B) ──|── Bs1 (TILE×TILE×4B) ──|

对于 128×128 FP32 tiles: 4 × 128 × 128 × 4B = 256 KB
A100 SM 最大 SMEM = 164 KB → 放不下！

解决：使用较小的 tile (如 64×64)，或用 FP16（省一半空间）
这正是选择 GEMM tile 大小时需要考虑的约束。
```

### 5.3 Triple Buffering

如果 Load 和 Compute 的时间不匹配——比如 Load 比 Compute 慢很多——可以增加 buffer 数量到 3 个（triple buffering），让 producer 在 consumer 慢的时候也不闲置。代价是 Shared Memory 用量进一步增加。

实际中，2-stage（double buffer）是最常用的——在 Load 和 Compute 时间大致匹配时已足够，且 SMEM 压力可控。

---

## 6. 实战应用

### 6.1 BF16 GEMM 的异步拷贝

在 [#11](11_tensor_core_gemm.md) 中展示的 `bf16TensorCoreGemm` 达到 90.9 TFLOPS 的关键因素之一就是 `cp.async`。官方 sample 的关键路径：

```cuda
// bf16TensorCoreGemm 的简化 core loop
// 来自 NVIDIA cuda-samples
for (int k = 0; k < K; k += TILE_K) {
    // 预取下一批 K tile（异步）
    #pragma unroll
    for (int i = 0; i < TILE_M / 16; i++) {
        cp.async.ca.shared.global [smem_a + i * 16 * lda], [gmem_a + k + i * 16 * K];
    }
    cp.async.commit_group();

    // 计算当前已有的 K tile（在 Tensor Core 上）
    wmma::mma_sync(acc, a_frag, b_frag, acc);

    // 等待预取的这一批就绪
    cp.async.wait_group 1;
    __syncthreads();
}
```

注意 `cp.async.commit_group() + wmma::mma_sync() + cp.async.wait_group 1` 的序列——Tensor Core 在算当前数据的同时，DMA 引擎在搬下一批数据。这就是"重叠"。

### 6.2 FlashAttention 的 Tile Pipeline

FlashAttention 将 QKV 切分为 tile，沿 seq 维度迭代。每个迭代：

1. Load Q tile（同步或异步）
2. Load K/V tile（异步——下一批在算当前批时搬）
3. Compute Attention Score on current K/V tile
4. Rescale + accumulate

Q tile 在整个外循环中不变——只需要一次同步 load。K/V tile 的 load 和 attention score 的计算可以用 pipeline 重叠：Compute[K₀] ‖ Load[K₁]。

### 6.3 通用模板：何时套用 Pipeline

任何满足以下条件的 kernel 都适合 pipeline：

1. **Tile-based**：数据可以切分为独立的 tile
2. **循环中有 Load 和 Compute 两个分离的阶段**
3. **Shared Memory 够用**（至少能装下 double buffer）
4. **SM80+**（老架构上 memcpy_async 回退到同步，无性能收益）

---

## 7. 权衡与决策

### 7.1 `memcpy_async` vs 手写 `cp.async` PTX

|            | `cuda::memcpy_async` | `cp.async` PTX inline asm             |
| ---------- | -------------------- | ------------------------------------- |
| 可读性     | 高（标准 C++）       | 低                                    |
| 可移植性   | 自动回退 SM70        | 仅 SM80+                              |
| 编译器优化 | 编译器可见、可优化   | 编译器屏障                            |
| 控制粒度   | 粗                   | 细（cache 策略、size 精确控制）       |
| 生产推荐   | 首选                 | 只在需要 `cg` cache mode 等精细控制时 |

**建议**：用 `cuda::memcpy_async` 写第一版——可读、可移植、正确。当 ncu 显示 Global→Shared Memory 带宽仍有显著空隙时，再考虑手写 PTX。

### 7.2 Pipeline Depth 决策

```text
Pipeline depth:
  1 stage   — 无重叠（相当于传统 __syncthreads）
  2 stages  — double buffer，最常见，SMEM 压力中等
  3+ stages — triple buffer 或更多，Load >> Compute 时收益最大，SMEM 压力高

选择:
  - 首先尝试 2-stage（double buffer）
  - 用 ncu 看 Load 和 Compute 的时间比例：
    - Load 时间 ≈ Compute 时间 → 2-stage 接近最优
    - Load 时间 >> Compute 时间 → 考虑 3-stage
    - Load 时间 << Compute 时间 → 使用 1-stage（生产太快，buffer 多了也无用）
  - 每次增加 stage 数都检查 Occupancy 是否下降
```

### 7.3 Shared Memory 预算

Double buffering 使 SMEM 用量 **~2×**。对于 A100（164KB/SM）：

- 64×64 FP32 GEMM double buffer: 64×64×4B×4 = **64KB** — 宽松
- 128×128 FP32 GEMM double buffer: 128×128×4B×4 = **256KB** — 放不下！

这时要么用 FP16（半精度省一半），要么缩 tile，要么接受单 buffer。这是 GEMM tile 选型时要和 SMEM 容量一起算的约束。

### 7.4 SM70 兼容性

`cuda::memcpy_async` 在 SM70 (V100) 上不会报错——它会自动回退到同步 `ld.shared`。性能当然不会提升，但代码不需要改。这意味着你可以用同一份源码，在 A100 上享受 pipeline 加速，在 V100 上保持正确（只是慢）。

---

## 8. 总结

### 8.1 技术演进路径

```text
传统同步 Load + __syncthreads
  → cp.async (SM80+) — 异步拷贝，DMA 引擎搬运
    → commit_group + wait_group — 手工管理批次
      → cuda::memcpy_async + pipeline — C++ 封装
        → Double Buffering — Load 和 Compute 重叠
          → Triple Buffering — 进一步隐藏 Load 延迟
```

### 8.2 速查

| 你想做什么                      | 用哪个                                   | 最低 CC              |
| ------------------------------- | ---------------------------------------- | -------------------- |
| 最简单的异步拷贝                | `cuda::memcpy_async` + `cuda::barrier`   | 7.0（SM70+回退同步） |
| 完整的 pipeline + double buffer | `cuda::memcpy_async` + `cuda::pipeline`  | 7.0（回退同步）      |
| 精细控制 cache 策略（`cg`）     | `cp.async.cg` PTX inline asm             | 8.0                  |
| 最快的手工 pipeline             | `cp.async` + `commit_group`/`wait_group` | 8.0                  |

### 8.3 与其他文章的衔接

| 本文涉及的                                               | 详见                                                                    |
| -------------------------------------------------------- | ----------------------------------------------------------------------- |
| Shared Memory 使用模式与 Bank Conflict                   | [#13 Shared Memory 与 Bank Conflict](13_shared_memory_bank_conflict.md) |
| Tensor Core GEMM 与 bf16 async copy                      | [#11 Tensor Core GEMM](11_tensor_core_gemm.md)                          |
| Global Memory 延迟与 HBM 带宽                            | [#12 GPU 内存管理](12_gpu_memory_management.md)                         |
| Cooperative Groups（`cuda::memcpy_async` 的 group 参数） | [#14 Warp-level Programming](14_warp_level_programming.md)              |
