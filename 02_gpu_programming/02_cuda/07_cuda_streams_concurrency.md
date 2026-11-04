# CUDA Streams 并发实战

> 基于 RTX 5090 (2 个 async copy engine) 和 A100-SXM4-80GB (3 个 async copy engine) 双平台实测。本文演示 CUDA Streams 如何实现 H2D + Kernel + D2H 重叠执行，并测量实际加速比。
>
> **验证代码**：本文所有实测数据来自配套 Benchmark：[07_streams_concurrency_bench.cu](code/07_streams_concurrency_bench.cu)。

---

## 1. 背景

### 1.1 Default Stream vs Explicit Streams

把全部工作塞进默认 stream（legacy stream 0）时，操作严格串行：H2D0 → Kernel0 → D2H0 → H2D1 → Kernel1 → D2H1。stream 内部按提交顺序执行是所有 stream 的通性；legacy 默认流额外的特殊性质是它与所有阻塞语义的 stream 互斥（隐式同步）。串行执行下，GPU 的 compute engine 在数据传输时空闲，copy engine 在 kernel 执行时空闲——两个 engine 交替闲置，硬件利用率低下。

显式 stream（`cudaStreamCreate`）打破这种约束。每个 stream 内部仍然保持顺序，但**不同 stream 之间可以并发**。当一个 stream 的 kernel 在执行时，另一个 stream 的 H2D 可以同时进行——GPU 的多个硬件引擎（copy engine + compute engine）可以同时忙。

### 1.2 硬件并发能力

RTX 5090 有 2 个 async copy engine，理论最多可实现 2 路并发数据传输；A100-SXM4-80GB 有 3 个，理论 3 路并发。加上 compute engine，多操作可同时进行。但实际并发度都受限于：

- PCIe 带宽共享：多个 copy engine 共享同一 PCIe 链路（RTX 5090: Gen 5 ×16 ~63 GB/s；A100: Gen 4 ×16 ~32 GB/s 理论值）。如果多个 engine 同时做 H2D，带宽被均分。
- 双向同时传输的实测吞吐低于两倍单向值：PCIe 是全双工链路，H2D/D2H 方向各自独立，但 nvbandwidth 实测同时双向时单方向从 56.3 掉到 50.3 GB/s（链路仲裁与编码开销）。
- Kernel 执行时长：只有当 kernel 执行时间与数据传输时间大致匹配时，才能最大化重叠。如果 kernel 太短，数据传输追不上；如果 kernel 太长，copy engine 闲置。

### 1.3 测试策略

以下程序用 4 个 stream 各处理 256 MB 数据（含 H2D + kernel + D2H），对比串行和并发的总耗时。kernel 的循环次数 (K=1024) 经过调整，使单 stream 的计算时间与传输时间大致相当——这是最大化重叠的前提。

---

## 2. 对比测试程序

```bash
cat > stream_overlap.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

#define N (64 * 1024 * 1024)   // 256 MB per buffer
#define K 1024                  // dummy compute iterations
#define STREAMS 4

#define CHECK(cmd) do {                                    \
    cudaError_t e = cmd;                                   \
    if (e != cudaSuccess) {                                \
        printf("Error: %s\n", cudaGetErrorString(e));      \
        exit(1);                                           \
    }                                                      \
} while(0)

__global__ void dummy_kernel(float *d, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = d[i];
        for (int j = 0; j < K; j++)
            x = x * s + 1.0f;
        d[i] = x;
    }
}

int main() {
    float *h[STREAMS], *d[STREAMS];
    cudaStream_t s[STREAMS];
    for (int i = 0; i < STREAMS; i++) {
        CHECK(cudaMallocHost(&h[i], N * sizeof(float)));
        CHECK(cudaMalloc(&d[i], N * sizeof(float)));
        CHECK(cudaStreamCreate(&s[i]));
    }

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // === Test 1: Sequential (no streams) ===
    CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < STREAMS; i++) {
        CHECK(cudaMemcpy(d[i], h[i], N * sizeof(float),
                        cudaMemcpyHostToDevice));
        dummy_kernel<<<(N + 255) / 256, 256>>>(d[i], N, 1.0001f);
        CHECK(cudaMemcpy(h[i], d[i], N * sizeof(float),
                        cudaMemcpyDeviceToHost));
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float t_seq;
    cudaEventElapsedTime(&t_seq, start, stop);

    // === Test 2: Concurrent with streams ===
    CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < STREAMS; i++) {
        CHECK(cudaMemcpyAsync(d[i], h[i], N * sizeof(float),
                             cudaMemcpyHostToDevice, s[i]));
        dummy_kernel<<<(N + 255) / 256, 256, 0, s[i]>>>(d[i], N, 1.0001f);
        CHECK(cudaMemcpyAsync(h[i], d[i], N * sizeof(float),
                             cudaMemcpyDeviceToHost, s[i]));
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float t_async;
    cudaEventElapsedTime(&t_async, start, stop);

    float total = ((float)N * sizeof(float) * 3 * STREAMS)
                / (1024.0 * 1024.0 * 1024.0);
    printf("Data: %.0f MB x %d streams = %.1f GB total\n",
           (float)N * sizeof(float) / (1024 * 1024), STREAMS, total);
    printf("Sequential:  %7.2f ms  (%.1f GB/s)\n",
           t_seq, total / t_seq * 1000);
    printf("Streamed:    %7.2f ms  (%.1f GB/s)\n",
           t_async, total / t_async * 1000);
    printf("Speedup:     %.2fx\n", t_seq / t_async);

    for (int i = 0; i < STREAMS; i++) {
        CHECK(cudaFreeHost(h[i]));
        CHECK(cudaFree(d[i]));
        CHECK(cudaStreamDestroy(s[i]));
    }
    return 0;
}
EOF

nvcc -o stream_overlap stream_overlap.cu
./stream_overlap
```

---

## 3. 实测结果

**RTX 5090 输出**：

```text
Data: 256 MB x 4 streams = 3.0 GB total
Sequential:   63.81 ms  (47.0 GB/s)
Streamed:     27.01 ms  (111.1 GB/s)
Speedup:     2.36x
```

### 3.1 为什么不是 3x

2 个 async copy engine + 1 个 compute engine 理论上可 3 路并发，但实际受限于：

- D2H 和 H2D 共享 copy engine：4 个 stream 的 8 次传输竞争 2 个 engine
- 双向传输的实测吞吐低于两倍单向值：nvbandwidth 实测单向 ~56 GB/s、同时双向各 ~50 GB/s。这是链路仲裁与编码开销——PCIe 本身是全双工链路，不存在"半双工"
- Kernel 执行时间不完美匹配：dummy kernel 时长可能不精确等于传输时间

---

## 4. A100 实测结果（官方 cuda-samples）

以下使用 NVIDIA 官方 cuda-samples 13.1 中的 `simpleStreams`、`simpleHyperQ`、`simpleMultiCopy` 在 A100-SXM4-80GB (CC 8.0, 3 copy engines) 上验证。

### 4.1 simpleStreams — 基础流并发

官方 simpleStreams（cuda-samples v13.1 路径 `cpp/0_Introduction/simpleStreams`）把同一组 H2D + kernel + D2H 分别用单 stream 顺序执行和多 stream 重叠执行，cudaEvent 计时后打印对比：

```text
Single stream = <X> ms
Multi-stream  = <Y> ms
Speedup       = <X/Y>x
```

以上是该 sample 的真实输出格式（具体数字随 GPU 与参数变化，直接编译运行即可得到自己平台的结果）。重叠收益取决于 kernel 时间与传输时间的比值：传输占比越大、stream 切分越均匀，重叠空间越大；kernel 占绝对主导时收益趋近于零。

### 4.2 simpleHyperQ — 32 路硬件队列并发

A100 支持 HyperQ——32 个独立的硬件工作队列，允许多个 stream 真正并行执行：

```text
[simpleHyperQ] - GPU: NVIDIA A100-SXM4-80GB
Expected serial:    0.330s - 0.640s
Expected concurrent: ~0.020s
Measured:           0.027s
```

实测 0.027s，接近理论并发时间。32 路 kernel 在执行时间上几乎完全重叠。HyperQ（32 个独立硬件工作队列）允许多个 stream 的 kernel 真并行；消费级 GPU 的硬件队列数量少得多，多 stream 并行度受限。

### 4.3 simpleMultiCopy — 多 copy engine 与负加速

simpleMultiCopy 用两个 stream + 两个 copy engine 对比串行与重叠执行。在 kernel 极轻（微秒级）而传输相对长的组合下，sample 自己的输出就可能给出负的 speedup——`Avg. speedup gained (serialized - overlapped)` 为负值，即重叠版比串行版更慢：多 stream 的管理开销超过了重叠收益。

结论：只有当 kernel 计算时间与传输时间量级相近时，重叠才有收益。kernel 比传输快数百倍时，GPU compute engine 几乎全程在等数据。判断标准以 Nsight Systems 的时间线为准，不要凭 engine 数量推断收益。

### 4.4 统一 Benchmark 实测 — 与 RTX 5090 同样代码的 A100 对比

使用配套 Benchmark（与第 3 节 RTX 5090 完全相同的测试逻辑：4 streams × 256 MB, K=1024 kernel 循环）在 A100 上的结果：

| 平台                                 | 串行耗时  | 并发耗时 | 加速比    |
| ------------------------------------ | --------- | -------- | --------- |
| RTX 5090 (Gen5 ×16, 2 copy engines)  | 63.81 ms  | 27.01 ms | **2.36x** |
| A100-SXM4 (Gen4 ×16, 3 copy engines) | 120.20 ms | 68.85 ms | **1.75x** |

A100 的绝对耗时更大（PCIe Gen4 带宽 ~32 GB/s vs RTX 5090 Gen5 ~63 GB/s），但仍有 1.75x 的显著重叠。3 个 copy engine 的优势被较低的 PCIe 带宽抵消了一部分——多 engine 的优势在 **kernel/传输比较大** 时才充分体现。

---

## 5. Nsight Systems 可视化 (概念)

在 Nsight Systems 中，stream 并发效果体现为时间线上各 stream 的操作互相交错：

```text
Stream 0:  [H2D ][Kernel       ][D2H ]
Stream 1:     [H2D ][Kernel       ][D2H ]
Stream 2:        [H2D ][Kernel       ][D2H ]
Stream 3:           [H2D ][Kernel       ][D2H ]
Time:      0ms ──────────────────────────────── 27ms
```

没有 stream 的时序：

```text
Default:   [H2D0][K0][D2H0][H2D1][K1][D2H1][H2D2][K2][D2H2][H2D3][K3][D2H3]
Time:      0ms ──────────────────────────────────────────────────── 64ms
```

---

## 6. Stream 最佳实践

| 实践                             | 说明                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| 使用 `cudaMemcpyAsync`           | 非阻塞拷贝是重叠的前提                                       |
| 每个 stream 独立 buffer          | 避免 data hazard，每个 stream 有独立的 H2D/Kernel/D2H buffer |
| pinned memory (`cudaMallocHost`) | pageable 源的 async 拷贝会被 staging 阻塞，拿不到重叠        |
| 先 issue 所有 work，再同步       | issue 阶段无阻塞，GPU 调度器自行安排并发                     |
| Nsight Systems 验证              | 用时间线确认重叠确实发生，不凭 engine 数量推断               |

---

## 参考

- [CUDA Streams 详解](03_cuda_streams.md)
- [CUDA C Programming Guide: Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
