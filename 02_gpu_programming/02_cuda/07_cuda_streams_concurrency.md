# CUDA Streams 并发实战

> 基于 RTX 5090 (2 个 async copy engine) 实测。本文演示 CUDA Streams 如何实现 H2D + Kernel + D2H 重叠执行，并测量实际加速比。

---

## 1. 背景

### 1.1 Default Stream vs Explicit Streams

CUDA 的默认 stream（stream 0）是**同步的**——所有操作按提交顺序串行执行：H2D0 → Kernel0 → D2H0 → H2D1 → Kernel1 → D2H1。这意味着 GPU 的 compute engine 在数据传输时空闲，copy engine 在 kernel 执行时空闲——两个 engine 交替闲置，硬件利用率低下。

显式 stream（`cudaStreamCreate`）打破了这种串行约束。每个 stream 内部仍然保持顺序，但**不同 stream 之间可以并发**。当 stream 0 的 kernel 在执行时，stream 1 的 H2D 可以同时进行——GPU 的三个硬件引擎（2 个 copy engine + 1 个 compute engine）可以同时忙。

### 1.2 硬件并发能力

RTX 5090 有 **2 个 async copy engine**，理论最多可实现 2 路并发数据传输。加上 compute engine，最多 3 个操作可同时进行。但实际并发度受限于：

- **PCIe 带宽共享**：2 个 copy engine 共享同一 PCIe Gen 5 ×16 链路（~63 GB/s 单向理论值）。如果两个 engine 同时做 H2D，带宽对半分。
- **H2D 和 D2H 的方向冲突**：H2D（写 GPU）和 D2H（读 GPU）共享 PCIe 双向带宽。同时做 H2D + D2H 时，双向带宽可能小于两个单向之和（nvbandwidth 实测：单向 56.3 GB/s，双向各 50.3 GB/s）。
- **Kernel 执行时长**：只有当 kernel 执行时间与数据传输时间大致匹配时，才能最大化重叠。如果 kernel 太短，数据传输追不上；如果 kernel 太长，copy engine 闲置。

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

- **D2H 和 H2D 共享 copy engine**：4 个 stream 的 8 次传输竞争 2 个 engine
- **PCIe 半双工特性**：同时 H2D + D2H 需要 PCIe 双向带宽（nvbandwidth 实测双向 ~50 GB/s vs 单向 ~56 GB/s）
- **Kernel 执行时间不完美匹配**：dummy kernel 时长可能不精确等于传输时间

---

## 4. Nsight Systems 可视化 (概念)

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

## 5. Stream 最佳实践

| 实践 | 说明 |
|------|------|
| 使用 `cudaMemcpyAsync` | 非阻塞拷贝是重叠的前提 |
| 每个 stream 独立 buffer | 避免 data hazard，每个 stream 有独立的 H2D/Kernel/D2H buffer |
| pinned memory (`cudaMallocHost`) | 普通 pageable memory 无法异步 H2D |
| 先 issue 所有 work，再同步 | issue 阶段无阻塞，GPU 调度器自行安排并发 |
| Nsight Systems 验证 | 眼见为实——工具确认重叠是否真的发生 |

---

## 参考

- [CUDA Streams 详解](03_cuda_streams.md)
- [CUDA C Programming Guide: Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
