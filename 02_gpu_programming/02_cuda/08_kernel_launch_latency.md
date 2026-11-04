# CUDA Kernel Launch 开销测量

> 基于 RTX 5090 实测。测量空 kernel 的 launch 延迟，以及不同 grid/block 配置对 launch 开销的影响。这是判断"计算该放 CPU 还是 GPU"的关键数据。
>
> **验证代码**：本文所有实测数据来自配套 Benchmark：[08_kernel_launch_latency.cu](code/08_kernel_launch_latency.cu)。

---

## 1. 为什么关心 Launch 开销

每个 CUDA kernel 的启动都有固定开销：

```text
CPU 提交命令 → CUDA Driver 处理 → GPU 调度 → 执行 → 完成通知
```

如果 kernel 计算量 < launch 开销，GPU 加速反而变成减速。本文实测单次 launch ~2.6 μs；加上一次 PCIe H2D/D2H 往返（~10 μs），一次小操作的固定开销 ~13 μs 起步，跨配置可到几十 μs。

---

## 2. 测试程序

```bash
cat > launch_lat.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(c) do {                                      \
    cudaError_t r = c;                                     \
    if (r != cudaSuccess) {                                \
        printf("Error: %s\n", cudaGetErrorString(r));      \
        exit(1);                                           \
    }                                                      \
} while(0)

__global__ void empty() {}

int main() {
    cudaEvent_t s, e;
    float t;
    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));

    // === Single empty kernel ===
    CHECK(cudaEventRecord(s, 0));
    empty<<<1, 1>>>();
    CHECK(cudaEventRecord(e, 0));
    CHECK(cudaEventSynchronize(e));
    CHECK(cudaEventElapsedTime(&t, s, e));
    printf("Empty kernel (1 thread): %.2f us\n", t * 1000);

    // === Average over 10000 launches ===
    for (int i = 0; i < 100; i++) empty<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(s, 0));
    for (int i = 0; i < 10000; i++)
        empty<<<1, 1>>>();
    CHECK(cudaEventRecord(e, 0));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventElapsedTime(&t, s, e));

    printf("Average (10k iters): %.3f us\n", t * 1000 / 10000);

    // === Different block counts ===
    int blocks[] = {1, 32, 256, 1024, 4096, 16384};
    printf("\n%-8s | %-22s\n", "Blocks", "Time (ms, 1000 launches)");
    printf("---------|----------------\n");
    for (int i = 0; i < 6; i++) {
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(s, 0));
        for (int j = 0; j < 1000; j++)
            empty<<<blocks[i], 256>>>();
        CHECK(cudaEventRecord(e, 0));
        CHECK(cudaEventSynchronize(e));
        CHECK(cudaEventElapsedTime(&t, s, e));
        printf("%-8d | %-15.2f\n", blocks[i], t);
    }

    CHECK(cudaEventDestroy(s));
    CHECK(cudaEventDestroy(e));
    return 0;
}
EOF

nvcc -o launch_lat launch_lat.cu
./launch_lat
```

---

## 3. RTX 5090 实测

> 数据出自内联脚本 `launch_lat.cu`（blocks × 256 线程）；§3.1 的 A100 表出自配套 `08_kernel_launch_latency.cu`（7 组 blocks×threads 配置）。两套程序配置集不同，数字不能逐行互换。

```text
Empty kernel (1 thread): 14363.71 us     ← 首次 launch，含 driver 初始化
Average (10k iters):     2.647 us        ← 稳定后每次 launch 开销

Blocks   | Time (ms, 1000 launches)   ← 每行 = 1000 次 launch 的总毫秒数
---------|----------------------------
1        | 2.34                        ← 每次 ~2.34 μs
32       | 2.35
256      | 2.36
1024     | 2.38
4096     | 2.40
16384    | 8.24                        ← 每次 ~8.2 μs，16k blocks 触发额外调度开销
```

### 3.1 A100 实测（配套 Benchmark）

配套 `08_kernel_launch_latency.cu` 在 A100 上的结果（每次 launch 均摊，μs）：

```text
Blocks  | Threads | A100 (μs)
--------|---------|----------
1       | 1       | 2.33
1       | 32      | 2.33
1       | 128     | 2.34
1       | 512     | 2.36
1       | 1024    | 2.34
10      | 256     | 2.33
100     | 256     | 2.31
```

A100 稳定 launch 开销约 **2.33 μs**，与 RTX 5090 的 ~2.6 μs 同一量级。两组实测中 blocks/threads 配置的差异对 launch 延迟的影响都小于事件计时的分辨率（约 0.5 μs）——launch 开销主要来自 CPU→GPU 的命令提交路径，而非 GPU 端调度。

### 3.2 关键解读

| 指标          | RTX 5090    | A100       | 说明                                |
| ------------- | ----------- | ---------- | ----------------------------------- |
| 首次 launch   | **14.4 ms** | —          | 含 CUDA driver 初始化、context 建立 |
| 稳定 launch   | **2.6 μs**  | **2.3 μs** | 正常 kernel 提交延迟                |
| block 数 < 4k | 每次 ~2.4 μs | 每次 ~2.3 μs | 调度负载可忽略                   |
| block 数 16k  | 每次 ~8.2 μs | —          | 大量 block 的硬件调度队列开销       |

---

## 4. CPU vs GPU 决策边界

以 RTX 5090 的 **2.6 μs** launch + ~10 μs PCIe H2D/D2H = **~13 μs** 总开销为基准：

```text
假设: CPU 单核 4 GHz, 4 FLOPs/cycle = 16 GFLOPS
      GPU RTX 5090 ≈ ~100 TFLOPS (BF16 Tensor Core)

不同计算密度下 GPU 胜出的最小数据量:
  算数密度 1 op/el   → GPU 胜出 > 200K 元素 (13μs × 16GFLOPs = 208K ops)
  算数密度 10 ops/el  → GPU 胜出 > 20K 元素
  算数密度 1000 ops/el → GPU 胜出 > 200 元素 (矩阵乘法等)
```

| 场景                    | 数据量 | 推荐 | 原因                        |
| ----------------------- | ------ | ---- | --------------------------- |
| 逐元素 ReLU (1 op/el)   | < 200K | CPU  | launch + PCIe > 计算        |
| 向量内积 (2 ops/el)     | > 100K | GPU  | 临界值约为 ReLU 场景的一半  |
| 矩阵乘法 1024×1024      | 任何   | GPU  | 计算时间远大于开销          |

**经验法则**：GPU kernel 总时间 < **20 μs** 则 CPU 可能更快；批量处理摊薄 13 μs 的固定开销——一次传输 100K 元素只付一次往返延迟，100 次 1K 传输要付 100 次。

---

## 5. 影响 Launch 开销的因素

| 因素                | 影响                                                    |
| ------------------- | ------------------------------------------------------- |
| CUDA context 初始化 | 首次 launch ~10-15 ms，后续 ~2-3 μs                     |
| Block 数量          | < 4096 无影响，> 16k 开始显著                           |
| Shared memory 大小  | 每个 block 的 shared mem 分配有微小开销                 |
| Stream 并发         | 多 stream 的 launch 可 pipeline，但单个 launch 延迟不变 |
| CUDA Graph          | 显著降低：通过预录制消除单次 launch 开销                |

### 5.1 用 CUDA Graph 消除 Launch 开销

对于重复执行的小 kernel，CUDA Graph 将多次 launch 合并为一次：

```c
cudaGraph_t graph;
cudaGraphExec_t instance;

// 录制
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (int i = 0; i < 1000; i++)
    kernel<<<1, 256, 0, stream>>>();
cudaStreamEndCapture(stream, &graph);

// 实例化
cudaGraphInstantiate(&instance, graph, 0);

// 执行（一次 launch 替代 1000 次）
cudaGraphLaunch(instance, stream);
```

---

## 6. 与 PCIe 延迟的关系

总 GPU 操作延迟 ≈ Launch 开销 + PCIe 传输时间：

| 操作                    | 延迟                                        |
| ----------------------- | ------------------------------------------- |
| Kernel launch           | **2.6 μs**                                  |
| PCIe H2D (1 KB)         | **11.1 μs**（含 ~10 μs 链路往返固定延迟）   |
| PCIe D2H (1 KB)         | **10-12 μs**                                |
| nvbandwidth latency 用例 | **621 ns**（纯 PCIe 链路往返口径，不含 launch 与驱动提交路径） |

> 每次 H2D/D2H 往返都有 ~10 μs 的固定延迟（驱动提交路径 + 链路往返），之后每字节传输接近线速。这是批量传输比逐次小传输高效几个数量级的原因。

---

## 参考

- [CUDA Streams 并发实战](07_cuda_streams_concurrency.md)
- [PCIe 链路状态与带宽实测](../04_profiling/02_pcie_bandwidth_measurement.md)
- [CUDA C Programming Guide: Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
