# GPU 显存带宽测试：片内 vs 片外

> 基于 RTX 5090 (GDDR7, 512-bit, 1792 GB/s 理论带宽) 实测。本文测量 device-to-device 内部带宽并与 PCIe 传输形成完整对比。

---

## 1. 为什么片内带宽如此重要

GPU 显存带宽（片内）和 PCIe 带宽（片外）之间的差距是 AI 系统设计的核心矛盾：

```text
GPU 内部:  GDDR7, 512-bit, 1792 GB/s
PCIe 5.0:  ×16, ~63 GB/s
差距:      约 28 倍
```

这 28 倍的差距决定了深度学习系统的几乎所有设计选择：

- **训练**：必须把所有参数、梯度、优化器状态放在 GPU 显存中。一次 PCIe 往返就可能让训练吞吐腰斩。
- **推理**：KV Cache 必须留在显存或通过高速方案（NVLink/NVSwitch/LMCache）在 GPU 间搬运——卸到 CPU 内存是下下策。
- **数据加载**：训练数据的 I/O 必须异步 prefetch 到 GPU 显存，绝不能在主循环中同步 H2D。

GPU 内部的 `cudaMemcpyDeviceToDevice` 走的是**内存控制器 → DRAM → 内存控制器**路径，不经过 PCIe 链路。测试 D2D 带宽可以验证：

1. GDDR7 的实际可用带宽（与理论值比较）
2. L2 Cache 对不同传输大小的加速效果
3. `cudaMemcpy` 是否用了正确的 copy engine 路径

---

## 2. 带宽分层全景

| 路径                  | 理论带宽  | 实测带宽      | 效率   |
| --------------------- | --------- | ------------- | ------ |
| **HBM (片内)**        | 1792 GB/s | 762-1341 GB/s | 43-75% |
| **PCIe Gen 5 (片外)** | ~63 GB/s  | 52-56 GB/s    | 83-89% |
| **片内/片外比**       | **~28:1** | **~14-24:1**  | —      |

带宽差距的本质：GPU 显存宽带 ≈ 1.8 TB/s，通过 PCIe 与 CPU 通信 ≈ 56 GB/s，相差一个数量级。这解释了为什么**深度学习训练/推理中数据应尽可能驻留在 GPU 显存**。

---

## 3. 测试程序

```bash
cat > hbm_bw.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(c) do {                                      \
    cudaError_t e = c;                                     \
    if (e != cudaSuccess) {                                \
        printf("Error: %s\n", cudaGetErrorString(e));      \
        exit(1);                                           \
    }                                                      \
} while(0)

int main() {
    const size_t sizes[] = {
        1 * 1024 * 1024,      // 1 MB
        16 * 1024 * 1024,     // 16 MB
        64 * 1024 * 1024,     // 64 MB
        256 * 1024 * 1024,    // 256 MB
        1024 * 1024 * 1024    // 1 GB
    };
    const int n = sizeof(sizes) / sizeof(sizes[0]);

    float *d_src, *d_dst;
    CHECK(cudaMalloc(&d_src, sizes[n - 1]));
    CHECK(cudaMalloc(&d_dst, sizes[n - 1]));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int theory_bw = 2.0 * prop.memoryClockRate
                  * (prop.memoryBusWidth / 8) / 1.0e6;
    printf("GPU: %s\n", prop.name);
    printf("Memory clock: %.1f MHz | Bus: %d-bit\n",
           (float)prop.memoryClockRate / 1000.0,
           prop.memoryBusWidth);
    printf("Theoretical peak: %d GB/s\n\n", theory_bw);

    printf("%-12s | %-15s | %-15s\n",
           "Size", "D2D (GB/s)", "% of peak");
    printf("-------------|------------------|------------------\n");

    for (int i = 0; i < n; i++) {
        size_t sz = sizes[i];
        cudaEvent_t start, stop;
        float ms;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        CHECK(cudaMemcpy(d_dst, d_src, sz, cudaMemcpyDeviceToDevice));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        float bw = (sz / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

        char b[16];
        if (sz >= 1073741824)
            snprintf(b, 16, "%lu GB", sz / 1073741824);
        else
            snprintf(b, 16, "%lu MB", sz / 1048576);

        printf("%-12s | %-15.2f | %-15.1f%%\n",
               b, bw, bw / theory_bw * 100);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    return 0;
}
EOF

nvcc -o hbm_bw hbm_bw.cu
./hbm_bw
```

---

## 4. 实测结果

**RTX 5090 (GDDR7, 512-bit, 14001 MHz)**：

```text
Size         | D2D (GB/s)     | % of peak
-------------|------------------|------------------
1 MB         | 33.72           | 1.9%
16 MB        | 887.78          | 49.5%
64 MB        | 1341.43         | 74.9%
256 MB       | 779.22          | 43.5%
1 GB         | 707.86          | 39.5%
```

**nvbandwidth 验证** (单向 device_local_copy)：

```text
762.33 GB/s
```

### 4.1 趋势解读

| 区间     | 现象                 | 原因                         |
| -------- | -------------------- | ---------------------------- |
| 1 MB     | 33.7 GB/s (1.9%)     | kernel launch 开销主导       |
| 16-64 MB | 888-1341 GB/s (峰值) | 适合 L2 cache (96 MB) 命中   |
| 256 MB+  | 707-779 GB/s         | 超出 L2，DRAM page miss 影响 |

### 4.2 为什么达不到理论值

- **cudaMemcpy D2D 瓶颈**：`cudaMemcpy` 走的是 copy engine 路径，不是 SM 的 load/store，受限于内存控制器的实际带宽
- **L2 Cache 效应**：64 MB 时数据部分命中 L2 (96 MB)，带宽最高 (1341 GB/s)；256 MB+ 完全 miss，降到 ~750 GB/s
- **DRAM 时序开销**：行激活、预充电等开销占理论峰值的 20-30%

---

## 5. 与 PCIe 带宽的完整对比

| 传输方向 | 工具           | 1 MB | 64 MB    | 1 GB          |
| -------- | -------------- | ---- | -------- | ------------- |
| **H2D**  | nvbandwidth CE | —    | —        | **56.3 GB/s** |
| **D2H**  | nvbandwidth CE | —    | —        | **56.8 GB/s** |
| **D2D**  | cudaMemcpy     | 33.7 | **1341** | 707.9         |
| **D2D**  | nvbandwidth    | —    | —        | **762.3**     |

**关键数字**：

- GPU 内部拷贝比 PCIe 传输快 **13-24 倍**（762 vs 56 GB/s）
- 如果你的算法需要频繁 H2D/D2H，考虑 Unified Memory + prefetch（见 [CUDA NUMA API](../02_cuda/05_cuda_numa_api.md)）

---

## 6. 编程启示

```text
✅ 尽量把数据和计算留在 GPU 显存
✅ 避免训练循环中的 H2D/D2H（每步 ~56 GB/s vs 内部 ~800 GB/s）
✅ 使用 cudaMallocManaged + cudaMemPrefetchAsync 做隐式数据迁移
✅ 用 nvbandwidth 做权威基准测试，cudaMemcpy 测趋势即可
```

---

## 参考

- [PCIe 链路状态与带宽实测](02_pcie_bandwidth_measurement.md)
- [nvbandwidth 深度解析](01_nvbandwidth_best_practices.md)
- [CUDA NUMA API 编程实践](../02_cuda/05_cuda_numa_api.md)
