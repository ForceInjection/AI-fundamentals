# GPU 显存带宽测试：片内 vs 片外

> 基于 RTX 5090 (GDDR7, 512-bit, 1792 GB/s 理论带宽) 和 A100-SXM4-80GB (HBM2e, 5120-bit, 2039 GB/s 理论带宽) 双平台实测。本文测量 device-to-device 内部带宽并与 PCIe 传输形成完整对比。

---

## 1. 为什么片内带宽如此重要

GPU 显存带宽（片内）和 PCIe 带宽（片外）之间的差距是 AI 系统设计的核心矛盾。以两个典型平台为例：

```text
A100-SXM4:  HBM2e, 5120-bit, 2039 GB/s  vs  PCIe Gen 4, ~28 GB/s  →  差距 ~73 倍
RTX 5090:   GDDR7,  512-bit, 1792 GB/s  vs  PCIe Gen 5, ~53 GB/s  →  差距 ~34 倍
```

这个数十倍的差距决定了深度学习系统的几乎所有设计选择：

- **训练**：必须把所有参数、梯度、优化器状态放在 GPU 显存中。一次 PCIe 往返就可能让训练吞吐腰斩。
- **推理**：KV Cache 必须留在显存或通过高速方案（NVLink/NVSwitch/LMCache）在 GPU 间搬运——卸到 CPU 内存是下下策。
- **数据加载**：训练数据的 I/O 必须异步 prefetch 到 GPU 显存，绝不能在主循环中同步 H2D。

GPU 内部的 `cudaMemcpyDeviceToDevice` 走的是**内存控制器 → DRAM → 内存控制器**路径，不经过 PCIe 链路。测试 D2D 带宽可以验证：

1. HBM2e/GDDR7 的实际可用带宽（与理论值比较）
2. L2 Cache 对不同传输大小的加速效果
3. `cudaMemcpy` 是否用了正确的 copy engine 路径

---

## 2. 带宽分层全景

| 路径                     | 理论带宽   | 实测带宽         | 效率    |
| ------------------------ | ---------- | ---------------- | ------- |
| **HBM2e 片内 (A100)**    | 2039 GB/s  | ~1188 GB/s (4MB) | 58%     |
| **GDDR7 片内 (RTX5090)** | 1792 GB/s  | 762-1341 GB/s    | 43-75%  |
| **PCIe Gen 4 (A100)**    | ~31.5 GB/s | ~25-28 GB/s      | ~80-89% |
| **PCIe Gen 5 (RTX5090)** | ~63 GB/s   | 52-56 GB/s       | 83-89%  |
| **A100 片内/片外比**     | **~65:1**  | **~42-47:1**     | —       |
| **RTX 5090 片内/片外比** | **~28:1**  | **~14-24:1**     | —       |

带宽差距的本质：A100 HBM2e ≈ 2.0 TB/s 通过 PCIe Gen 4 与 CPU 通信 ≈ 28 GB/s，相差 **73 倍**。RTX 5090 GDDR7 ≈ 1.8 TB/s 通过 PCIe Gen 5 与 CPU 通信 ≈ 53 GB/s，相差 **34 倍**。A100 片内/片外差距更大，但因为 HBM2e 带宽绝对值和 NVLink 的存在，多卡训练场景下数据搬运效率远高于消费级 GPU。无论哪种 GPU，**深度学习训练/推理中数据应尽可能驻留在 GPU 显存**。

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

### 4.3 A100-SXM4-80GB 实测（官方 transpose sample）

使用 NVIDIA 官方 cuda-samples 13.1 中的 `transpose` (6_Performance) 在 A100 上测试，1024×1024 fp32 矩阵 (4 MB)：

```text
GPU: NVIDIA A100-SXM4-80GB (CC 8.0, HBM2e 5120-bit, 1593 MHz)

transpose simple copy       , Throughput = 1188.38 GB/s
transpose shared memory copy, Throughput = 1130.28 GB/s
transpose naive             , Throughput =  215.28 GB/s
transpose coalesced         , Throughput =  530.19 GB/s
transpose optimized         , Throughput = 1168.36 GB/s
transpose coarse-grained    , Throughput = 1135.33 GB/s
transpose fine-grained      , Throughput = 1137.02 GB/s
transpose diagonal          , Throughput = 1135.33 GB/s
```

| 实现                | 带宽      | % of 理论峰值 (2039 GB/s) |
| ------------------- | --------- | ------------------------- |
| simple copy         | 1188 GB/s | 58.3%                     |
| optimized transpose | 1168 GB/s | 57.3%                     |
| coalesced           | 530 GB/s  | 26.0%                     |
| naive               | 215 GB/s  | 10.5%                     |

**关键观察**：

- 4 MB 矩阵完全容纳在 A100 的 40 MB L2 cache 中，因此所有实现都受益于 cache 命中
- Simple copy 和 optimized 达到 ~1188 GB/s (~58% 理论峰值)，主要受限于 `cudaMemcpy` 的 copy engine 路径而非 SM load/store
- 与 RTX 5090 比较：5120-bit HBM2e (A100) vs 512-bit GDDR7 (RTX 5090)，位宽 10 倍差距但 A100 时钟低约 9 倍，最终理论带宽差距仅 ~14%
- A100 的 D2D 带宽优势主要在大矩阵 (> 40 MB) 场景——更宽的位宽意味着更平稳的 DRAM page miss 处理

> **nvbandwidth 验证**：在 A100 上安装 nvbandwidth 可获得更权威的基准数据（device_local_copy）。本文 transpose 结果受限于 4 MB 矩阵和 L2 cache 效应，不代表全范围 D2D 带宽。

---

## 5. 与 PCIe 带宽的完整对比

| 传输方向     | 工具           | 1 MB       | 64 MB    | 1 GB          |
| ------------ | -------------- | ---------- | -------- | ------------- |
| **H2D**      | nvbandwidth CE | —          | —        | **56.3 GB/s** |
| **D2H**      | nvbandwidth CE | —          | —        | **56.8 GB/s** |
| **D2D**      | cudaMemcpy     | 33.7       | **1341** | 707.9         |
| **D2D**      | nvbandwidth    | —          | —        | **762.3**     |
| **D2D A100** | transpose (SM) | 1188 (4MB) | —        | —             |

**关键数字**：

- GPU 内部拷贝比 PCIe 传输快 **13-24 倍**（RTX 5090: 762 vs 56 GB/s）；A100 差距更大，约 **42-47 倍**（1188 vs 28 GB/s）
- 如果你的算法需要频繁 H2D/D2H，考虑 Unified Memory + prefetch（见 [CUDA NUMA API](../02_cuda/05_cuda_numa_api.md)）

---

## 6. 编程启示

```text
✅ 尽量把数据和计算留在 GPU 显存
✅ 避免训练循环中的 H2D/D2H（A100: ~28 GB/s vs 内部 ~1188 GB/s, 差距 42×）
✅ A100 的宽位宽 HBM2e 对大数据集更友好（5120-bit vs 512-bit GDDR7）
✅ 使用 cudaMallocManaged + cudaMemPrefetchAsync 做隐式数据迁移
✅ 用 nvbandwidth 做权威基准测试，cudaMemcpy 测趋势即可
```

---

## 参考

- [PCIe 链路状态与带宽实测](02_pcie_bandwidth_measurement.md)
- [nvbandwidth 深度解析](01_nvbandwidth_best_practices.md)
- [CUDA NUMA API 编程实践](../02_cuda/05_cuda_numa_api.md)
