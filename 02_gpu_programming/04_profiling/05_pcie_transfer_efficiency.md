# PCIe 传输效率曲线：从小包到大块

> 基于 RTX 5090 (PCIe Gen 5 x16) 实测。展示 1 KB → 1 GB 区间 PCIe H2D/D2H 的带宽爬升曲线，揭示"多大才够"的效率拐点。

---

## 1. 为什么关心小包传输

来自 [Kernel Launch 开销测量](../02_cuda/08_kernel_launch_latency.md) 的结论：单次 launch ~2.6 μs。但如果你的 H2D 传输只有 1 KB，耗时 ~11 μs——两者合计 ~14 μs，而 1 KB 的计算可能只需几十 ns。

理解**传输效率与大小的关系**，是决定"批量还是流式"的核心依据。

---

## 2. 测试程序

```bash
cat > pcie_micro.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(c) do {                                      \
    cudaError_t r = c;                                     \
    if (r != cudaSuccess) {                                \
        printf("Error: %s\n", cudaGetErrorString(r));      \
        exit(1);                                           \
    }                                                      \
} while(0)

int main() {
    const size_t sizes[] = {
        1024,             // 1 KB
        4 * 1024,         // 4 KB
        16 * 1024,        // 16 KB
        64 * 1024,        // 64 KB
        256 * 1024,       // 256 KB
        512 * 1024,       // 512 KB
        1024 * 1024       // 1 MB
    };
    const int n = 7;

    float *h, *d;
    cudaEvent_t s, e;
    float t;
    CHECK(cudaMallocHost(&h, 2 * 1024 * 1024));
    CHECK(cudaMalloc(&d, 2 * 1024 * 1024));

    printf("%-10s | %-12s | %-12s | %-12s\n",
           "Size", "H2D (GB/s)", "D2H (GB/s)", "Lat (us)");
    printf("-----------|--------------|--------------|-------------\n");

    for (int i = 0; i < n; i++) {
        size_t sz = sizes[i];
        int reps = sz < 65536 ? 100000 :
                   sz < 262144 ? 10000 :
                   sz < 1048576 ? 5000 : 2000;

        cudaEventCreate(&s);
        cudaEventCreate(&e);

        // H2D bandwidth
        cudaEventRecord(s, 0);
        for (int j = 0; j < reps; j++)
            CHECK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
        cudaEventRecord(e, 0);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&t, s, e);
        float h2d = (sz * reps / (t / 1000.0))
                  / (1024.0 * 1024.0 * 1024.0);

        // D2H bandwidth
        cudaEventRecord(s, 0);
        for (int j = 0; j < reps; j++)
            CHECK(cudaMemcpy(h, d, sz, cudaMemcpyDeviceToHost));
        cudaEventRecord(e, 0);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&t, s, e);
        float d2h = (sz * reps / (t / 1000.0))
                  / (1024.0 * 1024.0 * 1024.0);

        // Single transfer latency
        cudaEventRecord(s, 0);
        CHECK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
        cudaEventRecord(e, 0);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&t, s, e);
        float lat = t * 1000;

        char b[16];
        if (sz >= 1048576)
            snprintf(b, 16, "%lu MB", sz / 1048576);
        else if (sz >= 1024)
            snprintf(b, 16, "%lu KB", sz / 1024);
        printf("%-10s | %-12.2f | %-12.2f | %-12.1f\n",
               b, h2d, d2h, lat);

        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }

    CHECK(cudaFreeHost(h));
    CHECK(cudaFree(d));
    return 0;
}
EOF

nvcc -o pcie_micro pcie_micro.cu
./pcie_micro
```

---

## 3. RTX 5090 实测曲线

```text
Size       | H2D (GB/s)   | D2H (GB/s)   | Lat (us)
-----------|--------------|--------------|-------------
1 KB       | 0.16         | 0.17         | 11.1
4 KB       | 0.61         | 0.68         | 16.4
16 KB      | 2.08         | 2.58         | 12.2
64 KB      | 7.93         | 8.95         | 10.0        ← 最低延迟
256 KB     | 21.90        | 24.49        | 12.6
512 KB     | 30.84        | 33.56        | 19.4
1 MB       | 38.91        | 40.74        | 28.8
```

### 3.1 完整曲线（1 KB - 1 GB）

> 1 KB - 1 MB 来自本节微传输测试（多次迭代取均值），16 MB - 1 GB 来自 [PCIe 链路状态与带宽实测](02_pcie_bandwidth_measurement.md)（单次大块传输）。两者合并形成完整带宽曲线。

```text
Size       | H2D (GB/s)   | D2H (GB/s)   | Lat (us)
-----------|--------------|--------------|-------------
1 KB       | 0.16         | 0.17         | 11.1
4 KB       | 0.61         | 0.68         | 16.4
16 KB      | 2.08         | 2.58         | 12.2
64 KB      | 7.93         | 8.95         | 10.0
256 KB     | 21.90        | 24.49        | 12.6
512 KB     | 30.84        | 33.56        | 19.4
1 MB       | 38.91        | 40.74        | 28.8
16 MB      | 50.70        | 52.09        | —            ← 接近饱和
64 MB      | 52.09        | 52.92        | —            ← 饱和
1 GB       | 52.50        | 53.34        | —            ← 峰值
```

---

## 4. 效率拐点分析

### 4.1 带宽效率 (% of peak 52.5 GB/s)

| 大小   | H2D 效率  | 瓶颈                                         |
| ------ | --------- | -------------------------------------------- |
| 1 KB   | **0.3%**  | PCIe TLP header 开销 (~28B per TLP) 占比极大 |
| 4 KB   | 1.2%      | 仍有大量 header 开销                         |
| 16 KB  | 4.0%      | —                                            |
| 64 KB  | 15.1%     | **延迟最低点 (10.0 μs)**                     |
| 256 KB | 41.7%     | —                                            |
| 1 MB   | 74.1%     | —                                            |
| 16 MB  | **96.6%** | **接近饱和**                                 |

### 4.2 三阶段模型

这个曲线反映了 PCIe 传输的两个本质瓶颈：

1. **固定开销**（~10 μs）：每次 `cudaMemcpy` 都需要 TLP 打包、PCIe 链路仲裁、ACK 返回——这些开销与传输大小无关，是纯 latency 成本。
2. **带宽开销**（~19 ns/B at Gen 5 ×16）：一旦传输开始，数据以 ~52.5 GB/s 的速率流动。这是纯 throughput 成本。

总时间 = 固定开销 + 数据量 / 带宽。小传输时固定开销主导，大传输时带宽主导。

```text
阶段 1: 延迟主导 (1-64 KB)
  📉 带宽从 0.16 → 8 GB/s
  📍 每次传输 ~10 μs 固定开销（TLP 打包 + PCIe 链路往返）
  💡 启示：拼接多个小传输为一个批量传输

阶段 2: 混合 (64 KB - 16 MB)
  📈 带宽从 8 → 50 GB/s
  📍 传输时间开始主导，但还未完全饱和 PCIe 管道
  💡 启示：256 KB+ 已可接受，1 MB+ 效率 > 74%

阶段 3: 饱和 (16 MB+)
  📊 带宽 > 50 GB/s，接近理论极限 (~63 GB/s)
  📍 PCIe 管道填满，效率 > 95%
  💡 启示：16 MB+ 无需优化，已达硬件极限
```

---

## 5. 实用建议

| 场景         | 建议最小传输大小 | 原因                      |
| ------------ | ---------------- | ------------------------- |
| 超低延迟推理 | 64 KB            | 延迟最低 (10 μs)          |
| 高吞吐训练   | 16 MB+           | 带宽 > 96%                |
| 折中方案     | 256 KB - 1 MB    | 带宽 42-74%，延迟可接受   |
| 避免         | < 4 KB           | 效率 < 2%，浪费 PCIe 带宽 |

---

## 6. 与 Kernel 开销联动

从 [Kernel Launch 开销](../02_cuda/08_kernel_launch_latency.md)：单次 launch ~2.6 μs。

总开销 = launch + PCIe 传输：

| 计算量    | 最优执行位置 | 原因                          |
| --------- | ------------ | ----------------------------- |
| < 10 μs   | **CPU**      | launch + H2D + D2H > 计算本身 |
| 10-100 μs | GPU (批量)   | CPU 做 H2D batching           |
| > 100 μs  | **GPU**      | 计算时间主导，GPU 加速明显    |

---

## 参考

- [PCIe 链路状态与带宽实测](02_pcie_bandwidth_measurement.md)
- [Kernel Launch 开销测量](../02_cuda/08_kernel_launch_latency.md)
- [PCIe 技术大全](../../01_hardware_architecture/pcie/01_pcie_comprehensive_guide.md)
