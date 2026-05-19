# PCIe 链路状态与主机-设备带宽实测

> 基于 RTX 5090 (PCIe Gen 5 x16) 实际环境。本文提供一个零依赖 CUDA 程序，测量 Host↔Device 传输带宽，并分析 PCIe 链路状态在空闲/负载下的切换。

---

## 1. 为什么 PCIe 带宽测试不同于 nvbandwidth

`nvbandwidth` 功能强大但需要编译安装（依赖 Boost、CMake、GDS 等）。以下情况可能更适合本文的零依赖方案：

- 快速验证 GPU 是否插在正确的 PCIe 槽上
- 没有 root 权限编译 nvbandwidth
- 只需要 H2D/D2H 基础数据

nvbandwidth 的详细用法见 [nvbandwidth 深度解析](01_nvbandwidth_best_practices.md)。

---

## 2. PCIe 链路状态：为什么空闲时是 Gen 1

```bash
# 空闲时查询
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
# 输出: 1, 16
```

RTX 5090 最大支持 PCIe Gen 5 x16，但空闲时运行在 Gen 1。这是 **ASPM (Active State Power Management)** 机制：GPU 无负载时降级链路省电，一旦有数据传输，自动恢复到最高速率。

### 通过 sysfs 查看

```bash
# 当前速度
cat /sys/bus/pci/devices/0000:98:00.0/current_link_speed
# 输出: 2.5 GT/s PCIe (Gen 1)

# 最大能力
cat /sys/bus/pci/devices/0000:98:00.0/max_link_speed
# 输出: 32.0 GT/s PCIe (Gen 5)
```

### 负载下的链路恢复

运行带宽测试后，链路自动恢复：

```bash
# 运行测试后查询
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
# 输出: 5, 16
```

**结论**：看到 `Gen 1` 不要惊慌，这是正常行为。`nvidia-smi` 中的 `pcie.link.gen.max` 才是真实能力上限。

---

## 3. 零依赖带宽测试程序

```bash
cat > pcie_bw_test.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(cmd) do {                                    \
    cudaError_t e = cmd;                                   \
    if (e != cudaSuccess) {                                \
        printf("CUDA error: %s\n", cudaGetErrorString(e)); \
        exit(1);                                           \
    }                                                      \
} while(0)

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main() {
    const size_t sizes[] = {
        1 * 1024 * 1024,      // 1 MB
        16 * 1024 * 1024,     // 16 MB
        64 * 1024 * 1024,     // 64 MB
        256 * 1024 * 1024,    // 256 MB
        1024 * 1024 * 1024    // 1 GB
    };
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    float *h_buf, *d_buf;
    CHECK(cudaMallocHost(&h_buf, sizes[num_sizes - 1]));
    CHECK(cudaMalloc(&d_buf, sizes[num_sizes - 1]));

    printf("%-12s | %-15s | %-15s\n", "Size", "H2D (GB/s)", "D2H (GB/s)");
    printf("-------------|------------------|------------------\n");

    for (int i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];
        cudaEvent_t start, stop;
        float ms;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Host -> Device
        cudaEventRecord(start, 0);
        CHECK(cudaMemcpy(d_buf, h_buf, n, cudaMemcpyHostToDevice));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        float h2d = (n / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

        // Device -> Host
        cudaEventRecord(start, 0);
        CHECK(cudaMemcpy(h_buf, d_buf, n, cudaMemcpyDeviceToHost));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        float d2h = (n / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

        char size_str[16];
        if (n >= 1024 * 1024 * 1024)
            snprintf(size_str, 16, "%lu GB", n / (1024 * 1024 * 1024));
        else
            snprintf(size_str, 16, "%lu MB", n / (1024 * 1024));

        printf("%-12s | %-15.2f | %-15.2f\n", size_str, h2d, d2h);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    CHECK(cudaFreeHost(h_buf));
    CHECK(cudaFree(d_buf));
    return 0;
}
EOF

nvcc -o pcie_bw_test pcie_bw_test.cu
./pcie_bw_test
```

---

## 4. 实测结果 (RTX 5090)

```text
Size         | H2D (GB/s)      | D2H (GB/s)
-------------|------------------|------------------
1 MB         | 18.28           | 32.43
16 MB        | 50.70           | 52.09
64 MB        | 52.09           | 52.92
256 MB       | 52.44           | 53.27
1 GB         | 52.50           | 53.34
```

### 4.1 趋势分析

| 传输大小 | 现象 | 原因 |
|----------|------|------|
| 1 MB | 带宽低 (18-32 GB/s) | CUDA kernel launch 开销 + 链路 ramp-up 延迟主导 |
| 16 MB+ | 带宽稳定 (~52-53 GB/s) | 传输时间主导，链路已恢复到 Gen 5 |

### 4.2 与理论值对比

PCIe Gen 5 x16 理论单向带宽：32.0 GT/s × 16 lanes × 128b/130b 编码 = **~63.0 GB/s**。

实测 ~52.5 GB/s，效率约 **83%**。损耗来自：

- 128b/130b 编码开销（已计入理论值）
- PCIe TLP header 开销
- CUDA driver 和 runtime 开销
- Host 端内存控制器带宽限制

### 4.3 D2H vs H2D

D2H 略快于 H2D (~53.3 vs ~52.5 GB/s)。这是因为 GPU 是 DMA 发起方，D2H 时 GPU 直接 push 数据，而 H2D 时需要 GPU 主动 pull。

---

## 5. 诊断 PCIe 链路问题

### 5.1 确认是否跑在预期速率

```bash
# 方法 1: 最大能力查询
nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv,noheader
# 期望: 5, 16

# 方法 2: sysfs
cat /sys/bus/pci/devices/0000:98:00.0/max_link_speed
# 期望: 32.0 GT/s PCIe (即 Gen 5)
```

### 5.2 常见问题

| 问题 | 排查命令 | 可能原因 |
|------|----------|----------|
| max 显示 Gen 3 | `nvidia-smi --query-gpu=pcie.link.gen.max` | 主板/CPU 不支持更高，或插在低代槽位 |
| width 显示 x8 | `nvidia-smi --query-gpu=pcie.link.width.max` | 插槽物理宽度不足，或 lane 被其他设备共享 |
| 带宽远低于预期 (~25 GB/s) | 运行本文测试 + 检查 Gen | 运行在 Gen 3 x16 (~16 GB/s 理论) |
| 带宽异常低 (< 5 GB/s) | 检查 Gen+Width + 测小包 | 可能在 Gen 1 x1 或其他异常状态 |

### 5.3 PCIe 错误计数器

```bash
nvidia-smi --query-gpu=pcie.replay_counter,pcie.replay_rollover_counter --format=csv
# replay 计数器持续增长 = 链路信号质量问题
```

---

## 6. 进阶：GPU-GPU 带宽（多卡环境）

单卡环境不支持 P2P 带宽测试。如有双卡以上环境，参考 [GPUDirect P2P 技术详解](../../01_hardware_architecture/gpudirect/02_gpudirect_p2p.md) 或安装 nvbandwidth 测试 `Device to Device` 带宽。
