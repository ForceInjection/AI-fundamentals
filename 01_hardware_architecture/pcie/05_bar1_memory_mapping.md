# GPU BAR1 内存映射机制

> 基于 RTX 5090 (BAR1: 256 MB) 实测。BAR1 是 GPU 通过 PCIe 暴露给 Host 的内存窗口，直接影响 Unified Memory 和 P2P 性能。

---

## 1. BAR 基础

GPU 通过 PCIe 暴露两类 BAR (Base Address Register)：

| BAR      | 类型            | 大小 (RTX 5090) | 用途                                                                                        |
| -------- | --------------- | --------------- | ------------------------------------------------------------------------------------------- |
| **BAR0** | MMIO 寄存器映射 | 32 MB           | 驱动控制和配置 (NVML 指令通路)。NVML (NVIDIA Management Library) 是 nvidia-smi 的底层 C API |
| **BAR1** | GPU 显存窗口    | 256 MB          | Host 直接映射 GPU 显存                                                                      |

BAR1 是 Host 通过 PCIe **直接访问 GPU 显存**的窗口。CUDA Unified Memory、GPUDirect P2P 都依赖 BAR1。

---

## 2. 查看 BAR1 使用情况

```bash
nvidia-smi -q -i 0 | grep -A4 "BAR1"
```

RTX 5090 输出：

```text
BAR1 Memory Usage
    Total   : 256 MiB
    Used    : 1 MiB         ← 空闲时几乎不用
    Free    : 255 MiB
```

### 2.1 为什么 BAR1 只有 256 MB

| GPU 类型               | BAR1 大小 | 原因                           |
| ---------------------- | --------- | ------------------------------ |
| RTX 5090 (消费级)      | 256 MB    | 消费级固件限制                 |
| A100 / H100 (数据中心) | 64 GB+    | 支持大量并发 UM 映射           |
| GH200 (Grace Hopper)   | 512 GB    | C2C 直连，无传统 PCIe BAR 限制 |

消费级 GPU 的 BAR1 偏小，意味着**同一时刻只能映射 256 MB 的 Unified Memory 到 Host 可见**。

---

## 3. BAR1 不足时的影响

### 3.1 Unified Memory 性能退化

当申请 `cudaMallocManaged` 超过 BAR1 大小时：

- < 256 MB：GPU 显存直接映射到 Host 地址空间 → 最快
- > 256 MB：驱动需要动态迁移页面 → 触发 page fault → 性能退化

```c
// 小块 UM：全部在 BAR1 窗口内 → 快速
cudaMallocManaged(&data, 200 * 1024 * 1024);  // 200 MB → 快

// 大块 UM：超出 BAR1 → 页面迁移开销
cudaMallocManaged(&data, 2ULL * 1024 * 1024 * 1024);  // 2 GB → 慢
```

### 3.2 GPUDirect P2P 受限

单卡 RTX 5090 不支持 P2P，但在多卡消费级 GPU 环境中，BAR1 不足也会限制 P2P 映射窗口。

---

## 4. Resizable BAR (ReBAR)

### 4.1 什么是 ReBAR

Resizable BAR 是 PCIe 3.0+ 规范定义的功能，允许 BIOS/驱动协商扩大 BAR 大小。传统 BAR 固定为 256 MB，ReBAR 可扩展到整个显存。

### 4.2 检查 ReBAR 状态

```bash
# 查看 GPU 所有 PCIe BAR 资源
cat /sys/bus/pci/devices/0000:98:00.0/resource
# 输出示例:
# 0x00000000d4000000 0x00000000d7ffffff 0x0000000000040200  ← BAR0 (MMIO, 64 MB)
# 0x0000003800000000 0x000000381fffffff 0x000000000014220c  ← BAR1 (显存窗口)
# ...
```

各字段解读：

- `d4000000` — BAR 起始物理地址
- `d7ffffff` — BAR 结束物理地址
- `0x40200` — 属性标志（bit 1 = memory space, bit 9 = prefetchable, bit 18 = 64-bit）

BAR1 的大小 = `结束 - 起始 + 1`。如果 BAR1 大小等于全部显存（如 32 GB），则 ReBAR 已启用。

```bash
# dmesg 中查找 ReBAR 信息
dmesg | grep -i "BAR\|rebar\|resizable"
```

### 4.3 ReBAR 要求

| 条件      | 说明                              |
| --------- | --------------------------------- |
| BIOS 支持 | 需 Above 4G Decoding + ReBAR 开启 |
| GPU 支持  | RTX 30 系列+、A100+               |
| 驱动支持  | 525.xx+ (Linux)                   |

在云环境中（如本 Seetacloud），ReBAR 状态取决于宿主机 BIOS，容器内通常无法修改。

---

## 5. BAR1 vs FB vs System Memory

| 路径                | 带宽             | 延迟    | 容量   |
| ------------------- | ---------------- | ------- | ------ |
| GPU → FB (HBM/GDDR) | ~1792 GB/s       | ~100 ns | 32 GB  |
| Host → BAR1 (PCIe)  | ~56 GB/s         | ~620 ns | 256 MB |
| Host → System RAM   | ~100 GB/s (DDR5) | ~70 ns  | 385 GB |

设计启示：

```text
✅ GPU 频繁访问的数据 → 留在 FB，不经过 BAR1
✅ Host 偶尔访问 GPU 结果 → 通过 BAR1 窗口直接读
✅ 大量 Host↔GPU 迁移 → 用 cudaMemPrefetchAsync，驱动管理 BAR1
❌ 频繁通过 BAR1 逐字节访问 → 性能灾难
```

---

## 6. 查看和管理 BAR1

### 6.1 监控脚本

```bash
# 每 2 秒监控 BAR1 使用量
watch -n 2 'nvidia-smi -q -i 0 | grep -A4 BAR1'
```

### 6.2 CUDA 侧查询

```c
// 通过 cudaMemGetInfo 可以查看总/空闲显存
// 但 CUDA Runtime API 不直接暴露 BAR1 大小
// 需要 NVML:
#include <nvml.h>
nvmlInit();
nvmlDevice_t dev;
nvmlDeviceGetHandleByIndex(0, &dev);
nvmlBAR1Memory_t bar1;
nvmlDeviceGetBAR1MemoryInfo(dev, &bar1);
printf("BAR1: %llu MB total, %llu MB used\n",
       bar1.bar1Total / (1024*1024),
       bar1.bar1Used / (1024*1024));
```

编译：`nvcc -lnvidia-ml -o bar1_query bar1_query.cu`

---

## 参考

- [PCIe 技术大全](01_pcie_comprehensive_guide.md)
- [CUDA NUMA API 编程实践](../../02_gpu_programming/02_cuda/05_cuda_numa_api.md)
- [NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/)
