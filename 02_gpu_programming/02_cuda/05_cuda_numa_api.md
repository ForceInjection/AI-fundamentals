# CUDA NUMA API 编程实践

> 基于 Intel Xeon Platinum 8470Q (双路 NUMA) + RTX 5090 实际环境。本文聚焦 CUDA 中与 NUMA 相关的 API：pinned memory 分配策略、`cudaMemAdvise`、`cudaMemPrefetchAsync`，以及如何通过 CPU 亲和性绑定优化 H2D/D2H 带宽。

---

## 1. 背景：为什么 NUMA 影响 GPU 程序

GPU 通过 PCIe 连接到特定 NUMA 节点（本例为 node 1）。当 CUDA 程序：

1. 调用 `cudaMallocHost` 分配 pinned memory——内存从**当前线程所在的 NUMA 节点**分配
2. 如果线程运行在远端 NUMA (node 0)，H2D/D2H 需经过 UPI 跨 socket 传输
3. NUMA distance 为 21 vs 10，约 2.1× 延迟惩罚

---

## 2. Pinned Memory 的 NUMA 亲和性

### 2.1 `cudaMallocHost` 的默认行为

```c
// 内存从当前线程的 NUMA 节点分配
float *h_buf;
cudaMallocHost(&h_buf, size);  // NUMA 亲和性取决于线程位置
```

### 2.2 使用 `cudaHostAlloc` 指定 NUMA 策略

```c
// 可移植——允许从任意 NUMA 分配（降低性能但灵活）
cudaHostAlloc(&h_buf, size, cudaHostAllocPortable);

// 写入合并——优化 H2D 但牺牲随机读性能
cudaHostAlloc(&h_buf, size, cudaHostAllocWriteCombined);

// 组合使用
cudaHostAlloc(&h_buf, size,
    cudaHostAllocPortable | cudaHostAllocWriteCombined);
```

| Flag                         | 效果               | 适用场景              |
| ---------------------------- | ------------------ | --------------------- |
| 默认                         | 线程所在 NUMA 分配 | 线程已绑定到 GPU NUMA |
| `cudaHostAllocPortable`      | 跨 NUMA 可映射     | 多 GPU 位于不同 NUMA  |
| `cudaHostAllocWriteCombined` | 绕过 L1/L2 cache   | 纯 H2D 传输（不读回） |

---

## 3. CPU 亲和性控制

### 3.1 为什么需要

GPU 位于 NUMA node 1 (CPUs 52-103, 156-207)。程序应绑定到这些核，确保 pinned memory 从 node 1 分配。

### 3.2 通过 `taskset` 绑定

```bash
# 绑定到 GPU 所在 NUMA 节点 (node 1)
taskset -c 52-103,156-207 ./my_gpu_program

# 验证
taskset -c 52-103,156-207 nvidia-smi
```

### 3.3 通过 `numactl` 绑定

```bash
# 安装
apt install numactl

# 绑定 NUMA node 和内存策略
numactl --cpunodebind=1 --membind=1 ./my_gpu_program

# 查看拓扑
numactl --hardware
```

### 3.4 程序内查询 NUMA

```bash
# 查看 GPU 所在 NUMA node
cat /sys/bus/pci/devices/0000:98:00.0/numa_node
# 输出: 1

# 各 NUMA 节点的 CPU 范围
cat /sys/devices/system/node/node1/cpulist
# 输出: 52-103,156-207
```

---

## 4. CUDA Managed Memory 的 NUMA 优化

### 4.1 `cudaMemAdvise`

在统一内存 (Managed Memory) 模式下，给驱动 NUMA 放置提示：

```c
// 分配统一内存
float *data;
cudaMallocManaged(&data, size);

// 告知驱动：这块内存主要在 GPU 0 上被访问
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, 0);

// 告知驱动：这块内存会被 CPU 读取（只读）
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
```

### 4.2 `cudaMemPrefetchAsync`

主动将数据迁移到目标设备：

```c
// 预取到 GPU 0（消除首次访问的 page fault 延迟）
cudaMemPrefetchAsync(data, size, 0, stream);

// 预取回 CPU（GPU 处理完毕后）
cudaMemPrefetchAsync(data, size, cudaCpuDeviceId, stream);
```

### 4.3 完整示例

```c
#include <cuda_runtime.h>
#include <stdio.h>

#define N (1024 * 1024)

int main() {
    float *data;
    cudaMallocManaged(&data, N * sizeof(float));

    // 告知驱动：主要在 GPU 0 被访问
    cudaMemAdvise(data, N * sizeof(float),
                  cudaMemAdviseSetPreferredLocation, 0);

    // 预取到 GPU，避免首次 kernel launch 的 page fault
    cudaMemPrefetchAsync(data, N * sizeof(float), 0, 0);
    cudaDeviceSynchronize();

    // GPU kernel 计算...

    // 预取回 CPU
    cudaMemPrefetchAsync(data, N * sizeof(float), cudaCpuDeviceId, 0);
    cudaDeviceSynchronize();

    // CPU 读取结果
    for (int i = 0; i < 10; i++)
        printf("data[%d] = %f\n", i, data[i]);

    cudaFree(data);
    return 0;
}
```

编译：

```bash
nvcc -o managed_mem managed_mem.cu
```

---

## 5. 设备属性查询

RTX 5090 在统一内存方面的能力：

```c
cudaDeviceGetAttribute(&v, cudaDevAttrConcurrentManagedAccess, 0);
// = 1 → 支持 concurrent managed access (Hopper+)

cudaDeviceGetAttribute(&v, cudaDevAttrPageableMemoryAccess, 0);
// = 0 → 不支持 pageable memory access (仅数据中心 GPU 支持)

cudaDeviceGetAttribute(&v, cudaDevAttrDirectManagedMemAccessFromHost, 0);
// = 0 → 不支持从 Host 直接访问 managed memory
```

---

## 6. 最佳实践速查

| 场景               | 推荐做法                                          |
| ------------------ | ------------------------------------------------- |
| 单 GPU + H2D 为主  | `cudaHostAlloc(WriteCombined)` + `taskset` 绑核   |
| 单 GPU + D2H 频繁  | 默认 `cudaMallocHost` + 确保线程在 GPU NUMA       |
| 多 GPU 不同 NUMA   | `cudaHostAlloc(Portable)` + `cudaSetDevice` 切换  |
| Managed Memory     | `cudaMemAdvise` + `cudaMemPrefetchAsync` 主动放置 |
| 仅需知道 NUMA 信息 | `cat /sys/bus/pci/devices/<bdf>/numa_node`        |

---

## 参考

- [CUDA Runtime API: Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
- [CUDA C Programming Guide: UM](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)
- [单卡 GPU 拓扑与 NUMA 深入分析](../../01_hardware_architecture/performance/02_single_gpu_topology_analysis.md)
