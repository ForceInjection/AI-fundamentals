# CUDA NUMA API 编程实践

> 基于 Intel Xeon Platinum 8470Q (双路 NUMA) + RTX 5090 实际环境。本文聚焦 CUDA 中与 NUMA 相关的 API：pinned memory 分配策略、`cudaMemAdvise`、`cudaMemPrefetchAsync`，以及如何通过 CPU 亲和性绑定优化 H2D/D2H 带宽。
>
> **验证代码**：本文所有实测数据来自配套 Demo：[05_cuda_numa_demo.cu](code/05_cuda_numa_demo.cu)。用 `numactl --cpunodebind=0` vs `--cpunodebind=1` 分别运行，对比带宽差异即可验证 NUMA 效果。

---

## 1. 背景：为什么 NUMA 影响 GPU 程序

### 1.1 NUMA 架构简述

在双路 CPU 系统中，每个 CPU socket 拥有自己的内存控制器和本地 DDR5 内存。访问本地内存快（~70 ns），跨 socket 访问远端内存慢（~140 ns）。这种架构称为 **NUMA (Non-Uniform Memory Access)**——「非统一」指的是访问延迟因地址而异：

```text
Socket 0 (NUMA node 0)               Socket 1 (NUMA node 1)
  ├── CPU 0-51, 104-155                ├── CPU 52-103, 156-207
  ├── DDR5 385 GB (本地)               ├── DDR5 387 GB (本地)
  └── PCIe domain 7e/7f                └── PCIe domain 97/d7 (GPU + NVMe)
         ↓                                      ↓
    本地访问: 10 (distance)               GPU 在这里!
         ↓                                      ↓
    跨 socket 访问: 21 (distance) ←── UPI ──→ 跨 socket 访问: 21
```

### 1.2 GPU 受 NUMA 影响的原因

GPU 通过 PCIe 连接到特定 NUMA 节点（本例为 node 1）。当 CUDA 程序：

1. 调用 `cudaMallocHost` 分配 pinned memory——落在哪个 NUMA 由 Linux 的 first-touch 页分配策略决定，通常就是调用线程所在节点（CUDA 文档不承诺 NUMA 行为）
2. 如果线程运行在远端 NUMA (node 0)，H2D/D2H 需经过 UPI（Ultra Path Interconnect，Intel 双路 CPU 间的点对点互连总线）跨 socket 传输
3. 实测本地 ~70 ns、跨 socket ~140 ns，约 2 倍惩罚

要点：`cudaMallocHost` 不关心 GPU 在哪个 NUMA——分配位置跟着调用线程走。线程在 node 0 上分配，而 GPU 在 node 1，每次 DMA 都要穿过 UPI。所以要主动控制线程亲和性。

---

## 2. Pinned Memory 的 NUMA 亲和性

### 2.1 `cudaMallocHost` 的默认行为

```c
// 内存从当前线程的 NUMA 节点分配
float *h_buf;
cudaMallocHost(&h_buf, size);  // NUMA 亲和性取决于线程位置
```

### 2.2 `cudaHostAlloc` 的常用 flag

注意：`cudaHostAlloc` 本身**不提供 NUMA 指定能力**——它的 flag 解决的是另外两件事：

```c
// Portable——所有 CUDA context 都视其为 pinned memory（多 GPU 场景）
cudaHostAlloc(&h_buf, size, cudaHostAllocPortable);

// 写入合并——PCIe 写更快，CPU 侧读取低效
cudaHostAlloc(&h_buf, size, cudaHostAllocWriteCombined);

// 组合使用
cudaHostAlloc(&h_buf, size,
    cudaHostAllocPortable | cudaHostAllocWriteCombined);
```

| Flag                         | 效果                                    | 适用场景                    |
| ---------------------------- | --------------------------------------- | --------------------------- |
| 默认                         | 按 first-touch 落在调用线程所在 NUMA    | 线程已绑定到 GPU NUMA       |
| `cudaHostAllocPortable`      | 所有 CUDA context 均按 pinned 处理（与 NUMA 无关） | 多 GPU 多 context 共享同一块 pinned 内存 |
| `cudaHostAllocWriteCombined` | write-combined：PCIe 写更快，CPU 读取低效 | 纯 H2D 传输（CPU 不读回）   |

NUMA 粒度的放置控制要靠操作系统层（`numactl --membind`、线程绑定），或者第 4 节 Managed Memory 的 advice。

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

// 告知驱动：这块内存以读为主（任一处理器读时建只读副本；location 参数被忽略）
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

> CUDA 13.x 起 `cudaMemAdvise` / `cudaMemPrefetchAsync` 新增了 `cudaMemLocation` 形式的签名，可以显式指定 host NUMA 节点（`cudaMemLocationTypeHostNuma`）——NUMA 粒度的精细控制正是本文主题，旧的 int device 签名仍然可用。

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
// = 1 → 支持 concurrent managed access（CPU 与 GPU 并发一致访问；取决于平台）

cudaDeviceGetAttribute(&v, cudaDevAttrPageableMemoryAccess, 0);
// 设备能否直接访问 host 的 pageable memory（取决于平台/驱动的 ATS/HMM，不按 GPU 档位划分）

cudaDeviceGetAttribute(&v, cudaDevAttrDirectManagedMemAccessFromHost, 0);
// = 1 → Host 可免迁移直接访问驻留在设备上的 managed memory
// = 0 → Host 访问前需要把页迁到 host
```

---

## 6. 最佳实践速查

| 场景               | 推荐做法                                          |
| ------------------ | ------------------------------------------------- |
| 单 GPU + H2D 为主  | `cudaHostAlloc(WriteCombined)` + `taskset` 绑核   |
| 单 GPU + D2H 频繁  | 默认 `cudaMallocHost` + 确保线程在 GPU NUMA       |
| 多 GPU 共享 pinned | `cudaHostAlloc(Portable)`（多 context 按 pinned 处理）；NUMA 放置仍靠 numactl/线程绑定 |
| Managed Memory     | `cudaMemAdvise` + `cudaMemPrefetchAsync` 主动放置 |
| 仅需知道 NUMA 信息 | `cat /sys/bus/pci/devices/<bdf>/numa_node`        |

---

## 参考

- [CUDA Runtime API: Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
- [CUDA C Programming Guide: UM](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)
- [单卡 GPU 拓扑与 NUMA 深入分析](../../01_hardware_architecture/performance/02_single_gpu_topology_analysis.md)
