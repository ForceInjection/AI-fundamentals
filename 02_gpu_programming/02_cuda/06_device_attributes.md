# GPU 原子操作与 PCIe 能力查询

> 基于 RTX 5090 (Blackwell, CC 12.0) 实际环境。`cudaDeviceGetAttribute` 可查询 100+ 种设备属性，本文聚焦 PCIe 原子操作、Host Native Atomic 等底层硬件能力的查询方法。

---

## 1. `cudaDeviceGetAttribute` 概述

`cudaDeviceProp` 结构体只暴露了最常用的属性（SM 数量、显存大小、时钟频率等）。`cudaDeviceGetAttribute` 能查询更底层的硬件能力——这些属性不是简单的数字，而是**硬件特性的布尔开关或枚举值**，直接决定某些 CUDA API 是否可用。

为什么这很重要？很多 CUDA 高级功能在不同 GPU 上的支持状态不同：

- **GPU Direct P2P**：依赖 `HostNativeAtomicSupported`——如果为 0，无法用 GPU atomic 直接操作 Host 内存
- **Unified Memory**：`ConcurrentManagedAccess` 决定 Host 和 GPU 能否同时访问同一块 managed memory
- **Cooperative Launch**：`CooperativeLaunch` 决定能否使用 CUDA Cooperative Groups

这个 API 的正确使用模式是**运行时能力检查 + 优雅降级**——不是在文档上假设某个功能存在，而是每次运行前去查询，并准备 fallback 路径。

```c
int value;
cudaError_t err = cudaDeviceGetAttribute(&value, cudaDevAttrXxx, device_id);
```

---

## 2. PCIe 原子操作能力

### 2.1 什么是 PCIe Atomic

PCIe 原子操作允许 GPU 通过 PCIe 总线对 Host 内存执行不可分割的读写操作（无需锁总线），对 RDMA 和 GPUDirect 场景至关重要。

### 2.2 查询原子操作支持

```bash
cat > check_atomic.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_ATTR(attr, id) do {                              \
    int v;                                                      \
    cudaDeviceGetAttribute(&v, attr, id);                      \
    printf("  %-45s = %d%s\n", #attr, v, v ? " ✓" : "");        \
} while(0)

int main() {
    int count;
    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("GPU %d: %s (CC %d.%d)\n", i, prop.name,
               prop.major, prop.minor);

        printf("\n--- PCIe & Atomic ---\n");
        CHECK_ATTR(cudaDevAttrHostNativeAtomicSupported, i);
        CHECK_ATTR(cudaDevAttrCanUseHostPointerForRegisteredMem, i);

        printf("\n--- Managed Memory ---\n");
        CHECK_ATTR(cudaDevAttrConcurrentManagedAccess, i);
        CHECK_ATTR(cudaDevAttrPageableMemoryAccess, i);
        CHECK_ATTR(cudaDevAttrPageableMemoryAccessUsesHostPageTables, i);
        CHECK_ATTR(cudaDevAttrDirectManagedMemAccessFromHost, i);

        printf("\n--- Launch Capability ---\n");
        CHECK_ATTR(cudaDevAttrCooperativeLaunch, i);
        CHECK_ATTR(cudaDevAttrCooperativeMultiDeviceLaunch, i);

        printf("\n--- SM Resources ---\n");
        CHECK_ATTR(cudaDevAttrMaxRegistersPerMultiprocessor, i);
        CHECK_ATTR(cudaDevAttrMaxBlocksPerMultiprocessor, i);
        CHECK_ATTR(cudaDevAttrMaxThreadsPerMultiProcessor, i);
        CHECK_ATTR(cudaDevAttrAsyncEngineCount, i);

        printf("\n--- Precision ---\n");
        CHECK_ATTR(cudaDevAttrSingleToDoublePrecisionPerfRatio, i);
    }
    return 0;
}
EOF

nvcc -o check_atomic check_atomic.cu && ./check_atomic
```

**RTX 5090 实际输出：**

```text
GPU 0: NVIDIA GeForce RTX 5090 (CC 12.0)

--- PCIe & Atomic ---
  cudaDevAttrHostNativeAtomicSupported     = 0
  cudaDevAttrCanUseHostPointerForRegisteredMem = 1 ✓

--- Managed Memory ---
  cudaDevAttrConcurrentManagedAccess       = 1 ✓
  cudaDevAttrPageableMemoryAccess          = 0
  cudaDevAttrPageableMemoryAccessUsesHostPageTables = 0
  cudaDevAttrDirectManagedMemAccessFromHost = 0

--- Launch Capability ---
  cudaDevAttrCooperativeLaunch             = 1 ✓
  cudaDevAttrCooperativeMultiDeviceLaunch  = 1 ✓

--- SM Resources ---
  cudaDevAttrMaxRegistersPerMultiprocessor = 65536
  cudaDevAttrMaxBlocksPerMultiprocessor    = 24
  cudaDevAttrMaxThreadsPerMultiProcessor   = 1536
  cudaDevAttrAsyncEngineCount              = 2

--- Precision ---
  cudaDevAttrSingleToDoublePrecisionPerfRatio = 64
```

### 2.3 关键解读

| 属性                                | RTX 5090 | 数据中心 GPU    | 影响                                   |
| ----------------------------------- | -------- | --------------- | -------------------------------------- |
| `HostNativeAtomicSupported`         | 0 ❌     | 1 ✅            | 无法用 GPU atomic 直接操作 Host 内存   |
| `CanUseHostPointerForRegisteredMem` | 1 ✅     | 1 ✅            | 可以使用 registered host memory        |
| `PageableMemoryAccess`              | 0 ❌     | 1 ✅ (H100+)    | 不支持 pageable memory 的 GPU 直接访问 |
| `DirectManagedMemAccessFromHost`    | 0 ❌     | 1 ✅ (GH200)    | Host 无法直接访问 managed memory       |
| `ConcurrentManagedAccess`           | 1 ✅     | 1 ✅            | 支持 Host+GPU 同时访问 managed memory  |
| `CooperativeMultiDeviceLaunch`      | 1 ✅     | 1 ✅            | 支持 cooperative groups 跨设备         |
| `SingleToDoublePrecisionPerfRatio`  | 64:1     | 2:1 (A100/H100) | 消费级 GPU 双精度性能严重受限          |

---

## 3. `nvidia-smi` 侧确认 PCIe Atomic

`nvidia-smi -q` 的 PCI 部分会显示 atomic 能力：

```bash
nvidia-smi -q | grep -A2 "Atomic"
```

RTX 5090 输出：

```text
Atomic Caps Outbound    : N/A
Atomic Caps Inbound     : FETCHADD_32 FETCHADD_64 SWAP_32 SWAP_64 CAS_32 CAS_64
```

- **Inbound**：GPU 接受来自 PCIe 的 atomic 请求 → 支持 FetchAdd/Swap/CAS 32/64
- **Outbound**：GPU 向 Host 发起 atomic 请求 → N/A（消费级 GPU 不支持）

---

## 4. 常用属性速查表

### 4.1 内存相关

| 属性                                      | 说明                   | RTX 5090 |
| ----------------------------------------- | ---------------------- | -------- |
| `cudaDevAttrMaxSharedMemoryPerBlockOptin` | 可选最大 shared memory | 查询中   |
| `cudaDevAttrMemoryPoolsSupported`         | 内存池支持             | 支持     |
| `cudaDevAttrUnifiedAddressing`            | 统一寻址               | 支持     |

### 4.2 执行模型

| 属性                                       | 说明             | RTX 5090 |
| ------------------------------------------ | ---------------- | -------- |
| `cudaDevAttrMaxRegistersPerMultiprocessor` | SM 寄存器总数    | 65536    |
| `cudaDevAttrMaxBlocksPerMultiprocessor`    | SM 最大 block 数 | 24       |
| `cudaDevAttrMaxThreadsPerMultiProcessor`   | SM 最大线程数    | 1536     |
| `cudaDevAttrWarpSize` (from prop)          | Warp 大小        | 32       |

### 4.3 流与并发

| 属性                                   | RTX 5090 | 说明                        |
| -------------------------------------- | -------- | --------------------------- |
| `cudaDevAttrAsyncEngineCount`          | 2        | 2 个异步 copy engine        |
| `cudaDevAttrConcurrentKernels` (prop)  | 1        | 支持 concurrent kernel 执行 |
| `cudaDevAttrMaxSurface1DLayeredLayers` | ...      | 3D surface 层数上限         |

---

## 5. 编程实践

### 5.1 能力检查模式

```c
// 运行时检查能力，优雅降级
int host_atomic;
cudaDeviceGetAttribute(&host_atomic,
    cudaDevAttrHostNativeAtomicSupported, 0);

if (host_atomic) {
    // 使用 GPU atomic 直接操作 Host 内存
    launch_kernel_with_host_atomic<<<...>>>();
} else {
    // 回退方案：先 D2H → CPU atomic → H2D
    launch_kernel_without_host_atomic<<<...>>>();
    cudaMemcpy(...);
    cpu_atomic_op(...);
    cudaMemcpy(...);
}
```

### 5.2 设备选择

```c
// 选择支持特定能力的 GPU
for (int i = 0; i < count; i++) {
    int v;
    cudaDeviceGetAttribute(&v, cudaDevAttrCooperativeLaunch, i);
    if (v) {
        cudaSetDevice(i);
        break;
    }
}
```

---

## 参考

- [CUDA Device Attribute 枚举](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g49c516490dd8c1bf7aa84fc208f4ca02)
- [PCIe Atomic Operations Specification](https://pcisig.com/)
