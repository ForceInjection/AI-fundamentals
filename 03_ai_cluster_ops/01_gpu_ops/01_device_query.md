# GPU 设备属性查询——不是「看一眼」，而是「写对 Kernel 的前置条件」

写 CUDA 代码之前，你有两个选择：凭经验设 Grid/Block 大小，或者先查一下这张卡到底有几个 SM、每个 Block 最多放多少线程、共享内存多大。选前者的结果是 Kernel 在 A100 上跑得好好的，换到 T4 上直接 `cudaLaunch` 报错——因为 T4 的 `maxThreadsPerBlock` 是 1024，你写了 2048。

GPU 设备查询存在的意义不是「看看配置」，而是回答一个更实际的问题：**这段 Kernel 在这张卡上最多能怎么并行？** 本文从 Runtime API 和 Driver API 两条路径切入，最后落到一张速查表——只列和 Kernel 设计直接相关的那几个参数。

---

## 一、两套 API：Runtime vs Driver

CUDA 提供了两套查询接口，侧重不同：

| API 层      | 入口函数                    | 典型用途                                 |
| ----------- | --------------------------- | ---------------------------------------- |
| Runtime API | `cudaGetDeviceProperties()` | 应用开发首选——一行调用拿到所有属性       |
| Driver API  | `cuDeviceGetAttribute()`    | 需要逐属性查询的场景（如中间件、框架层） |

Runtime API 返回一个 `cudaDeviceProp` 结构体——几十个字段一次性填好。Driver API 按需逐项查询，更灵活但更繁琐。两者的信息源相同（驱动层），不存在精度差异。

NVIDIA 官方示例中，`deviceQuery` 用 Runtime API，`deviceQueryDrv` 用 Driver API。源码在 [cuda-samples](https://github.com/NVIDIA/cuda-samples) 的 `Samples/1_Utilities/` 下。

---

## 二、Runtime API：一行拿到全部属性

```cpp
#include <cuda_runtime.h>

int deviceCount;
cudaGetDeviceCount(&deviceCount);

for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device %d: %s (CC %d.%d, %d SMs, %zu MB global mem)\n",
           i, prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024 * 1024));
}
```

编译：

```bash
nvcc -o device_query device_query.cu
./device_query
```

以 Tesla T4 为例，关键输出：

```text
Device 0: "Tesla T4"
  CUDA Capability:      7.5
  SMs:                  40
  CUDA Cores:           2560  (40 × 64)
  Global Memory:        14931 MB
  Max Threads/Block:    1024
  Max Threads/SM:       1024
  Shared Mem/Block:     48 KB
  Registers/Block:      65536
  Warp Size:            32
  Max Grid Dim:         (2147483647, 65535, 65535)
  Max Block Dim:        (1024, 1024, 64)
  Memory Bus Width:     256-bit
  ECC:                  Enabled
  Concurrent Kernels:   Yes
  Concurrent Copy+Exec: Yes (3 copy engines)
  P2P Access:           Yes (GPU0↔GPU1)
```

---

## 三、哪些参数直接影响 Kernel 设计？

不是每个字段都需要关注。下面这 5 个决定了一段 Kernel 能否启动、能启动多大：

| 参数               | 字段                  | 含义                      | 踩坑场景                                                                                |
| ------------------ | --------------------- | ------------------------- | --------------------------------------------------------------------------------------- |
| **计算能力**       | `major`, `minor`      | 决定了哪些 CUDA 特性可用  | 编译时 `-arch=sm_80` 的代码无法在 CC 7.5 上运行                                         |
| **SM 数量**        | `multiProcessorCount` | 多少个 SM 可并行          | 决定了 Grid 的并行度上限；SM 数太少的卡上小 Grid 利用率极低                             |
| **最大线程/Block** | `maxThreadsPerBlock`  | Block 维度的硬上限        | 写了 2048 线程/Block，在 T4 (max 1024) 上直接 `cudaLaunch` 失败                         |
| **共享内存/Block** | `sharedMemPerBlock`   | Block 内的高速 Scratchpad | 申请超过 48KB 的 `extern __shared__` 时会报错（除非配置 `cudaFuncSetAttribute` 调上限） |
| **Warp Size**      | `warpSize`            | 硬件调度单位              | 所有 NVIDIA GPU 都是 32，但 Block Size 不整除 32 会浪费线程槽位                         |

其次重要的——决定能跑多快：

| 参数                          | 含义                                                        |
| ----------------------------- | ----------------------------------------------------------- |
| `maxThreadsPerMultiProcessor` | SM 上同时驻留的线程数上限 → 决定 Occupancy                  |
| `regsPerBlock`                | Block 可用的寄存器数 → 每个线程用太多寄存器，Occupancy 暴跌 |
| `memPitch`                    | 2D 内存拷贝的最大行距                                       |
| `concurrentKernels`           | 是否支持多 Stream 并行执行                                  |
| `deviceOverlap`               | 是否支持计算与数据拷贝重叠                                  |

其余参数（纹理维度、表面内存对齐等）属于图形/图像处理领域，CUDA 计算场景极少用到。

---

## 四、Driver API：按需逐属性查询

当不需要全部属性、只查一两个值时，Driver API 更高效：

```cpp
#include <cuda.h>

CUdevice device;
int smCount, maxThreadsPerBlock;
cuDeviceGet(&device, 0);
cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
printf("SM count: %d, Max threads/block: %d\n", smCount, maxThreadsPerBlock);
```

编译时需链接 `-lcuda`：

```bash
nvcc -o device_query_drv device_query_drv.cu -lcuda
```

> Driver API 无需 CUDA Runtime 初始化——它在加载 `libcuda.so` 后即可调用，适合需要在 `cudaSetDevice` 之前做硬件能力探测的场景（如框架层的设备筛选）。

---

## 五、速查：从 Kernel 设计视角看设备属性

| 你要做的事                 | 查哪个属性                                 | T4 示例值              |
| -------------------------- | ------------------------------------------ | ---------------------- |
| 确定 Block 最多放多少线程  | `maxThreadsPerBlock`                       | 1024                   |
| 确定 Grid 最多放多少 Block | `maxGridSize`                              | (2^31-1, 65535, 65535) |
| 估算最大 Occupancy         | `maxThreadsPerMultiProcessor` / `warpSize` | 32 warps/SM            |
| 静态共享内存上限           | `sharedMemPerBlock`                        | 48 KB                  |
| 动态共享内存上限           | `sharedMemPerBlockOptin`                   | 64 KB                  |
| 选 `-arch` 编译参数        | `major`, `minor`                           | `sm_75`                |
| 判断是否支持 P2P           | `cudaDeviceCanAccessPeer`                  | —                      |
| 多 Stream 是否真并行       | `concurrentKernels`                        | Yes                    |
| 是否支持 Unified Memory    | `unifiedAddressing` + `managedMemory`      | Yes                    |

---

## 六、相关资源

- [cuda-samples/deviceQuery](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/deviceQuery) — Runtime API 完整示例
- [cuda-samples/deviceQueryDrv](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/deviceQueryDrv) — Driver API 完整示例
- [CUDA Runtime API 文档 - cudaGetDeviceProperties](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)
- [CUDA Driver API 文档 - cuDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html)
