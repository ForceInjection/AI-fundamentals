# NCCL 单卡验证指南

> 本文提供一个最简 NCCL 验证程序，无需多卡、无需 Docker、无需 InfiniBand，在单张 GPU 上即可验证 NCCL 安装和基本功能。

---

## 0. NCCL 是什么

**NCCL (NVIDIA Collective Communications Library)** 是 NVIDIA 提供的多 GPU 集合通信库，负责 AllReduce、AllGather、Broadcast、ReduceScatter 等操作。它是分布式训练的基础设施——PyTorch DDP、DeepSpeed、Megatron-LM 等框架的跨 GPU 通信最终都会调用 NCCL。

NCCL 的核心能力：

| 能力 | 说明 |
|------|------|
| 拓扑感知 | 自动检测 NVLink、PCIe P2P、InfiniBand 路径，选择最优通信路由 |
| Ring / Tree 算法 | 根据数据量和 GPU 数量自动选择 Ring（大包）或 Tree（小包）算法 |
| 零拷贝 | 通过 GPUDirect P2P/RDMA 直接在 GPU 显存间搬运，不经 CPU 中转 |
| 异步执行 | 通信操作在 GPU stream 中异步进行，可与计算重叠 |

**为什么单卡也要验证 NCCL？** 即使只有一张 GPU，NCCL 的编译和链接环境也需要正确配置（nvcc 能找到 `nccl.h`，链接器能找到 `libnccl.so`）。单卡验证通过后，多卡部署只需关注网络和拓扑，排除了库本身的问题。

---

## 1. 前置条件

- NVIDIA GPU + 驱动 (nvidia-smi 可用)
- CUDA Toolkit (nvcc 可用)
- NCCL 库 (`libnccl2` + `libnccl-dev`)

检查方式：

```bash
nvidia-smi | head -5                 # 驱动和 GPU
nvcc --version                        # CUDA 编译器
dpkg -l | grep libnccl               # NCCL 库
ls /usr/include/nccl.h               # NCCL 头文件
```

---

## 2. 编译运行

```bash
cat > /tmp/nccl_hello.cu << 'EOF'
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    // 1. 查询 NCCL 版本
    int nccl_version;
    ncclGetVersion(&nccl_version);
    printf("NCCL version: %d.%d.%d\n",
        nccl_version / 10000,
        (nccl_version % 10000) / 100,
        nccl_version % 100);

    // 2. 检查 CUDA 设备
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("CUDA devices: %d\n", device_count);

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  [%d] %s (%d.%d, %d SMs, %d GB)\n",
            i, prop.name, prop.major, prop.minor,
            prop.multiProcessorCount,
            (int)(prop.totalGlobalMem / (1024*1024*1024)));
    }

    if (device_count == 0) {
        printf("No CUDA device found, exiting.\n");
        return 1;
    }

    // 3. 单卡初始化 NCCL communicator
    cudaSetDevice(0);

    ncclUniqueId id;
    ncclGetUniqueId(&id);

    ncclComm_t comm;
    ncclResult_t result = ncclCommInitRank(&comm, 1, id, 0);

    if (result == ncclSuccess) {
        printf("NCCL comm init: SUCCESS (single rank)\n");

        // 查询 communicator 信息
        int rank, nranks;
        ncclCommUserRank(comm, &rank);
        ncclCommCount(comm, &nranks);
        printf("  Rank: %d / %d\n", rank, nranks);

        ncclCommDestroy(comm);
    } else {
        printf("NCCL comm init: FAILED (error code %d)\n", result);
        const char* err = ncclGetErrorString(result);
        printf("  Error: %s\n", err);
        return 1;
    }

    printf("NCCL verification complete.\n");
    return 0;
}
EOF

nvcc -I/usr/include \
     -L/usr/lib/x86_64-linux-gnu \
     -lnccl \
     -o /tmp/nccl_hello /tmp/nccl_hello.cu

/tmp/nccl_hello
```

**单卡 RTX 5090 预期输出：**

```
NCCL version: 2.25.1
CUDA devices: 1
  [0] NVIDIA GeForce RTX 5090 (12.0, 170 SMs, 31 GB)
NCCL comm init: SUCCESS (single rank)
  Rank: 0 / 1
NCCL verification complete.
```

---

## 3. 进阶：单卡 AllReduce 测试

```bash
cat > /tmp/nccl_allreduce.cu << 'EOF'
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(cmd) do {                                  \
    ncclResult_t r = cmd;                                \
    if (r != ncclSuccess) {                              \
        printf("NCCL error at %s:%d: %s\n",              \
            __FILE__, __LINE__, ncclGetErrorString(r));  \
        exit(1);                                         \
    }                                                    \
} while(0)

int main() {
    int size = 1024 * 1024;  // 1M floats
    float *sendbuff, *recvbuff;
    cudaStream_t stream;

    cudaSetDevice(0);
    cudaStreamCreate(&stream);

    cudaMalloc(&sendbuff, size * sizeof(float));
    cudaMalloc(&recvbuff, size * sizeof(float));

    ncclUniqueId id;
    ncclGetUniqueId(&id);

    ncclComm_t comm;
    CHECK(ncclCommInitRank(&comm, 1, id, 0));

    // 单卡 AllReduce (无实际通信，验证 API 通路)
    CHECK(ncclAllReduce(
        (const void*)sendbuff, (void*)recvbuff,
        size, ncclFloat, ncclSum,
        comm, stream));

    cudaStreamSynchronize(stream);
    printf("ncclAllReduce (single GPU): SUCCESS\n");

    cudaFree(sendbuff);
    cudaFree(recvbuff);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);

    return 0;
}
EOF

nvcc -I/usr/include \
     -L/usr/lib/x86_64-linux-gnu \
     -lnccl \
     -o /tmp/nccl_allreduce /tmp/nccl_allreduce.cu

/tmp/nccl_allreduce
```

---

## 4. 常用 NCCL 环境变量

单卡场景基本不需要调参，了解即可：

| 变量 | 说明 | 示例 |
|------|------|------|
| `NCCL_DEBUG` | 日志级别 | `INFO`, `WARN`, `TRACE` |
| `NCCL_DEBUG_FILE` | 日志输出文件 | `/tmp/nccl-%h.log` |
| `NCCL_IB_DISABLE` | 禁用 InfiniBand | `0` / `1` |
| `NCCL_SOCKET_IFNAME` | 指定网络接口 | `eth0` |
| `NCCL_P2P_DISABLE` | 禁用 P2P 传输 | `0` / `1` |

调试示例：

```bash
NCCL_DEBUG=INFO /tmp/nccl_hello
```

---

## 5. 与 nccl-tests 的关系

`nccl-tests` 是 NVIDIA 官方的端到端性能测试套件，需要多 GPU 才能发挥价值。如果后续升级到多卡环境：

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make MPI=0 CUDA_HOME=/usr/local/cuda
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2  # 2 GPU 测试
```

本文档的 hello world 程序提供的是最基础的安装验证，确保 NCCL 库在编译和运行时路径均正确。
