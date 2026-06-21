# Multi-GPU CUDA 编程——从 P2P 到 NCCL 的多卡协作

你在 8×A100 上跑 TP=4 推理。GPU 0 算完一个 layer，需要把结果传给 GPU 1。你知道机器有 NVLink——但怎么让两个 GPU 直接通信？`cudaMemcpy` 从一个 device pointer 到另一个 device pointer 会不会静默走 CPU？NCCL 是不是比自己手写快？

这些问题有一个共同点：**CUDA 的默认编程模型是单 GPU 的**。当你 `cudaMalloc` 一个指针，它只对当前 `cudaSetDevice` 选中的 GPU 有效。让两个 GPU 共享数据，需要从 device 管理、peer access、P2P 传输、跨 GPU 同步四个层面逐一打通。

本文按这个顺序展开：多 device 管理 → Peer Access → P2P Memcpy → 跨 GPU 同步 → NCCL → 拓扑感知。建议按顺序阅读——后面章节的 API 建立在前面的 context 管理基础上。

> **前置要求**：你已经理解 Stream 并发（见 [CUDA Streams 并发实战](07_cuda_streams_concurrency.md)），了解 NVLink/PCIe 拓扑的基本概念（见 [GPU 内存管理](12_gpu_memory_management.md) 第 3 节）。
>
> **验证代码**：本文所有实测数据来自配套 Benchmark：[15_multi_gpu_bench.cu](code/15_multi_gpu_bench.cu)。包含 Peer Access 检查、P2P 带宽测量、跨 GPU Stream/Event 同步、NCCL AllReduce 四项测试。如需 NCCL 测试需加 `-DWITH_NCCL -lnccl` 编译。

---

## 1. 多 Device 管理——"为什么 GPU 0 的指针在 GPU 1 上不能用"

### 1.1 Per-device Context：每个 GPU 有独立的虚拟地址空间

CUDA 为每个 GPU 维护一个独立的 **primary context**——包含独立的虚拟地址空间、独立的 `cudaMalloc` 堆、独立的 stream/event 池。这意味着：

```cuda
cudaSetDevice(0);
float *d0;
cudaMalloc(&d0, N * sizeof(float));   // d0 在 GPU 0 的地址空间中

cudaSetDevice(1);
my_kernel<<<grid, block>>>(d0);        // ← 非法！d0 对 GPU 1 无意义
```

CPU 类比：两个进程各有独立的虚拟地址空间——进程 A 的 `malloc` 返回的指针在进程 B 中不可用。多 GPU 编程同理，只是"进程"变成了"device context"。

### 1.2 `cudaSetDevice` 影响所有后续 API

`cudaSetDevice` 切换的是**调用线程的当前 device**。它不是简单地选择一个物理 GPU——它改变了后续所有 CUDA API 的语义：

```cuda
cudaSetDevice(0);  // 从此，这个线程的：
cudaMalloc(...);   //   分配在 GPU 0 的 HBM 上
cudaStreamCreate(…); // stream 在 GPU 0 上
my_kernel<<<...>>>(); // kernel 在 GPU 0 上执行
cudaMemcpy(dst, src, ..., cudaMemcpyDefault);  // 默认为 GPU 0 ↔ Host
```

**关键约束**：一个线程同一时刻只能有一个 active device。多 GPU 程序通常采用以下两种模式之一：

- **单线程轮流切换**：`cudaSetDevice(0)` → 操作 → `cudaSetDevice(1)` → 操作 → ...
- **多线程各绑一个 GPU**：thread 0 绑定 GPU 0，thread 1 绑定 GPU 1（OpenMP / `std::thread`）

第二种模式避免了频繁 `cudaSetDevice` 的 overhead，是生产代码的推荐做法。

### 1.3 查询 GPU 数量与属性

```cuda
int deviceCount;
cudaGetDeviceCount(&deviceCount);

for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("GPU %d: %s, %.0f GB\n", i, prop.name,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}
```

生产代码中经常在程序启动时遍历一次，获取每个 GPU 的显存、SM 数、Compute Capability——后续做模型分片决策时依赖这些信息。

---

## 2. Peer Access——"P2P 通信的前提条件"

### 2.1 `cudaDeviceCanAccessPeer`：不是所有 GPU 对都支持

两个 GPU 能否直接访问对方的显存，取决于硬件拓扑和驱动配置。你必须**在运行时查询**，不能假设 NVLink 存在或者 P2P 一定可用：

```cuda
int canAccess01, canAccess10;
cudaDeviceCanAccessPeer(&canAccess01, 0, 1);  // GPU 0 → GPU 1？
cudaDeviceCanAccessPeer(&canAccess10, 1, 0);  // GPU 1 → GPU 0？
```

**关键点**：P2P 不是对称的。`canAccess01 == 1` 不意味着 `canAccess10 == 1`。每个方向需要独立开启。

### 2.2 `cudaDeviceEnablePeerAccess`：一次性启用

```cuda
cudaSetDevice(0);
if (canAccess01) {
    cudaDeviceEnablePeerAccess(1, 0);  // enable GPU 0 → GPU 1 access
}
cudaSetDevice(1);
if (canAccess10) {
    cudaDeviceEnablePeerAccess(0, 0);  // enable GPU 1 → GPU 0 access
}
```

启用是**幂等的、低开销的**——多次调用不会重复初始化。通常在程序初始化时全部开启。

### 2.3 为什么这很重要：静默降级

如果你做了 `cudaMemcpyPeer` 而没有启用 Peer Access，**CUDA runtime 不会报错**——它会静默地将数据通过 CPU 中转：

```text
期望（P2P 启用）:
  GPU 0 HBM ──NVLink──→ GPU 1 HBM               (~300 GB/s)

实际（P2P 未启用）:
  GPU 0 HBM → CPU → GPU 1 HBM                    (~13 GB/s Gen4)
```

你写的是同一行 `cudaMemcpyPeer`，带宽差了 **35 倍**。没有 warning、没有 error——只有在 `nvidia-smi dmon -s pucv` 上看到 PCIe 吞吐飙升时才意识到。**始终在程序启动时检查 `canAccessPeer` + `enablePeerAccess` 的结果**，或者更好的做法是打印拓扑确认信息到日志。

---

## 3. P2P Memcpy——"怎么让 NVLink 跑满"

### 3.1 `cudaMemcpyPeer`：直接的 GPU→GPU 传输

```cuda
// 将 GPU 0 上的 d_src 传输到 GPU 1 上的 d_dst
cudaMemcpyPeer(d_dst, 1,    // dst ptr + dst device
               d_src, 0,    // src ptr + src device
               N * sizeof(float));
```

这与 `cudaMemcpy` 的核心区别：**src 和 dst 都是 device pointer，不需要经过 host**。

### 3.2 `cudaMemcpyPeerAsync`：与 Stream 配合

```cuda
cudaStream_t s1;
cudaSetDevice(1);
cudaStreamCreate(&s1);

// 异步 P2P，GPU 0→1 的传输与 GPU 1 的 stream 绑定
cudaMemcpyPeerAsync(d_dst, 1, d_src, 0, N * sizeof(float), s1);

// 传输完成后立即执行 kernel
my_kernel<<<grid, block, 0, s1>>>(d_dst, N);
```

`cudaMemcpyPeerAsync` 将 P2P 传输与目标 GPU 的 stream 关联——数据到达后，同一 stream 上的后续 kernel 自动开始执行，不需要额外的同步。

### 3.3 性能实测：NVLink P2P vs PCIe P2P vs CPU 中转

```bash
# 测试程序
cat > p2p_bw.cu << 'BENCH'
#include <cuda_runtime.h>
#include <stdio.h>

#define N (256 * 1024 * 1024 / sizeof(float))  // 256 MB

int main() {
    int count;
    cudaGetDeviceCount(&count);
    printf("Found %d GPUs\n", count);

    // 只用 GPU 0 和 GPU 1
    cudaSetDevice(0);
    float *d0;
    cudaMalloc(&d0, N * sizeof(float));
    cudaDeviceEnablePeerAccess(1, 0);

    cudaSetDevice(1);
    float *d1;
    cudaMalloc(&d1, N * sizeof(float));
    cudaDeviceEnablePeerAccess(0, 0);

    // --- Test 1: P2P Direct ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaSetDevice(0);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        cudaMemcpyPeer(d1, 1, d0, 0, N * sizeof(float));
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    // ...

    // --- Test 2: CPU-mediated (disable peer access first) ---
    // (需要先 cudaDeviceDisablePeerAccess)
    // ...

    // --- Test 3: Staged via CPU ---
    float *h_buf;
    cudaMallocHost(&h_buf, N * sizeof(float));
    // GPU 0 → CPU → GPU 1
    // ...
}
BENCH
```

在 8×A100-SXM4 服务器上的实测（配套 Benchmark，64 MB × 200 次传输）：

| 路径                            | 实测带宽     | 加速比 | 说明                        |
| ------------------------------- | ------------ | ------ | --------------------------- |
| NVLink P2P (GPU 0→1)            | **269 GB/s** | 20.7×  | A100 NVLink 3.0, 12 条 link |
| CPU 中转 (GPU 0→CPU→GPU 1)      | 13.0 GB/s    | —      | DMA 两次串行 (Gen4 ×16)     |
| P2P 未启用却调用 cudaMemcpyPeer | ~13 GB/s     | —      | **静默降级，无 error！**    |

> **验证**：`nvidia-smi topo -m` 确认连接类型。A100-SXM4 的 NVLink 对等带宽约 300 GB/s 单向（12 条 NVLink 3.0 link），实测 269 GB/s 达到了 ~90% 的理论值。H100 升级到 NVLink 4.0（18 条 link），理论 ~450 GB/s 单向。

### 3.4 Unified Memory 在多 GPU 下的陷阱

在 [GPU 内存管理](12_gpu_memory_management.md) 第 4 节讨论过 UM 的 page fault 问题。多 GPU 场景中更严重：`cudaMallocManaged` 分配的页面在两个 GPU 之间迁移时，触发 **device-to-device page fault**——延迟 ~10 μs/page。

对于 LLM 推理的 KV cache 大小（~14 MB / 256-token chunk），跨 GPU page fault 的总延迟是致命的。这就是为什么 vLLM 用 NCCL 做 TP（显式的 peer-to-peer 传输 + 计算重叠），而不是 Unified Memory。

---

## 4. 跨 GPU 同步——"怎么确保 GPU 1 的数据已经就绪"

### 4.1 `cudaStreamWaitEvent`：让一个 GPU 等待另一个 GPU 的事件

解决的核心问题：GPU 0 算完了结果并 P2P 传到 GPU 1——GPU 1 怎么知道数据已经到了？

**方案**：在 GPU 0 上 record 一个 event，让 GPU 1 上某个 stream waiting on 那个 event：

```cuda
cudaSetDevice(0);
cudaEvent_t e0;
cudaEventCreate(&e0);

// GPU 0: 执行 kernel + P2P 传输
compute_kernel<<<grid, block, 0, s0>>>(d0, N);
cudaMemcpyPeerAsync(d1, 1, d0, 0, N * sizeof(float), s1);  // 从 s0 依赖?
cudaEventRecord(e0, s1);  // s1 完成时 e0 被标记
// 注意: P2P 操作仍然在 src device 的 stream 上提交

// GPU 1: 等待 GPU 0 的 event
cudaSetDevice(1);
cudaStreamWaitEvent(s1_consumer, e0, 0);  // s1_consumer 在 e0 完成前被阻塞
consume_kernel<<<grid, block, 0, s1_consumer>>>(d1, N);
```

**关键细节**：`cudaMemcpyPeerAsync` 在**源 GPU** 的 stream 中排队，在**目标 GPU** 的 stream 中完成。你只需要在源 GPU 的 stream 上 record event，然后在目标 GPU 的 stream 上 wait 那个 event。

### 4.2 跨 Device Event 的约束

- Event 必须在被 wait 之前被 record——否则 `cudaStreamWaitEvent` 是 no-op
- Event 可以通过 `cudaEventCreate` 创建（默认支持跨 device），不需要特殊 flag
- `cudaEventSynchronize` 可以在任意 device context 中调用——它等待的是 GPU 某个时间点，不依赖当前的 active device

### 4.3 实战：多 GPU TP 推理的同步流水线

```text
TP=4, 每个 GPU 算 1/4 layer

时间线:
  GPU 0: [Layer N] ──P2P──→ GPU 1: [Layer N] ──P2P──→ GPU 2: [Layer N]
      │  ← cudaStreamWaitEvent(e1)              │
      └─ [Layer N+1] 在收到 GPU 3 的结果后开始    └─ [Layer N+1]
```

TP 推理中每个 GPU 需要 all_reduce——等价于每个 GPU 将自己的 1/4 结果广播给其他 3 个 GPU。用 P2P + event 可以手写一个 ring-based all_reduce，但 NCCL 已经做了这件事（而且比你手写的好）。这就引出了下一节。

---

## 5. NCCL——"手写 P2P 的尽头"

### 5.1 为什么需要 NCCL

假设 TP=4，GPU 0-3 各持有一个 partial result，需要 all_reduce（求和并分发到所有 GPU）：

**手写 P2P 方案（ring all_reduce）**：

```text
Step 1: GPU 0→1, GPU 1→2, GPU 2→3, GPU 3→0  (每个发送 1/4 数据)
Step 2: GPU 0→1, GPU 1→2, GPU 2→3, GPU 3→0  (accumulate)
...共 2*(N-1) 步，每步一次 cudaMemcpyPeer + 一次 add kernel

代码量: ~200 行 CUDA，需手动管理每个方向的 stream + event
```

**NCCL 方案**：

```c
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);
// 一行。内部自动选最优算法（ring / tree / NVSwitch），
// 自动利用 NVLink 带宽，自动做 reduce+copy 的 kernel compute overlap
```

手写的 ring all_reduce 能工作，但 NCCL 做了以下你不太可能手动实现的事情：

- **根据拓扑选算法**：NVSwitch 全互联 → 直接 NVSwitch reduce；环形拓扑 → ring；树形拓扑 → tree
- **传输与计算重叠**：reduce_scatter 阶段一边传一边加，不是先传完再加
- **GPU Direct RDMA**：跨节点走 InfiniBand，不经 CPU

### 5.2 NCCL 初始化：`ncclCommInitRank` + UniqueId

```c
#include <nccl.h>

// 每个进程/线程调用
int rank = /* 0, 1, 2, ..., N-1 */;
int world_size = /* N */;

ncclUniqueId id;
if (rank == 0) ncclGetUniqueId(&id);
// 将 id 广播给所有 rank（MPI_Bcast / NCCL broadcast / shared file）

ncclComm_t comm;
ncclCommInitRank(&comm, world_size, id, rank);
```

#### 5.2.1 从单进程多线程到多进程（单机多卡）

上面的 `ncclCommInitRank` 示例假定每个 rank 是一个独立的进程/线程，通过 `ncclUniqueId` 来协商建立通信组。在生产环境中，这对应两种常见模式：

- **单进程多线程**：`main()` 中 spawn N 个线程（如 8 个），每个线程 `cudaSetDevice(i)` + `ncclCommInitRank`。这是 `torchrun --nproc_per_node=8` 的本质。
- **多进程**：`mpirun -np 8` 或者 `torchrun`，每个进程独占一个 GPU。`ncclUniqueId` 通过 MPI 广播或共享文件协调。

单机多卡的单进程多线程模式不需要 MPI——`ncclUniqueId` 可以通过 rank 0 生成后，以 shared memory / condition_variable 的方式交给其他线程。这也是本文配套 benchmark 采用的方式。

#### 5.2.2 NCCL 版本兼容性

```c
ncclGetVersion(&version);  // 返回类似 2.21.5 的整型编码
printf("NCCL version: %d\n", version);
```

运行时检查 NCCL 版本在处理多机异构环境（不同节点可能安装不同 CUDA/NCCL 版本）时是一个防御性编程的好习惯。NCCL 2.x 的主版本号保证了 API 向后兼容性，但某些 collective 算法在不同 minor 版本间可能有性能差异。如果跨节点通信，建议确保所有节点的主版本号一致。

### 5.3 核心集合通信原语

| 操作                    | 语义                                            | Tensor Parallel 用途           |
| ----------------------- | ----------------------------------------------- | ------------------------------ |
| `ncclAllReduce`         | 所有 GPU 的元素求和/最大值，结果分发到所有 GPU  | TP 后的 gradient 同步          |
| `ncclReduceScatter`     | 求和，但结果分散存储（GPU i 只拿到 1/N 的数据） | 优化的 all_reduce 第一步       |
| `ncclAllGather`         | 每个 GPU 的局部数据拼接，结果分发到所有 GPU     | TP 推理的 activation 分发      |
| `ncclBroadcast`         | GPU 0 的数据复制到所有 GPU                      | 模型权重广播                   |
| `ncclReduce`            | 求和到 root GPU                                 | 仅 rank 0 需要的 reduction     |
| `ncclSend` / `ncclRecv` | 点对点传输                                      | Pipeline Parallel 的 send/recv |

### 5.4 NCCL 与 CUDA Stream 集成

NCCL 的每个通信操作需要一个 CUDA stream 参数：

```c
cudaStream_t s;
cudaStreamCreate(&s);

// 在 stream s 中排队 all_reduce，该 stream 后续的 kernel 自动等待
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, s);
my_kernel<<<grid, block, 0, s>>>(recvbuff, N);

cudaStreamSynchronize(s);  // 或者用 event 来同步
```

**关键点**：NCCL 操作在 GPU 上执行，不是 CPU 阻塞——你需要用流同步来等待完成。

#### 5.4.1 NCCL 初始化与监控：从 `ncclResult_t` 到健康检查

```c
ncclResult_t res = ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, s);
if (res != ncclSuccess) {
    fprintf(stderr, "ncclAllReduce failed: %s\n", ncclGetErrorString(res));
}
```

NCCL 的所有 API 返回 `ncclResult_t`。常见错误：

- `ncclSystemError`：设备无法访问（P2P 不可用、NVLink 链路故障）
- `ncclInternalError`：通常意味着 NCCL 内部检测到拓扑异常
- `ncclInvalidArgument`：count 或 data type 不匹配

对于长时间运行的训练/推理任务，NCCL 还提供 **health check** 机制：

```c
ncclCommGetAsyncError(comm, &asyncErr);
if (asyncErr != ncclSuccess) {
    // 通信组中出现异步错误——需要重新初始化或 abort
}
```

在训练 loop 中每 N 步（如每 100 步）做一次 `ncclCommGetAsyncError` 检测是一种常见做法，可以在 NVLink 链路故障时提前发现而非等待 watch dog timeout。

---

## 6. 拓扑感知——"nvidia-smi topo -m 告诉你的事"

### 6.1 拓扑矩阵解读

在 [GPU 内存管理](12_gpu_memory_management.md) 第 3 节中展示过一个 8×H100 的拓扑。这里补充解读每种连接类型的含义：

```bash
nvidia-smi topo -m
```

| 标记   | 含义                   | 有效带宽        | 内核路径           |
| ------ | ---------------------- | --------------- | ------------------ |
| `NV18` | NVLink 4.0，18 条 link | ~900 GB/s 双向  | GPU↔GPU 直接       |
| `NV12` | NVLink 3.0，12 条 link | ~600 GB/s 双向  | GPU↔GPU 直接       |
| `NVL`  | NVLink（未指定代数）   | 取决于具体 GPU  | GPU↔GPU 直接       |
| `PIX`  | 同一 PCIe switch 下    | ~32 GB/s (Gen4) | 经 PCIe switch P2P |
| `PHB`  | 同一 PCIe host bridge  | ~32 GB/s (Gen4) | 经 host bridge P2P |
| `SYS`  | 需经 CPU / QPI / UPI   | ~13 GB/s (Gen4) | CPU 中转，最慢     |
| `NODE` | 同一 NUMA node         | 同 PIX/PHB      |                    |

**关键规则**：NVLink 直连 → 最好；PIX/PHB → 尚可；SYS → 避免跨 SYS 组做 TP。

#### 6.1.1 `nvidia-smi topo -p2p r`：P2P 可用性矩阵

`-m` 给你的是拓扑类型的"上限"（NVLink 存在），但**不保证 P2P 已启用**。用 `-p2p r` 查看实际的 P2P 状态：

```bash
nvidia-smi topo -p2p r
```

```text
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
 GPU0    X    OK    OK    OK    OK    OK    OK    OK
 GPU1    OK    X    OK    OK    OK    OK    OK    OK
 ...
```

`OK` = P2P 实际可用，`NS` = Not Supported（或者被 ACS 禁用）。这个矩阵是 `cudaDeviceCanAccessPeer` 的 `nvidia-smi` 版本——两者应该一致。

### 6.2 NVSwitch 域——A100/H100 的拓扑关键

8×A100-SXM4 的典型拓扑：

```text
NVSwitch Domain 0           NVSwitch Domain 1
┌─────────────────────┐    ┌─────────────────────┐
│ GPU0  GPU1  GPU2  GPU3│  │ GPU4  GPU5  GPU6  GPU7│
│  └──4×NVSwitch 全互联──┘  │  └──4×NVSwitch 全互联──┘
└─────────────────────┘    └─────────────────────┘
         │                          │
         └──── PCIe x16 ────────────┘
               跨 domain: SYS
```

- **Domain 内**：任意两个 GPU 通过 NVSwitch 以 NVLink 全互联，P2P 带宽约 300 GB/s 单向（A100 NVLink 3.0），实测 ~269 GB/s
- **跨 Domain**：必须经过 CPU（SYS），带宽骤降 35×
- **TP 分配原则**：TP 组内的 GPU 必须在同一个 NVSwitch domain 内——否则每个 token 的 all_reduce 都是瓶颈

### 6.3 `nvidia-smi topo -c`：查看各 GPU 的 CPU 亲和性

这告诉你 GPU 连接在哪个 CPU（影响 NUMA 分配——见 [CUDA NUMA API](05_cuda_numa_api.md)）：

```bash
nvidia-smi topo -c
```

```text
        CPU Affinity
GPU0    0-23,96-119
GPU1    0-23,96-119
...
GPU4    24-47,120-143
```

与 `numactl --hardware` 的输出对应——GPU 0-3 连接到 NUMA node 0，GPU 4-7 连接到 NUMA node 1。Pinned memory 分配时要确保绑在正确的 NUMA node 上。

---

## 7. 权衡——P2P vs NCCL vs CPU 中转

### 7.1 决策树

```text
跨 GPU 通信？
  ├─ 操作是 all_reduce / all_gather / reduce_scatter？
  │   └─ 直接用 NCCL。不要手写。
  │      （NCCL 自动选算法、自动利用拓扑、自动 overlap compute+copy）
  │
  ├─ 操作是点对点 send/recv？
  │   ├─ 仅 2 个 GPU、NVLink 直连、不需要 collective
  │   │   └─ 可以用 cudaMemcpyPeerAsync。简单、直接。
  │   ├─ ≥3 个 GPU
  │   │   └─ 用 NCCL ncclSend/ncclRecv。手写的 multi-P2P 同步很快失控。
  │   └─ 跨节点（无 NVLink）
  │       └─ 用 NCCL（GPU Direct RDMA over InfiniBand）
  │
  └─ 操作只是问"GPU X 能不能访问 GPU Y 的内存"？
      ├─ 是 → cudaDeviceEnablePeerAccess + cudaMemcpyPeer
      └─ 否 → CPU 中转（float *h_buf; cudaMemcpy D2H + H2D）
```

### 7.2 什么时候不能依赖 P2P

- **跨 NUMA node 的 GPU**：即使在单台 8×A100 中，NVSwitch domain 0↔1 之间仍然需要走 CPU
- **消费级 GPU**：RTX 系列不支持 P2P（`cudaDeviceCanAccessPeer` 返回 0），必须 CPU 中转
- **虚拟化环境**：GPU 被切分为 MIG 实例后，不同实例之间 P2P 被禁用
- **容器内的 ACS 限制**：即使硬件支持，容器的 PCIe ACS 策略可能阻止 P2P

---

## 8. 总结——多 GPU 编程选型速查

### 8.1 核心 API 速查

| 任务          | API                                      | 备注                    |
| ------------- | ---------------------------------------- | ----------------------- |
| 选择当前 GPU  | `cudaSetDevice(i)`                       | 影响后续所有 API        |
| 查询 P2P 可用 | `cudaDeviceCanAccessPeer(&ok, i, j)`     | 运行时查询，不能假设    |
| 启用 P2P      | `cudaDeviceEnablePeerAccess(j, 0)`       | 双向各开一次            |
| GPU→GPU 传输  | `cudaMemcpyPeer` / `cudaMemcpyPeerAsync` | 需先 enable peer access |
| 跨 GPU 等待   | `cudaStreamWaitEvent(s, e)`              | 让一个 GPU 等另一个     |
| 集合通信      | `ncclAllReduce` / `ncclAllGather` 等     | 多 GPU TP/DP 的标配     |
| 拓扑查询      | `nvidia-smi topo -m`                     | NVLink vs SYS 一目了然  |

### 8.2 A100 关键指标速查

| 路径                  | 有效带宽 (单向) | 适用                 |
| --------------------- | --------------- | -------------------- |
| NVLink 4.0 ×18 (H100) | ~450 GB/s       | 同一 NVSwitch domain |
| NVLink 3.0 ×12 (A100) | ~300 GB/s       | 同一 NVSwitch domain |
| PCIe Gen4 P2P         | ~32 GB/s        | 跨 PCIe switch       |
| CPU 中转 (Gen4)       | ~13 GB/s        | 最后手段             |
| CPU 中转 (Gen5)       | ~27 GB/s        | 最后手段             |

### 8.3 与其他文章的衔接

| 本文涉及的                         | 详见                                                    |
| ---------------------------------- | ------------------------------------------------------- |
| NVLink/PCIe 拓扑基础               | [GPU 内存管理](12_gpu_memory_management.md) 第 3 节     |
| NUMA 与 GPU 亲和性                 | [CUDA NUMA API](05_cuda_numa_api.md)                    |
| Stream 并发与 Event                | [CUDA Streams 并发实战](07_cuda_streams_concurrency.md) |
| CUDA Graph（多 stream P2P）        | [CUDA Graphs 编程](09_cuda_graphs.md)                   |
| Warp 内通信（单 GPU 内部的另一极） | [Warp-level Programming](14_warp_level_programming.md)  |
