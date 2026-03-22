# nvbandwidth 深度解析：NVIDIA GPU 带宽测量工具全指南

## 一、工具概述

### 1.1 什么是 nvbandwidth？

nvbandwidth 是 NVIDIA 官方开发的一款专门用于测量 GPU 内存带宽的高性能工具。简单来说，它把“带宽到底跑到多少”这件事讲清楚，也让不同传输路径的性能差异一目了然。它能够精确量化各种数据传输场景下的带宽表现，包括：

- **PCIe 带宽**：CPU 与 GPU 之间的数据传输
- **NVLink 带宽**：GPU 与 GPU 之间的高速互联
- **IMEX 支撑的跨节点访问**：多节点 GPU 内存访问场景

### 1.2 核心特点

| 特性       | 说明                                                                |
| ---------- | ------------------------------------------------------------------- |
| 双引擎测试 | 支持 Copy Engine (CE) 和 Streaming Multiprocessor (SM) 两种复制方式 |
| 多拓扑支持 | Host↔Device、Device↔Device 单/双向测试                              |
| 多节点能力 | 基于 MPI + IMEX 服务的跨节点 GPU 内存带宽测量                       |
| 精确测量   | Spin Kernel 排除调度开销，CUDA Events 计时，结果取中位数            |

**当前版本**：v0.8（基于 version.h）

---

## 二、环境搭建与使用说明

### 2.1 系统要求

**软件依赖**：

- CUDA Toolkit: `11.x+`（多节点需 `12.3+`）
- NVIDIA Driver: `550+`（多节点）
- 编译器: GCC `7.x+`（支持 C++17）
- CMake: `3.20+`
- Boost: `libboost-program-options-dev`

> **驱动兼容性说明**：CUDA Toolkit 版本必须与 NVIDIA 驱动版本兼容。例如：
>
> - 驱动 570.x 支持 CUDA ≤ 12.8
> - CUDA 13.x 需要更新的驱动版本
> - GDS (GPUDirect Storage) 用户可能需要特定 CUDA 版本
>
> 使用 `nvidia-smi` 查看驱动支持的 CUDA 版本。如需多版本共存，可安装多个 CUDA Toolkit 到不同目录（如 `/usr/local/cuda-12.8` 和 `/usr/local/cuda-13.1`），编译时指定路径：
>
> ```bash
> # 使用特定 CUDA 版本编译
> PATH=/usr/local/cuda-12.8/bin:$PATH cmake ..
> make -j$(nproc)
> ```

**硬件要求**：

- CUDA-enabled GPU
- 兼容的 NVIDIA 显示驱动

### 2.2 安装步骤

**Ubuntu/Debian**：

```bash
# 安装依赖
apt install libboost-program-options-dev

# 或使用提供的脚本一键安装（会同时安装依赖并编译）
sudo ./debian_install.sh
```

**Fedora**：

```bash
sudo dnf -y install boost-devel
```

**编译构建**：

```bash
# 方式一：源码目录内构建（单节点版本）
cmake .
make -j$(nproc)

# 方式二：独立构建目录（推荐，保持源码整洁）
mkdir build && cd build
cmake ..
make -j$(nproc)

# 多节点版本（需要 MPI 环境）
cmake -DMULTINODE=1 .
make -j$(nproc)
```

**构建输出示例**：

```text
-- The CUDA compiler identification is NVIDIA 13.1.115
-- The CXX compiler identification is GNU 13.3.0
-- Found Boost: /usr/lib/x86_64-linux-gnu/cmake/Boost-1.83.0
-- Detecting CUDA Arch to set CMAKE_CUDA_ARCHITECTURES
-- Configuring done
-- Generating done
-- Build files have been written to: /path/to/build

[  8%] Building CXX object CMakeFiles/nvbandwidth.dir/testcase.cpp.o
[ 33%] Building CUDA object CMakeFiles/nvbandwidth.dir/kernels.cu.o
...
[100%] Linking CXX executable nvbandwidth
[100%] Built target nvbandwidth
```

**构建产物**：

- 可执行文件：`./nvbandwidth`（构建目录内）
- 支持 CUDA 架构：70, 75, 80, 86, 89, 90, 100（CUDA 13.0+ 不再支持 52）

**依赖检查**：

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 CMake 版本
cmake --version

# 检查 Boost 是否安装
dpkg -l | grep boost  # Debian/Ubuntu
```

### 2.3 命令行参数详解

```bash
./nvbandwidth -h
```

| 参数                 | 简写 | 默认值 | 说明                               |
| -------------------- | ---- | ------ | ---------------------------------- |
| `--help`             | `-h` | -      | 显示帮助信息                       |
| `--bufferSize`       | `-b` | 512    | 拷贝缓冲区大小（MiB）              |
| `--list`             | `-l` | -      | 列出所有可用测试用例               |
| `--testcase`         | `-t` | -      | 指定要运行的测试用例（名称或索引） |
| `--testcasePrefixes` | `-p` | -      | 按前缀运行测试用例                 |
| `--verbose`          | `-v` | false  | 详细输出                           |
| `--skipVerification` | `-s` | false  | 跳过数据校验                       |
| `--disableAffinity`  | `-d` | false  | 禁用自动 CPU 亲和性控制            |
| `--testSamples`      | `-i` | 3      | 基准测试迭代次数（采样次数）       |
| `--useMean`          | `-m` | false  | 使用平均值代替中位数               |
| `--json`             | `-j` | false  | 以 JSON 格式输出而非纯文本         |

### 2.4 使用示例

**运行所有测试**：

```bash
./nvbandwidth
```

**运行特定测试**：

```bash
./nvbandwidth -t device_to_device_memcpy_read_ce
```

**指定缓冲区大小和迭代次数**：

```bash
./nvbandwidth -b 1024 -i 5
```

**多节点测试**（需先启动 IMEX 服务）：

```bash
# 启动 IMEX 服务
sudo systemctl start nvidia-imex.service

# 运行多节点测试
mpirun --allow-run-as-root --map-by ppr:4:node --bind-to core -np 8 \
  --hostfile /etc/nvidia-imex/nodes_config.cfg \
  ./nvbandwidth -p multinode
```

---

## 三、源码架构与原理分析

### 3.1 整体架构图

```text
nvbandwidth
├── 主程序入口 (nvbandwidth.cpp)
│   ├── 命令行参数解析（Boost program_options）
│   ├── 测试用例管理与调度
│   └── 结果输出（Output / JsonOutput）
├── 测试用例层 (testcase.h / testcase.cpp)
│   ├── Testcase 基类定义
│   └── 辅助方法（allToOneHelper / latencyHelper 等）
├── CE 测试用例 (testcases_ce.cpp)
│   ├── HostToDeviceCE / DeviceToHostCE
│   ├── DeviceToDeviceReadCE / WriteCE
│   └── AllToOne / OneToAll 等聚合测试
├── SM 测试用例 (testcases_sm.cpp)
│   ├── HostToDeviceSM / DeviceToHostSM
│   ├── DeviceToDeviceReadSM / WriteSM
│   └── Latency 测试
├── 内存操作层 (memcpy.h / memcpy.cpp)
│   ├── MemcpyBuffer（内存缓冲区抽象）
│   ├── HostBuffer（主机内存，NUMA 亲和性）
│   ├── DeviceBuffer（设备内存，Peer Access）
│   ├── MemcpyOperation（拷贝操作编排）
│   └── MemcpyInitiator（CE/SM 拷贝发起者）
├── CUDA 内核层 (kernels.cuh / kernels.cu)
│   ├── simpleCopyKernel（简单拷贝，小缓冲区）
│   ├── stridingMemcpyKernel（跨步拷贝，大缓冲区优化）
│   ├── splitWarpCopyKernel（双向拷贝）
│   ├── ptrChasingKernel（延迟测量）
│   ├── spinKernel（阻塞同步）
│   └── memsetKernel / memcmpKernel（数据校验）
├── 输出层 (output.h / output.cpp)
│   ├── Output（文本输出）
│   └── JsonOutput（JSON 格式输出）
└── 多节点支持 (multinode_memcpy.h / multinode_memcpy.cpp)
    ├── MultinodeDeviceBuffer（多节点缓冲区抽象）
    ├── MultinodeMemoryAllocationUnicast（单播内存）
    ├── MultinodeMemoryAllocationMulticast（多播内存）
    └── NodeHelperMulti（MPI 同步辅助）
```

### 3.2 核心类设计

#### 3.2.1 Testcase 基类

```cpp
class Testcase {
 protected:
    std::string key;      // 测试用例唯一标识
    std::string desc;     // 测试描述

    // 静态过滤方法
    static bool filterHasAccessiblePeerPairs();    // 检查是否有可访问的对等 GPU
    static bool filterSupportsMulticast();          // 检查是否支持多播

    // 辅助方法
    void allToOneHelper(...);    // AllToOne 测试辅助
    void oneToAllHelper(...);    // OneToAll 测试辅助
    void latencyHelper(...);     // 延迟测试辅助

 public:
    Testcase(std::string key, std::string desc);
    virtual ~Testcase() {}

    std::string testKey();
    std::string testDesc();

    virtual bool filter() { return true; }  // 过滤条件检查
    virtual void run(unsigned long long size, unsigned long long loopCount) = 0;
};
```

所有具体测试用例都继承自 `Testcase` 基类，实现了自己的 `run()` 方法。`filter()` 方法用于检查当前系统是否满足测试条件（如是否有对等 GPU 可访问）。

#### 3.2.2 内存缓冲区抽象

```cpp
// 基类
class MemcpyBuffer {
 protected:
    void* buffer{};
    size_t bufferSize;
 public:
    virtual int getBufferIdx() const = 0;
    virtual CUcontext getPrimaryCtx() const = 0;
};

// 主机内存
class HostBuffer : public MemcpyBuffer {
    // 使用 cuMemHostAlloc 分配可分页锁定内存
    // 自动设置 NUMA 亲和性
};

// 设备内存
class DeviceBuffer : public MemcpyBuffer {
    int deviceIdx;
    CUcontext primaryCtx;
    // 使用 cuMemAlloc 分配设备内存
};
```

#### 3.2.3 拷贝操作抽象

```cpp
class MemcpyOperation {
    std::shared_ptr<NodeHelper> nodeHelper;
    std::shared_ptr<MemcpyInitiator> memcpyInitiator;

 public:
    double doMemcpy(const MemcpyBuffer &src, const MemcpyBuffer &dst);
    std::vector<double> doMemcpyVector(...);
};
```

`MemcpyInitiator` 有两个主要实现：

- `MemcpyInitiatorCE`：使用 `cuMemcpyAsync` 进行拷贝
- `MemcpyInitiatorSM`：使用自定义 CUDA 内核进行拷贝

### 3.3 测量原理详解

#### 3.3.1 Spin Kernel 机制

为了排除 CUDA 内核启动和流队列的调度开销，nvbandwidth 采用了 **Spin Kernel** 技术：

```cpp
// 1. 重置阻塞变量
*blockingVarHost = 0;

// 2. 启动 Spin Kernel，GPU 在该流上等待信号
spinKernel(blockingVarHost, stream);  // GPU 自旋等待 blockingVarHost == 1

// 3. 在 Spin Kernel 运行期间，入队所有测量事件
//    这些操作会被 GPU 驱动排队，但尚未执行
cuEventRecord(startEvent, stream);
// ... 执行 loopCount 次拷贝操作 ...
cuEventRecord(endEvent, stream);

// 4. 释放信号，Spin Kernel 退出，测量真正开始
*blockingVarHost = 1;  // CPU 写入，GPU 检测到后退出 Spin Kernel

// 5. 等待所有操作完成
cuStreamSynchronize(stream);
```

**预热机制**：在正式测量前，会先执行 `WARMUP_COUNT`（默认 4 次）次预热拷贝，确保缓存和流水线处于稳定状态。

**关键点**：Spin Kernel 确保 CUDA Events 和所有拷贝操作在 GPU 上**同时就绪**后才真正开始执行，从而排除了 CPU 端入队开销对测量的影响。

**流程图**：

```text
CPU Side                    GPU Side
─────────────────────────────────────────────────
spinKernel(blockingVar) ──→ 等待 blockingVar == 1
                            │
cuEventRecord(start) ─────→ 记录开始时间戳
                            │
memcpy operations ────────→ 执行拷贝
                            │
cuEventRecord(end) ───────→ 记录结束时间戳
                            │
*blockingVar = 1 ─────────→ Spin Kernel 退出
                            │
cuStreamSynchronize ──────→ 等待完成
```

#### 3.3.2 带宽计算

**基本公式**：

```text
带宽 (GB/s) = (拷贝数据大小 (字节) × 循环次数) / 耗时 (微秒) / 1000
```

**推导过程**：

1. 总传输量 (字节) = 拷贝数据大小 × 循环次数
2. 传输速率 (字节/秒) = 总传输量 / (耗时 × 10⁻⁶) = 总传输量 × 10⁶ / 耗时
3. 传输速率 (GB/s) = 传输速率 (字节/秒) / 10⁹ = 总传输量 / 耗时 / 1000

**CE 拷贝带宽**：

```cpp
// 使用 CUDA Events 计时，elapsedWithEventsInUs 单位为微秒
float timeMs;
cuEventElapsedTime(&timeMs, start, end);
double elapsedWithEventsInUs = timeMs * 1000.0;  // 毫秒转微秒
// bandwidth 单位为 B/s，后续除以 1e9 转换为 GB/s
unsigned long long bandwidth = (size * loopCount * 1000ull * 1000ull) / elapsedWithEventsInUs;
```

**SM 拷贝带宽**：
SM 拷贝会将数据大小截断以适应 GPU 的 SM 数量，确保每个线程处理的数据量一致：

```cpp
// 实际拷贝大小计算
// 1. 获取 GPU 的 SM 数量和每块线程数
int numSm;
cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
unsigned int totalThreadCount = numSm * numThreadPerBlock;  // numThreadPerBlock = 512

// 2. 对于大缓冲区（>= 64 MiB），使用优化内核并截断
if (size >= (smallBufferThreshold * _MiB)) {
    size_t sizeInElement = size / sizeof(uint4);  // 转换为 uint4 元素数
    // 截断为 totalThreadCount 的整数倍
    sizeInElement = totalThreadCount * (sizeInElement / totalThreadCount);
    return sizeInElement * sizeof(uint4);
}
```

**截断原因**：优化的 `stridingMemcpyKernel` 使用跨步访问模式，要求每个线程处理相同数量的元素，以达到最佳带宽性能。

### 3.4 CUDA 内核实现

#### 3.4.1 简单拷贝内核

```cuda
__global__ void simpleCopyKernel(unsigned long long loopCount, uint4 *dst, uint4 *src) {
    for (unsigned int i = 0; i < loopCount; i++) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t offset = idx * sizeof(uint4);
        uint4* dst_uint4 = reinterpret_cast<uint4*>((char*)dst + offset);
        uint4* src_uint4 = reinterpret_cast<uint4*>((char*)src + offset);
        // 使用全局缓存加载/存储指令
        __stcg(dst_uint4, __ldcg(src_uint4));
    }
}
```

#### 3.4.2 跨步拷贝内核（优化版本）

```cuda
__global__ void stridingMemcpyKernel(unsigned int totalThreadCount,
                                      unsigned long long loopCount,
                                      uint4* dst, uint4* src,
                                      size_t chunkSizeInElement) {
    // 每个线程处理一个跨步的数据块
    unsigned long long from = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long bigChunkSizeInElement = chunkSizeInElement / 12;
    dst += from;
    src += from;
    uint4* dstBigEnd = dst + (bigChunkSizeInElement * 12) * totalThreadCount;
    uint4* dstEnd = dst + chunkSizeInElement * totalThreadCount;

    for (unsigned int i = 0; i < loopCount; i++) {
        uint4* cdst = dst;
        uint4* csrc = src;

        // 手动展开，每次处理 12 个 uint4（192 字节）
        while (cdst < dstBigEnd) {
            uint4 pipe_0 = *csrc; csrc += totalThreadCount;
            // ... 省略中间 pipe_1 ~ pipe_10 ...
            uint4 pipe_11 = *csrc; csrc += totalThreadCount;

            *cdst = pipe_0; cdst += totalThreadCount;
            // ... 省略中间写入 ...
            *cdst = pipe_11; cdst += totalThreadCount;
        }

        // 处理剩余尾部
        while (cdst < dstEnd) {
            *cdst = *csrc; cdst += totalThreadCount; csrc += totalThreadCount;
        }
    }
}
```

**优化要点**：

1. **循环展开**：减少分支预测失败
2. **流水线处理**：先批量加载，再批量存储
3. **跨步访问**：充分利用内存带宽

#### 3.4.3 双向拷贝内核（Split Warp）

```cuda
__global__ void splitWarpCopyKernel(unsigned long long loopCount, uint4 *dst, uint4 *src) {
    for (unsigned int i = 0; i < loopCount; i++) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int globalWarpId = idx / warpSize;
        unsigned int warpLaneId = idx % warpSize;

        // 奇偶 Warp 交替拷贝方向
        if (globalWarpId & 0x1) {
            // 奇数 Warp：src -> dst
            dst_uint4 = dst + (globalWarpId * warpSize + warpLaneId);
            src_uint4 = src + (globalWarpId * warpSize + warpLaneId);
        } else {
            // 偶数 Warp：dst -> src
            dst_uint4 = src + (globalWarpId * warpSize + warpLaneId);
            src_uint4 = dst + (globalWarpId * warpSize + warpLaneId);
        }

        __stcg(dst_uint4, __ldcg(src_uint4));
    }
}
```

### 3.5 延迟测量原理

使用**指针追踪（Pointer Chase）**技术测量访问延迟：

```cuda
struct LatencyNode {
    struct LatencyNode *next;
};

__global__ void ptrChasingKernel(struct LatencyNode *data, size_t size,
                                  unsigned int accesses, unsigned int targetBlock) {
    struct LatencyNode *p = data;
    if (blockIdx.x != targetBlock) return;

    // 顺序访问链表，产生依赖链
    for (auto i = 0; i < accesses; ++i) {
        p = p->next;  // 每次访问依赖上一次结果
    }

    // 防止编译器优化
    if (p == nullptr) __trap();
}
```

**链表初始化**：源码中使用**跨步模式（Stride Pattern）**初始化链表，跨步长度为 16：

```cpp
void Testcase::latencyHelper(const MemcpyBuffer &dataBuffer, bool measureDeviceToDeviceLatency) {
    uint64_t n_ptrs = dataBuffer.getBufferSize() / sizeof(struct LatencyNode);

    if (measureDeviceToDeviceLatency) {
        // 设备侧链表：写入设备地址
        for (uint64_t i = 0; i < n_ptrs; i++) {
            struct LatencyNode node;
            size_t nextOffset = ((i + strideLen) % n_ptrs) * sizeof(struct LatencyNode);
            node.next = (struct LatencyNode*)(dataBuffer.getBuffer() + nextOffset);
            cuMemcpyHtoD(dataBuffer.getBuffer() + i * sizeof(struct LatencyNode),
                         &node, sizeof(struct LatencyNode));
        }
    } else {
        // 主机侧链表：写入主机地址
        struct LatencyNode* hostMem = (struct LatencyNode*)dataBuffer.getBuffer();
        for (uint64_t i = 0; i < n_ptrs; i++) {
            hostMem[i].next = &hostMem[(i + strideLen) % n_ptrs];
        }
    }
}
```

**原理**：通过构建一个固定跨步的链表，强制 GPU 进行串行内存访问，从而测量真实的访问延迟。固定跨步模式可以减少缓存局部性影响，使测量结果更具代表性。

**注意**：延迟测试固定使用 2 MiB 缓冲区，`--bufferSize` 参数对延迟测试无效。源码中会强制设置：

```cpp
// nvbandwidth.cpp
if (test->testKey() == "host_device_latency_sm" || test->testKey() == "device_to_device_latency_sm") {
    test->run(2 * _MiB, loopCount);  // 固定使用 2MB 缓冲区
}
```

**两种延迟测试的区别**：

| 测试用例                      | 缓冲区位置       | 访问方式                                 |
| ----------------------------- | ---------------- | ---------------------------------------- |
| `host_device_latency_sm`      | 主机内存（Host） | GPU 执行指针追踪，访问 Host 内存         |
| `device_to_device_latency_sm` | 远端 GPU 内存    | 本地 GPU 执行指针追踪，访问远端 GPU 内存 |

---

## 四、测试用例详解

### 4.1 单节点测试用例

**Host ↔ Device 基础测试**：

| 测试用例                   | 描述                   |
| -------------------------- | ---------------------- |
| `host_to_device_memcpy_ce` | Host 到 Device CE 拷贝 |
| `device_to_host_memcpy_ce` | Device 到 Host CE 拷贝 |
| `host_to_device_memcpy_sm` | Host 到 Device SM 拷贝 |
| `device_to_host_memcpy_sm` | Device 到 Host SM 拷贝 |

**Host ↔ Device 双向测试**：

| 测试用例                                 | 描述                                              |
| ---------------------------------------- | ------------------------------------------------- |
| `host_to_device_bidirectional_memcpy_ce` | Host→Device 拷贝测量，同时运行 Device→Host 干扰流 |
| `device_to_host_bidirectional_memcpy_ce` | Device→Host 拷贝测量，同时运行 Host→Device 干扰流 |
| `host_to_device_bidirectional_memcpy_sm` | SM 双向拷贝（Split Warp）                         |
| `device_to_host_bidirectional_memcpy_sm` | SM 双向拷贝（Split Warp）                         |

**Device ↔ Device 测试**（需要 P2P 访问能力）：

| 测试用例                                         | 描述                                |
| ------------------------------------------------ | ----------------------------------- |
| `device_to_device_memcpy_read_ce`                | GPU 间 CE 读操作（从远端设备读取）  |
| `device_to_device_memcpy_write_ce`               | GPU 间 CE 写操作（写入远端设备）    |
| `device_to_device_memcpy_read_sm`                | GPU 间 SM 读操作                    |
| `device_to_device_memcpy_write_sm`               | GPU 间 SM 写操作                    |
| `device_to_device_bidirectional_memcpy_read_ce`  | CE 双向读（干扰流反向）             |
| `device_to_device_bidirectional_memcpy_write_ce` | CE 双向写（干扰流反向）             |
| `device_to_device_bidirectional_memcpy_read_sm`  | SM 双向读，报告 read1/read2/total   |
| `device_to_device_bidirectional_memcpy_write_sm` | SM 双向写，报告 write1/write2/total |

**聚合带宽测试**：

| 测试用例                              | 描述                                     |
| ------------------------------------- | ---------------------------------------- |
| `all_to_host_memcpy_ce`               | 所有 GPU 同时向 Host 拷贝，测量聚合带宽  |
| `all_to_host_memcpy_sm`               | SM 版本的 AllToHost                      |
| `host_to_all_memcpy_ce`               | Host 同时向所有 GPU 拷贝，测量聚合带宽   |
| `host_to_all_memcpy_sm`               | SM 版本的 HostToAll                      |
| `all_to_host_bidirectional_memcpy_ce` | 双向版本，其他 GPU 产生干扰流量          |
| `all_to_host_bidirectional_memcpy_sm` | SM 版本双向                              |
| `host_to_all_bidirectional_memcpy_ce` | 双向版本，其他 GPU 产生干扰流量          |
| `host_to_all_bidirectional_memcpy_sm` | SM 版本双向                              |
| `all_to_one_write_ce`                 | 所有 GPU 向单个 GPU 写入，测量总入站带宽 |
| `all_to_one_read_ce`                  | 单个 GPU 从所有 GPU 读取，测量总出站带宽 |
| `one_to_all_write_ce`                 | 单个 GPU 向所有 GPU 写入，测量总出站带宽 |
| `one_to_all_read_ce`                  | 所有 GPU 从单个 GPU 读取，测量总入站带宽 |
| `all_to_one_write_sm`                 | SM 版本 AllToOne Write                   |
| `all_to_one_read_sm`                  | SM 版本 AllToOne Read                    |
| `one_to_all_write_sm`                 | SM 版本 OneToAll Write                   |
| `one_to_all_read_sm`                  | SM 版本 OneToAll Read                    |

**延迟与本地拷贝测试**：

| 测试用例                      | 描述                                              |
| ----------------------------- | ------------------------------------------------- |
| `host_device_latency_sm`      | Host-Device 访问延迟（缓冲区在 Host，GPU 访问）   |
| `device_to_device_latency_sm` | GPU 间访问延迟（缓冲区在远端 GPU，本地 GPU 访问） |
| `device_local_copy`           | GPU 内部内存拷贝带宽                              |

说明：单节点版本共 35 个测试用例（索引 0-34），使用 `./nvbandwidth -l` 查看完整列表。

### 4.2 双向测试原理

**CE 双向测试**：

- 使用两个独立的 CUDA Stream
- Host↔Device 双向：仅第一条拷贝（被测量的流）计入结果，另一条为干扰流量
- Device↔Device 双向：仅测量流的带宽被报告，干扰流在相反方向同时运行但不被测量

**SM 双向测试**：

- 使用 Split Warp 内核（`splitWarpCopyKernel`）
- 奇数 Warp 执行正向拷贝，偶数 Warp 执行反向拷贝
- **Host↔Device 双向 SM 测试**：报告的带宽值为测量流带宽的一半（代码中 `getAdjustedBandwidth` 返回 `bandwidth / 2`），用于估算双向同时传输时每个方向的有效带宽
- **Device↔Device 双向 SM 测试**：输出三组数据
  - `read1` / `write1`：第一个方向的带宽
  - `read2` / `write2`：第二个方向的带宽
  - `total`：两个方向带宽之和

### 4.3 多节点测试

多节点版本需要：

1. **MPI 环境**：用于跨节点进程通信
2. **IMEX 服务**：NVIDIA Internode Memory Exchange Service
3. **NVLink 多节点部署**：物理连接支持

**多节点测试用例**（共 13 个，需要在编译时启用 `MULTINODE=1`）：

| 测试用例                                                   | 描述                     |
| ---------------------------------------------------------- | ------------------------ |
| `multinode_device_to_device_memcpy_read_ce`                | 跨节点 CE 读操作         |
| `multinode_device_to_device_memcpy_write_ce`               | 跨节点 CE 写操作         |
| `multinode_device_to_device_memcpy_read_sm`                | 跨节点 SM 读操作         |
| `multinode_device_to_device_memcpy_write_sm`               | 跨节点 SM 写操作         |
| `multinode_device_to_device_bidirectional_memcpy_read_ce`  | 跨节点双向 CE 读         |
| `multinode_device_to_device_bidirectional_memcpy_write_ce` | 跨节点双向 CE 写         |
| `multinode_device_to_device_bidirectional_memcpy_read_sm`  | 跨节点双向 SM 读         |
| `multinode_device_to_device_bidirectional_memcpy_write_sm` | 跨节点双向 SM 写         |
| `multinode_device_to_device_all_to_one_write_sm`           | 所有序节点向单个节点写入 |
| `multinode_device_to_device_all_from_one_read_sm`          | 单节点从所有序节点读取   |
| `multinode_device_to_device_broadcast_one_to_all_sm`       | 单节点广播到所有序节点   |
| `multinode_device_to_device_broadcast_all_to_all_sm`       | 所有节点同时广播         |
| `multinode_bisect_write_ce`                                | 对分带宽测试             |

**关键类**：

```cpp
// 多节点设备缓冲区基类
class MultinodeDeviceBuffer : public MemcpyBuffer {
    int MPI_rank;  // 内存所在的节点（MPI rank）

    CUcontext getPrimaryCtx() const override;  // 返回本地 GPU 上下文
    int getMPIRank() const override;            // 返回所属 MPI rank
};

// 单播内存（点对点访问）
// 内存物理上位于某个节点，其他节点可以远程访问
class MultinodeMemoryAllocationUnicast {
    CUmemGenericAllocationHandle handle;  // 内存分配句柄
    CUmemFabricHandle fh;                 // 跨节点共享句柄
    // 使用 cuMemExportToShareableHandle 导出，通过 MPI_Bcast 共享
};

// 多播内存（广播）
// 写入时自动传播到所有节点，适用于广播场景
class MultinodeMemoryAllocationMulticast {
    CUmemGenericAllocationHandle multicastHandle;  // 多播对象句柄
    CUmulticastObjectProp multicastProp;           // 多播属性
    // 使用 cuMulticastCreate 创建，cuMulticastBindMem 绑定
};

// 本地设备内存（仅单个节点可访问）
class MultinodeDeviceBufferLocal : public MultinodeDeviceBuffer {
    // 仅在 MPI_rank == worldRank 时分配实际内存
};
```

**内存共享流程**：

1. **单播内存**：
   - 源节点调用 `cuMemCreate` 分配内存
   - 通过 `cuMemExportToShareableHandle` 导出句柄
   - 使用 `MPI_Bcast` 广播句柄到其他节点
   - 其他节点通过 `cuMemImportFromShareableHandle` 导入并映射

2. **多播内存**：
   - 使用 `cuMulticastCreate` 创建多播对象
   - 所有节点调用 `cuMulticastAddDevice` 添加设备
   - 各节点分配本地内存并绑定到多播对象

---

## 五、实战案例

### 5.1 案例一：评估服务器 PCIe 带宽

**场景**：新采购的服务器，验证 PCIe 带宽是否达标

```bash
# 运行 Host-Device 相关测试
./nvbandwidth -t host_to_device_memcpy_ce -t device_to_host_memcpy_ce \
              -t host_to_device_bidirectional_memcpy_ce \
              -b 512 -i 5 -v
```

**单向测试预期输出**：

```text
Running host_to_device_memcpy_ce.
memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
           0         1         2         3         4         5         6         7
 0     56.54     56.53     56.53     56.56     56.29     56.47     56.51     53.61

SUM host_to_device_memcpy_ce 449.04

Running device_to_host_memcpy_ce.
memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)
           0         1         2         3         4         5         6         7
 0     57.40     57.40     57.38     57.39     57.36     57.11     57.38     57.38

SUM device_to_host_memcpy_ce 458.81
```

**双向测试预期输出**：

```text
Running host_to_device_bidirectional_memcpy_ce.
memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)
           0         1         2         3         4         5         6         7
 0     18.56     18.37     19.37     19.59     18.71     18.79     18.46     18.61
```

**矩阵解读**：

- 行索引（0）：代表 CPU（Host），单向测试只有一行
- 列索引（0-7）：代表 GPU 设备编号
- 数值：CPU 与对应 GPU 之间的带宽（GB/s）
- SUM：所有 GPU 带宽之和
- PCIe 3.0 x16 理论峰值约 16 GB/s，PCIe 4.0 x16 约 32 GB/s，PCIe 5.0 x16 约 64 GB/s
- 上述示例约 56-57 GB/s，表明使用 PCIe 5.0 x16 接口

说明：示例输出为实际运行结果（RTX 6000D，PCIe 5.0 x16，驱动 570.x）。实际数值因系统配置而异。

### 5.2 案例二：检测 NVLink 拓扑

**场景**：多 GPU 服务器，了解 GPU 间互联拓扑

```bash
# 运行 Device-Device 测试
./nvbandwidth -t device_to_device_memcpy_read_ce \
              -t device_to_device_memcpy_write_ce \
              -t device_to_device_bidirectional_memcpy_read_ce
```

**输出示例**（8 GPU NVLink 全互联系统）：

```text
Running device_to_device_memcpy_write_ce.
memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0      0.00    276.07    276.36    276.14    276.29    276.48    276.55    276.33
1    276.19      0.00    276.29    276.29    276.57    276.48    276.38    276.24
2    276.33    276.29      0.00    276.38    276.50    276.50    276.29    276.31
3    276.19    276.62    276.24      0.00    276.29    276.60    276.29    276.55
4    276.03    276.55    276.45    276.76      0.00    276.45    276.36    276.62
5    276.17    276.57    276.19    276.50    276.31      0.00    276.31    276.15
6    274.89    276.41    276.38    276.67    276.41    276.26      0.00    276.33
7    276.12    276.45    276.12    276.36    276.00    276.57    276.45      0.00
```

**矩阵解读**：

- 行索引：源 GPU（执行读取/写入操作的 GPU）
- 列索引：目标 GPU
- 对角线为 0.00：GPU 不与自身通信
- 非对角线值：GPU 间互联带宽（GB/s）
- NVLink 3.0 单链路约 25 GB/s，多链路聚合可达更高带宽

**无 NVLink 系统的情况**：

如果系统没有 NVLink 或 GPU 之间没有 P2P 访问能力，测试将被 WAIVED：

```text
Running device_to_device_memcpy_read_ce.
WAIVED device_to_device_memcpy_read_ce
  Reason: Test requires accessible peer pairs
```

此时可使用 `nvidia-smi topo -m` 查看拓扑：

```text
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0     X      NODE    NODE    NODE    SYS     SYS     SYS     SYS
GPU1    NODE     X      NODE    NODE    SYS     SYS     SYS     SYS
...
Legend:
  X    = Self
  SYS  = Connection via PCIe + SMP interconnect (跨 NUMA 节点)
  NODE = Connection via PCIe within NUMA node (NUMA 节点内)
  NV#  = Connection via # NVLinks
```

**拓扑判断方法**：

```bash
# 查看 NVLink 拓扑
nvidia-smi topo -m
```

说明：示例输出来自 README.md Usage，实际带宽取决于 NVLink 版本和链路数。

### 5.3 案例三：排查性能问题

**场景**：深度学习训练性能不达预期

```bash
# 全面测试所有带宽
./nvbandwidth -b 1024 -i 10 --json > bandwidth_report.json

# 或只测试特定类型
./nvbandwidth -t host_to_device_memcpy_ce -t device_to_host_memcpy_ce -v
```

**排查清单**：

1. **PCIe 带宽过低**：
   - 检查 GPU 是否插在正确的 PCIe 插槽（通常应使用 CPU 直连的插槽）
   - 使用 `lspci -vv` 检查 PCIe 链路速度和宽度是否降速
   - 确认 PCIe 版本（3.0/4.0/5.0）与预期一致

2. **NVLink 未启用**：
   - 检查 `nvidia-smi topo -m` 输出，确认 NVLink 连接状态
   - 确认驱动版本支持当前 GPU 的 NVLink 功能

3. **NUMA 亲和性问题**：
   - 使用 `nvidia-smi topo -m` 查看 GPU 与 NUMA 节点的对应关系
   - 使用 `-d` 禁用亲和性对比测试，确认是否影响结果

### 5.4 案例四：多节点集群验证

**场景**：NVLink 多节点集群部署后验证

**前置条件**：

- CUDA Toolkit 12.3+
- NVIDIA Driver 550+
- MPI 环境（OpenMPI 或 MPICH）
- NVLink 多节点物理连接

```bash
# 1. 配置节点列表
cat /etc/nvidia-imex/nodes_config.cfg
192.168.1.1
192.168.1.2

# 2. 启动 IMEX 服务（所有节点）
sudo systemctl start nvidia-imex.service

# 3. 运行多节点测试（示例：2 节点，每节点 4 GPU）
mpirun --allow-run-as-root --map-by ppr:4:node --bind-to core -np 8 \
       --hostfile /etc/nvidia-imex/nodes_config.cfg \
       ./nvbandwidth -p multinode

# 4. 本地单节点测试多节点功能（Ampere+ GPU）
mpirun -n 4 ./nvbandwidth -p multinode
```

**注意事项**：

- 每个进程对应一个 GPU，进程数不能超过 GPU 数量
- 所有 MPI rank 必须属于同一个 multinode clique
- 只有 MPI rank 0 会输出 stdout，stderr 由所有进程输出
- 建议只运行 `multinode*` 前缀的测试用例

说明：多节点运行示例与必要前置条件来自 README.md 的 Multinode benchmarks。

---

## 六、性能调优建议

### 6.1 缓冲区大小选择

```cpp
// 关键常量定义 (common.h)
const unsigned long long defaultBufferSize = 512;     // 默认缓冲区 512 MiB
const unsigned long long smallBufferThreshold = 64;   // 小缓冲区阈值 64 MiB
const unsigned long long defaultLoopCount = 16;       // 默认循环次数（每次采样内的 memcpy 迭代次数）
const unsigned int defaultAverageLoopCount = 3;       // 默认采样次数
const unsigned int numThreadPerBlock = 512;           // 每块线程数
const unsigned int strideLen = 16;                    // 延迟测试跨步长度
const unsigned long latencyMemAccessCnt = 1000000;    // 延迟测试访问次数（100 万次）
const unsigned int _MiB = 1024 * 1024;                // 1 MiB 字节数
const unsigned int _2MiB = 2 * _MiB;                  // 2 MiB 字节数
```

**建议**：

- 使用默认 512 MiB 以获得峰值带宽
- 小缓冲区（< 64 MiB）会触发 `simpleCopyKernel`，而非优化的 `stridingMemcpyKernel`
- 超大缓冲区可能受系统内存限制
- 延迟测试固定使用 2 MiB 缓冲区，`--bufferSize` 参数无效

### 6.2 迭代次数设置

```bash
# 默认 3 次采样，取中位数
./nvbandwidth -i 10  # 增加采样次数，结果更稳定
```

**说明**：

- `--testSamples` (-i)：控制采样次数，默认为 3 次
- 每次采样内部会执行 `defaultLoopCount`（默认 16 次）次 memcpy 操作
- 使用 `--useMean` (-m) 可改为取平均值而非中位数

### 6.3 CPU 亲和性

nvbandwidth 默认会自动设置 CPU 亲和性，确保 Host 内存分配在与 GPU 相同的 NUMA 节点上。

```bash
# 禁用亲和性（用于对比测试）
./nvbandwidth -d
```

---

## 七、总结

nvbandwidth 是一款功能强大的 GPU 带宽测量工具，其核心优势在于：

1. **精确性**：Spin Kernel 机制排除调度开销，CUDA Events 精确计时
2. **全面性**：覆盖 CE/SM、单向/双向、单节点/多节点各种场景
3. **专业性**：针对 NVIDIA GPU 架构深度优化，支持 NVLink、PCIe、IMEX 等多种互联

通过深入理解其源码实现，我们可以更好地：

- 诊断系统性能瓶颈
- 验证硬件配置是否正确
- 优化 GPU 应用程序性能

**扩展阅读**：

- [NVIDIA CUDA 编程指南](https://docs.nvidia.com/cuda/)
- [NVLink 技术白皮书](https://www.nvidia.com/en-us/data-center/nvlink/)
- [CUDA Bandwidth Test 示例](https://github.com/NVIDIA/cuda-samples)

---

_本文基于 nvbandwidth 源码分析撰写，如有疑问请参考官方文档或源码注释。_
