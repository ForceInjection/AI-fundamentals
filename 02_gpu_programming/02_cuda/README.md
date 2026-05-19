# CUDA 编程 (CUDA Programming)

这个目录给出了理解 CUDA 所需的一个完整的入门回路：从 GPU 的分层执行模型出发，看硬件是怎么组织线程的；再回到 CUDA 核心这个计算单元本身，看单条指令如何被并行执行；然后引入 Streams，理解 GPU 上的异步并发；最后从 SIMT 过渡到 Tile-Based，看当代 GPU 编程范式是如何被 Tensor Core 推着向前演进的。

下面的四篇文档基本就按这个顺序展开。

## 1. [GPU 编程导论](01_gpu_programming_introduction.md)

_GPU Architecture and Programming — An Introduction_：

- 介绍了 GPU 的分层执行模型：Grid, Block, Warp, Thread。
- 解释了 SIMT (Single-Instruction Multiple-Threads) 的基本原理。
- 包含架构图解与核心概念辨析。

## 2. [CUDA 核心详解](02_cuda_cores.md)

- 深入解析 Nvidia CUDA 核心（CUDA Cores）的硬件架构。
- 探讨计算单元的组成与工作方式。

## 3. [CUDA 流处理](03_cuda_streams.md)

- 详细介绍 CUDA Streams 的概念。
- 讲解如何利用流实现并发执行（计算与数据传输的重叠）。
- 异步编程模型的基础。

## 4. [SIMT 到 Tile-Based 编程范式](04_simt_vs_tile_based.md)

- **从 SIMT 到 Tile-Based：GPU 编程范式的演进与实战解析**
- 剖析 NVIDIA cuTile 编程模型。
- 对比传统 SIMT (Thread 视角) 与 Tile-Based (Block/Tile 视角) 的编程思维。
- 以矩阵乘法 (GEMM) 为例展示 Tensor Core 的抽象与使用。

## 5. [CUDA NUMA API 编程实践](05_cuda_numa_api.md)

- 单 GPU 环境下的 NUMA 亲和性管理。
- `cudaMallocHost` 与 NUMA 节点分配策略。
- `cudaMemAdvise` / `cudaMemPrefetchAsync` 在 Managed Memory 中的应用。
- CPU 亲和性绑定 (`taskset` / `numactl`) 的最佳实践。

## 6. [GPU 原子操作与 PCIe 能力查询](06_device_attributes.md)

- `cudaDeviceGetAttribute` 查询 100+ 种底层硬件能力。
- PCIe 原子操作支持确认（Inbound/Outbound Atomic）。
- Host Native Atomic 与数据中心 GPU 的能力差异。
- RTX 5090 关键属性实测。

## 7. [CUDA Streams 并发实战](07_cuda_streams_concurrency.md)

- 单 GPU 上 H2D + Kernel + D2H 重叠执行的完整 demo。
- 4 个 stream 的实测加速比 2.36x（RTX 5090，2 个 async copy engine）。
- Nsight Systems 可视化概念图与 stream 最佳实践。

## 8. [Kernel Launch 开销测量](08_kernel_launch_latency.md)

- 空 kernel launch 延迟实测：2.6 μs（RTX 5090）。
- 不同 block 数量对 launch 开销的影响。
- CPU vs GPU 决策边界与 CUDA Graph 替代方案。

## 参考资料

- [CUDA 编程简介 - 基础与实践.pdf](./references/CUDA%20%E7%BC%96%E7%A8%8B%E7%AE%80%E4%BB%8B%20-%20%E5%9F%BA%E7%A1%80%E4%B8%8E%E5%AE%9E%E8%B7%B5.pdf)
