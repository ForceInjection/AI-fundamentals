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

## 9. [CUDA Graphs 编程](09_cuda_graphs.md)

- 将多次 kernel launch + memcpy 合并为一次 graph launch。
- 覆盖 Stream Capture 与 Manual API 两种创建方式。
- 生命周期：录制 → 实例化 → 启动 → 更新。
- A100 实测：Instantiation ~33 μs, Repeat Launch ~2.3 μs。

## 10. [Reduction：从朴素实现到 Warp Shuffle](10_reduction.md)

- GPU 并行编程最经典的教学案例，展示从 0.048ms 到 0.022ms 的逐级优化（2.18× 加速）。
- 8 个 kernel 变体：interleaved → sequential → warp shuffle → template → Cooperative Groups。
- A100 实测 `--shmoo` 完整性能表。
- Warp shuffle 原理与 shared memory / register 性能对比。

## 11. [Tensor Core GEMM 性能实测](11_tensor_core_gemm.md)

- 运行官方 `cudaTensorCoreGemm` / `bf16TensorCoreGemm`，A100 实测 FP16 52.5 TFLOPS、BF16 90.9 TFLOPS。
- 理论峰值对标（156 TFLOPS dense）与矩阵大小对利用率的影响。
- WMMA API 编程模型 + "算力-带宽-算数密度"三角分析。

## 12. [GPU 内存管理——从推理工程师的日常问题出发](12_gpu_memory_management.md)

- 以 Linux 概念为类比，系统讲解 GPU 显存管理的核心机制：虚拟内存与碎片化、DMA 传输（pinned vs pageable）、NVLink/PCIe 拓扑、跨进程共享（CUDA IPC）、GPUDirect Storage、大页与 TLB、内存层级。
- 配套可交互概念图：[gpu-memory-visual.html](gpu-memory-visual.html)（6 个图层：物理拓扑 / 内存层级 / DMA 路径 / MMU 页表 / 碎片化 / 跨进程共享）。
- 包含 CPU ↔ GPU 命令对照附录和性能诊断决策树。

## 13. [Shared Memory 与 Bank Conflict](13_shared_memory_bank_conflict.md)

- 从 Bank 的硬件拓扑出发，系统讲解 Bank Conflict 的产生机制（stride 访问、列优先写入），以及三种消除方法：Padding（空间换时间）、Swizzle（地址 XOR 变换）、Reorder（算法层面重构）。
- 包含三个实战场景拆解：GEMM 的 tile 宽度为什么必须是 32 的倍数、FlashAttention 中的 XOR Swizzle 应用、Reduction 的 Sequential Addressing 为何天然无冲突。
- 跨架构迁移注意事项（V100 4B Bank → A100/H100 的 FP16 双通道 Bank 判定差异）。
- 附带 Bank Conflict 诊断 SOP 和 Python 验证脚本。

## 14. [Warp-level Programming——从 Shuffle 到 Cooperative Groups](14_warp_level_programming.md)

- 系统讲解 Warp 原语体系：Shuffle（`__shfl_sync`/`__shfl_down_sync`/`__shfl_up_sync`/`__shfl_xor_sync`）、Vote（`__ballot_sync`/`__all_sync`/`__any_sync`）、Match（`__match_any_sync`/`__match_all_sync`）、`__activemask`。
- 涵盖 Cooperative Groups `tiled_partition` 现代抽象与 A100 硬件 `__reduce_add_sync` 加速指令。
- A100 实测验证：Warp Reduce、Prefix Sum Scan、Ballot Filter、Match Dedup。

## 15. [Multi-GPU CUDA 编程——从 P2P 到 NCCL 的多卡协作](15_multi_gpu_programming.md)

- 系统讲解多 GPU 编程体系：多 device 管理（`cudaSetDevice`）、Peer Access（`cudaDeviceEnablePeerAccess`）、P2P Memcpy（NVLink 269 GB/s vs CPU 中转 13 GB/s）、跨 GPU Stream/Event 同步。
- 涵盖 NCCL 集合通信基础（`ncclAllReduce` 等）、拓扑感知（NVSwitch domain、`nvidia-smi topo -m`）。
- A100 实测验证：Peer Access、P2P 带宽 20.7× 加速比、跨 GPU 同步、NCCL AllReduce。

## 16. [异步拷贝与现代 Pipeline——从 cp.async 到 memcpy_async](16_async_copy_pipeline.md)

- 系统讲解异步数据搬运技术：传统同步 Load 的瓶颈、`cp.async` PTX 指令（SM80+）、Pipeline 原语（`cuda::pipeline`）、`cuda::memcpy_async` C++ API。
- 涵盖 double buffering / triple buffering 的 Shared Memory 布局与 buffer 管理。
- 实战：BF16 GEMM 的 `cp.async` loop、FlashAttention 的 tile pipeline、通用 tile-based kernel 的 pipeline 模板。
- A100 验证：1-stage sync vs 2-stage double buffer 对比（同步 `ld` 实现，加速比有限——证明异步 DMA 是 pipeline 收益的核心）。

## 17. [CUDA 性能调优方法论——从 Roofline Model 到 ncu 诊断](17_roofline_optimization.md)

- 系列收官文章，建立分级诊断框架：Roofline Model（天花板分析）→ ncu 三指标（瓶颈定位）→ 6 级逐级优化 Checklist。
- 案例：朴素 GEMM 从 Level 0 → Level 6 的完整调优过程（规划加速比 ~80×）。
- 包含优化边界分析（边际收益递减、"够好了"的判断标准）。

## 参考资料

- [CUDA 编程简介 - 基础与实践.pdf](./references/CUDA%20%E7%BC%96%E7%A8%8B%E7%AE%80%E4%BB%8B%20-%20%E5%9F%BA%E7%A1%80%E4%B8%8E%E5%AE%9E%E8%B7%B5.pdf)
