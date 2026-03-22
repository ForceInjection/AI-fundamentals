# CUDA 编程 (CUDA Programming)

本目录包含 NVIDIA CUDA 编程的核心概念介绍与基础实践指南。

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

## 5. [CUDA 编程简介 - 基础与实践.pdf](./references/CUDA%20%E7%BC%96%E7%A8%8B%E7%AE%80%E4%BB%8B%20-%20%E5%9F%BA%E7%A1%80%E4%B8%8E%E5%AE%9E%E8%B7%B5.pdf)

- 一份完整的 CUDA 编程入门讲义（PDF 格式）。
- 涵盖环境搭建、基础语法、内存管理与实战案例。
