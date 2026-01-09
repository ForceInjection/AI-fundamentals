# GPU 编程 (GPU Programming)

本章节涵盖 GPU 编程的多种范式与实践，从主流的 CUDA 到新兴的 Tile-Based 编程。

## 内容概览

### 1. [CUDA 编程](cuda/README.md)

NVIDIA 标准的并行计算架构与编程模型。

- **核心概念**：[CUDA 核心](cuda/cuda_cores_cn.md)、[流处理](cuda/cuda_streams.md)
- **入门指南**：[基础与实践](cuda/CUDA%20编程简介%20-%20基础与实践.pdf)
- **编程导论**：[GPU 编程导论](cuda/gpu_programming_introduction.md)
- **范式对比**：[SIMT vs Tile-Based](cuda/simt_vs_tile_based_programming.md)

### 2. [Tile-Based 编程](tilelang/README.md)

面向 Tensor Core 优化的新一代块级编程范式。

- **快速入门**：[TileLang 快速入门](tilelang/TileLang_快速入门.md)

### 3. [性能分析 (Profiling)](profiling/README.md)

GPU 程序性能分析与调优工具。

- **Nsight Compute**：内核级性能分析
