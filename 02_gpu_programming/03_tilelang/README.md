# TileLang 与 Tile-Based 编程

当 Tensor Core 这样的专用加速单元成为现代 GPU 上的主力计算单元后，传统 SIMT 从单个线程视角组织计算的思路就显得有些别扭了——手写一份逼近硬件极限的 GEMM 或 FlashAttention，往往要去拼 MMA 指令、Swizzle 排布和异步拷贝的大量底层细节。

Tile-Based 的思路是把视角从“线程”上移到“数据块（Tile）”，让开发者直接用更贴近张量计算的抽象来描述算子，底层细节交给编译器处理。

TileLang 是一种专为高性能 GPU 内核设计的领域特定语言（DSL）及编译器架构。它通过将数据块作为一等公民，帮助开发者在保持 Pythonic 语法的易用性的同时，自动生成深度优化的底层 CUDA 代码，极大地降低了编写极致性能算子的门槛。

## 1. [TileLang 快速入门](01_tilelang_quick_start.md)

从环境搭建到第一个矩阵乘法 Kernel：安装（PyPI / 源码 / Nightly）、核心 API 速览（`T.Kernel`、`T.alloc_shared`、`T.alloc_fragment`、`T.gemm`、`T.Pipelined`）、CUDA→TileLang 对照表，以及 JIT 编译流程和调试技巧。最终用 ~30 行代码实现一个接近手写 CUDA 性能的 GEMM 算子。

> **前置条件**：Python 3.8+、CUDA 11.0+、PyTorch。已验证硬件包括 H100、A100、V100、RTX 4090/3090/A6000 及 AMD MI250/MI300X。
