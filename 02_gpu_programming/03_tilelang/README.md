# Tile-Based 编程

当 Tensor Core 这样的专用加速单元成为现代 GPU 上的主力计算单元后，传统 SIMT 从单个线程视角组织计算的思路就显得有些别扭了——手写一份逼近硬件极限的 GEMM 或 FlashAttention，往往要去拼 MMA 指令、Swizzle 排布和异步拷贝的大量底层细节。

Tile-Based 的思路是把视角从“线程”上移到“数据块（Tile）”，让开发者直接用更贴近张量计算的抽象来描述算子，底层细节交给编译器处理。

## 1. [TileLang 快速入门](01_tilelang_quick_start.md)

了解 TileLang 的基本概念与使用方法。
