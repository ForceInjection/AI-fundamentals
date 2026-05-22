# FlashAttention 简化版

本模块手写 FlashAttention 的 forward pass，聚焦两个核心算法思想：**online softmax**（用动态 rescaling 单遍计算 softmax）和 **tiling**（分块避免 O(N²) 显存）。用纯 Python/PyTorch 实现，目标是理解算法原理而非追求性能。

实测：HBM 峰值从 130 MB 降至 1 MB（节省 95%），数值精度 max_diff < 1e-6。

## 1. [FlashAttention 简化版](01_flash_attention.md)

从 O(N²) 显存瓶颈出发，逐步推导 online softmax 的数学原理（`correction = exp(m_old - m_new)` 为什么能替代两遍 softmax），再到 tiling 分块策略和实测数据。包含精度验证、显存对比和 Python 实现不加速的原因分析。
