# GPU 架构深入理解

本文档集合旨在深入探讨 GPU 的硬件架构与工作原理，内容主要参考自 [Cornell Virtual Workshop: Understanding GPU Architecture](https://cvw.cac.cornell.edu/gpu-architecture/gpu-memory/index)。通过系统性地介绍 GPU 内存结构、核心特性及实例分析，帮助读者在高性能计算和人工智能开发中更好地优化程序性能。

## 1. 架构基础与特性

本章节主要介绍 GPU 的核心硬件特性以及与传统 CPU 架构的对比，重点分析其内存层次结构的设计逻辑与访问延迟。

- [01_gpu_characteristics.md](01_gpu_characteristics.md)：详细对比了 GPU 与 CPU 的设计理念差异，解析了并行计算架构的优势与局限性。
- [02_gpu_memory.md](02_gpu_memory.md)：深度剖析 GPU 的内存模型，包括全局内存、共享内存、寄存器等层次的访问延迟与带宽特性。

## 2. 经典硬件实例分析

本章节通过剖析具体的 GPU 型号，展示不同架构在实际硬件中的落地与演进过程，以加深对硬件原理的理解。

- [03_tesla_v100.md](03_tesla_v100.md)：以数据中心级 GPU 为例，解析 Volta 架构的核心创新，如 Tensor Core 加速单元和 HBM2 内存系统。
- [04_rtx_5000.md](04_rtx_5000.md)：以工作站级 GPU 为例，探讨 Turing 架构在图形与计算融合方面的设计，包括 RT Core 及其光线追踪能力。

## 3. 性能测试与实践练习

本章节提供了一些基于真实环境的练习脚本，帮助开发者动手验证 GPU 的设备信息与性能指标。

- [05_exer_device_query.md](05_exer_device_query.md)：介绍如何使用 CUDA API 获取当前系统的 GPU 设备属性与硬件规格。
- [06_exer_device_bandwidth.md](06_exer_device_bandwidth.md)：通过实际的基准测试，评估不同内存访问模式对实际带宽性能的影响。
