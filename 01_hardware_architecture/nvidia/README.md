# GPU 架构深度解析

## 1. 概述

本目录提供 GPU 架构的深度技术解析，帮助开发者和研究人员深入理解 GPU 硬件设计原理、内存层次结构和性能特征。内容涵盖从基础概念到具体硬件实例的全面分析。

## 2. 核心文档

### 2.1 架构基础

- **[GPU Characteristics（GPU 特性）](gpu_characteristics.md)** - GPU 硬件特性与 CPU 对比分析
  - 并行计算架构设计原理
  - GPU vs CPU 性能特征对比
  - 适用场景与局限性分析

- **[GPU Memory (GPU 内存)](gpu_memory.md)** - GPU 内存层次结构详解
  - 全局内存、共享内存、寄存器层次
  - 内存带宽与延迟特性
  - 内存访问模式优化策略

### 2.2 硬件实例分析

- **[GPU Example: Tesla V100](tesla_v100.md)** - 数据中心级 GPU 架构深度分析
  - Volta 架构技术特性
  - Tensor Core 加速单元
  - HBM2 内存系统设计

- **[GPU Example: RTX 5000](rtx_5000.md)** - 工作站级 GPU 架构特性
  - Turing 架构创新点
  - RT Core 光线追踪加速
  - GDDR6 内存系统

### 2.3 实践练习

- **[Exercise: Device Query](exer_device_query.md)** - GPU 设备信息获取与分析
  - CUDA 设备属性查询
  - 硬件规格解读
  - 性能参数分析

- **[Exercise: Device Bandwidth](exer_device_bandwidth.md)** - GPU 内存带宽性能测试
  - 内存带宽基准测试
  - 不同访问模式性能对比
  - 优化策略验证

## 3. 技术特色

- **理论与实践结合**：从架构原理到实际硬件分析
- **对比分析**：GPU 与 CPU 架构深度对比
- **实例驱动**：基于具体 GPU 型号的详细分析
- **动手实践**：提供可执行的测试和分析练习

## 4. 相关资源

- [CUDA 编程基础](../../02_gpu_programming/) - CUDA 核心概念与编程
- [GPU 编程实践](../../02_gpu_programming/) - GPU 编程入门指南
- [性能分析工具](../../02_gpu_programming/profiling/) - GPU 性能分析与优化
- [AI 系统架构](https://github.com/Infrasys-AI/AISystem) - AI 系统整体架构设计
