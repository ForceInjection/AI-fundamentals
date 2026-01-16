# 硬件与架构 (Hardware Architecture)

本章节深入探讨 AI 加速器的底层硬件架构，旨在帮助读者理解计算底座的设计原理与性能特征。

## 1. 硬件基础知识

在构建现代 AI 基础设施时，从 AI Infra 的视角来看，我们需要掌握以下核心硬件体系：

### 1.1 核心组件

1. **计算硬件 (Compute)**

   - **GPU 架构**：SIMT 线程模型、CUDA 核心原理、Tensor Core 加速单元。
   - **内存层次**：寄存器 (Registers) -> 共享内存 (Shared Memory) -> 全局内存 (Global Memory) -> HBM/GDDR。
   - **性能指标**：FLOPS (每秒浮点运算次数)、Memory Bandwidth (内存带宽)。

2. **互连技术 (Interconnect)**

   - **片间互连**：NVLink、CXL，解决单机多卡通信瓶颈。
   - **节点互连**：InfiniBand (IB)、RoCEv2，构建大规模分布式训练集群。
   - **系统总线**：PCIe 4.0/5.0，CPU 与 GPU 之间的高速通道。

3. **存储与网络 (Storage & Network)**
   - **高性能存储**：NVMe SSD、GPUDirect Storage (GDS)。
   - **网络架构**：Fat-Tree 拓扑、Rail-optimized 设计。

### 1.2 深度阅读

- [**PCIe 知识大全**](https://mp.weixin.qq.com/s/dHvKYcZoa4rcF90LLyo_0A) - 深入理解 PCIe 总线架构、带宽计算和性能优化
- [**NVLink 入门**](https://mp.weixin.qq.com/s/fP69UEgusOa_X4ZKLo30ig) - NVIDIA 高速互连技术的原理与应用场景
- [**NVIDIA DGX SuperPOD**](https://mp.weixin.qq.com/s/a64Qb6DuAAZnCTBy8g1p2Q) - 企业级 AI 超算集群的架构设计与部署实践

## 2. 芯片架构对比

针对不同场景（训练 vs 推理），选择合适的计算芯片至关重要。

- [**GPGPU vs NPU：大模型推理训练对比**](nvidia/GPGPU_vs_NPU_大模型推理训练对比.md)
  - 深入分析 GPGPU (通用图形处理器) 与 NPU (神经网络处理器) 的架构差异。
  - 探讨两者在 LLM 训练与推理场景下的性能表现与适用性。

## 3. 目录导航

### 3.1 [NVIDIA GPU 架构](nvidia/README.md)

深入解析 NVIDIA GPU 的硬件设计、内存层次结构及性能特性。

- **架构基础**：[GPU 特性](nvidia/gpu_characteristics.md)、[内存模型](nvidia/gpu_memory.md)
- **硬件实例**：[Tesla V100](nvidia/tesla_v100.md)、[RTX 5000](nvidia/rtx_5000.md) 架构分析
- **实践练习**：设备查询与带宽测试

### 3.2 [Google TPU 架构](tpu/tpu%20101.md)

探索 Google TPU (Tensor Processing Unit) 的设计哲学与脉动阵列架构。

- **TPU 101**：深度学习专用加速器架构解析
- **对比分析**：TPU vs GPU

### 3.3 [GPUDirect 技术](gpudirect/gpudirect_technology.md)

详细解析 NVIDIA GPUDirect 系列技术，重点关注解决“内存墙”与“IO 墙”问题的核心方案。

- **核心技术**：[GPUDirect RDMA](gpudirect/gpudirect_technology.md#2-gpudirect-rdma-技术) 与 [GPUDirect Storage (GDS)](gpudirect/gpudirect_technology.md#3-gpudirect-storage-gds-技术)
- **原理解析**：PCI BAR 映射、DMA 路径优化、Zero-Copy 机制
- **生态支持**：硬件（网卡/存储）、软件（通信库/文件系统）及 AI 框架集成
