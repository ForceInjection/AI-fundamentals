# 硬件架构与互连技术

## 1. 概述

本模块聚焦于 AI 算力基础设施的最底层，即硬件加速器与系统互连架构。通过从单机计算芯片（GPU、TPU）的设计哲学，到系统内的互连总线（PCIe、NVLink），再到跨节点的数据传输技术（GPUDirect）的系统性解析，帮助读者建立对 AI 计算底座从微观到宏观的完整认知。这对于后续的性能评估与架构调优至关重要。

---

## 2. 基础计算芯片架构

本章节探讨 AI 计算中最核心的加速器架构设计，这是理解并行计算与张量运算的基础。我们分别剖析了通用图形处理器（GPU）与专门为神经网络设计的处理器（TPU、NPU）的底层特性。

### 2.1 NVIDIA GPU 架构

NVIDIA GPU 是目前 AI 计算的主力。本小节提供了从架构特性到大模型算力选型的全面分析。

- **[深入理解 GPU 架构](nvidia/understand_gpu_architecture/README.md)**：包含 GPU 与 CPU 的特性对比、内存层次模型（全局内存、共享内存等），以及 Tesla V100、RTX 5000 等具体硬件实例的分析。
- **[GPGPU vs NPU：大模型推理训练对比](nvidia/GPGPU_vs_NPU_大模型推理训练对比.md)**：探讨在大语言模型时代，不同架构芯片在训练与推理场景下的优劣势与选型指南。

### 2.2 Google TPU 架构

TPU 代表了另一条专为深度学习优化的技术路线，通过脉动阵列（Systolic Array）实现极高的能效比。

- **[TPU 101：深度学习专用加速器架构解析](tpu/tpu%20101.md)**：探索 TPU 的设计哲学、核心计算单元原理及其与 GPU 的差异。

---

## 3. 高速互连与数据传输技术

随着模型规模的增长，单芯片的算力已无法满足需求，芯片间、节点间的数据传输成为系统的主要瓶颈（即“内存墙”与“IO 墙”）。本章节从基础总线到高级直通技术，系统解析现代互连架构。

### 3.1 基础系统总线与片间互连

系统总线与专用互连链路构成了单机多卡以及异构计算的通信基础。

- **[PCIe 总线技术大全](pcie/01_pcie_comprehensive_guide.md)**：从物理层到协议层全面解析 PCIe 总线架构及带宽演进。
- **[Linux PCIe P2PDMA 技术介绍](pcie/02_p2pdma_technology.md)**：详解设备直连 DMA 技术在 Linux 内核中的实现原理。
- **[NVLink 技术入门](nvlink/nvlink_intro.md)**：介绍 NVIDIA 为突破 PCIe 带宽瓶颈而设计的专有高速 GPU 互连方案。

### 3.2 高级直通技术（GPUDirect）

GPUDirect 是一系列旨在消除 CPU 与系统内存参与，实现设备间直接数据传输的高级技术。

- **[NVIDIA GPUDirect P2P 技术详解](gpudirect/02_gpudirect_p2p.md)**：探讨节点内多 GPU 之间如何通过 PCIe 或 NVLink 实现高速对等通信。
- **[NVIDIA GPUDirect RDMA 与 Storage 技术详解](gpudirect/01_gpudirect_technology.md)**：深入解析如何通过 RDMA 实现跨节点的网卡到 GPU 直接通信，以及通过 GDS 实现存储到 GPU 的直接数据加载。

---

## 4. 异构融合架构与系统性能评估

在掌握了基础芯片与互连技术后，本章节将视角提升至系统级与机架级，探讨下一代超级芯片架构以及如何对整体系统性能进行宏观评估。

### 4.1 AI Superchip 与机架级架构

随着 Blackwell 架构的推出，计算节点的边界正在被重新定义。

- **[NVLink-C2C 详解](superchips/nvlink_c2c.md)**：解析打破内存墙的关键——基于 `Chip-to-Chip` 的异构融合互连技术。
- **[NVIDIA GB300 NVL72 架构解析](superchips/nvidia_gb300.md)**：探讨基于下一代 Blackwell 架构的机架级（Rack-Scale）计算系统设计。

### 4.2 性能参考指标

在进行架构设计和性能调优时，建立对系统各层级延迟的数量级认知至关重要。

- **[AI 基础设施延迟金字塔](performance/ai_latency_pyramid.md)**：提供从寄存器访问、内存读写到跨节点网络通信的各级延迟参考基准数据。
