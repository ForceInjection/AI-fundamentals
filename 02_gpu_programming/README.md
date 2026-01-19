# GPU 编程

本章节构建了从基础 CUDA 编程到前沿 Tile-Based 优化的完整知识体系，旨在帮助开发者掌握高性能计算的核心编程范式。

## 1. 环境准备

工欲善其事，必先利其器。在开始 GPU 编程之前，首先需要构建一个标准化的容器开发环境。

- [**NVIDIA GPU 容器环境安装指南**](environment/nvidia_container_setup.md) - 从驱动安装到容器启动的完整实战手册。

---

## 2. 核心编程范式

### 2.1 [CUDA 编程基础](cuda/README.md)

NVIDIA 官方标准的并行计算架构与编程模型，是 GPU 编程的基石。

- **核心概念**：[CUDA 核心原理](cuda/cuda_cores_cn.md) | [流处理机制](cuda/cuda_streams.md)
- **入门指南**：[CUDA 编程简介 - 基础与实践](cuda/CUDA%20编程简介%20-%20基础与实践.pdf)
- **深度解析**：[GPU 编程导论](cuda/gpu_programming_introduction.md) | [SIMT vs Tile-Based 范式对比](cuda/simt_vs_tile_based_programming.md)

### 2.2 [Tile-Based 编程](tilelang/README.md)

面向 Tensor Core 优化的新一代块级编程范式，专为大模型时代的高性能算子开发设计。

- **快速入门**：[TileLang 快速入门](tilelang/TileLang_快速入门.md)

---

## 3. 开发工具链

### 3.1 [性能分析与调优](profiling/README.md)

- **Nsight Compute**：内核级性能分析工具，深入指令流水线挖掘性能瓶颈，参考文档：[Nsight Compute 核心分析指南 (PDF)](profiling/s9345-cuda-kernel-profiling-using-nvidia-nsight-compute.pdf)。

---

## 4. 学习资源库

### 4.1 快速入门

- [**并行计算、费林分类法和 CUDA 基本概念**](https://mp.weixin.qq.com/s/NL_Bz8JB-LdAtrQake7EdA)
- [**CUDA 编程模型入门**](https://mp.weixin.qq.com/s/IUYzzgt6DUYhfaDnbxoZuQ)
- [**CUDA 并发编程之 Stream 介绍**](cuda/cuda_streams.md)

### 4.2 进阶实战

面向专业开发者的深度优化指南。

- [**CUDA-Learn-Notes**](https://github.com/xlite-dev/CUDA-Learn-Notes) - 涵盖 200+ 个 Tensor Core/CUDA Core 极致优化内核示例 (HGEMM, FA2 via MMA and CuTe)。

### 4.3 参考资料大全

我们整理了从官方文档到社区精选的完整学习路径。

**书籍与文档**：

- [**《CUDA C++ Programming Guide 》**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) - NVIDIA 官方权威指南
- [**《CUDA C 编程权威指南》**](https://mp.weixin.qq.com/s/xJY5Znv3cuQi_UCd_XjJ4A) - 经典教材 (Professional CUDA C Programming)
- [**《CUDA 编程：基础与实践 by 樊哲勇》**](https://book.douban.com/subject/35252459/) - 中文经典实战教程
- [**《CUDA 编程简介: 基础与实践 by 李瑜》**](cuda/CUDA%20%E7%BC%96%E7%A8%8B%E7%AE%80%E4%BB%8B%20-%20%E5%9F%BA%E7%A1%80%E4%B8%8E%E5%AE%9E%E8%B7%B5.pdf)
- [**《CUDA 编程入门》**](https://hpcwiki.io/gpu/cuda/) - 改编自北京大学超算队 CUDA 教程讲义

**代码仓库与示例**：

- [**Nvidia 官方 CUDA 示例**](https://github.com/NVIDIA/cuda-samples) - 官方标准范例库
- [**书中示例代码 (Professional CUDA C)**](https://github.com/Eddie-Wang1120/Professional-CUDA-C-Programming-Code-and-Notes)
- [**学习笔记 (CudaSteps)**](https://github.com/QINZHAOYU/CudaSteps)
- [**示例代码 (CUDA_Programming)**](https://github.com/MAhaitao999/CUDA_Programming)
- [**Multi GPU Programming Models**](https://github.com/NVIDIA/multi-gpu-programming-models) - 多卡编程模型示例

**社区与讲座**：

- [**CUDA Reading Group 相关讲座**](https://mp.weixin.qq.com/s/6sOrNzG0UeVBes8stWSoWA)
- [**GPU Mode Reading Group**](https://github.com/gpu-mode) - 活跃的 GPU 编程社区
- [**樊哲勇主页**](https://wlkxyjsxy.bhu.edu.cn/engine2/general/4146630/detail?engineInstanceId=656243&typeId=2986094&pageId=85748&websiteId=63087&currentBranch=1)
- [**CUDA Processing Streams**](https://turing.une.edu.au/~cosc330/lectures/display_lecture.php?lecture=22#1)
