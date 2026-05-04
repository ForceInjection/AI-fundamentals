# GPU 编程

## 1. 概述

这一章想回答的是一个实际的问题：**从一张干净的 GPU 开始，到写出一个“跑得好”的 Kernel，中间要经历哪些环节？**

我们大致会走三步：

- **环境**：先把开发环境搭起来。依靠 NVIDIA Container Toolkit 和 CUDA 镜像，把驱动、工具链、框架依赖封装成可复现的容器环境。
- **范式**：先从 CUDA SIMT 的基础入手，然后接触 Streams 带来的异步并发，最后过渡到以 Tensor Core 为中心的 Tile-Based 编程范式。
- **工具**：在真正写出代码后，用 nvbandwidth、Nsight Compute / Systems 等工具链来回答“快不快、为什么不快”。

走完这三步，你将具备从写 Kernel 到定位带宽/算力瓶颈的完整能力。

---

## 2. 环境准备

“工欲善其事，必先利其器”。在开始写 CUDA 代码之前，需要先把开发环境整顺。在生产环境中，大家通常不会直接裸装驱动和 CUDA，而是用 **NVIDIA Container Toolkit 加 CUDA 基础镜像** 把开发环境标准化。

这里有几个经常踩坑的关键点：

- **版本对齐**：驱动和用户态库（像 `libcuda.so`）版本必须匹配，否则会出现 ABI 不一致问题。
- **镜像变体选型**：`base` / `runtime` / `devel` 各有适用场景，选错了要么镜像臃肿、要么缺工具链。
- **构建优化**：多阶段构建能显著瘦身镜像，同时需要明确指定 CUDA 架构符号（如 `sm_80` / `sm_90` / `sm_100`）以匹配目标 GPU。

这也是 vLLM、TGI、Llama.cpp、DeepSpeed 这类框架能在生产集群上“可复现”的前提——一旦环境一致，高层框架的部署才会真正变得可预测。

- [NVIDIA GPU 容器环境：原理与构建指南](01_environment/01_nvidia_container_setup.md) - 深入解析容器化架构原理，提供从驱动安装到实战配置的完整手册。
- [大模型训练与推理框架的 GPU 镜像构建深度解析](01_environment/02_cuda_image_build_analysis.md) - 剖析 vLLM、TGI、Llama.cpp、DeepSpeed 四大框架的 Dockerfile，详解 CUDA 镜像变体选择与构建优化策略。

---

## 3. 核心编程范式

GPU 编程发展到今天，主流上其实有两种思路，对应两种看待“计算负载”的视角：

- **SIMT 模型**是经典思路，从线程视角出发，以 Thread 和 Warp 为基本单位组织计算。
- **Tile-Based 模型**更贴近现代硬件，从数据块的视角出发，以 Block / Tile 为单位组织计算，直接面向 Tensor Core 这类现代张量加速单元。

两种范式并非互斥，而是从不同抽象层级描述同一块 GPU 上的计算。

### 3.1 [CUDA 编程基础](02_cuda/README.md)

不管你最终写的是 PyTorch 算子、vLLM 插件还是自定义核，CUDA 都是 NVIDIA GPU 上几乎所有深度学习框架与高性能算子的底层抽象，也是理解 GPU 行为的第一道关。

入门阶段主要要搞懂两件事：

- **执行层次**：Grid / Block / Warp / Thread 这四级结构有什么用、怎么落到硬件上；
- **异步并发**：如何用 Streams 让计算与 H2D/D2H 数据传输重叠，避免 GPU “空等数据”。

把这两点想通，后面再看任何 CUDA 代码基本都不会有理解障碍。

- **核心概念**：[CUDA 核心原理](02_cuda/02_cuda_cores.md) | [流处理机制](02_cuda/03_cuda_streams.md)
- **概念解析**：[GPU 编程导论](02_cuda/01_gpu_programming_introduction.md) | [SIMT 到 Tile-Based 编程范式的演进](02_cuda/04_simt_vs_tile_based.md)

### 3.2 [Tile-Based 编程与 TileLang](03_tilelang/README.md)

Tile-Based 的思路是把数据块（Tile）做为编程单元，把原本需要手写的底层能力封装成更高层的原语——比如 Tensor Core 的 MMA 指令、共享内存的 Swizzle 排布、以及异步拷贝（`cp.async`）。

实际的收益很直接：手写高性能 GEMM 或 FlashAttention 这类算子时，之前要难做对的一大堆底层细节，现在能被编译器帮你处理掉一大半，工程门槛明显降低。

- [TileLang 快速入门](03_tilelang/01_tilelang_quick_start.md)

---

## 4. [性能分析与优化](04_profiling/README.md)

GPU 性能分析不宜只看一个指标，实践上一般从三个视角交叉印证：

- **Nsight Compute 看单个 Kernel**：SM occupancy、L1/L2 命中率、memory throughput 等指标呈现中微观的执行质量。
- **Nsight Systems 看整个系统**：CPU 和 GPU 的时间线、NCCL 集合通信的调度和等待，看谁在等谁。
- **nvbandwidth 实测带宽**：验证 HBM 和 PCIe 实测带宽是否接近理论上限，以排除硬件或拓扑层面的瓶颈。

- [nvbandwidth 最佳实践](04_profiling/01_nvbandwidth_best_practices.md) - 深入了解和测量 GPU 的显存带宽与 PCIe 传输带宽。

---

## 5. 学习资源库

这里的资料大致按 **快速入门 → 进阶实战 → 权威参考** 三个梯度组织，适合不同阶段按需取用：

- **官方权威**：NVIDIA CUDA C++ Programming Guide，词典式查阅的第一去处。
- **中文教材**：樊哲勇《基础与实践》、Professional CUDA C，适合系统性读。
- **优化范例**：CUDA-Learn-Notes 收集了 200+ 个 Tensor Core / CUDA Core 的极致优化内核，适合“照着拆解”。
- **活跃社区**：GPU Mode、CUDA Reading Group 等，能看到最新的工程讨论。

### 5.1 快速入门

- [并行计算、费林分类法和 CUDA 基本概念](https://mp.weixin.qq.com/s/NL_Bz8JB-LdAtrQake7EdA)
- [CUDA 编程模型入门](https://mp.weixin.qq.com/s/IUYzzgt6DUYhfaDnbxoZuQ)
- [CUDA 并发编程之 Stream 介绍](./02_cuda/03_cuda_streams.md)

### 5.2 进阶实战

进阶阶段的重心通常放在 Tensor Core MMA、CuTe 布局、寄存器复用，以及 Warp-Specialized 流水这类极致优化技巧上，适合已经熟悉 CUDA 基础、想再抓一点性能的开发者。

- [CUDA-Learn-Notes](https://github.com/xlite-dev/CUDA-Learn-Notes) - 涵盖 200+ 个 Tensor Core/CUDA Core 极致优化内核示例 (HGEMM, FA2 via MMA and CuTe)。

### 5.3 参考资料大全

以 NVIDIA 官方 Programming Guide 为权威基准，再配上经典中文教材、开源范例仓库和学术/社区讲座，如下几类资源可以按需组合，形成自己的参考矩阵。

**书籍与文档**：

- [《CUDA C++ Programming Guide》](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) - NVIDIA 官方权威指南
- [《CUDA C 编程权威指南》](https://mp.weixin.qq.com/s/xJY5Znv3cuQi_UCd_XjJ4A) - 经典教材 (Professional CUDA C Programming)
- [《CUDA 编程：基础与实践 by 樊哲勇》](https://book.douban.com/subject/35252459/) - 中文经典实战教程
- [《CUDA 编程简介: 基础与实践 by 李瑜》](./02_cuda/references/CUDA%20%E7%BC%96%E7%A8%8B%E7%AE%80%E4%BB%8B%20-%20%E5%9F%BA%E7%A1%80%E4%B8%8E%E5%AE%9E%E8%B7%B5.pdf)
- [《CUDA 编程入门》](https://hpcwiki.io/gpu/cuda/) - 改编自北京大学超算队 CUDA 教程讲义

**代码仓库与示例**：

- [Nvidia 官方 CUDA 示例](https://github.com/NVIDIA/cuda-samples) - 官方标准范例库
- [书中示例代码 (Professional CUDA C)](https://github.com/Eddie-Wang1120/Professional-CUDA-C-Programming-Code-and-Notes)
- [学习笔记 (CudaSteps)](https://github.com/QINZHAOYU/CudaSteps)
- [示例代码 (CUDA_Programming)](https://github.com/MAhaitao999/CUDA_Programming)
- [Multi GPU Programming Models](https://github.com/NVIDIA/multi-gpu-programming-models) - 多卡编程模型示例

**社区与讲座**：

- [CUDA Reading Group 相关讲座](https://mp.weixin.qq.com/s/6sOrNzG0UeVBes8stWSoWA)
- [GPU Mode Reading Group](https://github.com/gpu-mode) - 活跃的 GPU 编程社区
- [樊哲勇主页](https://wlkxyjsxy.bhu.edu.cn/engine2/general/4146630/detail?engineInstanceId=656243&typeId=2986094&pageId=85748&websiteId=63087&currentBranch=1)
- [CUDA Processing Streams](https://turing.une.edu.au/~cosc330/lectures/display_lecture.php?lecture=22#1)
