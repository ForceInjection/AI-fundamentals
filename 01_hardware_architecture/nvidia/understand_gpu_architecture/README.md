# GPU 架构深入理解

这个目录是一套沿着 Cornell 的经典教程 [Understanding GPU Architecture](https://cvw.cac.cornell.edu/gpu-architecture/gpu-memory/index) 整理和扩展而来的笔记，重点回答一个问题：**GPU 内部到底长什么样，它和我们写的代码的性能之间究竟是怎么关联的？**

对做高性能计算或 AI 应用的开发者来说，看懂底层才能真正优化得动代码，而不是只能凭感觉调参。

## 1. 架构基础与特性

这一部分先从“为什么 GPU 和 CPU 不一样”出发，看 GPU 的核心特性和内存层次是怎么被设计出来的，重点理解各类存储的访问延迟和带宽特性。

- [01_gpu_characteristics.md](01_gpu_characteristics.md)：对比 GPU 与 CPU 的设计理念差异，解释并行计算架构的优势和局限性分别在哪里。
- [02_gpu_memory.md](02_gpu_memory.md)：把 GPU 内存模型里的全局内存、共享内存、寄存器等层级摘开，逐层说清楚它们的访问延迟和带宽特性。

## 2. 经典硬件实例分析

理论讲完之后，更有效的方式是拿真实的 GPU 型号“拆开看看”：相同的原理在不同代际、不同产品线上是如何落地的。

- [03_tesla_v100.md](03_tesla_v100.md)：以数据中心级 GPU 为例，看 Volta 架构的核心创新——Tensor Core 加速单元和 HBM2 内存系统是怎么回事。
- [04_rtx_5000.md](04_rtx_5000.md)：以工作站级 GPU 为例，看 Turing 架构在图形和计算融合上的设计思路，包含 RT Core 和光线追踪能力。

## 3. 性能测试与实践练习

光读文档容易流于表面，这里配上几个可以在真实环境里跑的脚本，目的是让你亲手验证“理论值”和“实测值”之间的差距。

- [05_exer_device_query.md](05_exer_device_query.md)：如何用 CUDA API 取出当前系统上 GPU 设备的属性和硬件规格。
- [06_exer_device_bandwidth.md](06_exer_device_bandwidth.md)：通过实际的基准测试，估算不同内存访问模式对实际带宽性能的影响。
