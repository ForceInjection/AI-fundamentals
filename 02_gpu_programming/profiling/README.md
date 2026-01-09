# AI 系统性能分析

本目录包含 AI 系统性能分析相关的工具、文档和最佳实践。性能分析是 AI 系统优化的关键环节，通过深入分析系统瓶颈，我们可以显著提升模型训练和推理的效率。

---

## 1. CUDA 性能分析工具

- **NVIDIA Nsight Compute**: CUDA 内核级性能分析器
- **NVIDIA Nsight Systems**: 系统级性能分析器
- **nvprof**: 传统 CUDA 性能分析工具

官方文档：

- [**CUDA 内核性能分析指南**](./s9345-cuda-kernel-profiling-using-nvidia-nsight-compute.pdf)：NVIDIA 官方 CUDA 内核性能分析详细指南

## 2. 性能分析实践

**CUDA 矩阵乘法性能优化案例**：

通过 Nsight 工具对 CUDA 矩阵乘法的不同实现进行定量分析，包括：

- 全局内存访问模式优化
- 共享内存（Shared Memory）优化
- 指令级并行（ILP）优化

详细分析请参考：[使用 Nsight 工具定量分析 CUDA 矩阵乘法几种实现](https://mp.weixin.qq.com/s/JK_bsvG-Y3wLJknZ4YKCYQ)

## 3. 参考资源

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [GPU 利用率是一个误导性指标](../../03_ai_cluster_ops/gpu_ops/GPU%20%E5%88%A9%E7%94%A8%E7%8E%87%E6%98%AF%E4%B8%80%E4%B8%AA%E8%AF%AF%E5%AF%BC%E6%80%A7%E6%8C%87%E6%A0%87.md)
- [CUDA 编程模型入门](../cuda/README.md)
