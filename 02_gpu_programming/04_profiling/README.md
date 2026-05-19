# AI 系统性能分析

写出一个可以运行的 CUDA Kernel 并不难，难的是回答一个问题：这个 Kernel、这条训练或推理流程，真的跑出了硬件该有的性能吗？如果没有，问题在哪里？

这个目录里的材料基本都是为了回答这类问题，涉及三类工具：

- **Nsight Compute** 抓单个 Kernel 的执行质量，看 SM occupancy、内存吞吐这些细节。
- **Nsight Systems** 拉远视角，展示 CPU 与 GPU 交互、NCCL 通信、内核启动间的间隙，寻找时间线上的气泡。
- **nvbandwidth** 负责把 HBM 和 PCIe 的实测带宽和理论上限做对比，验证硬件和拓扑是否存在短板。

把这三类工具配合起来，就可以较全面地定位训练/推理中的计算、内存和通信瓶颈。

---

## 1. CUDA 性能分析工具

- **NVIDIA Nsight Compute**: CUDA 内核级性能分析器
- **NVIDIA Nsight Systems**: 系统级性能分析器
- **nvprof**: 传统 CUDA 性能分析工具
- **nvbandwidth**: NVIDIA GPU 带宽测量工具

相关文档：

- [**CUDA 内核性能分析指南**](references/s9345-cuda-kernel-profiling-using-nvidia-nsight-compute.pdf)：NVIDIA 官方 CUDA 内核性能分析详细指南
- [**nvbandwidth 深度解析**](01_nvbandwidth_best_practices.md)：NVIDIA GPU 带宽测量工具全指南
- [**PCIe 链路状态与带宽实测**](02_pcie_bandwidth_measurement.md)：零依赖 CUDA H2D/D2H 带宽测试程序，基于 RTX 5090 PCIe Gen 5 实测
- [**GPU 显存带宽测试：片内 vs 片外**](03_hbm_bandwidth_test.md)：HBM D2D 带宽实测 (762-1341 GB/s)，与 PCIe 带宽形成完整对比
- [**GPU 功耗管理实战**](04_power_management.md)：RTX 5090 功耗限制、时钟控制、Power Smoothing/Power Profiles 兼容性实测
- [**PCIe 传输效率曲线**](05_pcie_transfer_efficiency.md)：1 KB - 1 GB 区间 PCIe 带宽爬升曲线，揭示效率拐点与最小传输大小建议

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
- [GPU 利用率是一个误导性指标](../../03_ai_cluster_ops/01_gpu_ops/02_gpu_utilization_myth.md)：解释为什么高 GPU 利用率并不总是意味着高效计算。
- [CUDA 编程模型入门](../02_cuda/README.md)
