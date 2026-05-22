# NPU 性能分析

本模块聚焦于 NPU 程序的性能分析——从"能跑"到"跑得快"。回答三个核心问题：**时间花在哪、硬件利用率如何、如何优化**。

工具按粒度从粗到细：`npu-smi`（卡级实时监控，对标 `nvidia-smi`）→ `torch_npu.profiler`（算子级 Chrome trace，对标 CUDA Profiler）→ `ascend-dmi --bw`（带宽基准，对标 `bandwidthTest`）。实测 16384² 矩阵乘法达 **71.84 TFLOPS**（FP32 理论峰值 ~80 TFLOPS），ResNet-50 吞吐 **180 img/s** (batch=8)。

## 1. [NPU 性能分析入门](01_npu_profiling.md)

从"计算 bound vs 访存 bound"的基本概念出发，覆盖 synchronize 异步陷阱、warmup 编译延迟、Chrome trace 读图方法、npu-smi 采样窗口限制等新手最常见的踩坑点。包含矩阵乘法/2D 卷积/ResNet-50 三类算子的 profiling 数据和决策清单。
