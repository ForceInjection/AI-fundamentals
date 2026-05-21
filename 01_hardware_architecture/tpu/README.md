# Google TPU 架构

TPU 是 Google 为深度学习量身打造的专用加速器。与 GPU 的通用 SIMT 路线不同，TPU 通过**脉动阵列（Systolic Array）**将矩阵乘法做到极致，在特定负载下实现远高于 GPU 的能效比。

## 文档

- [**TPU 101：深度学习专用加速器架构解析**](tpu%20101.md)：TPU 的设计哲学、Systolic Array 核心计算原理、与 GPU 的架构差异及适用场景对比。

## TPU vs GPU 速览

| 维度     | GPU (NVIDIA H100)     | TPU v5p                       |
| -------- | --------------------- | ----------------------------- |
| 计算核心 | SIMT + Tensor Core    | 脉动阵列                      |
| 编程模型 | CUDA / Triton         | JAX / XLA                     |
| 内存     | HBM3 80 GB            | HBM3 95 GB                    |
| 互连     | NVLink 4.0 + NVSwitch | ICI (Inter-Chip Interconnect) |
| 适用场景 | 训练 + 推理（通用）   | 训练 + 推理（Google 生态）    |

## 参考

- [NVIDIA GPU 架构](../nvidia/README.md) — 对比 NV GPU 路线
- [GPGPU vs NPU：大模型推理训练的算力选择指南](../nvidia/GPGPU_vs_NPU_大模型推理训练对比.md) — 架构选型讨论
