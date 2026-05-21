# PyTorch NPU 实战

本模块聚焦于用 PyTorch 在昇腾 NPU 上完成模型训练：从 CUDA 代码的一键迁移开始，到 ResNet-50 的 FP32/AMP 训练对比，最后覆盖 NPU 特有的编译延迟、TBE 依赖和环境配置陷阱。

## 1. [CUDA 到 NPU 的代码迁移](01_cuda_to_npu_migration.md)

只需三步——`import torch_npu` 注册后端、替换设备字符串（`npu()`）、替换同步与 AMP API——就能把 CUDA 代码搬到 NPU 上。附完整的 10 组常用 API 对照表和可直接运行的迁移代码示例。

## 2. [ResNet-50 训练与 AMP 实战](02_resnet50_amp_training.md)

FP32 545 img/s、AMP 1254 img/s (2.3× 加速，显存减半)。涵盖 Gradient Scaling 行为解析、CANN TBE 依赖报错排查、GE 图编译器的首次编译延迟分析和 `msprof` profiling 命令。
