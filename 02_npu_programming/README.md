# NPU 编程

## 1. 概述

这一章回答一个实际的问题：**从一张空闲的昇腾 NPU 开始，到用 PyTorch 或 MindSpore 跑通模型训练，中间需要了解哪些环节？**

与 GPU 编程以 CUDA 为核心不同，NPU 编程的入口有两条路径：

- **PyTorch NPU 适配**：通过 `torch_npu` 将现有 PyTorch 代码迁移到 Ascend 硬件，API 层面几乎只需将 `cuda` 替换为 `npu`。适合有 GPU/CUDA 经验的开发者快速上手。
- **MindSpore 原生**：华为自研框架，对 Ascend 有原生支持，提供 PyNative（动态图）和 Graph（静态图）两种执行模式，编程范式与 PyTorch 有明显差异。

我们大致走五步：

- **环境**：先把 CANN 工具链和虚拟环境搭起来，理解 `set_env.sh` 和 venv 的加载顺序为什么不能颠倒。
- **架构**：看达芬奇架构的 Cube/Vector/Scalar 三单元分工和 CANN 软件栈的七层结构，建立与 CUDA 生态的对照。
- **框架实战**：用 PyTorch NPU 和 MindSpore 分别跑 ResNet-50 训练，对比 FP32/AMP 的吞吐和显存差异。
- **工具链**：掌握 `npu-smi`（类似 `nvidia-smi`）和 `ascend-dmi`（类似 `deviceQuery` + `bandwidthTest`）的日常用法，以及 ATC 模型转换流程。
- **进阶**：Ascend C 自定义算子、GPU→NPU 迁移决策树。

## 2. 环境准备

在开始写 NPU 代码之前，需要先把 CANN（Compute Architecture for Neural Networks）开发环境配置好。CANN 是昇腾的异构计算架构，对应 NVIDIA CUDA 平台的角色。关键点是 `set_env.sh` 必须在 venv 激活之前 source——顺序颠倒会导致 `libhccl.so` 等动态库解析失败。

- [昇腾环境搭建](01_environment/01_ascend_environment_setup.md) — CANN 环境变量、虚拟环境创建、torch_npu 安装、关键依赖版本对齐、环境变量加载顺序
- [Hello NPU：第一个程序](01_environment/02_hello_npu_first_program.md) — `import torch_npu` 注册后端、CUDA→NPU API 映射速查、4096×4096 矩阵乘法实测 164.6× 加速比

## 3. 硬件架构与软件栈

昇腾 910B3 基于达芬奇 (Da Vinci) 架构，与 NVIDIA 的 SIMT 模型在计算组织方式上有根本差异。理解这种差异是从 CUDA 平滑过渡到 NPU 的关键。

- [达芬奇架构与 Ascend 910B3](02_ascend_architecture/01_davinci_architecture.md) — AI Core 的 Cube/Vector/Scalar 三单元分工、HBM 内存层次、HCCS 8 卡全互联拓扑、CUDA SIMT 对照
- [CANN 软件栈详解](02_ascend_architecture/02_cann_software_stack.md) — 驱动→Runtime→算子→图编译→AscendCL→HCCL 七层结构、CANN vs CUDA 工具对照表

## 4. 核心编程范式

### 4.1 PyTorch NPU

PyTorch NPU 是大多数有 GPU 经验的开发者的首选入口。通过 `torch_npu` 适配层，CUDA 代码的迁移成本极低——绝大多数场景只需替换设备字符串。但也需要了解 CANN 图编译器（GE）的首次编译延迟、TBE 算子依赖等 NPU 特有的行为。

- [CUDA 到 NPU 的代码迁移](03_pytorch_npu/01_cuda_to_npu_migration.md) — 三步迁移法、10 组常用 API 对照表、`cuda()`→`npu()` 替换规则
- [ResNet-50 训练与 AMP 实战](03_pytorch_npu/02_resnet50_amp_training.md) — FP32 545 img/s vs AMP 1254 img/s (2.3×)、Gradient Scaling 行为、TBE 依赖排查、编译延迟分析

### 4.2 MindSpore

MindSpore 是华为自研框架，采用函数式梯度 API（`ms.value_and_grad`），支持 PyNative 动态图和 Graph 静态图两种执行模式。API 风格与 PyTorch 差异明显，但对 Ascend 硬件的集成度更高。

- [MindSpore 与 PyTorch API 对照](04_mindspore/01_mindspore_vs_pytorch_api.md) — 15 组常用 API 对照表、模型定义（`nn.Cell` vs `nn.Module`）、训练循环（函数式梯度 vs `loss.backward()`）、动静态图模式对比
- [MindSpore Ascend 训练实战](04_mindspore/02_mindspore_ascend_training.md) — PyNative 165 img/s vs Graph 159 img/s、版本兼容性（MindSpore 2.6 vs 2.9 与 CANN 8.0.1 的匹配关系）、MindSpore Lite 定位

## 5. 工具链

昇腾生态的工具链与 CUDA 工具有清晰的对应关系。`npu-smi` 对标 `nvidia-smi`（轻量巡检），`ascend-dmi` 对标 `deviceQuery` + `bandwidthTest`（硬件诊断与性能基准），`atc` 对标 TensorRT（模型编译）。

- [npu-smi 使用参考](05_tools/01_npu_smi_reference.md) — 60+ 种查询类型、默认输出字段解读、拓扑/HCCS/ECC/功耗/温度等常用查询、与 ascend-dmi 分工对照
- [ascend-dmi 使用参考](05_tools/02_ascend_dmi_reference.md) — 设备详情、带宽测试（HBM 1538 GB/s / HCCS 26.2 GB/s / PCIe 24.8 GB/s）、算力测试（FP16 313.7 TFLOPS）、故障诊断 12 项
- [ATC 模型转换](05_tools/03_atc_model_conversion.md) — PyTorch → ONNX → OM 完整流程、关键参数说明、AscendCL 推理加载

## 6. 进阶主题

- [Ascend C 算子开发入门](06_advanced/01_ascend_c_intro.md) — 自定义算子的场景、`msopgen` + Ascend C Compiler 开发流程、简单 ReLU 示例
- [GPU 到 NPU 的迁移策略](06_advanced/02_gpu_to_npu_migration.md) — 迁移决策树、成本估算、不建议迁移的场景

## 7. 参考链接

- [昇腾社区官网](https://www.hiascend.com)
- [Ascend PyTorch 适配 (Gitee)](https://gitee.com/ascend/pytorch)
- [MindSpore 安装指南](https://www.mindspore.cn/install)
- [CANN 商业版文档](https://www.hiascend.com/document)
