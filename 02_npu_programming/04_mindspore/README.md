# MindSpore on Ascend

MindSpore 是华为自研深度学习框架，对 Ascend NPU 有原生支持。与 PyTorch NPU 的 "适配层" 思路不同，MindSpore 从设计之初就面向 Ascend 硬件。本模块覆盖 API 对照和训练实战两个维度。

## 1. [MindSpore 与 PyTorch API 对照](01_mindspore_vs_pytorch_api.md)

面向有 PyTorch 基础的开发者，以 15 组最常用 API 为线索，逐项对比两套框架在模型定义（`nn.Cell` vs `nn.Module`）、训练循环（函数式梯度 vs `loss.backward()`）、动态图/静态图等维度的差异。

## 2. [MindSpore Ascend 训练实战](02_mindspore_ascend_training.md)

手写简版 ResNet-50 在 Ascend 后端上对比 PyNative（动态图）和 Graph（静态图）两种模式的训练吞吐，同时梳理 MindSpore 与 CANN 的版本兼容性矩阵。
