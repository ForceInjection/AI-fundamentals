# 昇腾 NPU 环境搭建

本模块聚焦于华为昇腾 NPU 的开发环境搭建：从裸机开始，安装 CANN 工具链、配置虚拟环境、安装 PyTorch NPU 适配层，并运行第一个 NPU 程序验证环境可用性。

## 1. [昇腾环境搭建](01_ascend_environment_setup.md)

CANN 环境变量的加载顺序、虚拟环境创建、torch_npu 与 PyTorch 的版本对齐、关键依赖（TBE、decorator、numpy）的安装。回答了 "为什么 `source set_env.sh` 必须在 venv 激活之前" 这个最容易踩坑的问题。

## 2. [Hello NPU：第一个程序](02_hello_npu_first_program.md)

`import torch_npu` 注册后端 → CUDA↔NPU API 对照速查 → 4096×4096 矩阵乘法 CPU vs NPU 对比（实测 164.6× 加速比）。走通这一步，后续所有 PyTorch NPU 代码的调试都有了基准。
