# 昇腾硬件架构与 CANN 软件栈

本模块建立对昇腾 NPU 硬件和软件栈的结构化认知：从达芬奇架构的计算单元设计开始，到 CANN 七层软件栈的全貌，最后建立与 CUDA 生态的对照。

## 1. [达芬奇架构与 Ascend 910B3](01_davinci_architecture.md)

了解了达芬奇架构的 Cube/Vector/Scalar 三单元分工，才能真正理解为什么标准 PyTorch 算子可以直接迁移（它们落在 Cube Unit 的优化范围内），而自定义 CUDA kernel 需要改写为 Ascend C。

## 2. [CANN 软件栈详解](02_cann_software_stack.md)

CANN 是昇腾硬件上层的完整异构计算栈，对应 CUDA 平台。从底层的驱动和固件，到 Runtime 的任务调度，再到算子层（OPP / Ascend C / TBE）和图编译层（GE / AOE），最后到 AscendCL 应用接口和 HCCL 集合通信——每一层在 CUDA 生态中都能找到对应物，但设计思路有明显差异。
