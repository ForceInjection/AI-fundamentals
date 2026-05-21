# MindSpore Ascend 训练实战

## 1. 环境信息

| 项目      | 版本                                                                                    |
| --------- | --------------------------------------------------------------------------------------- |
| MindSpore | 2.6.0                                                                                   |
| CANN      | 8.0.1                                                                                   |
| Python    | 3.10.12                                                                                 |
| 设备      | Ascend 910B3 (NPU 7，空闲确认：`npu-smi info`，详见 `05_tools/01_npu_smi_reference.md`) |

## 2. 实验结果

在 NPU 7 上使用 ResNet-50 (batch_size=64, 224×224) 训练 2 epochs：

| 框架        | 模式              | 吞吐量    | 相对性能 |
| ----------- | ----------------- | --------- | -------- |
| PyTorch NPU | Eager             | 545 img/s | 基准     |
| MindSpore   | PyNative (动态图) | 165 img/s | 0.30×    |
| MindSpore   | Graph (静态图)    | 159 img/s | 0.29×    |

> [!NOTE]
> MindSpore 吞吐量低于 PyTorch NPU 约 3.3×。原因分析：
>
> 1. MindSpore 2.6.0 对 CANN 8.0.1 的适配不是最优目标版本（MindSpore 2.6 目标 CANN 8.2+）。
> 2. 自行编写的 ResNet 实现没有使用 MindSpore ModelZoo 的优化版本。
> 3. MindSpore 对 910B3 的编译优化在 8.0.1 上不如 PyTorch NPU 成熟。
> 4. PyNative 与 Graph 模式性能相近说明编译优化并未显著提升此用例。

**学习意义大于性能意义**：此实验的价值在于掌握 MindSpore 的编程范式和 API 差异，生产环境中应使用匹配的 CANN/MindSpore 版本组合。

## 3. 版本兼容性说明

MindSpore 版本必须与 CANN 版本匹配。从实验观察到：

| MindSpore | 需要的 CANN 版本 | 本环境 CANN | 状态                       |
| --------- | ---------------- | ----------- | -------------------------- |
| 2.9.0     | 8.2 / 8.3 / 9.0  | 8.0.1       | 版本不匹配，countdown 警告 |
| 2.6.0     | 8.0+             | 8.0.1       | 无版本警告，功能正常       |

PyPI 上可用的 MindSpore Ascend 版本（aarch64）：2.6.0 ~ 2.9.0。更早版本需从 MindSpore 官网或华为 Ascend 仓库获取特定安装包。

完整兼容性矩阵参考：[MindSpore 安装指南](https://www.mindspore.cn/install)。

## 4. MindSpore Lite

服务器系统环境中已安装 `mindspore-lite 2.3.0rc3`（系统 pip）。这是一个轻量级推理引擎，用于在端侧、边缘设备上运行 MindSpore 模型，不用于训练。与完整 MindSpore 包的关系：

- **mindspore**（完整版）= 训练 + 推理 + Ascend/Python 后端。
- **mindspore-lite** = 仅推理 + C++/Java API + 端侧优化。

## 5. 参考链接

- [MindSpore 安装指南与兼容性矩阵](https://www.mindspore.cn/install)
- [昇腾社区 — MindSpore 适配](https://www.hiascend.com/ecosystem/mindspore)
