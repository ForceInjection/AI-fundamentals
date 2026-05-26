# 14. 量化推理原理 (INT8 / INT4)

学习量化推理的数学原理和工程实践。

## 文件

| 文件                        | 说明                                                      |
| --------------------------- | --------------------------------------------------------- |
| `01_quantization_theory.md` | 量化推理理论（对称/非对称、per-tensor/per-channel、校准） |
| `quantization_demo.py`      | 纯 Python 量化演示（量化和反量化、误差分析）              |
| `quantization_viz.html`     | 量化精度对比面板（INT8/4/3/2 表格 + 误差分布图）          |
| `quantization_guide.html`   | 交互式学习指南（完整手算示例，逐步展示量化过程）          |

## 关键概念

- 对称量化 vs 非对称量化
- Per-tensor vs Per-channel 量化
- 校准数据的作用
- FP16/BF16 vs INT8/INT4 的精度-效率权衡

## 当前 NPU 栈限制

CANN 8.0.1 + torch_npu 2.1.0 不支持 HuggingFace 模型的 INT8/INT4 推理（bitsandbytes/GPTQ/AWQ 均为 CUDA 专用）。本章聚焦理论理解和 CPU 演示。
