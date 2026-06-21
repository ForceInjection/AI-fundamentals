# Profiling 示例代码

本目录包含 04_profiling/ 下各文章的配套可运行代码。

## 前置要求

- **CUDA Toolkit** ≥ 11.0（推荐 12.x）
- **GPU**：CC 8.0+（A100/H100）。部分 demo 可在更低 CC 上运行
- **系统**：Linux

## 文件清单

| 文件 | 配套文章 | 用途 | 编译 | 运行 |
|------|---------|------|------|------|
| `09_gpu_transfer_methods.cu` | 09_gpu_transfer_methods.md | 5 种 GPU→GPU 传输方法带宽对比（P2P/CPU relay/Zero-Copy/UM） | `nvcc -arch=sm_80 -O3 -o gpu_transfer_methods 09_gpu_transfer_methods.cu` | `CUDA_VISIBLE_DEVICES=0,1 ./gpu_transfer_methods` |

## 特殊说明

- **09_gpu_transfer_methods**：需要 ≥2 GPUs。P2P 方法（Method 1/2）仅在 Peer Access 可用时运行。`CUDA_VISIBLE_DEVICES` 用于选择特定的 GPU 对。
