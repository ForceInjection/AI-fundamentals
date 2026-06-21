# Profiling 示例代码

本目录包含 04_profiling/ 下各文章的配套可运行代码。

## 前置要求

- **CUDA Toolkit** ≥ 11.0（推荐 12.x）
- **GPU**：CC 8.0+（A100/H100）。部分 demo 可在更低 CC 上运行
- **系统**：Linux

## 文件清单

| 文件 | 配套文章 | 用途 | 编译 | 运行 |
|------|---------|------|------|------|
| `02_pcie_bandwidth_bench.cu` | 02_pcie_bandwidth_measurement.md | PCIe H2D/D2H 带宽测量 (1 MB → 1 GB) | `nvcc -arch=sm_80 -O3 -o pcie_bw_bench 02_pcie_bandwidth_bench.cu` | `./pcie_bw_bench` |
| `03_hbm_bandwidth_bench.cu` | 03_hbm_bandwidth_test.md | HBM D2D 内部 Copy 带宽 vs 理论峰值 | `nvcc -arch=sm_80 -O3 -o hbm_bw_bench 03_hbm_bandwidth_bench.cu` | `./hbm_bw_bench` |
| `05_pcie_transfer_efficiency.cu` | 05_pcie_transfer_efficiency.md | 1 KB → 1 MB 带宽爬升曲线 + 单次延迟 | `nvcc -arch=sm_80 -O3 -o pcie_transfer_efficiency 05_pcie_transfer_efficiency.cu` | `./pcie_transfer_efficiency` |
| `09_gpu_transfer_methods.cu` | 09_gpu_transfer_methods.md | 5 种 GPU→GPU 传输方法带宽对比（P2P/CPU relay/Zero-Copy/UM） | `nvcc -arch=sm_80 -O3 -o gpu_transfer_methods 09_gpu_transfer_methods.cu` | `CUDA_VISIBLE_DEVICES=0,1 ./gpu_transfer_methods` |

## 特殊说明

- **09_gpu_transfer_methods**：需要 ≥2 GPUs。P2P 方法（Method 1/2）仅在 Peer Access 可用时运行。`CUDA_VISIBLE_DEVICES` 用于选择特定的 GPU 对。
