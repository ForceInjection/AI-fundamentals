# CUDA 示例代码

本目录包含 02_gpu_programming/02_cuda/ 下各文章的配套可运行代码。每个 `.cu` 文件对应一篇文章，均可独立编译运行。

## 前置要求

- **CUDA Toolkit** ≥ 11.0（推荐 12.x）
- **GPU**：CC 8.0+（A100/H100/RTX 30xx+）。部分 demo 可在更低 CC 上运行，但实测数据以 A100 (CC 8.0) 为基准
- **系统**：Linux（05_numa_demo 需 `numactl`）

## 快速编译全部

```bash
# 在 A100 (CC 8.0) 上
for f in *.cu; do
    name=$(basename "$f" .cu)
    echo "=== $name ==="
    nvcc -arch=sm_80 -O3 -o "$name" "$f"
done
```

如需适配其他 GPU，将 `-arch=sm_80` 替换为你的 Compute Capability（如 RTX 4090 用 `sm_89`，H100 用 `sm_90`）。

## 文件清单

| 文件                              | 配套文章                          | 用途                                                                                | 编译                                                                               | 运行                                                                                              |
| --------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `05_cuda_numa_demo.cu`            | 05_cuda_numa_api.md               | NUMA 亲和性、Managed Memory API 演示                                                | `nvcc -arch=sm_80 -O3 -o numa_demo 05_cuda_numa_demo.cu`                           | `./numa_demo`（双路系统用 `numactl --cpunodebind=0 ./numa_demo` 和 `--cpunodebind=1` 分别跑对比） |
| `06_device_attributes.cu`         | 06_device_attributes.md           | 查询 20+ 种 GPU 硬件能力（原子操作、Managed Memory、P2P 等）                        | `nvcc -arch=sm_80 -O3 -o device_attrs 06_device_attributes.cu`                     | `./device_attrs`                                                                                  |
| `07_streams_concurrency_bench.cu` | 07_cuda_streams_concurrency.md    | 4 stream 串行 vs 并发对比（H2D + Kernel + D2H 重叠）                                | `nvcc -arch=sm_80 -O3 -o streams_bench 07_streams_concurrency_bench.cu`            | `./streams_bench`                                                                                 |
| `08_kernel_launch_latency.cu`     | 08_kernel_launch_latency.md       | 空 kernel launch 延迟测量 + CPU/GPU 决策边界                                        | `nvcc -arch=sm_80 -O3 -o launch_latency 08_kernel_launch_latency.cu`               | `./launch_latency`                                                                                |
| `09_cuda_graphs_demo.cu`          | 09_cuda_graphs.md                 | CUDA Graph：Stream Capture → Instantiate → Re-launch 全生命周期                     | `nvcc -arch=sm_80 -O3 -o cuda_graphs_demo 09_cuda_graphs_demo.cu`                  | `./cuda_graphs_demo`                                                                              |
| `10_reduction_bench.cu`           | 10_reduction.md                   | Reduction 7 个 kernel 变体（K0 Interleaved → K6 CG+Grid）                           | `nvcc -arch=sm_80 -O3 -o reduction_bench 10_reduction_bench.cu`                    | `./reduction_bench [--n 1048576]`                                                                 |
| `11_tensor_core_gemm_bench.cu`    | 11_tensor_core_gemm.md            | WMMA API FP16 GEMM + TFLOPS 测量                                                    | `nvcc -arch=sm_80 -O3 -o gemm_bench 11_tensor_core_gemm_bench.cu`                  | `./gemm_bench`                                                                                    |
| `13_bank_conflict_bench.cu`       | 13_shared_memory_bank_conflict.md | Bank Conflict 三项测试：Stride 带宽退化、Transpose Padding 加速、Reduction 寻址方式 | `nvcc -arch=sm_80 -O3 -o bank_conflict_bench 13_bank_conflict_bench.cu`            | `./bank_conflict_bench`                                                                           |
| `14_warp_level_bench.cu`          | 14_warp_level_programming.md      | Warp 原语四项测试：Shuffle/SMEM Reduce、Prefix Sum Scan、Ballot Filter、Match Dedup | `nvcc -arch=sm_80 -O3 -o warp_bench 14_warp_level_bench.cu`                        | `./warp_bench`                                                                                    |
| `15_multi_gpu_bench.cu`           | 15_multi_gpu_programming.md       | 多 GPU 四项测试：Peer Access、P2P 带宽、跨 GPU 同步、NCCL AllReduce                 | `nvcc -arch=sm_80 -O3 -DWITH_NCCL -o multi_gpu_bench 15_multi_gpu_bench.cu -lnccl` | `./multi_gpu_bench`                                                                               |
| `16_async_copy_bench.cu`          | 16_async_copy_pipeline.md         | Async Copy 对比：1-stage sync vs 2-stage double buffer tile 处理                    | `nvcc -arch=sm_80 -O3 -o async_copy_bench 16_async_copy_bench.cu`                   | `./async_copy_bench`                                                                              |

## 使用 Nsight Compute 分析

部分 benchmark 支持用 `ncu` 查看底层指标：

```bash
# 查看 Shared Memory Bank Conflict
ncu --set memory --section MemoryWorkloadAnalysis_Tables ./bank_conflict_bench

# 查看 Occupancy
ncu --section Occupancy ./reduction_bench

# 查看完整 Compute + Memory 分析
ncu --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis ./gemm_bench
```

## 特殊说明

- **05_numa_demo**：单路系统也可运行，但看不到 NUMA 带宽差异。双路系统上需安装 `numactl`（`apt install numactl`），分别绑到不同 NUMA 节点比较 H2D/D2H 带宽。
- **09_cuda_graphs_demo**：capture stream 和 launch stream **必须分开**，否则返回 "device not ready"。
- **10_reduction_bench**：默认 1M elements。用 `--n 2097152` 指定更大规模。数据量越大，各 kernel 之间的差异越显著。
- **11_tensor_core_gemm_bench**：使用 WMMA API（教学级），仅能达到 ~6% 峰值。cuBLAS 可达 ~58%。不要用这个数字评估 GPU 性能。
