# Nsight Systems CLI (`nsys`) 快速入门

> 基于 A100-SXM4-80GB + CUDA 13.1 实测。`nsys` 是 NVIDIA 系统级性能分析器，从时间线视角展示 CPU-GPU 交互、kernel 执行时长和 stream 并发效果。它与 `ncu`（kernel 内部）和 `nvbandwidth`（裸带宽）形成三件套工作流。

---

## 1. nsys 在三件套中的位置

[`06_nsight_compute_cli.md`](06_nsight_compute_cli.md) 的定位表在此展开：

| 工具          | 分析层级         | 核心问题                                                                   | 输出粒度                    |
| ------------- | ---------------- | -------------------------------------------------------------------------- | --------------------------- |
| `nvbandwidth` | 裸硬件           | PCIe/HBM 带宽是否达标？                                                    | 一个数字 (GB/s)             |
| **`ncu`**     | 单 kernel        | Memory bound 还是 Compute bound？                                          | 每个 kernel 的 200+ metrics |
| **`nsys`**    | **全系统时间线** | **CPU 在等 GPU 吗？stream 重叠了吗？第一个 cudaMalloc 为什么花了 200ms？** | 每个 CUDA API 调用的时间戳  |

**`ncu` 告诉你"kernel 为什么慢"，`nsys` 告诉你"谁在等谁"。**

对于以下问题，`nsys` 是首选工具：

- 为什么程序启动这么慢？→ `cuda_api_sum` 看 cudaMalloc 和 context 初始化
- 我的 stream 重叠真的发生了吗？→ 时间线直接可视化
- CPU 侧是 poll/ioctl 还是计算逻辑在拖慢整体？→ `osrt_sum` 看 OS runtime

---

## 2. 快速上手

### 2.1 安装

`nsys` 通常随 CUDA Toolkit 安装在独立路径。本环境：

```bash
/usr/local/bin/nsys --version
# NVIDIA Nsight Systems version 2025.5.2.266-255236693005v0
```

> 注意：`nsys` 不在 `<CUDA_PATH>/bin` 下，而是有独立的安装路径。在服务器上位于 `/usr/local/bin/nsys`。

### 2.2 基本用法

```bash
# 基础 profiling（生成 .nsys-rep 报告文件）
nsys profile -o my_report ./my_app

# 自动输出统计摘要（最常用）
nsys profile --stats=true -o my_report ./my_app

# 限制 trace 类型（减少 overhead 和文件体积）
nsys profile --trace=cuda,osrt -o my_report ./my_app

# 查看已有报告的统计摘要
nsys stats my_report.nsys-rep
```

### 2.3 默认报告类型

`nsys` 默认生成 6 个统计报告（即使不使用 `--stats=true`，`nsys stats` 也会输出它们）：

| 报告                      | 文件                    | 回答的问题                                      |
| ------------------------- | ----------------------- | ----------------------------------------------- |
| **osrt_sum**              | OS Runtime Summary      | CPU 侧时间花在哪了？poll/ioctl/mmap？           |
| **cuda_api_sum**          | CUDA API Summary        | cudaMalloc 花了多久？cudaMemcpy 多少次？        |
| **cuda_gpu_kern_sum**     | GPU Kernel Summary      | 每个 kernel 跑了多久、跑了多少次？              |
| **cuda_gpu_mem_time_sum** | GPU Memory Time Summary | H2D/D2H 总时间？                                |
| **cuda_gpu_mem_size_sum** | GPU Memory Size Summary | 传输了多少数据？                                |
| **nvtx_sum**              | NVTX Summary            | 自定义 annotation 区域的时间（需代码内嵌 NVTX） |

---

## 3. A100 实测：vectorAdd

以官方 cuda-samples 中的 `vectorAdd` 为目标程序：

```bash
cd cuda-samples/Samples/0_Introduction/vectorAdd
nvcc -arch=sm_80 -lineinfo -I../../../Common -o vectorAdd vectorAdd.cu -lcudart
nsys profile --stats=true -o vectorAdd_report ./vectorAdd
```

### 3.1 CUDA API Summary — CPU 侧 API 开销

```text
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)  Med (ns)    Name
 --------  ---------------  ---------  ----------  --------  --------------
     99.7      209,435,653          3  69,811,884   4,990    cudaMalloc
      0.1          233,650          3      77,883  37,340    cudaMemcpy
      0.1          194,810          3      64,936  17,670    cudaFree
      0.0           75,690          1      75,690  75,690    cudaLaunchKernel
```

**解读**：

- `cudaMalloc` 占 99.7% 的 CUDA API 时间——首次调用触发 CUDA context 初始化 + 物理显存分配，属于一次性开销。Median 仅 4,990 ns 说明后续 2 次分配是快的
- `cudaMemcpy`（3 次，avg 77.9 μs）和 `cudaLaunchKernel`（1 次，75.7 μs）是常态开销
- 如果看到 `cudaMemcpy` 的 avg >> med（如上表：77.9 μs vs 37.3 μs），说明某次 memcpy 特别大——查 `cuda_gpu_mem_size_sum`

### 3.2 GPU Kernel Summary — GPU 侧执行

```text
 Time (%)  Total Time (ns)  Instances  Avg (ns)    Name
 --------  ---------------  ---------  --------  --------------------
    100.0            3,040          1   3,040.0  vectorAdd(...)
```

vectorAdd kernel 在 GPU 上仅执行了 **3.04 μs**。对比前面 `cudaLaunchKernel` 的 75.7 μs，launch 开销是 kernel 执行时间的 25 倍——这就是为什么小 kernel 应该用 CUDA Graph（见 [`09_cuda_graphs.md`](../02_cuda/09_cuda_graphs.md)）。

### 3.3 GPU Memory Time Summary — H2D/D2H 时间线

```text
 Time (%)  Total Time (ns)  Count  Avg (ns)    Operation
 --------  ---------------  -----  --------  ----------------------------
     71.7           30,944      2   15,472   [CUDA memcpy Host-to-Device]
     28.3           12,192      1   12,192   [CUDA memcpy Device-to-Host]
```

2 次 H2D（input 数组拷贝，15.5 μs 每次），1 次 D2H（结果拷回，12.2 μs）。总计 memcpy 43.1 μs，占比远超 kernel 执行时间（3.0 μs）。

### 3.4 GPU Memory Size Summary — 数据传输量

```text
 Total (MB)  Count  Avg (MB)    Operation
 ----------  -----  --------  ----------------------------
      0.400      2     0.200  [CUDA memcpy Host-to-Device]
      0.200      1     0.200  [CUDA memcpy Device-to-Host]
```

2 次 H2D 每次 0.2 MB（50,000 floats × 2 arrays = 400 KB → 每个数组 200 KB），1 次 D2H 0.2 MB。总共 0.6 MB。

### 3.5 OS Runtime Summary — CPU 侧在干什么

```text
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)        Name
 --------  ---------------  ---------  ------------  --------------
     65.3      688,931,159         14  49,209,368    poll
     33.9      357,524,255      1,089     328,305    ioctl
      0.4        4,371,933         53      82,489    mmap64
```

**解读**：

- `poll`（65.3%, 689 ms）——程序在等 GPU 完成（`cudaDeviceSynchronize` 或隐式同步），这是正常的
- `ioctl`（33.9%, 358 ms）——CUDA driver 通信开销，1089 次 ioctl 调用均值 328 μs，中位数 28 μs，说明大部分很快但少数很大
- `mmap64` / `fopen`（< 0.5%）——程序加载和数据文件 I/O

> **经验法则**：如果 `poll` 占比 > 50%，说明程序是 GPU-bound（CPU 在等 GPU）。如果 `ioctl` 占比高且次数多，考虑 CUDA Graph 减少 driver 往返。

---

## 4. 常用分析场景

### 4.1 确认 stream 重叠

```bash
# 采集时间线
nsys profile --trace=cuda -o timeline ./stream_overlap

# CLI 无法直接看 Gantt 图，但可以从 kernel 时间戳推断
nsys stats --report cuda_gpu_kern_sum timeline.nsys-rep
```

**CLI 的局限**：时间线图需要 GUI。但通过对比 `cuda_gpu_kern_sum` 中各 kernel 的总执行时间和 wall-clock 时间，可以推断重叠效果。

### 4.2 找启动瓶颈

```bash
nsys profile --stats=true -o startup ./my_app
# 重点看 cuda_api_sum 的第一个 cudaMalloc 时间
```

首次 `cudaMalloc` 超 100ms → CUDA context 初始化。解决方案：预先 warmup 或使用 `cuInit` 提前初始化。

### 4.3 批量分析多个程序的对比

```bash
nsys profile -o baseline ./baseline_app
nsys profile -o optimized ./optimized_app
nsys stats baseline.nsys-rep > baseline.txt
nsys stats optimized.nsys-rep > optimized.txt
diff baseline.txt optimized.txt
```

---

## 5. 三件套工作流

```text
                    nvbandwidth
                    ┌──────────┐
                    │ 硬件基线  │ ← PCIe/HBM 带宽是否正常？
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │   nsys   │ ← 全局时间线：CPU 在等什么？stream 重叠了吗？
                    └────┬─────┘
                         │ 定位到具体的慢 kernel
                    ┌────▼─────┐
                    │   ncu    │ ← 进入 kernel：Memory bound？Occupancy 低？
                    └──────────┘
```

| 步骤 | 工具          | 问题                                    | 行动                                                                    |
| ---- | ------------- | --------------------------------------- | ----------------------------------------------------------------------- |
| 1    | `nvbandwidth` | 硬件基线正常吗？                        | 若否 → 查 PCIe 插槽/链路状态                                            |
| 2    | **`nsys`**    | CPU-GPU 时间线合理吗？stream 重叠了吗？ | 定位异常的 API 调用或 kernel 序列                                       |
| 3    | `ncu`         | 单个 kernel 内部瓶颈是什么？            | Memory → 查 coalescing；Compute → 查指令效率；Occupancy → 调 grid/block |

---

## 参考

- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute CLI 实战](06_nsight_compute_cli.md)
- [nvbandwidth 深度解析](01_nvbandwidth_best_practices.md)
- [CUDA Streams 并发实战](../02_cuda/07_cuda_streams_concurrency.md)
