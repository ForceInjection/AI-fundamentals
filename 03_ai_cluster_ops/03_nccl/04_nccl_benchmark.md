# NCCL 基准测试方法论

> NCCL 决定了多卡训练的通信效率。当吞吐"莫名掉一截"却找不到代码原因时，十有八九在 NCCL 这一层。本文覆盖从编译 `all_reduce_perf`、跑出真实带宽、到与拓扑对照诊断的全流程。
>
> as-of 2026-11-05：实测环境为 libnccl2 v2.29.3 + 8×A100-SXM4-80GB（NVSwitch Gen2，宿主驱动版本文中未记录）；NCCL 算法选择与带宽数字随版本、机型变化，引用前请按当前版本复核。

---

## 1. NCCL 基准测试是什么

NCCL 自带的 `nccl-tests` 提供标准化的集合通信性能测试：

| 测试                     | 测量对象           | 对应训练场景     |
| ------------------------ | ------------------ | ---------------- |
| `all_reduce_perf`        | AllReduce 带宽     | 数据并行梯度同步 |
| `all_gather_perf`        | AllGather 带宽     | ZeRO 参数收集    |
| `reduce_scatter_perf`    | ReduceScatter 带宽 | ZeRO 梯度分发    |
| `broadcast_perf`         | Broadcast 带宽     | 模型参数广播     |

每个测试在不同数据量（1B → 8 GB）和 GPU 数量（2 → 8）下各跑一轮，输出带宽 vs 数据量的完整曲线。

---

## 2. 编译 `nccl-tests`

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 CUDA_HOME=/usr/local/cuda
```

编译产物：`build/all_reduce_perf`、`build/all_gather_perf` 等。

> `NCCL_HOME` 不要指向 `/usr/lib/x86_64-linux-gnu`：nccl-tests 的 Makefile 会去 `$(NCCL_HOME)/include` 和 `$(NCCL_HOME)/lib64` 找头文件与库，装的是 Ubuntu 包（libnccl2 在 `/usr` 下）时直接编不过。包安装场景不设 `NCCL_HOME`，或显式设 `NCCL_HOME=/usr`。

> MPI 依赖：单节点测试不需要 MPI（`MPI=0`）。多节点需要 OpenMPI。

---

## 3. 运行 AllReduce 基准测试

### 3.1 基本命令

```bash
# 8 GPU 全部参与，数据范围 1M → 8G
./build/all_reduce_perf -b 1M -e 8G -f 2 -g 8

# 只测 NVLink 正常的 7 GPU（排除 GPU 7）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 ./build/all_reduce_perf -b 1M -e 8G -f 2 -g 7

# 单对 GPU（验证 NVLink 路径）
CUDA_VISIBLE_DEVICES=0,3 ./build/all_reduce_perf -b 64M -e 1G -g 2
```

| 参数 | 含义                     |
| ---- | ------------------------ |
| `-b` | 起始数据量               |
| `-e` | 结束数据量               |
| `-f` | 倍增因子（2 = 每次翻倍） |
| `-g` | GPU 数量                 |

### 3.2 期望带宽

A100 NVLink 3.0 (NV12) 的单向 P2P 带宽为 239 GB/s（实测，参见 [P2P 带宽实测](../../02_gpu_programming/04_profiling/08_p2p_bandwidth.md)）。A100 上大消息走 Ring，busbw 上限就是 NVLink 单向带宽；算法带宽（algbw）按 Ring 的流量系数 `2(N-1)/N` 折算：

| AllReduce 算法 | A100 (NVSwitch Gen2) 期望值                                 | 瓶颈           |
| -------------- | ------------------------------------------------------------ | -------------- |
| Ring           | busbw 上限 ≈ 239 GB/s；8 GPU algbw ≈ 239 × (8-1)/8 ≈ 209 GB/s | NVLink 单向带宽 |
| Tree           | 仅小消息回退使用；大消息 busbw 不会高于 Ring                 | 同上           |

> A100 的 NVSwitch 是 Gen2：没有 SHARP 在网归约，也没有 NVLS。NVLS 需要 NVSwitch Gen3 加 NCCL 2.17+（Hopper 世代起）。所以 A100 上超过 ~2 MB 的消息 NCCL 仍选 Ring（与 [NCCL 技术理论](01_nccl_theory.md) §1.2 一致）。实测 225 GB/s 约为 239 GB/s 上限的 94%，正是 Ring 的典型特征；「Tree 大消息接近 600 GB/s per GPU」是不可达的理论口径，不要拿它当期望值。

### 3.3 A100 实测数据

在 8 × A100-SXM4-80GB (NVSwitch Gen2) 上编译运行 nccl-tests，得到 2/4/7 GPU 的真实 AllReduce 带宽（`alg_bw` = 算法带宽 per GPU, `bus_bw` = 等效总线带宽）：

| GPU 数               | 1 GB bus_bw  | 256 MB bus_bw | 观测                                                      |
| -------------------- | ------------ | ------------- | --------------------------------------------------------- |
| **2** (GPU 0,3 NV12) | **197 GB/s** | 189 GB/s      | 接近 P2P 单向实测 (239 GB/s)，损耗来自 AllReduce 协议开销 |
| **4** (GPU 0-3 NV12) | **220 GB/s** | 213 GB/s      | Ring 的每跳经 NVSwitch 全互联转发，带宽不被组内 GPU 数摊薄 |
| **7** (GPU 0-6 NV12) | **225 GB/s** | 206 GB/s      | GPU 数增加但总线带宽不降：每 GPU 的 NVLink 带宽是独享的    |

完整 1 GB AllReduce 输出：

```text
# 2 GPU: 每个 GPU alg_bw ~197 GB/s, bus_bw = 197 GB/s
  1073741824     268435456     float     sum      -1  5438.06  197.45  197.45

# 4 GPU: alg_bw ~147 GB/s, bus_bw = 220 GB/s (N/(N-1) × alg_bw)
  1073741824     268435456     float     sum      -1  7304.94  146.99  220.48

# 7 GPU: alg_bw ~131 GB/s, bus_bw = 225 GB/s
  1073741824     268435456     float     sum      -1  8191.38  131.08  224.71
```

**关键发现**：

- `bus_bw` 从 197 → 220 → 225 GB/s 随 GPU 数量增长而**上升**：Ring 的每步通信都走独享 NVLink，链路随组扩大被利用得更充分
- 7 GPU 的 bus_bw (225 GB/s) 与 P2P 单向实测 (239 GB/s) 差距仅 ~6%：Ring 已把 NVLink 单向带宽用到 94%，离硬件上限不远
- 小数据量带宽：1 MB 仅 16.8 GB/s (8.5% 峰值), 8 MB 升至 89 GB/s (45%), 64 MB 达到 170 GB/s (86%)。数据量 < 64 MB 时 launch latency + 传输未饱和主导开销

---

## 4. 结果解读流程

拿到 `all_reduce_perf` 的带宽曲线后，对照 GPU 拓扑可以快速定位问题：

```text
理想的带宽曲线（8 × A100 NVSwitch 环境，busbw 口径，大消息走 Ring）：
          ▲
  225 ─── ┤                              ┌ Ring 饱和（单向带宽的 94%）
          ┤                         ╱╱╱
  150 ─── ┤                    ╱╱╱
          ┤               ╱╱╱
   75 ─── ┤          ╱╱╱
          ┤     ╱╱╱╱
    0 ─── ┼─────┬─────┬─────┬─────┬─────▶ 数据量
          1M    8M    64M  512M   4G   8G
```

**如果带宽远低于期望，可能的根因**：

| 现象                     | 可能原因                    | 排查命令                                                                                |
| ------------------------ | --------------------------- | --------------------------------------------------------------------------------------- |
| < 10 GB/s on any size    | 走了 PCIe 而非 NVLink       | `nvidia-smi topo -m` 检查连接类型                                                       |
| 某个 GPU 拖慢全组        | 该 GPU NVLink 故障          | `nvidia-smi nvlink --status -i <ID>`                                                    |
| < 100 GB/s on large data | 部分 GPU 对退化走 PCIe（NVLink 部分链路故障；A100 上 >2 MB 正常应走 Ring + NVLink） | `nvidia-smi topo -m` 逐对检查连接类型                                                   |
| 波动剧烈                 | 其他进程抢占 GPU            | `nvidia-smi` 检查 `compute processes`                                                   |
| 小数据量 (< 1M) 带宽极低 | launch latency 主导（正常） | 参照 [Kernel Launch 开销](../../02_gpu_programming/02_cuda/08_kernel_launch_latency.md) |

---

## 5. GPU 7 案例：MIG + NVLink 双重异常

本环境中 GPU 7 同时存在两个问题：**NVLink 硬件故障**（topo 显示 SYS/NODE/PXB）和 **MIG Enabled 但无实例**。实测影响：

```bash
# 尝试将 GPU 7 纳入 AllReduce 组
CUDA_VISIBLE_DEVICES=3,7 ./build/all_reduce_perf -b 1M -e 1G -g 2
# 输出: Invalid number of GPUs: 2 requested but only 1 were found.

# 8 GPU 全量
./build/all_reduce_perf -b 1M -e 1G -g 8
# 输出: Invalid number of GPUs: 8 requested but only 7 were found.
```

**根因分析**：

| GPU 7 问题                      | 表现                                               | 影响                           |
| ------------------------------- | -------------------------------------------------- | ------------------------------ |
| MIG Enabled, 无 GI/CI           | CUDA `cudaGetDeviceCount()` 返回 7（不包含 GPU 7） | NCCL 根本无法发现 GPU 7        |
| NVLink 全部 `_` (Not Supported) | 即使 MIG 修复，topo 也是 SYS/PXB                   | 带宽上限 ~28 GB/s (PCIe Gen 4) |

**两个问题的组合效果**：MIG 状态使 GPU 7 完全透明，比 NVLink 故障本身更隐蔽：`nvidia-smi` 能看到 GPU 7，但 `all_reduce_perf` 看不到。排查时先用 `cudaGetDeviceCount()` 检查设备数量是否与物理 GPU 数量一致。

```bash
#!/bin/bash
# 检查所有 GPU 的 NVLink 状态，任何异常直接退出
BAD_GPUS=$(nvidia-smi topo -m 2>/dev/null | grep -c "SYS" || true)
if [ "$BAD_GPUS" -gt 0 ]; then
    echo "WARNING: $BAD_GPUS SYS connections detected — possible NVLink fault"
fi
```

---

## 6. 参考

- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/)
- [NCCL 环境变量完整列表](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [nccl-tests GitHub](https://github.com/NVIDIA/nccl-tests)
- [`01_nccl_theory.md`](01_nccl_theory.md)：NCCL 算法理论（AllReduce/RDMA/性能建模）
- [`02_nccl_helloworld.md`](02_nccl_helloworld.md)：单卡 NCCL 验证
- [`03_nccl_tutorial.md`](03_nccl_tutorial.md)：完整使用教程与部署
- [GPU P2P 带宽实测](../../02_gpu_programming/04_profiling/08_p2p_bandwidth.md)：NCCL 期望带宽的硬件上限
- [NVLink 诊断与实操](../../01_hardware_architecture/nvlink/nvlink_diagnostics.md)：NVLink 故障对 NCCL 路径的影响
- [GPU 集群健康检查](../01_gpu_ops/06_gpu_health_check.md)：NCCL 验证作为 L3 压力测试
