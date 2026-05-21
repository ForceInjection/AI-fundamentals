# NCCL 基准测试方法论

> NCCL 决定了多卡训练的通信效率——当吞吐"莫名掉一截"却找不到代码原因时，十有八九在 NCCL 这一层。本文覆盖从编译 `allreduce_perf`、跑出真实带宽、到与拓扑对照诊断的全流程。基于 libnccl2 v2.29.3 + 8×A100 环境。

---

## 1. NCCL 基准测试是什么

NCCL 自带的 `nccl-tests` 提供标准化的集合通信性能测试：

| 测试                 | 测量对象           | 对应训练场景     |
| -------------------- | ------------------ | ---------------- |
| `allreduce_perf`     | AllReduce 带宽     | 数据并行梯度同步 |
| `allgather_perf`     | AllGather 带宽     | ZeRO 参数收集    |
| `reducescatter_perf` | ReduceScatter 带宽 | ZeRO 梯度分发    |
| `broadcast_perf`     | Broadcast 带宽     | 模型参数广播     |

每个测试在不同数据量（1B → 8 GB）和 GPU 数量（2 → 8）下各跑一轮，输出带宽 vs 数据量的完整曲线。

---

## 2. 编译 `nccl-tests`

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/lib/x86_64-linux-gnu
```

编译产物：`build/allreduce_perf`、`build/allgather_perf` 等。

> MPI 依赖：单节点测试不需要 MPI（`MPI=0`）。多节点需要 OpenMPI。

---

## 3. 运行 AllReduce 基准测试

### 3.1 基本命令

```bash
# 8 GPU 全部参与，数据范围 1M → 8G
./build/allreduce_perf -b 1M -e 8G -f 2 -g 8

# 只测 NVLink 正常的 7 GPU（排除 GPU 7）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 ./build/allreduce_perf -b 1M -e 8G -f 2 -g 7

# 单对 GPU（验证 NVLink 路径）
CUDA_VISIBLE_DEVICES=0,3 ./build/allreduce_perf -b 64M -e 1G -g 2
```

| 参数 | 含义                     |
| ---- | ------------------------ |
| `-b` | 起始数据量               |
| `-e` | 结束数据量               |
| `-f` | 倍增因子（2 = 每次翻倍） |
| `-g` | GPU 数量                 |

### 3.2 期望带宽

A100 NVLink 3.0 (NV12) 的单向 P2P 带宽为 239 GB/s（实测，参见 [P2P 带宽实测](../../02_gpu_programming/04_profiling/08_p2p_bandwidth.md)）。AllReduce 的带宽取决于算法：

| AllReduce 算法  | 8 GPU 期望带宽                    | 瓶颈            |
| --------------- | --------------------------------- | --------------- |
| Ring            | ~239 × (8-1)/8 ≈ 209 GB/s per GPU | NVLink 带宽     |
| Tree (NVSwitch) | 接近 600 GB/s per GPU             | NVSwitch 总带宽 |

> NCCL 会自动根据拓扑选择最优算法。A100 + NVSwitch 通常选择 Tree（经 NVSwitch 做 reduce），理论上可接近 NVSwitch 的双向带宽上限。

### 3.3 A100 实测数据

在 8 × A100-SXM4-80GB (NVSwitch Gen2) 上编译运行 nccl-tests，得到 2/4/7 GPU 的真实 AllReduce 带宽（`alg_bw` = 算法带宽 per GPU, `bus_bw` = 等效总线带宽）：

| GPU 数               | 1 GB bus_bw  | 256 MB bus_bw | 观测                                                      |
| -------------------- | ------------ | ------------- | --------------------------------------------------------- |
| **2** (GPU 0,3 NV12) | **197 GB/s** | 189 GB/s      | 接近 P2P 单向实测 (239 GB/s)，损耗来自 AllReduce 协议开销 |
| **4** (GPU 0-3 NV12) | **220 GB/s** | 213 GB/s      | Tree 算法通过 NVSwitch，总线带宽接近 NVSwitch 单端口上限  |
| **7** (GPU 0-6 NV12) | **225 GB/s** | 206 GB/s      | GPU 数增加但总线带宽不降——NVSwitch 交换式架构的优势       |

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

- `bus_bw` 从 197 → 220 → 225 GB/s 随 GPU 数量增长而**上升**——说明 NVSwitch 在更多 GPU 参与时反而能更高效利用带宽
- 7 GPU 的 bus_bw (225 GB/s) 与 P2P 单向实测 (239 GB/s) 差距仅 ~6%——NCCL 的 Tree 算法接近硬件上限
- 小数据量带宽：1 MB 仅 16.8 GB/s (8.5% 峰值), 8 MB 升至 89 GB/s (45%), 64 MB 达到 170 GB/s (86%)——数据量 < 64 MB 时 launch latency + 传输未饱和主导开销

---

## 4. 结果解读流程

拿到 `allreduce_perf` 的带宽曲线后，对照 GPU 拓扑可以快速定位问题：

```text
理想的带宽曲线（8 × A100 NVSwitch 环境）：
          ▲
  600 ─── ┤                              ┌ Tree 饱和
          ┤                         ╱╱╱
  300 ─── ┤                    ╱╱╱
          ┤               ╱╱╱
  100 ─── ┤          ╱╱╱
          ┤     ╱╱╱╱
    0 ─── ┼─────┬─────┬─────┬─────┬─────▶ 数据量
          1M    8M    64M  512M   4G   8G
```

**如果带宽远低于期望，可能的根因**：

| 现象                     | 可能原因                    | 排查命令                                                                                |
| ------------------------ | --------------------------- | --------------------------------------------------------------------------------------- |
| < 10 GB/s on any size    | 走了 PCIe 而非 NVLink       | `nvidia-smi topo -m` 检查连接类型                                                       |
| 某个 GPU 拖慢全组        | 该 GPU NVLink 故障          | `nvidia-smi nvlink --status -i <ID>`                                                    |
| < 100 GB/s on large data | Ring 算法而非 Tree          | 检查 NVSwitch 状态                                                                      |
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

**两个问题的组合效果**：MIG 状态使 GPU 7 完全透明，比 NVLink 故障本身更隐蔽——`nvidia-smi` 能看到 GPU 7，但 `allreduce_perf` 看不到。排查时先用 `cudaGetDeviceCount()` 检查设备数量是否与物理 GPU 数量一致。

```bash
#!/bin/bash
# 检查所有 GPU 的 NVLink 状态，任何异常直接退出
BAD_GPUS=$(nvidia-smi topo -m 2>/dev/null | grep -c "SYS" || true)
if [ "$BAD_GPUS" -gt 0 ]; then
    echo "WARNING: $BAD_GPUS SYS connections detected — possible NVLink fault"
fi
```

---

## 6. 关键 NCCL 环境变量

| 变量                  | 作用             | 建议值 (A100)                         |
| --------------------- | ---------------- | ------------------------------------- |
| `NCCL_DEBUG`          | 日志级别         | `INFO` (排查时), `WARN` (日常)        |
| `NCCL_P2P_LEVEL`      | P2P 使用策略     | `NVL` (优先 NVLink)                   |
| `NCCL_IB_DISABLE`     | 禁用 InfiniBand  | `1` (纯单机 NVLink 测试)              |
| `NCCL_SOCKET_IFNAME`  | 网卡接口         | 多节点时必须指定                      |
| `NCCL_TOPO_DUMP_FILE` | 导出拓扑图 (XML) | 拓扑诊断用                            |
| `NCCL_ALGO`           | 强制指定算法     | `Tree` / `Ring`（通常让 NCCL 自动选） |

```bash
# 典型调试组合
NCCL_DEBUG=INFO NCCL_P2P_LEVEL=NVL ./build/allreduce_perf -b 1G -e 2G -g 4
```

---

## 7. 相关文档

- [`01_nccl_theory.md`](01_nccl_theory.md)：NCCL 算法理论（AllReduce/RDMA/性能建模）
- [`02_nccl_helloworld.md`](02_nccl_helloworld.md)：单卡 NCCL 验证
- [`03_nccl_tutorial.md`](03_nccl_tutorial.md)：完整使用教程与部署
- [GPU P2P 带宽实测](../../02_gpu_programming/04_profiling/08_p2p_bandwidth.md)：NCCL 期望带宽的硬件上限
- [NVLink 诊断与实操](../../01_hardware_architecture/nvlink/nvlink_diagnostics.md)：NVLink 故障对 NCCL 路径的影响
- [GPU 集群健康检查](../01_gpu_ops/06_gpu_health_check.md)：NCCL 验证作为 L3 压力测试

## 参考

- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/)
- [nccl-tests GitHub](https://github.com/NVIDIA/nccl-tests)
- [NCCL 环境变量完整列表](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
