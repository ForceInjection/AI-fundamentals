# NCCL Debug 输出实战解读

> 基于 A100-SXM4-80GB (3 GPU, NVLink NV12) + NCCL 2.29.3 实测。`NCCL_DEBUG=INFO` 是排查分布式通信问题的"第一手线索"——但默认输出近 100 行，大部分是噪音。本文逐段拆解初始化阶段的关键信息，标注哪些行正常、哪些行是危险信号、哪些字眼出现就意味着问题。

---

## 1. 如何获取 NCCL Debug 输出

```bash
NCCL_DEBUG=INFO ./build/all_reduce_perf -b 1M -e 1M -g 3
```

`NCCL_DEBUG` 有三个级别：

| 级别       | 输出量                                             | 何时使用             |
| ---------- | -------------------------------------------------- | -------------------- |
| `WARN`     | 仅警告和错误（默认）                               | 日常运行             |
| **`INFO`** | 拓扑检测 + 通信路径选择 + 初始化参数（~80-120 行） | **性能异常排查首选** |
| `TRACE`    | 每次通信操作的详细时间戳（数千行）                 | 极深层问题定位       |

> 输出同时打印到 stdout 和 stderr。建议重定向到文件：`NCCL_DEBUG=INFO ./test 2>&1 | tee nccl_debug.log`。

---

## 2. 完整输出示例（A100 3 GPU, NVLink）

以下是在 A100 服务器上对 3 张空闲 GPU 运行 AllReduce 的完整 `NCCL_DEBUG=INFO` 输出，关键行已加注释。

### 2.1 版本与环境

```text
gpu-node:2674380:2674380 [0] NCCL INFO ENV/Plugin: Could not find: libnccl-env.so
gpu-node:2674380:2674380 [0] NCCL INFO Bootstrap: Using eth0:10.0.0.1<0>
gpu-node:2674380:2674380 [0] NCCL INFO cudaDriverVersion 13000
gpu-node:2674380:2674380 [0] NCCL INFO NCCL version 2.29.3+cuda13.1
gpu-node:2674380:2674380 [0] NCCL INFO NCCL git version stable dcf2a2fbe
```

| 行                               | 解读                                                 | 正常信号                                      |
| -------------------------------- | ---------------------------------------------------- | --------------------------------------------- |
| `Could not find: libnccl-env.so` | 可选插件未加载——**无影响**，大多数环境都会出现这一行 | ✅ 可忽略                                     |
| `Bootstrap: Using eth0...`       | NCCL 使用该网络接口做初始握手（不是数据面通信）      | ✅ 单节点可忽略；多节点需确认是正确的 IB 接口 |
| `cudaDriverVersion 13000`        | CUDA 驱动版本 13.0                                   | ✅                                            |
| `NCCL version 2.29.3`            | NCCL 库版本                                          | ✅ 与安装版本一致                             |

### 2.2 拓扑配置

```text
gpu-node:2674380:2674380 [0] NCCL INFO Comm config nvlinkCentricSched set to 1
gpu-node:2674380:2674380 [1] NCCL INFO Comm config nvlinkCentricSched set to 1
gpu-node:2674380:2674380 [2] NCCL INFO Comm config nvlinkCentricSched set to 1
```

**关键字 `nvlinkCentricSched`**：NCCL 检测到 NVLink 存在，启用了 NVLink 优先的调度策略。三个 rank（GPU 0,1,2 = 物理 GPU 3,4,5）均确认启用。

> ⚠️ 如果这里出现的是 `pcieCentricSched`，说明 NCCL 认为 NVLink 不可用或不足以构成主要通信路径——检查 `nvidia-smi topo -m` 和 NVLink 链路状态。

### 2.3 网络后端检测

```text
gpu-node:2674380:2674388 [0] NCCL INFO NET/Plugin: Could not find: libnccl-net.so
gpu-node:2674380:2674388 [0] NCCL INFO NET/IB : No device found.
gpu-node:2674380:2674388 [0] NCCL INFO NET/IB : Using [RO]; OOB eth0:10.0.0.1<0>
gpu-node:2674380:2674388 [0] NCCL INFO Failed to initialize NET plugin IB
```

| 行                                   | 解读                                                  |
| ------------------------------------ | ----------------------------------------------------- |
| `NET/Plugin: Could not find`         | 未安装自定义网络插件——单节点无需                      |
| `NET/IB : No device found`           | 未检测到 InfiniBand 设备——本机无 IB 硬件              |
| `Failed to initialize NET plugin IB` | IB 初始化失败——因为无硬件，**预期的正常行为**         |
| `Using [RO]`                         | RO = Ring Order。单节点 NVLink 环境下的默认 Ring 算法 |

> **关键区分**：有 IB 硬件的多节点环境应看到 `NET/IB : Using ...` + 具体的 HCA 设备名（如 `mlx5_0`）。如果硬件存在但这里显示 `No device found`，检查 `ibv_devinfo` 和 `NCCL_IB_HCA` 环境变量。

### 2.4 GPU 识别与设备分配

```text
# Using devices
#  Rank  0 Group  0 Pid 2674380 on    gpu-node device  0 [0000:4c:00] NVIDIA A100-SXM4-80GB
#  Rank  1 Group  0 Pid 2674380 on    gpu-node device  1 [0000:84:00] NVIDIA A100-SXM4-80GB
#  Rank  2 Group  0 Pid 2674380 on    gpu-node device  2 [0000:8a:00] NVIDIA A100-SXM4-80GB
```

**三列关键信息**：

| 字段                    | 示例                      | 含义                                             |
| ----------------------- | ------------------------- | ------------------------------------------------ |
| `device`                | `0, 1, 2`                 | CUDA device ID（受 `CUDA_VISIBLE_DEVICES` 影响） |
| `[0000:XX:00]`          | `4c:00`, `84:00`, `8a:00` | GPU 的 PCIe BDF (Bus:Device.Function)            |
| `NVIDIA A100-SXM4-80GB` | —                         | GPU 型号确认                                     |

> **诊断用途**：物理 GPU 3 的 PCIe BDF 是 `0000:4C:00`，GPU 4 是 `0000:84:00`。可以用 `nvidia-smi -q` 交叉验证——如果 NCCL 报告某个 rank 使用了预期之外的 GPU，问题出在 `CUDA_VISIBLE_DEVICES` 或容器 GPU 分配。

### 2.5 通信拓扑与 NVLink 检测

```text
gpu-node:2674380:2674380 [0] NCCL INFO NVLS multicast support is available on domain 0
gpu-node:2674380:2674380 [0] NCCL INFO NVLS multicast is available on this communicator
gpu-node:2674380:2674380 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/NVLink/0
gpu-node:2674380:2674380 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/NVLink/1
...
gpu-node:2674380:2674380 [0] NCCL INFO Connected all rings
gpu-node:2674380:2674380 [0] NCCL INFO Connected all trees
```

| 行                                             | 解读                                                                                                       |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `NVLS multicast support is available`          | NVL Switch 多播可用——**NVSwitch 正常工作的标志**。如果 A100+NVSwitch 环境看不到这行，NVSwitch 可能未被识别 |
| `Channel XX/0 : 0[0] -> 1[1] via P2P/NVLink/X` | 每个通信通道的物理路径。这里的 `via P2P/NVLink/0` 确认数据走 NVLink 而非 PCIe                              |
| `Connected all rings`                          | Ring 拓扑构建完成——所有 GPU 已加入通信环                                                                   |
| `Connected all trees`                          | Tree 拓扑构建完成——用于小数据量的 Tree 算法                                                                |

> ⚠️ **危险信号对照**：如果看到 `via P2P/PCIe` 或 `via NET` 而非 `via P2P/NVLink`，GPU 间走了慢速路径。此时检查 `nvidia-smi nvlink --status`。

### 2.6 计算完成

```text
gpu-node:2674380:2674380 [0] NCCL INFO comm 0x7ff13c005d40 rank 0 nranks 3 cudaDev 0 nid 0 busId 4c00 ...
# AllReduce test output:
  1048576   262144  float  sum  -1  52.27  20.06  30.09  0  56.34  18.61  31.90  0
```

最后一行是 AllReduce 性能数据——`bus_bw = 30.09 GB/s`（out-of-place）和 `31.90 GB/s`（in-place）。3 GPU × 1 MB 数据时 bus_bw 较低是正常的（launch latency 主导），参见 [§5.5.3 带宽效率曲线](03_nccl_tutorial.md#553-带宽效率曲线)。

---

## 3. 正常 vs 异常速查

### 3.1 正常环境（NVLink + NVSwitch）

```text
✅ Comm config nvlinkCentricSched set to 1     ← NVLink 优先调度
✅ NVLS multicast support is available          ← NVSwitch 正常
✅ Channel XX via P2P/NVLink/X                  ← NVLink 路径
✅ Connected all rings / Connected all trees     ← 拓扑完整
```

### 3.2 NVLink 故障环境（如 GPU 7）

```text
❌ Comm config pcieCentricSched set to 1        ← 被迫走 PCIe 调度
❌ Channel XX via P2P/PCIe                      ← 仅 PCIe 路径（或 via NET 走网络）
❌ NVLS multicast support is available (可能仍在) ← NVSwitch 自身正常，但该 GPU 未连接
```

### 3.3 IB 缺失/故障环境（多节点）

```text
❌ NET/IB : No device found                     ← 无 IB 或 IB 驱动未加载
❌ NET/IB : Failed to initialize                ← IB 初始化失败
❌ NET/Socket : Using ...                       ← 回退到 TCP Socket（带宽极低）
```

### 3.4 版本/配置不一致

```text
❌ NCCL version mismatch                         ← 不同节点 NCCL 版本不同
❌ cudaDriverVersion 不同值                      ← 驱动版本不一致
```

---

## 4. 排查流程

```text
NCCL_DEBUG=INFO 输出拿到后：

1. 找 "Channel XX via" → 确认是 P2P/NVLink 还是 P2P/PCIe
   └── 如果是 P2P/PCIe → nvidia-smi nvlink --status 检查 NVLink

2. 找 "nvlinkCentricSched" → 确认值为 1
   └── 如果是 pcieCentricSched → nvidia-smi topo -m 看是否有 SYS/NODE

3. 找 "NET/IB" → 多节点时确认有 "Using ..." + HCA 设备名
   └── 如果 No device found → ibv_devinfo 确认 IB 硬件存在

4. 找 "Connected all rings / trees" → 确认拓扑完整
   └── 如果缺少 → 可能某个 GPU 不可达或 rank 配置错误

5. 找 busId → 与 nvidia-smi -q 交叉验证 GPU 分配是否正确
```

---

## 5. 相关文档

- [`03_nccl_tutorial.md`](03_nccl_tutorial.md)：§5.1 环境变量、§5.5 性能分析——二者是本文输出解读的下游
- [`04_nccl_benchmark.md`](04_nccl_benchmark.md)：本文输出的 `bus_bw` 值需要与基准数据对比
- [`06_gpu_health_check.md`](../01_gpu_ops/06_gpu_health_check.md)：本文 §3 的正常/异常对照是健康检查 L2 的核心
- [`nvlink_diagnostics.md`](../../01_hardware_architecture/nvlink/nvlink_diagnostics.md)：NVLink 故障时的底层排查

---

## 参考

- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL Troubleshooting Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
