# GPU 集群健康检查

> 基于 A100-SXM4-80GB (8 GPU) 现场数据。将日常运维中零散的检查项整合为系统化的健康检查流程——从温度/功耗的快速扫一眼，到 NVLink 链路、ECC 错误、MIG 状态等深入诊断。GPU 7（NVLink 故障 + MIG Enabled）作为贯穿全文的真实异常案例。

---

## 1. 检查分层

健康检查按深度分为三层：

| 层级               | 耗时   | 内容                                       | 频率           |
| ------------------ | ------ | ------------------------------------------ | -------------- |
| **L1: 快速扫一眼** | < 10s  | 温度、功耗、显存使用、GPU-Util（仅参考）   | 日常登录时     |
| **L2: 结构检查**   | < 1min | NVLink 链路、ECC 错误、MIG 模式、PCIe 状态 | 部署前、故障后 |
| **L3: 压力验证**   | > 5min | DCGM diag、NCCL bandwidth、P2P test        | 硬件变更后     |

---

## 2. L1: 快速扫一眼

### 2.1 温度与功耗

```bash
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit --format=csv
```

本节点输出：

```text
0, NVIDIA A100-SXM4-80GB, 28, 63.28 W, 400 W
1, NVIDIA A100-SXM4-80GB, 27, 62.80 W, 400 W
...
7, NVIDIA A100-SXM4-80GB, 26, 48.75 W, 400 W
```

**判断标准**：

| 指标     | 正常范围        | 预警                    | 危险               |
| -------- | --------------- | ----------------------- | ------------------ |
| 空闲温度 | 25-40°C         | > 60°C（散热不足）      | > 80°C（触发降频） |
| 空闲功耗 | 45-70W (A100)   | > 100W（有残留进程）    | —                  |
| 满载功耗 | 250-400W (A100) | < 250W（降频/功率限制） | —                  |

> 本节点全部 8 GPU 温度在 25-29°C，功耗 48-64W——空闲状态健康。

### 2.2 显存使用

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free --format=csv
```

发现显存残留（`memory.used > 500 MB` 且无进程）→ 僵尸进程未释放显存：`fuser -v /dev/nvidia*` 排查。

---

## 3. L2: 结构检查

### 3.1 NVLink 链路拓扑

```bash
nvidia-smi topo -m
```

```text
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0     X    NV12  NV12  NV12  NV12  NV12  NV12  NV12
GPU3    NV12  NV12  NV12   X    NV12  NV12  NV12  NV12
GPU7    SYS   SYS   SYS   SYS   NODE  NODE  PXB    X
```

**检查点**：

- [ ] 除对角线外，所有 GPU 对是否都是 `NV12`？（如果有 `SYS`/`NODE` → 该 GPU NVLink 故障）
- [ ] 是否存在 `PIX`（PCIe P2P 可用但非 NVLink）？（单机多卡训练不应出现）
- [ ] 同一 NUMA node 的 GPU 是否形成连续 NV12 组？

> 本节点：GPU 0-6 全部 NV12 ↔ OK。GPU 7 全部 SYS/NODE/PXB ↔ **NVLink 故障**。

### 3.2 NVLink 链路详情

```bash
nvidia-smi nvlink --status        # 每条 link 速率
dcgmi nvlink -s                   # DCGM 紧凑输出（适合脚本）
```

```text
gpuId 0-6: U U U U U U U U U U U U   ← 12 links, all Up
gpuId 7:   _ _ _ _ _ _ _ _ _ _ _ _   ← 0 links, all Not Supported
```

**判断**：gpuId 0-6 的 12 条 link 全部 `U`（Up），速率 25 GB/s → NVLink 3.0 健康。gpuId 7 全部 `_` → 物理层故障。

### 3.3 ECC 错误与 Retired Pages

ECC 是数据中心 GPU 的标配——A100 上默认启用，覆盖 HBM 显存和内部 SRAM。

#### ECC 模式确认

```bash
nvidia-smi --query-gpu=index,ecc.mode.current --format=csv
```

本节点全部 8 GPU 均为 `Enabled`——数据中心 GPU 默认状态。

#### Volatile vs Aggregate

```bash
nvidia-smi --query-gpu=index,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total --format=csv
```

当前输出（全部 `0`，仅 GPU 2 有 1 个 aggregate corrected 历史记录）：

| 类型          | 含义                                  | 重置方式                  |
| ------------- | ------------------------------------- | ------------------------- |
| **Volatile**  | 自上次驱动加载/GPU 重置以来的错误计数 | 重启驱动或 GPU 重置后清零 |
| **Aggregate** | GPU 整个生命周期内的累计错误          | 不可清零（硬件记录）      |

> **关键区分**：Volatile = 0 但 Aggregate > 0 → GPU 过去曾遇到错误但当前正常，无需处理。**Volatile 持续增长**才是需要关注的信号——说明自上次重启以来 HBM 仍在产生新的单比特错误。

#### 可接受的错误频率

| 错误类型               | 正常           | 需关注                | 需立即处理 |
| ---------------------- | -------------- | --------------------- | ---------- |
| Corrected / Volatile   | 0-10/day       | > 100/day             | > 1000/day |
| Uncorrected / Volatile | **必须为 0**   | 任何非零值            | > 0        |
| Corrected / Aggregate  | 任何值（历史） | Volatile 与之同步增长 | —          |

#### SRAM vs DRAM 错误

```bash
nvidia-smi -i <ID> -q | grep -A8 'ECC Errors'
```

输出区分两类错误的来源：

```text
Volatile
    SRAM Correctable           : 0     ← GPU 内部 SRAM (寄存器文件、L1、共享内存)
    SRAM Uncorrectable         : 0
    DRAM Correctable           : 0     ← HBM 显存
    DRAM Uncorrectable         : 0
```

- **SRAM 错误**：影响计算单元内部缓存。少量 correctable 可接受，uncorrectable 可能导致静默数据损坏
- **DRAM 错误**：影响 HBM 显存。单比特自动纠正，双比特不可纠正——如果 DRAM Uncorrectable > 0，立即停用该 GPU

#### Retired Pages 与 Remapped Rows

当 HBM 某个内存页反复出现单比特错误，GPU 固件会将其标记为 "retired"（退役），用备用内存行替换：

```bash
nvidia-smi -i <ID> -q | grep -A20 'Retired'
```

```text
Retired Pages
    Single Bit ECC            : N/A    ← A100 上此字段不再使用
    Double Bit ECC            : N/A
    Pending Page Blacklist    : N/A
Remapped Rows
    Correctable Error         : 0      ← 因单比特错误被重映射的行数
    Uncorrectable Error       : 0      ← 因双比特错误被重映射的行数
    Pending                   : No     ← 有待处理的重映射？
    Remapping Failure Occurred: No     ← 重映射失败？（备用行耗尽）
```

| 字段                         | 含义                        | 危险信号             |
| ---------------------------- | --------------------------- | -------------------- |
| `Correctable Error`          | 已被 remap 的单比特错误行数 | > 100 说明 HBM 老化  |
| `Uncorrectable Error`        | 已被 remap 的双比特错误行数 | **> 0 立即停用**     |
| `Remapping Failure Occurred` | 备用行耗尽，无法再 remap    | **Yes = GPU 需更换** |

> 本节点 GPU 3 的 Remapped Rows 全部为 0，无重映射记录——HBM 健康。

### 3.4 MIG 模式一致性

```bash
nvidia-smi --query-gpu=index,mig.mode.current --format=csv
```

```text
0-6: Disabled
7:   Enabled
```

**不一致 → 影响**：DCGM diag 无法运行；CUDA 设备枚举行为不同（MIG Enabled 无 GI 时 GPU 不可见）。

**建议**：在训练集群中保持所有 GPU 的 MIG 模式一致。推理集群可按需配置。

### 3.5 PCIe 链路状态

```bash
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv
```

A100 期望：`4, 16`（PCIe Gen 4 x16）。如果看到 `1, 16` → ASPM 空闲降级，正常。如果 max 也低 → 插槽问题。

---

## 4. L3: 压力验证

### 4.1 DCGM 诊断（Level 1 快速模式）

```bash
dcgmi diag -r 1 -i <GPU_ID>
```

> 前提：所有 GPU MIG 模式一致。当前 GPU 7 为 Enabled、GPU 0-6 为 Disabled，无法运行。参见 [DCGM 监控实操](05_dcgm_monitoring.md#5-诊断测试dcgmi-diag)。

### 4.2 P2P 带宽验证

```bash
# 在每对 GPU 上运行 simpleP2P，确认 P2P 可用且带宽达标
CUDA_VISIBLE_DEVICES=0,1 ./simpleP2P   # 应输出 ~239 GB/s (NVLink)
CUDA_VISIBLE_DEVICES=3,6 ./simpleP2P   # 应输出 ~239 GB/s
CUDA_VISIBLE_DEVICES=3,7 ./simpleP2P   # 应失败（SYS 不支持 P2P）
```

> 详见 [GPU P2P 带宽实测](../../02_gpu_programming/04_profiling/08_p2p_bandwidth.md)

### 4.3 NCCL AllReduce 带宽

```bash
# 在所有 NVLink 正常的 GPU 上运行（排除 GPU 7）
allreduce_perf -b 1G -e 8G -g 7   # 7 张 GPU (GPU 0-6)
```

---

## 5. 健康检查脚本

将 L1+L2 整合为单次执行：

```bash
#!/bin/bash
# gpu_health_check.sh — GPU 集群 L1+L2 健康检查
set -e

echo "=== [L1] 温度与功耗 ==="
nvidia-smi --query-gpu=index,temperature.gpu,power.draw --format=csv,noheader | \
  awk -F',' '{ printf "GPU %s: %s°C  %sW\n", $1, $2, $3 }'

echo ""
echo "=== [L1] 显存使用 ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | \
  awk -F',' '{ printf "GPU %s: %.0f / %.0f MiB\n", $1, $2, $3 }'

echo ""
echo "=== [L2] NVLink 拓扑摘要 ==="
nv_count=$(nvidia-smi topo -m 2>/dev/null | grep -c "NV12" || true)
sys_count=$(nvidia-smi topo -m 2>/dev/null | grep -c "\bSYS\b" || true)
echo "NV12 连接数: $nv_count  SYS 连接数: $sys_count"

echo ""
echo "=== [L2] ECC 错误 ==="
nvidia-smi --query-gpu=index,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv,noheader

echo ""
echo "=== [L2] PCIe 链路 ==="
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader | \
  awk -F',' '{ printf "GPU %s: Gen %s x%s\n", $1, $2, $3 }'

echo ""
echo "=== [L2] MIG 模式 ==="
nvidia-smi --query-gpu=index,mig.mode.current --format=csv,noheader
```

### 5.1 异常输出示例：GPU 7

本节点运行该脚本时，GPU 7 相关的异常信号：

```text
GPU 7: 26°C  48.75W              ← 温度/功耗正常
GPU 7: 0 / 81920 MiB              ← 显存完全空闲
NV12 连接数: 42  SYS 连接数: 28   ← GPU 7 贡献了 14 条 SYS 连接
GPU 7, 0, 0                       ← ECC 无错误
GPU 7: Gen 4 x16                  ← PCIe 链路正常
MIG Mode: GPU 7 = Enabled         ← 与 GPU 0-6 不一致
```

**诊断结论**：GPU 7 的 NVLink 硬件故障 + MIG Enabled（可能是故障排查后的配置）。不适合放入训练 TP 组，可用于独立推理或 MIG 实验。

---

## 6. 相关文档

- [`03_nvidia_smi_guide.md`](03_nvidia_smi_guide.md)：本文用到的 nvidia-smi 命令详解
- [`05_dcgm_monitoring.md`](05_dcgm_monitoring.md)：L3 压力验证中 DCGM 诊断的使用
- [NVLink 诊断与实操](../../01_hardware_architecture/nvlink/nvlink_diagnostics.md)：NVLink 链路层面的深入排查
- [GPU P2P 带宽实测](../../02_gpu_programming/04_profiling/08_p2p_bandwidth.md)：P2P 带宽验证作为 L3 压力测试的一部分

---

## 参考

- [NVIDIA DCGM 官方文档](https://docs.nvidia.com/datacenter/dcgm/latest/)
- [NCCL 环境变量](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
