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

### 3.3 ECC 错误

```bash
nvidia-smi --query-gpu=index,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv
```

| 错误类型             | 可接受值 | 需关注     | 需立即处理 |
| -------------------- | -------- | ---------- | ---------- |
| Corrected (单比特)   | 0-10/day | > 100/day  | > 1000/day |
| Uncorrected (双比特) | **0**    | 任何非零值 | > 0        |

> ECC 单比特错误会被硬件自动纠正，不影响计算正确性。但快速增长的单比特错误暗示 HBM 模块可能退化。

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
