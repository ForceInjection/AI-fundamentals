# DCGM 监控实操

> 基于 A100-SXM4-80GB (8 GPU) + DCGM v3.3.9 实测。DCGM (Data Center GPU Manager) 是 NVIDIA 官方推荐的 GPU 集群监控方案——它提供了 `nvidia-smi` 不具备的能力：正确的 SM 利用率、显存带宽利用率、长时间趋势采集、以及与 Prometheus/Grafana 的原生集成。本文覆盖 `dcgmi` 命令行的诊断、监控、健康检查三大功能。

---

## 1. DCGM 与 nvidia-smi 的区别

`nvidia-smi` 的 `GPU-Util` 只告诉你"某段时间有 kernel 在跑"——1 个线程在跑也是 100%，全部 SM 满载也是 100%，无法区分。DCGM 在 `nvidia-smi` 的指标之上，补充了**真正反映硬件利用率的指标**：

| 指标            | nvidia-smi          | DCGM                             | 说明                                   |
| --------------- | ------------------- | -------------------------------- | -------------------------------------- |
| GPU 利用率      | GPU-Util (误导性)   | **SM Active** / **SM Occupancy** | DCGM 区分"有 kernel 在跑"和"SM 被用满" |
| 显存带宽利用率  | ❌ 无               | **DRAM Active** / **PCIe TX/RX** | 判断是否是 memory bound                |
| 功耗            | `power.draw`        | **Power Usage**                  | 两者都有，DCGM 可做趋势                |
| NVLink 状态     | `nvidia-smi nvlink` | **NVLink 链路 Up/Down**          | DCGM 输出更紧凑，适合脚本解析          |
| 历史数据        | ❌ 无               | ✅ 时间序列存储                  | 回溯分析的关键                         |
| Prometheus 集成 | ❌ 无               | ✅ 原生 exporter                 | 集群级监控的基础                       |

> 前置阅读：[GPU 利用率是一个误导性指标](02_gpu_utilization_myth.md)

---

## 2. 安装与基础发现

### 2.1 安装

```bash
# Ubuntu/Debian
apt-get install datacenter-gpu-manager

# 验证
dcgmi --version
# dcgmi  version: 3.3.9
```

### 2.2 GPU 发现

```bash
dcgmi discovery -l
```

```text
8 GPUs found.
+--------+--------------------------------------------+
| GPU ID | Device Information                         |
+--------+--------------------------------------------+
| 0      | Name: NVIDIA A100-SXM4-80GB                |
|        | PCI Bus ID: 00000000:07:00.0               |
+--------+--------------------------------------------+
... (8 GPUs listed)
0 NvSwitches found.
```

> 注意：`dcgmi discovery` 显示 0 NvSwitch——这并不意味着 NVSwitch 不存在，而是 DCGM v3.3.9 的 NVSwitch 发现依赖特定驱动接口。NVSwitch 的实际状态通过 NVLink 链路状态间接反映。

### 2.3 创建 GPU 组

DCGM 操作基于"组"（Group）模型。先创建组，再将 GPU 加入：

```bash
# 创建包含所有支持 GPU 的组
dcgmi group -c allgpus --default

# 查看所有组
dcgmi group -l
```

```text
3 groups found.
  0: DCGM_ALL_SUPPORTED_GPUS       → GPU 0-7 (系统默认)
  1: DCGM_ALL_SUPPORTED_NVSWITCHES → None (无 NVSwitch 被检测)
  2: allgpus                       → GPU 0-7 (用户创建)
```

---

## 3. 实时监控：`dcgmi dmon`

`dcgmi dmon` 是 DCGM 的实时监控命令——以固定间隔采样 GPU 指标。

### 3.1 关键 Field ID 速查

| Field ID | 短名  | 含义                                              | 重要性            |
| -------- | ----- | ------------------------------------------------- | ----------------- |
| **203**  | GRACT | **SM Active** — 至少 1 个 warp 活跃的 SM 周期占比 | 替代 GPU-Util     |
| **204**  | MCUTL | Mem Controller Utilization — 显存带宽利用率       | 判断 memory bound |
| **1001** | SMACT | SM Active 时间 (μs)                               | 精确 SM 使用量    |
| **1002** | DRAMA | DRAM Active 时间 (μs)                             | 精确显存使用量    |
| **1005** | PCIRX | PCIe RX 吞吐                                      | 跨机通信数据量    |
| **150**  | TMPTR | GPU 温度 (°C)                                     | 散热健康          |
| **155**  | PWR   | GPU 功耗 (W)                                      | 能效评估          |
| **210**  | FBCRX | Frame Buffer Ctx RX                               | 显存读访问        |

### 3.2 基础监控示例

```bash
# 每 3 秒采样，共 5 次。监控 SM Active、显存利用率、温度
dcgmi dmon -e 203,204,1001,1002,150 -d 3000 -c 5
```

当前节点输出（8 GPU 全部空闲）：

```text
#Entity   GRACT      MCUTL      SMACT       DRAMA       TMPTR
ID                                    (us)        (us)        (C)
GPU 7     N/A        N/A        0.000       0.000       26
GPU 6     0          0          0.000       0.000       26
GPU 5     0          0          0.000       0.000       25
GPU 4     0          0          0.000       0.000       26
GPU 3     0          0          0.000       0.000       27
GPU 2     0          0          0.000       0.000       27
GPU 1     0          0          0.000       0.000       27
GPU 0     0          0          0.000       0.000       28
```

**解读**：

- GPU 7 `GRACT` 和 `MCUTL` 显示 `N/A`，其他 GPU 显示 `0`——前者是 MIG Enabled 导致的监控异常（参见 [MIG 配置与实操](../../01_hardware_architecture/nvidia/mig_hands_on.md)），后者是真实空闲
- 所有 GPU 的 SMACT 和 DRAMA 为零 — 无负载
- 温度 25-29°C — 健康空闲温度范围

> **对比 nvidia-smi**：同样的空闲状态，`nvidia-smi` 的 GPU-Util 也会显示 0%。但关键区别发生在**有负载时**——nvidia-smi 的 GPU-Util 50% 可能意味着 SM Active 只有 20% 或 80%，DCGM 能精确区分。

---

## 4. NVLink 监控：`dcgmi nvlink`

DCGM 的 NVLink 输出比 `nvidia-smi nvlink --status` 更紧凑，适合脚本化采集：

```bash
dcgmi nvlink -s
```

```text
gpuId 0: U U U U U U U U U U U U _ _ _ _ _ _   ← 12 条 NVLink 全 Up
gpuId 1: U U U U U U U U U U U U _ _ _ _ _ _
gpuId 2: U U U U U U U U U U U U _ _ _ _ _ _
gpuId 3: U U U U U U U U U U U U _ _ _ _ _ _
gpuId 4: U U U U U U U U U U U U _ _ _ _ _ _
gpuId 5: U U U U U U U U U U U U _ _ _ _ _ _
gpuId 6: U U U U U U U U U U U U _ _ _ _ _ _
gpuId 7: _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _   ← 全部 Not Supported

Key: Up=U, Down=D, Disabled=X, Not Supported=_
```

**一眼定位故障**：GPU 7 的 18 个 link 位置全部为 `_`（Not Supported）——与 `nvidia-smi topo -m` 中 GPU 7 的 SYS/NODE/PXB 连接相互印证。GPU 0-6 的 12 条 link 全部为 `U`（Up），NVLink 健康。

---

## 5. 诊断测试：`dcgmi diag`

DCGM 内置三级诊断（Level 1-3），从快速冒烟测试到深度压力测试：

```bash
dcgmi diag -r 1 -i 3   # Level 1: 快速诊断 (GPU 3)
dcgmi diag -r 2 -i 3   # Level 2: 中等压力
dcgmi diag -r 3 -i 3   # Level 3: 全力压力测试
```

**本环境限制**：当集群中同时存在 MIG Enabled (GPU 7) 和 MIG Disabled (GPU 0-6) 的 GPU 时，`dcgmi diag` 会报错：

```text
Cannot run diagnostic: CUDA does not support enumerating GPUs when one or
more GPUs has MIG mode enabled and one or more GPUs has MIG mode disabled.
```

**解决**：在运行 `dcgmi diag` 前，确保所有 GPU 的 MIG 模式一致（全部 Disabled 或全部 Enabled + 创建 GI/CI）。

---

## 6. Prometheus + Grafana 集成（概念）

DCGM 提供 `dcgm-exporter` 组件，将 GPU 指标暴露为 Prometheus 格式：

```bash
# 启动 DCGM Exporter (独立容器或 systemd 服务)
docker run -d --gpus all --name dcgm-exporter \
  nvidia/dcgm-exporter:latest
```

Grafana Dashboard 推荐模板：

- **NVIDIA DCGM Exporter Dashboard** (ID: 12239) — 多 GPU 概览
- 关键面板：SM Active、DRAM Active、NVLink 吞吐、温度、功耗

> 本文聚焦 CLI，exporter 的完整部署留给 Prometheus 运维专题。

---

## 7. 监控脚本示例

将 DCGM 监控嵌入日常运维：

```bash
#!/bin/bash
# quick_gpu_check.sh — 快速 GPU 集群健康检查

echo "=== GPU 概览 ==="
dcgmi discovery -l | grep -E "GPU ID|Name"

echo ""
echo "=== NVLink 链路状态 ==="
dcgmi nvlink -s | grep -c "U U U U U U U U U U U U" | awk '{print $1 " / 8 GPUs have 12 NVLink Up"}'

echo ""
echo "=== 当前温度 ==="
dcgmi dmon -e 150 -d 1000 -c 1 2>/dev/null | tail -9 | awk '{printf "GPU %s: %s°C\n", $1, $2}'

echo ""
echo "=== SM Active (最近 3 秒) ==="
dcgmi dmon -e 203 -d 3000 -c 1 2>/dev/null
```

---

## 8. 相关文档

- [`02_gpu_utilization_myth.md`](02_gpu_utilization_myth.md)：解释了为什么 DCGM 的 SM Active 才是正确指标
- [`03_nvidia_smi_guide.md`](03_nvidia_smi_guide.md)：nvidia-smi 基本操作，DCGM 是其升级版
- [NVLink 诊断与实操](../../01_hardware_architecture/nvlink/nvlink_diagnostics.md)：nvidia-smi nvlink vs dcgmi nvlink 双工具对比
- [MIG 配置与实操](../../01_hardware_architecture/nvidia/mig_hands_on.md)：MIG Enabled GPU 对 DCGM 监控的影响

---

## 参考文档

- [NVIDIA DCGM 官方文档](https://docs.nvidia.com/datacenter/dcgm/latest/)
- [DCGM Exporter for Prometheus](https://github.com/NVIDIA/dcgm-exporter)
- [NVIDIA DCGM Grafana Dashboards](https://grafana.com/grafana/dashboards/12239)
