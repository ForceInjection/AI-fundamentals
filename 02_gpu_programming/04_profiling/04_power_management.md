# GPU 功耗管理实战

> 基于 RTX 5090 (575W TDP, 400-575W 可调范围) 实测。测试 nvidia-smi 的功耗管理功能，记录消费级 GPU 与数据中心 GPU 的能力差异。

---

## 0. 为什么关注功耗

GPU 功耗直接影响三个维度：

- **性能天花板**：功耗上限决定 Boost 时钟能跑多高。GPU 的 Boost 算法会持续推高频率直到触及功耗、温度或电压上限中的任意一个。如果功耗上限被卡住，即使温度和电压有余量，时钟也不会再升。
- **散热与稳定性**：575W 的 RTX 5090 在满负载下会产出巨大的热量。如果散热不足，GPU 会因温度墙（~83-85°C）而强制降频，功耗限制在这里反而起到了保护作用。
- **成本**：在电费敏感的机房或长期运行的训练任务中，适当降低功耗上限（如 575W → 450W）可以节省 20% 电力，而性能损失通常远小于 20%（因为电压-频率曲线在高端是非线性的——最后 100W 只换来 5-10% 的性能提升）。

GPU 的功耗管理基于 **P-State**（Performance State），从 P0（最高性能）到 P8（最低功耗）。GPU 根据负载自动切换 P-State：

```text
P0 (3090 MHz, 575W) ← 满负载
P2-P5              ← 中等负载
P8 (180 MHz, ~11W)  ← 空闲
```

---

## 1. RTX 5090 功耗规格

```bash
nvidia-smi --query-gpu=power.draw,power.limit,power.default_limit --format=csv
# 输出: 11.51 W, 575.00 W, 575.00 W
```

| 参数            | 值                               |
| --------------- | -------------------------------- |
| 默认功耗上限    | 575 W                            |
| 可设置范围      | 400 - 575 W                      |
| 空闲功耗        | ~11 W (P8 状态)                  |
| 最大 Boost 时钟 | 3090 MHz (GPU), 14001 MHz (显存) |

---

## 2. 功耗限制设置

降低功耗上限是最直接的控制手段。设置后会立即生效——GPU 硬件会将 Boost 时钟限制在不超过新功耗上限的范围内。`persistence mode` 必须启用，否则 GPU 在无进程使用时会被释放。

### 2.1 条件要求

```bash
# 必须启用 persistence mode
nvidia-smi -pm 1

# 设置功耗上限 (需管理员权限)
nvidia-smi -pl 500 -i 0

# 重置为默认值
nvidia-smi -pl 575 -i 0
```

### 2.2 云环境限制

在 Seetacloud 等云环境中，即使以 root 身份，`-pl` 可能返回 `Insufficient Permissions`。这是因为：

- 底层 Hypervisor 锁定了功耗管理
- GPU 驱动配置中禁用了 power management 接口
- 容器化环境中 power cgroup 限制

**本环境结果**：

```bash
nvidia-smi -pl 400
# Failed to set power management limit: Insufficient Permissions
```

> 提示：如果遇到此限制，仍可以通过 `nvidia-smi --query-gpu=power.draw` 监控实际功耗。

---

## 3. 时钟频率与 P-State

GPU 核心和显存的时钟频率不是固定值——它们随负载、温度和功耗动态调整。这个机制叫 **GPU Boost**：空闲时降到 P8（~180 MHz），负载时升到 P0（最高 3090 MHz）。显存频率通常固定在高位（14001 MHz）以保证显存带宽不成为瓶颈。

`--query-supported-clocks` 列出了 GPU 硬件支持的所有合法时钟组合。每种组合对应一个电压点——频率越高，电压越高，功耗呈二次方增长（P ∝ fV²）。这也是为什么锁频在略低于最高频的位置可以显著降低功耗。

### 3.1 查看当前状态

```bash
# 当前 P-State 和时钟
nvidia-smi --query-gpu=pstate,clocks.current.sm,clocks.current.graphics,clocks.current.memory --format=csv
```

空闲时 RTX 5090 运行在 P8：

```text
P8, 180 MHz, 180 MHz, 405 MHz
```

### 3.2 查看支持的所有时钟组合

```bash
nvidia-smi --query-supported-clocks=memory,graphics --format=csv | head -20
```

RTX 5090 部分输出：

```text
14001 MHz, 3090 MHz
14001 MHz, 3082 MHz
14001 MHz, 3075 MHz
...
14001 MHz, 2955 MHz
```

显存固定在最高频，GPU 核心从 2955 到 3090 MHz 以 7-8 MHz 步进可调。

### 3.3 锁定时钟（需管理员权限）

```bash
# 锁定 GPU 核心到 2500 MHz
nvidia-smi -lgc 2500

# 锁定显存时钟
nvidia-smi -lmc 12000

# 重置为默认
nvidia-smi -rgc
nvidia-smi -rmc
```

---

## 4. Power Smoothing

`power-smoothing` 是 v595 新增功能，允许预设功耗 ramp 策略。

```bash
# 查看帮助
nvidia-smi power-smoothing -h

# 启用功耗平滑
nvidia-smi power-smoothing -s 1

# 打印预设 profile 定义
nvidia-smi power-smoothing -ppd
```

**RTX 5090 实测**：该功能在消费级 GPU 上行为有限，`-ppd` 返回空。此功能主要面向数据中心 GPU (A100/H100/B200)。

---

## 5. Power Profiles

```bash
nvidia-smi power-profiles -l
```

**RTX 5090 输出**：

```text
Workload Power Profiles feature is not supported on this device.
```

Power Profiles 仅数据中心 GPU 支持。

---

## 6. 消费级 vs 数据中心功耗功能对比

| 功能                                | RTX 5090  | A100/H100 | 说明                      |
| ----------------------------------- | --------- | --------- | ------------------------- |
| 功耗监控 (`power.draw`)             | ✅        | ✅        | 基本功能                  |
| 功耗限制 (`-pl`)                    | ✅ 需权限 | ✅        | 云环境可能被锁定          |
| 时钟监控 (`clocks`)                 | ✅        | ✅        | nvidia-smi + NVML         |
| 锁定时钟 (`-lgc/-lmc`)              | ✅        | ✅        | 需管理员权限              |
| Power Smoothing (`power-smoothing`) | ⚠️ 受限   | ✅        | 数据中心完整支持          |
| Power Profiles (`power-profiles`)   | ❌        | ✅        | 消费级不支持              |
| GPM (`gpm`)                         | ✅        | ✅        | GPU 性能监控              |
| DCGM                                | ⚠️ 部分   | ✅        | 需 datacenter-gpu-manager |

---

## 7. 功耗感知编程建议

### 7.1 监控功耗趋势

```bash
# 每秒采样
nvidia-smi --query-gpu=timestamp,power.draw,temperature.gpu,clocks.current.sm \
    --format=csv -l 1
```

### 7.2 CUDA 侧查询功耗

```c
// 通过 NVML (libnvidia-ml)
#include <nvml.h>
nvmlInit();
nvmlDevice_t dev;
nvmlDeviceGetHandleByIndex(0, &dev);
unsigned int power;
nvmlDeviceGetPowerUsage(dev, &power);  // mW
printf("Power: %.1f W\n", power / 1000.0);
```

编译：`nvcc -lnvidia-ml -o power_mon power_mon.cu`

---

## 参考

- [nvidia-smi 快速入门](../../03_ai_cluster_ops/01_gpu_ops/03_nvidia_smi_guide.md)
- [GPU 利用率是一个误导性指标](../../03_ai_cluster_ops/01_gpu_ops/02_gpu_utilization_myth.md)
- [NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/)
