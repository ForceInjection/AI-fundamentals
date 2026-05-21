# MIG 配置与实操

> 基于 A100-SXM4-80GB (GPU 7, MIG Enabled) 的现场查询数据。MIG (Multi-Instance GPU) 将一张物理 A100 切分为最多 7 个独立 GPU 实例——每个有专用 SM、显存、L2 Cache 和内存带宽。本文覆盖查询、启用/禁用、创建实例的全流程，基于 `nvidia-smi mig` 命令集，不执行任何变更操作。

---

## 1. 什么是 MIG，为什么需要它

在共享 GPU 集群中，多个用户的 workload 跑在同一张 GPU 上时面临三个问题：

| 问题     | 传统方案           | 方案缺陷                      |
| -------- | ------------------ | ----------------------------- |
| 显存隔离 | 无（大家共享显存） | OOM 互相影响                  |
| 算力隔离 | GPU 时间片 (MPS)   | 无故障隔离，kernel 可能被阻塞 |
| QoS 保证 | 靠调度器优先级     | 无法保证延迟 SLO              |

**MIG 的解决方案**：在硬件层将一张 GPU 物理分区——每个实例有独立的内存通道、SM 集合、L2 Cache 分片和显存控制器。

```text
                    A100-80GB (MIG Disabled)
                    ┌───────────────────────┐
                    │    1 × 完整 GPU        │
                    │    108 SM, 80 GB      │
                    │    可被多个进程共享      │
                    └───────────────────────┘

                    A100-80GB (MIG Enabled)
                    ┌───────┬───────┬───────┐
                    │ 1g.10 │ 1g.10 │ 2g.20 │
                    │ 14 SM │ 14 SM │ 28 SM │
                    │ 9.5GB │ 9.5GB │19.5GB │
                    │隔离    │隔离    │隔离   │
                    └───────┴───────┴───────┘
```

**适用场景**：多租户推理服务、CI/CD 资源隔离、小模型并发推理。不适合大规模训练（需要整卡 SM 和显存）。

---

## 2. 查询 MIG 状态

### 2.1 确认 MIG 模式

```bash
nvidia-smi -i 7 --query-gpu=mig.mode.current,mig.mode.pending --format=csv
```

GPU 7 当前状态：

```text
mig.mode.current, mig.mode.pending
Enabled, Enabled
```

`Pending` 与 `Current` 一致说明 MIG 模式已稳定（无待处理的状态变更）。

### 2.2 列出可用的 GPU Instance Profile

```bash
nvidia-smi mig -i 7 --list-gpu-instance-profiles
```

GPU 7 的所有 profile（A100-80GB）：

```text
+-------------------------------------------------------------------------------+
| GPU instance profiles:                                                        |
| GPU   Name               ID    Instances   Memory     P2P    SM    DEC   ENC  |
|                                Free/Total   GiB              CE    JPEG  OFA  |
|===============================================================================|
|   7  MIG 1g.10gb         19     7/7        9.50       No     14     0     0   |
|                                                               1     0     0   |
|   7  MIG 1g.10gb+me      20     1/1        9.50       No     14     1     0   |
|                                                               1     1     1   |
|   7  MIG 1g.20gb         15     4/4        19.50      No     14     1     0   |
|                                                               1     0     0   |
|   7  MIG 2g.20gb         14     3/3        19.50      No     28     2     0   |
|                                                               2     0     0   |
|   7  MIG 3g.40gb          9     2/2        39.25      No     42     3     0   |
|                                                               3     0     0   |
|   7  MIG 4g.40gb          5     1/1        39.25      No     56     4     0   |
|                                                               4     0     0   |
|   7  MIG 7g.80gb          0     1/1        79.00      No     98     5     0   |
|                                                               7     1     1   |
+-------------------------------------------------------------------------------+
```

**字段解读**：

| 字段                   | 含义                                                          |
| ---------------------- | ------------------------------------------------------------- |
| `Name`                 | Profile 名称：N g.M GB — N 个 GPU 实例中的 SM 比例，M GB 显存 |
| `Instances Free/Total` | 还可以创建几个该 profile 的实例 / 最多几个                    |
| `P2P`                  | MIG 实例间不支持 P2P（全部为 No）                             |
| `SM`                   | 分配给该实例的 SM 数量（A100 总共 108 SM，MIG 可用 98）       |
| `CE`                   | Copy Engine 数量（1×H2D, 1×D2H）                              |

### 2.3 MIG 对 CUDA 设备枚举的影响

关键对比：

```bash
# 编译同一个 CUDA 查询程序
nvcc -arch=sm_80 -o check_device check_device.cu

# GPU 3 (MIG Disabled) → 正常
CUDA_VISIBLE_DEVICES=3 ./check_device
# 输出: CUDA device count: 1
#        Device 0: NVIDIA A100-SXM4-80GB (CC 8.0, SM 108, Mem 79 GB)

# GPU 7 (MIG Enabled, 无实例) → 错误！
CUDA_VISIBLE_DEVICES=7 ./check_device
# 输出: CUDA device count: <错误码>
```

**解读**：MIG Enabled 但没有任何 GI（GPU Instance）被创建时，该 GPU 对 CUDA 应用**完全不可见**。这是 MIG 与普通模式的本质区别——必须创建至少一个 GI，CUDA 才能"看到"这个 GPU。

---

## 3. 启用与禁用 MIG（命令参考，不执行）

> **重要**：启用/禁用 MIG 需要 GPU reset，会断开该 GPU 上的所有进程，且需要 root 权限。以下命令仅供参考，不在本文环境中执行。

### 3.1 启用 MIG

```bash
# 步骤 1：启用 MIG 模式（GPU 会重置）
nvidia-smi -i <GPU_ID> -mig 1

# 步骤 2：确认模式切换完成
nvidia-smi -i <GPU_ID> --query-gpu=mig.mode.current,mig.mode.pending --format=csv
# 期望：Enabled, Enabled

# 步骤 3：创建 GPU Instance (GI)
nvidia-smi mig -i <GPU_ID> -cgi 19,19,14  # 创建 2×1g.10gb + 1×2g.20gb
# 这里 19,14 是 profile ID (从 --list-gpu-instance-profiles 获取)

# 步骤 4：创建 Compute Instance (CI)
nvidia-smi mig -i <GPU_ID> -cci 0,0,0  # 为每个 GI 各创建 1 个 CI
```

**完整流程**：Enable MIG → Create GI → Create CI → CUDA 可见。GI 定义资源划分，CI 是可被 CUDA 程序使用的实例。

### 3.2 禁用 MIG

```bash
# 一次性销毁所有实例并关闭 MIG 模式
nvidia-smi -i <GPU_ID> -mig 0

# GPU 重置后恢复为完整设备
```

**MIG 禁用后**：之前创建的 GI/CI 全部被销毁，CUDA 重新看到完整的单 GPU 设备。GPU 在此期间需要数秒重置时间。

### 3.3 查询已创建的 MIG 实例

```bash
# 列出所有 GPU Instance (GI)
nvidia-smi mig -i <GPU_ID> --list-gpu-instances

# 列出所有 Compute Instance (CI)
nvidia-smi mig -i <GPU_ID> --list-compute-instances

# 销毁特定 CI
nvidia-smi mig -i <GPU_ID> -dci -ci 0,0

# 销毁特定 GI 及其 CI
nvidia-smi mig -i <GPU_ID> -dgi -gi 0
```

### 3.4 实例组合示例

以 GPU 7 的 profile 表为据，常见组合：

| 组合       | GI Profile                | 实例数 | 总显存  | 用途                   |
| ---------- | ------------------------- | ------ | ------- | ---------------------- |
| 最大化实例 | 1g.10gb × 7               | 7      | 66.5 GB | 7 个开发者各用一片 GPU |
| 混合       | 1g.10gb × 3 + 2g.20gb × 2 | 5      | 67.5 GB | 小模型 + 中等模型混部  |
| 均衡       | 3g.40gb × 2               | 2      | 78.5 GB | 两个独立推理服务       |
| 最大单实例 | 7g.80gb × 1               | 1      | 79 GB   | 接近完整 GPU（98 SM）  |

---

## 4. GPU 7 vs GPU 3 状态对比

| 属性              | GPU 3 (正常)              | GPU 7 (MIG Enabled)        |
| ----------------- | ------------------------- | -------------------------- |
| MIG Mode          | Disabled                  | Enabled                    |
| CUDA device count | **1**                     | **0**（无可用的 GI/CI）    |
| 可用 GI Profile   | 存在（需要先 Enable MIG） | 7 个 profile               |
| CUDA 程序可见     | 是                        | 否（需要创建至少 1 个 GI） |
| NVLink            | NV12 全互联               | 无（走 PCIe SYS）          |
| 当前显存使用      | 130 MiB                   | 0 MiB                      |

> **GPU 7 同时失去 NVLink 和 MIG Enabled 但无实例**——在 NCCL 拓扑中被标记为仅 PCIe 可达，且在 CUDA 中不可见。这是实验室环境中的典型配置：专门留给 MIG 实验而不影响训练 GPU。

---

## 5. MIG 与训练/推理的关系

| 场景               | 推荐配置                      | 原因                               |
| ------------------ | ----------------------------- | ---------------------------------- |
| 大规模训练 (TP/PP) | MIG Disabled + NVLink         | 需要整卡显存 + 跨卡高速通信        |
| 小模型推理         | MIG 1g.10gb × N               | 每实例 9.5 GB 足够 Llama-7B (INT8) |
| 中等模型推理       | MIG 2g.20gb 或 3g.40gb        | Llama-13B/70B 量化版本             |
| 并发推理服务       | MIG 多实例 + CUDA MPS（可选） | 硬件隔离 + 额外时间片共享          |
| 开发者环境         | MIG 1g.10gb × 7               | 多个用户互不干扰                   |

---

## 6. 相关文档

- [`07_a100_architecture.md`](understand_gpu_architecture/07_a100_architecture.md)：A100 MIG 能力的硬件基础
- [`nvlink_diagnostics.md`](../nvlink/nvlink_diagnostics.md)：GPU 7 的 NVLink 故障与 MIG 配置的关系

## 参考

- [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [nvidia-smi mig 命令参考](https://docs.nvidia.com/deploy/nvidia-smi/index.html#mig)
- [NVIDIA A100 架构白皮书](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) — MIG 章节
