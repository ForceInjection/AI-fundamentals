# NVLink 诊断与实操

> 基于 A100-SXM4-80GB (8 GPU, NVSwitch Gen 2) 服务器实测。本文覆盖 `nvidia-smi nvlink` 的完整命令集，从链路状态、能力查询、拓扑解读到错误检测，并提供一个真实的 NVLink 故障案例（GPU 7 走 PCIe 而非 NVLink）。

---

## 1. 命令速查

```bash
nvidia-smi nvlink --status         # 每条 link 的当前带宽 (GB/s)
nvidia-smi nvlink --capabilities   # P2P / atomics / SLI 能力
nvidia-smi nvlink --errorcounters  # 链路错误计数器 (Replay/Recovery/CRC)
nvidia-smi nvlink --getthroughput d # tx/rx 数据量 (KiB)
nvidia-smi topo -m                 # GPU 间拓扑矩阵 (NV12/SYS/NODE/PXB)
nvidia-smi nvlink --resetcounters  # 清零计数器（调试前先复位）
```

> 消费级 GPU (GeForce RTX 系列) 不支持 NVLink，执行 `nvidia-smi nvlink` 会输出 `Device does not have or support Nvlink`。

---

## 2. 链路状态：`--status`

```bash
nvidia-smi nvlink --status
```

A100 正常输出（每条 link 25 GB/s，共 12 条）：

```text
GPU 0: NVIDIA A100-SXM4-80GB
   Link 0: 25 GB/s
   Link 1: 25 GB/s
   Link 2: 25 GB/s
   ...
   Link 11: 25 GB/s
```

**解读**：

- A100 使用 NVLink 3.0，12 条 link，每条单向 25 GB/s → 总双向带宽 = 12 × 25 × 2 = **600 GB/s**，与 `nvlink_intro.md` 文档中 2.2 节记录一致
- 所有 link 均为 25 GB/s 表示链路处于最高速率（active state）
- 若出现 `Inactive` 或更低速率 → 物理层问题或链路未协商到满速

**只查某个 GPU**：

```bash
nvidia-smi nvlink --status -i 3   # 只查 GPU 3
```

---

## 3. 链路能力：`--capabilities`

```bash
nvidia-smi nvlink --capabilities
```

A100 输出：

```text
GPU 0: NVIDIA A100-SXM4-80GB
   Link 0, P2P is supported: true
   Link 0, Access to system memory supported: true
   Link 0, P2P atomics supported: true
   Link 0, System memory atomics supported: true
   Link 0, SLI is supported: true
   ... (12 条 link 全部为 true)
```

| 能力                        | 含义                                     | 依赖场景                            |
| --------------------------- | ---------------------------------------- | ----------------------------------- |
| **P2P is supported**        | GPU 间可直接访问对方显存 (GPUDirect P2P) | NCCL All-Reduce、Tensor Parallelism |
| **Access to system memory** | 可通过 NVLink 访问 Host 内存             | Unified Memory、GPUDirect RDMA      |
| **P2P atomics**             | 支持对远端 GPU 显存的原子操作            | 分布式梯度累加                      |
| **System memory atomics**   | 支持对 Host 内存的原子操作               | GPUDirect + atomic                  |
| **SLI is supported**        | 支持 SLI 桥接（图形渲染用途）            | 专业可视化                          |

> 12 条 link 全部能力为 `true` 是正常状态。如果某条 link 的 `P2P is supported: false`，应立刻检查该 link 的错误计数器。

---

## 4. 拓扑矩阵解读：`nvidia-smi topo -m`

这是诊断多 GPU 系统最重要的命令——它直接告诉你 GPU 对之间走的是 NVLink 还是 PCIe：

```text
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0     X    NV12  NV12  NV12  NV12  NV12  NV12  NV12
GPU1    NV12   X    NV12  NV12  NV12  NV12  NV12  NV12
GPU2    NV12  NV12   X    NV12  NV12  NV12  NV12  NV12
GPU3    NV12  NV12  NV12   X    NV12  NV12  NV12  NV12
GPU4    NV12  NV12  NV12  NV12   X    NV12  NV12  NV12
GPU5    NV12  NV12  NV12  NV12  NV12   X    NV12  NV12
GPU6    NV12  NV12  NV12  NV12  NV12  NV12   X    NV12
GPU7    SYS   SYS   SYS   SYS   NODE  NODE  PXB    X
```

### 4.1 连接类型含义

| 标识     | 含义                                   | 带宽层级          | 延迟      |
| -------- | -------------------------------------- | ----------------- | --------- |
| **NV12** | NVLink 3.0 (Ampere), 12 links 全连     | 600 GB/s 双向     | ~μs       |
| **NODE** | 同一 NUMA node 内，经 PCIe Host Bridge | PCIe 带宽         | ~10-20 μs |
| **SYS**  | 跨 NUMA node，经 SMP (QPI/UPI)         | PCIe + CPU 间互联 | 更高      |
| **PXB**  | PCIe Host Bridge 直连                  | PCIe 带宽         | ~10 μs    |

### 4.2 GPU 7 异常案例分析

GPU 7 与其他 GPU 的连接是 **SYS / NODE / PXB** 而非 NV12。这意味着 GPU 7 的 NVLink 未正常工作，所有 GPU 间通信走 PCIe + CPU 中转：

```text
GPU7 → GPU0-3: SYS  (跨 NUMA node，经 QPI/UPI → 延迟最高)
GPU7 → GPU4-5: NODE (同 NUMA node 1，经 PCIe Host Bridge)
GPU7 → GPU6:   PXB  (同一 PCIe Host Bridge 下)
```

**排查步骤**：

```bash
# 1. 确认 NVLink 状态
nvidia-smi nvlink --status -i 7      # 预期：输出为空（所有 link inactive）

# 2. 检查 NVLink 能力
nvidia-smi nvlink --capabilities -i 7 # 预期：无输出或全部 false

# 3. 检查物理层错误
nvidia-smi nvlink --errorcounters -i 7

# 4. 对比正常 GPU
nvidia-smi nvlink --status -i 0      # 预期：12 links × 25 GB/s
```

**常见原因**：

| 原因                        | 现象                     | 解决                |
| --------------------------- | ------------------------ | ------------------- |
| NVSwitch 端口故障           | 仅个别 GPU 断开 NVLink   | 替换 NVSwitch tray  |
| GPU 基板连接松动            | 拓扑中该 GPU 全部走 SYS  | 重新插拔 GPU tray   |
| NVLink 桥接器 (bridge) 损坏 | 物理层错误计数器持续增长 | 检查物理连接        |
| 固件/驱动版本不匹配         | 所有 link 显示 inactive  | 更新驱动或 GPU 固件 |

> 在本环境中，GPU 7 是 ECC enabled + MIG disabled 状态，且 `nvidia-smi nvlink --status -i 7` 返回空——确认是一个活跃的 NVLink 故障。但 GPU 7 仍通过 PCIe 可用（只是 P2P 带宽会大幅下降）。NCCL 在检测到拓扑后会走 PCIe fallback 路径，不会崩溃但性能会显著下降。

---

## 5. 链路错误检测：`--errorcounters`

NVLink 物理层使用 CRC 校验和 replay 机制保证数据完整性。错误计数器持续增长 → 物理层信号质量问题：

```bash
nvidia-smi nvlink --errorcounters
```

正常输出（健康链路）：

```text
GPU 0:
   Link 0: Replay Errors: 0
   Link 0: Recovery Errors: 0
   Link 0: CRC Errors: 0
   ... (12 条 link 全部为 0)
```

**三类错误**：

| 错误类型            | 含义             | 严重程度           |
| ------------------- | ---------------- | ------------------ |
| **CRC Errors**      | 数据校验失败     | 中——物理层比特错误 |
| **Replay Errors**   | 需要重传数据包   | 高——链路质量下降   |
| **Recovery Errors** | 链路需要重新训练 | 严重——物理层不稳定 |

**诊断流程**：

```bash
# 1. 清零计数器（开始测试前）
nvidia-smi nvlink --reseterrorcounters -i 0

# 2. 施加负载（运行 NCCL all-reduce 或 bandwidth test）
# ... 运行你的 GPU workload ...

# 3. 再次检查
nvidia-smi nvlink --errorcounters -i 0
# 如果 Replay > 100 / minute → 链路质量有问题
# 如果 Recovery > 0 → 硬件可能需要更换
```

---

## 6. GPU 0-6 健康 vs GPU 7 异常：全量对比

| 检查项                   | GPU 0-6 (正常)     | GPU 7 (异常) |
| ------------------------ | ------------------ | ------------ |
| `nvlink --status`        | 12 links × 25 GB/s | 无输出       |
| `nvlink --capabilities`  | 全部 true          | 无输出       |
| `topo -m` 连接类型       | NV12 全互联        | SYS/NODE/PXB |
| `nvlink --errorcounters` | 全部 0             | N/A          |

**生产环境建议**：部署时通过脚本检查所有 GPU 的 `nvidia-smi nvlink --status`，如果某 GPU 的输出为空，立即告警——这比依赖 NCCL 报错快得多（NCCL 可能在 NVLink 故障时自动回退到 PCIe，不会报错但训练吞吐会腰斩）。

---

## 7. 相关文档

- [`nvlink_intro.md`](nvlink_intro.md)：NVLink 1.0-6.0 版本演进、NVSwitch、SHARP 理论——本文是其诊断实操配套
- [`02_pcie_bandwidth_measurement.md`](../../02_gpu_programming/04_profiling/02_pcie_bandwidth_measurement.md)：PCIe 带宽测试，可与 NVLink 带宽形成对比
- [`06_device_attributes.md`](../../02_gpu_programming/02_cuda/06_device_attributes.md)：`cudaDeviceGetAttribute` 查 P2P 原子操作能力

---

## 参考

- [NVIDIA nvidia-smi NVLink 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html#nvlink)
- [NVIDIA NVLink & NVSwitch](https://www.nvidia.com/en-us/data-center/nvlink/)
- [NCCL 拓扑检测](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-topology-detection)
