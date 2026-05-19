# PCIe AER 错误监控与链路健康检查

> 基于 RTX 5090 实测——本环境所有 AER 计数器为 0，PCIe Replay = 0，链路信号质量完美。本文教读者如何通过 sysfs 和 nvidia-smi 监控 PCIe 链路健康状态。

---

## 1. 什么是 PCIe AER

**AER (Advanced Error Reporting)** 是 PCIe 规范定义的标准化错误报告机制。每条 PCIe 链路有两组计数器：

| 类别              | 说明                                | 影响                                   |
| ----------------- | ----------------------------------- | -------------------------------------- |
| **Correctable**   | 硬件自动纠正的错误（如单 bit 翻转） | 不影响数据正确性，但降低有效带宽       |
| **Uncorrectable** | 无法纠正的错误                      | 分 Fatal（致命）和 Non-Fatal（可恢复） |

---

## 2. sysfs 读取 AER 计数器

### 2.1 文件路径

```bash
GPU_BDF="0000:98:00.0"  # 从 nvidia-smi 获取

# 可纠正错误
cat /sys/bus/pci/devices/$GPU_BDF/aer_dev_correctable

# 致命错误
cat /sys/bus/pci/devices/$GPU_BDF/aer_dev_fatal

# 非致命错误
cat /sys/bus/pci/devices/$GPU_BDF/aer_dev_nonfatal
```

### 2.2 本环境实测

```bash
# 可纠正错误 — 全部为 0
cat /sys/bus/pci/devices/0000:98:00.0/aer_dev_correctable
```

```text
RxErr 0            # 接收错误
BadTLP 0           # 损坏的 TLP (Transaction Layer Packet)
BadDLLP 0          # 损坏的 DLLP (Data Link Layer Packet)
Rollover 0         # 计数器翻转（长时间运行后正常）
Timeout 0          # 超时
NonFatalErr 0      # 非致命错误
CorrIntErr 0       # 内部可纠正错误
HeaderOF 0         # Header 日志溢出
TOTAL_ERR_COR 0    # 可纠正错误总计：0
```

```bash
# 致命错误 — 全部为 0
cat /sys/bus/pci/devices/0000:98:00.0/aer_dev_fatal
```

```text
Undefined 0        # 未定义错误
DLP 0              # Data Link Protocol 错误
SDES 0             # Surprise Down 错误
TLP 0              # TLP 致命错误（如 Poisoned TLP）
FCP 0              # Flow Control Protocol 错误
CmpltTO 0          # Completion Timeout
CmpltAbrt 0        # Completer Abort
UnxCmplt 0         # Unexpected Completion
RxOF 0             # Receiver Overflow
MalfTLP 0          # Malformed TLP
ECRC 0             # ECRC 校验失败
UnsupReq 0         # Unsupported Request
ACSViol 0          # ACS 违反
UncorrIntErr 0     # 内部无法纠正错误
BlockedTLP 0       # 被阻止的 TLP
AtomicOpBlocked 0  # 被阻止的 Atomic 操作
TOTAL_ERR_FATAL 0  # 致命错误总计：0
```

---

## 3. nvidia-smi 侧 PCIe 指标

### 3.1 Replay 计数器

```bash
nvidia-smi -q -i 0 | grep -A3 "Replay"
```

```text
Replays Since Reset     : 0
Replay Number Rollovers : 0
```

- **Replay**：PCIe 链路层检测到错误后重传 TLP 的次数。增长意味着信号质量问题。
- **Rollover**：Replay 计数器溢出次数。非零说明链路曾经历大量重传。

### 3.2 实时吞吐

```bash
nvidia-smi -q -i 0 | grep -A2 "Throughput"
```

```text
Tx Throughput : 714 KB/s   (当前上行)
Rx Throughput : 525 KB/s   (当前下行)
```

空闲时吞吐量很低（KB/s 级别），负载下可达 GB/s 级别。

---

## 4. 链路健康诊断流程

```text
nvidia-smi -q → Replays > 0？
  ├── YES → 检查 AER 计数器 → 对应章节排查
  └── NO  → 链路健康 ✅

sysfs aer_dev_* → 非零计数？
  ├── Correctable 增长
  │     → 通常信号质量下降：检查线缆、插槽、PCIe 延长线
  │     → 如果 Rollover 频繁 → 需要降低 PCIe Gen
  ├── Fatal/Non-Fatal
  │     → 严重问题：检查驱动、固件、GPU 硬件状态
  │     → UnsupReq → 驱动/Firmware 兼容性问题
  │     → CmpltTO → GPU 无响应，检查供电和散热
  └── ALL 0 → 链路完美
```

### 4.1 常见错误场景

| 错误类型      | 计数器                          | 可能原因                        | 排查                    |
| ------------- | ------------------------------- | ------------------------------- | ----------------------- |
| 信号劣化      | `BadDLLP`, `RxErr`, Replay 增长 | PCIe 延长线质量差、插槽接触不良 | 换槽/降 Gen             |
| 链路不稳定    | `BadTLP`, Replay 频繁           | 时钟偏差、EMI 干扰              | 检查 BIOS PCIe 展频设置 |
| 驱动/FW BUG   | `UnsupReq`                      | GPU 驱动与内核不匹配            | 更新驱动                |
| 供电不足      | `CmpltTO`                       | GPU 瞬时功耗过高                | 检查电源/功耗限制       |
| ACS 配置错误  | `ACSViol`                       | IOMMU/ACS 配置错误              | 检查 BIOS VT-d/ACS 设置 |
| Atomic 被阻止 | `AtomicOpBlocked`               | PCIe switch 不支持 atomic       | 检查 PCIe topology      |

### 4.2 定期监控脚本

```bash
#!/bin/bash
# 每分钟记录一次 PCIe 链路健康状态
GPU_BDF="0000:98:00.0"
while true; do
    echo "$(date -Iseconds) | Replay: $(nvidia-smi -q -i 0 | grep 'Replays Since Reset' | awk '{print $NF}') | Corr: $(cat /sys/bus/pci/devices/$GPU_BDF/aer_dev_correctable | grep TOTAL | awk '{print $NF}') | Fatal: $(cat /sys/bus/pci/devices/$GPU_BDF/aer_dev_fatal | grep TOTAL | awk '{print $NF}')"
    sleep 60
done
```

---

## 5. 其他 PCIe 健康指标

### 5.1 PCIe 链路协商

```bash
# 当前速度 vs 最大能力
cat /sys/bus/pci/devices/0000:98:00.0/current_link_speed
cat /sys/bus/pci/devices/0000:98:00.0/max_link_speed
```

如果 `current` 持续低于 `max` 且 GPU 有负载，说明链路训练失败降级。

### 5.2 ASPM 状态

```bash
cat /sys/bus/pci/devices/0000:98:00.0/power/runtime_status
# active / suspended
```

GPU 的 `runtime_status` 通常为 `active`，`runtime_enabled` 通常为 `forbidden`（GPU 不允许 PCIe 休眠）。

---

## 6. 本环境结论

| 指标                     | 值                 | 判定        |
| ------------------------ | ------------------ | ----------- |
| AER Correctable Total    | 0                  | ✅ 完美     |
| AER Fatal Total          | 0                  | ✅ 完美     |
| AER Non-Fatal Total      | 0                  | ✅ 完美     |
| Replays Since Reset      | 0                  | ✅ 完美     |
| 当前 PCIe Gen / Width    | Gen 5 x16 (负载下) | ✅ 全速     |
| RxErr / BadTLP / BadDLLP | 0 / 0 / 0          | ✅ 信号完美 |

链路质量理想——无错误、无重传、全速协商。

---

## 参考

- [PCIe 技术大全](01_pcie_comprehensive_guide.md)
- [PCIe 拓扑可视化](03_pcie_topology_visualization.md)
- [PCI Express Base Specification: AER](https://pcisig.com/specifications)
