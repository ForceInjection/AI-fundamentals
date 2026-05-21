# NVLink 与 NVSwitch 高速互连

NVLink 是 NVIDIA 专为 GPU 间通信设计的私有高速链路，NVSwitch 将其扩展为交换式网络。本目录覆盖从理论演进到诊断实操的完整内容。

## 文档

- [**NVLink 技术入门**](nvlink_intro.md)：NVLink 1.0→6.0 版本演进、物理层与协议层原理、NVSwitch/SHARP/NVL72 架构、与 PCIe 协议对比、应用场景决策。
- [**NVLink 诊断与实操**](nvlink_diagnostics.md)：`nvidia-smi nvlink` 完整命令集（status/capabilities/errorcounters/topo）、A100 实测验证（12 links × 25 GB/s = 600 GB/s）、GPU 7 NVLink 故障案例排查。

## 关键数字速查

| GPU  | NVLink 版本 | Links | 单链路单向 | 总双向带宽 |
| ---- | ----------- | ----- | ---------- | ---------- |
| V100 | 2.0         | 6     | 25 GB/s    | 300 GB/s   |
| A100 | 3.0         | 12    | 25 GB/s    | 600 GB/s   |
| H100 | 4.0         | 18    | 25 GB/s    | 900 GB/s   |
| B200 | 5.0         | 18    | 50 GB/s    | 1.8 TB/s   |

> 消费级 GeForce RTX 系列不支持 NVLink。

## 参考

- [PCIe 总线体系](../pcie/README.md) — 通用互连标准
- [AI Superchip 与机架级架构](../superchips/nvidia_gb300.md) — NVLink-C2C 与 NVL72
