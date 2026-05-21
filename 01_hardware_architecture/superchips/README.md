# AI Superchip 与机架级架构

Blackwell 代际把 AI 机器边界从"单机"推到"机架级单域"：节点规模从 8-GPU HGX 扩展到 72-GPU NVL72；封装内 NVLink-C2C 让 CPU/GPU 共享一致性内存。

## 文档

- [**NVLink-C2C 技术详解**](nvlink_c2c.md)：Chip-to-Chip 异构融合互连——如何打破 CPU↔GPU 间的内存墙，实现高带宽低延迟的一致性互连。
- [**NVIDIA GB300 NVL72 架构解析**](nvidia_gb300.md)：基于 Blackwell 架构的机架级计算系统，含 nvbandwidth 实测数据与物理拓扑参数。

## 关键概念

| 概念             | 含义                   | 影响                                            |
| ---------------- | ---------------------- | ----------------------------------------------- |
| NVLink-C2C       | 芯片到芯片的一致性互连 | CPU 和 GPU 共享统一物理内存，消除 PCIe DMA 开销 |
| NVL72            | 72 GPU 机架级单域      | 130 TB/s 聚合带宽，单机架=一台逻辑 GPU          |
| Copper Backplane | 铜缆背板               | 短距离比光纤低 6 倍功耗 + 更低延迟              |

## 参考

- [NVLink 技术入门](../nvlink/nvlink_intro.md) — NVLink 1.0→6.0 演进
- [PCIe 总线体系](../pcie/README.md) — 传统互连标准对比
