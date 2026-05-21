# PCIe 总线体系

PCIe 是异构计算系统中最基础的互连标准——CPU↔GPU、GPU↔NIC、GPU↔NVMe 都走它。本目录从协议原理、拓扑可视化到生产运维全链路覆盖。

## 协议与基础

- [**PCIe 总线技术大全**](01_pcie_comprehensive_guide.md)：物理层到协议层全面解析，带宽演进（Gen3→Gen6），TLP/DLLP 包结构。
- [**Linux PCIe P2PDMA 技术**](02_p2pdma_technology.md)：设备直连 DMA 的硬件机制与内核实现，GDS 场景下的实践。
- [**GPU BAR1 内存映射**](05_bar1_memory_mapping.md)：BAR1 窗口对 Unified Memory 性能的影响，ReBAR 状态检查。

## 拓扑与可视化

- [**PCIe 拓扑层次**](06_pcie_topology_hierarchy.md)：Root Complex → Bridge/Switch → Device 四层模型，sysfs 识别方法，本环境 24 domain 完整拓扑。
- [**PCIe 拓扑可视化**](03_pcie_topology_visualization.md)：`nvidia-smi` + sysfs 交叉验证 GPU 在 PCIe 树中的位置。
- [**PCIe Switch 识别与验证**](07_pcie_switch_vs_bridge.md)：区分 Switch vs Bridge，多端口检测，ACS 验证。

## 运维与诊断

- [**PCIe AER 错误监控**](04_pcie_aer_monitoring.md)：sysfs AER 计数器解读，Replay 监控，链路健康诊断流程。

## 参考

- [NVLink 技术入门](../nvlink/nvlink_intro.md) — GPU 私有高速互连（PCIe 的高性能替代）
- [GPUDirect P2P 技术详解](../gpudirect/02_gpudirect_p2p.md) — PCIe 之上的 P2P 通信
