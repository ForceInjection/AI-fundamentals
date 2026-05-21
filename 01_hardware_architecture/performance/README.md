# 性能评估与拓扑分析

从芯片延迟到机架带宽的性能指标参考体系。覆盖系统延迟金字塔、NUMA 拓扑、PCIe/NVLink 带宽速查、CPU 矩阵加速对比。

## 文档

- [**HBM 显存技术演进**](01_hbm_evolution.md)：从 HBM2 到 HBM3e 的技术演进，A100 HBM2e vs RTX 5090 GDDR7 实测对比，3D 封装原理与 L2 Cache 加速效应。
- [**单卡 GPU 拓扑与 NUMA 深入分析**](02_single_gpu_topology_analysis.md)：`nvidia-smi topo -m` 输出解读，NUMA 亲和性验证，跨 socket 延迟分析（含 taskset 实测）。
- [**CPU AMX vs GPU Tensor Core**](03_amx_vs_tensorcore.md)：Intel AMX 与 NVIDIA Tensor Core 的硬件规格对比与小 batch 推理场景分析。
- [**多 PCIe Domain 与 NUMA 映射**](04_pcie_domain_numa.md)：Sapphire Rapids 多 domain 架构，BDF 编码的 NUMA 推断。
- [**PCIe & NVLink 带宽速查表**](05_pcie_nvlink_speed_reference.md)：PCIe/NVLink 各代带宽、主流 GPU 互连规格、NVMe SSD 速度。
- [**AI 基础设施延迟金字塔**](ai_latency_pyramid.md)：寄存器→HBM→跨节点 RDMA→NVMe 存储的六级延迟基准。

## 关键参考数字

| 层级        | 延迟量级 | 相对寄存器 |
| ----------- | -------- | ---------- |
| 寄存器 / L1 | ~1 ns    | 1×         |
| HBM         | ~100 ns  | ~100×      |
| NVLink P2P  | ~2 μs    | ~2,000×    |
| 跨节点 RDMA | ~2 μs    | ~2,000×    |
| NVMe 存储   | ~100 μs  | ~100,000×  |

## 参考

- [NVLink 技术入门](../nvlink/nvlink_intro.md) — PCIe & NVLink 带宽的理论基础
- [HBM 显存带宽测试](../../02_gpu_programming/04_profiling/03_hbm_bandwidth_test.md) — 片内带宽实测
