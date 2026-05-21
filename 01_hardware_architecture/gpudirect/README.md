# GPUDirect 家族

GPUDirect 让设备间直接 DMA，把 CPU Bounce Buffer 从数据路径上移除——既省带宽又省延迟。三个分支覆盖三种典型场景。

## 文档

| 技术          | 场景           | 数据路径                    | 文档                                                                 |
| ------------- | -------------- | --------------------------- | -------------------------------------------------------------------- |
| RDMA          | 跨节点 GPU↔NIC | NIC → 远端 GPU VRAM         | [**GPUDirect RDMA 与 Storage 技术详解**](01_gpudirect_technology.md) |
| P2P           | 同节点 GPU↔GPU | GPU ↔ GPU（经 PCIe/NVLink） | [**GPUDirect P2P 技术详解**](02_gpudirect_p2p.md)                    |
| Storage (GDS) | GPU↔NVMe 存储  | NVMe → GPU VRAM（绕过 CPU） | [**GPU Direct Storage 基础**](03_gds_basics.md)                      |

## 关键问题速查

| 问题                            | 文档                                                              |
| ------------------------------- | ----------------------------------------------------------------- |
| 如何检查两台 GPU 是否支持 P2P？ | [P2P](02_gpudirect_p2p.md) — `cudaDeviceCanAccessPeer` 和拓扑检查 |
| GPUDirect RDMA 需要哪些条件？   | [RDMA](01_gpudirect_technology.md) — BAR1 映射 + NIC 支持         |
| GDS 比传统路径快多少？          | [GDS](03_gds_basics.md) — 含 RTX 5090 + 3×NVMe 实测对比           |

## 参考

- [PCIe 总线体系](../pcie/README.md) — GPUDirect 依赖的底层 PCIe 能力（P2PDMA、ACS、BAR1）
- [NVLink 技术入门](../nvlink/nvlink_intro.md) — P2P 在 NVLink 上的高性能物理载体
