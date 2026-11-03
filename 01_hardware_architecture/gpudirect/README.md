# GPUDirect 家族

GPUDirect 让设备间直接 DMA，把 CPU Bounce Buffer 从数据路径上移除，省带宽也省延迟。三个分支覆盖三种典型场景。

## 文档

| 技术          | 场景           | 数据路径                    | 文档                                                                 |
| ------------- | -------------- | --------------------------- | -------------------------------------------------------------------- |
| P2P           | 同节点 GPU↔GPU | GPU ↔ GPU（经 PCIe/NVLink） | [**GPUDirect P2P 技术详解**](02_gpudirect_p2p.md)                    |
| RDMA          | 跨节点 GPU↔NIC | NIC → 远端 GPU VRAM         | [**GPUDirect RDMA 与 Storage 技术详解**](01_gpudirect_technology.md) |
| Storage (GDS) | GPU↔NVMe 存储  | NVMe → GPU VRAM（绕过 CPU） | [**GPU Direct Storage 基础**](03_gds_basics.md)                      |
| RDMA 实战     | 跨节点验证     | H100 → IB HDR → H100        | [**GPUDirect RDMA 跨节点验证**](04_gpudirect_rdma_verification.md)   |

建议按 **P2P → RDMA/GDS → 实战** 的顺序读：先建立节点内直连的概念，再看跨节点与存储的延伸，最后照验证文章上手。

## 关键问题速查

| 问题                            | 文档                                                                                   |
| ------------------------------- | -------------------------------------------------------------------------------------- |
| 如何检查两台 GPU 是否支持 P2P？ | [P2P](02_gpudirect_p2p.md) — `cudaDeviceCanAccessPeer` 和拓扑检查                      |
| GPUDirect RDMA 需要哪些条件？   | [RDMA](01_gpudirect_technology.md) — 同 root complex + IOMMU 关闭 + BAR1 映射 + NIC 支持 |
| 如何验证 RDMA 跨节点可用？      | [RDMA 实战](04_gpudirect_rdma_verification.md) — NCCL all_reduce 端到端验证 + 踩坑     |
| GDS 怎么上手？                  | [GDS 基础](03_gds_basics.md) — RTX 5090 + CUDA 12.8 环境检查与 cuFile 流程             |

## 参考

- [DMA engine](../dma/01_dma_engine.md) — 搬运本身的模型：描述符、地址映射、完成语义
- [PCIe 总线体系](../pcie/README.md) — GPUDirect 依赖的底层 PCIe 能力（P2PDMA、ACS、BAR1）
- [NVLink 技术入门](../nvlink/nvlink_intro.md) — P2P 在 NVLink 上的高性能物理载体
