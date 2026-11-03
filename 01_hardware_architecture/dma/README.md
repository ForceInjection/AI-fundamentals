# DMA engine

这一目录讨论设备侧的数据搬运执行单元：descriptor 和 doorbell 怎么启动一次传输，engine 怎么读写内存并报告完成，以及 GPU copy engine、NIC、NVMe、DPU 的 DMA 路径有什么不同。

它不是 CUDA `memcpy` API 教程，也不重复 [GPU 访问系统内存](../../02_gpu_programming/02_cuda/19_gpu_system_memory_access.md) 的地址翻译和 [GPUDirect](../gpudirect/README.md) 的协议细节。重点是把“谁在搬数据”讲清楚。

## 文章

- [DMA engine：从描述符到完成，一次搬运发生了什么](01_dma_engine.md)：从一次传输的生命周期出发，比较 GPU copy engine、kernel load/store、GPUDirect RDMA、GDS、DOCA DMA 和 CXL 共享内存的路径与边界。

## 相关目录

- [PCIe / P2PDMA](../pcie/02_p2pdma_technology.md)：PCIe 拓扑、P2PDMA、ACS 与 IOMMU。
- [GPUDirect](../gpudirect/README.md)：GPU 与 NIC、NVMe、其他 GPU 的直通路径。
- [GPU 访问系统内存](../../02_gpu_programming/02_cuda/19_gpu_system_memory_access.md)：MMU、IOMMU、MMIO、mmap、ioremap 的地址层。
- [GPU 显存互访](../../02_gpu_programming/02_cuda/20_gpu_peer_memory_access.md)：P2P、cudaIpc、VMM Fabric 与跨设备同步。
