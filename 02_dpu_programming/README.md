# DPU 编程

在典型的服务器里，网络转发、存储 I/O、加密解密这些“基础设施活”其实一直在不断蚕食主机 CPU 的算力。**数据处理单元（DPU）** 的出现，就是把这部分工作正式从主机卸载出来，交给一块专门的芯片去处理。

本章会围绕 **NVIDIA BlueField DPU 和其 DOCA 软件框架** 展开，除了介绍硬件本身，更重要的是解释开发者该怎么向它编程：

- **硬件角色**：把网络、存储、安全等基础设施工作从主机 CPU 卸载到 DPU 上完成，把主机算力留给用户业务。
- **编程抽象**：通过 DOCA SDK 统一访问 DPU 上的 ARM 核、硬件加速引擎（如 RegEx、SHA、Compress），以及 OVS、RDMA、NVMe-oF 等网络子系统，避免直接面对底层硬件细节。

## 1. 核心编程框架

### 1.1 DOCA 框架

**DOCA（Data Center-on-a-Chip Architecture）** 是 NVIDIA 提供的官方软件栈，也是解锁 BlueField DPU 潜力的主要入口，包含驱动、运行时、各种库和参考应用。

从使用场景看，DOCA 主要解决三类卸载需求：

- **网络卸载**：OVS、VXLAN、RoCE、时间同步等数据平面功能从 CPU 卸载到 DPU。
- **安全卸载**：IPsec、TLS、深包检测等安全能力在 DPU 上执行，不再占用主机资源。
- **存储卸载**：NVMe-oF、加密/压缩、存储初始化等存储路径上的热点负载。

- **编程指南**：[DOCA 编程入门](doca/01_doca_programming_guide.md) - 涵盖架构简介、核心组件、环境搭建及零拷贝传输、控制卸载等典型场景的编程实践。

---

## 2. 学习资源

- [NVIDIA DOCA SDK 文档](https://docs.nvidia.com/doca/sdk/index.html)
- [NVIDIA BlueField DPU 架构](https://www.nvidia.com/en-us/networking/products/data-processing-unit/)
