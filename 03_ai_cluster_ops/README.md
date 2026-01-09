# AI 集群运维与通信 (AI Cluster Operations & Communication)

本章节专注于 AI 基础设施的运维管理、网络通信与性能监控，构建稳定高效的 AI 算力集群。

## 内容概览

### 1. [GPU 基础运维](gpu_ops/README.md)

GPU 设备的基础监控与状态查询工具。

- **设备查询**：Device Query
- **状态监控**：nvidia-smi, nvtop
- **误区解读**：GPU 利用率指标分析

### 2. [InfiniBand 高性能网络](infiniband/README.md)

InfiniBand 网络技术的理论与实践。

- **理论基础**：IB 网络架构与协议
- **健康检查**：网络连通性与状态监测
- **性能监控**：带宽与延迟测试

### 3. [NCCL 分布式通信](nccl/README.md)

NVIDIA 分布式通信库 (NCCL) 的测试与部署。

- **基准测试**：NCCL Benchmark
- **多节点部署**：K8s 与原生环境部署
- **性能优化**：PXN 模式与网络调优
