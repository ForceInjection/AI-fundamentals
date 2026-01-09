# InfiniBand 高性能网络技术

## 1. 概述

InfiniBand 是一种高性能、低延迟的网络互连技术，广泛应用于高性能计算（HPC）和 AI 训练集群中。本目录包含 InfiniBand 技术的理论基础、健康检查工具和性能监控脚本。

## 2. 核心技术特性

- **超低延迟**：亚微秒级延迟（<1μs）
- **高带宽**：支持 100Gbps、200Gbps、400Gbps 等规格
- **RDMA 支持**：远程直接内存访问，绕过 CPU 和操作系统
- **硬件卸载**：网络协议栈硬件加速
- **可靠传输**：内置错误检测和恢复机制

## 3. 目录内容

### 3.1 理论文档

- [IB 网络理论与实践](IB%20网络理论与实践.md) - InfiniBand 网络技术的理论基础和实践应用

### 3.2 健康检查工具

- [health/](health/) - InfiniBand 网络健康检查工具集
  - 网络连通性检测
  - 设备状态监控
  - 性能指标检查
  - 自动化健康检查脚本

### 3.3 监控工具

- [monitor/](monitor/) - InfiniBand 网络监控工具集
  - 带宽监控脚本
  - 性能数据收集
  - 实时监控仪表板
  - 监控工具测试套件

## 4. 快速开始

### 4.1 健康检查

```bash
# 运行 InfiniBand 健康检查
cd health/
./ib_health_check.sh
```

### 4.2 性能监控

```bash
# 启动带宽监控
cd monitor/
./ib_bandwidth_monitor.sh
```

### 4.3 运行测试

```bash
# 运行集成测试
cd monitor/
./run_tests.sh
```

## 5. 参考资源

### 5.1 官方文档

- [InfiniBand Architecture Specification](https://www.infinibandta.org/)
- [NVIDIA Networking Documentation](https://docs.nvidia.com/networking/)

### 5.2 开源项目

- [RDMA Core](https://github.com/linux-rdma/rdma-core)
- [OpenSM](https://github.com/linux-rdma/opensm)
- [Perftest](https://github.com/linux-rdma/perftest)
