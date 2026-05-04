# InfiniBand 高性能网络技术

## 1. 概述

如果让单卡的 GPU 计算一路扩大到上百张卡的集群，网络近乎是唯一一个会从“能用”一路变成“满足不了需求”的东西。InfiniBand 之所以在 HPC 和 AI 集群里是默认答案，是因为它把 **亚微秒延迟、数百 Gbps 带宽、无损传输** 三件事在硬件层同时解决了。

这个目录把 IB 运维相关的三块内容放在一起：理论基础、健康检查工具、性能监控脚本，方便从概念到实战连起来看。

## 2. 核心技术特性

IB 值得看的几个核心特性其实互相关联：**用 RDMA 绕过 CPU**，才能把延迟压到亚微秒；**用硬件卸载协议栈**，才能撑起 400 Gbps 的带宽；**用基于信用的流控和错误恢复**，才能做到真正的无损传输。

- **超低延迟**：亚微秒级延迟（<1μs）
- **高带宽**：支持 100Gbps、200Gbps、400Gbps 等规格
- **RDMA 支持**：远程直接内存访问，绕过 CPU 和操作系统
- **硬件卸载**：网络协议栈硬件加速
- **可靠传输**：内置错误检测和恢复机制

## 3. 目录内容

### 3.1 理论文档

要看懂 IB 的监控信息和诊断输出，需要先向下一层走一步，理解 RDMA、Verbs、Subnet Manager 这些核心概念是怎么回事。

- [IB 网络理论与实践](01_ib_network_theory.md) - InfiniBand 网络技术的理论基础和实践应用

### 3.2 健康检查工具

刚接收一个 IB 集群、或者训练任务出了莫名其妙的性能问题时，先跑一轮健康检查远比猜疑代码来得有效。

- [health/](health/) - InfiniBand 网络健康检查工具集
  - 网络连通性检测
  - 设备状态监控
  - 性能指标检查
  - 自动化健康检查脚本

### 3.3 监控工具

与健康检查的“快照”不同，监控工具更关心长时间带宽、错误计数、拥塞情况的趋势，它们是判断“网络是不是被逐渐拖慢”的主要依据。

- [monitor/](monitor/) - InfiniBand 网络监控工具集
  - 带宽监控脚本
  - 性能数据收集
  - 实时监控仪表盘
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
