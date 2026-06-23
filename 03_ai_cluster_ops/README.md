# AI 集群运维与通信 (AI Cluster Operations & Communication)

## 1. 概述

单卡能跑起来的模型和一个几百卡的集群能稳定跑起来的模型，完全是两件事。这一章的目标就是回答一个很实际的问题：**当 GPU 数量从 1 张变成几十上百张之后，我们要看什么、要调什么、要防什么？**

大致分成三条主线：

- **看得见**：GPU 本身的状态、温度、利用率得有工具能实时观察，出问题才不会两眼一抹黑。
- **连得通**：多机之间要靠 InfiniBand / RoCE 这种高性能网络撑起集合通信的带宽与延迟，否则再强的算力也会被网络吃掉。
- **通得快**：NCCL 把 AllReduce、AllGather 这些集合操作做成标准抽象，真正决定分布式训练扩展效率的往往是它的调优。

把这三条线串起来，基本就是 AI 集群日常运维的主干。

---

## 2. [GPU 基础运维](01_gpu_ops/README.md)

集群里每一张卡的健康情况都得看得见。这一节按场景组织：查设备能力 → 判断利用率 → 日常监控 → 健康检查 → 进程管理 → 驱动故障。

一个常被忽视的点是：**`nvidia-smi` 里的 GPU-Util 并不等于 SM 真的在忙**——它只说明某段时间有 Kernel 在执行，并不代表算力被用满了。三行命令的正确判断方式见 [GPU 忙不忙怎么判断](01_gpu_ops/09_gpu_busy_check.md)。

- **设备能力**：[GPU 设备属性查询](01_gpu_ops/01_device_query.md) | [GPU 忙不忙怎么判断](01_gpu_ops/09_gpu_busy_check.md) ——前者查 Kernel 设计参数，后者三行命令判断 GPU 真忙假忙。
- **利用率**：[GPU 利用率是一个误导性指标](01_gpu_ops/02_gpu_utilization_myth.md)（原文翻译）——为什么 GPU-Util ≠ 算力利用率。
- **日常监控**：[nvidia-smi 场景速查](01_gpu_ops/03_nvidia_smi_guide.md) | [nvtop 监控工具](01_gpu_ops/04_nvtop_guide.md) ——前者按场景组织命令，后者提供交互式 TUI。
- **长期趋势**：[DCGM 监控实操](01_gpu_ops/05_dcgm_monitoring.md) ——NVIDIA 官方方案，含 SM Active、DRAM Active、Prometheus 集成。
- **故障排查**：[GPU 集群健康检查](01_gpu_ops/06_gpu_health_check.md) | [GPU 驱动故障速查](01_gpu_ops/08_gpu_driver_troubleshooting.md) ——前者 L1/L2/L3 三层检查，后者 nvidia-smi 不可用时的排查路径。
- **资源管理**：[GPU 进程与资源管理](01_gpu_ops/07_gpu_process_management.md)——Compute Mode、MPS、CUDA_VISIBLE_DEVICES、NUMA 亲和性。

---

## 3. [InfiniBand 高性能网络](02_infiniband/README.md)

当训练规模迈入多机多卡，网络就从“能通”变成了“决定吞吐”。InfiniBand 之所以在 AI 集群里占据主流，根本原因是它把 **超低延迟（μs 级）、高带宽（400 Gbps 起步）和无损传输** 同时做到位。

运维视角看 IB，主要关心三件事：

- **理论基础**：[IB 网络架构与协议](02_infiniband/01_ib_network_theory.md)——理解 RDMA、Verbs、SubnetManager 等核心概念，才能看懂诊断信息。
- **健康检查**：链路状态、端口错误计数、子网管理器状态，这些是排查通信异常时的第一手线索。
- **性能监控**：实时带宽、丢包、拥塞指标能够反映集合通信是否被网络拖慢。

一句话：**IB 出问题的时候，训练任务的表现往往是“莫名变慢”而不是“直接报错”**，所以监控比修复更重要。

---

## 4. [NCCL 分布式通信测试](03_nccl/README.md)

NCCL 是几乎所有主流训练框架（PyTorch DDP、Megatron、DeepSpeed、vLLM）背后实际在跑的集合通信库。它把 AllReduce、AllGather、Broadcast 这些操作按照 GPU 之间的物理拓扑自动选择最优路径（NVLink / PCIe / IB），开发者基本感受不到它的存在——直到它变慢。

这一节围绕实战展开：

- **理论入门**：[NCCL 技术理论](03_nccl/01_nccl_theory.md) | [NCCL 单卡验证](03_nccl/02_nccl_helloworld.md)——先理解原理，再动手跑通。
- **使用教程**：[NCCL 测试验证工具说明](03_nccl/03_nccl_tutorial.md)——单节点 → 容器化 → 多节点完整流程。
- **基准测试**：[NCCL 基准测试方法论](03_nccl/04_nccl_benchmark.md)——`allreduce_perf` 编译运行，A100 实测数据。
- **调试工具**：[NCCL Debug 输出解读](03_nccl/05_nccl_debug_output.md)——`NCCL_DEBUG=INFO` 完整输出的逐行拆解与异常速查。
- **通信路径压测**：[NCCL 通信路径逐层压测](03_nccl/06_nccl_path_benchmark.md)——H100 实测：NVLink (316 GB/s) → P2P Disable (24 GB/s)，跨 NUMA 无衰减验证。

当你看到训练吞吐”突然掉了一截”却找不到代码原因时，十有八九要去 NCCL 这一层找答案。
