# GPU 调度——从"够不够"到"快不快"

K8s 默认调度器把 GPU 当成标量资源（`nvidia.com/gpu: 1`），对于单 GPU 作业够用，但对于 AI 集群的真实负载有三个盲区：GPU 碎片化、拓扑不感知、Gang Scheduling 缺失。

本目录从这三个盲区出发，逐层展开解决方案。

## 文档

| #   | 文档                                                  | 内容                                                                |
| --- | ----------------------------------------------------- | ------------------------------------------------------------------- |
| 01  | [GPU 调度问题总览](01_gpu_scheduling_problem.md)      | 碎片化、拓扑不感知、Gang Scheduling 三个盲区的分析                  |
| 02  | [Gang Scheduling](02_gang_scheduling_for_training.md) | Volcano Coscheduling vs K8s SchedulingGates，All-or-Nothing 调度    |
| 03  | [拓扑感知调度](03_topology_aware_scheduling.md)       | NVLink/NUMA/跨节点三层拓扑感知 + NVIDIA Topology-Aware Scheduler    |
| 04  | [GPU 共享调度](04_gpu_sharing_scheduling.md)          | MIG（硬切分）、MPS（计算共享）、Time-slicing（时分）在 K8s 中的表达 |

## 相关资源

**概念层（本文）→ 实现层（进阶）**：

- [云原生 AI 平台 — GPU 管理与虚拟化](../../04_cloud_native_ai_platform/gpu_manager/README.md) — 从概念到落地：NVIDIA Container Toolkit、Device Plugin、HAMi、Kueue 实战部署
- [云原生 AI 平台 — K8s GPU 管理](../../04_cloud_native_ai_platform/k8s/README.md) — K8s 原生 AI 工作负载的容器运行时、调度器与推理框架
- [Kubernetes 调度器扩展案例：GPU 资源调度](https://github.com/ForceInjection/kubernetes-hands-on-course/tree/master/Advanced-Topics/调度/k8s-scheduler-gpu-case.md) — Device Plugin + 自定义调度器的完整实现
- [NVIDIA GPU Operator 文档](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/)
- [Volcano 官方文档](https://volcano.sh/docs/)
