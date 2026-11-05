# GPU 调度为什么比 CPU 调度难——碎片化、拓扑、Gang Scheduling

Kubernetes 默认调度器把 GPU 当成标量资源（`nvidia.com/gpu: 1`），和 CPU 的 `cpu: 4` 没什么区别。这个模型对于单 GPU 作业基本够用，但对于 AI 集群的真实负载有三个致命盲区：GPU 碎片化、拓扑不感知、Gang Scheduling 缺失。

本文从这三个问题出发，解释为什么 AI 集群需要 GPU 感知的调度策略。后文展开具体的解决方案：Gang Scheduling（§02）、拓扑感知（§03）、GPU 共享调度（§04）。

---

## 一、盲区 1：GPU 碎片化——4 张空闲 GPU ≠ 可以跑 4-GPU 的训练任务

K8s 默认调度器做的是「装箱」：把每个 Pod 放到一个满足资源请求的节点上。假设集群有 2 个 8-GPU 节点，节点 A 已经被 4 个单 GPU 作业各占了一张 GPU（GPU 0/2/4/6），剩余 4 张 GPU 空闲。此时一个请求 4 张 GPU 的训练 Pod 提交，默认调度器看到节点 A 有 4 张空闲 GPU，会把它调度上去。

然后 Pod 启动失败。

为什么？看剩余的卡落在哪。GPU 0/2/4/6 被占满后，剩下的是 GPU 1/3/5/7。如果这 4 张卡跨了 NVLink 域或 NUMA 边界（分散在不同 PCIe switch 下、不同 NUMA node 上），TP=4 训练的 NCCL 通信就要在 PCIe 和 NUMA 之间来回跳跃，带宽会从 NVLink 的数百 GB/s 掉到 PCIe 的几十 GB/s（PCIe Gen4 x16 单向约 32 GB/s、Gen5 单向约 64 GB/s，跨 socket 再叠加 UPI 一跳约 28 GB/s，均为数量级估计）；如果它们仍在同一个 NVLink 域内，训练就不受影响。默认调度器分辨不出这两种情况。

**真实案例：一台 8×H100 的 NVSwitch 全互联拓扑**（`nvidia-smi topo -m` 实测）：

```text
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0     X      NV18    NV18    NV18    NV18    NV18    NV18    NV18
GPU1    NV18     X      NV18    NV18    NV18    NV18    NV18    NV18
...（所有 8 张 GPU 两两之间均为 NV18，同一个 NVSwitch 域）
```

所有 GPU 共享一个 NVSwitch 域，任意 pair 都有完整的 450 GB/s 单向带宽。现在假设 4 个单 GPU 推理 Pod 分别占用了 GPU 0、2、4、6：

```text
节点 A (8×H100, NVSwitch 全互联)
┌─┬─┬─┬─┬─┬─┬─┬─┐
│■│ │■│ │■│ │■│ │  ← 4 个推理 Pod 各占一张 GPU
└─┴─┴─┴─┴─┴─┴─┴─┘
 0 1 2 3 4 5 6 7

剩余空闲: GPU 1,3,5,7 (4 张)
NVLink 实测带宽: 任意 pair ~316 GB/s (all_reduce bus_bw)
```

此时提交一个 TP=4 的训练作业，请求 4 张 GPU。K8s 默认调度器看到 `Allocatable: 4, Requested: 4 → OK`，调度成功。NVLink 实测也正常：这台 H100 的 8 张 GPU 全部在同一个 NVSwitch 域内，GPU 1/3/5/7 之间同样是 NVLink 全速互联，训练不受影响。

但如果把同样的场景放到一台 **无 NVSwitch、按 4 张一组 NVLink bridge 分组的 A100 服务器**（常见的 8 卡 A100 机型）上：

```text
节点 B (8×A100, 无 NVSwitch, NVLink bridge 分组)
组 1 (GPU 0-3):  两两 NVLink ~600 GB/s 双向 (bridge 互联)
组 2 (GPU 4-7):  两两 NVLink ~600 GB/s 双向 (bridge 互联)
跨组 (GPU 0↔4):  PCIe P2P ~32 GB/s (Gen4 x16 单向, 数量级估计)  ← 只有 NVLink 的 1/10-1/5
```

此时 GPU 0/2/4/6 被占用后，剩余的 GPU 1/3/5/7 分属两个 bridge 组。TP=4 作业被调度到 GPU 1,3（组 1）和 GPU 5,7（组 2），跨组通信只能走 PCIe，训练吞吐从 90% 利用率跌到 30%。

默认调度器在两个场景下看到的完全一样：`Allocatable: 4, Requested: 4 → OK`。但实际结果天差地别，前者正常训练，后者慢 3 倍。`nvidia.com/gpu` 表达不了 NVLink 域的拓扑边界。

解决方案方向：拓扑感知调度 + 碎片整理（重调度/deschedule 低优先级单 GPU 作业）。

---

## 二、盲区 2：拓扑不感知——最近的 GPU 不一定是挨着编号的

K8s 默认调度器不理解 GPU 之间的通信路径。它看到两张 GPU，不会知道它们之间走 NVLink 还是 PCIe、延迟差几个数量级。

```text
GPU 0 ← NVLink 450 GB/s       → GPU 1  ← 同一 NVLink 域，最快
GPU 0 ← PCIe ~32-64 GB/s      → GPU 4  ← 跨 NUMA，次快（Gen4/Gen5 单向，数量级估计）
GPU 0 ← UPI ~28 GB/s（经验值）→ GPU 8  ← 跨 socket（双路服务器），最慢
```

（示意图按双路服务器通用画法：单节点 8 卡通常编号 0-7，跨 socket 的卡只在 16 卡这类更大规格的节点上才会编到 8。）

对于单 GPU 推理任务，拓扑无关紧要。对于 TP=8 的训练，GPU 之间的通信路径决定了吞吐。如果 8 张 GPU 不在同一个 NVLink 域内、甚至跨了节点，训练吞吐可能从 90% 利用率跌到 30%。

节点内的拓扑信息来自 `nvidia-smi topo -m`。NVIDIA 早年提供一个 topology-aware scheduler 插件来消费这些信息，但该项目已下线（文档页与 GitHub 仓库均已 404），现行方案是 KAI-Scheduler（已入 CNCF sandbox）配合 Topograph 拓扑导出层，详见[拓扑感知调度](03_topology_aware_scheduling.md)。另外要注意，拓扑感知调度的效果取决于 GPU 资源的实际分配状态：在高负载混合集群中，拓扑最优的 GPU 很可能已经被其他作业占用了。

---

## 三、盲区 3：Gang Scheduling——要么全调度，要么全不等

训练任务是 All-or-Nothing 的：PyTorch DDP 初始化 NCCL communicator 时需要 **全部 rank 同时在线**。如果 8 GPU 的训练任务只有 6 个 Pod 被调度、另外 2 个还在 Pending，已有的 6 个 Pod 会阻塞在 `init_process_group` 上——**占着 GPU 但不计算**，直到超时。

这种部分调度的后果是多重浪费：

- 已调度的 Pod 占着 GPU 空转
- 未调度的 Pod 在队列中卡住
- 其他等待 GPU 的作业因为「GPU 看似被占满」而得不到调度

K8s 默认调度器是逐个 Pod 独立决策的，无法理解「只有全部 Pod 都能调度时，这个作业才有意义」。解决这个问题需要 Gang Scheduling：调度器在确认所有 Pod 的 GPU 需求都能满足之前，不为任何一个 Pod 做调度决策。Volcano 的 Coscheduling 和 Kubernetes 原生的 SchedulingGates 是实现这一需求的主流方案。详见 [Gang Scheduling：分布式训练的 All-or-Nothing 调度](02_gang_scheduling_for_training.md)。

---

## 四、三个盲区的解决路径

| 盲区               | 现象                                    | 解决方向                          | 详见                                      |
| ------------------ | --------------------------------------- | --------------------------------- | ----------------------------------------- |
| GPU 碎片化         | 多 GPU 训练无法找到连续 GPU 组          | 拓扑感知调度 + 碎片整理           | [§03](03_topology_aware_scheduling.md)    |
| 拓扑不感知         | 调度到跨 NUMA/跨 socket 的 GPU，通信慢  | NVLink/NUMA 感知的 Filter + Score | [§03](03_topology_aware_scheduling.md)    |
| 无 Gang Scheduling | 训练任务部分 Pod 占 GPU 空转            | Coscheduling / SchedulingGates    | [§02](02_gang_scheduling_for_training.md) |
| GPU 共享的粒度     | MIG/MPS/Timeslicing 如何暴露为 K8s 资源 | Device Plugin 的不同策略          | [§04](04_gpu_sharing_scheduling.md)       |

---

## 相关资源

- [Kubernetes 调度器介绍](https://github.com/ForceInjection/kubernetes-hands-on-course/tree/master/Advanced-Topics/调度/k8s-scheduler-intro-basic.md) — 默认调度器的工作原理
- [GPU 调度器扩展案例](https://github.com/ForceInjection/kubernetes-hands-on-course/tree/master/Advanced-Topics/调度/k8s-scheduler-gpu-case.md) — Device Plugin + 自定义调度器扩展
- [KAI-Scheduler（CNCF sandbox，NVIDIA 捐献）](https://github.com/kai-scheduler/KAI-Scheduler)
- [NVIDIA Topograph](https://docs.nvidia.com/topograph/) — 拓扑导出层，把集群拓扑暴露给调度器
- [NCCL 通信路径逐层压测](../03_nccl/06_nccl_path_benchmark.md) — NVLink 和 PCIe 到底差多少
