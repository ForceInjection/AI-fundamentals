# Gang Scheduling——分布式训练的 All-or-Nothing 调度

PyTorch DDP 初始化时，所有 rank 需要同时执行 `init_process_group`。如果 8 GPU 的训练任务只有 6 个 Pod 被调度，已有的 6 个 Pod 会阻塞在初始化上——占着 GPU、不计算、等待超时。这就是默认调度器逐个 Pod 决策的结构性问题：调度器不知道哪些 Pod 属于同一个训练作业。

本文解释 Gang Scheduling 的两种主流实现方式：Volcano Coscheduling 和 Kubernetes SchedulingGates。

---

## 一、问题：部分调度的三重浪费

一个典型场景：

```text
集群: 2 节点 × 8 GPU，节点 A 还剩 6 GPU，节点 B 还剩 4 GPU
作业: TP=8 训练，需要 8 GPU

K8s 默认调度器逐个处理 8 个 Pod:
  Pod 0-5: 调度到节点 A（占满剩余 6 GPU）
  Pod 6-7: Pending（没有足够 GPU 了）

结果:
  Pod 0-5 启动 → init_process_group → 等 Pod 6-7 → 阻塞
  节点 A 的 6 GPU 被占但零计算
  节点 B 的 4 GPU 空闲但 Pod 6-7 不够分配
  整个作业卡死
```

**三重浪费**：（1）已调度的 GPU 被无效占用；（2）其他作业因为节点 A "GPU 已满"得不到调度；（3）训练作业自身因超时而整体失败，GPU 释放后一切从头再来。

---

## 二、方案 A：Volcano Coscheduling

Volcano 是 CNCF 的批量计算调度器，在原生 K8s 调度能力之上增加了 Gang Scheduling、Fair-share、Queue 等特性。其 Coscheduling 的实现思路是：

```text
1. PodGroup 创建 → 声明 "这个作业需要 8 个 Pod"
2. 调度器不是逐个调度 Pod，而是为整个 PodGroup 做决策
3. 只有当 8 个 Pod 的资源需求都能满足时，才一次提交调度
4. 如果资源不够 → 整个 PodGroup 排队等待，不占任何 GPU
```

核心配置：

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: training-job-tp8
spec:
  minMember: 8 # 至少 8 个 Pod 全部就绪才启动
  minResources: # 每个 Pod 需要的资源
    nvidia.com/gpu: "1"
  queue: ai-training # 所属队列
---
# 每个训练 Pod 中指定所属的 PodGroup
apiVersion: v1
kind: Pod
metadata:
  labels:
    scheduling.volcano.sh/podgroup: training-job-tp8 # 关联到 PodGroup
spec:
  schedulerName: volcano # 使用 Volcano 调度器
  containers:
    - resources:
        limits:
          nvidia.com/gpu: 1
```

与默认调度器的关键区别：Volcano 在看到 8 个 Pod 时不会逐个提交——它先检查集群中是否存在能同时容纳 8 个 Pod 的节点/节点组。如果存在，一次性全调；如果不存在，PodGroup 进入 `Inqueue` 状态等待。

---

## 三、方案 B：Kubernetes SchedulingGates（1.26+）

K8s 1.26 引入了 SchedulingGates，允许外部控制器在 Pod 上设置「栅栏」，被栅住的 Pod 不会被调度器处理。配合一个 Gang Scheduling Controller，可以在不替换调度器的前提下实现 All-or-Nothing：

```text
1. Job Controller 创建 8 个 Pod，每个 Pod 带一个 scheduling gate
2. Gang Controller 检测到 8 个 Pod 的 PodGroup 已就绪
3. Gang Controller 移除所有 8 个 Pod 的 scheduling gate
4. 调度器同时处理 8 个 Pod → 逐个调度但不是逐个抢占
```

核心配置：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-worker-0
spec:
  schedulingGates: # K8s 1.26+
    - name: gang-scheduling # 有 gate 时调度器不处理此 Pod
  containers:
    - resources:
        limits:
          nvidia.com/gpu: 1
```

SchedulingGates 的优势是不需要替换调度器，但劣势也很明显：移除 gate 后调度器仍然是逐个 Pod 决策，可能出现「部分 Pod 调度成功、部分仍 Pending」——只是把「先到先得」变成了「同时开始」。要彻底解决部分调度问题，仍需要在调度器层做 All-or-Nothing 决策，即 Volcano 的 Coscheduling 模型。

---

## 四、选型建议

| 场景                                    | 推荐方案                         | 原因                                                 |
| --------------------------------------- | -------------------------------- | ---------------------------------------------------- |
| 纯训练集群，作业以多 GPU 分布式训练为主 | Volcano Coscheduling             | 保证 All-or-Nothing，配合 queue 实现优先级和公平调度 |
| 混合负载集群，训练 + 推理共存           | K8s SchedulingGates + 轻量控制器 | 不改调度器，渐进式引入 Gang Scheduling               |
| 已有 K8s 集群，不想引入额外调度器       | SchedulingGates                  | 最小改动，但效果不如 Volcano 彻底                    |
| 多租户 + 多队列 + 优先级                | Volcano                          | Queue + Fair-share 开箱即用                          |

---

## 五、相关资源

- [Volcano 官方文档](https://volcano.sh/docs/) — PodGroup / Queue / Coscheduling
- [K8s SchedulingGates KEP](https://github.com/kubernetes/enhancements/tree/master/keps/sig-scheduling/3521-pod-scheduling-readiness)
- [GPU 调度问题总览](01_gpu_scheduling_problem.md) — 碎片化、拓扑、Gang Scheduling 三个盲区
- [拓扑感知调度](03_topology_aware_scheduling.md) — Gang Scheduling 解决「够不够」的问题，拓扑感知解决「快不快」的问题
