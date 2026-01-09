# Kueue + HAMi：Kubernetes 原生的 AI 工作负载管理与 GPU 虚拟化解决方案

## 1. 概述与核心价值

`Kueue` 是一个专为 `Kubernetes` 设计的云原生作业队列系统，专门用于管理批处理、高性能计算（`HPC`）、人工智能/机器学习（`AI`/`ML`）以及类似应用的工作负载。作为 `Kubernetes SIG Scheduling` 的官方项目，`Kueue` 提供了一套完整的 `API` 和控制器，用于实现作业级别的队列管理和资源调度。

### 1.1 核心价值与定位

`Kueue` 的设计理念是构建一个多租户的批处理服务，通过配额管理和层次化的资源共享机制，帮助组织内的不同团队高效共享计算资源。基于可用配额，`Kueue` 能够智能决策作业的等待时机以及运行的时间和位置。

重要的是，`Kueue` 与标准的 `kube-scheduler`、`cluster-autoscaler` 以及整个 `Kubernetes` 生态系统协同工作。这种设计使得 `Kueue` 既可以在本地环境中运行，也可以在云环境中部署，支持异构、可替代和动态配置的资源。

### 1.2 生产就绪状态

`Kueue` 已经达到生产就绪状态，具备以下特征：

- ✅ **API 版本**：`v1beta1`，遵循 `Kubernetes` 弃用策略
- ✅ **完整文档**：提供详尽的使用文档和最佳实践
- ✅ **全面测试覆盖**：
  - 单元测试、集成测试、端到端测试
  - 支持 `Kubernetes 1.29`或更新版本
  - 性能测试验证可扩展性
- ✅ **安全性**：基于 `RBAC` 的访问控制
- ✅ **稳定发布周期**：2-3 个月的规律发布
- ✅ **生产采用者**：已有多个组织在生产环境中使用 `Kueue`

---

## 2. 核心概念与架构

### 2.0 核心资源对象关系图

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                Kueue 核心架构                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│      ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│      │   Namespace A   │    │   Namespace B   │    │   Namespace C   │          │
│      │                 │    │                 │    │                 │          │
│      │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │          │
│      │ │ LocalQueue  │ │    │ │ LocalQueue  │ │    │ │ LocalQueue  │ │          │
│      │ │   (team-a)  │ │    │ │   (team-b)  │ │    │ │   (team-c)  │ │          │
│      │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │          │
│      │        │        │    │        │        │    │        │        │          │
│      │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │          │
│      │ │    Jobs     │ │    │ │    Jobs     │ │    │ │    Jobs     │ │          │
│      │ │ (用户提交)   │ │    │ │ (用户提交)    │ │    │ │ (用户提交)   │ │          │
│      │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │          │
│      └─────────────────┘    └─────────────────┘    └─────────────────┘          │
│               │                       │                       │                 │
│               └───────────────────────┼───────────────────────┘                 │
│                                   │                                             │
│                                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                          Workload 抽象层                                 │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │  Workload   │  │  Workload   │  │  Workload   │  │  Workload   │     │    │
│  │  │   (Job-1)   │  │   (Job-2)   │  │   (Job-3)   │  │   (Job-N)   │     │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                   │                                             │
│                                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        ClusterQueue 调度层                               │    │
│  │                                                                         │    │
│  │      ┌─────────────────┐              ┌─────────────────┐               │    │
│  │      │  ClusterQueue   │              │  ClusterQueue   │               │    │
│  │      │   (gpu-queue)   │◄────────────►│  (cpu-queue)    │               │    │
│  │      │                 │   Cohort     │                 │               │    │
│  │      │ • 资源配额       │   资源共享     │ • 资源配额       │               │    │
│  │      │ • 调度策略       │               │ • 调度策略       │               │    │
│  │      │ • 抢占机制       │               │ • 抢占机制       │               │    │
│  │      └─────────────────┘              └─────────────────┘               │    │
│  │               │                                │                        │    │
│  │               ▼                                ▼                        │    │
│  │      ┌─────────────────┐              ┌─────────────────┐               │    │
│  │      │ ResourceFlavor  │              │ ResourceFlavor  │               │    │
│  │      │   (gpu-v100)    │              │   (cpu-intel)   │               │    │
│  │      │                 │              │                 │               │    │
│  │      │ • 节点标签       │              │ • 节点标签        │               │    │
│  │      │ • 污点容忍       │              │ • 污点容忍        │               │    │
│  │      │ • 资源特征       │              │ • 资源特征        │               │    │
│  │      └─────────────────┘              └─────────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                   │                                             │
│                                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        Kubernetes 调度层                                 │    │
│  │                                                                         │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │    Node     │  │    Node     │  │    Node     │  │    Node     │     │    │
│  │  │  (GPU节点)   │  │  (GPU节点)  │  │  (CPU节点)   │  │  (CPU节点)   │     │    │
│  │  │             │  │             │  │             │  │             │     │    │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │     │    │
│  │  │ │   Pod   │ │  │ │   Pod   │ │  │ │   Pod   │ │  │ │   Pod   │ │     │    │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │     │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

关系说明：

- `LocalQueue`：命名空间级别的作业提交入口，实现多租户隔离
- `Workload`：作业的抽象表示，包含资源需求和调度信息
- `ClusterQueue`：集群级别的资源配额和调度策略管理
- `ResourceFlavor`：定义具有特定特征的资源类型（如GPU型号、CPU架构），用于匹配节点资源
- `Cohort`：多个`ClusterQueue`组成的资源共享组，支持动态资源借用
- 调度流程：`Job` → `LocalQueue` → `Workload` → `ClusterQueue` → `ResourceFlavor` → `Node`

### 2.1 核心资源对象

#### 2.1.1 ClusterQueue（集群队列）

`ClusterQueue` 是 `Kueue` 的核心资源对象，定义了集群级别的资源配额和调度策略。它是整个队列系统的基础，负责管理和分配集群中的计算资源。主要特性包括：

- **资源组管理**：通过 `ResourceGroups` 定义不同类型资源的配额，如 CPU、内存、GPU 等，支持细粒度的资源控制
- **队列策略**：支持 `StrictFIFO`（严格先进先出）和 `BestEffortFIFO`（尽力而为先进先出）两种排队策略，满足不同场景的调度需求
- **命名空间选择器**：通过标签选择器控制哪些命名空间可以向该队列提交工作负载，实现多租户隔离
- **抢占策略**：支持多种抢占机制以优化资源利用率，包括同队列内抢占和跨队列抢占
- **资源借用**：允许队列在资源空闲时借用其他队列的配额，提高整体资源利用率
- **公平共享**：通过 Cohort 机制实现多个队列之间的公平资源共享

#### 2.1.2 LocalQueue（本地队列）

`LocalQueue` 是命名空间级别的队列，作为用户提交作业的入口点，与特定的 `ClusterQueue` 关联。它在 `Kueue` 架构中起到了关键的桥梁作用：

**主要作用：**

- **作业提交入口**：用户通过在 `Job`、`CronJob` 等工作负载中指定 `kueue.x-k8s.io/queue-name` 标签来将作业提交到 `LocalQueue`
- **命名空间隔离**：每个 `LocalQueue` 只能在特定命名空间中创建和使用，实现了天然的多租户隔离
- **队列映射**：将命名空间级别的作业请求映射到集群级别的 `ClusterQueue`，实现资源的统一管理
- **权限控制**：通过 `RBAC` 机制，可以精确控制不同用户或团队对特定 `LocalQueue` 的访问权限

**工作流程：**

1. 用户在特定命名空间中创建 `LocalQueue`，并关联到相应的 `ClusterQueue`
2. 用户提交作业时指定 `LocalQueue` 名称
3. `Kueue` 控制器将作业转换为 `Workload` 对象，并根据 `LocalQueue` 的配置进行调度
4. 作业在 `ClusterQueue` 中排队等待资源分配

#### 2.1.3 Workload（工作负载）

`Workload` 是 `Kueue` 中作业的抽象表示，是系统内部用于调度管理的核心对象。它包含了作业的完整调度信息：

**主要组成：**

- **资源需求**：详细描述作业所需的 `CPU`、`内存`、`GPU` 等资源规格
- **优先级信息**：基于 `Kubernetes PriorityClass` 的优先级设置
- **调度约束**：包括节点选择器、亲和性、反亲和性、容忍度等调度限制
- **队列信息**：关联的 `LocalQueue` 和 `ClusterQueue` 信息
- **状态管理**：跟踪作业的调度状态，如 `Pending`、`Admitted`、`Running`、`Finished` 等

**生命周期：**

1. **创建阶段**：当用户提交 `Job`、`CronJob` 等工作负载时，`Kueue` 自动创建对应的 `Workload` 对象
2. **排队阶段**：`Workload` 在 `ClusterQueue` 中排队等待资源分配
3. **准入阶段**：当资源可用时，`Workload` 被标记为 `Admitted`，原始作业开始执行
4. **完成阶段**：作业执行完成后，`Workload` 状态更新为 `Finished`，释放占用的资源

**状态转换详细说明：**

`Workload` 在其生命周期中会经历多个状态，每个状态转换都有特定的触发条件和对应的 `WorkloadCondition`：

- **Pending（等待）**：`Workload` 创建后的初始状态，等待 `ClusterQueue` 中的资源配额。此时 `Workload` 在队列中排队，等待资源预留成功。

- **QuotaReserved（配额预留）**：当 `ClusterQueue` 有足够资源时，系统为 `Workload` 预留资源配额，对应 `WorkloadQuotaReserved` 条件为 `True`。此阶段 `Workload` 等待通过所有准入检查。

- **Admitted（已准入）**：`Workload` 通过所有 `AdmissionCheck` 后进入此状态，对应 `WorkloadAdmitted` 条件为 `True`。此时 `Kubernetes` 开始调度和启动相关的 `Pod`。

- **Running（运行中）**：当所有 `Pod` 成功启动并准备就绪时，`WorkloadPodsReady` 条件变为 `True`，`Workload` 进入运行状态。

- **Finished（已完成）**：作业执行完成（成功或失败）后，`WorkloadFinished` 条件被设置，原因可能是 `Succeeded`、`Failed` 或 `OutOfSync`。

- **Evicted（被驱逐）**：`Workload` 可能因多种原因被驱逐，包括：
  - 资源不足或队列停止（`ClusterQueueStopped`、`LocalQueueStopped`）
  - 准入检查失败（`AdmissionCheckFailed`）
  - 运行超时（`PodsReadyTimeout`、`MaximumExecutionTimeExceeded`）
  - 节点故障（`NodeFailures`）
  对应 `WorkloadEvicted` 条件会被设置为 `True`，并包含具体的驱逐原因。

- **Preempted（被抢占）**：当高优先级 `Workload` 需要资源时，低优先级的 `Workload` 可能被抢占。抢占可能发生在队列内部（`InClusterQueueReason`）或 `Cohort` 级别的资源回收中。

- **Requeued（重新排队）**：被抢占或驱逐的 `Workload` 可以重新进入队列等待调度，对应 `WorkloadRequeued` 条件。系统会根据退避策略和重试限制决定是否允许重新排队。

**状态转换触发条件：**

- **资源可用性**：`ClusterQueue` 中的资源配额状况直接影响 `Pending` 到 `QuotaReserved` 的转换
- **准入检查**：所有配置的 `AdmissionCheck` 必须通过才能从 `QuotaReserved` 转换到 `Admitted`
- **Pod 调度**：`Kubernetes` 调度器的成功调度触发从 `Admitted` 到 `Running` 的转换
- **优先级抢占**：高优先级 `Workload` 的资源需求可能触发低优先级 `Workload` 的抢占
- **超时机制**：`maximumExecutionTimeSeconds` 等超时设置可能导致 `Workload` 被驱逐
- **外部事件**：节点故障、队列停止等外部事件可能导致状态转换

这种状态机制确保了 `Workload` 在复杂的集群环境中能够正确处理各种调度场景和异常情况，同时为用户提供了清晰的作业状态可见性。

#### 2.1.4 ResourceFlavor（资源风味）

`ResourceFlavor` 定义了具有特定特征的资源类型，是 `Kueue` 实现异构资源管理的关键机制。它允许集群管理员对不同类型的计算资源进行精细化管理：

**核心功能：**

- **节点标签匹配**：通过 `nodeLabels` 字段指定资源所在节点的标签，如 `GPU` 型号、`CPU` 架构等
- **污点容忍**：通过 `tolerations` 字段定义对特定节点污点的容忍度，确保作业能够调度到专用节点
- **资源替代性**：在 `ClusterQueue` 中可以定义多个 `ResourceFlavor`，实现资源的可替代性和灵活调度

**典型应用场景：**

- **GPU 类型管理**：为不同型号的 `GPU`（如 `V100`、`A100`、`T4`）创建不同的 `ResourceFlavor`
- **节点分组**：为不同可用区、不同硬件配置的节点创建专门的 `ResourceFlavor`
- **专用资源池**：为特定团队或项目创建专用的资源池，通过污点和容忍度实现隔离

### 2.2 高级特性

#### 2.2.1 Cohort（队列组）

`Cohort` 是 `Kueue` 中实现资源池化和动态共享的重要机制，允许多个 `ClusterQueue` 组成一个资源共享组：

**核心价值：**

- **资源池化**：将多个队列的资源配额汇聚成一个更大的资源池，提高整体资源利用率
- **动态借用**：当某个队列资源不足时，可以从同一 `Cohort` 内其他空闲队列借用资源
- **弹性扩展**：支持队列根据实际需求动态调整资源使用量，避免资源浪费

**工作机制：**

1. **资源聚合**：`Cohort` 内所有队列的资源配额被聚合为一个总的资源池
2. **优先级保障**：每个队列仍保持其基础资源配额的优先使用权
3. **借用机制**：当队列需要超出其配额的资源时，可以从 `Cohort` 的共享池中借用
4. **归还机制**：当资源所有者队列需要资源时，借用的资源会被自动归还

#### 2.2.2 Fair Sharing（公平共享）

在 `Cohort` 内部，`Kueue` 实现了基于权重的公平共享机制，确保各队列在资源竞争时获得合理的资源分配：

**公平性算法：**

- **权重分配**：每个 `ClusterQueue` 可以设置权重值，权重越高获得的资源比例越大
- **动态调整**：系统根据队列的实际使用情况和等待作业数量动态调整资源分配
- **饥饿防护**：确保低权重队列也能获得基本的资源保障，避免长期饥饿

**计算公式：**

```bash
队列资源份额 = (队列权重 / Cohort总权重) × Cohort总资源
```

**应用场景：**

- **多团队环境**：不同团队根据重要性或规模获得不同的资源权重
- **业务优先级**：生产环境队列获得更高权重，开发测试队列权重较低
- **时间段调整**：根据业务高峰期动态调整不同队列的权重

#### 2.2.3 Preemption（抢占）

`Kueue` 的抢占机制是一个多层次、智能化的资源回收系统，支持多种抢占策略以提高资源利用率：

**抢占类型：**

- **同队列内抢占**：高优先级作业可以抢占同一队列内低优先级作业的资源
- **跨队列抢占**：高优先级队列可以抢占低优先级队列的借用资源
- **Cohort 内抢占**：在 `Cohort` 内部，资源所有者可以抢占借用者的资源

**抢占策略：**

- **LowerPriority**：只抢占优先级更低的工作负载
- **LowerOrNewerEqualPriority**：抢占优先级更低或同等优先级但更新的工作负载
- **Any**：可以抢占任何工作负载（需谨慎使用）

**抢占流程：**

1. **资源需求评估**：系统评估新作业的资源需求
2. **候选者识别**：根据抢占策略识别可被抢占的工作负载
3. **影响最小化**：选择抢占影响最小的工作负载组合
4. **优雅终止**：被抢占的作业会收到终止信号，有机会进行清理工作
5. **重新排队**：被抢占的作业自动重新进入队列等待调度

**抢占保护：**

- **最小运行时间**：可以设置作业的最小运行时间，避免频繁抢占
- **抢占冷却期**：设置抢占操作之间的最小间隔，防止系统震荡
- **关键作业保护**：可以标记某些关键作业为不可抢占

### 2.3 详细调度流程

`Kueue` 的调度流程是一个多阶段、多层次的复杂过程，涉及作业提交、资源评估、队列管理、准入控制等多个环节。以下详细描述了从作业提交到最终调度到节点的完整流程：

#### 2.3.1 作业提交阶段

**1. 用户作业提交**：

用户通过标准的 `Kubernetes API` 提交作业（`Job`、`CronJob`、`TFJob` 等），并在作业规范中指定目标队列：

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: sample-job
  labels:
    kueue.x-k8s.io/queue-name: user-queue  # 指定目标 LocalQueue
spec:
  template:
    spec:
      containers:
      - name: main
        image: busybox
        resources:
          requests:
            cpu: "1"
            memory: "1Gi"
```

**2. Workload 对象创建**：

`Kueue` 控制器监听到新的作业后，会自动创建对应的 `Workload` 对象，该对象包含：

- **资源需求规范**：从原始作业中提取的 `CPU`、`内存`、`GPU` 等资源需求
- **调度约束**：节点选择器、亲和性、反亲和性、容忍度等
- **队列关联信息**：关联的 `LocalQueue` 和对应的 `ClusterQueue`
- **优先级信息**：基于 `PriorityClass` 的优先级设置

#### 2.3.2 队列路由阶段

**1. LocalQueue 验证**：

系统首先验证指定的 `LocalQueue` 是否存在且用户有权限访问：

- **命名空间检查**：确认 `LocalQueue` 在正确的命名空间中
- **RBAC 验证**：检查用户是否有向该队列提交作业的权限
- **队列状态检查**：确认队列处于活跃状态

**2. ClusterQueue 映射**：

`LocalQueue` 将作业路由到关联的 `ClusterQueue`：

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: user-queue
  namespace: team-a
spec:
  clusterQueue: gpu-cluster-queue  # 映射到集群队列
```

#### 2.3.3 资源评估与匹配阶段

**1. 资源需求分析**：

`Kueue` 分析 `Workload` 的资源需求，并将其转换为内部资源表示：

- **资源规范化**：将不同类型的资源需求统一为标准格式
- **资源聚合**：计算作业的总资源需求（考虑并行度）
- **特殊资源处理**：处理 `GPU`、`存储` 等特殊资源类型

**2. ResourceFlavor 匹配**:

`ClusterQueue` 中定义的 `ResourceFlavor` 用于匹配合适的资源：

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: gpu-cluster-queue
spec:
  resourceGroups:
  - coveredResources: ["nvidia.com/gpu", "cpu", "memory"]
    flavors:
    - name: gpu-v100
      resources:
      - name: "nvidia.com/gpu"
        nominalQuota: 8
      - name: "cpu"
        nominalQuota: 32
      - name: "memory"
        nominalQuota: 128Gi
```

**3. 节点兼容性检查**：

系统检查 `ResourceFlavor` 定义的节点标签和容忍度是否与集群中的节点兼容：

- **标签匹配**：验证集群中是否存在具有所需标签的节点
- **污点容忍**：检查作业是否能容忍目标节点的污点
- **资源可用性**：确认节点上有足够的物理资源

#### 2.3.4 队列调度阶段

**1. 队列内排序**：

在 `ClusterQueue` 内部，`Workload` 按照配置的排队策略进行排序：

- **StrictFIFO**：严格按照提交时间排序
- **BestEffortFIFO**：在 `FIFO` 基础上考虑资源可用性
- **优先级排序**：高优先级作业优先调度

**2. Cohort 级别调度**：

如果 `ClusterQueue` 属于某个 `Cohort`，还需要考虑 `Cohort` 级别的调度：

- **公平共享计算**：根据权重计算各队列的资源份额
- **借用机制评估**：判断是否可以从其他队列借用资源
- **全局优先级**：在 `Cohort` 内部进行跨队列的优先级比较

#### 2.3.5 准入控制阶段

`Kueue` 采用两阶段准入控制机制：

**1. 配额预留阶段（Quota Reservation）**：

- **资源配额检查**：验证 `ClusterQueue` 是否有足够的可用配额
- **配额预留**：为 `Workload` 预留所需的资源配额
- **借用评估**：如果本队列配额不足，评估是否可以从 `Cohort` 借用

**2. 准入检查阶段（Admission Checks）**：

并行执行所有配置的 AdmissionChecks：

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: gpu-cluster-queue
spec:
  admissionChecks:
  - "provisioning-check"  # 节点预配置检查
  - "security-scan"       # 安全扫描检查
```

每个 `AdmissionCheck` 必须返回 `Ready` 状态，`Workload` 才能被最终准入。

#### 2.3.6 作业准入与调度阶段

**1. Workload 准入**：

当所有准入条件满足后，`Workload` 状态变更为 `Admitted`：

- **资源分配确认**：最终确认分配给作业的具体资源
- **节点选择器注入**：根据 `ResourceFlavor` 修改作业的节点选择器
- **调度约束更新**：更新作业的调度约束以匹配分配的资源

**2. 原始作业恢复**：

`Kueue` 恢复原始作业的调度，使其能够被 `Kubernetes` 调度器处理：

```yaml
# Kueue 会修改原始 Job 的 nodeSelector
apiVersion: batch/v1
kind: Job
metadata:
  name: sample-job
spec:
  template:
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: "gpu-v100"  # 由 ResourceFlavor 注入
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

**3. Kubernetes 调度器接管**：

原始作业现在可以被标准的 `Kubernetes` 调度器调度到具体节点：

- **节点筛选**：根据更新后的节点选择器筛选候选节点
- **资源检查**：验证节点上的实际资源可用性
- **最终调度**：将 `Pod` 绑定到具体的节点

#### 2.3.7 执行监控阶段

**1. 作业状态跟踪**：

`Kueue` 持续监控作业的执行状态：

- **Pod 状态监控**：跟踪作业中各个 `Pod` 的运行状态
- **资源使用监控**：监控实际的资源使用情况
- **异常处理**：处理作业失败、节点故障等异常情况

**2. 动态资源管理**:

- **部分准入**：支持以降低的并行度运行作业
- **动态回收**：`Pod` 完成后立即释放对应的配额
- **弹性调整**：根据实际需求动态调整资源分配

#### 2.3.8 完成与清理阶段

**1. 作业完成处理**:

当作业执行完成后：

- **状态更新**：`Workload` 状态更新为 `Finished`
- **配额释放**：释放作业占用的所有资源配额
- **统计更新**：更新队列和 `Cohort` 的使用统计

**2. 资源回收**:

- **配额归还**：将借用的资源归还给原始队列
- **等待队列激活**：触发等待队列中下一个作业的调度
- **清理操作**：清理相关的临时资源和状态信息

#### 2.3.9 调度流程总结

完整的调度流程可以概括为以下关键路径：

```text
用户提交作业 → Workload创建 → LocalQueue路由 → ClusterQueue排队 
     ↓
资源需求分析 → ResourceFlavor匹配 → 节点兼容性检查
     ↓
队列内排序 → Cohort调度 → 配额预留 → 准入检查
     ↓
Workload准入 → 作业恢复 → Kubernetes调度 → 节点执行
     ↓
状态监控 → 动态管理 → 作业完成 → 资源清理
```

这个流程确保了：

- **资源的高效利用**：通过智能的资源匹配和借用机制
- **公平的资源分配**：通过队列权重和公平共享算法
- **灵活的调度策略**：支持多种排队和抢占策略
- **强大的扩展性**：通过 `AdmissionChecks` 支持自定义准入逻辑

### 2.4 资源配额同步机制

`Kueue` 的资源配额管理采用了一种独特的设计理念，它并不直接监控集群中所有 `Pod` 的实际资源使用情况，而是通过标签识别和手动配置相结合的方式来管理资源配额。这种设计在提供灵活性的同时，也对管理员提出了更高的要求。

#### 2.4.1 Kueue 管理的工作负载识别

Kueue 通过特定的标签来识别和管理工作负载：

**管理标签机制：**

- **标签标识**：`Kueue` 为其管理的所有 `Pod` 添加 `kueue.x-k8s.io/managed=true` 标签
- **资源跟踪**：只有带有此标签的 `Pod` 才会被 Kueue 纳入资源使用统计
- **隔离管理**：没有此标签的 `Pod`（如系统组件、其他调度器管理的工作负载）不会被 `Kueue` 直接感知

```yaml
# Kueue 管理的 Pod 示例
apiVersion: v1
kind: Pod
metadata:
  name: kueue-managed-pod
  labels:
    kueue.x-k8s.io/managed: "true"  # Kueue 管理标识
    kueue.x-k8s.io/queue-name: "user-queue"
spec:
  containers:
  - name: worker
    image: busybox
    resources:
      requests:
        cpu: "1"
        memory: "1Gi"
```

#### 2.4.2 nominalQuota 的设计理念

`nominalQuota` 是 `Kueue` 资源管理的核心概念，它代表了队列可以使用的"名义配额"：

**配额定义：**

根据 `Kueue` 官方文档，`nominalQuota` 的定义明确指出：

> **"The quantity of this resource that the ClusterQueue can admit, accounting for resources consumed by system components and pods not managed by kueue."**

这意味着 `nominalQuota` 应该是：

```text
nominalQuota = 集群总资源 - 系统组件资源 - 非Kueue管理的工作负载资源
```

**配置示例：**

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: production-queue
spec:
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: gpu-nodes
      resources:
      - name: "cpu"
        nominalQuota: 80    # 假设集群有100核，预留20核给系统和其他工作负载
      - name: "memory"
        nominalQuota: 320Gi # 假设集群有400Gi内存，预留80Gi
      - name: "nvidia.com/gpu"
        nominalQuota: 8     # 假设集群有10个GPU，预留2个
```

#### 2.4.3 资源同步的局限性

**非实时同步：**

`Kueue` 的设计存在以下重要特征：

1. **静态配额管理**：`nominalQuota` 需要管理员手动配置和维护
2. **有限感知能力**：`Kueue` 无法自动感知非其管理的工作负载的资源变化
3. **依赖管理员判断**：需要管理员根据集群实际情况合理设置配额

**其他调度器的影响：**

当集群中存在其他调度器（如默认调度器直接调度的 `Pod`、`Volcano`、`Yunikorn` 等）时：

```yaml
# 非 Kueue 管理的 Pod（缺少 kueue.x-k8s.io/managed 标签）
apiVersion: v1
kind: Pod
metadata:
  name: external-workload
  # 注意：没有 kueue.x-k8s.io/managed 标签
spec:
  containers:
  - name: worker
    image: nginx
    resources:
      requests:
        cpu: "2"      # 这2核CPU不会被Kueue感知
        memory: "4Gi"  # 这4Gi内存不会被Kueue感知
```

这些工作负载会消耗实际的集群资源，但 `Kueue` 无法感知，可能导致：

- **资源超分配**：`Kueue` 可能分配超过实际可用的资源
- **调度冲突**：多个调度器可能竞争相同的资源
- **性能下降**：节点资源不足导致的性能问题

#### 2.4.4 最佳实践建议

**统一调度管理：**

1. **集中管理**：尽可能将所有工作负载都通过 `Kueue` 进行管理
2. **标准化流程**：建立标准的作业提交流程，确保使用 `Kueue` 队列
3. **权限控制**：通过 `RBAC` 限制直接向集群提交工作负载的权限

**保守配额配置：**

```yaml
# 保守的配额配置示例
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: safe-queue
spec:
  resourceGroups:
  - coveredResources: ["cpu", "memory"]
    flavors:
    - name: standard-nodes
      resources:
      - name: "cpu"
        nominalQuota: 60    # 集群100核，预留40核（40%缓冲）
      - name: "memory"
        nominalQuota: 240Gi # 集群400Gi，预留160Gi（40%缓冲）
```

**监控和调整：**

1. **资源监控**：定期监控集群实际资源使用情况
2. **配额调整**：根据监控数据定期调整 `nominalQuota`
3. **告警机制**：设置资源使用率告警，及时发现问题

```bash
# 监控集群资源使用的示例命令
kubectl top nodes
kubectl describe nodes | grep -A 5 "Allocated resources"
```

**多调度器环境管理：**

如果必须在多调度器环境中运行：

1. **资源分区**：为不同调度器分配专用的节点池
2. **动态调整**：根据实际使用情况动态调整各调度器的资源配额
3. **监控集成**：集成监控系统，实时跟踪各调度器的资源使用

#### 2.4.5 技术实现细节

**资源使用计算：**

`Kueue` 使用几何平均数来计算资源使用情况，这些信息存储在 `FairSharingStatus` 中：

```yaml
# ClusterQueue 状态示例
status:
  fairSharing:
    weightedShare: 0.5
  usage:
    cpu: "10"      # 当前使用的CPU核数
    memory: "40Gi"  # 当前使用的内存
  pendingWorkloads: 5  # 等待中的工作负载数量
```

**准入惩罚机制：**

`Kueue` 在准入工作负载时会向 `FairSharingStatus` 添加条目惩罚，防止系统被恶意利用：

- **预防性计费**：在实际资源分配前就计入使用量
- **公平性保障**：确保资源分配的公平性
- **防止竞争**：避免多个工作负载同时竞争相同资源

这种设计确保了 `Kueue` 作为配额管理和调度决策系统的有效性，但也要求管理员具备足够的集群管理经验来合理配置和维护资源配额。

---

## 3. 主要功能特性

`Kueue` 提供了一套完整的批处理作业管理功能，以下是其核心能力矩阵和适用场景：

### 3.1 智能作业调度

**核心能力：**

- **优先级队列**：基于 `WorkloadPriorityClass` 的多级优先级调度
- **公平调度算法**：支持严格 `FIFO` 和最佳努力 `FIFO` 策略
- **抢占机制**：`LowerPriority`、`LowerOrNewerEqualPriority` 等多种抢占策略

**适用场景：**

- 多团队共享集群环境，需要优先级保障
- 紧急任务需要抢占低优先级作业资源
- 大规模批处理作业的有序执行

### 3.2 灵活资源管理

**核心能力：**

- **异构资源支持**：通过 `ResourceFlavor` 管理不同类型的 `GPU`、`CPU` 架构
- **资源借用机制**：`Cohort` 实现队列间动态资源共享
- **配额管理**：`nominalQuota` 和 `borrowingLimit` 的精细化控制

**适用场景：**

- 混合 `GPU` 型号的 `AI` 训练集群
- 需要资源弹性共享的多租户环境
- 成本敏感的资源利用率优化

### 3.3 广泛的工作负载集成

**原生支持的工作负载类型：**

| 类别 | 支持的工作负载 | 典型使用场景 |
|------|---------------|-------------|
| **Kubernetes 原生** | `Job`、`CronJob` | 批处理任务、定时任务 |
| **机器学习训练** | `TFJob`、`PyTorchJob`、`XGBoostJob`、`PaddleJob`、`JAXJob` | 分布式深度学习训练 |
| **高性能计算** | `MPIJob`（v2beta1） | 科学计算、仿真任务 |
| **分布式计算** | `RayJob`、`RayCluster` | 大数据处理、强化学习 |
| **作业编排** | `JobSet` | 复杂的多阶段作业流水线 |
| **服务工作负载** | `Deployment`、`StatefulSet`、`LeaderWorkerSet` | 长期运行的服务 |
| **自定义工作负载** | `AppWrapper`、普通 `Pod` | 特殊业务需求的包装 |

**集成优势：**

- 无需修改现有作业定义，仅需添加队列标签
- 统一的资源配额和调度策略
- 与 `Kubernetes` 生态无缝集成

### 3.4 高级调度特性

**差异化功能：**

#### 3.4.1 部分准入（Partial Admission）

- **功能**：当资源不足时，允许作业以降低的并行度运行
- **场景**：大规模训练任务的资源弹性调整

#### 3.4.2 动态资源回收

- **功能**：`Pod` 完成后立即释放配额，供其他作业使用
- **场景**：提高资源利用率，减少资源浪费

#### 3.4.3 全有或全无调度

- **功能**：基于超时机制的 `Gang Scheduling`
- **场景**：分布式训练任务的协调启动

#### 3.4.4 拓扑感知调度

- **功能**：优化跨节点、跨机架的 `Pod` 通信
- **场景**：高带宽要求的 `HPC` 和 `AI` 工作负载

### 3.5 企业级运维特性

#### 3.5.1 准入检查（AdmissionChecks）

**功能：**

- 自定义准入逻辑和外部系统集成
- 与 `cluster-autoscaler` 的 `provisioningRequest` 集成
- 支持资源预配置和动态扩容

**使用场景：**

- 成本控制和预算管理
- 合规性检查和安全策略
- 与企业 `ITSM` 系统集成

#### 3.5.2 监控与可观测性

**功能：**

- 丰富的 `Prometheus` 指标体系
- `KueueViz` 可视化管理界面
- 实时队列状态和作业监控

**监控维度：**

- 队列长度、等待时间、资源利用率
- 作业成功率、失败原因分析
- 多租户资源分配和使用统计

### 3.6 多集群资源编排

**MultiKueue 功能：**

- **跨集群作业分发**：智能选择最优执行集群
- **全局资源视图**：统一管理多个 `Kubernetes` 集群
- **故障转移**：集群故障时自动迁移作业

**适用场景：**

- 混合云和多云环境的资源统一调度
- 地理分布式的计算资源管理
- 成本优化的跨区域作业调度

### 3.7 核心竞争优势总结

| 特性 | `Kueue` 优势 | 竞争对手对比 |
|------|-----------|-------------|
| **Kubernetes 原生** | 无需替换调度器，渐进式采用 | `Volcano`、`YuniKorn` 需要替换默认调度器 |
| **简单易用** | 最小化配置，学习曲线平缓 | `Slurm` 配置复杂，运维成本高 |
| **生态集成** | 与 `K8s` 生态深度集成 | 传统 `HPC` 调度器生态割裂 |
| **多租户支持** | 原生多租户设计 | 部分方案多租户能力有限 |
| **资源效率** | 动态借用和回收机制 | 静态资源分配效率较低 |

---

## 4. 安装与使用

### 4.1 系统要求

**基础环境要求：**

- **Kubernetes 版本**：1.29 或更新版本
- **节点资源**：至少 2 个工作节点，每节点 2 核 4GB 内存
- **网络要求**：集群网络正常，支持 Pod 间通信
- **权限要求**：集群管理员权限，能够创建 CRD 和 RBAC 资源

**可选组件：**

- **Prometheus**：用于监控和指标收集
- **Grafana**：用于可视化监控面板
- **Cert-manager**：用于自动证书管理（如需 HTTPS）

### 4.2 快速安装

**安装最新稳定版本：**

```bash
# 安装 Kueue v0.13.2（当前稳定版本）
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.13.2/manifests.yaml
```

**验证安装：**

```bash
# 检查 Kueue 控制器状态
kubectl get pods -n kueue-system

# 验证 CRD 安装
kubectl get crd | grep kueue

# 检查控制器日志
kubectl logs -n kueue-system deployment/kueue-controller-manager
```

控制器将在 `kueue-system` 命名空间中运行，包含以下组件：

- **kueue-controller-manager**：核心控制器，负责队列管理和作业调度
- **kueue-webhook**：准入控制器，负责资源验证和默认值设置

### 4.3 基本使用示例

#### 4.3.1 创建基本队列设置

**步骤说明：**

1. **ResourceFlavor**：定义资源类型和节点选择策略
2. **ClusterQueue**：创建集群级资源队列，设置配额限制
3. **LocalQueue**：在命名空间中创建本地队列，连接到集群队列

**配置文件：**

```yaml
# ResourceFlavor：定义默认资源类型
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: default-flavor
spec:
  # 可选：指定节点标签选择器
  nodeLabels:
    node-type: "worker"
---
# ClusterQueue：定义集群级队列
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: cluster-queue
spec:
  # 命名空间选择器：匹配所有命名空间
  namespaceSelector: {}
  # 资源组配置
  resourceGroups:
  - coveredResources: ["cpu", "memory"]
    flavors:
    - name: default-flavor
      resources:
      - name: "cpu"
        nominalQuota: 9        # 9 核 CPU 配额
      - name: "memory"
        nominalQuota: 36Gi     # 36GB 内存配额
  # 队列策略：先进先出
  queueingStrategy: BestEffortFIFO
---
# LocalQueue：定义本地队列
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  namespace: default
  name: user-queue
spec:
  # 连接到集群队列
  clusterQueue: cluster-queue
```

#### 4.3.2 提交作业示例

**作业配置说明：**

- **suspend: true**：作业初始为暂停状态，由 Kueue 控制启动时机
- **queue-name 标签**：指定作业使用的队列，必须与 LocalQueue 名称匹配
- **资源请求**：定义作业所需的 CPU 和内存资源

**示例配置：**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  generateName: sample-job-
  namespace: default
  labels:
    # 关键配置：指定 Kueue 队列名称
    kueue.x-k8s.io/queue-name: user-queue
    # 可选：添加应用标签便于管理
    app: sample-workload
spec:
  # 并行度：同时运行的 Pod 数量
  parallelism: 3
  # 完成数：需要成功完成的 Pod 数量
  completions: 3
  # 重要：作业初始为暂停状态，由 Kueue 管理
  suspend: true
  template:
    metadata:
      labels:
        app: sample-workload
    spec:
      # 重启策略：失败时不重启 Pod
      restartPolicy: Never
      containers:
      - name: dummy-job
        image: registry.k8s.io/e2e-test-images/agnhost:2.53
        command: ["/bin/sh"]
        args: ["-c", "echo 'Job started'; sleep 60; echo 'Job completed'"]
        resources:
          requests:
            cpu: "1"           # 请求 1 核 CPU
            memory: "200Mi"    # 请求 200MB 内存
          limits:
            cpu: "2"           # 限制最大 2 核 CPU
            memory: "400Mi"    # 限制最大 400MB 内存
```

**提交和监控作业：**

```bash
# 提交作业
kubectl apply -f sample-job.yaml

# 查看作业状态
kubectl get jobs -w

# 查看队列状态
kubectl get localqueue user-queue -o yaml

# 查看 Workload 对象
kubectl get workloads

# 查看作业日志
kubectl logs job/sample-job-xxxxx
```

---

## 5. 与业界其他批处理调度器的对比

在 Kubernetes 批处理调度领域，Kueue 面临着来自多个成熟解决方案的竞争。本章将从技术架构、功能特性、易用性、生态集成等维度，对主要竞争对手进行客观分析，帮助用户根据实际需求做出合适的技术选型。

**对比维度说明：**

- **技术架构**：调度器设计、与 Kubernetes 的集成方式
- **功能特性**：支持的调度算法、资源管理能力、高级特性
- **易用性**：学习曲线、配置复杂度、运维成本
- **生态集成**：与现有工具链的兼容性、社区支持

### 5.1 Apache YuniKorn

**项目概述：**

Apache YuniKorn 是一个通用资源调度器，旨在为 Kubernetes 和 Apache Hadoop YARN 提供统一的调度语义。该项目由 Cloudera 主导开发，现已成为 Apache 顶级项目。

**技术架构：**

- **调度器替换**：完全替换 Kubernetes 默认调度器
- **双环境支持**：同时支持 Kubernetes 和 YARN 集群管理
- **分层队列**：提供层次化的队列管理和资源分配
- **插件化设计**：支持自定义调度策略和资源管理插件

**核心优势：**

- **统一调度语义**：为混合云环境提供一致的调度体验
- **自动资源预留**：智能预留资源，减少作业等待时间
- **强大的公平性保证**：支持多种公平性算法，确保资源公平分配
- **成熟的多租户支持**：完善的租户隔离和配额管理

**主要限制：**

- **部署复杂度高**：需要替换默认调度器，对集群侵入性强
- **学习曲线陡峭**：配置复杂，需要深入理解调度原理
- **生态集成度低**：与 Kubernetes 原生工具链兼容性有限
- **运维成本高**：需要专门的运维团队维护调度器

### 5.2 Volcano

**项目概述：**

Volcano 是华为开源的 Kubernetes 批处理调度器，专为高性能计算（HPC）和 AI/ML 工作负载设计。该项目是 CNCF 的孵化项目，在深度学习和科学计算领域有较好的应用实践。

**技术架构：**

- **自定义调度器**：替换 Kubernetes 默认调度器，提供专门的批处理调度能力
- **Gang 调度**：支持多 Pod 同时调度，确保分布式作业的完整性
- **作业依赖管理**：支持复杂的作业依赖关系和工作流调度
- **多种调度算法**：提供公平共享、优先级、回填等多种调度策略

**核心优势：**

- **先进的调度算法**：针对 HPC 和 AI 工作负载优化，支持复杂调度需求
- **Gang 调度支持**：确保分布式训练任务的所有 Pod 同时启动
- **作业依赖管理**：支持 DAG 工作流，适合复杂的科学计算场景
- **AI/ML 社区认可**：在深度学习领域有较多成功案例

**主要限制：**

- **调度器替换**：需要替换默认调度器，对集群侵入性强
- **配置复杂度高**：调度策略配置复杂，需要专业知识
- **通用性不足**：主要针对 HPC 和 AI 场景，通用批处理支持有限
- **运维成本高**：需要深入了解 HPC 调度原理进行运维

### 5.3 Slurm

**项目概述：**

Slurm（Simple Linux Utility for Resource Management）是传统 HPC 领域最广泛使用的集群管理和作业调度系统。该项目由劳伦斯利弗莫尔国家实验室开发，在全球超过一半的 Top 500 超级计算机中使用，拥有超过 20 年的发展历史。

**技术架构：**

- **独立集群管理**：提供完整的集群资源管理解决方案
- **分布式架构**：支持大规模集群的分布式管理
- **确定性调度**：提供精确的资源控制和调度保证
- **传统 HPC 模型**：基于作业提交和队列管理的经典模式

**核心优势：**

- **成熟稳定**：在 HPC 领域有超过 20 年的发展历史和丰富实践
- **强大的批处理能力**：支持复杂的作业依赖、资源预留和回收机制
- **原生 MPI 集成**：与科学计算框架深度集成，适合传统 HPC 应用
- **大规模支持**：能够管理数万节点的超大规模集群

**主要限制：**

- **云原生不兼容**：与 Kubernetes 生态完全割裂，无法利用云原生优势
- **用户体验传统**：缺乏现代化界面和 API，学习曲线陡峭
- **容器化支持弱**：主要面向传统 HPC 应用，容器化支持有限
- **现代化程度低**：缺乏云原生特性和现代运维工具支持

### 5.4 商业解决方案

**NVIDIA Run:ai：**

- **定位**：企业级 AI 工作负载和 GPU 编排平台，专注于 AI/ML 场景
- **优势**：动态资源分配、AI 原生调度、混合云支持、企业级功能
- **限制**：商业许可成本高、主要针对 GPU 工作负载

**IBM Spectrum LSF：**

- **定位**：企业级工作负载管理平台，提供全面的作业调度和资源管理
- **优势**：高级调度算法、企业级支持、丰富的管理功能
- **限制**：成本高昂、与云原生生态集成度低

**Altair PBS Professional：**

- **定位**：高性能计算工作负载管理器，专注于科学计算和研究场景
- **优势**：复杂调度策略支持、在学术界广泛应用、稳定可靠
- **限制**：商业许可成本、云原生支持不足

### 5.5 Kueue 的竞争优势

基于以上对比分析，Kueue 在 Kubernetes 批处理调度领域具有以下核心竞争优势：

**1. Kubernetes 原生设计**:

- **"作业优先，Pod 原生"设计理念**：通过 `spec.suspend` 字段管理作业生命周期，让默认调度器处理 Pod 调度
- **最小化侵入性**：无需替换现有调度器，对集群架构影响最小
- **渐进式采用路径**：支持逐步迁移，降低技术风险和运维成本
- **深度生态集成**：与 Kubernetes 原生 API 和工具链无缝协作

**2. 简单易用的运维体验**:

- **零调度器替换**：保持现有集群架构不变，降低部署风险
- **标准 Kubernetes API**：使用熟悉的 kubectl 和 YAML 配置方式
- **平缓学习曲线**：基于 Kubernetes 基础知识，无需额外的专业技能
- **低运维成本**：减少专门的调度器运维工作，降低人力成本

**3. 灵活的资源管理能力**:

- **多维度资源支持**：CPU、内存、GPU、自定义资源的统一管理
- **层次化队列设计**：支持复杂的组织架构和资源分配策略
- **智能公平共享**：基于权重的公平性算法，确保资源合理分配
- **抢占机制**：支持优先级抢占，提高资源利用效率

**4. 企业级生产就绪**:

- **官方项目保障**：由 Kubernetes SIG Scheduling 维护，质量和稳定性有保障
- **活跃社区支持**：Google、Red Hat、CERN 等知名机构的生产实践
- **持续演进发展**：紧跟 Kubernetes 发展趋势，功能持续完善
- **广泛工具集成**：与 Argo Workflows、Kubeflow、Ray 等主流工具良好集成

### 5.6 技术选型建议

基于不同的业务需求和技术环境，以下是各种批处理调度解决方案的适用场景建议：

#### 5.6.1 选择 Kueue 的场景

**核心适用场景：**

- **云原生环境**：已有 Kubernetes 集群，需要原生的批处理作业管理
- **渐进式迁移**：希望最小化对现有集群架构的影响
- **运维简化**：重视简单性和易维护性，团队 Kubernetes 经验丰富
- **生态集成**：需要与 Kubernetes 生态工具深度集成
- **多租户管理**：需要灵活的队列管理和资源配额控制

**典型用户画像：**

- 云原生技术团队
- 中小型 AI/ML 团队
- 需要批处理能力的微服务架构
- 重视技术债务控制的企业

#### 5.6.2 选择竞争对手的场景

**Apache YuniKorn：**

- **混合环境管理**：需要统一管理 Kubernetes 和 YARN 环境
- **大数据场景**：Spark、Hadoop 等大数据工作负载为主
- **复杂调度需求**：需要高级调度算法和资源优化

**Volcano：**

- **HPC 工作负载**：复杂的科学计算和高性能计算场景
- **Gang 调度需求**：分布式训练任务需要同时启动多个 Pod
- **作业依赖管理**：复杂的工作流和作业依赖关系

**Slurm：**

- **传统 HPC 环境**：现有 Slurm 基础设施，需要与科学计算工具集成
- **超大规模集群**：数万节点的超算中心
- **确定性调度**：对调度精确性有极高要求

**商业解决方案：**

- **企业级支持**：需要 7x24 技术支持和 SLA 保障
- **预算充足**：有足够的软件许可预算
- **合规要求**：需要商业软件的合规性保障

### 5.7 典型应用场景

以下是 Kueue 在实际生产环境中的几个典型应用场景：

#### 5.7.1 机器学习模型训练平台

**场景描述：**

某 AI 公司需要为多个研发团队提供统一的机器学习训练平台，支持 `PyTorch`、`TensorFlow` 等多种框架的分布式训练任务。

**传统痛点：**

- GPU资源争抢激烈，缺乏公平的分配机制
- 任务排队无序，重要项目可能被阻塞
- 资源利用率低，GPU空闲时无法被其他团队使用
- 缺乏多租户隔离，资源配额难以管理
- 任务失败后重启困难，影响训练效率

**Kueue 解决方案与好处：**

- **公平资源分配**：通过 ClusterQueue 为不同团队设置GPU配额，确保资源公平分配
- **智能排队调度**：基于优先级和公平性算法，确保重要任务优先执行
- **资源借用机制**：空闲资源可被其他队列借用，提高整体利用率
- **多租户隔离**：LocalQueue 提供项目级别的资源管理和隔离
- **自动重试机制**：任务失败后自动重新排队，提高训练成功率

**配置示例：**

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: ml-training-queue
spec:
  cohort: ml-cohort
  preemption:
    withinClusterQueue: LowerPriority
  resourceGroups:
  - coveredResources: ["nvidia.com/gpu", "cpu", "memory"]
    flavors:
    - name: gpu-a100
      resources:
      - name: "nvidia.com/gpu"
        nominalQuota: 32
        borrowingLimit: 16  # 允许借用额外资源
      - name: "cpu"
        nominalQuota: 256
      - name: "memory"
        nominalQuota: 2048Gi
```

#### 5.7.2 科学计算批处理系统

**场景描述：**

某研究机构需要处理大量的科学计算任务，包括基因测序、气候模拟、物理仿真等，这些任务具有不同的资源需求和优先级。

**传统痛点：**

- 计算任务资源需求差异巨大，难以统一调度
- 多节点并行任务启动不同步，导致资源浪费
- 长时间运行任务占用资源，影响其他任务执行
- 缺乏有效的资源共享机制，部门间资源孤岛严重
- 任务监控困难，无法及时发现和处理异常

**Kueue 解决方案与好处：**

- **灵活资源管理**：Cohort 机制实现多个研究组间的动态资源共享
- **Gang Scheduling**：确保多节点并行任务同时启动，避免部分节点等待
- **智能抢占策略**：低优先级长任务可被高优先级任务抢占，提高响应速度
- **资源借用优化**：空闲计算资源自动分配给有需求的研究组
- **完整监控体系**：集成 Prometheus 监控，实时跟踪任务状态和资源使用

**关键特性：**

- 支持长时间运行的计算任务
- 自动处理节点故障和任务重启
- 与 HPC 调度器（如 Slurm）的迁移兼容性

#### 5.7.3 CI/CD 构建农场

**场景描述：**

大型软件公司需要为数百个项目提供持续集成和部署服务，每天处理数千个构建任务，需要高效的资源调度和隔离。

**传统痛点：**

- 构建任务排队时间长，影响开发效率
- 资源分配不均，热门项目抢占大量资源
- 紧急修复无法快速构建，影响生产发布
- 构建资源利用率低，夜间和周末资源闲置
- 缺乏构建任务的可视化管理和监控

**Kueue 解决方案与好处：**

- **动态资源分配**：根据项目活跃度和提交频率自动调整资源配额
- **优先级管理**：生产修复和发布任务获得最高优先级，确保快速执行
- **弹性扩缩容**：构建高峰期自动扩展资源，低峰期回收节约成本
- **多项目隔离**：不同项目和分支使用独立队列，避免相互影响
- **智能调度**：基于历史数据预测构建时间，优化任务排序

**技术实现：**

- 为不同项目组创建隔离的队列
- 使用资源配额限制每个项目的资源使用
- 配置动态扩缩容，根据构建负载自动调整集群规模
- 集成 Tekton 或 Argo Workflows 实现完整的 CI/CD 流水线

#### 5.7.4 多租户数据处理平台

**场景描述：**

云服务提供商需要为客户提供数据处理服务，支持 Spark、Flink 等大数据框架，同时确保租户间的资源隔离和公平性。

**传统痛点：**

- 租户间资源争抢，服务质量难以保证
- 缺乏有效的资源隔离机制，存在安全风险
- 资源使用统计不准确，计费困难
- 无法根据租户等级提供差异化服务
- 资源超售风险高，可能导致服务中断

**Kueue 解决方案与好处：**

- **严格租户隔离**：基于 Namespace 和 LocalQueue 实现完全的资源和权限隔离
- **分级服务保障**：不同等级租户享受不同的资源配额和优先级
- **精确计费统计**：详细记录每个租户的资源使用情况，支持精确计费
- **弹性资源管理**：支持资源超售和动态调整，提高平台收益
- **SLA 保障**：通过优先级和抢占机制确保高级租户的服务质量

**技术实现：**

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: tenant-a-queue
spec:
  namespaceSelector:
    matchLabels:
      tenant: "tenant-a"
  admissionChecks:
  - billing-check
  - quota-validation
  resourceGroups:
  - coveredResources: ["cpu", "memory"]
    flavors:
    - name: standard-nodes
      resources:
      - name: "cpu"
        nominalQuota: 100
        borrowingLimit: 50
```

#### 5.7.5 边缘计算任务调度

**场景描述：**

物联网公司需要在边缘节点上运行数据处理和 AI 推理任务，这些节点资源有限且网络不稳定。

**传统痛点：**

- 边缘节点资源有限且异构，调度复杂度高
- 网络不稳定，任务执行可靠性差
- 数据传输成本高，缺乏本地性优化
- 节点故障频繁，任务容错能力不足
- 缺乏统一的边缘资源管理和监控

**Kueue 解决方案与好处：**

- **异构资源统一管理**：支持 CPU、GPU、FPGA 等多种计算资源的统一调度
- **地理位置感知调度**：基于节点标签实现就近调度，减少数据传输延迟
- **智能容错机制**：任务失败自动迁移到其他可用节点，提高执行成功率
- **混合负载优化**：离线批处理和在线服务任务合理混部，提高资源利用率
- **边缘集群联邦**：多个边缘集群统一管理，实现全局资源优化

**特殊考虑：**

- 网络带宽限制和延迟优化的调度策略
- 设备功耗管理和节点可靠性评估
- 数据本地性优化，减少跨节点数据传输
- 离线模式支持和断网续传机制

---

## 6. Kueue + HAMi 集成方案

### 6.1 架构概述

在现代 `AI` 基础设施中，`Kueue` 与 `HAMi`（`Heterogeneous AI Computing Virtualization Middleware`）的结合为企业提供了一套完整的 `AI` 工作负载管理解决方案。这种组合实现了从资源虚拟化到作业调度的全栈管理能力，为 `Kubernetes` 环境下的 `AI` 工作负载提供了强大的调度和资源管理能力。

#### 6.1.1 应用场景

`Kueue` 与 `HAMi` 的集成特别适用于以下 `AI` 工作负载场景：

**机器学习训练场景：**

- 大规模深度学习模型训练
- 分布式训练任务的资源协调
- GPU 资源的高效利用和共享
- 多租户环境下的资源隔离

**AI 推理服务场景：**

- 高并发推理任务的批处理
- GPU 算力的精细化分配
- 推理服务的弹性扩缩容
- 实时和离线推理的混合调度

**科研计算场景：**

- 科学计算和仿真任务
- 多用户共享 GPU 集群
- 长时间运行任务的资源管理
- 实验任务的优先级调度

#### 6.1.2 分层架构设计

**整体架构：**

```text
┌─────────────────────────────────────────────────────────┐
│                   AI 应用层                              │
│  (PyTorch, TensorFlow, Jupyter Notebook, MLflow)        │
├─────────────────────────────────────────────────────────┤
│                  Kueue 队列管理层                         │
│  • 作业队列管理    • 资源配额控制    • 公平调度              │
│  • 多租户隔离      • 优先级管理      • 抢占策略              │
├─────────────────────────────────────────────────────────┤
│                  HAMi 调度与虚拟化层                      │
│  • GPU 智能调度    • GPU 虚拟化      • 算力隔离            │
│  • 算力分片        • 设备共享        • 资源监控             │
├─────────────────────────────────────────────────────────┤
│                Kubernetes 调度层                         │
│  • 节点选择        • 亲和性调度      • 资源约束             │
└─────────────────────────────────────────────────────────┘
```

#### 6.1.3 HAMi 组件架构

`HAMi` 作为异构 `AI` 计算虚拟化中间件，由以下核心项目组成：

**核心组件：**

- **HAMi 主项目**：包含 `Scheduler Extender`（智能调度器）和 `Mutating Webhook`（准入控制器）
- **HAMi-WebUI**：提供 `GPU` 资源管理和可观测性的 `Web` 界面
- **HAMi-core**：负责容器内 `GPU` 算力切分和资源隔离的核心库
- **设备插件项目**：针对不同 `GPU` 厂商的 `Device Plugin` 实现，如 `ascend-device-plugin` 等

**组件职责：**

- **调度层**：`HAMi Scheduler Extender` 负责 `GPU` 资源的智能分配和优化
- **准入层**：`HAMi` Webhook 负责 Pod 创建时的资源转换和配置注入
- **运行时层**：`HAMi-core` 负责容器内的 `GPU` 资源隔离和监控
- **管理层**：`HAMi-WebUI` 提供可视化的资源管理和监控界面

#### 6.1.4 协作架构模式

Kueue 与 HAMi 的结合专门针对 AI 批处理工作负载的管理与调度需求，形成了分层协作的架构模式：

**分层职责：**

- **Kueue 层**：负责作业队列管理、资源配额控制和多租户隔离
- **HAMi 层**：负责 GPU 资源虚拟化、智能调度和设备共享
- **协作机制**：通过资源抽象映射和调度策略协调，实现高效的 AI 批处理负载管理

**协作优势：**

- **资源利用率提升**：HAMi 的 GPU 虚拟化技术提高硬件利用率
- **调度效率优化**：Kueue 的队列管理确保任务有序执行
- **多租户支持**：完善的资源隔离和配额管理机制
- **运维简化**：统一的管理界面和监控体系

```text
┌─────────────────────────────────────────────────────────┐
│                    HAMi 架构图                           │
├─────────────────┬───────────────────┬───────────────────┤
│  HAMi Webhook   │  HAMi Scheduler   │   HAMi-WebUI      │
│   (准入控制)     │   (智能调度)       │ (GPU资源监控界面)    │
├─────────────────┼───────────────────┼───────────────────┤
│              HAMi Device Plugin   HAMi-core             │
│                (设备管理)           (算力切分)             │
├─────────────────────────────────────────────────────────┤
│                GPU Driver & Runtime                     │
│              (NVIDIA, AMD, 昇腾等硬件驱动)                │
└─────────────────────────────────────────────────────────┘
```

### 6.2 核心工作机制

#### 6.2.1 资源抽象与映射

Kueue 与 HAMi 的集成通过资源抽象层实现了逻辑资源配额与物理 GPU 资源的映射关系。

**HAMi 物理资源定义：**

HAMi 在 Pod 规格中定义实际的 GPU 硬件资源需求：

```yaml
# HAMi 设备资源（实际硬件层面）
resources:
  limits:
    nvidia.com/gpu: 2              # 请求 2 个 vGPU
    nvidia.com/gpumem: 6000         # 每个 vGPU 6GB 显存
    nvidia.com/gpucores: 50         # GPU 算力占用 50%
```

**Kueue 逻辑配额定义：**

Kueue 在 ClusterQueue 中定义逻辑资源配额，实现资源的抽象管理：

```yaml
# Kueue 逻辑配额（在 ClusterQueue 中定义）
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: ai-training-queue
spec:
  resourceGroups:
  - coveredResources: ["nvidia.com/gpu", "nvidia.com/gpumem", "nvidia.com/gpucores"]
    flavors:
    - name: nvidia-gpu-flavor
      resources:
      - name: "nvidia.com/gpu"
        nominalQuota: 32     # GPU 卡数配额
      - name: "nvidia.com/gpumem"
        nominalQuota: 256000 # 显存配额 (MB)
      - name: "nvidia.com/gpucores"
        nominalQuota: 1600   # GPU 算力配额 (%)
```

**资源映射关系：**

| HAMi 物理资源 | Kueue 逻辑配额 | 映射说明 |
|--------------|---------------|----------|
| nvidia.com/gpu | nvidia.com/gpu | 1:1 映射，GPU 卡数 |
| nvidia.com/gpumem | nvidia.com/gpumem | 1:1 映射，显存大小 (MB) |
| nvidia.com/gpucores | nvidia.com/gpucores | 1:1 映射，算力百分比 |

#### 6.2.2 双层调度流程详解

Kueue 与 HAMi 的集成采用双层调度架构，实现了从逻辑队列管理到物理资源分配的完整调度链路。这种设计将队列管理与设备调度解耦，提供了更高的灵活性和扩展性。

##### 调度流程概述

**第一层：Kueue 队列调度**:

- 负责作业队列管理、资源配额控制和准入决策
- 实现多租户资源隔离和公平调度策略
- 提供作业优先级管理和抢占机制

**第二层：HAMi 设备调度**:

- 负责 GPU 等异构设备的智能分配和虚拟化
- 实现设备级别的拓扑感知和负载均衡
- 提供细粒度的资源隔离和监控能力

##### 详细调度阶段

**阶段一：作业提交与队列管理**:

```text
用户作业 → LocalQueue → Workload 对象 → 资源需求验证
```

- 用户通过 `kueue.x-k8s.io/queue-name` 标签将作业提交到指定的 LocalQueue
- Kueue 控制器自动创建 Workload 对象，封装作业的资源需求和调度信息
- 系统验证作业的资源请求格式和配额限制，确保请求的合理性

**阶段二：配额准入与优先级评估**:

```text
Workload → ClusterQueue → 配额检查 → 优先级排序 → 准入决策
```

- Kueue 根据 ClusterQueue 配置验证可用资源配额，包括 GPU、内存、算力等维度
- 应用队列优先级、公平共享策略和借用机制，确定作业的调度优先级
- 基于资源可用性和策略约束，做出准入决策（准入/排队/拒绝）

**阶段三：Kubernetes 调度与 HAMi 介入**:

```text
准入通过 → Pod 创建 → Kubernetes 调度器 → HAMi 扩展调度 → 节点选择
```

- Kueue 准入通过后，释放 Pod 到 Kubernetes 进行实际调度
- HAMi Scheduler Extender 作为调度器扩展介入，执行 GPU 专用的调度逻辑
- 考虑 GPU 拓扑结构、内存碎片化、设备利用率等因素进行最优节点选择
- HAMi Mutating Webhook 自动注入 GPU 配置和环境变量

**阶段四：资源分配与运行时管理**:

```text
Pod 调度 → 设备分配 → 资源隔离 → 运行时监控
```

- HAMi 设备插件完成具体的 GPU 设备分配和虚拟化配置
- HAMi-core 在容器运行时实现显存和算力的精确隔离
- HAMi-WebUI 提供实时的资源使用监控和管理界面
- Kueue 持续跟踪作业状态，支持动态抢占和资源回收

#### 6.2.3 设备插件项目

设备插件项目是 HAMi 生态系统的基础组件，负责异构计算设备的发现、注册和虚拟化管理。通过标准的 Kubernetes Device Plugin API，实现对不同厂商 GPU/NPU 设备的统一管理。

##### 核心功能特性

**设备发现与注册**:

- 自动扫描节点上的异构计算设备（NVIDIA GPU、华为昇腾 NPU、AMD GPU 等）
- 向 Kubelet 注册设备资源，使其在 Kubernetes 集群中可见和可调度
- 支持热插拔设备的动态发现和注册更新

**资源虚拟化管理**:

- 将物理设备虚拟化为多个逻辑设备，实现设备共享
- 支持显存、算力、带宽等多维度资源的细粒度分割
- 提供灵活的虚拟化策略配置，适应不同应用场景需求

**设备分配与隔离**:

- 响应 Pod 的设备资源请求，执行设备分配决策
- 生成设备挂载配置和环境变量注入
- 确保不同容器间的设备资源隔离和安全访问

##### 多厂商设备支持

**NVIDIA GPU 支持**:

```yaml
# NVIDIA GPU 资源配置示例
resources:
  nvidia.com/gpu: "1"           # GPU 设备数量
  nvidia.com/gpumem: "8192"     # GPU 显存 (MB)
  nvidia.com/gpucores: "50"     # GPU 算力百分比
```

**华为昇腾 NPU 支持**:

```yaml
# 昇腾 NPU 资源配置示例
resources:
  huawei.com/Ascend910B: "1"        # 昇腾 NPU 数量
  huawei.com/Ascend910B-memory: "4096"  # NPU 显存 (MB)
  huawei.com/Ascend910B-cores: "30"     # NPU 算力百分比
```

**AMD GPU 支持**:

```yaml
# AMD GPU 资源配置示例
resources:
  amd.com/gpu: "1"              # AMD GPU 设备数量
  amd.com/gpumem: "16384"       # GPU 显存 (MB)
```

#### 6.2.4 HAMi 调度器扩展

HAMi 调度器扩展是核心调度组件，通过扩展 Kubernetes 默认调度器，实现对异构计算资源的智能感知和优化分配。主要提供拓扑感知调度、智能碎片整理、负载均衡优化等能力。与 Kueue 协作时，Kueue 负责作业队列管理和资源配额控制，HAMi 负责 GPU 设备的具体分配和优化。

#### 6.2.5 HAMi Mutating Webhook

HAMi Mutating Webhook 是 Kubernetes 准入控制器的重要组成部分，在 Pod 创建阶段自动执行 GPU 资源配置的转换和注入。主要功能包括资源请求转换、环境变量注入（如 CUDA_VISIBLE_DEVICES）、容器配置修改等，确保用户提交的 GPU 资源请求能够正确转换为 HAMi 运行时所需的配置格式。

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
    nvidia.com/gpumem: "8192"
    nvidia.com/gpucores: "50"
```

**Webhook 注入后的配置**：

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
env:
- name: CUDA_VISIBLE_DEVICES
  value: "0"
- name: CUDA_DEVICE_MEMORY_LIMIT
  value: "8192"
- name: CUDA_DEVICE_SM_LIMIT
  value: "50"
- name: LD_PRELOAD
  value: "/usr/local/hami/libvgpu.so"
volumeMounts:
- name: hami-lib
  mountPath: /usr/local/hami
- name: nvidia-dev
  mountPath: /dev/nvidia0
```

#### 6.2.6 HAMi-core - 算力切分与隔离

HAMi-core 是容器内 GPU 资源控制的核心库，通过劫持 CUDA API 调用实现硬件资源的精确控制。主要功能包括显存虚拟化、算力限制、实时监控等，确保不同容器间的 GPU 资源隔离和精确分配。

#### 6.2.7 HAMi-WebUI - 管理与监控界面

HAMi-WebUI 提供直观的 Web 界面用于 GPU 资源的管理和监控，包括资源概览、节点管理、GPU 管理、任务管理等功能，帮助用户可视化集群资源使用情况和任务状态。

### 6.3 配置实践

本节将通过一个完整的AI训练场景，展示如何配置Kueue与HAMi的集成环境。我们将构建一个支持多团队、多GPU类型的AI训练平台，实现资源的高效调度和管理。

#### 6.3.1 场景描述

**业务场景：**
某AI公司拥有一个混合GPU集群，包含V100和A100两种GPU类型，需要支持多个研发团队进行模型训练。要求实现：

- **资源隔离**：不同团队拥有独立的资源配额，互不干扰，确保资源的公平分配。
- **GPU共享**：支持GPU虚拟化，提高资源利用率，避免资源浪费。
- **优先级调度**：生产任务优先，研发任务可被抢占，确保生产环境的及时响应。
- **弹性扩展**：支持资源借用机制，当集群资源不足时，允许临时借用其他团队的资源。

**集群环境：**

- 节点配置：4台V100节点（每台8卡32GB），2台A100节点（每台8卡80GB）
- 团队配置：ai-research团队（研发）、ai-production团队（生产）
- 资源需求：支持大模型训练、推理服务、实验任务

#### 6.3.2 HAMi 基础配置

**HAMi 设备插件配置：**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hami-device-plugin-config
data:
  config.yaml: |
    resourceName: "nvidia.com/gpu"
    resourceMemoryName: "nvidia.com/gpumem"
    resourceCoreName: "nvidia.com/gpucores"
    deviceSplitCount: 10
    migStrategy: "none"
```

**GPU 节点标签配置：**

```yaml
apiVersion: v1
kind: Node
metadata:
  name: gpu-node-01
  labels:
    nvidia.com/gpu.product: "Tesla-V100-32GB"
    nvidia.com/gpu.count: "8"
    zone: "gpu-zone-a"
```

#### 6.3.3 Kueue 队列配置

本节将配置 Kueue 的核心组件，包括 ResourceFlavor（资源类型）、ClusterQueue（集群队列）和 LocalQueue（本地队列）。这些配置将实现以下功能：

1. **ResourceFlavor**：定义不同类型的 GPU 资源（V100 和 A100），指定节点标签和容忍度
2. **ClusterQueue**：创建集群级别的资源队列，设置资源配额和调度策略
3. **LocalQueue**：在特定命名空间中创建本地队列，连接到集群队列

**ResourceFlavor 定义：**

```yaml
# 定义 V100 GPU 资源类型
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: gpu-v100-32gb
spec:
  nodeLabels:
    # 指定 GPU 产品型号，用于节点选择
    nvidia.com/gpu.product: "Tesla-V100-32GB"
    # 指定可用区，实现资源隔离
    zone: "gpu-zone-a"
  tolerations:
  # 容忍 GPU 节点的污点，确保 Pod 可以调度到 GPU 节点
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
---
# 定义 A100 GPU 资源类型
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: gpu-a100-80gb
spec:
  nodeLabels:
    # 指定 GPU 产品型号，A100 性能更强
    nvidia.com/gpu.product: "Tesla-A100-80GB"
    # 不同的可用区，实现负载均衡
    zone: "gpu-zone-b"
```

**ClusterQueue 配置：**

```yaml
# 创建 AI 训练专用的集群队列
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: ai-training-queue
spec:
  # 命名空间选择器：只有标记为 ai-research 团队的命名空间可以使用此队列
  namespaceSelector:
    matchLabels:
      team: "ai-research"
  resourceGroups:
  # 定义资源组，包含 HAMi 提供的三种 GPU 资源类型
  - coveredResources: ["nvidia.com/gpu", "nvidia.com/gpumem", "nvidia.com/gpucores"]
    flavors:
    # V100 GPU 资源配置
    - name: gpu-v100-32gb
      resources:
      - name: "nvidia.com/gpu"
        nominalQuota: 32        # 基础配额：32 个 GPU
        borrowingLimit: 16      # 可借用额度：16 个 GPU
      - name: "nvidia.com/gpumem"
        nominalQuota: 1024000   # 1TB 显存配额（32GB × 32 = 1024GB）
      - name: "nvidia.com/gpucores"
        nominalQuota: 1600      # 1600% GPU 算力（50% × 32 = 1600%）
    # A100 GPU 资源配置
    - name: gpu-a100-80gb
      resources:
      - name: "nvidia.com/gpu"
        nominalQuota: 16        # 基础配额：16 个 GPU
      - name: "nvidia.com/gpumem"
        nominalQuota: 1280000   # 1.25TB 显存配额（80GB × 16 = 1280GB）
      # A100 不设置 gpucores 限制，允许全性能使用
  # 队列策略：尽力而为的先进先出
  queueingStrategy: BestEffortFIFO
  # 抢占策略配置
  preemption:
    # 在同一 cohort 内可以抢占低优先级任务
    reclaimWithinCohort: LowerPriority
    # 在同一队列内可以抢占低优先级任务
    withinClusterQueue: LowerPriority
```

**LocalQueue 配置：**

```yaml
# 研发环境的本地队列
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: research-queue
  namespace: ai-research    # 部署在研发命名空间
spec:
  clusterQueue: ai-training-queue  # 连接到 AI 训练集群队列
---
# 生产环境的本地队列
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: production-queue
  namespace: ai-production  # 部署在生产命名空间
spec:
  clusterQueue: ai-training-queue  # 共享同一个集群队列资源
```

#### 6.3.4 命名空间和优先级配置

本节配置支持多租户环境的命名空间和任务优先级管理，实现以下功能：

1. **命名空间隔离**：为不同团队创建独立的命名空间，实现资源和权限隔离
2. **优先级管理**：定义不同优先级类别，确保重要任务优先调度
3. **标签管理**：通过标签实现命名空间与队列的关联

**命名空间配置：**

```yaml
# AI 研发团队命名空间
apiVersion: v1
kind: Namespace
metadata:
  name: ai-research
  labels:
    # 团队标识，用于 ClusterQueue 的 namespaceSelector
    team: "ai-research"
    # 环境标识，便于管理和监控
    environment: "development"
    # 资源类型标识
    workload-type: "research"
---
# AI 生产团队命名空间
apiVersion: v1
kind: Namespace
metadata:
  name: ai-production
  labels:
    # 生产团队的正确标签，与研发环境区分
    team: "ai-production"
    # 生产环境标识
    environment: "production"
    # 生产工作负载类型
    workload-type: "inference"
```

**优先级类配置：**

```yaml
# 高优先级类：用于生产环境的关键任务
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority
# 高优先级值，确保优先调度
value: 1000
# 不设为全局默认，避免所有任务都使用高优先级
globalDefault: false
description: "高优先级任务，用于生产环境的关键推理服务"
---
# 普通优先级类：用于研发环境的日常任务
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: normal-priority
# 普通优先级值
value: 100
# 设为全局默认，未指定优先级的任务使用此级别
globalDefault: true
description: "普通优先级任务，用于研发环境的训练和实验"
```

#### 6.3.5 完整任务示例

**大模型训练任务（研发环境）：**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: llm-training-research
  namespace: ai-research  # 指定研发团队的命名空间
  labels:
    # 关键配置：指定Kueue队列名称，必须与LocalQueue名称匹配
    kueue.x-k8s.io/queue-name: research-queue
spec:
  parallelism: 2    # 并行执行2个Pod，适合分布式训练
  completions: 1    # 任务完成条件：1个成功的Pod即可
  template:
    metadata:
      labels:
        app: llm-training
    spec:
      restartPolicy: Never  # Job失败时不重启Pod
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
        command: ["python", "/workspace/train.py"]
        resources:
          requests:  # 资源请求：调度器保证的最小资源
            cpu: "4"
            memory: "16Gi"
          limits:    # 资源限制：容器可使用的最大资源
            cpu: "8"
            memory: "32Gi"
            # HAMi GPU资源配置 - 关键部分
            nvidia.com/gpu: "1"        # 申请1个物理GPU
            nvidia.com/gpumem: "16000" # 申请16GB GPU显存（单位：MB）
            nvidia.com/gpucores: "50"  # 申请50%的GPU算力
        volumeMounts:
        - name: workspace
          mountPath: /workspace
      volumes:
      - name: workspace
        persistentVolumeClaim:
          claimName: training-data-pvc  # 挂载训练数据存储
      tolerations:  # 容忍GPU节点的污点，允许调度到GPU节点
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
```

**推理服务任务（生产环境）：**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-inference-prod
  namespace: ai-production  # 指定生产团队的命名空间
  labels:
    # 关键配置：指定生产环境的LocalQueue
    kueue.x-k8s.io/queue-name: production-queue
  annotations:
    # 可选配置：指定优先级类，用于Kueue调度优先级判断
    kueue.x-k8s.io/priority-class: "high-priority"
spec:
  parallelism: 4    # 并行4个Pod，提高推理吞吐量
  completions: 1    # 任务完成条件：1个成功的Pod即可
  template:
    metadata:
      labels:
        app: model-inference
    spec:
      restartPolicy: Never
      # 关键配置：Pod级别的优先级类，影响Kubernetes调度顺序
      priorityClassName: high-priority
      containers:
      - name: inference
        image: tensorflow/serving:2.13.0-gpu
        resources:
          requests:  # 推理任务通常资源需求较小
            cpu: "2"
            memory: "8Gi"
          limits:
            cpu: "4"
            memory: "16Gi"
            # HAMi GPU资源配置 - 推理优化配置
            nvidia.com/gpu: "1"       # 申请1个物理GPU
            nvidia.com/gpumem: "8000" # 申请8GB GPU显存（推理需求较小）
            nvidia.com/gpucores: "30" # 申请30%的GPU算力（推理计算量较小）
        ports:
        - containerPort: 8501  # TensorFlow Serving REST API端口
        env:
        - name: MODEL_NAME
          value: "llm-model"  # 指定要加载的模型名称
      tolerations:  # 容忍GPU节点的污点
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
```

#### 6.3.6 验证配置

**检查资源状态：**

```bash
# 检查ClusterQueue状态
kubectl get clusterqueue ai-training-queue -o yaml

# 检查LocalQueue状态
kubectl get localqueue -A

# 查看GPU节点资源
kubectl describe nodes -l nvidia.com/gpu.count

# 监控任务队列
kubectl get jobs -A --watch
```

**资源使用验证：**

```bash
# 提交测试任务
kubectl apply -f llm-training-research.yaml

# 查看任务调度状态
kubectl describe job llm-training-research -n ai-research

# 检查GPU资源分配
kubectl exec -it <pod-name> -n ai-research -- nvidia-smi
```

---

## 7. 总结

Kueue + HAMi 的组合为 Kubernetes 环境下的 AI 工作负载管理提供了一套完整、高效的解决方案。这种协作模式充分发挥了两个项目的优势，实现了从作业队列管理到 GPU 资源虚拟化的全栈覆盖。

---
