# Nvidia KAI Scheduler 功能架构解析

本文档深入解析了 NVIDIA KAI Scheduler 的核心架构与关键特性，并将其与异构计算虚拟化中间件 HAMi 进行多维度对比。通过分析两者在系统定位（调度器 vs 中间件）、资源隔离（软隔离 vs 硬隔离）及适用场景（大规模训练 vs 在线推理）上的差异，为构建高效、可靠的 AI 基础设施提供技术选型依据。

---

## 1. KAI Scheduler 架构与功能解析

KAI Scheduler 是一款专为大规模 AI 与机器学习工作负载定制的 Kubernetes 调度器。它构建于 **kube-batch** 架构体系之上，而非基于 Kubernetes 上游的 **Scheduling Framework** (即 kube-scheduler 的插件化架构)。这种架构选择使其能够采用 **Session (会话)** 模式与 **Action (动作)** 驱动机制，从而在全局视图下进行复杂的资源预留和回填（Backfill）。

KAI Scheduler 旨在突破原生 Kubernetes 调度器在高性能计算（HPC）场景下的瓶颈，重点解决了 **Gang Scheduling**（全量调度）、**细粒度资源共享**以及**多租户公平性策略**等关键问题。相比于关注单个 Pod 调度的原生框架，它更契合**批处理**（Batch）场景对全局最优解的需求。

### 1.1 架构渊源与组件交互

KAI Scheduler 的技术根基源于 **kube-batch** —— Kubernetes 社区早期的批处理调度孵化项目。理解 KAI 的架构，首先需要理解它在 Kubernetes 生态中的位置及其与核心组件的交互关系。

#### 1.1.1 kube-batch 的遗产与继承

`kube-batch` 是 Kubernetes 生态中首个引入 **PodGroup (任务组)** 概念的调度器，旨在解决 AI/HPC 场景下 "All-or-Nothing" 的调度需求。KAI Scheduler 继承并发扬了 `kube-batch` 的核心设计理念：

- **领域模型复用**: 沿用了 `Queue` (队列)、`PodGroup` (任务组) 等批处理原语，支持多租户环境下的资源配额管理。
- **架构扩展**: 在保留 kube-batch 优秀的插件化接口 (`Session-Plugin-Action`) 基础上，KAI 深度扩展了对异构硬件（GPU/NPU）的感知能力、拓扑感知调度 (TAS) 以及更复杂的抢占与公平性算法，使其从一个通用的批处理调度器进化为专用的 AI 算力调度引擎。

#### 1.1.2 核心组件交互逻辑

在运行态，KAI Scheduler 与 Kubernetes 集群组件保持着紧密而高效的交互：

1. **数据摄入 (Ingestion)**: KAI Scheduler 通过 Kubernetes 原生的 `List-Watch` 机制，实时监听 `Pod`、`Node`、`PodGroup`、`Queue` 等资源对象的变更事件。
2. **本地缓存 (Local Cache)**: 不同于直接查询 API Server，KAI 维护了一个高度优化的本地内存视图。所有监听到的事件首先更新本地缓存。这种设计大幅降低了 API Server 的负载，并允许调度器在微秒级完成成千上万次模拟调度计算。
3. **决策输出 (Binding)**: 一旦调度周期完成并产出决策，KAI Scheduler 会通过异步客户端向 API Server 发送 `Bind` 请求，将 Pod 绑定到选定的 Node，完成调度闭环。

### 1.2 核心调度流程

不同于原生 Kubernetes 调度器的事件驱动模型，KAI Scheduler 采用了周期性（Cycle-based）调度模型，通过快照机制确保了调度决策的全局一致性。其核心调度流程在一个闭环周期内完成，主要包含四个阶段：

1. **Cache Sync (缓存同步)**：实时同步 Kubernetes API Server 的资源状态至调度器内部缓存。
2. **Snapshot (快照)**：在每个调度周期伊始捕获集群状态快照。该机制有效隔离了并发修改，消除了调度计算过程中的竞态条件。
3. **Session (会话)**：基于快照构建独立的调度会话上下文，所有调度指令均在此上下文中封装执行。
4. **Actions (调度动作)**：通过插件化架构按序执行调度动作，核心动作包括：
   - `Allocate`: 执行核心资源分配逻辑。
   - `Preempt`: 触发高优先级任务对低优先级资源的抢占。
   - `Reclaim`: 基于公平性策略回收超额占用的资源。
   - `Backfill`: 利用碎片资源对小任务进行回填调度。

```go
// 调度主循环逻辑 (基于 pkg/scheduler/scheduler.go 简化)
func (s *Scheduler) Run(stopCh <-chan struct{}) {
    // 1. 启动缓存并等待同步完成
    s.cache.Run(stopCh)
    s.cache.WaitForCacheSync(stopCh)

    // 2. 周期性执行调度逻辑
    wait.Until(s.runOnce, s.schedulePeriod, stopCh)
}

func (s *Scheduler) runOnce() {
    // 3. 生成会话 ID (SessionID) 用于全链路追踪
    sessionId := generateSessionID()

    // 4. 打开会话 (Open Session): 构建快照并加载插件
    // OpenSession 会初始化插件并调用 OnSessionOpen
    ssn, err := framework.OpenSession(s.cache, s.config, sessionId)
    if err != nil {
        return
    }
    // 确保会话结束时清理资源并触发插件的 OnSessionClose
    defer framework.CloseSession(ssn)

    // 5. 按序执行配置的调度动作 (Actions)
    actions := conf_util.GetActionsFromConfig(s.config)
    for _, action := range actions {
        // 设置当前动作上下文 (用于日志与监控)
        log.SetAction(action.Name())
        metrics.SetCurrentAction(action.Name())

        // 执行具体动作 (如 Allocate, Preempt, Backfill)
        action.Execute(ssn)
    }
}
```

### 1.3 关键特性

#### 1.3.1 Gang Scheduling (PodGroups)

针对分布式训练任务（如 Parameter Server 架构），KAI Scheduler 引入了 `PodGroup` 资源对象以实现 "All-or-Nothing" 的调度语义。该特性确保任务的所有组件能够同时获得资源并启动，从根本上解决了因资源死锁导致的训练停滞问题，显著提升了集群的作业完成率。

#### 1.3.2 多级队列与公平性 (Hierarchical Queues & Fairness)

系统支持树状层级队列管理，并基于 DRF (Dominant Resource Fairness) 算法实现精细化的资源公平调度。这不仅通过应得配额（Deserved Quota）保障了队列的基础资源供给，还允许在集群空闲时进行超额借用（Over-quota）。此外，引入的时间感知公平（Time-aware Fairness）机制通过计算历史资源消耗的权重，防止特定租户长期霸占资源，从而实现了长周期维度的真正公平。

#### 1.3.3 拓扑感知调度 (Topology Aware Scheduling, TAS)

针对大模型训练对互联带宽的苛刻要求，原生集成物理拓扑感知能力（如 Rack, Zone 维度）：

- **Bin-packing (装箱策略)**: 优先将任务调度至同一拓扑域，最大化减少跨域通信。
- **Node Ordering (节点优选)**: 在选定拓扑域内，基于亲和性对节点进行排序，降低节点间通信延迟。

#### 1.3.4 GPU 共享与动态资源分配

在资源复用方面，KAI Scheduler 支持物理 GPU 的逻辑切分，允许用户通过注解声明算力比例或显存额度（Fractional Request）。其实现采用软隔离（Soft Isolation）机制，依赖应用层的协作或配合 NVIDIA MPS 技术。同时，它全面兼容 Kubernetes DRA (Dynamic Resource Allocation) 架构，为管理第三方异构硬件资源提供了标准化的扩展接口。

---

## 2. HAMi 架构与功能解析

HAMi (Heterogeneous AI Computing Virtualization Middleware) 定位为异构 AI 计算虚拟化中间件。与调度器不同，HAMi 聚焦于**设备层的虚拟化**与**异构硬件的统一管理**，旨在消除不同异构设备之间的差异，提供统一的管理接口。

### 2.1 核心架构与隔离机制

HAMi 采用 `MutatingWebhook` + `Scheduler Extender` + `Device Plugin` 的非侵入式架构，作为原生调度器的能力增强插件运行。其最显著的特性是**硬隔离 (Hard Isolation)**，通过底层库劫持（Library Interposition）等容器内虚拟化技术，实时拦截并审查 CUDA API 调用。这一机制强制实施了显存配额与算力核心限制，确保了多租户环境下的资源安全边界，有效防止了越界使用。

### 2.2 异构资源管理

在资源管理层面，HAMi 提供了极细粒度的控制能力。它支持按固定大小 (`nvidia.com/gpumem`) 或百分比 (`nvidia.com/gpumem-percentage`) 分配显存，以及按百分比分配 GPU 算力核心 (`nvidia.com/gpucores`)。

此外，HAMi 构建了统一的异构设备抽象层，除 NVIDIA GPU 外，还广泛支持华为 Ascend NPU、寒武纪 MLU、海光 DCU、天数智芯 CoreX、摩尔线程及沐曦等主流国产 AI 芯片。这种广泛的兼容性使其成为混合硬件环境下的理想管理工具。

### 2.3 调度增强

虽然作为中间件，HAMi 也提供了一系列调度增强能力以优化分配效率。这包括优先减少碎片的 Binpack 策略、实现负载均衡的 Spread 策略，以及确保 GPU 和 CPU 在同一 NUMA 节点以减少延迟的 NUMA-First 策略。它还允许用户通过注解指定所需的 GPU 型号或具体 UUID，提供了灵活的资源绑定能力。

此外，HAMi 注重系统的易用性与配置灵活性。它提供了可视化 WebUI (v2.4+) 以便管理员直观监控资源状态，并支持全局、节点级及 Pod 级的三级配置策略，允许针对不同集群区域灵活调整资源隔离与超卖策略。

---

## 3. KAI Scheduler 与 HAMi 对比分析

### 3.1 核心能力对比

| **特性维度**     | **KAI Scheduler**                                   | **HAMi**                                           |
| :--------------- | :-------------------------------------------------- | :------------------------------------------------- |
| **系统定位**     | 面向 AI/Batch 任务的**全量调度器** (Scheduler)      | 面向设备虚拟化与隔离的**中间件** (Middleware)      |
| **架构模式**     | 独立/替换式架构 (基于 kube-batch)                   | 插件/扩展式架构 (Extender + Webhook)               |
| **GPU 隔离级别** | **软隔离** (Soft Isolation)，侧重逻辑分配，依赖协作 | **硬隔离** (Hard Isolation)，基于 API 拦截强制限制 |
| **公平性策略**   | **作业级公平**，支持多级队列、DRF 及 Time-aware     | **设备级公平**，侧重物理设备切分利用率             |
| **拓扑感知**     | **原生集成** (TAS)，内建拓扑优选，路径更短          | **扩展支持**，通过策略实现，效率略逊原生           |
| **生态兼容性**   | **NVIDIA 深度优化**，官方维护                       | **多厂商异构**，广泛支持国产芯片                   |
| **作业编排**     | **强**，原生支持 Gang Scheduling 及抢占             | **中**，依赖底层调度器能力                         |
| **资源粒度**     | **粗/中粒度**，以 GPU 数量或大块显存为主            | **极细粒度**，支持按百分比、具体显存大小切分       |

### 3.2 深度技术差异解析

#### 3.2.1 调度哲学与决策范围

**KAI Scheduler** 扮演的是**全局决策者**的角色。它接管了 Pod 的全生命周期管理，具备为了全局最优（如 Gang Scheduling 的资源预留）而推迟部分任务调度的能力。其设计目标是优化整个集群在多租户、高并发场景下的作业吞吐量与资源公平性。相比之下，**HAMi** 扮演的是**资源增强层**的角色。它通过虚拟化技术将物理 GPU 映射为多个 vGPU，从而“欺骗”Kubernetes 原生调度器。它不改变 Kubernetes 的核心调度逻辑，而是致力于提升底层资源的分配灵活性与利用率。

#### 3.2.2 资源隔离机制的本质区别

**KAI** 的 GPU 共享机制本质上是一种**调度层面的记账（Accounting）**。例如，当分配 `0.5` GPU 时，调度器确保同一物理卡上仅调度两个此类 Pod。若应用发生显存泄漏或越界使用，KAI 调度器层面无法进行运行时的强制拦截。而 **HAMi** 的核心优势在于**运行时虚拟化**。它在容器启动时注入特定的动态链接库，实时拦截并审查 CUDA 内存申请。若申请量超过配额，API 调用将直接返回失败。这种机制在不可信的多租户环境中提供了必要的安全保障。

#### 3.2.3 公平性维度的侧重

**KAI** 聚焦于**组织/租户层面的公平性**，通过多级队列模型解决“部门 A 与部门 B 资源争抢”的问题，支持复杂的借用与归还机制。**HAMi** 则聚焦于**设备层面的切分效率**，致力于解决“如何将单张 GPU 卡高效切分给多个小任务”的问题，侧重于提升单卡的利用率。

### 3.3 场景化适用性分析

#### 3.3.1 AI 训练场景 (Training)

在模型训练，尤其是**大规模分布式训练**（如 LLM 预训练、微调）场景下，**KAI Scheduler** 具有压倒性优势。

- **核心挑战**: 资源死锁、网络带宽瓶颈、多租户资源争抢。
- **KAI Scheduler 优势**:
  - **Gang Scheduling**: 原生支持 PodGroup，确保分布式任务的所有 Worker 能够 "All-or-Nothing" 启动，彻底根除死锁风险。
  - **拓扑感知 (TAS)**: 对于通信密集型训练任务，KAI 能将 Pod 调度到同一交换机或机架下，显著降低 AllReduce 通信延迟，提升训练效率。
  - **公平性保障**: 基于 DRF 的队列管理机制，确保了不同部门或项目组在共享大集群时的资源公平性，避免算力被单一业务独占。
- **HAMi 局限性**: 缺乏全局视角的 Gang Scheduling 能力，容易导致分布式任务的部分 Pod 等待资源而部分 Pod 空转。

#### 3.3.2 AI 推理场景 (Inference)

在**在线推理**与**开发调试**场景下，**HAMi** 往往是更优的选择。

- **核心挑战**: 资源碎片化、服务间干扰、异构硬件适配。
- **HAMi 优势**:
  - **强隔离保障 SLA**: 推理服务通常对稳定性要求极高。HAMi 的显存硬隔离机制能防止因某个服务 OOM 而导致同一张卡上的其他推理服务崩溃，保障业务 SLA。
  - **极致资源利用率**: 推理任务通常无法占满整卡。HAMi 支持细粒度切分（如分配 10% 算力 + 2GB 显存），允许在单张高性能卡（如 A100）上混部数十个小模型推理服务。
  - **异构推理卡支持**: 推理侧硬件选型多样（如 T4, A10, Ascend 310 等）。HAMi 广泛的硬件兼容性使其能统一纳管各类异构推理芯片。
- **KAI Scheduler 局限性**: 软隔离机制在不可信的多租户推理场景下存在安全隐患；缺乏对非 NVIDIA 推理卡的深度支持。

### 3.4 选型建议与总结

在实际架构设计中，建议根据**业务核心属性**进行分层决策：

1. **大规模训练集群 (Training Cluster)**:

   - **首选 KAI Scheduler**。核心诉求是算力吞吐与作业编排，需依赖其 Gang Scheduling 与 TAS 能力。
   - **可选搭配**: 若需在训练集群中混部少量调试任务，可引入 HAMi 作为辅助，但需注意调度器冲突。

2. **推理与开发集群 (Inference/Dev Cluster)**:

   - **首选 HAMi**。核心诉求是资源利用率与安全隔离。利用其硬隔离特性实现高密度混部，降低单位推理成本。

3. **混合业务集群**:
   - 建议采用 **"KAI Scheduler (主) + HAMi (辅)"** 的联合部署模式。
   - 利用 KAI Scheduler 处理宏观的队列管理与大任务编排。
   - 利用 HAMi 处理节点内的细粒度切分与隔离。
   - **注意**: 需精细配置 Resource Name（如 KAI 使用 `nvidia.com/gpu`，HAMi 使用 `nvidia.com/gpumem`），避免资源核算的冲突。

### 3.5 前沿探索：社区融合方案 (Proposed)

HAMi 社区与 KAI Scheduler 社区正在探讨一种更深度的融合方案（参考 [PR #60](https://github.com/NVIDIA/KAI-Scheduler/pull/60)）。该方案旨在通过解耦的方式，在保留 KAI 调度决策权的同时，引入 HAMi 的底层硬隔离能力。

> **注意**: 该方案目前处于**提案与讨论阶段 (Under Discussion)**，尚未正式合入 KAI Scheduler 主干。

**方案架构核心**:

该方案不再依赖 HAMi 的 Scheduler Extender，而是将 HAMi 的核心隔离能力 (HAMi Core) 作为独立组件运行：

1. **组件解耦**:

   - **KAI Scheduler**: 继续负责核心调度逻辑（Gang Scheduling, TAS 等）及资源分配决策。
   - **HAMi Core**: 以 DaemonSet 形式独立部署，仅负责节点级的动态库分发与底层拦截。
   - **Mutating Webhook**: 负责拦截 Pod 创建请求，注入 HAMi Core 的挂载路径。

2. **协同工作流**:
   - **Step 1 (调度)**: KAI Scheduler 完成调度决策，并将分配的显存大小（Bytes）写入标准环境变量（如 `GPU_MEMORY_LIMIT`）。
   - **Step 2 (注入)**: Webhook 自动挂载 HAMi Core 的动态链接库到 Pod 容器中。
   - **Step 3 (运行时)**: 容器启动时加载拦截库，库函数读取 `GPU_MEMORY_LIMIT` 环境变量，在 CUDA API 调用层面强制实施显存硬隔离。

**方案价值**:

这种模式试图打破“二选一”的局面，实现 **“KAI 的全局编排能力 + HAMi 的运行时硬隔离能力”** 的完美结合，且对 KAI 调度器的代码侵入性极低。

---

## 4. 扩展对比：KAI Scheduler vs Volcano

鉴于 **KAI Scheduler** 与 CNCF 孵化项目 **[Volcano](https://github.com/volcano-sh/volcano)** 均源自 **kube-batch** 架构体系，两者在技术路线上存在一定的亲缘性，但在演进方向与适用场景上已展现出显著差异。

### 4.1 定位与演进方向差异

| **维度**     | **KAI Scheduler**                                                                | **Volcano**                                                                          |
| :----------- | :------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| **核心定位** | **垂直深耕的 AI 算力引擎**                                                       | **通用的云原生批处理平台**                                                           |
| **目标场景** | 聚焦 **LLM/GenAI** 大规模分布式训练，深度适配 NVIDIA 硬件栈 (GPU/NPU/InfiniBand) | 覆盖 **BigData** (Spark/Flink), **AI**, **HPC**, **基因计算** 等广泛的批处理工作负载 |
| **生态兼容** | 紧密集成 NVIDIA GPU Operator, DRA, PyTorch 等 AI 生态                            | 广泛支持 Spark Operator, Flink Operator, Argo, Kubeflow 等 CNCF/大数据生态           |

### 4.2 核心能力对比

- **拓扑感知 (Topology Awareness)**:

  - **KAI Scheduler**: 拥有 **Native TAS** 能力，针对 NVIDIA 硬件拓扑（NVLink, NVSwitch, InfiniBand 结构）进行了深度定制与优化，能够感知极其细微的硬件互联层级，以极致降低 AI 训练的通信延迟。
  - **Volcano**: 提供通用的拓扑感知插件，更侧重于 CPU/NUMA 或通用的网络距离，对特定 AI 芯片的私有互联协议优化不如 KAI 深入。

- **公平性算法 (Fairness)**:

  - **KAI Scheduler**: 引入了 **Time-aware Fairness** (时间感知公平性)，不仅关注当前的资源配额，还追溯历史使用时长，防止长期霸占资源，更适合昂贵的 GPU 算力租赁场景。
  - **Volcano**: 经典的 DRF (Dominant Resource Fairness) 算法实现，侧重于多资源维度的即时分配公平。

- **作业类型支持**:
  - **KAI Scheduler**: 针对 AI 训练任务（Training Jobs）进行了特化，对 `All-or-Nothing` 的 Gang Scheduling 做了针对大模型的死锁优化。
  - **Volcano**: 对数据密集型任务（Data-Intensive）有更好的支持，如 Spark 的 Driver/Executor 模型，支持更丰富的作业插件（Job Plugins）。

### 4.3 选型建议

- **选择 KAI Scheduler**: 如果您的集群是 **NVIDIA GPU 专用算力池**，且核心业务是 **LLM 预训练/微调**，追求极致的算力利用率与训练速度，KAI 的垂直优化能力将带来更高的 ROI。
- **选择 Volcano**: 如果您的集群是 **混合负载集群**（同时运行 Spark 数据处理、Flink 流计算、以及 AI 训练），需要一个统一的批处理调度器来管理多样化的工作负载，Volcano 是更通用且成熟的选择。

---
