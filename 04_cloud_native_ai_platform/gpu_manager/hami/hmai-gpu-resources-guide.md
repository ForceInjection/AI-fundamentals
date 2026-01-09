# HAMi GPU 资源管理完整指南

本指南面向对 HAMi 有一定了解的技术人员，提供从基础概念到高级特性的全方位技术说明。

---

## 1. 概述

本文档是 HAMi GPU 资源管理的完整技术指南，涵盖以下核心内容：

- **资源类型管理**: 详细介绍 NVIDIA GPU 的四种核心资源类型（`nvidia.com/gpu`、`nvidia.com/gpucores`、`nvidia.com/gpumem`、`nvidia.com/gpumem-percentage`）的定义、特性和使用方法
- **算力分配机制**: 深入解析基于反馈控制的令牌桶算法实现原理、分配流程和性能影响
- **高级特性**: 包括多种调度策略（binpack、spread、topology-aware 等）、设备选择、NUMA 绑定、拓扑感知、MIG 支持、运行时模式选择等
- **配置管理**: 全局配置、节点级配置和 Pod 级配置的详细说明
- **故障排查与最佳实践**: 常见问题诊断、性能优化建议、监控告警配置、版本升级指导
- **多厂商支持**: 统一管理 NVIDIA、寒武纪、海光、天数智芯、摩尔线程、华为昇腾、沐曦等 7 种厂商的 AI 加速设备
- **监控与可观测性**: Prometheus 指标、性能监控、资源使用率跟踪等

> **注意**: 文档中默认代码都是指向 `HAMi` 项目。

### 1.1 独占 vs 共享场景

- **独占模式 (Exclusive)**

  - **定义**：一个 Pod 独占一张或多张物理 GPU 卡。
  - **适用场景**：大型模型训练（LLM Pretraining/SFT）、高性能计算（HPC）、对延迟极其敏感的推理任务。
  - **优势**：性能无损耗，无资源争抢，故障完全隔离。
  - **配置特征**：申请资源量等于物理卡总量（如 100% 算力 + 100% 显存）。

- **共享模式 (Sharing/Fractional)**
  - **定义**：多个 Pod 共享同一张物理 GPU 卡，每个 Pod 占用一部分显存和算力。
  - **适用场景**：轻量级推理服务、开发/调试环境（Jupyter Notebook）、小模型微调、CI/CD 流水线。
  - **优势**：极大提升 GPU 利用率，降低成本，支持灵活的资源配置。
  - **配置特征**：申请资源量小于物理卡总量。

### 1.2 NVIDIA 共享模式深度对比

针对 NVIDIA GPU，HAMi 提供了三种主要的共享模式。请根据下表选择最适合的方案：

| **特性**     | **HAMi-core 模式 (默认)**                                           | **MIG 模式 (Multi-Instance GPU)**                           | **MPS 模式 (Multi-Process Service)**                               |
| :----------- | :------------------------------------------------------------------ | :---------------------------------------------------------- | :----------------------------------------------------------------- |
| **隔离级别** | 软件级 (显存硬隔离 + 算力时间片限制)                                | 硬件级 (物理隔离的计算单元与显存)                           | 进程级 (共享 CUDA Context)                                         |
| **硬件要求** | 所有 NVIDIA GPU                                                     | 仅限 Ampere (A100/A30) 及 Hopper (H100/H800) 架构           | 所有 NVIDIA GPU                                                    |
| **显存隔离** | 安全 (劫持 CUDA API 实现)                                           | 非常安全 (物理分区)                                         | 安全 (统一地址空间)                                                |
| **算力隔离** | 较好 (基于反馈控制的令牌桶算法)                                     | 极好 (物理计算单元隔离)                                     | 一般 (共享计算资源，提高并发度)                                    |
| **故障影响** | 单个任务崩溃通常不影响其他任务，但严重错误可能导致 XID 错误影响整卡 | 故障完全隔离                                                | 某个进程崩溃可能导致 MPS Server 重置，影响共享该 Server 的所有进程 |
| **适用场景** | 通用场景，需要灵活切分显存（如 1GB, 1.5GB）                         | 对隔离性要求极高（多租户），且接受固定切分规格（如 1g.5gb） | 小算力高并发任务，希望能最大化吞吐量                               |
| **启用方式** | 默认 (或 `nvidia.com/vgpu-mode: "hami-core"`)                       | `nvidia.com/vgpu-mode: "mig"`                               | `nvidia.com/vgpu-mode: "mps"`                                      |

**模式详解：**

1. **HAMi-core 模式**：

   - 这是 HAMi 的核心能力，通过拦截 CUDA 调用实现。
   - **优点**：切分粒度极其灵活（可以申请 1MB 显存），支持超卖（如果未开启严格限制），基于反馈控制的令牌桶算法实现精确的算力限制。
   - **缺点**：算力限制基于时间片，可能有微小的上下文切换开销。

2. **MIG 模式**：

   - 利用 NVIDIA GPU 的硬件特性将 GPU 物理切分为多个实例。
   - **优点**：提供 QoS 保证的硬件隔离，互不干扰。
   - **缺点**：切分规格固定（如 A100 只能切分为 7 个实例），不够灵活；不支持旧型号 GPU。

3. **MPS 模式**：
   - NVIDIA 官方提供的多进程服务。
   - **优点**：减少了多个 CUDA Context 切换的开销，显著提升小任务并发时的吞吐量；支持算力限制但内存相互可见。
   - **缺点**：故障隔离性较差（"一损俱损"风险），某个进程崩溃可能导致 MPS Server 重置，影响共享该 Server 的所有进程。

### 1.3 多厂商异构支持概览

HAMi 不仅支持 NVIDIA GPU，还通过统一的接口支持多种异构算力设备。详细的配置管理方法请参见第 4 章。

- **华为昇腾 (Ascend)**：支持 910A、910B (B2/B3/B4)、310P 系列 NPU 切分与共享。
- **寒武纪 (Cambricon)**：支持思元系列 MLU 加速卡。
- **海光 (Hygon)**：支持深算系列 DCU 加速卡。
- **天数智芯 (Iluvatar)**：支持 CoreX 系列 GPU。
- **摩尔线程 (Moore Threads)**：支持 MTT 系列 GPU。
- **沐曦 (MetaX)**：支持 MetaX 系列 GPU。

---

## 2. 资源配置实战

本章详细讲解 HAMi GPU 资源管理的具体配置方法，涵盖从基础资源参数到高级场景化配置的完整实践指南。

### 2.1 核心资源参数详解

在 Pod 的 YAML 中，通过 `resources.limits` 申请资源。以下是所有支持的 NVIDIA 资源类型：

| **资源名称**                   | **描述**             | **取值范围** | **版本要求** | **备注/冲突处理**                                                 |
| :----------------------------- | :------------------- | :----------- | :----------- | :---------------------------------------------------------------- |
| `nvidia.com/gpu`               | 申请的 **vGPU** 数量 | 正整数       | 所有版本     | **必填**。用于触发调度。                                          |
| `nvidia.com/gpumem`            | 申请的显存大小 (MB)  | 正整数       | 所有版本     | **显存硬限制**。若与 `gpumem-percentage` 同时存在，优先使用此值。 |
| `nvidia.com/gpumem-percentage` | 申请的显存百分比 (%) | 0-100        | v1.0.1.4+    | 动态适应不同显存大小的卡。**注意：MIG 模式不支持百分比配置**。    |
| `nvidia.com/gpucores`          | 申请的算力百分比 (%) | 0-100        | v1.0.1.3+    | 限制该容器最多使用多少比例的算力。                                |

> **注意**：`nvidia.com/gpu` 在 Pod 申请时代表“需要的 vGPU 实例数”，而在节点 Capacity 中通常代表“vGPU 的总配额”（取决于配置）。

### 2.2 场景化配置模板

#### 2.2.1 独占使用 (Exclusive)

希望完全占用一张卡，不与其他任务共享。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: exclusive-gpu-pod
spec:
  containers:
    - name: cuda-container
      image: nvidia/cuda:11.0-base
      resources:
        limits:
          nvidia.com/gpu: 1 # 申请 1 个 vGPU
          nvidia.com/gpumem-percentage: 100 # 占用 100% 显存
          nvidia.com/gpucores: 100 # 占用 100% 算力
```

#### 2.2.2 HAMi-core 共享模式 (Sharing)

一个轻量级推理服务，需要 4GB 显存和 20% 的算力。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: shared-gpu-pod
  annotations:
    nvidia.com/vgpu-mode: "hami-core" # 显式指定模式（默认可省略）
spec:
  containers:
    - name: inference-container
      image: pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
      resources:
        limits:
          nvidia.com/gpu: 1
          nvidia.com/gpumem: 4096 # 4096 MB 显存
          nvidia.com/gpucores: 20 # 20% 算力
```

#### 2.2.3 MIG 模式 (MIG)

申请一个 A100 的 MIG 实例。**注意：MIG 模式必须使用固定规格的显存配置，不支持百分比配置**。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mig-pod
  annotations:
    nvidia.com/vgpu-mode: "mig" # 指定使用 MIG 模式
spec:
  containers:
    - name: mig-container
      image: nvidia/cuda:11.0-base
      resources:
        limits:
          nvidia.com/gpu: 1
          nvidia.com/gpumem: 20480 # 必须使用MIG固定规格的显存大小 (对应 A100 3g.20gb 规格)
```

#### 2.2.4 MPS 模式 (MPS)

适用于小算力高并发任务，希望最大化吞吐量的场景。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mps-pod
  annotations:
    nvidia.com/vgpu-mode: "mps" # 指定使用 MPS 模式
spec:
  containers:
    - name: mps-container
      image: nvidia/cuda:11.0-base
      resources:
        limits:
          nvidia.com/gpu: 1
          nvidia.com/gpumem: 4096 # 申请 4GB 显存
          nvidia.com/gpucores: 25 # 申请 25% 算力
```

### 2.3 配置层级管理

HAMi 的配置生效优先级遵循：**Pod Annotation > Node Annotation > Global ConfigMap**。详细的配置管理方法请参见第 4 章。

#### 2.3.1 全局配置 (ConfigMap)

通过 `hami-scheduler-device` ConfigMap 管理（`device-config.yaml` 字段）：

| **字段名**                | **默认值** | **描述**                                      |
| :------------------------ | :--------- | :-------------------------------------------- |
| `deviceMemoryScaling`     | `1`        | 显存超卖比例。>1 表示开启超卖（实验性）。     |
| `deviceSplitCount`        | `10`       | 单个 GPU 可同时承载的最大容器数（并发度）。   |
| `defaultMemory`           | `0`        | Pod 未指定显存时的默认值（0 表示 100%）。     |
| `defaultCores`            | `0`        | Pod 未指定核心时的默认百分比（0 表示 100%）。 |
| `defaultGPUNum`           | `1`        | Pod 未指定 `nvidia.com/gpu` 时的默认数量。    |
| `strictMemoryEnforcement` | `false`    | 是否严格限制显存使用（防止超卖场景下的 OOM）  |

**ConfigMap 配置示例**：

```yaml
# device-config.yaml
deviceMemoryScaling: 1.0
deviceSplitCount: 10
defaultMemory: 0
defaultCores: 0
defaultGPUNum: 1
strictMemoryEnforcement: false
```

#### 2.3.2 节点级配置

节点级配置允许管理员针对特定节点设置调度策略和资源管理规则。

- **启用 GPU 调度**：为节点添加标签启用 HAMi 调度

  ```bash
  kubectl label nodes node1 gpu=on
  ```

- **节点调度策略**：设置节点级别的调度偏好

  ```bash
  # binpack: 尽量将任务集中到已使用的节点，减少碎片
  kubectl annotate node node1 hami.io/node-scheduler-policy=binpack

  # spread: 尽量将任务均匀分布到不同节点
  kubectl annotate node node1 hami.io/node-scheduler-policy=spread
  ```

- **设备选择策略**：限制节点上可使用的设备类型

  ```bash
  kubectl annotate node node1 nvidia.com/use-gputype="Tesla V100-PCIE-32GB"
  ```

#### 2.3.3 Pod 级配置

Pod 级配置允许开发者在应用层面指定调度和资源使用策略。

- **GPU 调度策略**：强制指定 Pod 的调度行为

  ```yaml
  apiVersion: v1
  kind: Pod
  metadata:
    name: example-pod
    annotations:
      hami.io/gpu-scheduler-policy: "spread" # 或 "binpack"
  spec:
    # ...
  ```

- **设备类型偏好**：指定 Pod 需要的特定设备型号

  ```yaml
  annotations:
    nvidia.com/use-gputype: "Tesla V100-PCIE-32GB"
  ```

- **GPU 模式选择**：显式指定使用的 vGPU 模式

  ```yaml
  annotations:
    nvidia.com/vgpu-mode: "hami-core" # 或 "mig", "mps"
  ```

---

## 3. 核心资源类型深度解析

### 3.1 nvidia.com/gpu

#### 3.1.1 定义

`nvidia.com/gpu` 是用于声明 Pod 所需物理 GPU 数量的资源类型。

#### 3.1.2 特性

- **必需性**: 必需的资源声明，用于触发 GPU 调度
- **取值范围**: 正整数（1, 2, 3...）
- **用途**: 指定 Pod 需要多少个物理 GPU 设备
- **调度作用**: HAMi 调度器根据此值进行 GPU 设备分配

#### 3.1.3 代码实现

**1. 资源请求解析**：

在 `pkg/device/nvidia/device.go` 的 `GenerateResourceRequests` 函数中：

```go
// GenerateResourceRequests 解析容器资源请求，生成设备请求结构体
// 参数: ctr - 容器对象，包含资源限制和请求
// 返回值: device.ContainerDeviceRequest - 设备请求结构体
func (dev *NvidiaGPUDevices) GenerateResourceRequests(ctr *corev1.Container) device.ContainerDeviceRequest {
    // 获取 nvidia.com/gpu 的值，优先从 Limits 获取，其次从 Requests 获取
    v, ok := ctr.Resources.Limits[resourceName]
    if !ok {
        v, ok = ctr.Resources.Requests[resourceName]
    }
    if ok {
        if n, ok := v.AsInt64(); ok {
            return device.ContainerDeviceRequest{
                Nums: int32(n),      // 请求的 GPU 数量
                Type: resourceName,   // 资源类型
                // ... 其他字段
            }
        }
    }
    // 如果没有找到资源请求，返回空结构体
    return device.ContainerDeviceRequest{}
}
```

**2. 调度分配**：

在 `Fit` 函数中，调度器会寻找满足数量要求的设备：

```go
// pkg/device/nvidia/device.go
// Fit 函数检查设备列表是否满足容器设备请求
// 参数: devices - 设备使用情况列表, request - 容器设备请求, pod - Pod对象, nodeInfo - 节点信息, allocated - 已分配设备
// 返回值: bool - 是否满足请求, map[string]device.ContainerDevices - 分配的设备映射, string - 失败原因
func (nv *NvidiaGPUDevices) Fit(devices []*device.DeviceUsage, request device.ContainerDeviceRequest, pod *corev1.Pod, nodeInfo *device.NodeInfo, allocated *device.PodDevices) (bool, map[string]device.ContainerDevices, string) {
    // ...
    // 检查是否需要拓扑感知调度策略
    needTopology := util.GetGPUSchedulerPolicyByPod(device.GPUSchedulerPolicy, pod) == util.GPUSchedulerPolicyTopology.String()
    // 从后向前遍历设备列表（优化分配策略）
    for i := len(devices) - 1; i >= 0; i-- {
        dev := devices[i]
        // ... 检查健康状态 / 设备类型 / UUID / 显存 / 核心等 ...

        tmpDevs[request.Type] = append(tmpDevs[request.Type], device.ContainerDevice{
            UUID:      dev.ID,
            Usedmem:   memreq,
            Usedcores: request.Coresreq,
        })

        // 非 topology-aware 策略：满足数量后立即返回
        if !needTopology {
            request.Nums--
            if request.Nums == 0 {
                return true, tmpDevs, ""
            }
        }
    }
    // topology-aware 策略：在候选集上做组合选择后返回
    // ...
}
```

### 3.2 nvidia.com/gpucores

#### 3.2.1 定义

`nvidia.com/gpucores` 是用于指定每个物理 GPU 分配给 Pod 的计算核心百分比的资源类型。

#### 3.2.2 特性

- **必需性**: 可选的资源声明
- **取值范围**: 0-100 的整数，表示百分比
- **默认值**: 由 `device-config.yaml` 中 `defaultCores` 配置项决定（来源：`charts/hami/templates/scheduler/device-configmap.yaml`）
- **限制类型**: 从 v1.0.1.3 版本开始在容器内部限制 GPU 利用率（来源：`HAMi/CHANGELOG.md` 中 `v1.0.1.3` 说明）
- **共享支持**: 支持多个 Pod 共享同一个物理 GPU 的不同算力份额

#### 3.2.3 配置选项

- `defaultCores`: 默认核心分配百分比（`device-config.yaml`）
- `devicePlugin.disablecorelimit`: 是否禁用核心限制（Helm Chart values，最终作为 `hami-device-plugin` 的 `--disable-core-limit` 参数）
- `GPU_CORE_UTILIZATION_POLICY`: 容器环境变量，控制核心利用率策略

#### 3.2.4 算力分配机制

##### 3.2.4.1 核心概念

- **总量(Totalcore)**: 每个 GPU 设备的总计算核心数，通常为 100（代表 100% 的 GPU 计算能力）
- **已分配(Usedcores)**: 当前已经分配给 Pod/容器的 GPU 核心数总和
- **算力**: 指 GPU 的计算核心资源，以百分比形式表示

##### 3.2.4.2 数据结构

在 `pkg/device/devices.go` 中的 `DeviceUsage` 结构体定义：

```go
type DeviceUsage struct {
    ID          string
    Index       uint
    Used        int32   // 使用该设备的容器数量
    Count       int32   // 设备可支持的最大容器数量
    Usedmem     int32   // 已使用内存
    Totalmem    int32   // 总内存
    Totalcore   int32   // 总核心数（通常为100）
    Usedcores   int32   // 已使用核心数
    Mode        string  // 设备模式（如MIG模式）
    Numa        int     // NUMA 节点编号
    Type        string  // GPU 型号
    Health      bool    // 设备健康状态
    // ... 其他字段
}
```

##### 3.2.4.3 分配流程

1. **资源声明**: Pod 通过 `nvidia.com/gpucores` 声明所需的 GPU 核心百分比
2. **分配检查**: 在 `Fit` 函数中检查可用资源

   ```go
   if dev.Totalcore-dev.Usedcores < k.Coresreq {
       // 核心资源不足
       continue
   }
   ```

3. **使用更新**: 在 `AddResourceUsage` 函数中更新已分配核心数

```go
// 更新设备的已使用核心数：累加容器的核心使用量
n.Usedcores += ctr.Usedcores
```

##### 3.2.4.4 实际示例

- **100/100**: GPU 被完全独占使用
- **50/100**: GPU 的 50% 算力已分配，还剩 50% 可用
- **80/100**: GPU 的 80% 算力已分配，还剩 20% 可用

##### 3.2.4.5 性能影响分析

`nvidia.com/gpucores` 表示 Pod 在调度与运行期可用的计算份额上限，但性能表现通常不满足严格的线性关系，主要受以下因素影响：

- 算子形态与并发（Kernel 粒度、Launch 频率、CPU 侧开销）
- 显存带宽与访存模式（训练/推理差异明显）
- 多租共享时的竞争（上下文切换、其他任务负载波动）

因此，建议在目标 GPU 型号与目标框架上做基准测试，使用 `HostCoreUtilization` 与容器级 `Device_utilization_desc_of_container` 指标对比评估资源配置是否合理。

#### 3.2.5 代码实现

**1. 资源请求解析**：

在 `pkg/device/nvidia/device.go` 中解析核心需求：

```go
// pkg/device/nvidia/device.go

// 设置默认核心数为配置的默认值
corenum := dev.config.DefaultCores
// 优先从 Limits 获取 nvidia.com/gpucores 的值
core, ok := ctr.Resources.Limits[resourceCores]
if !ok {
    // 如果 Limits 中没有，则从 Requests 获取
    core, ok = ctr.Resources.Requests[resourceCores]
}
if ok {
    // 将资源值转换为 int64
    corenums, ok := core.AsInt64()
    if ok {
        // 更新核心数为请求的值
        corenum = int32(corenums)
    }
}
```

**2. 调度分配**：

在 `Fit` 函数中，检查设备剩余算力是否满足请求：

```go
// pkg/device/nvidia/device.go

// 检查设备剩余核心数是否足够：总核心数 - 已使用核心数 >= 请求核心数
if dev.Totalcore-dev.Usedcores < k.Coresreq {
    // 核心资源不足，记录原因并继续检查其他设备
    reason[common.CardInsufficientCore]++
    continue
}
```

**3. 限制执行 (Enforcement)**：

核心利用率限制主要在 `HAMi-core` 项目中实现，采用了**基于反馈控制的令牌桶算法**。

**算法原理**：

1. **令牌桶 (Token Bucket)**：使用全局变量 `g_cur_cuda_cores` 作为令牌桶，代表当前允许使用的计算配额。该变量**初始化为 0**，后续由监控线程动态调整。
2. **消费 (Consume)**：当 CUDA Kernel 启动时，计算其所需的计算量（Grid × Block），并从令牌桶中扣除相应数量的令牌。如果桶中令牌不足（小于 0），则挂起当前线程等待，直到令牌回补。
3. **反馈调节 (Feedback Loop)**：后台监控线程定期检查 GPU 实际利用率。如果实际利用率低于用户设定的限制（`nvidia.com/gpucores`），则向桶中注入令牌；如果超过限制，则减少令牌，从而动态稳定 GPU 利用率。

> **注**：GPU 实际利用率是通过 NVML API `nvmlDeviceGetProcessUtilization` 获取的 **SM 利用率 (smUtil)**，统计了当前容器内所有进程的利用率之和。（来源：`HAMi-core/src/multiprocess/multiprocess_utilization_watcher.c` 调用 `nvmlDeviceGetProcessUtilization` 并累加 `smUtil`）
>
> **关于监控指标**：
>
> 1. **指标对应关系**：该指标在 **DCGM Exporter** 中**没有直接对应的容器级指标**。默认的 `DCGM_FI_DEV_GPU_UTIL` 仅反映整卡物理利用率。
> 2. **指标准确性说明**：标准的 GPU 利用率（如 `nvidia-smi`）通常是**基于时间（Temporal）**的度量，即“在过去采样周期内，GPU 是否有 Kernel 在执行”。（来源：NVML `nvmlDeviceGetProcessUtilization` 与 `nvmlProcessUtilizationSample_t.smUtil` 字段定义，见 `HAMi-core/src/include/nvml-subset.h`）
>    - 这意味着：即使只有一个 Kernel 在单个 SM 上运行，利用率也可能显示为 100%。
>    - 因此，该指标更适合衡量 **“GPU 是否繁忙”**（`Time Slicing`），而非 **“GPU 计算能力是否饱和”**（`Compute Saturation`/`SM Efficiency`）。
> 3. **HAMi 的做法**：HAMi 的核心限制本质上是**时间片调度（Time Slicing）**，因此依赖这种基于时间的利用率指标是符合其设计预期的。如果需要监控容器级别的核心利用率，建议使用 HAMi 自带的监控组件（`vGPU Monitor`）。

**代码实现细节**：

**1. Kernel 启动拦截与令牌消费**：

在 `HAMi-core/src/cuda/memory.c` 中，拦截了 `cuLaunchKernel` 等函数：

```c
// HAMi-core/src/cuda/memory.c

// cuLaunchKernel - CUDA Kernel 启动函数拦截实现
// 参数: f - CUDA 函数句柄, gridDimX/Y/Z - Grid 维度, blockDimX/Y/Z - Block 维度
//       sharedMemBytes - 共享内存大小, hStream - CUDA 流, kernelParams - 内核参数, extra - 额外参数
// 返回值: CUresult - CUDA 操作结果
CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra ){
    ENSURE_RUNNING();  // 确保运行时环境正常
    pre_launch_kernel();  // Kernel 启动前预处理
    if (pidfound==1){  // 检查是否为需要限制的进程
        // 调用速率限制器，计算 Kernel 的总计算量（Grid 大小 × Block 大小）
        rate_limiter(gridDimX * gridDimY * gridDimZ,
                   blockDimX * blockDimY * blockDimZ);
    }
    // 调用原始的 CUDA Kernel 启动函数
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuLaunchKernel,f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,hStream,kernelParams,extra);
    return res;  // 返回操作结果
}
```

**2. 速率限制逻辑**：

在 `HAMi-core/src/multiprocess/multiprocess_utilization_watcher.c` 中实现 `rate_limiter`，根据当前的利用率情况决定是否延迟 Kernel 的提交：

```c
// HAMi-core/src/multiprocess/multiprocess_utilization_watcher.c

// rate_limiter - 速率限制器函数，控制 CUDA Kernel 的提交速率
// 参数: grids - Grid 维度乘积, blocks - Block 维度乘积
// 返回值: void
void rate_limiter(int grids, int blocks) {
  long before_cuda_cores = 0;  // 限制前的核心数
  long after_cuda_cores = 0;   // 限制后的核心数
  long kernel_size = grids;    // Kernel 计算量（Grid 大小）

  // 等待初始化完成，确保监控系统已就绪
  while (get_recent_kernel()<0) {
    sleep(1);  // 每秒检查一次
  }
  set_recent_kernel(2);  // 设置最近 Kernel 状态

  // 检查是否需要限制：如果限制为100%（无限制）或0%（禁用限制），则直接返回
  if ((get_current_device_sm_limit(0)>=100) || (get_current_device_sm_limit(0)==0))
      return;
  // 检查利用率开关是否开启
  if (get_utilization_switch()==0)
      return;

  // 核心限制逻辑 - 使用 CAS（Compare And Swap）原子操作
  do {
CHECK:
      before_cuda_cores = g_cur_cuda_cores;  // 获取当前可用核心数
      if (before_cuda_cores < 0) {
        // 如果当前核心数不足，则睡眠等待令牌回补
        nanosleep(&g_cycle, NULL);  // 使用纳秒级睡眠
        goto CHECK;  // 重新检查
      }
      after_cuda_cores = before_cuda_cores - kernel_size;  // 计算扣除后的核心数
  } while (!CAS(&g_cur_cuda_cores, before_cuda_cores, after_cuda_cores));  // 原子更新
}
```

**3. 利用率监控与调节**：

后台线程 `utilization_watcher` 负责监控实际 GPU 利用率并动态调整可用 Token (`g_cur_cuda_cores`)：

```c
// HAMi-core/src/multiprocess/multiprocess_utilization_watcher.c

void* utilization_watcher() {
    // ... 初始化 ...
    int upper_limit = get_current_device_sm_limit(0);

    while (1){
        nanosleep(&g_wait, NULL);
        // ... 获取当前利用率 userutil ...

        // 动态调整可用核心数
        if ((userutil[0]<=100) && (userutil[0]>=0)){
          share = delta(upper_limit, userutil[0], share);
          change_token(share);
        }
    }
}
```

### 3.3 nvidia.com/gpumem

#### 3.3.1 定义

`nvidia.com/gpumem` 是用于指定 Pod 所需 GPU 内存大小的资源类型，以 MB 为单位。

#### 3.3.2 特性

- **必需性**: 可选的资源声明
- **取值范围**: 正整数，单位为 MB
- **用途**: 精确指定 Pod 需要的 GPU 内存量
- **互斥建议**: 建议不要与 `nvidia.com/gpumem-percentage` 同时设置；如同时设置，`gpumem` 优先生效（来源：`pkg/device/nvidia/device.go` 的 `Fit` 实现）
- **内存隔离**: 提供硬隔离保证，确保容器不会超出分配的内存边界

#### 3.3.3 代码实现

**1. 资源请求解析**：

在 `GenerateResourceRequests` 函数中处理内存资源请求：

```go
// pkg/device/nvidia/device.go

// 获取 nvidia.com/gpumem 的值
mem, ok := ctr.Resources.Limits[resourceMem]
if !ok {
    mem, ok = ctr.Resources.Requests[resourceMem]
}
if ok {
    memnums, ok := mem.AsInt64()
    if ok {
        memnum = int(memnums)
    }
}
```

**2. 调度分配**：

在 `Fit` 函数中，检查设备剩余显存是否满足请求：

```go
// pkg/device/nvidia/device.go

// 检查设备总显存减去已用显存是否小于请求显存
if dev.Totalmem-dev.Usedmem < memreq {
    reason[common.CardInsufficientMemory]++
    klog.V(5).InfoS(common.CardInsufficientMemory, ...)
    continue
}
```

### 3.4 nvidia.com/gpumem-percentage

#### 3.4.1 定义

`nvidia.com/gpumem-percentage` 是用于指定 Pod 所需 GPU 内存百分比的资源类型，在 v1.0.1.4 版本中引入。（来源：`HAMi/CHANGELOG.md` 中 `v1.0.1.4` 说明）

#### 3.4.2 特性

- **必需性**: 可选的资源声明
- **取值范围**: 0-100 的整数，表示百分比
- **用途**: 以百分比形式分配 GPU 内存
- **互斥建议**: 建议不要与 `nvidia.com/gpumem` 同时设置；如同时设置，`gpumem` 优先生效（来源：`pkg/device/nvidia/device.go` 的 `Fit` 实现）
- **独占支持**: 设置为 100 时配合 `nvidia.com/gpucores: 100` 可实现 GPU 独占

#### 3.4.3 使用示例

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
    nvidia.com/gpumem-percentage: 50 # 分配 50% 的 GPU 内存
    nvidia.com/gpucores: 50 # 分配 50% 的 GPU 算力
```

#### 3.4.4 代码分析

**1. 资源请求解析**：

在 `pkg/device/nvidia/device.go` 的 `GenerateResourceRequests` 函数中，HAMi 会解析 `gpumem-percentage` 并将其转换为内部请求结构。

- **读取配置与优先级处理**：代码首先尝试读取绝对内存值 (`gpumem`)，然后读取百分比值 (`gpumem-percentage`)。
- **默认行为**：如果两者都未设置，且未配置全局默认内存值，则默认设置为 100% 内存。

```go
// pkg/device/nvidia/device.go

func (dev *NvidiaGPUDevices) GenerateResourceRequests(ctr *corev1.Container) device.ContainerDeviceRequest {
    // ... (省略部分代码)

    // 读取 nvidia.com/gpumem-percentage
    mempnum := int32(101) // 101 为未设置的标记值
    mem, ok := ctr.Resources.Limits[resourceMemPercentage]
    if !ok {
        mem, ok = ctr.Resources.Requests[resourceMemPercentage]
    }
    if ok {
        mempnums, ok := mem.AsInt64()
        if ok {
            mempnum = int32(mempnums)
        }
    }

    // 默认逻辑：如果未设置百分比且未设置绝对内存，则根据配置决定默认行为
    if mempnum == 101 && memnum == 0 {
        if dev.config.DefaultMemory != 0 {
            memnum = int(dev.config.DefaultMemory)
        } else {
            mempnum = 100 // 默认为 100%
        }
    }

    // 返回标准化的设备请求结构
    return device.ContainerDeviceRequest{
        // ...
        Memreq:           int32(memnum),  // 绝对内存值
        MemPercentagereq: int32(mempnum), // 百分比值
        // ...
    }
}
```

**2. 调度与分配逻辑**：

在 `pkg/device/nvidia/device.go` 的 `Fit` 函数中，调度器会根据设备的总显存将百分比转换为实际的显存大小进行校验。

- **动态转换**：百分比值 (`MemPercentagereq`) 会根据当前遍历到的设备的总显存 (`dev.Totalmem`) 动态计算为绝对值 (`memreq`)。
- **互斥逻辑**：代码明确了 `Memreq` (来自 `gpumem`) 的优先级高于 `MemPercentagereq`。如果 `Memreq > 0`，则直接使用绝对值，忽略百分比设置。

```go
// pkg/device/nvidia/device.go

func (nv *NvidiaGPUDevices) Fit(devices []*device.DeviceUsage, request device.ContainerDeviceRequest, pod *corev1.Pod, nodeInfo *device.NodeInfo, allocated *device.PodDevices) (bool, map[string]device.ContainerDevices, string) {
    // ...
    k := request
    for i := len(devices) - 1; i >= 0; i-- {
        dev := devices[i]

        // ...

        memreq := int32(0)
        // 优先级 1: 如果设置了绝对内存值 (gpumem)，直接使用
        if k.Memreq > 0 {
            memreq = k.Memreq
        }
        // 优先级 2: 如果未设置绝对值且设置了百分比，则根据设备总显存计算
        if k.MemPercentagereq != 101 && k.Memreq == 0 {
            // 计算公式: 设备总显存 * 百分比 / 100
            memreq = dev.Totalmem * k.MemPercentagereq / 100
        }

        // 检查配额与剩余显存
        if dev.Totalmem-dev.Usedmem < memreq {
            // 显存不足，记录原因并跳过该设备
            reason[common.CardInsufficientMemory]++
            continue
        }

        // ...
    }
}
```

---

## 4. 配置管理

HAMi 提供了多层次的配置管理机制，支持从全局到 Pod 级别的精细控制。配置遵循层级优先级：Pod 注解 > 节点注解 > 全局配置。这种设计允许管理员在集群级别设置默认策略，同时在需要时进行细粒度的个性化配置。

### 4.1 全局配置

通过 `hami-scheduler-device` ConfigMap 管理调度器的设备配置（Helm Chart 默认渲染为 `<release>-scheduler-device`，来源：`charts/hami/templates/scheduler/device-configmap.yaml`）：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hami-scheduler-device # 示例名称，实际取决于 Helm Release 名称
data:
  device-config.yaml: |-
    nvidia:
      # GPU 设备内存虚拟化缩放比例，支持超分配（实验性功能）
      # 取值范围：大于 0 的浮点数，默认值：1
      # 当设置为大于 1 时，可实现 GPU 内存的虚拟化超分配
      deviceMemoryScaling: 1

      # 单个 GPU 设备可同时分配的最大任务数量
      # 取值范围：正整数，默认值：10
      # 控制 GPU 资源共享的并发度，影响调度器的分配策略
      deviceSplitCount: 10

      # Pod 未指定 GPU 内存时的默认内存分配量（单位：MB）
      # 取值范围：非负整数，默认值：0
      # 0 表示使用 GPU 的 100% 内存，非 0 值表示具体的内存分配量
      defaultMemory: 0

      # Pod 未指定 GPU 核心时的默认核心分配百分比
      # 取值范围：0-100 的整数，默认值：0
      # 0 表示可适配任何有足够内存的 GPU，100 表示独占整个 GPU
      defaultCores: 0

      # Pod 未指定 nvidia.com/gpu 时的默认 GPU 数量
      # 取值范围：正整数，默认值：1
      # 当 Pod 指定了 gpumem、gpumem-percentage 或 gpucores 时自动补齐此默认值
      defaultGPUNum: 1
```

`nvidia.disablecorelimit` 由 `hami-device-plugin` 的启动参数 `--disable-core-limit` 控制（来源：`charts/hami/templates/device-plugin/daemonsetnvidia.yaml`），不属于 `device-config.yaml` 的字段。

### 4.2 节点级配置

通过节点注解进行调度策略配置。详细的调度策略说明请参见第 5.1 节。

```yaml
apiVersion: v1
kind: Node
metadata:
  annotations:
    hami.io/node-scheduler-policy: "binpack" # 节点调度策略
    hami.io/gpu-scheduler-policy: "spread" # GPU调度策略
```

### 4.3 Pod 级配置

通过 Pod 注解进行精细控制：

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/use-gputype: "GeForce-RTX-3080" # 指定GPU类型
    nvidia.com/use-gpuuuid: "GPU-12345678" # 指定GPU UUID
    # nvidia.com/vgpu-mode: "mig" # 可选值: "mig", "mps", "hami-core"，不设置表示默认共享
spec:
  containers:
    - name: gpu-container
      env:
        - name: GPU_CORE_UTILIZATION_POLICY
          value: "default" # 核心利用率策略
```

---

## 5. 高级特性

本章详细介绍 HAMi 的高级功能特性，包括调度策略、设备选择、NUMA 绑定、MIG 支持、运行时模式选择和监控能力。这些特性为生产环境提供了灵活的配置选项和精细的控制能力。

### 5.1 调度策略

HAMi 支持多种 GPU 调度策略，可通过注解进行配置：

#### 5.1.1 节点级调度策略

```yaml
apiVersion: v1
kind: Node
metadata:
  annotations:
    hami.io/node-scheduler-policy: "binpack" # 紧凑分配策略
    # 可选值: "binpack", "spread"
```

#### 5.1.2 GPU 级调度策略

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    hami.io/gpu-scheduler-policy: "spread" # GPU 分散策略
    # 可选值: "binpack", "spread", "topology-aware"
spec:
  # Pod 规格定义
```

#### 5.1.3 调度策略说明

| **策略名称**     | **作用范围**  | **效果描述**                               |
| ---------------- | ------------- | ------------------------------------------ |
| `binpack`        | 节点级/GPU 级 | 优先使用已有负载的资源，提高资源利用率     |
| `spread`         | 节点级/GPU 级 | 分散分配，提高可用性和负载均衡             |
| `topology-aware` | GPU 级        | 优先选择拓扑更优（例如同一 NUMA 域）的 GPU |

策略名称来源：`pkg/util/types.go` 中 `NodeSchedulerPolicy*` 与 `GPUSchedulerPolicy*` 常量定义。

### 5.2 GPU 类型和设备选择

#### 5.2.1 指定 GPU 类型

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/use-gputype: "GeForce-RTX-3080,Tesla-V100" # 指定可用GPU类型
    nvidia.com/nouse-gputype: "GeForce-GTX-1080" # 排除特定GPU类型
spec:
  containers:
    - name: gpu-container
      resources:
        limits:
          nvidia.com/gpu: 1
```

#### 5.2.2 指定 GPU UUID

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/use-gpuuuid: "GPU-12345678-1234-1234-1234-123456789012" # 指定GPU UUID
    nvidia.com/nouse-gpuuuid: "GPU-87654321-4321-4321-4321-210987654321" # 排除特定GPU UUID
spec:
  # Pod 规格定义
```

### 5.3 NUMA 绑定和拓扑感知

#### 5.3.1 NUMA 绑定

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/numa-bind: "true" # 启用 NUMA 绑定
spec:
  containers:
    - name: gpu-container
      resources:
        limits:
          nvidia.com/gpu: 2 # 多 GPU 时会优先选择同一 NUMA 域的 GPU
```

#### 5.3.2 拓扑感知调度

HAMi 支持基于 GPU 拓扑的智能调度，在多 GPU 分配时会考虑：

- **NVLink 连接**: 优先分配有 NVLink 连接的 GPU 组合
- **PCIe 拓扑**: 考虑 PCIe 总线的带宽和延迟
- **NUMA 域**: 优先在同一 NUMA 域内分配 GPU

### 5.4 MIG 支持

#### 5.4.1 静态 MIG 配置

HAMi 支持 NVIDIA MIG (Multi-Instance GPU) 技术：

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/vgpu-mode: "mig" # 指定使用 MIG 模式
spec:
  containers:
    - name: mig-container
      resources:
        limits:
          nvidia.com/gpu: 1
          nvidia.com/gpumem: 5000 # MIG 实例内存大小
```

#### 5.4.2 动态 MIG 支持

HAMi 支持通过 `mig-parted` 进行动态 MIG 实例管理。（来源：`HAMi/docs/dynamic-mig-support_cn.md`）

- **自动切分与选择**: `hami-device-plugin` 会基于 device-config 中的 MIG template 选择并创建可用实例
- **动态调整**: 在任务调度与资源回收过程中按需调整 MIG template
- **统一 API**: 任务仍使用 `nvidia.com/gpu` 与 `nvidia.com/gpumem` 声明资源；可选通过 `nvidia.com/vgpu-mode: "mig"` 将任务限定在 MIG 节点

启用方式与前置条件来源：`HAMi/docs/dynamic-mig-support_cn.md` 的 “Enabling Dynamic-mig Support” 小节。

### 5.5 运行时模式选择

#### 5.5.1 vGPU 模式配置

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    # nvidia.com/vgpu-mode: "mig" # 可选值: "mig", "mps", "hami-core"，不设置表示默认共享
spec:
  containers:
    - name: gpu-container
      env:
        - name: GPU_CORE_UTILIZATION_POLICY
          value: "default" # 核心利用率策略
          # 可选值: "default", "force", "disable"
```

#### 5.5.2 运行时类配置

```yaml
apiVersion: v1
kind: Pod
spec:
  runtimeClassName: "nvidia" # 指定运行时类
  containers:
    - name: gpu-container
      resources:
        limits:
          nvidia.com/gpu: 1
```

#### 5.5.3 MPS 模式说明

MPS（Multi-Process Service）是 NVIDIA 提供的一种 GPU 共享机制，用于在多进程 CUDA 负载（例如 MPI 场景）下提升并发利用率。（来源：NVIDIA MPS 文档：<https://docs.nvidia.com/deploy/mps/index.html>）

HAMi 对 MPS 的处理可以拆为两层：

第一层是“调度约束”。HAMi 通过 Pod 注解 `nvidia.com/vgpu-mode: "mps"` 将任务约束到 `Mode` 匹配的设备/节点上（来源：`pkg/device/nvidia/device.go` 中 `AllocateMode` / `MpsMode` 常量与 `checkType()` 的模式过滤逻辑）。

第二层是“运行时能力”。当设备插件侧选择 MPS Sharing Strategy 时，启动参数会拒绝 `--mig-strategy=mixed`，并要求显式指定 `--mps-root`（来源：`cmd/device-plugin/nvidia/main.go` 的 `validateFlags()`）。但需要注意：本仓库内 `pkg/device-plugin/nvidiadevice/nvinternal/plugin/mps.go` 的 MPS 相关逻辑为占位实现（`getMPSOptions()` / `waitForDaemon()` / `updateReponse()` 均为空操作），因此 `nvidia.com/vgpu-mode: "mps"` 目前不应被理解为“HAMi 会自动完成 MPS control daemon 的部署、启动与环境注入”。（来源：`pkg/device-plugin/nvidiadevice/nvinternal/plugin/mps.go`）

下面给出 Pod 侧的声明方式示例（资源声明与共享模式一致，差异在于 `vgpu-mode` 约束）：

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/vgpu-mode: "mps" # 约束调度到 MPS 模式节点（若节点侧已启用对应模式）
spec:
  containers:
    - name: mps-container
      resources:
        limits:
          nvidia.com/gpu: 1 # 申请 1 张物理 GPU
          nvidia.com/gpucores: 50 # 可选：限制算力占比（0-100）
          nvidia.com/gpumem: 4096 # 可选：限制显存（MB）
```

可观测性方面，调度器监控端暴露的 `nodeGPUMigInstance` 指标描述中预留了 MPS 值语义（`0` 表示 hami-core，`1` 表示 mig，`2` 表示 mps）（来源：`cmd/scheduler/metrics.go` 的指标定义）。

### 5.6 监控和可观测性

#### 5.6.1 资源使用率监控

HAMi 提供多层次的 GPU 使用率监控：

- **节点级监控**: 通过 `HostCoreUtilization` 指标监控整个 GPU 卡的利用率
- **容器级监控**: 通过 `Device_utilization_desc_of_container` 指标监控容器维度的 SM 利用率（来源：`cmd/vGPUmonitor/metrics.go` 中 `DeviceSmUtil` 上报逻辑）
- **内存监控**: 通过 `vGPU_device_memory_usage_in_bytes` / `vGPU_device_memory_limit_in_bytes` 指标监控 vGPU 内存使用与限制

#### 5.6.2 Prometheus 指标

```yaml
# 主要监控指标
- HostGPUMemoryUsage: GPU 设备内存已用量（bytes）
- HostCoreUtilization: GPU 核心利用率（0-100）
- vGPU_device_memory_usage_in_bytes: vGPU 设备内存使用量（bytes）
- vGPU_device_memory_limit_in_bytes: vGPU 设备内存限制（bytes）
- Device_utilization_desc_of_container: 容器维度 SM 利用率（0-100）
```

指标名称与标签来源：`cmd/vGPUmonitor/metrics.go` 中 Prometheus 指标定义。

---

## 6. 故障排查和最佳实践

本章提供 HAMi GPU 资源管理的故障排查指南和性能优化最佳实践。内容包括常见问题诊断方法、性能优化建议以及生产环境中的配置经验。通过本章内容，您可以快速定位和解决 GPU 资源相关的各类问题。

### 6.1 常见问题诊断

#### 6.1.1 资源分配失败

**问题现象**: Pod 一直处于 Pending 状态，事件显示 GPU 资源不足

**排查步骤**:

```bash
# 1. 检查节点 GPU 资源状态
kubectl describe node <node-name> | grep nvidia.com

# 2. 查看 HAMi 调度器日志
kubectl logs -n kube-system deployment/hami-scheduler

# 3. 检查设备插件状态
kubectl get pods -n kube-system | grep hami-device-plugin
```

**常见原因**:

- GPU 内存碎片化：多个小任务占用导致无法分配大内存需求
- 算力超分配：`gpucores` 总和超过 100%
- 设备类型不匹配：指定的 GPU 类型在节点上不存在

#### 6.1.2 性能隔离问题

**问题现象**: 容器 GPU 利用率超出预期，影响其他容器

**解决方案**:

```yaml
# 启用严格的核心利用率限制
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/vgpu-mode: "hami-core"
spec:
  containers:
    - name: gpu-container
      env:
        - name: GPU_CORE_UTILIZATION_POLICY
          value: "force" # 强制执行利用率限制
```

#### 6.1.3 内存超限问题

**问题现象**: 容器因 GPU 内存不足被终止

**排查方法**:

```bash
# 查看容器 GPU 内存使用情况
kubectl exec <pod-name> -- nvidia-smi

# 检查HAMi监控指标
curl http://<node-ip>:9394/metrics | grep vGPU_device_memory
```

### 6.2 性能优化建议

#### 6.2.1 资源配置优化

**内存分配策略**:

```yaml
# 推荐：使用固定内存分配，避免碎片化
nvidia.com/gpumem: 4000 # 而非 nvidia.com/gpumem-percentage: 50

# 大内存任务：优先使用独占模式
nvidia.com/gpu: 1
nvidia.com/gpucores: 100
nvidia.com/gpumem-percentage: 100
```

**算力分配策略**:

```yaml
# 计算密集型任务：分配足够的核心资源
nvidia.com/gpucores: 80

# 推理任务：可以使用较少的核心资源
nvidia.com/gpucores: 25
```

#### 6.2.2 调度策略优化

```yaml
# 高性能计算场景：使用 NUMA 绑定
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/numa-bind: "true"
    hami.io/gpu-scheduler-policy: "topology-aware"

# 高可用场景：使用分散策略
apiVersion: v1
kind: Pod
metadata:
  annotations:
    hami.io/gpu-scheduler-policy: "spread"
```

---

## 7. 多厂商 GPU 支持

除了 NVIDIA GPU 外，HAMi 还支持多种其他厂商的 GPU 和 AI 加速设备：

### 7.1 支持的设备类型

| **厂商** | **设备类型** | **资源名称**                                                                                 | **特性支持**                     |
| -------- | ------------ | -------------------------------------------------------------------------------------------- | -------------------------------- |
| NVIDIA   | GPU          | `nvidia.com/gpu`, `nvidia.com/gpucores`, `nvidia.com/gpumem`, `nvidia.com/gpumem-percentage` | 完整虚拟化、MIG、监控            |
| 寒武纪   | MLU          | `cambricon.com/mlu`, `cambricon.com/mlu.smlu.vmemory`, `cambricon.com/mlu.smlu.vcore`        | 设备共享、内存限制、核心限制     |
| 海光     | DCU          | `hygon.com/dcunum`, `hygon.com/dcumem`, `hygon.com/dcucores`                                 | 设备虚拟化、资源隔离             |
| 天数智芯 | GPU          | `iluvatar.ai/vcuda-core`, `iluvatar.ai/vcuda-memory`                                         | 100 单元粒度切分                 |
| 摩尔线程 | GPU          | `mthreads.com/vgpu`, `mthreads.com/sgpu-memory`                                              | 设备共享、MT-CloudNative Toolkit |
| 华为昇腾 | NPU          | `huawei.com/Ascend910A`, `huawei.com/Ascend910A-memory`                                      | 模板化虚拟化、AI 核心控制        |
| 沐曦     | GPU          | `metax-tech.com/gpu`                                                                         | 设备复用、拓扑感知               |

### 7.2 通用资源声明模式

HAMi 为不同厂商的设备提供了统一的资源声明模式：

#### 7.2.1 基础设备分配

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
    - name: ai-workload
      resources:
        limits:
          # NVIDIA GPU
          nvidia.com/gpu: 1
          nvidia.com/gpumem: 4000

          # 寒武纪 MLU
          # cambricon.com/mlu: 1

          # 海光 DCU
          # hygon.com/dcu: 1

          # 天数智芯 GPU
          # iluvatar.ai/vcuda-core: 50
          # iluvatar.ai/vcuda-memory: 8000

          # 华为昇腾 NPU
          # huawei.com/Ascend910: 1
          # huawei.com/npu-memory: 16000
```

#### 7.2.2 厂商特定配置

不同厂商的设备可能需要特定的配置和依赖：

**天数智芯 GPU**:

```bash
# 需要部署 gpu-manager
helm install hami hami-charts/hami \
  --set iluvatarResourceMem=iluvatar.ai/vcuda-memory \
  --set iluvatarResourceCore=iluvatar.ai/vcuda-core
```

**摩尔线程 GPU**:

```bash
# 需要部署 MT-CloudNative Toolkit
# 联系厂商获取相关组件
```

**华为昇腾 NPU**:

```yaml
# 支持模板化虚拟化
apiVersion: v1
kind: Pod
metadata:
  annotations:
    huawei.com/npu-template: "1c8g" # 1核心8GB模板
spec:
  containers:
    - name: npu-container
      resources:
        limits:
          huawei.com/Ascend910: 1
```

### 7.3 设备粒度和虚拟化策略

不同厂商的设备采用不同的虚拟化粒度：

- **NVIDIA GPU**: 基于内存和核心百分比的灵活切分
- **天数智芯 GPU**: 100 单元粒度切分
- **华为昇腾 NPU**: 基于预定义模板的虚拟化
- **其他厂商**: 根据硬件特性采用相应的虚拟化策略

### 7.4 监控和可观测性

HAMi 为不同厂商的设备提供统一的监控接口：

```yaml
# 通用监控指标
- device_memory_usage_bytes: 设备内存使用量
- device_memory_limit_bytes: 设备内存限制
- device_utilization_percent: 设备利用率
- device_core_usage: 设备核心使用情况
```

---
