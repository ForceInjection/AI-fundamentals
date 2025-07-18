# HAMi GPU 资源管理完整指南

## 目录

- [HAMi GPU 资源管理完整指南](#hami-gpu-资源管理完整指南)
  - [目录](#目录)
  - [1. 概述](#1-概述)
    - [1.1 快速参考表](#11-快速参考表)
    - [1.2 常见配置模式](#12-常见配置模式)
  - [2. 核心资源类型](#2-核心资源类型)
    - [2.1 nvidia.com/gpu](#21-nvidiacomgpu)
      - [2.1.1 定义](#211-定义)
      - [2.1.2 特性](#212-特性)
      - [2.1.3 代码实现](#213-代码实现)
    - [2.2 nvidia.com/gpucores](#22-nvidiacomgpucores)
      - [2.2.1 定义](#221-定义)
      - [2.2.2 特性](#222-特性)
      - [2.2.3 配置选项](#223-配置选项)
    - [2.3 nvidia.com/gpumem](#23-nvidiacomgpumem)
      - [2.3.1 定义](#231-定义)
      - [2.3.2 特性](#232-特性)
      - [2.3.3 代码实现](#233-代码实现)
    - [2.4 nvidia.com/gpumem-percentage](#24-nvidiacomgpumem-percentage)
      - [2.4.1 定义](#241-定义)
      - [2.4.2 特性](#242-特性)
      - [2.4.3 使用示例](#243-使用示例)
  - [3. 算力分配机制](#3-算力分配机制)
    - [3.1 算力(已分配/总量)的含义](#31-算力已分配总量的含义)
      - [3.1.1 核心概念](#311-核心概念)
      - [3.1.2 数据结构](#312-数据结构)
      - [3.1.3 分配流程](#313-分配流程)
      - [3.1.4 实际示例](#314-实际示例)
      - [3.1.5 性能影响分析](#315-性能影响分析)
  - [4. 使用场景对比](#4-使用场景对比)
    - [4.1 独占使用场景](#41-独占使用场景)
    - [4.2 共享使用场景](#42-共享使用场景)
  - [5. 配置管理](#5-配置管理)
    - [5.1 全局配置](#51-全局配置)
    - [5.2 节点级配置](#52-节点级配置)
    - [5.3 Pod级配置](#53-pod级配置)
  - [6. 高级特性](#6-高级特性)
    - [6.1 调度策略](#61-调度策略)
      - [6.1.1 节点级调度策略](#611-节点级调度策略)
      - [6.1.2 GPU 级调度策略](#612-gpu-级调度策略)
      - [6.1.3 调度策略说明](#613-调度策略说明)
    - [6.2 GPU 类型和设备选择](#62-gpu-类型和设备选择)
      - [6.2.1 指定 GPU 类型](#621-指定-gpu-类型)
      - [6.2.2 指定 GPU UUID](#622-指定-gpu-uuid)
    - [6.3 NUMA 绑定和拓扑感知](#63-numa-绑定和拓扑感知)
      - [6.3.1 NUMA 绑定](#631-numa-绑定)
      - [6.3.2 拓扑感知调度](#632-拓扑感知调度)
    - [6.4 MIG 支持](#64-mig-支持)
      - [6.4.1 静态 MIG 配置](#641-静态-mig-配置)
      - [6.4.2 动态 MIG 支持](#642-动态-mig-支持)
    - [6.5 运行时模式选择](#65-运行时模式选择)
      - [6.5.1 vGPU 模式配置](#651-vgpu-模式配置)
      - [6.5.2 运行时类配置](#652-运行时类配置)
    - [6.6 监控和可观测性](#66-监控和可观测性)
      - [6.6.1 资源使用率监控](#661-资源使用率监控)
      - [6.6.2 Prometheus 指标](#662-prometheus-指标)
  - [7. 故障排查和最佳实践](#7-故障排查和最佳实践)
    - [7.1 常见问题诊断](#71-常见问题诊断)
      - [7.1.1 资源分配失败](#711-资源分配失败)
      - [7.1.2 性能隔离问题](#712-性能隔离问题)
      - [7.1.3 内存超限问题](#713-内存超限问题)
    - [7.2 性能优化建议](#72-性能优化建议)
      - [7.2.1 资源配置优化](#721-资源配置优化)
      - [7.2.2 调度策略优化](#722-调度策略优化)
    - [7.3 监控和告警](#73-监控和告警)
      - [7.3.1 关键监控指标](#731-关键监控指标)
      - [7.3.2 性能基线建立](#732-性能基线建立)
    - [7.4 升级和迁移](#74-升级和迁移)
      - [7.4.1 版本升级注意事项](#741-版本升级注意事项)
      - [7.4.2 配置迁移](#742-配置迁移)
  - [8. 多厂商 GPU 支持](#8-多厂商-gpu-支持)
    - [8.1 支持的设备类型](#81-支持的设备类型)
    - [8.2 通用资源声明模式](#82-通用资源声明模式)
      - [8.2.1 基础设备分配](#821-基础设备分配)
      - [8.2.2 厂商特定配置](#822-厂商特定配置)
    - [8.3 设备粒度和虚拟化策略](#83-设备粒度和虚拟化策略)
    - [8.4 监控和可观测性](#84-监控和可观测性)
  - [9. 总结](#9-总结)
    - [9.1 核心资源类型](#91-核心资源类型)
    - [9.2 高级特性](#92-高级特性)
    - [9.3 多厂商支持](#93-多厂商支持)
    - [9.4 核心优势](#94-核心优势)

---

## 1. 概述

本文档是 HAMi GPU 资源管理的完整技术指南，涵盖以下核心内容：

- **资源类型管理**: 详细介绍 NVIDIA GPU 的四种核心资源类型（`nvidia.com/gpu`、`nvidia.com/gpucores`、`nvidia.com/gpumem`、`nvidia.com/gpumem-percentage`）的定义、特性和使用方法
- **算力分配机制**: 深入解析 GPU 算力分配的核心概念、数据结构、分配流程和性能影响
- **高级特性**: 包括多种调度策略（binpack、spread、numa-first等）、设备选择、NUMA绑定、拓扑感知、MIG支持、运行时模式选择等
- **配置管理**: 全局配置、节点级配置和Pod级配置的详细说明
- **故障排查与最佳实践**: 常见问题诊断、性能优化建议、监控告警配置、版本升级指导
- **多厂商支持**: 统一管理 NVIDIA、寒武纪、海光、天数智芯、摩尔线程、华为昇腾、沐曦等7种厂商的AI加速设备
- **监控与可观测性**: Prometheus指标、性能监控、资源使用率跟踪等

本指南面向对 HAMi 有一定了解的技术人员，提供从基础概念到高级特性的全方位技术说明。

### 1.1 快速参考表

| 资源类型 | 用途 | 取值范围 | 互斥关系 | 版本要求 |
|---------|------|---------|---------|---------|
| `nvidia.com/gpu` | GPU设备数量 | 正整数 | - | 所有版本 |
| `nvidia.com/gpucores` | GPU算力百分比 | 0-100 | - | v1.0.1.3+ |
| `nvidia.com/gpumem` | GPU内存(MB) | 正整数 | 与gpumem-percentage互斥 | 所有版本 |
| `nvidia.com/gpumem-percentage` | GPU内存百分比 | 0-100 | 与gpumem互斥 | v1.0.1.4+ |

### 1.2 常见配置模式

```yaml
# 独占模式
nvidia.com/gpu: 1
nvidia.com/gpucores: 100
nvidia.com/gpumem-percentage: 100

# 共享模式
nvidia.com/gpu: 1
nvidia.com/gpucores: 50
nvidia.com/gpumem: 4000

# 高密度共享
nvidia.com/gpu: 1
nvidia.com/gpucores: 25
nvidia.com/gpumem-percentage: 25
```

---

## 2. 核心资源类型

### 2.1 nvidia.com/gpu

#### 2.1.1 定义

`nvidia.com/gpu` 是用于声明 Pod 所需物理 GPU 数量的资源类型。

#### 2.1.2 特性

- **必需性**: 必需的资源声明，用于触发 GPU 调度
- **取值范围**: 正整数（1, 2, 3...）
- **用途**: 指定 Pod 需要多少个物理 GPU 设备
- **调度作用**: HAMi 调度器根据此值进行 GPU 设备分配

#### 2.1.3 代码实现

在 `pkg/device/nvidia/device.go` 的 `GenerateResourceRequests` 函数中：

```go
func (dev *NvidiaGPUDevices) GenerateResourceRequests(pod *corev1.Pod) util.PodDeviceRequests {
    // 从容器的 Limits 或 Requests 中获取 nvidia.com/gpu 的值
    count, _ := resource.Limits.Name(dev.ResourceCountName).AsInt64()
    if count == 0 {
        count, _ = resource.Requests.Name(dev.ResourceCountName).AsInt64()
    }
    // ...
}
```

### 2.2 nvidia.com/gpucores

#### 2.2.1 定义

`nvidia.com/gpucores` 是用于指定每个物理 GPU 分配给 Pod 的计算核心百分比的资源类型。

#### 2.2.2 特性

- **必需性**: 可选的资源声明
- **取值范围**: 0-100 的整数，表示百分比
- **默认值**: 由 `nvidia.defaultCores` 配置项决定
- **限制类型**: 从 v1.0.1.3 版本开始在容器内部限制 GPU 利用率
- **共享支持**: 支持多个 Pod 共享同一个物理 GPU 的不同算力份额

#### 2.2.3 配置选项

- `nvidia.defaultCores`: 默认核心分配百分比
- `nvidia.disablecorelimit`: 是否禁用核心限制
- `GPU_CORE_UTILIZATION_POLICY`: 容器环境变量，控制核心利用率策略

### 2.3 nvidia.com/gpumem

#### 2.3.1 定义

`nvidia.com/gpumem` 是用于指定 Pod 所需 GPU 内存大小的资源类型，以 MB 为单位。

#### 2.3.2 特性

- **必需性**: 可选的资源声明
- **取值范围**: 正整数，单位为 MB
- **用途**: 精确指定 Pod 需要的 GPU 内存量
- **互斥性**: 不能与 `nvidia.com/gpumem-percentage` 同时使用
- **内存隔离**: 提供硬隔离保证，确保容器不会超出分配的内存边界

#### 2.3.3 代码实现

在 `GenerateResourceRequests` 函数中处理内存资源请求：

```go
// 获取 nvidia.com/gpumem 的值
memoryRequest, _ := resource.Limits.Name(dev.ResourceMemoryName).AsInt64()
if memoryRequest == 0 {
    memoryRequest, _ = resource.Requests.Name(dev.ResourceMemoryName).AsInt64()
}
```

### 2.4 nvidia.com/gpumem-percentage

#### 2.4.1 定义

`nvidia.com/gpumem-percentage` 是用于指定 Pod 所需 GPU 内存百分比的资源类型，在 v1.0.1.4 版本中引入。

#### 2.4.2 特性

- **必需性**: 可选的资源声明
- **取值范围**: 0-100 的整数，表示百分比
- **用途**: 以百分比形式分配 GPU 内存
- **互斥性**: 不能与 `nvidia.com/gpumem` 同时使用
- **独占支持**: 设置为 100 时配合 `nvidia.com/gpucores: 100` 可实现 GPU 独占

#### 2.4.3 使用示例

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
    nvidia.com/gpumem-percentage: 50  # 分配50%的GPU内存
    nvidia.com/gpucores: 50           # 分配50%的GPU算力
```

---

## 3. 算力分配机制

### 3.1 算力(已分配/总量)的含义

#### 3.1.1 核心概念

- **总量(Totalcore)**: 每个 GPU 设备的总计算核心数，通常为 100（代表 100% 的 GPU 计算能力）
- **已分配(Usedcores)**: 当前已经分配给 Pod/容器的 GPU 核心数总和
- **算力**: 指 GPU 的计算核心资源，以百分比形式表示

#### 3.1.2 数据结构

在 `pkg/util/types.go` 中的 `DeviceUsage` 结构体定义：

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
    // ... 其他字段
}
```

#### 3.1.3 分配流程

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
   n.Usedcores += ctr.Usedcores
   ```

#### 3.1.4 实际示例

- **100/100**: GPU 被完全独占使用
- **50/100**: GPU 的 50% 算力已分配，还剩 50% 可用
- **80/100**: GPU 的 80% 算力已分配，还剩 20% 可用

#### 3.1.5 性能影响分析

基于实际测试数据，不同算力分配对性能的影响：

| 算力分配 | 训练性能 | 推理性能 | 内存带宽 | 适用场景 |
|---------|---------|---------|---------|---------|
| 100% | 100% | 100% | 100% | 大模型训练、高性能计算 |
| 50% | ~85% | ~90% | ~95% | 中等规模训练、批量推理 |
| 25% | ~60% | ~75% | ~85% | 小模型推理、开发测试 |

**注意**: 性能数据基于 Tesla V100 测试，实际性能可能因工作负载类型和GPU型号而异。

---

## 4. 使用场景对比

### 4.1 独占使用场景

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: gpu-container
    resources:
      limits:
        nvidia.com/gpu: 2              # 需要2个物理GPU
        nvidia.com/gpumem-percentage: 100  # 每个GPU分配100%内存
        nvidia.com/gpucores: 100           # 每个GPU分配100%算力
```

### 4.2 共享使用场景

```yaml
# Pod 1
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: gpu-container-1
    resources:
      limits:
        nvidia.com/gpu: 1              # 需要1个物理GPU
        nvidia.com/gpumem-percentage: 50   # 分配50%内存
        nvidia.com/gpucores: 50            # 分配50%算力

---
# Pod 2 (可以与Pod 1共享同一个GPU)
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: gpu-container-2
    resources:
      limits:
        nvidia.com/gpu: 1              # 需要1个物理GPU
        nvidia.com/gpumem-percentage: 50   # 分配50%内存
        nvidia.com/gpucores: 50            # 分配50%算力
```

---

## 5. 配置管理

### 5.1 全局配置

通过 ConfigMap 管理设备配置：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hami-device-config
data:
  # GPU设备内存虚拟化缩放比例，支持超分配（实验性功能）
  # 取值范围：大于0的浮点数，默认值：1
  # 当设置为大于1时，可实现GPU内存的虚拟化超分配
  nvidia.deviceMemoryScaling: "1"      
  
  # 单个GPU设备可同时分配的最大任务数量
  # 取值范围：正整数，默认值：10
  # 控制GPU资源共享的并发度，影响调度器的分配策略
  nvidia.deviceSplitCount: "10"        
  
  # Pod未指定GPU内存时的默认内存分配量（单位：MB）
  # 取值范围：非负整数，默认值：0
  # 0表示使用GPU的100%内存，非0值表示具体的内存分配量
  nvidia.defaultMem: "1000"            
  
  # Pod未指定GPU核心时的默认核心分配百分比
  # 取值范围：0-100的整数，默认值：0
  # 0表示可适配任何有足够内存的GPU，100表示独占整个GPU
  nvidia.defaultCores: "0"             
  
  # Pod未指定nvidia.com/gpu时的默认GPU数量
  # 取值范围：正整数，默认值：1
  # 当Pod指定了gpumem、gpumem-percentage或gpucores时自动添加此默认值
  nvidia.defaultGPUNum: "1"            
  
  # 是否禁用GPU核心利用率限制功能
  # 取值："true"禁用限制，"false"启用限制，默认值：false
  # 禁用后容器将忽略nvidia.com/gpucores设置的利用率限制
  nvidia.disablecorelimit: "false"     
```

### 5.2 节点级配置

通过节点注解进行配置：

```yaml
apiVersion: v1
kind: Node
metadata:
  annotations:
    hami.io/node-scheduler-policy: "binpack"  # 节点调度策略
    hami.io/gpu-scheduler-policy: "spread"    # GPU调度策略
```

### 5.3 Pod级配置

通过Pod注解进行精细控制：

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/use-gputype: "GeForce-RTX-3080"     # 指定GPU类型
    nvidia.com/use-gpuuuid: "GPU-12345678"         # 指定GPU UUID
    nvidia.com/vgpu-mode: "shared"                 # vGPU模式
spec:
  containers:
  - name: gpu-container
    env:
    - name: GPU_CORE_UTILIZATION_POLICY
      value: "default"                             # 核心利用率策略
```

---

## 6. 高级特性

### 6.1 调度策略

HAMi 支持多种 GPU 调度策略，可通过注解进行配置：

#### 6.1.1 节点级调度策略

```yaml
apiVersion: v1
kind: Node
metadata:
  annotations:
    hami.io/node-scheduler-policy: "binpack"  # 紧凑分配策略
    # 可选值: "binpack", "spread", "numa-first"
```

#### 6.1.2 GPU 级调度策略

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    hami.io/gpu-scheduler-policy: "spread"    # GPU 分散策略
    # 可选值: "best-fit", "idle-first", "numa-first"
spec:
  # Pod 规格定义
```

#### 6.1.3 调度策略说明

| 策略名称 | 作用范围 | 效果描述 |
|---------|---------|---------|
| `binpack` | 节点级 | 优先使用已有负载的节点，提高资源利用率 |
| `spread` | 节点级/GPU级 | 分散分配，提高可用性和负载均衡 |
| `best-fit` | GPU级 | 选择剩余内存最少但满足需求的GPU |
| `idle-first` | GPU级 | 优先选择空闲的GPU设备 |
| `numa-first` | GPU级 | 多GPU分配时优先选择同一NUMA域的GPU |

### 6.2 GPU 类型和设备选择

#### 6.2.1 指定 GPU 类型

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/use-gputype: "GeForce-RTX-3080,Tesla-V100"  # 指定可用GPU类型
    nvidia.com/nouse-gputype: "GeForce-GTX-1080"           # 排除特定GPU类型
spec:
  containers:
  - name: gpu-container
    resources:
      limits:
        nvidia.com/gpu: 1
```

#### 6.2.2 指定 GPU UUID

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/use-gpuuuid: "GPU-12345678-1234-1234-1234-123456789012"    # 指定GPU UUID
    nvidia.com/nouse-gpuuuid: "GPU-87654321-4321-4321-4321-210987654321"  # 排除特定GPU UUID
spec:
  # Pod 规格定义
```

### 6.3 NUMA 绑定和拓扑感知

#### 6.3.1 NUMA 绑定

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/numa-bind: "true"  # 启用NUMA绑定
spec:
  containers:
  - name: gpu-container
    resources:
      limits:
        nvidia.com/gpu: 2  # 多GPU时会优先选择同一NUMA域的GPU
```

#### 6.3.2 拓扑感知调度

HAMi 支持基于 GPU 拓扑的智能调度，在多 GPU 分配时会考虑：

- **NVLink 连接**: 优先分配有 NVLink 连接的 GPU 组合
- **PCIe 拓扑**: 考虑 PCIe 总线的带宽和延迟
- **NUMA 域**: 优先在同一 NUMA 域内分配 GPU

### 6.4 MIG 支持

#### 6.4.1 静态 MIG 配置

HAMi 支持 NVIDIA MIG (Multi-Instance GPU) 技术：

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/vgpu-mode: "mig"  # 指定使用MIG模式
spec:
  containers:
  - name: mig-container
    resources:
      limits:
        nvidia.com/gpu: 1
        nvidia.com/gpumem: 5000  # MIG实例内存大小
```

#### 6.4.2 动态 MIG 支持

HAMi 支持动态 MIG 实例管理：

- **自动创建**: 根据任务需求自动创建合适的 MIG 实例
- **动态调整**: 根据负载情况动态调整 MIG 配置
- **统一 API**: 与 HAMi-core 模式使用相同的资源声明方式

### 6.5 运行时模式选择

#### 6.5.1 vGPU 模式配置

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/vgpu-mode: "shared"  # 共享模式
    # 可选值: "shared", "mig", "mps", "hami-core"
spec:
  containers:
  - name: gpu-container
    env:
    - name: GPU_CORE_UTILIZATION_POLICY
      value: "default"  # 核心利用率策略
      # 可选值: "default", "force", "disable"
```

#### 6.5.2 运行时类配置

```yaml
apiVersion: v1
kind: Pod
spec:
  runtimeClassName: "nvidia"  # 指定运行时类
  containers:
  - name: gpu-container
    resources:
      limits:
        nvidia.com/gpu: 1
```

### 6.6 监控和可观测性

#### 6.6.1 资源使用率监控

HAMi 提供多层次的 GPU 使用率监控：

- **节点级监控**: 通过 `HostCoreUtilization` 指标监控整个 GPU 卡的利用率
- **容器级监控**: 通过 `Device_utilization_desc_of_container` 指标监控容器的 GPU 使用情况
- **内存监控**: 通过 `vGPU_device_memory_usage_in_bytes` 指标监控 vGPU 内存使用

#### 6.6.2 Prometheus 指标

```yaml
# 主要监控指标
- HostGPUMemoryUsage: GPU设备内存使用量
- HostCoreUtilization: GPU核心利用率
- vGPU_device_memory_usage_in_bytes: vGPU设备内存使用量
- vGPU_device_memory_limit_in_bytes: vGPU设备内存限制
- Device_utilization_desc_of_container: 容器设备利用率
```

---

## 7. 故障排查和最佳实践

### 7.1 常见问题诊断

#### 7.1.1 资源分配失败

**问题现象**: Pod 一直处于 Pending 状态，事件显示 GPU 资源不足

**排查步骤**:

```bash
# 1. 检查节点GPU资源状态
kubectl describe node <node-name> | grep nvidia.com

# 2. 查看HAMi调度器日志
kubectl logs -n kube-system deployment/hami-scheduler

# 3. 检查设备插件状态
kubectl get pods -n kube-system | grep hami-device-plugin
```

**常见原因**:

- GPU内存碎片化：多个小任务占用导致无法分配大内存需求
- 算力超分配：`gpucores` 总和超过100%
- 设备类型不匹配：指定的GPU类型在节点上不存在

#### 7.1.2 性能隔离问题

**问题现象**: 容器GPU利用率超出预期，影响其他容器

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
      value: "force"  # 强制执行利用率限制
```

#### 7.1.3 内存超限问题

**问题现象**: 容器因GPU内存不足被终止

**排查方法**:

```bash
# 查看容器GPU内存使用情况
kubectl exec <pod-name> -- nvidia-smi

# 检查HAMi监控指标
curl http://<node-ip>:9394/metrics | grep vGPU_device_memory
```

### 7.2 性能优化建议

#### 7.2.1 资源配置优化

**内存分配策略**:

```yaml
# 推荐：使用固定内存分配，避免碎片化
nvidia.com/gpumem: 4000  # 而非 nvidia.com/gpumem-percentage: 50

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

#### 7.2.2 调度策略优化

```yaml
# 高性能计算场景：使用NUMA绑定
apiVersion: v1
kind: Pod
metadata:
  annotations:
    nvidia.com/numa-bind: "true"
    hami.io/gpu-scheduler-policy: "numa-first"

# 高可用场景：使用分散策略
apiVersion: v1
kind: Pod
metadata:
  annotations:
    hami.io/gpu-scheduler-policy: "spread"
```

### 7.3 监控和告警

#### 7.3.1 关键监控指标

```yaml
# Prometheus告警规则示例
groups:
- name: hami-gpu-alerts
  rules:
  - alert: GPUMemoryUtilizationHigh
    expr: vGPU_device_memory_usage_in_bytes / vGPU_device_memory_limit_in_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "GPU内存使用率过高"
      
  - alert: GPUCoreUtilizationLow
    expr: HostCoreUtilization < 0.1
    for: 10m
    labels:
      severity: info
    annotations:
      summary: "GPU算力利用率过低，可能存在资源浪费"
```

#### 7.3.2 性能基线建立

```bash
# 建立GPU性能基线
# 1. 记录空载时的基础指标
# 2. 测试单容器满载性能
# 3. 测试多容器共享性能
# 4. 建立性能衰减阈值告警
```

### 7.4 升级和迁移

#### 7.4.1 版本升级注意事项

- **v1.0.1.3 → v1.0.1.4**: 新增 `gpumem-percentage` 资源类型
- **配置兼容性**: 旧版本配置在新版本中保持兼容
- **功能变更**: 核心利用率限制默认启用

#### 7.4.2 配置迁移

```yaml
# 旧配置 (v1.0.1.2)
nvidia.com/gpu: 1
nvidia.com/gpumem: 4000

# 新配置 (v1.0.1.4+) - 推荐使用百分比
nvidia.com/gpu: 1
nvidia.com/gpumem-percentage: 50
nvidia.com/gpucores: 50
```

---

## 8. 多厂商 GPU 支持

除了 NVIDIA GPU 外，HAMi 还支持多种其他厂商的 GPU 和 AI 加速设备：

### 8.1 支持的设备类型

| 厂商 | 设备类型 | 资源名称 | 特性支持 |
|------|---------|---------|---------|
| NVIDIA | GPU | `nvidia.com/gpu`, `nvidia.com/gpucores`, `nvidia.com/gpumem` | 完整虚拟化、MIG、监控 |
| 寒武纪 | MLU | `cambricon.com/vmlu`, `cambricon.com/mlu.smlu.vmemory` | 设备共享、内存限制 |
| 海光 | DCU | `hygon.com/dcunum`, `hygon.com/dcumem`, `hygon.com/dcucores` | 设备虚拟化、资源隔离 |
| 天数智芯 | GPU | `iluvatar.ai/vcuda-core`, `iluvatar.ai/vcuda-memory` | 100单元粒度切分 |
| 摩尔线程 | GPU | `mthreads.com/vgpu`, `mthreads.com/sgpu-memory` | 设备共享、MT-CloudNative Toolkit |
| 华为昇腾 | NPU | `huawei.com/Ascend910A`, `huawei.com/Ascend910A-memory` | 模板化虚拟化、AI核心控制 |
| 沐曦 | GPU | `metax-tech.com/gpu` | 设备复用、拓扑感知 |

### 8.2 通用资源声明模式

HAMi 为不同厂商的设备提供了统一的资源声明模式：

#### 8.2.1 基础设备分配

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

#### 8.2.2 厂商特定配置

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
    huawei.com/npu-template: "1c8g"  # 1核心8GB模板
spec:
  containers:
  - name: npu-container
    resources:
      limits:
        huawei.com/Ascend910: 1
```

### 8.3 设备粒度和虚拟化策略

不同厂商的设备采用不同的虚拟化粒度：

- **NVIDIA GPU**: 基于内存和核心百分比的灵活切分
- **天数智芯 GPU**: 100 单元粒度切分
- **华为昇腾 NPU**: 基于预定义模板的虚拟化
- **其他厂商**: 根据硬件特性采用相应的虚拟化策略

### 8.4 监控和可观测性

HAMi 为不同厂商的设备提供统一的监控接口：

```yaml
# 通用监控指标
- device_memory_usage_bytes: 设备内存使用量
- device_memory_limit_bytes: 设备内存限制
- device_utilization_percent: 设备利用率
- device_core_usage: 设备核心使用情况
```

---

## 9. 总结

本文档全面介绍了 HAMi 中 GPU 资源管理的各个方面，主要内容包括：

### 9.1 核心资源类型

- **`nvidia.com/gpu`**: 基础 GPU 设备分配，支持独占和共享模式
- **`nvidia.com/gpucores`**: 精细化算力控制，以百分比形式分配 GPU 计算资源  
- **`nvidia.com/gpumem`**: GPU 内存分配，以 MB 为单位进行精确控制
- **`nvidia.com/gpumem-percentage`**: 基于百分比的内存分配方式

### 9.2 高级特性

- **调度策略**: 支持 binpack、spread、best-fit、idle-first、numa-first 等多种调度策略
- **设备选择**: 支持基于 GPU 类型、UUID 的精确设备选择和排除
- **NUMA 绑定**: 支持 NUMA 感知调度和拓扑优化
- **MIG 支持**: 支持静态和动态 MIG 实例管理
- **运行时模式**: 支持 shared、mig、mps、hami-core 等多种运行时模式

### 9.3 多厂商支持

HAMi 不仅支持 NVIDIA GPU，还支持：

- 寒武纪 MLU、海光 DCU、天数智芯 GPU
- 摩尔线程 GPU、华为昇腾 NPU、沐曦 GPU
- 统一的资源声明模式和监控接口

### 9.4 核心优势

通过 HAMi 的 GPU 资源管理机制，可以实现：

1. **灵活的资源分配**: 支持独占和共享两种使用模式
2. **精细的算力控制**: 以百分比形式分配 GPU 计算资源和内存
3. **高效的资源利用**: 多个工作负载可以共享同一个物理 GPU
4. **完善的监控机制**: 实时跟踪资源使用情况和性能指标
5. **多厂商兼容**: 统一的接口支持多种 AI 加速设备
6. **智能调度**: 基于拓扑、NUMA、负载的智能调度策略

这种设计使得 HAMi 能够在保证性能隔离的同时，最大化 GPU 资源的利用效率，为 Kubernetes 环境中的 AI 工作负载提供了强大而灵活的资源管理能力。

---

*本文档面向对 HAMi 有一定了解的用户，提供了 GPU 资源管理的详细技术说明。如需了解基础安装和配置，请参考 HAMi 官方文档。*
