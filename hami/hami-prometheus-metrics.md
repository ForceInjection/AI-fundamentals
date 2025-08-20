# HAMi-WebUI Prometheus 指标说明文档

## 1. 概述

HAMi-WebUI 项目通过 `/metrics` 端点暴露了丰富的 Prometheus 指标，用于监控 GPU 虚拟化环境中的资源使用情况、设备状态和系统健康状况。这些指标支持多种 GPU 厂商（NVIDIA、MLU、Hygon、Ascend），为 GPU 资源管理提供全面的可观测性。

## 2. 指标暴露配置

### 2.1 HTTP 端点

- **端点路径**: `/metrics`
- **格式**: Prometheus 标准格式
- **实现**: 基于 `github.com/prometheus/client_golang/prometheus/promhttp`

### 2.2 ServiceMonitor 配置

- **抓取间隔**: 15秒（可配置）
- **自动发现**: 支持 `Prometheus Operator` 自动发现
- **外部集成**: 支持外部 `Prometheus` 实例集成

#### 2.2.1 ServiceMonitor 配置示例

```yaml
serviceMonitor:
  enabled: true
  interval: 15s
  honorLabels: false
  additionalLabels:
    jobRelease: hami-webui-prometheus

hamiServiceMonitor:
  enabled: true
  interval: 15s
  honorLabels: false
  svcNamespace: kube-system
```

#### 2.2.2 外部 Prometheus 集成配置

```yaml
externalPrometheus:
  enabled: false
  address: "http://prometheus-kube-prometheus-prometheus.prometheus.svc.cluster.local:9090"
```

### 2.3 缓存机制

- 指标生成支持缓存机制，默认缓存时间为 1 分钟
- 可通过配置 `EnableMetricsCache` 启用或禁用缓存
- 缓存有助于减少对底层监控系统的查询压力

---

## 3. 指标分类

### 3.1 卡维度指标（设备级别）

#### 3.1.1 资源分配指标

| 指标名称 | 类型 | 描述 | 标签 |
|---------|------|------|------|
| `hami_vcore_scaling` | Gauge | 虚拟核心扩展比例 | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_vmemory_scaling` | Gauge | 虚拟内存扩展比例 | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_vgpu_count` | Gauge | 虚拟GPU数量 | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_vmemory_size` | Gauge | 虚拟内存大小（MB） | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_vcore_size` | Gauge | 虚拟核心大小 | node, provider, devicetype, deviceuuid, driver_version, device_no |

#### 3.1.2 物理资源指标

| 指标名称 | 类型 | 描述 | 标签 |
|---------|------|------|------|
| `hami_memory_size` | Gauge | 物理内存总量（MB） | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_memory_used` | Gauge | 物理内存使用量（MB） | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_memory_util` | Gauge | 内存利用率（0-100%） | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_core_size` | Gauge | 核心总量 | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_core_used` | Gauge | 核心使用量 | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_core_util` | Gauge | 核心利用率（0-100%） | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_core_used_avg` | Gauge | 平均核心使用量 | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_core_util_avg` | Gauge | 平均核心利用率（0-100%） | node, provider, devicetype, deviceuuid, driver_version, device_no |

#### 3.1.3 硬件监控指标

| 指标名称 | 类型 | 描述 | 标签 |
|---------|------|------|------|
| `hami_device_temperature` | Gauge | 设备温度（摄氏度） | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_device_memory_temperature` | Gauge | 内存温度（摄氏度） | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_device_power` | Gauge | 设备功耗（瓦特） | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_device_fan_speed_p` | Gauge | 风扇转速百分比（0-100%） | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_device_fan_speed_r` | Gauge | 风扇转速（RPM） | node, provider, devicetype, deviceuuid, driver_version, device_no |
| `hami_device_hardware_health` | Gauge | 硬件健康状态（0=异常，1=正常） | node, provider, devicetype, deviceuuid, driver_version, device_no |

### 3.2 任务维度指标（容器级别）

#### 3.2.1 资源分配指标

| 指标名称 | 类型 | 描述 | 标签 |
|---------|------|------|------|
| `hami_container_vgpu_allocated` | Gauge | 容器分配的虚拟GPU数量 | node, provider, devicetype, deviceuuid, pod_name, container_name, namespace_name, container_pod_uuid |
| `hami_container_vmemory_allocated` | Gauge | 容器分配的虚拟内存（MB） | node, provider, devicetype, deviceuuid, pod_name, container_name, namespace_name, container_pod_uuid |
| `hami_container_vcore_allocated` | Gauge | 容器分配的虚拟核心 | node, provider, devicetype, deviceuuid, pod_name, container_name, namespace_name, container_pod_uuid |

#### 3.2.2 资源使用指标

| 指标名称 | 类型 | 描述 | 标签 |
|---------|------|------|------|
| `hami_container_memory_used` | Gauge | 容器内存使用量（MB） | node, provider, devicetype, deviceuuid, pod_name, container_name, namespace_name |
| `hami_container_memory_util` | Gauge | 容器内存利用率（0-100%） | node, provider, devicetype, deviceuuid, pod_name, container_name, namespace_name |
| `hami_container_core_used` | Gauge | 容器核心使用量 | node, provider, devicetype, deviceuuid, pod_name, container_name, namespace_name |
| `hami_container_core_util` | Gauge | 容器核心利用率（0-100%） | node, provider, devicetype, deviceuuid, pod_name, container_name, namespace_name |

### 3.3 资源池维度指标

| 指标名称 | 类型 | 描述 | 标签 |
|---------|------|------|------|
| `hami_pool_vcore_size` | Gauge | 资源池虚拟核心总量 | pool |
| `hami_pool_vgpu_count` | Gauge | 资源池虚拟GPU总数 | pool |
| `hami_pool_vmemory_size` | Gauge | 资源池虚拟内存总量（MB） | pool |

### 3.4 系统组件健康指标

| 指标名称 | 类型 | 描述 | 标签 |
|---------|------|------|------|
| `hami_system_component_health` | Gauge | 系统组件健康状态（0=异常，1=正常） | component |

### 3.5 数据精度说明

- **温度指标**: 保留一位小数，单位为摄氏度
- **利用率指标**: 保留两位小数，范围 0-100%
- **内存指标**: 以MB为单位，整数值
- **功耗指标**: 以瓦特为单位，保留两位小数

---

## 4. 指标计算逻辑说明

### 4.1 设备级指标计算逻辑

#### 4.1.1 内存相关指标

**hami_memory_size（物理内存总量）**：

- **数据来源**: 通过 `deviceMemTotal()` 函数查询不同 GPU 提供商的内存总量指标
- **查询逻辑**:
  - NVIDIA: 查询 `DCGM_FI_DEV_FB_TOTAL` 指标
  - Cambricon: 查询 `cndev_memory_total` 指标
  - Ascend: 查询 `npu_chip_info_memory_total` 指标
  - Hygon: 查询 `dcu_memory_total_bytes` 指标
- **单位转换**: 统一转换为MB

**hami_memory_used（物理内存使用量）**：

- **数据来源**: 通过 `deviceMemUsed()` 函数查询内存使用量
- **查询逻辑**:
  - NVIDIA: 查询 `DCGM_FI_DEV_FB_USED` 指标
  - Cambricon: 查询 `cndev_memory_used` 指标
  - Ascend: 查询 `npu_chip_info_memory_used` 指标
  - Hygon: 查询 `dcu_memory_used_bytes` 指标
- **单位转换**: 统一转换为MB

**hami_memory_util（内存利用率）**：

- **计算公式**: `(hami_memory_used / hami_memory_size) * 100`
- **数据精度**: 保留两位小数，范围 0-100%

#### 4.1.2 核心相关指标

**hami_core_util（核心利用率）**：

- **数据来源**: 通过 `deviceCoreUtil()` 函数查询 GPU 利用率
- **查询逻辑**:
  - NVIDIA: 查询 `DCGM_FI_DEV_GPU_UTIL` 指标
  - Cambricon: 查询 `cndev_util` 指标
  - Ascend: 查询 `npu_chip_info_utilization` 指标
  - Hygon: 查询 `dcu_utilization` 指标
- **数据精度**: 保留两位小数，范围 0-100%

**hami_vcore_scaling（虚拟核心扩展比例）**：

- **计算逻辑**: 基于设备的虚拟化配置计算超分配比例
- **数据来源**: 从设备信息中获取虚拟核心分配情况
- **计算公式**: `虚拟核心总分配量 / 物理核心总量`

#### 4.1.3 硬件监控指标

**hami_device_temperature（设备温度）**：

- **数据来源**: 通过 `gpuTemperature()` 函数查询设备温度
- **查询逻辑**:
  - NVIDIA: 查询 `DCGM_FI_DEV_GPU_TEMP` 指标
  - Cambricon: 查询 `cndev_temperature` 指标
  - Ascend: 查询 `npu_chip_info_temperature` 指标
  - Hygon: 查询 `dcu_temperature` 指标
- **单位**: 摄氏度，保留一位小数

**hami_device_power（设备功耗）**：

- **数据来源**: 通过 `gpuPower()` 函数查询设备功耗
- **查询逻辑**:
  - NVIDIA: 查询 `DCGM_FI_DEV_POWER_USAGE` 指标
  - Cambricon: 查询 `cndev_power` 指标
  - Ascend: 查询 `npu_chip_info_power` 指标
  - Hygon: 查询 `dcu_power_usage` 指标
- **单位**: 瓦特（W），保留两位小数

### 4.2 任务级指标计算逻辑

#### 4.2.1 容器核心使用指标

**hami_container_core_used（容器核心使用量）**：

- **数据来源**: 通过 `taskCoreUsed()` 函数查询容器级别的 GPU 使用率
- **查询逻辑**: 基于容器标签过滤，查询对应的 GPU 利用率指标
- **标签过滤**: 使用 `pod_name`, `container_name`, `namespace_name` 等标签进行精确匹配

**hami_container_memory_used（容器内存使用量）**：

- **数据来源**: 通过 `taskMemoryUsed()` 函数查询容器内存使用情况
- **查询逻辑**: 根据不同 GPU 提供商查询对应的内存使用指标：
  - NVIDIA: 查询 `vGPU_device_memory_usage_in_bytes` 指标（已为字节单位）
  - Cambricon: 查询 `mlu_memory_utilization * mlu_container` 指标（百分比形式，需转换）
  - Ascend: 查询 `container_npu_used_memory` 指标（需乘以 1024*1024 转换为字节）
  - Hygon: 查询 `vdcu_usage_memory_size` 指标（已为字节单位）
- **单位处理**: 最终以 MB 为单位存储（除以 1024*1024），但在 Prometheus 中显示为 MB 值

### 4.3 关键代码实现

#### 4.3.1 指标生成核心函数

**代码位置**: `/server/internal/exporter/exporter.go:84`

```go
// GenerateDeviceMetrics 生成设备级指标
func (s *MetricsGenerator) GenerateDeviceMetrics(ctx context.Context) error {
    deviceInfos, err := s.nodeUsecase.ListAllDevices(ctx)
    if err != nil {
        return err
    }
    for _, device := range deviceInfos {
        provider := device.Provider
        // 查询device的驱动版本以及设备号
        deviceAdditional, err := s.queryDeviceAdditional(ctx, provider, device.Id)
        var driver, deviceNo = "", ""
        if err == nil && deviceAdditional != nil {
            driver = deviceAdditional.DriverVersion
            deviceNo = deviceAdditional.DeviceNo
        }

        // 分配率指标
        HamiVgpuCount.WithLabelValues(device.NodeName, provider, device.Type, device.Id, driver, deviceNo).Set(float64(device.Count))
        HamiVmemorySize.WithLabelValues(device.NodeName, provider, device.Type, device.Id, driver, deviceNo).Set(float64(device.Devmem))
        // 设置内存相关指标
        deviceMemUsed, err := s.deviceMemUsed(ctx, provider, device.Id)
        if err == nil {
            HamiMemoryUsed.WithLabelValues(device.NodeName, provider, device.Type, device.Id, driver, deviceNo).Set(float64(deviceMemUsed))
        }
        HamiMemorySize.WithLabelValues(device.NodeName, provider, device.Type, device.Id, driver, deviceNo).Set(float64(device.Devmem))
        // ...
    }
}
```

#### 4.3.2 数据查询函数

**代码位置**: `/server/internal/exporter/exporter.go:355`

```go
// taskMemoryUsed 查询容器内存使用量
func (s *MetricsGenerator) taskMemoryUsed(ctx context.Context, provider, namespace, pod, container, podUUID, deviceUUID string) (float32, error) {
    query := ""
    switch provider {
    case biz.NvidiaGPUDevice:
        query = fmt.Sprintf("avg(vGPU_device_memory_usage_in_bytes{deviceuuid=\"%s\", podnamespace=\"%s\", podname=\"%s\", ctrname=\"%s\"})", deviceUUID, namespace, pod, container)
    case biz.CambriconGPUDevice:
        query = fmt.Sprintf("avg(mlu_memory_utilization * on(uuid) group_right mlu_container{namespace=\"%s\",pod=\"%s\",container=\"%s\",type=\"mlu370.smlu.vmemory\"})", namespace, pod, container)
    case biz.AscendGPUDevice:
        query = fmt.Sprintf("avg(container_npu_used_memory{exported_namespace=\"%s\", pod_name=\"%s\", container_name=\"%s\"})", namespace, pod, container)
    case biz.HygonGPUDevice:
        query = fmt.Sprintf("avg(vdcu_usage_memory_size{pod_uuid=\"%s\", container_name=\"%s\"})", podUUID, container)
    default:
        return 0, errors.New("provider not exists")
    }
    return s.queryInstantVal(ctx, query)
}

// 容器内存使用量的单位转换处理
// 代码位置: /server/internal/exporter/exporter.go (GenerateContainerMetrics函数中)
switch provider {
case biz.CambriconGPUDevice:
    // Cambricon返回百分比，需要转换为实际字节数
    taskMemoryUsed = float32((taskMemoryUsed/100)*float32(memory)) * 1024 * 1024
case biz.AscendGPUDevice:
    // Ascend需要转换为字节
    taskMemoryUsed = float32(taskMemoryUsed) * 1024 * 1024
default:
    // NVIDIA和Hygon已经是字节单位
}
// 最终以MB为单位存储到Prometheus
HamiContainerMemoryUsed.WithLabelValues(...).Set(float64(taskMemoryUsed / 1024 / 1024))
```

#### 4.3.3 数据精度处理

**代码位置**: `/server/internal/exporter/exporter.go:36-42`

```go
// roundToTwoDecimal 将浮点数保留两位小数并进行四舍五入
func roundToTwoDecimal(value float64) float64 {
 return float64(math.Round(value*100) / 100)
}

// roundToOneDecimal 将浮点数保留一位小数并进行四舍五入
func roundToOneDecimal(value float64) float64 {
 return math.Round(value*10) / 10
}
```

### 4.4 数据来源说明

- **设备级指标**: 主要来源于各 GPU 厂商的监控接口（DCGM、cndev、npu-smi 等）
- **容器级指标**: 基于 Kubernetes 容器标签和 GPU 监控数据的关联查询
- **资源池指标**: 通过聚合设备级数据计算得出
- **系统组件指标**: 来源于 HAMi 系统组件的健康检查接口

---

## 5. 标签说明

### 5.1 通用标签

| 标签名称 | 描述 | 示例值 |
|---------|------|--------|
| `node` | Kubernetes 节点名称 | `worker-node-01` |
| `provider` | GPU 设备提供商 | `NVIDIA`, `MLU`, `Hygon`, `Ascend` |
| `devicetype` | 设备类型 | `GeForce RTX 3080`, `MLU270` |
| `deviceuuid` | 设备唯一标识符 | `GPU-12345678-1234-1234-1234-123456789012` |
| `driver_version` | 驱动程序版本 | `470.57.02` |
| `device_no` | 设备编号 | `0`, `1`, `2` |

### 5.2 容器相关标签

| 标签名称 | 描述 | 示例值 |
|---------|------|--------|
| `pod_name` | Pod 名称 | `training-job-12345` |
| `container_name` | 容器名称 | `pytorch-training` |
| `namespace_name` | 命名空间名称 | `ml-workloads` |
| `container_pod_uuid` | 容器 Pod UUID | `12345678-1234-1234-1234-123456789012` |

### 5.3 资源池标签

| 标签名称 | 描述 | 示例值 |
|---------|------|--------|
| `pool` | 资源池标识 | `default-pool`, `high-priority-pool` |

### 5.4 系统组件标签

| 标签名称 | 描述 | 示例值 |
|---------|------|--------|
| `component_type` | 组件类型 | `device-plugin`, `scheduler` |
| `component_name` | 组件名称 | `hami-device-plugin` |

---

## 6. 监控建议

### 6.1 关键告警指标

1. **设备温度过高**: `hami_device_temperature > 85`
2. **内存利用率过高**: `hami_memory_util > 90`
3. **硬件健康状态异常**: `hami_device_hardware_health == 0`
4. **系统组件不健康**: `hami_system_component_health == 0`

### 6.2 性能监控指标

1. **GPU 利用率**: `hami_core_util`
2. **内存利用率**: `hami_memory_util`
3. **容器资源使用效率**: `hami_container_memory_util`, `hami_container_core_util`

### 6.3 容量规划指标

1. **资源池总容量**: `hami_pool_vgpu_count`, `hami_pool_vmemory_size`
2. **设备分配情况**: `hami_vgpu_count`, `hami_vmemory_size`
3. **虚拟化扩展比例**: `hami_vcore_scaling`, `hami_vmemory_scaling`

---
