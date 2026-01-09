# NCCL Kubernetes 部署指南

## 概述

本指南介绍如何在 Kubernetes 集群中部署和运行 NCCL 多节点测试。相比传统的 Docker Compose 方案，Kubernetes 提供了更好的资源管理、调度和扩展能力。

## 架构优势

### 1. Kubernetes vs Docker Compose

| 特性         | Docker Compose | Kubernetes           |
| ------------ | -------------- | -------------------- |
| **节点调度** | 手动指定       | 自动调度到最优节点   |
| **资源管理** | 基础限制       | 精确的资源请求和限制 |
| **故障恢复** | 手动重启       | 自动重启和故障转移   |
| **扩展性**   | 静态配置       | 动态扩缩容           |
| **网络**     | Host/Bridge    | 丰富的网络策略       |
| **存储**     | 本地挂载       | 持久化存储抽象       |
| **监控**     | 基础日志       | 完整的可观测性       |

### 2. NCCL 在 Kubernetes 中的优势

- **智能调度**: 自动选择具有 GPU 和 InfiniBand 的节点
- **资源隔离**: 精确的 GPU、内存、CPU 资源分配
- **网络优化**: 支持 Host Network 和 InfiniBand 直通
- **故障处理**: 自动重试和故障转移机制
- **日志聚合**: 统一的日志收集和查看
- **配置管理**: ConfigMap 和 Secret 管理配置

## 部署架构

```text
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Node 1    │    │   Node 2    │    │   Node 3    │      │
│  │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │      │
│  │ │NCCL Pod │ │    │ │NCCL Pod │ │    │ │NCCL Pod │ │      │
│  │ │Rank: 0  │ │    │ │Rank: 1  │ │    │ │Rank: 2  │ │      │
│  │ │GPU: 2   │ │    │ │GPU: 2   │ │    │ │GPU: 2   │ │      │
│  │ └─────────┘ │    │ └─────────┘ │    │ └─────────┘ │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                 NCCL Master Service                         │
│              (Headless Service for P2P)                     │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 前置条件

确保你的 Kubernetes 集群满足以下要求：

```bash
# 检查集群状态
kubectl cluster-info

# 检查 GPU 节点
kubectl get nodes -l nvidia.com/gpu.present=true

# 注意：默认配置使用 hami-scheduler 调度器。
# 如果您的集群未使用该调度器，请修改 nccl-multinode-sts.yaml 中的 schedulerName 字段。
```

### 2. 部署测试

```bash
# 进入 NCCL 目录
cd /AI-fundermentals/nccl

# 部署默认配置
./k8s/deploy.sh deploy

# 或者自定义配置
./k8s/deploy.sh deploy --world-size 8 --gpus-per-node 4
```

### 3. 监控测试

```bash
# 查看测试状态
./k8s/deploy.sh status

# 查看实时日志
./k8s/deploy.sh logs

# 或者使用 kubectl 直接查看
kubectl get jobs,pods -l app=nccl-test
kubectl logs -l app=nccl-test --tail=100 -f
```

### 4. 清理资源

```bash
# 清理测试资源
./k8s/deploy.sh cleanup
```

## 配置说明

### 1. StatefulSet 配置 (nccl-multinode-sts.yaml)

```yaml
spec:
  serviceName: nccl-multinode
  replicas: 2 # 并行运行的 Pod 数量 (由 deploy.sh 动态替换)
  podManagementPolicy: Parallel
  template:
    spec:
      schedulerName: "hami-scheduler" # 使用 hami 调度器 (可选)
      restartPolicy: Always
```

### 2. 资源配置

```yaml
resources:
  requests:
    nvidia.com/gpu: 2 # 每个 Pod 请求的 GPU 数量 (可配置)
  limits:
    nvidia.com/gpu: 2
    memory: "16Gi" # 内存限制
    cpu: "8" # CPU 限制
```

### 3. 网络配置

```yaml
hostNetwork: true # 使用主机网络，支持 InfiniBand
hostIPC: true # 共享主机 IPC
securityContext:
  privileged: true # 特权模式，访问硬件设备
```

### 4. NCCL 环境变量

```yaml
env:
  - name: NCCL_DEBUG
    value: "INFO" # 调试级别
  - name: NCCL_SOCKET_IFNAME
    value: "eno*,eth*,ib*,enp*" # 指定网络接口
  - name: NCCL_IB_DISABLE
    value: "0" # 启用 InfiniBand
  - name: NCCL_NET_GDR_LEVEL
    value: "2" # 启用 GPUDirect RDMA
  - name: NCCL_P2P_DISABLE
    value: "0" # 启用 P2P 通信
  - name: NCCL_IGNORE_CPU_AFFINITY
    value: "1" # 忽略 CPU 亲和性限制
```

## 高级配置

### 1. 节点亲和性

如果需要将 Pod 调度到特定节点：

```yaml
nodeSelector:
  nvidia.com/gpu.present: "true"
  kubernetes.io/hostname: "gpu-node-1"

# 或使用节点亲和性
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: nvidia.com/gpu.count
              operator: Gt
              values: ["1"]
```

### 2. Pod 反亲和性

确保 Pod 分布在不同节点：

```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
            - key: app
              operator: In
              values: ["nccl-test"]
        topologyKey: kubernetes.io/hostname
```

### 3. 持久化存储

如果需要保存测试结果：

```yaml
volumes:
  - name: nccl-results
    persistentVolumeClaim:
      claimName: nccl-pvc

volumeMounts:
  - name: nccl-results
    mountPath: /workspace/results
```

## 故障排查

### 1. 常见问题

**Pod 无法调度**：

```bash
# 检查节点资源
kubectl describe nodes

# 检查 Pod 事件
kubectl describe pod <pod-name>
```

**GPU 不可用**：

```bash
# 检查 NVIDIA Device Plugin
kubectl get pods -n kube-system | grep nvidia

# 检查节点 GPU 标签
kubectl get nodes -o yaml | grep nvidia.com/gpu
```

**网络连接问题**：

```bash
# 检查 Service
kubectl get svc nccl-master-service

# 检查 DNS 解析
kubectl exec <pod-name> -- nslookup nccl-master-service
```

### 2. 调试命令

```bash
# 进入 Pod 调试
kubectl exec -it <pod-name> -- /bin/bash

# 查看详细日志
kubectl logs <pod-name> --previous

# 查看 Pod 详细信息
kubectl describe pod <pod-name>
```

## 性能优化

### 1. 资源调优

```yaml
# CPU 绑定
resources:
  requests:
    cpu: "8"
  limits:
    cpu: "8"

# 内存优化
env:
  - name: NCCL_BUFFSIZE
    value: "8388608" # 8MB buffer
```

### 2. 网络优化

```yaml
# InfiniBand 优化
env:
  - name: NCCL_IB_TIMEOUT
    value: "23"
  - name: NCCL_IB_RETRY_CNT
    value: "7"
  - name: NCCL_IB_GID_INDEX
    value: "3"
```

### 3. GPU 优化

```yaml
# GPU 直通优化
env:
  - name: NCCL_P2P_DISABLE
    value: "0" # 启用 P2P
  - name: NCCL_SHM_DISABLE
    value: "0" # 启用共享内存
```

## 监控和可观测性

### 1. 日志聚合

使用 Fluentd 或 Fluent Bit 收集日志：

```yaml
# 添加日志标签
metadata:
  labels:
    app: nccl-test
    component: worker
```

### 2. 指标监控

集成 Prometheus 监控：

```yaml
# 添加监控注解
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
```

### 3. 分布式追踪

使用 Jaeger 进行分布式追踪：

```yaml
env:
  - name: JAEGER_AGENT_HOST
    value: "jaeger-agent"
```

## 最佳实践

1. **资源规划**: 根据实际需求合理分配 GPU、CPU 和内存
2. **网络优化**: 优先使用 InfiniBand，配置合适的网络参数
3. **故障处理**: 设置合理的重试策略和超时时间
4. **监控告警**: 建立完善的监控和告警机制
5. **安全考虑**: 使用最小权限原则，避免不必要的特权模式
6. **版本管理**: 使用固定的镜像标签，避免使用 latest
7. **配置管理**: 使用 ConfigMap 和 Secret 管理配置
8. **测试验证**: 在生产环境部署前充分测试

通过 Kubernetes 部署 NCCL 测试，你可以获得更好的资源管理、故障恢复和扩展能力，这对于大规模的分布式训练环境尤其重要。
