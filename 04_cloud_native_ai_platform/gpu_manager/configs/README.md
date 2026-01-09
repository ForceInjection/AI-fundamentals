# GPU 管理器配置文件目录

本目录包含 GPU 资源管理和监控相关的配置文件模板，支持多云环境下的 GPU 集群部署、性能优化和监控告警。

## 目录结构

```bash
configs/
├── cloud/                 # 云平台 GPU 集群配置
│   ├── aws-eks-gpu-cluster.yaml      # AWS EKS GPU 集群配置
│   ├── azure-aks-gpu-setup.sh        # Azure AKS GPU 集群部署脚本
│   └── gcp-gke-gpu-setup.sh          # GCP GKE GPU 集群部署脚本
├── cuda/                  # CUDA 相关配置
│   ├── cuda-mps-config.yaml          # CUDA MPS 多进程服务配置
│   └── cuda-streams-config.yaml      # CUDA 流配置
├── docker/                # Docker GPU 配置
│   └── docker-gpu-config.yaml         # Docker GPU 资源预留配置
├── hami/                  # HAMi GPU 管理系统
│   └── hami-deployment.yaml           # HAMi v2.6.0 部署配置
├── kubernetes/            # Kubernetes GPU 配置
│   └── gpu-pod-templates.yaml        # GPU Pod 模板集合
├── monitoring/            # 监控和告警配置
│   ├── grafana-gpu-dashboard.json    # Grafana GPU 监控仪表板
│   └── prometheus-gpu-config.yaml     # Prometheus GPU 监控配置
└── scripts/               # 工具脚本
    ├── gpu-benchmark-suite.sh         # GPU 基准测试套件
    └── multicloud-gpu-scheduler.sh    # 多云 GPU 调度脚本
```

## 配置文件详细说明

### 云平台 GPU 集群配置 (`cloud/`)

#### `aws-eks-gpu-cluster.yaml`

- **功能**: AWS EKS GPU 集群自动化部署配置
- **核心配置**:
  - 使用 `eksctl` 配置 GPU 集群
  - 节点组配置: p3.2xlarge 实例 (NVIDIA Tesla V100)
  - 自动伸缩配置: 1-5 个 GPU 节点
  - 污点和标签配置用于 GPU 节点调度

#### `azure-aks-gpu-setup.sh`

- **功能**: Azure AKS GPU 集群部署脚本
- **核心功能**:
  - 使用 `az aks` 创建 GPU 集群
  - 配置 NC 系列 GPU 虚拟机
  - 安装 NVIDIA 设备插件
  - 设置 GPU 节点标签和污点

#### `gcp-gke-gpu-setup.sh`

- **功能**: GCP GKE GPU 集群部署脚本
- **核心功能**:
  - 使用 `gcloud container` 创建 GPU 集群
  - 配置 A100/V100/T4 GPU 节点
  - 部署 NVIDIA GPU 驱动
  - 设置节点池自动伸缩

### CUDA 配置 (`cuda/`)

#### `cuda-mps-config.yaml`

- **功能**: CUDA Multi-Process Service 配置
- **核心配置**:
  - MPS 管道和日志目录配置
  - MPS 用户和服务配置
  - 资源限制和 QoS 设置
  - Kubernetes ConfigMap 格式部署脚本

#### `cuda-streams-config.yaml`

- **功能**: CUDA 流并发配置
- **核心配置**:
  - 流优先级和同步配置
  - 内存分配策略
  - 并发执行参数优化
  - 性能监控指标

### Docker GPU 配置 (`docker/`)

#### `docker-gpu-config.yaml`

- **功能**: Docker GPU 资源预留和限制配置
- **核心配置**:
  - GPU 设备访问控制
  - 内存和计算资源限制
  - 容器运行时配置
  - 多 GPU 设备映射

### HAMi GPU 管理系统 (`hami/`)

#### `hami-deployment.yaml`

- **功能**: HAMi v2.6.0 GPU 管理系统部署
- **核心配置**:
  - `hami-system` 命名空间配置
  - ServiceAccount 和 ClusterRole 权限
  - 控制器和调度器部署
  - 自定义资源定义 (CRD)

### Kubernetes GPU 配置 (`kubernetes/`)

#### `gpu-pod-templates.yaml`

- **功能**: 多种 GPU 工作负载 Pod 模板
- **包含模板**:
  - 基础 GPU 测试 Pod (`nvidia-smi`)
  - GPU 训练工作负载 Pod (PyTorch)
  - 推理服务 Pod (TensorRT)
  - 多 GPU 并行训练 Pod
  - 监控和性能分析 Pod

### 监控配置 (`monitoring/`)

#### `prometheus-gpu-config.yaml`

- **功能**: Prometheus GPU 监控配置
- **核心配置**:
  - GPU 指标抓取配置 (DCGM exporter)
  - 告警规则定义
  - 服务发现和标签重写
  - 集群范围监控设置

#### `grafana-gpu-dashboard.json`

- **功能**: Grafana GPU 监控仪表板
- **核心功能**:
  - GPU 利用率监控
  - 内存使用情况
  - 温度和功耗监控
  - 多集群和多节点支持
  - 实时性能图表

### 工具脚本 (`scripts/`)

#### `gpu-benchmark-suite.sh`

- **功能**: 全面的 GPU 性能基准测试套件
- **测试类型**:
  - 计算性能测试 (FLOPS)
  - 内存带宽测试
  - 张量核心性能
  - 多 GPU 扩展性
  - 延迟和吞吐量测试

#### `multicloud-gpu-scheduler.sh`

- **功能**: 多云 GPU 资源调度脚本
- **支持平台**:
  - AWS EKS
  - Azure AKS
  - GCP GKE
  - 本地 Kubernetes 集群
- **调度功能**:
  - GPU 类型选择 (A100/V100/T4/P100)
  - 资源配额管理
  - 成本优化调度
  - 跨云集群联邦

## 使用指南

### 快速开始

1. **部署云 GPU 集群**:

   ```bash
   # AWS EKS
   eksctl create cluster -f configs/cloud/aws-eks-gpu-cluster.yaml

   # Azure AKS
   bash configs/cloud/azure-aks-gpu-setup.sh

   # GCP GKE
   bash configs/cloud/gcp-gke-gpu-setup.sh
   ```

2. **部署监控系统**:

   ```bash
   # 部署 Prometheus
   kubectl apply -f configs/monitoring/prometheus-gpu-config.yaml

   # 部署 Grafana 仪表板
   kubectl apply -f configs/monitoring/grafana-gpu-dashboard.json
   ```

3. **运行 GPU 基准测试**:

   ```bash
   bash configs/scripts/gpu-benchmark-suite.sh all ./results 5
   ```

4. **调度多云 GPU 任务**:

   ```bash
   bash configs/scripts/multicloud-gpu-scheduler.sh aws a100 2 training
   ```

### 配置自定义

所有配置文件都支持环境变量和参数化配置，主要配置方式：

1. **环境变量替换**: 使用 `${VARIABLE_NAME}` 语法
2. **命令行参数**: 脚本文件支持命令行参数
3. **配置文件覆盖**: 使用 `kubectl patch` 或 `kubectl edit`
4. **模板化部署**: 使用 Helm 或 Kustomize 进行配置管理

### 最佳实践

1. **资源限制**: 所有 GPU Pod 都应设置适当的资源限制
2. **监控告警**: 部署完整的监控栈并及时响应告警
3. **备份恢复**: 定期备份重要配置和集群状态
4. **安全加固**: 遵循最小权限原则，定期更新安全补丁
5. **成本优化**: 使用自动伸缩和 spot 实例降低成本

## 故障排除

### 常见问题

1. **GPU 设备未识别**:

   - 检查 NVIDIA 驱动安装
   - 验证设备插件部署
   - 检查节点污点配置

2. **性能问题**:

   - 运行基准测试确认硬件性能
   - 检查资源限制和配额
   - 监控温度和功耗

3. **调度失败**:
   - 检查资源可用性
   - 验证节点选择器和亲和性
   - 检查命名空间配额

### 日志和调试

```bash
# 查看 GPU 节点状态
kubectl describe nodes | grep -A 10 -B 10 nvidia

# 查看设备插件日志
kubectl logs -n kube-system -l name=nvidia-device-plugin

# 查看 HAMi 调度器日志
kubectl logs -n hami-system -l app=hami-scheduler
```

## 版本兼容性

| 组件          | 版本要求 | 测试版本 |
| ------------- | -------- | -------- |
| Kubernetes    | ≥ 1.20   | 1.28     |
| Docker        | ≥ 20.10  | 24.0     |
| NVIDIA Driver | ≥ 525.60 | 535.54   |
| CUDA          | ≥ 11.8   | 12.2     |
| HAMi          | ≥ 2.5.0  | 2.6.0    |
