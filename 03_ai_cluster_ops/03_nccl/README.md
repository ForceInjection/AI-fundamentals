# NCCL 分布式通信测试套件

## 1. 概述

本目录提供了完整的 NCCL (NVIDIA Collective Communications Library) 分布式通信测试解决方案，支持单节点和多节点的 GPU 通信性能测试，包含容器化部署和原生环境部署两种方式。

### 1.1 PXN 模式支持

本测试套件现已全面支持 **PXN (高性能多节点通信) 模式**，提供针对多节点环境优化的高性能通信解决方案：

- **专为多节点优化**：针对大规模分布式训练场景设计
- **三种优化级别**：保守、平衡、激进模式，满足不同性能需求
- **智能网络检测**：自动选择最佳网络配置和通信路径
- **详细性能分析**：提供 PXN 模式专用的性能指标和优化建议
- **全栈支持**：容器化、多节点原生、Kubernetes 部署全面支持

## 2. 文件结构

```text
nccl/
├── 🔧 核心测试工具
│   ├── nccl_benchmark.sh          # 主要的 NCCL 性能基准测试脚本 (支持 PXN 模式)
│   └── nccl_python_template.py    # Python 测试模板脚本
├── 🐳 容器化部署
│   ├── Dockerfile                 # NCCL 测试容器镜像定义
│   └── nccl_container_manager.sh  # 容器化测试管理脚本 (支持 PXN 多节点)
├── 🌐 多节点部署
│   ├── nccl_multinode_launcher.sh # 原生多节点测试启动器 (支持 PXN 模式)
│   └── k8s/                       # Kubernetes 多节点部署方案
│       ├── deploy.sh              # Kubernetes 部署管理脚本 (支持 PXN 模式)
│       ├── nccl-multinode-sts.yaml # NCCL 多节点 StatefulSet 配置
│       ├── nccl-service.yaml      # NCCL 服务配置
│       ├── nccl-configmap.yaml    # NCCL 配置映射
│       ├── nccl-rbac.yaml         # RBAC 权限配置
│       └── README.md              # Kubernetes 部署指南
├── 🔍 诊断工具
│   └── gpu_topology_detector.sh   # GPU 拓扑检测工具
├── 📚 配置文件
│   ├── requirements.txt           # Python 依赖包配置
│   └── tutorial.md               # 详细使用教程和最佳实践
└── 📁 测试数据
    └── test/                      # 测试脚本和数据 (包含 PXN 测试套件)
        ├── test_pxn_mode.sh       # PXN 模式功能测试
        └── run_all_tests.sh       # 全套测试运行器 (包含 PXN 测试)
```

## 3. 核心组件介绍

### 3.1 核心测试工具

#### 3.1.1 `nccl_benchmark.sh`

主要的 NCCL 性能基准测试脚本，提供以下功能：

- **多网络后端支持**：NVLink、InfiniBand、Ethernet、Socket、**PXN**
- **PXN 模式优化**：专为多节点高性能通信设计的优化模式
- **三种优化级别**：
  - `conservative`: 保守模式，稳定性优先
  - `balanced`: 平衡模式，性能与稳定性兼顾 (默认)
  - `aggressive`: 激进模式，最大性能优化
- **自动路径检测**：智能选择最佳通信路径，支持 PXN 自动检测
- **性能分析**：延迟、带宽、吞吐量等详细指标，包含 PXN 专用指标
- **环境验证**：自动检查依赖和配置，包含 PXN 环境验证

#### 3.1.2 `nccl_python_template.py`

基于 PyTorch 的分布式测试模板：

- **灵活配置**：支持自定义张量大小和测试时长
- **详细输出**：提供完整的性能指标和统计信息
- **容器友好**：可在容器环境中直接运行

### 3.2 容器化部署

#### 3.2.1 `Dockerfile`

优化的 NCCL 测试容器镜像：

- **基础镜像**：NVIDIA CUDA 官方镜像
- **预装依赖**：PyTorch、NCCL、InfiniBand 工具
- **网络优化**：GPUDirect RDMA 和网络配置

#### 3.2.2 `nccl_container_manager.sh`

容器化测试管理脚本，现已支持多节点 PXN 模式：

- **自动构建**：一键构建测试镜像
- **灵活配置**：支持多种 GPU 和网络配置
- **PXN 多节点支持**：支持容器化的多节点 PXN 测试
- **优化级别配置**：支持 PXN 三种优化级别
- **主从节点配置**：支持 `--master-addr`、`--master-port` 参数
- **交互模式**：提供调试和开发环境
- **单节点兼容**：保持原有单节点测试功能

### 3.3 多节点部署

#### 3.3.1 `nccl_multinode_launcher.sh`

原生多节点测试启动器，现已全面支持 PXN 模式：

- **简化部署**：一键启动多节点测试
- **PXN 模式支持**：完整的 PXN 高性能通信支持
- **智能网络检测**：默认使用 `auto` 模式，自动选择最佳网络配置
- **优化级别配置**：支持 `--optimization` 参数配置优化级别
- **环境检查**：自动验证集群环境，包含 PXN 环境验证
- **配置管理**：统一的节点配置，支持 PXN 参数传递
- **原生环境**：适用于传统裸机部署

#### 3.3.2 Kubernetes 多节点部署方案 (`k8s/`)

现代化的容器编排多节点部署方案：

**`deploy.sh`** - Kubernetes 部署管理脚本，现已支持 PXN 模式：

- **一键部署**：自动化 Kubernetes 资源创建
- **PXN 模式支持**：支持 `--network pxn` 参数启用 PXN 模式
- **优化级别配置**：支持 `--optimization` 参数配置优化级别
- **参数配置**：支持自定义 GPU 数量、测试参数
- **状态监控**：实时查看部署状态和测试进度
- **资源清理**：完整的资源生命周期管理

**`nccl-multinode-sts.yaml`** - NCCL 多节点 StatefulSet 配置：

- **并行执行**：支持多节点并行测试
- **资源管理**：GPU 资源请求和限制
- **网络配置**：Host Network 和 IPC 配置
- **环境变量**：完整的 NCCL 参数配置

**`nccl-service.yaml`** - NCCL 服务配置：

- **服务发现**：节点间通信服务
- **端口管理**：Master 和 Worker 端口配置
- **负载均衡**：集群内部通信优化

**`nccl-configmap.yaml`** - NCCL 配置映射：

- **参数管理**：集中化的 NCCL 环境变量配置
- **测试配置**：数据大小、测试时长等参数
- **网络优化**：InfiniBand 和网络后端配置

**`README.md`** - Kubernetes 部署指南：

- **完整教程**：从安装到部署的详细步骤
- **配置说明**：各种参数和选项的详细解释
- **故障排除**：常见问题和解决方案
- **最佳实践**：生产环境部署建议

### 3.4 诊断工具

#### 3.4.1 `gpu_topology_detector.sh`

GPU 拓扑检测工具：

- **硬件检测**：GPU 连接类型分析 (NVLink, PCIe P2P)
- **路径分析**：NCCL 通信路径优先级
- **性能建议**：网络配置优化建议

### 3.5 配置文件

#### 3.5.1 `requirements.txt`

Python 依赖包配置文件，包含：

- PyTorch 及相关组件
- NCCL 测试所需的 Python 库
- 容器化部署依赖

#### 3.5.2 `tutorial.md`

详细使用教程，包含：

- 完整的安装和配置说明
- 单节点和多节点测试详细步骤
- 性能调优和故障排除
- 最佳实践和参考资料

## 4. 前置条件

### 4.1 硬件要求

- **GPU**: 一个或多个 NVIDIA GPU (支持 CUDA)
- **网络**: InfiniBand 或高速以太网 (可选，用于多节点测试)
- **内存**: 建议 16 GB 以上系统内存

### 4.2 软件要求

- **操作系统**: Linux (Ubuntu 18.04+ / CentOS 7+ / RHEL 7+)
- **Docker**: 用于容器化部署 (推荐)
- **NVIDIA Container Toolkit**: GPU 容器支持
- **Python 3.7+**: 原生环境测试 (可选)
- **Kubernetes**: 用于多节点容器编排 (可选，推荐生产环境)
  - kubectl 1.20+
  - GPU Operator 或 NVIDIA Device Plugin

### 4.3 快速安装 (容器化部署)

```bash
# 1. 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 2. 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 3. 验证 GPU 容器支持
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi
```

## 5. 快速开始

### 5.1 构建测试环境

```bash
# 进入项目目录
cd /path/to/nccl/

# 构建容器镜像
./nccl_container_manager.sh --build
```

### 5.2 运行测试

#### 5.2.1 单节点测试 (推荐新手)

```bash
# 基础性能测试
./nccl_container_manager.sh --gpus all --size 100M --time 60

# PXN 模式单节点测试
./nccl_container_manager.sh --gpus all --size 100M --time 60 --network pxn --optimization balanced

# GPU 拓扑检测
./gpu_topology_detector.sh
```

#### 5.2.2 多节点测试

> 要准备 python 和 nccl 依赖的相关环境。

**方案一：原生多节点部署:**

```bash
# 标准多节点测试
# 在 Master 节点 (192.168.1.100) 上执行
./nccl_multinode_launcher.sh 0 192.168.1.100 --world-size 4 --nproc-per-node 2

# 在 Worker 节点 (192.168.1.101) 上执行  
./nccl_multinode_launcher.sh 1 192.168.1.100 --world-size 4 --nproc-per-node 2

# PXN 模式多节点测试（高性能）
# 在 Master 节点上执行
./nccl_multinode_launcher.sh 0 192.168.1.100 --world-size 4 --nproc-per-node 2 --network pxn --optimization aggressive

# 在 Worker 节点上执行
./nccl_multinode_launcher.sh 1 192.168.1.100 --world-size 4 --nproc-per-node 2 --network pxn --optimization aggressive
```

**方案二：Kubernetes 多节点部署（推荐）:**

```bash
# 进入 Kubernetes 配置目录
cd k8s/

# 快速部署（使用默认配置）
./deploy.sh deploy

# 自定义部署
./deploy.sh deploy --gpus 4 --test-size 1G --test-duration 120

# PXN 模式部署（高性能多节点）
./deploy.sh deploy --gpus 4 --test-size 1G --test-duration 120 --network pxn --optimization balanced

# PXN 激进模式部署（最大性能）
./deploy.sh deploy --gpus 8 --test-size 2G --test-duration 180 --network pxn --optimization aggressive

# 查看部署状态
./deploy.sh status

# 查看测试日志
./deploy.sh logs

# 清理资源
./deploy.sh cleanup
```

### 5.3 查看结果

测试完成后，查看生成的报告文件：

- 性能测试报告
- GPU 拓扑分析
- 网络配置建议

## 6. 使用场景

### 6.1 性能基准测试

- 建立 GPU 通信性能基线
- 验证硬件配置效果
- 对比不同网络后端性能
- **PXN 模式性能评估**：测试高性能多节点通信优化效果
- **优化级别对比**：评估 conservative、balanced、aggressive 三种优化级别的性能差异

### 6.2 环境验证

- 验证 NCCL 环境配置
- 检查 GPU 拓扑结构
- 诊断网络连接问题
- **PXN 环境检测**：验证 PXN 模式的网络环境和配置
- **智能网络检测**：自动检测最适合的网络后端（InfiniBand、以太网等）

### 6.3 生产部署准备

- 多节点集群性能验证
- 容器化部署测试
- 分布式训练环境准备
- Kubernetes 集群 NCCL 性能验证
- 云原生环境适配测试
- **PXN 模式生产部署**：大规模多节点集群的 PXN 模式部署和优化
- **多节点 PXN 调优**：针对不同硬件配置的 PXN 优化级别选择

### 6.4 云原生部署

- Kubernetes 集群 GPU 通信测试
- 容器编排环境性能基准
- 多租户环境隔离验证
- 自动化部署和监控
- **云原生 PXN 部署**：Kubernetes 环境下的 PXN 模式自动化部署
- **PXN 容器编排**：支持 PXN 模式的容器化多节点通信优化

## 7. 详细文档

- **[详细使用教程](./01_nccl_tutorial.md)** - 完整的安装、配置和使用说明
- **[Kubernetes 部署指南](./k8s/README.md)** - Kubernetes 多节点部署详细教程
- **[故障排除](./01_nccl_tutorial.md#故障排除)** - 常见问题和解决方案
- **[性能调优](./01_nccl_tutorial.md#性能调优)** - 最佳实践和优化建议

## 8. 参考资料

- [NVIDIA Container Toolkit 文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/)
- [PyTorch 分布式训练](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Kubernetes GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [Kubernetes 官方文档](https://kubernetes.io/docs/)
