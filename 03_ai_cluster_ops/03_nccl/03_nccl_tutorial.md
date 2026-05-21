# NCCL 测试验证工具说明文档

> 注意：**性能部分指标的解释和计算方法可能会根据具体的硬件配置和网络环境而有所不同。建议在实际测试中根据自己的环境进行调整和优化**。

---

## 目录

- [1. 概述与系统要求](#1-概述与系统要求)
- [2. 单节点测试](#2-单节点测试)
- [3. 容器化测试](#3-容器化测试)
- [4. 多节点测试](#4-多节点测试)
- [5. 网络配置详解](#5-网络配置详解)
- [6. 故障排除与诊断](#6-故障排除与诊断)
- [7. 附录](#7-附录)

---

## 1. 概述与系统要求

### 1.1 NCCL 测试背景

#### 1.1.1 为什么需要 NCCL 测试

在现代深度学习训练中，多 GPU 和分布式训练已成为处理大规模模型的标准方法。NCCL (NVIDIA Collective Communications Library) 作为 NVIDIA 提供的高性能集合通信库，负责 GPU 间的数据同步和通信。然而，NCCL 的性能高度依赖于：

- **硬件配置**：GPU 型号、内存带宽、PCIe 拓扑结构
- **网络环境**：InfiniBand、RoCE、以太网的配置和性能
- **软件栈**：CUDA 版本、驱动程序、NCCL 库版本的兼容性
- **环境变量**：数十个 NCCL 参数的正确配置（现已通过统一配置管理器自动化）

不当的配置可能导致：

- 训练吞吐腰斩：NVLink 故障时 AllReduce 带宽从 225 GB/s 降至 ~28 GB/s（[NCCL 基准测试](04_nccl_benchmark.md)），故障排查见 [NVLink 诊断](../../01_hardware_architecture/nvlink/nvlink_diagnostics.md)
- 通信延迟增加数倍：P2P 走 NVLink（239 GB/s）vs PCIe SYS（不支持 P2P，回退 CPU 中转），参见 [P2P 带宽实测](../../02_gpu_programming/04_profiling/08_p2p_bandwidth.md)
- 网络带宽利用率远低于硬件峰值
- 分布式训练失败或不稳定

#### 1.1.2 NCCL 核心概念

**AllReduce 操作**：NCCL 最重要的集合通信原语，用于梯度聚合

- 将所有 GPU 上的数据进行归约运算（如求和）
- 将结果广播到所有参与的 GPU
- 是分布式训练中梯度同步的核心操作

**通信算法**：

- **Ring AllReduce**：适用于带宽受限环境，通信量为 `2(N-1)/N × data_size`
- **Tree AllReduce**：适用于延迟敏感场景，通信深度为 `log₂(N)`
- **Double Binary Tree**：NCCL 2.4+ 的默认算法，平衡延迟和带宽

**网络后端**：

- **NVLink**：GPU 间直连，V100 300 / A100 600 / H100 900 GB/s 双向总带宽，A100 P2P 单向实测 **239 GB/s**（[基准测试](04_nccl_benchmark.md) | [NVLink 入门](../../01_hardware_architecture/nvlink/nvlink_intro.md)）
- **InfiniBand**：高性能网络，带宽 12.5-50 GB/s (100-400 Gbps)（[IB 理论基础](../02_infiniband/01_ib_network_theory.md)）
- **RoCE**：基于以太网的 RDMA，带宽 3.1-12.5 GB/s (25-100 Gbps)（[GPUDirect RDMA](../../01_hardware_architecture/gpudirect/01_gpudirect_technology.md)）
- **TCP/Socket**：通用网络，带宽 0.125-1.25 GB/s (1-10 Gbps)

### 1.2 工具概述

本工具套件提供了完整的 NCCL 测试和部署解决方案，包含以下核心工具：

#### 1.2.1 核心测试工具

**`nccl_benchmark.sh`** - 主要的 NCCL 性能测试工具：

- **性能基准测试**：测量真实的 NCCL 通信性能
- **智能配置管理**：统一配置管理器自动化 NCCL 环境变量设置
- **多级优化策略**：提供保守、平衡、激进三种优化级别（适用于 NVLink 和 PXN 网络后端）
- **自动路径检测**：按 NCCL 优先级自动选择最佳通信路径
- **问题诊断**：识别性能瓶颈和配置问题

**`gpu_topology_detector.sh`** - GPU 拓扑检测工具：

- **硬件拓扑分析**：检测 GPU 间的连接方式（NVLink、PCIe）
- **NCCL 通信路径验证**：确认 NCCL 实际使用的通信路径
- **性能预测**：基于硬件拓扑预测通信性能

#### 1.2.2 部署工具

**`nccl_container_manager.sh`** - 容器化测试管理工具
**`nccl_multinode_launcher.sh`** - 传统多节点部署工具
**`k8s/deploy.sh`** - Kubernetes 多节点部署工具
**`nccl_python_template.py`** - Python 测试模板

### 1.3 主要功能

- **系统检查**：验证 GPU 驱动、CUDA 版本、NCCL 库和 InfiniBand 工具是否就绪。相当于一键运行 `nvidia-smi` + `nvcc --version` + `dpkg -l | grep nccl` + `ibstat`。
- **统一配置管理**：根据检测到的硬件（NVLink 数量、IB 设备、PCIe 拓扑）自动设置最优 NCCL 环境变量，消除手动配置的错误。
- **性能测试**：运行 `allreduce_perf` 等标准基准，覆盖 1 MB 到 8 GB 数据范围，输出 bus_bw 并自动与理论峰值对比。
- **报告生成**：输出结构化测试报告，包含硬件拓扑摘要、每种网络后端的带宽曲线、异常标记和优化建议。

### 1.4 系统要求

#### 1.4.1 硬件要求

- **GPU**: 一个或多个 NVIDIA GPU (支持 CUDA Compute Capability 3.5+)
  - 推荐：V100、A100、H100 等数据中心 GPU
  - 最低：GTX 1080、RTX 2080 等消费级 GPU
- **网络**: InfiniBand 网卡 (原生 IB 或 RoCE)
  - InfiniBand：EDR (12.5 GB/s)、HDR (25 GB/s)、NDR (50 GB/s)
  - RoCE：3.1/6.25/12.5 GB/s (25/50/100 Gbps) 以太网卡
- **内存**: 建议 16 GB 以上系统内存
- **存储**: 至少 10 GB 可用空间用于日志和临时文件

#### 1.4.2 软件要求

- **操作系统**: Linux (Ubuntu 18.04+ / CentOS 7+ / RHEL 7+)
- **Python**: Python 3.7+ (推荐 3.8-3.11)
- **PyTorch**: 1.12.0+ 支持 CUDA 的版本
- **NCCL**: 2.12.0+ (推荐 2.18.0+)
- **CUDA**: 11.7+ (推荐 12.0+)
- **NVIDIA Driver**: 515.0+ (推荐 535.0+)
- **InfiniBand 工具**: `infiniband-diags`, `libibverbs-dev`, `rdma-core`

#### 1.4.3 依赖安装

**Ubuntu/Debian：**

```bash
# 更新包管理器
sudo apt-get update

# 安装 InfiniBand 工具和开发库
sudo apt-get install -y infiniband-diags ibverbs-utils libibverbs-dev rdma-core

# 验证 InfiniBand 安装
ibstat && ibv_devinfo

# 安装 Python 和 PyTorch (CUDA 11.8 示例)
pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 验证 PyTorch 和 NCCL
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'NCCL: {torch.cuda.nccl.version()}')"
```

**CentOS/RHEL：**

```bash
# 安装 InfiniBand 工具
sudo yum install -y infiniband-diags libibverbs-utils libibverbs-devel rdma-core-devel

# 启用 InfiniBand 服务
sudo systemctl enable rdma && sudo systemctl start rdma
```

> **阅读路径**：§2-4 覆盖三种部署模式的 NCCL 测试方法；§5 覆盖网络配置与性能分析；§6 是故障排查；§7 是速查附录。如果只需快速验证环境，直接进入 §2。

---

## 2. 单节点测试

### 2.1 单节点测试概述

单节点测试可在多节点部署前快速暴露 NVLink 故障、P2P 不可用和环境配置问题——这些问题在多节点场景下排查成本远高于单节点。

### 2.2 快速开始

#### 2.2.1 基础测试

```bash
# 自动检测最佳网络后端
./nccl_benchmark.sh

# 指定网络后端
./nccl_benchmark.sh --network nvlink    # NVLink (推荐用于单节点)
./nccl_benchmark.sh --network pcie      # PCIe P2P
./nccl_benchmark.sh --network ib        # InfiniBand
./nccl_benchmark.sh --network ethernet  # 以太网
./nccl_benchmark.sh --network socket    # Socket (调试用)
./nccl_benchmark.sh --network shm       # 共享内存 (调试用)

# 自定义测试参数
./nccl_benchmark.sh --size 100M --time 60 --network nvlink
```

#### 2.2.2 优化级别配置

**注意：优化级别适用于 NVLink 和 PXN 网络后端：**

```bash
# 保守级别 (默认，稳定性优先)
./nccl_benchmark.sh --network nvlink --optimization-level conservative

# 平衡级别 (性能与稳定性平衡)
./nccl_benchmark.sh --network nvlink --optimization-level balanced

# 激进级别 (最大性能，可能影响稳定性)
./nccl_benchmark.sh --network nvlink --optimization-level aggressive
```

**三种优化级别的参数差异**（基于 `nccl_benchmark.sh` 实际 NVLink 模式配置）：

| 参数                       | conservative                | balanced                      | aggressive                   |
| -------------------------- | --------------------------- | ----------------------------- | ---------------------------- |
| `NCCL_NTHREADS`            | 256                         | 384                           | 512                          |
| `NCCL_BUFFSIZE`            | 8 MB                        | 12 MB                         | 16 MB                        |
| `NCCL_MIN_NCHANNELS`       | 16                          | 16                            | 16                           |
| `NCCL_MAX_NCHANNELS`       | 32                          | 32                            | 32                           |
| `NCCL_ALGO` / `NCCL_PROTO` | 固定 `Ring,Tree` / `Simple` | **移除限制**（NCCL 自动选择） | **移除所有限制**（完全自动） |
| 核心策略                   | 稳定性优先，参数保守        | 性能与稳定性平衡              | 最大性能，依赖 NCCL 自动调优 |

> **注意**：NVLink 模式三级仅 `NCCL_NTHREADS`、`NCCL_BUFFSIZE` 和算法限制有所区别，通道数（MIN/MAX_NCHANNELS）在所有级别保持 16/32 不变。PXN 模式的通道数配置不同（conservative: 4/12, balanced: 6/16, aggressive: 8/20），详见 [§5.2.8 PXN 模式](#528-pxn-模式---network-pxn)。
>
> 选择建议：日常验证用 `conservative`；性能基准测试用 `balanced`；极致性能调优用 `aggressive`。如果在 `aggressive` 下出现 NCCL 初始化失败，回退到 `balanced`。

### 2.3 测试配置选项

#### 2.3.1 基本参数

| 参数                   | 默认值       | 说明                    | 示例                            |
| ---------------------- | ------------ | ----------------------- | ------------------------------- |
| `--size`               | 1M           | 测试数据大小            | `--size 100M`                   |
| `--time`               | 30           | 测试持续时间（秒）      | `--time 120`                    |
| `--network`            | auto         | 网络后端                | `--network nvlink`              |
| `--optimization-level` | conservative | 优化级别 (NVLink / PXN) | `--optimization-level balanced` |

#### 2.3.2 高级参数

```bash
# 环境变量展示
./nccl_benchmark.sh --env-only

# Dry-run 模式 (仅验证配置)
./nccl_benchmark.sh --dry-run

# 详细输出
./nccl_benchmark.sh --verbose
```

### 2.4 统一配置管理器

脚本使用统一配置管理器自动化 NCCL 环境变量设置，无需手动配置。

```bash
# 1. 自动配置所有 NCCL 环境变量
./nccl_benchmark.sh --network nvlink

# 2. 查看自动配置的环境变量
./nccl_benchmark.sh --env-only
```

**配置管理器功能**：

- 消除重复代码，统一管理所有 NCCL 配置项
- 智能缓存系统，避免重复检测
- 批量配置设置和管理
- 实时展示环境变量状态

---

> 延伸阅读：[NCCL 技术理论深度解析](01_nccl_theory.md)，涵盖 AllReduce 算法原理、RDMA 通信机制、性能建模、内存管理、容错与监控等主题。

### 2.5 Python 测试模板

#### 2.5.1 概述

`nccl_python_template.py` 提供可定制的 NCCL 测试代码模板，支持：

- 自定义测试脚本
- 性能监控和统计
- 多进程协调
- 详细的日志输出

#### 2.5.2 基本使用

```bash
# 通过 nccl_benchmark.sh 调用（推荐）
./nccl_benchmark.sh --network nvlink

# 直接调用（需要手动设置环境变量）
python3 nccl_python_template.py
```

#### 2.5.3 环境变量配置

**推荐方式：使用 nccl_benchmark.sh 统一配置管理器：**

```bash
# 自动配置所有 NCCL 环境变量（推荐）
./nccl_benchmark.sh --network nvlink

# 查看配置后的环境变量
./nccl_benchmark.sh --env-only
```

**统一配置管理器的优势**：

- 自动检测和配置
- 消除配置错误
- 批量设置相关环境变量
- 实时验证配置有效性

---

## 3. 容器化测试

### 3.1 容器化测试概述

容器化测试确保多节点间 NCCL 库版本、CUDA 版本和环境变量完全一致，消除"在我机器上能跑"的问题。推荐在 K8s 集群中使用。

**容器环境特有的注意事项**：

- **IPC_LOCK**：NCCL 使用共享内存和 P2P 需要 `--ipc=host` 或 `--shm-size=16g`。**这是最容易踩的坑**：Docker 默认 `/dev/shm` 仅 64 MB，不设置 `--shm-size` 或 `--ipc=host` 时，NCCL 初始化会因共享内存不足而**静默失败或崩溃**，错误信息不明显。推荐直接使用 `--ipc=host` 一次性解决
- **GPU 可见性**：容器内 `nvidia-smi` 默认能看到所有 GPU，通过 `--gpus` 参数或 `NVIDIA_VISIBLE_DEVICES` 限制
- **网络模式**：多节点容器必须使用 `--network=host`（Infiniband 要求）或正确配置 CNI 插件，否则 NCCL 无法发现对端
- **NCCL 调试**：容器内排查 NCCL 问题时，`NCCL_DEBUG=INFO` 输出的拓扑检测信息是判断通信路径是否正确的第一手线索

### 3.2 快速开始

#### 3.2.1 基础容器测试

```bash
# 基础测试 (使用默认配置)
./nccl_container_manager.sh

# 指定 GPU 数量和测试参数
./nccl_container_manager.sh --gpus 2 --size 100M --time 60

# 指定网络后端
./nccl_container_manager.sh --network nvlink --gpus 4
```

#### 3.2.2 高级配置

```bash
# 交互模式 (进入容器进行手动调试)
./nccl_container_manager.sh --interactive

# 自定义镜像
./nccl_container_manager.sh --image my-nccl:latest

# 详细日志
./nccl_container_manager.sh --log-level DEBUG
```

### 3.3 Docker 镜像构建

```bash
# 构建镜像
docker build -t nccl-test:latest .

# 验证镜像
docker run --rm --gpus all nccl-test:latest nvidia-smi
```

### 3.4 Kubernetes 部署

```bash
# 进入 Kubernetes 配置目录
cd k8s/

# 部署测试
./deploy.sh deploy

# 查看状态
./deploy.sh status

# 查看日志
./deploy.sh logs

# 清理资源
./deploy.sh cleanup
```

---

## 4. 多节点测试

### 4.1 多节点测试概述

多节点 NCCL 测试用于验证跨节点的分布式通信性能，提供两种部署方案：

- **Kubernetes 方案（推荐）**：云原生环境，自动调度和资源管理。适合已有 K8s 集群、需要多租户隔离的场景
- **原生方案**：传统 HPC 环境，直接控制。适合裸机部署、需要最小化中间层的场景

**多节点特有的前置检查**（单节点通过后再进行）：

| 检查项                | 命令                       | 通过标准                                                                                                                                     |
| --------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 节点间网络连通        | `ping <remote_host>`       | 延迟 < 1ms（同机房）                                                                                                                         |
| InfiniBand 链路       | `ibstatus`                 | 所有端口 `Active`，速率符合预期                                                                                                              |
| 时钟同步              | `ntpdate -q time.nist.gov` | 各节点偏差 < 1ms                                                                                                                             |
| 防火墙                | `sudo ufw status`          | NCCL 端口（默认 29500）未被拦截                                                                                                              |
| NCCL 版本一致         | `dpkg -l \| grep libnccl2` | 所有节点版本号完全相同                                                                                                                       |
| GPU 拓扑一致          | `nvidia-smi topo -m`       | 各节点 NVLink 拓扑结构一致（异构拓扑可能导致通信不均衡）                                                                                     |
| **NCCL 环境变量一致** | `env \| grep NCCL_`        | 所有节点的 `NCCL_*` 变量设置完全相同。即使一个节点的 `NCCL_P2P_LEVEL` 不同，也会导致通信路径不对称，出现"部分 GPU 快、部分 GPU 慢"的疑难问题 |

> **单节点先通过，再测多节点**——绝大多数多节点通信问题实际上是某个节点的 NVLink 或 PCIe 拓扑异常，单节点测试即可暴露。

### 4.2 快速开始

#### 4.2.1 Kubernetes 方案（推荐）

```bash
# 进入 Kubernetes 配置目录
cd k8s/

# 快速部署
./deploy.sh deploy

# 自定义配置
./deploy.sh deploy --gpus 4 --test-size 1G --network-backend ib

# 查看状态和日志
./deploy.sh status
./deploy.sh logs
```

#### 4.2.2 原生方案

原生方案通过脚本在每个节点上手动启动进程。**启动架构**：

- **`0` / `1`** 是 NCCL rank 编号：rank-0 为主节点（Master），rank-1 为第一个工作节点（Worker），依此类推
- **`192.168.1.100`** 是主节点的 IP 地址，所有工作节点通过 `MASTER_ADDR` 环境变量向主节点注册
- 脚本依赖 **SSH 免密登录**（或 `pdsh`/`mpirun`）在各节点上同步启动，确保所有 rank 几乎同时开始 NCCL 初始化
- **前置条件**：所有节点必须能通过 SSH 互访、使用相同的 NCCL 版本、且 `NCCL_*` 环境变量配置一致（见 §4.1 检查表）

```bash
# 在主节点上 (rank-0, IP 192.168.1.100)
./nccl_multinode_launcher.sh 0 192.168.1.100

# 在工作节点上 (rank-1, IP 192.168.1.101)
./nccl_multinode_launcher.sh 1 192.168.1.100

# rank-2, rank-3 等依此类推，每个节点执行一次，rank 编号递增

# 或直接使用 nccl_benchmark.sh（内部封装了 mpirun）
./nccl_benchmark.sh -m --master-addr 192.168.1.100 --network ib
```

### 4.3 PXN 模式多节点测试

PXN (Parallel eXecution Network) 模式是 NCCL 的高级网络优化功能，专为多节点分布式训练设计。

#### 4.3.1 PXN 模式概述

PXN 适用于节点内有 NVLink + 节点间有高速 IB 网络的混合拓扑——它让 NCCL 同时利用节点内 NVLink 和节点间 IB 做流水线通信，而不是串行执行"先节点内 AllReduce、再跨节点 AllReduce"。

**前置条件**：

- 节点内至少 2 GPU 通过 NVLink 连接
- 跨节点有 InfiniBand 或高速以太网（RoCE）
- NCCL 2.12+（推荐 2.18+）

**核心特性**：

- **并行网络执行**：NVLink 负责节点内 reduce，IB 同时做节点间数据传输
- **动态负载均衡**：根据数据量和 NVLink/IB 带宽比例自动调整流水线深度
- **容错机制**：IB 链路中断时自动回退到纯 NVLink 模式（节点内通信不受影响）

#### 4.3.2 快速开始

```bash
# 基础 PXN 测试
./nccl_benchmark.sh -m --master-addr 192.168.1.100 --network pxn

# 指定优化级别 (PXN 模式支持三种优化级别)
./nccl_benchmark.sh -m --master-addr 192.168.1.100 --network pxn --optimization-level conservative  # 保守模式
./nccl_benchmark.sh -m --master-addr 192.168.1.100 --network pxn --optimization-level balanced     # 平衡模式 (推荐)
./nccl_benchmark.sh -m --master-addr 192.168.1.100 --network pxn --optimization-level aggressive   # 激进模式

# 大规模测试
./nccl_benchmark.sh -m --master-addr 192.168.1.100 --network pxn --size 1G --time 300 --optimization-level balanced

```

---

## 5. 网络配置详解

### 5.1 环境变量调优基础

NCCL 的性能高度依赖环境变量配置。以下按**排查顺序**组织——从最常用的调试开关，到 P2P/NVLink 控制，再到内存和性能参数。

#### 第一层：诊断开关

| 变量                | 默认   | 作用                                                                              | 何时修改                                              |
| ------------------- | ------ | --------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `NCCL_DEBUG`        | `WARN` | 日志级别。`INFO` 输出拓扑检测结果和通信路径选择；`TRACE` 输出每次通信的详细时间戳 | 性能异常时设为 `INFO`，查看 NCCL 选择了什么算法和路径 |
| `NCCL_DEBUG_SUBSYS` | 全部   | 限制日志范围。常用值：`INIT`（初始化）、`NET`（网络）、`GRAPH`（拓扑图）          | 日志太多时缩小范围                                    |
| `NCCL_DEBUG_FILE`   | stderr | 日志输出文件。`%h` = 主机名, `%p` = PID                                           | 多节点排查时必须写文件，否则日志交错不可读            |

```bash
# 典型调试命令
NCCL_DEBUG=INFO NCCL_DEBUG_FILE=/tmp/nccl_debug.log ./nccl_benchmark.sh --network nvlink
```

#### 第二层：传输路径控制

| 变量                 | A100 建议值                 | 作用                                                                                           |
| -------------------- | --------------------------- | ---------------------------------------------------------------------------------------------- |
| `NCCL_P2P_LEVEL`     | `NVL`                       | 强制 P2P 走 NVLink（避免回退到 PCIe）。如果拓扑中有 SYS 连接的 GPU，NCCL 会为这些 GPU 自动降级 |
| `NCCL_P2P_DISABLE`   | `0`                         | 设为 `1` 强制禁用 P2P（仅调试用），此时所有 GPU 间通信走 CPU 中转                              |
| `NCCL_IB_DISABLE`    | `1` (单节点) / `0` (多节点) | 单节点测试时禁用 IB 可排除网络变量干扰                                                         |
| `NCCL_SOCKET_IFNAME` | `eth0` 或 `ib0`             | 多节点时指定 NCCL 使用的网络接口。支持正则排除：`^docker0,lo`                                  |
| `NCCL_NET_GDR_LEVEL` | `2`                         | GPUDirect RDMA 级别。0=禁用, 1=仅读, 2=读写(推荐), 3=强制。需要硬件支持                        |

#### 第三层：性能参数

| 变量                 | 默认 | 建议范围     | 作用                                                              |
| -------------------- | ---- | ------------ | ----------------------------------------------------------------- |
| `NCCL_NTHREADS`      | 自动 | 256-512      | CUDA 线程数。更多线程提高并发但增加开销。A100 (108 SM) 建议 384   |
| `NCCL_MIN_NCHANNELS` | 自动 | 4-8          | 最小通信通道数。通道越多带宽越高，但延迟增加。NVSwitch 环境建议 8 |
| `NCCL_MAX_NCHANNELS` | 自动 | 16-32        | 最大通道数。与 Min 配合使用，NCCL 在范围内自适应                  |
| `NCCL_BUFFSIZE`      | 4 MB | 8-16 MB      | 通信缓冲区。大数据传输（> 1 GB）建议 16 MB                        |
| `NCCL_ALGO`          | 自动 | `Ring,Tree`  | 强制指定算法。通常让 NCCL 自动选；排查时可以固定一个算法做对比    |
| `NCCL_CROSS_NIC`     | `0`  | `1` (多网卡) | 有多个 IB 网卡时启用，支持跨网卡负载均衡                          |

> 性能参数通常不需要手动设置——NCCL 的自动调优在大多数场景下已经是最优。只有当 `allreduce_perf` 的 bus_bw 明显低于 P2P 单向实测（A100: 239 GB/s）时，才需要调整上述参数。

### 5.2 网络后端配置策略

脚本支持 7 种网络后端模式，每种模式都有特定的配置策略和硬件检查机制：

#### 5.2.1 自动检测模式 (`--network auto`)

**检测优先级**：

1. NVLink (单节点多 GPU)
2. InfiniBand (多节点首选)
3. PCIe P2P (单节点备选)
4. 以太网 (通用选择)
5. Socket (兜底方案)

#### 5.2.2 InfiniBand 模式 (`--network ib`)

**适用场景**：高性能多节点通信，支持原生 InfiniBand 和 RoCE

**硬件检查**：

- 验证 InfiniBand 设备存在
- 检查设备状态 (Active/Down)
- 确认链路层类型 (InfiniBand/Ethernet)

**NCCL 参数配置**：

```bash
# 基础配置
NCCL_IB_DISABLE=0                   # 启用 InfiniBand 传输
NCCL_NET_GDR_LEVEL=2               # GPUDirect RDMA 级别 (0-3)
NCCL_P2P_DISABLE=0                 # 启用 P2P 通信
NCCL_P2P_LEVEL=PIX                 # P2P 级别：PIX (PCIe) 或 NVL (NVLink)

# InfiniBand 特定参数
NCCL_IB_HCA=mlx5_0                 # HCA 设备名 (自动检测)
NCCL_IB_TC=136                     # Traffic Class (流量类别)
NCCL_IB_SL=0                       # Service Level (服务级别)
NCCL_IB_TIMEOUT=22                 # 超时设置 (4.096 μs × 2^22)
NCCL_IB_RETRY_CNT=7                # 重试次数
NCCL_IB_GID_INDEX=0                # 原生 IB: 0, RoCE v2: 3
NCCL_IB_PKEY=0                     # Partition Key

# 性能优化参数
NCCL_BUFFSIZE=8388608              # 缓冲区大小 (8 MB)
NCCL_CROSS_NIC=0                   # 跨网卡通信 (0=禁用, 1=启用)
```

**参数含义详解**：

- **NCCL_NET_GDR_LEVEL**: GPUDirect RDMA 级别
  - `0`: 禁用 GPUDirect
  - `1`: 启用 GPUDirect 读取
  - `2`: 启用 GPUDirect 读写 (推荐)
  - `3`: 强制启用 GPUDirect

- **NCCL_IB_TC**: InfiniBand Traffic Class，用于 QoS 控制
- **NCCL_IB_TIMEOUT**: 超时值，计算公式：4.096 μs × 2^value
- **NCCL_IB_GID_INDEX**: 全局标识符索引，RoCE 需要设置为 3

#### 5.2.3 NVLink 模式 (`--network nvlink`)

**适用场景**：单节点多 GPU 环境，GPU 间直连通信

**硬件检查**：验证 NVLink 拓扑和连接状态

**NCCL 参数配置**：

```bash
# 基础配置
NCCL_P2P_DISABLE=0                 # 启用 P2P 通信
NCCL_P2P_LEVEL=NVL                 # 强制使用 NVLink
NCCL_IB_DISABLE=1                  # 禁用 InfiniBand
NCCL_NET_DISABLE=1                 # 禁用网络通信

# NVLink 特定参数
NCCL_NVLS_ENABLE=1                 # 启用 NVLink SHARP (仅 H100/H200/B200 支持, A100 上忽略此参数)
NCCL_NVLS_CHUNKSIZE=524288         # NVLink 块大小 (512 KB)
NCCL_TREE_THRESHOLD=0              # Tree 算法阈值

# 性能优化参数 (根据优化级别)
# 保守模式
NCCL_NTHREADS=256                  # 线程数
NCCL_MIN_NCHANNELS=16              # 最小通道数
NCCL_MAX_NCHANNELS=32              # 最大通道数

# 平衡模式
NCCL_NTHREADS=384                  # 线程数
NCCL_BUFFSIZE=12582912             # 缓冲区大小 (12 MB)

# 激进模式
NCCL_NTHREADS=512                  # 线程数
NCCL_BUFFSIZE=16777216             # 缓冲区大小 (16 MB)
NCCL_CHECK_POINTERS=1              # 启用指针检查
```

**参数含义详解**：

- **NCCL_NVLS_ENABLE**: NVLink SHARP 技术，在 NVSwitch 上直接完成归约（仅 H100/H200/B200 支持，A100 上设置此参数无效，参见 [NVLink 技术入门](../../01_hardware_architecture/nvlink/nvlink_intro.md)）
- **NCCL_NVLS_CHUNKSIZE**: NVLink 传输的数据块大小
- **NCCL_NTHREADS**: NCCL 使用的线程数，影响并发度
- **NCCL_MIN/MAX_NCHANNELS**: 通信通道数范围，影响带宽利用率

#### 5.2.4 PCIe P2P 模式 (`--network pcie`)

**适用场景**：单节点多 GPU，无 NVLink 连接的环境

**NCCL 参数配置**：

```bash
# 基础配置
NCCL_P2P_DISABLE=0                 # 启用 P2P 通信
NCCL_P2P_LEVEL=PIX                 # 使用 PCIe P2P
NCCL_IB_DISABLE=1                  # 禁用 InfiniBand
NCCL_NVLS_ENABLE=0                 # 禁用 NVLink SHARP

# PCIe 特定参数
NCCL_ALGO=Ring                     # 使用 Ring 算法
NCCL_MAX_NCHANNELS=16              # 最大通道数
NCCL_MIN_NCHANNELS=1               # 最小通道数
NCCL_P2P_NET_CHUNKSIZE=131072      # P2P 网络块大小 (128 KB)

# 性能优化参数
NCCL_NTHREADS=128                  # 线程数
NCCL_BUFFSIZE=8388608              # 缓冲区大小 (8 MB)
NCCL_DMABUF_ENABLE=1               # 启用 DMA 缓冲区
NCCL_REG_CACHE_ENABLE=1            # 启用注册缓存
NCCL_NET_GDR_LEVEL=1               # 基础 GPUDirect 支持
```

**参数含义详解**：

- **NCCL_P2P_NET_CHUNKSIZE**: P2P 传输的数据块大小
- **NCCL_DMABUF_ENABLE**: 启用 DMA 缓冲区，减少内存拷贝
- **NCCL_REG_CACHE_ENABLE**: 启用内存注册缓存，提高性能

#### 5.2.5 以太网模式 (`--network ethernet`)

**适用场景**：标准以太网环境，多节点通信

**NCCL 参数配置**：

```bash
# 基础配置
NCCL_IB_DISABLE=1                  # 禁用 InfiniBand
NCCL_P2P_DISABLE=0                 # 启用 P2P 通信
NCCL_P2P_LEVEL=PIX                 # 使用 PCIe P2P

# 以太网特定参数
NCCL_SOCKET_IFNAME=^docker0,lo,virbr0,veth,br-  # 排除虚拟接口
NCCL_NET_GDR_LEVEL=0               # 禁用 GPUDirect (以太网不支持)

# 性能优化参数
NCCL_NTHREADS=64                   # 线程数
NCCL_BUFFSIZE=4194304              # 缓冲区大小 (4 MB)
NCCL_MIN_NCHANNELS=1               # 最小通道数
NCCL_MAX_NCHANNELS=8               # 最大通道数
NCCL_SOCKET_NTHREADS=8             # Socket 线程数
NCCL_NSOCKS_PERTHREAD=1            # 每线程 Socket 数
```

**参数含义详解**：

- **NCCL_SOCKET_IFNAME**: 网络接口名称，支持正则表达式排除
- **NCCL_SOCKET_NTHREADS**: Socket 传输使用的线程数
- **NCCL_NSOCKS_PERTHREAD**: 每个线程使用的 Socket 连接数

#### 5.2.6 Socket 模式 (`--network socket`)

**适用场景**：调试和兼容性测试，强制使用 TCP Socket

**NCCL 参数配置**：

```bash
# 基础配置
NCCL_IB_DISABLE=1                  # 禁用 InfiniBand
NCCL_P2P_DISABLE=1                 # 禁用 P2P 通信
NCCL_SHM_DISABLE=1                 # 禁用共享内存
NCCL_NET_DISABLE=0                 # 启用网络传输

# Socket 特定参数
NCCL_SOCKET_IFNAME=^docker0,lo,virbr0,veth,br-  # 排除虚拟接口
NCCL_NET_GDR_LEVEL=0               # 禁用 GPUDirect

# 容器环境特殊配置
NCCL_SOCKET_FORCE=1                # 强制使用 Socket (容器环境)
NCCL_IGNORE_DISABLED_P2P=1         # 忽略禁用的 P2P
NCCL_CUMEM_ENABLE=0                # 禁用 CUDA 内存管理
NCCL_CHECK_DISABLE=0               # 启用检查
```

**参数含义详解**：

- **NCCL_SOCKET_FORCE**: 强制使用 Socket 传输，忽略其他选项
- **NCCL_IGNORE_DISABLED_P2P**: 忽略 P2P 禁用状态
- **NCCL_CUMEM_ENABLE**: CUDA 统一内存管理

#### 5.2.7 共享内存模式 (`--network shm`)

**适用场景**：单节点环境，调试和兼容性测试

**NCCL 参数配置**：

```bash
# 基础配置
NCCL_IB_DISABLE=1                  # 禁用 InfiniBand
NCCL_P2P_DISABLE=1                 # 禁用 P2P 通信
NCCL_SHM_DISABLE=0                 # 启用共享内存
NCCL_NET_GDR_LEVEL=0               # 禁用 GPUDirect

# 共享内存特定参数
NCCL_NTHREADS=32                   # 线程数
NCCL_BUFFSIZE=2097152              # 缓冲区大小 (2 MB)
NCCL_MIN_NCHANNELS=1               # 最小通道数
NCCL_MAX_NCHANNELS=4               # 最大通道数
NCCL_CUMEM_ENABLE=0                # 禁用 CUDA 内存管理
```

**参数含义详解**：

- **NCCL_SHM_DISABLE**: 控制共享内存传输的启用/禁用
- 共享内存模式性能较低，主要用于兼容性验证

#### 5.2.8 PXN 模式 (`--network pxn`)

**适用场景**：多节点高性能通信，Process Exchange Network

**智能 P2P 配置**：PXN 模式现在支持智能选择节点内 P2P 通信级别：

- **自动检测 NVLink**：如果检测到 NVLink 连接，自动设置 `NCCL_P2P_LEVEL=NVL`
- **PCIe 回退**：如果未检测到 NVLink，回退到 `NCCL_P2P_LEVEL=PIX`
- **节点间通信**：始终使用 PXN 集合通信 + 高速网络 (InfiniBand/以太网)

**NCCL 参数配置**：

```bash
# 基础配置
NCCL_ALGO=Ring,Tree,CollNet        # 支持的算法
NCCL_PROTO=Simple,LL,LL128         # 支持的协议
NCCL_NET_GDR_LEVEL=2               # GPUDirect RDMA
NCCL_P2P_DISABLE=0                 # 启用 P2P 通信
NCCL_P2P_LEVEL=NVL|PIX             # 智能选择: NVL (NVLink) 或 PIX (PCIe)
NCCL_IB_DISABLE=0                  # 启用 InfiniBand
NCCL_CROSS_NIC=1                   # 启用跨网卡通信

# PXN 特定参数
NCCL_PXN_DISABLE=0                 # 启用 PXN
NCCL_COLLNET_NODE_THRESHOLD=2      # 集合通信节点阈值
NCCL_COLLNET_CHAIN_THRESHOLD=2     # 链式通信阈值

# 性能优化参数 (根据优化级别)
# 保守模式
NCCL_NTHREADS=256                  # 线程数
NCCL_BUFFSIZE=8388608              # 缓冲区大小 (8 MB)
NCCL_MIN_NCHANNELS=4               # 最小通道数
NCCL_MAX_NCHANNELS=12              # 最大通道数

# 平衡模式
NCCL_NTHREADS=384                  # 线程数
NCCL_BUFFSIZE=12582912             # 缓冲区大小 (12 MB)
NCCL_MIN_NCHANNELS=6               # 最小通道数
NCCL_MAX_NCHANNELS=16              # 最大通道数
NCCL_P2P_NET_CHUNKSIZE=262144      # P2P 网络块大小 (256 KB)

# 激进模式 (启用完全自动优化)
NCCL_NTHREADS=512                  # 线程数
NCCL_BUFFSIZE=16777216             # 缓冲区大小 (16 MB)
NCCL_MIN_NCHANNELS=8               # 最小通道数
NCCL_MAX_NCHANNELS=20              # 最大通道数
NCCL_P2P_NET_CHUNKSIZE=524288      # P2P 网络块大小 (512 KB)
NCCL_CHECK_POINTERS=1              # 启用指针检查
NCCL_SOCKET_NTHREADS=16            # Socket 线程数
NCCL_NSOCKS_PERTHREAD=2            # 每线程 Socket 数
# 注意：激进模式会移除 NCCL_ALGO 和 NCCL_PROTO 限制，启用 NCCL 完全自动优化
```

**参数含义详解**：

- **NCCL_P2P_LEVEL (智能选择)**: 节点内 P2P 通信级别
  - `NVL`: 当检测到 NVLink 时自动选择，A100 NVLink 3.0 提供 600 GB/s 双向带宽，< 2 μs 延迟
  - `PIX`: 当未检测到 NVLink 时回退选择，PCIe Gen4 提供 ~32 GB/s 带宽，5-10 μs 延迟
  - 智能选择确保在不同硬件配置下都能获得最佳节点内通信性能
  - 脚本会自动检测 NVLink 连接数量并在日志中显示检测结果

- **NCCL_COLLNET_NODE_THRESHOLD**: 启用集合通信的最小节点数
- **NCCL_COLLNET_CHAIN_THRESHOLD**: 链式通信的阈值
- **NCCL_PXN_DISABLE**: 控制 PXN 功能的启用/禁用

**优化级别差异**：

- **保守模式 (conservative)**: 使用固定的算法和协议配置，稳定性优先
- **平衡模式 (balanced)**: 部分启用自动选择，平衡性能与稳定性
- **激进模式 (aggressive)**: 完全移除算法和协议限制，启用 NCCL 完全自动优化

**性能优势**：

- **节点内优化**：自动利用最快的节点内通信路径 (NVLink > PCIe P2P)。NVLink 环境下可达 600 GB/s 双向带宽（A100），< 2 μs 延迟；PCIe 环境下可达 ~32 GB/s 带宽（Gen4），5-10 μs 延迟。
- **节点间优化**：使用 PXN 集合通信算法优化多节点通信。
- **混合架构适配**：异构集群中自动为有 NVLink 的节点启用 NVL 路径，无 NVLink 的走 PIX。
- **自适应算法选择**：激进模式下启用 NCCL 完全自动优化，根据数据大小和网络拓扑动态选择最佳算法。
- **多级缓存优化**：优化数据传输路径，减少内存拷贝开销。

### 5.3 通用 NCCL 参数详解

> 参数的选择建议和调优流程见 [§5.1 环境变量调优基础](#51-环境变量调优基础)。本节是完整参数列表和含义参考。

除了各网络后端的特定参数外，以下是所有网络后端都会使用的通用 NCCL 参数：

#### 5.3.1 调试和日志参数

```bash
# 调试级别
NCCL_DEBUG=INFO                     # 调试级别: WARN, INFO, TRACE
NCCL_DEBUG_SUBSYS=INIT,NET         # 调试子系统: INIT, NET, GRAPH, COLL, P2P, SHM, BOOTSTRAP, ALL
NCCL_DEBUG_FILE=/tmp/nccl_%h_%p.log # 调试日志文件 (%h=主机名, %p=进程 ID)

# 性能分析
NCCL_ALGO_TRACE=1                   # 启用算法跟踪
NCCL_PROTO_TRACE=1                  # 启用协议跟踪
```

**参数含义**：

- **NCCL_DEBUG**: 控制调试信息的详细程度
  - `WARN`: 仅显示警告和错误
  - `INFO`: 显示基本信息、警告和错误
  - `TRACE`: 显示详细的跟踪信息（性能影响较大）

- **NCCL_DEBUG_SUBSYS**: 指定要调试的子系统
  - `INIT`: 初始化过程
  - `NET`: 网络通信
  - `GRAPH`: 通信图构建
  - `COLL`: 集合通信操作
  - `P2P`: 点对点通信
  - `SHM`: 共享内存
  - `BOOTSTRAP`: 引导过程

#### 5.3.2 性能优化参数

```bash
# 线程和通道配置
NCCL_NTHREADS=256                   # NCCL 使用的线程数 (32-512)
NCCL_MIN_NCHANNELS=1                # 最小通道数
NCCL_MAX_NCHANNELS=32               # 最大通道数

# 缓冲区配置
NCCL_BUFFSIZE=8388608               # 缓冲区大小 (字节)
NCCL_LL_BUFFSIZE=1048576            # Low-Latency 缓冲区大小
NCCL_LL128_BUFFSIZE=134217728       # LL128 缓冲区大小

# 算法选择
NCCL_ALGO=Ring,Tree,CollNet         # 允许的算法: Ring, Tree, CollNet
NCCL_PROTO=Simple,LL,LL128          # 允许的协议: Simple, LL, LL128
```

**参数含义**：

- **NCCL_NTHREADS**: NCCL 内部使用的线程数
  - 更多线程可以提高并发度，但也会增加开销
  - 推荐值：256-512 (根据 GPU 数量调整)

- **NCCL_MIN/MAX_NCHANNELS**: 通信通道数范围
  - 更多通道可以提高带宽利用率
  - 但也会增加延迟和内存开销

- **NCCL_BUFFSIZE**: 主缓冲区大小
  - 较大的缓冲区可以提高大数据传输的效率
  - 但会增加内存使用和延迟

- **NCCL_ALGO**: 集合通信算法
  - `Ring`: 环形算法，适合带宽受限环境
  - `Tree`: 树形算法，适合延迟敏感场景
  - `CollNet`: 集合网络算法，需要硬件支持

- **NCCL_PROTO**: 通信协议
  - `Simple`: 标准协议，兼容性最好
  - `LL`: Low-Latency 协议，降低延迟
  - `LL128`: 128 位 Low-Latency 协议，平衡延迟和带宽

#### 5.3.3 内存管理参数

```bash
# CUDA 内存管理
NCCL_CUMEM_ENABLE=0                 # CUDA 统一内存管理 (0=禁用, 1=启用)
NCCL_REG_CACHE_ENABLE=1             # 内存注册缓存 (0=禁用, 1=启用)
NCCL_DMABUF_ENABLE=1                # DMA 缓冲区 (0=禁用, 1=启用)

# 内存对齐
NCCL_MEM_ALIGN=4096                 # 内存对齐大小 (字节)
NCCL_LL_THRESHOLD=16384             # LL 协议阈值 (字节)
NCCL_TREE_THRESHOLD=0               # Tree 算法阈值 (字节, 0=自动)
```

**参数含义**：

- **NCCL_CUMEM_ENABLE**: CUDA 统一内存管理
  - 启用后可以自动管理 GPU 和 CPU 内存
  - 可能影响性能，建议在兼容性问题时启用

- **NCCL_REG_CACHE_ENABLE**: 内存注册缓存
  - 缓存已注册的内存区域，减少重复注册开销
  - 推荐启用以提高性能

- **NCCL_DMABUF_ENABLE**: DMA 缓冲区
  - 启用 DMA 缓冲区可以减少内存拷贝
  - 推荐启用以提高性能

#### 5.3.4 网络通用参数

```bash
# 网络传输控制
NCCL_NET_DISABLE=0                  # 禁用网络传输 (0=启用, 1=禁用)
NCCL_NET_GDR_LEVEL=2                # GPUDirect RDMA 级别 (0-3)
NCCL_NET_GDR_READ=1                 # GPUDirect 读取 (0=禁用, 1=启用)

# 跨设备通信
NCCL_CROSS_NIC=0                    # 跨网卡通信 (0=禁用, 1=启用)
NCCL_CHECK_POINTERS=0               # 指针检查 (0=禁用, 1=启用)
NCCL_IGNORE_CPU_AFFINITY=1          # 忽略 CPU 亲和性 (0=遵守, 1=忽略)
```

**参数含义**：

- **NCCL_NET_GDR_LEVEL**: GPUDirect RDMA 级别
  - `0`: 禁用 GPUDirect
  - `1`: 启用 GPUDirect 读取
  - `2`: 启用 GPUDirect 读写 (推荐)
  - `3`: 强制启用 GPUDirect

- **NCCL_CROSS_NIC**: 跨网卡通信
  - 启用后可以使用多个网卡进行通信
  - 可以提高带宽，但可能增加复杂性

- **NCCL_CHECK_POINTERS**: 指针有效性检查
  - 启用后会检查传入指针的有效性
  - 有助于调试，但会影响性能

#### 5.3.5 容错和重试参数

```bash
# 超时和重试
NCCL_TIMEOUT=1800                   # 操作超时时间 (秒)
NCCL_RETRY_COUNT=3                  # 重试次数
NCCL_ABORT_ON_ERROR=0               # 错误时是否中止 (0=继续, 1=中止)

# 健康检查
NCCL_HEALTH_CHECK_ENABLE=1          # 启用健康检查
NCCL_HEALTH_CHECK_TIMEOUT=30        # 健康检查超时 (秒)
```

**参数含义**：

- **NCCL_TIMEOUT**: 操作超时时间
  - 设置 NCCL 操作的最大等待时间
  - 过短可能导致误报，过长可能延迟错误检测

- **NCCL_RETRY_COUNT**: 失败重试次数
  - 网络不稳定时的重试机制
  - 适当的重试可以提高稳定性

### 5.4 环境变量优先级和覆盖规则

1. **用户预设变量**：不会被脚本覆盖
2. **硬件检测**：根据检测结果自动配置
3. **网络模式配置**：调用对应的配置函数
4. **配置验证**：验证配置的有效性
5. **实时展示**：显示当前环境变量状态

---

### 5.5 性能分析与基准

以下基于 A100-SXM4-80GB (8 GPU, NVSwitch Gen2) + NCCL 2.29.3 的 `allreduce_perf` 实测数据。详细测试方法和完整曲线见 [NCCL 基准测试方法论](04_nccl_benchmark.md)。

#### 5.5.1 输出解读

`allreduce_perf` 的输出格式：

```text
    size    count    type    op    root    time(us)  alg_bw  bus_bw  #wrong
1073741824  268435456 float   sum   -1      8191.38  131.08  224.71       0
```

| 字段     | 含义                                 | 7 GPU 示例值            |
| -------- | ------------------------------------ | ----------------------- |
| `size`   | 每 GPU 数据量 (bytes)                | 1 GB                    |
| `time`   | 操作耗时 (μs)                        | 8191 μs                 |
| `alg_bw` | 算法带宽 = size / time               | 131 GB/s                |
| `bus_bw` | **总线带宽** = alg_bw × 理论修正因子 | **225 GB/s** ← 关注这个 |
| `#wrong` | 校验错误数                           | 必须为 0                |

> `alg_bw` vs `bus_bw`：alg_bw 反映单个 GPU 的带宽；bus_bw 是等效总线带宽，消除了 AllReduce 的数据冗余因子。**bus_bw 才是和 P2P 带宽对标的值**。

**异常输出对照**：以下是 A100 上 2 GPU AllReduce 1 GB 数据时，正常 NVLink 与强制禁用 P2P（`NCCL_P2P_DISABLE=1`，所有通信走 CPU 内存中转）的实测对比：

```text
# 正常 (NVLink, bus_bw 196 GB/s)
  1073741824  268435456  float  sum  -1   5467.92  196.37  196.37  0

# 异常 (P2P 禁用, bus_bw 4.6 GB/s，降幅 97.7%)
  1073741824  268435456  float  sum  -1  232713.0    4.61    4.61  0
```

| 指标      | 正常 (NVLink) | 异常 (P2P 禁用)                                                             | 说明                                                                  |
| --------- | ------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| bus_bw    | **196 GB/s**  | **4.6 GB/s**                                                                | 差距 42 倍，远超 PCIe Gen4 极限 (~28 GB/s)，说明 CPU 内存拷贝是主瓶颈 |
| time (μs) | 5,468         | 232,713                                                                     | 耗时增加 42 倍                                                        |
| 排查路径  | —             | 检查 `nvidia-smi topo -m`→确认 NVLink 状态→检查 `NCCL_P2P_DISABLE` 环境变量 | —                                                                     |

> 当 bus_bw 在 1 GB 数据时低于 10 GB/s，优先怀疑 NCCL 根本没有使用 P2P 路径（而非 NVLink 速率不足）。`NCCL_DEBUG=INFO` 日志中搜索 `P2P` 和 `NET` 可确认实际路径。

#### 5.5.2 A100 实测数据

| GPU 数    | 1 GB bus_bw  | 64 MB bus_bw | 关键发现                            |
| --------- | ------------ | ------------ | ----------------------------------- |
| 2         | **197 GB/s** | 170 GB/s     | 接近 P2P 单向 (239 GB/s) 的 82%     |
| 4         | **220 GB/s** | 180 GB/s     | 超越 2 GPU 的 bus_bw——NVSwitch 优势 |
| 7         | **225 GB/s** | 160 GB/s     | bus_bw 继续上升，NVSwitch 全互连    |
| 7+1(故障) | **~28 GB/s** | —            | GPU 7 走 PCIe，拖垮全部带宽         |

#### 5.5.3 带宽效率曲线

AllReduce 的 bus_bw 随数据量变化的规律（7 GPU）：

| 数据量 | bus_bw       | 占 P2P 峰值比 | 瓶颈                    |
| ------ | ------------ | ------------- | ----------------------- |
| 1 MB   | 16.8 GB/s    | 7%            | Launch latency (α) 主导 |
| 8 MB   | 89 GB/s      | 37%           | 混合区                  |
| 64 MB  | 170 GB/s     | 71%           | 接近 β 主导             |
| 256 MB | 206 GB/s     | 86%           | 接近上限                |
| 1 GB   | **225 GB/s** | **94%**       | Ring 饱和               |

**判断标准**：

- `bus_bw < 50 GB/s` 且数据量 > 64 MB → 拓扑异常（走了 PCIe 而非 NVLink），检查 `nvidia-smi topo -m`
- `bus_bw 100-180 GB/s` 且数据量 > 64 MB → 可能有 GPU 走了 fallback 路径，或 NCCL 选用了 Ring 而非 Tree（NVSwitch 环境）
- `bus_bw > 200 GB/s` → 正常，NVLink + NVSwitch 工作正常

#### 5.5.4 优化决策树

```text
bus_bw < 期望？
  ├── 是 → 检查 topo（有无 SYS/NODE 连接）
  │       → 检查 NVLink 链路速率（是否所有 link 25 GB/s）
  │       → 检查 GPU 进程（是否有其他任务抢占 SM）
  │       → 尝试手动指定 NCCL_ALGO=Tree 或 Ring 对比
  │       → 增大 NCCL_MIN_NCHANNELS（NVSwitch 环境建议 8）
  └── 否 → 带宽已达标，关注计算/通信重叠优化
```

---

## 6. 故障排除与诊断

### 6.1 常见问题诊断

#### 6.1.1 环境依赖问题

**Python/PyTorch 问题**：

```bash
# 检查 Python 和 PyTorch 安装
python3 -c "import torch; print(torch.__version__)"

# 检查 CUDA 支持
python3 -c "import torch; print(torch.cuda.is_available())"

# 检查 NCCL 版本
python3 -c "import torch; print(torch.cuda.nccl.version())"
```

**GPU 驱动问题**：

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 CUDA 版本
nvcc --version
```

**InfiniBand 问题**：

```bash
# 检查 IB 设备
ibstat

# 检查 IB 详细信息
ibv_devinfo

# 检查网络连接
ping <remote_host>
```

#### 6.1.2 网络连接问题

**主节点连接失败**：

```bash
# 检查网络连通性
ping 192.168.1.100

# 检查端口可用性
telnet 192.168.1.100 29500

# 检查防火墙设置
sudo ufw status
```

**多节点同步问题**：

```bash
# 确保所有节点时间同步
sudo ntpdate -s time.nist.gov

# 检查节点间网络延迟
ping -c 10 <remote_host>
```

#### 6.1.3 性能问题

**吞吐量低于预期**：

1. 检查网络后端选择是否正确
2. 验证硬件配置和驱动版本
3. 检查 NCCL 环境变量配置
4. 排查网络拥塞和干扰

**延迟过高**：

1. 检查 GPU 拓扑结构 — 参见 [GPU 集群健康检查](../01_gpu_ops/06_gpu_health_check.md)
2. 验证 NVLink 连接状态 — 参见 [NVLink 诊断](../../01_hardware_architecture/nvlink/nvlink_diagnostics.md)
3. 检查系统负载和资源竞争 — 参见 [DCGM 监控实操](../01_gpu_ops/05_dcgm_monitoring.md)
4. 优化 NCCL 算法选择 — 参见 [NCCL 技术理论](01_nccl_theory.md)

### 6.2 调试技巧

#### 6.2.1 详细日志分析

```bash
# 启用详细调试信息
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 运行测试并保存日志
./nccl_benchmark.sh --network auto 2>&1 | tee debug.log

# 分析关键信息
grep -E "NCCL|ERROR|WARNING" debug.log
```

#### 6.2.2 性能瓶颈诊断流程

1. **硬件检查**：验证 GPU、网络硬件状态
2. **软件检查**：确认驱动、NCCL 版本兼容性
3. **配置检查**：验证环境变量和网络配置
4. **基准对比**：与理论性能进行对比分析

### 6.3 Docker 和容器问题

**Docker 权限问题**：

```bash
# 将用户添加到 docker 组
sudo usermod -aG docker $USER
newgrp docker
```

**NVIDIA Container Toolkit 问题**：

```bash
# 检查 NVIDIA 运行时
docker info | grep nvidia

# 测试 GPU 访问
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

**容器网络问题**：

```bash
# 使用主机网络模式
docker run --network host --gpus all <image>

# 检查容器内网络配置
docker exec -it <container> ip addr show
```

### 6.4 Kubernetes 问题

**Pod 调度问题**：

```bash
# 检查节点 GPU 资源
kubectl describe nodes

# 检查 Pod 状态
kubectl get pods -o wide

# 查看 Pod 事件
kubectl describe pod <pod-name>
```

**网络连接问题**：

```bash
# 检查 Service 和 Endpoint
kubectl get svc,ep

# 测试 Pod 间连接
kubectl exec -it <pod1> -- ping <pod2-ip>
```

---

## 7. 附录

### 7.1 环境变量参考

> 快速速查表。详细的参数选择建议、排查流程和 A100 建议值见 [§5.1 环境变量调优基础](#51-环境变量调优基础)。

#### 7.1.1 核心 NCCL 环境变量

| 变量名               | 默认值 | 说明                | 示例值            |
| -------------------- | ------ | ------------------- | ----------------- |
| `NCCL_DEBUG`         | WARN   | 调试级别            | INFO, WARN, ERROR |
| `NCCL_IB_DISABLE`    | 0      | 禁用 InfiniBand     | 0, 1              |
| `NCCL_NET_GDR_LEVEL` | 未设置 | GPUDirect RDMA 级别 | 0, 1, 2, 3        |
| `NCCL_P2P_DISABLE`   | 0      | 禁用 P2P 通信       | 0, 1              |
| `NCCL_SHM_DISABLE`   | 0      | 禁用共享内存        | 0, 1              |

#### 7.1.2 网络特定变量

**InfiniBand 相关**：

- `NCCL_IB_HCA`: HCA 设备名
- `NCCL_IB_GID_INDEX`: GID 索引
- `NCCL_IB_TIMEOUT`: 超时设置

**Socket 相关**：

- `NCCL_SOCKET_IFNAME`: 网络接口名
- `NCCL_SOCKET_FAMILY`: 地址族 (AF_INET/AF_INET6)

### 7.2 命令参考

#### 7.2.1 nccl_benchmark.sh 参数

| 参数                   | 短参数 | 默认值       | 说明         |
| ---------------------- | ------ | ------------ | ------------ |
| `--size`               | `-s`   | 1M           | 测试数据大小 |
| `--time`               | `-t`   | 30           | 测试时间(秒) |
| `--network`            | `-n`   | auto         | 网络后端     |
| `--multinode`          | `-m`   | false        | 多节点模式   |
| `--master-addr`        | 无     | 无           | 主节点地址   |
| `--optimization-level` | 无     | conservative | 优化级别     |

#### 7.2.2 网络后端选项

- `auto`: 自动检测
- `nvlink`: NVLink
- `ib`: InfiniBand
- `pcie`: PCIe P2P
- `ethernet`: 以太网
- `socket`: Socket
- `shm`: 共享内存
- `pxn`: PXN 模式

### 7.3 性能基准数据

> GPU 间 NVLink 带宽和 P2P 实测数据见 [§1.1.2 网络后端](#112-nccl-核心概念)。AllReduce 有效带宽受算法和 GPU 数量影响，详见 [NCCL 基准测试方法论](04_nccl_benchmark.md)。

#### 7.3.1 网络性能基准

| 网络类型       | 理论带宽 | 典型延迟 | 实际性能     |
| -------------- | -------- | -------- | ------------ |
| InfiniBand EDR | 100 Gbps | 1-2 μs   | 80-90 Gbps   |
| InfiniBand HDR | 200 Gbps | 1-2 μs   | 160-180 Gbps |
| 100GbE RoCE    | 100 Gbps | 2-5 μs   | 70-85 Gbps   |
| 10GbE          | 10 Gbps  | 10-50 μs | 8-9 Gbps     |

### 7.4 参考资料

[1] [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) — NVIDIA Collective Communications Library 用户指南。
[2] [NCCL 环境变量参考](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) — 完整环境变量列表及参数说明。
[3] [GPUDirect RDMA 与 Storage 技术详解](../../01_hardware_architecture/gpudirect/01_gpudirect_technology.md) — 本仓库 GPUDirect 内部文档，含 RDMA 和 GDS 的顶层解析。
[4] [NVLink 技术入门](../../01_hardware_architecture/nvlink/nvlink_intro.md) — NVLink 1.0-6.0 版本演进与 NVSwitch 架构。
[5] [GPU P2P 带宽实测](../../02_gpu_programming/04_profiling/08_p2p_bandwidth.md) — A100 NVLink P2P 239 GB/s 实测。
[6] [NVIDIA Networking 文档](https://docs.nvidia.com/networking/) — InfiniBand / Ethernet 网络配置指南。

---
