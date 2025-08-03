# NCCL InfiniBand 测试验证工具说明文档

> 注意：**性能部分指标的解释和计算方法可能会根据具体的硬件配置和网络环境而有所不同。建议在实际测试中根据自己的环境进行调整和优化**。

---

> 目前进展：单节点测试基本验证OK。

## 1. 概述与系统要求

### 1.1 NCCL 测试背景

#### 1.1.1 为什么需要 NCCL 测试

在现代深度学习训练中，多GPU和分布式训练已成为处理大规模模型的标准方法。NCCL (NVIDIA Collective Communications Library) 作为NVIDIA提供的高性能集合通信库，负责GPU间的数据同步和通信。然而，NCCL的性能高度依赖于：

- **硬件配置**：GPU型号、内存带宽、PCIe拓扑结构
- **网络环境**：InfiniBand、RoCE、以太网的配置和性能
- **软件栈**：CUDA版本、驱动程序、NCCL库版本的兼容性
- **环境变量**：数百个NCCL参数的正确配置

不当的配置可能导致：

- 训练速度下降50-90%
- 通信延迟增加10-100倍
- 网络带宽利用率低于10%
- 分布式训练失败或不稳定

#### 1.1.2 NCCL 核心概念

**AllReduce 操作**：NCCL最重要的集合通信原语，用于梯度聚合

- 将所有GPU上的数据进行归约运算（如求和）
- 将结果广播到所有参与的GPU
- 是分布式训练中梯度同步的核心操作

**通信算法**：

- **Ring AllReduce**：适用于带宽受限环境，通信量为 `2(N-1)/N × data_size`
- **Tree AllReduce**：适用于延迟敏感场景，通信深度为 `log₂(N)`
- **Double Binary Tree**：NCCL 2.4+的默认算法，平衡延迟和带宽

**网络后端**：

- **NVLink**：GPU间直连，带宽300-600 GB/s
- **InfiniBand**：高性能网络，带宽12.5-50 GB/s (100-400 Gbps)
- **RoCE**：基于以太网的RDMA，带宽3.1-12.5 GB/s (25-100 Gbps)
- **TCP/Socket**：通用网络，带宽0.125-1.25 GB/s (1-10 Gbps)

### 1.2 工具概述

本工具套件提供了完整的 NCCL 测试和部署解决方案，包含以下核心工具：

#### 1.2.1 核心测试工具

**`nccl_benchmark.sh`** - 主要的 NCCL 性能测试工具，专门设计用于：

- **性能基准测试**：测量真实的NCCL通信性能
- **配置验证**：验证NCCL环境变量和网络配置
- **问题诊断**：识别性能瓶颈和配置问题
- **环境优化**：提供最佳实践配置建议

**`gpu_topology_detector.sh`** - GPU 拓扑检测工具：

- **硬件拓扑分析**：检测 GPU 间的连接方式（NVLink、PCIe）
- **NCCL 通信路径验证**：确认 NCCL 实际使用的通信路径
- **性能预测**：基于硬件拓扑预测通信性能
- **配置建议**：提供最优的网络后端配置建议

#### 1.2.2 容器化部署工具

**`nccl_container_manager.sh`** - 容器化测试管理工具：

- **单节点容器测试**：使用 Docker 容器运行 NCCL 测试
- **环境隔离**：提供干净的测试环境，避免主机环境干扰
- **特权模式支持**：完整的硬件访问权限（GPU、InfiniBand）
- **配置灵活性**：支持多种网络后端和测试参数

#### 1.2.3 多节点部署工具

**`nccl_multinode_launcher.sh`** - 传统多节点部署工具：

- **裸机多节点测试**：直接在物理机上运行分布式测试
- **节点协调**：自动处理主节点和工作节点的启动顺序
- **参数同步**：确保所有节点使用一致的测试参数
- **故障恢复**：提供重试机制和错误处理

**`k8s/deploy.sh`** - Kubernetes 多节点部署工具：

- **云原生部署**：在 Kubernetes 集群中部署 NCCL 测试
- **StatefulSet 管理**：提供稳定的 Pod 命名和有序启动
- **资源管理**：自动管理 GPU 资源分配和调度
- **扩展性**：支持大规模多节点测试部署

#### 1.2.4 辅助工具

**`nccl_python_template.py`** - Python 测试模板：

- **自定义测试脚本**：提供可定制的 NCCL 测试代码模板
- **性能监控**：详细的延迟和吞吐量统计
- **调试支持**：丰富的日志输出和错误处理
- **多进程协调**：支持分布式测试的进程同步

### 1.3 主要功能

#### 1.3.1 系统检查

- **依赖检查**: 验证 Python3、PyTorch、CUDA、NCCL、NVIDIA GPU 等必要组件
- **InfiniBand 状态检查**: 检测 IB 设备状态、Link Layer 类型、网络拓扑

#### 1.3.2 环境配置

- **自动网络类型检测**: 智能识别原生 IB 或 RoCE 环境
- **NCCL 环境变量配置**: 自动设置最优的 NCCL 参数
- **GPUDirect RDMA 支持**: 启用高性能 GPU 直接内存访问

#### 1.3.3 性能测试

- **分布式 AllReduce 测试**: 使用多 GPU 进行 NCCL 通信测试
- **延迟和吞吐量分析**: 详细的性能指标测量
- **理论传输量计算**: Ring AllReduce 算法效率分析

#### 1.3.4 报告生成

- **详细测试报告**: 包含系统信息、配置参数、测试结果
- **性能数据分析**: 延迟、数据吞吐量、理论传输量统计
- **问题诊断**: 错误和警告的统计与分析

### 1.4 系统要求

#### 1.4.1 硬件要求

- **GPU**: 一个或多个 NVIDIA GPU (支持 CUDA Compute Capability 3.5+)
  - 推荐：V100、A100、H100 等数据中心GPU
  - 最低：GTX 1080、RTX 2080 等消费级GPU
- **网络**: InfiniBand 网卡 (原生 IB 或 RoCE)
  - InfiniBand：EDR (12.5 GB/s)、HDR (25 GB/s)、NDR (50 GB/s)
  - RoCE：3.1/6.25/12.5 GB/s (25/50/100 Gbps) 以太网卡
- **内存**: 建议 16GB 以上系统内存 (大规模测试需要更多)
- **存储**: 至少 10GB 可用空间用于日志和临时文件

#### 1.4.2 软件要求

- **操作系统**: Linux (Ubuntu 18.04+/CentOS 7+/RHEL 7+)
- **Python**: Python 3.7+ (推荐 3.8-3.11)
- **PyTorch**: 1.12.0+ 支持 CUDA 的版本
  - 推荐：PyTorch 2.0+ 以获得最佳NCCL支持
- **NCCL**: 2.12.0+ (推荐 2.18.0+)
  - 通过 `python -c "import torch; print(torch.cuda.nccl.version())"` 检查版本
- **CUDA**: 11.7+ (推荐 12.0+)
  - 必须与PyTorch版本兼容
- **NVIDIA Driver**: 515.0+ (推荐 535.0+)
- **InfiniBand 工具**:
  - `infiniband-diags` (包含 `ibstat`, `perfquery`)
  - `libibverbs-dev` (包含 `ibv_devinfo`)
  - `rdma-core` (RDMA 核心库)

#### 1.4.3 依赖安装

**Ubuntu/Debian：**

```bash
# 更新包管理器
sudo apt-get update

# 安装 InfiniBand 工具和开发库
sudo apt-get install -y infiniband-diags ibverbs-utils libibverbs-dev rdma-core

# 验证 InfiniBand 安装
ibstat
ibv_devinfo

# 安装 Python 和 PyTorch (CUDA 11.8 示例)
pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 验证 PyTorch 和 NCCL
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'NCCL: {torch.cuda.nccl.version()}')"
```

**CentOS/RHEL：**

```bash
# CentOS 7/RHEL 7
sudo yum install -y infiniband-diags libibverbs-utils libibverbs-devel rdma-core-devel

# CentOS 8+/RHEL 8+ (使用 dnf)
sudo dnf install -y infiniband-diags libibverbs-utils libibverbs-devel rdma-core-devel

# 启用 InfiniBand 服务
sudo systemctl enable rdma
sudo systemctl start rdma

# 验证安装
ibstat
ibv_devinfo
```

**验证完整环境：**

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 InfiniBand 设备
ibstat | grep -E "(CA type|Number of ports|State|Physical state)"

# 检查 NCCL 测试环境
python3 -c "
import torch
print(f'GPU 数量: {torch.cuda.device_count()}')
print(f'NCCL 可用: {torch.distributed.is_nccl_available()}')
print(f'NCCL 版本: {torch.cuda.nccl.version()}')
"
```

---

## 2. 单节点测试

### 2.1 单节点测试概述

单节点NCCL测试专注于验证单台服务器内多GPU之间的通信性能，是分布式训练环境搭建的第一步。

单节点测试具有以下特点：

#### 2.1.1 NCCL通信路径选择机制

**NCCL自动探测和选择逻辑**：

NCCL在启动时会自动探测系统硬件拓扑，并根据以下优先级选择最优通信路径：

| 通信方式 | 使用条件 | 带宽/延迟特性 | 典型标识 | 优先级 |
|----------|----------|---------------|----------|--------|
| **NVLink** | GPU间通过NVLink直连 | 高带宽，低延迟 | `NVL` | 1（最高） |
| **PCIe P2P** | GPU在同一PCIe根桥或支持P2P的桥下 | 中等带宽，低延迟 | `PXB` | 2 |
| **SHM（Shared Memory）** | 同节点GPU不支持P2P | 较低带宽，CPU参与数据拷贝 | `SHM` | 3 |
| **Loopback网络** | 强制配置或配置错误时 | 通常不用于单节点 | `NET` | 4（最低） |

**重要技术细节**：

- **单节点场景**：NCCL默认**不会**使用IB/RDMA网络，即使存在IB网卡
- **通信路径**：实际的GPU间数据传输仍然是NVLink或PCIe，IB仅在特殊配置下使用
- **自动选择**：`--network auto`让NCCL根据硬件拓扑自动选择最优路径

#### 2.1.2 测试范围与应用场景

**测试范围**：

- GPU间P2P通信（PCIe、NVLink）
- 本地网络栈性能（loopback）
- NCCL算法在单机环境下的效率
- GPU内存带宽和延迟特性

**应用场景**：

- 单机多GPU训练验证
- 硬件配置优化
- 性能基线建立
- 问题隔离和诊断

**性能期望**（参考值）：

- **NVLink连接**：延迟 < 10μs，带宽 > 200 GB/s
- **PCIe连接**：根据PCIe版本和配置：
  - **PCIe 3.0 x16**：延迟 < 100μs，带宽 > 12 GB/s
  - **PCIe 4.0 x16**：延迟 < 50μs，带宽 > 20 GB/s
  - **PCIe 5.0 x16**：延迟 < 30μs，带宽 > 40 GB/s
- **InfiniBand loopback**：延迟 < 5μs，带宽 > 80 GB/s

### 2.2 硬件环境检测

在进行NCCL性能测试前，首先需要了解系统的硬件拓扑和NCCL实际使用的通信路径。

#### 2.2.1 基础硬件拓扑检测

**查看GPU拓扑结构**：

```bash
nvidia-smi topo -m
```

**典型输出解读**：

```shm
        GPU0    GPU1    GPU2    GPU3    CPU Affinity
GPU0     X      NV2     SYS     SYS     0-31
GPU1    NV2      X      SYS     SYS     0-31  
GPU2    SYS     SYS      X      NV1     32-63
GPU3    SYS     SYS     NV1      X      32-63
```

**拓扑标识含义**：

- **NV1/NV2/NV4**：NVLink连接（数字表示链路数量）
- **PXB**：PCIe桥接连接
- **SYS**：跨NUMA节点或CPU连接
- **X**：自身

**PCIe配置检测**：

```bash
# 检查PCIe版本和链路宽度
lspci -vvv | grep -A10 NVIDIA | grep -E "(LnkCap|LnkSta)"

# 检查PCIe带宽利用率
nvidia-smi dmon -s pci -c 1
```

**PCIe配置建议**：

- **高端训练**：优先选择PCIe 4.0/5.0 x16，确保每个GPU独享x16链路
- **中端配置**：PCIe 3.0 x16可满足基本需求，避免x8配置
- **多GPU系统**：检查PCIe通道分配，避免带宽竞争
- **CPU选择**：确保CPU提供足够的PCIe通道数（如Intel Xeon或AMD EPYC）

#### 2.2.2 NCCL通信路径检测

我们提供了专门的检测脚本 [gpu_topology_detector.sh](gpu_topology_detector.sh)，该脚本按照NCCL的实际优先级进行检测：

**NCCL通信路径优先级**：

1. **NVLink** (最高优先级) - GPU间专用高速互联
2. **PCIe P2P** - 通过PCIe总线点对点通信  
3. **共享内存** - 通过CPU内存中转
4. **网络传输** (最低优先级) - 主要用于跨节点通信

**使用方法**：

```bash
# 运行NCCL通信路径检测脚本
chmod +x gpu_topology_detector.sh
./gpu_topology_detector.sh
```

**脚本功能特点**：

- **按NCCL优先级检测**：严格按照 NVLink > PCIe P2P > SHM > NET 的顺序进行检测
- **详细硬件分析**：检查GPU拓扑、NVLink状态、PCIe链路、IB设备等
- **实际路径验证**：运行NCCL测试验证实际使用的通信路径
- **智能配置建议**：根据检测结果提供最优的网络后端配置建议

**典型输出示例**：

```bash
=== NCCL通信路径检测 ===
1. GPU基本信息：
0, NVIDIA A100-SXM4-80GB, 81920

3. NVLink连接详情（优先级1 - 最高）：
✓ NVLink可用：检测到活跃的NVLink连接

9. 推荐配置（按NCCL优先级）：
🚀 优先级1 - NVLink直连（最优）：
   配置：--network auto（NCCL将自动选择NVLink）
   预期性能：延迟 < 10μs，带宽 > 200 GB/s
```

#### 2.2.3 NCCL通信路径验证

**使用NCCL调试信息**：

```bash
# 启用详细调试信息
export NCCL_DEBUG=INFO
./nccl_benchmark.sh --network auto -s 100M -t 30
```

**关键输出信息**：

```bash
NCCL INFO Channel 00 : 0[0] -> 1[0] via NVL  # NVLink连接
NCCL INFO Channel 01 : 1[0] -> 2[0] via PXB  # PCIe桥接
NCCL INFO Channel 02 : 2[0] -> 3[0] via SHM  # 共享内存
```

**路径标识说明**：

- **NVL**：NVLink直连
- **PXB**：PCIe桥接
- **SHM**：共享内存（CPU参与）
- **NET**：网络通信（罕见）

### 2.3 测试配置选项

#### 2.3.1 基本语法

```bash
./nccl_benchmark.sh [选项]
```

#### 2.3.2 关键技术背景

在了解具体参数之前，需要理解以下关键概念：

**实际物理路径**：无论选择哪种网络后端，GPU间的实际数据传输仍然通过NVLink或PCIe

**单节点NET测试的特殊意义**：

although single node environment, GPU interconnection via network transmission has no meaning, but NET test still has important value：

1. **硬件验证**：
   - 验证IB/以太网卡硬件功能正常
   - 测试GPUDirect RDMA是否正确配置
   - 确认网络驱动和固件版本兼容性

2. **多节点部署准备**：
   - 在单节点环境中预先验证网络配置
   - 建立网络性能基线，便于多节点对比
   - 排查网络相关的环境变量配置问题

3. **故障排除**：
   - 当NVLink/PCIe出现问题时，验证NCCL是否能正确回退到网络传输
   - 排查NCCL在特定网络配置下的行为
   - 验证容器化环境的网络隔离配置

4. **性能对比**：
   - 量化不同传输方式的性能差异
   - 为系统优化提供数据支撑
   - 验证硬件升级的性能提升效果

**IB的特殊性**：单节点中的`--network ib`是IB网卡的loopback测试，不改变GPU间物理连接

**自动选择机制**：`--network auto`在脚本层面配置网络环境，但NCCL仍按内部逻辑选择最优路径

**NET模式下的GPU间通信机制**：

当选择NET（网络传输）时，两块GPU之间的通信路径如下：

1. **数据流向**：

   ```text
   GPU1 显存 → CPU内存 → 网络栈 → 网卡 → Loopback → 网卡 → 网络栈 → CPU内存 → GPU2 显存
   ```

2. **具体步骤**：
   - **步骤1**：GPU1通过PCIe将数据传输到CPU内存（Host Memory）
   - **步骤2**：CPU将数据通过网络栈发送到网卡
   - **步骤3**：网卡进行本地回环（Loopback）处理
   - **步骤4**：数据通过网络栈返回到CPU内存
   - **步骤5**：CPU通过PCIe将数据传输到GPU2内存

3. **技术细节**：
   - **InfiniBand模式**：使用IB网卡的loopback功能，支持GPUDirect RDMA（如果配置正确）
   - **以太网模式**：使用以太网卡的loopback，通过TCP/IP协议栈
   - **Socket模式**：纯软件实现，通过TCP Socket进行本地通信

4. **GPUDirect RDMA的作用**：
   - **启用时**：GPU内存可以直接与网卡通信，减少CPU内存拷贝
   - **禁用时**：必须经过CPU内存中转，增加延迟和CPU负载

5. **性能影响因素**：
   - **网卡类型**：IB > 高速以太网 > 标准以太网
   - **GPUDirect支持**：RDMA > 标准DMA
   - **网络栈开销**：协议处理、中断处理、内存拷贝
   - **Loopback延迟**：网卡内部处理时间

**通信路径对比**：

| 通信模式 | 数据路径 | 延迟特征 | 带宽特征 | 适用场景 |
|----------|----------|----------|----------|----------|
| **NVLink** | `GPU1 ↔ GPU2` | < 10μs | > 200 GB/s | 生产环境首选 |
| **PCIe P2P** | `GPU1 ↔ PCIe ↔ GPU2` | 10-50μs | 32-63 GB/s | 无NVLink时的最佳选择 |
| **共享内存** | `GPU1 ↔ CPU内存 ↔ GPU2` | 50-200μs | 系统内存带宽 | 兼容性保障 |
| **IB网络** | `GPU1 ↔ CPU ↔ IB网卡 ↔ CPU ↔ GPU2` | > 200μs | 8-32 GB/s | IB功能验证 |
| **以太网** | `GPU1 ↔ CPU ↔ 以太网卡 ↔ CPU ↔ GPU2` | > 500μs | 4-16 GB/s | 兼容性测试 |
| **Socket** | `GPU1 ↔ CPU ↔ TCP栈 ↔ CPU ↔ GPU2` | > 1000μs | < 4 GB/s | 调试专用 |

**关键理解**：

- NET模式下，GPU间通信**不是直连**，而是通过网络栈的本地回环
- 这种通信方式主要用于**功能验证**而非性能优化
- 在实际应用中，NCCL会优先选择更高效的直连路径

**双重选择逻辑**：脚本选择网络配置策略，NCCL选择实际通信路径

#### 2.3.3 单节点专用参数

**数据大小配置**：

- `-s, --size SIZE`: 测试数据大小
  - `1M` = 1MB (262144 元素) - 延迟测试
  - `10M` = 10MB (2621440 元素) - 中等负载
  - `100M` = 100MB (26214400 元素) - 带宽测试  
  - `1G` = 1GB (268435456 元素) - 最大吞吐量
  - `10G` = 10GB (2684354560 元素) - 极限吞吐量测试

**测试时长**：

- `-t, --time SECONDS`: 测试持续时间 [默认: 30秒]
  - 建议延迟测试：10-30秒
  - 建议吞吐量测试：60-120秒

**网络后端选择**（NCCL通信路径控制）：

- `--network nvlink`: **强制NVLink路径**
  - 仅在确认存在NVLink连接时使用
  - 如果硬件不支持，测试将失败
  - 用于验证NVLink专用性能
  
- `--network auto`: **自动路径选择**（推荐）
  - **脚本层面**：自动检测IB设备，优先配置IB环境变量
  - **NCCL层面**：仍按NVLink > PCIe P2P > SHM > NET优先级选择实际通信路径
  - **实际效果**：GPU间通信仍遵循NCCL内部优先级，网络配置主要影响跨节点通信
  - 适用于大多数性能测试场景
  
- `--network ib`: **强制IB Loopback**（特殊测试场景）
  - 通过IB网卡进行本地回环通信
  - **重要说明**：这不是GPU间直连，而是测试IB网卡本身的性能
  - **应用场景**：
    - 验证IB网卡硬件功能（在多节点部署前）
    - 测试GPUDirect RDMA功能
    - 排查IB网卡驱动问题
    - 建立IB网络性能基线
  - **性能特点**：通常比NVLink/PCIe慢，但验证网络栈功能
  - 需要配合环境变量：`NCCL_IB_DISABLE=0`
  
- `--network ethernet`: **以太网Loopback**（兼容性测试）
  - 通过以太网卡进行本地回环
  - **应用场景**：
    - 验证NCCL在纯以太网环境的兼容性
    - 测试容器化环境的网络配置
    - 排查网络栈问题
    - 模拟低带宽网络环境
  - **性能特点**：性能较低，主要用于功能验证
  - 所有系统都支持
  
- `--network socket`: **Socket通信**（调试专用）
  - 通过TCP Socket进行通信
  - **应用场景**：
    - NCCL故障排除和调试
    - 验证最基础的通信功能
    - 测试防火墙和网络策略
    - 教学和演示用途
  - **性能特点**：性能最低，但兼容性最好

**调试选项**：

- `--check-only`: 仅检查环境，不运行测试
- `--env-only`: 显示环境变量配置
- `-q, --quiet`: 静默模式

### 2.4 单节点测试实践

#### 2.4.1 快速验证测试

```bash
# 基础环境检查
./nccl_benchmark.sh --check-only

# 快速性能测试（1MB数据，30秒）
./nccl_benchmark.sh

# 查看环境配置
./nccl_benchmark.sh --env-only
```

#### 2.4.2 NCCL环境变量配置

**强制指定通信路径**：

```bash
# 1. 强制使用InfiniBand（非典型，仅用于测试IB网卡）
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2    # 启用GPUDirect RDMA
export NCCL_IB_GID_INDEX=0     # 原生IB使用GID 0，RoCE使用GID 3
export NCCL_IB_TC=136          # Traffic Class
export NCCL_IB_SL=0            # Service Level
export NCCL_IB_TIMEOUT=22      # IB超时设置
export NCCL_IB_RETRY_CNT=7     # 重试次数
./nccl_benchmark.sh --network ib -s 100M -t 30

# 2. 禁用特定通信方式
export NCCL_P2P_DISABLE=1      # 禁用PCIe P2P
export NCCL_IB_DISABLE=1       # 禁用InfiniBand
export NCCL_NET_GDR_LEVEL=0    # 禁用GPUDirect RDMA

# 3. 调试和性能调优
export NCCL_DEBUG=INFO         # 详细调试信息
export NCCL_DEBUG_SUBSYS=INIT,NET  # 初始化和网络子系统调试
export NCCL_ALGO=Ring          # 强制使用Ring算法
export NCCL_PROTO=Simple       # 使用Simple协议

# 4. 清理环境变量
unset NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_NET_GDR_LEVEL
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_ALGO NCCL_PROTO
unset NCCL_IB_GID_INDEX NCCL_IB_TC NCCL_IB_SL NCCL_IB_TIMEOUT NCCL_IB_RETRY_CNT
```

**重要说明**：

- **默认行为**：NCCL会自动选择最优路径，通常无需手动配置
- **IB强制使用**：在单节点中强制使用IB是非典型配置，主要用于测试IB网卡性能
- **性能影响**：强制配置可能降低性能，建议仅用于调试和特殊测试

#### 2.4.3 延迟优化测试

```bash
# 小数据包延迟测试（NVLink）
./nccl_benchmark.sh --network nvlink -s 1M -t 30

# InfiniBand延迟测试
./nccl_benchmark.sh --network ib -s 1M -t 30

# 对比不同网络后端的延迟
for backend in nvlink ib ethernet socket; do
    echo "=== 测试 $backend 延迟 ==="
    ./nccl_benchmark.sh --network $backend -s 1M -t 10 -q
done
```

#### 2.4.4 带宽性能测试

```bash
# 大数据吞吐量测试（NVLink）
./nccl_benchmark.sh --network nvlink -s 1G -t 120

# InfiniBand带宽测试
./nccl_benchmark.sh --network ib -s 1G -t 120

# 渐进式数据大小测试
for size in 1M 10M 100M 1G 10G; do
    echo "=== 测试数据大小: $size ==="
    ./nccl_benchmark.sh --network auto -s $size -t 60
done
```

#### 2.4.5 性能基准建立

```bash
# 完整性能基准测试套件
echo "开始单节点NCCL性能基准测试..."

# 1. 延迟基准（小数据包）
echo "1. 延迟基准测试"
./nccl_benchmark.sh --network nvlink -s 1M -t 30 > baseline_latency_nvlink.log
./nccl_benchmark.sh --network ib -s 1M -t 30 > baseline_latency_ib.log

# 2. 带宽基准（大数据包）
echo "2. 带宽基准测试"  
./nccl_benchmark.sh --network nvlink -s 1G -t 120 > baseline_bandwidth_nvlink.log
./nccl_benchmark.sh --network ib -s 1G -t 120 > baseline_bandwidth_ib.log

# 3. 混合负载测试
echo "3. 混合负载测试"
./nccl_benchmark.sh --network auto -s 100M -t 300 > baseline_mixed_load.log

echo "基准测试完成，结果保存在 baseline_*.log 文件中"
```

### 2.5 单节点性能分析

#### 2.5.1 性能指标解读

**延迟指标**：

- **< 10μs**: 优秀（NVLink直连）
- **10-50μs**: 良好（PCIe或高速IB）
- **50-200μs**: 可接受（标准IB或以太网）
- **> 200μs**: 需要优化

**带宽指标**：

- **NVLink**: 期望 > 200 GB/s（双向）
- **PCIe连接**：
  - **PCIe 3.0 x16**: 期望 > 12 GB/s（理论15.75 GB/s）
  - **PCIe 4.0 x16**: 期望 > 20 GB/s（理论31.5 GB/s）
  - **PCIe 5.0 x16**: 期望 > 40 GB/s（理论63 GB/s）
- **InfiniBand（单节点loopback测试）**：
  - **EDR (100Gb/s)**: 期望 > 8 GB/s（受loopback限制）
  - **HDR (200Gb/s)**: 期望 > 16 GB/s（受loopback限制）
  - **NDR (400Gb/s)**: 期望 > 32 GB/s（受loopback限制）
  - **注意**: 单节点IB测试性能远低于理论值，主要用于功能验证
- **以太网（单节点loopback测试）**：
  - **100GbE**: 期望 > 4 GB/s（受loopback和协议栈限制）
  - **200GbE**: 期望 > 8 GB/s（受loopback和协议栈限制）
  - **400GbE**: 期望 > 16 GB/s（受loopback和协议栈限制）
  - **注意**: 单节点以太网测试主要用于兼容性验证，性能不具参考价值

**效率指标**：

- **算法效率**: 实际带宽 / 理论带宽 > 80%
- **网络利用率**: NCCL带宽 / 网络带宽 > 70%
- **GPU利用率**: 通信期间GPU使用率 > 90%

#### 2.5.2 常见性能问题

**延迟过高**：

- 检查GPU拓扑：`nvidia-smi topo -m`
- 验证NVLink状态：`nvidia-smi nvlink -s`
- 确认NCCL算法选择：查看日志中的算法信息

**带宽不足**：

- 检查PCIe链路状态：`lspci -vvv | grep -A5 NVIDIA`
- 验证内存带宽：运行GPU内存带宽测试
- 确认NCCL网络后端：检查环境变量设置

**性能不稳定**：

- 检查系统负载：`htop`, `nvidia-smi`
- 验证温度控制：避免GPU过热降频
- 确认电源管理：设置高性能模式

---

## 3. 容器化测试

### 3.1 容器化测试概述

`nccl_container_manager.sh` 提供了基于 Docker 容器的 NCCL 测试解决方案，具有以下优势：

#### 3.1.1 容器化的优势

**环境隔离**：

- 避免主机环境依赖冲突
- 提供一致的测试环境
- 简化部署和维护

**硬件访问**：

- 特权模式提供完整的硬件访问权限
- 支持 GPU、InfiniBand、NVLink 等设备
- Host Network 模式避免网络性能损失

**配置灵活性**：

- 支持多种网络后端配置
- 可定制的测试参数
- 交互模式用于调试

#### 3.1.2 技术特性

**网络模式**：

- **Host Network**：直接使用主机网络，无 Docker 网络层开销
- **特权模式**：完整的设备访问权限，无需手动挂载设备

**GPU 支持**：

- NVIDIA Container Toolkit 集成
- 支持指定 GPU 数量和 ID
- 完整的 CUDA 和 NCCL 支持

**存储挂载**：

- 自动挂载必要的系统目录
- 日志文件持久化到主机
- 配置文件共享

### 3.2 前置条件

#### 3.2.1 软件要求

```bash
# 1. 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 2. 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 3. 验证 GPU 容器支持
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

#### 3.2.2 构建测试镜像

```bash
# 构建 NCCL 测试镜像
docker build -t nccl-test:latest .

# 验证镜像构建成功
docker images | grep nccl-test
```

### 3.3 基本使用

#### 3.3.1 快速测试

```bash
# 基础测试（使用所有 GPU）
./nccl_container_manager.sh --gpus all --size 100M --time 60

# Dry-run 模式（检查配置但不执行测试）
./nccl_container_manager.sh --dry-run --gpus all --size 100M

# 交互模式（进入容器调试）
./nccl_container_manager.sh --interactive
```

#### 3.3.2 网络后端测试

```bash
# 自动选择最佳网络后端
./nccl_container_manager.sh --gpus 4 --network auto --size 1G

# 强制使用 NVLink（单节点多GPU）
./nccl_container_manager.sh --gpus 4 --network nvlink --size 500M

# 强制使用 InfiniBand
./nccl_container_manager.sh --gpus 2 --network ib --size 100M

# 使用以太网
./nccl_container_manager.sh --gpus 2 --network ethernet --size 50M

# 使用共享内存
./nccl_container_manager.sh --gpus 2 --network shm --size 10M
```

#### 3.3.3 GPU 配置选项

```bash
# 使用所有可用 GPU
./nccl_container_manager.sh --gpus all

# 使用指定数量的 GPU
./nccl_container_manager.sh --gpus 4

# 使用特定 GPU ID
./nccl_container_manager.sh --gpus 0,1,2,3

# 使用单个 GPU（测试基础功能）
./nccl_container_manager.sh --gpus 1
```

### 3.4 高级配置

#### 3.4.1 自定义容器配置

```bash
# 自定义容器名称和镜像
./nccl_container_manager.sh \
    --container-name my-nccl-test \
    --image-name my-nccl:v1.0 \
    --gpus 4 --size 1G

# 保留容器用于调试
./nccl_container_manager.sh \
    --no-cleanup \
    --gpus 2 --size 100M

# 启用详细日志
./nccl_container_manager.sh \
    --log-level DEBUG \
    --gpus 4 --size 500M
```

#### 3.4.2 性能测试套件

```bash
# 延迟测试套件
echo "=== 延迟测试套件 ==="
for backend in nvlink ib ethernet; do
    echo "测试 $backend 延迟..."
    ./nccl_container_manager.sh \
        --gpus 2 --network $backend \
        --size 1M --time 30
done

# 带宽测试套件
echo "=== 带宽测试套件 ==="
for size in 100M 500M 1G; do
    echo "测试数据大小: $size"
    ./nccl_container_manager.sh \
        --gpus 4 --network auto \
        --size $size --time 120
done

# 扩展性测试套件
echo "=== 扩展性测试套件 ==="
for gpus in 1 2 4 8; do
    echo "测试 GPU 数量: $gpus"
    ./nccl_container_manager.sh \
        --gpus $gpus --network nvlink \
        --size 100M --time 60
done
```

### 3.5 故障排除

#### 3.5.1 常见问题

**Docker 权限问题**：

```bash
# 将用户添加到 docker 组
sudo usermod -aG docker $USER
# 重新登录或执行
newgrp docker
```

**NVIDIA Container Toolkit 问题**：

```bash
# 检查 NVIDIA 运行时
docker info | grep nvidia

# 测试 GPU 访问
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

**镜像构建失败**：

```bash
# 检查 Dockerfile 是否存在
ls -la Dockerfile

# 手动构建并查看详细输出
docker build -t nccl-test:latest . --no-cache --progress=plain
```

#### 3.5.2 调试技巧

**交互模式调试**：

```bash
# 进入容器进行手动调试
./nccl_container_manager.sh --interactive

# 在容器内手动运行测试
cd /workspace/nccl_test
./nccl_benchmark.sh --network auto --size 100M
```

**日志分析**：

```bash
# 查看容器日志
docker logs nccl-test

# 查看详细的 NCCL 日志
./nccl_container_manager.sh --log-level DEBUG --gpus 2 --size 100M
```

### 3.6 最佳实践

#### 3.6.1 性能优化

- **使用 Host Network**：避免 Docker 网络层开销
- **特权模式**：确保完整的硬件访问权限
- **GPU 亲和性**：在多 GPU 系统中合理分配 GPU
- **内存管理**：确保容器有足够的内存用于大规模测试

#### 3.6.2 安全考虑

- **特权模式风险**：仅在受信任的环境中使用特权模式
- **网络隔离**：在生产环境中考虑网络安全策略
- **资源限制**：根据需要设置 CPU 和内存限制

#### 3.6.3 生产部署建议

- **镜像管理**：使用私有镜像仓库管理测试镜像
- **配置管理**：使用配置文件管理复杂的测试参数
- **监控集成**：集成监控系统跟踪测试性能
- **自动化**：结合 CI/CD 流水线进行自动化测试

---

## 4. 多节点测试

### 4.1 多节点测试概述

多节点 NCCL 测试用于验证跨节点的分布式通信性能，是大规模训练环境的重要验证手段。本工具提供两种多节点测试方案：

#### 4.1.1 原生多节点部署

使用 `nccl_multinode_launcher.sh` 脚本进行传统的裸机多节点部署：

- **适用场景**：传统 HPC 环境、裸机部署
- **优势**：配置简单、直接控制
- **劣势**：手动管理、扩展性有限

#### 4.1.2 Kubernetes 多节点部署（推荐）

使用 `k8s/deploy.sh` 脚本进行现代化的容器编排部署：

- **适用场景**：云原生环境、生产环境
- **优势**：自动调度、资源管理、故障恢复、易扩展
- **劣势**：需要 Kubernetes 集群

**推荐使用 Kubernetes 方案**，特别是在生产环境和云环境中。

### 4.2 快速开始

#### 4.2.1 方案选择

根据你的环境选择合适的部署方案：

**Kubernetes 方案（推荐）**：

```bash
# 进入 Kubernetes 配置目录
cd k8s/

# 快速部署（使用默认配置）
./deploy.sh deploy

# 查看部署状态
./deploy.sh status

# 查看测试日志
./deploy.sh logs

# 清理资源
./deploy.sh cleanup
```

**原生方案**：

```bash
# 节点1 (主节点 - 192.168.1.100)
./nccl_multinode_launcher.sh 0 192.168.1.100

# 节点2 (工作节点 - 192.168.1.101)  
./nccl_multinode_launcher.sh 1 192.168.1.100
```

#### 4.2.2 Kubernetes 自定义配置

```bash
# 自定义 GPU 数量和测试参数
./deploy.sh deploy --gpus 4 --test-size 1G --test-duration 120

# 指定命名空间和作业名称
./deploy.sh deploy --namespace nccl-test --job-name my-nccl-test

# 指定网络后端
./deploy.sh deploy --network-backend ib --test-size 500M

# 查看所有可用参数
./deploy.sh --help
```

#### 4.2.3 原生部署基础配置

**基础2节点测试**：

```bash
# 节点1 (主节点 - 192.168.1.100)
./nccl_multinode_launcher.sh 0 192.168.1.100

# 节点2 (工作节点 - 192.168.1.101)  
./nccl_multinode_launcher.sh 1 192.168.1.100
```

**IP 地址说明**:

- `192.168.1.100` 是示例主节点 IP 地址，**实际使用时需要替换为真实的主节点 IP**
- 当使用 IPoIB 时，必须使用 InfiniBand 网卡 (如 `ib0`) 上配置的 IP 地址
- 当使用以太网时，使用以太网网卡 (如 `eth0`) 上的 IP 地址
- 所有节点都需要指定相同的主节点 IP 地址进行协调

**自定义配置测试**：

```bash
# 4节点8GPU高性能测试
./nccl_multinode_launcher.sh 0 192.168.1.100 -w 8 -n 2 -s 100M -t 120

# 指定网络后端类型
./nccl_multinode_launcher.sh 0 192.168.1.100 --network ib -w 4 -n 2
```

**网络类型指定**：

`nccl_multinode_launcher.sh` 脚本支持通过 `--network` 参数指定网络后端：

```bash
# 使用 InfiniBand (推荐，默认)
./nccl_multinode_launcher.sh 0 192.168.1.100 --network ib

# 使用以太网
./nccl_multinode_launcher.sh 0 192.168.1.100 --network ethernet

# 使用 Socket (调试用)
./nccl_multinode_launcher.sh 0 192.168.1.100 --network socket
```

**网络后端说明**:

| 网络类型 | 参数值 | 性能特征 | 适用场景 |
|----------|--------|----------|----------|
| **InfiniBand** | `ib` | 延迟 < 1μs, 带宽最高 | **生产环境推荐** |
| 以太网 | `ethernet` | 延迟 ~10μs, 标准带宽 | 无 IB 环境 |
| Socket | `socket` | 延迟较高, 兼容性最好 | 调试和测试 |

**重要**:

- 默认使用 `ib` 网络后端，提供最佳性能
- 所有节点必须使用相同的网络后端配置
- 网络后端会传递给底层的 `nccl_benchmark.sh` 脚本

### 4.3 Kubernetes 多节点部署（推荐）

#### 4.3.1 Kubernetes 方案概述

Kubernetes 方案提供了现代化的容器编排多节点部署，具有以下优势：

- **自动化部署**: 一键部署多节点测试环境
- **资源管理**: 自动调度和资源分配
- **故障恢复**: 自动重启失败的 Pod
- **配置管理**: 统一的配置管理和版本控制
- **监控集成**: 与 Kubernetes 生态系统无缝集成

#### 4.3.2 快速开始

**部署测试环境**：

```bash
# 进入 Kubernetes 目录
cd k8s/

# 一键部署多节点测试
./deploy.sh

# 查看部署状态
kubectl get pods -l app=nccl-test

# 查看测试日志
kubectl logs -l app=nccl-test -f

# 清理资源
kubectl delete -f nccl-multinode-job.yaml
kubectl delete -f nccl-service.yaml
kubectl delete configmap nccl-config
```

#### 4.3.3 自定义配置

**修改测试参数**：

```bash
# 编辑配置文件
vim k8s/nccl-configmap.yaml

# 示例配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: nccl-config
data:
  WORLD_SIZE: "4"
  NPROC_PER_NODE: "2"
  TEST_SIZE: "100M"
  TEST_TIME: "120"
  NETWORK_TYPE: "ib"
```

**修改节点数量**：

```bash
# 编辑 Job 配置
vim k8s/nccl-multinode-job.yaml

# 修改 replicas 字段
spec:
  parallelism: 2    # 并行运行的 Pod 数量
  completions: 2    # 总共需要完成的 Pod 数量
```

#### 3.3.4 高级配置

**GPU 资源配置**：

```yaml
# 在 nccl-multinode-job.yaml 中配置
resources:
  limits:
    nvidia.com/gpu: 2    # 每个 Pod 使用的 GPU 数量
  requests:
    nvidia.com/gpu: 2
```

**网络配置**：

```yaml
# 使用 InfiniBand 网络
spec:
  template:
    spec:
      hostNetwork: true    # 使用主机网络
      dnsPolicy: ClusterFirstWithHostNet
```

**存储配置**：

```yaml
# 挂载共享存储
volumeMounts:
- name: shared-data
  mountPath: /shared
volumes:
- name: shared-data
  nfs:
    server: nfs-server.example.com
    path: /shared/nccl-data
```

#### 3.3.5 监控和调试

**查看集群状态**：

```bash
# 查看节点状态
kubectl get nodes -o wide

# 查看 GPU 资源
kubectl describe nodes | grep nvidia.com/gpu

# 查看网络插件状态
kubectl get pods -n kube-system | grep network
```

**调试测试问题**：

```bash
# 查看 Pod 详细信息
kubectl describe pod <pod-name>

# 进入 Pod 调试
kubectl exec -it <pod-name> -- /bin/bash

# 查看事件日志
kubectl get events --sort-by=.metadata.creationTimestamp
```

#### 3.3.6 性能优化

**节点亲和性配置**：

```yaml
# 确保 Pod 分布在不同节点
spec:
  template:
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - nccl-test
            topologyKey: kubernetes.io/hostname
```

**优先级和抢占**：

```yaml
# 设置高优先级
spec:
  template:
    spec:
      priorityClassName: high-priority
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

### 3.4 脚本参数详解

#### 3.4.1 完整参数列表

`nccl_multinode_launcher.sh` 脚本支持以下参数：

```bash
./nccl_multinode_launcher.sh <node_rank> <master_addr> [选项]
```

**必需参数**:

- `node_rank`: 节点编号 (0=主节点, 1,2,3...=工作节点)
- `master_addr`: 主节点 IP 地址

**可选参数**:

| 参数 | 长格式 | 默认值 | 说明 |
|------|--------|--------|------|
| `-w` | `--world-size` | 4 | 总 GPU 数量 |
| `-n` | `--nproc-per-node` | 2 | 每节点 GPU 数 |
| `-p` | `--master-port` | 29500 | 主节点通信端口 |
| `--network` | `--network` | ib | 网络后端 (ib/ethernet/socket) |
| `-s` | `--size` | 50M | 测试数据大小 |
| `-t` | `--time` | 90 | 测试时长 (秒) |
| `-h` | `--help` | - | 显示帮助信息 |

#### 3.4.2 参数使用示例

```bash
# 查看帮助信息
./nccl_multinode_launcher.sh --help

# 基础测试 (使用默认参数)
./nccl_multinode_launcher.sh 0 192.168.1.100

# 完整参数配置
./nccl_multinode_launcher.sh 0 192.168.1.100 \
  --world-size 8 \
  --nproc-per-node 2 \
  --master-port 29500 \
  --network ib \
  --size 100M \
  --time 120

# 以太网环境测试
./nccl_multinode_launcher.sh 0 192.168.1.100 \
  -w 4 -n 2 --network ethernet

# 调试模式 (Socket 网络)
./nccl_multinode_launcher.sh 0 192.168.1.100 \
  -w 2 -n 1 --network socket -s 10M -t 30
```

#### 3.4.3 参数验证规则

脚本会自动验证参数的有效性：

- `node_rank` 必须是非负整数且小于总节点数
- `world_size` 必须是大于等于 2 的整数
- `nproc_per_node` 必须是正整数
- `world_size` 必须能被 `nproc_per_node` 整除
- `master_port` 必须是有效的端口号
- `network` 必须是支持的网络类型

### 3.5 测试步骤

#### 3.5.1 环境准备

- ✅ 确保所有节点网络连通
- ✅ 同步所有节点时间
- ✅ 开放防火墙端口 (29500)
- ✅ 验证GPU和NCCL环境

#### 3.5.2 启动测试

1. **先启动主节点** (node_rank=0)
2. **再启动工作节点** (node_rank=1,2,3...)
3. **使用相同参数** 在所有节点

#### 3.5.3 监控测试

```bash
# 监控网络流量
watch -n 1 'ibstat | grep -A 5 "Port 1"'

# 监控GPU使用
watch -n 1 nvidia-smi
```

### 3.6 常用配置

| 场景 | 节点数 | GPU数 | 网络类型 | 命令示例 |
|------|--------|-------|----------|----------|
| 小规模验证 | 2 | 4 | IB (默认) | `./nccl_multinode_launcher.sh 0 192.168.1.100 -w 4 -n 2` |
| 中等规模 | 4 | 8 | IB | `./nccl_multinode_launcher.sh 0 192.168.1.100 -w 8 -n 2 -s 100M --network ib` |
| 大规模测试 | 8 | 16 | IB | `./nccl_multinode_launcher.sh 0 192.168.1.100 -w 16 -n 2 -s 500M -t 300 --network ib` |
| 以太网环境 | 2 | 4 | Ethernet | `./nccl_multinode_launcher.sh 0 192.168.1.100 -w 4 -n 2 --network ethernet` |
| 调试测试 | 2 | 2 | Socket | `./nccl_multinode_launcher.sh 0 192.168.1.100 -w 2 -n 1 --network socket` |

**参数说明**:

- `-w, --world-size`: 总 GPU 数量
- `-n, --nproc-per-node`: 每节点 GPU 数
- `-s, --size`: 测试数据大小 (如 10M, 100M, 1G)
- `-t, --time`: 测试时长 (秒)
- `--network`: 网络后端类型 (ib/ethernet/socket)

### 3.7 网络配置要求

#### 3.7.1 IP over InfiniBand (IPoIB) 地址要求

**重要**: 当使用 IPoIB 时，指定的 IP 地址**必须是 InfiniBand 网卡上配置的地址**，而非以太网网卡地址。

```bash
# 1. 查看 IB 网卡接口
ip addr show | grep ib

# 典型输出示例：
# ib0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 2044 qdisc mq state UP group default qlen 256
#     inet 192.168.1.100/24 brd 192.168.1.255 scope global ib0

# 2. 验证 IB 接口状态
ibstat ib0

# 3. 检查 IPoIB 模块加载
lsmod | grep ib_ipoib
```

#### 3.7.2 网络接口配置示例

```bash
# 配置 IPoIB 接口 (Ubuntu/Debian)
sudo ip addr add 192.168.1.100/24 dev ib0
sudo ip link set ib0 up

# 配置 IPoIB 接口 (CentOS/RHEL)
# 编辑 /etc/sysconfig/network-scripts/ifcfg-ib0
cat > /etc/sysconfig/network-scripts/ifcfg-ib0 << EOF
DEVICE=ib0
BOOTPROTO=static
IPADDR=192.168.1.100
NETMASK=255.255.255.0
ONBOOT=yes
TYPE=InfiniBand
CONNECTED_MODE=yes
EOF

# 重启网络服务
sudo systemctl restart network
```

#### 3.7.3 地址类型对比

| 网络类型 | 接口名称 | 地址示例 | MTU | 用途 |
|----------|----------|----------|-----|------|
| 以太网 | eth0, ens33 | 192.168.1.100 | 1500 | 管理网络 |
| IPoIB | ib0, ib1 | 192.168.1.100 | 2044/65520 | 高性能数据传输 |
| RoCE | eth0 (IB功能) | 192.168.1.100 | 1500 | 以太网上的IB |

#### 3.7.4 网络验证命令

```bash
# 验证 IB 接口配置
ip addr show ib0

# 检查 IB 设备状态
ibstat

# 测试 IPoIB 连通性
ping -I ib0 192.168.1.101

# 验证 IB 链路状态
ibstatus
```

#### 3.7.5 IPoIB 与原生 IB 的共存

**重要概念**: IPoIB 和原生 InfiniBand 可以在同一硬件上共存，NCCL 会自动选择最优通信方式。

##### 3.7.5.1 共存机制原理

```bash
# 同一个 IB 设备的双重身份
# 1. 原生 IB 设备 (用于 RDMA/Verbs)
ibv_devinfo mlx5_0

# 2. IPoIB 网络接口 (用于 TCP/IP)
ip addr show ib0

# 两者使用相同的物理硬件，但协议栈不同
```

##### 3.7.5.2 NCCL 网络选择优先级

NCCL 按以下优先级自动选择通信方式：

| 优先级 | 通信方式 | 性能特征 | 使用场景 |
|--------|----------|----------|----------|
| **1 (最高)** | **原生 IB Verbs** | 延迟 < 1μs, 零拷贝 | **推荐用于 NCCL** |
| 2 | IPoIB (Connected Mode) | 延迟 ~2μs, 高带宽 | 兼容性要求高时 |
| 3 | IPoIB (Datagram Mode) | 延迟 ~5μs, 中等带宽 | 网络配置简单时 |
| 4 (最低) | 以太网 TCP | 延迟 ~10μs, 标准带宽 | 回退选项 |

##### 3.7.5.3 强制使用原生 IB 的配置

```bash
# 方法1: 通过环境变量强制使用原生 IB
export NCCL_IB_DISABLE=0          # 启用 IB 支持
export NCCL_NET_GDR_LEVEL=2       # 启用 GPUDirect RDMA
export NCCL_IB_HCA=mlx5_0          # 指定 IB 设备
export NCCL_SOCKET_IFNAME=""       # 禁用 Socket 接口

# 方法2: 通过脚本参数和环境变量组合
./nccl_benchmark.sh --network ib -s 100M

# 验证 NCCL 使用的网络类型
export NCCL_DEBUG=INFO
# 查看日志中的 "NET/" 信息确认使用原生 IB
```

##### 3.7.5.4 网络类型验证方法

```bash
# 运行测试并检查网络选择
./nccl_benchmark.sh -s 10M 2>&1 | grep -E "NET/|Using|Selected"

# 期望看到类似输出：
# [nccl] INFO NET/IB : Using [0]mlx5_0:1/IB [RO]; OOB ib0:192.168.1.100<->192.168.1.101
# 这表明使用了原生 IB (NET/IB) 而非 TCP (NET/Socket)
```

##### 3.7.5.5 性能对比测试

```bash
# 测试1:# 强制使用原生 IB
export NCCL_SOCKET_IFNAME=""
./nccl_benchmark.sh -s 100M -t 60

# 测试2: 强制使用 IPoIB (TCP)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib0
./nccl_benchmark.sh -s 100M -t 60

# 比较两次测试的延迟和带宽差异
```

##### 3.7.5.6 最佳实践建议

1. **默认配置**: 让 NCCL 自动选择，通常会选择原生 IB
2. **性能优先**: 明确配置使用原生 IB Verbs
3. **兼容性优先**: 在网络配置复杂时可考虑 IPoIB
4. **监控验证**: 通过日志确认实际使用的网络类型

```bash
# 推荐的生产环境配置
export NCCL_IB_DISABLE=0          # 启用原生 IB
export NCCL_NET_GDR_LEVEL=2       # 启用 GPUDirect
export NCCL_IB_HCA=mlx5_0          # 指定设备
export NCCL_DEBUG=INFO             # 启用日志验证
```

### 3.8 多节点故障排除

#### 3.8.1 连接问题

```bash
# 检查网络连通性 (使用正确的接口)
ping -I ib0 192.168.1.100

# 检查端口开放
telnet 192.168.1.100 29500

# 检查防火墙
sudo ufw status

# 验证 IB 设备状态
ibstat
ibv_devinfo
```

#### 3.8.2 性能问题

```bash
# 检查网络类型
grep "NET/" /tmp/nccl_test_output.log

# 检查GPU利用率
nvidia-smi

# 检查IB状态
ibstat
```

### 3.9 性能基准

#### 3.9.1 InfiniBand (100Gbps / 12.5 GB/s)

- 2节点4GPU: 10-11.25 GB/s (80-90 Gbps)
- 4节点8GPU: 8.75-10.6 GB/s (70-85 Gbps)
- 8节点16GPU: 7.5-10 GB/s (60-80 Gbps)

#### 3.9.2 以太网 (25Gbps / 3.125 GB/s)

- 2节点4GPU: 2.5-2.9 GB/s (20-23 Gbps)
- 4节点8GPU: 2.25-2.75 GB/s (18-22 Gbps)
- 8节点16GPU: 1.9-2.5 GB/s (15-20 Gbps)

### 3.10 相关文件

- `nccl_benchmark.sh` - 主测试脚本
- `nccl_multinode_launcher.sh` - 多节点启动脚本
- `/tmp/nccl_test_output.log` - 测试输出日志

### 3.11 多节点最佳实践

1. **测试前**: 先运行单节点测试验证环境
2. **启动顺序**: 主节点 → 工作节点 (间隔10-15秒)
3. **参数一致**: 所有节点使用完全相同的参数
4. **监控**: 同时监控网络和GPU使用情况
5. **日志**: 保存所有节点的测试日志用于分析

---

## 5. 关键技术说明

### 5.1 NCCL AllReduce 算法原理

#### 5.1.1 Ring AllReduce 算法

NCCL 默认使用 Ring AllReduce 算法，该算法具有以下特点：

```python
# Ring AllReduce 算法伪代码
def ring_allreduce(data, rank, world_size):
    """
    Ring AllReduce 算法实现
    
    参数:
        data: 本地数据张量
        rank: 当前进程的排名 (0 到 world_size-1)
        world_size: 总进程数
    
    算法步骤:
        1. Reduce-Scatter 阶段: 将数据分块，每个节点负责一个块的归约
        2. AllGather 阶段: 将归约结果广播到所有节点
    """
    
    # 第一阶段: Reduce-Scatter
    # 每个节点将其数据块发送给下一个节点，接收上一个节点的数据块
    for step in range(world_size - 1):
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1 + world_size) % world_size
        
        # 发送数据块到下一个节点，从上一个节点接收数据块
        # 并进行归约操作 (求和)
        
    # 第二阶段: AllGather  
    # 将归约后的数据块广播到所有节点
    for step in range(world_size - 1):
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1 + world_size) % world_size
        
        # 发送完整的数据块到下一个节点
```

**算法复杂度分析**:

- **通信复杂度**: O(N) 其中 N 是数据大小
- **时间复杂度**: 2 × (N-1)/N × (数据传输时间)
- **带宽利用率**: 接近 100% (理论最优)

#### 5.1.2 理论传输量计算

```python
def calculate_theoretical_transfer(data_size_mb, world_size):
    """
    计算 Ring AllReduce 的理论传输量
    
    公式: 2 × (N-1)/N × data_size
    
    解释:
        - 第一个 (N-1)/N: Reduce-Scatter 阶段的传输比例
        - 第二个 (N-1)/N: AllGather 阶段的传输比例  
        - 系数 2: 两个阶段的总和
    """
    transfer_ratio = 2 * (world_size - 1) / world_size
    theoretical_transfer_mb = transfer_ratio * data_size_mb
    
    return theoretical_transfer_mb

# 示例计算
# 4个GPU，100MB数据: 2 × (4-1)/4 × 100 = 150MB 理论传输量
```

### 5.2 GPUDirect RDMA 技术原理

#### 5.2.1 传统数据路径 vs GPUDirect RDMA

```bash
# 传统数据路径 (无 GPUDirect RDMA)
GPU1 Memory → CPU Memory → Network → CPU Memory → GPU2 Memory
#           ↑ PCIe      ↑ System Bus  ↑ System Bus ↑ PCIe
#           4次内存拷贝，高延迟

# GPUDirect RDMA 数据路径  
GPU1 Memory → Network → GPU2 Memory
#           ↑ 直接连接，零拷贝，低延迟
```

#### 5.2.2 NCCL 环境变量技术细节

```bash
# 关键环境变量及其技术含义

# 1. 启用 InfiniBand 支持
export NCCL_IB_DISABLE=0
# 技术说明: 
#   - 0: 启用 IB 传输
#   - 1: 禁用 IB，回退到 Socket/Ethernet
#   - 影响: 直接决定是否使用高性能 IB 网络

# 2. GPUDirect RDMA 级别控制
export NCCL_NET_GDR_LEVEL=2  
# 技术说明:
#   - 0: 禁用 GPUDirect RDMA
#   - 1: 启用 GPUDirect RDMA (读取)
#   - 2: 启用 GPUDirect RDMA (读取+写入) - 最高性能
#   - 影响: 控制 GPU 内存与网络的直接访问程度

# 3. HCA 设备选择
export NCCL_IB_HCA=mlx5_0
# 技术说明:
#   - 指定使用的 InfiniBand 网卡设备
#   - 多网卡环境下的负载均衡和性能优化
#   - 格式: 设备名称 (通过 ibstat -l 获取)

# 4. GID 索引配置 (关键性能参数)
export NCCL_IB_GID_INDEX=0    # 原生 IB
export NCCL_IB_GID_INDEX=3    # RoCE v2
# 技术说明:
#   - GID (Global Identifier): IB 网络中的全局标识符
#   - 不同 GID 索引对应不同的网络协议栈
#   - 错误配置会导致连接失败或性能下降

# 5. 流控制参数
export NCCL_IB_TC=136         # Traffic Class
export NCCL_IB_SL=0           # Service Level  
# 技术说明:
#   - TC: 流量分类，用于 QoS 控制
#   - SL: 服务级别，影响数据包优先级
#   - 136: 常用的高优先级 TC 值

# 6. 超时和重试机制
export NCCL_IB_TIMEOUT=22     # 超时时间 (对数刻度)
export NCCL_IB_RETRY_CNT=7    # 重试次数
# 技术说明:
#   - TIMEOUT: 2^22 微秒 ≈ 4.2 秒
#   - 网络不稳定时的容错机制
#   - 过小值可能导致误报，过大值影响故障检测
```

### 5.3 网络类型自动检测机制

#### 5.3.1 检测算法实现

```bash
# 脚本中的网络类型检测逻辑
detect_network_type() {
    local is_roce=false
    local network_type="未知"
    
    # 使用 ibv_devinfo 检测链路层类型
    if command -v ibv_devinfo >/dev/null 2>&1; then
        # 获取第一个设备的链路层信息
        local link_layer=$(ibv_devinfo | grep "link_layer:" | head -1 | awk '{print $2}')
        
        case "$link_layer" in
            "Ethernet")
                is_roce=true
                network_type="RoCE (Ethernet over IB)"
                # RoCE 特定配置
                configure_roce_parameters
                ;;
            "InfiniBand") 
                network_type="原生 InfiniBand"
                # 原生 IB 特定配置
                configure_native_ib_parameters
                ;;
            *)
                network_type="未知类型: $link_layer"
                # 使用保守的默认配置
                configure_default_parameters
                ;;
        esac
    fi
    
    echo "检测到网络类型: $network_type"
}
```

#### 5.3.2 配置差异对比

| 参数 | 原生 InfiniBand | RoCE v2 | 技术说明 |
|------|----------------|---------|----------|
| `NCCL_IB_GID_INDEX` | 0 | 3 | GID 索引选择不同的协议栈 |
| `NCCL_SOCKET_IFNAME` | 未设置 | `""` | RoCE 需禁用 Socket 接口避免冲突 |
| 网络层 | IB Verbs | Ethernet + IB Verbs | 底层传输协议不同 |
| 性能特征 | 超低延迟 | 较低延迟，更好兼容性 | 硬件和协议栈差异 |

### 5.4 性能测试核心代码解析

#### 5.4.1 动态张量大小计算

```python
# 脚本生成的 Python 测试代码关键部分
def calculate_tensor_size(test_size_str):
    """
    根据用户输入计算张量元素数量
    
    技术细节:
        - float32 张量: 每个元素 4 字节
        - 内存对齐: GPU 内存分配需要考虑对齐要求
        - 大小限制: 受 GPU 内存容量限制
    """
    size_mapping = {
        "1M":  262144,      # 1MB / 4 bytes = 262,144 elements
        "10M": 2621440,     # 10MB / 4 bytes = 2,621,440 elements  
        "100M": 26214400,   # 100MB / 4 bytes = 26,214,400 elements
        "1G": 268435456     # 1GB / 4 bytes = 268,435,456 elements
    }
    
    return size_mapping.get(test_size_str, 262144)

# GPU 内存安全检查
def check_gpu_memory_safety(tensor_size, device):
    """
    检查 GPU 内存是否足够容纳测试数据
    
    安全策略:
        - 预留 20% 内存用于 CUDA 上下文和其他开销
        - 考虑 NCCL 内部缓冲区需求
        - 多 GPU 环境下的内存碎片问题
    """
    gpu_memory_bytes = torch.cuda.get_device_properties(device).total_memory
    required_memory_bytes = tensor_size * 4  # float32
    
    # 安全阈值: 80% GPU 内存
    safety_threshold = gpu_memory_bytes * 0.8
    
    if required_memory_bytes > safety_threshold:
        print(f"警告: 所需内存 {required_memory_bytes/1024**3:.2f}GB "
              f"超过安全阈值 {safety_threshold/1024**3:.2f}GB")
        return False
    
    return True
```

#### 5.4.2 性能指标计算详解

```python
def calculate_performance_metrics(tensor, duration_ms, world_size):
    """
    计算详细的性能指标
    
    返回指标说明:
        - 延迟: AllReduce 操作的端到端时间
        - 数据吞吐量: 应用层数据处理速率 (非网络带宽)
        - 理论传输量: Ring AllReduce 算法的理论网络传输量
        - 算法效率: 实际性能与理论最优的比值
    """
    
    # 1. 基础数据计算
    data_size_bytes = tensor.numel() * tensor.element_size()
    data_size_mb = data_size_bytes / (1024 * 1024)
    
    # 2. 延迟计算 (毫秒)
    latency_ms = duration_ms
    
    # 3. 数据吞吐量计算 (Gbps)
    # 注意: 这是应用层吞吐量，包含了算法开销
    throughput_gbps = (data_size_mb * 8) / (duration_ms / 1000) / 1000
    
    # 4. 理论传输量计算 (Ring AllReduce)
    if world_size > 1:
        # Ring AllReduce 公式: 2 * (N-1) / N * data_size
        transfer_efficiency = 2 * (world_size - 1) / world_size
        theoretical_transfer_mb = transfer_efficiency * data_size_mb
    else:
        theoretical_transfer_mb = 0  # 单 GPU 无网络传输
    
    # 5. 算法效率分析
    # 理论最优: 每个字节只传输一次
    # Ring AllReduce: 每个字节传输 2*(N-1)/N 次
    algorithm_overhead = transfer_efficiency if world_size > 1 else 1.0
    
    return {
        'latency_ms': latency_ms,
        'throughput_gbps': throughput_gbps, 
        'data_size_mb': data_size_mb,
        'theoretical_transfer_mb': theoretical_transfer_mb,
        'algorithm_overhead': algorithm_overhead
    }
```

### 5.5 调试和诊断技术

#### 5.5.1 NCCL 调试级别详解

```bash
# NCCL 调试级别配置
export NCCL_DEBUG=TRACE    # 最详细的调试信息
export NCCL_DEBUG=INFO     # 标准调试信息 (推荐)
export NCCL_DEBUG=WARN     # 仅警告和错误 (生产环境)

# 调试子系统选择
export NCCL_DEBUG_SUBSYS=ALL        # 所有子系统
export NCCL_DEBUG_SUBSYS=INIT,NET   # 初始化和网络子系统
export NCCL_DEBUG_SUBSYS=GRAPH      # 通信图构建
export NCCL_DEBUG_SUBSYS=TUNING     # 性能调优信息

# 调试输出示例解析
# [nccl] INFO NCCL_IB_HCA set by environment to mlx5_0
#        ↑     ↑    ↑
#     组件  级别  消息内容
```

#### 5.5.2 性能瓶颈诊断流程

```bash
# 1. 网络连通性检查
ping_test() {
    # IB 网络 ping 测试
    ibping -S    # 服务端
    ibping -c 1000 -f <server_lid>  # 客户端
}

# 2. 带宽基准测试  
bandwidth_test() {
    # 使用 ib_write_bw 测试原始 IB 带宽
    ib_write_bw -d mlx5_0 -i 1 -s 1048576
    
    # 对比 NCCL 测试结果与原始带宽
    # 如果差距过大，可能存在配置问题
}

# 3. 延迟基准测试
latency_test() {
    # 使用 ib_write_lat 测试原始 IB 延迟
    ib_write_lat -d mlx5_0 -i 1000
    
    # NCCL 延迟通常比原始 IB 延迟高 2-5 倍
    # 这是正常的算法和协议开销
}

# 4. GPU 内存带宽测试
gpu_memory_test() {
    # 使用 bandwidthTest (需要安装 CUDA Samples)
    # 如果已安装: /usr/local/cuda/samples/1_Utilities/bandwidthTest/bandwidthTest
    if command -v bandwidthTest >/dev/null 2>&1; then
        bandwidthTest --memory=pinned --mode=quick
    else
        echo "bandwidthTest 未找到，请安装 CUDA Samples"
        echo "或使用: nvidia-smi --query-gpu=memory.total,memory.used --format=csv"
    fi
}    
    # GPU 内存带宽应远高于网络带宽
    # 确保不是 GPU 内存成为瓶颈
}
```

---

## 6. Python 测试模板

### 6.1 Python 测试模板概述

`nccl_python_template.py` 提供了一个简洁高效的 NCCL 分布式通信测试模板，专注于核心性能测试功能：

#### 6.1.1 功能特性

**系统信息收集**：

- 自动检测和显示系统硬件信息（Python版本、PyTorch版本、CUDA版本、GPU信息）
- NCCL 版本信息和环境变量配置展示
- 本机IP地址获取
- GPU内存信息显示

**性能测试**：

- NCCL AllReduce 操作的持续性能测试
- 基于时间的测试控制（默认30秒）
- 实时延迟和吞吐量测量
- 详细的性能统计分析

**核心特点**：

- 使用环境变量进行参数配置
- 支持多进程分布式测试
- 固定使用float32数据类型
- 自动进程同步和资源清理

#### 6.1.2 技术架构

**分布式初始化**：

- 使用 PyTorch 分布式后端（torch.distributed）
- 支持 NCCL 后端的 GPU 通信
- 基于环境变量的进程组管理
- 完善的错误处理和超时机制

**测试执行流程**：

- 系统信息检测和环境验证
- 分布式环境初始化和设备分配
- 预热阶段（5次AllReduce操作）
- 基于时间的持续性能测试
- 统计数据收集和结果输出

**性能监控**：

- 毫秒级精确时间测量
- 实时吞吐量计算（GB/s）
- 延迟统计（平均值、最小值、最大值）
- AllReduce理论传输量计算

### 6.2 环境要求

#### 6.2.1 软件依赖

```bash
# Python 环境要求
Python >= 3.7

# 核心依赖包（必需）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 注意：模板仅使用标准库和PyTorch，无需额外依赖包
```

#### 6.2.2 硬件要求

- **GPU**: NVIDIA GPU with CUDA support
- **NCCL**: NCCL 2.0+ (通常随 PyTorch 安装)
- **网络**: InfiniBand, Ethernet, 或 NVLink
- **内存**: 足够的 GPU 内存用于测试数据

### 6.3 基本使用

**重要说明**：该模板通过环境变量进行配置，通常由 `nccl_benchmark.sh` 脚本调用，不支持命令行参数。

#### 6.3.1 环境变量配置

```bash
# 分布式配置（必需）
export RANK=0                    # 进程排名
export WORLD_SIZE=2              # 总进程数
export LOCAL_RANK=0              # 本地GPU排名
export MASTER_ADDR=localhost     # 主节点地址
export MASTER_PORT=29500         # 主节点端口

# 测试参数配置（可选）
export TENSOR_ELEMENTS=1000000   # 张量元素数量（默认1M）
export TEST_DURATION=30          # 测试时长（秒，默认30）
export NCCL_BACKEND=nccl         # 后端类型（默认nccl）

# 运行测试
python nccl_python_template.py
```

#### 6.3.2 单节点多GPU测试

```bash
# 2 GPU测试示例
export WORLD_SIZE=2
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 启动第一个进程
export RANK=0
export LOCAL_RANK=0
python nccl_python_template.py &

# 启动第二个进程
export RANK=1
export LOCAL_RANK=1
python nccl_python_template.py &

wait  # 等待所有进程完成
```

#### 6.3.3 多节点分布式测试

```bash
# 节点1 (192.168.1.100) - 运行2个进程
export WORLD_SIZE=4
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500

# 节点1的第一个进程
export RANK=0
export LOCAL_RANK=0
python nccl_python_template.py &

# 节点1的第二个进程
export RANK=1
export LOCAL_RANK=1
python nccl_python_template.py &

# 节点2 (192.168.1.101) - 运行2个进程
export WORLD_SIZE=4
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500

# 节点2的第一个进程
export RANK=2
export LOCAL_RANK=0
python nccl_python_template.py &

# 节点2的第二个进程
export RANK=3
export LOCAL_RANK=1
python nccl_python_template.py &
```

### 6.4 高级配置

#### 6.4.1 测试参数调整

```bash
# 大数据量测试（10M元素，约40MB）
export TENSOR_ELEMENTS=10000000
export TEST_DURATION=60
python nccl_python_template.py

# 小数据量延迟测试（100K元素，约400KB）
export TENSOR_ELEMENTS=100000
export TEST_DURATION=30
python nccl_python_template.py

# 长时间稳定性测试
export TENSOR_ELEMENTS=1000000
export TEST_DURATION=300  # 5分钟
python nccl_python_template.py
```

#### 6.4.2 NCCL环境变量配置

```bash
# NCCL 调试配置
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 网络配置
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=^docker0,lo

# 性能优化
export NCCL_BUFFSIZE=8388608
export NCCL_NTHREADS=16

# 运行测试
export RANK=0
export WORLD_SIZE=2
export LOCAL_RANK=0
python nccl_python_template.py
```

#### 6.4.3 批量测试脚本

```bash
#!/bin/bash
# batch_test.sh - 批量性能测试脚本

# 测试不同数据大小（以元素数量为单位）
tensor_sizes=(100000 1000000 10000000)  # 100K, 1M, 10M 元素
world_sizes=(2 4 8)

echo "开始批量性能测试..."
echo "时间戳: $(date)"

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export TEST_DURATION=30

for size in "${tensor_sizes[@]}"; do
    for ws in "${world_sizes[@]}"; do
        echo "测试配置: ${ws} 进程, 张量大小: ${size} 元素"
        
        export TENSOR_ELEMENTS=$size
        export WORLD_SIZE=$ws
        
        # 启动多个进程
        for ((rank=0; rank<ws; rank++)); do
            export RANK=$rank
            export LOCAL_RANK=$((rank % $(nvidia-smi -L | wc -l)))
            python nccl_python_template.py > "log_${ws}proc_${size}elem_rank${rank}.txt" 2>&1 &
        done
        
        wait  # 等待所有进程完成
        echo "完成: ${ws} 进程, ${size} 元素"
        sleep 5
    done
done

echo "批量测试完成"
```

### 6.5 输出解读

#### 6.5.1 系统信息输出

模板首先输出系统环境信息：

```bash
=== 系统信息 ===
Python 版本: 3.12.11 | packaged by Anaconda, Inc. | (main, Jun  5 2025, 13:09:17) [GCC 11.2.0]
PyTorch 版本: 2.7.0+cu126
CUDA 版本: 12.6
NCCL 版本: (2, 26, 2)
可用 GPU 数量: 3
GPU 0: NVIDIA GPU (79.2 GB)
GPU 1: NVIDIA GPU (79.2 GB)
GPU 2: NVIDIA GPU (79.2 GB)
本机 IP: 192.168.1.100

=== NCCL 环境信息 ===
NCCL 版本: (2, 26, 2)
NCCL 环境变量:
  NCCL_IB_DISABLE: 1
  NCCL_NET_DISABLE: 1
  NCCL_NET_GDR_LEVEL: 0
  NCCL_IB_HCA: 未设置
  NCCL_IB_GID_INDEX: 未设置
  NCCL_DEBUG: INFO
  NCCL_DEBUG_SUBSYS: INIT,NET
  NCCL_P2P_DISABLE: 0
  NCCL_P2P_LEVEL: NVL
  NCCL_NVLS_ENABLE: 1
  NCCL_SOCKET_IFNAME: lo
  NCCL_IB_TIMEOUT: 未设置
  NCCL_IB_RETRY_CNT: 未设置
  NCCL_SHM_DISABLE: 未设置
  NCCL_SOCKET_FAMILY: 未设置
  NCCL_SOCKET_NTHREADS: 未设置
```

#### 6.5.2 测试配置输出

```bash
[Rank 0] 开始初始化分布式环境...
[Rank 0] 后端: nccl
[Rank 0] 世界大小: 3
[Rank 0] 本地排名: 0
[Rank 0] 张量大小: 262144 个元素 (1.00 MB)
[Rank 0] 分布式环境初始化成功
[Rank 0] 使用设备: cuda:0
[Rank 0] GPU: NVIDIA GPU

[Rank 1] 开始初始化分布式环境...
[Rank 1] 后端: nccl
[Rank 1] 世界大小: 3
[Rank 1] 本地排名: 1
[Rank 1] 张量大小: 262144 个元素 (1.00 MB)
[Rank 1] 分布式环境初始化成功
[Rank 1] 使用设备: cuda:1
[Rank 1] GPU: NVIDIA GPU

[Rank 2] 开始初始化分布式环境...
[Rank 2] 后端: nccl
[Rank 2] 世界大小: 3
[Rank 2] 本地排名: 2
[Rank 2] 张量大小: 262144 个元素 (1.00 MB)
[Rank 2] 分布式环境初始化成功
[Rank 2] 使用设备: cuda:2
[Rank 2] GPU: NVIDIA GPU
```

#### 6.5.3 性能测试结果

```bash
=== 性能测试开始 ===
[Rank 0] 迭代 100: 0.05 ms, 74.47 GB/s
[Rank 1] 迭代 100: 0.05 ms, 73.47 GB/s
[Rank 2] 迭代 100: 0.05 ms, 74.81 GB/s

[Rank 0] 迭代 200: 0.05 ms, 75.16 GB/s
[Rank 1] 迭代 200: 0.05 ms, 74.81 GB/s
[Rank 2] 迭代 200: 0.05 ms, 75.50 GB/s

...

[Rank 0] 迭代 374200: 0.05 ms, 75.16 GB/s
[Rank 1] 迭代 374200: 0.05 ms, 74.81 GB/s
[Rank 2] 迭代 374200: 0.05 ms, 75.50 GB/s

[Rank 0] 测试时间到达，设置停止标志
[Rank 0] 收到停止信号，退出测试循环
[Rank 1] 收到停止信号，退出测试循环
[Rank 2] 收到停止信号，退出测试循环

=== 测试完成 ===
[Rank 0] 总迭代次数: 374250
[Rank 0] 总测试时间: 30.01 秒
[Rank 0] 平均延迟: 0.08 ms
[Rank 0] 最小延迟: 0.05 ms
[Rank 0] 最大延迟: 21.14 ms
[Rank 0] 平均吞吐量: 48.71 GB/s
[Rank 0] 数据大小: 262144 个元素 (1.00 MB)
[Rank 0] 理论传输量: 1461.91 GB
[Rank 0] 网络传输倍数: 4.0x

[Rank 1] 总迭代次数: 374250
[Rank 1] 总测试时间: 30.01 秒
[Rank 1] 平均延迟: 0.08 ms
[Rank 1] 最小延迟: 0.05 ms
[Rank 1] 最大延迟: 21.74 ms
[Rank 1] 平均吞吐量: 48.71 GB/s

[Rank 2] 总迭代次数: 374250
[Rank 2] 总测试时间: 30.01 秒
[Rank 2] 平均延迟: 0.08 ms
[Rank 2] 最小延迟: 0.05 ms
[Rank 2] 最大延迟: 21.51 ms
[Rank 2] 平均吞吐量: 48.71 GB/s

✅ NCCL AllReduce 测试成功完成
```

#### 6.5.4 指标含义

- **总迭代次数**: 在指定时间内完成的AllReduce操作次数
- **总测试时间**: 实际测试运行的总时长
- **平均延迟**: 单次AllReduce操作的平均耗时（毫秒）
- **最小延迟**: 测试期间记录的最低延迟
- **最大延迟**: 测试期间记录的最高延迟（包含初始化开销）
- **平均吞吐量**: 数据传输的平均速率（GB/s）
- **数据大小**: 每次传输的数据量（元素数量，1.00 MB）
- **理论传输量**: 测试期间的总数据传输量
- **网络传输倍数**: AllReduce操作的网络传输倍数（通常为 2×(n-1)/n，其中n为GPU数量）

### 6.6 注意事项

#### 6.6.1 环境要求

- **PyTorch版本**: 确保安装了支持NCCL的PyTorch版本
- **CUDA环境**: 需要正确配置CUDA环境变量
- **分布式环境**: 多节点测试需要配置SSH免密登录
- **网络连通**: 确保节点间网络连通性

#### 6.6.2 常见问题

**环境变量未设置**：

```bash
# 检查必需的环境变量
echo $RANK $WORLD_SIZE $LOCAL_RANK $MASTER_ADDR $MASTER_PORT

# 设置默认值
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
```

**GPU内存不足**：

```bash
# 减少张量大小
export TENSOR_ELEMENTS=100000  # 减少到100K元素

# 检查GPU内存使用
nvidia-smi
```

**分布式初始化超时**：

```bash
# 增加超时时间（代码中已设置为300秒）
# 检查网络连通性
ping $MASTER_ADDR

# 检查端口是否被占用
netstat -an | grep $MASTER_PORT
```

#### 6.6.3 调试建议

- **单进程测试**: 先在单进程环境下验证代码正确性
- **逐步扩展**: 从单节点多GPU扩展到多节点测试
- **日志分析**: 查看每个进程的输出日志
- **性能对比**: 与官方nccl-tests结果进行对比

---

## 7. 配置说明

### 7.1 NCCL 环境变量

脚本会自动配置以下关键的 NCCL 环境变量：

#### 7.1.1 基础配置

- `NCCL_IB_DISABLE=0`: 启用 InfiniBand 支持
- `NCCL_NET_GDR_LEVEL=2`: 启用 GPUDirect RDMA
- `NCCL_IB_HCA`: 自动检测并设置 HCA 设备名

#### 7.1.2 网络类型特定配置

**原生 InfiniBand**:

- `NCCL_IB_GID_INDEX=0`: 使用默认 GID
- `NCCL_IB_TC=136`: 流控制参数
- `NCCL_IB_SL=0`: 服务级别

**RoCE (Ethernet over IB)**:

- `NCCL_IB_GID_INDEX=3`: RoCE v2 GID
- `NCCL_IB_TC=136`: 流控制参数
- `NCCL_IB_SL=0`: 服务级别
- `NCCL_SOCKET_IFNAME=""`: 禁用 Socket 接口

### 7.2 调试和性能优化

- `NCCL_DEBUG=INFO`: 启用详细日志
- `NCCL_DEBUG_SUBSYS=INIT,NET`: 网络初始化调试
- `NCCL_IB_TIMEOUT=22`: IB 超时设置
- `NCCL_IB_RETRY_CNT=7`: 重试次数

### 7.3 不同网络模式下的 NCCL 环境变量配置

脚本支持 7 种网络后端模式，每种模式都有特定的 NCCL 环境变量配置策略和严格的硬件检查机制。脚本会根据选择的网络模式进行相应的硬件验证，如果硬件不支持或配置不当，会直接退出并提供详细的解决方案。

**支持的网络模式**：

- **auto**: 自动检测最优网络（按 NCCL 优先级）
- **ib**: 强制使用 InfiniBand（包括原生 IB 和 RoCE）
- **nvlink**: 强制使用 NVLink（仅单节点多GPU）
- **pcie**: 强制使用 PCIe P2P（仅单节点多GPU）
- **ethernet**: 使用标准以太网
- **socket**: 强制使用 TCP Socket
- **shm**: 仅使用共享内存（仅单节点）

**重要特性**：

- **硬件检查**: 每种强制模式都会验证硬件支持
- **错误处理**: 硬件不匹配时提供详细的解决方案
- **配置冲突检测**: 防止多节点模式使用单节点专用网络
- **环境变量展示**: 实时显示当前 NCCL 环境变量配置

以下是各种模式的详细配置说明：

#### 7.3.1 Auto 模式 (`--network auto`)

**配置策略**: 按照 NCCL 优先级自动检测并选择最优网络

Auto 模式实现了智能的硬件检测和配置选择，严格按照 NCCL 的性能优先级进行检测：

**检测优先级和逻辑**:

1. **NVLink 检测** (最高优先级，仅单节点多GPU)

   ```bash
   # 检测活跃的 NVLink 连接
   nvidia-smi nvlink --status | grep "GB/s"
   # 验证带宽信息（如 "26.562 GB/s"）
   # 备选：检测 GPU 拓扑中的 NVLink 硬件
   nvidia-smi topo -m | grep "NV[0-9]+"
   ```

2. **PCIe P2P 检测** (第二优先级，仅单节点多GPU)

   ```bash
   # 检查 P2P 拓扑矩阵
   nvidia-smi topo -p2p r | grep "OK"
   # 基于 GPU 型号推测（Tesla/Quadro/A系列/H系列支持）
   ```

3. **共享内存** (第三优先级，仅单节点)
   - 单节点模式下自动可用，无需额外检测

4. **网络传输** (最低优先级)

   ```bash
   # InfiniBand 优先检测
   ibv_devinfo | grep "link_layer"
   # 区分原生 InfiniBand 和 RoCE
   # 以太网备选：InfiniBand 不可用时回退
   ```

**环境变量设置**: 根据检测结果调用对应的配置函数

```bash
# 通用调试配置（所有检测到的模式）
NCCL_DEBUG=INFO                    # 启用调试信息
NCCL_DEBUG_SUBSYS=INIT,NET        # 初始化和网络子系统调试

# 多节点配置（如果启用多节点模式）
NCCL_SOCKET_IFNAME=^docker0,lo     # 排除容器接口

# 具体的环境变量配置取决于检测结果：
# - 检测到 NVLink → 调用 configure_nvlink_settings
# - 检测到 PCIe P2P → 调用 configure_pcie_p2p_settings  
# - 单节点无高速互连 → 调用 configure_shm_settings
# - 检测到 InfiniBand → 调用 configure_infiniband_settings
# - 回退到以太网 → 调用 configure_ethernet_settings
```

**检测日志示例**:

```bash
[INFO] 自动检测网络环境 (按 NCCL 优先级: NVLink > PCIe P2P > 共享内存 > 网络传输)...
[SUCCESS] 检测到 4 个活跃的 NVLink 连接 (带宽: 25.781 GB/s)
[SUCCESS] 自动选择 NVLink 网络 (最高优先级)
```

#### 7.3.2 InfiniBand 模式 (`--network ib`)

**配置策略**: 强制使用 InfiniBand 网络，包含严格的硬件检查

InfiniBand 模式会进行全面的硬件验证，确保 InfiniBand 环境可用：

**硬件检查流程**:

1. **工具可用性检查**

   ```bash
   # 验证 ibv_devinfo 命令可用性
   command -v ibv_devinfo >/dev/null 2>&1
   # 失败时提供安装指导：apt-get install infiniband-diags
   ```

2. **设备检测和类型识别**

   ```bash
   # 检测 InfiniBand 设备
   ibv_devinfo | grep "link_layer:"
   # 区分链路层类型：
   # - "InfiniBand" → 原生 InfiniBand
   # - "Ethernet" → RoCE (RDMA over Converged Ethernet)
   ```

3. **设备状态验证**

   ```bash
   # 验证 HCA 设备状态（如果 ibstat 可用）
   ibstat | grep "State: Active"
   # 确保设备处于活跃状态
   ```

**环境变量设置**:

```bash
# 核心 InfiniBand 配置
NCCL_IB_DISABLE=0                  # 启用 InfiniBand
NCCL_SOCKET_DISABLE=1              # 禁用 Socket 传输
NCCL_NET_GDR_LEVEL=5               # 最高级别的 GPUDirect RDMA

# InfiniBand 特定参数
NCCL_IB_HCA=mlx5_0                 # HCA 设备（自动检测）
NCCL_IB_GID_INDEX=3                # GID 索引（RoCE 使用 3，原生 IB 使用 0）
NCCL_IB_TIMEOUT=22                 # 超时设置
NCCL_IB_RETRY_CNT=7                # 重试次数

# 性能优化
NCCL_IB_USE_CUDA_MEM=1             # 启用 CUDA 内存
NCCL_IB_CUDA_SUPPORT=1             # 启用 CUDA 支持
NCCL_BUFFSIZE=8388608              # 8MB 缓冲区

# 调试和监控
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET
NCCL_GRAPH_DUMP_FILE=/tmp/nccl_graph_ib.xml
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo_ib.xml

# 其他配置
NCCL_IGNORE_CPU_AFFINITY=1
NCCL_CROSS_NIC=0
```

**错误处理**: 硬件检查失败时会直接退出并提供解决方案：

```bash
[ERROR] 硬件检查失败: ibv_devinfo 命令不可用，请安装 InfiniBand 驱动和工具
[ERROR] 无法强制使用 InfiniBand 网络
[INFO] 解决方案:
[INFO]   1. 安装 InfiniBand 驱动: apt-get install infiniband-diags
[INFO]   2. 或使用其他网络后端: --network ethernet 或 --network auto
```

#### 7.3.3 NVLink 模式 (`--network nvlink`)

**配置策略**: 强制使用 NVLink 传输，仅支持单节点多GPU

NVLink 模式包含严格的硬件检查和配置限制：

**基础条件检查**:

1. **节点模式验证**

   ```bash
   # NVLink 仅适用于单节点多GPU场景
   if [ "$MULTI_NODE_MODE" = true ]; then
       # 直接退出，提供解决方案
   fi
   ```

2. **GPU 数量检查**

   ```bash
   # 检查 nvidia-smi 可用性
   command -v nvidia-smi >/dev/null 2>&1
   # 验证 GPU 数量（至少 2 个）
   gpu_count=$(nvidia-smi -L | wc -l)
   ```

**NVLink 硬件检测**:

1. **活跃连接检测**（主要方法）

   ```bash
   # 检测活跃的 NVLink 连接
   nvidia-smi nvlink --status | grep -c "GB/s"
   # 验证带宽信息（如 "26.562 GB/s"）
   ```

2. **硬件拓扑检测**（备选方法）

   ```bash
   # 检查 NVLink 硬件拓扑
   nvidia-smi topo -m | grep -E "NV[0-9]+"
   ```

**环境变量设置**:

```bash
# NVLink 核心配置
NCCL_NET_DISABLE=1                 # 禁用网络传输
NCCL_P2P_DISABLE=0                 # 启用 P2P 传输
NCCL_SHM_DISABLE=1                 # 禁用共享内存

# NVLink 特定参数
NCCL_NVLS_ENABLE=1                 # 启用 NVLink Sharp
NCCL_NVLS_CHUNKSIZE=524288         # 块大小 (512KB)
NCCL_ALGO=NVLS,Tree,Ring           # 算法优先级

# 性能优化
NCCL_MAX_NCHANNELS=16              # 最大通道数
NCCL_MIN_NCHANNELS=8               # 最小通道数

# 调试配置
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET
NCCL_GRAPH_DUMP_FILE=/tmp/nccl_graph_nvlink.xml
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo_nvlink.xml

# 其他配置
NCCL_IGNORE_CPU_AFFINITY=1
NCCL_CROSS_NIC=0
```

**错误处理**: 不满足条件时会直接退出并提供解决方案：

```bash
[ERROR] NVLink 仅支持单节点多GPU模式，不支持多节点
[ERROR] 无法强制使用 NVLink 网络
[INFO] 解决方案:
[INFO]   1. 使用单节点模式 (移除 -m 或 --multi-node 参数)
[INFO]   2. 或使用多节点兼容的网络后端: --network ib 或 --network ethernet
```

#### 7.3.4 PCIe 模式 (`--network pcie`)

**配置策略**: 强制使用 PCIe P2P 传输，仅支持单节点多GPU

PCIe 模式与 NVLink 模式类似，包含严格的硬件检查和配置限制：

**基础条件检查**:

1. **节点模式验证**

   ```bash
   # PCIe P2P 仅适用于单节点多GPU场景
   if [ "$MULTI_NODE_MODE" = true ]; then
       # 直接退出，提供解决方案
   fi
   ```

2. **GPU 数量和 P2P 支持检查**

   ```bash
   # 验证 GPU 数量（至少 2 个）
   gpu_count=$(nvidia-smi -L | wc -l)
   # 检查 P2P 拓扑可用性
   nvidia-smi topo -p2p r | grep "OK"
   ```

**PCIe P2P 硬件检测**:

1. **P2P 拓扑验证**

   ```bash
   # 检查 P2P 拓扑矩阵
   nvidia-smi topo -p2p r | grep "OK"
   ```

2. **ACS 状态检测**

   ```bash
   # 检测 Access Control Services (ACS) 状态
   # ACS 启用可能阻止 P2P 通信
   ```

3. **GPU 型号推测**

   ```bash
   # 基于 GPU 型号推测 P2P 支持
   # Tesla/Quadro/A系列/H系列通常支持 P2P
   ```

**环境变量设置**:

```bash
# PCIe P2P 核心配置
NCCL_NET_DISABLE=1                 # 禁用网络传输
NCCL_P2P_DISABLE=0                 # 启用 P2P 传输
NCCL_SHM_DISABLE=1                 # 禁用共享内存

# PCIe 特定参数
NCCL_P2P_LEVEL=SYS                 # 系统级别 P2P
NCCL_ALGO=Tree,Ring                # 算法选择

# 性能优化
NCCL_MAX_NCHANNELS=8               # 最大通道数
NCCL_MIN_NCHANNELS=4               # 最小通道数

# 调试配置
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET
NCCL_GRAPH_DUMP_FILE=/tmp/nccl_graph_pcie.xml
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo_pcie.xml

# 其他配置
NCCL_IGNORE_CPU_AFFINITY=1
NCCL_CROSS_NIC=0
```

**错误处理**: 不满足条件时会直接退出并提供解决方案：

```bash
[ERROR] PCIe P2P 仅支持单节点多GPU模式，不支持多节点
[ERROR] 无法强制使用 PCIe P2P 网络
[INFO] 解决方案:
[INFO]   1. 使用单节点模式 (移除 -m 或 --multi-node 参数)
[INFO]   2. 或使用多节点兼容的网络后端: --network ib 或 --network ethernet
```

#### 7.3.5 以太网模式 (`--network ethernet`)

**配置策略**: 使用标准以太网进行通信（兼容性模式）

以太网模式提供最大的兼容性，适用于各种网络环境：

**网络接口检测**:

1. **可用接口检测**

   ```bash
   # 检测可用的以太网接口
   ip link show | grep "state UP" | grep -v "lo\|docker"
   # 排除回环和虚拟接口
   ```

2. **接口状态验证**

   ```bash
   # 验证接口状态和连通性
   # 选择第一个可用的以太网接口
   ```

**环境变量设置**:

```bash
# 以太网核心配置
NCCL_NET_DISABLE=0                 # 启用网络传输
NCCL_IB_DISABLE=1                  # 禁用 InfiniBand
NCCL_SOCKET_DISABLE=0              # 启用 Socket 传输

# 网络接口配置
NCCL_SOCKET_IFNAME=eth0            # 指定以太网接口（自动检测）
NCCL_SOCKET_NTHREADS=4             # Socket 线程数

# 性能配置
NCCL_NET_GDR_LEVEL=0               # 禁用 GPUDirect（以太网不支持）
NCCL_BUFFSIZE=8388608              # 8MB 缓冲区

# 调试配置
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET
NCCL_GRAPH_DUMP_FILE=/tmp/nccl_graph_ethernet.xml
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo_ethernet.xml

# 其他配置
NCCL_IGNORE_CPU_AFFINITY=1
NCCL_CROSS_NIC=0
```

**特点**:

- **高兼容性**: 适用于所有有以太网的环境
- **自动检测**: 自动选择可用的以太网接口
- **性能适中**: 性能低于 InfiniBand 但高于 Socket 模式
- **多节点支持**: 支持多节点分布式训练

#### 7.3.6 Socket 模式 (`--network socket`)

**配置策略**: 强制使用 TCP Socket 进行通信（调试和测试模式）

Socket 模式强制使用 TCP Socket 传输，主要用于调试和兼容性测试：

**配置特点**:

- **强制 Socket**: 禁用所有其他传输方式
- **容器友好**: 自动排除容器网络接口
- **调试优化**: 适合网络问题排查

**环境变量设置**:

```bash
# Socket 核心配置
NCCL_NET_DISABLE=0                 # 启用网络传输
NCCL_IB_DISABLE=1                  # 禁用 InfiniBand
NCCL_SOCKET_DISABLE=0              # 启用 Socket 传输
NCCL_SOCKET_FORCE=1                # 强制使用 Socket

# 容器环境适配
NCCL_SOCKET_IFNAME=^docker0,lo     # 排除容器接口（多节点）
# 或单节点时使用 loopback

# 性能配置
NCCL_NET_GDR_LEVEL=0               # 禁用 GPUDirect
NCCL_BUFFSIZE=8388608              # 8MB 缓冲区
NCCL_CHECK_DISABLE=0               # 启用检查

# 调试配置
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET
NCCL_GRAPH_DUMP_FILE=/tmp/nccl_graph_socket.xml
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo_socket.xml

# 其他配置
NCCL_IGNORE_CPU_AFFINITY=1
NCCL_CROSS_NIC=0
```

**使用场景**:

- **网络调试**: 排查网络配置问题
- **兼容性测试**: 验证基础 TCP 连通性
- **容器环境**: 容器化部署的备选方案
- **开发测试**: 开发环境的简单配置

#### 7.3.7 共享内存模式 (`--network shm`)

**配置策略**: 仅使用共享内存进行 GPU 间通信（仅单节点）

共享内存模式是最基础的传输方式，仅适用于单节点环境：

**限制检查**:

1. **节点模式验证**

   ```bash
   # 共享内存仅适用于单节点
   if [ "$MULTI_NODE_MODE" = true ]; then
       # 直接退出，提供解决方案
   fi
   ```

2. **性能警告**

   ```bash
   # 警告性能限制
   log_warning "共享内存模式性能有限，建议使用 NVLink 或 PCIe P2P"
   ```

**环境变量设置**:

```bash
# 共享内存核心配置
NCCL_NET_DISABLE=1                 # 禁用网络传输
NCCL_P2P_DISABLE=1                 # 禁用 P2P 传输
NCCL_SHM_DISABLE=0                 # 启用共享内存

# 禁用高级功能
NCCL_NET_GDR_LEVEL=0               # 禁用 GPUDirect
NCCL_NVLS_ENABLE=0                 # 禁用 NVLink SHARP

# 性能配置
NCCL_BUFFSIZE=8388608              # 8MB 缓冲区

# 网络接口清理
NCCL_SOCKET_IFNAME=                # 清除接口限制

# 调试配置
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET
NCCL_GRAPH_DUMP_FILE=/tmp/nccl_graph_shm.xml
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo_shm.xml

# 其他配置
NCCL_IGNORE_CPU_AFFINITY=1
NCCL_CROSS_NIC=0
```

**特点和限制**:

- **仅单节点**: 不支持多节点分布式训练
- **性能最低**: 所有传输方式中性能最低
- **高兼容性**: 适用于所有单节点环境
- **调试用途**: 主要用于排查硬件问题

**错误处理**: 多节点模式下会直接退出：

```bash
[ERROR] 共享内存模式仅支持单节点，不支持多节点
[ERROR] 无法使用共享内存网络
[INFO] 解决方案:
[INFO]   1. 使用单节点模式 (移除 -m 或 --multi-node 参数)
[INFO]   2. 或使用多节点兼容的网络后端: --network ib 或 --network ethernet
```

#### 7.3.8 环境变量优先级和覆盖规则

**配置流程**:

1. **网络模式选择**: 根据 `--network` 参数选择配置策略
2. **硬件检查**: 验证硬件支持（强制模式下）
3. **环境变量配置**: 调用对应的配置函数
4. **通用配置**: 应用调试和多节点配置
5. **环境变量展示**: 实时显示当前配置

**优先级顺序**:

1. **命令行参数** (`--network` 选项) - 最高优先级
2. **脚本配置函数** (基于硬件检测和模式选择)
3. **通用配置** (调试、多节点等)
4. **NCCL 默认值** - 最低优先级

**重要特性**:

- **不覆盖用户变量**: 脚本不会覆盖用户预设的 NCCL 环境变量
- **配置一致性**: 每种模式都有独立的配置函数，确保配置一致
- **错误处理**: 硬件不匹配时直接退出，避免错误配置
- **实时展示**: 配置完成后立即显示环境变量状态

**环境变量展示功能**:

脚本在配置完成后会调用 `display_nccl_environment_variables` 函数，提供：

- **分类展示**: 按核心配置、网络配置、性能优化、调试配置分类
- **状态标识**:
  - `[SUCCESS]` - 已设置的变量
  - `[INFO] 未设置` - 未设置的变量  
  - `[WARNING]` - 额外的变量（非预定义但已设置）
- **统计信息**: 显示已设置变量的数量和比例
- **实时更新**: 反映当前环境的实际状态

**验证方法**:

```bash
# 查看配置过程和环境变量
./nccl_benchmark.sh --network auto --dry-run

# 查看特定模式的配置
./nccl_benchmark.sh --network ib --dry-run

# 启用详细日志查看硬件检测过程
./nccl_benchmark.sh --network nvlink --verbose --dry-run
```

---

## 8. 测试结果与运维指南

### 8.1 输出文件说明

#### 8.1.1 日志文件

- **位置**: `/tmp/nccl_test_YYYYMMDD_HHMMSS.log`
- **内容**: 详细的执行日志，包括所有检查、配置和测试过程

#### 8.1.2 测试输出

- **位置**: `/tmp/nccl_test_output.log`
- **内容**: NCCL 测试的原始输出，包含性能数据和调试信息

#### 8.1.3 测试报告

- **位置**: `/tmp/nccl_test_report_YYYYMMDD_HHMMSS.txt`
- **内容**: 格式化的测试报告，包含系统信息、配置参数、测试结果

#### 8.1.4 临时文件

- `/tmp/nccl_test.py`: 动态生成的 NCCL 测试脚本

### 8.2 性能指标说明

#### 8.2.1 延迟 (Latency)

- **单位**: 毫秒 (ms)
- **含义**: AllReduce 操作的端到端延迟
- **典型值**: 0.1-2.0 ms (取决于数据大小和网络配置)

#### 8.2.2 数据吞吐量 (Data Throughput)

- **单位**: Gbps (比特每秒)
- **含义**: NCCL 操作的数据处理速率
- **注意**: 这不等同于网络带宽，包含了算法开销
- **换算**: 1 Gbps = 0.125 GB/s (字节每秒)

#### 8.2.3 理论传输量 (Theoretical Transfer)

- **单位**: MB
- **计算公式**: Ring AllReduce: `2 * (N-1) / N * data_size`
- **含义**: 理论上需要传输的数据量

---

### 8.3 故障排除

#### 8.3.1 常见问题

##### 8.3.1.1 依赖检查失败

```bash
[ERROR] Python3 未安装
[ERROR] PyTorch 未安装或不可用
```

**解决方案**: 安装缺失的依赖包

##### 8.3.1.2 InfiniBand 设备不可用

```bash
[ERROR] InfiniBand 设备不可用或未配置
```

**解决方案**:

- 检查 IB 驱动是否正确安装
- 确认 IB 网卡硬件连接正常
- 运行 `ibstat` 检查设备状态

##### 8.3.1.3 NCCL 测试失败

```bash
[ERROR] NCCL 测试执行失败
```

**解决方案**:

- 检查 GPU 内存是否充足
- 确认 NCCL 环境变量配置正确
- 查看详细日志文件排查具体错误

##### 8.3.1.4 网络性能不佳

```bash
[WARNING] 数据吞吐量低于预期
```

**解决方案**:

- 检查 IB 网络拓扑和连接
- 验证 GPUDirect RDMA 是否启用
- 调整 NCCL 参数优化性能

#### 8.3.2 调试技巧

##### 8.3.2.1 启用详细日志

```bash
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
./nccl_benchmark.sh --network auto -s 1M -t 30
```

##### 8.3.2.2 检查 IB 状态

```bash
# 查看 IB 设备状态
ibstat

# 查看设备详细信息
ibv_devinfo

# 检查性能计数器
perfquery -a
```

##### 8.3.2.3 监控网络流量

```bash
# 在测试前后比较计数器
perfquery -a > before.txt
# 运行测试
perfquery -a > after.txt
diff before.txt after.txt
```

---

### 8.4 最佳实践

#### 8.4.1 测试前准备

- 建议先运行 `../InfiniBand/health/ib_health_check.sh` 确保 InfiniBand 网络正常
- 使用 `../InfiniBand/monitor/ib_bandwidth_monitor.sh` 监控测试期间的网络性能
- 确保所有节点的时间同步
- 确保所有 GPU 可用且内存充足
- 验证 InfiniBand 网络连接正常
- 关闭不必要的后台进程

#### 8.4.2 性能监控

- 在测试期间，可以使用专门的 `ib_bandwidth_monitor.sh` 脚本监控 InfiniBand 网络性能
- 建议在另一个终端窗口运行监控脚本：

  ```bash
  # 在测试期间监控网络性能 (需要先安装 InfiniBand 工具包)
  # 脚本位置: ../InfiniBand/monitor/ib_bandwidth_monitor.sh
  ../InfiniBand/monitor/ib_bandwidth_monitor.sh -d mlx5_0 -p 1 -t 300
  ```

#### 8.4.3 性能优化

- 使用多 GPU 环境进行测试
- 确保 GPUDirect RDMA 正确配置
- 根据网络类型调整 NCCL 参数

#### 8.4.4 定期监控

- 定期运行测试以监控系统性能变化
- 建立性能基线，及时发现性能退化
- 记录配置变更对性能的影响
