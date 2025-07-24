# InfiniBand 网卡健康检查脚本使用说明

## 1. 概述

`ib_health_check.sh` 是一个专业的 InfiniBand 网络健康检查脚本，基于 Ubuntu 服务器 IB 网络分析报告开发。该脚本专注于**检查和诊断**，不会修改任何系统配置，所有优化建议需要用户手工执行。

## 2. 主要功能

### 2.1 全面检查 (10个核心模块)

- **依赖检查** (`check_dependencies`): 验证必要命令和 MLNX_OFED 驱动包
- **系统信息** (`check_system_info`): 主机名、操作系统、内核版本、CPU、内存、Swap状态
- **硬件检测** (`check_ib_hardware`): PCI 设备、IB 网卡、固件版本、Node GUID
- **驱动验证** (`check_ofed_driver`): OFED 版本、内核模块状态、ConnectX-7兼容性
- **端口状态** (`check_port_status`): 物理连接、链路速率、活跃端口统计
- **网络拓扑** (`check_network_topology`): 网络发现、交换机、子网管理器状态
- **性能计数器** (`check_performance_counters`): 错误计数器、拥塞检测、数据传输统计
- **网络接口** (`check_network_interfaces`): IPoIB 接口、MTU 设置、IP 配置
- **性能工具** (`check_performance_tools`): 测试工具可用性验证
- **系统优化** (`check_system_optimization`): 内核参数、CPU 调节器、NUMA 配置

### 2.2 智能分析

- **状态评估**: 自动判断配置是否符合最佳实践
- **问题识别**: 发现潜在的性能瓶颈和配置问题
- **优化建议**: 提供具体的命令和配置指导（使用独立的建议标识）
- **风险评估**: 按严重程度分类问题（错误/警告/信息/建议）

## 3. 使用方法

### 3.1 基本语法

```bash
sudo ./ib_health_check.sh [选项]
```

### 3.2 命令选项

| 选项 | 说明 |
|------|------|
| `-h, --help` | 显示帮助信息 |
| `-v, --version` | 显示版本信息 |
| `-q, --quiet` | 静默模式（仅输出错误和警告） |
| `-s, --summary` | 摘要模式（仅显示关键检查结果） |
| `--no-color` | 禁用彩色输出 |

### 3.3 运行模式详解

#### 3.3.1 完整模式（默认）

**特点**：

- 执行所有 **10 项检查**（硬件、驱动、网络、优化等）
- 显示详细的检查过程和结果
- 提供完整的优化建议和故障排查指导
- 输出包含所有成功、警告、错误和建议信息

**适用场景**：

- 新环境部署后的全面诊断
- 故障排查和问题定位
- 系统配置验证
- 性能优化评估

**检查项目**：
系统信息 + IB硬件 + OFED驱动 + 端口状态 + 网络拓扑 + 性能计数器 + 网络接口 + 性能工具 + 系统优化 + 优化建议

#### 3.3.2 静默模式（-q）

**特点**：

- 执行所有 **10 项检查**
- 仅显示警告和错误信息，隐藏正常状态
- 无输出表示系统状态良好
- 有输出表示需要关注的问题

**适用场景**：

- 自动化监控脚本
- 定期巡检任务
- 脚本集成和批处理
- 快速问题识别

**输出特点**：

- 🔴 显示错误信息（[FAIL]）
- 🟡 显示警告信息（[WARN]）
- 🟡 显示优化建议（[建议]）
- ✅ 隐藏成功信息（[PASS]）
- ✅ 隐藏一般信息（[INFO]）

#### 3.3.3 摘要模式（-s）

**特点**：

- 仅执行 **4 项关键检查**（依赖、硬件、端口、性能计数器）
- 显示简洁的总结报告
- 专注于网络基本功能验证
- 不包含系统优化检查和建议

**适用场景**：

- 快速健康检查
- 日常巡检的状态概览
- 基本功能验证
- 网络连通性确认

**检查项目**：
依赖检查 + IB硬件 + 端口状态 + 性能计数器

**输出特点**：

- 重点关注硬件状态和网络连接
- 不检查系统优化配置
- 提供简洁的总结信息

### 3.4 模式对比表

| 特性 | 完整模式 | 静默模式 | 摘要模式 |
|------|----------|----------|----------|
| 检查项目数 | 10项 | 10项 | 4项 |
| 输出详细程度 | 完整 | 仅问题 | 简洁总结 |
| 优化建议 | ✅ 包含 | ✅ 包含 | ❌ 不包含 |
| 系统优化检查 | ✅ 包含 | ✅ 包含 | ❌ 不包含 |
| 适用场景 | 全面诊断 | 自动监控 | 快速检查 |
| 执行时间 | 较长 | 较长 | 较短 |

### 3.5 使用示例

#### 3.5.1 完整健康检查

```bash
sudo ./ib_health_check.sh
```

执行所有检查项目，输出详细信息和优化建议。

#### 3.5.2 静默模式检查

```bash
sudo ./ib_health_check.sh -q
```

仅显示警告和错误信息，适合脚本化使用。

#### 3.5.3 摘要模式检查

```bash
sudo ./ib_health_check.sh -s
```

仅执行关键检查（硬件、端口状态、性能计数器），快速了解系统状态。

#### 3.5.4 无彩色输出

```bash
sudo ./ib_health_check.sh --no-color
```

禁用彩色输出，适合日志记录或不支持彩色的终端。

#### 3.5.5 定期监控

```bash
# 添加到 crontab 进行定期检查
sudo crontab -e
# 添加以下行（每天上午8点执行）
0 8 * * * /path/to/ib_health_check.sh -q > /var/log/ib_daily_check.log 2>&1
```

### 3.6 输出示例

#### 3.6.1 完整模式输出示例

```bash
$ sudo ./ib_health_check.sh

=== InfiniBand 健康检查工具 v2.0 ===
检查时间: 2024-01-15 10:30:45
日志文件: /tmp/ib_health_check_20240115_103045.log

[INFO] 开始执行 InfiniBand 健康检查...

=== 1. 依赖检查 ===
[PASS] lspci 命令可用
[PASS] ibstat 命令可用
[PASS] ibv_devinfo 命令可用
[PASS] perfquery 命令可用

=== 2. 系统基本信息 ===
[INFO] 主机名: gpu-node-01
[INFO] 操作系统: Ubuntu 20.04.6 LTS
[INFO] 内核版本: 5.4.0-150-generic
[PASS] 系统负载正常: 0.45

=== 3. InfiniBand 硬件 ===
[PASS] 检测到 1 个 InfiniBand 设备
[INFO] 设备: mlx5_0 (ConnectX-7)
[INFO] 固件版本: 28.38.1002

=== 4. 端口状态 ===
[PASS] 端口 mlx5_0:1 状态: Active (HDR 200 Gb/s)

=== 总结 ===
总检查项目: 10
通过模块: 10/10
警告事件: 17
失败事件: 0

详细日志: /tmp/ib_health_check_20240115_103045.log
```

#### 3.6.2 静默模式输出示例

```bash
$ sudo ./ib_health_check.sh -q

[WARN] 网络内核参数 net.core.rmem_max 当前值: 212992, 建议值: 134217728
[WARN] 网络内核参数 net.core.wmem_max 当前值: 212992, 建议值: 134217728
[WARN] CPU调节器未设置为性能模式 (当前: powersave)
[WARN] IPoIB接口 ib0 MTU为数据报模式 (2044), 建议使用连接模式 (65520)
[建议] 检测到IPoIB接口，建议优化网络内核参数
[建议] CPU调节器建议设置为性能模式以获得最佳性能
[建议] 建议配置NUMA亲和性以优化性能
```

#### 3.6.3 摘要模式输出示例

```bash
$ sudo ./ib_health_check.sh -s

=== InfiniBand 健康检查摘要 ===
检查时间: 2024-01-15 10:35:20

=== 关键检查结果 ===
✅ 依赖检查: 通过
✅ IB硬件: 1个设备正常
✅ 端口状态: 1个端口活跃 (HDR)
✅ 性能计数器: 无错误

=== 总结 ===
总检查项目: 4
通过模块: 4/4
警告事件: 0
失败事件: 0

InfiniBand 网络基本功能正常
详细日志: /tmp/ib_health_check_20240115_103520.log
```

## 4. 输出说明

### 4.1 状态标识

- 🟢 **[PASS]**: 检查通过，配置正确
- 🟡 **[WARN]**: 警告，建议优化（计入警告事件统计）
- 🔴 **[FAIL]**: 错误，需要立即处理（计入失败事件统计）
- 🔵 **[INFO]**: 信息，供参考
- 🟡 **[建议]**: 优化建议，不计入警告统计

### 4.2 计数逻辑说明

脚本采用模块化计数方式：

- **总检查项目**: 10个功能模块（固定数量）
- **通过模块**: 总模块数 - 失败模块数
- **警告事件**: 具体的警告数量（可能超过模块数）
- **失败事件**: 具体的失败数量

**示例输出**:

```bash
总检查项目: 10
通过模块: 10/10
警告事件: 21
失败事件: 0
```

这表示所有10个检查模块都通过了（没有失败），但在检查过程中发现了21个需要关注的警告事件。

**重要说明**:

- 一个检查模块可能产生多个警告事件，这是正常现象
- 通过模块数 = 总模块数 - 失败模块数（不受警告数量影响）
- 优化建议使用 `[建议]` 标识，不计入警告统计
- 关注重点：优先处理失败事件，然后关注警告事件

### 4.3 退出码

- `0`: 所有检查通过（无错误且无警告）
- `1`: 发现严重错误（有失败事件）
- `2`: 发现警告项目（无失败事件但有警告事件）

### 4.4 日志文件

脚本会自动生成详细日志文件：

```text
/tmp/ib_health_check_YYYYMMDD_HHMMSS.log
```

## 5. 检查项目详解 (10个核心模块)

### 5.1 依赖检查 (`check_dependencies`)

- 验证必要命令可用性：`lspci`、`ibstat`、`ibv_devinfo`、`perfquery`、`ibnetdiscover`、`ofed_info`
- 检查 MLNX_OFED 驱动包安装状态
- 确保后续检查能够正常执行

### 5.2 系统基本信息 (`check_system_info`)

- 主机名、操作系统、内核版本、架构
- CPU 核心数、型号、内存使用情况
- 系统负载、Swap 状态检查
- HPC 环境最佳实践验证

### 5.3 InfiniBand 硬件 (`check_ib_hardware`)

- PCI 设备检测和识别
- IB 网卡型号、数量统计
- 固件版本、Node GUID 信息
- 设备状态和端口初步检查

### 5.4 OFED 驱动 (`check_ofed_driver`)

- OFED 版本检查和兼容性验证
- 关键内核模块加载状态：`mlx5_core`、`mlx5_ib`、`ib_core`、`ib_ipoib`、`ib_uverbs`、`rdma_cm`
- ConnectX-7 支持验证
- 驱动版本推荐检查

### 5.5 端口状态 (`check_port_status`)

- 物理连接状态检查
- 链路速率分析（HDR/EDR/FDR）
- 端口活跃状态统计
- 物理状态验证（LinkUp/LinkDown）

### 5.6 网络拓扑 (`check_network_topology`)

- 网络发现和节点统计
- 交换机检测和冗余分析
- 子网管理器状态检查
- 网络拓扑完整性验证

### 5.7 性能计数器 (`check_performance_counters`)

- 错误计数器监控：符号错误、链路错误、接收错误
- 网络拥塞检测：XmitWait 计数器
- 数据传输统计：发送/接收包统计
- 链路质量评估

### 5.8 网络接口 (`check_network_interfaces`)

- IPoIB 接口配置检查
- MTU 设置验证（数据报模式 2044 vs 连接模式 65520）
- IP 地址配置状态
- 接口状态监控

### 5.9 性能工具 (`check_performance_tools`)

- 性能测试工具可用性检查：
  - **带宽测试工具**：`ib_send_bw`、`ib_write_bw`、`ib_read_bw`
  - **延迟测试工具**：`ib_send_lat`、`ib_write_lat`、`ib_read_lat`
- 检查每个工具的可用性状态（可用/不可用）
- 统计可用和缺失的性能测试工具
- 提供 perftest 包安装建议（当有工具缺失时）
- 基准测试工具完整性验证

### 5.10 系统优化 (`check_system_optimization`)

- 网络内核参数检查：`net.core.rmem_max`、`net.core.wmem_max`
- CPU 频率调节器设置
- NUMA 配置和亲和性检查
- HPC 环境优化验证

---

## 6. 优化建议说明

脚本会根据检查结果提供以下类型的建议：

### 6.1 配置优化

- **MTU 设置优化**：推荐从数据报模式 2044 调整为连接模式 65520
  - **适用场景**：仅适用于 IPoIB 网络接口
  - **注意事项**：连接模式需要特殊配置支持
- **网络内核参数调优**：优化 `net.core.rmem_max` 和 `net.core.wmem_max` 参数
  - **适用场景**：仅适用于 IPoIB 场景（IP over InfiniBand）
  - **不适用场景**：GPU 直接使用 IB 网络（RDMA）时无需修改这些参数
- **CPU 性能模式设置**：建议使用 `performance` 调节器
  - **适用场景**：所有高性能计算场景（包括 GPU 计算、IPoIB 网络等）

### 6.2 性能调优

- **NUMA 亲和性配置**：提供 NUMA 节点绑定建议
  - **适用场景**：所有高性能计算场景（GPU 计算、IB 网络、高性能应用）
  - **包含内容**：IB 设备、GPU 设备、CPU 的 NUMA 节点检查
- **应用程序绑定建议**：CPU 和内存亲和性优化
- **性能基线测试指导**：带宽和延迟测试命令

### 6.3 监控维护

- **定期检查脚本设置**：crontab 配置示例
- **实时监控命令**：性能计数器和端口状态监控
- **故障排查工具**：诊断和修复命令

### 6.4 工具安装

- **性能测试工具**：perftest 包安装建议
- **诊断工具**：ibdiagnet、ibping 等工具推荐
- **监控脚本**：自动化监控解决方案

### 6.5 重要说明

- **脚本仅提供建议，不执行任何修改操作**
- **所有优化命令需要用户手工执行**
- **建议在测试环境中验证配置更改**
- **优化建议使用 `[建议]` 标识，不计入警告统计**
- **网络内核参数优化仅在检测到 IPoIB 接口时提供**

### 6.6 建议输出示例

```bash
[建议] MTU为数据报模式标准值，建议优化为连接模式65520
优化命令:
  sudo ip link set ib0 mtu 65520
  
[建议] CPU调节器未设置为性能模式 (当前: powersave)
适用场景: 所有高性能计算场景 (包括GPU计算、IPoIB网络等)
优化命令:
  # 临时设置 (重启后失效)
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  # 永久设置 (安装cpufrequtils)
  sudo apt install cpufrequtils
  echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
  sudo systemctl restart cpufrequtils

[建议] 网络内核参数未优化 (适用于IPoIB场景)
适用场景: 仅当使用IPoIB网络接口时需要优化
注意: 如果GPU直接使用IB网络(RDMA)，则无需修改这些参数
优化命令 (仅在使用IPoIB时执行):
  # 创建优化配置文件
  sudo tee /etc/sysctl.d/99-ipoib-network.conf << EOF
# IPoIB 网络优化参数 (仅适用于IP over InfiniBand)
net.core.rmem_max = 268435456
net.core.wmem_max = 268435456
net.core.rmem_default = 67108864
net.core.wmem_default = 67108864
net.core.netdev_max_backlog = 5000
EOF
  # 应用配置
  sudo sysctl -p /etc/sysctl.d/99-ipoib-network.conf

内核参数优化建议:
  未检测到IPoIB接口，无需修改网络内核参数
  注意: 如果GPU直接使用IB网络(RDMA)，内核网络参数优化无效

NUMA亲和性优化建议:
适用场景: 所有高性能计算场景 (GPU计算、IB网络、高性能应用)
  # 检查IB设备的NUMA节点
  cat /sys/class/infiniband/*/device/numa_node
  # 检查GPU设备的NUMA节点 (如果有GPU)
  nvidia-smi topo -m
  # 检查CPU NUMA拓扑
  numactl --hardware
  # 绑定应用到特定NUMA节点 (示例)
  numactl --cpunodebind=0 --membind=0 your_application
  # GPU应用NUMA绑定示例
  numactl --cpunodebind=0 --membind=0 python gpu_training.py
```

---

## 7. 故障排查

### 7.1 常见问题

#### 7.1.1 权限不足

```bash
错误: 此脚本需要root权限运行
解决: 使用 sudo 运行脚本
```

#### 7.1.2 命令未找到

```bash
错误: 命令 ibstat 未找到
解决: 安装 MLNX_OFED 驱动包
```

#### 7.1.3 网络拓扑发现失败

```bash
错误: 网络拓扑发现失败或超时
原因: 网络连接问题或子网管理器未运行
解决: 检查网络连接和子网管理器状态
```

---

## 8. 最佳实践

### 8.1 定期检查

建议每日运行健康检查，及时发现问题：

```bash
# 添加到 crontab 进行定期检查
sudo crontab -e
# 添加以下行（每天上午8点执行）
0 8 * * * /path/to/ib_health_check.sh -q > /var/log/ib_daily_check.log 2>&1
```

### 8.2 基线建立

首次部署后建立性能基线，便于后续对比：

```bash
# 完整检查并保存基线
sudo ./ib_health_check.sh > /var/log/ib_baseline_$(date +%Y%m%d).log 2>&1

# 性能基线测试
ib_send_bw -d mlx5_0 --report_gbits > /var/log/ib_bandwidth_baseline.log
ib_send_lat -d mlx5_0 > /var/log/ib_latency_baseline.log
```

### 8.3 变更管理

配置变更前后都要运行检查，确保变更效果：

```bash
# 变更前检查
sudo ./ib_health_check.sh -s > /var/log/ib_pre_change.log

# 执行变更操作
# ...

# 变更后检查
sudo ./ib_health_check.sh -s > /var/log/ib_post_change.log

# 对比结果
diff /var/log/ib_pre_change.log /var/log/ib_post_change.log
```

### 8.4 文档记录

保存检查日志和优化记录，建立运维档案：

```bash
# 创建运维日志目录
sudo mkdir -p /var/log/ib_health/

# 定期归档日志
sudo find /tmp -name "ib_health_check_*.log" -mtime +7 -exec mv {} /var/log/ib_health/ \;
```

### 8.5 理解计数逻辑

正确理解脚本的计数逻辑：

- **检查模块 vs 警告事件**：10个检查模块是固定的功能单元，警告事件是具体的问题数量
- **通过模块计算**：通过模块 = 总模块 - 失败模块（不受警告数量影响）
- **警告事件意义**：一个模块可能产生多个警告，这是正常现象
- **关注重点**：优先关注失败事件，然后处理警告事件

---
