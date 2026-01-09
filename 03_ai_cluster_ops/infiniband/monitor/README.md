# InfiniBand 网络带宽监控脚本使用说明

## 1. 概述与功能特性

`ib_bandwidth_monitor.sh` 是一个专门用于监控和计算 InfiniBand 网络实际传输速率的脚本。它通过读取 InfiniBand 设备的性能计数器，实时计算网络的发送和接收速率，并以 SAR 风格格式输出。

### 1.1 核心功能

- **实时速率监控**: 计算实际的发送/接收速率 (MB/s)
- **链路利用率**: 显示当前速率相对于链路最大速率的利用率
- **性能计数器**: 监控数据包统计和传输速率
- **SAR 风格输出**: 固定使用 SAR 格式输出，便于系统监控集成
- **多端口支持**: 同时监控多个端口或端口范围
- **多设备支持**: 支持同时监控多个设备的多个端口
- **端口发现**: 自动列出所有可监控的 InfiniBand 端口
- **自动设备检测**: 自动发现可用的 InfiniBand 设备
- **时间控制监控**: 支持指定监控时长或无限监控模式
- **智能错误处理**: 多种 perfquery 调用方式，提高兼容性
- **彩色日志输出**: 支持彩色日志和静默模式

### 1.2 监控指标

- 实时传输速率 (发送/接收 MB/s)
- 数据包速率 (发送/接收 包/秒)
- 链路利用率百分比
- 累积数据传输量
- 数据包统计

---

## 2. 安装要求与依赖

### 2.1 系统要求

- Linux 操作系统
- InfiniBand 硬件和驱动
- 管理员权限 (用于重置计数器)

### 2.2 依赖软件安装

```bash
# Ubuntu/Debian
sudo apt install infiniband-diags perftest bc

# RHEL/CentOS/Rocky Linux
sudo yum install infiniband-diags perftest bc
```

### 2.3 必要命令

- `perfquery`: 查询性能计数器
- `ibstat`: 查询设备状态
- `ibv_devinfo`: 设备信息 (可选，用于设备发现)
- `bc`: 数学计算
- `timeout`: 命令超时控制 (通常系统自带)

---

## 3. 使用方法与示例

### 3.1 基本语法

```bash
./ib_bandwidth_monitor.sh [选项]
```

### 3.2 命令行选项

| 选项 | 长选项 | 参数 | 说明 |
|------|--------|------|------|
| `-h` | `--help` | - | 显示帮助信息 |
| `-v` | `--version` | - | 显示版本信息 |
| `-d` | `--device` | DEVICE | 指定设备名称 (如: mlx5_0) |
| `-p` | `--port` | PORT | 指定端口号 (默认: 1) |
| | `--ports` | PORTS | 指定多个端口 (如: 1,2,3 或 1-4) |
| | `--multi-devices` | CONFIG | 指定多设备多端口配置 (如: mlx5_0:1,2;mlx5_1:1,3) |
| | `--list-ports` | - | 列出所有可监控的端口 |
| `-i` | `--interval` | SECONDS | 监控间隔秒数 (默认: 5) |
| `-t` | `--time` | SECONDS | 监控总时长秒数 (默认: 0，表示无限监控) |
| `-q` | `--quiet` | - | 静默模式 (仅输出结果) |

### 3.3 使用示例

#### 3.3.1 端口发现和基本监控

```bash
# 列出所有可监控的端口
./ib_bandwidth_monitor.sh --list-ports

# 自动检测设备并监控（无限监控）
./ib_bandwidth_monitor.sh

# 指定设备和端口监控
./ib_bandwidth_monitor.sh -d mlx5_0 -p 1

# 监控指定时长
./ib_bandwidth_monitor.sh -d mlx5_0 -p 1 -t 60
```

#### 3.3.2 多端口监控

```bash
# 同时监控多个指定端口
./ib_bandwidth_monitor.sh -d mlx5_0 --ports 1,2,3

# 监控端口范围
./ib_bandwidth_monitor.sh -d mlx5_0 --ports 1-4

# 混合指定端口和范围
./ib_bandwidth_monitor.sh -d mlx5_0 --ports 1,3-5,8
```

#### 3.3.3 多设备多端口监控

```bash
# 监控多个设备的多个端口
./ib_bandwidth_monitor.sh --multi-devices "mlx5_0:1,2;mlx5_1:1,3"

# 复杂的多设备配置
./ib_bandwidth_monitor.sh --multi-devices "mlx5_0:1-3;mlx5_1:2,4;mlx5_2:1"

# 多设备监控指定时长
./ib_bandwidth_monitor.sh --multi-devices "mlx5_0:1,2;mlx5_1:1" -t 300
```

#### 3.3.4 自定义监控参数

```bash
# 监控 2 分钟，每 10 秒采样一次
./ib_bandwidth_monitor.sh -d mlx5_0 -i 10 -t 120

# 高频率监控（1秒间隔）
./ib_bandwidth_monitor.sh -d mlx5_0 -i 1 -t 60

# 静默模式输出（仅数据）
./ib_bandwidth_monitor.sh -d mlx5_0 -q
```

#### 3.3.5 数据收集和分析

```bash
# 长时间监控并保存数据
./ib_bandwidth_monitor.sh -d mlx5_0 -q -t 3600 > bandwidth_data.txt

# 短间隔高精度监控
./ib_bandwidth_monitor.sh -i 1 -t 300 -q

# 多端口数据收集
./ib_bandwidth_monitor.sh -d mlx5_0 --ports 1-4 -q -t 1800 > multi_port_data.txt
```

---

## 4. 输出格式与性能计算

### 4.1 SAR 格式输出

脚本固定使用 SAR 风格的输出格式，便于与系统监控工具集成：

```bash
时间            接口               rxpck/s   txpck/s    rxMB/s    txMB/s   %ifutil
14:30:25        mlx5_0:1              1234      1098   5654.32   4765.43     45.23
14:30:30        mlx5_0:1              1245      1109   5765.43   4876.54     46.12
14:30:35        mlx5_0:1              1256      1120   5876.54   4987.65     47.01
```

### 4.2 输出字段说明

| 字段 | 说明 |
|------|------|
| 时间 | 采样时间戳 (HH:MM:SS) |
| 接口 | 设备名:端口号 (如: mlx5_0:1) |
| rxpck/s | 接收包速率 (包/秒) |
| txpck/s | 发送包速率 (包/秒) |
| rxMB/s | 接收数据速率 (MB/秒) |
| txMB/s | 发送数据速率 (MB/秒) |
| %ifutil | 链路利用率百分比 |

### 4.3 多设备输出示例

```bash
时间            接口               rxpck/s   txpck/s    rxMB/s    txMB/s   %ifutil
14:30:25        mlx5_0:1              1234      1098   5654.32   4765.43     45.23
14:30:25        mlx5_0:2               987       876   4321.10   3654.21     35.67
14:30:25        mlx5_1:1              1567      1432   6789.45   5432.10     52.34
14:30:30        mlx5_0:1              1245      1109   5765.43   4876.54     46.12
14:30:30        mlx5_0:2               998       887   4432.21   3765.32     36.45
14:30:30        mlx5_1:1              1578      1443   6890.56   5543.21     53.12
```

### 4.4 性能计算原理

#### 4.4.1 带宽计算公式

```bash
实际速率 (MB/s) = (当前计数器 - 前次计数器) × 4字节 / 时间差 / (1024 * 1024)
```

#### 4.4.2 包速率计算

```bash
包速率 (包/s) = (当前包计数器 - 前次包计数器) / 时间差
```

#### 4.4.3 利用率计算

```bash
利用率 (%) = max(发送速率, 接收速率) / 链路最大速率 × 100
链路最大速率 (MB/s) = 链路速率 (Gbps) × 125
```

#### 4.4.4 关键计数器说明

- **PortXmitData**: 发送的数据量 (4字节单位)
- **PortRcvData**: 接收的数据量 (4字节单位)
- **PortXmitPkts**: 发送的数据包数量
- **PortRcvPkts**: 接收的数据包数量

#### 4.4.5 计数器回绕处理

脚本支持32位和64位计数器的回绕处理：

- 首先尝试64位回绕检测
- 如果64位回绕无效，则使用32位回绕
- 确保计数器溢出时数据的连续性

---

## 5. 故障排除与优化

### 5.1 常见错误及解决方法

#### 5.1.1 未检测到 InfiniBand 设备

```bash
[ERROR] 未检测到 InfiniBand 设备
```

**解决方法**:

- 检查 InfiniBand 硬件是否正确安装
- 确认驱动程序已加载: `lsmod | grep ib`
- 检查设备状态: `ibstat -l`
- 加载驱动: `sudo modprobe mlx5_ib`

#### 5.1.2 端口状态不是 Active

```bash
[ERROR] 端口 mlx5_0:1 状态不是 Active (当前: Down)
```

**解决方法**:

- 检查网络连接
- 确认对端设备状态
- 检查端口配置: `ibstat mlx5_0 1`
- 检查子网管理器是否运行

#### 5.1.3 无法获取性能计数器

```bash
[ERROR] 无法获取性能计数器
```

**解决方法**:

脚本会自动尝试多种 perfquery 调用方式：

1. 使用 CA 名称和端口: `perfquery -C device -P port`
2. 使用 LID: `perfquery lid port`

如果所有方式都失败：

- 确认有足够权限 (尝试使用 sudo)
- 检查 perfquery 命令可用性
- 确认设备和端口参数正确
- 检查 InfiniBand 子网管理器状态

#### 5.1.4 缺少必要命令

```bash
[ERROR] 缺少必要的命令: perfquery ibstat
```

**解决方法**:

- 安装 InfiniBand 工具包
- Ubuntu/Debian: `sudo apt install infiniband-diags bc`
- RHEL/CentOS: `sudo yum install infiniband-diags bc`

#### 5.1.5 权限问题

```bash
[ERROR] Permission denied
```

**解决方法**:

- 使用 sudo 运行脚本: `sudo ./ib_bandwidth_monitor.sh`
- 将用户添加到 rdma 组: `sudo usermod -a -G rdma $USER`
- 重新登录以使组权限生效

#### 5.1.6 多设备配置错误

```bash
[ERROR] 无效的设备配置格式
```

**解决方法**:

- 检查配置格式: `device1:port1,port2;device2:port3,port4`
- 确保设备名称正确
- 确保端口号在有效范围内
- 使用 `--list-ports` 查看可用端口

### 5.2 信号处理和清理

脚本支持优雅的信号处理机制：

- **Ctrl+C (SIGINT)**: 优雅停止监控，显示停止信息
- **SIGTERM**: 终止信号处理，用于系统关闭或进程管理
- **自动清理**: 监控停止时自动清理资源和显示状态

```bash
# 正常停止示例
$ ./ib_bandwidth_monitor.sh -d mlx5_0
^C
监控已停止
```

### 5.3 性能优化建议

#### 5.3.1 监控间隔选择

- **高精度监控**: 1-2 秒间隔 (短期测试)
- **常规监控**: 5-10 秒间隔 (日常监控)
- **长期监控**: 30-60 秒间隔 (趋势分析)

#### 5.3.2 时间控制

- 使用 `-t` 选项指定监控时长，避免无限监控占用资源
- 对于长期监控，建议使用较大的间隔值
- 使用静默模式 (`-q`) 减少输出开销

#### 5.3.3 多设备监控

- 合理配置多设备监控，避免过多端口同时监控
- 考虑系统负载，适当调整监控间隔
- 使用静默模式输出到文件进行数据分析

---

## 6. 集成和自动化

### 6.1 与监控系统集成

```bash
# 数据收集示例 - 输出到文件
./ib_bandwidth_monitor.sh -d mlx5_0 -q -t 300 > /var/log/ib_bandwidth.log

# 实时监控集成
./ib_bandwidth_monitor.sh -d mlx5_0 -q -i 10 | \
  while read line; do
    echo "$(date): $line" >> /var/log/ib_monitoring.log
  done
```

### 6.2 定时监控

```bash
# 添加到 crontab
# 每5分钟记录一次带宽数据（监控1分钟）
*/5 * * * * /path/to/ib_bandwidth_monitor.sh -d mlx5_0 -q -t 60 >> /var/log/ib_bandwidth.log
```

### 6.3 告警脚本

```bash
#!/bin/bash
# 带宽告警脚本示例
THRESHOLD=80  # 利用率阈值 80%

./ib_bandwidth_monitor.sh -d mlx5_0 -q -t 30 | \
  tail -n +2 | \  # 跳过表头
  while read timestamp interface rxpps txpps rxkbps txkbps util; do
    if (( $(echo "$util > $THRESHOLD" | bc -l) )); then
      echo "WARNING: High bandwidth utilization on $interface: ${util}%"
      # 发送告警通知
    fi
  done
```

### 6.4 多设备监控脚本

```bash
#!/bin/bash
# 多设备监控脚本
DEVICES_CONFIG="mlx5_0:1,2;mlx5_1:1,3;mlx5_2:1"
LOG_FILE="/var/log/multi_device_ib.log"

./ib_bandwidth_monitor.sh --multi-devices "$DEVICES_CONFIG" -q -t 3600 >> "$LOG_FILE"
```

---
