# 07 — npu-smi 使用参考

`npu-smi` v24.1.0.3 是昇腾 NPU 的轻量级设备管理工具，对应 NVIDIA 生态中的 `nvidia-smi`。它不依赖 CANN toolkit，直接通过驱动层获取设备状态，因此即使在 CANN 未安装或配置错误的场景下仍可正常工作。适合日常巡检、状态监控和进程管理。

## 1. 命令结构

`npu-smi` 采用 `npu-smi info <子命令>` 的二级参数结构：

```text
npu-smi                           # 等价于 npu-smi info（默认输出）
npu-smi info                      # 所有设备概览 + 进程表
npu-smi info -l                   # 卡间拓扑
npu-smi info -m                   # Chip ID 映射
npu-smi info -t <type> [-i <id>]  # 指定类型详细信息
npu-smi info watch                # 滚动刷新模式
```

`-t` 支持 60+ 种查询类型，`-i` 指定 NPU 编号（0-7），`-c` 指定 Chip ID（默认 0 为 AI 芯片，1 为 MCU 管理芯片）。

## 2. 默认输出解读

执行 `npu-smi` 或 `npu-smi info`，输出分为上下两部分：设备概览表 + 进程表。

```text
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.0.3                 Version: 24.1.0.3                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B3               | OK            | 92.9        37                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          3379 / 65536         |
+---------------------------+---------------+----------------------------------------------------+
...
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
+===========================+===============+====================================================+
```

上半部分每行对应一个 Chip：第一行为 AI 芯片（Chip ID=0），字段含义如下：

| 字段             | 含义           | 解读                                                                                         |
| ---------------- | -------------- | -------------------------------------------------------------------------------------------- |
| Name             | 芯片型号       | 910B3 / 910B2 等                                                                             |
| Health           | 健康状态       | OK 正常；若有异常则显示具体告警码                                                            |
| Power(W)         | 实时功耗       | 空闲 ~93 W；FP16 算力测试满载 ~231 W；实际训练可达 ~273 W（此前观测到 5 号卡训练时达此功耗） |
| Temp(C)          | 芯片温度       | 正常范围 30-60°C；持续 >85°C 触发降频                                                        |
| Hugepages-Usage  | 大页内存       | 显示已用/总量（page）；本服务器为 0，表示未使用 DDR 大页                                     |
| Bus-Id           | PCIe 总线地址  | 格式 `domain:bus:device.function`，可据此定位物理插槽                                        |
| AICore(%)        | AI Core 利用率 | 0% 表示空闲；若持续为 0 但进程存在，可能进程在等待数据                                       |
| Memory-Usage(MB) | DDR 内存使用   | 已用/总量；本服务器 DDR 容量为 0，因此始终为 0/0                                             |
| HBM-Usage(MB)    | HBM 使用量     | 已用/总量；空闲时约 3300-3400 MB 为驱动保留（约 5%），65536 MB 为 64 GB 总量                 |

下半部分为进程表，列出每张 NPU 上运行的进程 PID、进程名和 NPU 内存占用。默认模式只显示当前时刻快照。

## 3. 实时监控 (`watch` / `proc`)

```bash
npu-smi info watch          # 设备概览滚动刷新（类似 nvidia-smi -l 1）
npu-smi info proc           # 进程表滚动刷新
```

`watch` 和 `proc` 以 1 秒为间隔持续刷新，Ctrl+C 退出。适合在训练任务启动后观察 NPU 状态变化——例如确认 AI Core 利用率是否从 0% 上升到预期值，或监控温度上升趋势。

## 4. 拓扑查询 (`-l` / `-m`)

### 卡间拓扑 (`-l`)

```bash
npu-smi info -l
```

输出 8×8 矩阵，每格表示两卡间的连接类型：

| 标识 | 含义                                                 |
| ---- | ---------------------------------------------------- |
| X    | 自身                                                 |
| HCCS | 专用直连链路（Huawei Cache Coherence System）        |
| SYS  | 经过 PCIe + NUMA 互联（跨 SMP 系统总线，如 QPI/UPI） |
| PHB  | 经过 PCIe Host Bridge                                |
| PIX  | 经过单个 PCIe Switch                                 |
| PXB  | 经过多个 PCIe Switch                                 |
| NA   | 无法判定                                             |

本服务器 8 卡之间全部为 HCCS 直连（8 卡全互联 full mesh），不存在经过 PCIe 或 NUMA 的间接链路。这是训练集群的理想拓扑——任何两卡之间的 AllReduce 延迟完全对称，不存在跨 NUMA node 的通信瓶颈。

### Chip ID 映射 (`-m`)

```bash
npu-smi info -m
```

每张物理 NPU 包含两个 Chip：Chip ID=0 为 AI 芯片（Ascend 910B3），Chip ID=1 为 MCU（管理控制单元）。Chip Logic ID 是对用户可见的逻辑编号（0-7），即 `-i` 参数中使用的设备号。

## 5. 常用查询类型速查

以下为本服务器实测数据的精选子集，按使用频率排序。

### 5.1 资源使用 (`-t usages`)

```bash
npu-smi info -t usages -i 7
```

| 字段                                     | 含义                                                       |
| ---------------------------------------- | ---------------------------------------------------------- |
| `HBM Capacity(MB)` / `HBM Usage Rate(%)` | 64 GB 总量，Usage Rate 为使用百分比（空闲 ~5% 为驱动保留） |
| `Aicore Usage Rate(%)`                   | AI Core 利用率；0% 表示空闲，训练中可达 60-99%             |
| `Aivector Usage Rate(%)`                 | Vector Unit 利用率；通常与 AI Core 利用率同步变化          |
| `Aicpu Usage Rate(%)`                    | AI CPU 利用率；处理控制流和非矩阵算子                      |
| `Ctrlcpu Usage Rate(%)`                  | 管理 CPU 利用率；负责设备调度和通信                        |
| `HBM Bandwidth Usage Rate(%)`            | HBM 带宽利用率；memory-bound 算子可使此值接近 100%         |

判断 NPU 是否可用的核心命令：确认 Usage Rate 为 0% 或接近 0%，且有足够 HBM 余量。

### 5.2 健康与错误 (`-t health` / `-t ecc` / `-t pcie-err`)

```bash
npu-smi info -t health -i 7     # 健康状态：OK / 异常告警码
npu-smi info -t ecc -i 7        # HBM ECC 错误详情
npu-smi info -t pcie-err -i 7   # PCIe 链路错误计数器
```

**ECC 输出字段解读**：

| 字段                       | 含义                 | 告警阈值               |
| -------------------------- | -------------------- | ---------------------- |
| HBM Single Bit Error Count | 可纠正单比特错误累计 | > 1000/day 需关注      |
| HBM Double Bit Error Count | 不可纠正双比特错误   | 任何 >0 都需要硬件检查 |
| Isolated Pages Count       | 已被驱动隔离的坏页数 | 持续增长表示 HBM 退化  |

本服务器所有卡 ECC 计数均为 0，PCIe TX/RX/LCRC/ECRC/Retry 计数均为 0，硬件状态健康。

### 5.3 物理信息 (`-t board` / `-t memory`)

```bash
npu-smi info -t board -i 7      # 卡级别硬件信息
npu-smi info -t memory -i 7     # 内存规格
```

**`board` 关键字段**：

| 字段                 | 示例值          | 用途                                   |
| -------------------- | --------------- | -------------------------------------- |
| Product Name         | IT21HMDC_Bin6   | 产品型号（IT21 表示训练卡）            |
| Serial Number        | 1025A6785593    | 序列号，用于资产管理                   |
| PCI Vendor/Device ID | 0x19E5 / 0xD802 | 华为 PCI Vendor ID，lspci 中可据此定位 |
| Software Version     | 24.1.0.3        | 驱动版本                               |
| Firmware Version     | 7.5.0.5.220     | 固件版本                               |

**`memory` 关键字段**：

| 字段                | 值       | 说明                                |
| ------------------- | -------- | ----------------------------------- |
| HBM Capacity        | 65536 MB | 64 GB HBM2e                         |
| HBM Clock Speed     | 1600 MHz | 1.6 GHz，对应 ~1.54 TB/s 实测带宽   |
| HBM Temperature     | 38°C     | 仅 HBM 颗粒温度，不同于芯片整体温度 |
| HBM Manufacturer ID | 0x57     | Samsung                             |
| DDR Capacity        | 0 MB     | 本服务器 DDR 不可用                 |

### 5.4 HCCS 链路状态 (`-t hccs` / `-t hccs-bw`)

```bash
npu-smi info -t hccs -i 7 -c 0    # HCCS 链路详情（需同时指定 -i 和 -c）
npu-smi info -t hccs-bw -i 7 -c 0 -time 100  # HCCS 带宽实时探测（100 ms）
```

**`hccs` 输出解读**（8 个 lane 按数组格式排列，`[lane0 lane1 ... lane7]`）：

| 字段                     | 本服务器值          | 说明                           |
| ------------------------ | ------------------- | ------------------------------ |
| hccs health status       | OK                  | 链路健康                       |
| hccs lane mode           | `[4 4 4 4 4 4 4 4]` | 每个 lane 模式为 4             |
| hccs link lane list      | `[1111 1111 ...]`   | 4 个 lane 全部在线（1=active） |
| hccs link speed          | `[224 224 ...]`     | 每 lane 速率 224 Gbps          |
| hccs retry / error count | 全部 0              | 链路无误码                     |

lane mode 值 4 表示 4 条物理 lane 组网，link speed 224 表示单 lane 224 Gbps，合计每条 HCCS 链路带宽 = 4 × 224 Gbps = 896 Gbps ≈ 112 GB/s（全双工）。但实际 P2P 有效带宽（ascend-dmi 实测 ~26.2 GB/s）受限于协议开销、数据打包效率和内存带宽，与链路物理带宽差距较大。

> [!TIP]
> **`hccs` vs `ascend-dmi --bw -t p2p`**：`hccs` 查看物理链路层的原始状态（lane 数、速率、误码），`ascend-dmi --bw -t p2p` 测量应用层的有效 P2P 带宽。如果 P2P 带宽异常偏低，先用 `hccs` 检查 lane 是否全部在线、是否有 retry/error 计数增长。

### 5.5 温度与功耗 (`-t temp` / `-t power` / `-t volt`)

```bash
npu-smi info -t temp -i 7       # 当前温度
npu-smi info -t power -i 7      # 实时功耗
npu-smi info -t volt -i 7       # 当前电压
```

温度输出包含 NPU 芯片温度（~40°C 空闲，~60°C 满载）和 MCU 温度。LM75A_TE/LM75B_TE 为板上 LM75 温度传感器的读数，用于交叉校验。持续 >85°C 会触发频率限制。

### 5.6 综合信息 (`-t common`)

```bash
npu-smi info -t common -i 7
```

将 `usages` + `temp` + `power` 的核心字段合并为一次查询，适合脚本中一键获取关键指标。额外提供 AI Core 标称频率（1800 MHz）和当前实际频率（800 MHz 空闲，1800 MHz 满载），可用于确认是否因温度或功耗限制而降频。

## 6. `npu-smi` vs `ascend-dmi` 分工对照

| 维度     | `npu-smi`               | `ascend-dmi`                     |
| -------- | ----------------------- | -------------------------------- |
| 启动速度 | < 0.5 s                 | 2-5 s（需加载 DCMI 和 AscendCL） |
| 依赖     | 仅驱动                  | 驱动 + CANN toolkit              |
| 核心用途 | 实时状态、进程、拓扑    | 性能基线、压测、故障诊断         |
| 算力测试 | 不提供                  | `-f` 实测 TFLOPS                 |
| 带宽测试 | `-t hccs-bw`（仅 HCCS） | `--bw`（HBM/PCIe/HCCS 全覆盖）   |
| ECC 详情 | `-t ecc` 完整统计       | `--info --detail` 中有汇总       |
| 输出格式 | 纯文本                  | 支持 `--fmt json`                |
| 最佳场景 | 日常巡检、确认卡状态    | 装机验收、性能基准、硬件排障     |

日常流程：`npu-smi` 快速确认所有卡 OK → 发现异常时用 `npu-smi info -t <type> -i <id>` 深入查看 → 需要定量诊断或压测时切到 `ascend-dmi`（详见 `02_ascend_dmi_reference.md`）。

## 7. 参考链接

- [CANN 文档 — npu-smi 工具](https://www.hiascend.com/document/detail/en/canncommercial/800/devtool/npu/npu_0001.html)
