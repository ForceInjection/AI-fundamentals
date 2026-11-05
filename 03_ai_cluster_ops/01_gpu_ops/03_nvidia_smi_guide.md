# nvidia-smi 场景速查

`nvidia-smi` 有 100+ 个选项，但真正遇到问题时你只需要 5-6 个。本文按运维场景组织：从「看一眼」到「改配置」到「脚本化」，每个场景给出最精简的命令和输出解读。完整选项参考 `nvidia-smi -h`。

---

## 一、快速扫一眼：「有什么 GPU？状态怎么样？」

```bash
nvidia-smi                          # 概要
nvidia-smi -L                       # 仅型号 + UUID
```

`nvidia-smi` 默认输出的每一列：

```text
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:81:00.0 Off |                    0 |
| N/A   40C    P8              10W /  70W |      2MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

| 列                     | 含义                             | 关注点                                      |
| ---------------------- | -------------------------------- | ------------------------------------------- |
| `Fan`                  | 风扇转速百分比，`N/A` = 被动散热 | —                                           |
| `Temp`                 | GPU 核心温度 (°C)                | T4 < 75°C，H100 < 70°C（满载时）            |
| `Perf`                 | P0-P12 性能状态，P0 = 最高频率   | 如果一直是 P8，GPU 没有计算负载             |
| `Pwr:Usage/Cap`        | 当前功耗 / 功耗上限 (W)          | Usage 远小于 Cap = 没跑满                   |
| `Memory-Usage`         | 已用 / 总显存                    | OOM 前兆：`Used` 接近 `Total`               |
| `GPU-Util`             | GPU 利用率（kernel 执行时间占比） | ≠ 算力利用率，详见下文「常见误区」          |
| `Compute M.`           | 计算模式                         | `Default` = 多进程共享，`E. Process` = 独占 |
| `MIG M.`               | MIG 模式                         | A100/H100 支持 GPU 切分，`N/A` = 不支持     |
| `Volatile Uncorr. ECC` | 本次运行中未纠正的 ECC 错误      | > 0 立即关注                                |
| `Persistence-M`        | 持久化模式                       | `On` 可避免 GPU 空闲时驱动卸载              |

**常见误区**：`GPU-Util` 只表示采样周期内有 Kernel 在跑，1 个 SM 在跑显示 100%，全部 SM 都在跑也显示 100%。注意 nvidia-smi 里没有单独的「SM 利用率」查询：`GPU-Util`、`--query-gpu=utilization.gpu`、`dmon` 的 `sm` 列同源于 NVML `nvmlDeviceGetUtilizationRates`，口径都是 kernel 执行时间占比。判断 GPU 是否真的被用满，用 §三 的 util + 显存利用率 + 功耗三件套，要量化 SM 占用则用 DCGM 的 SMACT (1002)。详见 [GPU 利用率是一个误导性指标](02_gpu_utilization_myth.md) 和 [GPU 忙不忙怎么判断](09_gpu_busy_check.md)。

---

## 二、谁在用什么？「哪个进程占了多少显存？」

```bash
nvidia-smi pmon -i 0                # 进程级实时
nvidia-smi --query-compute-apps=pid,used_memory --format=csv  # 脚本化
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

`pmon` 输出：

```text
# gpu   pid  type   sm   mem   enc   dec   command
    0  2635     C    0     0     0     0   python
    0  4210     C   45    12     0     0   python3
```

列含义：

| 列        | 含义                                 |
| --------- | ------------------------------------ |
| `type`    | `C` = 计算，`G` = 图形               |
| `sm`      | 该进程的 kernel 执行时间占比（口径同 GPU-Util，进程级） |
| `mem`     | 该进程的显存带宽利用率               |
| `enc/dec` | 编码/解码引擎利用率                  |

**脚本化示例**（查找显存占用 TOP 3）：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader | sort -t',' -k3 -rn | head -3
```

---

## 三、跑满了没？「GPU 在干活还是闲着？」

### 3.1 实时滚动

```bash
nvidia-smi dmon -s mucp -d 2        # 显存用量 + 利用率四列 + 时钟 + 功耗温度，每 2 秒
```

`-s` 选项控制显示哪些列（按 nvidia-smi 手册）：

| -s 选项 | 列含义                                                  |
| ------- | ------------------------------------------------------- |
| `m`     | 显存用量：fb（framebuffer）+ bar1                       |
| `u`     | 利用率四列：sm / mem / enc / dec（sm、mem 口径同 GPU-Util） |
| `c`     | SM 时钟 + 显存时钟 (MHz)                                |
| `p`     | 功耗 + 温度                                             |
| `v`     | power/thermal violations（限频事件计数）                |
| `t`     | PCIe 吞吐 (MB/s)；dmon 没有 NVLink 吞吐列               |

常用组合：`-s mucp`（显存用量 + 利用率四列 + 时钟 + 功耗温度），一行看 GPU 负载全景。

### 3.2 脚本化单次查询

```bash
# GPU 利用率 + 显存利用率
nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader

# 功耗
nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader

# 时钟
nvidia-smi --query-gpu=clocks.current.sm,clocks.current.memory --format=csv,noheader
```

### 3.3 解读

判读靠三件套：`utilization.gpu`（kernel 活跃）+ `utilization.memory`（显存带宽）+ 功耗。注意 `utilization.gpu` 高只说明 kernel 执行时间占比高，不能单独推出计算饱和。

| 现象                                              | 含义             | 方向                                                       |
| ------------------------------------------------- | ---------------- | ---------------------------------------------------------- |
| util.gpu < 20%，util.memory > 80%                 | Memory-bound 信号 | 检查是否未用 pinned memory 或 batch 太小                   |
| util.gpu > 80%，util.memory < 30%，功耗接近 power limit | 在真算，疑似计算重负载 | 确认饱和用 DCGM SMACT (1002)；想更快就换更强的 GPU        |
| util.gpu 低，`Perf` 始终是 P8，功耗贴近空闲值     | GPU 无计算负载   | 进程可能卡在 CPU 端                                       |

---

## 四、健康吗？「温度/功耗/ECC/拓扑有没有异常？」

### 4.1 温度和功耗

```bash
nvidia-smi -q -d TEMPERATURE,POWER
```

关键字段：`GPU Current Temp`、`GPU Slowdown Temp`（撞此温度后 GPU 自动降频）、`Power Limit`、`Power Draw`。

正常范围：H100 满载 ~65-72°C、T4 ~70-78°C。功耗应显著低于 `Power Limit`：如果 `Power Draw` 一直等于 `Power Limit`，GPU 在功耗墙上被限制，换更强的散热或降低功耗配置。

### 4.2 ECC 错误

```bash
nvidia-smi -q -d ECC
```

- **Volatile ECC > 0**：本次驱动加载以来已纠正的错误。如果是 DRAM 错误且持续增长，GPU 可能有硬件问题
- **Aggregate ECC 缓慢增长**：出厂以来的累计值，单 GPU 累计 < 100 属正常老化
- **Uncorrectable ECC > 0**：严重，数据完整性风险。Ampere 及之后架构优先看 Remapped Rows（row remapping 兜底），如需清计数可 `nvidia-smi -i <ID> --reset-ecc-errors=volatile`，仍在增长则更换 GPU

区分 SRAM 和 DRAM 错误：SRAM 错误通常来自宇宙射线（单粒子翻转），单次出现无影响；DRAM 错误频繁出现则表明显存颗粒老化。

### 4.3 NVLink 和拓扑

```bash
nvidia-smi topo -m                       # 连接矩阵
nvidia-smi nvlink --status               # 每条 link 的状态 + 带宽
nvidia-smi nvlink --capabilities         # NVLink 版本和能力
nvidia-smi nvlink --error-counters       # 错误计数
```

拓扑标识解读：

| 标识   | 含义                                | 性能            |
| ------ | ----------------------------------- | --------------- |
| `NV12` | 12 条 NVLink 直连                   | ~600 GB/s 双向  |
| `PIX`  | 同一 PCIe switch，P2P 可用          | ~32 GB/s (Gen4) |
| `NODE` | 同一 NUMA node，经 PCIe Host Bridge | ~28 GB/s        |
| `SYS`  | 跨 NUMA，经 QPI/UPI                 | 最慢            |

详见 [GPU 集群健康检查](06_gpu_health_check.md) L2 流程和 [NCCL 通信路径逐层压测](../03_nccl/06_nccl_path_benchmark.md) 的实测数据。

### 4.4 PCIe 链路

```bash
nvidia-smi -q -d PCIE
```

确保 `Current Link Width` = `Max Link Width`（通常都是 x16）。如果 Gen4 卡运行在 Gen3 或 x16 掉到 x8：检查主板 BIOS 设置、PCIe 扩展器、Riser 卡。

---

## 五、出错了？「XID 错误和 GPU 重置」

```bash
dmesg | grep -i "NVRM\|Xid"                      # 内核日志中的 XID 错误
nvidia-smi -i <ID> --reset-ecc-errors=volatile   # 清零 volatile ECC 计数（aggregate 不可清）
nvidia-smi -i 0 --gpu-reset                      # 重置 GPU：需 root，目标 GPU 及 NVLink peer 空闲；Ampere 之前架构的多卡互联 GPU 须用同一命令一起重置。官方注明生产环境不建议常规使用
```

常见 XID 错误速查（含义按 NVIDIA Xid Errors 目录，Release 620，as-of 2026-11）：

| XID | 官方含义                                                            | 处理                                              |
| --- | ------------------------------------------------------------------- | ------------------------------------------------- |
| 31  | GPU memory page fault（MMU 报告的页面错误，多为应用非法地址访问）   | 单次出现多为应用层问题，复现则排查应用            |
| 43  | GPU stopped processing（应用触发软件故障，GPU 仍健康）              | 多为应用错误；复现时定位应用，不是硬件问题        |
| 45  | Preemptive cleanup, due to previous errors（此前错误的连带清理）    | 伴随其他 XID 时先处理根因；单独出现重启进程/GPU   |
| 48  | Double Bit ECC Error（双比特 ECC 错误）                             | Ampere+ 走 row remapping；持续出现评估更换        |
| 79  | GPU has fallen off the bus（驱动经 PCIe 访问 GPU 失败，多指向 PCIe 链路/硬件故障） | 官方建议整机重启；复现时检查 PCIe 插槽、供电、Riser，跑 DCGM diag |
| 119 | GSP RPC Timeout（GSP 固件响应超时）                                 | 记录日志并观察；高频复现先更新驱动，仍复现报支持  |

---

## 六、我要改点什么？「功耗 / 计算模式 / 时钟 / 持久化」

```bash
# 功耗
nvidia-smi -i 0 -pl 300              # 限制 GPU 0 最大 300W
nvidia-smi -i 0 -q -d POWER          # 确认生效

# 计算模式
nvidia-smi -i 0 -c 0                 # 0=Default（共享）
#                                    2=Prohibited（禁止新进程，测试用）
#                                    3=EXCLUSIVE_PROCESS（单进程独占）

# 时钟（基准测试需要稳定 clock 时）
nvidia-smi -lgc 1500,2100            # 锁定 GPU 时钟在 1500-2100 MHz
nvidia-smi -lmc 5001,5001            # 锁定显存时钟
nvidia-smi -rgc && nvidia-smi -rmc   # 恢复默认

# 持久化模式
nvidia-smi -pm 1                     # 启用（推荐生产环境）
nvidia-smi -q -i 0 | grep Persistence # 确认
```

**持久化模式**值得单独说明：关闭时，GPU 在最后一个进程退出后会被驱动卸载（`nvidia-smi` 仍然能看到但延迟显著升高）。启用后驱动常驻，避免空闲 GPU 的冷启动延迟（通常 1-2 秒）。生产环境建议 `nvidia-persistenced` 服务。

---

## 七、多实例 GPU（MIG）

A100/H100 支持将一张物理 GPU 切分为多个独立的 GPU 实例，每个实例有专属的显存、SM 和带宽。

```bash
nvidia-smi mig -lgip                  # 列出可用的 GPU 实例配置
nvidia-smi mig -cgi 19,19,19 -C       # 创建 3 个 2g.20gb 实例（A100-80GB 上限即 3 个；
                                      # profile ID 19 等取值以现场 -lgip 输出为准）
nvidia-smi mig -dci                   # 销毁所有实例
nvidia-smi mig -dci -gi 0             # 销毁指定 GPU 实例
```

> MIG 模式需要重启 GPU 才能切换（`nvidia-smi -i <ID> -r`：需 root，GPU 上无运行中的进程，NVLink peer 也须空闲），且与 P2P 互斥：切分为 MIG 后 GPU 间不再支持 P2P 通信。

---

## 八、实时监控

```bash
nvidia-smi -l 5                       # 每 5 秒刷新概要
nvidia-smi dmon -s mucp -d 2          # 每 2 秒滚动
watch -n 1 nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader
```

容器环境（Docker/K8s）中，`nvidia-smi` 默认只显示容器可见的 GPU（通过 `NVIDIA_VISIBLE_DEVICES` 控制）。如果容器内看不到预期的 GPU，检查 `NVIDIA_VISIBLE_DEVICES` 或容器的 `--gpus` 参数。

---

## 九、`--query-gpu` 脚本化速查

脚本化场景下，`--query-gpu` + `--format=csv,noheader,nounits` 是最佳组合：输出干净、可管道、不依赖 `-q` 的冗长格式。

```bash
# 通用格式
nvidia-smi --query-gpu=<属性1>,<属性2> --format=csv,noheader,nounits
```

| 场景     | `--query-gpu=` 属性                                                         |
| -------- | --------------------------------------------------------------------------- |
| 显存     | `memory.total,memory.used,memory.free`                                      |
| 利用率   | `utilization.gpu,utilization.memory`                                        |
| 温度     | `temperature.gpu,temperature.memory`                                        |
| 功耗     | `power.draw,power.limit`                                                    |
| 时钟     | `clocks.current.sm,clocks.current.memory`                                   |
| PCIe     | `pcie.link.gen.current,pcie.link.width.current`                             |
| 风扇     | `fan.speed`                                                                 |
| ECC      | `ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total` |
| 计算模式 | `compute_mode`                                                              |
| MIG      | `mig.mode.current`                                                          |
| 进程     | 用 `--query-compute-apps` 替代                                              |

**一行监控脚本示例**（打印所有 GPU 的 ID + GPU 利用率 + 显存占用 + 温度）：

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu \
  --format=csv,noheader,nounits | \
  awk -F', ' '{printf "GPU %s: %s%% util | %s MiB | %s°C\n", $1, $2, $3, $4}'
```

---

## 十、相关资源

- [GPU 利用率是一个误导性指标](02_gpu_utilization_myth.md)：读完再回头看 `GPU-Util` 这列
- [GPU 忙不忙怎么判断](09_gpu_busy_check.md)：util + 功耗温度排假忙，DCGM SMACT 看真实 SM 占用
- [GPU 集群健康检查](06_gpu_health_check.md)：L1/L2/L3 三层检查流程，含 GPU 7 真实异常案例
- [NVLink 诊断与实操](../../01_hardware_architecture/nvlink/nvlink_diagnostics.md)：`nvlink --status` 输出的深度解读
- [DCGM 监控实操](05_dcgm_monitoring.md)：长期趋势和 Prometheus 集成
- [GPU 进程与资源管理](07_gpu_process_management.md)：Compute Mode、CUDA_VISIBLE_DEVICES、NUMA 亲和性
- [NVIDIA nvidia-smi 文档](https://docs.nvidia.com/deploy/driver-nvidia-smi/index.html)
