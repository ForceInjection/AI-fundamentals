# 06 — ascend-dmi 使用参考

`ascend-dmi` v6.0.0 是昇腾设备管理接口工具，集成 `nvidia-smi`（设备状态）、`deviceQuery`（硬件规格枚举）和 `bandwidthTest`（带宽基准）三类功能于一身。它通过底层 DCMI（设备控制管理接口）获取驱动级信息，通过 AscendCL 执行算力和带宽测试。二进制位于 `/usr/local/Ascend/toolbox/6.0.0/Ascend-DMI/bin/ascend-dmi`，运行日志在 `/var/log/ascend-dmi/ascend-dmi.log`。

## 1. 环境变量

`ascend-dmi` 依赖两套动态库：CANN toolkit 的 AscendCL 库（用于算力和带宽测试）和驱动层的 `libdcmi.so`（用于设备信息查询和诊断）。缺少任一套都会导致对应功能不可用。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/toolbox/6.0.0/Ascend-DMI/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
```

缺少 `libdcmi.so` 时报 `Error code [0x13]`。此时 `--info --brief` 等基础查询仍可用（通过 sysfs），但 `--bandwidth`、`--flops`、`--diagnosis` 会因无法加载驱动接口而失败。

## 2. 功能速查表

`ascend-dmi` 采用 `ascend-dmi <主命令> <子选项>` 的两级参数结构。每个主命令对应一类功能，子选项控制具体行为。以下是完整功能矩阵以及与 CUDA 工具的对照：

| 命令                                        | 对应 CUDA 工具                              | 功能                                                                             |
| ------------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------- |
| `--info --detail`                           | `deviceQuery`                               | 每张卡完整硬件规格：AI Core 数量、HBM 参数、PCIe LnkCap/LnkSta、ECC 状态、DIE ID |
| `--info --brief`                            | `nvidia-smi`                                | 所有卡实时状态一览：健康、温度、功耗、内存占用、AI Core 利用率                   |
| `--bandwidth -t {d2d\|h2d\|d2h\|p2p}`       | `bandwidthTest` / `p2pBandwidthLatencyTest` | 各级存储带宽与延迟的微基准测试                                                   |
| `--flops -t {fp16\|bf16\|fp32\|hf32\|int8}` | 无直接对应                                  | AI Core Cube Unit 的矩阵乘实测算力，反映满负载下的实际 TFLOPS                    |
| `--compatible`                              | 无直接对应                                  | 驱动/固件/CANN/toolbox 四层之间的版本兼容性校验                                  |
| `--diagnosis -i {item}`                     | `nvidia-smi -q` 部分功能                    | 软硬件综合诊断，12 个独立诊断项，部分支持压测模式                                |
| `--signal-quality -t {pcie\|hccs\|roce}`    | 无直接对应                                  | 物理层链路信号质量检测，排查间歇性通信异常                                       |
| `--cardinfo`                                | `nvidia-smi -q` 部分                        | 卡级别信息概览                                                                   |
| `--power`                                   | `nvidia-smi -q -d POWER`                    | 整卡或芯片级功耗实时检测                                                         |

所有支持 `--fmt` 的命令均可加 `--fmt json` 输出结构化 JSON，便于脚本解析和监控系统集成。

## 3. 设备详情 (`--info --detail`)

`--info --detail` 是 `deviceQuery` 的直接对应物，它遍历每个 NPU 设备并从驱动层枚举全部硬件属性，包括 `npu-smi` 不暴露的 DIE ID（芯片晶圆级唯一标识）、Vector Unit 数量、ECC 配置和 PCIe 电气能力等。

910B3 单卡实测结构：

| 类别   | 字段                             | 实测值                                | 解读                                                                    |
| ------ | -------------------------------- | ------------------------------------- | ----------------------------------------------------------------------- |
| 标识   | DIE ID                           | 每卡唯一                              | 芯片晶圆级唯一标识，用于硬件追溯和 RMA（退货授权）流程                  |
| 计算   | AI Core / Cube / Vector          | 20 / 20 / **40**                      | 每个 AI Core 含 1 个 Cube Unit（矩阵乘）和 2 个 Vector Unit（向量运算） |
| CPU    | AI CPU / Control CPU / Ctrl 频率 | 7 / 1 / 2000 MHz                      | AI CPU 处理非矩阵算子（如控制流、数据预处理），Control CPU 负责设备管理 |
| 内存   | HBM 总量 / 频率                  | 65536 MB (64 GB) / 1600 MHz           | 单卡 64 GB HBM2e，1.6 GHz 提供约 1.54 TB/s 实测带宽                     |
| PCIe   | LnkSta / LnkCap                  | Gen4 ×16 (16GT/s) / Gen5 ×16 (32GT/s) | LnkSta 为当前协商速率，LnkCap 为硬件支持的最高速率                      |
| ECC    | HBM ECC                          | Enabled                               | Single-bit/Double-bit Error 计数见完整 `--detail` 输出                  |
| 功耗   | 实时（空闲/满载）                | ~93 W / ~231 W                        | 空闲功耗约 93 W，FP16 满载时上升至 ~231 W                               |
| 亲和性 | CPU Affinity                     | 各卡不同（如 NPU 0: 144-167）         | NPU 对应的 NUMA node CPU 核心范围，影响数据加载性能                     |

LnkCap vs LnkSta 的差异值得关注：910B3 硬件支持 PCIe Gen5 ×16（32GT/s），但本服务器当前协商在 Gen4 ×16（16GT/s），说明 PCIe 速率上限由主板和 CPU 决定而非 NPU 本身。如果需要 Gen5 带宽，需要确认服务器主板和 BIOS 是否支持。

## 4. 版本兼容性检查 (`--compatible`)

昇腾软件栈由四层独立组件构成：驱动、固件、CANN toolkit 和 toolbox。各层有独立的版本号体系，且存在严格的兼容性约束——例如 toolkit 8.0.1 要求驱动版本在 23.0.0 ~ 24.1.x 范围内。版本不匹配的常见后果包括 CANN 初始化失败、部分算子编译错误或运行时异常。

```bash
ascend-dmi --compatible
```

本服务器输出：

| 组件         | 版本        | 状态 | 内部版本              |
| ------------ | ----------- | ---- | --------------------- |
| npu-driver   | 24.1.0.3    | OK   | V100R001C19SPC005B220 |
| npu-firmware | 7.5.0.5.220 | OK   | -                     |
| toolkit      | 8.0.1       | OK   | V100R001C20SPC002B220 |
| toolbox      | 6.0.0       | OK   | -                     |

输出中有两个版本号维度：商业版本号（如 `24.1.0.3`）面向用户，用于日常查阅和兼容性判定；内部版本号（如 `V100R001C19SPC005B220`）是华为内部的版本追踪标识，在与华为技术支持沟通时需要提供。两者不是一一映射的——一个商业版本可能对应多个内部 build。

当 NPU 出现异常行为时，首先执行 `ascend-dmi --compatible`。版本不匹配是最常见但也最容易忽视的根因——多见于升级 toolkit 后未同步升级驱动、或 Docker 容器内外 toolkit 版本不一致的场景。

## 5. 带宽测试 (`--bandwidth`)

`--bandwidth` 执行微基准测试，测量 NPU 各层存储之间的数据传输带宽和延迟。它覆盖三条关键数据通路：HBM 内部（d2d，决定 AI Core 的数据供给速率）、主机与设备之间（h2d/d2h，走 PCIe，决定数据加载和多卡 AllReduce 的通信效率）、设备与设备之间（p2p，走 HCCS 直连链路，决定分布式训练的卡间通信能力）。

默认不加 `-t` 时，工具按 d2d → d2h → h2d 的顺序串行执行全部三种测试，完整跑完约需 2-3 分钟。建议通过 `-t` 指定单一类型。

| 类型  | 数据通路                   | 对应 CUDA                       | 关键参数                            |
| ----- | -------------------------- | ------------------------------- | ----------------------------------- |
| `d2d` | HBM 内部读写               | `bandwidthTest` device 部分     | `-d` 设备号                         |
| `h2d` | 主机内存 → 设备 HBM (PCIe) | `bandwidthTest --memory=pinned` | `-d` 设备号，`-s` 数据量 (1B~512MB) |
| `d2h` | 设备 HBM → 主机内存 (PCIe) | 同上                            | 同上                                |
| `p2p` | 设备间直接传输 (HCCS)      | `p2pBandwidthLatencyTest`       | `--ds` 源卡，`--dd` 目标卡          |

### 实测数据（910B3，NPU 7）

测试采用阶梯式模式：从小数据量逐步增大到大数据量（2 B → 32 MB / 20.97 GB），每个数据量重复多次（默认 5 次），最终取带宽饱和后的稳定值。

| 测试     | 稳定带宽      | 饱和数据量      | 说明                                                                |
| -------- | ------------- | --------------- | ------------------------------------------------------------------- |
| d2d      | **1538 GB/s** | 20.97 GB (固定) | HBM 内部带宽。这是 AI Core 读写 HBM 的峰值速度，延迟 ~13.6 ms       |
| p2p 单向 | **26.2 GB/s** | ≥ 2 MB          | HCCS 单向（NPU 7→6）。< 64 KB 时带宽随数据量正比增长，≥ 2 MB 后饱和 |
| p2p 双向 | **50.8 GB/s** | ≥ 2 MB          | HCCS 全双工（NPU 7↔6）。双向约为单向的 2×，说明 HCCS 链路为全双工   |
| h2d      | **24.8 GB/s** | ≥ 16 MB         | PCIe Gen4 ×16 上行（主机→设备）                                     |
| d2h      | **27.6 GB/s** | ≥ 16 MB         | PCIe Gen4 ×16 下行（设备→主机），略高于 h2d（DMA 读通常优于写）     |

PCIe 带宽与理论值吻合：16 GT/s × 16 lanes × 128b/130b（编码效率）≈ 25.2 GB/s，实测 24.8-27.6 GB/s 在此范围内。

**HBM 带宽（1538 GB/s）的工程意义**：这是 AI Core 从本地 HBM 取数的峰值速度。以 ResNet-50 FP32 训练为例，单步计算密度约 8 GFLOPS，假设 50% 算力利用率时需要的 HBM 带宽约 70 GB/s——远低于 1538 GB/s 上限。但对于 memory-bound 算子（LayerNorm、Softmax、Attention 中的 softmax/QKV 投影），HBM 带宽是训练吞吐的关键瓶颈，决定了这些算子的执行时间下界。

**HCCS P2P 带宽（26.2 GB/s）的工程意义**：在多卡数据并行训练中，AllReduce 的通信瓶颈取决于卡间带宽 × 拓扑效率（8 卡全互联拓扑效率接近 100%）。以每步 AllReduce 通信量 25 MB（对应 25M 参数的梯度 × FP32）为例，单次通信时间约 25 MB / 26.2 GB/s ≈ 1 ms。对于通信计算比良好的 workload，这个开销是可接受的；对于小模型（< 10M 参数），通信时间占比会显著上升。

### 命令

```bash
ascend-dmi --bw -t d2d -d 7 -q                     # HBM 带宽（固定传输 20.97 GB）
ascend-dmi --bw -t p2p --ds 7 --dd 6 -q             # 卡间 P2P（阶梯式：2 B → 32 MB）
ascend-dmi --bw -t h2d -d 7 -s 1048576 --et 10 -q   # 自定义：传输 1 MB，重复 10 次
```

> [!WARNING]
> 带宽和算力测试会独占 NPU 的计算和带宽资源，运行期间可能导致同卡上的训练任务性能下降或 OOM。执行前确认目标卡空闲，使用 `-q` 跳过交互确认。

## 6. 算力测试 (`--flops`)

`--flops` 通过反复执行大规模矩阵乘法（`M×K × K×N`）测量 AI Core Cube Unit 的实测峰值吞吐。与 `deviceQuery` 只报告理论峰值不同，这里的数字是真实运行后反推的——已包含 Cube Unit 的时钟频率、温度降频、指令调度开销等实际因素。

测试流程：工具在目标卡的每个 AI Core 上执行固定规模的矩阵乘，默认执行 60 万次（`--et 60`，单位：十万次），然后根据总运算量和总耗时计算单卡 TFLOPS。注意这个测试只测试 Cube Unit，不涉及 Vector/Scalar Unit，也不经过 GE 图编译器——因此它代表的是纯矩阵运算的硬件峰值，而非端到端模型算力。

```bash
ascend-dmi -f -d 7 -t fp16 -q          # FP16（默认矩阵规模，60 万次）
ascend-dmi -f -d 7 -t bf16 -q          # BF16（需 CANN 8.0+）
ascend-dmi -f -d 7 -t fp32 -q          # FP32（峰值约为 FP16 的 1/8 ~ 1/4）
ascend-dmi -f -d 7 -t int8 -q          # INT8（推理场景基准）
ascend-dmi -f -d 7 -t fp16 --et 60 -q  # 自定义执行次数（60 × 100,000 = 6M 次）
```

### 实测数据（NPU 7，FP16）

| 执行次数    | 耗时    | TFLOPS    | 功耗    |
| ----------- | ------- | --------- | ------- |
| 360,000,000 | 1702 ms | **313.7** | 231.1 W |

解读：NPU 7 的 FP16 实测峰值为 313.7 TFLOPS，对标 NVIDIA A100 SXM 的 FP16 非稀疏峰值 312 TFLOPS。测试期间功耗从空闲的 ~93 W 上升到 ~231 W，增幅约 138 W——这部分功率主要由 Cube Unit 在满载时消耗。

> [!NOTE]
> 该测试反映的是纯矩阵乘理想负载下的 Cube Unit 峰值。实际模型训练中，算力的全局利用率（MFU: Model FLOPS Utilization）通常在 30%-60%，受限于内存带宽、算子编译效率、Vector/Scalar Unit 的负载均衡以及通信开销。

## 7. 故障诊断 (`--diagnosis`)

`--diagnosis` 是一个涵盖软件和硬件的综合检查框架。它不像 `--compatible` 只检查版本，而是深入到驱动健康、CANN 内部模块（runtime/compiler/hccl/opp 之间的兼容性）、设备健康、信号质量，甚至 HBM 和 AI Core 的压测。诊断结果以 PASS/FAIL 形式输出，帮助快速定位问题在软件层还是硬件层。

诊断项分为**轻量检查**（不执行压测，耗时短，可在生产环境在线执行）和**压测检查**（执行实际负载，耗时较长，建议在隔离或维护窗口执行）。

| 诊断项          | 类别 | 可压测          | 检查内容                                                   |
| --------------- | ---- | --------------- | ---------------------------------------------------------- |
| `driver`        | 软件 | -               | 驱动内核模块是否正常运行、与固件通信是否正常               |
| `cann`          | 软件 | -               | CANN runtime/compiler/hccl/opp 内部各模块之间的 API 兼容性 |
| `device`        | 硬件 | -               | 设备是否在线、健康状态是否为 OK                            |
| `network`       | 硬件 | -               | 网络接口（如 RoCE 网卡）状态检查                           |
| `bandwidth`     | 硬件 | -               | 本地带宽是否达到基准值（PASS/FAIL 判定，非数值输出）       |
| `aiflops`       | 硬件 | -               | 算力是否达到基准值                                         |
| `hbm`           | 硬件 | 可（`-s --st`） | HBM 读写正确性检查，支持持续压测以验证 HBM 可靠性          |
| `aicore`        | 硬件 | 可              | AI Core 计算正确性检查，支持压测                           |
| `signalQuality` | 硬件 | -               | HCCS/PCIe/RoCE 物理层信号质量                              |
| `p2p`           | 硬件 | 可              | 卡间通信正确性及带宽压测                                   |
| `tdp`           | 功耗 | 可              | TDP (Thermal Design Power) 压力测试                        |
| `edp`           | 功耗 | 可              | EDP (Electrical Design Power) 压力测试                     |

```bash
ascend-dmi --dg -i driver,cann                         # 纯软件层诊断（1-2 分钟）
ascend-dmi --dg -i device,hbm,aicore -d 7 -q           # 指定卡 + 指定诊断项
ascend-dmi --dg -i hbm -d 7 -s --st 60 -q              # HBM 60 秒压测（内存可靠性验证）
ascend-dmi --dg -i aicore -d 7 -s --st 300 -q          # AI Core 5 分钟压测（计算稳定性验证）
```

> [!TIP]
> 推荐的排障顺序：`--compatible`（排除版本问题） → `--dg -i driver,cann,device`（快速健康检查） → 针对异常层面单独诊断或压测。注意：`aicore` / `prbs` / `edp` / `tdp` 四个诊断项互斥，不能与其他项同时使用。

## 8. 信号质量 (`--signal-quality`)

检测 NPU 间 HCCS 链路和 NPU 与主机间 PCIe/RoCE 链路的物理层信号质量。在训练出现间歇性 HCCS 超时、集合通信性能抖动、或设备偶发离线时，信号质量检测是硬件排障的第一站。异常通常表现为误码率升高、链路降速或频繁的链路重训练。

```bash
ascend-dmi --sq -d 7                  # NPU 7 的 PCIe + RoCE 信号（默认）
ascend-dmi --sq -d 7,6 -t hccs        # NPU 7↔6 的 HCCS 信号（需至少 2 张卡）
ascend-dmi --sq -t hccs               # 全局 HCCS 信号质量扫描
```

如果信号质量检测发现异常，需要检查物理连接（线缆是否插紧、插槽是否松动）、交换机配置和散热条件。

## 9. 典型场景命令组合

| 场景             | 命令                                                        | 耗时    |
| ---------------- | ----------------------------------------------------------- | ------- |
| 日常巡检         | `ascend-dmi --info --brief`                                 | < 5 s   |
| 装机验收         | `ascend-dmi --info --detail && ascend-dmi --compatible`     | ~30 s   |
| 性能基线         | `ascend-dmi -f -d 7 -q && ascend-dmi --bw -d 7 -q`          | ~5 min  |
| 故障排查（快速） | `ascend-dmi --dg -i driver,cann,device -q`                  | ~2 min  |
| 故障排查（深度） | `ascend-dmi --dg -i device,hbm,aicore,bandwidth,aiflops -q` | ~10 min |
| 脚本采集         | `ascend-dmi --info --detail --fmt json`                     | < 5 s   |

## 10. 昇腾工具定位区分

昇腾生态中，`npu-smi`、`ascend-dmi`、`msprof`、`atc` 四者各司其职，覆盖 "运维监控 → 装机诊断 → 性能调优 → 模型部署" 的完整链条。`npu-smi` 的完整用法见 `01_npu_smi_reference.md`。

| 工具         | 定位           | 偏重                               | 典型用法                                                              |
| ------------ | -------------- | ---------------------------------- | --------------------------------------------------------------------- |
| `npu-smi`    | 轻量设备管理   | 实时状态、进程管理、拓扑           | `npu-smi info` 日常巡检；`npu-smi info -t usages -i 7` 确认卡是否空闲 |
| `ascend-dmi` | 重型设备管理   | 性能基准测试、故障诊断、兼容性检查 | `--compatible` 版本排查；`-f -q` 算力基准；`--dg` 综合诊断            |
| `msprof`     | 模型 Profiling | 模型级算子耗时、时间线分析         | `msprof --application="python train.py" --output=./prof`              |
| `atc`        | 模型编译器     | 模型转换 (ONNX→OM)、图编译、量化   | `atc --model=model.onnx --framework=5 --soc_version=Ascend910B3`      |

一个典型的排障路径：`npu-smi info` 发现某卡状态异常 → `ascend-dmi --dg -d <id>` 定位问题层面 → 软件问题用 `--compatible` 做版本检查，硬件问题用 `--dg -i hbm,aicore` 做压测 → 如果是模型性能问题，转用 `msprof` 深入算子级分析。

## 11. 参考链接

- [昇腾社区 — ascend-dmi 工具](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devtool/dmi/dmi_0001.html)
- [CANN 商业版文档 — 硬件检测](https://www.hiascend.com/document/detail/en/canncommercial/800/devtool/dmi/dmi_0001.html)
