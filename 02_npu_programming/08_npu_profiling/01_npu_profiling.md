# NPU 性能分析

## 1. 背景

### 1.1 从"能跑"到"跑得快"

前面 6 个 phase 解决了"能不能在 NPU 上跑"的问题。但工程上，能跑只是第一步。接下来要问：

- NPU 真的在满负荷工作吗，还是在"摸鱼"？
- 时间都花在计算上，还是花在数据搬运上？
- 如果慢，慢在哪个算子？怎么优化？

这就是性能分析要回答的问题。

**先建立两个基本概念：**

```text
计算 bound（瓶颈在算力）:
  程序大部分时间在做计算（矩阵乘、卷积），AI Core 利用率高，
  HBM 带宽用不完。优化方向：用 FP16 混合精度、换更高效的算子。

访存 bound（瓶颈在带宽）:
  程序大部分时间在读写数据（逐元素操作、归一化），AI Core 空闲等数据。
  优化方向：算子融合减少读写次数、增大 batch 让每次读写更"划算"。
```

大部分神经网络是**计算 bound**（尤其是大矩阵乘法和卷积），但归一化层（BatchNorm、LayerNorm）和激活函数通常是访存 bound。一个模型中两类瓶颈往往并存。

### 1.2 工具全景：我应该用哪个？

CANN 提供了从粗到细的多层工具。初学者容易陷入"所有工具都试试但哪个都没用明白"的困境。建议按以下顺序渐进：

```text
第 1 步: npu-smi（卡级监控）
  → 花 10 秒看一眼：AI Core 利用率很低？先确认代码真的在 NPU 上跑。

第 2 步: torch_npu.profiler（算子级追踪）
  → 主要工具。输出 Chrome trace，像"录像"一样回放每个算子的执行时间。

第 3 步: ascend-dmi（带宽基准）
  → 需要知道"这台 NPU 的 HBM 理论最快能多快"时用。
```

| 工具                 | 粒度   | 用多久   | 一句话                           |
| -------------------- | ------ | -------- | -------------------------------- |
| `npu-smi`            | 卡级   | 10 秒    | "NPU 在干活吗？利用率多少？"     |
| `torch_npu.profiler` | 算子级 | 主要工具 | "哪个算子最慢？时间花在哪？"     |
| `ascend-dmi --bw`    | 卡级   | 基准测试 | "HBM 理论带宽是多少？"           |
| `msprof` (CANN 原生) | 系统级 | 进阶使用 | "AI Core 内部的流水线效率如何？" |

> [!NOTE]
> AI Core 硬件指标（PipeUtilization、ArithmeticUtilization、L2Cache 等）需要 CANN >= 8.2.RC1。当前环境 CANN 8.0.1 仅支持算子级追踪（Level0）。升级 CANN 后可以解锁更深层的分析能力。

### 1.3 本次分析范围

以 phase 1（矩阵乘法）和 phase 3（ResNet-50）为基础，对三类核心算子进行系统性 profiling，并结合 npu-smi 实时监控观察负载状态。

---

## 2. 工具详解

### 2.1 torch_npu.profiler：最重要的工具

#### 2.1.1 它做了什么

在代码运行时"录像"，记录每个算子在 NPU 上的开始时间、结束时间、以及对应的 CPU 调用栈。录完后输出一个 JSON 文件，用 Chrome 浏览器打开可以看到这样的时间线：

```text
 NPU: |=Conv2D===|==BatchNorm==|==ReLU==|==MaxPool==|====Conv2D====|==ReLU==|...
 CPU:  | launch ||            ||       ||          ||   launch    ||       |...
       +--------++------------++-------++----------++------------++-------++------
       0ms      5ms          10ms     15ms        20ms          25ms     30ms
```

每个色块是一个算子。色块越长 = 算子越慢。你可以放大到微秒级精度，找到最慢的那个算子。

#### 2.1.2 `synchronize`：一个让新手困惑的点

NPU 的执行是**异步**的：CPU 发送指令后不等待 NPU 完成就继续执行下一条 Python 代码。这意味着：

```python
# 错误示范：测量的是 CPU"下发指令"的时间，不是 NPU 实际执行时间
t0 = time.time()
output = model(input)
print(f"耗时: {time.time() - t0}")  # 可能只有 0.001s，但 NPU 还在跑！

# 正确做法：等待 NPU 完成再计时
t0 = time.time()
output = model(input)
torch.npu.synchronize()  # 阻塞 CPU，直到 NPU 完成所有排队任务
print(f"耗时: {time.time() - t0}")  # 这才是真正的 NPU 执行时间
```

如果不加 `synchronize`，你测量的时间可能比实际 NPU 执行时间短 10-100 倍——这就是为什么很多初学者觉得"NPU 好快"，其实是根本没测到 NPU 的时间。

**什么时候不需要 `synchronize`：**

- `.item()` 获取标量时会自动同步
- `loss.backward()` 会触发同步
- `print(tensor)` 会触发同步
- profiler 内部已经做了同步，不需要额外加

#### 2.1.3 warmup：第一次总是特别的

```python
# NPU 第一次执行某个算子时，CANN 会在后台编译优化这个算子
# 这个过程叫"图编译"(graph compilation)，耗时可能是正常执行的 10-100 倍
output = model(input)  # 第一次：可能 5 秒（包含了编译时间）

# 之后每次都是正常速度
output = model(input)  # 第二次：0.05 秒
output = model(input)  # 第三次：0.05 秒
```

**profiling 前必须 warmup 3-5 次**，否则 trace 里会混入编译时间，数据完全不可用。

#### 2.1.4 基本用法

```python
from torch_npu.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# warmup 3 次
for _ in range(3):
    model(input)

# 开始录像
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    on_trace_ready=tensorboard_trace_handler("./trace_output"),
):
    output = model(input)      # 只录这一次
    loss = criterion(output, target)
    loss.backward()

# 录像自动保存到 ./trace_output/ 目录
# 用 Chrome 打开 chrome://tracing → Load → 选择目录下的 .json 文件
```

### 2.2 npu-smi：快速看一眼 NPU 状态

```bash
npu-smi info -t usages -i 7
```

输出示例及含义：

```text
Aicore Usage Rate(%)           : 85     ← AI Core 计算单元在忙的比例
Aivector Usage Rate(%)         : 70     ← 向量运算单元的利用率
Aicpu Usage Rate(%)            : 12     ← AI CPU（负责任务调度）的利用率
Ctrlcpu Usage Rate(%)          : 3      ← 控制 CPU 的利用率
HBM Capacity(MB)               : 65536  ← 每张卡 64GB HBM 显存
HBM Usage Rate(%)              : 5      ← 当前 HBM 占用比例
HBM Bandwidth Usage Rate(%)    : 45     ← HBM 读写带宽利用率
DDR Bandwidth Usage Rate(%)    : 0      ← 主机内存带宽利用率
```

**初学者容易误解的点：**

npu-smi 大约每秒采样一次。如果你的算子只跑了 0.1 秒，采样发生时算子多半已经结束了，利用率显示 0% 是正常的——不代表 NPU 空闲，只是"采样没赶上"。只有持续运行 1 秒以上的任务才能被 npu-smi 准确反映。

**正确的使用姿势：**

- 训练脚本运行时，另开一个终端，每秒 `watch -n 1 npu-smi info -t usages -i 7`
- 对于短算子，忽略 npu-smi，直接看 profiler 的 trace
- 对于持续运行的训练/推理服务，npu-smi 的利用率数据才可信

### 2.3 ascend-dmi：带宽基准测试

```bash
# HBM 内部带宽（device-to-device）
ascend-dmi --bw -t d2d -d 7

# CPU→NPU 传输带宽（host-to-device）
ascend-dmi --bw -t h2d -d 7

# NPU→CPU 传输带宽（device-to-host）
ascend-dmi --bw -t d2h -d 7
```

> [!WARNING]
> ascend-dmi 的带宽测试需要独占 NPU（测试期间其他进程不能使用该 NPU），且会弹出交互确认提示。在训练任务运行时做带宽测试会影响训练。建议在没有任务时单独运行。

910B3 的理论 HBM 带宽约 **1.2 TB/s**。实际可达带宽取决于数据大小和对齐方式，通常能达到理论值的 70-85%。

---

## 3. 性能测试结果

测试环境：Ascend 910B3 (64GB HBM), CANN 8.0.1, torch_npu 2.1.0.post13, NPU 7。

### 3.1 矩阵乘法：看计算效率

矩阵乘法是神经网络中最核心的运算——全连接层、注意力机制、LSTM 的门控计算本质上都是矩阵乘法。如果矩阵乘法跑不满算力，整个模型都不会快。

| 尺寸        | NPU 耗时 | TFLOPS    | 说明                                         |
| ----------- | -------- | --------- | -------------------------------------------- |
| 4096×4096   | 2.5 ms   | 55.73     | 矩阵太小，kernel launch 开销占比较高         |
| 8192×8192   | 15.6 ms  | 70.35     | 开始进入高效区间                             |
| 16384×16384 | 122.4 ms | **71.84** | 接近 FP32 理论峰值 (~80 TFLOPS)，利用率 ~90% |

**什么叫 TFLOPS？**

TFLOPS = Tera (万亿) Floating-point Operations Per Second，即每秒执行多少万亿次浮点运算。矩阵乘法 C = A × B 的计算量为 `2 × M × N × K` 次浮点运算（一次乘 + 一次加 = 两次运算）。16384×16384 的矩阵乘 = 2 × 16384³ ≈ 8.8 万亿次运算，122ms 完成 → 71.84 TFLOPS。

**如何读这些数据：**

- 4096 时 TFLOPS 只有 55.7，因为矩阵"不够大"。NPU 有上千个 AI Core，大矩阵才能让它们全部忙起来。类比：一个小任务给 1000 个人做，分配任务的时间比干活的时间还长。
- 8192→16384，TFLOPS 从 70.3 缓慢提升到 71.8，说明已接近 FP32 算力天花板。
- 910B3 的 FP32 理论算力约 80 TFLOPS，实测 71.84 达到 ~90%，属于正常水平。

### 3.2 2D 卷积：看不同配置的影响

卷积是 CNN 的核心算子，不同 kernel size 和 stride 的计算量差异巨大。

| 配置                    | 耗时    | MACs | 场景                     |
| ----------------------- | ------- | ---- | ------------------------ |
| Conv(3×3, s=1, 64→64)   | 1.26 ms | 3.7G | ResNet 主体卷积          |
| Conv(3×3, s=2, 64→64)   | 0.71 ms | 0.9G | 带下采样，计算量减至 1/4 |
| Conv(7×7, s=2, 3→64)    | 0.30 ms | 0.2G | ResNet 第一层（stem）    |
| Conv(1×1, s=1, 256→256) | 4.47 ms | 6.6G | Bottleneck 降维/升维     |

**什么叫 MACs？**

MACs = Multiply-Accumulate operations，即乘加运算次数。一次 MAC = 一次乘法 + 一次加法，与 2 次 FLOPs 等价。卷积的 MACs = `2 × in_ch × k² × out_ch × H_out × W_out`。6.6G MACs 就是 66 亿次乘加运算。

**如何读这些数据：**

- **耗时和 MACs 基本成正比**：1×1 卷积虽然参数量只有 0.07M，但 MACs 高达 6.6G，耗时最长（4.47ms）。这说明卷积的计算瓶颈在 MACs 而非参数量——不要被"1×1 是小卷积"误导。
- **stride=2 把计算量减到 1/4**：分辨率减半 → 输出像素数减为 1/4 → MACs 减为 1/4。下采样层不会成为瓶颈。
- 7×7 大 kernel 但输入通道只有 3（RGB），所以总 MACs 很小。
- 这些 trace 文件都可以用 Chrome 打开查看每个卷积的内部算子分解。

### 3.3 ResNet-50：端到端模型

| 指标          | 数值        | 含义                         |
| ------------- | ----------- | ---------------------------- |
| 参数量        | 25.56M      | 模型本身占 ~100MB 存储       |
| Forward       | 22.1 ms     | 推理一张图的延迟（batch=8）  |
| Backward      | 22.3 ms     | 反向传播计算梯度的时间       |
| 总计 (1 iter) | 44.4 ms     | 一次训练迭代的总时间         |
| 吞吐          | 180.2 img/s | 每秒可处理的图片数           |
| HBM 稳态占用  | 204 MB      | 模型参数 + 优化器状态的内存  |
| HBM 峰值占用  | **4116 MB** | 含中间激活 (activation maps) |

**为什么 HBM 峰值是稳态的 20 倍？**

```text
HBM 占用的构成:
┌────────────────────────────────────────────────────────┐
│ 模型参数 (204 MB)                                       │
│ 优化器状态 (204 MB，SGD 的 momentum buffer)             │
│ 输入数据 (batch=8 × 3 × 224 × 224 × 4 bytes ≈ 5 MB)   │
│ 中间激活 (约 3,700 MB) ← 这是大头！                      │
│   - 每一层的输出都要保存，供反向传播计算梯度用            │
│   - ResNet-50 有 50 层，每层输出 8 × C × H × W          │
│   - 浅层分辨率高 (112×112, 56×56)，激活占用大            │
└────────────────────────────────────────────────────────┘
```

这就是为什么大 batch 训练会 OOM——不是参数存不下，是中间激活太大了。解决办法：

- **减小 batch size**：最简单
- **梯度检查点 (gradient checkpointing)**：不保存全部中间激活，反向传播时重新计算一部分。用时间换空间，HBM 峰值可降低 50-70%
- **混合精度 (AMP)**：FP16 存储，激活占用减半

**为什么 Forward 和 Backward 时间几乎相等？**

这不是巧合，而是 ResNet 这类"干净"网络的特征：梯度计算涉及相同的卷积操作，计算量约为 forward 的 1-2 倍。如果 backward 远大于 forward，可能是某些算子没有高效的 NPU 反向实现（需要回退到 CPU）。

### 3.4 npu-smi 实时监控

在持续 3 秒的 16384×16384 矩阵乘法负载下观察：

```text
指标                       空闲时       负载中        变化
────────────────────────────────────────────────────────
AI Core 利用率              0%          0%           —
AI Vector 利用率            0%          0%           —
HBM 带宽利用率              0%         33%         +33%
DDR 带宽利用率              0%          0%           —
────────────────────────────────────────────────────────
3 秒内完成 6165 次 16384×16384 matmul
```

**AI Core 为什么显示 0%？**

这不代表 NPU 没干活（3 秒算了 6165 次大矩阵乘法，显然在满负荷运转）。原因是 npu-smi 每秒采样一次，而单次 matmul 只有 122ms。采样发生的瞬间，大概率 kernel 刚好运行完。采样频率跟不上执行频率。

**什么情况下 npu-smi 的 AI Core 数据才可信？**

- 长时间运行的训练（每个 iteration > 1 秒）
- 持续的推理服务
- 用 `watch -n 0.1` 提高采样频率（但可能不被支持）

**HBM 带宽利用率 33% 说明什么？**

矩阵乘法是计算密集型，数据读写量相对于计算量来说较小（16384² 的矩阵读 1GB，但算了 8.8 万亿次运算），所以 HBM 带宽用不满。这恰好验证了它是**计算 bound**。

如果是逐元素加法这种访存密集型操作（每读一个数只做一次加法），HBM 带宽利用率会显著升高。

---

## 4. 从数据到优化：决策清单

有了 profiling 数据后，下一步是决定怎么优化。按以下决策树：

```text
找到最慢的算子
    │
    ├── 是矩阵乘/卷积？
    │   ├── TFLOPS 利用率 < 50%？ → 增大 batch size、用 AMP (FP16)
    │   └── TFLOPS 利用率 > 80%？ → 这个算子已经很快了，看下一个
    │
    ├── 是逐元素操作 (Add, ReLU, Norm)？
    │   ├── 单个耗时短，但数量多？ → 算子融合（让编译器自动合并）
    │   └── 某个特别慢？ → 检查是否回退到 CPU 执行了
    │
    ├── 是数据传输 (to/from/copy)？
    │   └── 存在 Host↔Device 传输 → 减少不必要的 .cpu()/.item() 调用
    │
    └── 是算子启动 (kernel launch)？
        └── 大量细碎的小算子 → 用 torch.compile 或 ATC 做图编译优化
```

### 4.1 通用优化手段

| 手段                | 效果                             | 代价                     |
| ------------------- | -------------------------------- | ------------------------ |
| AMP (FP16 混合精度) | 算力翻倍 (~320 TFLOPS)，内存减半 | 需验证精度损失           |
| 增大 batch size     | 提高 GPU 利用率                  | 内存压力增大             |
| 算子融合            | 减少 kernel launch 开销          | 编译器自动完成，无需手动 |
| 梯度检查点          | HBM 峰值降低 50-70%              | 训练时间增加 20-30%      |
| 梯度累积            | 等效大 batch，内存不变           | 训练时间略微增加         |

### 4.2 Profiling 最佳实践（避坑指南）

1. **先 warmup，再 profiling**：首次执行会触发图编译，耗时是正常的 10-100 倍
2. **只 profile 关心的代码段**：trace 文件可能很大（ResNet-50 一个 iteration 的 trace 有 2.3MB），profile 整个训练循环会生成几百 MB 的文件
3. **加 `synchronize` 确认时间**：`time.time()` 测的是 CPU 端时间，加 `torch.npu.synchronize()` 才是 NPU 执行时间
4. **Level0 足够入门**：Level1（AI Core 指标）需要 CANN >= 8.2.RC1，但 Level0 的算子耗时数据已经能定位 90% 的性能问题
5. **Chrome trace 小技巧**：在 `chrome://tracing` 中按 `W`/`S` 放大/缩小，按 `A`/`D` 左右移动，点击色块看详细信息

---

## 5. 文件清单

```text
08_npu_profiling/
└── profile_ops.py    # NPU 性能分析脚本，包含以下函数:
                          ProfileRunner         — 封装 profiler，输出 Chrome trace
                          profile_matmul()      — 多尺寸矩阵乘法对比
                          profile_conv2d()      — 不同 kernel/stride 卷积对比
                          profile_resnet50()    — 端到端模型 forward+backward
                          npu_smi_snapshot()    — npu-smi 指标快照
                          bandwidth_benchmark() — ascend-dmi 带宽基准测试
                          monitor_during_workload() — 负载前后利用率对比
```

---

## 6. 与后续方向的关系

- **多卡 DDP 训练**：单卡 profiling 建立性能基线后，多卡场景中可通过 profiler 的 HCCL 通信追踪来定位通信瓶颈
- **混合精度 (AMP)**：当前 TP32 达 71.84 TFLOPS；开启 AMP 后，FP16 理论峰值 ~320 TFLOPS，实际期望 ~200+ TFLOPS
- **CANN 升级后**：升级到 CANN >= 8.2.RC1 后，可启用 ProfilerLevel.Level1 采集 AI Core 硬件指标（PipeUtilization、ArithmeticUtilization、L2Cache、ResourceConflictRatio），这些指标直接告诉你 AI Core 内部各单元的忙闲程度和缓存命中率

---

## 参考链接

- [Ascend CANN 性能分析文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/analysis/atlasprofiling_16_0006.html)
- [PyTorch Profiler 文档](https://pytorch.org/docs/stable/profiler.html)
- [Ascend-DMI 带宽测试](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devtools/ascenddmi/ascend_dmi_01_0074.html)
