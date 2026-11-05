# GPU 忙不忙怎么判断

`nvidia-smi` 的 `GPU-Util` 只回答"有没有 Kernel 在跑"：任何 Kernel 在跑它就报 100%，即便只有一个 SM 在执行一个 `memcpy`。它回答不了另一个更重要的问题：算力到底被用掉了多少。

本文给一套判断流程：先用 nvidia-smi 的 util、功耗、温度排除假忙，再上 DCGM 的 SM Active 量化算力占用。

## 先澄清：nvidia-smi 没有 SM 利用率这个指标

流传较广的一种说法是：`--query-gpu=utilization.gpu` 和默认输出的 `GPU-Util` 不是同一个东西，前者是 SM 利用率。这是错的。官方口径只有一条（NVML 的 `nvmlDeviceGetUtilizationRates`）：GPU utilization 指采样周期内有一个或多个 kernel 在执行的时间占比。默认输出的 `GPU-Util` 列、`utilization.gpu` 查询、`dmon` 的 `sm` 列全部出自这一个接口，NVML 里没有第二个利用率查询。

真正的 SM Active（SM 上至少一个 warp 活跃的周期占比）在 NVML 里查不到，只能从 DCGM 的 profiling 指标取：Field ID 1002（`DCGM_FI_PROF_SM_ACTIVE`）。详见 [DCGM 监控实操](05_dcgm_monitoring.md)。

## nvidia-smi 能看的：util + 功耗 + 温度

```bash
nvidia-smi dmon -s u -d 2 -c 3
```

输出示例：

```text
# gpu   sm   mem   enc   dec
    0   95    42     0     0
    0   92    38     0     0
    0   97    45     0     0
```

`sm` 列就是 GPU-Util 那个口径（kernel 执行时间占比），`mem` 列是显存带宽利用率。两个一起看能区分几类状态：

| sm   | mem   | 状态             | 怎么回事                                                                                   |
| ---- | ----- | ---------------- | ------------------------------------------------------------------------------------------ |
| > 80% | < 30% | 疑似计算重负载   | kernel 执行时间占比高且不走显存带宽，像 GEMM 这类计算密集负载。是否真饱和要看功耗、温度或 DCGM SMACT |
| < 30% | > 80% | Memory-bound 信号 | GPU 时间花在数据搬运上。查 pinned memory、增大 batch、用 kernel fusion                      |
| < 30% | < 30% | 空闲             | 没有有效负载                                                                               |
| 任意 | 任意  | 不能直接判饱和   | `sm > 80%` 只说明 kernel 执行时间占比高：一个持续跑的 `memcpy` 同样能把这列推到 100%        |

功耗和温度是性价比最高的佐证。硅片发热和功耗直接反映晶体管翻转量，`memcpy` 骗不过它们：

```bash
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit --format=csv,noheader
```

| 温度               | 功耗                | 可能状态                                                  |
| ------------------ | ------------------- | --------------------------------------------------------- |
| 空闲温度 (25-35°C) | 空闲功耗 (45-70W)   | GPU 没干活，即使 `GPU-Util` 显示 100%                     |
| 满载温度 (55-75°C) | 满载功耗 (250-400W) | GPU 真的在算                                              |
| 空闲温度           | 中等功耗 (100-200W) | 有残留进程占显存但没跑计算                                |
| 满载温度           | 低功耗 (< 200W)     | 被功耗墙限制：`nvidia-smi -pl` 设得太低或散热不足触发降频 |

典型假忙场景：`GPU-Util` = 100%，但温度 30°C、功耗 60W，说明 GPU 没在算。正常的满载计算一定伴随温升和功耗提升。

## 经典误判：GPU-Util 100% 但实际没在算

一个 `memcpy` 操作持续跑：`GPU-Util` 显示 100%，CUDA Core 一个没用，功耗和温度纹丝不动。靠功耗、温度就能排除这类假忙；要量化计算单元的真实占用，需要 DCGM 的 SMACT（Field ID 1002）。

Kernel 很短但频繁 launch 也会造成类似假象：`GPU-Util` 的采样窗口看到"有 Kernel 在跑"，其实每个 Kernel 只占 GPU 的千分之一时间。

## 要量化算力占用：DCGM SM Active

`nvidia-smi` 的 util 回答不了"SM 被用到了几成"。DCGM 在 util 之外提供 profiling 指标，两者对照如下：

| nvidia-smi 字段                  | 含义                                        | 对应 DCGM                                                                                              |
| -------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 默认输出 `GPU-Util`              | 采样周期内有 kernel 执行的时间占比          | 203 `DCGM_FI_DEV_GPU_UTIL`（同口径）；1001 `DCGM_FI_PROF_GR_ENGINE_ACTIVE`（比例形式）                 |
| `--query-gpu=utilization.gpu`    | 同 `GPU-Util`，同一 NVML 接口               | 同上                                                                                                   |
| `--query-gpu=utilization.memory` | 显存带宽利用率                              | 204 `DCGM_FI_DEV_MEM_COPY_UTIL`；1005 `DCGM_FI_PROF_DRAM_ACTIVE`（比例形式）                           |
| （NVML 无对应查询）              | SM Active：SM 上至少一个 warp 活跃的周期占比 | 1002 `DCGM_FI_PROF_SM_ACTIVE`                                                                          |

```bash
# SMACT 是 SM Active，DRAMA 是 DRAM Active
dcgmi dmon -e 1002,1005,150 -d 3000 -c 3
```

Field ID 的完整对照见 [DCGM 监控实操](05_dcgm_monitoring.md)。

## 速决流程

```text
SSH 上去 → nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw
         ├─ util 100% + 温度 30°C + 功耗 60W?   → 🛑 假忙，GPU 没在实际计算
         ├─ util < 30% 且 mem > 80%?            → ⚠️ Memory-bound 信号，优化数据搬运
         ├─ util > 80% 且功耗接近 power limit?  → ✅ 在真算；是否饱和看 DCGM SMACT (1002)
         └─ 需要量化 SM 占用?                   → dcgmi dmon -e 1002 看 SM Active
```

## 相关资源

- [GPU 利用率是一个误导性指标（原文翻译）](02_gpu_utilization_myth.md)：MFU vs SM Efficiency 的深入对比
- [DCGM 监控实操](05_dcgm_monitoring.md)：DCGM 安装和完整 Field ID 参考
- [nvidia-smi 场景速查](03_nvidia_smi_guide.md)：nvidia-smi 全部命令
