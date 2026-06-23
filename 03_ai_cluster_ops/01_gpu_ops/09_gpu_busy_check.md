# GPU 忙不忙怎么判断——三行命令替代 GPU-Util

`nvidia-smi` 的 `GPU-Util` 是一个历史遗留指标：任何 Kernel 在跑它就报 100%，即便只有一个 SM 在工作。仅靠这一个数字判断 GPU 是否吃满，相当于用车速表判断发动机转速——踩油门就显示有速度，但可能只是一档。

本文给一个三行命令的判断流程：一眼看清 GPU 是真忙还是假忙。

## 一行替代：看 SM 利用率，别看 GPU-Util

```bash
nvidia-smi dmon -s m -d 2 -c 3
```

输出示例：

```text
# gpu   sm   mem   enc   dec
    0   95    42     0     0
    0   92    38     0     0
    0   97    45     0     0
```

`sm` 是 SM 利用率，`mem` 是显存带宽利用率。**两个一起看才有意义**：

| SM 利用率 | 显存带宽利用率 | 状态                    | 怎么回事                                                             |
| --------- | -------------- | ----------------------- | -------------------------------------------------------------------- |
| > 80%     | 任意           | **真忙** — 计算饱和     | GPU 在全力算。想更快只能换 GPU                                       |
| < 30%     | > 80%          | **假忙** — Memory-bound | GPU 在等数据搬来搬去。查 pinned memory、增大 batch、用 kernel fusion |
| < 30%     | < 30%          | **空闲** — 没人用       | GPU 闲着                                                             |
| > 80%     | < 10%          | **纯计算** — 理想       | 计算密集（GEMM 等），显存带宽不是瓶颈                                |

## 辅助判断：温度和功耗不会撒谎

SM 利用率和显存带宽是主指标。但如果不想装 DCGM，温度和功耗是最快的辅助判断——硅片发热和功耗直接反映晶体管翻转量，`memcpy` 骗不了它们：

```bash
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit --format=csv,noheader
```

| 温度               | 功耗                | 可能状态                                                  |
| ------------------ | ------------------- | --------------------------------------------------------- |
| 空闲温度 (25-35°C) | 空闲功耗 (45-70W)   | GPU 没干活——即使 `GPU-Util` 显示 100%                     |
| 满载温度 (55-75°C) | 满载功耗 (250-400W) | GPU 真的在算                                              |
| 空闲温度           | 中等功耗 (100-200W) | 有残留进程占显存但没跑计算                                |
| 满载温度           | 低功耗 (< 200W)     | 被功耗墙限制——`nvidia-smi -pl` 设得太低或散热不足触发降频 |

**典型假忙场景**：`GPU-Util` = 100%，但温度 30°C、功耗 60W——100% 是假的，GPU 根本没在算。正常的满载计算一定伴随显著的温升和功耗提升。

## 经典误判：GPU-Util 100% 但 SM 只有 10%

一个 `memcpy` 操作持续跑——`GPU-Util` 显示 100%，但 CUDA Core 一个没用，`sm` 列显示 0-5%。此时温度和功耗也暴露真相：风扇没加速、功耗纹丝不动。Kernel 很短但频繁 launch 也会造成类似假象——`GPU-Util` 的采样窗口看到"有 Kernel 在跑"，但其实每个 Kernel 只占 GPU 的千分之一时间。

## 没有 DCGM 怎么办？

如果环境里只有 `nvidia-smi`，用 `--query-gpu`（注意：这里的 `utilization.gpu` **不是**默认输出的 `GPU-Util`，而是 SM 利用率）：

```bash
# 每 2 秒采样一次，共 3 次
for i in 1 2 3; do
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory \
    --format=csv,noheader
  sleep 2
done
```

| `nvidia-smi` 字段                | 对应含义             | 对应 DCGM                             |
| -------------------------------- | -------------------- | ------------------------------------- |
| 默认输出 `GPU-Util`              | "有 Kernel 在跑吗？" | 无直接对应（DCGM 没有这个误导性指标） |
| `--query-gpu=utilization.gpu`    | SM 利用率            | `dcgmi dmon -e 203` (SM Active)       |
| `--query-gpu=utilization.memory` | 显存带宽利用率       | `dcgmi dmon -e 204` (MCUTL)           |

## 速决流程

```text
SSH 上去 → nvidia-smi dmon -s m -d 2 -c 3
         ├─ sm > 80%?               → ✅ GPU 吃满了（温度 + 功耗应同步升高确认）
         ├─ sm < 30% 且 mem > 80%?  → ⚠️ Memory-bound，优化数据搬运
         ├─ sm < 30% 且 mem < 30%?  → 🛑 GPU 空闲
         └─ 不确定？→ nvidia-smi --query-gpu=temperature.gpu,power.draw 看一眼
             温度 30°C + 功耗 60W → GPU-Util 100% 是假的
```

## 相关资源

- [GPU 利用率是一个误导性指标（原文翻译）](02_gpu_utilization_myth.md) — MFU vs SM Efficiency 的深入对比
- [DCGM 监控实操](05_dcgm_monitoring.md) — DCGM 安装和完整 Field ID 参考
- [nvidia-smi 场景速查](03_nvidia_smi_guide.md) — nvidia-smi 全部命令
