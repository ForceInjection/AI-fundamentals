# ResNet-50 训练与 AMP 实战

## 1. 实验结果

在 NPU 7 上使用 ResNet-50 (batch_size=64, 224×224 输入) 训练 2 epochs：

| 指标          | FP32      | AMP (FP16) | 变化             |
| ------------- | --------- | ---------- | ---------------- |
| 吞吐量        | 545 img/s | 1254 img/s | **+130% (2.3×)** |
| HBM 峰值      | 5651 MB   | 2924 MB    | **-48%**         |
| 单 batch 耗时 | 117.4 ms  | 50.6 ms    | **-57%**         |

FP32 训练约占用 5.5 GB HBM（64 GB 的 8.6%），AMP 降至 2.9 GB。

## 2. Gradient Scaling 观察

AMP 训练初期出现了 gradient overflow，loss scaler 自动调整：

```text
Gradient overflow. Skipping step
Loss scaler reducing loss scale to 32768.0
Loss scaler reducing loss scale to 16384.0
Loss scaler reducing loss scale to 8192.0
```

这是 AMP 的正常行为——scaler 自动搜索合适的 loss scale 值（初始 65536，最终稳定在 8192）。实际训练中可以通过 warmup 或手动设定 scale 来避免前几步的震荡。

## 3. 与 CUDA 训练的差异点

### 3.1 编译延迟

首次运行 ResNet-50 时，CANN 的图编译器 (GE) 需要对 PyTorch 计算图进行图融合和优化（算子合并、内存复用、数据布局转换），生成 Ascend 可执行的任务序列。首次迭代有明显编译延迟（约 10-30 秒），后续运行相同模型会使用缓存的编译结果。这是 NPU 训练的常态——类比于 GPU 的 JIT kernel compilation（CUDA 侧为 NVRTC，或 Triton 等第三方编译器的首次运行开销）。

### 3.2 算子覆盖率

ResNet-50 的所有标准算子（Conv2D、BatchNorm、ReLU、Linear、AdaptiveAvgPool2d）在 CANN 8.0.1 中均有优化实现，训练过程无任何 Fallback 到 CPU。对于自定义模型，先在不使用 AMP 的情况下确认所有算子均在 NPU 上执行（无 CPU fallback）；通过后再开启 `torch.npu.amp.autocast()` 提升训练速度——AMP 只控制精度转换，不直接影响算子是否 fallback。

### 3.3 数据加载

NPU 训练的数据加载仍发生在 CPU 端（ARM CPU）。由于 ARM 架构的 CPU 性能通常低于 x86，`num_workers` 设置可能需要调整。本实验中 `num_workers=2` 运行良好。

### 3.4 随机数精度

NPU 的浮点运算结果与 GPU 存在可忽略的微小差异（与 01_environment 中矩阵乘法实验一致，误差 < 10⁻³），不影响训练收敛。

## 4. 关键环境配置

CANN TBE 依赖安装和环境变量加载顺序是 NPU 训练最常见的两个配置陷阱，已在 [昇腾环境搭建](../01_environment/01_ascend_environment_setup.md) 中详细说明（§3 TBE 依赖、§4 加载顺序），此处不赘述。

## 5. Profiling

使用 CANN 自带的 `msprof` 进行性能分析。进行 profiling 前，先用 `npu-smi info` 确认目标卡空闲（AICore 0%、HBM 余量充足，详见 [01_npu_smi_reference.md](../05_tools/01_npu_smi_reference.md)）。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/user/npu-learning/venv/bin/activate
ASCEND_RT_VISIBLE_DEVICES=7 msprof \
  --output=/tmp/prof_resnet50 \
  --application="python3 train_resnet50.py --epochs 1"
```

Profiling 结果会导出到 `/tmp/prof_resnet50/`，包含算子执行时间线、AI Core 利用率、内存带宽使用率和算子耗时排序。

## 6. 参考链接

- [昇腾社区 — PyTorch 训练适配指南](https://gitee.com/ascend/pytorch)
- [昇腾社区 — msprof 工具](https://www.hiascend.com/document/detail/en/canncommercial/800/devtool/profiling/profiling_0001.html)
- [PyTorch 官档 — AMP](https://pytorch.org/docs/stable/amp.html)
