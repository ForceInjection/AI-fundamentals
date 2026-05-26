# 03 — PyTorch NPU 实战: ResNet-50 训练与 AMP

## 1. 目标

在 Ascend 910B3 上完成 ResNet-50 模型训练，掌握：

- PyTorch 模型从 CUDA 到 NPU 的迁移方法
- AMP 混合精度训练在 NPU 上的配置
- FP32 vs AMP 的性能与显存对比

## 2. 模型迁移: CUDA → NPU

### 2.1 迁移步骤

将现有 PyTorch CUDA 代码迁移到 NPU 只需三步：

**1. 导入 torch_npu**：

```python
import torch
import torch_npu  # 注册 NPU 后端
```

**2. 替换设备字符串**：

```python
# CUDA
model = models.resnet50().cuda()
images = images.cuda()

# NPU — 只需改设备字符串
model = models.resnet50().npu()
images = images.npu()
```

**3. 替换同步和 AMP API**：

```python
# 设备同步
torch.cuda.synchronize()  → torch.npu.synchronize()

# AMP
torch.cuda.amp.autocast() → torch.npu.amp.autocast()
torch.cuda.amp.GradScaler() → torch.npu.amp.GradScaler()
```

### 2.2 完整代码示例

```python
import torch
import torch_npu
import torchvision.models as models

device = "npu:0"
model = models.resnet50(weights=None).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.npu.amp.GradScaler()

for images, labels in dataloader:
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    with torch.npu.amp.autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

完整训练脚本见 [train_resnet50.py](train_resnet50.py)。

## 3. 实验结果

在 NPU 7 上使用 ResNet-50 (batch_size=64, 224×224 输入) 训练 2 epochs：

| 指标          | FP32      | AMP (FP16) | 变化             |
| ------------- | --------- | ---------- | ---------------- |
| 吞吐量        | 545 img/s | 1254 img/s | **+130% (2.3×)** |
| HBM 峰值      | 5651 MB   | 2924 MB    | **-48%**         |
| 单 batch 耗时 | 117.4 ms  | 50.6 ms    | **-57%**         |

FP32 训练约占用 5.5 GB HBM（64 GB 的 8.6%），AMP 降至 2.9 GB。

### 3.1 Gradient Scaling 观察

AMP 训练初期出现了 gradient overflow，loss scaler 自动调整：

```text
Gradient overflow. Skipping step
Loss scaler reducing loss scale to 32768.0
Loss scaler reducing loss scale to 16384.0
Loss scaler reducing loss scale to 8192.0
```

这是 AMP 的正常行为——scaler 自动搜索合适的 loss scale 值（初始 65536，最终稳定在 8192）。实际训练中可以通过 warmup 或手动设定 scale 来避免前几步的震荡。

## 4. 关键环境配置

### 4.1 CANN TBE 依赖

CANN 的 Tensor Boost Engine (TBE) 依赖于以下 Python 包，在虚拟环境中必须完整安装：

```bash
pip install attrs cloudpickle decorator ml-dtypes psutil scipy tornado
```

缺少任何一个都会导致 GE (Graph Engine) 初始化失败：

```text
ModuleNotFoundError: No module named 'attr'
GELib::InnerInitialize failed
GEInitialize failed
```

这是因为即使使用 PyTorch NPU，底层 CANN 的图编译器仍需要 TBE 的 Python 环境才能将计算图编译为 Ascend 可执行指令。

### 4.2 环境变量加载顺序

`source set_env.sh` 必须在 venv 激活之前：

```bash
# 正确顺序
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/npu-learning/venv/bin/activate
```

`set_env.sh` 将 CANN 的 Python 路径（`/usr/local/Ascend/ascend-toolkit/latest/python/site-packages`）注入 `PYTHONPATH`，使 TBE 和 AscendCL Python 绑定可被导入。如果顺序反了，venv 的 Python 无法找到这些模块。

## 5. 与 CUDA 训练的差异点

### 5.1 编译延迟

首次运行 ResNet-50 时，CANN 的图编译器 (GE) 需要将 PyTorch 计算图编译为 Ascend 可执行算子。首次迭代有明显编译延迟（约 10-30 秒），后续运行相同模型会使用缓存的编译结果。这是 NPU 训练的常态——类比于 GPU 的 JIT kernel compilation（CUDA 侧为 NVRTC，或 Triton 等第三方编译器的首次运行开销）。

### 5.2 算子覆盖率

ResNet-50 的所有标准算子（Conv2D、BatchNorm、ReLU、Linear、AdaptiveAvgPool2d）在 CANN 8.0.1 中均有优化实现，训练过程无任何 Fallback 到 CPU。对于自定义模型，建议先用 `torch.npu.amp.autocast()` 包裹计算，未适配的算子会自动 fallback。

### 5.3 数据加载

NPU 训练的数据加载仍发生在 CPU 端（ARM CPU）。由于 ARM 架构的 CPU 性能通常低于 x86，`num_workers` 设置可能需要调整。本实验中 `num_workers=2` 运行良好。

### 5.4 随机数精度

NPU 的浮点运算结果与 GPU 存在可忽略的微小差异（与 01-hello-npu 中的矩阵乘法实验一致，误差 < 10⁻³），不影响训练收敛。

## 6. Profiling

进行 profiling 前，先用 `npu-smi info` 确认目标卡空闲（AICore 0%、HBM 余量充足，详见 [01_npu_smi_reference.md](../05_tools/01_npu_smi_reference.md)）。使用 CANN 自带的 `msprof` 进行性能分析：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/npu-learning/venv/bin/activate

# 收集 profiling 数据
ASCEND_RT_VISIBLE_DEVICES=7 msprof \
  --output=/tmp/prof_resnet50 \
  --application="python3 /tmp/train_resnet50.py --epochs 1"
```

Profiling 结果会导出到 `/tmp/prof_resnet50/`，包含：

- 算子执行时间线 (timeline)
- AI Core 利用率
- 内存带宽使用率
- 算子耗时排序

## 7. 参考链接

- [昇腾社区 — PyTorch 训练适配指南](https://gitee.com/ascend/pytorch)
- [CANN 文档 — AMP](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/appdevgapi/aclpythondevg_0019.html)
- [昇腾社区 — msprof 工具](https://www.hiascend.com/document/detail/en/canncommercial/800/devtool/profiling/profiling_0001.html)
- [PyTorch 官档 — AMP](https://pytorch.org/docs/stable/amp.html)
