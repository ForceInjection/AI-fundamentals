# CUDA 到 NPU 的代码迁移

## 1. 迁移步骤

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

## 2. CUDA → NPU API 快速迁移

基础 API 对照（`cuda` → `npu` 替换）见 [Hello NPU — 第一个程序 §4 API 对照表](../01_environment/02_hello_npu_first_program.md)。本文聚焦迁移中真正容易出错的地方——以下详细展开。

## 3. 关键差异

- **驱动层 API 不同**：NPU 使用 `torch_npu` 包作为 PyTorch 与 CANN 之间的适配层，必须在代码中 `import torch_npu` 来注册 NPU 后端。
- **`torch_npu` 导包**：`import torch_npu` 的副作用是向 PyTorch 注册 `npu` 设备后端。如果不 import，`torch.npu.is_available()` 返回 `False`，`tensor.npu()` 会报错。在 torch_npu 2.5.1+ 中可省略，但仍建议显式 import。
- **内存模型**：`torch.npu.memory_allocated()` 和 `memory_reserved()` 行为与 CUDA 对应 API 完全一致——`allocated` 为正在使用的张量内存，`reserved` 为 PyTorch 缓存分配器保留的内存。
- **分布式训练**：`dist.init_process_group(backend='hccl')` 替代 `backend='nccl'`。HCCL 是昇腾集合通信库，API 层面与 NCCL 兼容——`DistributedDataParallel` 可直接使用，无需额外改动。`torchrun --nproc_per_node=N` 的 rank 映射逻辑不变。

## 4. 完整迁移示例

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

完整训练脚本见 `train_resnet50.py`。

## 5. 参考链接

- [昇腾社区 — PyTorch 训练适配指南](https://gitee.com/ascend/pytorch)
- [PyTorch 官档 — AMP](https://pytorch.org/docs/stable/amp.html)
