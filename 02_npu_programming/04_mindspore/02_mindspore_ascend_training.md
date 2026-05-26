# 04 — MindSpore 实战: 在 Ascend 上定义与训练模型

## 1. 环境信息

| 项目      | 版本                                                                           |
| --------- | ------------------------------------------------------------------------------ |
| MindSpore | 2.6.0                                                                          |
| CANN      | 8.0.1                                                                          |
| Python    | 3.10.12                                                                        |
| 设备      | Ascend 910B3 (NPU 7，空闲确认：`npu-smi info`，详见 [01_npu_smi_reference.md](../05_tools/01_npu_smi_reference.md)) |
| 兼容性    | MindSpore ≥2.6 对应 CANN ≥8.0，当前组合无版本冲突警告                          |

---

## 2. MindSpore vs PyTorch 核心编程范式对比

### 2.1 模型定义

**PyTorch** (`nn.Module`):

```python
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))
```

**MindSpore** (`nn.Cell`):

```python
import mindspore.nn as nn

class Bottleneck(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.relu = nn.ReLU()

    def construct(self, x):
        return self.relu(self.bn1(self.conv1(x)))
```

关键差异：

- PyTorch 的 `forward` → MindSpore 的 `construct`
- PyTorch 的 `nn.Module` → MindSpore 的 `nn.Cell`
- MindSpore 的 `nn.Conv2d` 默认无 bias（`has_bias=False`），PyTorch 默认有 bias

### 2.2 训练循环

**PyTorch**:

```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

**MindSpore**:

```python
# 定义前向+损失函数
def forward_fn(data, label):
    logits = net(data)
    loss = loss_fn(logits, label)
    return loss

# 获取梯度函数
grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
loss, grads = grad_fn(images, labels)
optimizer(grads)  # optimizer 作为函数调用
```

关键差异：

- MindSpore 使用函数式梯度：`ms.value_and_grad()` 返回梯度函数
- 不需要显式 `zero_grad()`，梯度在每次 `value_and_grad` 调用时自动清零
- `optimizer(grads)` 而不是 `optimizer.step()`

### 2.3 动态图 vs 静态图

| 特性         | PyTorch                                    | MindSpore                            |
| ------------ | ------------------------------------------ | ------------------------------------ |
| 动态图       | 默认 (eager)                               | `PYNATIVE_MODE`                      |
| 静态图       | `torch.compile()` (Python 字节码→FX Graph) | `GRAPH_MODE` (源码级 JIT 编译)       |
| 设置方式     | `torch.compile(model)`                     | `ms.set_context(mode=ms.GRAPH_MODE)` |
| 编译时机     | 运行期根据输入形状触发                     | 模型定义后首次执行时编译             |
| debug 友好度 | 高（用 `print`/`pdb`）                     | Graph 模式下不支持动态 `print`       |

### 2.4 常用 API 对照速查

| 功能       | PyTorch                                      | MindSpore                                                      |
| ---------- | -------------------------------------------- | -------------------------------------------------------------- |
| 模块基类   | `nn.Module`                                  | `nn.Cell`                                                      |
| 前向方法   | `forward()`                                  | `construct()`                                                  |
| 卷积       | `nn.Conv2d(c_in, c_out, k, stride, padding)` | `nn.Conv2d(c_in, c_out, k, stride, pad_mode='pad', padding=p)` |
| 批归一化   | `nn.BatchNorm2d(c)`                          | `nn.BatchNorm2d(c)`                                            |
| 激活函数   | `nn.ReLU(inplace=True)`                      | `nn.ReLU()` (无 inplace)                                       |
| 最大池化   | `nn.MaxPool2d(k, s, p)`                      | `nn.MaxPool2d(k, s, pad_mode='pad', padding=p)`                |
| 全连接     | `nn.Linear(in_f, out_f)`                     | `nn.Dense(in_f, out_f)`                                        |
| Flatten    | `nn.Flatten()`                               | `nn.Flatten()`                                                 |
| 序列容器   | `nn.Sequential(*layers)`                     | `nn.SequentialCell(layers)`                                    |
| 优化器     | `optim.SGD(params, lr, momentum)`            | `nn.SGD(params, learning_rate, momentum)`                      |
| 交叉熵     | `nn.CrossEntropyLoss()`                      | `nn.SoftmaxCrossEntropyWithLogits(sparse=True)`                |
| 梯度计算   | `loss.backward()`                            | `ms.value_and_grad(forward_fn)(...)`                           |
| 设备同步   | `torch.cuda.synchronize()`                   | `ms.hal.synchronize()`                                         |
| 设置上下文 | `torch.cuda.set_device(i)`                   | `ms.set_context(device_target='Ascend', device_id=i)`          |
| 张量创建   | `torch.randn(3, 4)`                          | `ms.Tensor(np.random.randn(3, 4).astype(np.float32))`          |
| 张量到设备 | `tensor.to('cuda')`                          | (上下文设置后自动)                                             |

---

## 3. 实验结果

在 NPU 7 上使用 ResNet-50 (batch_size=64, 224×224) 训练 2 epochs：

| 框架        | 模式              | 吞吐量    | 相对性能 |
| ----------- | ----------------- | --------- | -------- |
| PyTorch NPU | Eager             | 545 img/s | 基准     |
| MindSpore   | PyNative (动态图) | 165 img/s | 0.30×    |
| MindSpore   | Graph (静态图)    | 159 img/s | 0.29×    |

> MindSpore 吞吐量低于 PyTorch NPU 约 3.3×。原因分析：
>
> 1. MindSpore 2.6.0 对 CANN 8.0.1 的适配不是最优目标版本（MindSpore 2.6 目标 CANN 8.2+）
> 2. 手写的 ResNet 实现没有使用 MindSpore ModelZoo 的优化版本
> 3. MindSpore 对 910B3 的编译优化在 8.0.1 上不如 PyTorch NPU 成熟
> 4. PyNative 与 Graph 模式性能相近说明编译优化并未显著提升此用例

**学习意义大于性能意义**：此实验的价值在于掌握 MindSpore 的编程范式和 API 差异，生产环境中应使用匹配的 CANN/MindSpore 版本组合。

---

## 4. 版本兼容性说明

MindSpore 版本必须与 CANN 版本匹配。从实验观察到：

| MindSpore | 需要的 CANN 版本 | 本环境 CANN | 状态                       |
| --------- | ---------------- | ----------- | -------------------------- |
| 2.9.0     | 8.2 / 8.3 / 9.0  | 8.0.1       | 版本不匹配，countdown 警告 |
| 2.6.0     | 8.0+ (?)         | 8.0.1       | 无版本警告，功能正常       |

PyPI 上可用的 MindSpore Ascend 版本（aarch64）：2.6.0 ~ 2.9.0。更早版本需从 MindSpore 官网或华为 Ascend 仓库获取特定安装包。

完整兼容性矩阵参考：[MindSpore 安装指南](https://www.mindspore.cn/install)

---

## 5. MindSpore Lite

服务器系统环境中已安装 `mindspore-lite 2.3.0rc3`（系统 pip）。这是一个轻量级推理引擎，用于在端侧、边缘设备上运行 MindSpore 模型，不用于训练。与完整 MindSpore 包的关系：

- **mindspore**（完整版）= 训练 + 推理 + Ascend/Python 后端
- **mindspore-lite** = 仅推理 + C++/Java API + 端侧优化

---

## 6. 参考链接

- [MindSpore 安装指南与兼容性矩阵](https://www.mindspore.cn/install)
- [MindSpore API 文档](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- [MindSpore 编程指南](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
- [昇腾社区 — MindSpore 适配](https://www.hiascend.com/ecosystem/mindspore)
